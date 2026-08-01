from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import html
import json
import math
import shutil
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf


SCRIPT_PATH = Path(__file__).resolve()
ROOT = SCRIPT_PATH.parents[2]

LEGACY_IMPORT_DIRS = [
    ROOT,
    ROOT / "experiments/E0202_anchor_replay_audit",
    ROOT / "experiments/E0203_mcp_short_training",
    ROOT / "experiments/E0205B_mcp_quality_gate",
    ROOT / "experiments/E0206B_depth1_single_state_overfit",
]

for directory in reversed(LEGACY_IMPORT_DIRS):
    text = str(directory)
    if text in sys.path:
        sys.path.remove(text)
    sys.path.insert(0, text)

import run_overfit  # type: ignore  # noqa: E402
from decode_quality import save_video  # type: ignore  # noqa: E402
from train_short import (  # type: ignore  # noqa: E402
    BLOCK_FRAMES,
    CONFIG_PATH,
    build_steps,
    move_to_cpu,
    move_to_device,
    run_state,
)
from utils.wan_wrapper import (  # noqa: E402
    WanTextEncoder,
    WanVAEWrapper,
)


FORMAL_MANIFEST_PATH = (
    ROOT
    / "experiments/E0208C_teacher_rollout_formal/manifest.json"
)
FORMAL_TRAINING_REPORT_PATH = (
    ROOT
    / "experiments/E0209_depth1_formal_training/report.json"
)
FORMAL_CHECKPOINT_PATH = (
    ROOT
    / "experiments/E0209_depth1_formal_training/mcp_depth1_best.pt"
)
OLD_CHECKPOINT_PATH = (
    ROOT
    / "experiments/E0207C_depth1_continuation/mcp_depth1_best.pt"
)
LEGACY_SAMPLE_DIR = (
    ROOT
    / "experiments/E0201_teacher_rollout_smoke"
)
DEFAULT_OUTPUT_ROOT = (
    ROOT
    / "experiments/E0210_depth1_visual_gate"
)

EXPECTED_BACKBONE_SHA256 = (
    "a0413986d9734e02c09504e1520f5697"
    "ba6df731bb2f0f35577485e9cc8f56a3"
)
ANCHORS = (0, 1, 2, 3)
VARIANT_ORDER = ("target", "step32", "e0207c", "e0209")


@dataclass(frozen=True)
class SampleSpec:
    key: str
    source: str
    sample_index: int
    split_index: int | None
    selection_label: str
    payload_path: Path
    reference_mean_mse: float | None = None
    reference_min_mse: float | None = None
    reference_max_mse: float | None = None
    reference_anchor_spread: float | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--suite",
        choices=("regression", "full"),
        default="regression",
        help=(
            "regression: only legacy samples 004/005; "
            "full: legacy samples plus eight formal-validation samples"
        ),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    parser.add_argument("--skip-old", action="store_true")
    parser.add_argument("--save-latents", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json_write(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def finite_number(value: float) -> bool:
    return math.isfinite(float(value))


def _flatten_tensors(value: Any) -> list[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, dict):
        result: list[torch.Tensor] = []
        for child in value.values():
            result.extend(_flatten_tensors(child))
        return result
    if isinstance(value, (list, tuple)):
        result = []
        for child in value:
            result.extend(_flatten_tensors(child))
        return result
    return []


def resolve_manifest_file(manifest_path: Path, record: dict[str, Any]) -> Path:
    value = None
    for key in (
        "file",
        "path",
        "payload_path",
        "artifact_path",
    ):
        candidate = record.get(key)
        if candidate:
            value = str(candidate)
            break
    if value is None:
        raise RuntimeError(
            f"Manifest record has no payload path: keys={sorted(record)}"
        )

    path = Path(value)
    candidates = [path]
    if not path.is_absolute():
        candidates.extend(
            [
                manifest_path.parent / path,
                ROOT / path,
            ]
        )

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    raise FileNotFoundError(
        f"Cannot resolve payload for sample={record.get('sample_index')}: "
        f"tried={[str(item) for item in candidates]}"
    )


def formal_manifest_record_map() -> dict[int, dict[str, Any]]:
    if not FORMAL_MANIFEST_PATH.is_file():
        raise FileNotFoundError(FORMAL_MANIFEST_PATH)
    manifest = json.loads(
        FORMAL_MANIFEST_PATH.read_text(encoding="utf-8")
    )
    if manifest.get("status") != "PASS":
        raise RuntimeError("Formal teacher manifest is not PASS.")
    records = manifest.get("samples")
    if not isinstance(records, list):
        raise RuntimeError("Formal manifest has no samples list.")
    result: dict[int, dict[str, Any]] = {}
    for record in records:
        sample_index = int(record["sample_index"])
        if sample_index in result:
            raise RuntimeError(f"Duplicate formal sample_index={sample_index}")
        result[sample_index] = record
    return result


def select_formal_validation_samples() -> list[SampleSpec]:
    if not FORMAL_TRAINING_REPORT_PATH.is_file():
        raise FileNotFoundError(FORMAL_TRAINING_REPORT_PATH)

    report = json.loads(
        FORMAL_TRAINING_REPORT_PATH.read_text(encoding="utf-8")
    )
    states = report["metrics"]["best_validation"]["states"]

    grouped: dict[int, list[float]] = defaultdict(list)
    for state in states:
        sample_index = int(state["sample_index"])
        loss = float(state["depth_losses"][0])
        grouped[sample_index].append(loss)

    if len(grouped) != 256:
        raise RuntimeError(
            f"Expected 256 formal validation samples, found {len(grouped)}."
        )
    for sample_index, losses in grouped.items():
        if len(losses) != 4:
            raise RuntimeError(
                f"sample={sample_index} has {len(losses)} anchors, expected 4."
            )

    summaries = []
    for sample_index, losses in grouped.items():
        summaries.append(
            {
                "sample_index": sample_index,
                "mean": sum(losses) / len(losses),
                "min": min(losses),
                "max": max(losses),
                "spread": max(losses) - min(losses),
            }
        )
    summaries.sort(key=lambda item: item["mean"])

    def quantile_record(q: float) -> dict[str, Any]:
        index = round(q * (len(summaries) - 1))
        return summaries[index]

    selected_with_labels: list[tuple[str, dict[str, Any]]] = [
        ("easiest", quantile_record(0.00)),
        ("p10", quantile_record(0.10)),
        ("p25", quantile_record(0.25)),
        ("median", quantile_record(0.50)),
        ("p75", quantile_record(0.75)),
        ("p90", quantile_record(0.90)),
    ]

    used = {item["sample_index"] for _, item in selected_with_labels}
    spread_record = max(
        (item for item in summaries if item["sample_index"] not in used),
        key=lambda item: item["spread"],
    )
    selected_with_labels.append(("largest_anchor_spread", spread_record))
    used.add(spread_record["sample_index"])

    worst_record = max(
        (item for item in summaries if item["sample_index"] not in used),
        key=lambda item: item["mean"],
    )
    selected_with_labels.append(("worst", worst_record))

    record_map = formal_manifest_record_map()
    specs: list[SampleSpec] = []

    for label, summary in selected_with_labels:
        sample_index = int(summary["sample_index"])
        record = record_map[sample_index]
        if record.get("split") != "validation":
            raise RuntimeError(
                f"Selected sample={sample_index} is not validation."
            )
        split_index = int(record["split_index"])
        specs.append(
            SampleSpec(
                key=f"formal_validation_{split_index:06d}",
                source="formal_validation",
                sample_index=sample_index,
                split_index=split_index,
                selection_label=label,
                payload_path=resolve_manifest_file(
                    FORMAL_MANIFEST_PATH,
                    record,
                ),
                reference_mean_mse=float(summary["mean"]),
                reference_min_mse=float(summary["min"]),
                reference_max_mse=float(summary["max"]),
                reference_anchor_spread=float(summary["spread"]),
            )
        )

    return specs


def legacy_sample_specs() -> list[SampleSpec]:
    specs = []
    for index, label in ((4, "legacy_failure_person_store"), (5, "legacy_failure_cat")):
        path = LEGACY_SAMPLE_DIR / f"teacher_sample_{index:03d}.pt"
        if not path.is_file():
            raise FileNotFoundError(path)
        specs.append(
            SampleSpec(
                key=f"legacy_{index:03d}",
                source="legacy_regression",
                sample_index=index,
                split_index=None,
                selection_label=label,
                payload_path=path.resolve(),
            )
        )
    return specs


def load_payloads(specs: list[SampleSpec]) -> dict[str, dict[str, Any]]:
    payloads: dict[str, dict[str, Any]] = {}
    for spec in specs:
        payload = torch.load(
            spec.payload_path,
            map_location="cpu",
            weights_only=False,
        )
        if int(payload["sample_index"]) != spec.sample_index:
            raise RuntimeError(
                f"Payload mismatch for {spec.key}: "
                f"expected={spec.sample_index}, got={payload.get('sample_index')}"
            )
        for tensor_key in ("source_noise", "target_latent"):
            tensor = payload[tensor_key]
            if tuple(tensor.shape) != (1, 21, 16, 60, 104):
                raise RuntimeError(
                    f"{spec.key} {tensor_key} shape={tuple(tensor.shape)}"
                )
            if tensor.dtype != torch.bfloat16:
                raise RuntimeError(
                    f"{spec.key} {tensor_key} dtype={tensor.dtype}"
                )
            if not bool(torch.isfinite(tensor.float()).all().item()):
                raise RuntimeError(f"{spec.key} {tensor_key} is non-finite")
        payloads[spec.key] = payload
    return payloads


def precompute_prompt_embeddings(
    specs: list[SampleSpec],
    payloads: dict[str, dict[str, Any]],
    device: torch.device,
) -> dict[str, Any]:
    print("===== PRECOMPUTE PROMPT EMBEDDINGS =====", flush=True)
    encoder = WanTextEncoder()
    encoder.eval().requires_grad_(False)
    encoder.to(device=device, dtype=torch.bfloat16)
    result: dict[str, Any] = {}
    for spec in specs:
        with torch.inference_mode():
            embedding = encoder(
                text_prompts=[payloads[spec.key]["prompt"]]
            )
        result[spec.key] = move_to_cpu(embedding)
        del embedding
        print(f"embedding={spec.key}", flush=True)
    encoder.to("cpu")
    del encoder
    gc.collect()
    torch.cuda.empty_cache()
    return result


def checkpoint_metadata(path: Path) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != "mcp_depth1_multistate_v1":
        raise RuntimeError(
            f"Unexpected checkpoint format at {path}: {payload.get('format')}"
        )
    if payload.get("backbone_sha256") != EXPECTED_BACKBONE_SHA256:
        raise RuntimeError(f"Backbone SHA mismatch in {path}")
    state = payload.get("mcp")
    if not isinstance(state, dict):
        raise RuntimeError(f"Checkpoint {path} has no MCP state.")
    metadata = {
        "path": str(path.resolve()),
        "sha256": file_sha256(path),
        "format": payload.get("format"),
        "step": int(payload.get("step", -1)),
        "trainable_scope": payload.get("trainable_scope"),
        "backbone_sha256": payload.get("backbone_sha256"),
    }
    return metadata, state


def restore_checkpoint(
    generator: torch.nn.Module,
    path: Path,
) -> dict[str, Any]:
    metadata, state = checkpoint_metadata(path)
    result = generator.mcp.load_state_dict(state, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(
            f"Strict MCP restore failed for {path}: {result}"
        )
    del state
    gc.collect()
    return metadata


def copy_state(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in module.state_dict().items()
    }


def audit_formal_checkpoint_scope(
    base_state: dict[str, torch.Tensor],
    formal_state: dict[str, torch.Tensor],
) -> dict[str, Any]:
    if base_state.keys() != formal_state.keys():
        raise RuntimeError("Base/formal MCP state keys differ.")
    depth1_changed = []
    frozen_changed = []
    for key in base_state:
        equal = torch.equal(base_state[key], formal_state[key])
        if key.startswith("mcp_modules.0."):
            if not equal:
                depth1_changed.append(key)
        elif not equal:
            frozen_changed.append(key)
    report = {
        "depth1_changed_tensor_count": len(depth1_changed),
        "depth1_changed_examples": depth1_changed[:10],
        "frozen_depth23_changed_tensor_count": len(frozen_changed),
        "frozen_depth23_changed_examples": frozen_changed[:10],
    }
    if not depth1_changed:
        raise RuntimeError("Formal checkpoint did not change depth-1 tensors.")
    if frozen_changed:
        raise RuntimeError(
            f"Formal checkpoint changed frozen depth-2/3 tensors: {frozen_changed[:10]}"
        )
    return report


def evaluate_variant(
    *,
    name: str,
    generator: torch.nn.Module,
    specs: list[SampleSpec],
    payloads: dict[str, dict[str, Any]],
    prompt_embeddings: dict[str, Any],
    denoising_steps: list[float],
    device: torch.device,
) -> tuple[dict[str, dict[int, torch.Tensor]], list[dict[str, Any]]]:
    print(f"===== GENERATE {name} =====", flush=True)
    run_overfit.resolve_depth_module(generator.mcp, 0)

    drafts: dict[str, dict[int, torch.Tensor]] = {}
    records: list[dict[str, Any]] = []

    for spec in specs:
        payload = payloads[spec.key]
        source_noise = payload["source_noise"].to(
            device=device,
            dtype=torch.bfloat16,
        )
        target_latent = payload["target_latent"].to(
            device=device,
            dtype=torch.bfloat16,
        )
        conditional_dict = move_to_device(
            prompt_embeddings[spec.key],
            device,
        )
        drafts[spec.key] = {}

        for anchor in ANCHORS:
            target_block = anchor + 1
            start = target_block * BLOCK_FRAMES
            end = start + BLOCK_FRAMES
            noise_chunk = source_noise[:, start:end]
            target_chunk = target_latent[:, start:end]
            captures: list[torch.Tensor] = []

            def capture_hook(_module: Any, _inputs: Any, output: Any) -> None:
                for tensor in _flatten_tensors(output):
                    captures.append(tensor.detach().clone())

            handle = generator.register_forward_hook(capture_hook)
            try:
                with torch.inference_mode():
                    result = run_state(
                        generator=generator,
                        payload=payload,
                        conditional_dict=conditional_dict,
                        denoising_steps=denoising_steps,
                        anchor_block=anchor,
                        require_grad=False,
                    )
            finally:
                handle.remove()

            expected_loss = float(result["depth_losses"][0])
            candidates = []
            for predicted_flow in captures:
                if tuple(predicted_flow.shape) != tuple(noise_chunk.shape):
                    continue
                draft = noise_chunk - predicted_flow
                observed_loss = float(
                    torch.nn.functional.mse_loss(
                        draft.float(),
                        target_chunk.float(),
                    ).item()
                )
                candidates.append(
                    (
                        abs(observed_loss - expected_loss),
                        observed_loss,
                        draft,
                        predicted_flow,
                    )
                )

            if not candidates:
                raise RuntimeError(
                    f"No depth-1 flow capture for {spec.key} anchor={anchor}; "
                    f"capture_count={len(captures)}"
                )
            candidates.sort(key=lambda item: item[0])
            path_difference, observed_loss, draft, predicted_flow = candidates[0]
            if path_difference > 1.0e-5:
                raise RuntimeError(
                    f"Capture mismatch for {spec.key} anchor={anchor}: "
                    f"expected={expected_loss}, observed={observed_loss}, "
                    f"difference={path_difference}"
                )

            metrics = run_overfit.metric_bundle(
                draft=draft,
                target=target_chunk,
                noise=noise_chunk,
                predicted_flow=predicted_flow,
            )
            if not all(
                finite_number(float(value))
                for key, value in metrics.items()
                if key != "finite" and isinstance(value, (int, float))
            ):
                raise RuntimeError(f"Non-finite metrics for {spec.key} {name}")
            if not bool(metrics["finite"]):
                raise RuntimeError(f"Non-finite tensor for {spec.key} {name}")

            drafts[spec.key][anchor] = draft.detach().cpu().clone()
            record = {
                "sample_key": spec.key,
                "source": spec.source,
                "sample_index": spec.sample_index,
                "split_index": spec.split_index,
                "selection_label": spec.selection_label,
                "variant": name,
                "anchor_block": anchor,
                "target_block": target_block,
                "run_state_depth1_loss": expected_loss,
                "capture_path_difference": path_difference,
                "capture_tensor_count": len(captures),
                **metrics,
            }
            records.append(record)
            print(
                f"{name} sample={spec.key} anchor={anchor} "
                f"mse={metrics['draft_target_mse']:.8f} "
                f"progress={metrics['progress_to_target']:.6f} "
                f"cos={metrics['flow_cosine_with_oracle']:.6f}",
                flush=True,
            )

            del result, candidates, draft, predicted_flow

        del source_noise, target_latent, conditional_dict
        gc.collect()
        torch.cuda.empty_cache()

    return drafts, records


def aggregate_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise RuntimeError("Cannot aggregate empty records.")
    keys = (
        "noise_target_mse",
        "draft_target_mse",
        "draft_noise_mse",
        "progress_to_target",
        "flow_cosine_with_oracle",
        "flow_norm_ratio",
    )
    result: dict[str, Any] = {"state_count": len(records)}
    for key in keys:
        result[key] = sum(float(record[key]) for record in records) / len(records)
    result["all_finite"] = all(bool(record["finite"]) for record in records)
    return result


def per_sample_variant_summary(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["sample_key"])].append(record)
    return {
        key: aggregate_records(value)
        for key, value in sorted(grouped.items())
    }


def splice_variant(
    target_latent: torch.Tensor,
    drafts: dict[int, torch.Tensor],
) -> torch.Tensor:
    variant = target_latent.detach().cpu().clone()
    for anchor in ANCHORS:
        target_block = anchor + 1
        start = target_block * BLOCK_FRAMES
        end = start + BLOCK_FRAMES
        variant[:, start:end] = drafts[anchor]
    return variant


def decode_videos(
    *,
    output_dir: Path,
    specs: list[SampleSpec],
    payloads: dict[str, dict[str, Any]],
    all_drafts: dict[str, dict[str, dict[int, torch.Tensor]]],
    device: torch.device,
    save_latents: bool,
) -> list[dict[str, Any]]:
    print("===== FULL VAE DECODE =====", flush=True)
    video_dir = output_dir / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    latent_dir = output_dir / "latents"
    if save_latents:
        latent_dir.mkdir(parents=True, exist_ok=True)

    vae = WanVAEWrapper()
    vae.eval().requires_grad_(False)
    vae.to(device=device, dtype=torch.bfloat16)
    records: list[dict[str, Any]] = []

    for spec in specs:
        target_latent = payloads[spec.key]["target_latent"].cpu()
        variants: dict[str, torch.Tensor] = {"target": target_latent}
        for name, draft_by_sample in all_drafts.items():
            variants[name] = splice_variant(
                target_latent,
                draft_by_sample[spec.key],
            )

        for name in VARIANT_ORDER:
            if name not in variants:
                continue
            latent = variants[name]
            if save_latents:
                torch.save(
                    {
                        "format": "e0210_fixed_teacher_history_latent_v1",
                        "sample_key": spec.key,
                        "variant": name,
                        "latent": latent,
                    },
                    latent_dir / f"{spec.key}_{name}.pt",
                )
            print(f"decode sample={spec.key} variant={name}", flush=True)
            latent_gpu = latent.to(device=device, dtype=torch.bfloat16)
            with torch.inference_mode():
                pixels = vae.decode_to_pixel(latent_gpu)
            path = video_dir / f"{spec.key}_{name}.mp4"
            artifact = save_video(pixels=pixels, path=path)
            records.append(
                {
                    "sample_key": spec.key,
                    "sample_index": spec.sample_index,
                    "variant": name,
                    "relative_path": str(path.relative_to(output_dir)),
                    **artifact,
                }
            )
            del latent_gpu, pixels
            gc.collect()
            torch.cuda.empty_cache()

    vae.to("cpu")
    del vae
    gc.collect()
    torch.cuda.empty_cache()
    return records


def write_metrics_csv(records: list[dict[str, Any]], path: Path) -> None:
    fields = [
        "sample_key",
        "source",
        "sample_index",
        "split_index",
        "selection_label",
        "variant",
        "anchor_block",
        "target_block",
        "draft_target_mse",
        "noise_target_mse",
        "progress_to_target",
        "flow_cosine_with_oracle",
        "flow_norm_ratio",
        "capture_path_difference",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            writer.writerow(record)


def write_review_html(
    *,
    output_dir: Path,
    specs: list[SampleSpec],
    payloads: dict[str, dict[str, Any]],
    video_records: list[dict[str, Any]],
    per_sample_metrics: dict[str, dict[str, dict[str, Any]]],
) -> None:
    video_map = {
        (record["sample_key"], record["variant"]): record["relative_path"]
        for record in video_records
    }
    cards = []
    for spec in specs:
        cells = []
        for variant in VARIANT_ORDER:
            path = video_map.get((spec.key, variant))
            if path is None:
                continue
            metric = per_sample_metrics.get(variant, {}).get(spec.key)
            metric_text = "Target teacher"
            if metric is not None:
                metric_text = (
                    f"mean MSE={metric['draft_target_mse']:.6f}; "
                    f"progress={metric['progress_to_target']:.4f}; "
                    f"cos={metric['flow_cosine_with_oracle']:.4f}"
                )
            cells.append(
                f"""
                <div class="variant">
                  <h3>{html.escape(variant)}</h3>
                  <video controls preload="metadata" src="{html.escape(path)}"></video>
                  <p>{html.escape(metric_text)}</p>
                </div>
                """
            )
        reference = ""
        if spec.reference_mean_mse is not None:
            reference = (
                f"Formal best-validation reference: mean={spec.reference_mean_mse:.6f}, "
                f"min={spec.reference_min_mse:.6f}, max={spec.reference_max_mse:.6f}, "
                f"spread={spec.reference_anchor_spread:.6f}"
            )
        prompt = str(payloads[spec.key]["prompt"])
        cards.append(
            f"""
            <section>
              <h2>{html.escape(spec.key)} — {html.escape(spec.selection_label)}</h2>
              <p><b>sample_index:</b> {spec.sample_index} &nbsp; <b>source:</b> {html.escape(spec.source)}</p>
              <p>{html.escape(reference)}</p>
              <p><b>Prompt:</b> {html.escape(prompt)}</p>
              <div class="grid">{''.join(cells)}</div>
            </section>
            """
        )

    content = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>E0210 depth-1 fixed-teacher-history visual gate</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; background: #f5f5f5; }}
section {{ background: white; padding: 18px; margin-bottom: 24px; border-radius: 10px; }}
.grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; }}
.variant {{ border: 1px solid #ddd; padding: 10px; border-radius: 8px; }}
video {{ width: 100%; background: black; }}
code {{ background: #eee; padding: 2px 4px; }}
@media (max-width: 900px) {{ .grid {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>
<h1>E0210 depth-1 fixed-teacher-history visual gate</h1>
<p>Each MCP variant replaces target blocks 1–4, while every draft was computed from the correct teacher-generated history. This is not an always-accept closed-loop test.</p>
<p>Manual gate: compare <code>e0209</code> against <code>target</code>, <code>step32</code>, and <code>e0207c</code>. Check subject preservation, sharpness, temporal continuity, flicker, and scene collapse.</p>
{''.join(cards)}
</body>
</html>
"""
    (output_dir / "review.html").write_text(content, encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else (DEFAULT_OUTPUT_ROOT / args.suite).resolve()
    )

    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Output exists: {output_dir}. Pass --overwrite to replace it."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    required = [
        FORMAL_CHECKPOINT_PATH,
        FORMAL_TRAINING_REPORT_PATH,
        FORMAL_MANIFEST_PATH,
    ]
    if not args.skip_old:
        required.append(OLD_CHECKPOINT_PATH)
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    specs = legacy_sample_specs()
    if args.suite == "full":
        specs.extend(select_formal_validation_samples())
    payloads = load_payloads(specs)

    device = torch.device(args.device)
    if device.type != "cuda":
        raise RuntimeError("E0210 requires a CUDA device.")
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = True

    started = time.time()
    contract = {
        "status": "RUNNING",
        "experiment": "E0210_depth1_visual_gate",
        "suite": args.suite,
        "scope": "fixed teacher-generated history",
        "output_dir": str(output_dir),
        "formal_manifest": str(FORMAL_MANIFEST_PATH),
        "formal_manifest_sha256": file_sha256(FORMAL_MANIFEST_PATH),
        "formal_training_report": str(FORMAL_TRAINING_REPORT_PATH),
        "formal_training_report_sha256": file_sha256(FORMAL_TRAINING_REPORT_PATH),
        "formal_checkpoint": str(FORMAL_CHECKPOINT_PATH),
        "formal_checkpoint_sha256": file_sha256(FORMAL_CHECKPOINT_PATH),
        "old_checkpoint": None if args.skip_old else str(OLD_CHECKPOINT_PATH),
        "old_checkpoint_sha256": None if args.skip_old else file_sha256(OLD_CHECKPOINT_PATH),
        "anchors": list(ANCHORS),
        "samples": [
            {
                "key": spec.key,
                "source": spec.source,
                "sample_index": spec.sample_index,
                "split_index": spec.split_index,
                "selection_label": spec.selection_label,
                "payload_path": str(spec.payload_path),
                "reference_mean_mse": spec.reference_mean_mse,
                "reference_min_mse": spec.reference_min_mse,
                "reference_max_mse": spec.reference_max_mse,
                "reference_anchor_spread": spec.reference_anchor_spread,
            }
            for spec in specs
        ],
        "started_unix": started,
    }
    atomic_json_write(contract, output_dir / "contract.json")

    prompt_embeddings = precompute_prompt_embeddings(specs, payloads, device)
    config = OmegaConf.merge(
        OmegaConf.load(ROOT / "configs/default_config.yaml"),
        OmegaConf.load(ROOT / CONFIG_PATH),
    )

    print("===== LOAD OFFICIAL BACKBONE + STEP32 MCP =====", flush=True)
    generator = run_overfit.load_generator(
        config=config,
        device=device,
        train_depth1=False,
    )
    raw_steps, warped_steps = build_steps(config, generator.get_scheduler())
    base_state = copy_state(generator.mcp)

    all_drafts: dict[str, dict[str, dict[int, torch.Tensor]]] = {}
    all_records: dict[str, list[dict[str, Any]]] = {}
    checkpoint_reports: dict[str, Any] = {}

    step32_drafts, step32_records = evaluate_variant(
        name="step32",
        generator=generator,
        specs=specs,
        payloads=payloads,
        prompt_embeddings=prompt_embeddings,
        denoising_steps=warped_steps,
        device=device,
    )
    all_drafts["step32"] = step32_drafts
    all_records["step32"] = step32_records

    if not args.skip_old:
        print("===== LOAD E0207C BEST MCP =====", flush=True)
        checkpoint_reports["e0207c"] = restore_checkpoint(
            generator,
            OLD_CHECKPOINT_PATH,
        )
        old_drafts, old_records = evaluate_variant(
            name="e0207c",
            generator=generator,
            specs=specs,
            payloads=payloads,
            prompt_embeddings=prompt_embeddings,
            denoising_steps=warped_steps,
            device=device,
        )
        all_drafts["e0207c"] = old_drafts
        all_records["e0207c"] = old_records

    print("===== LOAD E0209 FORMAL BEST MCP =====", flush=True)
    checkpoint_reports["e0209"] = restore_checkpoint(
        generator,
        FORMAL_CHECKPOINT_PATH,
    )
    formal_state = copy_state(generator.mcp)
    scope_audit = audit_formal_checkpoint_scope(base_state, formal_state)

    formal_drafts, formal_records = evaluate_variant(
        name="e0209",
        generator=generator,
        specs=specs,
        payloads=payloads,
        prompt_embeddings=prompt_embeddings,
        denoising_steps=warped_steps,
        device=device,
    )
    all_drafts["e0209"] = formal_drafts
    all_records["e0209"] = formal_records

    generator.to("cpu")
    del generator, base_state, formal_state
    gc.collect()
    torch.cuda.empty_cache()

    video_records = decode_videos(
        output_dir=output_dir,
        specs=specs,
        payloads=payloads,
        all_drafts=all_drafts,
        device=device,
        save_latents=args.save_latents,
    )

    flat_records = [
        record
        for records in all_records.values()
        for record in records
    ]
    write_metrics_csv(flat_records, output_dir / "metrics.csv")

    aggregate = {
        name: aggregate_records(records)
        for name, records in all_records.items()
    }
    per_sample = {
        name: per_sample_variant_summary(records)
        for name, records in all_records.items()
    }

    write_review_html(
        output_dir=output_dir,
        specs=specs,
        payloads=payloads,
        video_records=video_records,
        per_sample_metrics=per_sample,
    )

    finished = time.time()
    report = {
        **contract,
        "status": "ARTIFACTS_READY",
        "finished_unix": finished,
        "elapsed_seconds": finished - started,
        "schedule": {
            "raw_steps": raw_steps,
            "warped_steps": warped_steps,
        },
        "checkpoints": checkpoint_reports,
        "formal_checkpoint_scope_audit": scope_audit,
        "metrics": {
            "aggregate": aggregate,
            "per_sample": per_sample,
            "records": flat_records,
        },
        "artifacts": {
            "review_html": str((output_dir / "review.html").resolve()),
            "metrics_csv": str((output_dir / "metrics.csv").resolve()),
            "videos": video_records,
            "video_count": len(video_records),
            "latents_saved": bool(args.save_latents),
        },
        "manual_gate": {
            "status": "PENDING",
            "criteria": [
                "E0209 replacement blocks are not pure noise or severely blurred.",
                "Legacy sample 005 preserves the cat, rainbow, and background.",
                "Legacy sample 004 preserves a stable person and store layout.",
                "Four consecutive replaced blocks do not cause catastrophic flicker or scene collapse.",
                "Formal hard-tail samples remain semantically recognizable and temporally coherent.",
            ],
            "decision": {
                "fixed_history_fail": "Do not enter depth-2; inspect loss/latent splice/decode semantics.",
                "fixed_history_pass": "Proceed to E0211 depth-1 always-accept closed-loop gate.",
            },
        },
    }
    atomic_json_write(report, output_dir / "report.json")

    print("===== E0210 RESULT =====", flush=True)
    print("status=ARTIFACTS_READY", flush=True)
    print(f"suite={args.suite}", flush=True)
    print(f"sample_count={len(specs)}", flush=True)
    print(f"video_count={len(video_records)}", flush=True)
    for name, metrics in aggregate.items():
        print(
            f"{name}_mse={metrics['draft_target_mse']:.8f} "
            f"{name}_progress={metrics['progress_to_target']:.6f}",
            flush=True,
        )
    print(f"review={output_dir / 'review.html'}", flush=True)
    print(f"report={output_dir / 'report.json'}", flush=True)
    print("E0210_ARTIFACTS_READY=PASS", flush=True)


if __name__ == "__main__":
    main()
