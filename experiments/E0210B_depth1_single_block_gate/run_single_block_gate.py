from __future__ import annotations

import argparse
import csv
import gc
import html
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf


SCRIPT_PATH = Path(__file__).resolve()
ROOT = SCRIPT_PATH.parents[2]
COMMON_DIR = ROOT / "experiments/E0210_depth1_visual_gate"

for directory in (ROOT, COMMON_DIR):
    text = str(directory)
    if text in sys.path:
        sys.path.remove(text)
    sys.path.insert(0, text)

import run_visual_gate as common  # type: ignore  # noqa: E402
from decode_quality import save_video  # type: ignore  # noqa: E402
from train_short import BLOCK_FRAMES, CONFIG_PATH, build_steps  # type: ignore  # noqa: E402
from utils.wan_wrapper import WanVAEWrapper  # noqa: E402


DEFAULT_OUTPUT_DIR = (
    ROOT
    / "experiments/E0210B_depth1_single_block_gate/results"
)
ANCHORS = (0, 1, 2, 3)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument("--save-latents", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def atomic_json_write(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def splice_one_block(
    target_latent: torch.Tensor,
    draft: torch.Tensor,
    anchor: int,
) -> torch.Tensor:
    target_block = anchor + 1
    start = target_block * BLOCK_FRAMES
    end = start + BLOCK_FRAMES
    result = target_latent.detach().cpu().clone()
    result[:, start:end] = draft.detach().cpu()
    return result


def splice_all_blocks(
    target_latent: torch.Tensor,
    drafts: dict[int, torch.Tensor],
) -> torch.Tensor:
    result = target_latent.detach().cpu().clone()
    for anchor in ANCHORS:
        target_block = anchor + 1
        start = target_block * BLOCK_FRAMES
        end = start + BLOCK_FRAMES
        result[:, start:end] = drafts[anchor].detach().cpu()
    return result


def decode_variants(
    *,
    output_dir: Path,
    specs: list[common.SampleSpec],
    payloads: dict[str, dict[str, Any]],
    drafts: dict[str, dict[int, torch.Tensor]],
    device: torch.device,
    save_latents: bool,
) -> list[dict[str, Any]]:
    print("===== DECODE SINGLE-BLOCK VARIANTS =====", flush=True)
    video_dir = output_dir / "videos"
    latent_dir = output_dir / "latents"
    video_dir.mkdir(parents=True, exist_ok=True)
    if save_latents:
        latent_dir.mkdir(parents=True, exist_ok=True)

    vae = WanVAEWrapper()
    vae.eval().requires_grad_(False)
    vae.to(device=device, dtype=torch.bfloat16)
    records: list[dict[str, Any]] = []

    for spec in specs:
        target_latent = payloads[spec.key]["target_latent"].cpu()
        variants: list[tuple[str, torch.Tensor, int | None]] = [
            ("target", target_latent, None),
            (
                "e0209_all4",
                splice_all_blocks(target_latent, drafts[spec.key]),
                None,
            ),
        ]
        for anchor in ANCHORS:
            variants.append(
                (
                    f"e0209_anchor{anchor}",
                    splice_one_block(
                        target_latent,
                        drafts[spec.key][anchor],
                        anchor,
                    ),
                    anchor,
                )
            )

        for variant, latent, anchor in variants:
            if save_latents:
                torch.save(
                    {
                        "format": "e0210b_single_block_latent_v1",
                        "sample_key": spec.key,
                        "variant": variant,
                        "anchor_block": anchor,
                        "target_block": None if anchor is None else anchor + 1,
                        "latent": latent,
                    },
                    latent_dir / f"{spec.key}_{variant}.pt",
                )

            print(
                f"decode sample={spec.key} variant={variant}",
                flush=True,
            )
            latent_gpu = latent.to(
                device=device,
                dtype=torch.bfloat16,
            )
            with torch.inference_mode():
                pixels = vae.decode_to_pixel(latent_gpu)
            path = video_dir / f"{spec.key}_{variant}.mp4"
            artifact = save_video(pixels=pixels, path=path)
            records.append(
                {
                    "sample_key": spec.key,
                    "sample_index": spec.sample_index,
                    "variant": variant,
                    "anchor_block": anchor,
                    "target_block": None if anchor is None else anchor + 1,
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


def write_metrics_csv(
    records: list[dict[str, Any]],
    path: Path,
) -> None:
    fields = [
        "sample_key",
        "sample_index",
        "selection_label",
        "anchor_block",
        "target_block",
        "draft_target_mse",
        "progress_to_target",
        "flow_cosine_with_oracle",
        "flow_norm_ratio",
        "finite",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({key: record.get(key) for key in fields})


def write_review_html(
    *,
    output_dir: Path,
    specs: list[common.SampleSpec],
    payloads: dict[str, dict[str, Any]],
    video_records: list[dict[str, Any]],
    metric_records: list[dict[str, Any]],
) -> None:
    videos_by_sample: dict[str, dict[str, dict[str, Any]]] = {}
    for record in video_records:
        videos_by_sample.setdefault(record["sample_key"], {})[
            record["variant"]
        ] = record

    metrics_by_sample_anchor = {
        (record["sample_key"], int(record["anchor_block"])): record
        for record in metric_records
    }

    order = [
        "target",
        "e0209_all4",
        "e0209_anchor0",
        "e0209_anchor1",
        "e0209_anchor2",
        "e0209_anchor3",
    ]
    cards = []
    for spec in specs:
        cells = []
        sample_videos = videos_by_sample[spec.key]
        for variant in order:
            record = sample_videos[variant]
            if variant.startswith("e0209_anchor"):
                anchor = int(variant[-1])
                metric = metrics_by_sample_anchor[(spec.key, anchor)]
                detail = (
                    f"Only target block {anchor + 1} is replaced; "
                    f"MSE={float(metric['draft_target_mse']):.6f}, "
                    f"progress={float(metric['progress_to_target']):.4f}"
                )
            elif variant == "e0209_all4":
                detail = "Reference: target blocks 1–4 are all replaced."
            else:
                detail = "Unmodified teacher target."
            cells.append(
                f"""
                <div class="variant">
                  <h3>{html.escape(variant)}</h3>
                  <p>{html.escape(detail)}</p>
                  <video controls loop muted preload="metadata">
                    <source src="{html.escape(record['relative_path'])}" type="video/mp4">
                  </video>
                </div>
                """
            )

        prompt = str(payloads[spec.key]["prompt"])
        cards.append(
            f"""
            <section>
              <h2>{html.escape(spec.key)} — {html.escape(spec.selection_label)}</h2>
              <p><b>Prompt:</b> {html.escape(prompt)}</p>
              <div class="grid">{''.join(cells)}</div>
            </section>
            """
        )

    content = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>E0210B depth-1 single-block diagnostic</title>
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
<h1>E0210B depth-1 single-block replacement diagnostic</h1>
<p>For each <code>e0209_anchorK</code> video, exactly one future block is replaced by the E0209 depth-1 draft. Every other block remains the teacher target. The <code>e0209_all4</code> video reproduces the previous four-block composite as a reference.</p>
<p>Decision rule: if individual anchor videos already lose the subject or become heavily blurred, the single draft block itself is inadequate. If all four individual replacements are acceptable but <code>e0209_all4</code> fails, the main issue is cross-block incompatibility rather than per-block reconstruction.</p>
{''.join(cards)}
</body>
</html>
"""
    (output_dir / "review.html").write_text(content, encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Output exists: {output_dir}. Pass --overwrite to replace it."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    required = [
        COMMON_DIR / "run_visual_gate.py",
        common.FORMAL_CHECKPOINT_PATH,
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    specs = common.legacy_sample_specs()
    payloads = common.load_payloads(specs)

    device = torch.device(args.device)
    if device.type != "cuda":
        raise RuntimeError("E0210B requires a CUDA device.")
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = True

    started = time.time()
    contract = {
        "status": "RUNNING",
        "experiment": "E0210B_depth1_single_block_gate",
        "scope": "fixed teacher history, one replaced future block per video",
        "purpose": (
            "Separate per-block draft quality from incompatibility caused by "
            "concatenating four independently teacher-conditioned drafts."
        ),
        "output_dir": str(output_dir),
        "formal_checkpoint": str(common.FORMAL_CHECKPOINT_PATH),
        "formal_checkpoint_sha256": common.file_sha256(
            common.FORMAL_CHECKPOINT_PATH
        ),
        "samples": [
            {
                "key": spec.key,
                "sample_index": spec.sample_index,
                "selection_label": spec.selection_label,
                "payload_path": str(spec.payload_path),
            }
            for spec in specs
        ],
        "anchors": list(ANCHORS),
        "started_unix": started,
    }
    atomic_json_write(contract, output_dir / "contract.json")

    prompt_embeddings = common.precompute_prompt_embeddings(
        specs,
        payloads,
        device,
    )
    config = OmegaConf.merge(
        OmegaConf.load(ROOT / "configs/default_config.yaml"),
        OmegaConf.load(ROOT / CONFIG_PATH),
    )

    print("===== LOAD OFFICIAL BACKBONE + E0209 BEST MCP =====", flush=True)
    generator = common.run_overfit.load_generator(
        config=config,
        device=device,
        train_depth1=False,
    )
    _raw_steps, warped_steps = build_steps(
        config,
        generator.get_scheduler(),
    )
    base_state = common.copy_state(generator.mcp)
    checkpoint_report = common.restore_checkpoint(
        generator,
        common.FORMAL_CHECKPOINT_PATH,
    )
    formal_state = common.copy_state(generator.mcp)
    scope_audit = common.audit_formal_checkpoint_scope(
        base_state,
        formal_state,
    )

    drafts, metric_records = common.evaluate_variant(
        name="e0209",
        generator=generator,
        specs=specs,
        payloads=payloads,
        prompt_embeddings=prompt_embeddings,
        denoising_steps=warped_steps,
        device=device,
    )

    generator.to("cpu")
    del generator, base_state, formal_state
    gc.collect()
    torch.cuda.empty_cache()

    video_records = decode_variants(
        output_dir=output_dir,
        specs=specs,
        payloads=payloads,
        drafts=drafts,
        device=device,
        save_latents=args.save_latents,
    )
    write_metrics_csv(metric_records, output_dir / "metrics.csv")
    write_review_html(
        output_dir=output_dir,
        specs=specs,
        payloads=payloads,
        video_records=video_records,
        metric_records=metric_records,
    )

    finished = time.time()
    report = {
        **contract,
        "status": "ARTIFACTS_READY",
        "finished_unix": finished,
        "elapsed_seconds": finished - started,
        "checkpoint": checkpoint_report,
        "formal_checkpoint_scope_audit": scope_audit,
        "metrics": {
            "aggregate": common.aggregate_records(metric_records),
            "per_sample": common.per_sample_variant_summary(metric_records),
            "records": metric_records,
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
                "Inspect each e0209_anchorK video only at the replaced block.",
                "The person/store remain stable in legacy_004.",
                "The cat/rainbow/background remain identifiable in legacy_005.",
                "No severe blur, ghosting, noise, or structural collapse occurs inside a single replaced block.",
            ],
            "decision": {
                "single_block_fail": (
                    "Per-block draft quality is insufficient. Stay at depth-1 and "
                    "audit objective/target representation/model capacity."
                ),
                "single_block_pass_all4_fail": (
                    "Per-block reconstruction is acceptable but drafts are mutually "
                    "incompatible. Next test sequential/on-policy history at depth-1."
                ),
                "single_block_and_all4_pass": (
                    "Revisit the previous manual judgment, then run the depth-1 "
                    "closed-loop gate before considering depth-2."
                ),
            },
        },
    }
    atomic_json_write(report, output_dir / "report.json")

    print("===== E0210B RESULT =====", flush=True)
    print("status=ARTIFACTS_READY", flush=True)
    print(f"sample_count={len(specs)}", flush=True)
    print(f"state_count={len(metric_records)}", flush=True)
    print(f"video_count={len(video_records)}", flush=True)
    print(f"review={output_dir / 'review.html'}", flush=True)
    print(f"report={output_dir / 'report.json'}", flush=True)
    print("E0210B_ARTIFACTS_READY=PASS", flush=True)


if __name__ == "__main__":
    main()
