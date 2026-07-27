from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import subprocess
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Optional, Sequence


SCHEMA_VERSION = "f4b2_wan_vae_transaction_gpu_v1"

FIXED_NUM_FRAMES = 9
FIXED_NUM_FRAME_PER_BLOCK = 3
FIXED_MCP_DEPTH = 1
FIXED_BATCH_SIZE = 1
FIXED_SCHEDULE = (1000,)
FIXED_LOCAL_ATTN_SIZE = -1
FIXED_SINK_SIZE = 0

EXPECTED_FIRST_PIXEL_FRAMES = 9
EXPECTED_CACHED_PIXEL_FRAMES = 12
DEFAULT_REPEAT_LOOPS = 20
WARMUP_LOOPS = 2
DEFAULT_PIXEL_MAX_ABS_TOLERANCE = 0.0


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "A100 Wan VAE cached-decode transaction validation. This script "
            "loads the real runtime, freezes four real latents, then compares "
            "Path A and Path B VAE cached-decode behavior. It does not save "
            "latents, videos, or checkpoint-derived artifacts."
        )
    )
    parser.add_argument("--config", required=True, help="Path to the run config YAML.")
    parser.add_argument("--checkpoint", required=True, help="Path to the MCP-complete checkpoint.")
    parser.add_argument("--prompt", required=True, help="Single T2V prompt.")
    parser.add_argument("--seed", type=int, default=0, help="Runtime seed.")
    parser.add_argument("--device", default="cuda", help="CUDA device, e.g. cuda or cuda:0.")
    parser.add_argument(
        "--output_json",
        required=True,
        help="Path for the JSON validation report. Written even when validation fails.",
    )
    parser.add_argument(
        "--repeat_loops",
        type=int,
        default=DEFAULT_REPEAT_LOOPS,
        help="Measured reject/rollback loop count after warmup.",
    )
    parser.add_argument(
        "--pixel_max_abs_tolerance",
        type=float,
        default=DEFAULT_PIXEL_MAX_ABS_TOLERANCE,
        help=(
            "Maximum allowed Path A/B pixel max-abs difference. Default 0.0 "
            "requires exact equality."
        ),
    )

    args = parser.parse_args(argv)
    if args.repeat_loops < 5:
        parser.error("--repeat_loops must be >= 5.")
    if args.pixel_max_abs_tolerance < 0.0 or not math.isfinite(args.pixel_max_abs_tolerance):
        parser.error("--pixel_max_abs_tolerance must be a finite non-negative number.")
    return args


def helper_args_from(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        config=args.config,
        checkpoint=args.checkpoint,
        prompt=args.prompt,
        seed=int(args.seed),
        num_frames=FIXED_NUM_FRAMES,
        mcp_depth=FIXED_MCP_DEPTH,
        disable_mcp=False,
        save_trace=None,
        fps=16,
        device=args.device,
    )


def git_head() -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def base_report(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "git_commit": git_head(),
        "config": str(Path(args.config)),
        "checkpoint": str(Path(args.checkpoint)),
        "prompt": args.prompt,
        "seed": int(args.seed),
        "device": args.device,
        "output_json": str(Path(args.output_json)),
        "repeat_loops": int(args.repeat_loops),
        "pixel_max_abs_tolerance": float(args.pixel_max_abs_tolerance),
        "fixed_experiment": {
            "num_frames": FIXED_NUM_FRAMES,
            "num_frame_per_block": FIXED_NUM_FRAME_PER_BLOCK,
            "mcp_depth": FIXED_MCP_DEPTH,
            "batch_size": FIXED_BATCH_SIZE,
            "schedule": list(FIXED_SCHEDULE),
            "local_attn_size": FIXED_LOCAL_ATTN_SIZE,
            "sink_size": FIXED_SINK_SIZE,
        },
        "torch": None,
        "cuda": None,
        "gpu_name": None,
        "compute_capability": None,
        "running_on_a100": None,
        "latent": None,
        "source_noise_identity_check": None,
        "frame_count_checks": None,
        "path_a": None,
        "path_b": None,
        "target_comparison": None,
        "following_comparison": None,
        "rollback_fingerprint_comparison": None,
        "old_tensor_inplace_mutation_check": None,
        "memory_loop": None,
        "blocking_checks": [],
        "overall_pass": False,
        "remaining_limitations": [
            "Does not solve evaluator, committer, or controller transaction ownership.",
            "Does not connect accepted candidates to production inference.",
            "Does not add ImageReward or any learned quality scorer.",
            "Does not set or validate acceptance thresholds.",
            "Does not claim visual quality equivalence.",
            "Does not claim speedup.",
        ],
    }


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return str(value)


def write_report(path: str, report: Mapping[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = json_safe(report)
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
    output.write_text(text + "\n", encoding="utf-8")


def add_blocking_check(
    report: dict[str, Any],
    name: str,
    passed: bool,
    details: Optional[Mapping[str, Any]] = None,
) -> None:
    report["blocking_checks"].append(
        {
            "name": name,
            "passed": bool(passed),
            "details": {} if details is None else dict(details),
        }
    )


def set_overall_pass(report: dict[str, Any]) -> bool:
    overall = all(bool(check.get("passed")) for check in report["blocking_checks"])
    report["overall_pass"] = bool(overall)
    return bool(overall)


def exception_payload(exc: BaseException) -> dict[str, Any]:
    return {
        "type": type(exc).__name__,
        "message": str(exc),
        "traceback": traceback.format_exception(type(exc), exc, exc.__traceback__),
    }


def config_get(value: Any, key: str, default: Any = None) -> Any:
    if value is None:
        return default
    if isinstance(value, Mapping):
        return value.get(key, default)
    get = getattr(value, "get", None)
    if callable(get):
        try:
            return get(key, default)
        except Exception:
            pass
    return getattr(value, key, default)


def validate_fixed_experiment_config(config: Any) -> None:
    block_frames = int(config_get(config, "num_frame_per_block"))
    if block_frames != FIXED_NUM_FRAME_PER_BLOCK:
        raise ValueError(
            "F4B2 requires config.num_frame_per_block == "
            f"{FIXED_NUM_FRAME_PER_BLOCK}, got {block_frames}."
        )

    model_kwargs = config_get(config, "model_kwargs", {}) or {}
    local_attn_size = int(config_get(model_kwargs, "local_attn_size", FIXED_LOCAL_ATTN_SIZE))
    sink_size = int(config_get(model_kwargs, "sink_size", FIXED_SINK_SIZE))
    if local_attn_size != FIXED_LOCAL_ATTN_SIZE:
        raise ValueError(
            "F4B2 requires local_attn_size == "
            f"{FIXED_LOCAL_ATTN_SIZE}, got {local_attn_size}."
        )
    if sink_size != FIXED_SINK_SIZE:
        raise ValueError(f"F4B2 requires sink_size == {FIXED_SINK_SIZE}, got {sink_size}.")


def block_to_dict(block: Any) -> dict[str, Optional[int]]:
    return {
        "index": int(block.index),
        "start_frame": None if block.start_frame is None else int(block.start_frame),
        "num_frames": None if block.num_frames is None else int(block.num_frames),
    }


def tensor_record(tensor: Any) -> dict[str, Any]:
    return {
        "shape": [int(dim) for dim in tensor.shape],
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
    }


def freeze_latent_cpu(latent: Any, torch: Any, label: str) -> Any:
    if not torch.is_tensor(latent):
        raise TypeError(f"{label} must be a torch.Tensor.")
    if latent.ndim != 5:
        raise ValueError(f"{label} must have rank 5 [B,F,C,H,W], got {tuple(latent.shape)}.")
    if latent.shape[0] != FIXED_BATCH_SIZE:
        raise ValueError(f"{label} batch must be {FIXED_BATCH_SIZE}, got {latent.shape[0]}.")
    if not bool(torch.isfinite(latent).all().item()):
        raise ValueError(f"{label} contains non-finite values.")
    return latent.detach().to(device="cpu").clone()


def collect_real_latents(
    *,
    args: argparse.Namespace,
    config: Any,
    model: Any,
    device: Any,
    torch: Any,
    np: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    from inference_mcp import build_rollout_pipeline, make_noise, reset_runtime_seed
    from inference_speculative import build_runtime
    from speculative.types import CommitRequest, ControlRequest

    helper_args = helper_args_from(args)
    rollout_pipeline = None
    runtime = None
    conditional_dict = None
    noise = None
    output = None

    try:
        model.generator.eval().requires_grad_(False)
        model.text_encoder.eval().requires_grad_(False)
        model.generator.to(device=device, dtype=torch.bfloat16)
        model.text_encoder.to(device=device, dtype=torch.bfloat16)

        with torch.inference_mode():
            conditional_dict = model.text_encoder(text_prompts=[args.prompt])

        rollout_pipeline = build_rollout_pipeline(config, model, helper_args)
        actual_denoising_schedule = tuple(
            int(step.item()) if hasattr(step, "item") else int(step)
            for step in rollout_pipeline.denoising_step_list
        )
        if actual_denoising_schedule != FIXED_SCHEDULE:
            raise RuntimeError(
                "rollout_pipeline.denoising_step_list must be exactly "
                f"{FIXED_SCHEDULE}, got {actual_denoising_schedule}."
            )
        if int(rollout_pipeline.num_frame_per_block) != FIXED_NUM_FRAME_PER_BLOCK:
            raise RuntimeError("rollout_pipeline.num_frame_per_block is not fixed at 3.")
        if int(rollout_pipeline.mcp_num_modules) != FIXED_MCP_DEPTH:
            raise RuntimeError("rollout_pipeline.mcp_num_modules is not fixed at 1.")
        if int(rollout_pipeline.mcp_accel_depths) != FIXED_MCP_DEPTH:
            raise RuntimeError("rollout_pipeline.mcp_accel_depths is not fixed at 1.")

        noise = make_noise(helper_args, device, torch)
        if int(noise.shape[0]) != FIXED_BATCH_SIZE:
            raise RuntimeError("source noise batch size is not fixed at 1.")

        rollout_pipeline._initialize_kv_cache(
            batch_size=FIXED_BATCH_SIZE,
            dtype=noise.dtype,
            device=noise.device,
        )
        rollout_pipeline._initialize_crossattn_cache(
            batch_size=FIXED_BATCH_SIZE,
            dtype=noise.dtype,
            device=noise.device,
        )

        output = torch.zeros_like(noise)
        runtime = build_runtime(
            rollout_pipeline=rollout_pipeline,
            conditional_dict=conditional_dict,
            noise=noise,
            output=output,
            mcp_depth=FIXED_MCP_DEPTH,
        )
        reset_runtime_seed(args.seed, np, torch)

        with torch.inference_mode():
            runtime.prepare()
            blocks = tuple(runtime.rollout_plan.blocks)
            if len(blocks) != 3:
                raise RuntimeError(f"F4B2 expects exactly 3 rollout blocks, got {len(blocks)}.")

            request_0 = ControlRequest(anchor_block=blocks[0], max_depth=FIXED_MCP_DEPTH)
            batch_0 = runtime.propose_window(request_0)
            if batch_0.anchor.block.index != 0:
                raise RuntimeError("batch_0.anchor.block.index is not 0.")
            if len(batch_0.drafts) != 1:
                raise RuntimeError(f"batch_0 must contain one draft, got {len(batch_0.drafts)}.")
            draft = batch_0.drafts[0]
            if draft.block.index != 1:
                raise RuntimeError("batch_0 draft block index is not 1.")
            if draft.depth != 1:
                raise RuntimeError("batch_0 draft depth is not 1.")

            anchor_latent = freeze_latent_cpu(batch_0.anchor.latent, torch, "anchor_latent")
            draft_latent = freeze_latent_cpu(draft.latent, torch, "draft_latent")

            runtime.begin_window()
            runtime.commit_block(batch_0.anchor)

            fallback = runtime.generate_target_fallback(draft)
            source_noise_identity = fallback.source_noise is draft.source_noise
            if fallback.block != draft.block:
                raise RuntimeError("fallback.block does not match draft.block.")
            if not source_noise_identity:
                raise RuntimeError("fallback.source_noise does not preserve draft.source_noise identity.")
            target_latent = freeze_latent_cpu(fallback.latent, torch, "target_latent")

            runtime.commit_block(
                CommitRequest(
                    block=fallback.block,
                    latent=fallback.latent,
                    source="fallback",
                    depth=draft.depth,
                    source_noise=fallback.source_noise,
                )
            )
            runtime.complete_window()

            request_2 = ControlRequest(anchor_block=blocks[2], max_depth=0)
            batch_2 = runtime.propose_window(request_2)
            if batch_2.anchor.block.index != 2:
                raise RuntimeError("batch_2.anchor.block.index is not 2.")
            if batch_2.drafts:
                raise RuntimeError("batch_2 must not contain drafts when max_depth=0.")
            following_latent = freeze_latent_cpu(
                batch_2.anchor.latent,
                torch,
                "following_latent",
            )

        report = {
            "rollout_blocks": [block_to_dict(block) for block in blocks],
            "denoising_step_list": list(actual_denoising_schedule),
            "request_0": {
                "anchor_block": block_to_dict(request_0.anchor_block),
                "max_depth": int(request_0.max_depth),
            },
            "request_2": {
                "anchor_block": block_to_dict(request_2.anchor_block),
                "max_depth": int(request_2.max_depth),
            },
            "batch_0_checks": {
                "anchor_block_index": int(batch_0.anchor.block.index),
                "draft_count": int(len(batch_0.drafts)),
                "draft_block_index": int(draft.block.index),
                "draft_depth": int(draft.depth),
            },
            "batch_2_checks": {
                "anchor_block_index": int(batch_2.anchor.block.index),
                "draft_count": int(len(batch_2.drafts)),
            },
            "source_noise_identity_check": {
                "fallback_source_noise_is_draft_source_noise": bool(source_noise_identity),
                "draft_source_noise_object_id": int(id(draft.source_noise)),
                "fallback_source_noise_object_id": int(id(fallback.source_noise)),
            },
            "frozen_latents_on_cpu": True,
        }
        latents = {
            "anchor": anchor_latent,
            "draft": draft_latent,
            "target": target_latent,
            "following": following_latent,
        }
        return latents, report
    finally:
        if runtime is not None and getattr(runtime, "has_active_window", False):
            try:
                runtime.rollback_window()
            except Exception:
                pass
        try:
            if rollout_pipeline is not None:
                rollout_pipeline.generator = None
        except Exception:
            pass
        runtime = None
        rollout_pipeline = None
        conditional_dict = None
        noise = None
        output = None
        gc.collect()


def move_non_vae_to_cpu(model: Any) -> None:
    if model is None:
        return
    if hasattr(model, "generator") and model.generator is not None:
        model.generator.to("cpu")
    if hasattr(model, "text_encoder") and model.text_encoder is not None:
        model.text_encoder.to("cpu")


def validate_pixel_tensor(pixels: Any, *, expected_frames: int, torch: Any) -> dict[str, Any]:
    is_tensor = bool(torch.is_tensor(pixels))
    shape = [int(dim) for dim in pixels.shape] if is_tensor else None
    rank = int(pixels.ndim) if is_tensor else None
    batch = int(pixels.shape[0]) if is_tensor and pixels.ndim >= 1 else None
    frames = int(pixels.shape[1]) if is_tensor and pixels.ndim >= 2 else None
    channels = int(pixels.shape[2]) if is_tensor and pixels.ndim >= 3 else None
    finite = None
    if is_tensor:
        try:
            finite = bool(torch.isfinite(pixels).all().item())
        except Exception:
            finite = False
    passed = (
        is_tensor
        and rank == 5
        and batch == FIXED_BATCH_SIZE
        and channels == 3
        and frames == int(expected_frames)
        and finite is True
    )
    return {
        "is_tensor": is_tensor,
        "shape": shape,
        "rank": rank,
        "batch": batch,
        "frames": frames,
        "channels": channels,
        "expected_frames": int(expected_frames),
        "finite": finite,
        "passed": bool(passed),
    }


def decode_cached_block(
    *,
    vae: Any,
    latent: Any,
    expected_frames: int,
    label: str,
    torch: Any,
) -> tuple[Any, dict[str, Any]]:
    pixels = vae.decode_to_pixel(latent, use_cache=True)
    check = validate_pixel_tensor(pixels, expected_frames=expected_frames, torch=torch)
    check["label"] = label
    return pixels, check


def readonly_tensor_digest(tensor: Any, torch: Any) -> str:
    value = tensor.detach().contiguous().cpu()
    digest_value = value.to(torch.float32) if value.dtype == torch.bfloat16 else value
    digest = hashlib.sha256()
    header = (
        f"shape={tuple(tensor.shape)};"
        f"dtype={tensor.dtype};"
        f"digest_dtype={digest_value.dtype};"
    )
    digest.update(header.encode("utf-8"))
    digest.update(digest_value.numpy().tobytes())
    return digest.hexdigest()


def cache_entry_digest_records(entries: Sequence[Any], torch: Any) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index, entry in enumerate(entries):
        if torch.is_tensor(entry):
            try:
                finite = bool(torch.isfinite(entry.detach()).all().item())
            except Exception:
                finite = False
            records.append(
                {
                    "index": int(index),
                    "kind": "tensor",
                    "object_id": int(id(entry)),
                    "data_ptr": int(entry.data_ptr()),
                    "shape": [int(dim) for dim in entry.shape],
                    "dtype": str(entry.dtype),
                    "device": str(entry.device),
                    "finite": finite,
                    "digest": readonly_tensor_digest(entry, torch),
                }
            )
        elif entry is None:
            records.append(
                {
                    "index": int(index),
                    "kind": "none",
                    "object_id": int(id(entry)),
                }
            )
        elif isinstance(entry, str):
            records.append(
                {
                    "index": int(index),
                    "kind": "sentinel",
                    "object_id": int(id(entry)),
                    "value": entry,
                }
            )
        else:
            records.append(
                {
                    "index": int(index),
                    "kind": type(entry).__name__,
                    "object_id": int(id(entry)),
                    "repr": repr(entry),
                }
            )
    return records


def old_tensor_mutation_report(
    before: Sequence[Mapping[str, Any]],
    during: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    before_by_index = {int(record["index"]): record for record in before}
    during_by_index = {int(record["index"]): record for record in during}
    changed: list[int] = []
    tensor_count = 0
    for index, before_record in before_by_index.items():
        if before_record.get("kind") != "tensor":
            continue
        tensor_count += 1
        during_record = during_by_index.get(index)
        if during_record is None or during_record.get("kind") != "tensor":
            changed.append(index)
            continue
        if before_record.get("digest") != during_record.get("digest"):
            changed.append(index)
    return {
        "tensor_entry_count": int(tensor_count),
        "mutated_indices": changed,
        "old_cache_tensor_inplace_mutation": bool(changed),
    }


def compare_pixels(left: Any, right: Any, *, torch: Any) -> dict[str, Any]:
    left_is_tensor = bool(torch.is_tensor(left))
    right_is_tensor = bool(torch.is_tensor(right))
    left_expected_frames = int(left.shape[1]) if left_is_tensor and left.ndim >= 2 else 0
    right_expected_frames = int(right.shape[1]) if right_is_tensor and right.ndim >= 2 else 0
    left_check = validate_pixel_tensor(left, expected_frames=left_expected_frames, torch=torch)
    right_check = validate_pixel_tensor(right, expected_frames=right_expected_frames, torch=torch)
    same_shape = bool(left_is_tensor and right_is_tensor and left.shape == right.shape)
    finite = bool(left_check["finite"] is True and right_check["finite"] is True)
    exact_equal = bool(same_shape and finite and torch.equal(left, right))

    max_abs_diff = None
    mean_abs_diff = None
    mse = None
    psnr_db = None
    if same_shape and finite:
        diff = left.detach().to(torch.float32) - right.detach().to(torch.float32)
        abs_diff = diff.abs()
        max_abs_diff = float(abs_diff.max().item())
        mean_abs_diff = float(abs_diff.mean().item())
        mse = float((diff * diff).mean().item())
        if mse > 0.0 and math.isfinite(mse):
            psnr_db = float(20.0 * math.log10(2.0) - 10.0 * math.log10(mse))
        else:
            psnr_db = None

    return {
        "shape_equal": same_shape,
        "left_shape": [int(dim) for dim in left.shape] if torch.is_tensor(left) else None,
        "right_shape": [int(dim) for dim in right.shape] if torch.is_tensor(right) else None,
        "finite": finite,
        "exact_equal": exact_equal,
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
        "mse": mse,
        "psnr_db": psnr_db,
        "psnr_peak_value": 2.0,
    }


def cache_comparison(
    *,
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    fingerprints_structurally_equal: Any,
    fingerprints_numerically_equal: Any,
    fingerprints_identity_equal: Any = None,
) -> dict[str, Any]:
    result = {
        "structural_equal": bool(fingerprints_structurally_equal(left, right)),
        "numerical_equal": bool(fingerprints_numerically_equal(left, right)),
    }
    if fingerprints_identity_equal is not None:
        result["identity_equal"] = bool(fingerprints_identity_equal(left, right))
    return result


def run_path_experiment(
    *,
    vae: Any,
    latents: Mapping[str, Any],
    device: Any,
    torch: Any,
) -> dict[str, Any]:
    from speculative.adapters.wan_vae_transaction import (
        WanVAECacheTransaction,
        fingerprint_wan_vae_cache,
        fingerprints_identity_equal,
        fingerprints_numerically_equal,
        fingerprints_structurally_equal,
    )

    path_a: dict[str, Any] = {}
    path_b: dict[str, Any] = {}

    vae.model.clear_cache()
    anchor_pixels_a, path_a_anchor_check = decode_cached_block(
        vae=vae,
        latent=latents["anchor"],
        expected_frames=EXPECTED_FIRST_PIXEL_FRAMES,
        label="path_a_anchor",
        torch=torch,
    )
    del anchor_pixels_a

    target_pixels_a, path_a_target_check = decode_cached_block(
        vae=vae,
        latent=latents["target"],
        expected_frames=EXPECTED_CACHED_PIXEL_FRAMES,
        label="path_a_target",
        torch=torch,
    )
    target_post_cache_a = fingerprint_wan_vae_cache(vae, include_digest=True)

    following_pixels_a, path_a_following_check = decode_cached_block(
        vae=vae,
        latent=latents["following"],
        expected_frames=EXPECTED_CACHED_PIXEL_FRAMES,
        label="path_a_following",
        torch=torch,
    )
    following_post_cache_a = fingerprint_wan_vae_cache(vae, include_digest=True)
    path_a.update(
        {
            "anchor": {"pixel_check": path_a_anchor_check},
            "target": {"pixel_check": path_a_target_check},
            "following": {"pixel_check": path_a_following_check},
            "target_post_cache": target_post_cache_a,
            "following_post_cache": following_post_cache_a,
        }
    )

    vae.model.clear_cache()
    anchor_pixels_b, path_b_anchor_check = decode_cached_block(
        vae=vae,
        latent=latents["anchor"],
        expected_frames=EXPECTED_FIRST_PIXEL_FRAMES,
        label="path_b_anchor",
        torch=torch,
    )
    del anchor_pixels_b
    pre_draft_cache_b = fingerprint_wan_vae_cache(vae, include_digest=True)
    old_cache_entries = tuple(vae.model._feat_map)
    old_entry_digests_before = cache_entry_digest_records(old_cache_entries, torch)

    tx = WanVAECacheTransaction(vae).begin()
    draft_pixels_b = None
    try:
        draft_pixels_b, path_b_draft_check = decode_cached_block(
            vae=vae,
            latent=latents["draft"],
            expected_frames=EXPECTED_CACHED_PIXEL_FRAMES,
            label="path_b_draft",
            torch=torch,
        )
        torch.cuda.synchronize(device)
        old_entry_digests_during = cache_entry_digest_records(old_cache_entries, torch)
    finally:
        if draft_pixels_b is not None:
            del draft_pixels_b
        if tx.is_active:
            tx.rollback()

    post_rollback_cache_b = fingerprint_wan_vae_cache(vae, include_digest=True)
    rollback_comparison = cache_comparison(
        left=pre_draft_cache_b,
        right=post_rollback_cache_b,
        fingerprints_structurally_equal=fingerprints_structurally_equal,
        fingerprints_numerically_equal=fingerprints_numerically_equal,
        fingerprints_identity_equal=fingerprints_identity_equal,
    )
    old_mutation = old_tensor_mutation_report(
        old_entry_digests_before,
        old_entry_digests_during,
    )

    target_pixels_b, path_b_target_check = decode_cached_block(
        vae=vae,
        latent=latents["target"],
        expected_frames=EXPECTED_CACHED_PIXEL_FRAMES,
        label="path_b_target",
        torch=torch,
    )
    target_post_cache_b = fingerprint_wan_vae_cache(vae, include_digest=True)

    following_pixels_b, path_b_following_check = decode_cached_block(
        vae=vae,
        latent=latents["following"],
        expected_frames=EXPECTED_CACHED_PIXEL_FRAMES,
        label="path_b_following",
        torch=torch,
    )
    following_post_cache_b = fingerprint_wan_vae_cache(vae, include_digest=True)

    target_comparison = compare_pixels(target_pixels_a, target_pixels_b, torch=torch)
    following_comparison = compare_pixels(following_pixels_a, following_pixels_b, torch=torch)

    target_cache_comparison = cache_comparison(
        left=target_post_cache_a,
        right=target_post_cache_b,
        fingerprints_structurally_equal=fingerprints_structurally_equal,
        fingerprints_numerically_equal=fingerprints_numerically_equal,
    )
    following_cache_comparison = cache_comparison(
        left=following_post_cache_a,
        right=following_post_cache_b,
        fingerprints_structurally_equal=fingerprints_structurally_equal,
        fingerprints_numerically_equal=fingerprints_numerically_equal,
    )

    path_b.update(
        {
            "anchor": {"pixel_check": path_b_anchor_check},
            "pre_draft_cache": pre_draft_cache_b,
            "draft": {"pixel_check": path_b_draft_check},
            "old_entry_digests_before": old_entry_digests_before,
            "old_entry_digests_during": old_entry_digests_during,
            "post_rollback_cache": post_rollback_cache_b,
            "target": {"pixel_check": path_b_target_check},
            "following": {"pixel_check": path_b_following_check},
            "target_post_cache": target_post_cache_b,
            "following_post_cache": following_post_cache_b,
        }
    )

    del target_pixels_a
    del following_pixels_a
    del target_pixels_b
    del following_pixels_b
    gc.collect()

    return {
        "path_a": path_a,
        "path_b": path_b,
        "target_comparison": target_comparison,
        "following_comparison": following_comparison,
        "rollback_fingerprint_comparison": rollback_comparison,
        "old_tensor_inplace_mutation_check": old_mutation,
        "target_cache_comparison": target_cache_comparison,
        "following_cache_comparison": following_cache_comparison,
    }


def memory_stats(torch: Any, device: Any) -> dict[str, int]:
    return {
        "memory_allocated": int(torch.cuda.memory_allocated(device)),
        "memory_reserved": int(torch.cuda.memory_reserved(device)),
        "max_memory_allocated": int(torch.cuda.max_memory_allocated(device)),
    }


def strictly_increasing(values: Sequence[int]) -> bool:
    return len(values) >= 2 and all(right > left for left, right in zip(values, values[1:]))


def run_memory_loop(
    *,
    vae: Any,
    latents: Mapping[str, Any],
    device: Any,
    repeat_loops: int,
    torch: Any,
) -> dict[str, Any]:
    from speculative.adapters.wan_vae_transaction import (
        WanVAECacheTransaction,
        fingerprint_wan_vae_cache,
        fingerprints_identity_equal,
        fingerprints_numerically_equal,
        fingerprints_structurally_equal,
    )

    vae.model.clear_cache()
    anchor_pixels, anchor_check = decode_cached_block(
        vae=vae,
        latent=latents["anchor"],
        expected_frames=EXPECTED_FIRST_PIXEL_FRAMES,
        label="memory_anchor",
        torch=torch,
    )
    del anchor_pixels
    context_fingerprint = fingerprint_wan_vae_cache(vae, include_digest=True)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    baseline = memory_stats(torch, device)

    warmup_records = []
    for index in range(WARMUP_LOOPS):
        tx = WanVAECacheTransaction(vae).begin()
        draft_pixels = None
        try:
            draft_pixels, draft_check = decode_cached_block(
                vae=vae,
                latent=latents["draft"],
                expected_frames=EXPECTED_CACHED_PIXEL_FRAMES,
                label=f"memory_warmup_{index}",
                torch=torch,
            )
        finally:
            if draft_pixels is not None:
                del draft_pixels
            if tx.is_active:
                tx.rollback()
        torch.cuda.synchronize(device)
        warmup_fingerprint = fingerprint_wan_vae_cache(vae, include_digest=True)
        warmup_records.append(
            {
                "index": int(index),
                "draft_pixel_check": draft_check,
                "rollback_structural_equal": bool(
                    fingerprints_structurally_equal(context_fingerprint, warmup_fingerprint)
                ),
                "rollback_numerical_equal": bool(
                    fingerprints_numerically_equal(context_fingerprint, warmup_fingerprint)
                ),
                "rollback_identity_equal": bool(
                    fingerprints_identity_equal(context_fingerprint, warmup_fingerprint)
                ),
                **memory_stats(torch, device),
            }
        )

    torch.cuda.synchronize(device)
    post_warmup_baseline = memory_stats(torch, device)

    torch.cuda.reset_peak_memory_stats(device)

    loop_records = []
    for index in range(int(repeat_loops)):
        tx = WanVAECacheTransaction(vae).begin()
        draft_pixels = None
        try:
            draft_pixels, draft_check = decode_cached_block(
                vae=vae,
                latent=latents["draft"],
                expected_frames=EXPECTED_CACHED_PIXEL_FRAMES,
                label=f"memory_loop_{index}",
                torch=torch,
            )
        finally:
            if draft_pixels is not None:
                del draft_pixels
            if tx.is_active:
                tx.rollback()
        gc.collect()
        torch.cuda.synchronize(device)

        loop_fingerprint = fingerprint_wan_vae_cache(vae, include_digest=True)
        loop_records.append(
            {
                "index": int(index),
                "draft_pixel_check": draft_check,
                "rollback_structural_equal": bool(
                    fingerprints_structurally_equal(context_fingerprint, loop_fingerprint)
                ),
                "rollback_numerical_equal": bool(
                    fingerprints_numerically_equal(context_fingerprint, loop_fingerprint)
                ),
                "rollback_identity_equal": bool(
                    fingerprints_identity_equal(context_fingerprint, loop_fingerprint)
                ),
                **memory_stats(torch, device),
            }
        )

    allocated_values = [int(record["memory_allocated"]) for record in loop_records]
    reserved_values = [int(record["memory_reserved"]) for record in loop_records]
    max_allocated_values = [int(record["max_memory_allocated"]) for record in loop_records]
    post_warmup_allocated_baseline = int(post_warmup_baseline["memory_allocated"])
    post_warmup_reserved_baseline = int(post_warmup_baseline["memory_reserved"])
    allocated_deltas = [
        value - post_warmup_allocated_baseline
        for value in allocated_values
    ]
    reserved_deltas = [
        value - post_warmup_reserved_baseline
        for value in reserved_values
    ]
    tail_allocated_deltas = allocated_deltas[-min(5, len(allocated_deltas)) :]
    tail_reserved = reserved_values[-min(5, len(reserved_values)) :]

    allocated_final_minus_post_warmup = (
        allocated_deltas[-1] if allocated_deltas else None
    )
    allocated_max_post_rollback_delta = (
        max(allocated_deltas) if allocated_deltas else None
    )
    allocated_min_post_rollback_delta = (
        min(allocated_deltas) if allocated_deltas else None
    )
    allocated_tail_range = (
        max(tail_allocated_deltas) - min(tail_allocated_deltas)
        if tail_allocated_deltas
        else None
    )
    reserved_final_minus_post_warmup = (
        reserved_deltas[-1] if reserved_deltas else None
    )
    reserved_tail_range = (
        max(tail_reserved) - min(tail_reserved)
        if tail_reserved
        else None
    )

    return {
        "warmup_loops": WARMUP_LOOPS,
        "repeat_loops": int(repeat_loops),
        "context_cache_fingerprint": context_fingerprint,
        "anchor_pixel_check": anchor_check,
        "baseline": baseline,
        "warmup_records": warmup_records,
        "post_warmup_baseline": post_warmup_baseline,
        "loop_records": loop_records,
        "post_warmup_allocated_baseline": post_warmup_allocated_baseline,
        "post_warmup_allocated_baseline_bytes": post_warmup_allocated_baseline,
        "allocated_deltas_from_post_warmup": allocated_deltas,
        "allocated_final_minus_post_warmup_baseline_bytes": (
            allocated_final_minus_post_warmup
        ),
        "allocated_max_post_rollback_delta_bytes": allocated_max_post_rollback_delta,
        "allocated_min_post_rollback_delta_bytes": allocated_min_post_rollback_delta,
        "allocated_tail_range_bytes": allocated_tail_range,
        "allocated_all_return_to_post_warmup_baseline": all(
            delta == 0 for delta in allocated_deltas
        ),
        "reserved_deltas_from_post_warmup": reserved_deltas,
        "reserved_final_minus_post_warmup_baseline_bytes": reserved_final_minus_post_warmup,
        "reserved_tail_range_bytes": reserved_tail_range,
        "reserved_memory_is_diagnostic_only": True,
        "reserved_monotonic_continuous_growth_after_warmup": bool(
            strictly_increasing(reserved_values)
        ),
        "loop_cache_structural_all_equal": all(
            bool(record["rollback_structural_equal"]) for record in loop_records
        ),
        "loop_cache_numerical_all_equal": all(
            bool(record["rollback_numerical_equal"]) for record in loop_records
        ),
        "loop_cache_identity_all_equal": all(
            bool(record["rollback_identity_equal"]) for record in loop_records
        ),
        "loop_draft_pixel_checks_all_passed": all(
            bool(record["draft_pixel_check"]["passed"]) for record in loop_records
        ),
    }


def collect_frame_count_checks(path_report: Mapping[str, Any]) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    for path_name in ("path_a", "path_b"):
        path = path_report[path_name]
        for label, payload in path.items():
            if isinstance(payload, Mapping) and "pixel_check" in payload:
                check = payload["pixel_check"]
                checks[f"{path_name}_{label}"] = {
                    "frames": check["frames"],
                    "expected_frames": check["expected_frames"],
                    "passed": bool(check["passed"]),
                }
    return checks


def add_result_blocking_checks(
    *,
    report: dict[str, Any],
    path_report: Mapping[str, Any],
    memory_report: Mapping[str, Any],
    pixel_tolerance: float,
) -> None:
    source_identity = report["source_noise_identity_check"]
    add_blocking_check(
        report,
        "source_noise_identity_preserved",
        bool(source_identity["fallback_source_noise_is_draft_source_noise"]),
        source_identity,
    )

    for check_name, check in report["frame_count_checks"].items():
        add_blocking_check(
            report,
            f"pixel_check_{check_name}",
            bool(check["passed"]),
            check,
        )

    rollback = path_report["rollback_fingerprint_comparison"]
    add_blocking_check(
        report,
        "rollback_cache_structural_equal",
        bool(rollback["structural_equal"]),
        rollback,
    )
    add_blocking_check(
        report,
        "rollback_cache_numerical_equal",
        bool(rollback["numerical_equal"]),
        rollback,
    )
    add_blocking_check(
        report,
        "rollback_cache_identity_equal",
        bool(rollback["identity_equal"]),
        rollback,
    )

    old_mutation = path_report["old_tensor_inplace_mutation_check"]
    add_blocking_check(
        report,
        "old_cache_tensor_digest_available",
        int(old_mutation["tensor_entry_count"]) > 0,
        old_mutation,
    )
    add_blocking_check(
        report,
        "old_cache_tensor_not_mutated_in_place",
        not bool(old_mutation["old_cache_tensor_inplace_mutation"]),
        old_mutation,
    )

    for label in ("target", "following"):
        comparison = path_report[f"{label}_comparison"]
        max_abs = comparison["max_abs_diff"]
        difference_within_tolerance = (
            max_abs is not None
            and bool(comparison["shape_equal"])
            and bool(comparison["finite"])
            and float(max_abs) <= float(pixel_tolerance)
        )
        add_blocking_check(
            report,
            f"{label}_pixels_within_tolerance",
            difference_within_tolerance,
            {
                "max_abs_diff": max_abs,
                "pixel_max_abs_tolerance": float(pixel_tolerance),
                "shape_equal": comparison["shape_equal"],
                "finite": comparison["finite"],
                "exact_equal": comparison["exact_equal"],
            },
        )

        cache_comparison_payload = path_report[f"{label}_cache_comparison"]
        add_blocking_check(
            report,
            f"{label}_post_cache_structural_equal_path_a_b",
            bool(cache_comparison_payload["structural_equal"]),
            cache_comparison_payload,
        )
        add_blocking_check(
            report,
            f"{label}_post_cache_numerical_equal_path_a_b",
            bool(cache_comparison_payload["numerical_equal"]),
            cache_comparison_payload,
        )

    add_blocking_check(
        report,
        "memory_loop_anchor_pixel_check",
        bool(memory_report["anchor_pixel_check"]["passed"]),
        memory_report["anchor_pixel_check"],
    )
    add_blocking_check(
        report,
        "memory_loop_draft_pixel_checks",
        bool(memory_report["loop_draft_pixel_checks_all_passed"]),
        {
            "loop_draft_pixel_checks_all_passed": memory_report[
                "loop_draft_pixel_checks_all_passed"
            ]
        },
    )
    add_blocking_check(
        report,
        "memory_loop_cache_structural_equal",
        bool(memory_report["loop_cache_structural_all_equal"]),
        {"loop_cache_structural_all_equal": memory_report["loop_cache_structural_all_equal"]},
    )
    add_blocking_check(
        report,
        "memory_loop_cache_numerical_equal",
        bool(memory_report["loop_cache_numerical_all_equal"]),
        {"loop_cache_numerical_all_equal": memory_report["loop_cache_numerical_all_equal"]},
    )
    add_blocking_check(
        report,
        "memory_loop_cache_identity_equal",
        bool(memory_report["loop_cache_identity_all_equal"]),
        {"loop_cache_identity_all_equal": memory_report["loop_cache_identity_all_equal"]},
    )
    add_blocking_check(
        report,
        "allocated_memory_returns_to_post_warmup_baseline",
        bool(memory_report["allocated_all_return_to_post_warmup_baseline"]),
        {
            "post_warmup_allocated_baseline": memory_report[
                "post_warmup_allocated_baseline"
            ],
            "allocated_deltas_from_post_warmup": memory_report[
                "allocated_deltas_from_post_warmup"
            ],
            "allocated_final_minus_post_warmup_baseline_bytes": memory_report[
                "allocated_final_minus_post_warmup_baseline_bytes"
            ],
            "allocated_max_post_rollback_delta_bytes": memory_report[
                "allocated_max_post_rollback_delta_bytes"
            ],
            "allocated_tail_range_bytes": memory_report["allocated_tail_range_bytes"],
        },
    )


def print_summary(report: Mapping[str, Any]) -> None:
    target = report.get("target_comparison") or {}
    following = report.get("following_comparison") or {}
    rollback = report.get("rollback_fingerprint_comparison") or {}
    mutation = report.get("old_tensor_inplace_mutation_check") or {}
    memory = report.get("memory_loop") or {}
    print(f"overall_pass={report.get('overall_pass')}")
    print(f"target_max_abs_diff={target.get('max_abs_diff')}")
    print(f"following_max_abs_diff={following.get('max_abs_diff')}")
    print(f"rollback_structural_equal={rollback.get('structural_equal')}")
    print(f"rollback_numerical_equal={rollback.get('numerical_equal')}")
    print(f"rollback_identity_equal={rollback.get('identity_equal')}")
    print(
        "old_cache_tensor_inplace_mutation="
        f"{mutation.get('old_cache_tensor_inplace_mutation')}"
    )
    print(
        "allocated_growth_bytes="
        f"{memory.get('allocated_final_minus_post_warmup_baseline_bytes')}"
    )
    print(f"output_json={report.get('output_json')}")


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    report = base_report(args)
    model = None
    device = None
    torch = None
    exit_code = 1

    try:
        import numpy as np
        import torch as torch_module
        from inference_mcp import (
            ANCHOR_DENOISING_STEPS,
            EXPECTED_MCP_TENSOR_COUNT,
            merge_config,
            require_single_gpu_runtime,
            reset_runtime_seed,
            validate_checkpoint_restore,
            validate_config,
        )
        from model.ode_regression import ODERegression
        from utils.checkpoint import (
            MCP_COMPLETE_STRICT_RESTORE,
            is_mcp_state_key,
            load_state_dict_allowing_mcp_mismatch,
        )

        torch = torch_module
        device = require_single_gpu_runtime(torch, args.device)
        torch.set_grad_enabled(False)
        reset_runtime_seed(args.seed, np, torch)

        report["torch"] = str(torch.__version__)
        report["cuda"] = {
            "torch_cuda_version": str(torch.version.cuda),
            "is_available": bool(torch.cuda.is_available()),
            "device": str(device),
        }
        report["gpu_name"] = torch.cuda.get_device_name(device)
        compute_capability = tuple(int(part) for part in torch.cuda.get_device_capability(device))
        report["compute_capability"] = list(compute_capability)
        report["running_on_a100"] = (
            "A100" in str(report["gpu_name"]) and compute_capability == (8, 0)
        )
        add_blocking_check(
            report,
            "running_on_real_a100",
            bool(report["running_on_a100"]),
            {
                "gpu_name": report["gpu_name"],
                "compute_capability": report["compute_capability"],
                "running_on_a100": report["running_on_a100"],
            },
        )
        if not report["running_on_a100"]:
            raise RuntimeError(
                "F4B2 GPU validation requires a real A100 with compute capability "
                f"(8, 0); got gpu_name={report['gpu_name']!r}, "
                f"compute_capability={compute_capability}."
            )

        if tuple(ANCHOR_DENOISING_STEPS) != FIXED_SCHEDULE:
            raise RuntimeError(f"Expected denoising schedule [1000], got {ANCHOR_DENOISING_STEPS}.")

        config = merge_config(args.config)
        config.generator_ckpt = args.checkpoint
        config.gradient_checkpointing = False
        helper_args = helper_args_from(args)
        validate_config(config, helper_args)
        validate_fixed_experiment_config(config)

        model = ODERegression(config, device)
        load_mode, mcp_tensor_count = validate_checkpoint_restore(
            model=model,
            load_helper=load_state_dict_allowing_mcp_mismatch,
            is_mcp_state_key=is_mcp_state_key,
            strict_mode=MCP_COMPLETE_STRICT_RESTORE,
            torch=torch,
        )
        if mcp_tensor_count != EXPECTED_MCP_TENSOR_COUNT:
            raise RuntimeError(
                f"Expected {EXPECTED_MCP_TENSOR_COUNT} MCP tensors, got {mcp_tensor_count}."
            )
        report["checkpoint_restore"] = {
            "load_mode": load_mode,
            "mcp_tensor_count": int(mcp_tensor_count),
        }

        cpu_latents, latent_collection_report = collect_real_latents(
            args=args,
            config=config,
            model=model,
            device=device,
            torch=torch,
            np=np,
        )
        report["latent_collection"] = latent_collection_report
        report["source_noise_identity_check"] = latent_collection_report[
            "source_noise_identity_check"
        ]
        report["latent"] = {
            name: tensor_record(latent)
            for name, latent in cpu_latents.items()
        }

        move_non_vae_to_cpu(model)
        gc.collect()
        torch.cuda.empty_cache()

        model.vae.eval().requires_grad_(False)
        model.vae.to(device=device, dtype=torch.bfloat16)
        latents = {
            name: latent.to(device=device, dtype=torch.bfloat16)
            for name, latent in cpu_latents.items()
        }
        report["latent_on_vae_device"] = {
            name: tensor_record(latent)
            for name, latent in latents.items()
        }

        with torch.inference_mode():
            path_report = run_path_experiment(
                vae=model.vae,
                latents=latents,
                device=device,
                torch=torch,
            )
            report["path_a"] = path_report["path_a"]
            report["path_b"] = path_report["path_b"]
            report["target_comparison"] = path_report["target_comparison"]
            report["following_comparison"] = path_report["following_comparison"]
            report["rollback_fingerprint_comparison"] = path_report[
                "rollback_fingerprint_comparison"
            ]
            report["old_tensor_inplace_mutation_check"] = path_report[
                "old_tensor_inplace_mutation_check"
            ]
            report["target_cache_comparison"] = path_report["target_cache_comparison"]
            report["following_cache_comparison"] = path_report[
                "following_cache_comparison"
            ]
            report["frame_count_checks"] = collect_frame_count_checks(path_report)

            memory_report = run_memory_loop(
                vae=model.vae,
                latents=latents,
                device=device,
                repeat_loops=int(args.repeat_loops),
                torch=torch,
            )
            report["memory_loop"] = memory_report

        add_result_blocking_checks(
            report=report,
            path_report=path_report,
            memory_report=memory_report,
            pixel_tolerance=float(args.pixel_max_abs_tolerance),
        )
        exit_code = 0 if set_overall_pass(report) else 2
    except BaseException as exc:
        if isinstance(exc, SystemExit):
            raise
        report["exception"] = exception_payload(exc)
        add_blocking_check(report, "script_completed_without_exception", False, report["exception"])
        set_overall_pass(report)
        exit_code = 1
    finally:
        try:
            if model is not None:
                move_non_vae_to_cpu(model)
                if hasattr(model, "vae") and model.vae is not None:
                    model.vae.to("cpu")
        except Exception as cleanup_exc:
            report["cleanup_warning"] = exception_payload(cleanup_exc)
        finally:
            model = None
            gc.collect()
            try:
                if torch is not None and torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

        write_report(args.output_json, report)
        print_summary(report)

    if not report.get("overall_pass"):
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
