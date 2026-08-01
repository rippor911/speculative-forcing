#!/usr/bin/env python3
"""E0211: MCP depth-1 draft + Target refinement curve.

This is an experiment-only entry point. It does not modify the frozen Target
implementation or any repository source file.

For each selected teacher sample and anchor, it fixes:
- prompt
- seed
- source noise
- Target history
- anchor
- VAE decode path

It compares:
- draft + 0 Target refinement steps
- draft + 1 Target refinement step
- draft + 2 Target refinement steps
- draft + 3 Target refinement steps
- full Target 4-step generation

The refinement bridge is:
    x_t = scheduler.add_noise(draft_x0, fixed_noise, start_t)
followed by the suffix of the official four-step Target schedule.

Outputs:
    report.json
    metrics.csv
    timing.json
    checkpoint_contract.json
    review.html
    videos/
    latents.pt
    manual_review_template.csv
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import html
import json
import math
import os
import random
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from omegaconf import OmegaConf

from model import ODERegression
from pipeline.self_forcing_training import SelfForcingTrainingPipeline


EXPERIMENT_NAME = "E0211_target_refinement_curve_v2_shared_noise"
SCHEMA_VERSION = 1

DEFAULT_TARGET_CHECKPOINT = Path("checkpoints/self_forcing_dmd.pt")
DEFAULT_MCP_CHECKPOINT = Path(
    "experiments/E0209_depth1_formal_training/mcp_depth1_best.pt"
)
DEFAULT_SAMPLE_FILES = (
    Path("experiments/E0201_teacher_rollout_smoke/teacher_sample_004.pt"),
    Path("experiments/E0201_teacher_rollout_smoke/teacher_sample_005.pt"),
)
DEFAULT_OUTPUT_DIR = Path("experiments/E0211_target_refinement_curve_v2")

DEFAULT_CONFIGS = (
    Path("configs/default_config.yaml"),
    Path("configs/self_forcing_dmd_mcp.yaml"),
)

EXPECTED_TARGET_SHA256 = (
    "a0413986d9734e02c09504e1520f5697ba6df731bb2f0f35577485e9cc8f56a3"
)

NUM_FRAMES = 21
BLOCK_FRAMES = 3
FRAME_SEQ_LENGTH = 1560
LATENT_CHANNELS = 16
LATENT_HEIGHT = 60
LATENT_WIDTH = 104
EXPECTED_WARPED_STEPS = (1000.0, 937.5, 833.3333129882812, 625.0)


@dataclass(frozen=True)
class TensorSnapshot:
    """Index-only snapshot for append-only Self-Forcing KV cache.

    E0211 only runs the next block at the current cache end. Restoring the visible
    indices is sufficient because stale values beyond the visible end are ignored
    and overwritten by the next variant. This avoids cloning the large Wan KV
    tensors.
    """

    global_end_indices: tuple[int, ...]
    local_end_indices: tuple[int, ...]


@dataclass(frozen=True)
class HistoryState:
    pipeline: SelfForcingTrainingPipeline
    draft: torch.Tensor
    mcp_flow: torch.Tensor
    target_chunk: torch.Tensor
    future_noise: torch.Tensor
    future_start: int
    anchor_block: int
    cache_snapshot: TensorSnapshot
    cpu_rng_state: torch.Tensor
    cuda_rng_state: torch.Tensor
    proposal_ms: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run E0211 MCP draft + Target refinement curve."
    )
    parser.add_argument(
        "--sample-file",
        action="append",
        dest="sample_files",
        default=None,
        help=(
            "Teacher .pt file. Repeat for multiple samples. "
            "Defaults to old visual-gate samples 004 and 005."
        ),
    )
    parser.add_argument(
        "--anchor",
        action="append",
        type=int,
        dest="anchors",
        default=None,
        help="Anchor block in [0, 3]. Repeat as needed. Defaults to 0,1,2,3.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--target-checkpoint",
        type=Path,
        default=DEFAULT_TARGET_CHECKPOINT,
    )
    parser.add_argument(
        "--mcp-checkpoint",
        type=Path,
        default=DEFAULT_MCP_CHECKPOINT,
    )
    parser.add_argument(
        "--config",
        action="append",
        dest="configs",
        default=None,
        help=(
            "YAML config path. Repeat to merge in order. Defaults to "
            "configs/default_config.yaml and configs/self_forcing_dmd_mcp.yaml."
        ),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument(
        "--timing-repeats",
        type=int,
        default=3,
        help="Repeated refinement timing runs; median is reported.",
    )
    parser.add_argument(
        "--skip-video",
        action="store_true",
        help="Run latent/timing experiment without VAE decode.",
    )
    parser.add_argument(
        "--save-every-anchor-video",
        action="store_true",
        help=(
            "Decode every requested anchor. By default, all anchors are measured "
            "but only the first requested anchor per sample is decoded."
        ),
    )
    parser.add_argument(
        "--allow-target-hash-mismatch",
        action="store_true",
        help="Diagnostic escape hatch; mismatch is still recorded in the report.",
    )
    return parser.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_output(*args: str) -> str:
    try:
        completed = subprocess.run(
            ["git", *args],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except Exception as exc:
        return f"UNAVAILABLE: {type(exc).__name__}: {exc}"
    return completed.stdout.strip()


def reset_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def finite_float(value: float) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError(f"Non-finite scalar produced: {result}")
    return result


def tensor_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    source_noise: torch.Tensor,
) -> dict[str, float]:
    pred = prediction.detach().float()
    ref = target.detach().float()
    noise = source_noise.detach().float()

    error = pred - ref
    mse = error.square().mean()
    rmse = mse.sqrt()
    target_rms = ref.square().mean().sqrt().clamp_min(1e-12)
    relative_rmse = rmse / target_rms

    pred_flat = pred.reshape(-1)
    ref_flat = ref.reshape(-1)
    cosine = torch.nn.functional.cosine_similarity(
        pred_flat.unsqueeze(0),
        ref_flat.unsqueeze(0),
        dim=1,
        eps=1e-12,
    )[0]

    predicted_flow = noise - pred
    target_flow = noise - ref
    flow_cosine = torch.nn.functional.cosine_similarity(
        predicted_flow.reshape(1, -1),
        target_flow.reshape(1, -1),
        dim=1,
        eps=1e-12,
    )[0]
    flow_norm_ratio = (
        predicted_flow.norm() / target_flow.norm().clamp_min(1e-12)
    )

    return {
        "latent_mse": finite_float(mse.item()),
        "latent_rmse": finite_float(rmse.item()),
        "relative_rmse": finite_float(relative_rmse.item()),
        "latent_cosine": finite_float(cosine.item()),
        "flow_cosine": finite_float(flow_cosine.item()),
        "flow_norm_ratio": finite_float(flow_norm_ratio.item()),
        "prediction_mean": finite_float(pred.mean().item()),
        "prediction_std": finite_float(pred.std(unbiased=False).item()),
        "target_mean": finite_float(ref.mean().item()),
        "target_std": finite_float(ref.std(unbiased=False).item()),
    }


def normalize_state_key(name: str) -> str:
    prefixes = (
        "module.",
        "_orig_mod.",
        "_checkpoint_wrapped_module.",
        "generator.",
        "model.generator.",
    )
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if name.startswith(prefix):
                name = name[len(prefix):]
                changed = True
    if name.startswith("model.mcp."):
        name = name[len("model."):]
    return name


def tensor_mapping_candidates(payload: Any) -> list[tuple[str, Mapping[str, Any]]]:
    candidates: list[tuple[str, Mapping[str, Any]]] = []
    if isinstance(payload, Mapping):
        candidates.append(("root", payload))
        for key in (
            "generator",
            "model",
            "state_dict",
            "mcp",
            "mcp_state",
            "mcp_state_dict",
            "generator_state_dict",
            "depth1_state",
            "mcp_depth1_state",
            "trainable_state",
        ):
            value = payload.get(key)
            if isinstance(value, Mapping):
                candidates.append((key, value))
    return candidates


def possible_model_keys(raw_name: str, source_name: str) -> tuple[str, ...]:
    """Map common full-generator, MCP-stack, and depth-1-only save layouts."""

    normalized = normalize_state_key(raw_name)
    candidates = [normalized]

    if normalized.startswith("mcp_modules.") or normalized.startswith("fusion."):
        candidates.append("mcp." + normalized)

    if source_name in {"mcp_state", "mcp_state_dict"}:
        candidates.append("mcp." + normalized)

    if source_name in {"depth1_state", "mcp_depth1_state", "trainable_state"}:
        candidates.append("mcp.mcp_modules.0." + normalized)

    # Some depth-1 checkpoints store keys as mcp_modules.0.* without the
    # surrounding MCPStack prefix.
    if normalized.startswith("mcp_modules.0."):
        suffix = normalized[len("mcp_modules.0."):]
        candidates.append("mcp.mcp_modules.0." + suffix)

    return tuple(dict.fromkeys(candidates))


def load_mcp_weights(
    generator: torch.nn.Module,
    checkpoint_path: Path,
) -> dict[str, Any]:
    payload = torch.load(checkpoint_path, map_location="cpu")

    model_state = generator.state_dict()
    selected: dict[str, torch.Tensor] = {}
    selected_source = None

    for source_name, mapping in tensor_mapping_candidates(payload):
        current: dict[str, torch.Tensor] = {}
        for raw_name, value in mapping.items():
            if not isinstance(raw_name, str) or not torch.is_tensor(value):
                continue
            for name in possible_model_keys(raw_name, source_name):
                if not name.startswith("mcp."):
                    continue
                if (
                    name in model_state
                    and tuple(value.shape) == tuple(model_state[name].shape)
                ):
                    current[name] = value
                    break
        if len(current) > len(selected):
            selected = current
            selected_source = source_name

    if not selected:
        raise RuntimeError(
            "No shape-compatible mcp.* tensors were found in "
            f"{checkpoint_path}. Inspect the checkpoint layout before changing code."
        )

    missing, unexpected = generator.load_state_dict(selected, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected MCP keys during restore: {unexpected}")

    depth1_keys = [
        name for name in selected if name.startswith("mcp.mcp_modules.0.")
    ]
    if not depth1_keys:
        raise RuntimeError(
            "Checkpoint restore found MCP tensors, but none under "
            "mcp.mcp_modules.0.*."
        )

    with torch.no_grad():
        max_restore_diff = 0.0
        restored = generator.state_dict()
        for name, expected in selected.items():
            difference = (
                restored[name].detach().cpu().float()
                - expected.detach().cpu().float()
            ).abs().max().item()
            max_restore_diff = max(max_restore_diff, float(difference))

    return {
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "selected_source": selected_source,
        "loaded_mcp_tensor_count": len(selected),
        "loaded_depth1_tensor_count": len(depth1_keys),
        "max_restore_abs_diff": finite_float(max_restore_diff),
        "missing_key_count_after_partial_restore": len(missing),
        "unexpected_keys": list(unexpected),
    }


def validate_teacher_payload(payload: Mapping[str, Any], path: Path) -> None:
    required = {
        "prompt",
        "seed",
        "source_noise",
        "target_latent",
        "valid_anchor_blocks",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise RuntimeError(f"{path} misses teacher fields: {missing}")

    source_noise = payload["source_noise"]
    target_latent = payload["target_latent"]
    if not torch.is_tensor(source_noise) or not torch.is_tensor(target_latent):
        raise TypeError(f"{path}: source_noise and target_latent must be tensors.")
    expected_shape = (1, NUM_FRAMES, LATENT_CHANNELS, LATENT_HEIGHT, LATENT_WIDTH)
    if tuple(source_noise.shape) != expected_shape:
        raise RuntimeError(
            f"{path}: source_noise shape {tuple(source_noise.shape)} != {expected_shape}"
        )
    if tuple(target_latent.shape) != expected_shape:
        raise RuntimeError(
            f"{path}: target_latent shape {tuple(target_latent.shape)} != {expected_shape}"
        )
    if not bool(torch.isfinite(source_noise.float()).all()):
        raise RuntimeError(f"{path}: source_noise contains NaN/Inf.")
    if not bool(torch.isfinite(target_latent.float()).all()):
        raise RuntimeError(f"{path}: target_latent contains NaN/Inf.")


def resolve_steps(payload: Mapping[str, Any]) -> tuple[float, ...]:
    values = payload.get("warped_denoising_steps", EXPECTED_WARPED_STEPS)
    steps = tuple(float(value) for value in values)
    if len(steps) != 4:
        raise RuntimeError(f"E0211 requires four Target steps, got {steps}.")
    return steps


def timestep_tensor(
    batch_size: int,
    num_frames: int,
    timestep: float,
    device: torch.device,
) -> torch.Tensor:
    # This deliberately mirrors SelfForcingTrainingPipeline, whose timestep
    # tensors are int64. Fractional warped values are therefore truncated.
    return torch.full(
        (batch_size, num_frames),
        int(float(timestep)),
        device=device,
        dtype=torch.int64,
    )


def add_noise(
    scheduler: Any,
    x0: torch.Tensor,
    noise: torch.Tensor,
    timestep: float,
) -> torch.Tensor:
    if x0.shape != noise.shape:
        raise RuntimeError(
            f"add_noise shape mismatch: x0={tuple(x0.shape)} noise={tuple(noise.shape)}"
        )
    batch_size, num_frames = x0.shape[:2]
    timestep_flat = torch.full(
        (batch_size * num_frames,),
        int(float(timestep)),
        device=x0.device,
        dtype=torch.long,
    )
    return scheduler.add_noise(
        x0.flatten(0, 1),
        noise.flatten(0, 1),
        timestep_flat,
    ).unflatten(0, x0.shape[:2])


def make_pipeline(
    generator: torch.nn.Module,
    steps: Sequence[float],
) -> SelfForcingTrainingPipeline:
    pipeline = SelfForcingTrainingPipeline(
        denoising_step_list=list(steps),
        scheduler=generator.get_scheduler(),
        generator=generator,
        num_frame_per_block=BLOCK_FRAMES,
        independent_first_frame=False,
        same_step_across_blocks=False,
        last_step_only=True,
        num_max_frames=NUM_FRAMES,
        context_noise=0,
        memory_gap_blocks=0,
        mcp_num_modules=3,
        mcp_accel_depths=1,
    )
    return pipeline


def generator_forward(
    *,
    generator: torch.nn.Module,
    noisy_input: torch.Tensor,
    conditional_dict: Mapping[str, Any],
    timestep: float,
    kv_cache: Any,
    crossattn_cache: Any,
    start_frame: int,
    mcp_future_noises: Sequence[torch.Tensor | None] | None = None,
    mcp_future_starts: Sequence[int | None] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor] | None]:
    current_timestep = timestep_tensor(
        noisy_input.shape[0],
        noisy_input.shape[1],
        timestep,
        noisy_input.device,
    )
    kwargs: dict[str, Any] = {}
    if mcp_future_noises is not None:
        kwargs["mcp_future_noises"] = list(mcp_future_noises)
        kwargs["mcp_future_start_frames"] = list(mcp_future_starts or ())

    output = generator(
        noisy_image_or_video=noisy_input,
        conditional_dict=dict(conditional_dict),
        timestep=current_timestep,
        kv_cache=kv_cache,
        crossattn_cache=crossattn_cache,
        current_start=start_frame * FRAME_SEQ_LENGTH,
        **kwargs,
    )

    if not isinstance(output, tuple):
        raise RuntimeError(f"Generator returned {type(output).__name__}, expected tuple.")
    if len(output) == 2:
        flow, x0 = output
        return flow, x0, None
    if len(output) == 3:
        flow, x0, mcp_flows = output
        return flow, x0, list(mcp_flows)
    raise RuntimeError(f"Unexpected generator output length: {len(output)}")


def commit_context(
    *,
    pipeline: SelfForcingTrainingPipeline,
    latent: torch.Tensor,
    conditional_dict: Mapping[str, Any],
    start_frame: int,
) -> None:
    batch_size, num_frames = latent.shape[:2]
    context_timestep = timestep_tensor(
        batch_size,
        num_frames,
        0.0,
        latent.device,
    )
    context_latent = add_noise(
        pipeline.scheduler,
        latent,
        torch.randn_like(latent),
        0.0,
    )
    with torch.no_grad():
        pipeline.generator(
            noisy_image_or_video=context_latent,
            conditional_dict=dict(conditional_dict),
            timestep=context_timestep,
            kv_cache=pipeline.kv_cache1,
            crossattn_cache=pipeline.crossattn_cache,
            current_start=start_frame * FRAME_SEQ_LENGTH,
        )


def run_target_schedule(
    *,
    pipeline: SelfForcingTrainingPipeline,
    conditional_dict: Mapping[str, Any],
    initial_noisy: torch.Tensor,
    start_frame: int,
    steps: Sequence[float],
    transition_noises: Sequence[torch.Tensor] | None = None,
) -> torch.Tensor:
    if not steps:
        raise ValueError("Target schedule cannot be empty.")
    noisy_input = initial_noisy
    if transition_noises is not None and len(transition_noises) != len(steps) - 1:
        raise ValueError(
            "transition_noises must contain one tensor between adjacent steps."
        )

    for index, current_timestep in enumerate(steps):
        with torch.no_grad():
            _, x0, _ = generator_forward(
                generator=pipeline.generator,
                noisy_input=noisy_input,
                conditional_dict=conditional_dict,
                timestep=current_timestep,
                kv_cache=pipeline.kv_cache1,
                crossattn_cache=pipeline.crossattn_cache,
                start_frame=start_frame,
            )
        if index + 1 < len(steps):
            next_timestep = steps[index + 1]
            transition_noise = (
                torch.randn_like(x0)
                if transition_noises is None
                else transition_noises[index]
            )
            noisy_input = add_noise(
                pipeline.scheduler,
                x0,
                transition_noise,
                next_timestep,
            )
    return x0


def capture_kv_snapshot(kv_cache: Sequence[Mapping[str, Any]]) -> TensorSnapshot:
    return TensorSnapshot(
        global_end_indices=tuple(
            int(layer["global_end_index"].item()) for layer in kv_cache
        ),
        local_end_indices=tuple(
            int(layer["local_end_index"].item()) for layer in kv_cache
        ),
    )


def restore_kv_snapshot(
    kv_cache: Sequence[Mapping[str, Any]],
    snapshot: TensorSnapshot,
) -> None:
    if len(kv_cache) != len(snapshot.global_end_indices):
        raise RuntimeError("KV layer count changed after snapshot.")
    with torch.no_grad():
        for index, layer in enumerate(kv_cache):
            layer["global_end_index"].fill_(snapshot.global_end_indices[index])
            layer["local_end_index"].fill_(snapshot.local_end_indices[index])


def restore_rng(
    cpu_state: torch.Tensor,
    cuda_state: torch.Tensor,
    device: torch.device,
) -> None:
    torch.random.set_rng_state(cpu_state)
    if device.type == "cuda":
        torch.cuda.set_rng_state(cuda_state, device=device)


def replay_target_history_and_make_draft(
    *,
    generator: torch.nn.Module,
    payload: Mapping[str, Any],
    conditional_dict: Mapping[str, Any],
    anchor_block: int,
    steps: Sequence[float],
    device: torch.device,
    dtype: torch.dtype,
) -> HistoryState:
    source_noise = payload["source_noise"].to(device=device, dtype=dtype)
    target_latent = payload["target_latent"].to(device=device, dtype=dtype)
    seed = int(payload["seed"])

    pipeline = make_pipeline(generator, steps)
    pipeline._initialize_kv_cache(  # noqa: SLF001 - experiment mirrors frozen runtime
        batch_size=source_noise.shape[0],
        dtype=source_noise.dtype,
        device=source_noise.device,
    )
    pipeline._initialize_crossattn_cache(  # noqa: SLF001
        batch_size=source_noise.shape[0],
        dtype=source_noise.dtype,
        device=source_noise.device,
    )

    reset_seed(seed)
    proposal_ms = 0.0
    draft = None
    mcp_flow = None

    for block_index in range(anchor_block + 1):
        start = block_index * BLOCK_FRAMES
        end = start + BLOCK_FRAMES
        noisy_input = source_noise[:, start:end]

        for step_index, current_timestep in enumerate(steps):
            is_last = step_index == len(steps) - 1
            use_mcp = block_index == anchor_block and is_last

            if use_mcp:
                future_start = end
                future_end = future_start + BLOCK_FRAMES
                future_noise = source_noise[:, future_start:future_end]

                synchronize(device)
                started = time.perf_counter()
                with torch.no_grad():
                    _, x0, mcp_flows = generator_forward(
                        generator=generator,
                        noisy_input=noisy_input,
                        conditional_dict=conditional_dict,
                        timestep=current_timestep,
                        kv_cache=pipeline.kv_cache1,
                        crossattn_cache=pipeline.crossattn_cache,
                        start_frame=start,
                        mcp_future_noises=(future_noise, None, None),
                        mcp_future_starts=(future_start, None, None),
                    )
                synchronize(device)
                proposal_ms = (time.perf_counter() - started) * 1000.0

                if not mcp_flows:
                    raise RuntimeError("Depth-1 MCP forward returned no MCP flow.")
                mcp_flow = mcp_flows[0]
                draft = future_noise - mcp_flow
            else:
                with torch.no_grad():
                    _, x0, _ = generator_forward(
                        generator=generator,
                        noisy_input=noisy_input,
                        conditional_dict=conditional_dict,
                        timestep=current_timestep,
                        kv_cache=pipeline.kv_cache1,
                        crossattn_cache=pipeline.crossattn_cache,
                        start_frame=start,
                    )

            if not is_last:
                noisy_input = add_noise(
                    pipeline.scheduler,
                    x0,
                    torch.randn_like(x0),
                    steps[step_index + 1],
                )

        commit_context(
            pipeline=pipeline,
            latent=x0,
            conditional_dict=conditional_dict,
            start_frame=start,
        )

    if draft is None or mcp_flow is None:
        raise RuntimeError("Internal error: draft was not produced.")

    future_start = (anchor_block + 1) * BLOCK_FRAMES
    future_end = future_start + BLOCK_FRAMES
    target_chunk = target_latent[:, future_start:future_end]
    future_noise = source_noise[:, future_start:future_end]

    expected_cache_end = future_start * FRAME_SEQ_LENGTH
    observed_global = {
        int(layer["global_end_index"].item()) for layer in pipeline.kv_cache1
    }
    observed_local = {
        int(layer["local_end_index"].item()) for layer in pipeline.kv_cache1
    }
    if observed_global != {expected_cache_end} or observed_local != {expected_cache_end}:
        raise RuntimeError(
            "Target-history cache does not end exactly at the future block: "
            f"expected={expected_cache_end}, global={sorted(observed_global)}, "
            f"local={sorted(observed_local)}"
        )

    snapshot = capture_kv_snapshot(pipeline.kv_cache1)
    cpu_state = torch.random.get_rng_state().clone()
    cuda_state = (
        torch.cuda.get_rng_state(device=device).clone()
        if device.type == "cuda"
        else torch.empty(0, dtype=torch.uint8)
    )

    return HistoryState(
        pipeline=pipeline,
        draft=draft.detach(),
        mcp_flow=mcp_flow.detach(),
        target_chunk=target_chunk.detach(),
        future_noise=future_noise.detach(),
        future_start=future_start,
        anchor_block=anchor_block,
        cache_snapshot=snapshot,
        cpu_rng_state=cpu_state,
        cuda_rng_state=cuda_state,
        proposal_ms=finite_float(proposal_ms),
    )


def deterministic_noise_like(
    tensor: torch.Tensor,
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator(device=tensor.device)
    generator.manual_seed(int(seed))
    return torch.randn(
        tensor.shape,
        generator=generator,
        device=tensor.device,
        dtype=tensor.dtype,
    )


def run_refinement_variant(
    *,
    history: HistoryState,
    conditional_dict: Mapping[str, Any],
    num_refinement_steps: int,
    full_steps: Sequence[float],
    shared_transition_noises: Sequence[torch.Tensor],
    timing_repeats: int,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Refine Draft using a suffix of one shared Target trajectory.

    For four Target steps [t0, t1, t2, t3], shared_transition_noises are
    [z0, z1, z2], where zi is used to form the state at t{i+1}.

    Therefore:
      r3 bridges with z0 and continues with z1,z2;
      r2 bridges with z1 and continues with z2;
      r1 bridges with z2.

    All variants consequently share the same suffix randomness and are directly
    comparable to the controlled full-Target baseline.
    """
    if num_refinement_steps not in (1, 2, 3):
        raise ValueError("num_refinement_steps must be 1, 2, or 3.")
    if len(shared_transition_noises) != len(full_steps) - 1:
        raise ValueError(
            "shared_transition_noises must have len(full_steps)-1 entries."
        )

    suffix = tuple(full_steps[-num_refinement_steps:])
    start_timestep = suffix[0]
    bridge_index = len(full_steps) - num_refinement_steps - 1
    bridge_noise = shared_transition_noises[bridge_index]
    transition_noises = list(shared_transition_noises[bridge_index + 1:])

    if len(transition_noises) != num_refinement_steps - 1:
        raise RuntimeError("Internal shared-noise suffix indexing error.")

    timings: list[float] = []
    output = None

    for _ in range(max(1, timing_repeats)):
        restore_kv_snapshot(history.pipeline.kv_cache1, history.cache_snapshot)
        restore_rng(history.cpu_rng_state, history.cuda_rng_state, device)

        synchronize(device)
        started = time.perf_counter()

        noisy = add_noise(
            history.pipeline.scheduler,
            history.draft,
            bridge_noise,
            start_timestep,
        )
        output = run_target_schedule(
            pipeline=history.pipeline,
            conditional_dict=conditional_dict,
            initial_noisy=noisy,
            start_frame=history.future_start,
            steps=suffix,
            transition_noises=transition_noises,
        )

        synchronize(device)
        timings.append((time.perf_counter() - started) * 1000.0)

    assert output is not None
    return output.detach(), {
        "num_target_refinement_steps": num_refinement_steps,
        "start_timestep": finite_float(start_timestep),
        "bridge_noise_index": bridge_index,
        "shared_suffix_noise_indices": list(
            range(bridge_index + 1, len(shared_transition_noises))
        ),
        "bridge_mode": "shared_noise_renoise_then_target_suffix",
        "timing_repeats_ms": [finite_float(value) for value in timings],
        "wall_clock_ms_median": finite_float(statistics.median(timings)),
        "wall_clock_ms_min": finite_float(min(timings)),
        "wall_clock_ms_max": finite_float(max(timings)),
    }

def run_full_target_variant(
    *,
    history: HistoryState,
    conditional_dict: Mapping[str, Any],
    full_steps: Sequence[float],
    shared_transition_noises: Sequence[torch.Tensor],
    timing_repeats: int,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Run the controlled four-step Target baseline with shared noises."""
    if len(shared_transition_noises) != len(full_steps) - 1:
        raise ValueError(
            "shared_transition_noises must have len(full_steps)-1 entries."
        )

    timings: list[float] = []
    output = None

    for _ in range(max(1, timing_repeats)):
        restore_kv_snapshot(history.pipeline.kv_cache1, history.cache_snapshot)
        restore_rng(history.cpu_rng_state, history.cuda_rng_state, device)

        synchronize(device)
        started = time.perf_counter()
        output = run_target_schedule(
            pipeline=history.pipeline,
            conditional_dict=conditional_dict,
            initial_noisy=history.future_noise,
            start_frame=history.future_start,
            steps=full_steps,
            transition_noises=shared_transition_noises,
        )
        synchronize(device)
        timings.append((time.perf_counter() - started) * 1000.0)

    assert output is not None
    return output.detach(), {
        "num_target_steps": len(full_steps),
        "trajectory": "controlled_shared_noise",
        "timing_repeats_ms": [finite_float(value) for value in timings],
        "wall_clock_ms_median": finite_float(statistics.median(timings)),
        "wall_clock_ms_min": finite_float(min(timings)),
        "wall_clock_ms_max": finite_float(max(timings)),
    }


def run_teacher_rng_replay(
    *,
    history: HistoryState,
    conditional_dict: Mapping[str, Any],
    full_steps: Sequence[float],
    device: torch.device,
) -> torch.Tensor:
    """Attempt exact replay of the old teacher writer from the saved RNG state.

    This is only a parity diagnostic. The controlled curve does not depend on
    parity with the old stored teacher sample.
    """
    restore_kv_snapshot(history.pipeline.kv_cache1, history.cache_snapshot)
    restore_rng(history.cpu_rng_state, history.cuda_rng_state, device)
    output = run_target_schedule(
        pipeline=history.pipeline,
        conditional_dict=conditional_dict,
        initial_noisy=history.future_noise,
        start_frame=history.future_start,
        steps=full_steps,
        transition_noises=None,
    )
    return output.detach()

def splice_variant(
    target_latent: torch.Tensor,
    replacement: torch.Tensor,
    start: int,
) -> torch.Tensor:
    result = target_latent.detach().clone()
    end = start + replacement.shape[1]
    result[:, start:end] = replacement.to(
        device=result.device,
        dtype=result.dtype,
    )
    return result


def normalize_pixels(pixels: Any) -> torch.Tensor:
    if isinstance(pixels, (list, tuple)):
        if len(pixels) != 1:
            raise RuntimeError(
                f"Expected one decoded sample, got list length {len(pixels)}."
            )
        pixels = pixels[0]
    if not torch.is_tensor(pixels):
        raise TypeError(f"VAE returned {type(pixels).__name__}, expected tensor.")

    value = pixels.detach().float().cpu()
    if value.ndim == 5:
        if value.shape[0] != 1:
            raise RuntimeError(f"Expected batch 1 pixels, got {tuple(value.shape)}.")
        value = value[0]
    if value.ndim != 4:
        raise RuntimeError(f"Expected 4D decoded sample, got {tuple(value.shape)}.")

    # Convert to [T, C, H, W].
    if value.shape[1] == 3:
        frames = value
    elif value.shape[0] == 3:
        frames = value.permute(1, 0, 2, 3)
    elif value.shape[-1] == 3:
        frames = value.permute(0, 3, 1, 2)
    else:
        raise RuntimeError(
            f"Cannot identify RGB channel in decoded shape {tuple(value.shape)}."
        )

    minimum = float(frames.min().item())
    maximum = float(frames.max().item())
    if minimum < -0.05:
        frames = (frames + 1.0) / 2.0
    elif maximum > 1.5:
        frames = frames / 255.0
    return frames.clamp(0.0, 1.0)


def save_video(path: Path, frames_tchw: torch.Tensor, fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames_uint8 = (
        frames_tchw.permute(0, 2, 3, 1)
        .mul(255.0)
        .round()
        .clamp(0, 255)
        .to(torch.uint8)
    )
    errors: list[str] = []

    try:
        from torchvision.io import write_video

        write_video(
            str(path),
            frames_uint8,
            fps=float(fps),
            video_codec="libx264",
            options={"crf": "18"},
        )
        return
    except Exception as exc:
        errors.append(f"torchvision: {type(exc).__name__}: {exc}")

    try:
        import imageio.v3 as iio

        iio.imwrite(
            path,
            frames_uint8.numpy(),
            fps=fps,
            codec="libx264",
        )
        return
    except Exception as exc:
        errors.append(f"imageio: {type(exc).__name__}: {exc}")

    raise RuntimeError(
        f"Could not save {path}. Backends failed: " + " | ".join(errors)
    )


def block_pixel_span(block_index: int, total_pixel_frames: int) -> tuple[int, int]:
    # Wan causal VAE temporal compression: 3 latent frames decode to 9 pixels
    # for block 0; each later 3-latent block contributes 12 pixels.
    if block_index == 0:
        start, end = 0, 9
    else:
        start = 12 * block_index - 3
        end = 12 * block_index + 9
    if start < 0 or end > total_pixel_frames or start >= end:
        raise RuntimeError(
            "Causal VAE block/pixel mapping is incompatible with decoded length: "
            f"block={block_index}, span=({start},{end}), total={total_pixel_frames}"
        )
    return start, end


def decode_and_save_variants(
    *,
    model: ODERegression,
    payload: Mapping[str, Any],
    sample_label: str,
    anchor_block: int,
    variants: Mapping[str, torch.Tensor],
    output_dir: Path,
    device: torch.device,
    dtype: torch.dtype,
    fps: int,
) -> list[dict[str, Any]]:
    target_latent = payload["target_latent"].to(device=device, dtype=dtype)
    future_block = anchor_block + 1
    future_start = future_block * BLOCK_FRAMES

    records: list[dict[str, Any]] = []
    model.vae.eval().requires_grad_(False)
    model.vae.to(device=device, dtype=dtype)

    for variant_name, chunk in variants.items():
        full_latent = splice_variant(target_latent, chunk, future_start)
        with torch.no_grad():
            decoded = model.vae.decode_to_pixel(full_latent, use_cache=False)
        frames = normalize_pixels(decoded)

        stem = f"{sample_label}_anchor{anchor_block}_{variant_name}"
        full_path = output_dir / "videos" / f"{stem}_full.mp4"
        save_video(full_path, frames, fps)

        start_pixel, end_pixel = block_pixel_span(future_block, frames.shape[0])
        block_path = output_dir / "videos" / f"{stem}_block.mp4"
        save_video(block_path, frames[start_pixel:end_pixel], fps)

        records.append(
            {
                "variant": variant_name,
                "full_video": str(full_path.relative_to(output_dir)),
                "block_video": str(block_path.relative_to(output_dir)),
                "decoded_pixel_frames": int(frames.shape[0]),
                "block_pixel_start": start_pixel,
                "block_pixel_end": end_pixel,
            }
        )

        del decoded, frames, full_latent
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    model.vae.to("cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return records


def make_review_html(
    *,
    output_dir: Path,
    records: Sequence[Mapping[str, Any]],
    video_records: Sequence[Mapping[str, Any]],
) -> None:
    video_lookup: dict[tuple[str, int, str], Mapping[str, Any]] = {}
    for item in video_records:
        video_lookup[
            (
                str(item["sample_label"]),
                int(item["anchor_block"]),
                str(item["variant"]),
            )
        ] = item

    rows = []
    for record in records:
        key = (
            str(record["sample_label"]),
            int(record["anchor_block"]),
            str(record["variant"]),
        )
        video = video_lookup.get(key)
        links = "not decoded"
        if video is not None:
            links = (
                f'<video controls preload="metadata" width="360" '
                f'src="{html.escape(str(video["block_video"]))}"></video>'
                f'<br><a href="{html.escape(str(video["full_video"]))}">full video</a>'
            )
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(record['sample_label']))}</td>"
            f"<td>{record['anchor_block']}</td>"
            f"<td>{html.escape(str(record['variant']))}</td>"
            f"<td>{record['num_target_steps']}</td>"
            f"<td>{record['latent_mse']:.6f}</td>"
            f"<td>{record['flow_cosine']:.6f}</td>"
            f"<td>{record['flow_norm_ratio']:.6f}</td>"
            f"<td>{record.get('wall_clock_ms_median', 0.0):.2f}</td>"
            f"<td>{record.get('measured_step_speedup', 0.0):.3f}×</td>"
            f"<td>{links}</td>"
            "<td>□ clear / □ blurry</td>"
            "<td>□ kept / □ lost</td>"
            "<td>□ stable / □ flicker</td>"
            "</tr>"
        )

    document = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>E0211 Target refinement review</title>
<style>
body {{ font-family: sans-serif; margin: 24px; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #bbb; padding: 8px; vertical-align: top; }}
th {{ position: sticky; top: 0; background: #eee; }}
code {{ background: #f2f2f2; padding: 2px 4px; }}
.note {{ max-width: 1100px; line-height: 1.6; }}
</style>
</head>
<body>
<h1>E0211：MCP Draft + Target refinement</h1>
<div class="note">
<p>所有视频仅替换目标 future block；其余 block 使用同一条教师 Target latent。</p>
<p>人工验收重点：最差帧清晰度、主体是否保留、替换边界是否闪烁。
请把结论填写到 <code>manual_review_template.csv</code>。</p>
</div>
<table>
<thead>
<tr>
<th>sample</th><th>anchor</th><th>variant</th><th>Target steps</th>
<th>MSE</th><th>flow cosine</th><th>flow norm ratio</th>
<th>wall-clock ms</th><th>step speedup</th><th>video</th>
<th>worst frame</th><th>subject</th><th>boundary</th>
</tr>
</thead>
<tbody>
{''.join(rows)}
</tbody>
</table>
</body>
</html>
"""
    (output_dir / "review.html").write_text(document, encoding="utf-8")


def write_metrics_csv(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    fields = [
        "sample_label",
        "sample_file",
        "sample_index",
        "anchor_block",
        "future_block",
        "variant",
        "num_target_steps",
        "latent_mse",
        "latent_rmse",
        "relative_rmse",
        "latent_cosine",
        "flow_cosine",
        "flow_norm_ratio",
        "prediction_mean",
        "prediction_std",
        "target_mean",
        "target_std",
        "proposal_ms",
        "wall_clock_ms_median",
        "full_target_wall_clock_ms_median",
        "measured_step_speedup",
        "worst_frame_quality_manual",
        "subject_retained_manual",
        "boundary_flicker_manual",
        "manual_notes",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({field: record.get(field, "") for field in fields})


def write_manual_template(
    path: Path,
    records: Sequence[Mapping[str, Any]],
) -> None:
    fields = [
        "sample_label",
        "anchor_block",
        "variant",
        "worst_frame_quality",
        "subject_retained",
        "boundary_flicker",
        "pass_visual_gate",
        "notes",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "sample_label": record["sample_label"],
                    "anchor_block": record["anchor_block"],
                    "variant": record["variant"],
                }
            )


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("E0211 requires CUDA.")
    device = torch.device(args.device)
    dtype = torch.bfloat16

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "videos").mkdir(exist_ok=True)

    sample_files = (
        tuple(Path(value) for value in args.sample_files)
        if args.sample_files
        else DEFAULT_SAMPLE_FILES
    )
    anchors = tuple(args.anchors) if args.anchors else (0, 1, 2, 3)
    if not anchors:
        raise ValueError("At least one anchor is required.")
    if any(anchor not in (0, 1, 2, 3) for anchor in anchors):
        raise ValueError(f"Anchors must be within 0..3, got {anchors}.")
    if args.timing_repeats <= 0:
        raise ValueError("--timing-repeats must be positive.")

    config_paths = (
        tuple(Path(value) for value in args.configs)
        if args.configs
        else DEFAULT_CONFIGS
    )
    required_paths = [
        *sample_files,
        *config_paths,
        args.target_checkpoint,
        args.mcp_checkpoint,
    ]
    missing = [str(path) for path in required_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Required E0211 inputs are missing:\n  " + "\n  ".join(missing)
        )

    target_hash = sha256_file(args.target_checkpoint)
    target_hash_matches = target_hash == EXPECTED_TARGET_SHA256
    if not target_hash_matches and not args.allow_target_hash_mismatch:
        raise RuntimeError(
            "Official Target checkpoint hash mismatch.\n"
            f"expected={EXPECTED_TARGET_SHA256}\nactual={target_hash}\n"
            "Do not continue unless the checkpoint change is intentional."
        )

    payloads: list[tuple[Path, Mapping[str, Any]]] = []
    for path in sample_files:
        payload = torch.load(path, map_location="cpu")
        if not isinstance(payload, Mapping):
            raise TypeError(f"{path} must contain a mapping.")
        validate_teacher_payload(payload, path)
        payloads.append((path.resolve(), payload))

    merged_config = OmegaConf.merge(
        *[OmegaConf.load(path) for path in config_paths]
    )
    merged_config.generator_ckpt = str(args.target_checkpoint)
    merged_config.gradient_checkpointing = False
    merged_config.num_frame_per_block = BLOCK_FRAMES
    merged_config.num_max_frames = NUM_FRAMES
    merged_config.last_step_only = True
    merged_config.memory_gap_blocks = 0
    merged_config.memory_gap_min_blocks = 0
    merged_config.memory_gap_max_blocks = 0
    merged_config.mcp_num_modules = 3
    merged_config.mcp_accel_depths = 1

    print("===== E0211 LOAD MODEL =====", flush=True)
    model = ODERegression(merged_config, device=device)
    model.generator.eval().requires_grad_(False)
    model.text_encoder.eval().requires_grad_(False)
    model.vae.eval().requires_grad_(False)
    model.generator.to(device=device, dtype=dtype)
    model.text_encoder.to(device=device, dtype=dtype)
    model.vae.to("cpu")

    mcp_contract = load_mcp_weights(model.generator, args.mcp_checkpoint)
    checkpoint_contract = {
        "schema_version": SCHEMA_VERSION,
        "target": {
            "path": str(args.target_checkpoint.resolve()),
            "sha256": target_hash,
            "expected_sha256": EXPECTED_TARGET_SHA256,
            "hash_matches": target_hash_matches,
        },
        "mcp": mcp_contract,
        "configs": [str(path.resolve()) for path in config_paths],
        "git": {
            "branch": git_output("branch", "--show-current"),
            "head": git_output("rev-parse", "HEAD"),
            "status_short": git_output("status", "--short"),
        },
    }
    write_json(output_dir / "checkpoint_contract.json", checkpoint_contract)

    all_metric_records: list[dict[str, Any]] = []
    all_timing_records: list[dict[str, Any]] = []
    all_video_records: list[dict[str, Any]] = []
    latent_archive: dict[str, Any] = {}

    for sample_position, (sample_path, payload) in enumerate(payloads):
        sample_index = int(payload.get("sample_index", sample_position))
        sample_label = f"sample{sample_index:06d}"
        if sample_path.name.startswith("teacher_sample_"):
            sample_label = sample_path.stem.replace("teacher_sample_", "sample")

        steps = resolve_steps(payload)
        valid_anchors = {int(value) for value in payload["valid_anchor_blocks"]}
        requested_valid = [anchor for anchor in anchors if anchor in valid_anchors]
        if not requested_valid:
            raise RuntimeError(
                f"{sample_path} has no requested valid anchors. "
                f"available={sorted(valid_anchors)} requested={anchors}"
            )

        print(f"===== PRECOMPUTE TEXT {sample_label} =====", flush=True)
        with torch.no_grad():
            conditional_dict = model.text_encoder(
                text_prompts=[str(payload["prompt"])]
            )

        for anchor_position, anchor_block in enumerate(requested_valid):
            print(
                f"===== {sample_label} anchor={anchor_block} BUILD DRAFT =====",
                flush=True,
            )
            history = replay_target_history_and_make_draft(
                generator=model.generator,
                payload=payload,
                conditional_dict=conditional_dict,
                anchor_block=anchor_block,
                steps=steps,
                device=device,
                dtype=dtype,
            )

            variants: dict[str, torch.Tensor] = {
                "draft_r0": history.draft,
            }
            timing_by_variant: dict[str, dict[str, Any]] = {
                "draft_r0": {
                    "num_target_refinement_steps": 0,
                    "wall_clock_ms_median": 0.0,
                    "wall_clock_ms_min": 0.0,
                    "wall_clock_ms_max": 0.0,
                    "timing_repeats_ms": [0.0],
                    "bridge_mode": "none",
                }
            }

            controlled_seed = (
                int(payload["seed"]) + 400_000 + anchor_block * 1_000
            )
            shared_transition_noises = [
                deterministic_noise_like(
                    history.draft,
                    controlled_seed + noise_index,
                )
                for noise_index in range(len(steps) - 1)
            ]

            for refinement_steps in (1, 2, 3):
                name = f"draft_r{refinement_steps}"
                output, timing = run_refinement_variant(
                    history=history,
                    conditional_dict=conditional_dict,
                    num_refinement_steps=refinement_steps,
                    full_steps=steps,
                    shared_transition_noises=shared_transition_noises,
                    timing_repeats=args.timing_repeats,
                    device=device,
                )
                variants[name] = output
                timing_by_variant[name] = timing

            target_rerun, target_timing = run_full_target_variant(
                history=history,
                conditional_dict=conditional_dict,
                full_steps=steps,
                shared_transition_noises=shared_transition_noises,
                timing_repeats=args.timing_repeats,
                device=device,
            )
            teacher_rng_replay = run_teacher_rng_replay(
                history=history,
                conditional_dict=conditional_dict,
                full_steps=steps,
                device=device,
            )
            variants["target4_controlled"] = target_rerun
            variants["target_teacher"] = history.target_chunk
            variants["target_teacher_rng_replay"] = teacher_rng_replay
            timing_by_variant["target4_controlled"] = target_timing
            timing_by_variant["target_teacher"] = {
                "num_target_steps": 4,
                "wall_clock_ms_median": target_timing["wall_clock_ms_median"],
                "wall_clock_ms_min": target_timing["wall_clock_ms_min"],
                "wall_clock_ms_max": target_timing["wall_clock_ms_max"],
                "timing_repeats_ms": target_timing["timing_repeats_ms"],
                "source": "stored_teacher_quality_oracle; rerun_timing_reused",
            }
            timing_by_variant["target_teacher_rng_replay"] = {
                "num_target_steps": 4,
                "wall_clock_ms_median": 0.0,
                "wall_clock_ms_min": 0.0,
                "wall_clock_ms_max": 0.0,
                "timing_repeats_ms": [0.0],
                "source": "old_teacher_rng_parity_diagnostic",
            }

            target4_ms = float(target_timing["wall_clock_ms_median"])
            anchor_key = f"{sample_label}_anchor{anchor_block}"
            latent_archive[anchor_key] = {
                name: tensor.detach().cpu()
                for name, tensor in variants.items()
            }
            latent_archive[anchor_key]["future_noise"] = (
                history.future_noise.detach().cpu()
            )
            latent_archive[anchor_key]["mcp_flow"] = history.mcp_flow.detach().cpu()

            for name, tensor in variants.items():
                metrics = tensor_metrics(
                    tensor,
                    target_rerun,
                    source_noise=history.future_noise,
                )
                if name.startswith("draft_r"):
                    num_target_steps = int(name.split("r")[-1])
                else:
                    num_target_steps = 4

                variant_ms = float(
                    timing_by_variant[name]["wall_clock_ms_median"]
                )
                # Proposal is shared with anchor generation. This ratio is a
                # measured step-level diagnostic, not an end-to-end video claim.
                hybrid_extra_ms = (
                    history.proposal_ms + variant_ms
                    if name.startswith("draft_r")
                    else target4_ms
                )
                measured_speedup = (
                    target4_ms / hybrid_extra_ms
                    if hybrid_extra_ms > 0
                    else 0.0
                )

                record = {
                    "sample_label": sample_label,
                    "sample_file": str(sample_path),
                    "sample_index": sample_index,
                    "anchor_block": anchor_block,
                    "future_block": anchor_block + 1,
                    "variant": name,
                    "reference_variant": "target4_controlled",
                    "controlled_trajectory_seed": controlled_seed,
                    "num_target_steps": num_target_steps,
                    "proposal_ms": history.proposal_ms,
                    "wall_clock_ms_median": variant_ms,
                    "full_target_wall_clock_ms_median": target4_ms,
                    "measured_step_speedup": finite_float(measured_speedup),
                    **metrics,
                }
                all_metric_records.append(record)
                all_timing_records.append(
                    {
                        "sample_label": sample_label,
                        "anchor_block": anchor_block,
                        "variant": name,
                        "proposal_ms": history.proposal_ms,
                        **timing_by_variant[name],
                    }
                )
                print(
                    f"{sample_label} anchor={anchor_block} variant={name} "
                    f"mse={metrics['latent_mse']:.6f} "
                    f"flow_cos={metrics['flow_cosine']:.6f} "
                    f"ms={variant_ms:.2f}",
                    flush=True,
                )

            controlled_vs_teacher = tensor_metrics(
                target_rerun,
                history.target_chunk,
                source_noise=history.future_noise,
            )
            rng_replay_vs_teacher = tensor_metrics(
                teacher_rng_replay,
                history.target_chunk,
                source_noise=history.future_noise,
            )
            print(
                f"{sample_label} anchor={anchor_block} "
                f"target4_controlled_vs_teacher_mse="
                f"{controlled_vs_teacher['latent_mse']:.6f} "
                f"target_teacher_rng_replay_mse="
                f"{rng_replay_vs_teacher['latent_mse']:.6f}",
                flush=True,
            )

            should_decode = (
                not args.skip_video
                and (
                    args.save_every_anchor_video
                    or anchor_position == 0
                )
            )
            if should_decode:
                print(
                    f"===== {sample_label} anchor={anchor_block} VAE DECODE =====",
                    flush=True,
                )
                decoded = decode_and_save_variants(
                    model=model,
                    payload=payload,
                    sample_label=sample_label,
                    anchor_block=anchor_block,
                    variants=variants,
                    output_dir=output_dir,
                    device=device,
                    dtype=dtype,
                    fps=args.fps,
                )
                for item in decoded:
                    all_video_records.append(
                        {
                            "sample_label": sample_label,
                            "anchor_block": anchor_block,
                            **item,
                        }
                    )
                model.generator.to(device=device, dtype=dtype)
                model.text_encoder.to(device=device, dtype=dtype)

            del history, variants, timing_by_variant
            gc.collect()
            torch.cuda.empty_cache()

        del conditional_dict
        gc.collect()
        torch.cuda.empty_cache()

    torch.save(latent_archive, output_dir / "latents.pt")

    write_metrics_csv(output_dir / "metrics.csv", all_metric_records)
    write_manual_template(
        output_dir / "manual_review_template.csv",
        all_metric_records,
    )
    timing_payload = {
        "schema_version": SCHEMA_VERSION,
        "note": (
            "measured_step_speedup is a step-level diagnostic. It uses "
            "Target-4-step time divided by MCP proposal forward plus refinement "
            "time; it is not an end-to-end video speedup claim."
        ),
        "records": all_timing_records,
    }
    write_json(output_dir / "timing.json", timing_payload)

    report = {
        "schema_version": SCHEMA_VERSION,
        "experiment": EXPERIMENT_NAME,
        "status": "ARTIFACTS_READY",
        "question": (
            "Can the blurry depth-1 MCP draft serve as a coarse initialization "
            "that Target repairs in one or two denoising steps?"
        ),
        "fixed_conditions": [
            "prompt",
            "seed",
            "source_noise",
            "Target history",
            "anchor",
            "VAE decode settings",
        ],
        "variants": [
            "draft_r0",
            "draft_r1",
            "draft_r2",
            "draft_r3",
            "target4_controlled",
            "target_teacher_rng_replay",
            "target_teacher",
        ],
        "refinement_bridge": (
            "All variants use one controlled shared-noise Target trajectory. "
            "Draft r3/r2/r1 bridge with z0/z1/z2 respectively and reuse the "
            "same remaining suffix noises as target4_controlled."
        ),
        "decision_rule": {
            "r1_visual_pass": (
                "Proceed to MCP proposal + Target one-step correction."
            ),
            "r2_visual_pass": (
                "Compute full wall-clock benefit before continuing."
            ),
            "only_r3_or_target_pass": (
                "Current MCP has limited acceleration value; investigate "
                "training target/loss/architecture."
            ),
        },
        "checkpoint_contract": checkpoint_contract,
        "sample_files": [str(path) for path, _ in payloads],
        "anchors": list(anchors),
        "metrics": all_metric_records,
        "videos": all_video_records,
        "limitations": [
            (
                "Visual PASS/FAIL is intentionally manual. Fill "
                "manual_review_template.csv after viewing review.html."
            ),
            (
                "The reported speedup is a local step-level diagnostic, not a "
                "complete end-to-end speculative runtime measurement."
            ),
            (
                "All curve metrics use target4_controlled as the local reference. "
                "The stored teacher and exact-RNG replay are reported separately."
            ),
        ],
    }
    write_json(output_dir / "report.json", report)
    make_review_html(
        output_dir=output_dir,
        records=all_metric_records,
        video_records=all_video_records,
    )

    print("===== E0211 RESULT =====", flush=True)
    print("status=ARTIFACTS_READY", flush=True)
    print(f"output_dir={output_dir}", flush=True)
    print(f"report={output_dir / 'report.json'}", flush=True)
    print(f"review={output_dir / 'review.html'}", flush=True)
    print(
        "next=Open review.html and fill manual_review_template.csv; "
        "do not enter depth-2/verifier before the visual decision.",
        flush=True,
    )


if __name__ == "__main__":
    main()
