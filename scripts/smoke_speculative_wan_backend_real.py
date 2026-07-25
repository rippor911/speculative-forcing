from __future__ import annotations

import argparse
import gc
import subprocess
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Sequence

import torch

from inference_mcp import (
    ANCHOR_DENOISING_STEPS,
    EXPECTED_MCP_TENSOR_COUNT,
    build_rollout_pipeline,
    make_noise,
    merge_config,
    require_single_gpu_runtime,
    reset_runtime_seed,
    validate_checkpoint_restore,
    validate_config,
)
from speculative.adapters.self_forcing_runtime import (
    SelfForcingMCPRuntime,
    SelfForcingMCPRuntimeConfig,
)
from speculative.adapters.self_forcing_wan_backend import SelfForcingWanMCPBackend
from speculative.adapters.wan_state_planner import (
    SUPPORTED_ATTENTION_MODE,
    WanCacheLayout,
    WanTouchedRangePlanner,
)
from speculative.types import BlockRef, CommitRequest, ControlRequest, DraftCandidate


@dataclass(frozen=True)
class TensorFingerprint:
    shape: tuple[int, ...]
    dtype: str
    device: str
    sum: float
    abs_sum: float
    square_sum: float


@dataclass(frozen=True)
class CrossAttentionIdentity:
    list_id: int
    layer_ids: tuple[int, ...]
    k_ids: tuple[int, ...]
    v_ids: tuple[int, ...]


@dataclass(frozen=True)
class CrossAttentionSnapshot:
    identity: CrossAttentionIdentity
    is_init: tuple[bool, ...]
    k_fingerprints: tuple[TensorFingerprint, ...]
    v_fingerprints: tuple[TensorFingerprint, ...]


@dataclass(frozen=True)
class RuntimeSnapshot:
    kv_indices: tuple[tuple[int, int], ...]
    kv_fingerprints: tuple[tuple[int, str, int, int, TensorFingerprint], ...]
    cross_attention: CrossAttentionSnapshot
    output: torch.Tensor
    committed_blocks: tuple[BlockRef, ...]
    has_active_window: bool
    is_prepared: bool
    cuda_rng: torch.Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Real-checkpoint smoke for SelfForcingWanMCPBackend under "
            "SelfForcingMCPRuntime. No VAE decode, no video output."
        )
    )
    parser.add_argument("--config", required=True, help="Path to run config YAML.")
    parser.add_argument("--checkpoint", required=True, help="Generator checkpoint.")
    parser.add_argument("--prompt", required=True, help="Single T2V prompt.")
    parser.add_argument("--seed", type=int, default=0, help="Runtime seed.")
    parser.add_argument("--num_frames", type=int, default=6, help="Latent frame count.")
    parser.add_argument(
        "--mcp_depth",
        type=int,
        choices=(1, 2, 3),
        default=1,
        help="MCP draft depth to smoke.",
    )
    parser.add_argument("--device", default="cuda:0", help="CUDA device, e.g. cuda:0.")
    return parser.parse_args()


def helper_args_from(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        config=args.config,
        checkpoint=args.checkpoint,
        prompt=args.prompt,
        seed=args.seed,
        num_frames=args.num_frames,
        mcp_depth=args.mcp_depth,
        disable_mcp=False,
        fps=16,
        device=args.device,
    )


def require_exactly_one_visible_cuda(device: torch.device) -> None:
    visible = torch.cuda.device_count()
    if visible != 1:
        raise RuntimeError(
            "This smoke requires exactly one visible CUDA device; set "
            f"CUDA_VISIBLE_DEVICES=0. Visible CUDA device count: {visible}."
        )
    if device.index not in (None, 0):
        raise RuntimeError(f"Single-GPU smoke requires cuda:0, got {device}.")


def require_anchor_and_draft_frames(num_frames: int, block_frames: int) -> None:
    if num_frames < block_frames * 2:
        raise ValueError(
            "--num_frames must include at least one anchor block and one draft "
            f"block: got {num_frames}, block size {block_frames}."
        )


def git_head() -> str:
    try:
        output = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except Exception:
        return "UNKNOWN"
    return output.strip()


def tensor_fingerprint(tensor: torch.Tensor) -> TensorFingerprint:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(tensor).__name__}.")
    detached = tensor.detach()
    stats = detached.to(dtype=torch.float32)
    return TensorFingerprint(
        shape=tuple(int(dim) for dim in detached.shape),
        dtype=str(detached.dtype),
        device=str(detached.device),
        sum=float(stats.sum().item()),
        abs_sum=float(stats.abs().sum().item()),
        square_sum=float(stats.square().sum().item()),
    )


def require_finite_tensor(tensor: torch.Tensor, name: str) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if not bool(torch.isfinite(tensor).all().item()):
        raise RuntimeError(f"{name} contains non-finite values.")


def require_exact_tensor_equal(actual: torch.Tensor, expected: torch.Tensor, name: str) -> None:
    if actual.shape != expected.shape:
        raise RuntimeError(
            f"{name} shape mismatch: {tuple(actual.shape)} != {tuple(expected.shape)}."
        )
    if actual.dtype != expected.dtype:
        raise RuntimeError(f"{name} dtype mismatch: {actual.dtype} != {expected.dtype}.")
    if actual.device != expected.device:
        raise RuntimeError(f"{name} device mismatch: {actual.device} != {expected.device}.")
    if actual.layout != expected.layout:
        raise RuntimeError(f"{name} layout mismatch: {actual.layout} != {expected.layout}.")
    if not bool(torch.equal(actual, expected)):
        raise RuntimeError(f"{name} values differ.")


def require_output_zero(output: torch.Tensor, name: str) -> None:
    if not bool((output == 0).all().item()):
        raise RuntimeError(f"{name} is not all zeros.")


def tensor_shares_storage(left: torch.Tensor, right: torch.Tensor) -> bool:
    return left.untyped_storage().data_ptr() == right.untyped_storage().data_ptr()


def block_slice(tensor: torch.Tensor, block: BlockRef) -> torch.Tensor:
    if block.start_frame is None or block.num_frames is None:
        raise RuntimeError("BlockRef start_frame and num_frames are required.")
    start = block.start_frame
    end = start + block.num_frames
    return tensor[:, start:end]


def block_token_range(block: BlockRef, frame_seq_length: int) -> tuple[int, int]:
    if block.start_frame is None or block.num_frames is None:
        raise RuntimeError("BlockRef start_frame and num_frames are required.")
    start = block.start_frame * frame_seq_length
    end = (block.start_frame + block.num_frames) * frame_seq_length
    return start, end


def merge_ranges(ranges: Sequence[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    ordered = sorted(ranges)
    merged: list[tuple[int, int]] = []
    for start, end in ordered:
        if start == end:
            continue
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            previous_start, previous_end = merged[-1]
            merged[-1] = (previous_start, max(previous_end, end))
    return tuple(merged)


def kv_ranges_for_blocks(
    blocks: Sequence[BlockRef],
    frame_seq_length: int,
) -> tuple[tuple[int, int], ...]:
    return merge_ranges(tuple(block_token_range(block, frame_seq_length) for block in blocks))


def kv_indices(kv_cache: Sequence[dict[str, Any]]) -> tuple[tuple[int, int], ...]:
    indices = []
    for layer_index, layer in enumerate(kv_cache):
        try:
            global_index = int(layer["global_end_index"].item())
            local_index = int(layer["local_end_index"].item())
        except KeyError as exc:
            raise KeyError(f"kv_cache layer {layer_index} is missing {exc.args[0]}.") from exc
        indices.append((global_index, local_index))
    return tuple(indices)


def require_all_indices(
    kv_cache: Sequence[dict[str, Any]],
    expected: int,
    label: str,
) -> None:
    for layer_index, (global_index, local_index) in enumerate(kv_indices(kv_cache)):
        if global_index != expected:
            raise RuntimeError(
                f"{label}: layer {layer_index} global_end_index "
                f"{global_index} != expected {expected}."
            )
        if local_index != expected:
            raise RuntimeError(
                f"{label}: layer {layer_index} local_end_index "
                f"{local_index} != expected {expected}."
            )


def consistent_index(kv_cache: Sequence[dict[str, Any]], field: str) -> int:
    values = [int(layer[field].item()) for layer in kv_cache]
    if not values:
        raise RuntimeError("KV cache is empty.")
    first = values[0]
    if any(value != first for value in values):
        raise RuntimeError(f"KV cache {field} differs across layers: {values}.")
    return first


def kv_fingerprints(
    kv_cache: Sequence[dict[str, Any]],
    ranges: Sequence[tuple[int, int]],
) -> tuple[tuple[int, str, int, int, TensorFingerprint], ...]:
    fingerprints = []
    for layer_index, layer in enumerate(kv_cache):
        for field_name in ("k", "v"):
            tensor = layer[field_name]
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"kv_cache layer {layer_index}.{field_name} must be a tensor.")
            for start, end in ranges:
                if start < 0 or end < start or end > tensor.shape[1]:
                    raise RuntimeError(
                        f"KV fingerprint range [{start}, {end}) is outside "
                        f"layer {layer_index}.{field_name} capacity {tensor.shape[1]}."
                    )
                fingerprints.append(
                    (
                        layer_index,
                        field_name,
                        start,
                        end,
                        tensor_fingerprint(tensor[:, start:end]),
                    )
                )
    return tuple(fingerprints)


def cross_attention_identity(cache: Sequence[dict[str, Any]]) -> CrossAttentionIdentity:
    return CrossAttentionIdentity(
        list_id=id(cache),
        layer_ids=tuple(id(layer) for layer in cache),
        k_ids=tuple(id(layer["k"]) for layer in cache),
        v_ids=tuple(id(layer["v"]) for layer in cache),
    )


def cross_attention_snapshot(cache: Sequence[dict[str, Any]]) -> CrossAttentionSnapshot:
    return CrossAttentionSnapshot(
        identity=cross_attention_identity(cache),
        is_init=tuple(layer["is_init"] for layer in cache),
        k_fingerprints=tuple(tensor_fingerprint(layer["k"]) for layer in cache),
        v_fingerprints=tuple(tensor_fingerprint(layer["v"]) for layer in cache),
    )


def require_cross_attention_finite(cache: Sequence[dict[str, Any]]) -> None:
    for layer_index, layer in enumerate(cache):
        require_finite_tensor(layer["k"], f"crossattn_cache layer {layer_index}.k")
        require_finite_tensor(layer["v"], f"crossattn_cache layer {layer_index}.v")


def require_cross_attention_is_init(cache: Sequence[dict[str, Any]]) -> None:
    for layer_index, layer in enumerate(cache):
        if layer["is_init"] is not True:
            raise RuntimeError(f"crossattn_cache layer {layer_index}.is_init is not True.")


def cuda_rng_snapshot(device: torch.device) -> torch.Tensor:
    torch.cuda.synchronize(device)
    return torch.cuda.get_rng_state(device).clone()


def runtime_snapshot(
    runtime: SelfForcingMCPRuntime,
    kv_cache: Sequence[dict[str, Any]],
    crossattn_cache: Sequence[dict[str, Any]],
    kv_ranges: Sequence[tuple[int, int]],
    device: torch.device,
) -> RuntimeSnapshot:
    torch.cuda.synchronize(device)
    return RuntimeSnapshot(
        kv_indices=kv_indices(kv_cache),
        kv_fingerprints=kv_fingerprints(kv_cache, kv_ranges),
        cross_attention=cross_attention_snapshot(crossattn_cache),
        output=runtime.output.detach().clone(),
        committed_blocks=runtime.committed_blocks,
        has_active_window=runtime.has_active_window,
        is_prepared=runtime.is_prepared,
        cuda_rng=cuda_rng_snapshot(device),
    )


def require_runtime_matches_snapshot(
    *,
    runtime: SelfForcingMCPRuntime,
    kv_cache: Sequence[dict[str, Any]],
    crossattn_cache: Sequence[dict[str, Any]],
    kv_ranges: Sequence[tuple[int, int]],
    device: torch.device,
    expected: RuntimeSnapshot,
    label: str,
) -> None:
    actual = runtime_snapshot(runtime, kv_cache, crossattn_cache, kv_ranges, device)
    if actual.kv_indices != expected.kv_indices:
        raise RuntimeError(f"{label}: KV indices differ from snapshot.")
    if actual.kv_fingerprints != expected.kv_fingerprints:
        raise RuntimeError(f"{label}: KV fingerprints differ from snapshot.")
    if actual.cross_attention != expected.cross_attention:
        raise RuntimeError(f"{label}: cross-attention snapshot differs.")
    require_exact_tensor_equal(actual.output, expected.output, f"{label} output")
    if actual.committed_blocks != expected.committed_blocks:
        raise RuntimeError(f"{label}: committed_blocks differ from snapshot.")
    if actual.has_active_window != expected.has_active_window:
        raise RuntimeError(f"{label}: has_active_window differs from snapshot.")
    if actual.is_prepared != expected.is_prepared:
        raise RuntimeError(f"{label}: is_prepared differs from snapshot.")
    if not bool(torch.equal(actual.cuda_rng, expected.cuda_rng)):
        raise RuntimeError(f"{label}: CUDA RNG state differs from snapshot.")


def requested_window_blocks(
    rollout_blocks: Sequence[BlockRef],
    request: ControlRequest,
    mcp_depth: int,
) -> tuple[BlockRef, ...]:
    remaining = len(rollout_blocks) - request.anchor_block.index - 1
    depth_count = min(request.max_depth, mcp_depth, remaining)
    start = request.anchor_block.index
    return tuple(rollout_blocks[start:start + depth_count + 1])


def require_draft_source_noise_view(
    draft: DraftCandidate,
    source_noise: torch.Tensor,
) -> None:
    candidate_noise = draft.source_noise
    if not isinstance(candidate_noise, torch.Tensor):
        raise TypeError("draft.source_noise must be a torch.Tensor.")
    expected = block_slice(source_noise, draft.block)
    if not tensor_shares_storage(candidate_noise, source_noise):
        raise RuntimeError(f"draft depth {draft.depth} source_noise is not a source-noise view.")
    if candidate_noise.data_ptr() != expected.data_ptr():
        raise RuntimeError(f"draft depth {draft.depth} source_noise starts at the wrong offset.")
    require_exact_tensor_equal(
        candidate_noise,
        expected,
        f"draft depth {draft.depth} source_noise",
    )


def require_contiguous_drafts(drafts: Sequence[DraftCandidate]) -> None:
    expected_depths = tuple(range(1, len(drafts) + 1))
    actual_depths = tuple(draft.depth for draft in drafts)
    if actual_depths != expected_depths:
        raise RuntimeError(f"Draft depths are not contiguous: {actual_depths}.")


def require_output_slice_equals(
    output: torch.Tensor,
    block: BlockRef,
    latent: torch.Tensor,
    label: str,
) -> None:
    require_exact_tensor_equal(block_slice(output, block), latent, label)


def expected_token_end(block: BlockRef, frame_seq_length: int) -> int:
    if block.start_frame is None or block.num_frames is None:
        raise RuntimeError("BlockRef start_frame and num_frames are required.")
    return (block.start_frame + block.num_frames) * frame_seq_length


def smoke_runtime(
    *,
    runtime: SelfForcingMCPRuntime,
    pipeline: Any,
    source_noise: torch.Tensor,
    output: torch.Tensor,
    mcp_depth: int,
    frame_seq_length: int,
    device: torch.device,
) -> dict[str, Any]:
    kv_cache = pipeline.kv_cache1
    crossattn_cache = pipeline.crossattn_cache
    initial_cross_identity = cross_attention_identity(crossattn_cache)
    initial_output = output.detach().clone()

    runtime.prepare()
    torch.cuda.synchronize(device)
    if not runtime.is_prepared:
        raise RuntimeError("runtime.is_prepared is not True after prepare.")
    if pipeline.kv_cache1 is not kv_cache:
        raise RuntimeError("pipeline.kv_cache1 was rebound during prepare.")
    if pipeline.crossattn_cache is not crossattn_cache:
        raise RuntimeError("pipeline.crossattn_cache was rebound during prepare.")
    if cross_attention_identity(crossattn_cache) != initial_cross_identity:
        raise RuntimeError("Cross-attention identity changed during prepare.")
    require_cross_attention_is_init(crossattn_cache)
    require_cross_attention_finite(crossattn_cache)
    require_all_indices(kv_cache, 0, "prepare")
    require_exact_tensor_equal(output, initial_output, "prepare output")
    require_output_zero(output, "prepare output")
    if runtime.committed_blocks:
        raise RuntimeError("committed_blocks is not empty after prepare.")

    request = ControlRequest(
        anchor_block=runtime.rollout_plan.blocks[0],
        max_depth=mcp_depth,
    )
    baseline_blocks = requested_window_blocks(
        runtime.rollout_plan.blocks,
        request,
        mcp_depth,
    )
    baseline_kv_ranges = kv_ranges_for_blocks(
        baseline_blocks,
        frame_seq_length,
    )
    s0 = runtime_snapshot(runtime, kv_cache, crossattn_cache, baseline_kv_ranges, device)

    batch = runtime.propose_window(request)
    torch.cuda.synchronize(device)
    require_finite_tensor(batch.anchor.latent, "proposal anchor latent")
    if len(batch.drafts) < 1:
        raise RuntimeError("Proposal returned no drafts; smoke needs at least one draft.")
    require_contiguous_drafts(batch.drafts)
    for draft in batch.drafts:
        require_finite_tensor(draft.latent, f"proposal draft depth {draft.depth} latent")
        require_draft_source_noise_view(draft, source_noise)
    require_runtime_matches_snapshot(
        runtime=runtime,
        kv_cache=kv_cache,
        crossattn_cache=crossattn_cache,
        kv_ranges=baseline_kv_ranges,
        device=device,
        expected=s0,
        label="proposal rollback",
    )
    if runtime.has_active_window:
        raise RuntimeError("runtime.has_active_window is True after proposal rollback.")

    runtime.begin_window()
    runtime.commit_block(batch.anchor)
    torch.cuda.synchronize(device)
    if not runtime.has_active_window:
        raise RuntimeError("runtime.has_active_window is False after begin+anchor commit.")
    if runtime.committed_blocks != (batch.anchor.block,):
        raise RuntimeError("committed_blocks does not contain only the anchor block.")
    require_output_slice_equals(
        output,
        batch.anchor.block,
        batch.anchor.latent,
        "anchor output slice",
    )
    anchor_end = expected_token_end(batch.anchor.block, frame_seq_length)
    require_all_indices(kv_cache, anchor_end, "anchor commit")

    s1 = runtime_snapshot(runtime, kv_cache, crossattn_cache, baseline_kv_ranges, device)
    first_draft = batch.drafts[0]
    fallback = runtime.generate_target_fallback(first_draft)
    torch.cuda.synchronize(device)
    if fallback.block != first_draft.block:
        raise RuntimeError("fallback.block does not match draft.block.")
    same_noise_fallback = fallback.source_noise is first_draft.source_noise
    if not same_noise_fallback:
        raise RuntimeError("fallback.source_noise does not preserve draft.source_noise identity.")
    require_finite_tensor(fallback.latent, "fallback latent")
    require_runtime_matches_snapshot(
        runtime=runtime,
        kv_cache=kv_cache,
        crossattn_cache=crossattn_cache,
        kv_ranges=baseline_kv_ranges,
        device=device,
        expected=s1,
        label="fallback rollback",
    )

    runtime.rollback_window()
    torch.cuda.synchronize(device)
    require_runtime_matches_snapshot(
        runtime=runtime,
        kv_cache=kv_cache,
        crossattn_cache=crossattn_cache,
        kv_ranges=baseline_kv_ranges,
        device=device,
        expected=s0,
        label="window rollback",
    )
    if runtime.has_active_window:
        raise RuntimeError("runtime.has_active_window is True after window rollback.")

    second_batch = runtime.propose_window(request)
    torch.cuda.synchronize(device)
    require_exact_tensor_equal(
        second_batch.anchor.latent,
        batch.anchor.latent,
        "second proposal anchor latent",
    )
    if len(second_batch.drafts) != len(batch.drafts):
        raise RuntimeError("Second proposal draft count differs from the first proposal.")
    for first, second in zip(batch.drafts, second_batch.drafts):
        require_exact_tensor_equal(
            second.latent,
            first.latent,
            f"second proposal draft depth {second.depth} latent",
        )

    runtime.begin_window()
    all_commits = [
        second_batch.anchor,
        *(
            CommitRequest(
                block=draft.block,
                latent=draft.latent,
                source="draft",
                depth=draft.depth,
                source_noise=draft.source_noise,
            )
            for draft in second_batch.drafts
        ),
    ]
    for commit_request in all_commits:
        require_finite_tensor(
            commit_request.latent,
            f"commit block {commit_request.block.index} latent",
        )
        runtime.commit_block(commit_request)
    runtime.complete_window()
    torch.cuda.synchronize(device)
    if runtime.has_active_window:
        raise RuntimeError("runtime.has_active_window is True after complete_window.")
    final_blocks = tuple(commit.block for commit in all_commits)
    if runtime.committed_blocks != final_blocks:
        raise RuntimeError("committed_blocks does not match committed anchor+drafts.")
    for commit_request in all_commits:
        require_output_slice_equals(
            output,
            commit_request.block,
            commit_request.latent,
            f"completed output block {commit_request.block.index}",
        )
    last_block = all_commits[-1].block
    final_token_end = expected_token_end(
        last_block,
        frame_seq_length,
    )
    require_all_indices(kv_cache, final_token_end, "window complete")

    final_snapshot = runtime_snapshot(
        runtime,
        kv_cache,
        crossattn_cache,
        baseline_kv_ranges,
        device,
    )
    if final_snapshot.kv_fingerprints == s0.kv_fingerprints:
        raise RuntimeError("KV fingerprints did not change after completed commits.")
    if bool(torch.equal(final_snapshot.cuda_rng, s0.cuda_rng)):
        raise RuntimeError("CUDA RNG did not advance after completed commits.")
    if final_snapshot.cross_attention != s0.cross_attention:
        raise RuntimeError("Cross-attention snapshot changed after completed commits.")
    if final_snapshot.cross_attention.identity != initial_cross_identity:
        raise RuntimeError("Cross-attention identity changed before final complete.")
    for commit_request in all_commits:
        require_finite_tensor(
            block_slice(output, commit_request.block),
            f"output block {commit_request.block.index}",
        )

    return {
        "prepare": "PASS",
        "proposal_rollback": "PASS",
        "fallback_rollback": "PASS",
        "window_rollback": "PASS",
        "window_complete": "PASS",
        "crossattn_identity_preserved": True,
        "same_noise_fallback": same_noise_fallback,
        "final_committed_blocks": [block.index for block in runtime.committed_blocks],
        "final_global_index": consistent_index(kv_cache, "global_end_index"),
        "final_local_index": consistent_index(kv_cache, "local_end_index"),
        "cuda_rng_rollback": True,
        "cuda_rng_completed_changed": True,
    }


def print_result(key: str, value: Any) -> None:
    print(f"{key}={value}", flush=True)


def main() -> None:
    args = parse_args()
    helper_args = helper_args_from(args)

    import numpy as np
    from model.ode_regression import ODERegression
    from utils.checkpoint import (
        MCP_COMPLETE_STRICT_RESTORE,
        is_mcp_state_key,
        load_state_dict_allowing_mcp_mismatch,
    )

    model = None
    rollout_pipeline = None
    backend = None
    runtime = None
    conditional_dict = None
    noise = None
    output = None
    device = None

    try:
        device = require_single_gpu_runtime(torch, helper_args.device)
        require_exactly_one_visible_cuda(device)
        torch.cuda.reset_peak_memory_stats(device)
        torch.set_grad_enabled(False)

        reset_runtime_seed(helper_args.seed, np, torch)

        config = merge_config(helper_args.config)
        config.generator_ckpt = helper_args.checkpoint
        config.gradient_checkpointing = False
        validate_config(config, helper_args)
        require_anchor_and_draft_frames(
            int(helper_args.num_frames),
            int(config.num_frame_per_block),
        )

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
                f"Expected {EXPECTED_MCP_TENSOR_COUNT} MCP tensors, got "
                f"{mcp_tensor_count}."
            )

        model.generator.eval().requires_grad_(False)
        model.text_encoder.eval().requires_grad_(False)
        model.generator.to(device=device, dtype=torch.bfloat16)
        model.text_encoder.to(device=device, dtype=torch.bfloat16)

        with torch.no_grad():
            conditional_dict = model.text_encoder(text_prompts=[helper_args.prompt])

        rollout_pipeline = build_rollout_pipeline(config, model, helper_args)
        noise = make_noise(helper_args, device, torch)
        rollout_pipeline._initialize_kv_cache(
            batch_size=1,
            dtype=noise.dtype,
            device=noise.device,
        )
        rollout_pipeline._initialize_crossattn_cache(
            batch_size=1,
            dtype=noise.dtype,
            device=noise.device,
        )

        cache_layout = WanCacheLayout(
            num_layers=len(rollout_pipeline.kv_cache1),
            cache_capacity=int(rollout_pipeline.kv_cache_size),
            cross_attention_capacity=int(rollout_pipeline.crossattn_cache[0]["k"].shape[1]),
            self_attention_dim=1,
            cross_attention_dim=1,
            batch_size=1,
            cfg_batch_multiplier=1,
        )
        planner = WanTouchedRangePlanner(
            frame_seq_length=int(rollout_pipeline.frame_seq_length),
            num_frame_per_block=int(rollout_pipeline.num_frame_per_block),
            mcp_depth=int(helper_args.mcp_depth),
            cache_layout=cache_layout,
            local_attn_size=-1,
            sink_size=0,
            runtime_device_type="cuda",
        )
        backend = SelfForcingWanMCPBackend(
            pipeline=rollout_pipeline,
            conditional_dict=conditional_dict,
            planner=planner,
        )
        output = torch.zeros_like(noise)
        runtime_config = SelfForcingMCPRuntimeConfig(
            anchor_denoising_steps=ANCHOR_DENOISING_STEPS,
            frame_seq_length=int(rollout_pipeline.frame_seq_length),
            num_frame_per_block=int(rollout_pipeline.num_frame_per_block),
            all_num_frames=int(helper_args.num_frames),
            mcp_depth=int(helper_args.mcp_depth),
            attention_mode=SUPPORTED_ATTENTION_MODE,
            validated_attention_mode=SUPPORTED_ATTENTION_MODE,
        )
        runtime = SelfForcingMCPRuntime(
            config=runtime_config,
            backend=backend,
            source_noise=noise,
            output=output,
            kv_cache=rollout_pipeline.kv_cache1,
            cross_attention_cache=rollout_pipeline.crossattn_cache,
        )

        reset_runtime_seed(helper_args.seed, np, torch)
        smoke_results = smoke_runtime(
            runtime=runtime,
            pipeline=rollout_pipeline,
            source_noise=noise,
            output=output,
            mcp_depth=int(helper_args.mcp_depth),
            frame_seq_length=int(rollout_pipeline.frame_seq_length),
            device=device,
        )

        torch.cuda.synchronize(device)
        peak_gib = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        print_result("f2a_commit_hash", git_head())
        print_result("checkpoint_load_mode", load_mode)
        print_result("mcp_tensor_count", int(mcp_tensor_count))
        print_result("device", str(device))
        print_result("gpu_name", torch.cuda.get_device_name(device))
        print_result("dtype", str(noise.dtype))
        print_result("num_frames", int(helper_args.num_frames))
        print_result("mcp_depth", int(helper_args.mcp_depth))
        for key in (
            "prepare",
            "proposal_rollback",
            "fallback_rollback",
            "window_rollback",
            "window_complete",
            "crossattn_identity_preserved",
            "same_noise_fallback",
            "cuda_rng_rollback",
            "cuda_rng_completed_changed",
            "final_committed_blocks",
            "final_global_index",
            "final_local_index",
        ):
            print_result(key, smoke_results[key])
        print_result("peak_cuda_memory_gib", f"{peak_gib:.3f}")
        print_result("F2B_REAL_SMOKE", "PASS")
    finally:
        try:
            if model is not None:
                if hasattr(model, "generator"):
                    model.generator.to("cpu")
                if hasattr(model, "text_encoder"):
                    model.text_encoder.to("cpu")
        except Exception as cleanup_exc:
            print_result(
                "cleanup_warning",
                f"{type(cleanup_exc).__name__}: {cleanup_exc}",
            )

        del runtime
        del backend
        del rollout_pipeline
        del conditional_dict
        del output
        del noise
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
