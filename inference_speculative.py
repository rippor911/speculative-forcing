from __future__ import annotations

import argparse
import gc
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional, Sequence

from inference_mcp import (
    ANCHOR_DENOISING_STEPS,
    EXPECTED_MCP_TENSOR_COUNT,
    build_rollout_pipeline,
    make_noise,
    merge_config,
    require_consistent_cache_index,
    require_single_gpu_runtime,
    reset_runtime_seed,
    save_video,
    validate_checkpoint_restore,
    validate_config as validate_mcp_config,
    validate_latent,
    validate_pixels,
    write_trace,
)
from speculative.adapters.self_forcing_mcp import (
    SelfForcingMCPCommitter,
    SelfForcingMCPFallbackGenerator,
    SelfForcingMCPProposalSource,
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
from speculative.controller import SpeculativeController
from speculative.factory import create_policy
from speculative.trace import TRACE_SCHEMA_VERSION
from speculative.types import BlockRef, ControlRequest, DraftCandidate, Evaluation


SUPPORTED_POLICIES = ("always_accept", "always_reject", "reject_at_depth")
MODE = "speculative"


class ScriptedCandidateEvaluator:
    """Minimal production evaluator for scripted F3 policies.

    It returns the existing candidate inside `Evaluation` and does not decode,
    score, call models, touch caches, or mutate generation state.
    """

    def evaluate(self, candidate: DraftCandidate) -> Evaluation:
        return Evaluation(
            candidate=candidate,
            metadata={
                "evaluator_name": "scripted_noop",
                "block_index": candidate.block.index,
                "depth": candidate.depth,
            },
        )


@dataclass(frozen=True)
class ReferenceRNGPlan:
    mode: str
    draw_count: int
    exit_flags: tuple[int, ...]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Speculative Self-Forcing MCP rollout with scripted policies. "
            "No candidate VAE decode, scorer, or learned verifier."
        )
    )
    parser.add_argument("--config", required=True, help="Path to the run config YAML.")
    parser.add_argument("--checkpoint", required=True, help="Path to the generator checkpoint.")
    parser.add_argument("--prompt", required=True, help="Text prompt for T2V generation.")
    parser.add_argument("--output", required=True, help="Output MP4 path.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--num_frames", type=int, default=21, help="Number of latent frames.")
    parser.add_argument(
        "--mcp_depth",
        type=int,
        choices=(1, 2, 3),
        default=3,
        help="Maximum MCP draft depth to request per speculative window.",
    )
    parser.add_argument(
        "--policy",
        choices=SUPPORTED_POLICIES,
        default="always_accept",
        help="Scripted acceptance policy.",
    )
    parser.add_argument(
        "--reject_depth",
        type=int,
        default=None,
        help="Positive reject depth required only by --policy reject_at_depth.",
    )
    parser.add_argument(
        "--save_trace",
        default=None,
        help="Optional path for a JSON speculative trace.",
    )
    parser.add_argument("--fps", type=int, default=16, help="Output MP4 frame rate.")
    parser.add_argument("--device", default="cuda", help="CUDA device, e.g. cuda or cuda:0.")
    return parser.parse_args(argv)


def helper_args_from(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        config=args.config,
        checkpoint=args.checkpoint,
        prompt=args.prompt,
        seed=args.seed,
        num_frames=args.num_frames,
        mcp_depth=args.mcp_depth,
        disable_mcp=False,
        save_trace=args.save_trace,
        fps=args.fps,
        device=args.device,
    )


def validate_policy_args(args: argparse.Namespace) -> None:
    if args.policy not in SUPPORTED_POLICIES:
        known = ", ".join(SUPPORTED_POLICIES)
        raise ValueError(f"--policy must be one of {known}, got {args.policy!r}.")

    if not isinstance(args.mcp_depth, int) or isinstance(args.mcp_depth, bool):
        raise ValueError("--mcp_depth must be an integer.")
    if args.mcp_depth not in (1, 2, 3):
        raise ValueError("--mcp_depth must be one of 1, 2, or 3.")

    reject_depth = getattr(args, "reject_depth", None)
    if args.policy == "reject_at_depth":
        if reject_depth is None:
            raise ValueError("--policy reject_at_depth requires --reject_depth.")
        if not isinstance(reject_depth, int) or isinstance(reject_depth, bool):
            raise ValueError("--reject_depth must be a positive integer.")
        if reject_depth <= 0:
            raise ValueError("--reject_depth must be a positive integer.")
        if reject_depth > args.mcp_depth:
            raise ValueError("--reject_depth must be <= --mcp_depth.")
    elif reject_depth is not None:
        raise ValueError(f"--policy {args.policy} does not accept --reject_depth.")


def validate_speculative_config(config: Any, args: argparse.Namespace) -> None:
    validate_policy_args(args)
    validate_mcp_config(config, helper_args_from(args))


def build_policy(args: argparse.Namespace) -> Any:
    validate_policy_args(args)
    kwargs = {}
    if args.reject_depth is not None:
        kwargs["reject_depth"] = args.reject_depth
    return create_policy(args.policy, **kwargs)


def build_controller(runtime: Any, policy: Any) -> SpeculativeController:
    return SpeculativeController(
        proposer=SelfForcingMCPProposalSource(runtime),
        evaluator=ScriptedCandidateEvaluator(),
        policy=policy,
        fallback=SelfForcingMCPFallbackGenerator(runtime),
        committer=SelfForcingMCPCommitter(runtime),
    )


def next_control_request(runtime: Any, mcp_depth: int) -> Optional[ControlRequest]:
    blocks = tuple(runtime.rollout_plan.blocks)
    committed = tuple(runtime.committed_blocks)
    if len(committed) > len(blocks):
        raise RuntimeError("Runtime committed more blocks than the rollout plan contains.")
    for expected_index, block in enumerate(committed):
        if block != blocks[expected_index]:
            raise RuntimeError("Runtime committed_blocks must be a contiguous rollout prefix.")
    if len(committed) == len(blocks):
        return None

    anchor = blocks[len(committed)]
    remaining_future_blocks = len(blocks) - anchor.index - 1
    max_depth = min(int(mcp_depth), remaining_future_blocks)
    return ControlRequest(anchor_block=anchor, max_depth=max_depth)


def run_speculative_rollout(
    *,
    runtime: Any,
    controller: SpeculativeController,
    mcp_depth: int,
    kv_cache: Any = None,
) -> list[dict[str, Any]]:
    runtime.prepare()
    windows: list[dict[str, Any]] = []
    while True:
        request = next_control_request(runtime, mcp_depth)
        if request is None:
            return windows
        result = controller.run(request)
        windows.append(
            serialize_window(
                window_index=len(windows),
                request=request,
                result=result,
                runtime=runtime,
                kv_cache=kv_cache,
            )
        )


def build_cache_layout(rollout_pipeline: Any) -> WanCacheLayout:
    return WanCacheLayout(
        num_layers=len(rollout_pipeline.kv_cache1),
        cache_capacity=int(rollout_pipeline.kv_cache_size),
        cross_attention_capacity=int(rollout_pipeline.crossattn_cache[0]["k"].shape[1]),
        self_attention_dim=1,
        cross_attention_dim=1,
        batch_size=1,
        cfg_batch_multiplier=1,
    )


def build_planner(
    *,
    rollout_pipeline: Any,
    mcp_depth: int,
    runtime_device_type: str,
) -> WanTouchedRangePlanner:
    return WanTouchedRangePlanner(
        frame_seq_length=int(rollout_pipeline.frame_seq_length),
        num_frame_per_block=int(rollout_pipeline.num_frame_per_block),
        mcp_depth=int(mcp_depth),
        cache_layout=build_cache_layout(rollout_pipeline),
        local_attn_size=-1,
        sink_size=0,
        runtime_device_type=runtime_device_type,  # type: ignore[arg-type]
    )


def build_runtime(
    *,
    rollout_pipeline: Any,
    conditional_dict: dict[str, Any],
    noise: Any,
    output: Any,
    mcp_depth: int,
) -> SelfForcingMCPRuntime:
    planner = build_planner(
        rollout_pipeline=rollout_pipeline,
        mcp_depth=mcp_depth,
        runtime_device_type=noise.device.type,
    )
    backend = SelfForcingWanMCPBackend(
        pipeline=rollout_pipeline,
        conditional_dict=conditional_dict,
        planner=planner,
    )
    runtime_config = SelfForcingMCPRuntimeConfig(
        anchor_denoising_steps=ANCHOR_DENOISING_STEPS,
        frame_seq_length=int(rollout_pipeline.frame_seq_length),
        num_frame_per_block=int(rollout_pipeline.num_frame_per_block),
        all_num_frames=int(noise.shape[1]),
        mcp_depth=int(mcp_depth),
        attention_mode=SUPPORTED_ATTENTION_MODE,
        validated_attention_mode=SUPPORTED_ATTENTION_MODE,
    )
    return SelfForcingMCPRuntime(
        config=runtime_config,
        backend=backend,
        source_noise=noise,
        output=output,
        kv_cache=rollout_pipeline.kv_cache1,
        cross_attention_cache=rollout_pipeline.crossattn_cache,
    )


def consume_reference_rollout_setup_rng(
    *,
    rollout_pipeline: Any,
    runtime: Any,
    device: Any,
    policy_name: str,
    reject_depth: Optional[int],
) -> ReferenceRNGPlan:
    num_denoising_steps = len(rollout_pipeline.denoising_step_list)
    if tuple(rollout_pipeline.denoising_step_list) != tuple(ANCHOR_DENOISING_STEPS):
        raise RuntimeError("Speculative F3 supports only denoising_step_list == [1000].")
    reference_mode, draw_count = reference_rng_mode_and_draw_count(
        rollout_plan=runtime.rollout_plan,
        policy_name=policy_name,
        reject_depth=reject_depth,
    )
    exit_flags = rollout_pipeline.generate_and_sync_list(
        draw_count,
        num_denoising_steps,
        device=device,
    )
    if len(exit_flags) != draw_count:
        raise RuntimeError(
            f"Reference RNG setup returned {len(exit_flags)} flags for "
            f"draw_count={draw_count}."
        )
    expected_flag = num_denoising_steps - 1
    if any(int(flag) != expected_flag for flag in exit_flags):
        raise RuntimeError(f"Unexpected rollout exit flags: {exit_flags}.")
    return ReferenceRNGPlan(
        mode=reference_mode,
        draw_count=draw_count,
        exit_flags=tuple(int(flag) for flag in exit_flags),
    )


def reference_rng_mode_and_draw_count(
    *,
    rollout_plan: Any,
    policy_name: str,
    reject_depth: Optional[int],
) -> tuple[str, int]:
    block_count = len(tuple(rollout_plan.blocks))
    if block_count <= 0:
        raise RuntimeError("Rollout plan has no blocks.")

    if policy_name == "always_accept":
        return "frozen_mcp_always_accept", len(tuple(rollout_plan.anchor_block_indices))
    if policy_name == "always_reject":
        return "vanilla_target_only", block_count
    if policy_name == "reject_at_depth":
        if reject_depth is None or reject_depth <= 0:
            raise ValueError("reject_at_depth requires a positive reject_depth.")
        return "scripted_reject_at_depth", (block_count + reject_depth) // (reject_depth + 1)
    raise ValueError(f"Unsupported policy for reference RNG setup: {policy_name!r}.")


def serialize_block(block: Optional[BlockRef]) -> Optional[dict[str, Optional[int]]]:
    if block is None:
        return None
    return {
        "index": block.index,
        "start_frame": block.start_frame,
        "num_frames": block.num_frames,
    }


def _optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if hasattr(value, "item"):
        return int(value.item())
    return int(value)


def _cache_indices(cache: Any, field: str) -> list[int]:
    if not cache:
        return []
    return [_optional_int(layer[field]) for layer in cache]  # type: ignore[list-item]


def _cache_trace_value(cache: Any, field: str) -> Any:
    values = _cache_indices(cache, field)
    if not values:
        return None
    first = values[0]
    if all(value == first for value in values):
        return first
    return values


def kv_index_summary(kv_cache: Any) -> dict[str, Any]:
    return {
        "global": _cache_trace_value(kv_cache, "global_end_index"),
        "local": _cache_trace_value(kv_cache, "local_end_index"),
    }


def _events_by_name(events: Sequence[Any], name: str) -> list[Any]:
    return [event for event in events if event.name == name]


def _drafts_from_result(request: ControlRequest, result: Any, runtime: Any) -> list[dict[str, Any]]:
    draft_depths: list[int] = []
    proposal_ready = _events_by_name(result.trace, "proposal_ready")
    if proposal_ready:
        draft_depths = [int(depth) for depth in proposal_ready[-1].metadata["draft_depths"]]
    decisions = {
        int(event.depth): event.decision
        for event in _events_by_name(result.trace, "decision")
        if event.depth is not None
    }
    rollout_blocks = tuple(runtime.rollout_plan.blocks)
    drafts = []
    for depth in draft_depths:
        block_index = request.anchor_block.index + depth
        block = rollout_blocks[block_index] if block_index < len(rollout_blocks) else None
        drafts.append(
            {
                "depth": depth,
                "block": serialize_block(block),
                "decision": decisions.get(depth),
            }
        )
    return drafts


def serialize_window(
    *,
    window_index: int,
    request: ControlRequest,
    result: Any,
    runtime: Any,
    kv_cache: Any,
) -> dict[str, Any]:
    committed_blocks = tuple(runtime.committed_blocks)
    rollout_blocks = tuple(runtime.rollout_plan.blocks)
    next_index = len(committed_blocks)
    next_anchor = rollout_blocks[next_index] if next_index < len(rollout_blocks) else None
    fallback_commits = [commit for commit in result.committed if commit.source == "fallback"]
    return {
        "window_index": int(window_index),
        "anchor_block": serialize_block(request.anchor_block),
        "max_depth": int(request.max_depth),
        "drafts": _drafts_from_result(request, result, runtime),
        "accepted_depth": result.accepted_depth,
        "rejected_depth": result.rejected_depth,
        "decisions": [
            event.to_dict()
            for event in _events_by_name(result.trace, "decision")
        ],
        "fallback": [
            {
                "block": serialize_block(commit.block),
                "depth": commit.depth,
                "source_noise_reused": True,
            }
            for commit in fallback_commits
        ],
        "invalidated": [
            {
                "block": serialize_block(candidate.block),
                "depth": candidate.depth,
            }
            for candidate in result.invalidated
        ],
        "commits": [
            {
                "block": serialize_block(commit.block),
                "source": commit.source,
                "depth": commit.depth,
            }
            for commit in result.committed
        ],
        "committed_block_sequence": [
            serialize_block(block)
            for block in committed_blocks
        ],
        "kv_after_window": kv_index_summary(kv_cache),
        "next_anchor": serialize_block(next_anchor),
        "controller_events": [event.to_dict() for event in result.trace],
    }


def build_trace_payload(
    *,
    args: argparse.Namespace,
    config: Any,
    load_mode: str,
    mcp_tensor_count: int,
    runtime: Any,
    windows: Sequence[dict[str, Any]],
    kv_cache: Any,
    reference_rng: ReferenceRNGPlan,
    expected_cache_index: Optional[int] = None,
) -> dict[str, Any]:
    final_kv = kv_index_summary(kv_cache)
    committed_blocks = tuple(runtime.committed_blocks)
    block_frames = int(config.num_frame_per_block)
    payload = {
        "schema_version": TRACE_SCHEMA_VERSION,
        "mode": MODE,
        "seed": int(args.seed),
        "num_frames": int(args.num_frames),
        "block_frames": block_frames,
        "mcp_depth": int(args.mcp_depth),
        "policy": args.policy,
        "reject_depth": args.reject_depth,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "load_mode": load_mode,
        "mcp_tensor_count": int(mcp_tensor_count),
        "anchor_denoising_steps": [int(step) for step in ANCHOR_DENOISING_STEPS],
        "reference_rng_mode": reference_rng.mode,
        "reference_rng_draw_count": reference_rng.draw_count,
        "rollout_setup_exit_flags": [int(flag) for flag in reference_rng.exit_flags],
        "windows": list(windows),
        "anchor_frames": [
            window["anchor_block"]["start_frame"]
            for window in windows
            if window["anchor_block"] is not None
        ],
        "commit_frames": [
            block.start_frame
            for block in committed_blocks
        ],
        "committed_blocks": [
            serialize_block(block)
            for block in committed_blocks
        ],
        "final_cache_global": final_kv["global"],
        "final_cache_local": final_kv["local"],
        "final_kv_index": final_kv,
        "expected_cache_index": expected_cache_index,
    }
    json.dumps(payload, allow_nan=False)
    return payload


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    validate_policy_args(args)
    helper_args = helper_args_from(args)

    import numpy as np
    import torch
    from model.ode_regression import ODERegression
    from utils.checkpoint import (
        MCP_COMPLETE_STRICT_RESTORE,
        is_mcp_state_key,
        load_state_dict_allowing_mcp_mismatch,
    )

    model = None
    rollout_pipeline = None
    runtime = None
    controller = None
    conditional_dict = None
    noise = None
    output = None
    trace_payload = None

    device = require_single_gpu_runtime(torch, args.device)
    reset_runtime_seed(args.seed, np, torch)
    torch.set_grad_enabled(False)

    config = merge_config(args.config)
    config.generator_ckpt = args.checkpoint
    config.gradient_checkpointing = False
    validate_speculative_config(config, args)

    try:
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
            conditional_dict = model.text_encoder(text_prompts=[args.prompt])

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

        output = torch.zeros_like(noise)
        runtime = build_runtime(
            rollout_pipeline=rollout_pipeline,
            conditional_dict=conditional_dict,
            noise=noise,
            output=output,
            mcp_depth=int(args.mcp_depth),
        )
        controller = build_controller(runtime, build_policy(args))

        reset_runtime_seed(args.seed, np, torch)
        reference_rng = consume_reference_rollout_setup_rng(
            rollout_pipeline=rollout_pipeline,
            runtime=runtime,
            device=noise.device,
            policy_name=args.policy,
            reject_depth=args.reject_depth,
        )
        windows = run_speculative_rollout(
            runtime=runtime,
            controller=controller,
            mcp_depth=int(args.mcp_depth),
            kv_cache=rollout_pipeline.kv_cache1,
        )

        if len(runtime.committed_blocks) != len(runtime.rollout_plan.blocks):
            raise RuntimeError("Speculative rollout ended before all blocks were committed.")
        validate_latent(output, noise, torch)

        final_cache_global = require_consistent_cache_index(
            rollout_pipeline.kv_cache1,
            "global_end_index",
        )
        final_cache_local = require_consistent_cache_index(
            rollout_pipeline.kv_cache1,
            "local_end_index",
        )
        expected_cache_index = int(args.num_frames) * int(rollout_pipeline.frame_seq_length)
        if final_cache_global != expected_cache_index:
            raise RuntimeError(
                f"final_cache_global {final_cache_global} does not equal expected "
                f"{expected_cache_index}."
            )
        if final_cache_local != expected_cache_index:
            raise RuntimeError(
                f"final_cache_local {final_cache_local} does not equal expected "
                f"{expected_cache_index}."
            )

        if args.save_trace:
            trace_payload = build_trace_payload(
                args=args,
                config=config,
                load_mode=load_mode,
                mcp_tensor_count=mcp_tensor_count,
                runtime=runtime,
                windows=windows,
                kv_cache=rollout_pipeline.kv_cache1,
                reference_rng=reference_rng,
                expected_cache_index=expected_cache_index,
            )

        latent = output.detach().to("cpu")
        controller = None
        runtime = None
        rollout_pipeline.generator = None
        rollout_pipeline = None
        conditional_dict = None
        noise = None
        output = None
        model.generator.to("cpu")
        model.text_encoder.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()

        model.vae.eval().requires_grad_(False)
        model.vae.to(device=device, dtype=torch.bfloat16)
        latent = latent.to(device=device, dtype=torch.bfloat16)
        with torch.no_grad():
            pixels = model.vae.decode_to_pixel(latent, use_cache=False)
        validate_pixels(pixels, torch)

        save_video(args.output, pixels, args.fps)
        print(f"mode={MODE}")
        print(f"acceptance_policy={args.policy}")
        print(f"saved_output={Path(args.output).resolve()}")

        if args.save_trace:
            write_trace(args.save_trace, trace_payload)
    finally:
        try:
            if model is not None:
                if hasattr(model, "generator"):
                    model.generator.to("cpu")
                if hasattr(model, "text_encoder"):
                    model.text_encoder.to("cpu")
        finally:
            controller = None
            runtime = None
            rollout_pipeline = None
            conditional_dict = None
            output = None
            noise = None
            gc.collect()
            if "torch" in locals() and torch.cuda.is_available():
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
