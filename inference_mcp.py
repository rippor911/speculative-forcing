import argparse
import gc
import json
import os
import random
from pathlib import Path


EXPECTED_MCP_TENSOR_COUNT = 172
ANCHOR_DENOISING_STEPS = [1000]
ACCEPTANCE_POLICY = "always_accept"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Single-GPU, single-prompt MCP accelerated rollout inference "
            "(always-accept; no verifier)."
        )
    )
    parser.add_argument("--config", required=True, help="Path to the run config YAML.")
    parser.add_argument("--checkpoint", required=True, help="Path to the generator checkpoint.")
    parser.add_argument("--prompt", required=True, help="Text prompt for T2V generation.")
    parser.add_argument("--output", required=True, help="Output MP4 path.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--num_frames", type=int, default=21, help="Number of latent frames.")
    parser.add_argument("--mcp_depth", type=int, choices=[1, 2, 3], default=3,
                        help="Number of MCP depths to deploy.")
    parser.add_argument("--disable_mcp", action="store_true",
                        help="Run vanilla rollout while loading the MCP-complete checkpoint.")
    parser.add_argument("--save_trace", default=None,
                        help="Optional path for a lightweight JSON trace.")
    parser.add_argument("--fps", type=int, default=16, help="Output MP4 frame rate.")
    parser.add_argument("--device", default="cuda", help="CUDA device, e.g. cuda or cuda:0.")
    return parser.parse_args()


def require_single_gpu_runtime(torch, device_arg):
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError("Distributed inference is not supported by inference_mcp.py.")

    device = torch.device(device_arg)
    if device.type != "cuda":
        raise RuntimeError("inference_mcp.py requires a CUDA device.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    if device.index is None:
        torch.cuda.set_device(0)
        return torch.device("cuda:0")

    torch.cuda.set_device(device)
    return device


def reset_runtime_seed(seed, np, torch):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def merge_config(config_path):
    from omegaconf import OmegaConf

    default_config = OmegaConf.load("configs/default_config.yaml")
    run_config = OmegaConf.load(config_path)
    return OmegaConf.merge(default_config, run_config)


def count_mcp_tensors(state_dict, is_mcp_state_key, torch):
    count = 0
    for key, value in state_dict.items():
        if is_mcp_state_key(key) and torch.is_tensor(value):
            count += 1
    return count


def validate_config(config, args):
    if args.num_frames <= 0:
        raise ValueError(f"--num_frames must be > 0, got {args.num_frames}.")
    if args.fps <= 0:
        raise ValueError(f"--fps must be > 0, got {args.fps}.")
    if not Path(args.checkpoint).is_file():
        raise ValueError(f"--checkpoint must be an existing file: {args.checkpoint}")

    if bool(getattr(config, "i2v", False)):
        raise ValueError("inference_mcp.py supports T2V only; config.i2v must be false.")

    block_frames = int(config.num_frame_per_block)
    if block_frames <= 0:
        raise ValueError(
            f"config.num_frame_per_block must be > 0, got {block_frames}."
        )
    if args.num_frames % block_frames != 0:
        raise ValueError(
            f"--num_frames ({args.num_frames}) must be divisible by "
            f"config.num_frame_per_block ({block_frames})."
        )

    configured_modules = int(getattr(config, "mcp_num_modules", 0))
    if configured_modules <= 0:
        raise ValueError(
            "config.mcp_num_modules must be > 0 so the MCP-complete checkpoint "
            "can be strictly restored."
        )
    if not args.disable_mcp and args.mcp_depth > configured_modules:
        raise ValueError(
            f"--mcp_depth ({args.mcp_depth}) must be <= config.mcp_num_modules "
            f"({configured_modules})."
        )


def validate_checkpoint_restore(model, load_helper, is_mcp_state_key, strict_mode, torch):
    load_mode = load_helper.last_load_mode
    if load_mode != strict_mode:
        raise RuntimeError(
            "Checkpoint did not restore a complete MCP generator strictly. "
            f"Expected load mode {strict_mode}, got {load_mode!r}."
        )

    mcp_tensor_count = count_mcp_tensors(
        model.generator.state_dict(),
        is_mcp_state_key,
        torch
    )
    if mcp_tensor_count != EXPECTED_MCP_TENSOR_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_MCP_TENSOR_COUNT} mcp.* tensors in the loaded "
            f"generator state_dict, found {mcp_tensor_count}."
        )

    return load_mode, mcp_tensor_count


def build_rollout_pipeline(config, model, args):
    from pipeline.self_forcing_training import SelfForcingTrainingPipeline

    if args.disable_mcp:
        rollout_mcp_modules = 0
        rollout_mcp_depths = 0
    else:
        rollout_mcp_modules = int(args.mcp_depth)
        rollout_mcp_depths = int(args.mcp_depth)

    return SelfForcingTrainingPipeline(
        denoising_step_list=ANCHOR_DENOISING_STEPS,
        scheduler=model.generator.get_scheduler(),
        generator=model.generator,
        num_frame_per_block=int(config.num_frame_per_block),
        independent_first_frame=False,
        same_step_across_blocks=False,
        last_step_only=True,
        num_max_frames=int(args.num_frames),
        context_noise=0,
        memory_gap_blocks=0,
        memory_gap_sample_mode="fixed",
        memory_gap_min_blocks=0,
        memory_gap_max_blocks=0,
        mcp_num_modules=rollout_mcp_modules,
        mcp_accel_depths=rollout_mcp_depths,
    )


def make_noise(args, device, torch):
    cpu_generator = torch.Generator(device="cpu")
    cpu_generator.manual_seed(args.seed)
    noise = torch.randn(
        [1, args.num_frames, 16, 60, 104],
        generator=cpu_generator,
        device="cpu",
        dtype=torch.float32,
    )
    noise = noise.to(dtype=torch.bfloat16)
    return noise.to(device=device)


def cache_indices(cache, field):
    if not cache:
        return []
    return [int(layer[field].item()) for layer in cache]


def cache_trace_value(cache, field):
    values = cache_indices(cache, field)
    if not values:
        return None
    first = values[0]
    if all(value == first for value in values):
        return first
    return values


def require_consistent_cache_index(cache, field):
    values = cache_indices(cache, field)
    if not values:
        raise RuntimeError(f"KV cache has no {field} values.")
    first = values[0]
    if any(value != first for value in values):
        raise RuntimeError(f"KV cache {field} differs across layers: {values}")
    return first


def expected_anchor_frames(num_frames, block_frames, mcp_depth):
    block_starts = list(range(0, num_frames, block_frames))
    if mcp_depth <= 0:
        return block_starts
    period = mcp_depth + 1
    return block_starts[::period]


def expected_commit_frames(num_frames, block_frames):
    return list(range(0, num_frames, block_frames))


def to_int_or_float(value):
    number = float(value)
    if number.is_integer():
        return int(number)
    return number


def serialize_timestep(timestep):
    if timestep is None:
        return None
    if hasattr(timestep, "detach"):
        values = timestep.detach().flatten().cpu().unique(sorted=True).tolist()
        values = [to_int_or_float(value) for value in values]
        if len(values) == 1:
            return values[0]
        return values
    return to_int_or_float(timestep)


def optional_int(value):
    if value is None:
        return None
    if hasattr(value, "item"):
        return int(value.item())
    return int(value)


def serialize_optional_int_list(values):
    if values is None:
        return None
    return [optional_int(value) for value in values]


def install_generator_trace_hooks(generator, frame_seq_length):
    events = []
    pending_events = []

    def pre_hook(module, hook_args, kwargs):
        current_start = optional_int(kwargs.get("current_start"))
        kv_cache = kwargs.get("kv_cache")
        event = {
            "current_start": current_start,
            "start_frame": None if current_start is None else current_start // frame_seq_length,
            "timestep": serialize_timestep(kwargs.get("timestep")),
            "mcp_requested": kwargs.get("mcp_future_noises") is not None,
            "mcp_future_start_frames": serialize_optional_int_list(
                kwargs.get("mcp_future_start_frames")
            ),
            "pre_global": cache_trace_value(kv_cache, "global_end_index"),
            "pre_local": cache_trace_value(kv_cache, "local_end_index"),
        }
        pending_events.append(event)

    def post_hook(module, hook_args, kwargs, output):
        if pending_events:
            event = pending_events.pop()
        else:
            event = {
                "current_start": optional_int(kwargs.get("current_start")),
                "start_frame": None,
                "timestep": serialize_timestep(kwargs.get("timestep")),
                "mcp_requested": kwargs.get("mcp_future_noises") is not None,
                "mcp_future_start_frames": serialize_optional_int_list(
                    kwargs.get("mcp_future_start_frames")
                ),
                "pre_global": None,
                "pre_local": None,
            }
            if event["current_start"] is not None:
                event["start_frame"] = event["current_start"] // frame_seq_length

        kv_cache = kwargs.get("kv_cache")
        event["post_global"] = cache_trace_value(kv_cache, "global_end_index")
        event["post_local"] = cache_trace_value(kv_cache, "local_end_index")
        returned_depth_count = 0
        if isinstance(output, tuple) and len(output) >= 3:
            returned_depth_count = len(output[2])
        event["returned_depth_count"] = int(returned_depth_count)
        events.append(event)

    pre_handle = generator.register_forward_pre_hook(pre_hook, with_kwargs=True)
    post_handle = generator.register_forward_hook(post_hook, with_kwargs=True)
    return [pre_handle, post_handle], events


def observed_frames_for_timestep(events, timestep):
    return [
        int(event["start_frame"])
        for event in events
        if event["start_frame"] is not None and event["timestep"] == timestep
    ]


def mcp_calls_from_events(events):
    calls = []
    for event in events:
        if not event["mcp_requested"]:
            continue
        calls.append({
            "anchor": event["start_frame"],
            "future_starts": event["mcp_future_start_frames"],
            "returned_depth_count": event["returned_depth_count"],
        })
    return calls


def validate_rollout_events(events, args, block_frames, frame_seq_length):
    effective_mcp_depth = 0 if args.disable_mcp else int(args.mcp_depth)
    expected_anchors = expected_anchor_frames(
        args.num_frames,
        block_frames,
        effective_mcp_depth,
    )
    expected_commits = expected_commit_frames(args.num_frames, block_frames)
    observed_anchors = observed_frames_for_timestep(events, 1000)
    observed_commits = observed_frames_for_timestep(events, 0)

    if observed_anchors != expected_anchors:
        raise RuntimeError(
            f"Observed anchor_frames {observed_anchors} do not match expected "
            f"{expected_anchors}."
        )
    if observed_commits != expected_commits:
        raise RuntimeError(
            f"Observed commit_frames {observed_commits} do not match expected "
            f"{expected_commits}."
        )

    mcp_calls = mcp_calls_from_events(events)
    if args.disable_mcp:
        if mcp_calls:
            raise RuntimeError(f"Vanilla rollout unexpectedly requested MCP: {mcp_calls}")
    else:
        num_blocks = args.num_frames // block_frames
        expected_mcp_call_anchors = [
            anchor
            for anchor in expected_anchors
            if (num_blocks - anchor // block_frames - 1) > 0
        ]
        observed_mcp_call_anchors = [call["anchor"] for call in mcp_calls]
        if observed_mcp_call_anchors != expected_mcp_call_anchors:
            raise RuntimeError(
                f"Observed MCP call anchors {observed_mcp_call_anchors} do not "
                f"match expected {expected_mcp_call_anchors}."
            )
        for call in mcp_calls:
            anchor_block = int(call["anchor"]) // block_frames
            remaining_future_blocks = num_blocks - anchor_block - 1
            expected_depth_count = min(int(args.mcp_depth), remaining_future_blocks)
            if call["returned_depth_count"] != expected_depth_count:
                raise RuntimeError(
                    f"MCP call at anchor {call['anchor']} returned "
                    f"{call['returned_depth_count']} depths, expected "
                    f"{expected_depth_count}."
                )

    expected_cache_index = args.num_frames * frame_seq_length
    return {
        "anchor_frames": observed_anchors,
        "commit_frames": observed_commits,
        "expected_anchor_frames": expected_anchors,
        "expected_commit_frames": expected_commits,
        "mcp_calls": mcp_calls,
        "expected_cache_index": expected_cache_index,
    }


def validate_latent(latent, noise, torch):
    if tuple(latent.shape) != tuple(noise.shape):
        raise RuntimeError(
            f"latent.shape {tuple(latent.shape)} does not match noise.shape "
            f"{tuple(noise.shape)}."
        )
    if not bool(torch.isfinite(latent).all().item()):
        raise RuntimeError("Rollout latent contains non-finite values.")


def validate_pixels(pixels, torch):
    if pixels.ndim != 5:
        raise RuntimeError(f"Decoded pixels must be 5-D, got shape {tuple(pixels.shape)}.")
    if pixels.shape[0] != 1:
        raise RuntimeError(f"Decoded pixels batch size must be 1, got {pixels.shape[0]}.")
    if pixels.shape[2] != 3:
        raise RuntimeError(f"Decoded pixels channel count must be 3, got {pixels.shape[2]}.")
    if not bool(torch.isfinite(pixels).all().item()):
        raise RuntimeError("Decoded pixels contain non-finite values.")


def write_trace(path, trace):
    trace_path = Path(path)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    with trace_path.open("w", encoding="utf-8") as handle:
        json.dump(trace, handle, indent=2)
        handle.write("\n")


def save_video(output_path, pixels, fps):
    import torch
    from torchvision.io import write_video

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    video = (pixels[0] * 0.5 + 0.5).clamp(0, 1)
    video = video.permute(0, 2, 3, 1).mul(255).round().to("cpu", dtype=pixels.dtype)
    video = video.to(dtype=torch.uint8)
    write_video(str(output), video, fps=fps)


def main():
    args = parse_args()

    import numpy as np
    import torch
    from model.ode_regression import ODERegression
    from utils.checkpoint import (
        MCP_COMPLETE_STRICT_RESTORE,
        is_mcp_state_key,
        load_state_dict_allowing_mcp_mismatch,
    )

    device = require_single_gpu_runtime(torch, args.device)

    reset_runtime_seed(args.seed, np, torch)
    torch.set_grad_enabled(False)

    config = merge_config(args.config)
    config.generator_ckpt = args.checkpoint
    config.gradient_checkpointing = False
    validate_config(config, args)

    model = ODERegression(config, device)
    load_mode, mcp_tensor_count = validate_checkpoint_restore(
        model=model,
        load_helper=load_state_dict_allowing_mcp_mismatch,
        is_mcp_state_key=is_mcp_state_key,
        strict_mode=MCP_COMPLETE_STRICT_RESTORE,
        torch=torch,
    )
    print(f"checkpoint_load_mode={load_mode}")
    print(f"mcp_tensor_count={mcp_tensor_count}")

    model.generator.eval().requires_grad_(False)
    model.text_encoder.eval().requires_grad_(False)
    model.generator.to(device=device, dtype=torch.bfloat16)
    model.text_encoder.to(device=device, dtype=torch.bfloat16)

    mode = "vanilla" if args.disable_mcp else "mcp"
    effective_mcp_depth = 0 if args.disable_mcp else int(args.mcp_depth)
    block_frames = int(config.num_frame_per_block)

    with torch.no_grad():
        conditional_dict = model.text_encoder(text_prompts=[args.prompt])
    rollout_pipeline = build_rollout_pipeline(config, model, args)
    if args.disable_mcp and rollout_pipeline.mcp_num_modules != 0:
        raise RuntimeError("Vanilla rollout must disable MCP modules.")
    noise = make_noise(args, device, torch)

    trace_hooks = []
    generator_events = []
    if args.save_trace:
        trace_hooks, generator_events = install_generator_trace_hooks(
            model.generator,
            rollout_pipeline.frame_seq_length,
        )

    reset_runtime_seed(args.seed, np, torch)
    try:
        with torch.no_grad():
            latent, _, _ = rollout_pipeline.inference_with_trajectory(
                noise=noise,
                **conditional_dict,
            )
    finally:
        for trace_hook in trace_hooks:
            trace_hook.remove()

    validate_latent(latent, noise, torch)
    observed_trace = None
    if args.save_trace:
        observed_trace = validate_rollout_events(
            events=generator_events,
            args=args,
            block_frames=block_frames,
            frame_seq_length=rollout_pipeline.frame_seq_length,
        )

    final_cache_global = require_consistent_cache_index(
        rollout_pipeline.kv_cache1,
        "global_end_index",
    )
    final_cache_local = require_consistent_cache_index(
        rollout_pipeline.kv_cache1,
        "local_end_index",
    )
    expected_cache_index = args.num_frames * rollout_pipeline.frame_seq_length
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

    latent = latent.detach().to("cpu")

    del conditional_dict
    del noise
    rollout_pipeline.generator = None
    del rollout_pipeline
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
    print(f"mode={mode}")
    print(f"acceptance_policy={ACCEPTANCE_POLICY}")
    print(f"saved_output={Path(args.output).resolve()}")

    if args.save_trace:
        trace = {
            "mode": mode,
            "seed": int(args.seed),
            "num_frames": int(args.num_frames),
            "block_frames": block_frames,
            "mcp_depth": effective_mcp_depth,
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "load_mode": load_mode,
            "mcp_tensor_count": int(mcp_tensor_count),
            "anchor_frames": observed_trace["anchor_frames"],
            "commit_frames": observed_trace["commit_frames"],
            "expected_anchor_frames": observed_trace["expected_anchor_frames"],
            "expected_commit_frames": observed_trace["expected_commit_frames"],
            "mcp_calls": observed_trace["mcp_calls"],
            "generator_events": generator_events,
            "final_cache_global": final_cache_global,
            "final_cache_local": final_cache_local,
            "acceptance_policy": ACCEPTANCE_POLICY,
            "anchor_denoising_steps": ANCHOR_DENOISING_STEPS,
        }
        write_trace(args.save_trace, trace)


if __name__ == "__main__":
    main()
