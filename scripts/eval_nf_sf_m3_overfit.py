from __future__ import annotations

import argparse
import gc
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inference_mcp import merge_config, require_single_gpu_runtime
from utils.nf_sf_m3 import (
    M3_CHUNK_FRAMES,
    M3_DEPTHS,
    M3_DEPTH_WEIGHTS,
    atomic_json_write,
    audit_parameter_changes,
    compare_loss_dicts,
    compare_probe_outputs,
    compare_serialized_probe_tensors,
    load_m3_checkpoint,
    load_m3_teacher_sample,
    m3_mode_from_checkpoint_payload,
    make_m3_probe,
    move_tensors_to_device,
    reconstruct_main_current,
    reconstruct_mcp1_next,
    reconstruction_metrics,
    run_m3_probe_forward,
    selected_state_to_device,
    serialize_noisy_batch,
    solver_schedule_to_json,
    validate_m3_checkpoint_pair,
    validate_m3_checkpoint_git_sha,
    validate_m3_eval_config_matches_checkpoint,
    validate_git_sha,
)
from utils.nf_sf_tensors import DEFAULT_S_MAIN
from utils.scheduler import FlowMatchScheduler


TAP_LAYERS = (3, 11, 19, 29)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="NF-SF v1 M3 eval-only fresh-process restore and decode."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--m3_checkpoint", required=True, type=Path)
    parser.add_argument("--initial_m3_checkpoint", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--dataset_root", type=Path, default=None)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bf16", "float32"), default=None)
    parser.add_argument("--restore_tolerance", type=float, default=None)
    parser.add_argument("--decode", action="store_true")
    parser.add_argument("--solver_steps", type=int, default=None)
    parser.add_argument("--allow_solver_override", action="store_true")
    parser.add_argument("--fps", type=int, default=16)
    return parser.parse_args()


def dtype_from_arg(value: str) -> torch.dtype:
    return torch.bfloat16 if value == "bf16" else torch.float32


def git_head() -> str:
    return validate_git_sha(
        subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip(),
        name="current_git_sha",
    )


def resolved_config_dict(config: Any) -> dict[str, Any]:
    from omegaconf import OmegaConf

    return OmegaConf.to_container(config, resolve=True)


def validate_config(config: Any) -> None:
    if bool(getattr(config, "i2v", False)):
        raise ValueError("NF-SF M3 eval supports T2V only")
    if int(getattr(config, "num_frame_per_block", 0)) != M3_CHUNK_FRAMES:
        raise ValueError("NF-SF M3 requires chunk_frames=3")
    if int(getattr(config, "mcp_num_modules", 0)) != len(M3_DEPTHS):
        raise ValueError("NF-SF M3 requires mcp_num_modules=3")
    if int(getattr(config, "mcp_num_layers", 0)) != 3:
        raise ValueError("NF-SF M3 requires mcp_num_layers=3")
    if tuple(int(x) for x in getattr(config, "mcp_tap_layers", ())) != TAP_LAYERS:
        raise ValueError("NF-SF M3 requires mcp_tap_layers=[3, 11, 19, 29]")
    if tuple(float(x) for x in getattr(config, "mcp_depth_weights", ())) != M3_DEPTH_WEIGHTS:
        raise ValueError("NF-SF M3 requires depth weights [0.5, 0.2, 0.1]")
    model_kwargs = getattr(config, "model_kwargs", {})
    if float(model_kwargs.get("timestep_shift", DEFAULT_S_MAIN)) != DEFAULT_S_MAIN:
        raise ValueError("NF-SF M3 requires main timestep shift s_main=5.0")


def load_generator_from_m3_checkpoint(
    *,
    config: Any,
    checkpoint_payload: dict[str, Any],
    device: torch.device,
    dtype: torch.dtype,
) -> Any:
    from utils.wan_wrapper import WanDiffusionWrapper

    model_kwargs = dict(getattr(config, "model_kwargs", {}))
    generator = WanDiffusionWrapper(**model_kwargs, is_causal=True)
    generator.add_mcp_modules(
        num_modules=len(M3_DEPTHS),
        num_layers=3,
        tap_layers=TAP_LAYERS,
    )
    result = generator.load_state_dict(checkpoint_payload["generator"], strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(f"M3 generator restore mismatch: {result}")
    generator.eval().requires_grad_(False)
    generator.to(device=device, dtype=dtype)
    return generator


def conditional_dict_for_checkpoint(
    *,
    payload: dict[str, Any],
    prompt: str,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    saved = payload.get("probe_prompt_embedding")
    if saved is not None:
        return move_tensors_to_device(saved, device=device, floating_dtype=dtype)

    from utils.wan_wrapper import WanTextEncoder

    text_encoder = WanTextEncoder().to(device=device, dtype=dtype).eval().requires_grad_(False)
    try:
        with torch.no_grad():
            return text_encoder([prompt])
    finally:
        text_encoder.to("cpu")
        del text_encoder
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def compare_sample_metadata(
    checkpoint_metadata: dict[str, Any],
    sample_metadata: dict[str, Any],
) -> None:
    for key in ("sample_index", "split", "split_index", "prompt", "latent_file_sha256"):
        if checkpoint_metadata.get(key) != sample_metadata.get(key):
            raise RuntimeError(f"selected sample metadata differs for {key!r}")
    checkpoint_target = checkpoint_metadata.get("target_latent", {})
    current_target = sample_metadata.get("target_latent", {})
    if checkpoint_target.get("sha256") != current_target.get("sha256"):
        raise RuntimeError("selected target_latent tensor SHA256 differs")


def load_optimizer_audit_for_checkpoint(checkpoint_path: Path) -> dict[str, Any]:
    audit_path = checkpoint_path.resolve().parent / "optimizer_audit.json"
    if not audit_path.is_file():
        raise FileNotFoundError(f"optimizer audit not found: {audit_path}")
    payload = json.loads(audit_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("optimizer_audit.json must contain a JSON object")
    return payload


def normalize_pixels(decoded: torch.Tensor) -> torch.Tensor:
    frames = decoded.detach().cpu().float()[0]
    frames = frames.clamp(-1, 1)
    frames = ((frames + 1.0) * 127.5).round().clamp(0, 255).to(torch.uint8)
    return frames.permute(0, 2, 3, 1).contiguous()


def save_video(path: Path, frames_thwc: torch.Tensor, fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    errors = []
    try:
        from torchvision.io import write_video

        write_video(str(path), frames_thwc, fps=fps, video_codec="libx264")
        return
    except Exception as exc:
        errors.append(f"torchvision: {type(exc).__name__}: {exc}")
    try:
        import imageio.v3 as iio

        iio.imwrite(path, frames_thwc.numpy(), fps=fps, codec="libx264")
        return
    except Exception as exc:
        errors.append(f"imageio: {type(exc).__name__}: {exc}")
    raise RuntimeError("video save failed: " + " | ".join(errors))


def block_pixel_span(block_index: int, total_pixel_frames: int) -> tuple[int, int]:
    if block_index == 0:
        start, end = 0, 9
    else:
        start = 12 * block_index - 3
        end = 12 * block_index + 9
    if start < 0 or end > total_pixel_frames or start >= end:
        raise RuntimeError(
            "VAE block span incompatible with decoded video: "
            f"block={block_index}, span=({start},{end}), total={total_pixel_frames}"
        )
    return start, end


def splice_chunk(full_latent: torch.Tensor, chunk: torch.Tensor, start_frame: int) -> torch.Tensor:
    result = full_latent.detach().cpu().clone()
    chunk_cpu = chunk.detach().cpu().to(dtype=result.dtype)
    end_frame = start_frame + chunk_cpu.shape[1]
    if result[:, start_frame:end_frame].shape != chunk_cpu.shape:
        raise ValueError("chunk shape does not match target splice span")
    result[:, start_frame:end_frame] = chunk_cpu
    return result


def decode_chunk_variant(
    *,
    vae: Any,
    full_target_latent: torch.Tensor,
    chunk: torch.Tensor,
    block_index: int,
    output_path: Path,
    device: torch.device,
    dtype: torch.dtype,
    fps: int,
) -> dict[str, Any]:
    full_latent = splice_chunk(
        full_target_latent,
        chunk,
        start_frame=block_index * M3_CHUNK_FRAMES,
    )
    with torch.no_grad():
        decoded = vae.decode_to_pixel(
            full_latent.to(device=device, dtype=dtype),
            use_cache=False,
        )
    frames = normalize_pixels(decoded)
    start, end = block_pixel_span(block_index, frames.shape[0])
    cropped = frames[start:end]
    save_video(output_path, cropped, fps=fps)
    return {
        "path": str(output_path),
        "block_index": int(block_index),
        "decoded_pixel_frames": int(frames.shape[0]),
        "saved_pixel_start": int(start),
        "saved_pixel_end": int(end),
        "saved_frames": int(cropped.shape[0]),
    }


def run_reconstructions(
    *,
    config: Any,
    initial_payload: dict[str, Any],
    final_payload: dict[str, Any],
    teacher_payload: dict[str, Any],
    conditional_dict: dict[str, Any],
    state,
    probe,
    device: torch.device,
    dtype: torch.dtype,
    solver_steps_override: int | None,
    allow_solver_override: bool,
) -> dict[str, Any]:
    initial_generator = load_generator_from_m3_checkpoint(
        config=config,
        checkpoint_payload=initial_payload,
        device=device,
        dtype=dtype,
    )
    try:
        initial_main = reconstruct_main_current(
            initial_generator,
            conditional_dict=conditional_dict,
            state=state,
            initial_noise=probe.noisy_batch.epsilon_main,
            teacher_payload=teacher_payload,
            solver_steps_override=solver_steps_override,
            allow_solver_override=allow_solver_override,
        )
        initial_mcp1 = reconstruct_mcp1_next(
            initial_generator,
            conditional_dict=conditional_dict,
            state=state,
            next_initial_noise=probe.noisy_batch.epsilon_depths[0],
            current_condition_noise=probe.noisy_batch.epsilon_main,
            teacher_payload=teacher_payload,
            solver_steps_override=solver_steps_override,
            allow_solver_override=allow_solver_override,
        )
    finally:
        initial_generator.to("cpu")
        del initial_generator
        gc.collect()
        torch.cuda.empty_cache()

    final_generator = load_generator_from_m3_checkpoint(
        config=config,
        checkpoint_payload=final_payload,
        device=device,
        dtype=dtype,
    )
    try:
        final_main = reconstruct_main_current(
            final_generator,
            conditional_dict=conditional_dict,
            state=state,
            initial_noise=probe.noisy_batch.epsilon_main,
            teacher_payload=teacher_payload,
            solver_steps_override=solver_steps_override,
            allow_solver_override=allow_solver_override,
        )
        final_mcp1 = reconstruct_mcp1_next(
            final_generator,
            conditional_dict=conditional_dict,
            state=state,
            next_initial_noise=probe.noisy_batch.epsilon_depths[0],
            current_condition_noise=probe.noisy_batch.epsilon_main,
            teacher_payload=teacher_payload,
            solver_steps_override=solver_steps_override,
            allow_solver_override=allow_solver_override,
        )
    finally:
        final_generator.to("cpu")
        del final_generator
        gc.collect()
        torch.cuda.empty_cache()

    return {
        "initial_main": initial_main.latent.detach().cpu(),
        "final_main": final_main.latent.detach().cpu(),
        "initial_mcp1": initial_mcp1.latent.detach().cpu(),
        "final_mcp1": final_mcp1.latent.detach().cpu(),
        "solver_schedule": solver_schedule_to_json(final_main.solver_schedule),
        "initial_main_solver_schedule": solver_schedule_to_json(
            initial_main.solver_schedule
        ),
        "initial_mcp1_solver_schedule": solver_schedule_to_json(
            initial_mcp1.solver_schedule
        ),
        "final_main_solver_schedule": solver_schedule_to_json(
            final_main.solver_schedule
        ),
        "final_mcp1_solver_schedule": solver_schedule_to_json(
            final_mcp1.solver_schedule
        ),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = require_single_gpu_runtime(torch, args.device)
    current_git_sha = git_head()
    final_payload = load_m3_checkpoint(args.m3_checkpoint)
    checkpoint_mode = m3_mode_from_checkpoint_payload(final_payload)
    validate_m3_checkpoint_git_sha(
        final_payload,
        current_git_sha=current_git_sha,
    )
    checkpoint_dtype = final_payload["resolved_config"].get("m3", {}).get("dtype", "bf16")
    dtype_name = args.dtype or str(checkpoint_dtype)
    dtype = dtype_from_arg("bf16" if dtype_name in ("bf16", "torch.bfloat16") else dtype_name)
    tolerance = args.restore_tolerance
    if tolerance is None:
        tolerance = 5.0e-3 if dtype is torch.bfloat16 else 1.0e-6

    config = merge_config(str(args.config))
    validate_config(config)
    current_model_config = resolved_config_dict(config)
    validate_m3_eval_config_matches_checkpoint(
        final_payload,
        current_model_config,
    )
    metadata = final_payload["selected_sample_metadata"]
    manifest_path = args.manifest or Path(metadata["manifest_path"])
    sample = load_m3_teacher_sample(
        manifest_path=manifest_path,
        dataset_root=args.dataset_root,
        sample_index=int(metadata["sample_index"]),
        reference_checkpoint_path=final_payload["reference_checkpoint"]["path"],
    )
    compare_sample_metadata(metadata, sample.metadata)

    state = selected_state_to_device(sample.selected_state, device=device, dtype=dtype)
    generator = load_generator_from_m3_checkpoint(
        config=config,
        checkpoint_payload=final_payload,
        device=device,
        dtype=dtype,
    )
    conditional_dict = conditional_dict_for_checkpoint(
        payload=final_payload,
        prompt=sample.metadata["prompt"],
        device=device,
        dtype=dtype,
    )
    try:
        scheduler_main = generator.get_scheduler()
        scheduler_main.sigmas = scheduler_main.sigmas.to(device)
        scheduler_main.timesteps = scheduler_main.timesteps.to(device)
        scheduler_mcp = FlowMatchScheduler(shift=10.0, sigma_min=0.0, extra_one_step=True)
        scheduler_mcp.set_timesteps(1000, training=True)
        scheduler_mcp.sigmas = scheduler_mcp.sigmas.to(device)
        scheduler_mcp.timesteps = scheduler_mcp.timesteps.to(device)

        rebuilt_probe = make_m3_probe(
            state,
            scheduler_main=scheduler_main,
            scheduler_mcp=scheduler_mcp,
            seed=int(final_payload["probe_seed"]),
        )
        probe_tensor_comparison = compare_serialized_probe_tensors(
            serialize_noisy_batch(rebuilt_probe.noisy_batch),
            final_payload["probe_tensors"],
        )
        actual_probe = run_m3_probe_forward(
            generator,
            conditional_dict=conditional_dict,
            noisy_batch=rebuilt_probe.noisy_batch,
        )
    finally:
        generator.to("cpu")
        del generator
        gc.collect()
        torch.cuda.empty_cache()

    saved_losses = dict(final_payload["probe_summary"]["probe_losses"])
    loss_comparison = compare_loss_dicts(actual_probe.losses, saved_losses)
    probe_output_comparison = compare_probe_outputs(
        actual_probe.outputs,
        final_payload["probe_outputs"],
    )
    restore_pass = (
        loss_comparison["max_abs_diff"] <= tolerance
        and probe_tensor_comparison["max_abs_diff"] == 0.0
        and probe_output_comparison["max_abs_diff"] <= tolerance
    )
    report = {
        "status": "PASS" if restore_pass else "FAIL",
        "m3_checkpoint": str(args.m3_checkpoint.resolve()),
        "current_git_sha": current_git_sha,
        "checkpoint_git_sha": final_payload["git_sha"],
        "global_step": int(final_payload["global_step"]),
        "dtype": str(dtype),
        "tolerance": tolerance,
        "saved_probe_losses": saved_losses,
        "restored_probe_losses": actual_probe.losses,
        "loss_comparison": loss_comparison,
        "probe_tensor_comparison": probe_tensor_comparison,
        "probe_output_comparison": probe_output_comparison,
        "selected_sample_metadata": sample.metadata,
    }
    atomic_json_write(report, args.output_dir / "restore_validation.json")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    if not restore_pass:
        raise SystemExit(2)

    initial_payload = None
    checkpoint_pair_report = None
    parameter_change_report = None
    if args.initial_m3_checkpoint is not None:
        initial_payload = load_m3_checkpoint(args.initial_m3_checkpoint)
        checkpoint_pair_report = validate_m3_checkpoint_pair(
            initial_payload=initial_payload,
            final_payload=final_payload,
            current_model_config=current_model_config,
            current_git_sha=current_git_sha,
        )
        optimizer_audit = load_optimizer_audit_for_checkpoint(args.m3_checkpoint)
        parameter_change_report = audit_parameter_changes(
            initial_state_dict=initial_payload["generator"],
            final_state_dict=final_payload["generator"],
            optimizer_audit=optimizer_audit,
            mode=checkpoint_mode,
        )
        atomic_json_write(
            {
                "status": parameter_change_report["status"],
                "m3_checkpoint": str(args.m3_checkpoint.resolve()),
                "initial_m3_checkpoint": str(args.initial_m3_checkpoint.resolve()),
                "mode": checkpoint_mode,
                "checkpoint_pair": checkpoint_pair_report,
                **parameter_change_report,
            },
            args.output_dir / "parameter_change_audit.json",
        )
        if not parameter_change_report["all_groups_match_mode_contract"]:
            raise RuntimeError(
                f"{checkpoint_mode} M3 parameter change audit contract failed"
            )

    if not args.decode:
        return
    if initial_payload is None:
        raise ValueError("--decode requires --initial_m3_checkpoint")
    assert checkpoint_pair_report is not None
    recon = run_reconstructions(
        config=config,
        initial_payload=initial_payload,
        final_payload=final_payload,
        teacher_payload=sample.payload,
        conditional_dict=conditional_dict,
        state=state,
        probe=rebuilt_probe,
        device=device,
        dtype=dtype,
        solver_steps_override=args.solver_steps,
        allow_solver_override=args.allow_solver_override,
    )
    metrics = reconstruction_metrics(
        initial_main=recon["initial_main"],
        final_main=recon["final_main"],
        initial_mcp1=recon["initial_mcp1"],
        final_mcp1=recon["final_mcp1"],
        state=sample.selected_state,
    )
    atomic_json_write(
        {
            "status": "PASS",
            "checkpoint_pair": checkpoint_pair_report,
            "parameter_change_audit": parameter_change_report,
            **metrics,
        },
        args.output_dir / "reconstruction_metrics.json",
    )

    decoded_dir = args.output_dir / "decoded"
    from utils.wan_wrapper import WanVAEWrapper

    vae = WanVAEWrapper().eval().requires_grad_(False)
    vae.to(device=device, dtype=dtype)
    try:
        decode_records = {
            "target_current": decode_chunk_variant(
                vae=vae,
                full_target_latent=sample.target_latent,
                chunk=sample.selected_state.current_target,
                block_index=1,
                output_path=decoded_dir / "target_current.mp4",
                device=device,
                dtype=dtype,
                fps=args.fps,
            ),
            "target_next1": decode_chunk_variant(
                vae=vae,
                full_target_latent=sample.target_latent,
                chunk=sample.selected_state.future_targets[0],
                block_index=2,
                output_path=decoded_dir / "target_next1.mp4",
                device=device,
                dtype=dtype,
                fps=args.fps,
            ),
            "initial_main": decode_chunk_variant(
                vae=vae,
                full_target_latent=sample.target_latent,
                chunk=recon["initial_main"],
                block_index=1,
                output_path=decoded_dir / "initial_main.mp4",
                device=device,
                dtype=dtype,
                fps=args.fps,
            ),
            "final_main": decode_chunk_variant(
                vae=vae,
                full_target_latent=sample.target_latent,
                chunk=recon["final_main"],
                block_index=1,
                output_path=decoded_dir / "final_main.mp4",
                device=device,
                dtype=dtype,
                fps=args.fps,
            ),
            "initial_mcp1": decode_chunk_variant(
                vae=vae,
                full_target_latent=sample.target_latent,
                chunk=recon["initial_mcp1"],
                block_index=2,
                output_path=decoded_dir / "initial_mcp1.mp4",
                device=device,
                dtype=dtype,
                fps=args.fps,
            ),
            "final_mcp1": decode_chunk_variant(
                vae=vae,
                full_target_latent=sample.target_latent,
                chunk=recon["final_mcp1"],
                block_index=2,
                output_path=decoded_dir / "final_mcp1.mp4",
                device=device,
                dtype=dtype,
                fps=args.fps,
            ),
        }
    finally:
        vae.to("cpu")
        del vae
        gc.collect()
        torch.cuda.empty_cache()

    atomic_json_write(
        {
            "status": "PASS",
            "decode_mode": "target_context_splice",
            "solver_schedule": recon["solver_schedule"],
            "solver_override_allowed": bool(args.allow_solver_override),
            "solver_steps_override": args.solver_steps,
            "checkpoint_pair": checkpoint_pair_report,
            "parameter_change_audit": parameter_change_report,
            "records": decode_records,
        },
        decoded_dir / "decode_manifest.json",
    )


if __name__ == "__main__":
    main()
