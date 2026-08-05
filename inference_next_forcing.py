from __future__ import annotations

import argparse
import gc
import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from utils.nf_sf_m3 import M3_CHUNK_FRAMES, move_tensors_to_device
from utils.nf_sf_m6 import (
    M6_WAN_FRAME_SEQ_LENGTH,
    M6OracleRuntime,
    build_common_inputs,
    canonical_json_sha256,
    compare_latents,
    conditioning_json_summary,
    current_git_head,
    file_sha256,
    finalize_oracle_gate,
    load_oracle_checkpoint,
    oracle_stdout_payload,
    resolve_m6_schedule,
    run_main_only_oracle,
    select_m6_teacher_sample,
    validate_json_payload,
    validate_oracle_a_artifact_dir,
    write_oracle_artifacts,
)

TAP_LAYERS = (3, 11, 19, 29)
M6_MCP_MODULE_COUNT = 3


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="NF-SF v1 M6.0 controlled four-step Oracle A/B entry."
    )
    parser.add_argument("--oracle", required=True, choices=("A", "B"))
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--teacher_manifest", required=True, type=Path)
    parser.add_argument("--dataset_root", required=True, type=Path)
    parser.add_argument("--sample_index", type=int, default=None)
    parser.add_argument("--sample_id", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--split_index", type=int, default=None)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bf16", "float32"), default="bf16")
    parser.add_argument("--decode", action="store_true")
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--tolerance", type=float, default=None)
    parser.add_argument(
        "--oracle_a_dir",
        type=Path,
        default=None,
        help="Optional Oracle A output directory for Oracle B latent comparison.",
    )
    return parser.parse_args(argv)


def dtype_from_arg(value: str) -> torch.dtype:
    if value == "bf16":
        return torch.bfloat16
    if value == "float32":
        return torch.float32
    raise ValueError(f"unsupported dtype: {value}")


def merge_config(config_path: Path) -> Any:
    from omegaconf import OmegaConf

    default_config = OmegaConf.load("configs/default_config.yaml")
    run_config = OmegaConf.load(config_path)
    return OmegaConf.merge(default_config, run_config)


def resolved_config_dict(config: Any) -> dict[str, Any]:
    from omegaconf import OmegaConf

    return OmegaConf.to_container(config, resolve=True)


def validate_cli_config(config: Any) -> None:
    if bool(config.i2v):
        raise ValueError("M6.0 Oracle A/B supports T2V only")
    if int(config.num_frame_per_block) != M3_CHUNK_FRAMES:
        raise ValueError("M6.0 Oracle A/B requires chunk_frames=3")
    if int(config.mcp_num_modules) not in (0, M6_MCP_MODULE_COUNT):
        raise ValueError("M6.0 Oracle B formal restore expects three MCP modules")


def validate_oracle_b_cli_contract(args: argparse.Namespace) -> None:
    if args.oracle != "B":
        return
    if args.oracle_a_dir is None:
        raise ValueError("Oracle B requires --oracle_a_dir before output_dir creation")
    if args.tolerance is None:
        raise ValueError("Oracle B requires --tolerance before output_dir creation")
    if float(args.tolerance) < 0:
        raise ValueError("--tolerance must be non-negative")
    for name in ("oracle_trace.json", "oracle_summary.json", "output_latent.pt"):
        path = args.oracle_a_dir / name
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(f"Oracle B requires non-empty Oracle A artifact: {path}")


def runtime_device(device_arg: str) -> tuple[torch.device, dict[str, Any]]:
    if os.environ.get("WORLD_SIZE", "1") != "1":
        raise RuntimeError("M6.0 Oracle A/B requires WORLD_SIZE == 1")
    if device_arg != "cuda:0":
        raise RuntimeError("M6.0 Oracle A/B requires --device cuda:0")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    cuda_device_count = int(torch.cuda.device_count())
    if cuda_device_count != 1:
        raise RuntimeError(
            "M6.0 Oracle A/B requires torch.cuda.device_count() == 1, "
            f"actual={cuda_device_count}"
        )
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    return device, {
        "WORLD_SIZE": "1",
        "device": str(device),
        "cuda_available": True,
        "cuda_device_count": cuda_device_count,
        "cuda_set_device": 0,
        "runtime": "single_cuda0",
    }


def build_conditioning(
    *,
    prompt: str,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    from utils.nf_sf_m3 import move_tensors_to_cpu
    from utils.wan_wrapper import WanTextEncoder

    text_encoder = WanTextEncoder().eval().requires_grad_(False)
    text_encoder.to(device=device, dtype=dtype)
    try:
        with torch.no_grad():
            return move_tensors_to_cpu(text_encoder(text_prompts=[prompt]))
    finally:
        text_encoder.to("cpu")
        del text_encoder
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def build_generator(
    *,
    oracle: str,
    config: Any,
    checkpoint_record,
    device: torch.device,
    dtype: torch.dtype,
) -> Any:
    from utils.wan_wrapper import WanDiffusionWrapper

    model_kwargs = dict(getattr(config, "model_kwargs", {}))
    generator = WanDiffusionWrapper(**model_kwargs, is_causal=True)
    if oracle == "B":
        generator.add_mcp_modules(
            num_modules=M6_MCP_MODULE_COUNT,
            num_layers=int(config.mcp_num_layers),
            tap_layers=tuple(int(value) for value in config.mcp_tap_layers),
        )
    result = generator.load_state_dict(checkpoint_record.generator_state_dict, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(f"generator strict restore mismatch: {result}")
    generator.eval().requires_grad_(False)
    generator.to(device=device, dtype=dtype)
    return generator


def initialize_runtime(
    *,
    generator,
    config: Any,
    source_noise: torch.Tensor,
) -> M6OracleRuntime:
    from pipeline.self_forcing_training import SelfForcingTrainingPipeline

    scheduler = generator.get_scheduler()
    pipeline = SelfForcingTrainingPipeline(
        denoising_step_list=[1000, 750, 500, 250],
        scheduler=scheduler,
        generator=generator,
        num_frame_per_block=int(config.num_frame_per_block),
        independent_first_frame=False,
        same_step_across_blocks=False,
        last_step_only=False,
        num_max_frames=int(source_noise.shape[1]),
        context_noise=int(config.context_noise),
        mcp_num_modules=0,
        mcp_accel_depths=0,
    )
    pipeline._initialize_kv_cache(
        batch_size=int(source_noise.shape[0]),
        dtype=source_noise.dtype,
        device=source_noise.device,
    )
    pipeline._initialize_crossattn_cache(
        batch_size=int(source_noise.shape[0]),
        dtype=source_noise.dtype,
        device=source_noise.device,
    )
    return M6OracleRuntime(
        generator=generator,
        scheduler=scheduler,
        kv_cache=pipeline.kv_cache1,
        crossattn_cache=pipeline.crossattn_cache,
        frame_seq_length=int(pipeline.frame_seq_length),
        num_frame_per_block=int(pipeline.num_frame_per_block),
        context_noise=int(pipeline.context_noise),
    )


def save_decoded_video(
    *,
    latent: torch.Tensor,
    output_dir: Path,
    device: torch.device,
    dtype: torch.dtype,
    fps: int,
) -> dict[str, Any]:
    from utils.nf_sf_m3 import atomic_json_write
    from utils.wan_wrapper import WanVAEWrapper

    vae = WanVAEWrapper().eval().requires_grad_(False)
    vae.to(device=device, dtype=dtype)
    latent = latent.to(device=device, dtype=dtype)
    try:
        with torch.no_grad():
            pixels = vae.decode_to_pixel(latent, use_cache=False)
        frames = ((pixels.detach().cpu().float()[0].clamp(-1, 1) + 1.0) * 127.5)
        frames = frames.round().clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1)
        output_path = output_dir / "output.mp4"
        try:
            from torchvision.io import write_video

            write_video(str(output_path), frames, fps=fps, video_codec="libx264")
        except (ImportError, ModuleNotFoundError, RuntimeError, OSError):
            import imageio.v3 as iio

            iio.imwrite(output_path, frames.numpy(), fps=fps, codec="libx264")
        report = {
            "decode": True,
            "path": str(output_path.resolve()),
            "sha256": file_sha256(output_path),
            "fps": int(fps),
        }
        validate_json_payload(report)
        atomic_json_write(report, output_dir / "decode_summary.json")
        return report
    finally:
        vae.to("cpu")
        del vae
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    torch.set_grad_enabled(False)
    validate_oracle_b_cli_contract(args)

    device, device_runtime_contract = runtime_device(args.device)
    dtype = dtype_from_arg(args.dtype)
    git_sha = current_git_head()
    config = merge_config(args.config)
    validate_cli_config(config)
    schedule = resolve_m6_schedule(config)
    resolved_config = resolved_config_dict(config)
    resolved_config_sha = canonical_json_sha256(resolved_config)

    teacher_sample = select_m6_teacher_sample(
        teacher_manifest=args.teacher_manifest,
        dataset_root=args.dataset_root,
        sample_index=args.sample_index,
        sample_id=args.sample_id,
        split=args.split,
        split_index=args.split_index,
        reference_checkpoint_path=args.checkpoint if args.oracle == "A" else None,
    )
    conditioning_cpu = build_conditioning(
        prompt=str(teacher_sample.payload["prompt"]),
        device=device,
        dtype=dtype,
    )
    if teacher_sample.source_noise.dtype != dtype:
        raise ValueError(
            "requested --dtype must match stored teacher source_noise dtype; "
            f"requested={dtype}, stored={teacher_sample.source_noise.dtype}"
        )
    source_noise = teacher_sample.source_noise.to(device=device)
    conditional_dict = move_tensors_to_device(
        conditioning_cpu,
        device=device,
        floating_dtype=dtype,
    )
    conditioning_summary = conditioning_json_summary(conditional_dict)
    common_inputs, common_fingerprint = build_common_inputs(
        teacher_metadata=teacher_sample.metadata,
        teacher_payload=teacher_sample.payload,
        source_noise=source_noise,
        conditioning_summary=conditioning_summary,
        schedule=schedule,
        rollout_seed=int(teacher_sample.payload["rollout_seed"]),
        context_noise=int(config.context_noise),
        chunk_frames=int(config.num_frame_per_block),
        frame_seq_length=M6_WAN_FRAME_SEQ_LENGTH,
        device_runtime_contract=device_runtime_contract,
        resolved_config_canonical_sha256=resolved_config_sha,
        runtime_git_sha=git_sha,
    )
    oracle_a_artifacts = None
    if args.oracle == "B":
        assert args.oracle_a_dir is not None
        oracle_a_artifacts = validate_oracle_a_artifact_dir(
            args.oracle_a_dir,
            expected_common_inputs_fingerprint_sha256=common_fingerprint,
        )

    expected_official_sha = teacher_sample.metadata.get("generation_source", {}).get(
        "checkpoint_sha256"
    )
    checkpoint = load_oracle_checkpoint(
        path=args.checkpoint,
        oracle_kind=args.oracle,
        expected_official_sha256=expected_official_sha if args.oracle == "A" else None,
    )
    generator = build_generator(
        oracle=args.oracle,
        config=config,
        checkpoint_record=checkpoint,
        device=device,
        dtype=dtype,
    )
    runtime = initialize_runtime(
        generator=generator,
        config=config,
        source_noise=source_noise,
    )
    result = run_main_only_oracle(
        oracle_kind=args.oracle,
        runtime=runtime,
        source_noise=source_noise,
        teacher_payload=teacher_sample.payload,
        teacher_metadata=teacher_sample.metadata,
        conditional_dict=conditional_dict,
        schedule=schedule,
        checkpoint=checkpoint,
        git_sha=git_sha,
        resolved_config_canonical_sha256=resolved_config_sha,
        device_runtime_contract=device_runtime_contract,
        expected_common_inputs=common_inputs,
        tolerance=args.tolerance,
    )

    comparison = None
    if args.oracle == "B":
        assert oracle_a_artifacts is not None
        comparison = compare_latents(
            result.latent,
            oracle_a_artifacts.latent,
            tolerance=args.tolerance,
            chunk_frames=M3_CHUNK_FRAMES,
        )
    result = finalize_oracle_gate(
        result,
        oracle_a_comparison=comparison,
    )

    resolved_payload = {
        "config": resolved_config,
        "m6": {
            "schema": result.trace["schema"],
            "oracle_kind": args.oracle,
            "schedule": schedule.to_json(),
            "checkpoint": checkpoint.to_json(),
            "common_inputs": result.trace["common_inputs"],
            "common_inputs_fingerprint_sha256": result.trace[
                "common_inputs_fingerprint_sha256"
            ],
            "cli": {
                "config": str(args.config.resolve()),
                "checkpoint": str(args.checkpoint.resolve()),
                "teacher_manifest": str(args.teacher_manifest.resolve()),
                "dataset_root": str(args.dataset_root.resolve()),
                "sample_index": args.sample_index,
                "sample_id": args.sample_id,
                "split": args.split,
                "split_index": args.split_index,
                "device": str(device),
                "dtype": args.dtype,
                "decode": bool(args.decode),
                "tolerance": args.tolerance,
                "oracle_a_dir": None if args.oracle_a_dir is None else str(args.oracle_a_dir.resolve()),
            },
        },
    }
    artifact_hashes = write_oracle_artifacts(
        output_dir=args.output_dir,
        resolved_config=resolved_payload,
        result=result,
        oracle_comparison=comparison,
    )

    if args.decode:
        save_decoded_video(
            latent=result.latent,
            output_dir=args.output_dir,
            device=device,
            dtype=dtype,
            fps=args.fps,
        )

    print(
        json.dumps(
            oracle_stdout_payload(
                result=result,
                output_dir=args.output_dir,
                artifact_hashes=artifact_hashes,
                comparison=comparison,
            ),
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return 0 if result.summary["status"] in ("PASS", "REPORT_ONLY") else 1


if __name__ == "__main__":
    raise SystemExit(main())
