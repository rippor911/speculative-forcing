from __future__ import annotations

import argparse
import gc
import json
import os
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from utils.nf_sf_full_sequence_eval import (
    FULL_SEQUENCE_CHUNK_FRAMES,
    FULL_SEQUENCE_FRAME_COUNT,
    FULL_SEQUENCE_FRAME_SEQ_LENGTH,
    FULL_SEQUENCE_MCP_LAYERS,
    FULL_SEQUENCE_MCP_MODULES,
    FULL_SEQUENCE_TAP_LAYERS,
    MODE_OFFICIAL_MAIN,
    MODE_TRAINED_MAIN,
    MODE_TRAINED_MCP1,
    OFFICIAL_SELF_FORCING_CHECKPOINT_SHA256,
    TRAINING_CHECKPOINT_GIT_SHA,
    DeploymentCheckpointRecord,
    DeploymentResult,
    DeploymentRuntime,
    assert_common_input_fingerprints,
    build_common_inputs_record,
    build_comparison_report,
    build_eval_manifest,
    current_git_head,
    file_sha256,
    load_full_sequence_checkpoint_record,
    load_official_checkpoint_record,
    resolve_deployment_schedule,
    run_main_only_deployment,
    run_mcp1_deployment,
    validate_eval_artifact_identity,
    validate_repo_preflight,
    write_mode_outputs,
)
from utils.nf_sf_m3 import atomic_json_write, move_tensors_to_device
from utils.nf_sf_m4 import load_m4_sample_plan
from utils.nf_sf_m5_samples import M5TeacherSampleStore


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="NF-SF Full-Sequence Next-Forcing v1 deployment eval."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/self_forcing_dmd_mcp.yaml"),
    )
    parser.add_argument(
        "--official_checkpoint",
        type=Path,
        default=Path("checkpoints/self_forcing_dmd.pt"),
    )
    parser.add_argument("--full_sequence_checkpoint", required=True, type=Path)
    parser.add_argument("--sample_plan", required=True, type=Path)
    parser.add_argument("--teacher_manifest", required=True, type=Path)
    parser.add_argument("--dataset_root", required=True, type=Path)
    parser.add_argument("--sample_identity", action="append", default=None)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--expected_runtime_git_sha", required=True)
    parser.add_argument(
        "--expected_training_git_sha",
        default=TRAINING_CHECKPOINT_GIT_SHA,
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bf16",), default="bf16")
    parser.add_argument("--fps", type=int, default=16)
    return parser.parse_args(argv)


def dtype_from_arg(value: str) -> torch.dtype:
    if value != "bf16":
        raise ValueError("deployment evaluator supports only bf16")
    return torch.bfloat16


def merge_config(config_path: Path) -> Any:
    from omegaconf import OmegaConf

    default_config = OmegaConf.load("configs/default_config.yaml")
    run_config = OmegaConf.load(config_path)
    return OmegaConf.merge(default_config, run_config)


def runtime_device(device_arg: str) -> tuple[torch.device, dict[str, Any]]:
    if os.environ.get("WORLD_SIZE", "1") != "1":
        raise RuntimeError("deployment evaluator requires WORLD_SIZE=1")
    if str(device_arg) != "cuda:0":
        raise RuntimeError("deployment evaluator requires --device cuda:0")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != "0":
        raise RuntimeError("deployment evaluator requires CUDA_VISIBLE_DEVICES=0")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    return device, {
        "WORLD_SIZE": "1",
        "CUDA_VISIBLE_DEVICES": visible,
        "device": str(device),
        "cuda_available": True,
        "cuda_device_count": int(torch.cuda.device_count()),
        "dtype": "bf16",
        "runtime": "single_cuda0",
    }


def validate_cli_contract(args: argparse.Namespace, *, git_sha: str) -> dict[str, Any]:
    if git_sha != str(args.expected_runtime_git_sha):
        raise RuntimeError("current git SHA differs from --expected_runtime_git_sha")
    repo_preflight = validate_repo_preflight(
        expected_runtime_git_sha=str(args.expected_runtime_git_sha),
        output_dir=args.output_dir,
    )
    if args.config.resolve() != (
        Path.cwd() / "configs" / "self_forcing_dmd_mcp.yaml"
    ).resolve():
        raise RuntimeError("deployment evaluator requires canonical config path")
    if int(args.num_samples) <= 0:
        raise ValueError("--num_samples must be positive")
    identities = args.sample_identity or []
    requested_count = len(identities) if identities else int(args.num_samples)
    if requested_count != 1:
        raise RuntimeError("v1 deployment evaluator runs exactly one sample per invocation")
    prepare_output_dir(args.output_dir)
    return repo_preflight


def validate_config(config: Any) -> None:
    if bool(getattr(config, "i2v", False)):
        raise ValueError("deployment evaluator supports T2V only")
    if int(getattr(config, "num_frame_per_block", 0)) != FULL_SEQUENCE_CHUNK_FRAMES:
        raise ValueError("deployment evaluator requires chunk_frames=3")
    if int(getattr(config, "context_noise", -1)) < 0:
        raise ValueError("config.context_noise must be non-negative")


def prepare_output_dir(output_dir: Path) -> None:
    if output_dir.exists():
        if not output_dir.is_dir():
            raise RuntimeError("--output_dir exists and is not a directory")
        if any(output_dir.iterdir()):
            raise RuntimeError("--output_dir must be new or empty")
    output_dir.mkdir(parents=True, exist_ok=True)


def select_eval_identity(
    sample_plan: dict[str, Any],
    *,
    sample_identity: Sequence[str] | None,
    num_samples: int,
) -> str:
    if sample_identity:
        if len(sample_identity) != 1:
            raise RuntimeError("v1 deployment evaluator accepts exactly one --sample_identity")
        return str(sample_identity[0])
    if int(num_samples) != 1:
        raise RuntimeError("v1 deployment evaluator currently supports only --num_samples 1")
    return str(sample_plan["fixed_decode_validation_identity"])


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
        torch.cuda.empty_cache()


def build_generator(
    *,
    config: Any,
    checkpoint: DeploymentCheckpointRecord,
    mode: str,
    device: torch.device,
    dtype: torch.dtype,
) -> Any:
    from utils.wan_wrapper import WanDiffusionWrapper

    model_kwargs = dict(getattr(config, "model_kwargs", {}) or {})
    generator = WanDiffusionWrapper(
        **model_kwargs,
        is_causal=True,
        local_attn_size=int(getattr(config, "local_attn_size", -1)),
        sink_size=int(getattr(config, "sink_size", 0)),
    )
    if mode in (MODE_TRAINED_MAIN, MODE_TRAINED_MCP1):
        generator.add_mcp_modules(
            num_modules=FULL_SEQUENCE_MCP_MODULES,
            num_layers=FULL_SEQUENCE_MCP_LAYERS,
            tap_layers=FULL_SEQUENCE_TAP_LAYERS,
        )
    result = generator.load_state_dict(checkpoint.generator_state_dict, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(f"generator strict restore mismatch: {result}")
    generator.model.num_frame_per_block = FULL_SEQUENCE_CHUNK_FRAMES
    generator.eval().requires_grad_(False)
    generator.to(device=device, dtype=dtype)
    return generator


def initialize_runtime(
    *,
    generator: Any,
    config: Any,
    source_noise: torch.Tensor,
) -> DeploymentRuntime:
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
    return DeploymentRuntime(
        generator=generator,
        scheduler=scheduler,
        kv_cache=pipeline.kv_cache1,
        crossattn_cache=pipeline.crossattn_cache,
        frame_seq_length=int(pipeline.frame_seq_length),
        num_frame_per_block=int(pipeline.num_frame_per_block),
        context_noise=int(pipeline.context_noise),
    )


def build_mcp_scheduler(*, device: torch.device) -> Any:
    from utils.nf_sf_tensors import DEFAULT_NUM_TRAIN_TIMESTEPS, DEFAULT_S_MCP
    from utils.scheduler import FlowMatchScheduler

    scheduler = FlowMatchScheduler(
        shift=DEFAULT_S_MCP,
        sigma_min=0.0,
        extra_one_step=True,
    )
    scheduler.set_timesteps(DEFAULT_NUM_TRAIN_TIMESTEPS, training=True)
    scheduler.sigmas = scheduler.sigmas.to(device)
    scheduler.timesteps = scheduler.timesteps.to(device)
    return scheduler


def run_mode(
    *,
    mode: str,
    config: Any,
    checkpoint: DeploymentCheckpointRecord,
    source_noise: torch.Tensor,
    teacher_payload: dict[str, Any],
    teacher_metadata: dict[str, Any],
    conditional_dict: dict[str, Any],
    common_inputs: dict[str, Any],
    common_inputs_fingerprint_sha256: str,
    git_sha: str,
    device: torch.device,
    dtype: torch.dtype,
) -> DeploymentResult:
    generator = build_generator(
        config=config,
        checkpoint=checkpoint,
        mode=mode,
        device=device,
        dtype=dtype,
    )
    try:
        runtime = initialize_runtime(
            generator=generator,
            config=config,
            source_noise=source_noise,
        )
        if mode == MODE_TRAINED_MCP1:
            return run_mcp1_deployment(
                runtime=runtime,
                mcp_scheduler=build_mcp_scheduler(device=device),
                source_noise=source_noise,
                teacher_payload=teacher_payload,
                teacher_metadata=teacher_metadata,
                conditional_dict=conditional_dict,
                checkpoint=checkpoint,
                git_sha=git_sha,
                common_inputs=common_inputs,
                common_inputs_fingerprint_sha256=common_inputs_fingerprint_sha256,
            )
        return run_main_only_deployment(
            mode=mode,
            runtime=runtime,
            source_noise=source_noise,
            teacher_payload=teacher_payload,
            teacher_metadata=teacher_metadata,
            conditional_dict=conditional_dict,
            checkpoint=checkpoint,
            git_sha=git_sha,
            common_inputs=common_inputs,
            common_inputs_fingerprint_sha256=common_inputs_fingerprint_sha256,
        )
    finally:
        generator.to("cpu")
        del generator
        gc.collect()
        torch.cuda.empty_cache()


def decode_and_write_videos(
    *,
    latents: Mapping[str, torch.Tensor],
    output_dir: Path,
    device: torch.device,
    dtype: torch.dtype,
    fps: int,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    from utils.wan_wrapper import WanVAEWrapper

    vae = WanVAEWrapper().eval().requires_grad_(False)
    vae.to(device=device, dtype=dtype)
    frames_by_mode: dict[str, torch.Tensor] = {}
    elapsed_by_mode: dict[str, float] = {}
    total_start = time.perf_counter()
    try:
        for mode, latent in latents.items():
            mode_start = time.perf_counter()
            frames = decode_latent_to_video_frames(
                vae,
                latent,
                device=device,
                dtype=dtype,
            )
            mode_dir = output_dir / mode
            mode_dir.mkdir(parents=True, exist_ok=True)
            write_video_frames(mode_dir / "output.mp4", frames, fps=fps)
            frames_by_mode[mode] = frames.detach().cpu()
            elapsed_by_mode[mode] = (time.perf_counter() - mode_start) * 1000.0
    finally:
        vae.to("cpu")
        del vae
        gc.collect()
        torch.cuda.empty_cache()
    return frames_by_mode, {
        "runtime_measurement_status": "SANITY_ONLY_NOT_BENCHMARK",
        "decode_elapsed_ms": (time.perf_counter() - total_start) * 1000.0,
        "decode_elapsed_ms_by_mode": elapsed_by_mode,
    }


def decode_latent_to_video_frames(
    vae: Any,
    latent: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    latent = latent.to(device=device, dtype=dtype)
    with torch.no_grad():
        pixels = vae.decode_to_pixel(latent, use_cache=False)
    frames = ((pixels.detach().cpu().float()[0].clamp(-1, 1) + 1.0) * 127.5)
    return frames.round().clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1)


def write_video_frames(output_path: Path, frames: torch.Tensor, *, fps: int) -> None:
    try:
        from torchvision.io import write_video

        write_video(str(output_path), frames, fps=fps, video_codec="libx264")
    except (ImportError, ModuleNotFoundError, RuntimeError, OSError):
        import imageio.v3 as iio

        iio.imwrite(output_path, frames.numpy(), fps=fps, codec="libx264")


def run_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    torch.set_grad_enabled(False)
    git_sha = current_git_head()
    repo_preflight = validate_cli_contract(args, git_sha=git_sha)
    device, runtime_contract = runtime_device(args.device)
    dtype = dtype_from_arg(args.dtype)
    config = merge_config(args.config)
    validate_config(config)
    schedule = resolve_deployment_schedule()

    sample_plan = load_m4_sample_plan(
        args.sample_plan,
        manifest_path=args.teacher_manifest,
    )
    sample_identity = select_eval_identity(
        sample_plan,
        sample_identity=args.sample_identity,
        num_samples=int(args.num_samples),
    )
    teacher_manifest_sha256 = file_sha256(args.teacher_manifest)
    full_checkpoint = load_full_sequence_checkpoint_record(
        args.full_sequence_checkpoint,
        expected_training_git_sha=str(args.expected_training_git_sha),
    )
    if full_checkpoint.payload is None:
        raise RuntimeError("full-sequence checkpoint payload missing after validation")
    artifact_identity = validate_eval_artifact_identity(
        sample_plan=sample_plan,
        teacher_manifest_sha256=teacher_manifest_sha256,
        checkpoint_payload=full_checkpoint.payload,
        selected_identity=sample_identity,
    )
    teacher_store = M5TeacherSampleStore(
        sample_plan=sample_plan,
        manifest_path=args.teacher_manifest,
        dataset_root=args.dataset_root,
        reference_checkpoint_path=None,
        expected_reference_sha256=OFFICIAL_SELF_FORCING_CHECKPOINT_SHA256,
    )
    with teacher_store.acquire(sample_identity) as teacher_sample:
        teacher_payload = dict(teacher_sample.payload)
        teacher_metadata = dict(teacher_sample.metadata)
        source_noise_cpu = teacher_sample.source_noise.detach().cpu()

    if int(source_noise_cpu.shape[1]) != FULL_SEQUENCE_FRAME_COUNT:
        raise RuntimeError("selected teacher sample must contain 21 latent frames")
    conditioning_cpu = build_conditioning(
        prompt=str(teacher_payload["prompt"]),
        device=device,
        dtype=dtype,
    )
    if source_noise_cpu.dtype != dtype:
        raise RuntimeError(
            "requested dtype must match stored teacher source_noise dtype; "
            f"requested={dtype}, stored={source_noise_cpu.dtype}"
        )
    source_noise = source_noise_cpu.to(device=device)
    teacher_payload["source_noise"] = source_noise
    conditional_dict = move_tensors_to_device(
        conditioning_cpu,
        device=device,
        floating_dtype=dtype,
    )
    common_inputs, common_fingerprint = build_common_inputs_record(
        sample_identity=sample_identity,
        teacher_metadata=teacher_metadata,
        teacher_payload=teacher_payload,
        source_noise=source_noise,
        conditioning=conditional_dict,
        runtime_git_sha=git_sha,
        training_checkpoint_git_sha=str(full_checkpoint.training_git_sha),
        fps=int(args.fps),
        sample_plan_sha256=str(artifact_identity["sample_plan_sha256"]),
        teacher_manifest_sha256=str(artifact_identity["teacher_manifest_sha256"]),
        selected_validation_position=int(
            artifact_identity["selected_validation_position"]
        ),
    )
    common_inputs.update(
        {
            "runtime_contract": runtime_contract,
            "repo_preflight": repo_preflight,
            "artifact_identity": artifact_identity,
            "config_path": str(args.config.resolve()),
            "sample_plan_path": str(args.sample_plan.resolve()),
            "teacher_manifest_path": str(args.teacher_manifest.resolve()),
            "dataset_root": str(args.dataset_root.resolve()),
            "frame_seq_length": FULL_SEQUENCE_FRAME_SEQ_LENGTH,
            "deployment_schedule_summary": schedule.to_json(include_mcp=True),
        }
    )
    common_fingerprint = assert_common_payload_fingerprint(common_inputs)
    atomic_json_write(common_inputs, args.output_dir / "common_inputs.json")

    official_checkpoint = load_official_checkpoint_record(args.official_checkpoint)
    official_result = run_mode(
        mode=MODE_OFFICIAL_MAIN,
        config=config,
        checkpoint=official_checkpoint,
        source_noise=source_noise,
        teacher_payload=teacher_payload,
        teacher_metadata=teacher_metadata,
        conditional_dict=conditional_dict,
        common_inputs=common_inputs,
        common_inputs_fingerprint_sha256=common_fingerprint,
        git_sha=git_sha,
        device=device,
        dtype=dtype,
    )
    del official_checkpoint
    torch.cuda.empty_cache()

    trained_main_result = run_mode(
        mode=MODE_TRAINED_MAIN,
        config=config,
        checkpoint=full_checkpoint,
        source_noise=source_noise,
        teacher_payload=teacher_payload,
        teacher_metadata=teacher_metadata,
        conditional_dict=conditional_dict,
        common_inputs=common_inputs,
        common_inputs_fingerprint_sha256=common_fingerprint,
        git_sha=git_sha,
        device=device,
        dtype=dtype,
    )
    trained_mcp1_result = run_mode(
        mode=MODE_TRAINED_MCP1,
        config=config,
        checkpoint=full_checkpoint,
        source_noise=source_noise,
        teacher_payload=teacher_payload,
        teacher_metadata=teacher_metadata,
        conditional_dict=conditional_dict,
        common_inputs=common_inputs,
        common_inputs_fingerprint_sha256=common_fingerprint,
        git_sha=git_sha,
        device=device,
        dtype=dtype,
    )

    results = {
        MODE_OFFICIAL_MAIN: official_result,
        MODE_TRAINED_MAIN: trained_main_result,
        MODE_TRAINED_MCP1: trained_mcp1_result,
    }
    frames, decode_timing = decode_and_write_videos(
        latents={mode: result.latent for mode, result in results.items()},
        output_dir=args.output_dir,
        device=device,
        dtype=dtype,
        fps=int(args.fps),
    )
    for mode, result in results.items():
        elapsed = float(decode_timing["decode_elapsed_ms_by_mode"][mode])
        result.trace["decode_elapsed_ms"] = elapsed
        result.summary["decode_elapsed_ms"] = elapsed
        result.trace["runtime_measurement_status"] = "SANITY_ONLY_NOT_BENCHMARK"
        result.summary["runtime_measurement_status"] = "SANITY_ONLY_NOT_BENCHMARK"
    mode_summaries = {
        mode: write_mode_outputs(
            mode_dir=args.output_dir / mode,
            result=result,
            video_path=args.output_dir / mode / "output.mp4",
            fps=int(args.fps),
        )
        for mode, result in results.items()
    }
    for summary in mode_summaries.values():
        video = summary.get("video")
        if isinstance(video, dict):
            video["size_bytes"] = Path(video["path"]).stat().st_size

    comparisons_dir = args.output_dir / "comparisons"
    comparisons_dir.mkdir(parents=True, exist_ok=True)
    comparisons = {
        "official_vs_trained_main": build_comparison_report(
            name="official_vs_trained_main",
            left_mode=MODE_OFFICIAL_MAIN,
            right_mode=MODE_TRAINED_MAIN,
            latent_left=official_result.latent,
            latent_right=trained_main_result.latent,
            pixel_left=frames[MODE_OFFICIAL_MAIN],
            pixel_right=frames[MODE_TRAINED_MAIN],
        ),
        "trained_main_vs_trained_mcp1": build_comparison_report(
            name="trained_main_vs_trained_mcp1",
            left_mode=MODE_TRAINED_MAIN,
            right_mode=MODE_TRAINED_MCP1,
            latent_left=trained_main_result.latent,
            latent_right=trained_mcp1_result.latent,
            pixel_left=frames[MODE_TRAINED_MAIN],
            pixel_right=frames[MODE_TRAINED_MCP1],
            role_map=trained_mcp1_result.summary["role_map"],
        ),
    }
    comparison_paths = {
        "official_vs_trained_main": comparisons_dir
        / "official_vs_trained_main.json",
        "trained_main_vs_trained_mcp1": comparisons_dir
        / "trained_main_vs_trained_mcp1.json",
    }
    for name, path in comparison_paths.items():
        atomic_json_write(comparisons[name], path)
    assert_common_input_fingerprints(mode_summaries)
    manifest = build_eval_manifest(
        common_inputs=common_inputs,
        common_inputs_fingerprint_sha256=common_fingerprint,
        mode_summaries=mode_summaries,
        comparisons=comparisons,
        output_dir=args.output_dir,
        git_sha=git_sha,
    )
    manifest["checkpoint_inputs"] = {
        "official": {
            "path": str(args.official_checkpoint.resolve()),
            "sha256": file_sha256(args.official_checkpoint),
        },
        "full_sequence": {
            "path": str(args.full_sequence_checkpoint.resolve()),
            "sha256": full_checkpoint.sha256,
        },
    }
    manifest["comparison_paths"] = {
        key: str(value.resolve()) for key, value in comparison_paths.items()
    }
    manifest["repo_preflight"] = repo_preflight
    manifest["artifact_identity"] = artifact_identity
    manifest["decode_timing"] = decode_timing
    atomic_json_write(manifest, args.output_dir / "eval_manifest.json")
    return manifest


def assert_common_payload_fingerprint(common_inputs: dict[str, Any]) -> str:
    from utils.nf_sf_full_sequence_eval import canonical_json_sha256

    return canonical_json_sha256(common_inputs)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = run_evaluation(args)
    print(
        json.dumps(
            {
                "schema": manifest["schema"],
                "status": manifest["status"],
                "output_dir": manifest["output_dir"],
                "common_inputs_fingerprint_sha256": manifest[
                    "common_inputs_fingerprint_sha256"
                ],
                "visual_review_status": manifest["visual_review_status"],
            },
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0 if manifest["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
