from __future__ import annotations

import argparse
import gc
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from scripts.eval_nf_sf_full_sequence_deployment import (
    assert_common_payload_fingerprint,
    build_conditioning,
    build_generator,
    build_mcp_scheduler,
    decode_and_write_videos,
    dtype_from_arg,
    initialize_runtime,
    merge_config,
    runtime_device,
    select_eval_identity,
    validate_cli_contract,
    validate_config,
)
from utils.nf_sf_full_sequence_eval import (
    DIAG_MODE_MCP1_LIVE_HISTORY,
    DIAG_MODE_MCP1_MAIN_HISTORY_REPAIR,
    DIAG_MODE_MCP1_TEACHER_HISTORY_ORACLE,
    DIAG_MODE_TRAINED_MAIN_REFERENCE,
    FULL_SEQUENCE_FRAME_COUNT,
    FULL_SEQUENCE_FRAME_SEQ_LENGTH,
    MODE_TRAINED_MAIN,
    MODE_TRAINED_MCP1,
    OFFICIAL_SELF_FORCING_CHECKPOINT_SHA256,
    TRAINING_CHECKPOINT_GIT_SHA,
    DeploymentCheckpointRecord,
    DeploymentResult,
    assert_common_input_fingerprints,
    build_common_inputs_record,
    build_history_diagnostic_comparison,
    build_history_diagnostic_manifest,
    current_git_head,
    file_sha256,
    load_full_sequence_checkpoint_record,
    resolve_deployment_schedule,
    run_main_only_deployment,
    run_mcp1_history_intervention_deployment,
    tensor_sha256,
    validate_eval_artifact_identity,
    write_mode_outputs,
)
from utils.nf_sf_m3 import atomic_json_write, move_tensors_to_device
from utils.nf_sf_m4 import load_m4_sample_plan
from utils.nf_sf_m5_samples import M5TeacherSampleStore


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "NF-SF Full-Sequence history intervention diagnostic evaluator."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/self_forcing_dmd_mcp.yaml"),
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


def _run_diagnostic_mode(
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
    history_recache_tensor: torch.Tensor | None = None,
    history_source: str = "generated_output",
) -> DeploymentResult:
    restore_mode = MODE_TRAINED_MAIN
    if mode != DIAG_MODE_TRAINED_MAIN_REFERENCE:
        restore_mode = MODE_TRAINED_MCP1
    generator = build_generator(
        config=config,
        checkpoint=checkpoint,
        mode=restore_mode,
        device=device,
        dtype=dtype,
    )
    try:
        runtime = initialize_runtime(
            generator=generator,
            config=config,
            source_noise=source_noise,
        )
        if mode == DIAG_MODE_TRAINED_MAIN_REFERENCE:
            return run_main_only_deployment(
                mode=DIAG_MODE_TRAINED_MAIN_REFERENCE,
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
        return run_mcp1_history_intervention_deployment(
            mode=mode,  # type: ignore[arg-type]
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
            history_recache_tensor=history_recache_tensor,
            history_source=history_source,
        )
    finally:
        generator.to("cpu")
        del generator
        gc.collect()
        torch.cuda.empty_cache()


def _comparison(
    *,
    name: str,
    left_mode: str,
    right_mode: str,
    results: Mapping[str, DeploymentResult],
    frames: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    return build_history_diagnostic_comparison(
        name=name,
        left_mode=left_mode,
        right_mode=right_mode,
        latent_left=results[left_mode].latent,
        latent_right=results[right_mode].latent,
        pixel_left=frames.get(left_mode),
        pixel_right=frames.get(right_mode),
    )


def run_diagnostic(args: argparse.Namespace) -> dict[str, Any]:
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
        teacher_target_cpu = teacher_sample.target_latent.detach().cpu()

    if int(source_noise_cpu.shape[1]) != FULL_SEQUENCE_FRAME_COUNT:
        raise RuntimeError("selected teacher sample must contain 21 latent frames")
    if tuple(teacher_target_cpu.shape) != tuple(source_noise_cpu.shape):
        raise RuntimeError("teacher target_latent must match source_noise shape")
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
    teacher_target = teacher_target_cpu.to(device=device, dtype=dtype)
    teacher_target_sha256 = tensor_sha256(teacher_target_cpu)
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
            "diagnostic_schema": "nf_sf_full_sequence_history_intervention_diagnostic_v1",
            "diagnostic_only": True,
            "runtime_contract": runtime_contract,
            "repo_preflight": repo_preflight,
            "artifact_identity": artifact_identity,
            "config_path": str(args.config.resolve()),
            "sample_plan_path": str(args.sample_plan.resolve()),
            "teacher_manifest_path": str(args.teacher_manifest.resolve()),
            "dataset_root": str(args.dataset_root.resolve()),
            "frame_seq_length": FULL_SEQUENCE_FRAME_SEQ_LENGTH,
            "teacher_target_sha256": teacher_target_sha256,
            "deployment_schedule_summary": schedule.to_json(include_mcp=True),
            "history_interventions": {
                DIAG_MODE_MCP1_LIVE_HISTORY: "generated_output",
                DIAG_MODE_MCP1_MAIN_HISTORY_REPAIR: "trained_main_reference",
                DIAG_MODE_MCP1_TEACHER_HISTORY_ORACLE: "teacher_target",
            },
        }
    )
    common_fingerprint = assert_common_payload_fingerprint(common_inputs)
    atomic_json_write(common_inputs, args.output_dir / "common_inputs.json")

    trained_main_reference = _run_diagnostic_mode(
        mode=DIAG_MODE_TRAINED_MAIN_REFERENCE,
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
    live_history = _run_diagnostic_mode(
        mode=DIAG_MODE_MCP1_LIVE_HISTORY,
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
        history_recache_tensor=None,
        history_source="generated_output",
    )
    main_history_repair = _run_diagnostic_mode(
        mode=DIAG_MODE_MCP1_MAIN_HISTORY_REPAIR,
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
        history_recache_tensor=trained_main_reference.latent.to(device=device, dtype=dtype),
        history_source="trained_main_reference",
    )
    teacher_history_oracle = _run_diagnostic_mode(
        mode=DIAG_MODE_MCP1_TEACHER_HISTORY_ORACLE,
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
        history_recache_tensor=teacher_target,
        history_source="teacher_target",
    )

    results = {
        DIAG_MODE_TRAINED_MAIN_REFERENCE: trained_main_reference,
        DIAG_MODE_MCP1_LIVE_HISTORY: live_history,
        DIAG_MODE_MCP1_MAIN_HISTORY_REPAIR: main_history_repair,
        DIAG_MODE_MCP1_TEACHER_HISTORY_ORACLE: teacher_history_oracle,
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
    mode_summaries = {
        mode: write_mode_outputs(
            mode_dir=args.output_dir / mode,
            result=result,
            video_path=args.output_dir / mode / "output.mp4",
            fps=int(args.fps),
        )
        for mode, result in results.items()
    }

    comparisons_dir = args.output_dir / "comparisons"
    comparisons_dir.mkdir(parents=True, exist_ok=True)
    comparisons = {
        "live_vs_trained_main": _comparison(
            name="live_vs_trained_main",
            left_mode=DIAG_MODE_MCP1_LIVE_HISTORY,
            right_mode=DIAG_MODE_TRAINED_MAIN_REFERENCE,
            results=results,
            frames=frames,
        ),
        "main_history_repair_vs_trained_main": _comparison(
            name="main_history_repair_vs_trained_main",
            left_mode=DIAG_MODE_MCP1_MAIN_HISTORY_REPAIR,
            right_mode=DIAG_MODE_TRAINED_MAIN_REFERENCE,
            results=results,
            frames=frames,
        ),
        "teacher_history_oracle_vs_trained_main": _comparison(
            name="teacher_history_oracle_vs_trained_main",
            left_mode=DIAG_MODE_MCP1_TEACHER_HISTORY_ORACLE,
            right_mode=DIAG_MODE_TRAINED_MAIN_REFERENCE,
            results=results,
            frames=frames,
        ),
        "live_vs_main_history_repair": _comparison(
            name="live_vs_main_history_repair",
            left_mode=DIAG_MODE_MCP1_LIVE_HISTORY,
            right_mode=DIAG_MODE_MCP1_MAIN_HISTORY_REPAIR,
            results=results,
            frames=frames,
        ),
        "live_vs_teacher_history_oracle": _comparison(
            name="live_vs_teacher_history_oracle",
            left_mode=DIAG_MODE_MCP1_LIVE_HISTORY,
            right_mode=DIAG_MODE_MCP1_TEACHER_HISTORY_ORACLE,
            results=results,
            frames=frames,
        ),
    }
    for name, report in comparisons.items():
        atomic_json_write(report, comparisons_dir / f"{name}.json")
    assert_common_input_fingerprints(mode_summaries)
    manifest = build_history_diagnostic_manifest(
        common_inputs=common_inputs,
        common_inputs_fingerprint_sha256=common_fingerprint,
        mode_summaries=mode_summaries,
        mode_traces={mode: result.trace for mode, result in results.items()},
        comparisons=comparisons,
        output_dir=args.output_dir,
        git_sha=git_sha,
    )
    manifest["teacher_target_sha256"] = teacher_target_sha256
    manifest["checkpoint_inputs"] = {
        "full_sequence": {
            "path": str(args.full_sequence_checkpoint.resolve()),
            "sha256": full_checkpoint.sha256,
        },
    }
    manifest["comparison_paths"] = {
        key: str((comparisons_dir / f"{key}.json").resolve())
        for key in comparisons
    }
    manifest["repo_preflight"] = repo_preflight
    manifest["artifact_identity"] = artifact_identity
    manifest["decode_timing"] = decode_timing
    atomic_json_write(manifest, args.output_dir / "diagnostic_manifest.json")
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = run_diagnostic(args)
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
