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
    decode_and_write_videos,
    dtype_from_arg,
    initialize_runtime,
    merge_config,
    runtime_device,
    select_eval_identity,
    validate_cli_contract,
    validate_config,
)
from utils.nf_sf_first_mcp_route_equivalence import (
    load_route_equivalence_checkpoint_record,
    route_equivalence_checkpoint_loader_mode,
)
from utils.nf_sf_first_mcp_step_sweep import (
    DEFAULT_STEP_COUNTS,
    FIRST_MCP_STEP_SWEEP_SCHEMA,
    LIVE_JOINT_PREDICTED,
    ORACLE_FLOW,
    TEACHER_CURRENT_PREDICTED_MCP,
    run_first_mcp_step_sweep,
)
from utils.nf_sf_full_sequence_eval import (
    FULL_SEQUENCE_FRAME_COUNT,
    FULL_SEQUENCE_FRAME_SEQ_LENGTH,
    MODE_TRAINED_MCP1,
    OFFICIAL_SELF_FORCING_CHECKPOINT_SHA256,
    build_common_inputs_record,
    current_git_head,
    file_sha256,
    tensor_sha256,
    validate_eval_artifact_identity,
)
from utils.nf_sf_m3 import atomic_json_write, atomic_torch_save, move_tensors_to_device
from utils.nf_sf_m4 import load_m4_sample_plan
from utils.nf_sf_m5_samples import M5TeacherSampleStore
from utils.nf_sf_tensors import DEFAULT_S_MAIN, DEFAULT_S_MCP


STEP6500_CHECKPOINT_SHA256 = (
    "9ef57cb2d3e5f20b244129317af4a0e1d2b1c810ba65ec970892e60ccbd34f4f"
)
STEP6500_CHECKPOINT_GIT_SHA = "c3f89888bf6da31b48650f0a680dd6534943f56f"
STEP6500_GLOBAL_STEP = 6500


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="NF-SF First-MCP diagnostic-only denoising-step sweep."
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
        "--expected_checkpoint_git_sha",
        default=STEP6500_CHECKPOINT_GIT_SHA,
    )
    parser.add_argument(
        "--expected_checkpoint_sha256",
        default=STEP6500_CHECKPOINT_SHA256,
    )
    parser.add_argument(
        "--step_counts",
        nargs="+",
        type=int,
        default=list(DEFAULT_STEP_COUNTS),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bf16",), default="bf16")
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--decode_extremes", action="store_true")
    return parser.parse_args(argv)


def run_sweep(args: argparse.Namespace) -> dict[str, Any]:
    torch.set_grad_enabled(False)
    git_sha = current_git_head()
    repo_preflight = validate_cli_contract(args, git_sha=git_sha)
    device, runtime_contract = runtime_device(args.device)
    dtype = dtype_from_arg(args.dtype)
    config = merge_config(args.config)
    validate_config(config)
    validate_step_sweep_config(config)

    sample_plan = load_m4_sample_plan(
        args.sample_plan,
        manifest_path=args.teacher_manifest,
    )
    sample_identity = select_eval_identity(
        sample_plan,
        sample_identity=args.sample_identity,
        num_samples=int(args.num_samples),
    )
    fixed_identity_contract = enforce_fixed_validation_identity_contract(
        sample_plan,
        selected_identity=sample_identity,
    )
    teacher_manifest_sha256 = file_sha256(args.teacher_manifest)
    full_checkpoint = load_route_equivalence_checkpoint_record(
        args.full_sequence_checkpoint,
        expected_checkpoint_step=STEP6500_GLOBAL_STEP,
        expected_training_git_sha=str(args.expected_checkpoint_git_sha),
    )
    _validate_step6500_checkpoint(
        full_checkpoint,
        expected_checkpoint_sha256=str(args.expected_checkpoint_sha256),
    )
    if full_checkpoint.payload is None:
        raise RuntimeError("step6500 checkpoint payload missing after validation")
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
    if tuple(source_noise_cpu.shape) != tuple(teacher_target_cpu.shape):
        raise RuntimeError("teacher target_latent must match source_noise shape")
    if source_noise_cpu.dtype != dtype:
        raise RuntimeError(
            "requested dtype must match stored teacher source_noise dtype; "
            f"requested={dtype}, stored={source_noise_cpu.dtype}"
        )

    conditioning_cpu = build_conditioning(
        prompt=str(teacher_payload["prompt"]),
        device=device,
        dtype=dtype,
    )
    source_noise = source_noise_cpu.to(device=device)
    teacher_target = teacher_target_cpu.to(device=device, dtype=dtype)
    teacher_payload["source_noise"] = source_noise
    conditional_dict = move_tensors_to_device(
        conditioning_cpu,
        device=device,
        floating_dtype=dtype,
    )
    teacher_target_sha256 = tensor_sha256(teacher_target_cpu)
    common_inputs, _ = build_common_inputs_record(
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
    checkpoint_loader_mode = route_equivalence_checkpoint_loader_mode(
        STEP6500_GLOBAL_STEP
    )
    common_inputs.update(
        {
            "audit_schema": FIRST_MCP_STEP_SWEEP_SCHEMA,
            "diagnostic_only": True,
            "non_deployable": True,
            "training_eligible": False,
            "canonical_training_eligible": False,
            "canonical_deployment_eligible": False,
            **fixed_identity_contract,
            "expected_checkpoint_step": STEP6500_GLOBAL_STEP,
            "loaded_checkpoint_global_step": int(full_checkpoint.global_step),
            "checkpoint_loader_mode": checkpoint_loader_mode,
            "diagnostic_intermediate_checkpoint": True,
            "checkpoint_sha256": str(full_checkpoint.sha256),
            "expected_checkpoint_sha256": str(args.expected_checkpoint_sha256),
            "expected_checkpoint_git_sha": str(args.expected_checkpoint_git_sha),
            "runtime_contract": runtime_contract,
            "repo_preflight": repo_preflight,
            "artifact_identity": artifact_identity,
            "config_path": str(args.config.resolve()),
            "sample_plan_path": str(args.sample_plan.resolve()),
            "teacher_manifest_path": str(args.teacher_manifest.resolve()),
            "dataset_root": str(args.dataset_root.resolve()),
            "frame_seq_length": FULL_SEQUENCE_FRAME_SEQ_LENGTH,
            "teacher_target_sha256": teacher_target_sha256,
            "first_mcp_step_sweep_contract": {
                "teacher_history_chunks": [0],
                "current_chunk": 1,
                "future_chunk": 2,
                "depths_used": [1],
                "forbidden_depths": [2, 3],
                "step_counts": [int(value) for value in args.step_counts],
                "main_shift": DEFAULT_S_MAIN,
                "mcp_shift": DEFAULT_S_MCP,
                "sigma_min": 0.0,
                "extra_one_step": True,
                "context_noise": 0,
                "decode_extremes": bool(args.decode_extremes),
                "video_decode_main_metric": False,
            },
        }
    )
    common_fingerprint = assert_common_payload_fingerprint(common_inputs)
    atomic_json_write(common_inputs, args.output_dir / "common_inputs.json")

    generator = build_generator(
        config=config,
        checkpoint=full_checkpoint,
        mode=MODE_TRAINED_MCP1,
        device=device,
        dtype=dtype,
    )
    try:
        result = run_first_mcp_step_sweep(
            runtime_factory=lambda: initialize_runtime(
                generator=generator,
                config=config,
                source_noise=source_noise,
            ),
            source_noise=source_noise,
            teacher_target=teacher_target,
            teacher_payload=teacher_payload,
            conditional_dict=conditional_dict,
            checkpoint_summary={
                **full_checkpoint.to_json(),
                "expected_checkpoint_step": STEP6500_GLOBAL_STEP,
                "loaded_checkpoint_global_step": int(full_checkpoint.global_step),
                "checkpoint_loader_mode": checkpoint_loader_mode,
                "diagnostic_intermediate_checkpoint": True,
                "expected_checkpoint_sha256": str(args.expected_checkpoint_sha256),
            },
            common_inputs=common_inputs,
            common_inputs_fingerprint_sha256=common_fingerprint,
            runtime_git_sha=git_sha,
            training_checkpoint_git_sha=str(full_checkpoint.training_git_sha),
            step_counts=tuple(int(value) for value in args.step_counts),
        )
    finally:
        generator.to("cpu")
        del generator
        gc.collect()
        torch.cuda.empty_cache()

    tensors_dir = args.output_dir / "tensors"
    tensors_dir.mkdir(parents=True, exist_ok=True)
    tensor_path = tensors_dir / "first_mcp_step_sweep_tensors.pt"
    atomic_torch_save(result.tensors, tensor_path)

    hybrid_paths = {}
    for name, latent in result.hybrid_latents.items():
        path = args.output_dir / f"{name}.pt"
        atomic_torch_save({"latent": latent, "source": LIVE_JOINT_PREDICTED}, path)
        hybrid_paths[name] = {
            "path": str(path.resolve()),
            "sha256": file_sha256(path),
        }

    decode_timing = None
    video_outputs = {}
    if bool(args.decode_extremes):
        _, decode_timing = decode_and_write_videos(
            latents=result.hybrid_latents,
            output_dir=args.output_dir,
            device=device,
            dtype=dtype,
            fps=int(args.fps),
        )
        video_outputs = {
            mode: {
                "path": str((args.output_dir / mode / "output.mp4").resolve()),
                "sha256": file_sha256(args.output_dir / mode / "output.mp4"),
            }
            for mode in result.hybrid_latents.keys()
        }

    manifest = dict(result.manifest)
    manifest["output_dir"] = str(args.output_dir.resolve())
    manifest.update(fixed_identity_contract)
    manifest["repo_preflight"] = repo_preflight
    manifest["runtime_contract"] = runtime_contract
    manifest["artifact_identity"] = artifact_identity
    manifest["tensor_archive"] = {
        "path": str(tensor_path.resolve()),
        "sha256": file_sha256(tensor_path),
    }
    manifest["hybrid_latent_artifacts"] = hybrid_paths
    manifest["decode_extremes"] = bool(args.decode_extremes)
    manifest["decode_timing"] = decode_timing
    manifest["video_outputs"] = video_outputs
    atomic_json_write(manifest, args.output_dir / "first_mcp_step_sweep.json")
    return manifest


def validate_step_sweep_config(config: Any) -> None:
    if int(getattr(config, "context_noise", -1)) != 0:
        raise RuntimeError("First-MCP step sweep requires canonical context_noise=0")


def enforce_fixed_validation_identity_contract(
    sample_plan: Mapping[str, Any],
    *,
    selected_identity: str,
) -> dict[str, Any]:
    if "fixed_decode_validation_identity" not in sample_plan:
        raise RuntimeError("sample plan missing fixed_decode_validation_identity")
    expected = str(sample_plan["fixed_decode_validation_identity"])
    selected = str(selected_identity)
    if selected != expected:
        raise RuntimeError(
            "First-MCP step sweep requires fixed_decode_validation_identity; "
            f"selected={selected}, expected={expected}"
        )
    return {
        "fixed_identity_contract": True,
        "fixed_decode_validation_identity": expected,
    }


def _validate_step6500_checkpoint(
    checkpoint: Any,
    *,
    expected_checkpoint_sha256: str,
) -> None:
    if int(checkpoint.global_step) != STEP6500_GLOBAL_STEP:
        raise RuntimeError("First-MCP step sweep requires step6500 checkpoint")
    if str(checkpoint.sha256) != str(expected_checkpoint_sha256):
        raise RuntimeError("step6500 checkpoint SHA256 mismatch")
    if str(checkpoint.training_git_sha) != STEP6500_CHECKPOINT_GIT_SHA:
        raise RuntimeError("step6500 checkpoint training git SHA mismatch")
    expected_mode = route_equivalence_checkpoint_loader_mode(STEP6500_GLOBAL_STEP)
    if str(checkpoint.load_mode) != expected_mode:
        raise RuntimeError("step6500 checkpoint loader mode mismatch")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = run_sweep(args)
    print(
        json.dumps(
            {
                "schema": manifest["schema"],
                "status": manifest["status"],
                "output_dir": manifest["output_dir"],
                "primary_decision": manifest["primary_decision"],
                "final_mse": {
                    count: {
                        LIVE_JOINT_PREDICTED: run["branches"][
                            LIVE_JOINT_PREDICTED
                        ]["final_chunk2_mse_to_teacher"],
                        TEACHER_CURRENT_PREDICTED_MCP: run["branches"][
                            TEACHER_CURRENT_PREDICTED_MCP
                        ]["final_chunk2_mse_to_teacher"],
                        ORACLE_FLOW: run["branches"][ORACLE_FLOW][
                            "final_oracle_chunk2_mse_to_teacher"
                        ],
                    }
                    for count, run in manifest["runs"].items()
                },
            },
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0 if manifest["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
