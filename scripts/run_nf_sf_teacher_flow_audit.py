from __future__ import annotations

import argparse
import gc
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from scripts.eval_nf_sf_full_sequence_deployment import (
    assert_common_payload_fingerprint,
    build_conditioning,
    build_generator,
    dtype_from_arg,
    initialize_runtime,
    merge_config,
    runtime_device,
    validate_cli_contract,
    validate_config,
)
from utils.nf_sf_full_sequence_eval import (
    FULL_SEQUENCE_FRAME_COUNT,
    FULL_SEQUENCE_FRAME_SEQ_LENGTH,
    MODE_OFFICIAL_MAIN,
    MODE_TRAINED_MCP1,
    OFFICIAL_SELF_FORCING_CHECKPOINT_SHA256,
    TRAINING_CHECKPOINT_GIT_SHA,
    build_common_inputs_record,
    current_git_head,
    file_sha256,
    load_official_checkpoint_record,
)
from utils.nf_sf_m3 import atomic_json_write, move_tensors_to_device
from utils.nf_sf_m4 import load_m4_sample_plan
from utils.nf_sf_m5_samples import M5TeacherSampleStore
from utils.nf_sf_teacher_flow_audit import (
    TEACHER_FLOW_AUDIT_SCHEMA,
    build_flow_match_scheduler,
    build_teacher_flow_audit_result,
    build_teacher_flow_audit_states,
    load_teacher_flow_student_checkpoint_record,
    run_student_mcp_full_sequence_predictions,
    run_teacher_branch_predictions,
    select_validation_zero_identity,
    validate_frozen_teacher_model,
    validate_teacher_flow_artifact_identity,
)
from utils.nf_sf_tensors import DEFAULT_S_MAIN, DEFAULT_S_MCP


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="NF-SF inference-only Teacher conditional-flow audit."
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
    parser.add_argument("--expected_checkpoint_step", type=int, required=True)
    parser.add_argument("--sample_plan", required=True, type=Path)
    parser.add_argument("--teacher_manifest", required=True, type=Path)
    parser.add_argument("--dataset_root", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--expected_runtime_git_sha", required=True)
    parser.add_argument(
        "--expected_training_git_sha",
        default=None,
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bf16",), default="bf16")
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument(
        "--student_direct_clean_context_kv",
        action="store_true",
        help=(
            "Use only when auditing a full-sequence validation route whose MCP "
            "forward explicitly enabled direct_clean_context_kv."
        ),
    )
    parser.set_defaults(sample_identity=None, num_samples=1)
    return parser.parse_args(argv)


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    torch.set_grad_enabled(False)
    git_sha = current_git_head()
    repo_preflight = validate_cli_contract(args, git_sha=git_sha)
    device, runtime_contract = runtime_device(args.device)
    dtype = dtype_from_arg(args.dtype)
    config = merge_config(args.config)
    validate_config(config)

    sample_plan = load_m4_sample_plan(
        args.sample_plan,
        manifest_path=args.teacher_manifest,
    )
    sample_identity = select_validation_zero_identity(sample_plan)
    teacher_manifest_sha256 = file_sha256(args.teacher_manifest)
    expected_training_git_sha = _expected_training_git_sha(args, git_sha=git_sha)
    student_checkpoint = load_teacher_flow_student_checkpoint_record(
        args.full_sequence_checkpoint,
        expected_checkpoint_step=int(args.expected_checkpoint_step),
        expected_training_git_sha=expected_training_git_sha,
    )
    if student_checkpoint.payload is None:
        raise RuntimeError("student checkpoint payload missing after validation")
    artifact_identity = validate_teacher_flow_artifact_identity(
        sample_plan=sample_plan,
        teacher_manifest_sha256=teacher_manifest_sha256,
        checkpoint_payload=student_checkpoint.payload,
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
    if source_noise_cpu.dtype != dtype or teacher_target_cpu.dtype != dtype:
        raise RuntimeError("requested dtype must match stored teacher tensors")

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
    common_inputs, _ = build_common_inputs_record(
        sample_identity=sample_identity,
        teacher_metadata=teacher_metadata,
        teacher_payload=teacher_payload,
        source_noise=source_noise,
        conditioning=conditional_dict,
        runtime_git_sha=git_sha,
        training_checkpoint_git_sha=str(student_checkpoint.training_git_sha),
        fps=int(args.fps),
        sample_plan_sha256=str(artifact_identity["sample_plan_sha256"]),
        teacher_manifest_sha256=str(artifact_identity["teacher_manifest_sha256"]),
        selected_validation_position=int(
            artifact_identity["selected_validation_position"]
        ),
    )
    common_inputs.update(
        {
            "audit_schema": TEACHER_FLOW_AUDIT_SCHEMA,
            "diagnostic_only": True,
            "non_deployable": True,
            "runtime_contract": runtime_contract,
            "repo_preflight": repo_preflight,
            "artifact_identity": artifact_identity,
            "config_path": str(args.config.resolve()),
            "sample_plan_path": str(args.sample_plan.resolve()),
            "teacher_manifest_path": str(args.teacher_manifest.resolve()),
            "dataset_root": str(args.dataset_root.resolve()),
            "frame_seq_length": FULL_SEQUENCE_FRAME_SEQ_LENGTH,
            "selected_identity_policy": "validation_sample_identities[0]",
            "student_direct_clean_context_kv": bool(
                args.student_direct_clean_context_kv
            ),
            "expected_student_checkpoint_git_sha": str(expected_training_git_sha),
        }
    )
    common_fingerprint = assert_common_payload_fingerprint(common_inputs)

    main_scheduler = build_flow_match_scheduler(shift=DEFAULT_S_MAIN, device=device)
    mcp_scheduler = build_flow_match_scheduler(shift=DEFAULT_S_MCP, device=device)
    states = build_teacher_flow_audit_states(
        source_noise=source_noise,
        teacher_target=teacher_target,
        main_scheduler=main_scheduler,
        mcp_scheduler=mcp_scheduler,
    )

    student_generator = build_generator(
        config=config,
        checkpoint=student_checkpoint,
        mode=MODE_TRAINED_MCP1,
        device=device,
        dtype=dtype,
    )
    try:
        student_predictions = run_student_mcp_full_sequence_predictions(
            student_generator,
            states=states,
            teacher_target=teacher_target,
            conditional_dict=conditional_dict,
            direct_clean_context_kv=bool(args.student_direct_clean_context_kv),
        )
    finally:
        student_generator.to("cpu")
        del student_generator
        gc.collect()

    teacher_checkpoint = load_official_checkpoint_record(args.official_checkpoint)
    teacher_generator = build_generator(
        config=config,
        checkpoint=teacher_checkpoint,
        mode=MODE_OFFICIAL_MAIN,
        device=device,
        dtype=dtype,
    )
    teacher_summary = validate_frozen_teacher_model(
        teacher_generator,
        checkpoint=teacher_checkpoint,
    )
    try:
        teacher_predictions = run_teacher_branch_predictions(
            runtime_factory=lambda: initialize_runtime(
                generator=teacher_generator,
                config=config,
                source_noise=source_noise,
            ),
            states=states,
            source_noise=source_noise,
            teacher_target=teacher_target,
            teacher_payload=teacher_payload,
            conditional_dict=conditional_dict,
        )
    finally:
        teacher_generator.to("cpu")
        del teacher_generator
        gc.collect()

    result = build_teacher_flow_audit_result(
        states=states,
        student_predictions=student_predictions,
        teacher_predictions=teacher_predictions,
        sample_identity=sample_identity,
        checkpoint_summary={
            **student_checkpoint.to_json(),
            "expected_checkpoint_step": int(args.expected_checkpoint_step),
        },
        teacher_summary=teacher_summary,
        common_inputs=common_inputs,
        common_inputs_fingerprint_sha256=common_fingerprint,
        runtime_git_sha=git_sha,
        training_checkpoint_git_sha=str(student_checkpoint.training_git_sha),
    )
    manifest = dict(result.manifest)
    manifest["output_dir"] = str(args.output_dir.resolve())
    atomic_json_write(manifest, args.output_dir / "teacher_flow_audit.json")
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = run_audit(args)
    print(
        json.dumps(
            {
                "schema": manifest["schema"],
                "status": manifest["status"],
                "diagnostic_label": manifest["diagnostic_label"],
                "state_count": manifest["state_count"],
                "output_dir": manifest["output_dir"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _expected_training_git_sha(args: argparse.Namespace, *, git_sha: str) -> str:
    if args.expected_training_git_sha is not None:
        return str(args.expected_training_git_sha)
    if int(args.expected_checkpoint_step) == 5000:
        return TRAINING_CHECKPOINT_GIT_SHA
    return str(git_sha)


if __name__ == "__main__":
    raise SystemExit(main())
