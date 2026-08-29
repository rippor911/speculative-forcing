from __future__ import annotations

import argparse
import gc
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Mapping

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
    global_rng_state_hash,
    load_official_checkpoint_record,
)
from utils.nf_sf_m3 import atomic_json_write, move_tensors_to_device
from utils.nf_sf_m4 import load_m4_sample_plan
from utils.nf_sf_m5_samples import M5TeacherSampleStore
from utils.nf_sf_teacher_flow_audit import (
    TEACHER_FLOW_AUDIT_SCHEMA,
    TEACHER_FLOW_AUDIT_MULTI_NOISE_REALIZATIONS_PER_RAW,
    BF16_QUANTIZED_STATE_CONTRACT,
    EXACT_PASS,
    PREDICTED_CURRENT_ORACLE_RECHECK_MODE,
    PREDICTED_CURRENT_ORACLE_RECHECK_NOISE_INDEX,
    PREDICTED_CURRENT_ORACLE_RECHECK_RAW_TIMESTEP,
    aggregate_teacher_flow_metrics,
    build_predicted_current_oracle_recheck_artifact,
    build_predicted_current_oracle_recheck_state,
    build_teacher_flow_multi_identity_manifest,
    build_flow_match_scheduler,
    build_teacher_flow_audit_result,
    build_teacher_flow_audit_states,
    exact_current_flow_conversion_oracle,
    build_teacher_flow_state_records,
    load_teacher_flow_student_checkpoint_record,
    parameter_sha256_report,
    require_no_parameter_mutation,
    run_student_predicted_current_predictions,
    run_student_mcp_full_sequence_predictions,
    run_teacher_branch_predictions,
    select_validation32_identities,
    select_validation_zero_identity,
    validate_frozen_student_model,
    validate_frozen_teacher_model,
    validate_multi_identity_student_checkpoint_contract,
    validate_teacher_flow_artifact_identity,
    validate_teacher_flow_artifact_identity_selection,
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
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--multi_identity_validation32",
        action="store_true",
        help=(
            "Run the strict 32-identity validation audit at positions "
            "0,8,16,...,248 with one deterministic noise per raw timestep."
        ),
    )
    mode_group.add_argument(
        "--predicted_current_oracle_recheck_only",
        action="store_true",
        help=(
            "Run only the validation0/raw999/noise0 exact-current oracle "
            "diagnostic and write predicted_current_oracle_recheck.json."
        ),
    )
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
    _validate_multi_identity_cli_contract(args)
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
    formal_validation_selection = bool(
        getattr(args, "multi_identity_validation32", False)
    ) or bool(getattr(args, "predicted_current_oracle_recheck_only", False))
    identity_selection = (
        select_validation32_identities(sample_plan)
        if formal_validation_selection
        else None
    )
    sample_identity = (
        str(identity_selection["identity_strings"][0])
        if identity_selection is not None
        else select_validation_zero_identity(sample_plan)
    )
    teacher_manifest_sha256 = file_sha256(args.teacher_manifest)
    expected_training_git_sha = _expected_training_git_sha(args, git_sha=git_sha)
    student_checkpoint = load_teacher_flow_student_checkpoint_record(
        args.full_sequence_checkpoint,
        expected_checkpoint_step=int(args.expected_checkpoint_step),
        expected_training_git_sha=expected_training_git_sha,
    )
    if student_checkpoint.payload is None:
        raise RuntimeError("student checkpoint payload missing after validation")
    multi_student_checkpoint_contract = (
        validate_multi_identity_student_checkpoint_contract(student_checkpoint)
        if identity_selection is not None
        else None
    )
    artifact_identity = (
        validate_teacher_flow_artifact_identity_selection(
            sample_plan=sample_plan,
            teacher_manifest_sha256=teacher_manifest_sha256,
            checkpoint_payload=student_checkpoint.payload,
            identity_selection=identity_selection,
        )
        if identity_selection is not None
        else validate_teacher_flow_artifact_identity(
            sample_plan=sample_plan,
            teacher_manifest_sha256=teacher_manifest_sha256,
            checkpoint_payload=student_checkpoint.payload,
            selected_identity=sample_identity,
        )
    )
    teacher_store = M5TeacherSampleStore(
        sample_plan=sample_plan,
        manifest_path=args.teacher_manifest,
        dataset_root=args.dataset_root,
        reference_checkpoint_path=None,
        expected_reference_sha256=OFFICIAL_SELF_FORCING_CHECKPOINT_SHA256,
    )
    if bool(getattr(args, "predicted_current_oracle_recheck_only", False)):
        return _run_predicted_current_oracle_recheck_only(
            args=args,
            git_sha=git_sha,
            repo_preflight=repo_preflight,
            runtime_contract=runtime_contract,
            device=device,
            dtype=dtype,
            config=config,
            teacher_store=teacher_store,
            student_checkpoint=student_checkpoint,
            identity_selection=identity_selection,
            artifact_identity=artifact_identity,
            student_checkpoint_contract=multi_student_checkpoint_contract,
            expected_training_git_sha=expected_training_git_sha,
        )
    if identity_selection is not None:
        return _run_multi_identity_validation32(
            args=args,
            git_sha=git_sha,
            repo_preflight=repo_preflight,
            runtime_contract=runtime_contract,
            device=device,
            dtype=dtype,
            config=config,
            teacher_store=teacher_store,
            student_checkpoint=student_checkpoint,
            identity_selection=identity_selection,
            artifact_identity=artifact_identity,
            student_checkpoint_contract=multi_student_checkpoint_contract,
            expected_training_git_sha=expected_training_git_sha,
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
    student_summary = validate_frozen_student_model(
        student_generator,
        checkpoint=student_checkpoint,
    )
    student_parameters_before = parameter_sha256_report(
        student_generator,
        role="student_main_mcp",
    )
    try:
        student_predictions = run_student_mcp_full_sequence_predictions(
            student_generator,
            states=states,
            teacher_target=teacher_target,
            conditional_dict=conditional_dict,
            direct_clean_context_kv=bool(args.student_direct_clean_context_kv),
        )
        student_current_predictions = run_student_predicted_current_predictions(
            runtime_factory=lambda: initialize_runtime(
                generator=student_generator,
                config=config,
                source_noise=source_noise,
            ),
            states=states,
            source_noise=source_noise,
            teacher_target=teacher_target,
            teacher_payload=teacher_payload,
            conditional_dict=conditional_dict,
            main_scheduler=main_scheduler,
        )
        student_parameters_after = parameter_sha256_report(
            student_generator,
            role="student_main_mcp",
        )
        student_summary["parameter_mutation_proof"] = require_no_parameter_mutation(
            student_parameters_before,
            student_parameters_after,
            role="Student",
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
    teacher_parameters_before = parameter_sha256_report(
        teacher_generator,
        role="teacher_official_main",
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
            student_current_predictions=student_current_predictions,
        )
        teacher_parameters_after = parameter_sha256_report(
            teacher_generator,
            role="teacher_official_main",
        )
        teacher_summary["parameter_mutation_proof"] = require_no_parameter_mutation(
            teacher_parameters_before,
            teacher_parameters_after,
            role="Teacher",
        )
    finally:
        teacher_generator.to("cpu")
        del teacher_generator
        gc.collect()

    result = build_teacher_flow_audit_result(
        states=states,
        student_predictions=student_predictions,
        student_current_predictions=student_current_predictions,
        teacher_predictions=teacher_predictions,
        sample_identity=sample_identity,
        checkpoint_summary={
            **student_checkpoint.to_json(),
            "expected_checkpoint_step": int(args.expected_checkpoint_step),
        },
        student_summary=student_summary,
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


def _run_predicted_current_oracle_recheck_only(
    *,
    args: argparse.Namespace,
    git_sha: str,
    repo_preflight: Mapping[str, Any],
    runtime_contract: Mapping[str, Any],
    device: torch.device,
    dtype: torch.dtype,
    config: Mapping[str, Any],
    teacher_store: M5TeacherSampleStore,
    student_checkpoint: Any,
    identity_selection: Mapping[str, Any] | None,
    artifact_identity: Mapping[str, Any],
    student_checkpoint_contract: Mapping[str, Any] | None,
    expected_training_git_sha: str,
) -> dict[str, Any]:
    if identity_selection is None:
        raise RuntimeError("predicted-current oracle recheck requires identity selection")
    if student_checkpoint_contract is None:
        raise RuntimeError("predicted-current oracle recheck requires step6500 contract")
    sample_identity = str(identity_selection["identity_strings"][0])
    identity_index = 0
    validation_position = int(identity_selection["positions"][0])
    if validation_position != 0:
        raise RuntimeError("predicted-current oracle recheck requires validation0")

    main_scheduler = build_flow_match_scheduler(shift=DEFAULT_S_MAIN, device=device)
    mcp_scheduler = build_flow_match_scheduler(shift=DEFAULT_S_MCP, device=device)

    student_generator = build_generator(
        config=config,
        checkpoint=student_checkpoint,
        mode=MODE_TRAINED_MCP1,
        device=device,
        dtype=dtype,
    )
    student_summary = validate_frozen_student_model(
        student_generator,
        checkpoint=student_checkpoint,
    )
    student_parameters_before = parameter_sha256_report(
        student_generator,
        role="student_main_mcp",
    )
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
    teacher_parameters_before = parameter_sha256_report(
        teacher_generator,
        role="teacher_official_main",
    )
    try:
        with teacher_store.acquire(sample_identity) as teacher_sample:
            teacher_payload = dict(teacher_sample.payload)
            teacher_metadata = dict(teacher_sample.metadata)
            source_noise_cpu = teacher_sample.source_noise.detach().cpu()
            teacher_target_cpu = teacher_sample.target_latent.detach().cpu()

        _validate_sample_tensors(
            source_noise=source_noise_cpu,
            teacher_target=teacher_target_cpu,
            dtype=dtype,
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
            selected_validation_position=validation_position,
        )
        common_inputs.update(
            {
                "audit_schema": TEACHER_FLOW_AUDIT_SCHEMA,
                "mode": PREDICTED_CURRENT_ORACLE_RECHECK_MODE,
                "diagnostic_only": True,
                "non_deployable": True,
                "runtime_contract": runtime_contract,
                "repo_preflight": repo_preflight,
                "artifact_identity": artifact_identity,
                "student_checkpoint_contract": student_checkpoint_contract,
                "config_path": str(args.config.resolve()),
                "sample_plan_path": str(args.sample_plan.resolve()),
                "teacher_manifest_path": str(args.teacher_manifest.resolve()),
                "dataset_root": str(args.dataset_root.resolve()),
                "frame_seq_length": FULL_SEQUENCE_FRAME_SEQ_LENGTH,
                "selected_identity_policy": "validation positions 0,8,16,...,248",
                "identity_index": identity_index,
                "identity_selection_fingerprint_sha256": str(
                    artifact_identity["identity_selection"][
                        "selection_fingerprint_sha256"
                    ]
                ),
                "fixed_raw_timestep": PREDICTED_CURRENT_ORACLE_RECHECK_RAW_TIMESTEP,
                "fixed_noise_index": PREDICTED_CURRENT_ORACLE_RECHECK_NOISE_INDEX,
                "student_direct_clean_context_kv": False,
                "expected_student_checkpoint_git_sha": str(expected_training_git_sha),
            }
        )
        common_fingerprint = assert_common_payload_fingerprint(common_inputs)

        rng_before = global_rng_state_hash(device)
        state = build_predicted_current_oracle_recheck_state(
            source_noise=source_noise,
            teacher_target=teacher_target,
            main_scheduler=main_scheduler,
            mcp_scheduler=mcp_scheduler,
            state_id_prefix="id00_pos000",
            sample_identity=sample_identity,
            validation_position=validation_position,
            identity_index=identity_index,
        )
        original_bf16_oracle_pass = True
        try:
            oracle = exact_current_flow_conversion_oracle(
                main_scheduler,
                state=state,
                teacher_target=teacher_target,
            )
            diagnostic = dict(oracle["diagnostic"])
        except RuntimeError as exc:
            original_bf16_oracle_pass = False
            diagnostic = _current_oracle_failure_diagnostic(exc)
        rng_after = global_rng_state_hash(device)

        student_parameters_after = parameter_sha256_report(
            student_generator,
            role="student_main_mcp",
        )
        teacher_parameters_after = parameter_sha256_report(
            teacher_generator,
            role="teacher_official_main",
        )
        artifact = build_predicted_current_oracle_recheck_artifact(
            diagnostic=diagnostic,
            original_bf16_oracle_pass=original_bf16_oracle_pass,
            runtime_git_sha=git_sha,
            sample_identity=sample_identity,
            identity_index=identity_index,
            validation_position=validation_position,
            student_parameters_before=student_parameters_before,
            student_parameters_after=student_parameters_after,
            teacher_parameters_before=teacher_parameters_before,
            teacher_parameters_after=teacher_parameters_after,
            rng_before=rng_before,
            rng_after=rng_after,
            common_inputs_fingerprint_sha256=common_fingerprint,
            artifact_identity=artifact_identity,
            student_checkpoint_contract=student_checkpoint_contract,
            checkpoint_summary={
                **student_checkpoint.to_json(),
                "expected_checkpoint_step": int(args.expected_checkpoint_step),
            },
            student_summary=student_summary,
            teacher_summary=teacher_summary,
        )
        artifact["output_dir"] = str(args.output_dir.resolve())
        atomic_json_write(
            artifact,
            args.output_dir / "predicted_current_oracle_recheck.json",
        )
        return artifact
    finally:
        student_generator.to("cpu")
        teacher_generator.to("cpu")
        del student_generator
        del teacher_generator
        gc.collect()


def _run_multi_identity_validation32(
    *,
    args: argparse.Namespace,
    git_sha: str,
    repo_preflight: Mapping[str, Any],
    runtime_contract: Mapping[str, Any],
    device: torch.device,
    dtype: torch.dtype,
    config: Mapping[str, Any],
    teacher_store: M5TeacherSampleStore,
    student_checkpoint: Any,
    identity_selection: Mapping[str, Any],
    artifact_identity: Mapping[str, Any],
    student_checkpoint_contract: Mapping[str, Any],
    expected_training_git_sha: str,
) -> dict[str, Any]:
    main_scheduler = build_flow_match_scheduler(shift=DEFAULT_S_MAIN, device=device)
    mcp_scheduler = build_flow_match_scheduler(shift=DEFAULT_S_MCP, device=device)

    student_generator = build_generator(
        config=config,
        checkpoint=student_checkpoint,
        mode=MODE_TRAINED_MCP1,
        device=device,
        dtype=dtype,
    )
    student_summary = validate_frozen_student_model(
        student_generator,
        checkpoint=student_checkpoint,
    )
    student_parameters_before = parameter_sha256_report(
        student_generator,
        role="student_main_mcp",
    )
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
    teacher_parameters_before = parameter_sha256_report(
        teacher_generator,
        role="teacher_official_main",
    )

    state_records: list[Mapping[str, Any]] = []
    identity_records: list[dict[str, Any]] = []
    common_fingerprints: dict[str, str] = {}
    try:
        for identity_index, identity in enumerate(identity_selection["identity_strings"]):
            validation_position = int(identity_selection["positions"][identity_index])
            identity_state_records, common_fingerprint = _run_one_identity_records(
                args=args,
                git_sha=git_sha,
                repo_preflight=repo_preflight,
                runtime_contract=runtime_contract,
                device=device,
                dtype=dtype,
                config=config,
                teacher_store=teacher_store,
                student_generator=student_generator,
                teacher_generator=teacher_generator,
                sample_identity=str(identity),
                validation_position=validation_position,
                identity_index=int(identity_index),
                student_checkpoint=student_checkpoint,
                artifact_identity=artifact_identity,
                student_checkpoint_contract=student_checkpoint_contract,
                main_scheduler=main_scheduler,
                mcp_scheduler=mcp_scheduler,
                expected_training_git_sha=expected_training_git_sha,
            )
            state_records.extend(identity_state_records)
            common_fingerprints[str(validation_position)] = common_fingerprint
            identity_records.append(
                {
                    "identity_index": int(identity_index),
                    "sample_identity": str(identity),
                    "validation_position": validation_position,
                    "state_count": len(identity_state_records),
                    "common_inputs_fingerprint_sha256": common_fingerprint,
                    "metrics": aggregate_teacher_flow_metrics(
                        identity_state_records
                    )["all_states"],
                }
            )
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        student_parameters_after = parameter_sha256_report(
            student_generator,
            role="student_main_mcp",
        )
        teacher_parameters_after = parameter_sha256_report(
            teacher_generator,
            role="teacher_official_main",
        )
        student_summary["parameter_mutation_proof"] = require_no_parameter_mutation(
            student_parameters_before,
            student_parameters_after,
            role="Student",
        )
        teacher_summary["parameter_mutation_proof"] = require_no_parameter_mutation(
            teacher_parameters_before,
            teacher_parameters_after,
            role="Teacher",
        )
    finally:
        student_generator.to("cpu")
        teacher_generator.to("cpu")
        del student_generator
        del teacher_generator
        gc.collect()

    manifest = build_teacher_flow_multi_identity_manifest(
        state_records=state_records,
        identity_records=identity_records,
        identity_selection=identity_selection,
        student_checkpoint_contract=student_checkpoint_contract,
        checkpoint_summary={
            **student_checkpoint.to_json(),
            "expected_checkpoint_step": int(args.expected_checkpoint_step),
        },
        student_summary=student_summary,
        teacher_summary=teacher_summary,
        common_inputs_fingerprints_sha256=common_fingerprints,
        runtime_git_sha=git_sha,
        training_checkpoint_git_sha=str(student_checkpoint.training_git_sha),
    )
    manifest["output_dir"] = str(args.output_dir.resolve())
    atomic_json_write(manifest, args.output_dir / "teacher_flow_audit.json")
    return manifest


def _run_one_identity_records(
    *,
    args: argparse.Namespace,
    git_sha: str,
    repo_preflight: Mapping[str, Any],
    runtime_contract: Mapping[str, Any],
    device: torch.device,
    dtype: torch.dtype,
    config: Mapping[str, Any],
    teacher_store: M5TeacherSampleStore,
    student_generator: Any,
    teacher_generator: Any,
    sample_identity: str,
    validation_position: int,
    identity_index: int,
    student_checkpoint: Any,
    artifact_identity: Mapping[str, Any],
    student_checkpoint_contract: Mapping[str, Any],
    main_scheduler: Any,
    mcp_scheduler: Any,
    expected_training_git_sha: str,
) -> tuple[list[dict[str, Any]], str]:
    with teacher_store.acquire(sample_identity) as teacher_sample:
        teacher_payload = dict(teacher_sample.payload)
        teacher_metadata = dict(teacher_sample.metadata)
        source_noise_cpu = teacher_sample.source_noise.detach().cpu()
        teacher_target_cpu = teacher_sample.target_latent.detach().cpu()

    _validate_sample_tensors(
        source_noise=source_noise_cpu,
        teacher_target=teacher_target_cpu,
        dtype=dtype,
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
        selected_validation_position=int(validation_position),
    )
    common_inputs.update(
        {
            "audit_schema": TEACHER_FLOW_AUDIT_SCHEMA,
            "diagnostic_only": True,
            "non_deployable": True,
            "runtime_contract": runtime_contract,
            "repo_preflight": repo_preflight,
            "artifact_identity": artifact_identity,
            "student_checkpoint_contract": student_checkpoint_contract,
            "config_path": str(args.config.resolve()),
            "sample_plan_path": str(args.sample_plan.resolve()),
            "teacher_manifest_path": str(args.teacher_manifest.resolve()),
            "dataset_root": str(args.dataset_root.resolve()),
            "frame_seq_length": FULL_SEQUENCE_FRAME_SEQ_LENGTH,
            "selected_identity_policy": "validation positions 0,8,16,...,248",
            "identity_index": int(identity_index),
            "identity_selection_fingerprint_sha256": str(
                artifact_identity["identity_selection"]["selection_fingerprint_sha256"]
            ),
            "student_direct_clean_context_kv": False,
            "expected_student_checkpoint_git_sha": str(expected_training_git_sha),
        }
    )
    common_fingerprint = assert_common_payload_fingerprint(common_inputs)
    states = build_teacher_flow_audit_states(
        source_noise=source_noise,
        teacher_target=teacher_target,
        main_scheduler=main_scheduler,
        mcp_scheduler=mcp_scheduler,
        noise_realizations_per_raw=TEACHER_FLOW_AUDIT_MULTI_NOISE_REALIZATIONS_PER_RAW,
        state_id_prefix=f"id{identity_index:02d}_pos{validation_position:03d}",
        sample_identity=sample_identity,
        validation_position=int(validation_position),
        identity_index=int(identity_index),
    )
    student_predictions = run_student_mcp_full_sequence_predictions(
        student_generator,
        states=states,
        teacher_target=teacher_target,
        conditional_dict=conditional_dict,
        direct_clean_context_kv=False,
    )
    student_current_predictions = run_student_predicted_current_predictions(
        runtime_factory=lambda: initialize_runtime(
            generator=student_generator,
            config=config,
            source_noise=source_noise,
        ),
        states=states,
        source_noise=source_noise,
        teacher_target=teacher_target,
        teacher_payload=teacher_payload,
        conditional_dict=conditional_dict,
        main_scheduler=main_scheduler,
    )
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
        student_current_predictions=student_current_predictions,
    )
    state_records = build_teacher_flow_state_records(
        states=states,
        student_predictions=student_predictions,
        student_current_predictions=student_current_predictions,
        teacher_predictions=teacher_predictions,
        sample_identity=sample_identity,
        validation_position=int(validation_position),
        identity_index=int(identity_index),
    )
    del states
    del student_predictions
    del student_current_predictions
    del teacher_predictions
    del source_noise
    del teacher_target
    del conditional_dict
    del teacher_payload
    del teacher_metadata
    return state_records, common_fingerprint


def _validate_sample_tensors(
    *,
    source_noise: torch.Tensor,
    teacher_target: torch.Tensor,
    dtype: torch.dtype,
) -> None:
    if int(source_noise.shape[1]) != FULL_SEQUENCE_FRAME_COUNT:
        raise RuntimeError("selected teacher sample must contain 21 latent frames")
    if tuple(source_noise.shape) != tuple(teacher_target.shape):
        raise RuntimeError("teacher target_latent must match source_noise shape")
    if source_noise.dtype != dtype or teacher_target.dtype != dtype:
        raise RuntimeError("requested dtype must match stored teacher tensors")


def _current_oracle_failure_diagnostic(exc: BaseException) -> dict[str, Any]:
    marker = "diagnostic="
    text = str(exc)
    if marker not in text:
        raise exc
    return json.loads(text.split(marker, 1)[1])


def _validate_multi_identity_cli_contract(args: argparse.Namespace) -> None:
    requires_formal_step6500 = bool(
        getattr(args, "multi_identity_validation32", False)
    ) or bool(getattr(args, "predicted_current_oracle_recheck_only", False))
    if not requires_formal_step6500:
        return
    mode_name = (
        "multi-identity Teacher-flow audit"
        if bool(getattr(args, "multi_identity_validation32", False))
        else "predicted-current oracle recheck"
    )
    if int(args.expected_checkpoint_step) != 6500:
        raise RuntimeError(f"{mode_name} requires step6500")
    if bool(args.student_direct_clean_context_kv):
        raise RuntimeError(f"{mode_name} requires direct_clean_context_kv=false")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = run_audit(args)
    if bool(getattr(args, "predicted_current_oracle_recheck_only", False)):
        print(
            json.dumps(
                {
                    "schema": manifest["schema"],
                    "status": manifest["status"],
                    "diagnostic_classification": manifest[
                        "diagnostic_classification"
                    ],
                    "original_bf16_oracle_pass": manifest[
                        "original_bf16_oracle_pass"
                    ],
                    "output_dir": manifest["output_dir"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return _predicted_current_oracle_recheck_exit_code(manifest)
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


def _predicted_current_oracle_recheck_exit_code(manifest: Mapping[str, Any]) -> int:
    if manifest.get("status") != "PASS":
        return 1
    if manifest.get("diagnostic_classification") in {
        EXACT_PASS,
        BF16_QUANTIZED_STATE_CONTRACT,
    }:
        return 0
    return 1


def _expected_training_git_sha(args: argparse.Namespace, *, git_sha: str) -> str:
    if args.expected_training_git_sha is not None:
        return str(args.expected_training_git_sha)
    if int(args.expected_checkpoint_step) == 5000:
        return TRAINING_CHECKPOINT_GIT_SHA
    return str(git_sha)


if __name__ == "__main__":
    raise SystemExit(main())
