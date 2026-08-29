from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any, Mapping, Sequence

import torch

import utils.nf_sf_first_mcp_flow_audit as flow_audit
import utils.nf_sf_first_mcp_route_equivalence as route_eq
import utils.nf_sf_full_sequence_eval as deployment
import utils.nf_sf_mcp1_memorization_probe as memorization
from utils.nf_sf_m3 import file_sha256, tensor_sha256, tensor_summary
from utils.nf_sf_mcp1_only_continuation import NF_SF_MCP1_ONLY_CONTINUATION_SCHEMA
from utils.nf_sf_mcp_direct_context_ablation import (
    NF_SF_MCP_DIRECT_CLEAN_KV_ABLATION_SCHEMA,
)
from utils.nf_sf_tensors import (
    DEFAULT_NUM_TRAIN_TIMESTEPS,
    DEFAULT_S_MAIN,
    DEFAULT_S_MCP,
    FULL_SEQUENCE_CHUNK_FRAMES,
    FULL_SEQUENCE_DEPTHS,
    FULL_SEQUENCE_NUM_CHUNKS,
)
from utils.nf_sf_training import (
    NFSFFullSequenceNoisyBatch,
    build_full_sequence_mcp_anchor_inputs,
    build_full_sequence_mcp_anchor_specs,
    nf_sf_full_sequence_train_cursor,
    _add_noise_for_anchor_chunks,
    _add_noise_like_scheduler,
    _training_target_for_anchor_chunks,
    _training_target_like_scheduler,
)


TEACHER_FLOW_AUDIT_SCHEMA = "nf_sf_teacher_conditional_flow_audit_v1"
TEACHER_FLOW_AUDIT_STUDENT_LOAD_MODE = "TEACHER_FLOW_AUDIT_STUDENT_STRICT"
TEACHER_FLOW_AUDIT_SUPPORTED_STUDENT_SCHEMAS = (
    route_eq.FULL_SEQUENCE_TRAINER_SCHEMA,
    NF_SF_MCP_DIRECT_CLEAN_KV_ABLATION_SCHEMA,
    NF_SF_MCP1_ONLY_CONTINUATION_SCHEMA,
)
TEACHER_FLOW_AUDIT_RAW_TIMESTEPS = (999, 750, 500, 250)
TEACHER_FLOW_AUDIT_NOISE_REALIZATIONS_PER_RAW = 4
TEACHER_FLOW_AUDIT_MULTI_NOISE_REALIZATIONS_PER_RAW = 1
TEACHER_FLOW_AUDIT_VALIDATION_COUNT = 256
TEACHER_FLOW_AUDIT_MULTI_VALIDATION_STRIDE = 8
TEACHER_FLOW_AUDIT_MULTI_VALIDATION_COUNT = 32
TEACHER_FLOW_AUDIT_MODE_SINGLE = "single_identity_validation0"
TEACHER_FLOW_AUDIT_MODE_MULTI_VALIDATION32 = "multi_identity_validation32"
PREDICTED_CURRENT_ORACLE_RECHECK_SCHEMA = (
    "nf_sf_predicted_current_oracle_recheck_v1"
)
PREDICTED_CURRENT_ORACLE_RECHECK_ALL_RAW_SCHEMA = (
    "nf_sf_predicted_current_oracle_recheck_validation0_all_raw_v1"
)
PREDICTED_CURRENT_ORACLE_RECHECK_MODE = (
    "predicted_current_oracle_recheck_only"
)
PREDICTED_CURRENT_ORACLE_RECHECK_ALL_RAW_MODE = (
    "predicted_current_oracle_recheck_validation0_all_raw"
)
PREDICTED_CURRENT_ORACLE_RECHECK_RAW_TIMESTEP = 999
PREDICTED_CURRENT_ORACLE_RECHECK_NOISE_INDEX = 0
EXACT_PASS = "exact_pass"
BF16_QUANTIZED_STATE_CONTRACT = "bf16_quantized_state_contract"
SCHEDULER_MISMATCH = "scheduler_mismatch"
STATE_PROVENANCE_MISMATCH = "state_provenance_mismatch"
SEMANTIC_MISMATCH = "semantic_mismatch"
PREDICTED_CURRENT_ORACLE_RECHECK_CLASSIFICATIONS = (
    EXACT_PASS,
    BF16_QUANTIZED_STATE_CONTRACT,
    SCHEDULER_MISMATCH,
    STATE_PROVENANCE_MISMATCH,
    SEMANTIC_MISMATCH,
)
PREDICTED_CURRENT_ORACLE_RECHECK_SCHEDULER_ATOL = 1.0e-6
PREDICTED_CURRENT_ORACLE_RECHECK_FLOAT32_ATOL = 1.0e-5
TEACHER_MATCHED_CURRENT_BRANCH = "teacher_matched_current"
TEACHER_PREDICTED_CURRENT_BRANCH = "teacher_predicted_current"
TEACHER_PRIVILEGED_CURRENT_BRANCH = "teacher_privileged_current"
TEACHER_CLEAN_CURRENT_BRANCH = "teacher_clean_current"
STUDENT_MCP_BRANCH = "student_mcp1_full_sequence"
STUDENT_PREDICTED_CURRENT_BRANCH = "student_main_predicted_current"

HISTORY_CHUNK_INDEX = 0
CURRENT_CHUNK_INDEX = 1
FUTURE_CHUNK_INDEX = 2
FUTURE_START_FRAME = FUTURE_CHUNK_INDEX * FULL_SEQUENCE_CHUNK_FRAMES

TEACHER_MATCHED_STRONGLY_BETTER = "TEACHER_MATCHED_STRONGLY_BETTER"
TEACHER_PRIVILEGED_ONLY_BETTER = "TEACHER_PRIVILEGED_ONLY_BETTER"
TEACHER_NOT_BETTER = "TEACHER_NOT_BETTER"
STRONG_PRIVILEGED_CURRENT_SUPPORT = "STRONG_PRIVILEGED_CURRENT_SUPPORT"
NO_PRIVILEGED_CURRENT_SUPPORT = "NO_PRIVILEGED_CURRENT_SUPPORT"
STRONG_PREDICTED_CURRENT_BRIDGE_SUPPORT = (
    "STRONG_PREDICTED_CURRENT_BRIDGE_SUPPORT"
)
NO_SUPPORT = "NO_SUPPORT"
MATCHED_TEACHER_TIMESTEP_DEPENDENCE = "MATCHED_TEACHER_TIMESTEP_DEPENDENCE"
INCONCLUSIVE = "INCONCLUSIVE"
PREDICTED_CURRENT_CLEARLY_WORSE_MARGIN = 0.05
PREDICTED_CURRENT_THRESHOLD_ATOL = 1.0e-12
CURRENT_X0_ORACLE_ATOL = 2.0e-2
CURRENT_X0_ORACLE_RTOL = 1.0e-2


@dataclass(frozen=True)
class TeacherFlowAuditState:
    state_id: str
    raw_timestep: int
    noise_index: int
    main_warped_timestep: float
    future_warped_timestep: float
    teacher_future_timestep: float
    main_sigma: float
    future_sigma: float
    current_noise: torch.Tensor
    future_noise: torch.Tensor
    current_state: torch.Tensor
    future_state: torch.Tensor
    exact_mcp_target: torch.Tensor
    noisy_batch: NFSFFullSequenceNoisyBatch
    provenance: dict[str, Any]


@dataclass(frozen=True)
class FlowPrediction:
    state_id: str
    branch: str
    flow: torch.Tensor
    x0: torch.Tensor
    proof: dict[str, Any]


@dataclass(frozen=True)
class TeacherFlowAuditResult:
    manifest: dict[str, Any]
    tensors: dict[str, Any]


def select_validation_zero_identity(sample_plan: Mapping[str, Any]) -> str:
    identities = sample_plan.get("validation_sample_identities")
    if not isinstance(identities, Sequence) or isinstance(identities, (str, bytes)):
        raise RuntimeError("sample plan validation_sample_identities must be a sequence")
    if not identities:
        raise RuntimeError("sample plan has no validation identities")
    identity = str(identities[0])
    fixed = sample_plan.get("fixed_decode_validation_identity")
    if fixed is not None and str(fixed) != identity:
        raise RuntimeError("fixed decode identity must equal validation identity 0")
    return identity


def select_validation32_identities(sample_plan: Mapping[str, Any]) -> dict[str, Any]:
    identities = deployment.sample_plan_validation_identities(sample_plan)
    if len(identities) != TEACHER_FLOW_AUDIT_VALIDATION_COUNT:
        raise RuntimeError(
            "multi-identity Teacher-flow audit requires exactly "
            f"{TEACHER_FLOW_AUDIT_VALIDATION_COUNT} validation identities"
        )
    if len(set(identities)) != len(identities):
        raise RuntimeError("multi-identity Teacher-flow audit requires unique identities")
    positions = list(
        range(
            0,
            TEACHER_FLOW_AUDIT_VALIDATION_COUNT,
            TEACHER_FLOW_AUDIT_MULTI_VALIDATION_STRIDE,
        )
    )
    selected = [str(identities[position]) for position in positions]
    if len(selected) != TEACHER_FLOW_AUDIT_MULTI_VALIDATION_COUNT:
        raise RuntimeError("multi-identity Teacher-flow selection count mismatch")
    identity_list_sha256 = deployment.canonical_json_sha256(
        {"validation_sample_identities": list(identities)}
    )
    selection_payload = {
        "mode": TEACHER_FLOW_AUDIT_MODE_MULTI_VALIDATION32,
        "validation_identity_count": TEACHER_FLOW_AUDIT_VALIDATION_COUNT,
        "selected_identity_count": TEACHER_FLOW_AUDIT_MULTI_VALIDATION_COUNT,
        "stride": TEACHER_FLOW_AUDIT_MULTI_VALIDATION_STRIDE,
        "positions": positions,
        "identity_strings": selected,
        "identity_list_sha256": identity_list_sha256,
        "selection_rule": "validation positions 0,8,16,...,248 from exact 256 list",
    }
    return {
        **selection_payload,
        "selection_fingerprint_sha256": deployment.canonical_json_sha256(
            selection_payload
        ),
    }


def build_flow_match_scheduler(*, shift: float, device: torch.device | str):
    return route_eq.build_flow_match_scheduler(shift=float(shift), device=device)


def load_teacher_flow_student_checkpoint_record(
    path: Path | str,
    *,
    expected_checkpoint_step: int,
    expected_training_git_sha: str | None = None,
    expected_official_sha256: str = deployment.OFFICIAL_SELF_FORCING_CHECKPOINT_SHA256,
) -> deployment.DeploymentCheckpointRecord:
    checkpoint_path = Path(path)
    step = int(expected_checkpoint_step)
    if step == deployment.FULL_SEQUENCE_GLOBAL_STEP:
        return deployment.load_full_sequence_checkpoint_record(
            checkpoint_path,
            expected_training_git_sha=(
                deployment.TRAINING_CHECKPOINT_GIT_SHA
                if expected_training_git_sha is None
                else str(expected_training_git_sha)
            ),
            expected_official_sha256=str(expected_official_sha256),
        )
    expected_name = f"checkpoint_step{step:06d}.pt"
    if checkpoint_path.name != expected_name:
        raise RuntimeError(
            "teacher-flow audit checkpoint filename mismatch: "
            f"expected {expected_name}"
        )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"teacher-flow checkpoint not found: {checkpoint_path}")
    actual_sha = file_sha256(checkpoint_path)
    validation = _validate_student_checkpoint_sidecars(
        checkpoint_path,
        expected_sha256=actual_sha,
        expected_checkpoint_step=step,
    )
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    _validate_student_checkpoint_payload(
        payload,
        checkpoint_sha256=actual_sha,
        expected_checkpoint_step=step,
        expected_training_git_sha=expected_training_git_sha,
        expected_official_sha256=str(expected_official_sha256),
    )
    state_dict = payload["generator"]
    if int(validation.get("generator_key_count", len(state_dict))) != len(state_dict):
        raise RuntimeError("teacher-flow validation generator_key_count mismatch")
    return deployment.DeploymentCheckpointRecord(
        path=str(checkpoint_path.resolve()),
        sha256=actual_sha,
        checkpoint_type=f"teacher_flow_student_step{step}",
        load_mode=TEACHER_FLOW_AUDIT_STUDENT_LOAD_MODE,
        generator_state_dict=state_dict,
        global_step=int(payload["global_step"]),
        training_git_sha=str(payload["git_sha"]),
        payload=_slim_student_checkpoint_payload(payload),
        validation_sidecar=validation,
    )


def validate_frozen_teacher_model(
    teacher_generator: Any,
    *,
    checkpoint: deployment.DeploymentCheckpointRecord,
) -> dict[str, Any]:
    state_dict = checkpoint.generator_state_dict
    mcp_tensor_count = deployment.count_mcp_tensors(state_dict)
    if mcp_tensor_count != 0:
        raise RuntimeError("frozen Teacher checkpoint contains MCP tensors")
    if getattr(teacher_generator, "training", False):
        raise RuntimeError("frozen Teacher must be in eval mode")
    parameters = list(teacher_generator.parameters())
    trainable = [
        name for name, param in teacher_generator.named_parameters() if param.requires_grad
    ]
    if trainable:
        raise RuntimeError("frozen Teacher has trainable parameters")
    if getattr(teacher_generator, "mcp", None) is not None:
        raise RuntimeError("frozen Teacher generator must be Main-only")
    return {
        "checkpoint_type": str(checkpoint.checkpoint_type),
        "checkpoint_sha256": str(checkpoint.sha256),
        "load_mode": str(checkpoint.load_mode),
        "mcp_tensor_count": int(mcp_tensor_count),
        "eval_mode": True,
        "requires_grad_false": True,
        "parameter_count": int(len(parameters)),
    }


def validate_frozen_student_model(
    student_generator: Any,
    *,
    checkpoint: deployment.DeploymentCheckpointRecord,
) -> dict[str, Any]:
    if getattr(student_generator, "training", False):
        raise RuntimeError("frozen Student must be in eval mode")
    trainable = [
        name
        for name, param in student_generator.named_parameters()
        if param.requires_grad
    ]
    if trainable:
        raise RuntimeError("frozen Student has trainable parameters")
    parameters = list(student_generator.parameters())
    return {
        "checkpoint_type": str(checkpoint.checkpoint_type),
        "checkpoint_sha256": str(checkpoint.sha256),
        "load_mode": str(checkpoint.load_mode),
        "mcp_tensor_count": int(
            deployment.count_mcp_tensors(checkpoint.generator_state_dict)
        ),
        "eval_mode": True,
        "requires_grad_false": True,
        "parameter_count": int(len(parameters)),
    }


def parameter_sha256_report(generator: Any, *, role: str) -> dict[str, Any]:
    records = {}
    for name, parameter in generator.named_parameters():
        tensor = parameter.detach().cpu()
        records[str(name)] = {
            "sha256": tensor_sha256(tensor),
            "shape": [int(dim) for dim in tensor.shape],
            "dtype": str(tensor.dtype),
            "requires_grad": bool(parameter.requires_grad),
        }
    payload = {
        "role": str(role),
        "parameter_count": int(len(records)),
        "parameters": records,
    }
    return {
        "role": str(role),
        "parameter_count": int(len(records)),
        "fingerprint_sha256": deployment.canonical_json_sha256(payload),
        "parameters": records,
    }


def compare_parameter_sha256_reports(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
) -> dict[str, Any]:
    mismatches = []
    before_parameters = before.get("parameters", {})
    after_parameters = after.get("parameters", {})
    if not isinstance(before_parameters, Mapping) or not isinstance(
        after_parameters,
        Mapping,
    ):
        raise RuntimeError("parameter SHA reports must contain parameter maps")
    for name, record in before_parameters.items():
        after_record = after_parameters.get(name)
        if not isinstance(after_record, Mapping):
            mismatches.append(str(name))
        elif str(record.get("sha256")) != str(after_record.get("sha256")):
            mismatches.append(str(name))
    extra_after = sorted(set(str(name) for name in after_parameters) - set(
        str(name) for name in before_parameters
    ))
    mismatches.extend(extra_after)
    return {
        "role": str(before.get("role", after.get("role", ""))),
        "before_fingerprint_sha256": str(before.get("fingerprint_sha256")),
        "after_fingerprint_sha256": str(after.get("fingerprint_sha256")),
        "parameter_count_before": int(before.get("parameter_count", 0)),
        "parameter_count_after": int(after.get("parameter_count", 0)),
        "all_sha256_exact_match": len(mismatches) == 0,
        "mismatch_parameter_names": mismatches,
    }


def require_no_parameter_mutation(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    *,
    role: str,
) -> dict[str, Any]:
    comparison = compare_parameter_sha256_reports(before, after)
    if comparison["all_sha256_exact_match"] is not True:
        raise RuntimeError(f"{role} parameter mutation detected")
    return {
        **comparison,
        "parameter_mutation_detected": False,
    }


def validate_teacher_flow_artifact_identity(
    *,
    sample_plan: Mapping[str, Any],
    teacher_manifest_sha256: str,
    checkpoint_payload: Mapping[str, Any],
    selected_identity: str,
) -> dict[str, Any]:
    sample_plan_sha = str(sample_plan.get("sample_plan_sha256", ""))
    checkpoint_sample_plan_sha = _payload_string(
        checkpoint_payload,
        "sample_plan_sha256",
    )
    checkpoint_manifest_sha = _payload_string(
        checkpoint_payload,
        "manifest_sha256",
    )
    if sample_plan_sha != checkpoint_sample_plan_sha:
        raise RuntimeError("current sample_plan SHA differs from student checkpoint")
    if str(teacher_manifest_sha256) != checkpoint_manifest_sha:
        raise RuntimeError("current teacher manifest SHA differs from student checkpoint")
    validation_position = deployment.selected_validation_position(
        sample_plan,
        selected_identity,
    )
    return {
        "status": "PASS",
        "sample_plan_sha256": sample_plan_sha,
        "teacher_manifest_sha256": str(teacher_manifest_sha256),
        "selected_identity": str(selected_identity),
        "selected_validation_position": int(validation_position),
        "default_fixed_decode_validation_identity": str(
            sample_plan.get("fixed_decode_validation_identity")
        ),
        "checkpoint_sample_plan_sha256_source": _payload_string_source(
            checkpoint_payload,
            "sample_plan_sha256",
        ),
        "checkpoint_manifest_sha256_source": _payload_string_source(
            checkpoint_payload,
            "manifest_sha256",
        ),
    }


def validate_teacher_flow_artifact_identity_selection(
    *,
    sample_plan: Mapping[str, Any],
    teacher_manifest_sha256: str,
    checkpoint_payload: Mapping[str, Any],
    identity_selection: Mapping[str, Any],
) -> dict[str, Any]:
    sample_plan_sha = str(sample_plan.get("sample_plan_sha256", ""))
    checkpoint_sample_plan_sha = _payload_string(
        checkpoint_payload,
        "sample_plan_sha256",
    )
    checkpoint_manifest_sha = _payload_string(
        checkpoint_payload,
        "manifest_sha256",
    )
    if sample_plan_sha != checkpoint_sample_plan_sha:
        raise RuntimeError("current sample_plan SHA differs from student checkpoint")
    if str(teacher_manifest_sha256) != checkpoint_manifest_sha:
        raise RuntimeError("current teacher manifest SHA differs from student checkpoint")
    _validate_identity_selection(identity_selection)
    per_identity = []
    for index, identity in enumerate(identity_selection["identity_strings"]):
        validation_position = deployment.selected_validation_position(
            sample_plan,
            str(identity),
        )
        expected_position = int(identity_selection["positions"][index])
        if validation_position != expected_position:
            raise RuntimeError("selected validation identity position mismatch")
        per_identity.append(
            {
                "identity_index": int(index),
                "sample_identity": str(identity),
                "validation_position": int(validation_position),
            }
        )
    return {
        "status": "PASS",
        "sample_plan_sha256": sample_plan_sha,
        "teacher_manifest_sha256": str(teacher_manifest_sha256),
        "identity_selection": dict(identity_selection),
        "per_identity": per_identity,
        "checkpoint_sample_plan_sha256_source": _payload_string_source(
            checkpoint_payload,
            "sample_plan_sha256",
        ),
        "checkpoint_manifest_sha256_source": _payload_string_source(
            checkpoint_payload,
            "manifest_sha256",
        ),
    }


def validate_multi_identity_student_checkpoint_contract(
    checkpoint: deployment.DeploymentCheckpointRecord,
) -> dict[str, Any]:
    if int(checkpoint.global_step) != 6500:
        raise RuntimeError("multi-identity Teacher-flow audit requires step6500")
    payload = checkpoint.payload
    if not isinstance(payload, Mapping):
        raise RuntimeError("multi-identity student checkpoint payload missing")
    if payload.get("schema") != route_eq.FULL_SEQUENCE_TRAINER_SCHEMA:
        raise RuntimeError(
            "multi-identity Teacher-flow audit requires formal full-sequence "
            "trainer checkpoint"
        )
    if deployment.count_mcp_tensors(checkpoint.generator_state_dict) <= 0:
        raise RuntimeError("multi-identity student checkpoint missing MCP tensors")
    return {
        "status": "PASS",
        "required_global_step": 6500,
        "actual_global_step": int(checkpoint.global_step),
        "required_schema": route_eq.FULL_SEQUENCE_TRAINER_SCHEMA,
        "actual_schema": str(payload["schema"]),
        "checkpoint_sha256": str(checkpoint.sha256),
    }


def build_teacher_flow_audit_states(
    *,
    source_noise: torch.Tensor,
    teacher_target: torch.Tensor,
    main_scheduler: Any,
    mcp_scheduler: Any,
    noise_seed: int = memorization.DEFAULT_NOISE_SEED,
    raw_timesteps: Sequence[int] = TEACHER_FLOW_AUDIT_RAW_TIMESTEPS,
    noise_realizations_per_raw: int = TEACHER_FLOW_AUDIT_NOISE_REALIZATIONS_PER_RAW,
    state_id_prefix: str | None = None,
    sample_identity: str | None = None,
    validation_position: int | None = None,
    identity_index: int | None = None,
) -> tuple[TeacherFlowAuditState, ...]:
    flow_audit._validate_source_and_teacher(source_noise, teacher_target)
    if tuple(int(value) for value in raw_timesteps) != TEACHER_FLOW_AUDIT_RAW_TIMESTEPS:
        raise ValueError("teacher-flow audit raw timestep grid is locked")
    noise_count = int(noise_realizations_per_raw)
    if noise_count not in (
        TEACHER_FLOW_AUDIT_NOISE_REALIZATIONS_PER_RAW,
        TEACHER_FLOW_AUDIT_MULTI_NOISE_REALIZATIONS_PER_RAW,
    ):
        raise ValueError("teacher-flow audit requires one or four noises per raw")

    states: list[TeacherFlowAuditState] = []
    for raw_timestep in raw_timesteps:
        raw = int(raw_timestep)
        for noise_index in range(noise_count):
            states.append(
                _build_teacher_flow_audit_state(
                    source_noise=source_noise,
                    teacher_target=teacher_target,
                    main_scheduler=main_scheduler,
                    mcp_scheduler=mcp_scheduler,
                    noise_seed=int(noise_seed),
                    raw_timestep=raw,
                    noise_index=int(noise_index),
                    state_id_prefix=state_id_prefix,
                    sample_identity=sample_identity,
                    validation_position=validation_position,
                    identity_index=identity_index,
                )
            )
    _require_teacher_flow_state_contract(
        states,
        expected_noise_realizations_per_raw=noise_count,
    )
    return tuple(states)


def build_predicted_current_oracle_recheck_state(
    *,
    source_noise: torch.Tensor,
    teacher_target: torch.Tensor,
    main_scheduler: Any,
    mcp_scheduler: Any,
    noise_seed: int = memorization.DEFAULT_NOISE_SEED,
    state_id_prefix: str | None = None,
    sample_identity: str | None = None,
    validation_position: int | None = None,
    identity_index: int | None = None,
) -> TeacherFlowAuditState:
    flow_audit._validate_source_and_teacher(source_noise, teacher_target)
    state = _build_teacher_flow_audit_state(
        source_noise=source_noise,
        teacher_target=teacher_target,
        main_scheduler=main_scheduler,
        mcp_scheduler=mcp_scheduler,
        noise_seed=int(noise_seed),
        raw_timestep=PREDICTED_CURRENT_ORACLE_RECHECK_RAW_TIMESTEP,
        noise_index=PREDICTED_CURRENT_ORACLE_RECHECK_NOISE_INDEX,
        state_id_prefix=state_id_prefix,
        sample_identity=sample_identity,
        validation_position=validation_position,
        identity_index=identity_index,
    )
    if int(state.raw_timestep) != PREDICTED_CURRENT_ORACLE_RECHECK_RAW_TIMESTEP:
        raise RuntimeError("predicted-current oracle recheck raw timestep mismatch")
    if int(state.noise_index) != PREDICTED_CURRENT_ORACLE_RECHECK_NOISE_INDEX:
        raise RuntimeError("predicted-current oracle recheck noise index mismatch")
    return state


def build_predicted_current_oracle_recheck_validation0_all_raw_states(
    *,
    source_noise: torch.Tensor,
    teacher_target: torch.Tensor,
    main_scheduler: Any,
    mcp_scheduler: Any,
    noise_seed: int = memorization.DEFAULT_NOISE_SEED,
    state_id_prefix: str | None = None,
    sample_identity: str | None = None,
    validation_position: int | None = None,
    identity_index: int | None = None,
) -> tuple[TeacherFlowAuditState, ...]:
    states = build_teacher_flow_audit_states(
        source_noise=source_noise,
        teacher_target=teacher_target,
        main_scheduler=main_scheduler,
        mcp_scheduler=mcp_scheduler,
        noise_seed=int(noise_seed),
        raw_timesteps=TEACHER_FLOW_AUDIT_RAW_TIMESTEPS,
        noise_realizations_per_raw=TEACHER_FLOW_AUDIT_MULTI_NOISE_REALIZATIONS_PER_RAW,
        state_id_prefix=state_id_prefix,
        sample_identity=sample_identity,
        validation_position=validation_position,
        identity_index=identity_index,
    )
    actual_plan = tuple(
        (int(state.raw_timestep), int(state.noise_index)) for state in states
    )
    expected_plan = tuple((int(raw), 0) for raw in TEACHER_FLOW_AUDIT_RAW_TIMESTEPS)
    if actual_plan != expected_plan:
        raise RuntimeError("validation0 all-raw recheck state plan mismatch")
    return states


def _build_teacher_flow_audit_state(
    *,
    source_noise: torch.Tensor,
    teacher_target: torch.Tensor,
    main_scheduler: Any,
    mcp_scheduler: Any,
    noise_seed: int,
    raw_timestep: int,
    noise_index: int,
    state_id_prefix: str | None,
    sample_identity: str | None,
    validation_position: int | None,
    identity_index: int | None,
) -> TeacherFlowAuditState:
    clean_current = _chunk(teacher_target, CURRENT_CHUNK_INDEX)
    clean_future = _chunk(teacher_target, FUTURE_CHUNK_INDEX)
    source_current = _chunk(source_noise, CURRENT_CHUNK_INDEX)
    source_future = _chunk(source_noise, FUTURE_CHUNK_INDEX)
    raw = int(raw_timestep)
    noise_idx = int(noise_index)
    main_t = memorization._warp_raw_timestep(raw, shift=DEFAULT_S_MAIN)
    future_t = memorization._warp_raw_timestep(raw, shift=DEFAULT_S_MCP)
    current_noise, current_noise_record = memorization._noise_for_realization(
        template=source_current,
        source_noise=source_noise,
        teacher_target=teacher_target,
        raw_timestep=raw,
        noise_index=noise_idx,
        role="current_chunk1",
        base_seed=int(noise_seed),
    )
    future_noise, future_noise_record = memorization._noise_for_realization(
        template=source_future,
        source_noise=source_noise,
        teacher_target=teacher_target,
        raw_timestep=raw,
        noise_index=noise_idx,
        role="future_chunk2",
        base_seed=int(noise_seed),
    )
    main_timestep = route_eq._timestep(main_t, clean_current)
    future_timestep = route_eq._timestep(future_t, clean_future)
    current_state = route_eq._add_noise_chunk(
        main_scheduler,
        clean=clean_current,
        noise=current_noise,
        timestep=main_timestep,
        name="teacher_flow_current_state",
    )
    future_state = route_eq._add_noise_chunk(
        mcp_scheduler,
        clean=clean_future,
        noise=future_noise,
        timestep=future_timestep,
        name="teacher_flow_future_state",
    )
    exact_mcp_target = route_eq._training_target_chunk(
        mcp_scheduler,
        clean=clean_future,
        noise=future_noise,
        timestep=future_timestep,
        name="teacher_flow_exact_mcp_target",
    )
    main_sigma = flow_audit._resolved_sigma(
        main_scheduler,
        main_timestep,
        clean_current,
    )
    future_sigma = flow_audit._resolved_sigma(
        mcp_scheduler,
        future_timestep,
        clean_future,
    )
    teacher_future_timestep = float(future_sigma) * float(
        DEFAULT_NUM_TRAIN_TIMESTEPS
    )
    noisy_batch = _build_full_sequence_validation_noisy_batch(
        source_noise=source_noise,
        teacher_target=teacher_target,
        main_scheduler=main_scheduler,
        mcp_scheduler=mcp_scheduler,
        raw_timestep=raw,
        main_timestep_value=main_t,
        future_timestep_value=future_t,
        current_noise=current_noise,
        future_noise=future_noise,
        current_state=current_state,
        future_state=future_state,
        exact_mcp_target=exact_mcp_target,
    )
    base_state_id = f"raw{raw:03d}_noise{noise_idx}"
    if state_id_prefix:
        state_id = f"{state_id_prefix}_{base_state_id}"
    else:
        state_id = base_state_id
    provenance = {
        "state_id": state_id,
        "base_state_id": base_state_id,
        "raw_timestep": raw,
        "noise_index": noise_idx,
        "main_shift": DEFAULT_S_MAIN,
        "mcp_shift": DEFAULT_S_MCP,
        "main_warped_timestep": float(main_t),
        "future_warped_timestep": float(future_t),
        "teacher_future_timestep": float(teacher_future_timestep),
        "main_sigma": float(main_sigma),
        "future_sigma": float(future_sigma),
        "teacher_timestep_contract": "teacher_timestep = physical_sigma * 1000",
        "current_noise": dict(current_noise_record),
        "future_noise": dict(future_noise_record),
        "history_chunk0": _tensor_record(
            _chunk(teacher_target, HISTORY_CHUNK_INDEX)
        ),
        "clean_current": _tensor_record(clean_current),
        "clean_future": _tensor_record(clean_future),
        "current_state": _tensor_record(current_state),
        "future_state": _tensor_record(future_state),
        "exact_mcp_target": _tensor_record(exact_mcp_target),
        "main_timestep_sha256": tensor_sha256(main_timestep.detach().cpu()),
        "future_timestep_sha256": tensor_sha256(future_timestep.detach().cpu()),
    }
    if sample_identity is not None:
        provenance["sample_identity"] = str(sample_identity)
    if validation_position is not None:
        provenance["validation_position"] = int(validation_position)
    if identity_index is not None:
        provenance["identity_index"] = int(identity_index)
    return TeacherFlowAuditState(
        state_id=state_id,
        raw_timestep=raw,
        noise_index=noise_idx,
        main_warped_timestep=float(main_t),
        future_warped_timestep=float(future_t),
        teacher_future_timestep=float(teacher_future_timestep),
        main_sigma=float(main_sigma),
        future_sigma=float(future_sigma),
        current_noise=current_noise,
        future_noise=future_noise,
        current_state=current_state,
        future_state=future_state,
        exact_mcp_target=exact_mcp_target,
        noisy_batch=noisy_batch,
        provenance=provenance,
    )


def run_student_mcp_full_sequence_predictions(
    generator: Any,
    *,
    states: Sequence[TeacherFlowAuditState],
    teacher_target: torch.Tensor,
    conditional_dict: Mapping[str, Any],
    direct_clean_context_kv: bool = False,
) -> dict[str, FlowPrediction]:
    _require_teacher_flow_state_contract(states)
    result: dict[str, FlowPrediction] = {}
    model = route_eq._model_with_block_mask(generator)
    was_training = bool(getattr(generator, "training", False))
    if hasattr(generator, "eval"):
        generator.eval()
    try:
        for state in states:
            anchors = build_full_sequence_mcp_anchor_inputs(state.noisy_batch)
            selected_anchor = _selected_anchor1_depth1(anchors)
            if tensor_sha256(selected_anchor["future_noises"][0].detach().cpu()) != tensor_sha256(
                state.future_state.detach().cpu()
            ):
                raise RuntimeError("student MCP future input differs from audit state")
            original_block_mask = model.block_mask
            try:
                model.block_mask = None
                with route_eq._capture_mcp_pre_hook(generator.mcp) as capture:
                    outputs, rng_guard = deployment._call_with_rng_guard(
                        device=teacher_target.device,
                        label="teacher_flow_student_full_sequence_forward",
                        fn=lambda: _student_full_sequence_call(
                            generator,
                            state=state,
                            teacher_target=teacher_target,
                            conditional_dict=conditional_dict,
                            anchors=anchors,
                            direct_clean_context_kv=bool(direct_clean_context_kv),
                        ),
                    )
            finally:
                model.block_mask = original_block_mask
            selected = route_eq._selected_mcp_call(capture, route="training_route")
            mcp_by_depth = tuple(route_eq._output_field(outputs, "mcp_flow_preds_by_depth"))
            if len(mcp_by_depth) != len(FULL_SEQUENCE_DEPTHS):
                raise RuntimeError("student full-sequence route must return depth1/2/3")
            mcp_flow = mcp_by_depth[0][:, CURRENT_CHUNK_INDEX]
            _require_finite_tensor(mcp_flow, name="student_mcp_flow")
            mcp_x0 = manual_flow_to_x0(
                future_state=state.future_state,
                flow=mcp_flow,
                sigma=state.future_sigma,
                name="student_mcp_x0",
            )
            proof = {
                "branch": STUDENT_MCP_BRANCH,
                "route": "forward_full_sequence_next_forcing",
                "direct_clean_context_kv": bool(direct_clean_context_kv),
                "uses_deployment_serial_rollout": False,
                "uses_wrapper_auto_x0": False,
                "mcp_current_input_sha256": tensor_sha256(
                    _chunk(state.noisy_batch.noisy_main, CURRENT_CHUNK_INDEX).detach().cpu()
                ),
                "audit_current_state_sha256": tensor_sha256(
                    state.current_state.detach().cpu()
                ),
                "mcp_future_input_sha256": tensor_sha256(
                    selected_anchor["future_noises"][0].detach().cpu()
                ),
                "audit_future_state_sha256": tensor_sha256(
                    state.future_state.detach().cpu()
                ),
                "mcp_timestep_sha256": tensor_sha256(
                    selected_anchor["timesteps"][0].detach().cpu()
                ),
                "mcp_pre_hook_timestep_sha256": tensor_sha256(
                    selected["timestep"].detach().cpu()
                ),
                "selected_mcp_future_start_frames": list(selected["future_start_frames"]),
                "selected_mcp_call_index": int(selected["call_index"]),
                "forward_rng": dict(rng_guard),
                "student_frozen": _parameters_are_frozen(generator),
                "optimizer_step_executed": False,
            }
            proof["current_state_exact"] = (
                proof["mcp_current_input_sha256"]
                == proof["audit_current_state_sha256"]
            )
            proof["future_state_exact"] = (
                proof["mcp_future_input_sha256"]
                == proof["audit_future_state_sha256"]
            )
            if not proof["current_state_exact"] or not proof["future_state_exact"]:
                raise RuntimeError("student same-state proof failed")
            if proof["student_frozen"] is not True:
                raise RuntimeError("student MCP branch parameters are not frozen")
            result[state.state_id] = FlowPrediction(
                state_id=state.state_id,
                branch=STUDENT_MCP_BRANCH,
                flow=mcp_flow.detach().clone(),
                x0=mcp_x0.detach().clone(),
                proof=proof,
            )
    finally:
        if hasattr(generator, "train"):
            generator.train(was_training)
    return result


def run_student_predicted_current_predictions(
    *,
    runtime_factory: Any,
    states: Sequence[TeacherFlowAuditState],
    source_noise: torch.Tensor,
    teacher_target: torch.Tensor,
    teacher_payload: Mapping[str, Any],
    conditional_dict: Mapping[str, Any],
    main_scheduler: Any,
) -> dict[str, FlowPrediction]:
    _require_teacher_flow_state_contract(states)
    rng_plan = deployment.build_absolute_chunk_rng_plan(
        source_noise=source_noise,
        rollout_seed=int(teacher_payload["rollout_seed"]),
        num_denoising_steps=len(TEACHER_FLOW_AUDIT_RAW_TIMESTEPS),
        chunk_frames=FULL_SEQUENCE_CHUNK_FRAMES,
    )
    result: dict[str, FlowPrediction] = {}
    for state in states:
        result[state.state_id] = _run_student_predicted_current_state(
            runtime_factory=runtime_factory,
            state=state,
            teacher_target=teacher_target,
            conditional_dict=conditional_dict,
            rng_plan=rng_plan,
            main_scheduler=main_scheduler,
        )
    return result


def run_teacher_branch_predictions(
    *,
    runtime_factory: Any,
    states: Sequence[TeacherFlowAuditState],
    source_noise: torch.Tensor,
    teacher_target: torch.Tensor,
    teacher_payload: Mapping[str, Any],
    conditional_dict: Mapping[str, Any],
    student_current_predictions: Mapping[str, FlowPrediction],
) -> dict[str, dict[str, FlowPrediction]]:
    _require_teacher_flow_state_contract(states)
    rng_plan = deployment.build_absolute_chunk_rng_plan(
        source_noise=source_noise,
        rollout_seed=int(teacher_payload["rollout_seed"]),
        num_denoising_steps=len(TEACHER_FLOW_AUDIT_RAW_TIMESTEPS),
        chunk_frames=FULL_SEQUENCE_CHUNK_FRAMES,
    )
    predictions: dict[str, dict[str, FlowPrediction]] = {}
    for state in states:
        predicted_current = _prediction_for(
            student_current_predictions,
            state.state_id,
            STUDENT_PREDICTED_CURRENT_BRANCH,
        )
        predictions[state.state_id] = {
            TEACHER_MATCHED_CURRENT_BRANCH: _run_teacher_matched_current_branch(
                runtime_factory=runtime_factory,
                state=state,
                teacher_target=teacher_target,
                conditional_dict=conditional_dict,
                rng_plan=rng_plan,
            ),
            TEACHER_PREDICTED_CURRENT_BRANCH: _run_teacher_predicted_current_branch(
                runtime_factory=runtime_factory,
                state=state,
                teacher_target=teacher_target,
                conditional_dict=conditional_dict,
                rng_plan=rng_plan,
                predicted_current=predicted_current,
            ),
            TEACHER_PRIVILEGED_CURRENT_BRANCH: _run_teacher_clean_current_branch(
                runtime_factory=runtime_factory,
                state=state,
                teacher_target=teacher_target,
                conditional_dict=conditional_dict,
                rng_plan=rng_plan,
            ),
        }
    return predictions


def build_teacher_flow_audit_result(
    *,
    states: Sequence[TeacherFlowAuditState],
    student_predictions: Mapping[str, FlowPrediction],
    student_current_predictions: Mapping[str, FlowPrediction],
    teacher_predictions: Mapping[str, Mapping[str, FlowPrediction]],
    sample_identity: str,
    checkpoint_summary: Mapping[str, Any],
    student_summary: Mapping[str, Any] | None = None,
    teacher_summary: Mapping[str, Any],
    common_inputs: Mapping[str, Any],
    common_inputs_fingerprint_sha256: str,
    runtime_git_sha: str,
    training_checkpoint_git_sha: str,
) -> TeacherFlowAuditResult:
    _require_16_state_contract(states)
    state_records = build_teacher_flow_state_records(
        states=states,
        student_predictions=student_predictions,
        student_current_predictions=student_current_predictions,
        teacher_predictions=teacher_predictions,
        sample_identity=sample_identity,
        validation_position=_selected_validation_position_from_common(common_inputs),
    )
    tensors: dict[str, Any] = {
        "schema": f"{TEACHER_FLOW_AUDIT_SCHEMA}_tensors_v1",
        "states": {},
    }
    for state in states:
        student = _prediction_for(student_predictions, state.state_id, STUDENT_MCP_BRANCH)
        current = _prediction_for(
            student_current_predictions,
            state.state_id,
            STUDENT_PREDICTED_CURRENT_BRANCH,
        )
        matched = _teacher_prediction_for(
            teacher_predictions,
            state.state_id,
            TEACHER_MATCHED_CURRENT_BRANCH,
        )
        predicted = _teacher_prediction_for(
            teacher_predictions,
            state.state_id,
            TEACHER_PREDICTED_CURRENT_BRANCH,
        )
        clean = _teacher_prediction_for(
            teacher_predictions,
            state.state_id,
            TEACHER_PRIVILEGED_CURRENT_BRANCH,
        )
        tensors["states"][state.state_id] = {
            "current_state": state.current_state.detach().cpu(),
            "future_state": state.future_state.detach().cpu(),
            "exact_mcp_target": state.exact_mcp_target.detach().cpu(),
            "student_mcp_flow": student.flow.detach().cpu(),
            "student_mcp_x0": student.x0.detach().cpu(),
            "student_predicted_current_flow": current.flow.detach().cpu(),
            "student_predicted_current_x0_hat": current.x0.detach().cpu(),
            "teacher_matched_flow": matched.flow.detach().cpu(),
            "teacher_matched_x0": matched.x0.detach().cpu(),
            "teacher_predicted_current_flow": predicted.flow.detach().cpu(),
            "teacher_predicted_current_x0": predicted.x0.detach().cpu(),
            "teacher_privileged_flow": clean.flow.detach().cpu(),
            "teacher_privileged_x0": clean.x0.detach().cpu(),
        }
    aggregates = aggregate_teacher_flow_metrics(state_records)
    bridge_statistics = predicted_current_bridge_statistics(aggregates=aggregates)
    label = diagnostic_label_from_metrics(state_records)
    manifest = {
        "schema": TEACHER_FLOW_AUDIT_SCHEMA,
        "status": "PASS",
        "diagnostic_only": True,
        "non_deployable": True,
        "canonical_training_eligible": False,
        "canonical_deployment_eligible": False,
        "writes_checkpoint": False,
        "runs_backward": False,
        "uses_optimizer": False,
        "mode": TEACHER_FLOW_AUDIT_MODE_SINGLE,
        "sample_identity": str(sample_identity),
        "state_count": len(state_records),
        "raw_timesteps": list(TEACHER_FLOW_AUDIT_RAW_TIMESTEPS),
        "noise_realizations_per_raw": TEACHER_FLOW_AUDIT_NOISE_REALIZATIONS_PER_RAW,
        "diagnostic_label": label,
        "diagnostic_policy": _diagnostic_policy(),
        "predicted_current_bridge_statistics": bridge_statistics,
        "predicted_current_bridge_policy": _predicted_current_bridge_policy(),
        "state_sigma_matching_contract": _state_sigma_matching_contract(),
        "teacher_routes": _teacher_route_contracts(),
        "student_route": _student_route_contract(student_predictions),
        "student_predicted_current_route": _student_predicted_current_route_contract(
            student_current_predictions
        ),
        "conversion_contract": _conversion_contract(),
        "forbidden_comparisons": _forbidden_comparisons(),
        "scientific_interpretation_boundaries": _scientific_boundaries(),
        "checkpoint": dict(checkpoint_summary),
        "student": dict(student_summary or {}),
        "teacher": dict(teacher_summary),
        "common_inputs": dict(common_inputs),
        "common_inputs_fingerprint_sha256": str(common_inputs_fingerprint_sha256),
        "runtime_git_sha": str(runtime_git_sha),
        "training_checkpoint_git_sha": str(training_checkpoint_git_sha),
        "states": state_records,
        "aggregates": aggregates,
    }
    validate_teacher_flow_audit_manifest(manifest)
    return TeacherFlowAuditResult(manifest=manifest, tensors=tensors)


def build_teacher_flow_state_records(
    *,
    states: Sequence[TeacherFlowAuditState],
    student_predictions: Mapping[str, FlowPrediction],
    student_current_predictions: Mapping[str, FlowPrediction],
    teacher_predictions: Mapping[str, Mapping[str, FlowPrediction]],
    sample_identity: str,
    validation_position: int | None = None,
    identity_index: int | None = None,
) -> list[dict[str, Any]]:
    _require_teacher_flow_state_contract(states)
    state_records = []
    for state in states:
        student = _prediction_for(student_predictions, state.state_id, STUDENT_MCP_BRANCH)
        current = _prediction_for(
            student_current_predictions,
            state.state_id,
            STUDENT_PREDICTED_CURRENT_BRANCH,
        )
        matched = _teacher_prediction_for(
            teacher_predictions,
            state.state_id,
            TEACHER_MATCHED_CURRENT_BRANCH,
        )
        predicted = _teacher_prediction_for(
            teacher_predictions,
            state.state_id,
            TEACHER_PREDICTED_CURRENT_BRANCH,
        )
        clean = _teacher_prediction_for(
            teacher_predictions,
            state.state_id,
            TEACHER_PRIVILEGED_CURRENT_BRANCH,
        )
        privileged_record = _prediction_record(clean)
        record = {
            **state.provenance,
            "sample_identity": str(sample_identity),
            "student": _prediction_record(student),
            "mcp": _prediction_record(student),
            "student_predicted_current": _prediction_record(current),
            "teacher_matched_current": _prediction_record(matched),
            "teacher_predicted_current": _prediction_record(predicted),
            "teacher_privileged_current": privileged_record,
            "teacher_clean_current": privileged_record,
            "metrics": _state_metrics(
                state=state,
                student=student,
                student_current=current,
                teacher_matched=matched,
                teacher_predicted=predicted,
                teacher_clean=clean,
            ),
            "same_state_sigma_proof": _same_state_sigma_proof(
                state=state,
                student=student,
                student_current=current,
                teacher_matched=matched,
                teacher_predicted=predicted,
                teacher_clean=clean,
            ),
        }
        if validation_position is not None:
            record["validation_position"] = int(validation_position)
        if identity_index is not None:
            record["identity_index"] = int(identity_index)
        state_records.append(record)
    return state_records


def build_teacher_flow_multi_identity_manifest(
    *,
    state_records: Sequence[Mapping[str, Any]],
    identity_records: Sequence[Mapping[str, Any]],
    identity_selection: Mapping[str, Any],
    student_checkpoint_contract: Mapping[str, Any],
    checkpoint_summary: Mapping[str, Any],
    student_summary: Mapping[str, Any] | None = None,
    teacher_summary: Mapping[str, Any],
    common_inputs_fingerprints_sha256: Mapping[str, str],
    runtime_git_sha: str,
    training_checkpoint_git_sha: str,
) -> dict[str, Any]:
    _validate_identity_selection(identity_selection)
    _require_multi_identity_state_records(state_records)
    _validate_multi_identity_records(identity_records, identity_selection)
    _validate_common_fingerprints(
        common_inputs_fingerprints_sha256,
        identity_selection,
    )
    aggregates = aggregate_teacher_flow_metrics(state_records)
    paired_statistics = paired_teacher_flow_statistics(state_records)
    privileged_label = privileged_current_generalization_label(
        aggregates=aggregates,
        paired_statistics=paired_statistics,
    )
    bridge_statistics = predicted_current_bridge_statistics(aggregates=aggregates)
    primary_label = predicted_current_bridge_label(
        bridge_statistics=bridge_statistics,
    )
    manifest = {
        "schema": TEACHER_FLOW_AUDIT_SCHEMA,
        "status": "PASS",
        "diagnostic_only": True,
        "non_deployable": True,
        "canonical_training_eligible": False,
        "canonical_deployment_eligible": False,
        "writes_checkpoint": False,
        "runs_backward": False,
        "uses_optimizer": False,
        "mode": TEACHER_FLOW_AUDIT_MODE_MULTI_VALIDATION32,
        "state_count": len(state_records),
        "raw_timesteps": list(TEACHER_FLOW_AUDIT_RAW_TIMESTEPS),
        "noise_realizations_per_raw": (
            TEACHER_FLOW_AUDIT_MULTI_NOISE_REALIZATIONS_PER_RAW
        ),
        "identity_selection": dict(identity_selection),
        "identity_records": [dict(record) for record in identity_records],
        "diagnostic_label": primary_label,
        "primary_diagnostic_label": primary_label,
        "primary_diagnostic_policy": _predicted_current_bridge_policy(),
        "privileged_current_diagnostic_label": privileged_label,
        "privileged_current_diagnostic_policy": _multi_identity_primary_policy(),
        "predicted_current_bridge_statistics": bridge_statistics,
        "matched_teacher_timestep_diagnostic": matched_teacher_timestep_diagnostic(
            state_records
        ),
        "state_sigma_matching_contract": _state_sigma_matching_contract(
            identity_rule="validation positions 0,8,16,...,248",
            noise_realizations_per_raw=(
                TEACHER_FLOW_AUDIT_MULTI_NOISE_REALIZATIONS_PER_RAW
            ),
        ),
        "teacher_routes": _teacher_route_contracts(),
        "student_route": {
            "route": "forward_full_sequence_next_forcing",
            "direct_clean_context_kv": False,
            "uses_deployment_serial_rollout": False,
            "same_route_as_full_sequence_validation": True,
        },
        "conversion_contract": _conversion_contract(),
        "forbidden_comparisons": _forbidden_comparisons(),
        "scientific_interpretation_boundaries": _multi_identity_boundaries(),
        "streaming_contract": _multi_identity_streaming_contract(),
        "checkpoint": dict(checkpoint_summary),
        "student_checkpoint_contract": dict(student_checkpoint_contract),
        "student": dict(student_summary or {}),
        "teacher": dict(teacher_summary),
        "common_inputs_fingerprints_sha256": dict(common_inputs_fingerprints_sha256),
        "runtime_git_sha": str(runtime_git_sha),
        "training_checkpoint_git_sha": str(training_checkpoint_git_sha),
        "states": [dict(record) for record in state_records],
        "aggregates": aggregates,
        "paired_statistics": paired_statistics,
    }
    validate_teacher_flow_audit_manifest(manifest)
    return manifest


def run_teacher_flow_audit(
    *,
    student_generator: Any,
    student_runtime_factory: Any,
    teacher_runtime_factory: Any,
    source_noise: torch.Tensor,
    teacher_target: torch.Tensor,
    teacher_payload: Mapping[str, Any],
    conditional_dict: Mapping[str, Any],
    main_scheduler: Any,
    mcp_scheduler: Any,
    sample_identity: str,
    checkpoint_summary: Mapping[str, Any],
    teacher_summary: Mapping[str, Any],
    common_inputs: Mapping[str, Any],
    common_inputs_fingerprint_sha256: str,
    runtime_git_sha: str,
    training_checkpoint_git_sha: str,
    student_direct_clean_context_kv: bool = False,
) -> TeacherFlowAuditResult:
    states = build_teacher_flow_audit_states(
        source_noise=source_noise,
        teacher_target=teacher_target,
        main_scheduler=main_scheduler,
        mcp_scheduler=mcp_scheduler,
    )
    student_predictions = run_student_mcp_full_sequence_predictions(
        student_generator,
        states=states,
        teacher_target=teacher_target,
        conditional_dict=conditional_dict,
        direct_clean_context_kv=bool(student_direct_clean_context_kv),
    )
    student_current_predictions = run_student_predicted_current_predictions(
        runtime_factory=student_runtime_factory,
        states=states,
        source_noise=source_noise,
        teacher_target=teacher_target,
        teacher_payload=teacher_payload,
        conditional_dict=conditional_dict,
        main_scheduler=main_scheduler,
    )
    teacher_predictions = run_teacher_branch_predictions(
        runtime_factory=teacher_runtime_factory,
        states=states,
        source_noise=source_noise,
        teacher_target=teacher_target,
        teacher_payload=teacher_payload,
        conditional_dict=conditional_dict,
        student_current_predictions=student_current_predictions,
    )
    return build_teacher_flow_audit_result(
        states=states,
        student_predictions=student_predictions,
        student_current_predictions=student_current_predictions,
        teacher_predictions=teacher_predictions,
        sample_identity=sample_identity,
        checkpoint_summary=checkpoint_summary,
        teacher_summary=teacher_summary,
        common_inputs=common_inputs,
        common_inputs_fingerprint_sha256=common_inputs_fingerprint_sha256,
        runtime_git_sha=runtime_git_sha,
        training_checkpoint_git_sha=training_checkpoint_git_sha,
    )


def manual_flow_to_x0(
    *,
    future_state: torch.Tensor,
    flow: torch.Tensor,
    sigma: float,
    name: str,
) -> torch.Tensor:
    if tuple(future_state.shape) != tuple(flow.shape):
        raise RuntimeError(f"{name} flow/state shape mismatch")
    if not (float(sigma) > 0.0):
        raise RuntimeError(f"{name} requires positive physical sigma")
    value = future_state.float() - float(sigma) * flow.float()
    value = value.to(device=future_state.device, dtype=future_state.dtype)
    _require_finite_tensor(value, name=name)
    return value


def reconstruct_x0_from_flow_matching(
    scheduler: Any,
    *,
    state: torch.Tensor,
    flow: torch.Tensor,
    timestep: torch.Tensor,
    name: str,
) -> torch.Tensor:
    if tuple(state.shape) != tuple(flow.shape):
        raise RuntimeError(f"{name} flow/state shape mismatch")
    _require_finite_tensor(state, name=f"{name}_state")
    _require_finite_tensor(flow, name=f"{name}_flow")
    _require_finite_tensor(timestep, name=f"{name}_timestep")
    original_shape = state.shape
    value = scheduler.step(
        flow.flatten(0, 1),
        timestep.flatten(0, 1),
        state.flatten(0, 1),
        to_final=True,
    ).unflatten(0, original_shape[:2])
    value = value.to(device=state.device, dtype=state.dtype)
    _require_finite_tensor(value, name=name)
    return value


def _explicit_x0_like_scheduler_step_arithmetic(
    scheduler: Any,
    *,
    state: torch.Tensor,
    flow: torch.Tensor,
    timestep: torch.Tensor,
    name: str,
) -> torch.Tensor:
    if tuple(state.shape) != tuple(flow.shape):
        raise RuntimeError(f"{name} flow/state shape mismatch")
    _require_finite_tensor(state, name=f"{name}_state")
    _require_finite_tensor(flow, name=f"{name}_flow")
    _require_finite_tensor(timestep, name=f"{name}_timestep")
    original_shape = state.shape
    state_flat = state.flatten(0, 1)
    flow_flat = flow.flatten(0, 1)
    timestep_flat = timestep.flatten(0, 1) if timestep.ndim == 2 else timestep
    sigmas = scheduler.sigmas.to(flow.device)
    timesteps = scheduler.timesteps.to(flow.device)
    timestep_id = torch.argmin(
        (timesteps.unsqueeze(0) - timestep_flat.unsqueeze(1)).abs(),
        dim=1,
    )
    sigma = sigmas[timestep_id].reshape(-1, 1, 1, 1)
    sigma_final = 1 if (
        bool(getattr(scheduler, "inverse_timesteps", False))
        or bool(getattr(scheduler, "reverse_sigmas", False))
    ) else 0
    value = state_flat + flow_flat * (sigma_final - sigma)
    value = value.unflatten(0, original_shape[:2])
    value = value.to(device=state.device, dtype=state.dtype)
    _require_finite_tensor(value, name=name)
    return value


def exact_current_flow_conversion_oracle(
    main_scheduler: Any,
    *,
    state: TeacherFlowAuditState,
    teacher_target: torch.Tensor,
) -> dict[str, Any]:
    clean_current = _chunk(teacher_target, CURRENT_CHUNK_INDEX)
    current_timestep = route_eq._timestep(
        state.main_warped_timestep,
        state.current_state,
    )
    diagnostic = _exact_current_flow_oracle_diagnostic(
        main_scheduler,
        audit_state=state,
        clean_current=clean_current,
        timestep=current_timestep,
    )
    exact_current_flow = diagnostic["tensors"]["exact_current_flow"]
    reconstructed = diagnostic["tensors"]["reconstructed"]
    max_abs = float((reconstructed.float() - clean_current.float()).abs().max().item())
    mse = route_eq._mse(reconstructed, clean_current)
    passed = bool(
        torch.allclose(
            reconstructed.float(),
            clean_current.float(),
            atol=CURRENT_X0_ORACLE_ATOL,
            rtol=CURRENT_X0_ORACLE_RTOL,
        )
    )
    formula_matches_scheduler = bool(
        torch.allclose(
            diagnostic["tensors"]["recon_formula_actual"].float(),
            reconstructed.float(),
            atol=CURRENT_X0_ORACLE_ATOL,
            rtol=CURRENT_X0_ORACLE_RTOL,
        )
    )
    failure_reasons = _current_oracle_failure_reasons(
        diagnostic=diagnostic,
        scheduler_path_pass=passed,
        formula_matches_scheduler=formula_matches_scheduler,
    )
    if failure_reasons:
        _raise_current_oracle_failure(
            failure_reasons[0],
            diagnostic=_strip_oracle_diagnostic_tensors(diagnostic),
        )
    if not formula_matches_scheduler:
        _raise_current_oracle_failure(
            "explicit_formula_differs_from_scheduler_step",
            diagnostic=_strip_oracle_diagnostic_tensors(diagnostic),
        )
    return {
        "status": "PASS",
        "source": "FlowMatchScheduler.step(..., to_final=True)",
        "training_target_source": "FlowMatchScheduler.training_target(clean, noise, timestep)",
        "derived_formula": "x0 = x_t - sigma * flow",
        "atol": float(CURRENT_X0_ORACLE_ATOL),
        "rtol": float(CURRENT_X0_ORACLE_RTOL),
        "max_abs_error": max_abs,
        "mse": float(mse),
        "scheduler_step_to_final": True,
        "formula_matches_scheduler": True,
        "diagnostic": _strip_oracle_diagnostic_tensors(diagnostic),
        "exact_current_flow": _tensor_record(exact_current_flow),
        "reconstructed_current_x0": _tensor_record(reconstructed),
        "clean_current": _tensor_record(clean_current),
    }


def _exact_current_flow_oracle_diagnostic(
    scheduler: Any,
    *,
    audit_state: TeacherFlowAuditState,
    clean_current: torch.Tensor,
    timestep: torch.Tensor,
) -> dict[str, Any]:
    expected_main_timestep = memorization._warp_raw_timestep(
        audit_state.raw_timestep,
        shift=DEFAULT_S_MAIN,
    )
    scheduler_shift = _scheduler_shift(scheduler)
    scheduler_sigma = flow_audit._resolved_sigma(
        scheduler,
        timestep,
        audit_state.current_state,
    )
    exact_flow_sched = route_eq._training_target_chunk(
        scheduler,
        clean=clean_current,
        noise=audit_state.current_noise,
        timestep=timestep,
        name="teacher_flow_exact_current_target",
    )
    recon_sched = reconstruct_x0_from_flow_matching(
        scheduler,
        state=audit_state.current_state,
        flow=exact_flow_sched,
        timestep=timestep,
        name="teacher_flow_exact_current_x0_oracle",
    )
    recon_same_dtype_explicit = _explicit_x0_like_scheduler_step_arithmetic(
        scheduler,
        state=audit_state.current_state,
        flow=exact_flow_sched,
        timestep=timestep,
        name="teacher_flow_exact_current_same_dtype_formula",
    )
    recon_formula_actual = audit_state.current_state.float() - (
        float(scheduler_sigma) * exact_flow_sched.float()
    )
    regenerated_state = route_eq._add_noise_chunk(
        scheduler,
        clean=clean_current,
        noise=audit_state.current_noise,
        timestep=timestep,
        name="teacher_flow_regenerated_current_state",
    )
    recon_regenerated = reconstruct_x0_from_flow_matching(
        scheduler,
        state=regenerated_state,
        flow=exact_flow_sched,
        timestep=timestep,
        name="teacher_flow_regenerated_current_x0_oracle",
    )
    clean32 = clean_current.detach().float()
    noise32 = audit_state.current_noise.detach().float()
    timestep32 = route_eq._timestep(audit_state.main_warped_timestep, clean32)
    state32 = route_eq._add_noise_chunk(
        scheduler,
        clean=clean32,
        noise=noise32,
        timestep=timestep32,
        name="teacher_flow_current_state_float32_reference",
    )
    flow32 = route_eq._training_target_chunk(
        scheduler,
        clean=clean32,
        noise=noise32,
        timestep=timestep32,
        name="teacher_flow_current_target_float32_reference",
    )
    recon32 = reconstruct_x0_from_flow_matching(
        scheduler,
        state=state32,
        flow=flow32,
        timestep=timestep32,
        name="teacher_flow_current_x0_float32_reference",
    )
    scheduler_sigma32 = flow_audit._resolved_sigma(
        scheduler,
        timestep32,
        clean32,
    )
    recon32_explicit = state32.float() - (
        float(scheduler_sigma32) * flow32.float()
    )
    recon_sched_vs_clean = _tensor_error_stats(recon_sched, clean_current)
    recon_same_dtype_explicit_vs_clean = _tensor_error_stats(
        recon_same_dtype_explicit,
        clean_current,
    )
    recon_formula_actual_vs_clean = _tensor_error_stats(
        recon_formula_actual,
        clean_current,
    )
    recon_regenerated_vs_clean = _tensor_error_stats(
        recon_regenerated,
        clean_current,
    )
    recon32_vs_clean32 = _tensor_error_stats(recon32, clean32)
    noisy_state_vs_regenerated = _tensor_error_stats(
        audit_state.current_state,
        regenerated_state,
    )
    noisy_state_vs_regenerated["torch_equal"] = bool(
        torch.equal(audit_state.current_state, regenerated_state)
    )
    recon_sched_vs_formula_actual = _tensor_error_stats(
        recon_sched,
        recon_formula_actual,
    )
    recon_sched_vs_same_dtype_explicit = _tensor_error_stats(
        recon_sched,
        recon_same_dtype_explicit,
    )
    recon32_explicit_vs_clean32 = _tensor_error_stats(recon32_explicit, clean32)
    recon32_scheduler_vs_explicit = _tensor_error_stats(recon32, recon32_explicit)
    return {
        "schema": "teacher_flow_exact_current_oracle_diagnostic_v1",
        "identity": audit_state.provenance.get("sample_identity"),
        "identity_index": audit_state.provenance.get("identity_index"),
        "validation_position": audit_state.provenance.get("validation_position"),
        "state_id": audit_state.state_id,
        "raw_timestep": int(audit_state.raw_timestep),
        "noise_index": int(audit_state.noise_index),
        "warped_current_timestep": float(audit_state.main_warped_timestep),
        "expected_main_warped_timestep": float(expected_main_timestep),
        "future_warped_timestep": float(audit_state.future_warped_timestep),
        "teacher_future_timestep": float(audit_state.teacher_future_timestep),
        "resolved_sigma": float(scheduler_sigma),
        "state_main_sigma": float(audit_state.main_sigma),
        "state_future_sigma": float(audit_state.future_sigma),
        "main_scheduler_expected_contract": {
            "shift": float(DEFAULT_S_MAIN),
            "timestep": float(audit_state.main_warped_timestep),
            "sigma": float(audit_state.main_sigma),
        },
        "scheduler_actually_passed": {
            "class": type(scheduler).__name__,
            "id": int(id(scheduler)),
            "shift": scheduler_shift,
            "timestep": float(timestep.flatten()[0].detach().cpu().item()),
            "sigma": float(scheduler_sigma),
        },
        "sources": {
            "clean_current_tensor": "teacher_target chunk1",
            "current_noise_tensor": audit_state.provenance.get("current_noise"),
            "current_noisy_state": (
                "main_scheduler.add_noise(clean_current, current_noise, "
                "main_timestep)"
            ),
            "exact_current_flow": (
                "main_scheduler.training_target(clean_current, current_noise, "
                "main_timestep)"
            ),
            "reconstruction": (
                "main_scheduler.step(exact_current_flow, main_timestep, "
                "current_noisy_state, to_final=True)"
            ),
            "explicit_same_dtype_reference": (
                "manual FlowMatchScheduler.step arithmetic with the same "
                "state/flow/sigma dtype conversion and final state dtype cast"
            ),
            "explicit_float32_reference": (
                "current_noisy_state.float() - sigma * exact_current_flow.float()"
            ),
        },
        "tensor_records": {
            "current_noise": _tensor_record(audit_state.current_noise),
            "current_state": _tensor_record(audit_state.current_state),
            "regenerated_state": _tensor_record(regenerated_state),
        },
        "tensor_dtypes_devices": {
            "clean": _tensor_meta(clean_current),
            "noise": _tensor_meta(audit_state.current_noise),
            "state": _tensor_meta(audit_state.current_state),
            "exact_flow": _tensor_meta(exact_flow_sched),
            "reconstructed": _tensor_meta(recon_sched),
            "same_dtype_explicit": _tensor_meta(recon_same_dtype_explicit),
            "float32_explicit": _tensor_meta(recon_formula_actual),
            "regenerated_state": _tensor_meta(regenerated_state),
            "float32_clean": _tensor_meta(clean32),
            "float32_state": _tensor_meta(state32),
            "float32_flow": _tensor_meta(flow32),
            "float32_reconstructed": _tensor_meta(recon32),
            "float32_explicit_reference": _tensor_meta(recon32_explicit),
        },
        "noisy_state_vs_regenerated": noisy_state_vs_regenerated,
        "recon_sched_vs_clean": recon_sched_vs_clean,
        "recon_same_dtype_explicit_vs_clean": recon_same_dtype_explicit_vs_clean,
        "recon_formula_actual_vs_clean": recon_formula_actual_vs_clean,
        "recon_regenerated_vs_clean": recon_regenerated_vs_clean,
        "recon_sched_vs_same_dtype_explicit": recon_sched_vs_same_dtype_explicit,
        "recon_sched_vs_formula_actual": recon_sched_vs_formula_actual,
        "recon_sched_vs_float32_explicit": recon_sched_vs_formula_actual,
        "explicit_same_dtype_reference": {
            "source": (
                "manual FlowMatchScheduler.step arithmetic, preserving scheduler "
                "sigma lookup and final cast to current state dtype"
            ),
            "state_dtype": str(audit_state.current_state.dtype),
            "flow_dtype": str(exact_flow_sched.dtype),
            "sigma_tensor_dtype": str(scheduler.sigmas.dtype),
            "result_dtype": str(recon_same_dtype_explicit.dtype),
            "recon_vs_clean": recon_same_dtype_explicit_vs_clean,
            "scheduler_vs_explicit": recon_sched_vs_same_dtype_explicit,
        },
        "float32_explicit_reference": {
            "state32_source": "current_noisy_state.float()",
            "flow32_source": "exact_current_flow.float()",
            "sigma": float(scheduler_sigma),
            "recon32_formula_source": "state.float() - sigma * flow.float()",
            "recon32_formula_vs_clean": recon_formula_actual_vs_clean,
            "scheduler_vs_explicit": recon_sched_vs_formula_actual,
        },
        "float32_reference": {
            "state32_source": "scheduler.add_noise(clean32, noise32, timestep)",
            "flow32_source": "scheduler.training_target(clean32, noise32, timestep)",
            "recon32_source": "scheduler.step(flow32, timestep, state32, to_final=True)",
            "recon32_vs_clean32": recon32_vs_clean32,
            "recon32_explicit_source": "state32 - sigma32 * flow32",
            "recon32_explicit_vs_clean32": recon32_explicit_vs_clean32,
            "recon32_scheduler_vs_explicit": recon32_scheduler_vs_explicit,
            "resolved_sigma32": float(scheduler_sigma32),
            "passes_existing_oracle_tolerance": bool(
                torch.allclose(
                    recon32.float(),
                    clean32.float(),
                    atol=CURRENT_X0_ORACLE_ATOL,
                    rtol=CURRENT_X0_ORACLE_RTOL,
                )
            ),
        },
        "tensors": {
            "exact_current_flow": exact_flow_sched,
            "reconstructed": recon_sched,
            "recon_same_dtype_explicit": recon_same_dtype_explicit,
            "recon_formula_actual": recon_formula_actual,
        },
    }


def _current_oracle_failure_reasons(
    *,
    diagnostic: Mapping[str, Any],
    scheduler_path_pass: bool,
    formula_matches_scheduler: bool,
) -> list[str]:
    reasons: list[str] = []
    actual_shift = diagnostic["scheduler_actually_passed"]["shift"]
    if actual_shift is None or abs(float(actual_shift) - float(DEFAULT_S_MAIN)) > 1.0e-9:
        reasons.append("scheduler_shift_not_main")
    if (
        abs(
            float(diagnostic["warped_current_timestep"])
            - float(diagnostic["expected_main_warped_timestep"])
        )
        > 1.0e-5
    ):
        reasons.append("current_timestep_not_main_shift5_raw_warp")
    if (
        abs(
            float(diagnostic["resolved_sigma"])
            - float(diagnostic["state_main_sigma"])
        )
        > 1.0e-7
    ):
        reasons.append("scheduler_sigma_differs_from_state_main_sigma")
    if diagnostic["noisy_state_vs_regenerated"]["torch_equal"] is not True:
        reasons.append("current_state_noise_timestep_round_trip_mismatch")
    if (
        diagnostic["float32_reference"]["passes_existing_oracle_tolerance"]
        is not True
    ):
        reasons.append("float32_reference_oracle_failed")
    if not scheduler_path_pass:
        reasons.append("scheduler_reconstruction_exceeds_existing_tolerance")
    if not formula_matches_scheduler:
        reasons.append("explicit_formula_differs_from_scheduler_step")
    return reasons


def _strip_oracle_diagnostic_tensors(
    diagnostic: Mapping[str, Any],
) -> dict[str, Any]:
    payload = {key: value for key, value in diagnostic.items() if key != "tensors"}
    payload["failure_case"] = _classify_current_oracle_failure(payload)
    return payload


def _classify_current_oracle_failure(diagnostic: Mapping[str, Any]) -> str:
    if diagnostic["noisy_state_vs_regenerated"]["torch_equal"] is not True:
        return "state_noise_timestep_provenance_bug"
    sched_formula_error = float(
        diagnostic["recon_sched_vs_same_dtype_explicit"]["max_abs"]
    )
    if sched_formula_error > CURRENT_X0_ORACLE_ATOL:
        return "scheduler_identity_or_timestep_lookup_bug"
    if diagnostic["float32_reference"]["passes_existing_oracle_tolerance"] is True:
        sched_error = float(diagnostic["recon_sched_vs_clean"]["max_abs"])
        if sched_error > CURRENT_X0_ORACLE_ATOL:
            return "bf16_quantized_state_contract"
    return "scheduler_or_flow_semantic_error"


def _raise_current_oracle_failure(
    reason: str,
    *,
    diagnostic: Mapping[str, Any],
) -> None:
    payload = json.dumps(diagnostic, sort_keys=True)
    raise RuntimeError(
        "exact current flow-to-x0 conversion oracle failed: "
        f"{reason}; diagnostic={payload}"
    )


def classify_predicted_current_oracle_recheck(
    diagnostic: Mapping[str, Any],
    *,
    original_bf16_oracle_pass: bool,
) -> str:
    if not _recheck_main_scheduler_contract_ok(diagnostic):
        return SCHEDULER_MISMATCH
    if not _recheck_raw_warp_contract_ok(diagnostic):
        return SCHEDULER_MISMATCH
    if not _recheck_state_provenance_contract_ok(diagnostic):
        return STATE_PROVENANCE_MISMATCH
    if not _recheck_scheduler_explicit_contract_ok(diagnostic):
        return SCHEDULER_MISMATCH
    if not _recheck_float32_reference_ok(diagnostic):
        return SEMANTIC_MISMATCH
    if bool(original_bf16_oracle_pass):
        return EXACT_PASS
    if _recheck_actual_bf16_reconstruction_only_failed(diagnostic):
        return BF16_QUANTIZED_STATE_CONTRACT
    return SEMANTIC_MISMATCH


def build_predicted_current_oracle_recheck_artifact(
    *,
    diagnostic: Mapping[str, Any],
    original_bf16_oracle_pass: bool,
    runtime_git_sha: str,
    sample_identity: str,
    identity_index: int,
    validation_position: int,
    student_parameters_before: Mapping[str, Any],
    student_parameters_after: Mapping[str, Any],
    teacher_parameters_before: Mapping[str, Any],
    teacher_parameters_after: Mapping[str, Any],
    rng_before: str,
    rng_after: str,
    common_inputs_fingerprint_sha256: str | None = None,
    artifact_identity: Mapping[str, Any] | None = None,
    student_checkpoint_contract: Mapping[str, Any] | None = None,
    checkpoint_summary: Mapping[str, Any] | None = None,
    student_summary: Mapping[str, Any] | None = None,
    teacher_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    classification = classify_predicted_current_oracle_recheck(
        diagnostic,
        original_bf16_oracle_pass=bool(original_bf16_oracle_pass),
    )
    student_comparison = compare_parameter_sha256_reports(
        student_parameters_before,
        student_parameters_after,
    )
    teacher_comparison = compare_parameter_sha256_reports(
        teacher_parameters_before,
        teacher_parameters_after,
    )
    rng_unchanged = str(rng_before) == str(rng_after)
    safety_pass = (
        student_comparison["all_sha256_exact_match"] is True
        and teacher_comparison["all_sha256_exact_match"] is True
        and rng_unchanged
    )
    diagnostic_pass = classification in (EXACT_PASS, BF16_QUANTIZED_STATE_CONTRACT)
    status = "PASS" if diagnostic_pass and safety_pass else "FAIL"
    scheduler_actual = diagnostic["scheduler_actually_passed"]
    return {
        "schema": PREDICTED_CURRENT_ORACLE_RECHECK_SCHEMA,
        "status": status,
        "mode": PREDICTED_CURRENT_ORACLE_RECHECK_MODE,
        "diagnostic_classification": classification,
        "runtime_git_sha": str(runtime_git_sha),
        "identity_index": int(identity_index),
        "validation_position": int(validation_position),
        "raw_timestep": int(diagnostic["raw_timestep"]),
        "noise_index": int(diagnostic["noise_index"]),
        "sample_identity": str(sample_identity),
        "state_id": str(diagnostic.get("state_id", "")),
        "main_scheduler_class": str(scheduler_actual["class"]),
        "main_scheduler_shift": scheduler_actual["shift"],
        "main_warped_timestep": float(diagnostic["warped_current_timestep"]),
        "main_sigma": float(diagnostic["state_main_sigma"]),
        "main_scheduler_sigma": float(diagnostic["resolved_sigma"]),
        "current_state_vs_regenerated": _stats_subset(
            diagnostic["noisy_state_vs_regenerated"],
            keys=("torch_equal", "mse", "max_abs", "mean_abs"),
        ),
        "bf16_scheduler_reconstruction": _stats_subset(
            diagnostic["recon_sched_vs_clean"],
            keys=("mse", "max_abs", "mean_abs", "p99_abs"),
        ),
        "bf16_explicit_formula_reconstruction": _stats_subset(
            diagnostic["recon_same_dtype_explicit_vs_clean"],
            keys=("mse", "max_abs", "mean_abs", "p99_abs"),
        ),
        "bf16_explicit_same_dtype": _stats_subset(
            diagnostic["recon_same_dtype_explicit_vs_clean"],
            keys=("mse", "max_abs", "mean_abs", "p99_abs"),
        ),
        "scheduler_vs_explicit": _stats_subset(
            diagnostic["recon_sched_vs_same_dtype_explicit"],
            keys=("mse", "max_abs", "mean_abs"),
        ),
        "scheduler_vs_same_dtype_explicit": _stats_subset(
            diagnostic["recon_sched_vs_same_dtype_explicit"],
            keys=("mse", "max_abs", "mean_abs"),
        ),
        "scheduler_vs_float32_explicit": _stats_subset(
            diagnostic["recon_sched_vs_float32_explicit"],
            keys=("mse", "max_abs", "mean_abs"),
        ),
        "float32_explicit_reference": _stats_subset(
            diagnostic["float32_explicit_reference"][
                "recon32_formula_vs_clean"
            ],
            keys=("mse", "max_abs", "mean_abs", "p99_abs"),
        ),
        "float32_reference": _stats_subset(
            diagnostic["float32_reference"]["recon32_vs_clean32"],
            keys=("mse", "max_abs", "mean_abs", "p99_abs"),
        ),
        "precision_diagnostic": {
            "scheduler_step_result": diagnostic["tensor_dtypes_devices"][
                "reconstructed"
            ],
            "same_dtype_explicit_result": diagnostic["tensor_dtypes_devices"][
                "same_dtype_explicit"
            ],
            "float32_explicit_result": diagnostic["tensor_dtypes_devices"][
                "float32_explicit"
            ],
            "semantic_scheduler_gate": "scheduler_vs_same_dtype_explicit",
            "float32_explicit_is_diagnostic_only": True,
        },
        "oracle_atol": float(CURRENT_X0_ORACLE_ATOL),
        "oracle_rtol": float(CURRENT_X0_ORACLE_RTOL),
        "original_bf16_oracle_pass": bool(original_bf16_oracle_pass),
        "diagnostic_failure_case": diagnostic.get("failure_case"),
        "student_parameter_sha_before": str(
            student_parameters_before["fingerprint_sha256"]
        ),
        "student_parameter_sha_after": str(
            student_parameters_after["fingerprint_sha256"]
        ),
        "student_parameters_unchanged": bool(
            student_comparison["all_sha256_exact_match"]
        ),
        "student_parameter_comparison": student_comparison,
        "teacher_parameter_sha_before": str(
            teacher_parameters_before["fingerprint_sha256"]
        ),
        "teacher_parameter_sha_after": str(
            teacher_parameters_after["fingerprint_sha256"]
        ),
        "teacher_parameters_unchanged": bool(
            teacher_comparison["all_sha256_exact_match"]
        ),
        "teacher_parameter_comparison": teacher_comparison,
        "rng_before": str(rng_before),
        "rng_after": str(rng_after),
        "rng_unchanged": bool(rng_unchanged),
        "backward_executed": False,
        "optimizer_step_executed": False,
        "checkpoint_written": False,
        "common_inputs_fingerprint_sha256": common_inputs_fingerprint_sha256,
        "artifact_identity": (
            None if artifact_identity is None else dict(artifact_identity)
        ),
        "student_checkpoint_contract": (
            None
            if student_checkpoint_contract is None
            else dict(student_checkpoint_contract)
        ),
        "checkpoint": None if checkpoint_summary is None else dict(checkpoint_summary),
        "student": None if student_summary is None else dict(student_summary),
        "teacher": None if teacher_summary is None else dict(teacher_summary),
        "exit_code_contract": (
            "rc=0 for exact_pass or bf16_quantized_state_contract when "
            "parameter and RNG safety checks pass; mismatch or safety failure rc=1"
        ),
    }


def build_predicted_current_oracle_recheck_state_record(
    *,
    diagnostic: Mapping[str, Any],
    original_bf16_oracle_pass: bool,
) -> dict[str, Any]:
    classification = classify_predicted_current_oracle_recheck(
        diagnostic,
        original_bf16_oracle_pass=bool(original_bf16_oracle_pass),
    )
    tensor_records = diagnostic["tensor_records"]
    return {
        "state_id": str(diagnostic.get("state_id", "")),
        "identity_index": diagnostic.get("identity_index"),
        "validation_position": diagnostic.get("validation_position"),
        "raw_timestep": int(diagnostic["raw_timestep"]),
        "noise_index": int(diagnostic["noise_index"]),
        "warped_timestep": float(diagnostic["warped_current_timestep"]),
        "sigma": float(diagnostic["state_main_sigma"]),
        "current_noise_sha256": str(tensor_records["current_noise"]["sha256"]),
        "current_state_sha256": str(tensor_records["current_state"]["sha256"]),
        "regenerated_state_sha256": str(
            tensor_records["regenerated_state"]["sha256"]
        ),
        "state_vs_regenerated": _stats_subset(
            diagnostic["noisy_state_vs_regenerated"],
            keys=("torch_equal", "mse", "max_abs", "mean_abs"),
        ),
        "original_bf16_oracle_pass": bool(original_bf16_oracle_pass),
        "bf16_scheduler_reconstruction": _stats_subset(
            diagnostic["recon_sched_vs_clean"],
            keys=("mse", "max_abs", "mean_abs", "p99_abs"),
        ),
        "bf16_explicit_same_dtype": _stats_subset(
            diagnostic["recon_same_dtype_explicit_vs_clean"],
            keys=("mse", "max_abs", "mean_abs", "p99_abs"),
        ),
        "bf16_or_same_dtype_scheduler_difference": _stats_subset(
            diagnostic["recon_sched_vs_same_dtype_explicit"],
            keys=("mse", "max_abs", "mean_abs", "p99_abs"),
        ),
        "float32_explicit_reference": _stats_subset(
            diagnostic["float32_explicit_reference"][
                "recon32_formula_vs_clean"
            ],
            keys=("mse", "max_abs", "mean_abs", "p99_abs"),
        ),
        "scheduler_vs_float32_explicit": _stats_subset(
            diagnostic["recon_sched_vs_float32_explicit"],
            keys=("mse", "max_abs", "mean_abs", "p99_abs"),
        ),
        "diagnostic_classification": classification,
    }


def build_predicted_current_oracle_recheck_validation0_all_raw_artifact(
    *,
    state_records: Sequence[Mapping[str, Any]],
    runtime_git_sha: str,
    sample_identity: str,
    identity_index: int,
    validation_position: int,
    student_parameters_before: Mapping[str, Any],
    student_parameters_after: Mapping[str, Any],
    teacher_parameters_before: Mapping[str, Any],
    teacher_parameters_after: Mapping[str, Any],
    rng_before: str,
    rng_after: str,
    common_inputs_fingerprint_sha256: str | None = None,
    artifact_identity: Mapping[str, Any] | None = None,
    student_checkpoint_contract: Mapping[str, Any] | None = None,
    checkpoint_summary: Mapping[str, Any] | None = None,
    student_summary: Mapping[str, Any] | None = None,
    teacher_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    records = [dict(record) for record in state_records]
    plan = tuple(
        (int(record["raw_timestep"]), int(record["noise_index"]))
        for record in records
    )
    expected_plan = tuple((int(raw), 0) for raw in TEACHER_FLOW_AUDIT_RAW_TIMESTEPS)
    if plan != expected_plan:
        raise RuntimeError("validation0 all-raw artifact state plan mismatch")

    classifications = [str(record["diagnostic_classification"]) for record in records]
    if any(value == SCHEDULER_MISMATCH for value in classifications):
        classification = SCHEDULER_MISMATCH
    elif any(value == STATE_PROVENANCE_MISMATCH for value in classifications):
        classification = STATE_PROVENANCE_MISMATCH
    elif any(value == SEMANTIC_MISMATCH for value in classifications):
        classification = SEMANTIC_MISMATCH
    elif any(value == BF16_QUANTIZED_STATE_CONTRACT for value in classifications):
        classification = BF16_QUANTIZED_STATE_CONTRACT
    else:
        classification = EXACT_PASS

    student_comparison = compare_parameter_sha256_reports(
        student_parameters_before,
        student_parameters_after,
    )
    teacher_comparison = compare_parameter_sha256_reports(
        teacher_parameters_before,
        teacher_parameters_after,
    )
    rng_unchanged = str(rng_before) == str(rng_after)
    safety_pass = (
        student_comparison["all_sha256_exact_match"] is True
        and teacher_comparison["all_sha256_exact_match"] is True
        and rng_unchanged
    )
    diagnostic_pass = classification in (EXACT_PASS, BF16_QUANTIZED_STATE_CONTRACT)
    status = "PASS" if diagnostic_pass and safety_pass else "FAIL"
    return {
        "schema": PREDICTED_CURRENT_ORACLE_RECHECK_ALL_RAW_SCHEMA,
        "status": status,
        "mode": PREDICTED_CURRENT_ORACLE_RECHECK_ALL_RAW_MODE,
        "diagnostic_classification": classification,
        "runtime_git_sha": str(runtime_git_sha),
        "identity_index": int(identity_index),
        "validation_position": int(validation_position),
        "noise_index": PREDICTED_CURRENT_ORACLE_RECHECK_NOISE_INDEX,
        "sample_identity": str(sample_identity),
        "state_count": len(records),
        "raw_timesteps": [int(raw) for raw in TEACHER_FLOW_AUDIT_RAW_TIMESTEPS],
        "formal_identity0_state_order": [
            {
                "state_index": index,
                "raw_timestep": int(record["raw_timestep"]),
                "noise_index": int(record["noise_index"]),
                "state_id": str(record["state_id"]),
            }
            for index, record in enumerate(records)
        ],
        "states": records,
        "all_original_bf16_oracle_pass": all(
            bool(record["original_bf16_oracle_pass"]) for record in records
        ),
        "oracle_atol": float(CURRENT_X0_ORACLE_ATOL),
        "oracle_rtol": float(CURRENT_X0_ORACLE_RTOL),
        "student_parameter_sha_before": str(
            student_parameters_before["fingerprint_sha256"]
        ),
        "student_parameter_sha_after": str(
            student_parameters_after["fingerprint_sha256"]
        ),
        "student_parameters_unchanged": bool(
            student_comparison["all_sha256_exact_match"]
        ),
        "student_parameter_comparison": student_comparison,
        "teacher_parameter_sha_before": str(
            teacher_parameters_before["fingerprint_sha256"]
        ),
        "teacher_parameter_sha_after": str(
            teacher_parameters_after["fingerprint_sha256"]
        ),
        "teacher_parameters_unchanged": bool(
            teacher_comparison["all_sha256_exact_match"]
        ),
        "teacher_parameter_comparison": teacher_comparison,
        "rng_before": str(rng_before),
        "rng_after": str(rng_after),
        "rng_unchanged": bool(rng_unchanged),
        "backward_executed": False,
        "optimizer_step_executed": False,
        "checkpoint_written": False,
        "common_inputs_fingerprint_sha256": common_inputs_fingerprint_sha256,
        "artifact_identity": (
            None if artifact_identity is None else dict(artifact_identity)
        ),
        "student_checkpoint_contract": (
            None
            if student_checkpoint_contract is None
            else dict(student_checkpoint_contract)
        ),
        "checkpoint": None if checkpoint_summary is None else dict(checkpoint_summary),
        "student": None if student_summary is None else dict(student_summary),
        "teacher": None if teacher_summary is None else dict(teacher_summary),
        "exit_code_contract": (
            "rc=0 only when all four state classifications are exact_pass or "
            "bf16_quantized_state_contract and parameter/RNG safety checks pass; "
            "any scheduler/state/semantic mismatch or safety failure rc=1"
        ),
    }


def _stats_subset(
    stats: Mapping[str, Any],
    *,
    keys: Sequence[str],
) -> dict[str, Any]:
    result = {}
    for key in keys:
        value = stats[key]
        if isinstance(value, bool):
            result[key] = bool(value)
        elif value is None:
            result[key] = None
        else:
            result[key] = float(value)
    return result


def _recheck_main_scheduler_contract_ok(diagnostic: Mapping[str, Any]) -> bool:
    actual = diagnostic["scheduler_actually_passed"]
    return (
        str(actual.get("class")) == "FlowMatchScheduler"
        and _near_optional(actual.get("shift"), DEFAULT_S_MAIN, atol=1.0e-9)
    )


def _recheck_raw_warp_contract_ok(diagnostic: Mapping[str, Any]) -> bool:
    return _near_optional(
        diagnostic.get("warped_current_timestep"),
        diagnostic.get("expected_main_warped_timestep"),
        atol=1.0e-5,
    ) and _near_optional(
        diagnostic.get("resolved_sigma"),
        diagnostic.get("state_main_sigma"),
        atol=1.0e-7,
    )


def _recheck_state_provenance_contract_ok(diagnostic: Mapping[str, Any]) -> bool:
    state = diagnostic["noisy_state_vs_regenerated"]
    return state.get("torch_equal") is True


def _recheck_scheduler_explicit_contract_ok(diagnostic: Mapping[str, Any]) -> bool:
    return _max_abs_leq(
        diagnostic["recon_sched_vs_same_dtype_explicit"],
        PREDICTED_CURRENT_ORACLE_RECHECK_SCHEDULER_ATOL,
    )


def _recheck_float32_reference_ok(diagnostic: Mapping[str, Any]) -> bool:
    reference = diagnostic["float32_reference"]
    return (
        reference.get("passes_existing_oracle_tolerance") is True
        and _max_abs_leq(
            reference["recon32_vs_clean32"],
            PREDICTED_CURRENT_ORACLE_RECHECK_FLOAT32_ATOL,
        )
    )


def _recheck_actual_bf16_reconstruction_only_failed(
    diagnostic: Mapping[str, Any],
) -> bool:
    if not _recheck_actual_dtype_is_bf16(diagnostic):
        return False
    scheduler_failed = not _max_abs_leq(
        diagnostic["recon_sched_vs_clean"],
        CURRENT_X0_ORACLE_ATOL,
    )
    formula_failed = not _max_abs_leq(
        diagnostic["recon_same_dtype_explicit_vs_clean"],
        CURRENT_X0_ORACLE_ATOL,
    )
    return scheduler_failed or formula_failed


def _recheck_actual_dtype_is_bf16(diagnostic: Mapping[str, Any]) -> bool:
    metadata = diagnostic.get("tensor_dtypes_devices")
    if not isinstance(metadata, Mapping):
        return False
    for key in ("clean", "noise", "state", "exact_flow", "reconstructed"):
        value = metadata.get(key)
        if not isinstance(value, Mapping):
            return False
        if str(value.get("dtype")) != "torch.bfloat16":
            return False
    return True


def _max_abs_leq(stats: Mapping[str, Any], threshold: float) -> bool:
    return float(stats["max_abs"]) <= float(threshold)


def _near_optional(left: Any, right: Any, *, atol: float) -> bool:
    if left is None or right is None:
        return False
    return abs(float(left) - float(right)) <= float(atol)


def _tensor_meta(tensor: torch.Tensor) -> dict[str, Any]:
    return {
        "shape": [int(value) for value in tensor.shape],
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
    }


def _tensor_error_stats(left: torch.Tensor, right: torch.Tensor) -> dict[str, Any]:
    _require_finite_tensor(left, name="error_stats_left")
    _require_finite_tensor(right, name="error_stats_right")
    diff = (left.detach().float() - right.detach().float()).abs()
    _require_finite_tensor(diff, name="error_stats_abs_diff")
    flat = diff.reshape(-1)
    return {
        "mse": float((diff.square()).mean().detach().cpu().item()),
        "max_abs": float(diff.max().detach().cpu().item()),
        "mean_abs": float(diff.mean().detach().cpu().item()),
        "p99_abs": float(torch.quantile(flat, 0.99).detach().cpu().item()),
    }


def _scheduler_shift(scheduler: Any) -> float | None:
    value = getattr(scheduler, "shift", None)
    if value is None:
        return None
    return float(value)


def aggregate_teacher_flow_metrics(
    state_records: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    grouped: dict[int, list[Mapping[str, Any]]] = {}
    grouped_by_identity: dict[str, list[Mapping[str, Any]]] = {}
    for record in state_records:
        raw = int(record["raw_timestep"])
        grouped.setdefault(raw, []).append(record)
        identity_key = _identity_group_key(record)
        grouped_by_identity.setdefault(identity_key, []).append(record)
    return {
        "by_state": {
            str(record["state_id"]): {
                "sample_identity": str(record.get("sample_identity", "")),
                "validation_position": _optional_int(
                    record.get("validation_position")
                ),
                "raw_timestep": int(record["raw_timestep"]),
                "noise_index": int(record["noise_index"]),
                "metrics": _aggregate_group([record]),
            }
            for record in state_records
        },
        "by_identity": {
            key: {
                "sample_identity": str(records[0].get("sample_identity", key)),
                "validation_position": _optional_int(
                    records[0].get("validation_position")
                ),
                "identity_index": _optional_int(records[0].get("identity_index")),
                "state_count": int(len(records)),
                "metrics": _aggregate_group(records),
            }
            for key, records in sorted(grouped_by_identity.items())
        },
        "by_raw": {
            str(raw): _aggregate_group(records)
            for raw, records in sorted(grouped.items())
        },
        "all_states": _aggregate_group(list(state_records)),
    }


def paired_teacher_flow_statistics(
    state_records: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    _require_non_empty_records(state_records)
    identity_groups: dict[str, list[Mapping[str, Any]]] = {}
    raw_groups: dict[int, list[Mapping[str, Any]]] = {}
    for record in state_records:
        identity_groups.setdefault(_identity_group_key(record), []).append(record)
        raw_groups.setdefault(int(record["raw_timestep"]), []).append(record)

    by_identity = {}
    for key, records in sorted(identity_groups.items()):
        values = _paired_values(records)
        by_identity[key] = {
            "sample_identity": str(records[0].get("sample_identity", key)),
            "validation_position": _optional_int(records[0].get("validation_position")),
            "identity_index": _optional_int(records[0].get("identity_index")),
            "state_count": int(len(records)),
            **values,
        }

    identity_values = list(by_identity.values())
    predicted_identity_wins = sum(
        1 for value in identity_values if value["predicted_better_than_matched"]
    )
    by_raw = {}
    for raw, records in sorted(raw_groups.items()):
        raw_values = [_paired_values([record]) for record in records]
        by_raw[str(raw)] = _paired_summary(raw_values)

    return {
        "by_identity": by_identity,
        "by_raw": by_raw,
        "privileged_identity_flow_win_count": int(
            sum(1 for value in identity_values if value["privileged_flow_win"])
        ),
        "privileged_identity_flow_win_rate": _rate(
            sum(1 for value in identity_values if value["privileged_flow_win"]),
            len(identity_values),
        ),
        "matched_identity_flow_win_count": int(
            sum(1 for value in identity_values if value["matched_flow_win"])
        ),
        "matched_identity_flow_win_rate": _rate(
            sum(1 for value in identity_values if value["matched_flow_win"]),
            len(identity_values),
        ),
        "predicted_better_than_matched_win_count": int(predicted_identity_wins),
        "predicted_better_than_matched_win_rate": _rate(
            predicted_identity_wins,
            len(identity_values),
        ),
        "privileged_identity_flow_reduction_mean": _mean(
            value["privileged_flow_reduction"] for value in identity_values
        ),
        "privileged_identity_flow_reduction_median": _median(
            value["privileged_flow_reduction"] for value in identity_values
        ),
        "privileged_identity_x0_reduction_mean": _mean(
            value["privileged_x0_reduction"] for value in identity_values
        ),
        "privileged_identity_x0_reduction_median": _median(
            value["privileged_x0_reduction"] for value in identity_values
        ),
        "matched_identity_flow_reduction_mean": _mean(
            value["matched_flow_reduction"] for value in identity_values
        ),
        "matched_identity_flow_reduction_median": _median(
            value["matched_flow_reduction"] for value in identity_values
        ),
        "matched_identity_x0_reduction_mean": _mean(
            value["matched_x0_reduction"] for value in identity_values
        ),
        "matched_identity_x0_reduction_median": _median(
            value["matched_x0_reduction"] for value in identity_values
        ),
        "predicted_vs_matched_identity_flow_reduction_mean": _mean(
            value["predicted_vs_matched_flow_reduction"]
            for value in identity_values
        ),
        "predicted_vs_matched_identity_flow_reduction_median": _median(
            value["predicted_vs_matched_flow_reduction"]
            for value in identity_values
        ),
        "predicted_vs_matched_flow_reduction_mean": _mean(
            value["predicted_vs_matched_flow_reduction"]
            for value in identity_values
        ),
        "predicted_vs_matched_flow_reduction_median": _median(
            value["predicted_vs_matched_flow_reduction"]
            for value in identity_values
        ),
        "predicted_vs_matched_identity_x0_reduction_mean": _mean(
            value["predicted_vs_matched_x0_reduction"] for value in identity_values
        ),
        "predicted_vs_matched_identity_x0_reduction_median": _median(
            value["predicted_vs_matched_x0_reduction"] for value in identity_values
        ),
        "gap_recovery_ratio_mean": _mean_defined(
            value["gap_recovery_ratio"] for value in identity_values
        ),
        "gap_recovery_ratio_median": _median_defined(
            value["gap_recovery_ratio"] for value in identity_values
        ),
    }


def predicted_current_bridge_statistics(
    *,
    aggregates: Mapping[str, Any],
) -> dict[str, Any]:
    all_states = _bridge_group_metrics(aggregates["all_states"])
    by_identity = {
        key: {
            "sample_identity": str(value.get("sample_identity", key)),
            "validation_position": _optional_int(value.get("validation_position")),
            "identity_index": _optional_int(value.get("identity_index")),
            "state_count": int(value["state_count"]),
            **_bridge_group_metrics(value["metrics"]),
        }
        for key, value in sorted(aggregates["by_identity"].items())
    }
    by_raw = {
        str(raw): _bridge_group_metrics(metrics)
        for raw, metrics in sorted(aggregates["by_raw"].items())
    }
    identity_values = list(by_identity.values())
    raw_values = list(by_raw.values())
    identity_wins = sum(
        1 for value in identity_values if value["predicted_better_than_matched"]
    )
    raw_clearly_worse = sum(
        1 for value in raw_values if value["predicted_clearly_worse_than_matched"]
    )
    raw_not_worse = sum(
        1
        for value in raw_values
        if _passes_max(value["predicted_flow_mse"], value["matched_flow_mse"])
    )
    return {
        "policy": _predicted_current_bridge_policy(),
        "all_states": all_states,
        "by_identity": by_identity,
        "by_raw": by_raw,
        "identity_predicted_better_than_matched_win_count": int(identity_wins),
        "identity_predicted_better_than_matched_win_rate": _rate(
            identity_wins,
            len(identity_values),
        ),
        "raw_predicted_not_worse_count": int(raw_not_worse),
        "raw_predicted_clearly_worse_than_matched_count": int(raw_clearly_worse),
    }


def predicted_current_bridge_label(
    *,
    bridge_statistics: Mapping[str, Any],
) -> str:
    all_states = bridge_statistics["all_states"]
    policy = bridge_statistics["policy"]
    flow_improvement = float(
        all_states["predicted_vs_matched_flow_reduction"]
    )
    x0_improvement = float(all_states["predicted_vs_matched_x0_reduction"])
    gap_recovery = all_states.get("gap_recovery_ratio")
    identity_win_rate = float(
        bridge_statistics["identity_predicted_better_than_matched_win_rate"]
    )
    identity_win_count = int(
        bridge_statistics["identity_predicted_better_than_matched_win_count"]
    )
    raw_not_worse_count = int(bridge_statistics["raw_predicted_not_worse_count"])
    raw_clearly_worse_count = int(
        bridge_statistics["raw_predicted_clearly_worse_than_matched_count"]
    )
    if (
        _passes_min(
            flow_improvement,
            float(policy["strong"]["all_state_flow_reduction_min"]),
        )
        and _passes_min(
            identity_win_rate,
            float(policy["strong"]["identity_win_rate_min"]),
        )
        and identity_win_count >= int(policy["strong"]["identity_win_count_min"])
        and raw_not_worse_count == len(TEACHER_FLOW_AUDIT_RAW_TIMESTEPS)
        and gap_recovery is not None
        and _passes_min(
            float(gap_recovery),
            float(policy["strong"]["gap_recovery_ratio_min"]),
        )
        and _passes_min(
            x0_improvement,
            float(policy["strong"]["future_x0_reduction_min"]),
        )
    ):
        return STRONG_PREDICTED_CURRENT_BRIDGE_SUPPORT
    if (
        _below_min(
            flow_improvement,
            float(policy["no_support"]["all_state_flow_reduction_lt"]),
        )
        or _below_min(
            identity_win_rate,
            float(policy["no_support"]["identity_win_rate_lt"]),
        )
        or raw_clearly_worse_count
        >= int(policy["no_support"]["raw_clearly_worse_count_gte"])
    ):
        return NO_SUPPORT
    return INCONCLUSIVE


def privileged_current_generalization_label(
    *,
    aggregates: Mapping[str, Any],
    paired_statistics: Mapping[str, Any],
) -> str:
    all_states = aggregates["all_states"]
    mcp_flow = float(all_states["mcp_flow_vs_exact_mse"]["mean"])
    privileged_flow = float(all_states["teacher_clean_flow_vs_exact_mse"]["mean"])
    mcp_x0 = float(all_states["mcp_x0_vs_clean_future_mse"]["mean"])
    privileged_x0 = float(all_states["teacher_clean_x0_vs_clean_future_mse"]["mean"])
    flow_reduction = _relative_reduction(mcp_flow, privileged_flow)
    x0_reduction = _relative_reduction(mcp_x0, privileged_x0)
    identity_win_rate = float(
        paired_statistics["privileged_identity_flow_win_rate"]
    )
    raw_not_worse = all(
        float(raw_metrics["teacher_clean_flow_vs_exact_mse"]["mean"])
        <= float(raw_metrics["mcp_flow_vs_exact_mse"]["mean"])
        for raw_metrics in aggregates["by_raw"].values()
    )
    if (
        flow_reduction >= 0.25
        and identity_win_rate >= 0.75
        and x0_reduction >= 0.25
        and raw_not_worse
    ):
        return STRONG_PRIVILEGED_CURRENT_SUPPORT
    if flow_reduction < 0.10 or identity_win_rate < 0.60:
        return NO_PRIVILEGED_CURRENT_SUPPORT
    return INCONCLUSIVE


def matched_teacher_timestep_diagnostic(
    state_records: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    grouped: dict[int, list[Mapping[str, Any]]] = {}
    for record in state_records:
        grouped.setdefault(int(record["raw_timestep"]), []).append(record)
    by_raw = {}
    for raw, records in sorted(grouped.items()):
        values = [_paired_values([record]) for record in records]
        summary = _paired_summary(values)
        reduction = float(summary["matched_flow_reduction_mean"])
        if reduction > 0.0:
            direction = "better_than_mcp"
        elif reduction < 0.0:
            direction = "worse_than_mcp"
        else:
            direction = "equal_to_mcp"
        by_raw[str(raw)] = {
            "matched_flow_reduction_mean": reduction,
            "matched_flow_reduction_median": float(
                summary["matched_flow_reduction_median"]
            ),
            "matched_flow_win_count": int(summary["matched_flow_win_count"]),
            "matched_flow_win_rate": float(summary["matched_flow_win_rate"]),
            "direction": direction,
        }
    return {
        "label": MATCHED_TEACHER_TIMESTEP_DEPENDENCE,
        "not_primary_gate": True,
        "by_raw": by_raw,
    }


def diagnostic_label_from_metrics(
    state_records: Sequence[Mapping[str, Any]]
) -> str:
    matched_better = _branch_strictly_dominates(
        state_records,
        flow_key="teacher_matched_flow_vs_exact_mse",
        x0_key="teacher_matched_x0_vs_clean_future_mse",
    )
    clean_better = _branch_strictly_dominates(
        state_records,
        flow_key="teacher_clean_flow_vs_exact_mse",
        x0_key="teacher_clean_x0_vs_clean_future_mse",
    )
    if matched_better:
        return TEACHER_MATCHED_STRONGLY_BETTER
    if clean_better:
        return TEACHER_PRIVILEGED_ONLY_BETTER
    matched_partial = _branch_has_any_joint_improvement(
        state_records,
        flow_key="teacher_matched_flow_vs_exact_mse",
        x0_key="teacher_matched_x0_vs_clean_future_mse",
    )
    clean_partial = _branch_has_any_joint_improvement(
        state_records,
        flow_key="teacher_clean_flow_vs_exact_mse",
        x0_key="teacher_clean_x0_vs_clean_future_mse",
    )
    if not matched_partial and not clean_partial:
        return TEACHER_NOT_BETTER
    return INCONCLUSIVE


def validate_teacher_flow_audit_manifest(manifest: Mapping[str, Any]) -> None:
    if manifest.get("schema") != TEACHER_FLOW_AUDIT_SCHEMA:
        raise RuntimeError("teacher-flow audit schema mismatch")
    if manifest.get("status") != "PASS":
        raise RuntimeError("teacher-flow audit status must be PASS")
    if manifest.get("diagnostic_only") is not True:
        raise RuntimeError("teacher-flow audit must be diagnostic-only")
    for field in ("writes_checkpoint", "runs_backward", "uses_optimizer"):
        if manifest.get(field) is not False:
            raise RuntimeError(f"teacher-flow audit forbidden runtime flag set: {field}")
    mode = str(manifest.get("mode", TEACHER_FLOW_AUDIT_MODE_SINGLE))
    if tuple(manifest.get("raw_timesteps", ())) != TEACHER_FLOW_AUDIT_RAW_TIMESTEPS:
        raise RuntimeError("teacher-flow audit raw grid mismatch")
    states = manifest.get("states")
    if not isinstance(states, Sequence) or isinstance(states, (str, bytes)):
        raise RuntimeError("teacher-flow audit states missing")
    if mode == TEACHER_FLOW_AUDIT_MODE_MULTI_VALIDATION32:
        if int(manifest.get("state_count", -1)) != 128:
            raise RuntimeError("multi-identity Teacher-flow audit must contain 128 states")
        if int(manifest.get("noise_realizations_per_raw", -1)) != 1:
            raise RuntimeError("multi-identity Teacher-flow audit noise count mismatch")
        if len(states) != 128:
            raise RuntimeError("multi-identity Teacher-flow state record count mismatch")
        _validate_identity_selection(manifest.get("identity_selection"))
        _require_multi_identity_state_records(states)
        _validate_multi_identity_records(
            manifest.get("identity_records"),
            manifest.get("identity_selection"),
        )
        _validate_common_fingerprints(
            manifest.get("common_inputs_fingerprints_sha256"),
            manifest.get("identity_selection"),
        )
        if manifest.get("primary_diagnostic_label") not in {
            STRONG_PREDICTED_CURRENT_BRIDGE_SUPPORT,
            NO_SUPPORT,
            INCONCLUSIVE,
        }:
            raise RuntimeError("multi-identity Teacher-flow diagnostic label invalid")
        if manifest.get("privileged_current_diagnostic_label") not in {
            STRONG_PRIVILEGED_CURRENT_SUPPORT,
            NO_PRIVILEGED_CURRENT_SUPPORT,
            INCONCLUSIVE,
        }:
            raise RuntimeError("multi-identity privileged diagnostic label invalid")
    else:
        if int(manifest.get("state_count", -1)) != 16:
            raise RuntimeError("teacher-flow audit must contain 16 states")
        if int(manifest.get("noise_realizations_per_raw", -1)) != 4:
            raise RuntimeError("teacher-flow audit noise count mismatch")
        if len(states) != 16:
            raise RuntimeError("teacher-flow audit state record count mismatch")
    for record in states:
        _validate_state_record(record)
    aggregates = manifest.get("aggregates")
    if not isinstance(aggregates, Mapping) or "all_states" not in aggregates:
        raise RuntimeError("teacher-flow audit aggregates missing")
    if "by_state" not in aggregates or "by_identity" not in aggregates:
        raise RuntimeError("teacher-flow audit aggregate hierarchy missing")
    bridge = manifest.get("predicted_current_bridge_statistics")
    if not isinstance(bridge, Mapping) or "all_states" not in bridge:
        raise RuntimeError("predicted-current bridge statistics missing")


def _validate_student_checkpoint_sidecars(
    path: Path,
    *,
    expected_sha256: str,
    expected_checkpoint_step: int,
) -> dict[str, Any]:
    sidecars = deployment.checkpoint_sidecar_paths(path)
    if not sidecars["sha256"].is_file():
        raise RuntimeError("teacher-flow checkpoint SHA256 sidecar is missing")
    if not sidecars["validation"].is_file():
        raise RuntimeError("teacher-flow checkpoint validation sidecar is missing")
    actual_sha = file_sha256(path)
    if actual_sha != str(expected_sha256):
        raise RuntimeError("teacher-flow checkpoint SHA256 mismatch")
    sha_tokens = sidecars["sha256"].read_text(encoding="utf-8").strip().split()
    if not sha_tokens or sha_tokens[0] != actual_sha:
        raise RuntimeError("teacher-flow checkpoint SHA256 sidecar mismatch")
    validation = json.loads(sidecars["validation"].read_text(encoding="utf-8"))
    if validation.get("status") != "PASS":
        raise RuntimeError("teacher-flow checkpoint validation status must be PASS")
    if validation.get("sha256") != actual_sha:
        raise RuntimeError("teacher-flow checkpoint validation SHA mismatch")
    if "path" in validation and validation["path"] != str(path.resolve()):
        raise RuntimeError("teacher-flow checkpoint validation path mismatch")
    if int(validation.get("size_bytes", -1)) != int(path.stat().st_size):
        raise RuntimeError("teacher-flow checkpoint validation size mismatch")
    if int(validation.get("global_step", -1)) != int(expected_checkpoint_step):
        raise RuntimeError("teacher-flow checkpoint validation global_step mismatch")
    return validation


def _validate_student_checkpoint_payload(
    payload: Any,
    *,
    checkpoint_sha256: str,
    expected_checkpoint_step: int,
    expected_training_git_sha: str | None,
    expected_official_sha256: str,
) -> None:
    if not isinstance(payload, Mapping):
        raise TypeError("teacher-flow student checkpoint payload must be a mapping")
    required = {"schema", "status", "global_step", "git_sha", "generator"}
    missing = required - set(payload.keys())
    if missing:
        raise RuntimeError(
            f"teacher-flow student checkpoint missing fields: {sorted(missing)}"
        )
    schema = str(payload["schema"])
    if schema not in TEACHER_FLOW_AUDIT_SUPPORTED_STUDENT_SCHEMAS:
        raise RuntimeError("teacher-flow student checkpoint schema unsupported")
    if int(payload["global_step"]) != int(expected_checkpoint_step):
        raise RuntimeError("teacher-flow student checkpoint global_step mismatch")
    if expected_training_git_sha is not None and str(payload["git_sha"]) != str(
        expected_training_git_sha
    ):
        raise RuntimeError("teacher-flow student checkpoint git SHA mismatch")
    if not _is_sha256(checkpoint_sha256):
        raise RuntimeError("teacher-flow student checkpoint SHA must be valid")
    state_dict = payload["generator"]
    if not isinstance(state_dict, Mapping):
        raise TypeError("teacher-flow student checkpoint generator must be a state dict")
    if deployment.count_mcp_tensors(state_dict) <= 0:
        raise RuntimeError("teacher-flow student checkpoint missing MCP tensors")
    if schema == route_eq.FULL_SEQUENCE_TRAINER_SCHEMA:
        _validate_full_sequence_student_payload(
            payload,
            expected_official_sha256=expected_official_sha256,
        )
    elif schema == NF_SF_MCP_DIRECT_CLEAN_KV_ABLATION_SCHEMA:
        if payload.get("status") != "DIAGNOSTIC_ABLATION":
            raise RuntimeError("teacher-flow ablation checkpoint status mismatch")
        if payload.get("canonical_training_eligible") is not False:
            raise RuntimeError("teacher-flow ablation checkpoint must be non-canonical")
    elif schema == NF_SF_MCP1_ONLY_CONTINUATION_SCHEMA:
        if payload.get("status") != "DIAGNOSTIC_MCP1_ONLY":
            raise RuntimeError("teacher-flow MCP1-only checkpoint status mismatch")
        if payload.get("diagnostic_only") is not True:
            raise RuntimeError("teacher-flow MCP1-only checkpoint must be diagnostic")
    resolved = payload.get("resolved_config")
    if isinstance(resolved, Mapping):
        if int(resolved.get("num_frame_per_block", FULL_SEQUENCE_CHUNK_FRAMES)) != (
            FULL_SEQUENCE_CHUNK_FRAMES
        ):
            raise RuntimeError("teacher-flow student checkpoint nfpb mismatch")


def _validate_full_sequence_student_payload(
    payload: Mapping[str, Any],
    *,
    expected_official_sha256: str,
) -> None:
    if payload.get("run_kind") != route_eq.FULL_SEQUENCE_RUN_KIND:
        raise RuntimeError("teacher-flow full-sequence checkpoint run_kind mismatch")
    if payload.get("objective_version") != route_eq.FULL_SEQUENCE_OBJECTIVE_VERSION:
        raise RuntimeError("teacher-flow full-sequence objective_version mismatch")
    if payload.get("objective_mode") != deployment.FULL_SEQUENCE_OBJECTIVE_MODE:
        raise RuntimeError("teacher-flow full-sequence objective_mode mismatch")
    reference = payload.get("reference_checkpoint")
    if not isinstance(reference, Mapping):
        raise RuntimeError("teacher-flow full-sequence reference checkpoint missing")
    if reference.get("sha256") != str(expected_official_sha256):
        raise RuntimeError("teacher-flow official parent SHA mismatch")
    if payload.get("sample_cursor") != nf_sf_full_sequence_train_cursor(
        int(payload["global_step"])
    ):
        raise RuntimeError("teacher-flow full-sequence sample_cursor mismatch")


def _slim_student_checkpoint_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    keep = (
        "schema",
        "status",
        "diagnostic_only",
        "non_canonical",
        "canonical_training_eligible",
        "canonical_deployment_eligible",
        "deployment_eligible",
        "global_step",
        "git_sha",
        "sample_plan_sha256",
        "manifest_sha256",
        "conditionals_artifact_sha256",
        "resolved_config",
        "metadata",
        "provenance",
        "reference_checkpoint",
    )
    return {key: payload[key] for key in keep if key in payload}


def _build_full_sequence_validation_noisy_batch(
    *,
    source_noise: torch.Tensor,
    teacher_target: torch.Tensor,
    main_scheduler: Any,
    mcp_scheduler: Any,
    raw_timestep: int,
    main_timestep_value: float,
    future_timestep_value: float,
    current_noise: torch.Tensor,
    future_noise: torch.Tensor,
    current_state: torch.Tensor,
    future_state: torch.Tensor,
    exact_mcp_target: torch.Tensor,
) -> NFSFFullSequenceNoisyBatch:
    main_timestep = route_eq._timestep(float(main_timestep_value), teacher_target)
    epsilon_main = source_noise.detach().clone()
    _set_chunk(epsilon_main, CURRENT_CHUNK_INDEX, current_noise)
    noisy_main = _add_noise_like_scheduler(
        main_scheduler,
        teacher_target,
        epsilon_main,
        main_timestep,
    )
    target_flow_main = _training_target_like_scheduler(
        main_scheduler,
        teacher_target,
        epsilon_main,
        main_timestep,
    )
    raw_main = torch.full(
        (teacher_target.shape[0], FULL_SEQUENCE_NUM_CHUNKS),
        int(raw_timestep),
        device=teacher_target.device,
        dtype=torch.int64,
    )
    noisy_depths = []
    target_depths = []
    epsilon_depths = []
    raw_depths = []
    timestep_depths = []
    for depth in FULL_SEQUENCE_DEPTHS:
        count = FULL_SEQUENCE_NUM_CHUNKS - int(depth)
        epsilons = []
        timesteps = []
        clean_chunks = []
        for anchor_index in range(count):
            target_chunk = int(anchor_index) + int(depth)
            clean = _chunk(teacher_target, target_chunk)
            if int(depth) == 1 and int(anchor_index) == CURRENT_CHUNK_INDEX:
                epsilon = future_noise
            else:
                epsilon = _chunk(source_noise, target_chunk).detach().clone()
            clean_chunks.append(clean)
            epsilons.append(epsilon)
            timesteps.append(route_eq._timestep(float(future_timestep_value), clean))
        clean_tensor = torch.stack(clean_chunks, dim=1)
        epsilon_tensor = torch.stack(epsilons, dim=1)
        timestep_tensor = torch.stack(timesteps, dim=1)
        noisy_depths.append(
            _add_noise_for_anchor_chunks(
                mcp_scheduler,
                clean_tensor,
                epsilon_tensor,
                timestep_tensor,
            )
        )
        target_depths.append(
            _training_target_for_anchor_chunks(
                mcp_scheduler,
                clean_tensor,
                epsilon_tensor,
                timestep_tensor,
            )
        )
        epsilon_depths.append(epsilon_tensor)
        raw_depths.append(
            torch.full(
                (teacher_target.shape[0], count),
                int(raw_timestep),
                device=teacher_target.device,
                dtype=torch.int64,
            )
        )
        timestep_depths.append(timestep_tensor)
    if not torch.equal(_chunk(noisy_main, CURRENT_CHUNK_INDEX), current_state):
        raise RuntimeError("full-sequence noisy_main chunk1 does not match current_state")
    if not torch.equal(noisy_depths[0][:, CURRENT_CHUNK_INDEX], future_state):
        raise RuntimeError("full-sequence MCP depth1 anchor1 does not match future_state")
    if not torch.equal(target_depths[0][:, CURRENT_CHUNK_INDEX], exact_mcp_target):
        raise RuntimeError("full-sequence MCP exact target mismatch")
    return NFSFFullSequenceNoisyBatch(
        clean_target=teacher_target,
        noisy_main=noisy_main,
        target_flow_main=target_flow_main,
        epsilon_main=epsilon_main,
        raw_timestep_main=raw_main,
        timestep_main=main_timestep,
        noisy_mcp_depths=tuple(noisy_depths),
        target_flow_mcp_depths=tuple(target_depths),
        epsilon_mcp_depths=tuple(epsilon_depths),
        raw_timestep_mcp_depths=tuple(raw_depths),
        timestep_mcp_depths=tuple(timestep_depths),
        anchor_specs=build_full_sequence_mcp_anchor_specs(
            num_chunks=FULL_SEQUENCE_NUM_CHUNKS,
            chunk_frames=FULL_SEQUENCE_CHUNK_FRAMES,
            depths=FULL_SEQUENCE_DEPTHS,
        ),
    )


def _run_teacher_matched_current_branch(
    *,
    runtime_factory: Any,
    state: TeacherFlowAuditState,
    teacher_target: torch.Tensor,
    conditional_dict: Mapping[str, Any],
    rng_plan: Mapping[str, Any],
) -> FlowPrediction:
    runtime = runtime_factory()
    _validate_teacher_runtime(runtime)
    history = _recache_clean_chunk(
        runtime=runtime,
        teacher_target=teacher_target,
        conditional_dict=conditional_dict,
        rng_plan=rng_plan,
        chunk_index=HISTORY_CHUNK_INDEX,
    )
    current_timestep = route_eq._timestep(
        state.main_warped_timestep,
        state.current_state,
    )
    _, current_guard = _teacher_forward_chunk(
        runtime=runtime,
        conditional_dict=conditional_dict,
        chunk=state.current_state,
        timestep=current_timestep,
        start_frame=CURRENT_CHUNK_INDEX * FULL_SEQUENCE_CHUNK_FRAMES,
        label="teacher_matched_current_forward",
    )
    future_timestep = route_eq._timestep(
        state.teacher_future_timestep,
        state.future_state,
    )
    flow, future_guard = _teacher_forward_chunk(
        runtime=runtime,
        conditional_dict=conditional_dict,
        chunk=state.future_state,
        timestep=future_timestep,
        start_frame=FUTURE_START_FRAME,
        label="teacher_matched_future_forward",
    )
    x0 = manual_flow_to_x0(
        future_state=state.future_state,
        flow=flow,
        sigma=state.future_sigma,
        name="teacher_matched_x0",
    )
    proof = _teacher_branch_proof(
        branch=TEACHER_MATCHED_CURRENT_BRANCH,
        state=state,
        history=history,
        current_state=state.current_state,
        current_timestep=current_timestep,
        future_timestep=future_timestep,
        current_guard=current_guard,
        future_guard=future_guard,
        privileged_clean_current=False,
        same_information_as_mcp=True,
    )
    proof.update(
        {
            "inference_information_available": True,
            "deployment_proof": False,
            "teacher_frozen": _parameters_are_frozen(runtime.generator),
        }
    )
    if proof["teacher_frozen"] is not True:
        raise RuntimeError("matched-current Teacher is not frozen")
    return FlowPrediction(
        state_id=state.state_id,
        branch=TEACHER_MATCHED_CURRENT_BRANCH,
        flow=flow.detach().clone(),
        x0=x0.detach().clone(),
        proof=proof,
    )


def _run_student_predicted_current_state(
    *,
    runtime_factory: Any,
    state: TeacherFlowAuditState,
    teacher_target: torch.Tensor,
    conditional_dict: Mapping[str, Any],
    rng_plan: Mapping[str, Any],
    main_scheduler: Any,
) -> FlowPrediction:
    runtime = runtime_factory()
    was_training = bool(getattr(runtime.generator, "training", False))
    if hasattr(runtime.generator, "eval"):
        runtime.generator.eval()
    try:
        _validate_student_current_runtime(runtime)
        history = _recache_clean_chunk(
            runtime=runtime,
            teacher_target=teacher_target,
            conditional_dict=conditional_dict,
            rng_plan=rng_plan,
            chunk_index=HISTORY_CHUNK_INDEX,
        )
        current_timestep = route_eq._timestep(
            state.main_warped_timestep,
            state.current_state,
        )
        flow, current_guard = _teacher_forward_chunk(
            runtime=runtime,
            conditional_dict=conditional_dict,
            chunk=state.current_state,
            timestep=current_timestep,
            start_frame=CURRENT_CHUNK_INDEX * FULL_SEQUENCE_CHUNK_FRAMES,
            label="student_predicted_current_main_forward",
        )
        x0_hat = reconstruct_x0_from_flow_matching(
            main_scheduler,
            state=state.current_state,
            flow=flow,
            timestep=current_timestep,
            name="student_predicted_current_x0_hat",
        ).detach()
        if x0_hat.requires_grad:
            raise RuntimeError("predicted-current x0_hat must be detached")
        oracle = exact_current_flow_conversion_oracle(
            main_scheduler,
            state=state,
            teacher_target=teacher_target,
        )
        clean_current = _chunk(teacher_target, CURRENT_CHUNK_INDEX)
        history_rng_unchanged = bool(history["forward_rng"]["unchanged"])
        current_rng_unchanged = bool(current_guard["unchanged"])
        proof = {
            "branch": STUDENT_PREDICTED_CURRENT_BRANCH,
            "route": "student_main_single_chunk_kv_forward",
            "history_policy": "same clean chunk0 recache as Teacher audit",
            "history_chunk0_clean_sha256": tensor_sha256(
                _chunk(teacher_target, HISTORY_CHUNK_INDEX).detach().cpu()
            ),
            "history_context_latent_sha256": history["context_latent"]["sha256"],
            "history_context_noise": int(history["context_noise"]),
            "current_noisy_tensor_sha256": tensor_sha256(
                state.current_state.detach().cpu()
            ),
            "mcp_current_state_sha256": tensor_sha256(state.current_state.detach().cpu()),
            "main_timestep_sha256": tensor_sha256(current_timestep.detach().cpu()),
            "main_sigma": float(state.main_sigma),
            "main_forward_uses_clean_x": False,
            "main_forward_uses_mcp_future": False,
            "predicted_current_uses_gt_current": False,
            "predicted_current_uses_gt_future": False,
            "uses_teacher_prediction": False,
            "uses_privileged_current": False,
            "uses_wrapper_auto_x0": False,
            "x0_hat_source": (
                "scheduler.step(main_flow, main_timestep, current_state, "
                "to_final=True)"
            ),
            "x0_hat_detached": True,
            "student_frozen": _parameters_are_frozen(runtime.generator),
            "optimizer_step_executed": False,
            "history_forward_rng": dict(history["forward_rng"]),
            "current_forward_rng": dict(current_guard),
            "student_rng_unchanged": bool(
                history_rng_unchanged and current_rng_unchanged
            ),
            "current_x0_hat": _tensor_record(x0_hat),
            "current_x0_hat_mse_to_gt_current": route_eq._mse(
                x0_hat,
                clean_current,
            ),
            "exact_flow_conversion_oracle": oracle,
        }
        if proof["student_frozen"] is not True:
            raise RuntimeError("predicted-current Student is not frozen")
        if proof["student_rng_unchanged"] is not True:
            raise RuntimeError("predicted-current Student RNG changed")
        return FlowPrediction(
            state_id=state.state_id,
            branch=STUDENT_PREDICTED_CURRENT_BRANCH,
            flow=flow.detach().clone(),
            x0=x0_hat.detach().clone(),
            proof=proof,
        )
    finally:
        if hasattr(runtime.generator, "train"):
            runtime.generator.train(was_training)


def _run_teacher_clean_current_branch(
    *,
    runtime_factory: Any,
    state: TeacherFlowAuditState,
    teacher_target: torch.Tensor,
    conditional_dict: Mapping[str, Any],
    rng_plan: Mapping[str, Any],
) -> FlowPrediction:
    runtime = runtime_factory()
    _validate_teacher_runtime(runtime)
    history = _recache_clean_chunk(
        runtime=runtime,
        teacher_target=teacher_target,
        conditional_dict=conditional_dict,
        rng_plan=rng_plan,
        chunk_index=HISTORY_CHUNK_INDEX,
    )
    clean_current = _recache_clean_chunk(
        runtime=runtime,
        teacher_target=teacher_target,
        conditional_dict=conditional_dict,
        rng_plan=rng_plan,
        chunk_index=CURRENT_CHUNK_INDEX,
    )
    future_timestep = route_eq._timestep(
        state.teacher_future_timestep,
        state.future_state,
    )
    flow, future_guard = _teacher_forward_chunk(
        runtime=runtime,
        conditional_dict=conditional_dict,
        chunk=state.future_state,
        timestep=future_timestep,
        start_frame=FUTURE_START_FRAME,
        label="teacher_clean_current_future_forward",
    )
    x0 = manual_flow_to_x0(
        future_state=state.future_state,
        flow=flow,
        sigma=state.future_sigma,
        name="teacher_clean_x0",
    )
    proof = _teacher_branch_proof(
        branch=TEACHER_CLEAN_CURRENT_BRANCH,
        state=state,
        history=history,
        current_state=_chunk(teacher_target, CURRENT_CHUNK_INDEX),
        current_timestep=None,
        future_timestep=future_timestep,
        current_guard=clean_current["forward_rng"],
        future_guard=future_guard,
        privileged_clean_current=True,
        same_information_as_mcp=False,
    )
    proof["clean_current_recache"] = clean_current
    proof.update(
        {
            "branch": TEACHER_PRIVILEGED_CURRENT_BRANCH,
            "inference_information_available": False,
            "deployment_proof": False,
            "teacher_frozen": _parameters_are_frozen(runtime.generator),
        }
    )
    if proof["teacher_frozen"] is not True:
        raise RuntimeError("privileged-current Teacher is not frozen")
    return FlowPrediction(
        state_id=state.state_id,
        branch=TEACHER_PRIVILEGED_CURRENT_BRANCH,
        flow=flow.detach().clone(),
        x0=x0.detach().clone(),
        proof=proof,
    )


def _run_teacher_predicted_current_branch(
    *,
    runtime_factory: Any,
    state: TeacherFlowAuditState,
    teacher_target: torch.Tensor,
    conditional_dict: Mapping[str, Any],
    rng_plan: Mapping[str, Any],
    predicted_current: FlowPrediction,
) -> FlowPrediction:
    runtime = runtime_factory()
    _validate_teacher_runtime(runtime)
    history = _recache_clean_chunk(
        runtime=runtime,
        teacher_target=teacher_target,
        conditional_dict=conditional_dict,
        rng_plan=rng_plan,
        chunk_index=HISTORY_CHUNK_INDEX,
    )
    current_x0_hat = predicted_current.x0.detach()
    if current_x0_hat.requires_grad:
        raise RuntimeError("predicted-current Teacher recache got attached x0_hat")
    current_recache = _recache_supplied_chunk(
        runtime=runtime,
        conditional_dict=conditional_dict,
        rng_plan=rng_plan,
        chunk=current_x0_hat,
        chunk_index=CURRENT_CHUNK_INDEX,
        label="predicted_current_recache",
    )
    future_timestep = route_eq._timestep(
        state.teacher_future_timestep,
        state.future_state,
    )
    flow, future_guard = _teacher_forward_chunk(
        runtime=runtime,
        conditional_dict=conditional_dict,
        chunk=state.future_state,
        timestep=future_timestep,
        start_frame=FUTURE_START_FRAME,
        label="teacher_predicted_current_future_forward",
    )
    x0 = manual_flow_to_x0(
        future_state=state.future_state,
        flow=flow,
        sigma=state.future_sigma,
        name="teacher_predicted_current_x0",
    )
    proof = _teacher_branch_proof(
        branch=TEACHER_PREDICTED_CURRENT_BRANCH,
        state=state,
        history=history,
        current_state=current_x0_hat,
        current_timestep=None,
        future_timestep=future_timestep,
        current_guard=current_recache["forward_rng"],
        future_guard=future_guard,
        privileged_clean_current=False,
        same_information_as_mcp=False,
    )
    clean_current = _chunk(teacher_target, CURRENT_CHUNK_INDEX)
    proof.update(
        {
            "inference_information_available": True,
            "deployment_proof": False,
            "current_recache": current_recache,
            "current_x0_hat": _tensor_record(current_x0_hat),
            "current_x0_hat_mse_to_gt_current": route_eq._mse(
                current_x0_hat,
                clean_current,
            ),
            "current_x0_hat_detached": True,
            "student_current_prediction": _prediction_record(predicted_current),
            "predicted_current_uses_gt_current": False,
            "predicted_current_uses_gt_future": False,
            "same_future_tensor_as_other_branches": True,
            "teacher_frozen": _parameters_are_frozen(runtime.generator),
            "student_frozen": bool(
                predicted_current.proof.get("student_frozen") is True
            ),
            "optimizer_step_executed": False,
            "teacher_rng_unchanged": bool(
                proof["teacher_rng_unchanged"]
                and current_recache["forward_rng"]["unchanged"]
                and future_guard["unchanged"]
            ),
        }
    )
    if proof["teacher_frozen"] is not True or proof["student_frozen"] is not True:
        raise RuntimeError("predicted-current branch frozen proof failed")
    if proof["teacher_rng_unchanged"] is not True:
        raise RuntimeError("predicted-current Teacher RNG changed")
    return FlowPrediction(
        state_id=state.state_id,
        branch=TEACHER_PREDICTED_CURRENT_BRANCH,
        flow=flow.detach().clone(),
        x0=x0.detach().clone(),
        proof=proof,
    )


def _student_full_sequence_call(
    generator: Any,
    *,
    state: TeacherFlowAuditState,
    teacher_target: torch.Tensor,
    conditional_dict: Mapping[str, Any],
    anchors: Sequence[Mapping[str, Any]],
    direct_clean_context_kv: bool,
) -> Any:
    with torch.no_grad():
        return generator.forward_full_sequence_next_forcing(
            noisy_image_or_video=state.noisy_batch.noisy_main,
            clean_x=teacher_target,
            conditional_dict=dict(conditional_dict),
            timestep_main=state.noisy_batch.timestep_main,
            mcp_anchor_inputs=tuple(anchors),
            direct_clean_context_kv=bool(direct_clean_context_kv),
        )


def _recache_clean_chunk(
    *,
    runtime: deployment.DeploymentRuntime,
    teacher_target: torch.Tensor,
    conditional_dict: Mapping[str, Any],
    rng_plan: Mapping[str, Any],
    chunk_index: int,
) -> dict[str, Any]:
    chunk = _chunk(teacher_target, int(chunk_index))
    counts = {"clean_recache_forward_count": 0}
    return deployment._clean_recache(
        runtime=runtime,
        conditional_dict=conditional_dict,
        rng_plan=rng_plan,
        counts=counts,
        clean_chunk=chunk,
        chunk_index=int(chunk_index),
        start_frame=int(chunk_index) * FULL_SEQUENCE_CHUNK_FRAMES,
        expected_before=None,
    )


def _recache_supplied_chunk(
    *,
    runtime: deployment.DeploymentRuntime,
    conditional_dict: Mapping[str, Any],
    rng_plan: Mapping[str, Any],
    chunk: torch.Tensor,
    chunk_index: int,
    label: str,
) -> dict[str, Any]:
    if chunk.requires_grad:
        raise RuntimeError(f"{label} recache chunk must be detached")
    counts = {"clean_recache_forward_count": 0}
    record = deployment._clean_recache(
        runtime=runtime,
        conditional_dict=conditional_dict,
        rng_plan=rng_plan,
        counts=counts,
        clean_chunk=chunk,
        chunk_index=int(chunk_index),
        start_frame=int(chunk_index) * FULL_SEQUENCE_CHUNK_FRAMES,
        expected_before=None,
    )
    return {
        **record,
        "label": str(label),
        "recache_chunk": _tensor_record(chunk),
    }


def _teacher_forward_chunk(
    *,
    runtime: deployment.DeploymentRuntime,
    conditional_dict: Mapping[str, Any],
    chunk: torch.Tensor,
    timestep: torch.Tensor,
    start_frame: int,
    label: str,
) -> tuple[torch.Tensor, dict[str, Any]]:
    def call_teacher():
        return runtime.generator(
            noisy_image_or_video=chunk,
            conditional_dict=dict(conditional_dict),
            timestep=timestep,
            kv_cache=runtime.kv_cache,
            crossattn_cache=runtime.crossattn_cache,
            current_start=int(start_frame) * int(runtime.frame_seq_length),
        )

    outputs, rng_guard = deployment._call_with_rng_guard(
        device=chunk.device,
        label=label,
        fn=lambda: _no_grad_call(call_teacher),
    )
    flow, _ = deployment._unpack_main_outputs(outputs)
    if tuple(flow.shape) != tuple(chunk.shape):
        raise RuntimeError(f"{label} flow shape mismatch")
    _require_finite_tensor(flow, name=label)
    return flow, dict(rng_guard)


def _no_grad_call(fn):
    with torch.no_grad():
        return fn()


def _teacher_branch_proof(
    *,
    branch: str,
    state: TeacherFlowAuditState,
    history: Mapping[str, Any],
    current_state: torch.Tensor,
    current_timestep: torch.Tensor | None,
    future_timestep: torch.Tensor,
    current_guard: Mapping[str, Any],
    future_guard: Mapping[str, Any],
    privileged_clean_current: bool,
    same_information_as_mcp: bool,
) -> dict[str, Any]:
    current_record = _tensor_record(current_state)
    history_rng_unchanged = bool(history["forward_rng"]["unchanged"])
    current_rng_unchanged = bool(current_guard["unchanged"])
    future_rng_unchanged = bool(future_guard["unchanged"])
    proof = {
        "branch": branch,
        "teacher_checkpoint_role": "official_main_only_self_forcing",
        "route": "kv_cache_single_chunk_forward",
        "history_chunk0_clean_sha256": tensor_sha256(
            _chunk(state.noisy_batch.clean_target, HISTORY_CHUNK_INDEX).detach().cpu()
        ),
        "history_context_latent_sha256": history["context_latent"]["sha256"],
        "history_context_noise": int(history["context_noise"]),
        "current_state_sha256": current_record["sha256"],
        "mcp_current_state_sha256": tensor_sha256(state.current_state.detach().cpu()),
        "future_state_sha256": tensor_sha256(state.future_state.detach().cpu()),
        "mcp_future_state_sha256": tensor_sha256(state.future_state.detach().cpu()),
        "main_sigma": float(state.main_sigma),
        "future_sigma": float(state.future_sigma),
        "teacher_future_timestep": float(state.teacher_future_timestep),
        "expected_teacher_future_timestep": float(
            state.future_sigma * DEFAULT_NUM_TRAIN_TIMESTEPS
        ),
        "future_timestep_sha256": tensor_sha256(future_timestep.detach().cpu()),
        "privileged_clean_current": bool(privileged_clean_current),
        "same_information_as_mcp": bool(same_information_as_mcp),
        "uses_ground_truth_future_x0_for_conversion": False,
        "uses_wrapper_auto_x0": False,
        "history_forward_rng": dict(history["forward_rng"]),
        "current_forward_rng": dict(current_guard),
        "future_forward_rng": dict(future_guard),
        "teacher_rng_unchanged": bool(
            history_rng_unchanged and current_rng_unchanged and future_rng_unchanged
        ),
        "optimizer_step_executed": False,
    }
    proof["history_chunk0_identity_exact"] = (
        proof["history_chunk0_clean_sha256"]
        == tensor_sha256(
            _chunk(state.noisy_batch.clean_target, HISTORY_CHUNK_INDEX).detach().cpu()
        )
    )
    proof["history_context_latent_exact_clean"] = (
        proof["history_context_latent_sha256"] == proof["history_chunk0_clean_sha256"]
    )
    if not proof["history_chunk0_identity_exact"]:
        raise RuntimeError("Teacher history identity is not clean chunk0")
    if current_timestep is not None:
        proof["current_timestep_sha256"] = tensor_sha256(current_timestep.detach().cpu())
    proof["future_timestep_matches_physical_sigma"] = abs(
        proof["teacher_future_timestep"] - proof["expected_teacher_future_timestep"]
    ) <= 1.0e-5
    if branch == TEACHER_MATCHED_CURRENT_BRANCH:
        proof["matched_current_state_exact"] = (
            proof["current_state_sha256"] == proof["mcp_current_state_sha256"]
        )
        if not proof["matched_current_state_exact"]:
            raise RuntimeError("Teacher matched-current state does not match MCP")
    else:
        proof["matched_current_state_exact"] = False
    if not proof["future_timestep_matches_physical_sigma"]:
        raise RuntimeError("Teacher future timestep does not match physical sigma")
    if proof["teacher_rng_unchanged"] is not True:
        raise RuntimeError("Teacher branch RNG changed")
    return proof


def _state_metrics(
    *,
    state: TeacherFlowAuditState,
    student: FlowPrediction,
    student_current: FlowPrediction,
    teacher_matched: FlowPrediction,
    teacher_predicted: FlowPrediction,
    teacher_clean: FlowPrediction,
) -> dict[str, float]:
    clean_future = _chunk(state.noisy_batch.clean_target, FUTURE_CHUNK_INDEX)
    clean_current = _chunk(state.noisy_batch.clean_target, CURRENT_CHUNK_INDEX)
    return {
        "mcp_flow_vs_exact_mse": route_eq._mse(student.flow, state.exact_mcp_target),
        "mcp_x0_vs_clean_future_mse": route_eq._mse(student.x0, clean_future),
        "student_predicted_current_flow_vs_exact_mse": route_eq._mse(
            student_current.flow,
            _chunk(state.noisy_batch.target_flow_main, CURRENT_CHUNK_INDEX),
        ),
        "teacher_predicted_current_x0_hat_vs_clean_current_mse": route_eq._mse(
            student_current.x0,
            clean_current,
        ),
        "teacher_matched_flow_vs_exact_mse": route_eq._mse(
            teacher_matched.flow,
            state.exact_mcp_target,
        ),
        "teacher_matched_x0_vs_clean_future_mse": route_eq._mse(
            teacher_matched.x0,
            clean_future,
        ),
        "teacher_predicted_flow_vs_exact_mse": route_eq._mse(
            teacher_predicted.flow,
            state.exact_mcp_target,
        ),
        "teacher_predicted_x0_vs_clean_future_mse": route_eq._mse(
            teacher_predicted.x0,
            clean_future,
        ),
        "mcp_flow_vs_teacher_predicted_flow_mse": route_eq._mse(
            student.flow,
            teacher_predicted.flow,
        ),
        "mcp_flow_vs_teacher_matched_flow_mse": route_eq._mse(
            student.flow,
            teacher_matched.flow,
        ),
        "teacher_clean_flow_vs_exact_mse": route_eq._mse(
            teacher_clean.flow,
            state.exact_mcp_target,
        ),
        "teacher_clean_x0_vs_clean_future_mse": route_eq._mse(
            teacher_clean.x0,
            clean_future,
        ),
        "mcp_flow_vs_teacher_clean_flow_mse": route_eq._mse(
            student.flow,
            teacher_clean.flow,
        ),
    }


def _same_state_sigma_proof(
    *,
    state: TeacherFlowAuditState,
    student: FlowPrediction,
    student_current: FlowPrediction,
    teacher_matched: FlowPrediction,
    teacher_predicted: FlowPrediction,
    teacher_clean: FlowPrediction,
) -> dict[str, Any]:
    future_sha = tensor_sha256(state.future_state.detach().cpu())
    current_sha = tensor_sha256(state.current_state.detach().cpu())
    proof = {
        "student_future_state_exact": student.proof["mcp_future_input_sha256"]
        == future_sha,
        "teacher_matched_future_state_exact": teacher_matched.proof[
            "future_state_sha256"
        ]
        == future_sha,
        "teacher_clean_future_state_exact": teacher_clean.proof["future_state_sha256"]
        == future_sha,
        "teacher_predicted_future_state_exact": teacher_predicted.proof[
            "future_state_sha256"
        ]
        == future_sha,
        "teacher_matched_current_state_exact": teacher_matched.proof[
            "current_state_sha256"
        ]
        == current_sha,
        "student_predicted_current_input_exact": student_current.proof[
            "current_noisy_tensor_sha256"
        ]
        == current_sha,
        "main_sigma": float(state.main_sigma),
        "future_sigma": float(state.future_sigma),
        "teacher_future_timestep": float(state.teacher_future_timestep),
        "raw_timestep_directly_used_for_teacher": False,
        "predicted_current_uses_gt_current": bool(
            teacher_predicted.proof["predicted_current_uses_gt_current"]
        ),
        "predicted_current_uses_gt_future": bool(
            teacher_predicted.proof["predicted_current_uses_gt_future"]
        ),
        "same_future_tensor_as_other_branches": bool(
            teacher_predicted.proof["same_future_tensor_as_other_branches"]
        ),
        "teacher_frozen": bool(
            teacher_matched.proof.get("teacher_frozen") is True
            and teacher_predicted.proof.get("teacher_frozen") is True
            and teacher_clean.proof.get("teacher_frozen") is True
        ),
        "student_frozen": bool(
            student.proof.get("student_frozen", True) is True
            and student_current.proof.get("student_frozen") is True
        ),
        "optimizer_step_executed": False,
        "teacher_rng_unchanged": bool(
            teacher_matched.proof.get("teacher_rng_unchanged") is True
            and teacher_predicted.proof.get("teacher_rng_unchanged") is True
            and teacher_clean.proof.get("teacher_rng_unchanged") is True
        ),
    }
    proof["all_future_states_exact"] = (
        proof["student_future_state_exact"]
        and proof["teacher_matched_future_state_exact"]
        and proof["teacher_predicted_future_state_exact"]
        and proof["teacher_clean_future_state_exact"]
    )
    if not proof["all_future_states_exact"]:
        raise RuntimeError("future state exact proof failed")
    if not proof["teacher_matched_current_state_exact"]:
        raise RuntimeError("matched current state exact proof failed")
    if not proof["student_predicted_current_input_exact"]:
        raise RuntimeError("predicted-current Main input mismatch")
    if proof["predicted_current_uses_gt_current"] is not False:
        raise RuntimeError("predicted-current branch used GT current")
    if proof["predicted_current_uses_gt_future"] is not False:
        raise RuntimeError("predicted-current branch used GT future")
    if proof["teacher_frozen"] is not True or proof["student_frozen"] is not True:
        raise RuntimeError("predicted-current frozen proof failed")
    if proof["teacher_rng_unchanged"] is not True:
        raise RuntimeError("Teacher RNG unchanged proof failed")
    return proof


def _prediction_record(prediction: FlowPrediction) -> dict[str, Any]:
    return {
        "branch": prediction.branch,
        "flow": _tensor_record(prediction.flow),
        "x0": _tensor_record(prediction.x0),
        "proof": dict(prediction.proof),
    }


def _prediction_for(
    predictions: Mapping[str, FlowPrediction],
    state_id: str,
    branch: str,
) -> FlowPrediction:
    prediction = predictions.get(state_id)
    if prediction is None:
        raise RuntimeError(f"missing prediction for state {state_id}")
    if prediction.branch != branch:
        raise RuntimeError(f"prediction branch mismatch for state {state_id}")
    return prediction


def _teacher_prediction_for(
    predictions: Mapping[str, Mapping[str, FlowPrediction]],
    state_id: str,
    branch: str,
) -> FlowPrediction:
    by_branch = predictions.get(state_id)
    if not isinstance(by_branch, Mapping):
        raise RuntimeError(f"missing teacher predictions for state {state_id}")
    prediction = by_branch.get(branch)
    if prediction is None:
        raise RuntimeError(f"missing {branch} prediction for state {state_id}")
    if prediction.branch != branch:
        raise RuntimeError(f"teacher branch mismatch for state {state_id}")
    return prediction


def _aggregate_group(records: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, float]]:
    if not records:
        raise RuntimeError("cannot aggregate empty teacher-flow metric group")
    metric_names = tuple(records[0]["metrics"].keys())
    aggregate = {}
    for name in metric_names:
        values = [float(record["metrics"][name]) for record in records]
        aggregate[name] = {
            "mean": float(sum(values) / len(values)),
            "max": float(max(values)),
        }
    return aggregate


def _metric_mean(aggregate: Mapping[str, Any], key: str) -> float:
    metric = aggregate.get(key)
    if not isinstance(metric, Mapping):
        raise RuntimeError(f"bridge aggregate metric missing: {key}")
    return float(metric["mean"])


def _bridge_group_metrics(aggregate: Mapping[str, Any]) -> dict[str, Any]:
    matched_flow = _metric_mean(aggregate, "teacher_matched_flow_vs_exact_mse")
    predicted_flow = _metric_mean(aggregate, "teacher_predicted_flow_vs_exact_mse")
    privileged_flow = _metric_mean(aggregate, "teacher_clean_flow_vs_exact_mse")
    mcp_flow = _metric_mean(aggregate, "mcp_flow_vs_exact_mse")
    matched_x0 = _metric_mean(aggregate, "teacher_matched_x0_vs_clean_future_mse")
    predicted_x0 = _metric_mean(aggregate, "teacher_predicted_x0_vs_clean_future_mse")
    privileged_x0 = _metric_mean(aggregate, "teacher_clean_x0_vs_clean_future_mse")
    mcp_x0 = _metric_mean(aggregate, "mcp_x0_vs_clean_future_mse")
    flow_reduction = _relative_reduction(matched_flow, predicted_flow)
    x0_reduction = _relative_reduction(matched_x0, predicted_x0)
    gap_recovery = _gap_recovery_ratio(
        matched_flow=matched_flow,
        predicted_flow=predicted_flow,
        privileged_flow=privileged_flow,
    )
    return {
        "matched_flow_mse": matched_flow,
        "predicted_flow_mse": predicted_flow,
        "privileged_flow_mse": privileged_flow,
        "mcp_flow_mse": mcp_flow,
        "matched_x0_mse": matched_x0,
        "predicted_x0_mse": predicted_x0,
        "privileged_x0_mse": privileged_x0,
        "mcp_x0_mse": mcp_x0,
        "current_x0_hat_mse_to_gt_current": _metric_mean(
            aggregate,
            "teacher_predicted_current_x0_hat_vs_clean_current_mse",
        ),
        "predicted_vs_matched_flow_reduction": flow_reduction,
        "predicted_vs_matched_flow_reduction_pct": flow_reduction * 100.0,
        "predicted_vs_matched_x0_reduction": x0_reduction,
        "predicted_vs_matched_x0_reduction_pct": x0_reduction * 100.0,
        "gap_recovery_ratio": gap_recovery,
        "predicted_better_than_matched": predicted_flow < matched_flow,
        "predicted_clearly_worse_than_matched": (
            _passes_min(
                _relative_increase(matched_flow, predicted_flow),
                PREDICTED_CURRENT_CLEARLY_WORSE_MARGIN,
            )
        ),
    }


def _paired_values(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    aggregate = _aggregate_group(records)
    mcp_flow = float(aggregate["mcp_flow_vs_exact_mse"]["mean"])
    matched_flow = float(aggregate["teacher_matched_flow_vs_exact_mse"]["mean"])
    predicted_flow = float(aggregate["teacher_predicted_flow_vs_exact_mse"]["mean"])
    privileged_flow = float(aggregate["teacher_clean_flow_vs_exact_mse"]["mean"])
    mcp_x0 = float(aggregate["mcp_x0_vs_clean_future_mse"]["mean"])
    matched_x0 = float(aggregate["teacher_matched_x0_vs_clean_future_mse"]["mean"])
    predicted_x0 = float(aggregate["teacher_predicted_x0_vs_clean_future_mse"]["mean"])
    privileged_x0 = float(aggregate["teacher_clean_x0_vs_clean_future_mse"]["mean"])
    return {
        "mcp_flow_mse": mcp_flow,
        "matched_flow_mse": matched_flow,
        "predicted_flow_mse": predicted_flow,
        "privileged_flow_mse": privileged_flow,
        "mcp_x0_mse": mcp_x0,
        "matched_x0_mse": matched_x0,
        "predicted_x0_mse": predicted_x0,
        "privileged_x0_mse": privileged_x0,
        "privileged_flow_reduction": _relative_reduction(
            mcp_flow,
            privileged_flow,
        ),
        "matched_flow_reduction": _relative_reduction(mcp_flow, matched_flow),
        "predicted_vs_matched_flow_reduction": _relative_reduction(
            matched_flow,
            predicted_flow,
        ),
        "privileged_x0_reduction": _relative_reduction(mcp_x0, privileged_x0),
        "matched_x0_reduction": _relative_reduction(mcp_x0, matched_x0),
        "predicted_vs_matched_x0_reduction": _relative_reduction(
            matched_x0,
            predicted_x0,
        ),
        "gap_recovery_ratio": _gap_recovery_ratio(
            matched_flow=matched_flow,
            predicted_flow=predicted_flow,
            privileged_flow=privileged_flow,
        ),
        "privileged_flow_win": privileged_flow < mcp_flow,
        "matched_flow_win": matched_flow < mcp_flow,
        "predicted_better_than_matched": predicted_flow < matched_flow,
        "predicted_clearly_worse_than_matched": (
            _passes_min(
                _relative_increase(matched_flow, predicted_flow),
                PREDICTED_CURRENT_CLEARLY_WORSE_MARGIN,
            )
        ),
    }


def _paired_summary(values: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not values:
        raise RuntimeError("cannot summarize empty paired values")
    privileged_wins = sum(1 for value in values if value["privileged_flow_win"])
    matched_wins = sum(1 for value in values if value["matched_flow_win"])
    predicted_wins = sum(
        1 for value in values if value["predicted_better_than_matched"]
    )
    predicted_clearly_worse = sum(
        1 for value in values if value["predicted_clearly_worse_than_matched"]
    )
    return {
        "state_count": int(len(values)),
        "privileged_flow_win_count": int(privileged_wins),
        "privileged_flow_win_rate": _rate(privileged_wins, len(values)),
        "matched_flow_win_count": int(matched_wins),
        "matched_flow_win_rate": _rate(matched_wins, len(values)),
        "predicted_better_than_matched_win_count": int(predicted_wins),
        "predicted_better_than_matched_win_rate": _rate(
            predicted_wins,
            len(values),
        ),
        "predicted_clearly_worse_than_matched_count": int(
            predicted_clearly_worse
        ),
        "privileged_flow_reduction_mean": _mean(
            value["privileged_flow_reduction"] for value in values
        ),
        "privileged_flow_reduction_median": _median(
            value["privileged_flow_reduction"] for value in values
        ),
        "matched_flow_reduction_mean": _mean(
            value["matched_flow_reduction"] for value in values
        ),
        "matched_flow_reduction_median": _median(
            value["matched_flow_reduction"] for value in values
        ),
        "predicted_vs_matched_flow_reduction_mean": _mean(
            value["predicted_vs_matched_flow_reduction"] for value in values
        ),
        "predicted_vs_matched_flow_reduction_median": _median(
            value["predicted_vs_matched_flow_reduction"] for value in values
        ),
        "privileged_x0_reduction_mean": _mean(
            value["privileged_x0_reduction"] for value in values
        ),
        "privileged_x0_reduction_median": _median(
            value["privileged_x0_reduction"] for value in values
        ),
        "matched_x0_reduction_mean": _mean(
            value["matched_x0_reduction"] for value in values
        ),
        "matched_x0_reduction_median": _median(
            value["matched_x0_reduction"] for value in values
        ),
        "predicted_vs_matched_x0_reduction_mean": _mean(
            value["predicted_vs_matched_x0_reduction"] for value in values
        ),
        "predicted_vs_matched_x0_reduction_median": _median(
            value["predicted_vs_matched_x0_reduction"] for value in values
        ),
        "gap_recovery_ratio_mean": _mean_defined(
            value["gap_recovery_ratio"] for value in values
        ),
        "gap_recovery_ratio_median": _median_defined(
            value["gap_recovery_ratio"] for value in values
        ),
    }


def _relative_reduction(baseline: float, candidate: float) -> float:
    baseline_value = float(baseline)
    if baseline_value <= 0.0:
        return 0.0
    return float((baseline_value - float(candidate)) / baseline_value)


def _passes_min(
    value: float,
    threshold: float,
    *,
    atol: float = PREDICTED_CURRENT_THRESHOLD_ATOL,
) -> bool:
    return float(value) + float(atol) >= float(threshold)


def _passes_max(
    value: float,
    threshold: float,
    *,
    atol: float = PREDICTED_CURRENT_THRESHOLD_ATOL,
) -> bool:
    return float(value) <= float(threshold) + float(atol)


def _below_min(
    value: float,
    threshold: float,
    *,
    atol: float = PREDICTED_CURRENT_THRESHOLD_ATOL,
) -> bool:
    return float(value) < float(threshold) - float(atol)


def _relative_increase(baseline: float, candidate: float) -> float:
    baseline_value = float(baseline)
    candidate_value = float(candidate)
    if baseline_value <= 0.0:
        return float("inf") if candidate_value > baseline_value else 0.0
    return float((candidate_value - baseline_value) / baseline_value)


def _gap_recovery_ratio(
    *,
    matched_flow: float,
    predicted_flow: float,
    privileged_flow: float,
) -> float | None:
    denominator = float(matched_flow) - float(privileged_flow)
    if denominator <= 0.0:
        return None
    return float((float(matched_flow) - float(predicted_flow)) / denominator)


def _rate(count: int, total: int) -> float:
    if int(total) <= 0:
        raise RuntimeError("rate denominator must be positive")
    return float(int(count) / int(total))


def _mean(values: Any) -> float:
    materialized = [float(value) for value in values]
    if not materialized:
        raise RuntimeError("cannot average empty values")
    return float(sum(materialized) / len(materialized))


def _median(values: Any) -> float:
    materialized = [float(value) for value in values]
    if not materialized:
        raise RuntimeError("cannot compute median of empty values")
    return float(median(materialized))


def _mean_defined(values: Any) -> float | None:
    materialized = [float(value) for value in values if value is not None]
    if not materialized:
        return None
    return float(sum(materialized) / len(materialized))


def _median_defined(values: Any) -> float | None:
    materialized = [float(value) for value in values if value is not None]
    if not materialized:
        return None
    return float(median(materialized))


def _identity_group_key(record: Mapping[str, Any]) -> str:
    if "validation_position" in record:
        return f"validation_position_{int(record['validation_position']):03d}"
    return str(record.get("sample_identity", "single_identity"))


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _require_non_empty_records(records: Sequence[Mapping[str, Any]]) -> None:
    if not records:
        raise RuntimeError("teacher-flow metric records must be non-empty")


def _selected_validation_position_from_common(common_inputs: Mapping[str, Any]) -> int | None:
    value = common_inputs.get("selected_validation_position")
    if value is not None:
        return int(value)
    artifact = common_inputs.get("artifact_identity")
    if isinstance(artifact, Mapping) and artifact.get("selected_validation_position") is not None:
        return int(artifact["selected_validation_position"])
    return None


def _branch_strictly_dominates(
    state_records: Sequence[Mapping[str, Any]],
    *,
    flow_key: str,
    x0_key: str,
) -> bool:
    if not state_records:
        return False
    for record in state_records:
        metrics = record["metrics"]
        if not (
            float(metrics[flow_key]) < float(metrics["mcp_flow_vs_exact_mse"])
            and float(metrics[x0_key]) < float(metrics["mcp_x0_vs_clean_future_mse"])
        ):
            return False
    return True


def _branch_has_any_joint_improvement(
    state_records: Sequence[Mapping[str, Any]],
    *,
    flow_key: str,
    x0_key: str,
) -> bool:
    for record in state_records:
        metrics = record["metrics"]
        if (
            float(metrics[flow_key]) < float(metrics["mcp_flow_vs_exact_mse"])
            and float(metrics[x0_key]) < float(metrics["mcp_x0_vs_clean_future_mse"])
        ):
            return True
    return False


def _diagnostic_policy() -> dict[str, Any]:
    return {
        "label_policy": "strict_all_state_dominance_no_ad_hoc_numeric_margin",
        TEACHER_MATCHED_STRONGLY_BETTER: (
            "matched Teacher lower than MCP on flow-vs-exact and x0-vs-clean "
            "for every state"
        ),
        TEACHER_PRIVILEGED_ONLY_BETTER: (
            "clean-current Teacher strictly dominates MCP, matched Teacher does not"
        ),
        TEACHER_NOT_BETTER: "no state has joint flow and x0 Teacher improvement",
        INCONCLUSIVE: "some Teacher states improve, but strict dominance is absent",
    }


def _multi_identity_primary_policy() -> dict[str, Any]:
    return {
        "primary_hypothesis": "PRIVILEGED_CURRENT_GENERALIZES",
        STRONG_PRIVILEGED_CURRENT_SUPPORT: (
            "all-state privileged flow reduction >=25%, privileged identity "
            "flow win-rate >=75%, all-state privileged x0 reduction >=25%, "
            "and no raw aggregate privileged flow MSE is worse than MCP"
        ),
        NO_PRIVILEGED_CURRENT_SUPPORT: (
            "overall privileged flow reduction <10% or privileged identity "
            "flow win-rate <60%"
        ),
        INCONCLUSIVE: "pre-registered strong/no-support criteria are not met",
        "matched_teacher_is_not_primary_gate": True,
    }


def _predicted_current_bridge_policy() -> dict[str, Any]:
    return {
        "primary_hypothesis": "PREDICTED_CURRENT_BRIDGES_PRIVILEGED_CURRENT_GAP",
        "scope": "diagnostic current bridge only; not a deployment proof",
        "threshold_comparison_atol": PREDICTED_CURRENT_THRESHOLD_ATOL,
        "strong": {
            "label": STRONG_PREDICTED_CURRENT_BRIDGE_SUPPORT,
            "all_state_flow_reduction_min": 0.15,
            "identity_win_rate_min": 0.75,
            "identity_win_count_min": 24,
            "raw_aggregates_predicted_not_worse_than_matched": True,
            "gap_recovery_ratio_min": 0.30,
            "future_x0_reduction_min": 0.10,
        },
        "no_support": {
            "label": NO_SUPPORT,
            "all_state_flow_reduction_lt": 0.05,
            "identity_win_rate_lt": 0.60,
            "raw_clearly_worse_count_gte": 2,
            "clearly_worse_relative_margin": PREDICTED_CURRENT_CLEARLY_WORSE_MARGIN,
        },
        INCONCLUSIVE: "pre-registered strong/no-support criteria are not met",
    }


def _state_sigma_matching_contract(
    *,
    identity_rule: str = "validation_sample_identities[0]",
    noise_realizations_per_raw: int = TEACHER_FLOW_AUDIT_NOISE_REALIZATIONS_PER_RAW,
) -> dict[str, Any]:
    return {
        "identity": str(identity_rule),
        "history_chunk": HISTORY_CHUNK_INDEX,
        "current_chunk": CURRENT_CHUNK_INDEX,
        "future_chunk": FUTURE_CHUNK_INDEX,
        "raw_timesteps": list(TEACHER_FLOW_AUDIT_RAW_TIMESTEPS),
        "noise_realizations_per_raw": int(noise_realizations_per_raw),
        "current_state": "Main scheduler shift=5 add_noise(clean_chunk1, current_noise)",
        "future_state": "MCP scheduler shift=10 add_noise(clean_chunk2, future_noise)",
        "teacher_timestep": "same physical future sigma times 1000",
        "raw_timestep_not_used_for_teacher": True,
    }


def _teacher_route_contracts() -> dict[str, Any]:
    return {
        TEACHER_MATCHED_CURRENT_BRANCH: {
            "fresh_teacher_kv": True,
            "history": "clean chunk0 recache",
            "current": "same noisy current_state as MCP context",
            "future": "same noisy future_state as MCP target",
            "privileged_clean_current": False,
            "same_information_as_mcp": True,
            "inference_information_available": True,
        },
        TEACHER_PREDICTED_CURRENT_BRANCH: {
            "fresh_teacher_kv": True,
            "history": "clean chunk0 recache",
            "current": "student Main predicted current x0_hat recache",
            "future": "same noisy future_state as MCP target",
            "privileged_clean_current": False,
            "same_information_as_mcp": False,
            "inference_information_available": True,
            "deployment_proof": False,
        },
        TEACHER_PRIVILEGED_CURRENT_BRANCH: {
            "fresh_teacher_kv": True,
            "history": "clean chunk0 recache",
            "current": "clean chunk1 recache",
            "future": "same noisy future_state as MCP target",
            "privileged_clean_current": True,
            "same_information_as_mcp": False,
            "inference_information_available": False,
        },
    }


def _student_route_contract(
    student_predictions: Mapping[str, FlowPrediction]
) -> dict[str, Any]:
    values = {
        bool(prediction.proof.get("direct_clean_context_kv", False))
        for prediction in student_predictions.values()
    }
    if len(values) != 1:
        raise RuntimeError("student direct_clean_context_kv contract changed across states")
    return {
        "route": "forward_full_sequence_next_forcing",
        "direct_clean_context_kv": values.pop() if values else False,
        "uses_deployment_serial_rollout": False,
        "same_route_as_full_sequence_validation": True,
    }


def _student_predicted_current_route_contract(
    student_predictions: Mapping[str, FlowPrediction]
) -> dict[str, Any]:
    if not student_predictions:
        raise RuntimeError("student predicted-current predictions missing")
    for prediction in student_predictions.values():
        if prediction.branch != STUDENT_PREDICTED_CURRENT_BRANCH:
            raise RuntimeError("student predicted-current branch mismatch")
        proof = prediction.proof
        if proof.get("main_forward_uses_clean_x") is not False:
            raise RuntimeError("student predicted-current used clean_x")
        if proof.get("main_forward_uses_mcp_future") is not False:
            raise RuntimeError("student predicted-current used MCP future")
        if proof.get("x0_hat_detached") is not True:
            raise RuntimeError("student predicted-current x0_hat not detached")
    return {
        "route": "student_main_single_chunk_kv_forward",
        "history": "same clean chunk0 recache as Teacher audit",
        "current": "same noisy current_state as MCP and matched-current",
        "uses_clean_x": False,
        "uses_mcp_future": False,
        "uses_teacher_prediction": False,
        "uses_privileged_current": False,
        "uses_gt_current_to_construct_x0_hat": False,
        "uses_gt_future_to_construct_x0_hat": False,
        "x0_hat_detached": True,
        "inference_information_available": True,
    }


def _conversion_contract() -> dict[str, Any]:
    return {
        "teacher_x0": "future_state - sigma_future * teacher_flow",
        "mcp_x0": "future_state - sigma_future * mcp_flow",
        "predicted_current_x0_hat": (
            "FlowMatchScheduler.step(main_flow, main_timestep, current_state, "
            "to_final=True)"
        ),
        "current_flow_matching_derivation": (
            "add_noise gives x_t=(1-sigma)*x0+sigma*noise; training_target "
            "is noise-x0; final scheduler step gives x0=x_t-sigma*flow"
        ),
        "current_conversion_oracle_required": True,
        "current_conversion_oracle_atol": CURRENT_X0_ORACLE_ATOL,
        "current_conversion_oracle_rtol": CURRENT_X0_ORACLE_RTOL,
        "uses_wrapper_auto_x0": False,
        "uses_clean_future_to_derive_teacher_flow": False,
    }


def _forbidden_comparisons() -> list[str]:
    return [
        "raw timestep across scheduler shifts",
        "different actual future states",
        "ground-truth-derived oracle flow as Teacher flow",
        "clean-current Teacher reported as same-condition matched Teacher",
        "deployment serial rollout MCP flow mixed with full-sequence validation MCP flow",
    ]


def _scientific_boundaries() -> dict[str, str]:
    return {
        TEACHER_MATCHED_STRONGLY_BETTER: (
            "supports potential teacher-flow distillation target, but does not prove "
            "target variance as the cause"
        ),
        TEACHER_PRIVILEGED_ONLY_BETTER: (
            "clean-current information matters; distillation may exploit privileged "
            "information, but this does not show the same-condition objective is wrong"
        ),
        TEACHER_NOT_BETTER: (
            "does not support teacher-flow distillation as the next priority"
        ),
        INCONCLUSIVE: "mixed evidence; do not route science decisions from the label alone",
    }


def _multi_identity_boundaries() -> dict[str, str]:
    return {
        STRONG_PREDICTED_CURRENT_BRIDGE_SUPPORT: (
            "student Main predicted current information closes a preregistered "
            "fraction of the matched-to-privileged Teacher-flow gap in this "
            "diagnostic context setup"
        ),
        NO_SUPPORT: "does not support the predicted-current bridge as the next priority",
        STRONG_PRIVILEGED_CURRENT_SUPPORT: (
            "privileged near-clean current information produces a substantially "
            "better future conditional target across validation identities and is "
            "a promising training-time distillation signal"
        ),
        NO_PRIVILEGED_CURRENT_SUPPORT: (
            "does not support privileged-current Teacher flow as the next priority"
        ),
        INCONCLUSIVE: (
            "mixed evidence; do not claim conditional ambiguity or objective failure"
        ),
        "forbidden_claims": (
            "do not claim conditional ambiguity is proven, exact FM is wrong, "
            "the privileged Teacher is oracle, predicted-current is a deployment "
            "proof, or Teacher flow should directly replace FM target"
        ),
    }


def _multi_identity_streaming_contract() -> dict[str, Any]:
    return {
        "identity_order": "one selected validation identity at a time",
        "states_per_identity": len(TEACHER_FLOW_AUDIT_RAW_TIMESTEPS),
        "retains_full_state_tensors": False,
        "retains_full_flow_tensors": False,
        "final_json_contains": "provenance, tensor SHA summaries, metrics, aggregates",
    }


def _validate_state_record(record: Any) -> None:
    if not isinstance(record, Mapping):
        raise RuntimeError("teacher-flow state record must be a mapping")
    metrics = record.get("metrics")
    if not isinstance(metrics, Mapping):
        raise RuntimeError("teacher-flow state metrics missing")
    required_metrics = {
        "mcp_flow_vs_exact_mse",
        "mcp_x0_vs_clean_future_mse",
        "student_predicted_current_flow_vs_exact_mse",
        "teacher_predicted_current_x0_hat_vs_clean_current_mse",
        "teacher_matched_flow_vs_exact_mse",
        "teacher_matched_x0_vs_clean_future_mse",
        "teacher_predicted_flow_vs_exact_mse",
        "teacher_predicted_x0_vs_clean_future_mse",
        "mcp_flow_vs_teacher_predicted_flow_mse",
        "mcp_flow_vs_teacher_matched_flow_mse",
        "teacher_clean_flow_vs_exact_mse",
        "teacher_clean_x0_vs_clean_future_mse",
        "mcp_flow_vs_teacher_clean_flow_mse",
    }
    missing = required_metrics - set(metrics.keys())
    if missing:
        raise RuntimeError(f"teacher-flow state metrics missing: {sorted(missing)}")
    for key in required_metrics:
        value = float(metrics[key])
        if not torch.isfinite(torch.tensor(value)):
            raise RuntimeError(f"teacher-flow metric is nonfinite: {key}")
    matched = record.get("teacher_matched_current")
    predicted = record.get("teacher_predicted_current")
    privileged = record.get("teacher_privileged_current")
    if not isinstance(matched, Mapping) or not isinstance(predicted, Mapping):
        raise RuntimeError("teacher-flow branch records missing")
    if not isinstance(privileged, Mapping):
        raise RuntimeError("teacher-flow branch records missing")
    matched_proof = matched.get("proof")
    predicted_proof = predicted.get("proof")
    privileged_proof = privileged.get("proof")
    if not isinstance(matched_proof, Mapping) or not isinstance(
        predicted_proof,
        Mapping,
    ) or not isinstance(privileged_proof, Mapping):
        raise RuntimeError("teacher-flow branch proof missing")
    if matched_proof.get("privileged_clean_current") is not False:
        raise RuntimeError("matched Teacher branch is marked privileged")
    if matched_proof.get("same_information_as_mcp") is not True:
        raise RuntimeError("matched Teacher branch information flag mismatch")
    if predicted_proof.get("inference_information_available") is not True:
        raise RuntimeError("predicted-current availability flag mismatch")
    if predicted_proof.get("predicted_current_uses_gt_current") is not False:
        raise RuntimeError("predicted-current used GT current")
    if predicted_proof.get("predicted_current_uses_gt_future") is not False:
        raise RuntimeError("predicted-current used GT future")
    if predicted_proof.get("current_x0_hat_detached") is not True:
        raise RuntimeError("predicted-current x0_hat detach proof missing")
    if predicted_proof.get("teacher_frozen") is not True:
        raise RuntimeError("predicted-current Teacher frozen proof missing")
    if predicted_proof.get("student_frozen") is not True:
        raise RuntimeError("predicted-current Student frozen proof missing")
    if predicted_proof.get("optimizer_step_executed") is not False:
        raise RuntimeError("predicted-current optimizer step proof mismatch")
    if privileged_proof.get("privileged_clean_current") is not True:
        raise RuntimeError("clean-current Teacher branch privilege flag missing")
    if privileged_proof.get("same_information_as_mcp") is not False:
        raise RuntimeError("clean-current Teacher branch information flag mismatch")
    if privileged_proof.get("inference_information_available") is not False:
        raise RuntimeError("privileged-current availability flag mismatch")
    proof = record.get("same_state_sigma_proof")
    if not isinstance(proof, Mapping) or proof.get("all_future_states_exact") is not True:
        raise RuntimeError("teacher-flow same future state proof missing")
    if proof.get("raw_timestep_directly_used_for_teacher") is not False:
        raise RuntimeError("teacher-flow Teacher used raw timestep directly")
    if proof.get("same_future_tensor_as_other_branches") is not True:
        raise RuntimeError("predicted-current future tensor proof missing")


def _validate_teacher_runtime(runtime: deployment.DeploymentRuntime) -> None:
    deployment._validate_runtime(runtime)
    if getattr(runtime.generator, "mcp", None) is not None:
        raise RuntimeError("Teacher runtime must be Main-only")
    if int(runtime.context_noise) != 0:
        raise RuntimeError("Teacher audit requires clean-history context_noise=0")


def _validate_student_current_runtime(runtime: deployment.DeploymentRuntime) -> None:
    deployment._validate_runtime(runtime)
    if int(runtime.context_noise) != 0:
        raise RuntimeError("predicted-current Student requires context_noise=0")
    if getattr(runtime.generator, "training", False):
        raise RuntimeError("predicted-current Student must be eval")
    if not _parameters_are_frozen(runtime.generator):
        raise RuntimeError("predicted-current Student parameters must be frozen")


def _parameters_are_frozen(generator: Any) -> bool:
    return all(not parameter.requires_grad for parameter in generator.parameters())


def _validate_identity_selection(selection: Any) -> None:
    if not isinstance(selection, Mapping):
        raise RuntimeError("multi-identity selection must be a mapping")
    positions = selection.get("positions")
    identities = selection.get("identity_strings")
    if not isinstance(positions, Sequence) or isinstance(positions, (str, bytes)):
        raise RuntimeError("multi-identity selection positions missing")
    if not isinstance(identities, Sequence) or isinstance(identities, (str, bytes)):
        raise RuntimeError("multi-identity selection identities missing")
    expected_positions = list(
        range(
            0,
            TEACHER_FLOW_AUDIT_VALIDATION_COUNT,
            TEACHER_FLOW_AUDIT_MULTI_VALIDATION_STRIDE,
        )
    )
    if [int(value) for value in positions] != expected_positions:
        raise RuntimeError("multi-identity selection positions mismatch")
    if int(selection.get("stride", -1)) != TEACHER_FLOW_AUDIT_MULTI_VALIDATION_STRIDE:
        raise RuntimeError("multi-identity selection stride mismatch")
    expected_rule = "validation positions 0,8,16,...,248 from exact 256 list"
    if str(selection.get("selection_rule")) != expected_rule:
        raise RuntimeError("multi-identity selection rule mismatch")
    if len(identities) != TEACHER_FLOW_AUDIT_MULTI_VALIDATION_COUNT:
        raise RuntimeError("multi-identity selection identity count mismatch")
    if len(set(str(value) for value in identities)) != len(identities):
        raise RuntimeError("multi-identity selection contains duplicate identities")
    if int(selection.get("validation_identity_count", -1)) != (
        TEACHER_FLOW_AUDIT_VALIDATION_COUNT
    ):
        raise RuntimeError("multi-identity validation count mismatch")
    if int(selection.get("selected_identity_count", -1)) != (
        TEACHER_FLOW_AUDIT_MULTI_VALIDATION_COUNT
    ):
        raise RuntimeError("multi-identity selected count mismatch")
    if selection.get("mode") != TEACHER_FLOW_AUDIT_MODE_MULTI_VALIDATION32:
        raise RuntimeError("multi-identity selection mode mismatch")
    if not _is_sha256(selection.get("identity_list_sha256")):
        raise RuntimeError("multi-identity identity-list SHA missing")
    if not _is_sha256(selection.get("selection_fingerprint_sha256")):
        raise RuntimeError("multi-identity selection fingerprint missing")
    expected_payload = {
        "mode": selection["mode"],
        "validation_identity_count": int(selection["validation_identity_count"]),
        "selected_identity_count": int(selection["selected_identity_count"]),
        "stride": int(selection.get("stride", -1)),
        "positions": [int(value) for value in positions],
        "identity_strings": [str(value) for value in identities],
        "identity_list_sha256": str(selection["identity_list_sha256"]),
        "selection_rule": expected_rule,
    }
    if deployment.canonical_json_sha256(expected_payload) != str(
        selection["selection_fingerprint_sha256"]
    ):
        raise RuntimeError("multi-identity selection fingerprint mismatch")


def _require_teacher_flow_state_contract(
    states: Sequence[TeacherFlowAuditState],
    *,
    expected_noise_realizations_per_raw: int | None = None,
) -> None:
    if not states:
        raise RuntimeError("teacher-flow audit state list is empty")
    seen = set()
    by_raw = {raw: 0 for raw in TEACHER_FLOW_AUDIT_RAW_TIMESTEPS}
    for state in states:
        if state.state_id in seen:
            raise RuntimeError("duplicate teacher-flow audit state id")
        seen.add(state.state_id)
        raw = int(state.raw_timestep)
        if raw not in TEACHER_FLOW_AUDIT_RAW_TIMESTEPS:
            raise RuntimeError("teacher-flow audit unexpected raw timestep")
        by_raw[raw] += 1
    counts = set(by_raw.values())
    if len(counts) != 1:
        raise RuntimeError("teacher-flow audit raw state counts differ")
    noise_count = counts.pop()
    if expected_noise_realizations_per_raw is not None:
        if noise_count != int(expected_noise_realizations_per_raw):
            raise RuntimeError("teacher-flow audit noise realization count mismatch")
    elif noise_count not in (
        TEACHER_FLOW_AUDIT_NOISE_REALIZATIONS_PER_RAW,
        TEACHER_FLOW_AUDIT_MULTI_NOISE_REALIZATIONS_PER_RAW,
    ):
        raise RuntimeError("teacher-flow audit unsupported noise realization count")
    expected_states = len(TEACHER_FLOW_AUDIT_RAW_TIMESTEPS) * int(noise_count)
    if len(states) != expected_states:
        raise RuntimeError("teacher-flow audit state count mismatch")


def _require_multi_identity_state_records(
    records: Sequence[Mapping[str, Any]],
) -> None:
    if len(records) != (
        TEACHER_FLOW_AUDIT_MULTI_VALIDATION_COUNT
        * len(TEACHER_FLOW_AUDIT_RAW_TIMESTEPS)
    ):
        raise RuntimeError("multi-identity Teacher-flow state count mismatch")
    grouped: dict[int, list[Mapping[str, Any]]] = {}
    for record in records:
        if int(record.get("noise_index", -1)) != 0:
            raise RuntimeError("multi-identity Teacher-flow requires noise_index 0")
        position = int(record.get("validation_position", -1))
        grouped.setdefault(position, []).append(record)
    expected_positions = set(
        range(
            0,
            TEACHER_FLOW_AUDIT_VALIDATION_COUNT,
            TEACHER_FLOW_AUDIT_MULTI_VALIDATION_STRIDE,
        )
    )
    if set(grouped.keys()) != expected_positions:
        raise RuntimeError("multi-identity Teacher-flow validation positions mismatch")
    for position, position_records in grouped.items():
        raws = sorted(int(record["raw_timestep"]) for record in position_records)
        if raws != sorted(TEACHER_FLOW_AUDIT_RAW_TIMESTEPS):
            raise RuntimeError(
                f"multi-identity Teacher-flow raw grid mismatch at {position}"
            )


def _validate_multi_identity_records(
    records: Any,
    selection: Any,
) -> None:
    _validate_identity_selection(selection)
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        raise RuntimeError("multi-identity records must be a sequence")
    if len(records) != TEACHER_FLOW_AUDIT_MULTI_VALIDATION_COUNT:
        raise RuntimeError("multi-identity records count mismatch")
    by_position = {
        int(record["validation_position"]): record
        for record in records
    }
    for index, position in enumerate(selection["positions"]):
        position_int = int(position)
        record = by_position.get(position_int)
        if record is None:
            raise RuntimeError("multi-identity record validation position missing")
        if str(record.get("sample_identity")) != str(
            selection["identity_strings"][index]
        ):
            raise RuntimeError("multi-identity record identity mismatch")
        if int(record.get("identity_index", -1)) != int(index):
            raise RuntimeError("multi-identity record identity_index mismatch")
        if int(record.get("state_count", -1)) != len(TEACHER_FLOW_AUDIT_RAW_TIMESTEPS):
            raise RuntimeError("multi-identity record state_count mismatch")
        if not _is_sha256(record.get("common_inputs_fingerprint_sha256")):
            raise RuntimeError("multi-identity common input fingerprint missing")


def _validate_common_fingerprints(
    fingerprints: Any,
    selection: Any,
) -> None:
    _validate_identity_selection(selection)
    if not isinstance(fingerprints, Mapping):
        raise RuntimeError("multi-identity common fingerprints must be a mapping")
    expected_keys = {str(position) for position in selection["positions"]}
    if set(str(key) for key in fingerprints.keys()) != expected_keys:
        raise RuntimeError("multi-identity common fingerprint positions mismatch")
    for value in fingerprints.values():
        if not _is_sha256(value):
            raise RuntimeError("multi-identity common fingerprint invalid")


def _require_16_state_contract(states: Sequence[TeacherFlowAuditState]) -> None:
    _require_teacher_flow_state_contract(
        states,
        expected_noise_realizations_per_raw=TEACHER_FLOW_AUDIT_NOISE_REALIZATIONS_PER_RAW,
    )


def _selected_anchor1_depth1(anchors: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    for anchor in anchors:
        if int(anchor["anchor_index"]) == CURRENT_CHUNK_INDEX:
            depths = tuple(int(value) for value in anchor["depths"])
            if depths[:3] != FULL_SEQUENCE_DEPTHS:
                raise RuntimeError("anchor1 must expose MCP depths 1/2/3")
            starts = tuple(int(value) for value in anchor["future_start_frames"])
            if starts[:3] != (
                FUTURE_START_FRAME,
                FUTURE_START_FRAME + FULL_SEQUENCE_CHUNK_FRAMES,
                FUTURE_START_FRAME + 2 * FULL_SEQUENCE_CHUNK_FRAMES,
            ):
                raise RuntimeError("anchor1 future start frames mismatch")
            return anchor
    raise RuntimeError("teacher-flow audit missing anchor1 MCP inputs")


def _chunk(tensor: torch.Tensor, chunk_index: int) -> torch.Tensor:
    return flow_audit._chunk(tensor, int(chunk_index))


def _set_chunk(tensor: torch.Tensor, chunk_index: int, value: torch.Tensor) -> None:
    start = int(chunk_index) * FULL_SEQUENCE_CHUNK_FRAMES
    stop = start + FULL_SEQUENCE_CHUNK_FRAMES
    tensor[:, start:stop] = value


def _tensor_record(tensor: torch.Tensor) -> dict[str, Any]:
    _require_finite_tensor(tensor, name="recorded_tensor")
    summary = tensor_summary(tensor.detach().cpu())
    value = tensor.detach().float()
    return {
        "shape": summary["shape"],
        "dtype": summary["dtype"],
        "finite": summary["finite"],
        "sha256": summary["sha256"],
        "mean_abs": float(value.abs().mean().item()),
        "max_abs": float(value.abs().max().item()),
    }


def _require_finite_tensor(tensor: torch.Tensor, *, name: str) -> None:
    deployment._ensure_finite_tensor(tensor, name=name)


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    text = value.lower()
    return len(text) == 64 and all(char in "0123456789abcdef" for char in text)


def _payload_string(payload: Mapping[str, Any], key: str) -> str:
    value = _payload_nested_value(payload, key)
    if not _is_sha256(value):
        raise RuntimeError(f"student checkpoint {key} missing or invalid")
    return str(value)


def _payload_string_source(payload: Mapping[str, Any], key: str) -> str:
    for source, value in _payload_nested_candidates(payload, key):
        if _is_sha256(value):
            return source
    raise RuntimeError(f"student checkpoint {key} source missing")


def _payload_nested_value(payload: Mapping[str, Any], key: str) -> Any:
    for _, value in _payload_nested_candidates(payload, key):
        if _is_sha256(value):
            return value
    return None


def _payload_nested_candidates(payload: Mapping[str, Any], key: str):
    yield key, payload.get(key)
    resolved = payload.get("resolved_config")
    if isinstance(resolved, Mapping):
        yield f"resolved_config.{key}", resolved.get(key)
    metadata = payload.get("metadata")
    if isinstance(metadata, Mapping):
        yield f"metadata.{key}", metadata.get(key)
        provenance = metadata.get("provenance")
        if isinstance(provenance, Mapping):
            yield f"metadata.provenance.{key}", provenance.get(key)


__all__ = [
    "CURRENT_CHUNK_INDEX",
    "FUTURE_CHUNK_INDEX",
    "HISTORY_CHUNK_INDEX",
    "BF16_QUANTIZED_STATE_CONTRACT",
    "EXACT_PASS",
    "INCONCLUSIVE",
    "MATCHED_TEACHER_TIMESTEP_DEPENDENCE",
    "NO_SUPPORT",
    "NO_PRIVILEGED_CURRENT_SUPPORT",
    "PREDICTED_CURRENT_CLEARLY_WORSE_MARGIN",
    "PREDICTED_CURRENT_ORACLE_RECHECK_CLASSIFICATIONS",
    "PREDICTED_CURRENT_ORACLE_RECHECK_MODE",
    "PREDICTED_CURRENT_ORACLE_RECHECK_NOISE_INDEX",
    "PREDICTED_CURRENT_ORACLE_RECHECK_RAW_TIMESTEP",
    "PREDICTED_CURRENT_ORACLE_RECHECK_SCHEMA",
    "PREDICTED_CURRENT_THRESHOLD_ATOL",
    "SCHEDULER_MISMATCH",
    "SEMANTIC_MISMATCH",
    "STATE_PROVENANCE_MISMATCH",
    "STUDENT_MCP_BRANCH",
    "STUDENT_PREDICTED_CURRENT_BRANCH",
    "STRONG_PREDICTED_CURRENT_BRIDGE_SUPPORT",
    "STRONG_PRIVILEGED_CURRENT_SUPPORT",
    "TEACHER_CLEAN_CURRENT_BRANCH",
    "TEACHER_FLOW_AUDIT_NOISE_REALIZATIONS_PER_RAW",
    "TEACHER_FLOW_AUDIT_MODE_MULTI_VALIDATION32",
    "TEACHER_FLOW_AUDIT_MODE_SINGLE",
    "TEACHER_FLOW_AUDIT_MULTI_NOISE_REALIZATIONS_PER_RAW",
    "TEACHER_FLOW_AUDIT_RAW_TIMESTEPS",
    "TEACHER_FLOW_AUDIT_SCHEMA",
    "TEACHER_MATCHED_CURRENT_BRANCH",
    "TEACHER_MATCHED_STRONGLY_BETTER",
    "TEACHER_NOT_BETTER",
    "TEACHER_PREDICTED_CURRENT_BRANCH",
    "TEACHER_PRIVILEGED_ONLY_BETTER",
    "TEACHER_PRIVILEGED_CURRENT_BRANCH",
    "FlowPrediction",
    "TeacherFlowAuditResult",
    "TeacherFlowAuditState",
    "aggregate_teacher_flow_metrics",
    "build_predicted_current_oracle_recheck_artifact",
    "build_predicted_current_oracle_recheck_state_record",
    "build_predicted_current_oracle_recheck_state",
    "build_predicted_current_oracle_recheck_validation0_all_raw_artifact",
    "build_predicted_current_oracle_recheck_validation0_all_raw_states",
    "build_flow_match_scheduler",
    "build_teacher_flow_multi_identity_manifest",
    "build_teacher_flow_audit_result",
    "build_teacher_flow_audit_states",
    "classify_predicted_current_oracle_recheck",
    "build_teacher_flow_state_records",
    "diagnostic_label_from_metrics",
    "exact_current_flow_conversion_oracle",
    "load_teacher_flow_student_checkpoint_record",
    "matched_teacher_timestep_diagnostic",
    "manual_flow_to_x0",
    "parameter_sha256_report",
    "paired_teacher_flow_statistics",
    "predicted_current_bridge_label",
    "predicted_current_bridge_statistics",
    "privileged_current_generalization_label",
    "reconstruct_x0_from_flow_matching",
    "require_no_parameter_mutation",
    "run_student_mcp_full_sequence_predictions",
    "run_student_predicted_current_predictions",
    "run_teacher_branch_predictions",
    "run_teacher_flow_audit",
    "select_validation32_identities",
    "select_validation_zero_identity",
    "validate_frozen_student_model",
    "validate_frozen_teacher_model",
    "validate_multi_identity_student_checkpoint_contract",
    "validate_teacher_flow_artifact_identity_selection",
    "validate_teacher_flow_artifact_identity",
    "validate_teacher_flow_audit_manifest",
]
