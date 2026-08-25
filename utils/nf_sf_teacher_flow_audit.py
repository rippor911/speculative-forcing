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
TEACHER_MATCHED_CURRENT_BRANCH = "teacher_matched_current"
TEACHER_CLEAN_CURRENT_BRANCH = "teacher_clean_current"
STUDENT_MCP_BRANCH = "student_mcp1_full_sequence"

HISTORY_CHUNK_INDEX = 0
CURRENT_CHUNK_INDEX = 1
FUTURE_CHUNK_INDEX = 2
FUTURE_START_FRAME = FUTURE_CHUNK_INDEX * FULL_SEQUENCE_CHUNK_FRAMES

TEACHER_MATCHED_STRONGLY_BETTER = "TEACHER_MATCHED_STRONGLY_BETTER"
TEACHER_PRIVILEGED_ONLY_BETTER = "TEACHER_PRIVILEGED_ONLY_BETTER"
TEACHER_NOT_BETTER = "TEACHER_NOT_BETTER"
STRONG_PRIVILEGED_CURRENT_SUPPORT = "STRONG_PRIVILEGED_CURRENT_SUPPORT"
NO_PRIVILEGED_CURRENT_SUPPORT = "NO_PRIVILEGED_CURRENT_SUPPORT"
MATCHED_TEACHER_TIMESTEP_DEPENDENCE = "MATCHED_TEACHER_TIMESTEP_DEPENDENCE"
INCONCLUSIVE = "INCONCLUSIVE"


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

    clean_current = _chunk(teacher_target, CURRENT_CHUNK_INDEX)
    clean_future = _chunk(teacher_target, FUTURE_CHUNK_INDEX)
    source_current = _chunk(source_noise, CURRENT_CHUNK_INDEX)
    source_future = _chunk(source_noise, FUTURE_CHUNK_INDEX)

    states: list[TeacherFlowAuditState] = []
    for raw_timestep in raw_timesteps:
        raw = int(raw_timestep)
        main_t = memorization._warp_raw_timestep(raw, shift=DEFAULT_S_MAIN)
        future_t = memorization._warp_raw_timestep(raw, shift=DEFAULT_S_MCP)
        for noise_index in range(noise_count):
            current_noise, current_noise_record = memorization._noise_for_realization(
                template=source_current,
                source_noise=source_noise,
                teacher_target=teacher_target,
                raw_timestep=raw,
                noise_index=noise_index,
                role="current_chunk1",
                base_seed=int(noise_seed),
            )
            future_noise, future_noise_record = memorization._noise_for_realization(
                template=source_future,
                source_noise=source_noise,
                teacher_target=teacher_target,
                raw_timestep=raw,
                noise_index=noise_index,
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
            base_state_id = f"raw{raw:03d}_noise{int(noise_index)}"
            if state_id_prefix:
                state_id = f"{state_id_prefix}_{base_state_id}"
            else:
                state_id = base_state_id
            provenance = {
                "state_id": state_id,
                "base_state_id": base_state_id,
                "raw_timestep": raw,
                "noise_index": int(noise_index),
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
            states.append(
                TeacherFlowAuditState(
                    state_id=state_id,
                    raw_timestep=raw,
                    noise_index=int(noise_index),
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
            )
    _require_teacher_flow_state_contract(
        states,
        expected_noise_realizations_per_raw=noise_count,
    )
    return tuple(states)


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


def run_teacher_branch_predictions(
    *,
    runtime_factory: Any,
    states: Sequence[TeacherFlowAuditState],
    source_noise: torch.Tensor,
    teacher_target: torch.Tensor,
    teacher_payload: Mapping[str, Any],
    conditional_dict: Mapping[str, Any],
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
        predictions[state.state_id] = {
            TEACHER_MATCHED_CURRENT_BRANCH: _run_teacher_matched_current_branch(
                runtime_factory=runtime_factory,
                state=state,
                teacher_target=teacher_target,
                conditional_dict=conditional_dict,
                rng_plan=rng_plan,
            ),
            TEACHER_CLEAN_CURRENT_BRANCH: _run_teacher_clean_current_branch(
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
    teacher_predictions: Mapping[str, Mapping[str, FlowPrediction]],
    sample_identity: str,
    checkpoint_summary: Mapping[str, Any],
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
        matched = _teacher_prediction_for(
            teacher_predictions,
            state.state_id,
            TEACHER_MATCHED_CURRENT_BRANCH,
        )
        clean = _teacher_prediction_for(
            teacher_predictions,
            state.state_id,
            TEACHER_CLEAN_CURRENT_BRANCH,
        )
        tensors["states"][state.state_id] = {
            "current_state": state.current_state.detach().cpu(),
            "future_state": state.future_state.detach().cpu(),
            "exact_mcp_target": state.exact_mcp_target.detach().cpu(),
            "student_mcp_flow": student.flow.detach().cpu(),
            "student_mcp_x0": student.x0.detach().cpu(),
            "teacher_matched_flow": matched.flow.detach().cpu(),
            "teacher_matched_x0": matched.x0.detach().cpu(),
            "teacher_clean_flow": clean.flow.detach().cpu(),
            "teacher_clean_x0": clean.x0.detach().cpu(),
        }
    aggregates = aggregate_teacher_flow_metrics(state_records)
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
        "state_sigma_matching_contract": _state_sigma_matching_contract(),
        "teacher_routes": _teacher_route_contracts(),
        "student_route": _student_route_contract(student_predictions),
        "conversion_contract": _conversion_contract(),
        "forbidden_comparisons": _forbidden_comparisons(),
        "scientific_interpretation_boundaries": _scientific_boundaries(),
        "checkpoint": dict(checkpoint_summary),
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
    teacher_predictions: Mapping[str, Mapping[str, FlowPrediction]],
    sample_identity: str,
    validation_position: int | None = None,
    identity_index: int | None = None,
) -> list[dict[str, Any]]:
    _require_teacher_flow_state_contract(states)
    state_records = []
    for state in states:
        student = _prediction_for(student_predictions, state.state_id, STUDENT_MCP_BRANCH)
        matched = _teacher_prediction_for(
            teacher_predictions,
            state.state_id,
            TEACHER_MATCHED_CURRENT_BRANCH,
        )
        clean = _teacher_prediction_for(
            teacher_predictions,
            state.state_id,
            TEACHER_CLEAN_CURRENT_BRANCH,
        )
        record = {
            **state.provenance,
            "sample_identity": str(sample_identity),
            "student": _prediction_record(student),
            "teacher_matched_current": _prediction_record(matched),
            "teacher_clean_current": _prediction_record(clean),
            "metrics": _state_metrics(
                state=state,
                student=student,
                teacher_matched=matched,
                teacher_clean=clean,
            ),
            "same_state_sigma_proof": _same_state_sigma_proof(
                state=state,
                student=student,
                teacher_matched=matched,
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
    primary_label = privileged_current_generalization_label(
        aggregates=aggregates,
        paired_statistics=paired_statistics,
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
        "primary_diagnostic_policy": _multi_identity_primary_policy(),
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
    teacher_predictions = run_teacher_branch_predictions(
        runtime_factory=teacher_runtime_factory,
        states=states,
        source_noise=source_noise,
        teacher_target=teacher_target,
        teacher_payload=teacher_payload,
        conditional_dict=conditional_dict,
    )
    return build_teacher_flow_audit_result(
        states=states,
        student_predictions=student_predictions,
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
    }


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
            STRONG_PRIVILEGED_CURRENT_SUPPORT,
            NO_PRIVILEGED_CURRENT_SUPPORT,
            INCONCLUSIVE,
        }:
            raise RuntimeError("multi-identity Teacher-flow diagnostic label invalid")
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
    return FlowPrediction(
        state_id=state.state_id,
        branch=TEACHER_MATCHED_CURRENT_BRANCH,
        flow=flow.detach().clone(),
        x0=x0.detach().clone(),
        proof=proof,
    )


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
    return FlowPrediction(
        state_id=state.state_id,
        branch=TEACHER_CLEAN_CURRENT_BRANCH,
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
        fn=call_teacher,
    )
    flow, _ = deployment._unpack_main_outputs(outputs)
    if tuple(flow.shape) != tuple(chunk.shape):
        raise RuntimeError(f"{label} flow shape mismatch")
    _require_finite_tensor(flow, name=label)
    return flow, dict(rng_guard)


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
        "current_forward_rng": dict(current_guard),
        "future_forward_rng": dict(future_guard),
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
    return proof


def _state_metrics(
    *,
    state: TeacherFlowAuditState,
    student: FlowPrediction,
    teacher_matched: FlowPrediction,
    teacher_clean: FlowPrediction,
) -> dict[str, float]:
    clean_future = _chunk(state.noisy_batch.clean_target, FUTURE_CHUNK_INDEX)
    return {
        "mcp_flow_vs_exact_mse": route_eq._mse(student.flow, state.exact_mcp_target),
        "mcp_x0_vs_clean_future_mse": route_eq._mse(student.x0, clean_future),
        "teacher_matched_flow_vs_exact_mse": route_eq._mse(
            teacher_matched.flow,
            state.exact_mcp_target,
        ),
        "teacher_matched_x0_vs_clean_future_mse": route_eq._mse(
            teacher_matched.x0,
            clean_future,
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
    teacher_matched: FlowPrediction,
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
        "teacher_matched_current_state_exact": teacher_matched.proof[
            "current_state_sha256"
        ]
        == current_sha,
        "main_sigma": float(state.main_sigma),
        "future_sigma": float(state.future_sigma),
        "teacher_future_timestep": float(state.teacher_future_timestep),
        "raw_timestep_directly_used_for_teacher": False,
    }
    proof["all_future_states_exact"] = (
        proof["student_future_state_exact"]
        and proof["teacher_matched_future_state_exact"]
        and proof["teacher_clean_future_state_exact"]
    )
    if not proof["all_future_states_exact"]:
        raise RuntimeError("future state exact proof failed")
    if not proof["teacher_matched_current_state_exact"]:
        raise RuntimeError("matched current state exact proof failed")
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


def _paired_values(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    aggregate = _aggregate_group(records)
    mcp_flow = float(aggregate["mcp_flow_vs_exact_mse"]["mean"])
    matched_flow = float(aggregate["teacher_matched_flow_vs_exact_mse"]["mean"])
    privileged_flow = float(aggregate["teacher_clean_flow_vs_exact_mse"]["mean"])
    mcp_x0 = float(aggregate["mcp_x0_vs_clean_future_mse"]["mean"])
    matched_x0 = float(aggregate["teacher_matched_x0_vs_clean_future_mse"]["mean"])
    privileged_x0 = float(aggregate["teacher_clean_x0_vs_clean_future_mse"]["mean"])
    return {
        "mcp_flow_mse": mcp_flow,
        "matched_flow_mse": matched_flow,
        "privileged_flow_mse": privileged_flow,
        "mcp_x0_mse": mcp_x0,
        "matched_x0_mse": matched_x0,
        "privileged_x0_mse": privileged_x0,
        "privileged_flow_reduction": _relative_reduction(
            mcp_flow,
            privileged_flow,
        ),
        "matched_flow_reduction": _relative_reduction(mcp_flow, matched_flow),
        "privileged_x0_reduction": _relative_reduction(mcp_x0, privileged_x0),
        "matched_x0_reduction": _relative_reduction(mcp_x0, matched_x0),
        "privileged_flow_win": privileged_flow < mcp_flow,
        "matched_flow_win": matched_flow < mcp_flow,
    }


def _paired_summary(values: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not values:
        raise RuntimeError("cannot summarize empty paired values")
    privileged_wins = sum(1 for value in values if value["privileged_flow_win"])
    matched_wins = sum(1 for value in values if value["matched_flow_win"])
    return {
        "state_count": int(len(values)),
        "privileged_flow_win_count": int(privileged_wins),
        "privileged_flow_win_rate": _rate(privileged_wins, len(values)),
        "matched_flow_win_count": int(matched_wins),
        "matched_flow_win_rate": _rate(matched_wins, len(values)),
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
    }


def _relative_reduction(baseline: float, candidate: float) -> float:
    baseline_value = float(baseline)
    if baseline_value <= 0.0:
        return 0.0
    return float((baseline_value - float(candidate)) / baseline_value)


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
        },
        TEACHER_CLEAN_CURRENT_BRANCH: {
            "fresh_teacher_kv": True,
            "history": "clean chunk0 recache",
            "current": "clean chunk1 recache",
            "future": "same noisy future_state as MCP target",
            "privileged_clean_current": True,
            "same_information_as_mcp": False,
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


def _conversion_contract() -> dict[str, Any]:
    return {
        "teacher_x0": "future_state - sigma_future * teacher_flow",
        "mcp_x0": "future_state - sigma_future * mcp_flow",
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
            "the privileged Teacher is oracle, or Teacher flow should directly "
            "replace FM target"
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
        "teacher_matched_flow_vs_exact_mse",
        "teacher_matched_x0_vs_clean_future_mse",
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
    clean = record.get("teacher_clean_current")
    if not isinstance(matched, Mapping) or not isinstance(clean, Mapping):
        raise RuntimeError("teacher-flow branch records missing")
    matched_proof = matched.get("proof")
    clean_proof = clean.get("proof")
    if not isinstance(matched_proof, Mapping) or not isinstance(clean_proof, Mapping):
        raise RuntimeError("teacher-flow branch proof missing")
    if matched_proof.get("privileged_clean_current") is not False:
        raise RuntimeError("matched Teacher branch is marked privileged")
    if matched_proof.get("same_information_as_mcp") is not True:
        raise RuntimeError("matched Teacher branch information flag mismatch")
    if clean_proof.get("privileged_clean_current") is not True:
        raise RuntimeError("clean-current Teacher branch privilege flag missing")
    if clean_proof.get("same_information_as_mcp") is not False:
        raise RuntimeError("clean-current Teacher branch information flag mismatch")
    proof = record.get("same_state_sigma_proof")
    if not isinstance(proof, Mapping) or proof.get("all_future_states_exact") is not True:
        raise RuntimeError("teacher-flow same future state proof missing")
    if proof.get("raw_timestep_directly_used_for_teacher") is not False:
        raise RuntimeError("teacher-flow Teacher used raw timestep directly")


def _validate_teacher_runtime(runtime: deployment.DeploymentRuntime) -> None:
    deployment._validate_runtime(runtime)
    if getattr(runtime.generator, "mcp", None) is not None:
        raise RuntimeError("Teacher runtime must be Main-only")
    if int(runtime.context_noise) != 0:
        raise RuntimeError("Teacher audit requires clean-history context_noise=0")


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
    "INCONCLUSIVE",
    "MATCHED_TEACHER_TIMESTEP_DEPENDENCE",
    "NO_PRIVILEGED_CURRENT_SUPPORT",
    "STUDENT_MCP_BRANCH",
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
    "TEACHER_PRIVILEGED_ONLY_BETTER",
    "FlowPrediction",
    "TeacherFlowAuditResult",
    "TeacherFlowAuditState",
    "aggregate_teacher_flow_metrics",
    "build_flow_match_scheduler",
    "build_teacher_flow_multi_identity_manifest",
    "build_teacher_flow_audit_result",
    "build_teacher_flow_audit_states",
    "build_teacher_flow_state_records",
    "diagnostic_label_from_metrics",
    "load_teacher_flow_student_checkpoint_record",
    "matched_teacher_timestep_diagnostic",
    "manual_flow_to_x0",
    "paired_teacher_flow_statistics",
    "privileged_current_generalization_label",
    "run_student_mcp_full_sequence_predictions",
    "run_teacher_branch_predictions",
    "run_teacher_flow_audit",
    "select_validation32_identities",
    "select_validation_zero_identity",
    "validate_frozen_teacher_model",
    "validate_multi_identity_student_checkpoint_contract",
    "validate_teacher_flow_artifact_identity_selection",
    "validate_teacher_flow_artifact_identity",
    "validate_teacher_flow_audit_manifest",
]
