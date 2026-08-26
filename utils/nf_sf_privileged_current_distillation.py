from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

import utils.nf_sf_first_mcp_flow_audit as flow_audit
import utils.nf_sf_teacher_flow_audit as teacher_audit
from utils.nf_sf_full_sequence_eval import (
    OFFICIAL_SELF_FORCING_CHECKPOINT_SHA256,
    build_absolute_chunk_rng_plan,
    global_rng_state_hash,
)
from utils.nf_sf_m3 import tensor_sha256, tensor_summary
from utils.nf_sf_mcp1_only_continuation import load_matching_control_artifact_bundle
from utils.nf_sf_tensors import (
    DEFAULT_NUM_TRAIN_TIMESTEPS,
    FULL_SEQUENCE_CHUNK_FRAMES,
    FULL_SEQUENCE_NUM_CHUNKS,
)
from utils.nf_sf_training import (
    FULL_SEQUENCE_DEPTH_WEIGHTS,
    FULL_SEQUENCE_OBJECTIVE_VERSION,
    FULL_SEQUENCE_RUN_KIND,
    FULL_SEQUENCE_TRAINER_SCHEMA,
    NFSFFullSequenceForwardResult,
    NFSFFullSequenceNoisyBatch,
    build_full_sequence_mcp_anchor_inputs,
    collect_nf_sf_parameter_groups,
    nf_sf_full_sequence_train_cursor,
    run_nf_sf_full_sequence_forward_loss,
)


PRIVILEGED_CURRENT_DISTILLATION_SCHEMA = (
    "nf_sf_privileged_current_distillation_v1"
)
PRIVILEGED_CURRENT_PARENT_STEP = 6500
PRIVILEGED_CURRENT_TARGET_STEP = 7000
PRIVILEGED_CURRENT_UPDATE_COUNT = 500
PRIVILEGED_CURRENT_PARENT_CHECKPOINT_SHA256 = (
    "9ef57cb2d3e5f20b244129317af4a0e1d2b1c810ba65ec970892e60ccbd34f4f"
)
PRIVILEGED_CURRENT_PARENT_GIT_SHA = (
    "c3f89888bf6da31b48650f0a680dd6534943f56f"
)
PRIVILEGED_CURRENT_TEACHER_CHECKPOINT_SHA256 = (
    "a0413986d9734e02c09504e1520f5697ba6df731bb2f0f35577485e9cc8f56a3"
)
PRIVILEGED_CURRENT_LAMBDA = 0.25
PRIVILEGED_CURRENT_OBJECTIVE_MODE = "next_forcing_full"
PRIVILEGED_CURRENT_AUX_DEPTH = 1

STRONG_SUPPORT = "SUPPORT_PRIVILEGED_CURRENT_DISTILLATION"
NO_SUPPORT = "NO_SUPPORT"
INCONCLUSIVE = "INCONCLUSIVE"


@dataclass(frozen=True)
class MCP1AnchorSemantics:
    anchor_index: int
    current_chunk_index: int
    future_chunk_index: int
    future_start_frame: int
    recache_chunk_indices: tuple[int, ...]
    future_state: torch.Tensor
    future_timestep: torch.Tensor
    physical_sigma: float
    teacher_timestep: torch.Tensor
    proof: dict[str, Any]


@dataclass(frozen=True)
class PrivilegedTeacherTargets:
    target_flows: torch.Tensor
    anchor_records: tuple[dict[str, Any], ...]
    target_summary: dict[str, Any]
    rng_guard: dict[str, Any]


@dataclass(frozen=True)
class PrivilegedForwardResult:
    canonical: NFSFFullSequenceForwardResult
    total_loss: torch.Tensor
    canonical_loss: torch.Tensor
    privileged_loss: torch.Tensor
    mcp1_exact_loss: torch.Tensor
    teacher_targets: PrivilegedTeacherTargets
    loss_record: dict[str, Any]


def privileged_run_plan() -> dict[str, Any]:
    return {
        "schema": PRIVILEGED_CURRENT_DISTILLATION_SCHEMA,
        "parent_step": PRIVILEGED_CURRENT_PARENT_STEP,
        "target_step": PRIVILEGED_CURRENT_TARGET_STEP,
        "update_count": PRIVILEGED_CURRENT_UPDATE_COUNT,
        "first_step": PRIVILEGED_CURRENT_PARENT_STEP + 1,
        "last_step": PRIVILEGED_CURRENT_TARGET_STEP,
        "objective_mode": PRIVILEGED_CURRENT_OBJECTIVE_MODE,
        "canonical_objective_unchanged": True,
        "inference_graph_changed": False,
        "teacher_trained": False,
        "auxiliary_depths": [PRIVILEGED_CURRENT_AUX_DEPTH],
        "lambda_priv": PRIVILEGED_CURRENT_LAMBDA,
        "lambda_fixed_formal_contract": True,
        "lambda_rationale": (
            "MCP1 canonical weight=0.5 and exact/teacher-distance losses are "
            "expected to be same scale; lambda=0.25 starts at half the direct "
            "MCP1 canonical weight as a conservative first intervention."
        ),
        "ab_decision_rule": privileged_ab_decision_rule(),
        "diagnostic_only": True,
        "non_canonical": True,
        "canonical_training_eligible": False,
        "deployment_eligible": False,
    }


def validate_lambda_priv(lambda_priv: float, *, formal: bool = True) -> float:
    value = float(lambda_priv)
    if value < 0.0 or not torch.isfinite(torch.tensor(value)):
        raise ValueError("lambda_priv must be finite and non-negative")
    if formal and value != PRIVILEGED_CURRENT_LAMBDA:
        raise RuntimeError("formal privileged-current distillation fixes lambda=0.25")
    return value


def validate_control_reuse(summary_path: Any) -> dict[str, Any]:
    if summary_path is None:
        return {
            "schema": f"{PRIVILEGED_CURRENT_DISTILLATION_SCHEMA}_control_reuse_v1",
            "CONTROL_REUSABLE": False,
            "failures": ["matching control summary path missing"],
        }
    audit = load_matching_control_artifact_bundle(summary_path)
    failures = list(audit.get("failures", ()))
    if audit.get("CONTROL_REUSABLE") is not True:
        failures.append("strict matching control validator did not pass")
    return {
        **dict(audit),
        "schema": f"{PRIVILEGED_CURRENT_DISTILLATION_SCHEMA}_control_reuse_v1",
        "CONTROL_REUSABLE": not failures,
        "failures": failures,
        "required_parent_checkpoint_sha256": PRIVILEGED_CURRENT_PARENT_CHECKPOINT_SHA256,
        "required_parent_git_sha": PRIVILEGED_CURRENT_PARENT_GIT_SHA,
        "required_update_count": PRIVILEGED_CURRENT_UPDATE_COUNT,
        "requires_no_direct_clean_context_kv": True,
    }


def mcp1_anchor_semantics(
    *,
    noisy_batch: NFSFFullSequenceNoisyBatch,
    mcp_scheduler: Any,
) -> tuple[MCP1AnchorSemantics, ...]:
    anchors = build_full_sequence_mcp_anchor_inputs(noisy_batch)
    semantics: list[MCP1AnchorSemantics] = []
    for anchor in anchors:
        anchor_index = int(anchor["anchor_index"])
        depths = tuple(int(value) for value in anchor["depths"])
        if PRIVILEGED_CURRENT_AUX_DEPTH not in depths:
            continue
        depth_position = depths.index(PRIVILEGED_CURRENT_AUX_DEPTH)
        future_chunk = anchor_index + PRIVILEGED_CURRENT_AUX_DEPTH
        future_state = anchor["future_noises"][depth_position]
        future_timestep = anchor["timesteps"][depth_position]
        canonical_future = noisy_batch.noisy_mcp_depths[0][:, anchor_index]
        canonical_timestep = noisy_batch.timestep_mcp_depths[0][:, anchor_index]
        if not torch.equal(future_state, canonical_future):
            raise RuntimeError("MCP1 anchor future tensor differs from noisy_batch")
        if not torch.equal(future_timestep, canonical_timestep):
            raise RuntimeError("MCP1 anchor timestep differs from noisy_batch")
        sigma = flow_audit._resolved_sigma(
            mcp_scheduler,
            future_timestep,
            future_state,
        )
        teacher_timestep = torch.full(
            future_state.shape[:2],
            float(sigma) * float(DEFAULT_NUM_TRAIN_TIMESTEPS),
            device=future_state.device,
            dtype=torch.float32,
        )
        recache_chunks = tuple(range(anchor_index + 1))
        proof = {
            "anchor_index": anchor_index,
            "depth": PRIVILEGED_CURRENT_AUX_DEPTH,
            "current_chunk_index": anchor_index,
            "future_chunk_index": future_chunk,
            "future_start_frame": int(future_chunk * FULL_SEQUENCE_CHUNK_FRAMES),
            "recache_chunk_indices": list(recache_chunks),
            "context_boundary": f"clean/canonical chunks <= {anchor_index}",
            "future_clean_leakage": False,
            "future_state_sha256": tensor_sha256(future_state.detach().cpu()),
            "canonical_future_state_sha256": tensor_sha256(
                canonical_future.detach().cpu()
            ),
            "same_future_tensor_as_student": True,
            "physical_sigma": float(sigma),
            "teacher_timestep": float(sigma) * float(DEFAULT_NUM_TRAIN_TIMESTEPS),
            "teacher_timestep_contract": "physical_sigma * 1000",
            "raw_timestep_directly_used_for_teacher": False,
        }
        semantics.append(
            MCP1AnchorSemantics(
                anchor_index=anchor_index,
                current_chunk_index=anchor_index,
                future_chunk_index=future_chunk,
                future_start_frame=int(future_chunk * FULL_SEQUENCE_CHUNK_FRAMES),
                recache_chunk_indices=recache_chunks,
                future_state=future_state,
                future_timestep=future_timestep,
                physical_sigma=float(sigma),
                teacher_timestep=teacher_timestep,
                proof=proof,
            )
        )
    expected = FULL_SEQUENCE_NUM_CHUNKS - PRIVILEGED_CURRENT_AUX_DEPTH
    if len(semantics) != expected:
        raise RuntimeError("canonical MCP1 anchor count mismatch")
    for expected_anchor, semantic in enumerate(semantics):
        if semantic.anchor_index != expected_anchor:
            raise RuntimeError("canonical MCP1 anchor order mismatch")
    return tuple(semantics)


def build_privileged_mcp1_teacher_targets(
    *,
    teacher_runtime_factory: Any,
    noisy_batch: NFSFFullSequenceNoisyBatch,
    source_noise: torch.Tensor,
    teacher_payload: Mapping[str, Any],
    conditional_dict: Mapping[str, Any],
    mcp_scheduler: Any,
) -> PrivilegedTeacherTargets:
    if tuple(source_noise.shape) != tuple(noisy_batch.clean_target.shape):
        raise RuntimeError("Teacher source_noise must match clean training target")
    if str(PRIVILEGED_CURRENT_TEACHER_CHECKPOINT_SHA256) != (
        OFFICIAL_SELF_FORCING_CHECKPOINT_SHA256
    ):
        raise RuntimeError("Teacher checkpoint SHA contract mismatch")
    active_rng_before = global_rng_state_hash(source_noise.device)
    rng_plan = build_absolute_chunk_rng_plan(
        source_noise=source_noise,
        rollout_seed=int(teacher_payload["rollout_seed"]),
        num_denoising_steps=4,
        chunk_frames=FULL_SEQUENCE_CHUNK_FRAMES,
    )
    semantics = mcp1_anchor_semantics(
        noisy_batch=noisy_batch,
        mcp_scheduler=mcp_scheduler,
    )
    target_flows = []
    anchor_records = []
    with torch.no_grad():
        for semantic in semantics:
            runtime = teacher_runtime_factory()
            teacher_audit._validate_teacher_runtime(runtime)
            recache_records = []
            for chunk_index in semantic.recache_chunk_indices:
                recache_records.append(
                    teacher_audit._recache_clean_chunk(
                        runtime=runtime,
                        teacher_target=noisy_batch.clean_target,
                        conditional_dict=conditional_dict,
                        rng_plan=rng_plan,
                        chunk_index=int(chunk_index),
                    )
                )
            flow, future_guard = teacher_audit._teacher_forward_chunk(
                runtime=runtime,
                conditional_dict=conditional_dict,
                chunk=semantic.future_state,
                timestep=semantic.teacher_timestep,
                start_frame=semantic.future_start_frame,
                label="privileged_current_teacher_future_forward",
            )
            if flow.requires_grad:
                raise RuntimeError("privileged Teacher target requires grad")
            target_flows.append(flow.detach())
            anchor_records.append(
                _anchor_record(
                    semantic=semantic,
                    flow=flow,
                    recache_records=recache_records,
                    future_guard=future_guard,
                )
            )
    target = torch.stack(target_flows, dim=1).detach()
    expected_shape = tuple(noisy_batch.noisy_mcp_depths[0].shape)
    if tuple(target.shape) != expected_shape:
        raise RuntimeError("privileged Teacher MCP1 target shape mismatch")
    active_rng_after = global_rng_state_hash(source_noise.device)
    if active_rng_after != active_rng_before:
        raise RuntimeError("privileged Teacher target construction changed RNG")
    return PrivilegedTeacherTargets(
        target_flows=target,
        anchor_records=tuple(anchor_records),
        target_summary=_tensor_record(target),
        rng_guard={
            "state_before_hash": active_rng_before,
            "state_after_hash": active_rng_after,
            "unchanged": True,
        },
    )


def run_privileged_current_forward_loss(
    generator: Any,
    *,
    teacher_runtime_factory: Any,
    conditional_dict: Mapping[str, Any],
    noisy_batch: NFSFFullSequenceNoisyBatch,
    source_noise: torch.Tensor,
    teacher_payload: Mapping[str, Any],
    mcp_scheduler: Any,
    lambda_priv: float = PRIVILEGED_CURRENT_LAMBDA,
) -> PrivilegedForwardResult:
    lambda_value = validate_lambda_priv(lambda_priv, formal=False)
    teacher_targets = build_privileged_mcp1_teacher_targets(
        teacher_runtime_factory=teacher_runtime_factory,
        noisy_batch=noisy_batch,
        source_noise=source_noise,
        teacher_payload=teacher_payload,
        conditional_dict=conditional_dict,
        mcp_scheduler=mcp_scheduler,
    )
    canonical = run_nf_sf_full_sequence_forward_loss(
        generator,
        conditional_dict=dict(conditional_dict),
        noisy_batch=noisy_batch,
        objective_mode=PRIVILEGED_CURRENT_OBJECTIVE_MODE,
    )
    if len(canonical.mcp_flow_preds_by_depth) != 3:
        raise RuntimeError("canonical full-sequence output must include MCP1/2/3")
    mcp1_pred = canonical.mcp_flow_preds_by_depth[0]
    if tuple(mcp1_pred.shape) != tuple(teacher_targets.target_flows.shape):
        raise RuntimeError("MCP1 prediction shape differs from Teacher target")
    privileged_loss = F.mse_loss(
        mcp1_pred.float(),
        teacher_targets.target_flows.float().detach(),
        reduction="mean",
    )
    canonical_loss = canonical.losses.total_loss
    total_loss = canonical_loss + lambda_value * privileged_loss
    record = privileged_loss_record(
        canonical=canonical,
        teacher_targets=teacher_targets,
        privileged_loss=privileged_loss,
        total_loss=total_loss,
        lambda_priv=lambda_value,
    )
    return PrivilegedForwardResult(
        canonical=canonical,
        total_loss=total_loss,
        canonical_loss=canonical_loss,
        privileged_loss=privileged_loss,
        mcp1_exact_loss=canonical.losses.mcp_depth_losses[0],
        teacher_targets=teacher_targets,
        loss_record=record,
    )


def privileged_loss_record(
    *,
    canonical: NFSFFullSequenceForwardResult,
    teacher_targets: PrivilegedTeacherTargets,
    privileged_loss: torch.Tensor,
    total_loss: torch.Tensor,
    lambda_priv: float,
) -> dict[str, Any]:
    mcp1_pred = canonical.mcp_flow_preds_by_depth[0]
    exact = canonical.noisy_batch.target_flow_mcp_depths[0]
    teacher = teacher_targets.target_flows
    teacher_vs_exact = F.mse_loss(
        teacher.float(),
        exact.float(),
        reduction="mean",
    )
    return {
        "canonical_loss": float(canonical.losses.total_loss.detach().float().item()),
        "main_loss": float(canonical.losses.main_loss.detach().float().item()),
        "mcp1_exact_loss": float(
            canonical.losses.mcp_depth_losses[0].detach().float().item()
        ),
        "mcp2_exact_loss": float(
            canonical.losses.mcp_depth_losses[1].detach().float().item()
        ),
        "mcp3_exact_loss": float(
            canonical.losses.mcp_depth_losses[2].detach().float().item()
        ),
        "privileged_mcp1_loss": float(
            privileged_loss.detach().float().item()
        ),
        "student_vs_teacher_flow_mse": float(
            privileged_loss.detach().float().item()
        ),
        "teacher_vs_exact_flow_mse": float(
            teacher_vs_exact.detach().float().item()
        ),
        "lambda_priv": float(lambda_priv),
        "weighted_privileged_loss": float(
            (float(lambda_priv) * privileged_loss.detach().float()).item()
        ),
        "total_loss": float(total_loss.detach().float().item()),
        "formula": (
            "L_total = L_main + 0.5*L_mcp1_exact + 0.2*L_mcp2_exact "
            "+ 0.1*L_mcp3_exact + lambda_priv*L_mcp1_privileged"
        ),
        "canonical_depth_weights": list(FULL_SEQUENCE_DEPTH_WEIGHTS),
        "auxiliary_depths": [PRIVILEGED_CURRENT_AUX_DEPTH],
        "teacher_target_summary": dict(teacher_targets.target_summary),
        "anchor_records": [dict(record) for record in teacher_targets.anchor_records],
    }


def teacher_frozen_report(teacher: torch.nn.Module) -> dict[str, Any]:
    trainable = [
        name for name, param in teacher.named_parameters() if param.requires_grad
    ]
    mcp_tensors = [
        name for name, tensor in teacher.state_dict().items()
        if str(name).startswith("mcp.") and torch.is_tensor(tensor)
    ]
    if trainable:
        raise RuntimeError("Teacher has trainable parameters")
    if getattr(teacher, "training", False):
        raise RuntimeError("Teacher must be eval")
    if getattr(teacher, "mcp", None) is not None:
        raise RuntimeError("Teacher must be Main-only")
    if mcp_tensors:
        raise RuntimeError("Teacher state_dict contains MCP tensors")
    return {
        "eval_mode": True,
        "requires_grad_false": True,
        "mcp_tensor_count": len(mcp_tensors),
        "parameter_sha256": parameter_sha256_report(teacher),
    }


def parameter_sha256_report(module: torch.nn.Module) -> dict[str, Any]:
    entries = []
    for name, tensor in module.state_dict().items():
        if torch.is_tensor(tensor):
            entries.append(
                {
                    "name": str(name),
                    "sha256": _sha256_tensor(tensor.detach().cpu()),
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                }
            )
    digest = hashlib.sha256(
        "\n".join(f"{item['name']}:{item['sha256']}" for item in entries)
        .encode("utf-8")
    ).hexdigest()
    return {
        "tensor_count": len(entries),
        "aggregate_sha256": digest,
        "entries": entries,
    }


def optimizer_state_fingerprint(optimizer: torch.optim.Optimizer) -> str:
    entries = []
    state = optimizer.state_dict()
    for group_index, group in enumerate(state.get("param_groups", ())):
        entries.append(
            f"group:{group_index}:{group.get('name')}:{group.get('lr')}:"
            f"{group.get('weight_decay')}:{list(group.get('params', ()))}"
        )
    for key, value in sorted(state.get("state", {}).items(), key=lambda item: str(item[0])):
        entries.append(f"state:{key}")
        if isinstance(value, Mapping):
            for state_key, state_value in sorted(value.items(), key=lambda item: str(item[0])):
                if torch.is_tensor(state_value):
                    entries.append(
                        f"{state_key}:tensor:{list(state_value.shape)}:"
                        f"{state_value.dtype}:{_sha256_tensor(state_value.detach().cpu())}"
                    )
                else:
                    entries.append(f"{state_key}:{state_value!r}")
    return hashlib.sha256("\n".join(entries).encode("utf-8")).hexdigest()


def gradient_group_report(module: torch.nn.Module) -> dict[str, dict[str, Any]]:
    report = {}
    for group_name, named_params in collect_nf_sf_parameter_groups(module).items():
        values = []
        missing = 0
        finite = True
        for _, param in named_params:
            if not param.requires_grad:
                continue
            if param.grad is None:
                missing += 1
                continue
            grad = param.grad.detach().float().reshape(-1)
            finite = finite and bool(torch.isfinite(grad).all().item())
            values.append(grad.cpu())
        if values:
            flat = torch.cat(values)
            norm = float(flat.norm().item())
        else:
            flat = torch.empty(0)
            norm = 0.0
        report[group_name] = {
            "grad_tensors": int(len(values)),
            "missing_grad_tensors": int(missing),
            "finite": bool(finite),
            "norm": norm,
            "_flat": flat,
        }
    return report


def compare_gradient_reports(
    canonical: Mapping[str, Mapping[str, Any]],
    privileged: Mapping[str, Mapping[str, Any]],
    *,
    lambda_priv: float = PRIVILEGED_CURRENT_LAMBDA,
) -> dict[str, Any]:
    groups = ("backbone", "patch_embedding", "mcp_fusion", "mcp_depth1")
    result = {}
    for group in groups:
        left = canonical[group]["_flat"]
        right = privileged[group]["_flat"]
        if left.numel() != right.numel():
            cosine = None
        elif left.numel() == 0 or float(left.norm().item()) == 0.0:
            cosine = None
        elif float(right.norm().item()) == 0.0:
            cosine = None
        else:
            cosine = float(F.cosine_similarity(left, right, dim=0).item())
        canonical_norm = float(canonical[group]["norm"])
        privileged_norm = float(privileged[group]["norm"])
        ratio = None
        if canonical_norm > 0.0:
            ratio = float(float(lambda_priv) * privileged_norm / canonical_norm)
        result[group] = {
            "canonical_norm": canonical_norm,
            "privileged_norm": privileged_norm,
            "canonical_finite": bool(canonical[group]["finite"]),
            "privileged_finite": bool(privileged[group]["finite"]),
            "cosine": cosine,
            "lambda_scaled_aux_to_canonical_norm_ratio": ratio,
        }
    return result


def strip_gradient_flats(report: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    return {
        group: {
            key: value
            for key, value in data.items()
            if key != "_flat"
        }
        for group, data in report.items()
    }


def classify_privileged_current_ab(
    *,
    control_raw999_mcp1_mse: float,
    treatment_raw999_mcp1_mse: float,
    control_validation_mcp1_mse: float,
    treatment_validation_mcp1_mse: float,
    control_main_mse: float,
    treatment_main_mse: float,
) -> dict[str, Any]:
    raw_reduction = _relative_improvement(
        control_raw999_mcp1_mse,
        treatment_raw999_mcp1_mse,
    )
    validation_reduction = _relative_improvement(
        control_validation_mcp1_mse,
        treatment_validation_mcp1_mse,
    )
    main_degradation = _relative_degradation(control_main_mse, treatment_main_mse)
    if (
        raw_reduction >= 0.10
        and validation_reduction >= 0.05
        and main_degradation < 0.05
    ):
        decision = STRONG_SUPPORT
    elif (
        (raw_reduction < 0.05 and validation_reduction < 0.05)
        or main_degradation >= 0.05
    ):
        decision = NO_SUPPORT
    else:
        decision = INCONCLUSIVE
    return {
        "decision": decision,
        "raw999_mcp1_reduction": raw_reduction,
        "validation_mcp1_reduction": validation_reduction,
        "main_degradation": main_degradation,
        "thresholds": privileged_ab_decision_rule(),
    }


def privileged_ab_decision_rule() -> dict[str, Any]:
    return {
        STRONG_SUPPORT: {
            "primary_metric": "fixed raw999 MCP1 MSE",
            "raw999_mcp1_reduction_min": 0.10,
            "validation_mcp1_reduction_min": 0.05,
            "main_degradation_max_exclusive": 0.05,
        },
        NO_SUPPORT: {
            "raw999_mcp1_reduction_below": 0.05,
            "validation_mcp1_reduction_below": 0.05,
            "main_degradation_min": 0.05,
        },
        INCONCLUSIVE: {
            "condition": "all other mixed cases",
        },
    }


def first_step_contract(global_step: int, sample_cursor: Mapping[str, Any]) -> dict[str, Any]:
    if int(global_step) != PRIVILEGED_CURRENT_PARENT_STEP + 1:
        raise RuntimeError("privileged-current first step mismatch")
    expected = nf_sf_full_sequence_train_cursor(global_step)
    if dict(sample_cursor) != dict(expected):
        raise RuntimeError("privileged-current sample cursor mismatch")
    return {
        "status": "PASS",
        "first_global_step": int(global_step),
        "first_sample_cursor": dict(sample_cursor),
    }


def _anchor_record(
    *,
    semantic: MCP1AnchorSemantics,
    flow: torch.Tensor,
    recache_records: Sequence[Mapping[str, Any]],
    future_guard: Mapping[str, Any],
) -> dict[str, Any]:
    if not recache_records:
        raise RuntimeError("privileged Teacher anchor has no recached context")
    if any(int(record["context_noise"]) != 0 for record in recache_records):
        raise RuntimeError("privileged Teacher recache context_noise mismatch")
    proof = dict(semantic.proof)
    proof.update(
        {
            "privileged_clean_current": True,
            "same_information_as_mcp": False,
            "teacher_route": "teacher_clean_current_generalized_to_mcp1_anchor",
            "uses_ground_truth_future_x0_for_conversion": False,
            "uses_wrapper_auto_x0": False,
            "future_forward_rng": dict(future_guard),
            "recache_count": len(recache_records),
            "recache_context_noises": [
                int(record["context_noise"]) for record in recache_records
            ],
            "history_context_latent_exact_clean": False,
        }
    )
    return {
        "anchor_index": int(semantic.anchor_index),
        "current_chunk_index": int(semantic.current_chunk_index),
        "future_chunk_index": int(semantic.future_chunk_index),
        "future_state": _tensor_record(semantic.future_state),
        "teacher_flow": _tensor_record(flow),
        "proof": proof,
    }


def _tensor_record(tensor: torch.Tensor) -> dict[str, Any]:
    summary = tensor_summary(tensor.detach().cpu())
    return {
        "shape": list(summary["shape"]),
        "dtype": summary["dtype"],
        "finite": summary["finite"],
        "sha256": summary["sha256"],
        "mean_abs": float(tensor.detach().float().abs().mean().item()),
        "max_abs": float(tensor.detach().float().abs().max().item()),
    }


def _sha256_tensor(tensor: torch.Tensor) -> str:
    normalized = tensor.detach().cpu().contiguous().reshape(-1)
    return tensor_sha256(normalized)


def _relative_improvement(baseline: float, candidate: float) -> float:
    base = _finite_nonnegative(baseline, name="baseline")
    value = _finite_nonnegative(candidate, name="candidate")
    if base == 0.0:
        return 0.0
    return float((base - value) / base)


def _relative_degradation(baseline: float, candidate: float) -> float:
    base = _finite_nonnegative(baseline, name="baseline")
    value = _finite_nonnegative(candidate, name="candidate")
    if base == 0.0:
        return 0.0
    return float((value - base) / base)


def _finite_nonnegative(value: float, *, name: str) -> float:
    number = float(value)
    if not torch.isfinite(torch.tensor(number)) or number < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return number


def provenance_contract() -> dict[str, Any]:
    return {
        "schema": PRIVILEGED_CURRENT_DISTILLATION_SCHEMA,
        "run_kind": FULL_SEQUENCE_RUN_KIND,
        "objective_version": FULL_SEQUENCE_OBJECTIVE_VERSION,
        "objective_mode": PRIVILEGED_CURRENT_OBJECTIVE_MODE,
        "canonical_objective_preserved": True,
        "exact_fm_replaced": False,
        "inference_graph_changed": False,
        "teacher_trained": False,
        "teacher_checkpoint_sha256": PRIVILEGED_CURRENT_TEACHER_CHECKPOINT_SHA256,
        "parent_checkpoint_sha256": PRIVILEGED_CURRENT_PARENT_CHECKPOINT_SHA256,
        "parent_checkpoint_git_sha": PRIVILEGED_CURRENT_PARENT_GIT_SHA,
        "depth_weights": list(FULL_SEQUENCE_DEPTH_WEIGHTS),
        "lambda_priv": PRIVILEGED_CURRENT_LAMBDA,
        "ab_decision_rule": privileged_ab_decision_rule(),
        "privileged_branch": "teacher_clean_current",
        "privileged_current_is_near_clean_canonical_recache": True,
        "privileged_current_is_oracle": False,
        "forbidden": {
            "ground_truth_future_x0_to_make_teacher_flow": False,
            "raw_timestep_as_teacher_timestep": False,
            "wrapper_auto_x0": False,
            "teacher_rng_stream_mutation": False,
            "mcp2_privileged_auxiliary_loss": False,
            "mcp3_privileged_auxiliary_loss": False,
            "deployment_graph_change": False,
        },
        "interpretation_boundary": (
            "Even if SUPPORT holds, only claim that privileged near-clean "
            "current information is a promising training-time distillation "
            "signal; do not claim conditional ambiguity is proven, exact FM is "
            "wrong, the Teacher is oracle, or Teacher target should replace FM."
        ),
    }


__all__ = [
    "INCONCLUSIVE",
    "MCP1AnchorSemantics",
    "NO_SUPPORT",
    "PRIVILEGED_CURRENT_DISTILLATION_SCHEMA",
    "PRIVILEGED_CURRENT_LAMBDA",
    "PRIVILEGED_CURRENT_PARENT_CHECKPOINT_SHA256",
    "PRIVILEGED_CURRENT_PARENT_GIT_SHA",
    "PRIVILEGED_CURRENT_TARGET_STEP",
    "PRIVILEGED_CURRENT_TEACHER_CHECKPOINT_SHA256",
    "PRIVILEGED_CURRENT_UPDATE_COUNT",
    "PrivilegedForwardResult",
    "PrivilegedTeacherTargets",
    "STRONG_SUPPORT",
    "build_privileged_mcp1_teacher_targets",
    "classify_privileged_current_ab",
    "compare_gradient_reports",
    "first_step_contract",
    "gradient_group_report",
    "mcp1_anchor_semantics",
    "optimizer_state_fingerprint",
    "parameter_sha256_report",
    "privileged_loss_record",
    "privileged_run_plan",
    "privileged_ab_decision_rule",
    "provenance_contract",
    "run_privileged_current_forward_loss",
    "strip_gradient_flats",
    "teacher_frozen_report",
    "validate_control_reuse",
    "validate_lambda_priv",
]
