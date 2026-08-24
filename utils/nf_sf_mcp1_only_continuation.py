from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from utils.nf_sf_full_sequence_continuation import (
    ADAMW_BETAS,
    ADAMW_EPS,
    CONTINUATION_OBJECTIVE_MODE,
)
from utils.nf_sf_m3 import file_sha256, tensor_sha256
from utils.nf_sf_mcp_direct_context_ablation import (
    NF_SF_MCP_DIRECT_CLEAN_KV_ABLATION_PARENT_STEP,
    NF_SF_MCP_DIRECT_CLEAN_KV_ABLATION_TARGET_STEP,
    NF_SF_MCP_DIRECT_CLEAN_KV_PARENT_CHECKPOINT_SHA256,
    NF_SF_MCP_DIRECT_CLEAN_KV_PARENT_GIT_SHA,
)
from utils.nf_sf_training import (
    FULL_SEQUENCE_DEPTH_WEIGHTS,
    FULL_SEQUENCE_RNG_DRAW_ORDER_VERSION,
    NFSFFullSequenceNoisyBatch,
    build_full_sequence_mcp_anchor_inputs,
    collect_nf_sf_parameter_groups,
    nf_sf_full_sequence_train_cursor,
    _output_field,
)


NF_SF_MCP1_ONLY_CONTINUATION_SCHEMA = "nf_sf_mcp1_only_continuation_v1"
NF_SF_MCP1_ONLY_CONTINUATION_PARENT_STEP = (
    NF_SF_MCP_DIRECT_CLEAN_KV_ABLATION_PARENT_STEP
)
NF_SF_MCP1_ONLY_CONTINUATION_TARGET_STEP = (
    NF_SF_MCP_DIRECT_CLEAN_KV_ABLATION_TARGET_STEP
)
NF_SF_MCP1_ONLY_CONTINUATION_UPDATE_COUNT = 500
NF_SF_MCP1_ONLY_PARENT_CHECKPOINT_SHA256 = (
    NF_SF_MCP_DIRECT_CLEAN_KV_PARENT_CHECKPOINT_SHA256
)
NF_SF_MCP1_ONLY_PARENT_GIT_SHA = NF_SF_MCP_DIRECT_CLEAN_KV_PARENT_GIT_SHA
DIRECT_CLEAN_KV_CONTROL_SCHEMA = "nf_sf_mcp_direct_clean_kv_ablation_v1"
MCP1_ONLY_TRAINABLE_GROUPS = ("mcp_fusion", "mcp_depth1")
MCP1_ONLY_FROZEN_GROUPS = (
    "backbone",
    "patch_embedding",
    "mcp_depth2",
    "mcp_depth3",
)
MCP1_ONLY_REQUIRED_GROUPS = MCP1_ONLY_FROZEN_GROUPS + MCP1_ONLY_TRAINABLE_GROUPS
MCP1_ONLY_PRIMARY_METRIC = "fixed_probe_raw999_step007000.mcp1_flow_mse_to_exact"
MCP1_ONLY_BASELINE_STEP6500_METRICS = {
    "raw999_mcp1": 0.11986814439296722,
}
SUPPORT_JOINT_TRAINING_INTERFERENCE = "SUPPORT_JOINT_TRAINING_INTERFERENCE"
NO_SUPPORT = "NO_SUPPORT"
INCONCLUSIVE = "INCONCLUSIVE"


@dataclass(frozen=True)
class MCP1OnlyRunPlan:
    schema: str
    parent_step: int
    first_step: int
    target_step: int
    update_count: int
    checkpoint_steps: tuple[int, ...]
    validation_steps: tuple[int, ...]
    diagnostic_only: bool
    non_canonical: bool
    canonical_training_eligible: bool
    deployment_eligible: bool
    objective_mode: str
    loss_scope: str
    trainable_groups: tuple[str, ...]
    frozen_groups: tuple[str, ...]


@dataclass(frozen=True)
class MCP1OnlyParamSelection:
    trainable_named_parameters: tuple[tuple[str, torch.nn.Parameter], ...]
    optimizer_param_groups: tuple[dict[str, Any], ...]
    allowed_param_ids: frozenset[int]
    summary: dict[str, Any]


@dataclass(frozen=True)
class MCP1OnlyForwardResult:
    noisy_batch: NFSFFullSequenceNoisyBatch
    total_loss: torch.Tensor
    mcp1_loss: torch.Tensor
    main_observation_loss: torch.Tensor
    mcp_depth_observation_losses: tuple[torch.Tensor, ...]
    mcp_anchor_observation_losses: tuple[tuple[torch.Tensor, ...], ...]
    main_flow_pred: torch.Tensor
    mcp_flow_preds_by_depth: tuple[torch.Tensor, ...]
    tap_shapes: tuple[tuple[int, ...], ...]
    anchor_token_slices: tuple[tuple[int, int], ...]
    main_backbone_forward_count: int
    future_embedding_order: str | None


def mcp1_only_step_numbers(
    *,
    parent_step: int = NF_SF_MCP1_ONLY_CONTINUATION_PARENT_STEP,
    target_step: int = NF_SF_MCP1_ONLY_CONTINUATION_TARGET_STEP,
) -> tuple[int, ...]:
    parent = int(parent_step)
    target = int(target_step)
    if parent != NF_SF_MCP1_ONLY_CONTINUATION_PARENT_STEP:
        raise RuntimeError("MCP1-only continuation must fork from step6500")
    if target != NF_SF_MCP1_ONLY_CONTINUATION_TARGET_STEP:
        raise RuntimeError("MCP1-only continuation target must be step7000")
    steps = tuple(range(parent + 1, target + 1))
    if len(steps) != NF_SF_MCP1_ONLY_CONTINUATION_UPDATE_COUNT:
        raise RuntimeError("MCP1-only continuation must run exactly 500 updates")
    return steps


def build_mcp1_only_run_plan() -> MCP1OnlyRunPlan:
    steps = mcp1_only_step_numbers()
    return MCP1OnlyRunPlan(
        schema=NF_SF_MCP1_ONLY_CONTINUATION_SCHEMA,
        parent_step=NF_SF_MCP1_ONLY_CONTINUATION_PARENT_STEP,
        first_step=steps[0],
        target_step=steps[-1],
        update_count=len(steps),
        checkpoint_steps=(steps[-1],),
        validation_steps=(steps[-1],),
        diagnostic_only=True,
        non_canonical=True,
        canonical_training_eligible=False,
        deployment_eligible=False,
        objective_mode=CONTINUATION_OBJECTIVE_MODE,
        loss_scope="only_mcp_depth1_exact_flow_matching",
        trainable_groups=MCP1_ONLY_TRAINABLE_GROUPS,
        frozen_groups=MCP1_ONLY_FROZEN_GROUPS,
    )


def mcp1_only_first_step_contract(
    *,
    train_identity: str,
    sample_cursor: Mapping[str, Any],
) -> dict[str, Any]:
    expected_cursor = nf_sf_full_sequence_train_cursor(
        NF_SF_MCP1_ONLY_CONTINUATION_PARENT_STEP + 1
    )
    if dict(sample_cursor) != dict(expected_cursor):
        raise RuntimeError("MCP1-only first continuation sample_cursor mismatch")
    return {
        "status": "PASS",
        "parent_global_step": NF_SF_MCP1_ONLY_CONTINUATION_PARENT_STEP,
        "target_global_step": NF_SF_MCP1_ONLY_CONTINUATION_TARGET_STEP,
        "first_global_step": NF_SF_MCP1_ONLY_CONTINUATION_PARENT_STEP + 1,
        "first_sample_identity": str(train_identity),
        "first_sample_cursor": dict(sample_cursor),
    }


def configure_mcp1_only_trainable_parameters(
    generator: Any,
    *,
    mcp_lr: float,
    weight_decay: float,
) -> MCP1OnlyParamSelection:
    if getattr(generator, "mcp", None) is None:
        raise RuntimeError("MCP1-only continuation requires attached MCP modules")
    groups = collect_nf_sf_parameter_groups(generator)
    missing = sorted(set(MCP1_ONLY_REQUIRED_GROUPS).difference(groups.keys()))
    if missing:
        raise RuntimeError(f"MCP1-only parameter group(s) missing: {missing}")
    if not math.isfinite(float(mcp_lr)) or float(mcp_lr) <= 0.0:
        raise ValueError("mcp_lr must be positive and finite")
    if not math.isfinite(float(weight_decay)) or float(weight_decay) < 0.0:
        raise ValueError("weight_decay must be non-negative and finite")

    alias_audit = _object_id_alias_audit(groups)
    allowed_ids = {
        id(param)
        for group_name in MCP1_ONLY_TRAINABLE_GROUPS
        for _, param in groups[group_name]
    }
    forbidden_ids = {
        id(param)
        for group_name in MCP1_ONLY_FROZEN_GROUPS
        for _, param in groups[group_name]
    }
    if allowed_ids & forbidden_ids:
        raise RuntimeError("MCP1-only allowed parameters alias frozen parameters")

    generator.requires_grad_(False)
    trainable_named: list[tuple[str, torch.nn.Parameter]] = []
    optimizer_groups: list[dict[str, Any]] = []
    group_records: dict[str, Any] = {}
    for group_name, named_params in groups.items():
        allow = group_name in MCP1_ONLY_TRAINABLE_GROUPS
        params = []
        for name, param in named_params:
            param.requires_grad_(allow)
            if allow:
                trainable_named.append((name, param))
                params.append(param)
        if allow:
            if not params:
                raise RuntimeError(f"MCP1-only allowed group {group_name} is empty")
            optimizer_groups.append(
                {
                    "name": group_name,
                    "params": params,
                    "lr": float(mcp_lr),
                    "weight_decay": float(weight_decay),
                }
            )
        group_records[group_name] = {
            "parameter_names": [name for name, _ in named_params],
            "tensor_count": len(named_params),
            "parameter_count": int(sum(param.numel() for _, param in named_params)),
            "trainable_tensor_count": int(
                sum(1 for _, param in named_params if param.requires_grad)
            ),
            "trainable_parameter_count": int(
                sum(param.numel() for _, param in named_params if param.requires_grad)
            ),
            "requires_grad": bool(named_params)
            and all(param.requires_grad for _, param in named_params),
            "in_optimizer": bool(allow),
        }

    optimizer_ids = [
        id(param) for group in optimizer_groups for param in group["params"]
    ]
    if len(set(optimizer_ids)) != len(optimizer_ids):
        raise RuntimeError("MCP1-only optimizer plan contains duplicate parameter ids")
    if set(optimizer_ids) != allowed_ids:
        raise RuntimeError("MCP1-only optimizer planned ids differ from allowed ids")
    assert_mcp1_only_trainable_contract(generator)
    summary = {
        "schema": f"{NF_SF_MCP1_ONLY_CONTINUATION_SCHEMA}_trainable_contract_v1",
        "trainable_groups": list(MCP1_ONLY_TRAINABLE_GROUPS),
        "frozen_groups": list(MCP1_ONLY_FROZEN_GROUPS),
        "trainable_parameter_names": [name for name, _ in trainable_named],
        "trainable_tensor_count": len(trainable_named),
        "trainable_parameter_count": int(
            sum(param.numel() for _, param in trainable_named)
        ),
        "optimizer_group_names": [group["name"] for group in optimizer_groups],
        "object_id_alias_audit": alias_audit,
        "parameter_identity_contract": {
            "allowed_unique_id_count": len(allowed_ids),
            "allowed_named_parameter_count": len(trainable_named),
            "optimizer_unique_id_count": len(set(optimizer_ids)),
            "optimizer_named_parameter_count": len(optimizer_ids),
            "optimizer_ids_equal_allowed_ids": True,
            "allowed_ids_intersect_main_or_patch_embedding": False,
            "allowed_ids_intersect_mcp_depth2_or_depth3": False,
        },
        "group_records": group_records,
    }
    return MCP1OnlyParamSelection(
        trainable_named_parameters=tuple(trainable_named),
        optimizer_param_groups=tuple(optimizer_groups),
        allowed_param_ids=frozenset(allowed_ids),
        summary=summary,
    )


def build_mcp1_only_optimizer_from_canonical_state(
    generator: Any,
    canonical_optimizer: torch.optim.Optimizer,
    *,
    mcp_lr: float,
    weight_decay: float,
    require_existing_state: bool = True,
) -> tuple[torch.optim.AdamW, MCP1OnlyParamSelection, dict[str, Any]]:
    selection = configure_mcp1_only_trainable_parameters(
        generator,
        mcp_lr=float(mcp_lr),
        weight_decay=float(weight_decay),
    )
    optimizer = torch.optim.AdamW(
        list(selection.optimizer_param_groups),
        betas=ADAMW_BETAS,
        eps=ADAMW_EPS,
        weight_decay=float(weight_decay),
    )
    missing_state = []
    transferred_state = []
    for name, param in selection.trainable_named_parameters:
        source_state = canonical_optimizer.state.get(param)
        if not source_state:
            missing_state.append(name)
            continue
        optimizer.state[param] = _clone_optimizer_state(
            source_state,
            device=param.device,
        )
        transferred_state.append(
            {
                "name": name,
                "state_keys": sorted(str(key) for key in source_state.keys()),
            }
        )
    if require_existing_state and missing_state:
        raise RuntimeError(
            "MCP1-only optimizer could not inherit parent AdamW state for: "
            + ", ".join(missing_state)
        )
    optimizer_report = validate_mcp1_only_optimizer_isolation(
        generator,
        optimizer,
        allowed_param_ids=set(selection.allowed_param_ids),
    )
    report = {
        "schema": f"{NF_SF_MCP1_ONLY_CONTINUATION_SCHEMA}_optimizer_state_v1",
        "optimizer_class": optimizer.__class__.__name__,
        "betas": [float(value) for value in optimizer.defaults["betas"]],
        "eps": float(optimizer.defaults["eps"]),
        "weight_decay": float(weight_decay),
        "group_lrs": {
            str(group["name"]): float(group["lr"]) for group in optimizer.param_groups
        },
        "inherited_from_canonical_parent_optimizer": True,
        "require_existing_state": bool(require_existing_state),
        "transferred_state_entry_count": len(transferred_state),
        "missing_state_parameter_names": missing_state,
        "transferred_state": transferred_state,
        "optimizer_isolation": optimizer_report,
    }
    return optimizer, selection, report


def validate_mcp1_only_optimizer_isolation(
    generator: Any,
    optimizer: torch.optim.Optimizer,
    *,
    allowed_param_ids: set[int] | frozenset[int],
) -> dict[str, Any]:
    groups = collect_nf_sf_parameter_groups(generator)
    frozen_ids = {
        id(param)
        for group_name in MCP1_ONLY_FROZEN_GROUPS
        for _, param in groups[group_name]
    }
    optimizer_ids_list = [
        id(param)
        for group in optimizer.param_groups
        for param in group.get("params", ())
    ]
    if len(set(optimizer_ids_list)) != len(optimizer_ids_list):
        raise RuntimeError("MCP1-only optimizer contains duplicate parameter ids")
    optimizer_ids = set(optimizer_ids_list)
    if optimizer_ids != set(allowed_param_ids):
        raise RuntimeError("MCP1-only optimizer ids differ from allowed ids")
    if optimizer_ids & frozen_ids:
        raise RuntimeError("MCP1-only optimizer includes frozen parameters")
    group_names = tuple(str(group.get("name")) for group in optimizer.param_groups)
    if group_names != MCP1_ONLY_TRAINABLE_GROUPS:
        raise RuntimeError("MCP1-only optimizer group names/order mismatch")
    return {
        "status": "PASS",
        "optimizer_group_names": list(group_names),
        "optimizer_param_id_count": len(optimizer_ids),
        "allowed_param_id_count": len(set(allowed_param_ids)),
        "optimizer_ids_equal_allowed_ids": True,
        "optimizer_ids_intersect_frozen_ids": False,
    }


def assert_mcp1_only_trainable_contract(generator: Any) -> None:
    groups = collect_nf_sf_parameter_groups(generator)
    for group_name in MCP1_ONLY_TRAINABLE_GROUPS:
        params = groups[group_name]
        if not params:
            raise RuntimeError(f"MCP1-only trainable group {group_name} is empty")
        for name, param in params:
            if not param.requires_grad:
                raise RuntimeError(f"MCP1-only expected trainable parameter frozen: {name}")
    for group_name in MCP1_ONLY_FROZEN_GROUPS:
        for name, param in groups[group_name]:
            if param.requires_grad:
                raise RuntimeError(f"MCP1-only expected frozen parameter trainable: {name}")


def assert_no_forbidden_mcp1_only_gradients(generator: Any) -> None:
    groups = collect_nf_sf_parameter_groups(generator)
    for group_name in MCP1_ONLY_FROZEN_GROUPS:
        for name, param in groups[group_name]:
            if param.grad is None:
                continue
            grad = param.grad.detach()
            if bool(torch.isfinite(grad).all().item()) and float(
                grad.float().abs().max().item()
            ) == 0.0:
                continue
            raise RuntimeError(f"MCP1-only forbidden gradient present on {name}")


def audit_mcp1_only_gradients(generator: Any) -> dict[str, Any]:
    assert_no_forbidden_mcp1_only_gradients(generator)
    groups = collect_nf_sf_parameter_groups(generator)
    records = []
    nonzero = 0
    for group_name in MCP1_ONLY_TRAINABLE_GROUPS:
        for name, param in groups[group_name]:
            grad = param.grad
            if grad is None:
                raise RuntimeError(f"MCP1-only missing gradient on {name}")
            detached = grad.detach()
            if not bool(torch.isfinite(detached).all().item()):
                raise RuntimeError(f"MCP1-only non-finite gradient on {name}")
            norm = float(detached.float().square().sum().item()) ** 0.5
            max_abs = float(detached.float().abs().max().item())
            if max_abs > 0.0:
                nonzero += 1
            records.append(
                {
                    "group": group_name,
                    "name": name,
                    "finite": True,
                    "l2": norm,
                    "max_abs": max_abs,
                }
            )
    if nonzero <= 0:
        raise RuntimeError("MCP1-only trainable parameters received no nonzero gradients")
    return {
        "schema": f"{NF_SF_MCP1_ONLY_CONTINUATION_SCHEMA}_gradient_audit_v1",
        "trainable_groups": list(MCP1_ONLY_TRAINABLE_GROUPS),
        "frozen_groups": list(MCP1_ONLY_FROZEN_GROUPS),
        "trainable_gradient_records": records,
        "trainable_nonzero_gradient_tensor_count": int(nonzero),
        "forbidden_gradients_absent_or_zero": True,
    }


def has_nonfinite_trainable_grad(optimizer: torch.optim.Optimizer) -> bool:
    for group in optimizer.param_groups:
        for param in group.get("params", ()):
            if param.grad is None:
                continue
            if not bool(torch.isfinite(param.grad.detach().float()).all().item()):
                return True
    return False


def run_mcp1_only_forward_loss(
    generator: Any,
    *,
    conditional_dict: Mapping[str, Any],
    noisy_batch: NFSFFullSequenceNoisyBatch,
) -> MCP1OnlyForwardResult:
    mcp_anchor_inputs = build_full_sequence_mcp_anchor_inputs(noisy_batch)
    outputs = generator.forward_full_sequence_next_forcing(
        noisy_image_or_video=noisy_batch.noisy_main,
        clean_x=noisy_batch.clean_target,
        conditional_dict=dict(conditional_dict),
        timestep_main=noisy_batch.timestep_main,
        mcp_anchor_inputs=mcp_anchor_inputs,
    )
    main_flow_pred = _output_field(outputs, "main_flow_pred")
    mcp_flow_preds_by_depth = tuple(_output_field(outputs, "mcp_flow_preds_by_depth"))
    (
        mcp1_loss,
        main_observation_loss,
        mcp_depth_observation_losses,
        mcp_anchor_observation_losses,
    ) = compute_mcp1_only_losses(
        main_flow_pred=main_flow_pred,
        mcp_flow_preds_by_depth=mcp_flow_preds_by_depth,
        noisy_batch=noisy_batch,
    )
    return MCP1OnlyForwardResult(
        noisy_batch=noisy_batch,
        total_loss=mcp1_loss,
        mcp1_loss=mcp1_loss,
        main_observation_loss=main_observation_loss,
        mcp_depth_observation_losses=mcp_depth_observation_losses,
        mcp_anchor_observation_losses=mcp_anchor_observation_losses,
        main_flow_pred=main_flow_pred,
        mcp_flow_preds_by_depth=mcp_flow_preds_by_depth,
        tap_shapes=tuple(_output_field(outputs, "tap_shapes")),
        anchor_token_slices=tuple(_output_field(outputs, "anchor_token_slices")),
        main_backbone_forward_count=int(_output_field(outputs, "main_backbone_forward_count")),
        future_embedding_order=_output_field(outputs, "future_embedding_order"),
    )


def compute_mcp1_only_losses(
    *,
    main_flow_pred: torch.Tensor,
    mcp_flow_preds_by_depth: tuple[torch.Tensor, ...],
    noisy_batch: NFSFFullSequenceNoisyBatch,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    tuple[torch.Tensor, ...],
    tuple[tuple[torch.Tensor, ...], ...],
]:
    if len(mcp_flow_preds_by_depth) != 3:
        raise RuntimeError("MCP1-only training expects canonical MCP1/2/3 outputs")
    if len(noisy_batch.target_flow_mcp_depths) != 3:
        raise RuntimeError("MCP1-only training expects canonical MCP1/2/3 targets")
    if tuple(mcp_flow_preds_by_depth[0].shape) != tuple(
        noisy_batch.target_flow_mcp_depths[0].shape
    ):
        raise ValueError("MCP1 prediction shape mismatch")
    mcp1_loss = F.mse_loss(
        mcp_flow_preds_by_depth[0].float(),
        noisy_batch.target_flow_mcp_depths[0].float(),
        reduction="mean",
    )
    main_observation = _detached_mse(main_flow_pred, noisy_batch.target_flow_main)
    depth_observations = tuple(
        _detached_mse(pred, target)
        for pred, target in zip(
            mcp_flow_preds_by_depth,
            noisy_batch.target_flow_mcp_depths,
        )
    )
    anchor_observations = []
    for pred, target in zip(
        mcp_flow_preds_by_depth,
        noisy_batch.target_flow_mcp_depths,
    ):
        if tuple(pred.shape) != tuple(target.shape):
            raise ValueError("MCP observation shape mismatch")
        anchor_observations.append(
            tuple(
                _detached_mse(pred[:, anchor_index], target[:, anchor_index])
                for anchor_index in range(int(pred.shape[1]))
            )
        )
    return (
        mcp1_loss,
        main_observation,
        depth_observations,
        tuple(anchor_observations),
    )


def mcp1_only_loss_metrics(result: MCP1OnlyForwardResult) -> dict[str, Any]:
    return {
        "total_loss": float(result.total_loss.detach().float().item()),
        "mcp1_loss": float(result.mcp1_loss.detach().float().item()),
        "main_loss_observation_only": float(
            result.main_observation_loss.detach().float().item()
        ),
        "mcp_depth_losses_observation_only": [
            float(loss.detach().float().item())
            for loss in result.mcp_depth_observation_losses
        ],
        "mcp_anchor_losses_observation_only": [
            [float(loss.detach().float().item()) for loss in anchor_losses]
            for anchor_losses in result.mcp_anchor_observation_losses
        ],
        "canonical_depth_weights_not_used_for_backward": list(FULL_SEQUENCE_DEPTH_WEIGHTS),
        "loss_scope": "mcp_depth1_exact_flow_matching_only",
    }


def parameter_sha256_report(
    generator: Any,
    *,
    groups: Sequence[str],
) -> dict[str, Any]:
    all_groups = collect_nf_sf_parameter_groups(generator)
    records: dict[str, dict[str, Any]] = {}
    for group_name in groups:
        if group_name not in all_groups:
            raise RuntimeError(f"parameter group {group_name!r} missing")
        group_records = {}
        for name, param in all_groups[group_name]:
            tensor = param.detach().cpu()
            group_records[name] = {
                "sha256": tensor_sha256(tensor),
                "shape": [int(dim) for dim in tensor.shape],
                "dtype": str(tensor.dtype),
                "requires_grad": bool(param.requires_grad),
            }
        records[group_name] = group_records
    payload = {
        "groups": list(groups),
        "parameter_count": int(sum(len(records[name]) for name in records)),
        "parameters": records,
    }
    return {
        **payload,
        "fingerprint_sha256": _json_sha256(payload),
    }


def compare_parameter_sha256_reports(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
) -> dict[str, Any]:
    mismatches = []
    for group_name, group_records in before.get("parameters", {}).items():
        after_group = after.get("parameters", {}).get(group_name, {})
        for name, record in group_records.items():
            after_record = after_group.get(name)
            if not isinstance(after_record, Mapping):
                mismatches.append(name)
            elif str(record.get("sha256")) != str(after_record.get("sha256")):
                mismatches.append(name)
    return {
        "checked_groups": list(before.get("groups", ())),
        "parameter_count": int(before.get("parameter_count", 0)),
        "before_fingerprint_sha256": str(before.get("fingerprint_sha256")),
        "after_fingerprint_sha256": str(after.get("fingerprint_sha256")),
        "all_sha256_exact_match": len(mismatches) == 0,
        "mismatch_parameter_names": mismatches,
    }


def trainable_parameter_delta_report(
    named_parameters: Sequence[tuple[str, torch.nn.Parameter]],
    before: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    total_sq = 0.0
    by_parameter = {}
    nonzero = 0
    for name, param in named_parameters:
        if name not in before:
            raise RuntimeError(f"missing trainable snapshot for {name}")
        delta = param.detach().cpu().float() - before[name].float()
        norm = float(delta.square().sum().item()) ** 0.5
        if norm > 0.0:
            nonzero += 1
        total_sq += norm * norm
        by_parameter[name] = norm
    return {
        "aggregate_l2": float(total_sq ** 0.5),
        "nonzero_parameter_delta_count": int(nonzero),
        "by_parameter_l2": by_parameter,
        "parameter_count": len(by_parameter),
    }


def trainable_parameter_snapshot(
    named_parameters: Sequence[tuple[str, torch.nn.Parameter]],
) -> dict[str, torch.Tensor]:
    return {
        name: param.detach().cpu().clone()
        for name, param in named_parameters
    }


def build_mcp1_only_provenance(
    *,
    runtime_git_sha: str,
    semantic_lock_fingerprint: str,
    parent_checkpoint_sha256: str = NF_SF_MCP1_ONLY_PARENT_CHECKPOINT_SHA256,
    parent_git_sha: str = NF_SF_MCP1_ONLY_PARENT_GIT_SHA,
) -> dict[str, Any]:
    return {
        "schema": NF_SF_MCP1_ONLY_CONTINUATION_SCHEMA,
        "diagnostic_only": True,
        "non_canonical": True,
        "canonical_training_eligible": False,
        "deployment_eligible": False,
        "canonical_deployment_eligible": False,
        "parent_global_step": NF_SF_MCP1_ONLY_CONTINUATION_PARENT_STEP,
        "target_global_step": NF_SF_MCP1_ONLY_CONTINUATION_TARGET_STEP,
        "update_count": NF_SF_MCP1_ONLY_CONTINUATION_UPDATE_COUNT,
        "runtime_git_sha": str(runtime_git_sha),
        "parent_checkpoint_sha256": str(parent_checkpoint_sha256),
        "parent_checkpoint_git_sha": str(parent_git_sha),
        "semantic_lock_fingerprint": str(semantic_lock_fingerprint),
        "objective_mode": CONTINUATION_OBJECTIVE_MODE,
        "canonical_full_data_pipeline_reused": True,
        "dataset_changed": False,
        "sample_stream_changed": False,
        "loader_semantics_changed": False,
        "rng_semantics_changed": False,
        "timestep_distribution_changed": False,
        "noise_distribution_changed": False,
        "teacher_target_construction_changed": False,
        "mcp1_fm_target_changed": False,
        "precision_changed": False,
        "optimizer_family_changed": False,
        "optimizer_hparams_changed": False,
        "training_mode_forward_semantics_changed": False,
        "trainable_scope_changed": True,
        "loss_scope_changed": True,
        "trainable_groups": list(MCP1_ONLY_TRAINABLE_GROUPS),
        "frozen_groups": list(MCP1_ONLY_FROZEN_GROUPS),
        "loss": "MCP1 exact Flow Matching MSE only",
        "forbidden_features": forbidden_feature_contract(),
        "primary_metric": MCP1_ONLY_PRIMARY_METRIC,
        "rng_draw_order_version": FULL_SEQUENCE_RNG_DRAW_ORDER_VERSION,
    }


def forbidden_feature_contract() -> dict[str, bool]:
    return {
        "main_loss_backward": False,
        "mcp2_loss_backward": False,
        "mcp3_loss_backward": False,
        "x0_loss": False,
        "dmd": False,
        "teacher_flow_distillation": False,
        "on_policy_self_rollout": False,
        "verifier": False,
        "architecture_changes": False,
        "clean_history_treatment": False,
        "new_timestep_distribution": False,
    }


def validate_mcp1_only_manifest(manifest: Mapping[str, Any]) -> None:
    if manifest.get("schema") != NF_SF_MCP1_ONLY_CONTINUATION_SCHEMA:
        raise RuntimeError("MCP1-only manifest schema mismatch")
    for key in ("diagnostic_only", "non_canonical"):
        if manifest.get(key) is not True:
            raise RuntimeError(f"MCP1-only manifest must mark {key}=True")
    if manifest.get("canonical_training_eligible") is not False:
        raise RuntimeError("MCP1-only manifest must not be canonical-training eligible")
    if manifest.get("deployment_eligible") is not False:
        raise RuntimeError("MCP1-only manifest must not be deployment eligible")
    if manifest.get("train_record_count") not in (
        1,
        NF_SF_MCP1_ONLY_CONTINUATION_UPDATE_COUNT,
    ):
        raise RuntimeError("MCP1-only manifest train_record_count mismatch")
    scope = manifest.get("optimization_scope")
    if not isinstance(scope, Mapping):
        raise RuntimeError("MCP1-only optimization_scope missing")
    if tuple(scope.get("trainable_groups", ())) != MCP1_ONLY_TRAINABLE_GROUPS:
        raise RuntimeError("MCP1-only trainable scope mismatch")
    if tuple(scope.get("frozen_groups", ())) != MCP1_ONLY_FROZEN_GROUPS:
        raise RuntimeError("MCP1-only frozen scope mismatch")
    forbidden = manifest.get("forbidden_features")
    if not isinstance(forbidden, Mapping):
        raise RuntimeError("MCP1-only forbidden feature contract missing")
    if any(bool(value) for value in forbidden.values()):
        raise RuntimeError("MCP1-only forbidden feature enabled")


def load_matching_control_artifact_bundle(
    summary_path: Path | str,
    *,
    expected_control_runtime_git_sha: str | None = None,
) -> dict[str, Any]:
    summary_path = Path(summary_path)
    paths = {
        "control_summary_path": summary_path,
        "run_metadata_path": summary_path.parent / "run_metadata.json",
        "metrics_path": summary_path.parent / "metrics.jsonl",
        "checkpoint_validation_path": (
            summary_path.parent / "checkpoint_step007000.validation.json"
        ),
    }
    failures: list[str] = []
    source_sha256: dict[str, str | None] = {}
    summary = _load_json_companion(
        paths["control_summary_path"],
        label="control summary",
        failures=failures,
        source_sha256=source_sha256,
    )
    schema = summary.get("schema") if isinstance(summary, Mapping) else None
    if schema == DIRECT_CLEAN_KV_CONTROL_SCHEMA:
        run_metadata = _load_json_companion(
            paths["run_metadata_path"],
            label="run_metadata.json",
            failures=failures,
            source_sha256=source_sha256,
        )
        metrics_records = _load_jsonl_companion(
            paths["metrics_path"],
            failures=failures,
            source_sha256=source_sha256,
        )
        checkpoint_validation = _load_json_companion(
            paths["checkpoint_validation_path"],
            label="checkpoint validation",
            failures=failures,
            source_sha256=source_sha256,
        )
        audit = validate_matching_control_provenance(
            summary if isinstance(summary, Mapping) else {},
            run_metadata=run_metadata if isinstance(run_metadata, Mapping) else None,
            metrics_records=metrics_records,
            checkpoint_validation=(
                checkpoint_validation
                if isinstance(checkpoint_validation, Mapping)
                else None
            ),
            control_summary_path=paths["control_summary_path"],
            run_metadata_path=paths["run_metadata_path"],
            metrics_path=paths["metrics_path"],
            checkpoint_validation_path=paths["checkpoint_validation_path"],
            source_sha256=source_sha256,
            expected_control_runtime_git_sha=expected_control_runtime_git_sha,
        )
        if failures:
            audit["failures"] = [*audit["failures"], *failures]
            audit["CONTROL_REUSABLE"] = False
        return audit

    audit = validate_matching_control_provenance(
        summary if isinstance(summary, Mapping) else {},
        control_summary_path=paths["control_summary_path"],
        source_sha256=source_sha256,
        expected_control_runtime_git_sha=expected_control_runtime_git_sha,
    )
    if failures:
        audit["failures"] = [*audit["failures"], *failures]
        audit["CONTROL_REUSABLE"] = False
    return audit


def validate_matching_control_provenance(
    control: Mapping[str, Any],
    *,
    run_metadata: Mapping[str, Any] | None = None,
    metrics_records: Sequence[Mapping[str, Any]] | None = None,
    checkpoint_validation: Mapping[str, Any] | None = None,
    control_summary_path: Path | str | None = None,
    run_metadata_path: Path | str | None = None,
    metrics_path: Path | str | None = None,
    checkpoint_validation_path: Path | str | None = None,
    source_sha256: Mapping[str, str | None] | None = None,
    expected_control_runtime_git_sha: str | None = None,
) -> dict[str, Any]:
    failures: list[str] = []
    schema = str(control.get("schema", ""))
    if schema not in (DIRECT_CLEAN_KV_CONTROL_SCHEMA, NF_SF_MCP1_ONLY_CONTINUATION_SCHEMA):
        failures.append("schema is not a recognized step6500->7000 control")
    if schema == DIRECT_CLEAN_KV_CONTROL_SCHEMA:
        if control.get("status") != "PASS":
            failures.append("direct-clean-KV control status is not PASS")
        if control.get("arm") != "control":
            failures.append("direct-clean-KV control arm is not control")
    else:
        if control.get("status") not in ("PASS", "DONE"):
            failures.append("control status is not PASS/DONE")
        if control.get("arm", "control") != "control":
            failures.append("control arm is not control")
    _check_equal(
        control,
        failures,
        "parent_step",
        NF_SF_MCP1_ONLY_CONTINUATION_PARENT_STEP,
    )
    _check_equal(
        control,
        failures,
        "target_step",
        NF_SF_MCP1_ONLY_CONTINUATION_TARGET_STEP,
    )
    _check_equal(
        control,
        failures,
        "train_record_count",
        NF_SF_MCP1_ONLY_CONTINUATION_UPDATE_COUNT,
    )
    _validate_control_summary_validation_and_probe(control, failures=failures)

    if schema == DIRECT_CLEAN_KV_CONTROL_SCHEMA:
        if run_metadata is None:
            failures.append("direct-clean-KV control run_metadata.json missing")
        else:
            _validate_direct_clean_control_run_metadata(
                run_metadata,
                failures=failures,
                expected_control_runtime_git_sha=expected_control_runtime_git_sha,
            )
        if metrics_records is None:
            failures.append("direct-clean-KV control metrics.jsonl missing")
        else:
            _validate_direct_clean_control_metrics(metrics_records, failures=failures)
        if checkpoint_validation is None:
            failures.append("direct-clean-KV checkpoint validation missing")
        else:
            _validate_direct_clean_control_checkpoint_validation(
                control,
                checkpoint_validation,
                failures=failures,
            )
    else:
        _validate_native_control_embedded_provenance(control, failures=failures)

    return {
        "schema": f"{NF_SF_MCP1_ONLY_CONTINUATION_SCHEMA}_control_reuse_audit_v1",
        "CONTROL_REUSABLE": len(failures) == 0,
        "control_summary_path": _path_string(control_summary_path),
        "run_metadata_path": _path_string(run_metadata_path),
        "metrics_path": _path_string(metrics_path),
        "checkpoint_validation_path": _path_string(checkpoint_validation_path),
        "source_sha256": dict(source_sha256 or {}),
        "failures": failures,
        "checked_items": [
            "summary schema/status/arm/step/train count",
            "parent checkpoint SHA",
            "parent global step",
            "parent git",
            "run plan 6500->7000",
            "500 optimizer steps",
            "metrics sample-stream proof",
            "restore RNG fingerprint proof",
            "validation identity/noise contract",
            "fixed raw999 MCP1 probe",
            "checkpoint validation sidecar",
        ],
    }


def _validate_control_summary_validation_and_probe(
    control: Mapping[str, Any],
    *,
    failures: list[str],
) -> None:
    validation = control.get("validation")
    if isinstance(validation, Mapping):
        if _int_or_none(validation.get("global_step")) != 7000:
            failures.append("validation global_step mismatch")
        if validation.get("paired_identity_noise_across_steps") is not True:
            failures.append("validation paired identity noise proof missing")
    else:
        failures.append("validation proof missing")
    fixed_probe = control.get("fixed_probe")
    if isinstance(fixed_probe, Mapping):
        if _int_or_none(fixed_probe.get("raw_timestep")) != 999:
            failures.append("fixed raw999 probe missing")
        if _int_or_none(fixed_probe.get("depth")) != 1:
            failures.append("fixed MCP1 probe missing")
    else:
        failures.append("fixed raw999 MCP1 probe missing")


def _validate_direct_clean_control_run_metadata(
    run_metadata: Mapping[str, Any],
    *,
    failures: list[str],
    expected_control_runtime_git_sha: str | None,
) -> None:
    if run_metadata.get("schema") != DIRECT_CLEAN_KV_CONTROL_SCHEMA:
        failures.append("run_metadata schema mismatch")
    provenance = run_metadata.get("provenance")
    if not isinstance(provenance, Mapping):
        failures.append("run_metadata provenance missing")
    else:
        _check_equal(
            provenance,
            failures,
            "parent_checkpoint_sha256",
            NF_SF_MCP1_ONLY_PARENT_CHECKPOINT_SHA256,
        )
        _check_equal(
            provenance,
            failures,
            "parent_git_sha",
            NF_SF_MCP1_ONLY_PARENT_GIT_SHA,
        )
        _check_equal(
            provenance,
            failures,
            "parent_step",
            NF_SF_MCP1_ONLY_CONTINUATION_PARENT_STEP,
        )
        _check_equal(
            provenance,
            failures,
            "target_step",
            NF_SF_MCP1_ONLY_CONTINUATION_TARGET_STEP,
        )
        _check_equal(
            provenance,
            failures,
            "update_count",
            NF_SF_MCP1_ONLY_CONTINUATION_UPDATE_COUNT,
        )
        _check_equal(provenance, failures, "arm", "control")
        _check_equal(provenance, failures, "direct_clean_context_kv", False)
        for key in ("data_changed", "rng_changed", "objective_changed", "optimizer_changed"):
            _check_equal(provenance, failures, key, False)
        runtime_git = str(provenance.get("runtime_git_sha", ""))
        if not _is_git_sha(runtime_git):
            failures.append("control runtime_git_sha missing or invalid")
        elif expected_control_runtime_git_sha is not None:
            expected = str(expected_control_runtime_git_sha)
            if not _is_git_sha(expected) or runtime_git != expected:
                failures.append("control runtime_git_sha mismatch")

    restore = run_metadata.get("restore_contract")
    if not isinstance(restore, Mapping):
        failures.append("run_metadata restore_contract missing")
    else:
        if restore.get("status") != "PASS":
            failures.append("restore_contract status mismatch")
        rng = restore.get("rng_fingerprint")
        if not isinstance(rng, Mapping):
            failures.append("restore_contract rng_fingerprint missing")
        else:
            for key in (
                "train_rng_state_sha256",
                "validation_base_rng_state_sha256",
                "python_random_state_sha256",
                "torch_cpu_global_rng_state_sha256",
                "torch_cuda_global_rng_state_sha256",
            ):
                if not _is_sha256(str(rng.get(key, ""))):
                    failures.append(f"restore_contract missing valid {key}")

    run_plan = run_metadata.get("run_plan")
    if not isinstance(run_plan, Mapping):
        failures.append("run_metadata run_plan missing")
    else:
        _check_equal(
            run_plan,
            failures,
            "first_step",
            NF_SF_MCP1_ONLY_CONTINUATION_PARENT_STEP + 1,
        )
        _check_equal(
            run_plan,
            failures,
            "target_step",
            NF_SF_MCP1_ONLY_CONTINUATION_TARGET_STEP,
        )
        _check_equal(
            run_plan,
            failures,
            "update_count",
            NF_SF_MCP1_ONLY_CONTINUATION_UPDATE_COUNT,
        )
        _check_equal(run_plan, failures, "arm", "control")
        _check_equal(run_plan, failures, "direct_clean_context_kv", False)


def _validate_direct_clean_control_metrics(
    records: Sequence[Mapping[str, Any]],
    *,
    failures: list[str],
) -> None:
    train_records = [
        record
        for record in records
        if record.get("schema") == DIRECT_CLEAN_KV_CONTROL_SCHEMA
        and record.get("arm") == "control"
        and _int_or_none(record.get("global_step")) is not None
        and isinstance(record.get("sample_cursor"), Mapping)
        and "train_rng_before_sha256" in record
        and "train_rng_after_sha256" in record
    ]
    expected_steps = list(
        range(
            NF_SF_MCP1_ONLY_CONTINUATION_PARENT_STEP + 1,
            NF_SF_MCP1_ONLY_CONTINUATION_TARGET_STEP + 1,
        )
    )
    actual_steps = [_int_or_none(record.get("global_step")) for record in train_records]
    if len(train_records) != NF_SF_MCP1_ONLY_CONTINUATION_UPDATE_COUNT:
        failures.append("metrics training record count mismatch")
    if actual_steps != expected_steps:
        failures.append("metrics training global_step sequence mismatch")
    if not train_records:
        return
    first = train_records[0]
    last = train_records[-1]
    if _int_or_none(first.get("global_step")) != expected_steps[0]:
        failures.append("metrics first training step mismatch")
    if _int_or_none(last.get("global_step")) != expected_steps[-1]:
        failures.append("metrics last training step mismatch")
    if dict(first.get("sample_cursor", {})) != nf_sf_full_sequence_train_cursor(6501):
        failures.append("metrics first sample_cursor mismatch")
    if dict(last.get("sample_cursor", {})) != nf_sf_full_sequence_train_cursor(7000):
        failures.append("metrics last sample_cursor mismatch")
    for record in train_records:
        identity = record.get("sample_identity") or record.get("identity")
        if not isinstance(identity, str) or not identity:
            failures.append("metrics training sample identity missing")
            break
        for key in ("train_rng_before_sha256", "train_rng_after_sha256"):
            if not _is_sha256(str(record.get(key, ""))):
                failures.append(f"metrics training {key} missing or invalid")
                return


def _validate_direct_clean_control_checkpoint_validation(
    control: Mapping[str, Any],
    checkpoint_validation: Mapping[str, Any],
    *,
    failures: list[str],
) -> None:
    if checkpoint_validation.get("status") != "PASS":
        failures.append("checkpoint validation status mismatch")
    if _int_or_none(checkpoint_validation.get("global_step")) != 7000:
        failures.append("checkpoint validation global_step mismatch")
    summary_sha = str(control.get("checkpoint_sha256", ""))
    sidecar_sha = str(checkpoint_validation.get("sha256", ""))
    if not _is_sha256(summary_sha):
        failures.append("summary checkpoint_sha256 missing or invalid")
    if not _is_sha256(sidecar_sha):
        failures.append("checkpoint validation sha256 missing or invalid")
    if _is_sha256(summary_sha) and _is_sha256(sidecar_sha) and summary_sha != sidecar_sha:
        failures.append("checkpoint validation SHA mismatch")


def _validate_native_control_embedded_provenance(
    control: Mapping[str, Any],
    *,
    failures: list[str],
) -> None:
    parent_sha = _nested_get(
        control,
        ("parent_checkpoint", "sha256"),
        ("provenance", "parent_checkpoint_sha256"),
        ("metadata", "provenance", "parent_checkpoint_sha256"),
    )
    if parent_sha != NF_SF_MCP1_ONLY_PARENT_CHECKPOINT_SHA256:
        failures.append("parent checkpoint SHA mismatch or missing")
    parent_git = _nested_get(
        control,
        ("parent_checkpoint", "git_sha"),
        ("provenance", "parent_git_sha"),
        ("provenance", "parent_checkpoint_git_sha"),
        ("metadata", "provenance", "parent_checkpoint_git_sha"),
    )
    if parent_git != NF_SF_MCP1_ONLY_PARENT_GIT_SHA:
        failures.append("parent checkpoint git mismatch or missing")
    run_plan = control.get("run_plan")
    if isinstance(run_plan, Mapping):
        if run_plan.get("first_step") != NF_SF_MCP1_ONLY_CONTINUATION_PARENT_STEP + 1:
            failures.append("control first_step mismatch")
        if run_plan.get("update_count") != NF_SF_MCP1_ONLY_CONTINUATION_UPDATE_COUNT:
            failures.append("control update_count mismatch")
    else:
        failures.append("control run_plan missing")
    first = control.get("first_continuation_step") or control.get("first_step_contract")
    if isinstance(first, Mapping):
        if first.get("first_sample_cursor") != nf_sf_full_sequence_train_cursor(6501):
            failures.append("first sample cursor mismatch")
    else:
        failures.append("first-step/sample-stream proof missing")


def classify_mcp1_only_comparison(
    *,
    baseline_step6500_mse: float,
    control_step7000_mse: float,
    treatment_step7000_mse: float,
    control_paired_mcp1_mse: float | None = None,
    treatment_paired_mcp1_mse: float | None = None,
) -> dict[str, Any]:
    baseline = _finite_nonnegative(baseline_step6500_mse, name="baseline_step6500_mse")
    control = _finite_nonnegative(control_step7000_mse, name="control_step7000_mse")
    treatment = _finite_nonnegative(treatment_step7000_mse, name="treatment_step7000_mse")
    primary_extra = _relative_improvement(control, treatment)
    control_improvement = _relative_improvement(baseline, control)
    treatment_improvement = _relative_improvement(baseline, treatment)
    paired_extra = None
    paired_same_direction = False
    paired_clear = False
    if control_paired_mcp1_mse is not None and treatment_paired_mcp1_mse is not None:
        control_paired = _finite_nonnegative(
            control_paired_mcp1_mse,
            name="control_paired_mcp1_mse",
        )
        treatment_paired = _finite_nonnegative(
            treatment_paired_mcp1_mse,
            name="treatment_paired_mcp1_mse",
        )
        paired_extra = _relative_improvement(control_paired, treatment_paired)
        paired_same_direction = paired_extra > 0.0
        paired_clear = paired_extra >= 0.05
    if primary_extra >= 0.10 and paired_same_direction:
        decision = SUPPORT_JOINT_TRAINING_INTERFERENCE
    elif primary_extra < 0.05 and not paired_clear:
        decision = NO_SUPPORT
    else:
        decision = INCONCLUSIVE
    return {
        "schema": f"{NF_SF_MCP1_ONLY_CONTINUATION_SCHEMA}_decision_v1",
        "decision": decision,
        "baseline_step6500_mse": baseline,
        "control_step7000_mse": control,
        "treatment_step7000_mse": treatment,
        "control_improvement_vs_baseline": control_improvement,
        "treatment_improvement_vs_baseline": treatment_improvement,
        "treatment_additional_improvement_vs_control": primary_extra,
        "paired_mcp1_additional_improvement_vs_control": paired_extra,
        "thresholds": {
            "support_primary_relative_min": 0.10,
            "no_support_primary_relative_max_exclusive": 0.05,
            "inconclusive_band": [0.05, 0.10],
            "support_requires_paired_same_direction": True,
        },
    }


def _load_json_companion(
    path: Path,
    *,
    label: str,
    failures: list[str],
    source_sha256: dict[str, str | None],
) -> Mapping[str, Any] | None:
    key = str(path.resolve())
    if not path.is_file():
        source_sha256[key] = None
        failures.append(f"{label} companion artifact missing")
        return None
    source_sha256[key] = file_sha256(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        failures.append(f"{label} JSON parse failed: {exc}")
        return None
    if not isinstance(payload, Mapping):
        failures.append(f"{label} must contain a JSON object")
        return None
    return payload


def _load_jsonl_companion(
    path: Path,
    *,
    failures: list[str],
    source_sha256: dict[str, str | None],
) -> tuple[Mapping[str, Any], ...] | None:
    key = str(path.resolve())
    if not path.is_file():
        source_sha256[key] = None
        failures.append("metrics.jsonl companion artifact missing")
        return None
    source_sha256[key] = file_sha256(path)
    records = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception as exc:
        failures.append(f"metrics.jsonl read failed: {exc}")
        return None
    for line_index, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except Exception as exc:
            failures.append(f"metrics.jsonl line {line_index} parse failed: {exc}")
            continue
        if not isinstance(payload, Mapping):
            failures.append(f"metrics.jsonl line {line_index} is not an object")
            continue
        records.append(payload)
    return tuple(records)


def _clone_optimizer_state(
    source_state: Mapping[str, Any],
    *,
    device: torch.device,
) -> dict[str, Any]:
    cloned = {}
    for key, value in source_state.items():
        if torch.is_tensor(value):
            cloned[key] = value.detach().clone().to(device=device)
        else:
            cloned[key] = copy.deepcopy(value)
    return cloned


def _detached_mse(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(
        left.detach().float(),
        right.detach().float(),
        reduction="mean",
    )


def _object_id_alias_audit(
    groups: Mapping[str, Sequence[tuple[str, torch.nn.Parameter]]],
) -> dict[str, Any]:
    by_id: dict[int, list[dict[str, Any]]] = {}
    for group_name, named_params in groups.items():
        for name, param in named_params:
            by_id.setdefault(id(param), []).append(
                {"group": group_name, "name": name}
            )
    aliases = [records for records in by_id.values() if len(records) > 1]
    return {
        "group_count": len(groups),
        "unique_object_id_count": len(by_id),
        "alias_count": len(aliases),
        "aliases": aliases,
    }


def _json_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _check_equal(
    record: Mapping[str, Any],
    failures: list[str],
    key: str,
    expected: Any,
) -> None:
    if record.get(key) != expected:
        failures.append(f"{key} mismatch")


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _is_sha256(value: str) -> bool:
    text = str(value)
    return len(text) == 64 and all(char in "0123456789abcdef" for char in text)


def _is_git_sha(value: str) -> bool:
    text = str(value)
    return len(text) == 40 and all(char in "0123456789abcdef" for char in text)


def _path_string(value: Path | str | None) -> str | None:
    if value is None:
        return None
    return str(Path(value).resolve())


def _nested_get(record: Mapping[str, Any], *paths: tuple[str, ...]) -> Any:
    for path in paths:
        value: Any = record
        ok = True
        for key in path:
            if not isinstance(value, Mapping) or key not in value:
                ok = False
                break
            value = value[key]
        if ok:
            return value
    return None


def _finite_nonnegative(value: float, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _relative_improvement(reference: float, candidate: float) -> float:
    if float(reference) == 0.0:
        return 0.0
    return float((float(reference) - float(candidate)) / float(reference))


__all__ = [
    "INCONCLUSIVE",
    "MCP1OnlyForwardResult",
    "MCP1OnlyParamSelection",
    "MCP1OnlyRunPlan",
    "MCP1_ONLY_BASELINE_STEP6500_METRICS",
    "MCP1_ONLY_FROZEN_GROUPS",
    "MCP1_ONLY_PRIMARY_METRIC",
    "MCP1_ONLY_TRAINABLE_GROUPS",
    "NF_SF_MCP1_ONLY_CONTINUATION_PARENT_STEP",
    "NF_SF_MCP1_ONLY_CONTINUATION_SCHEMA",
    "NF_SF_MCP1_ONLY_CONTINUATION_TARGET_STEP",
    "NF_SF_MCP1_ONLY_CONTINUATION_UPDATE_COUNT",
    "NF_SF_MCP1_ONLY_PARENT_CHECKPOINT_SHA256",
    "NF_SF_MCP1_ONLY_PARENT_GIT_SHA",
    "NO_SUPPORT",
    "SUPPORT_JOINT_TRAINING_INTERFERENCE",
    "assert_mcp1_only_trainable_contract",
    "assert_no_forbidden_mcp1_only_gradients",
    "audit_mcp1_only_gradients",
    "build_mcp1_only_optimizer_from_canonical_state",
    "build_mcp1_only_provenance",
    "build_mcp1_only_run_plan",
    "classify_mcp1_only_comparison",
    "compare_parameter_sha256_reports",
    "compute_mcp1_only_losses",
    "configure_mcp1_only_trainable_parameters",
    "forbidden_feature_contract",
    "has_nonfinite_trainable_grad",
    "load_matching_control_artifact_bundle",
    "mcp1_only_first_step_contract",
    "mcp1_only_loss_metrics",
    "mcp1_only_step_numbers",
    "parameter_sha256_report",
    "run_mcp1_only_forward_loss",
    "trainable_parameter_delta_report",
    "trainable_parameter_snapshot",
    "validate_matching_control_provenance",
    "validate_mcp1_only_manifest",
    "validate_mcp1_only_optimizer_isolation",
]
