from __future__ import annotations

import hashlib
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping

import torch

from utils.nf_sf_training import (
    FULL_SEQUENCE_DEPTH_WEIGHTS,
    NFSFFullSequenceForwardResult,
    build_full_sequence_mcp_anchor_inputs,
    compute_nf_sf_full_sequence_losses,
    nf_sf_full_sequence_train_cursor,
    run_nf_sf_full_sequence_forward_loss,
    validate_nf_sf_full_sequence_objective_mode,
    _output_field,
)


NF_SF_MCP_DIRECT_CLEAN_KV_ABLATION_SCHEMA = "nf_sf_mcp_direct_clean_kv_ablation_v1"
NF_SF_MCP_DIRECT_CLEAN_KV_ABLATION_PARENT_STEP = 6500
NF_SF_MCP_DIRECT_CLEAN_KV_ABLATION_TARGET_STEP = 7000
NF_SF_MCP_DIRECT_CLEAN_KV_ABLATION_UPDATE_COUNT = 500
NF_SF_MCP_DIRECT_CLEAN_KV_FIXED_PROBE_SCHEMA = (
    f"{NF_SF_MCP_DIRECT_CLEAN_KV_ABLATION_SCHEMA}_fixed_probe_raw999_v1"
)
NF_SF_MCP_DIRECT_CLEAN_KV_FIXED_PROBE_FILENAME = (
    "fixed_probe_raw999_step007000.json"
)
PRIMARY_FIXED_PROBE_RAW_TIMESTEP = 999
PRIMARY_FIXED_PROBE_DEPTH = 1
PRIMARY_FIXED_PROBE_ANCHOR_INDEX = 1
PRIMARY_FIXED_PROBE_TARGET_CHUNK = 2
PRIMARY_FIXED_PROBE_TARGET_START_FRAME = 6
NF_SF_MCP_DIRECT_CLEAN_KV_PARENT_CHECKPOINT_SHA256 = (
    "9ef57cb2d3e5f20b244129317af4a0e1d2b1c810ba65ec970892e60ccbd34f4f"
)
NF_SF_MCP_DIRECT_CLEAN_KV_PARENT_GIT_SHA = (
    "c3f89888bf6da31b48650f0a680dd6534943f56f"
)
NF_SF_MCP_DIRECT_CLEAN_KV_BASE_TRAINING_GIT_SHA = (
    "2ab9b3a7c08b09140b6cbae23df21107817fe3be"
)

AblationArm = Literal["control", "direct_clean_kv"]
ABLATION_ARMS: tuple[AblationArm, ...] = ("control", "direct_clean_kv")

BASELINE_STEP6500_METRICS = {
    "raw999_mcp1": 0.11986814439296722,
    "paired_mcp1": 0.21171082811633823,
    "paired_mcp2": 0.2519458762835711,
    "paired_mcp3": 0.2634747195697855,
    "paired_main": 0.09161755588866072,
}


@dataclass(frozen=True)
class AblationRunPlan:
    schema: str
    arm: AblationArm
    parent_step: int
    first_step: int
    target_step: int
    update_count: int
    checkpoint_steps: tuple[int, ...]
    validation_steps: tuple[int, ...]
    direct_clean_context_kv: bool
    depth1_direct_context_only: bool
    no_validation6500: bool


def validate_ablation_arm(arm: str) -> AblationArm:
    if arm not in ABLATION_ARMS:
        raise ValueError("--arm must be one of: control, direct_clean_kv")
    return arm  # type: ignore[return-value]


def direct_clean_context_kv_enabled(arm: str) -> bool:
    return validate_ablation_arm(arm) == "direct_clean_kv"


def ablation_step_numbers(
    *,
    parent_step: int = NF_SF_MCP_DIRECT_CLEAN_KV_ABLATION_PARENT_STEP,
    target_step: int = NF_SF_MCP_DIRECT_CLEAN_KV_ABLATION_TARGET_STEP,
) -> tuple[int, ...]:
    parent = int(parent_step)
    target = int(target_step)
    steps = tuple(range(parent + 1, target + 1))
    if len(steps) != NF_SF_MCP_DIRECT_CLEAN_KV_ABLATION_UPDATE_COUNT:
        raise RuntimeError("direct clean-context ablation must run exactly 500 updates")
    if steps[0] != 6501 or steps[-1] != 7000:
        raise RuntimeError("direct clean-context ablation step range must be 6501..7000")
    return steps


def build_ablation_run_plan(arm: str) -> AblationRunPlan:
    resolved_arm = validate_ablation_arm(arm)
    steps = ablation_step_numbers()
    return AblationRunPlan(
        schema=NF_SF_MCP_DIRECT_CLEAN_KV_ABLATION_SCHEMA,
        arm=resolved_arm,
        parent_step=NF_SF_MCP_DIRECT_CLEAN_KV_ABLATION_PARENT_STEP,
        first_step=steps[0],
        target_step=steps[-1],
        update_count=len(steps),
        checkpoint_steps=(steps[-1],),
        validation_steps=(steps[-1],),
        direct_clean_context_kv=direct_clean_context_kv_enabled(resolved_arm),
        depth1_direct_context_only=True,
        no_validation6500=True,
    )


def build_ablation_smoke_plan(arm: str) -> dict[str, Any]:
    resolved_arm = validate_ablation_arm(arm)
    return {
        "schema": NF_SF_MCP_DIRECT_CLEAN_KV_ABLATION_SCHEMA,
        "arm": resolved_arm,
        "engineering_smoke": True,
        "parent_step": NF_SF_MCP_DIRECT_CLEAN_KV_ABLATION_PARENT_STEP,
        "first_step": NF_SF_MCP_DIRECT_CLEAN_KV_ABLATION_PARENT_STEP + 1,
        "target_step": NF_SF_MCP_DIRECT_CLEAN_KV_ABLATION_PARENT_STEP + 1,
        "update_count": 1,
        "checkpoint_steps": (),
        "validation_steps": (),
        "direct_clean_context_kv": direct_clean_context_kv_enabled(resolved_arm),
        "depth1_direct_context_only": True,
        "formal_ablation_schedule_unchanged": True,
        "formal_update_count": NF_SF_MCP_DIRECT_CLEAN_KV_ABLATION_UPDATE_COUNT,
    }


def build_ablation_provenance(
    *,
    arm: str,
    runtime_git_sha: str,
    semantic_lock_fingerprint: str,
    parent_checkpoint_sha256: str = NF_SF_MCP_DIRECT_CLEAN_KV_PARENT_CHECKPOINT_SHA256,
    parent_git_sha: str = NF_SF_MCP_DIRECT_CLEAN_KV_PARENT_GIT_SHA,
    base_training_git_sha: str = NF_SF_MCP_DIRECT_CLEAN_KV_BASE_TRAINING_GIT_SHA,
) -> dict[str, Any]:
    resolved_arm = validate_ablation_arm(arm)
    treatment = direct_clean_context_kv_enabled(resolved_arm)
    return {
        "schema": NF_SF_MCP_DIRECT_CLEAN_KV_ABLATION_SCHEMA,
        "arm": resolved_arm,
        "diagnostic_training_ablation": True,
        "paper_exact_reproduction": False,
        "canonical_training_eligible": False,
        "canonical_deployment_eligible": False,
        "direct_clean_context_kv": treatment,
        "target_query_only": treatment,
        "target_query_direct_clean_kv": treatment,
        "new_trainable_parameters": False,
        "objective_changed": False,
        "optimizer_changed": False,
        "data_changed": False,
        "rng_changed": False,
        "main_attention_changed": False,
        "depth2_depth3_direct_context_changed": False,
        "mcp_depth1_direct_context_changed": treatment,
        "mcp_depth2_direct_context_changed": False,
        "mcp_depth3_direct_context_changed": False,
        "opd_changed": False,
        "on_policy_changed": False,
        "noisy_history_augmentation_changed": False,
        "parent_step": NF_SF_MCP_DIRECT_CLEAN_KV_ABLATION_PARENT_STEP,
        "target_step": NF_SF_MCP_DIRECT_CLEAN_KV_ABLATION_TARGET_STEP,
        "update_count": NF_SF_MCP_DIRECT_CLEAN_KV_ABLATION_UPDATE_COUNT,
        "primary_fixed_probe_required": True,
        "primary_fixed_probe_raw_timestep": PRIMARY_FIXED_PROBE_RAW_TIMESTEP,
        "primary_fixed_probe_depth": PRIMARY_FIXED_PROBE_DEPTH,
        "primary_fixed_probe_anchor_index": PRIMARY_FIXED_PROBE_ANCHOR_INDEX,
        "primary_fixed_probe_target_chunk": PRIMARY_FIXED_PROBE_TARGET_CHUNK,
        "paired_validation_step": NF_SF_MCP_DIRECT_CLEAN_KV_ABLATION_TARGET_STEP,
        "parent_checkpoint_sha256": str(parent_checkpoint_sha256),
        "parent_git_sha": str(parent_git_sha),
        "base_training_git_sha": str(base_training_git_sha),
        "runtime_git_sha": str(runtime_git_sha),
        "semantic_lock_fingerprint": str(semantic_lock_fingerprint),
        "parent_cursor": dict(
            nf_sf_full_sequence_train_cursor(
                NF_SF_MCP_DIRECT_CLEAN_KV_ABLATION_PARENT_STEP
            )
        ),
    }


def ablation_decision_rule_metadata() -> dict[str, Any]:
    return {
        "schema": NF_SF_MCP_DIRECT_CLEAN_KV_ABLATION_SCHEMA,
        "auto_declare_go": False,
        "baseline_step6500": dict(BASELINE_STEP6500_METRICS),
        "primary_metric": "fixed_probe_raw999_step007000.mcp1_flow_mse_to_exact",
        "primary_fixed_probe_required": True,
        "primary_fixed_probe_raw_timestep": PRIMARY_FIXED_PROBE_RAW_TIMESTEP,
        "primary_fixed_probe_depth": PRIMARY_FIXED_PROBE_DEPTH,
        "primary_fixed_probe_anchor_index": PRIMARY_FIXED_PROBE_ANCHOR_INDEX,
        "primary_fixed_probe_target_chunk": PRIMARY_FIXED_PROBE_TARGET_CHUNK,
        "paired_validation_step": NF_SF_MCP_DIRECT_CLEAN_KV_ABLATION_TARGET_STEP,
        "supports_context_pathway_ceiling": {
            "treatment_raw999_mcp1_relative_drop_vs_step6500_min": 0.10,
            "treatment_vs_control_raw999_extra_improvement_pp_min": 7.0,
            "paired_mcp1_treatment_vs_control_extra_improvement_pp_min": 5.0,
            "mcp2_mcp3_relative_worse_than_control_pp_max": 2.0,
            "main_catastrophic_regression_forbidden": True,
        },
        "supports_objective_or_capacity_ceiling": {
            "treatment_vs_control_raw999_extra_improvement_pp_max": 2.0,
            "both_raw999_remain_near_0p12_plateau": True,
            "paired_mcp1_mcp2_mcp3_no_treatment_advantage": True,
        },
        "otherwise": "INCONCLUSIVE",
    }


def parameter_key_tuple(module: torch.nn.Module) -> tuple[str, ...]:
    return tuple(name for name, _ in module.named_parameters())


def assert_parameter_keys_unchanged(
    before: Iterable[str],
    after: Iterable[str],
) -> None:
    before_tuple = tuple(before)
    after_tuple = tuple(after)
    if before_tuple != after_tuple:
        raise RuntimeError("direct clean-context ablation changed parameter keys")


def run_nf_sf_full_sequence_forward_loss_for_ablation(
    generator,
    *,
    arm: str,
    conditional_dict: dict,
    noisy_batch,
    depth_weights: Iterable[float] = FULL_SEQUENCE_DEPTH_WEIGHTS,
    objective_mode: str = "next_forcing_full",
) -> NFSFFullSequenceForwardResult:
    objective_mode = validate_nf_sf_full_sequence_objective_mode(objective_mode)
    if not direct_clean_context_kv_enabled(arm):
        return run_nf_sf_full_sequence_forward_loss(
            generator,
            conditional_dict=conditional_dict,
            noisy_batch=noisy_batch,
            depth_weights=depth_weights,
            objective_mode=objective_mode,
        )

    mcp_anchor_inputs = (
        build_full_sequence_mcp_anchor_inputs(noisy_batch)
        if objective_mode == "next_forcing_full"
        else ()
    )
    outputs = generator.forward_full_sequence_next_forcing(
        noisy_image_or_video=noisy_batch.noisy_main,
        clean_x=noisy_batch.clean_target,
        conditional_dict=conditional_dict,
        timestep_main=noisy_batch.timestep_main,
        mcp_anchor_inputs=mcp_anchor_inputs,
        direct_clean_context_kv=True,
    )
    main_flow_pred = _output_field(outputs, "main_flow_pred")
    mcp_flow_preds_by_depth = tuple(_output_field(outputs, "mcp_flow_preds_by_depth"))
    losses = compute_nf_sf_full_sequence_losses(
        main_flow_pred=main_flow_pred,
        mcp_flow_preds_by_depth=mcp_flow_preds_by_depth,
        noisy_batch=noisy_batch,
        depth_weights=tuple(float(weight) for weight in depth_weights),
        objective_mode=objective_mode,
    )
    return NFSFFullSequenceForwardResult(
        noisy_batch=noisy_batch,
        main_flow_pred=main_flow_pred,
        mcp_flow_preds_by_depth=mcp_flow_preds_by_depth,
        losses=losses,
        tap_shapes=tuple(_output_field(outputs, "tap_shapes")),
        anchor_token_slices=tuple(_output_field(outputs, "anchor_token_slices")),
        main_backbone_forward_count=int(_output_field(outputs, "main_backbone_forward_count")),
        future_embedding_order=_output_field(outputs, "future_embedding_order"),
    )


def tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(str(tuple(value.shape)).encode("ascii"))
    digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def noisy_batch_draw_fingerprint(noisy_batch) -> dict[str, Any]:
    return {
        "rng_draw_order_version": str(noisy_batch.rng_draw_order_version),
        "epsilon_main_sha256": tensor_sha256(noisy_batch.epsilon_main),
        "raw_timestep_main_sha256": tensor_sha256(noisy_batch.raw_timestep_main),
        "timestep_main_sha256": tensor_sha256(noisy_batch.timestep_main),
        "epsilon_mcp_depth_sha256": [
            tensor_sha256(tensor) for tensor in noisy_batch.epsilon_mcp_depths
        ],
        "raw_timestep_mcp_depth_sha256": [
            tensor_sha256(tensor) for tensor in noisy_batch.raw_timestep_mcp_depths
        ],
        "timestep_mcp_depth_sha256": [
            tensor_sha256(tensor) for tensor in noisy_batch.timestep_mcp_depths
        ],
    }


def train_rng_state_sha256(train_rng: torch.Generator) -> str:
    return tensor_sha256(train_rng.get_state())


@contextmanager
def _capture_selected_mcp_stack_kwargs(mcp_module: torch.nn.Module):
    capture = _SelectedMCPStackKwargsCapture()
    handle = mcp_module.register_forward_pre_hook(capture.hook, with_kwargs=True)
    try:
        yield capture
    finally:
        handle.remove()


class _SelectedMCPStackKwargsCapture:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.selected: list[dict[str, Any]] = []

    def hook(
        self,
        _module: torch.nn.Module,
        _args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> None:
        starts = [None if value is None else int(value) for value in kwargs["future_start_frames"]]
        record = {
            "call_index": len(self.calls),
            "future_start_frames": starts,
            "future_count": len(starts),
            "direct_clean_context_kv": bool(kwargs.get("direct_clean_context_kv", False)),
        }
        self.calls.append(record)
        if starts and starts[0] == PRIMARY_FIXED_PROBE_TARGET_START_FRAME:
            self.selected.append(_clone_selected_mcp_stack_kwargs(kwargs, call_index=record["call_index"]))


def _clone_selected_mcp_stack_kwargs(
    kwargs: Mapping[str, Any],
    *,
    call_index: int,
) -> dict[str, Any]:
    clean_context_features = kwargs.get("clean_context_features")
    return {
        "call_index": int(call_index),
        "features": tuple(_clone_tensor(tensor) for tensor in kwargs["features"]),
        "future_embeds": tuple(_clone_tensor(tensor) for tensor in kwargs["future_embeds"]),
        "future_grid_sizes": tuple(_clone_tensor(tensor) for tensor in kwargs["future_grid_sizes"]),
        "future_start_frames": [
            None if value is None else int(value)
            for value in kwargs["future_start_frames"]
        ],
        "timesteps": tuple(
            None if tensor is None else _clone_tensor(tensor)
            for tensor in kwargs["timesteps"]
        ),
        "direct_clean_context_kv": bool(kwargs.get("direct_clean_context_kv", False)),
        "clean_context_features": (
            None
            if clean_context_features is None
            else tuple(_clone_tensor(tensor) for tensor in clean_context_features)
        ),
        "clean_context_grid_sizes": (
            None
            if kwargs.get("clean_context_grid_sizes") is None
            else _clone_tensor(kwargs["clean_context_grid_sizes"])
        ),
        "clean_context_start_frame": int(kwargs.get("clean_context_start_frame", 0)),
    }


def _clone_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(tensor):
        raise TypeError("selected MCP probe input must be a tensor")
    return tensor.detach().cpu().clone()


def _selected_probe_call(capture: _SelectedMCPStackKwargsCapture) -> dict[str, Any]:
    if len(capture.selected) != 1:
        raise RuntimeError(
            "fixed raw999 probe must capture exactly one anchor1 MCP call, "
            f"got {len(capture.selected)}"
        )
    selected = capture.selected[0]
    if selected["future_start_frames"][:3] != [6, 9, 12]:
        raise RuntimeError("fixed raw999 probe selected anchor1 must expose starts [6, 9, 12]")
    return selected


def _mse(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(torch.mean((left.detach().float() - right.detach().float()).square()).item())


def _fused_clean_context_sha256(
    *,
    mcp_module: torch.nn.Module,
    clean_context_features: tuple[torch.Tensor, ...],
) -> str:
    fusion = getattr(mcp_module, "fusion", None)
    concatenated = torch.cat(clean_context_features, dim=-1)
    if fusion is None:
        value = concatenated
    else:
        parameter = next(fusion.parameters(), None)
        if parameter is not None:
            concatenated = concatenated.to(device=parameter.device, dtype=parameter.dtype)
        value = fusion(concatenated)
    return tensor_sha256(value.detach().cpu())


@torch.no_grad()
def run_fixed_raw999_probe_for_ablation(
    generator,
    *,
    arm: str,
    main_scheduler,
    mcp_scheduler,
    source_noise: torch.Tensor,
    teacher_target: torch.Tensor,
    conditional_dict: Mapping[str, Any],
    sample_identity: str,
    global_step: int = NF_SF_MCP_DIRECT_CLEAN_KV_ABLATION_TARGET_STEP,
) -> dict[str, Any]:
    import utils.nf_sf_first_mcp_route_equivalence as route_eq
    import utils.nf_sf_full_sequence_eval as deployment

    if int(global_step) != NF_SF_MCP_DIRECT_CLEAN_KV_ABLATION_TARGET_STEP:
        raise RuntimeError("fixed raw999 ablation probe must run at step7000")
    route_eq._validate_source_and_teacher(source_noise, teacher_target)
    point = route_eq.build_route_equivalence_point(PRIMARY_FIXED_PROBE_RAW_TIMESTEP)
    noisy_batch = route_eq._build_training_noisy_batch(
        main_scheduler=main_scheduler,
        mcp_scheduler=mcp_scheduler,
        source_noise=source_noise,
        teacher_target=teacher_target,
        point=point,
    )
    anchors = build_full_sequence_mcp_anchor_inputs(noisy_batch)
    model = route_eq._model_with_block_mask(generator)
    original_block_mask = model.block_mask
    block_mask_before_was_none = original_block_mask is None
    block_mask_recreated_for_probe = False
    treatment = direct_clean_context_kv_enabled(arm)
    was_training = generator.training
    generator.eval()
    try:
        model.block_mask = None
        with _capture_selected_mcp_stack_kwargs(generator.mcp) as capture:
            def call_training():
                return generator.forward_full_sequence_next_forcing(
                    noisy_image_or_video=noisy_batch.noisy_main,
                    clean_x=teacher_target,
                    conditional_dict=dict(conditional_dict),
                    timestep_main=noisy_batch.timestep_main,
                    mcp_anchor_inputs=anchors,
                    direct_clean_context_kv=treatment,
                )

            outputs, rng_guard = deployment._call_with_rng_guard(
                device=teacher_target.device,
                label="direct_clean_kv_ablation_fixed_raw999_training_route_probe",
                fn=call_training,
            )
        block_mask_recreated_for_probe = model.block_mask is not None
        if not block_mask_recreated_for_probe:
            raise RuntimeError("fixed raw999 probe did not recreate teacher-forcing block_mask")
    finally:
        model.block_mask = original_block_mask
        generator.train(was_training)
    block_mask_restored_exact = (
        model.block_mask is None
        if original_block_mask is None
        else model.block_mask is original_block_mask
    )
    if not block_mask_restored_exact:
        raise RuntimeError("fixed raw999 probe failed to restore original block_mask")

    selected = _selected_probe_call(capture)
    main_flow_full = _output_field(outputs, "main_flow_pred")
    mcp_by_depth = tuple(_output_field(outputs, "mcp_flow_preds_by_depth"))
    if len(mcp_by_depth) < PRIMARY_FIXED_PROBE_DEPTH:
        raise RuntimeError("fixed raw999 probe requires depth1 MCP output")
    main_flow = route_eq._chunk(main_flow_full, route_eq.CURRENT_CHUNK_INDEX)
    mcp_flow = mcp_by_depth[0][:, PRIMARY_FIXED_PROBE_ANCHOR_INDEX]
    exact = route_eq._exact_point_targets(
        main_scheduler=main_scheduler,
        mcp_scheduler=mcp_scheduler,
        source_noise=source_noise,
        teacher_target=teacher_target,
        point=point,
    )
    target_mcp_input = noisy_batch.noisy_mcp_depths[0][:, PRIMARY_FIXED_PROBE_ANCHOR_INDEX]
    clean_context_features = selected["clean_context_features"]
    clean_context_token_count = 0
    clean_context_sha256 = None
    clean_context_grid_frames = None
    clean_context_start_frame = None
    if treatment:
        if clean_context_features is None:
            raise RuntimeError("treatment fixed probe did not pass clean context features")
        clean_context_token_count = int(clean_context_features[0].shape[1])
        clean_context_sha256 = _fused_clean_context_sha256(
            mcp_module=generator.mcp,
            clean_context_features=clean_context_features,
        )
        clean_grid = selected["clean_context_grid_sizes"]
        if clean_grid is None:
            raise RuntimeError("treatment fixed probe missing clean context grid")
        clean_context_grid_frames = int(clean_grid[0, 0].item())
        clean_context_start_frame = int(selected["clean_context_start_frame"])
        if clean_context_token_count != 4680:
            raise RuntimeError("treatment fixed probe clean context token count must be 4680")
        if clean_context_grid_frames != 3 or clean_context_start_frame != 0:
            raise RuntimeError("treatment fixed probe clean context RoPE metadata mismatch")

    return {
        "schema": NF_SF_MCP_DIRECT_CLEAN_KV_FIXED_PROBE_SCHEMA,
        "experiment_schema": NF_SF_MCP_DIRECT_CLEAN_KV_ABLATION_SCHEMA,
        "arm": validate_ablation_arm(arm),
        "global_step": int(global_step),
        "checkpoint_global_step": int(global_step),
        "fixed_validation_identity": str(sample_identity),
        "raw_timestep": PRIMARY_FIXED_PROBE_RAW_TIMESTEP,
        "depth": PRIMARY_FIXED_PROBE_DEPTH,
        "anchor_index": PRIMARY_FIXED_PROBE_ANCHOR_INDEX,
        "target_chunk": PRIMARY_FIXED_PROBE_TARGET_CHUNK,
        "solver_loop": False,
        "mcp1_flow_mse_to_exact": _mse(mcp_flow, exact["mcp_target"]),
        "main_flow_mse_to_exact": _mse(main_flow, exact["main_target"]),
        "source_noise_sha256": tensor_sha256(source_noise.detach().cpu()),
        "source_noise_chunk2_sha256": tensor_sha256(
            route_eq._chunk(source_noise, PRIMARY_FIXED_PROBE_TARGET_CHUNK).detach().cpu()
        ),
        "target_sha256": tensor_sha256(
            route_eq._chunk(teacher_target, PRIMARY_FIXED_PROBE_TARGET_CHUNK).detach().cpu()
        ),
        "exact_flow_sha256": tensor_sha256(exact["mcp_target"].detach().cpu()),
        "exact_main_flow_sha256": tensor_sha256(exact["main_target"].detach().cpu()),
        "target_mcp_input_sha256": tensor_sha256(target_mcp_input.detach().cpu()),
        "target_mcp_embed_sha256": tensor_sha256(selected["future_embeds"][0]),
        "direct_clean_context_kv": treatment,
        "clean_context_token_count": clean_context_token_count,
        "clean_context_sha256": clean_context_sha256,
        "clean_context_grid_frames": clean_context_grid_frames,
        "clean_context_start_frame": clean_context_start_frame,
        "target_start_frame": PRIMARY_FIXED_PROBE_TARGET_START_FRAME,
        "selected_mcp_call_index": int(selected["call_index"]),
        "selected_mcp_future_start_frames": list(selected["future_start_frames"]),
        "forward_rng": dict(rng_guard),
        "block_mask_before_was_none": bool(block_mask_before_was_none),
        "block_mask_recreated_for_probe": bool(block_mask_recreated_for_probe),
        "block_mask_restored_exact": bool(block_mask_restored_exact),
        "block_mask_policy": "reset_to_none_recreate_teacher_forcing_then_restore",
        "paper_exact_reproduction": False,
        "canonical_training_eligible": False,
        "canonical_deployment_eligible": False,
    }


def path_is_outside_repo(path: Path, *, repo_root: Path) -> bool:
    resolved_path = Path(path).resolve()
    resolved_root = Path(repo_root).resolve()
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError:
        return True
    return False


def validate_ablation_real_run_guards(
    *,
    arm: str,
    parent_step: int,
    parent_checkpoint_sha256: str,
    output_dir: Path,
    repo_root: Path,
    staged_changes: bool,
    tracked_changes: bool,
) -> None:
    validate_ablation_arm(arm)
    if int(parent_step) != NF_SF_MCP_DIRECT_CLEAN_KV_ABLATION_PARENT_STEP:
        raise RuntimeError("ablation must fork only from step6500")
    if str(parent_checkpoint_sha256) != NF_SF_MCP_DIRECT_CLEAN_KV_PARENT_CHECKPOINT_SHA256:
        raise RuntimeError("parent checkpoint SHA256 mismatch for direct clean-context ablation")
    if not path_is_outside_repo(Path(output_dir), repo_root=Path(repo_root)):
        raise RuntimeError("real ablation output_dir must be outside the repo")
    if staged_changes:
        raise RuntimeError("real ablation requires no staged changes")
    if tracked_changes:
        raise RuntimeError("real ablation requires tracked worktree clean")


def build_checkpoint_metadata(
    *,
    arm: str,
    runtime_git_sha: str,
    semantic_lock_fingerprint: str,
) -> dict[str, Any]:
    return {
        "schema": NF_SF_MCP_DIRECT_CLEAN_KV_ABLATION_SCHEMA,
        "canonical_training_eligible": False,
        "canonical_deployment_eligible": False,
        "provenance": build_ablation_provenance(
            arm=arm,
            runtime_git_sha=runtime_git_sha,
            semantic_lock_fingerprint=semantic_lock_fingerprint,
        ),
        "decision_rule": ablation_decision_rule_metadata(),
        "run_plan": build_ablation_run_plan(arm).__dict__,
    }
