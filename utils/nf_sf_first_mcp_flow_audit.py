from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

import utils.nf_sf_full_sequence_eval as deployment
from utils.nf_sf_m3 import tensor_sha256, tensor_summary
from utils.nf_sf_tensors import (
    DEFAULT_NUM_TRAIN_TIMESTEPS,
    DEFAULT_S_MAIN,
    DEFAULT_S_MCP,
)
from utils.scheduler import FlowMatchScheduler


FIRST_MCP_FLOW_AUDIT_SCHEMA = "nf_sf_full_sequence_first_mcp_flow_audit_v1"
FIRST_MCP_TENSOR_SCHEMA = "nf_sf_full_sequence_first_mcp_flow_audit_tensors_v1"
PREDICTED_ROLLOUT = "predicted_rollout"
ORACLE_FLOW_ROLLOUT = "oracle_flow_rollout"
TEACHER_STATE_PROBE = "teacher_state_probe"
TEACHER_REFERENCE = "teacher_reference"
PREDICTED_FIRST_MCP = "predicted_first_mcp"
ORACLE_FLOW_FIRST_MCP = "oracle_flow_first_mcp"
HISTORY_CHUNK_INDEX = 0
CURRENT_CHUNK_INDEX = 1
FUTURE_CHUNK_INDEX = 2


@dataclass(frozen=True)
class FirstMCPFlowAuditResult:
    manifest: dict[str, Any]
    tensors: dict[str, Any]
    hybrid_latents: dict[str, torch.Tensor]
    comparisons: dict[str, dict[str, Any]]


def build_flow_match_scheduler(*, shift: float, device: torch.device | str) -> FlowMatchScheduler:
    scheduler = FlowMatchScheduler(
        shift=float(shift),
        sigma_min=0.0,
        extra_one_step=True,
    )
    scheduler.set_timesteps(DEFAULT_NUM_TRAIN_TIMESTEPS, training=True)
    scheduler.sigmas = scheduler.sigmas.to(device)
    scheduler.timesteps = scheduler.timesteps.to(device)
    return scheduler


def run_first_mcp_flow_audit(
    *,
    runtime_factory: Callable[[], deployment.DeploymentRuntime],
    mcp_scheduler: Any,
    source_noise: torch.Tensor,
    teacher_target: torch.Tensor,
    teacher_payload: Mapping[str, Any],
    conditional_dict: Mapping[str, Any],
    checkpoint_summary: Mapping[str, Any],
    common_inputs: Mapping[str, Any],
    common_inputs_fingerprint_sha256: str,
    runtime_git_sha: str,
    training_checkpoint_git_sha: str,
) -> FirstMCPFlowAuditResult:
    _validate_source_and_teacher(source_noise, teacher_target)
    schedule = deployment.resolve_deployment_schedule()
    rng_plan = deployment.build_absolute_chunk_rng_plan(
        source_noise=source_noise,
        rollout_seed=int(teacher_payload["rollout_seed"]),
        num_denoising_steps=len(schedule.raw_schedule),
        chunk_frames=deployment.FULL_SEQUENCE_CHUNK_FRAMES,
    )
    predicted = _run_predicted_rollout(
        runtime=runtime_factory(),
        mcp_scheduler=mcp_scheduler,
        source_noise=source_noise,
        teacher_target=teacher_target,
        conditional_dict=conditional_dict,
        schedule=schedule,
        rng_plan=rng_plan,
    )
    oracle = _run_oracle_flow_rollout(
        runtime=runtime_factory(),
        mcp_scheduler=mcp_scheduler,
        source_noise=source_noise,
        teacher_target=teacher_target,
        conditional_dict=conditional_dict,
        schedule=schedule,
        rng_plan=rng_plan,
    )
    probe = _run_teacher_state_probe(
        runtime_factory=runtime_factory,
        mcp_scheduler=mcp_scheduler,
        source_noise=source_noise,
        teacher_target=teacher_target,
        conditional_dict=conditional_dict,
        schedule=schedule,
        rng_plan=rng_plan,
    )
    _require_main_trajectory_exact(
        predicted["main_trajectory"],
        oracle["main_trajectory"],
    )
    oracle["trace"]["main_trajectory_matches_predicted"] = True
    step0 = _step0_decisive_metrics(predicted, oracle, probe)
    teacher_reference = teacher_target.detach().cpu()
    predicted_hybrid = build_chunk2_hybrid_latent(
        teacher_target,
        replacement_chunk=predicted["final_chunk2"],
    )
    oracle_hybrid = build_chunk2_hybrid_latent(
        teacher_target,
        replacement_chunk=oracle["final_chunk2"],
    )
    comparisons = {
        "predicted_vs_teacher": deployment.build_comparison_report(
            name="predicted_vs_teacher",
            left_mode=PREDICTED_FIRST_MCP,
            right_mode=TEACHER_REFERENCE,
            latent_left=predicted_hybrid,
            latent_right=teacher_reference,
        ),
        "oracle_vs_teacher": deployment.build_comparison_report(
            name="oracle_vs_teacher",
            left_mode=ORACLE_FLOW_FIRST_MCP,
            right_mode=TEACHER_REFERENCE,
            latent_left=oracle_hybrid,
            latent_right=teacher_reference,
        ),
    }
    manifest = {
        "schema": FIRST_MCP_FLOW_AUDIT_SCHEMA,
        "status": "PASS",
        "diagnostic_only": True,
        "non_deployable": True,
        "runtime_git_sha": str(runtime_git_sha),
        "training_checkpoint_git_sha": str(training_checkpoint_git_sha),
        "checkpoint": dict(checkpoint_summary),
        "common_inputs_fingerprint_sha256": str(common_inputs_fingerprint_sha256),
        "common_inputs": dict(common_inputs),
        "rng_plan_fingerprint_sha256": rng_plan["trace"][
            "rng_plan_fingerprint_sha256"
        ],
        "raw_schedule": list(schedule.raw_schedule),
        "main_warped_schedule": list(schedule.main_warped_schedule),
        "mcp_warped_schedule": list(schedule.mcp_warped_schedule),
        "main_shift": DEFAULT_S_MAIN,
        "mcp_shift": DEFAULT_S_MCP,
        "chunk_contract": {
            "history_chunks": [HISTORY_CHUNK_INDEX],
            "current_chunk": CURRENT_CHUNK_INDEX,
            "future_chunk": FUTURE_CHUNK_INDEX,
            "depths_used": [1],
            "solver_steps": len(schedule.raw_schedule),
        },
        "input_tensors": _input_tensor_provenance(source_noise, teacher_target),
        "branches": {
            PREDICTED_ROLLOUT: predicted["trace"],
            ORACLE_FLOW_ROLLOUT: oracle["trace"],
            TEACHER_STATE_PROBE: probe["trace"],
        },
        "step0_decisive_metrics": step0,
        "comparisons": {
            key: {
                "latent": value["latent"],
                "visual_review_status": value["visual_review_status"],
                "visual_quality_pass": value["visual_quality_pass"],
            }
            for key, value in comparisons.items()
        },
        "forbidden_features": {
            "mcp_depth2": False,
            "mcp_depth3": False,
            "target_refinement": False,
            "verifier": False,
            "dmd": False,
            "routing": False,
            "self_rollout_training": False,
            "speed_benchmark": False,
        },
        "interpretation_contract": {
            "model_flow_failure_supported": None,
            "solver_semantics_failure_supported": None,
            "solver_state_distribution_drift_supported": None,
            "training_like_state_failure_supported": None,
            "case_descriptions": {
                "case_a": (
                    "oracle_flow rollout restores teacher chunk2, but step0 or "
                    "teacher-state probe model flow is already wrong: MCP "
                    "model/objective/capacity/fusion is the primary suspect."
                ),
                "case_b": (
                    "model is accurate on teacher-state probes, but predicted "
                    "rollout flow rapidly worsens: solver-state train/test "
                    "distribution drift is important."
                ),
                "case_c": (
                    "exact oracle flow through current flow-to-x0/re-noise "
                    "cannot preserve teacher chunk2: MCP inference scheduler, "
                    "timestep, or flow-to-x0 semantics are wrong."
                ),
                "case_d": "multiple symptoms improve or fail: mechanisms are mixed.",
            },
        },
    }
    tensors = {
        "schema": FIRST_MCP_TENSOR_SCHEMA,
        "teacher_chunk1": _chunk(teacher_target, CURRENT_CHUNK_INDEX).detach().cpu(),
        "teacher_chunk2": _chunk(teacher_target, FUTURE_CHUNK_INDEX).detach().cpu(),
        "predicted_final_chunk2": predicted["final_chunk2"].detach().cpu(),
        "oracle_final_chunk2": oracle["final_chunk2"].detach().cpu(),
        "predicted_mcp_flows": [
            step["predicted_flow"].detach().cpu()
            for step in predicted["tensor_steps"]
        ],
        "predicted_teacher_directed_flows": [
            step["teacher_directed_flow"].detach().cpu()
            for step in predicted["tensor_steps"]
        ],
        "oracle_state_model_mcp_flows": [
            step["model_predicted_flow"].detach().cpu()
            for step in oracle["tensor_steps"]
        ],
        "exact_target_flows": [
            step["exact_flow"].detach().cpu()
            for step in oracle["tensor_steps"]
        ],
        "teacher_state_probe": {
            "main_flows": [
                step["main_predicted_flow"].detach().cpu()
                for step in probe["tensor_steps"]
            ],
            "mcp_flows": [
                step["mcp_predicted_flow"].detach().cpu()
                for step in probe["tensor_steps"]
            ],
            "main_x0": [
                step["main_predicted_x0"].detach().cpu()
                for step in probe["tensor_steps"]
            ],
            "mcp_x0": [
                step["mcp_predicted_x0"].detach().cpu()
                for step in probe["tensor_steps"]
            ],
        },
    }
    validate_first_mcp_flow_audit_manifest(manifest)
    return FirstMCPFlowAuditResult(
        manifest=manifest,
        tensors=tensors,
        hybrid_latents={
            TEACHER_REFERENCE: teacher_reference,
            PREDICTED_FIRST_MCP: predicted_hybrid,
            ORACLE_FLOW_FIRST_MCP: oracle_hybrid,
        },
        comparisons=comparisons,
    )


def build_chunk2_hybrid_latent(
    teacher_target: torch.Tensor,
    *,
    replacement_chunk: torch.Tensor,
) -> torch.Tensor:
    _require_finite_tensor(teacher_target, name="teacher_target")
    _require_finite_tensor(replacement_chunk, name="replacement_chunk2")
    if tuple(replacement_chunk.shape) != tuple(_chunk(teacher_target, FUTURE_CHUNK_INDEX).shape):
        raise RuntimeError("replacement chunk2 shape mismatch")
    hybrid = teacher_target.detach().clone()
    start = FUTURE_CHUNK_INDEX * deployment.FULL_SEQUENCE_CHUNK_FRAMES
    hybrid[:, start:start + deployment.FULL_SEQUENCE_CHUNK_FRAMES] = replacement_chunk
    return hybrid.detach().cpu()


def teacher_directed_flow_for_state(
    scheduler: Any,
    *,
    teacher_x0: torch.Tensor,
    future_state: torch.Tensor,
    timestep: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    _require_finite_tensor(teacher_x0, name="teacher_directed_teacher_x0")
    _require_finite_tensor(future_state, name="teacher_directed_future_state")
    _require_finite_tensor(timestep, name="teacher_directed_timestep")
    sigma = _resolved_sigma(scheduler, timestep, future_state)
    if abs(float(sigma)) <= 1.0e-12:
        raise RuntimeError("cannot infer noise at sigma=0")
    implied_noise = (
        future_state.float() - (1.0 - float(sigma)) * teacher_x0.float()
    ) / float(sigma)
    implied_noise = implied_noise.to(device=future_state.device, dtype=future_state.dtype)
    _require_finite_tensor(implied_noise, name="teacher_directed_implied_noise")
    flow = _training_target_chunk(
        scheduler,
        clean=teacher_x0,
        noise=implied_noise,
        timestep=timestep,
    )
    return flow, implied_noise, float(sigma)


def validate_first_mcp_flow_audit_manifest(manifest: Mapping[str, Any]) -> None:
    if manifest.get("schema") != FIRST_MCP_FLOW_AUDIT_SCHEMA:
        raise RuntimeError("first MCP audit schema mismatch")
    branches = manifest.get("branches")
    if not isinstance(branches, Mapping):
        raise RuntimeError("first MCP audit branches missing")
    for key in (PREDICTED_ROLLOUT, ORACLE_FLOW_ROLLOUT, TEACHER_STATE_PROBE):
        branch = branches.get(key)
        if not isinstance(branch, Mapping):
            raise RuntimeError(f"{key} branch missing")
        steps = branch.get("steps")
        if not isinstance(steps, Sequence) or len(steps) != 4:
            raise RuntimeError(f"{key} must contain four raw solver steps")
        for step_index, step in enumerate(steps):
            if not isinstance(step, Mapping):
                raise RuntimeError(f"{key} step must be a mapping")
            if int(step.get("raw_index", -1)) != step_index:
                raise RuntimeError(f"{key} raw index mismatch")
            _validate_joint_forward_rng_guard(
                step.get("joint_forward_rng"),
                branch=key,
                step_index=step_index,
            )
    if branches[PREDICTED_ROLLOUT].get("mcp_depths_used") != [1]:
        raise RuntimeError("predicted branch must use depth1 only")
    if branches[ORACLE_FLOW_ROLLOUT].get("mcp_depths_used") != [1]:
        raise RuntimeError("oracle branch must use depth1 only")
    if branches[TEACHER_STATE_PROBE].get("mcp_depths_used") != [1]:
        raise RuntimeError("teacher probe must use depth1 only")
    if branches[ORACLE_FLOW_ROLLOUT].get("used_exact_flow_for_transition") is not True:
        raise RuntimeError("oracle branch must use exact flow for transition")
    if branches[PREDICTED_ROLLOUT].get("used_model_flow_for_transition") is not True:
        raise RuntimeError("predicted branch must use model flow for transition")
    if branches[ORACLE_FLOW_ROLLOUT].get("main_trajectory_matches_predicted") is not True:
        raise RuntimeError("oracle branch Main trajectory diverged from predicted branch")
    step0 = manifest.get("step0_decisive_metrics")
    if not isinstance(step0, Mapping):
        raise RuntimeError("step0 decisive metrics missing")
    if step0.get("predicted_vs_teacher_probe_input_exact") is not True:
        raise RuntimeError("step0 predicted/probe input mismatch")
    if step0.get("predicted_vs_teacher_probe_flow_sha_exact") is not True:
        raise RuntimeError("step0 predicted/probe MCP flow SHA mismatch")
    hypotheses = manifest.get("interpretation_contract")
    if not isinstance(hypotheses, Mapping):
        raise RuntimeError("interpretation contract missing")
    for field in (
        "model_flow_failure_supported",
        "solver_semantics_failure_supported",
        "solver_state_distribution_drift_supported",
        "training_like_state_failure_supported",
    ):
        if hypotheses.get(field) is not None:
            raise RuntimeError("first MCP audit hypotheses must remain null")


def _run_predicted_rollout(
    *,
    runtime: deployment.DeploymentRuntime,
    main_transition_scheduler: Any | None = None,
    mcp_scheduler: Any,
    source_noise: torch.Tensor,
    teacher_target: torch.Tensor,
    conditional_dict: Mapping[str, Any],
    schedule: deployment.DeploymentSchedule,
    rng_plan: Mapping[str, Any],
) -> dict[str, Any]:
    _recache_teacher_history0(
        runtime=runtime,
        source_noise=source_noise,
        teacher_target=teacher_target,
        conditional_dict=conditional_dict,
        rng_plan=rng_plan,
    )
    current_state = _chunk(source_noise, CURRENT_CHUNK_INDEX).detach().clone()
    future_state = _chunk(source_noise, FUTURE_CHUNK_INDEX).detach().clone()
    teacher_chunk2 = _chunk(teacher_target, FUTURE_CHUNK_INDEX)
    trace_steps: list[dict[str, Any]] = []
    tensor_steps: list[dict[str, torch.Tensor]] = []
    main_trajectory: list[dict[str, str]] = []
    final_chunk2 = future_state
    for step_index, (raw_t, main_t, mcp_t) in enumerate(
        zip(schedule.raw_schedule, schedule.main_warped_schedule, schedule.mcp_warped_schedule)
    ):
        future_input = future_state.detach().clone()
        main_flow, main_x0, mcp_flow, call_record = _call_joint_depth1(
            runtime=runtime,
            conditional_dict=conditional_dict,
            current_state=current_state,
            future_state=future_state,
            current_start_frame=CURRENT_CHUNK_INDEX * deployment.FULL_SEQUENCE_CHUNK_FRAMES,
            future_start_frame=FUTURE_CHUNK_INDEX * deployment.FULL_SEQUENCE_CHUNK_FRAMES,
            main_timestep_value=float(main_t),
            mcp_timestep_value=float(mcp_t),
        )
        mcp_timestep = _timestep(float(mcp_t), future_state)
        teacher_directed_flow, implied_noise, sigma = teacher_directed_flow_for_state(
            mcp_scheduler,
            teacher_x0=teacher_chunk2,
            future_state=future_state,
            timestep=mcp_timestep,
        )
        predicted_x0 = _flow_to_x0_chunk(
            mcp_scheduler,
            flow=mcp_flow,
            state=future_state,
            timestep=mcp_timestep,
        )
        transition_state = None
        if step_index < len(schedule.raw_schedule) - 1:
            next_main_t = float(schedule.main_warped_schedule[step_index + 1])
            current_noise = _transition_noise(
                rng_plan,
                chunk_index=CURRENT_CHUNK_INDEX,
                step_index=step_index,
                template=main_x0,
            )
            current_state = _add_noise_chunk(
                runtime.scheduler
                if main_transition_scheduler is None
                else main_transition_scheduler,
                clean=main_x0,
                noise=current_noise,
                timestep=_timestep(next_main_t, main_x0),
            )
            next_mcp_t = float(schedule.mcp_warped_schedule[step_index + 1])
            future_noise = _transition_noise(
                rng_plan,
                chunk_index=FUTURE_CHUNK_INDEX,
                step_index=step_index,
                template=predicted_x0,
            )
            future_state = _add_noise_chunk(
                mcp_scheduler,
                clean=predicted_x0,
                noise=future_noise,
                timestep=_timestep(next_mcp_t, predicted_x0),
            )
            transition_state = _tensor_record(future_state)
        else:
            final_chunk2 = predicted_x0.detach().clone()
        main_trajectory.append(
            {
                "main_input_sha256": tensor_sha256(call_record["current_input"].detach().cpu()),
                "main_predicted_flow_sha256": tensor_sha256(main_flow.detach().cpu()),
                "main_x0_sha256": tensor_sha256(main_x0.detach().cpu()),
            }
        )
        trace_steps.append(
            {
                "raw_index": int(step_index),
                "raw_timestep": float(raw_t),
                "main_warped_timestep": float(main_t),
                "mcp_warped_timestep": float(mcp_t),
                "resolved_sigma": sigma,
                "future_state": _tensor_record(future_input),
                "predicted_flow": _tensor_record(mcp_flow),
                "teacher_directed_oracle_flow": _tensor_record(teacher_directed_flow),
                "teacher_directed_target_source": "actual_state_implied_noise",
                "implied_noise": _tensor_record(implied_noise),
                "predicted_flow_vs_teacher_directed_flow_mse": _mse(mcp_flow, teacher_directed_flow),
                "predicted_x0_vs_teacher_mse": _mse(predicted_x0, teacher_chunk2),
                "predicted_x0": _tensor_record(predicted_x0),
                "transition_state": transition_state,
                "main": main_trajectory[-1],
                "depths_requested": [1],
                "joint_forward_rng": call_record["joint_forward_rng"],
            }
        )
        tensor_steps.append(
            {
                "predicted_flow": mcp_flow.detach().clone(),
                "teacher_directed_flow": teacher_directed_flow.detach().clone(),
                "predicted_x0": predicted_x0.detach().clone(),
            }
        )
    return {
        "trace": {
            "mode": PREDICTED_ROLLOUT,
            "steps": trace_steps,
            "mcp_depths_used": [1],
            "used_model_flow_for_transition": True,
            "used_exact_flow_for_transition": False,
            "history_chunks": [HISTORY_CHUNK_INDEX],
            "current_chunk": CURRENT_CHUNK_INDEX,
            "future_chunk": FUTURE_CHUNK_INDEX,
        },
        "tensor_steps": tensor_steps,
        "main_trajectory": main_trajectory,
        "final_chunk2": final_chunk2.detach().clone(),
    }


def _run_oracle_flow_rollout(
    *,
    runtime: deployment.DeploymentRuntime,
    main_transition_scheduler: Any | None = None,
    mcp_scheduler: Any,
    source_noise: torch.Tensor,
    teacher_target: torch.Tensor,
    conditional_dict: Mapping[str, Any],
    schedule: deployment.DeploymentSchedule,
    rng_plan: Mapping[str, Any],
) -> dict[str, Any]:
    _recache_teacher_history0(
        runtime=runtime,
        source_noise=source_noise,
        teacher_target=teacher_target,
        conditional_dict=conditional_dict,
        rng_plan=rng_plan,
    )
    current_state = _chunk(source_noise, CURRENT_CHUNK_INDEX).detach().clone()
    teacher_chunk2 = _chunk(teacher_target, FUTURE_CHUNK_INDEX)
    future_state = _chunk(source_noise, FUTURE_CHUNK_INDEX).detach().clone()
    state_noise = _chunk(source_noise, FUTURE_CHUNK_INDEX).detach().clone()
    trace_steps: list[dict[str, Any]] = []
    tensor_steps: list[dict[str, torch.Tensor]] = []
    main_trajectory: list[dict[str, str]] = []
    final_chunk2 = future_state
    for step_index, (raw_t, main_t, mcp_t) in enumerate(
        zip(schedule.raw_schedule, schedule.main_warped_schedule, schedule.mcp_warped_schedule)
    ):
        future_input = future_state.detach().clone()
        main_flow, main_x0, model_mcp_flow, call_record = _call_joint_depth1(
            runtime=runtime,
            conditional_dict=conditional_dict,
            current_state=current_state,
            future_state=future_state,
            current_start_frame=CURRENT_CHUNK_INDEX * deployment.FULL_SEQUENCE_CHUNK_FRAMES,
            future_start_frame=FUTURE_CHUNK_INDEX * deployment.FULL_SEQUENCE_CHUNK_FRAMES,
            main_timestep_value=float(main_t),
            mcp_timestep_value=float(mcp_t),
        )
        mcp_timestep = _timestep(float(mcp_t), future_state)
        exact_flow = _training_target_chunk(
            mcp_scheduler,
            clean=teacher_chunk2,
            noise=state_noise,
            timestep=mcp_timestep,
        )
        model_x0 = _flow_to_x0_chunk(
            mcp_scheduler,
            flow=model_mcp_flow,
            state=future_state,
            timestep=mcp_timestep,
        )
        oracle_x0 = _flow_to_x0_chunk(
            mcp_scheduler,
            flow=exact_flow,
            state=future_state,
            timestep=mcp_timestep,
        )
        transition_state = None
        planned_noise_record = _tensor_record(state_noise)
        if step_index < len(schedule.raw_schedule) - 1:
            next_main_t = float(schedule.main_warped_schedule[step_index + 1])
            current_noise = _transition_noise(
                rng_plan,
                chunk_index=CURRENT_CHUNK_INDEX,
                step_index=step_index,
                template=main_x0,
            )
            current_state = _add_noise_chunk(
                runtime.scheduler
                if main_transition_scheduler is None
                else main_transition_scheduler,
                clean=main_x0,
                noise=current_noise,
                timestep=_timestep(next_main_t, main_x0),
            )
            state_noise = _transition_noise(
                rng_plan,
                chunk_index=FUTURE_CHUNK_INDEX,
                step_index=step_index,
                template=teacher_chunk2,
            )
            next_mcp_t = float(schedule.mcp_warped_schedule[step_index + 1])
            future_state = _add_noise_chunk(
                mcp_scheduler,
                clean=teacher_chunk2,
                noise=state_noise,
                timestep=_timestep(next_mcp_t, teacher_chunk2),
            )
            transition_state = _tensor_record(future_state)
        else:
            final_chunk2 = oracle_x0.detach().clone()
        main_trajectory.append(
            {
                "main_input_sha256": tensor_sha256(call_record["current_input"].detach().cpu()),
                "main_predicted_flow_sha256": tensor_sha256(main_flow.detach().cpu()),
                "main_x0_sha256": tensor_sha256(main_x0.detach().cpu()),
            }
        )
        trace_steps.append(
            {
                "raw_index": int(step_index),
                "raw_timestep": float(raw_t),
                "main_warped_timestep": float(main_t),
                "mcp_warped_timestep": float(mcp_t),
                "resolved_sigma": _resolved_sigma(mcp_scheduler, mcp_timestep, future_state),
                "teacher_corrupted_state": _tensor_record(future_input),
                "planned_noise": planned_noise_record,
                "model_predicted_flow": _tensor_record(model_mcp_flow),
                "model_predicted_flow_vs_exact_training_target_mse": _mse(model_mcp_flow, exact_flow),
                "model_predicted_x0_vs_teacher_mse": _mse(model_x0, teacher_chunk2),
                "exact_oracle_flow": _tensor_record(exact_flow),
                "oracle_x0_vs_teacher_mse": _mse(oracle_x0, teacher_chunk2),
                "oracle_x0": _tensor_record(oracle_x0),
                "transition_state": transition_state,
                "main": main_trajectory[-1],
                "depths_requested": [1],
                "joint_forward_rng": call_record["joint_forward_rng"],
            }
        )
        tensor_steps.append(
            {
                "model_predicted_flow": model_mcp_flow.detach().clone(),
                "exact_flow": exact_flow.detach().clone(),
                "oracle_x0": oracle_x0.detach().clone(),
            }
        )
    return {
        "trace": {
            "mode": ORACLE_FLOW_ROLLOUT,
            "steps": trace_steps,
            "mcp_depths_used": [1],
            "used_exact_flow_for_transition": True,
            "used_model_flow_for_transition": False,
            "history_chunks": [HISTORY_CHUNK_INDEX],
            "current_chunk": CURRENT_CHUNK_INDEX,
            "future_chunk": FUTURE_CHUNK_INDEX,
            "main_trajectory_matches_predicted": None,
        },
        "tensor_steps": tensor_steps,
        "main_trajectory": main_trajectory,
        "final_chunk2": final_chunk2.detach().clone(),
    }


def _run_teacher_state_probe(
    *,
    runtime_factory: Callable[[], deployment.DeploymentRuntime],
    mcp_scheduler: Any,
    source_noise: torch.Tensor,
    teacher_target: torch.Tensor,
    conditional_dict: Mapping[str, Any],
    schedule: deployment.DeploymentSchedule,
    rng_plan: Mapping[str, Any],
) -> dict[str, Any]:
    teacher_chunk1 = _chunk(teacher_target, CURRENT_CHUNK_INDEX)
    teacher_chunk2 = _chunk(teacher_target, FUTURE_CHUNK_INDEX)
    trace_steps: list[dict[str, Any]] = []
    tensor_steps: list[dict[str, torch.Tensor]] = []
    for step_index, (raw_t, main_t, mcp_t) in enumerate(
        zip(schedule.raw_schedule, schedule.main_warped_schedule, schedule.mcp_warped_schedule)
    ):
        runtime = runtime_factory()
        _recache_teacher_history0(
            runtime=runtime,
            source_noise=source_noise,
            teacher_target=teacher_target,
            conditional_dict=conditional_dict,
            rng_plan=rng_plan,
        )
        current_noise = _state_noise_for_step(
            rng_plan,
            source_noise=source_noise,
            chunk_index=CURRENT_CHUNK_INDEX,
            step_index=step_index,
            template=teacher_chunk1,
        )
        future_noise = _state_noise_for_step(
            rng_plan,
            source_noise=source_noise,
            chunk_index=FUTURE_CHUNK_INDEX,
            step_index=step_index,
            template=teacher_chunk2,
        )
        main_timestep = _timestep(float(main_t), teacher_chunk1)
        mcp_timestep = _timestep(float(mcp_t), teacher_chunk2)
        current_state = _add_noise_chunk(
            runtime.scheduler,
            clean=teacher_chunk1,
            noise=current_noise,
            timestep=main_timestep,
        )
        future_state = _add_noise_chunk(
            mcp_scheduler,
            clean=teacher_chunk2,
            noise=future_noise,
            timestep=mcp_timestep,
        )
        main_flow, main_x0, mcp_flow, call_record = _call_joint_depth1(
            runtime=runtime,
            conditional_dict=conditional_dict,
            current_state=current_state,
            future_state=future_state,
            current_start_frame=CURRENT_CHUNK_INDEX * deployment.FULL_SEQUENCE_CHUNK_FRAMES,
            future_start_frame=FUTURE_CHUNK_INDEX * deployment.FULL_SEQUENCE_CHUNK_FRAMES,
            main_timestep_value=float(main_t),
            mcp_timestep_value=float(mcp_t),
        )
        main_exact = _training_target_chunk(
            runtime.scheduler,
            clean=teacher_chunk1,
            noise=current_noise,
            timestep=main_timestep,
        )
        mcp_exact = _training_target_chunk(
            mcp_scheduler,
            clean=teacher_chunk2,
            noise=future_noise,
            timestep=mcp_timestep,
        )
        mcp_x0 = _flow_to_x0_chunk(
            mcp_scheduler,
            flow=mcp_flow,
            state=future_state,
            timestep=mcp_timestep,
        )
        trace_steps.append(
            {
                "raw_index": int(step_index),
                "raw_timestep": float(raw_t),
                "main_warped_timestep": float(main_t),
                "mcp_warped_timestep": float(mcp_t),
                "fresh_kv_teacher_history_chunks": [HISTORY_CHUNK_INDEX],
                "current_state": _tensor_record(current_state),
                "future_state": _tensor_record(future_state),
                "current_noise": _tensor_record(current_noise),
                "future_noise": _tensor_record(future_noise),
                "main_pred_flow_vs_exact_target_mse": _mse(main_flow, main_exact),
                "main_pred_x0_vs_teacher_chunk1_mse": _mse(main_x0, teacher_chunk1),
                "mcp_pred_flow_vs_exact_target_mse": _mse(mcp_flow, mcp_exact),
                "mcp_pred_x0_vs_teacher_chunk2_mse": _mse(mcp_x0, teacher_chunk2),
                "main_predicted_flow": _tensor_record(main_flow),
                "mcp_predicted_flow": _tensor_record(mcp_flow),
                "main_exact_target_flow": _tensor_record(main_exact),
                "mcp_exact_target_flow": _tensor_record(mcp_exact),
                "main_predicted_x0": _tensor_record(main_x0),
                "mcp_predicted_x0": _tensor_record(mcp_x0),
                "depths_requested": [1],
                "independent_probe": True,
                "joint_forward_rng": call_record["joint_forward_rng"],
            }
        )
        tensor_steps.append(
            {
                "main_predicted_flow": main_flow.detach().clone(),
                "mcp_predicted_flow": mcp_flow.detach().clone(),
                "main_predicted_x0": main_x0.detach().clone(),
                "mcp_predicted_x0": mcp_x0.detach().clone(),
                "main_exact_flow": main_exact.detach().clone(),
                "mcp_exact_flow": mcp_exact.detach().clone(),
            }
        )
    return {
        "trace": {
            "mode": TEACHER_STATE_PROBE,
            "steps": trace_steps,
            "mcp_depths_used": [1],
            "history_chunks": [HISTORY_CHUNK_INDEX],
            "current_chunk": CURRENT_CHUNK_INDEX,
            "future_chunk": FUTURE_CHUNK_INDEX,
            "independent_fresh_kv_per_step": True,
            "exact_targets_use_scheduler_training_target": True,
        },
        "tensor_steps": tensor_steps,
    }


def _call_joint_depth1(
    *,
    runtime: deployment.DeploymentRuntime,
    conditional_dict: Mapping[str, Any],
    current_state: torch.Tensor,
    future_state: torch.Tensor,
    current_start_frame: int,
    future_start_frame: int,
    main_timestep_value: float,
    mcp_timestep_value: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    snapshot = deployment.KVSnapshot.capture(runtime.kv_cache)
    kv_before = deployment.kv_boundary_summary(runtime.kv_cache)
    main_timestep = _timestep(main_timestep_value, current_state)
    mcp_timestep = _timestep(mcp_timestep_value, future_state)

    def call_joint():
        return runtime.generator(
            noisy_image_or_video=current_state,
            conditional_dict=dict(conditional_dict),
            timestep=main_timestep,
            kv_cache=runtime.kv_cache,
            crossattn_cache=runtime.crossattn_cache,
            current_start=int(current_start_frame) * int(runtime.frame_seq_length),
            mcp_future_noises=[future_state],
            mcp_future_start_frames=[int(future_start_frame)],
            mcp_timesteps=[mcp_timestep],
        )

    outputs, rng_guard = deployment._call_with_rng_guard(
        device=current_state.device,
        label="first_mcp_joint_forward",
        fn=call_joint,
    )
    if not isinstance(outputs, (tuple, list)) or len(outputs) != 3:
        raise RuntimeError("first MCP audit joint forward must return depth1 MCP output")
    main_flow, main_x0 = deployment._unpack_main_outputs(outputs)
    _require_finite_tensor(main_flow, name="main_flow")
    _require_finite_tensor(main_x0, name="main_x0")
    mcp_outputs = outputs[2]
    if not isinstance(mcp_outputs, (tuple, list)) or len(mcp_outputs) != 1:
        raise RuntimeError("first MCP audit must request MCP depth1 only")
    mcp_flow = mcp_outputs[0]
    if not torch.is_tensor(mcp_flow):
        raise TypeError("first MCP audit MCP flow must be a tensor")
    _require_finite_tensor(mcp_flow, name="mcp_flow")
    restored = snapshot.restore(runtime.kv_cache)
    if not restored:
        raise RuntimeError("first MCP audit KV rollback failed")
    kv_after = deployment.kv_boundary_summary(runtime.kv_cache)
    deployment._require_kv_rollback_matches(kv_before, kv_after)
    return main_flow, main_x0, mcp_flow, {
        "current_input": current_state.detach().clone(),
        "future_input": future_state.detach().clone(),
        "joint_forward_rng": rng_guard,
    }


def _recache_teacher_history0(
    *,
    runtime: deployment.DeploymentRuntime,
    source_noise: torch.Tensor,
    teacher_target: torch.Tensor,
    conditional_dict: Mapping[str, Any],
    rng_plan: Mapping[str, Any],
) -> None:
    counts = {"clean_recache_forward_count": 0}
    deployment._clean_recache(
        runtime=runtime,
        conditional_dict=conditional_dict,
        rng_plan=rng_plan,
        counts=counts,
        clean_chunk=_chunk(teacher_target, HISTORY_CHUNK_INDEX),
        chunk_index=HISTORY_CHUNK_INDEX,
        start_frame=0,
        expected_before=None,
    )


def _step0_decisive_metrics(
    predicted: Mapping[str, Any],
    oracle: Mapping[str, Any],
    probe: Mapping[str, Any],
) -> dict[str, Any]:
    predicted_step0 = predicted["trace"]["steps"][0]
    oracle_step0 = oracle["trace"]["steps"][0]
    probe_step0 = probe["trace"]["steps"][0]
    predicted_input = (
        predicted_step0["main"]["main_input_sha256"],
        predicted_step0["future_state"]["sha256"],
        predicted_step0["main_warped_timestep"],
        predicted_step0["mcp_warped_timestep"],
    )
    probe_input = (
        probe_step0["current_state"]["sha256"],
        probe_step0["future_state"]["sha256"],
        probe_step0["main_warped_timestep"],
        probe_step0["mcp_warped_timestep"],
    )
    input_exact = predicted_input == probe_input
    predicted_flow_sha = predicted_step0["predicted_flow"]["sha256"]
    probe_flow_sha = probe_step0["mcp_predicted_flow"]["sha256"]
    flow_exact = predicted_flow_sha == probe_flow_sha
    return {
        "predicted_mcp_flow_mse_to_exact_target": predicted_step0[
            "predicted_flow_vs_teacher_directed_flow_mse"
        ],
        "predicted_mcp_x0_mse_to_teacher": predicted_step0[
            "predicted_x0_vs_teacher_mse"
        ],
        "oracle_flow_x0_mse_to_teacher": oracle_step0["oracle_x0_vs_teacher_mse"],
        "teacher_state_probe_mcp_flow_mse": probe_step0[
            "mcp_pred_flow_vs_exact_target_mse"
        ],
        "teacher_state_probe_mcp_x0_mse": probe_step0[
            "mcp_pred_x0_vs_teacher_chunk2_mse"
        ],
        "predicted_vs_teacher_probe_input_exact": input_exact,
        "predicted_vs_teacher_probe_flow_sha_exact": flow_exact,
        "input_mismatch_detail": None
        if input_exact
        else {"predicted": predicted_input, "teacher_state_probe": probe_input},
        "predicted_flow_sha256": predicted_flow_sha,
        "teacher_state_probe_mcp_flow_sha256": probe_flow_sha,
    }


def _require_main_trajectory_exact(
    predicted: Sequence[Mapping[str, str]],
    oracle: Sequence[Mapping[str, str]],
) -> None:
    if len(predicted) != len(oracle) or len(predicted) <= 0:
        raise RuntimeError("Main trajectory comparison requires equal nonempty steps")
    for index, (left, right) in enumerate(zip(predicted, oracle)):
        if dict(left) != dict(right):
            raise RuntimeError(f"oracle Main trajectory diverged at step {index}")


def _input_tensor_provenance(
    source_noise: torch.Tensor,
    teacher_target: torch.Tensor,
) -> dict[str, Any]:
    return {
        "teacher_chunk0_sha256": tensor_sha256(
            _chunk(teacher_target, HISTORY_CHUNK_INDEX).detach().cpu()
        ),
        "teacher_chunk1_sha256": tensor_sha256(
            _chunk(teacher_target, CURRENT_CHUNK_INDEX).detach().cpu()
        ),
        "teacher_chunk2_sha256": tensor_sha256(
            _chunk(teacher_target, FUTURE_CHUNK_INDEX).detach().cpu()
        ),
        "source_noise_chunk1_sha256": tensor_sha256(
            _chunk(source_noise, CURRENT_CHUNK_INDEX).detach().cpu()
        ),
        "source_noise_chunk2_sha256": tensor_sha256(
            _chunk(source_noise, FUTURE_CHUNK_INDEX).detach().cpu()
        ),
        "source_noise_sha256": tensor_sha256(source_noise.detach().cpu()),
        "teacher_target_sha256": tensor_sha256(teacher_target.detach().cpu()),
    }


def _validate_source_and_teacher(
    source_noise: torch.Tensor,
    teacher_target: torch.Tensor,
) -> None:
    if not torch.is_tensor(source_noise) or not torch.is_tensor(teacher_target):
        raise TypeError("source_noise and teacher_target must be tensors")
    if tuple(source_noise.shape) != tuple(teacher_target.shape):
        raise RuntimeError("source_noise and teacher_target shapes must match")
    if source_noise.ndim != 5:
        raise RuntimeError("source_noise must have shape [B,21,C,H,W]")
    if int(source_noise.shape[1]) != deployment.FULL_SEQUENCE_FRAME_COUNT:
        raise RuntimeError("first MCP audit requires 21 latent frames")
    _require_finite_tensor(source_noise, name="source_noise")
    _require_finite_tensor(teacher_target, name="teacher_target")


def _chunk(tensor: torch.Tensor, chunk_index: int) -> torch.Tensor:
    start = int(chunk_index) * deployment.FULL_SEQUENCE_CHUNK_FRAMES
    return tensor[:, start:start + deployment.FULL_SEQUENCE_CHUNK_FRAMES]


def _state_noise_for_step(
    rng_plan: Mapping[str, Any],
    *,
    source_noise: torch.Tensor,
    chunk_index: int,
    step_index: int,
    template: torch.Tensor,
) -> torch.Tensor:
    if int(step_index) == 0:
        noise = _chunk(source_noise, chunk_index).detach().clone()
        _require_finite_tensor(noise, name="state_step_source_noise")
        return noise
    return _transition_noise(
        rng_plan,
        chunk_index=chunk_index,
        step_index=int(step_index) - 1,
        template=template,
    )


def _transition_noise(
    rng_plan: Mapping[str, Any],
    *,
    chunk_index: int,
    step_index: int,
    template: torch.Tensor,
) -> torch.Tensor:
    transitions = rng_plan.get("transition_noises")
    if not isinstance(transitions, Mapping):
        raise RuntimeError("RNG plan transition noises missing")
    noise = transitions[(int(chunk_index), int(step_index))]
    if noise.device != template.device:
        noise = noise.to(device=template.device)
    value = noise.unflatten(0, template.shape[:2]).to(dtype=template.dtype)
    _require_finite_tensor(value, name="transition_noise")
    return value


def _add_noise_chunk(
    scheduler: Any,
    *,
    clean: torch.Tensor,
    noise: torch.Tensor,
    timestep: torch.Tensor,
) -> torch.Tensor:
    _require_finite_tensor(clean, name="add_noise_clean")
    _require_finite_tensor(noise, name="add_noise_noise")
    _require_finite_tensor(timestep, name="add_noise_timestep")
    original_shape = clean.shape
    value = scheduler.add_noise(
        clean.flatten(0, 1),
        noise.flatten(0, 1),
        timestep.flatten(0, 1),
    ).unflatten(0, original_shape[:2])
    value = value.to(device=clean.device, dtype=clean.dtype)
    _require_finite_tensor(value, name="re_noised_transition_state")
    return value


def _training_target_chunk(
    scheduler: Any,
    *,
    clean: torch.Tensor,
    noise: torch.Tensor,
    timestep: torch.Tensor,
) -> torch.Tensor:
    _require_finite_tensor(clean, name="training_target_clean")
    _require_finite_tensor(noise, name="training_target_noise")
    _require_finite_tensor(timestep, name="training_target_timestep")
    original_shape = clean.shape
    value = scheduler.training_target(
        clean.flatten(0, 1),
        noise.flatten(0, 1),
        timestep.flatten(0, 1),
    ).unflatten(0, original_shape[:2])
    value = value.to(device=clean.device, dtype=clean.dtype)
    _require_finite_tensor(value, name="exact_oracle_flow")
    return value


def _flow_to_x0_chunk(
    scheduler: Any,
    *,
    flow: torch.Tensor,
    state: torch.Tensor,
    timestep: torch.Tensor,
) -> torch.Tensor:
    _require_finite_tensor(flow, name="flow_to_x0_flow")
    _require_finite_tensor(state, name="flow_to_x0_state")
    _require_finite_tensor(timestep, name="flow_to_x0_timestep")
    original_shape = state.shape
    value = scheduler.step(
        flow.flatten(0, 1),
        timestep.flatten(0, 1),
        state.flatten(0, 1),
        to_final=True,
    ).unflatten(0, original_shape[:2])
    value = value.to(device=state.device, dtype=state.dtype)
    _require_finite_tensor(value, name="flow_to_x0_result")
    return value


def _timestep(value: float, target: torch.Tensor) -> torch.Tensor:
    if not bool(torch.isfinite(torch.tensor(float(value), dtype=torch.float32)).item()):
        raise RuntimeError("first MCP audit timestep must be finite")
    timestep = torch.full(
        target.shape[:2],
        float(value),
        device=target.device,
        dtype=torch.float32,
    )
    _require_finite_tensor(timestep, name="timestep")
    return timestep


def _resolved_sigma(scheduler: Any, timestep: torch.Tensor, template: torch.Tensor) -> float:
    _require_finite_tensor(timestep, name="resolved_sigma_timestep")
    timestep_flat = timestep.flatten(0, 1).float()
    scheduler.sigmas = scheduler.sigmas.to(template.device)
    scheduler.timesteps = scheduler.timesteps.to(template.device)
    timestep_id = torch.argmin(
        (scheduler.timesteps.unsqueeze(0) - timestep_flat.unsqueeze(1)).abs(),
        dim=1,
    )
    sigma = scheduler.sigmas[timestep_id].reshape(-1)
    _require_finite_tensor(sigma, name="resolved_sigma")
    if not bool(torch.allclose(sigma, sigma[:1])):
        raise RuntimeError("first MCP audit expected shared chunk sigma")
    return float(sigma[0].detach().cpu().item())


def _validate_joint_forward_rng_guard(
    guard: Any,
    *,
    branch: str,
    step_index: int,
) -> None:
    if not isinstance(guard, Mapping):
        raise RuntimeError(f"{branch} step {step_index} joint-forward RNG guard missing")
    if guard.get("unchanged") is not True:
        raise RuntimeError(f"{branch} step {step_index} joint-forward RNG guard changed")
    for field in ("state_before_hash", "state_after_hash"):
        value = guard.get(field)
        if not isinstance(value, str) or len(value) != 64:
            raise RuntimeError(
                f"{branch} step {step_index} joint-forward RNG guard {field} invalid"
            )
    if guard["state_before_hash"] != guard["state_after_hash"]:
        raise RuntimeError(f"{branch} step {step_index} joint-forward RNG guard hash mismatch")


def _require_finite_tensor(tensor: torch.Tensor, *, name: str) -> None:
    deployment._ensure_finite_tensor(tensor, name=f"first_mcp_{name}")


def _tensor_record(tensor: torch.Tensor) -> dict[str, Any]:
    _require_finite_tensor(tensor, name="recorded_tensor")
    summary = tensor_summary(tensor.detach().cpu())
    return {
        "shape": summary["shape"],
        "dtype": summary["dtype"],
        "finite": summary["finite"],
        "sha256": summary["sha256"],
        "max_abs": float(tensor.detach().float().abs().max().item()),
        "mean_abs": float(tensor.detach().float().abs().mean().item()),
    }


def _mse(left: torch.Tensor, right: torch.Tensor) -> float:
    _require_finite_tensor(left, name="mse_left")
    _require_finite_tensor(right, name="mse_right")
    value = (left.detach().float() - right.detach().float()).square().mean()
    _require_finite_tensor(value, name="mse_value")
    return float(value.item())


__all__ = [
    "CURRENT_CHUNK_INDEX",
    "FIRST_MCP_FLOW_AUDIT_SCHEMA",
    "FIRST_MCP_TENSOR_SCHEMA",
    "FUTURE_CHUNK_INDEX",
    "HISTORY_CHUNK_INDEX",
    "ORACLE_FLOW_FIRST_MCP",
    "ORACLE_FLOW_ROLLOUT",
    "PREDICTED_FIRST_MCP",
    "PREDICTED_ROLLOUT",
    "TEACHER_REFERENCE",
    "TEACHER_STATE_PROBE",
    "FirstMCPFlowAuditResult",
    "build_chunk2_hybrid_latent",
    "build_flow_match_scheduler",
    "run_first_mcp_flow_audit",
    "teacher_directed_flow_for_state",
    "validate_first_mcp_flow_audit_manifest",
]
