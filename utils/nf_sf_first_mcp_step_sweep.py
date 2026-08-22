from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

import utils.nf_sf_first_mcp_flow_audit as flow_audit
import utils.nf_sf_full_sequence_eval as deployment
from utils.nf_sf_m3 import tensor_sha256
from utils.nf_sf_tensors import (
    DEFAULT_NUM_TRAIN_TIMESTEPS,
    DEFAULT_S_MAIN,
    DEFAULT_S_MCP,
)
from utils.scheduler import FlowMatchScheduler


FIRST_MCP_STEP_SWEEP_SCHEMA = "nf_sf_full_sequence_first_mcp_step_sweep_v1"
FIRST_MCP_STEP_SWEEP_TENSOR_SCHEMA = (
    "nf_sf_full_sequence_first_mcp_step_sweep_tensors_v1"
)
LIVE_JOINT_PREDICTED = "live_joint_predicted"
TEACHER_CURRENT_PREDICTED_MCP = "teacher_current_predicted_mcp"
ORACLE_FLOW = "oracle_flow"
DEFAULT_STEP_COUNTS = (4, 8, 16, 32)
SEMANTIC_RNG_COUPLING_SCHEMA = "first_mcp_step_sweep_semantic_rng_coupling_v1"
SEMANTIC_RAW_TICK_DENOMINATOR = max(DEFAULT_STEP_COUNTS)
CANONICAL_4_ANCHORED_RNG_MODE = "canonical_4_anchored_semantic_rng"
CANONICAL_RAW_4 = deployment.RAW_DEPLOYMENT_SCHEDULE
CANONICAL_MAIN_4 = deployment.MAIN_DEPLOYMENT_SCHEDULE
CANONICAL_MCP_4 = deployment.MCP_DEPLOYMENT_SCHEDULE
ORACLE_GATE_TOLERANCE = 1.0e-4
EXPECTED_STEP6500_GLOBAL_STEP = 6500
EXPECTED_STEP6500_LOADER_MODE = "DIAGNOSTIC_INTERMEDIATE_STRICT"
SUPPORT_FEW_STEP_INFERENCE_GAP = "SUPPORT_FEW_STEP_INFERENCE_GAP"
NO_SUPPORT = "NO_SUPPORT"
INCONCLUSIVE = "INCONCLUSIVE"
INVALID_ORACLE_GATE = "INVALID_ORACLE_GATE"


@dataclass(frozen=True)
class FirstMCPStepSweepResult:
    manifest: dict[str, Any]
    tensors: dict[str, Any]
    hybrid_latents: dict[str, torch.Tensor]


def build_step_sweep_scheduler(
    *,
    step_count: int,
    shift: float,
    device: torch.device | str,
) -> FlowMatchScheduler:
    _validate_step_count(step_count)
    scheduler = FlowMatchScheduler(
        num_train_timesteps=DEFAULT_NUM_TRAIN_TIMESTEPS,
        shift=float(shift),
        sigma_min=0.0,
        extra_one_step=True,
    )
    scheduler.set_timesteps(int(step_count))
    scheduler.sigmas = scheduler.sigmas.to(device)
    scheduler.timesteps = scheduler.timesteps.to(device)
    return scheduler


def build_step_sweep_schedule(
    *,
    step_count: int,
    device: torch.device | str = "cpu",
) -> deployment.DeploymentSchedule:
    raw_scheduler = build_step_sweep_scheduler(
        step_count=int(step_count),
        shift=1.0,
        device=device,
    )
    main_scheduler = build_step_sweep_scheduler(
        step_count=int(step_count),
        shift=DEFAULT_S_MAIN,
        device=device,
    )
    mcp_scheduler = build_step_sweep_scheduler(
        step_count=int(step_count),
        shift=DEFAULT_S_MCP,
        device=device,
    )
    schedule = deployment.DeploymentSchedule(
        raw_schedule=_float_tuple(raw_scheduler.timesteps),
        main_warped_schedule=_float_tuple(main_scheduler.timesteps),
        mcp_warped_schedule=_float_tuple(mcp_scheduler.timesteps),
    )
    validate_step_sweep_schedule(schedule, expected_step_count=int(step_count))
    return schedule


def validate_step_sweep_schedule(
    schedule: deployment.DeploymentSchedule,
    *,
    expected_step_count: int,
) -> None:
    count = _validate_step_count(expected_step_count)
    for label, values in (
        ("raw", schedule.raw_schedule),
        ("main", schedule.main_warped_schedule),
        ("mcp", schedule.mcp_warped_schedule),
    ):
        if len(values) != count:
            raise RuntimeError(f"{label} schedule length mismatch")
        _require_finite_descending(values, label=f"{label} schedule")
    if abs(float(schedule.raw_schedule[0]) - 1000.0) > 1.0e-6:
        raise RuntimeError("first sweep raw schedule must start at timestep 1000")
    if abs(float(schedule.main_warped_schedule[0]) - 1000.0) > 1.0e-6:
        raise RuntimeError("first sweep Main schedule must start at timestep 1000")
    if abs(float(schedule.mcp_warped_schedule[0]) - 1000.0) > 1.0e-6:
        raise RuntimeError("first sweep MCP schedule must start at timestep 1000")
    if count == 4:
        _require_close_tuple(schedule.raw_schedule, CANONICAL_RAW_4, "raw 4-step")
        _require_close_tuple(schedule.main_warped_schedule, CANONICAL_MAIN_4, "main 4-step")
        _require_close_tuple(schedule.mcp_warped_schedule, CANONICAL_MCP_4, "mcp 4-step")


def schedule_fingerprint(schedule: deployment.DeploymentSchedule) -> str:
    return deployment.canonical_json_sha256(
        {
            "raw_schedule": list(schedule.raw_schedule),
            "main_warped_schedule": list(schedule.main_warped_schedule),
            "mcp_warped_schedule": list(schedule.mcp_warped_schedule),
            "main_shift": DEFAULT_S_MAIN,
            "mcp_shift": DEFAULT_S_MCP,
            "sigma_min": 0.0,
            "extra_one_step": True,
            "num_train_timesteps": DEFAULT_NUM_TRAIN_TIMESTEPS,
        }
    )


def build_step_sweep_rng_plan(
    *,
    source_noise: torch.Tensor,
    rollout_seed: int,
    schedule: deployment.DeploymentSchedule,
    step_count: int,
    chunk_frames: int = deployment.FULL_SEQUENCE_CHUNK_FRAMES,
) -> dict[str, Any]:
    count = _validate_step_count(step_count)
    validate_step_sweep_schedule(schedule, expected_step_count=count)
    if source_noise.ndim != 5:
        raise ValueError("source_noise must have layout [B, F, C, H, W]")
    if int(source_noise.shape[1]) % int(chunk_frames) != 0:
        raise ValueError("source_noise frame count must be chunk-aligned")
    device = source_noise.device
    source_sha = tensor_sha256(source_noise.detach().cpu())
    num_chunks = int(source_noise.shape[1]) // int(chunk_frames)
    active_before = deployment.global_rng_state_hash(device)
    canonical_plan = deployment.build_absolute_chunk_rng_plan(
        source_noise=source_noise,
        rollout_seed=int(rollout_seed),
        num_denoising_steps=4,
        chunk_frames=int(chunk_frames),
    )
    compatibility = (
        dict(canonical_plan["trace"]["compatibility_draw"])
        if count == 4
        else _semantic_compatibility_draw(
            source_noise_sha256=source_sha,
            rollout_seed=int(rollout_seed),
            num_chunks=num_chunks,
            num_denoising_steps=count,
            device=device,
        )
    )
    canonical_transition_steps = {
        _destination_raw_timestep_tick(
            deployment.resolve_deployment_schedule(),
            step_index=step_index,
        ): step_index
        for step_index in range(len(deployment.RAW_DEPLOYMENT_SCHEDULE) - 1)
    }
    transitions: dict[tuple[int, int], torch.Tensor] = {}
    contexts: dict[int, torch.Tensor] = {}
    draws: list[dict[str, Any]] = []
    draw_order = 1
    for chunk_index in range(num_chunks):
        start = int(chunk_index) * int(chunk_frames)
        template = source_noise[:, start:start + int(chunk_frames)].flatten(0, 1)
        for step_index in range(count - 1):
            destination_tick = _destination_raw_timestep_tick(
                schedule,
                step_index=step_index,
            )
            canonical_step_index = canonical_transition_steps.get(destination_tick)
            if canonical_step_index is None:
                noise, record = _semantic_noise_like(
                    template,
                    source_noise_sha256=source_sha,
                    rollout_seed=int(rollout_seed),
                    purpose="transition_re_noise",
                    draw_order=draw_order,
                    chunk_index=chunk_index,
                    solver_step_index=step_index,
                    destination_raw_timestep_tick_32=destination_tick,
                )
            else:
                noise = canonical_plan["transition_noises"][
                    (int(chunk_index), int(canonical_step_index))
                ].detach().clone()
                record = _anchored_rng_record(
                    noise,
                    purpose="transition_re_noise",
                    draw_order=draw_order,
                    chunk_index=chunk_index,
                    solver_step_index=step_index,
                    destination_raw_timestep_tick_32=destination_tick,
                    canonical_4_solver_step_index=canonical_step_index,
                )
            record["absolute_chunk_index"] = int(chunk_index)
            transitions[(int(chunk_index), int(step_index))] = noise.detach().clone()
            draws.append(record)
            draw_order += 1
        noise = canonical_plan["context_noises"][int(chunk_index)].detach().clone()
        record = _anchored_rng_record(
            noise,
            purpose="context_clean_recache_noise",
            draw_order=draw_order,
            chunk_index=chunk_index,
            solver_step_index=None,
            destination_raw_timestep_tick_32=None,
            canonical_4_solver_step_index=None,
        )
        record["absolute_chunk_index"] = int(chunk_index)
        contexts[int(chunk_index)] = noise.detach().clone()
        draws.append(record)
        draw_order += 1
    active_after = deployment.global_rng_state_hash(device)
    if active_after != active_before:
        raise RuntimeError("step sweep semantic RNG plan changed active RNG state")
    coupling = _semantic_rng_coupling_contract()
    trace = {
        "schema": deployment.EVAL_RNG_PLAN_SCHEMA,
        "rollout_seed": int(rollout_seed),
        "source_noise_sha256": source_sha,
        "canonical_4_rng_plan_fingerprint_sha256": canonical_plan["trace"][
            "rng_plan_fingerprint_sha256"
        ],
        "post_reset_global_rng_state_hash": active_before,
        "active_rng_unchanged": True,
        "compatibility_draw": compatibility,
        "draws": draws,
        "draw_count": len(draws),
        "semantic_coupling": coupling,
    }
    trace["rng_plan_fingerprint_sha256"] = deployment.rng_plan_fingerprint(trace)
    return {
        "schema": deployment.EVAL_RNG_PLAN_SCHEMA,
        "semantic_coupling_schema": SEMANTIC_RNG_COUPLING_SCHEMA,
        "coupling_mode": CANONICAL_4_ANCHORED_RNG_MODE,
        "canonical_4_rng_preserved": True,
        "rollout_seed": int(rollout_seed),
        "num_chunks": int(num_chunks),
        "chunk_frames": int(chunk_frames),
        "num_denoising_steps": int(count),
        "source_noise_sha256": source_sha,
        "transition_noises": transitions,
        "context_noises": contexts,
        "trace": trace,
    }


def _semantic_rng_coupling_contract() -> dict[str, Any]:
    return {
        "schema": SEMANTIC_RNG_COUPLING_SCHEMA,
        "mode": CANONICAL_4_ANCHORED_RNG_MODE,
        "scope": "First-MCP denoising-step diagnostic sweep only",
        "canonical_deployment_rng_unchanged": True,
        "canonical_4_rng_preserved": True,
        "context_noise_key": [
            "canonical_4_context_clean_recache_noise",
            "chunk_index",
        ],
        "transition_noise_key": [
            "rollout_seed",
            "source_noise_sha256",
            "chunk_index",
            "destination_raw_timestep_tick_32",
        ],
        "raw_tick_denominator": SEMANTIC_RAW_TICK_DENOMINATOR,
        "canonical_4_anchor_destination_raw_timestep_ticks_32": [24, 16, 8],
        "raw_tick_unit_timestep": (
            float(DEFAULT_NUM_TRAIN_TIMESTEPS) / float(SEMANTIC_RAW_TICK_DENOMINATOR)
        ),
        "transition_alignment_semantics": (
            "Noise is anchored to the canonical 4-step RNG plan at raw 750, "
            "500, and 250; finer-only raw destinations use semantic keyed RNG."
        ),
    }


def _semantic_compatibility_draw(
    *,
    source_noise_sha256: str,
    rollout_seed: int,
    num_chunks: int,
    num_denoising_steps: int,
    device: torch.device | str,
) -> dict[str, Any]:
    key = {
        "schema": SEMANTIC_RNG_COUPLING_SCHEMA,
        "rollout_seed": int(rollout_seed),
        "source_noise_sha256": str(source_noise_sha256),
        "purpose": "teacher_exit_flag_randint_compatibility",
    }
    generator = _semantic_torch_generator(key, device=device)
    state_before = _generator_state_hash(generator)
    values = torch.randint(
        low=0,
        high=int(num_denoising_steps),
        size=(int(num_chunks),),
        device=device,
        dtype=torch.long,
        generator=generator,
    )
    state_after = _generator_state_hash(generator)
    return {
        "draw_order": 0,
        "purpose": "teacher_exit_flag_randint_compatibility",
        "operation": "semantic_keyed_torch.randint",
        "low": 0,
        "high": int(num_denoising_steps),
        "size": [int(num_chunks)],
        "dtype": str(values.dtype),
        "device": str(values.device),
        "state_before_hash": state_before,
        "state_after_hash": state_after,
        "semantic_rng_key": key,
        "values_sha256": tensor_sha256(values.detach().cpu()),
        "values_discarded": True,
    }


def _anchored_rng_record(
    noise: torch.Tensor,
    *,
    purpose: str,
    draw_order: int,
    chunk_index: int,
    solver_step_index: int | None,
    destination_raw_timestep_tick_32: int | None,
    canonical_4_solver_step_index: int | None,
) -> dict[str, Any]:
    return {
        "draw_order": int(draw_order),
        "purpose": str(purpose),
        "chunk_index": int(chunk_index),
        "solver_step_index": (
            None if solver_step_index is None else int(solver_step_index)
        ),
        "destination_raw_timestep_tick_32": destination_raw_timestep_tick_32,
        "canonical_4_anchor": True,
        "canonical_4_solver_step_index": (
            None
            if canonical_4_solver_step_index is None
            else int(canonical_4_solver_step_index)
        ),
        "noise": deployment.tensor_json_summary(noise),
    }


def _semantic_noise_like(
    template: torch.Tensor,
    *,
    source_noise_sha256: str,
    rollout_seed: int,
    purpose: str,
    draw_order: int,
    chunk_index: int,
    solver_step_index: int | None,
    destination_raw_timestep_tick_32: int | None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    semantic_key: dict[str, Any] = {
        "schema": SEMANTIC_RNG_COUPLING_SCHEMA,
        "rollout_seed": int(rollout_seed),
        "source_noise_sha256": str(source_noise_sha256),
        "purpose": str(purpose),
        "chunk_index": int(chunk_index),
    }
    if destination_raw_timestep_tick_32 is None:
        semantic_key["semantic_position"] = "context_clean_recache"
    else:
        semantic_key["destination_raw_timestep_tick_32"] = int(
            destination_raw_timestep_tick_32
        )
        semantic_key["raw_tick_denominator"] = SEMANTIC_RAW_TICK_DENOMINATOR
    generator = _semantic_torch_generator(semantic_key, device=template.device)
    state_before = _generator_state_hash(generator)
    noise = torch.randn(
        tuple(template.shape),
        device=template.device,
        dtype=template.dtype,
        generator=generator,
    )
    state_after = _generator_state_hash(generator)
    return noise, {
        "draw_order": int(draw_order),
        "purpose": str(purpose),
        "chunk_index": int(chunk_index),
        "solver_step_index": (
            None if solver_step_index is None else int(solver_step_index)
        ),
        "destination_raw_timestep_tick_32": destination_raw_timestep_tick_32,
        "semantic_rng_key": semantic_key,
        "state_before_hash": state_before,
        "state_after_hash": state_after,
        "noise": deployment.tensor_json_summary(noise),
    }


def _semantic_torch_generator(
    semantic_key: Mapping[str, Any],
    *,
    device: torch.device | str,
) -> torch.Generator:
    device = torch.device(device)
    generator_device = device if device.type == "cuda" else torch.device("cpu")
    generator = torch.Generator(device=generator_device)
    generator.manual_seed(_semantic_seed(semantic_key))
    return generator


def _semantic_seed(semantic_key: Mapping[str, Any]) -> int:
    digest = deployment.canonical_json_sha256(semantic_key)
    return int(digest[:16], 16)


def _generator_state_hash(generator: torch.Generator) -> str:
    return tensor_sha256(generator.get_state().detach().cpu())


def _destination_raw_timestep_tick(
    schedule: deployment.DeploymentSchedule,
    *,
    step_index: int,
) -> int:
    destination = float(schedule.raw_schedule[int(step_index) + 1])
    scaled = destination * float(SEMANTIC_RAW_TICK_DENOMINATOR) / float(
        DEFAULT_NUM_TRAIN_TIMESTEPS
    )
    tick = int(round(scaled))
    expected = (
        float(DEFAULT_NUM_TRAIN_TIMESTEPS)
        * float(tick)
        / float(SEMANTIC_RAW_TICK_DENOMINATOR)
    )
    if abs(destination - expected) > 1.0e-4:
        raise RuntimeError("step sweep raw destination timestep is off 32-tick grid")
    return tick


def run_first_mcp_step_sweep(
    *,
    runtime_factory: Callable[[], deployment.DeploymentRuntime],
    source_noise: torch.Tensor,
    teacher_target: torch.Tensor,
    teacher_payload: Mapping[str, Any],
    conditional_dict: Mapping[str, Any],
    checkpoint_summary: Mapping[str, Any],
    common_inputs: Mapping[str, Any],
    common_inputs_fingerprint_sha256: str,
    runtime_git_sha: str,
    training_checkpoint_git_sha: str,
    step_counts: Sequence[int] = DEFAULT_STEP_COUNTS,
    oracle_gate_tolerance: float = ORACLE_GATE_TOLERANCE,
) -> FirstMCPStepSweepResult:
    flow_audit._validate_source_and_teacher(source_noise, teacher_target)
    counts = _normalize_step_counts(step_counts)
    teacher_chunk2 = flow_audit._chunk(teacher_target, flow_audit.FUTURE_CHUNK_INDEX)
    runs: dict[str, Any] = {}
    tensor_runs: dict[str, Any] = {}
    hybrid_latents: dict[str, torch.Tensor] = {}
    schedule_records: dict[str, Any] = {}
    rng_records: dict[str, Any] = {}

    with torch.no_grad():
        for count in counts:
            schedule = build_step_sweep_schedule(
                step_count=count,
                device=source_noise.device,
            )
            main_transition_scheduler = build_step_sweep_scheduler(
                step_count=count,
                shift=DEFAULT_S_MAIN,
                device=source_noise.device,
            )
            mcp_scheduler = build_step_sweep_scheduler(
                step_count=count,
                shift=DEFAULT_S_MCP,
                device=source_noise.device,
            )
            rng_plan = build_step_sweep_rng_plan(
                source_noise=source_noise,
                rollout_seed=int(teacher_payload["rollout_seed"]),
                schedule=schedule,
                step_count=count,
                chunk_frames=deployment.FULL_SEQUENCE_CHUNK_FRAMES,
            )
            live = flow_audit._run_predicted_rollout(
                runtime=runtime_factory(),
                main_transition_scheduler=main_transition_scheduler,
                mcp_scheduler=mcp_scheduler,
                source_noise=source_noise,
                teacher_target=teacher_target,
                conditional_dict=conditional_dict,
                schedule=schedule,
                rng_plan=rng_plan,
            )
            oracle = flow_audit._run_oracle_flow_rollout(
                runtime=runtime_factory(),
                main_transition_scheduler=main_transition_scheduler,
                mcp_scheduler=mcp_scheduler,
                source_noise=source_noise,
                teacher_target=teacher_target,
                conditional_dict=conditional_dict,
                schedule=schedule,
                rng_plan=rng_plan,
            )
            flow_audit._require_main_trajectory_exact(
                live["main_trajectory"],
                oracle["main_trajectory"],
            )
            oracle["trace"]["main_trajectory_matches_predicted"] = True
            teacher_current = _run_teacher_current_predicted_mcp(
                runtime=runtime_factory(),
                main_transition_scheduler=main_transition_scheduler,
                mcp_scheduler=mcp_scheduler,
                source_noise=source_noise,
                teacher_target=teacher_target,
                conditional_dict=conditional_dict,
                schedule=schedule,
                rng_plan=rng_plan,
            )
            oracle_final_mse = flow_audit._mse(oracle["final_chunk2"], teacher_chunk2)
            oracle_gate_pass = bool(oracle_final_mse <= float(oracle_gate_tolerance))
            key = str(count)
            schedule_records[key] = _schedule_record(schedule, count=count)
            rng_records[key] = {
                "rng_plan_fingerprint_sha256": rng_plan["trace"][
                    "rng_plan_fingerprint_sha256"
                ],
                "draw_count": int(rng_plan["trace"]["draw_count"]),
                "num_denoising_steps": int(rng_plan["num_denoising_steps"]),
                "source_noise_sha256": str(rng_plan["source_noise_sha256"]),
                "semantic_coupling": dict(rng_plan["trace"]["semantic_coupling"]),
            }
            runs[key] = {
                "step_count": int(count),
                "schedule_fingerprint_sha256": schedule_records[key][
                    "schedule_fingerprint_sha256"
                ],
                "rng_plan_fingerprint_sha256": rng_records[key][
                    "rng_plan_fingerprint_sha256"
                ],
                "branches": {
                    LIVE_JOINT_PREDICTED: _predicted_branch_record(
                        live,
                        branch_name=LIVE_JOINT_PREDICTED,
                        teacher_chunk2=teacher_chunk2,
                    ),
                    TEACHER_CURRENT_PREDICTED_MCP: _predicted_branch_record(
                        teacher_current,
                        branch_name=TEACHER_CURRENT_PREDICTED_MCP,
                        teacher_chunk2=teacher_chunk2,
                    ),
                    ORACLE_FLOW: _oracle_branch_record(
                        oracle,
                        teacher_chunk2=teacher_chunk2,
                        gate_pass=oracle_gate_pass,
                        gate_tolerance=float(oracle_gate_tolerance),
                    ),
                },
            }
            tensor_runs[key] = {
                LIVE_JOINT_PREDICTED: _tensor_branch(live),
                TEACHER_CURRENT_PREDICTED_MCP: _tensor_branch(teacher_current),
                ORACLE_FLOW: _tensor_branch(oracle),
            }
            if count in (4, 32):
                hybrid_latents[f"hybrid_{count}step"] = (
                    flow_audit.build_chunk2_hybrid_latent(
                        teacher_target,
                        replacement_chunk=live["final_chunk2"],
                    )
                )

    step0_fairness = _step0_single_variable_fairness_report(runs)
    manifest = {
        "schema": FIRST_MCP_STEP_SWEEP_SCHEMA,
        "status": "PASS",
        "diagnostic_only": True,
        "non_deployable": True,
        "training_eligible": False,
        "canonical_training_eligible": False,
        "canonical_deployment_eligible": False,
        "runtime_git_sha": str(runtime_git_sha),
        "training_checkpoint_git_sha": str(training_checkpoint_git_sha),
        "checkpoint": dict(checkpoint_summary),
        "common_inputs_fingerprint_sha256": str(common_inputs_fingerprint_sha256),
        "common_inputs": dict(common_inputs),
        "conditional_fingerprint_sha256": str(
            common_inputs.get("conditioning_sha256", "")
        ),
        "step_counts_requested": [int(value) for value in step_counts],
        "step_counts": [int(value) for value in counts],
        "rng_coupling_contract": _semantic_rng_coupling_contract(),
        "schedule_by_step_count": schedule_records,
        "rng_plan_by_step_count": rng_records,
        "input_tensors": flow_audit._input_tensor_provenance(
            source_noise,
            teacher_target,
        ),
        "chunk_contract": {
            "history_chunks": [flow_audit.HISTORY_CHUNK_INDEX],
            "current_chunk": flow_audit.CURRENT_CHUNK_INDEX,
            "future_chunk": flow_audit.FUTURE_CHUNK_INDEX,
            "depths_used": [1],
            "forbidden_depths": [2, 3],
        },
        "branches_implemented": {
            LIVE_JOINT_PREDICTED: True,
            TEACHER_CURRENT_PREDICTED_MCP: True,
            ORACLE_FLOW: True,
        },
        "step0_single_variable_fairness": step0_fairness,
        "runs": runs,
        "preregistered_decision_rule": _preregistered_decision_rule(),
        "primary_decision": evaluate_few_step_decision(runs),
        "forbidden_features": {
            "mcp_depth2": False,
            "mcp_depth3": False,
            "verifier": False,
            "target_refinement": False,
            "routing": False,
            "dmd": False,
            "self_rollout": False,
            "training": False,
            "optimizer": False,
            "checkpoint_save": False,
            "video_decode_main_metric": False,
            "speed_benchmark": False,
        },
        "interpretation_limits": {
            "objective_or_capacity_confirmed": False,
            "failed_sweep_only_means": (
                "few-step inference is insufficient to explain the primary "
                "First-MCP failure under this fixed input and checkpoint"
            ),
        },
    }
    validate_first_mcp_step_sweep_manifest(manifest)
    tensors = {
        "schema": FIRST_MCP_STEP_SWEEP_TENSOR_SCHEMA,
        "teacher_chunk1": flow_audit._chunk(
            teacher_target,
            flow_audit.CURRENT_CHUNK_INDEX,
        ).detach().cpu(),
        "teacher_chunk2": teacher_chunk2.detach().cpu(),
        "runs": tensor_runs,
    }
    return FirstMCPStepSweepResult(
        manifest=manifest,
        tensors=tensors,
        hybrid_latents={key: value.detach().cpu() for key, value in hybrid_latents.items()},
    )


def evaluate_few_step_decision(runs: Mapping[str, Any]) -> dict[str, Any]:
    if not all(str(count) in runs for count in DEFAULT_STEP_COUNTS):
        return {
            "status": INVALID_ORACLE_GATE,
            "reason": "primary comparison requires 4, 8, 16, and 32-step runs",
        }
    oracle_gates = {
        str(count): bool(
            run["branches"][ORACLE_FLOW]["oracle_gate_pass"]
        )
        for count, run in runs.items()
    }
    if not all(oracle_gates.values()):
        return {
            "status": INVALID_ORACLE_GATE,
            "oracle_gates": oracle_gates,
        }
    live_improvements = _improvement_pct_by_step_count(
        runs,
        branch_name=LIVE_JOINT_PREDICTED,
    )
    teacher_current_improvements = _improvement_pct_by_step_count(
        runs,
        branch_name=TEACHER_CURRENT_PREDICTED_MCP,
    )
    live_32 = float(live_improvements["32"])
    teacher_current_32 = float(teacher_current_improvements["32"])
    teacher_current_direction_consistent = teacher_current_32 > 0.0
    live_has_any_10pp = any(value >= 10.0 for value in live_improvements.values())
    teacher_current_has_any_10pp = any(
        value >= 10.0 for value in teacher_current_improvements.values()
    )
    if live_32 >= 30.0 and teacher_current_direction_consistent:
        status = SUPPORT_FEW_STEP_INFERENCE_GAP
    elif (
        live_32 < 10.0
        and not live_has_any_10pp
        and teacher_current_32 < 10.0
        and not teacher_current_has_any_10pp
    ):
        status = NO_SUPPORT
    else:
        status = INCONCLUSIVE
    return {
        "status": status,
        "primary_branch": LIVE_JOINT_PREDICTED,
        "live_improvement_pct_by_step_count": live_improvements,
        "teacher_current_improvement_pct_by_step_count": teacher_current_improvements,
        "live_32_vs_4_improvement_pct": live_32,
        "teacher_current_32_vs_4_improvement_pct": teacher_current_32,
        "teacher_current_direction_consistent": bool(
            teacher_current_direction_consistent
        ),
        "improvement_pct": live_32,
        "support_threshold_pct": 30.0,
        "no_support_threshold_pct": 10.0,
        "oracle_gates": oracle_gates,
    }


def _improvement_pct_by_step_count(
    runs: Mapping[str, Any],
    *,
    branch_name: str,
) -> dict[str, float]:
    baseline = _branch_final_mse(runs, count=4, branch_name=branch_name)
    return {
        str(count): _improvement_pct(
            baseline,
            _branch_final_mse(runs, count=count, branch_name=branch_name),
        )
        for count in (8, 16, 32)
    }


def _branch_final_mse(
    runs: Mapping[str, Any],
    *,
    count: int,
    branch_name: str,
) -> float:
    value = float(
        runs[str(count)]["branches"][branch_name]["final_chunk2_mse_to_teacher"]
    )
    _require_finite_number(value, label=f"{branch_name} {count}-step final MSE")
    return value


def _improvement_pct(baseline: float, candidate: float) -> float:
    _require_finite_number(baseline, label="improvement baseline")
    _require_finite_number(candidate, label="improvement candidate")
    if baseline <= 0.0:
        return 0.0 if candidate <= baseline else -100.0
    return float((baseline - candidate) / baseline * 100.0)


def _step0_single_variable_fairness_report(
    runs: Mapping[str, Any],
) -> dict[str, Any]:
    keys = tuple(str(value) for value in DEFAULT_STEP_COUNTS)
    if set(runs.keys()) != set(keys):
        raise RuntimeError("step0 fairness requires all sweep step counts")
    live_ref = _live_step0_identity(runs["4"])
    teacher_ref = _teacher_current_step0_identity(runs["4"])
    per_count: dict[str, Any] = {}
    for key in keys:
        live = _live_step0_identity(runs[key])
        teacher_current = _teacher_current_step0_identity(runs[key])
        _require_step0_identity_exact(
            "live",
            count_key=key,
            reference=live_ref,
            candidate=live,
        )
        _require_step0_identity_exact(
            "teacher-current",
            count_key=key,
            reference=teacher_ref,
            candidate=teacher_current,
        )
        _require_step0_field_exact(
            "live-vs-teacher-current",
            count_key=key,
            field="current_input_sha256",
            reference=live["current_input_sha256"],
            candidate=teacher_current["current_input_sha256"],
        )
        _require_step0_field_exact(
            "live-vs-teacher-current",
            count_key=key,
            field="future_state_sha256",
            reference=live["future_state_sha256"],
            candidate=teacher_current["future_state_sha256"],
        )
        _require_step0_field_exact(
            "live-vs-teacher-current",
            count_key=key,
            field="main_warped_timestep",
            reference=live["main_warped_timestep"],
            candidate=teacher_current["main_warped_timestep"],
        )
        _require_step0_field_exact(
            "live-vs-teacher-current",
            count_key=key,
            field="mcp_warped_timestep",
            reference=live["mcp_warped_timestep"],
            candidate=teacher_current["mcp_warped_timestep"],
        )
        _require_step0_field_exact(
            "live-vs-teacher-current",
            count_key=key,
            field="history_kv_fingerprint_sha256",
            reference=live["history_kv_fingerprint_sha256"],
            candidate=teacher_current["history_kv_fingerprint_sha256"],
        )
        _require_step0_field_exact(
            "live-vs-teacher-current",
            count_key=key,
            field="crossattn_cache_fingerprint_sha256",
            reference=live["crossattn_cache_fingerprint_sha256"],
            candidate=teacher_current["crossattn_cache_fingerprint_sha256"],
        )
        _require_step0_field_exact(
            "live-vs-teacher-current",
            count_key=key,
            field="predicted_flow_sha256",
            reference=live["predicted_flow_sha256"],
            candidate=teacher_current["predicted_flow_sha256"],
        )
        per_count[key] = {
            "live_vs_teacher_current_current_input_exact": True,
            "live_vs_teacher_current_future_state_exact": True,
            "live_vs_teacher_current_timestep_exact": True,
            "live_vs_teacher_current_history_kv_exact": True,
            "live_vs_teacher_current_crossattn_cache_exact": True,
            "live_vs_teacher_current_predicted_flow_exact": True,
        }
    return {
        "step0_single_variable_fairness_gate_pass": True,
        "counts": list(DEFAULT_STEP_COUNTS),
        "live_reference": live_ref,
        "teacher_current_reference": teacher_ref,
        "cross_count_live_exact": True,
        "cross_count_teacher_current_exact": True,
        "within_count_live_vs_teacher_current_exact": True,
        "per_count": per_count,
    }


def _live_step0_identity(run: Mapping[str, Any]) -> dict[str, str]:
    branch = run["branches"][LIVE_JOINT_PREDICTED]
    step0 = branch["steps"][0]
    return {
        "current_input_sha256": str(step0["main"]["main_input_sha256"]),
        "future_state_sha256": str(step0["future_state"]["sha256"]),
        "main_warped_timestep": str(step0["main_warped_timestep"]),
        "mcp_warped_timestep": str(step0["mcp_warped_timestep"]),
        "history_kv_fingerprint_sha256": str(
            branch["history_kv_fingerprint_sha256"]
        ),
        "crossattn_cache_fingerprint_sha256": str(
            branch["crossattn_cache_fingerprint_sha256"]
        ),
        "predicted_flow_sha256": str(step0["predicted_flow"]["sha256"]),
    }


def _teacher_current_step0_identity(run: Mapping[str, Any]) -> dict[str, str]:
    branch = run["branches"][TEACHER_CURRENT_PREDICTED_MCP]
    step0 = branch["steps"][0]
    return {
        "current_input_sha256": str(step0["current_state"]["sha256"]),
        "future_state_sha256": str(step0["future_state"]["sha256"]),
        "main_warped_timestep": str(step0["main_warped_timestep"]),
        "mcp_warped_timestep": str(step0["mcp_warped_timestep"]),
        "history_kv_fingerprint_sha256": str(
            branch["history_kv_fingerprint_sha256"]
        ),
        "crossattn_cache_fingerprint_sha256": str(
            branch["crossattn_cache_fingerprint_sha256"]
        ),
        "predicted_flow_sha256": str(step0["predicted_flow"]["sha256"]),
    }


def _require_step0_identity_exact(
    branch_label: str,
    *,
    count_key: str,
    reference: Mapping[str, str],
    candidate: Mapping[str, str],
) -> None:
    for field, reference_value in reference.items():
        _require_step0_field_exact(
            branch_label,
            count_key=count_key,
            field=field,
            reference=reference_value,
            candidate=candidate.get(field),
        )


def _require_step0_field_exact(
    branch_label: str,
    *,
    count_key: str,
    field: str,
    reference: Any,
    candidate: Any,
) -> None:
    if candidate != reference:
        raise RuntimeError(
            f"{branch_label} step0 fairness mismatch for {count_key}-step "
            f"field {field}: reference={reference} candidate={candidate}"
        )


def _validate_checkpoint_summary(checkpoint: Any) -> None:
    if not isinstance(checkpoint, Mapping):
        raise RuntimeError("step sweep checkpoint summary missing")
    if int(checkpoint.get("global_step", -1)) != EXPECTED_STEP6500_GLOBAL_STEP:
        raise RuntimeError("step sweep checkpoint global_step must be 6500")
    load_mode = str(
        checkpoint.get(
            "load_mode",
            checkpoint.get("checkpoint_loader_mode", ""),
        )
    )
    if load_mode != EXPECTED_STEP6500_LOADER_MODE:
        raise RuntimeError("step sweep checkpoint loader mode mismatch")
    if not _is_sha256(checkpoint.get("sha256")):
        raise RuntimeError("step sweep checkpoint SHA invalid")


def validate_first_mcp_step_sweep_manifest(manifest: Mapping[str, Any]) -> None:
    if manifest.get("schema") != FIRST_MCP_STEP_SWEEP_SCHEMA:
        raise RuntimeError("first MCP step sweep schema mismatch")
    if manifest.get("status") != "PASS":
        raise RuntimeError("first MCP step sweep status must be PASS")
    if manifest.get("diagnostic_only") is not True:
        raise RuntimeError("first MCP step sweep must be diagnostic-only")
    if manifest.get("training_eligible") is not False:
        raise RuntimeError("first MCP step sweep must not be training eligible")
    if manifest.get("canonical_training_eligible") is not False:
        raise RuntimeError("first MCP step sweep must not be canonical training eligible")
    if manifest.get("canonical_deployment_eligible") is not False:
        raise RuntimeError("first MCP step sweep must not be canonical deployment eligible")
    requested = [int(value) for value in manifest.get("step_counts_requested", ())]
    if len(requested) != len(set(requested)):
        raise RuntimeError("step sweep contains duplicate requested step counts")
    if requested != list(DEFAULT_STEP_COUNTS):
        raise RuntimeError("step sweep requested counts must be exactly 4, 8, 16, 32")
    counts = [int(value) for value in manifest.get("step_counts", ())]
    if counts != list(DEFAULT_STEP_COUNTS):
        raise RuntimeError("step sweep must contain exactly 4, 8, 16, 32")
    schedules = manifest.get("schedule_by_step_count")
    rng_records = manifest.get("rng_plan_by_step_count")
    runs = manifest.get("runs")
    if (
        not isinstance(schedules, Mapping)
        or not isinstance(rng_records, Mapping)
        or not isinstance(runs, Mapping)
    ):
        raise RuntimeError("step sweep schedules/rng/runs missing")
    if set(schedules.keys()) != {str(value) for value in DEFAULT_STEP_COUNTS}:
        raise RuntimeError("step sweep schedule set mismatch")
    if set(rng_records.keys()) != {str(value) for value in DEFAULT_STEP_COUNTS}:
        raise RuntimeError("step sweep RNG plan set mismatch")
    if set(runs.keys()) != {str(value) for value in DEFAULT_STEP_COUNTS}:
        raise RuntimeError("step sweep run set mismatch")
    common_inputs = manifest.get("common_inputs")
    if not isinstance(common_inputs, Mapping):
        raise RuntimeError("step sweep common inputs missing")
    if str(manifest.get("common_inputs_fingerprint_sha256")) != (
        deployment.canonical_json_sha256(common_inputs)
    ):
        raise RuntimeError("step sweep common inputs fingerprint mismatch")
    if str(manifest.get("conditional_fingerprint_sha256")) != str(
        common_inputs.get("conditioning_sha256", "")
    ):
        raise RuntimeError("step sweep conditioning fingerprint mismatch")
    if common_inputs.get("fixed_identity_contract") is not True:
        raise RuntimeError("step sweep fixed identity contract missing")
    if str(common_inputs.get("sample_identity")) != str(
        common_inputs.get("fixed_decode_validation_identity")
    ):
        raise RuntimeError("step sweep fixed identity mismatch")
    if dict(manifest.get("rng_coupling_contract") or {}) != (
        _semantic_rng_coupling_contract()
    ):
        raise RuntimeError("step sweep RNG coupling contract mismatch")
    input_tensors = manifest.get("input_tensors")
    if not isinstance(input_tensors, Mapping):
        raise RuntimeError("step sweep input tensor provenance missing")
    source_sha = str(input_tensors.get("source_noise_sha256", ""))
    if not _is_sha256(source_sha):
        raise RuntimeError("step sweep source noise SHA invalid")
    _validate_checkpoint_summary(manifest.get("checkpoint"))
    for count in DEFAULT_STEP_COUNTS:
        key = str(count)
        schedule = schedules[key]
        if not isinstance(schedule, Mapping):
            raise RuntimeError("step sweep schedule entry must be a mapping")
        _validate_schedule_record(schedule, expected_step_count=count)
        rng = rng_records[key]
        if not isinstance(rng, Mapping):
            raise RuntimeError("step sweep RNG plan entry must be a mapping")
        if int(rng.get("num_denoising_steps", -1)) != count:
            raise RuntimeError("step sweep RNG num_denoising_steps mismatch")
        if str(rng.get("source_noise_sha256", "")) != source_sha:
            raise RuntimeError("step sweep RNG source noise SHA mismatch")
        if not _is_sha256(rng.get("rng_plan_fingerprint_sha256")):
            raise RuntimeError("step sweep RNG fingerprint invalid")
        if dict(rng.get("semantic_coupling") or {}) != (
            _semantic_rng_coupling_contract()
        ):
            raise RuntimeError("step sweep semantic RNG coupling mismatch")
        run = runs[key]
        if not isinstance(run, Mapping):
            raise RuntimeError("step sweep run entry must be a mapping")
        if int(run.get("step_count", -1)) != count:
            raise RuntimeError("step sweep run step_count mismatch")
        if str(run.get("schedule_fingerprint_sha256")) != str(
            schedule.get("schedule_fingerprint_sha256")
        ):
            raise RuntimeError("step sweep run/schedule fingerprint mismatch")
        if str(run.get("rng_plan_fingerprint_sha256")) != str(
            rng.get("rng_plan_fingerprint_sha256")
        ):
            raise RuntimeError("step sweep run/RNG fingerprint mismatch")
        branches = run.get("branches")
        if not isinstance(branches, Mapping):
            raise RuntimeError("step sweep branches missing")
        if set(branches.keys()) != {
            LIVE_JOINT_PREDICTED,
            TEACHER_CURRENT_PREDICTED_MCP,
            ORACLE_FLOW,
        }:
            raise RuntimeError("step sweep branch set mismatch")
        for branch_name in (LIVE_JOINT_PREDICTED, TEACHER_CURRENT_PREDICTED_MCP):
            _validate_predicted_branch(
                branches[branch_name],
                branch_name=branch_name,
                expected_step_count=count,
            )
        _validate_oracle_branch(branches[ORACLE_FLOW], expected_step_count=count)
    decision = manifest.get("primary_decision")
    if not isinstance(decision, Mapping):
        raise RuntimeError("step sweep primary decision missing")
    if decision.get("status") not in {
        SUPPORT_FEW_STEP_INFERENCE_GAP,
        NO_SUPPORT,
        INCONCLUSIVE,
    }:
        raise RuntimeError("step sweep primary decision invalid")
    expected_decision = evaluate_few_step_decision(runs)
    if dict(decision) != dict(expected_decision):
        raise RuntimeError("step sweep primary decision mismatch")
    expected_fairness = _step0_single_variable_fairness_report(runs)
    if dict(manifest.get("step0_single_variable_fairness") or {}) != dict(
        expected_fairness
    ):
        raise RuntimeError("step sweep step0 fairness mismatch")


def _run_teacher_current_predicted_mcp(
    *,
    runtime: deployment.DeploymentRuntime,
    main_transition_scheduler: Any,
    mcp_scheduler: Any,
    source_noise: torch.Tensor,
    teacher_target: torch.Tensor,
    conditional_dict: Mapping[str, Any],
    schedule: deployment.DeploymentSchedule,
    rng_plan: Mapping[str, Any],
) -> dict[str, Any]:
    history_recache = flow_audit._recache_teacher_history0(
        runtime=runtime,
        source_noise=source_noise,
        teacher_target=teacher_target,
        conditional_dict=conditional_dict,
        rng_plan=rng_plan,
    )
    teacher_chunk1 = flow_audit._chunk(teacher_target, flow_audit.CURRENT_CHUNK_INDEX)
    teacher_chunk2 = flow_audit._chunk(teacher_target, flow_audit.FUTURE_CHUNK_INDEX)
    future_state = flow_audit._chunk(
        source_noise,
        flow_audit.FUTURE_CHUNK_INDEX,
    ).detach().clone()
    trace_steps: list[dict[str, Any]] = []
    tensor_steps: list[dict[str, torch.Tensor]] = []
    final_chunk2 = future_state
    for step_index, (raw_t, main_t, mcp_t) in enumerate(
        zip(
            schedule.raw_schedule,
            schedule.main_warped_schedule,
            schedule.mcp_warped_schedule,
        )
    ):
        current_noise = flow_audit._state_noise_for_step(
            rng_plan,
            source_noise=source_noise,
            chunk_index=flow_audit.CURRENT_CHUNK_INDEX,
            step_index=step_index,
            template=teacher_chunk1,
        )
        current_state = flow_audit._add_noise_chunk(
            main_transition_scheduler,
            clean=teacher_chunk1,
            noise=current_noise,
            timestep=flow_audit._timestep(float(main_t), teacher_chunk1),
        )
        future_input = future_state.detach().clone()
        main_flow, main_x0, mcp_flow, call_record = flow_audit._call_joint_depth1(
            runtime=runtime,
            conditional_dict=conditional_dict,
            current_state=current_state,
            future_state=future_state,
            current_start_frame=(
                flow_audit.CURRENT_CHUNK_INDEX
                * deployment.FULL_SEQUENCE_CHUNK_FRAMES
            ),
            future_start_frame=(
                flow_audit.FUTURE_CHUNK_INDEX
                * deployment.FULL_SEQUENCE_CHUNK_FRAMES
            ),
            main_timestep_value=float(main_t),
            mcp_timestep_value=float(mcp_t),
        )
        mcp_timestep = flow_audit._timestep(float(mcp_t), future_state)
        teacher_directed_flow, implied_noise, sigma = (
            flow_audit.teacher_directed_flow_for_state(
                mcp_scheduler,
                teacher_x0=teacher_chunk2,
                future_state=future_state,
                timestep=mcp_timestep,
            )
        )
        predicted_x0 = flow_audit._flow_to_x0_chunk(
            mcp_scheduler,
            flow=mcp_flow,
            state=future_state,
            timestep=mcp_timestep,
        )
        transition_state = None
        if step_index < len(schedule.raw_schedule) - 1:
            next_mcp_t = float(schedule.mcp_warped_schedule[step_index + 1])
            future_noise = flow_audit._transition_noise(
                rng_plan,
                chunk_index=flow_audit.FUTURE_CHUNK_INDEX,
                step_index=step_index,
                template=predicted_x0,
            )
            future_state = flow_audit._add_noise_chunk(
                mcp_scheduler,
                clean=predicted_x0,
                noise=future_noise,
                timestep=flow_audit._timestep(next_mcp_t, predicted_x0),
            )
            transition_state = flow_audit._tensor_record(future_state)
        else:
            final_chunk2 = predicted_x0.detach().clone()
        trace_steps.append(
            {
                "raw_index": int(step_index),
                "raw_timestep": float(raw_t),
                "main_warped_timestep": float(main_t),
                "mcp_warped_timestep": float(mcp_t),
                "resolved_sigma": sigma,
                "current_state_source": "teacher_corrupted_current_chunk1",
                "current_noise": flow_audit._tensor_record(current_noise),
                "current_state": flow_audit._tensor_record(current_state),
                "main_predicted_x0": flow_audit._tensor_record(main_x0),
                "main_predicted_x0_ignored_for_next_current": True,
                "future_state": flow_audit._tensor_record(future_input),
                "future_state_serial_predicted": True,
                "predicted_flow": flow_audit._tensor_record(mcp_flow),
                "teacher_directed_oracle_flow": flow_audit._tensor_record(
                    teacher_directed_flow
                ),
                "teacher_directed_target_source": "actual_state_implied_noise",
                "implied_noise": flow_audit._tensor_record(implied_noise),
                "predicted_flow_vs_teacher_directed_flow_mse": flow_audit._mse(
                    mcp_flow,
                    teacher_directed_flow,
                ),
                "predicted_x0_vs_teacher_mse": flow_audit._mse(
                    predicted_x0,
                    teacher_chunk2,
                ),
                "predicted_x0": flow_audit._tensor_record(predicted_x0),
                "transition_state": transition_state,
                "depths_requested": [1],
                "joint_forward_rng": call_record["joint_forward_rng"],
            }
        )
        tensor_steps.append(
            {
                "current_state": current_state.detach().clone(),
                "current_noise": current_noise.detach().clone(),
                "main_predicted_flow": main_flow.detach().clone(),
                "main_predicted_x0": main_x0.detach().clone(),
                "predicted_flow": mcp_flow.detach().clone(),
                "teacher_directed_flow": teacher_directed_flow.detach().clone(),
                "predicted_x0": predicted_x0.detach().clone(),
            }
        )
    return {
        "trace": {
            "mode": TEACHER_CURRENT_PREDICTED_MCP,
            "steps": trace_steps,
            "history_recache": history_recache,
            "history_kv_fingerprint_sha256": history_recache[
                "history_kv_fingerprint_sha256"
            ],
            "crossattn_cache_fingerprint_sha256": history_recache[
                "crossattn_cache_fingerprint_sha256"
            ],
            "mcp_depths_used": [1],
            "used_model_flow_for_transition": True,
            "used_exact_flow_for_transition": False,
            "history_chunks": [flow_audit.HISTORY_CHUNK_INDEX],
            "current_chunk": flow_audit.CURRENT_CHUNK_INDEX,
            "future_chunk": flow_audit.FUTURE_CHUNK_INDEX,
            "current_state_each_step": "teacher_corrupted_current_chunk1",
            "current_state_depends_on_previous_main_x0": False,
            "future_state_depends_on_previous_mcp_x0": True,
        },
        "tensor_steps": tensor_steps,
        "final_chunk2": final_chunk2.detach().clone(),
    }


def _predicted_branch_record(
    rollout: Mapping[str, Any],
    *,
    branch_name: str,
    teacher_chunk2: torch.Tensor,
) -> dict[str, Any]:
    trace = dict(rollout["trace"])
    trace["mode"] = str(branch_name)
    final_chunk2 = rollout["final_chunk2"]
    first = trace["steps"][0]
    return {
        **trace,
        "final_chunk2_mse_to_teacher": flow_audit._mse(final_chunk2, teacher_chunk2),
        "final_chunk2_sha256": tensor_sha256(final_chunk2.detach().cpu()),
        "first_step": {
            "predicted_flow_vs_teacher_directed_flow_mse": first[
                "predicted_flow_vs_teacher_directed_flow_mse"
            ],
            "predicted_x0_vs_teacher_mse": first["predicted_x0_vs_teacher_mse"],
        },
    }


def _oracle_branch_record(
    rollout: Mapping[str, Any],
    *,
    teacher_chunk2: torch.Tensor,
    gate_pass: bool,
    gate_tolerance: float,
) -> dict[str, Any]:
    trace = dict(rollout["trace"])
    trace["mode"] = ORACLE_FLOW
    final_chunk2 = rollout["final_chunk2"]
    final_mse = flow_audit._mse(final_chunk2, teacher_chunk2)
    return {
        **trace,
        "final_oracle_chunk2_mse_to_teacher": final_mse,
        "final_chunk2_sha256": tensor_sha256(final_chunk2.detach().cpu()),
        "oracle_gate_pass": bool(gate_pass),
        "oracle_gate_tolerance": float(gate_tolerance),
        "exact_flow_contract": {
            "flow_source": "scheduler.training_target(teacher_chunk2, planned_or_implied_noise)",
            "flow_to_x0": "scheduler.step(..., to_final=True)",
            "transition": "next-step re-noise with FlowMatchScheduler.add_noise",
        },
    }


def _tensor_branch(rollout: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "final_chunk2": rollout["final_chunk2"].detach().cpu(),
        "steps": [
            {
                key: value.detach().cpu()
                for key, value in step.items()
                if torch.is_tensor(value)
            }
            for step in rollout["tensor_steps"]
        ],
    }


def _schedule_record(
    schedule: deployment.DeploymentSchedule,
    *,
    count: int,
) -> dict[str, Any]:
    raw_sigmas = [float(value) / DEFAULT_NUM_TRAIN_TIMESTEPS for value in schedule.raw_schedule]
    main_sigmas = [
        float(value) / DEFAULT_NUM_TRAIN_TIMESTEPS
        for value in schedule.main_warped_schedule
    ]
    mcp_sigmas = [
        float(value) / DEFAULT_NUM_TRAIN_TIMESTEPS
        for value in schedule.mcp_warped_schedule
    ]
    return {
        "step_count": int(count),
        "raw_schedule": list(schedule.raw_schedule),
        "main_warped_schedule": list(schedule.main_warped_schedule),
        "mcp_warped_schedule": list(schedule.mcp_warped_schedule),
        "raw_sigma_schedule": raw_sigmas,
        "main_warped_sigma_schedule": main_sigmas,
        "mcp_warped_sigma_schedule": mcp_sigmas,
        "main_shift": DEFAULT_S_MAIN,
        "mcp_shift": DEFAULT_S_MCP,
        "sigma_min": 0.0,
        "extra_one_step": True,
        "schedule_fingerprint_sha256": schedule_fingerprint(schedule),
    }


def _validate_schedule_record(
    record: Mapping[str, Any],
    *,
    expected_step_count: int,
) -> None:
    schedule = deployment.DeploymentSchedule(
        raw_schedule=tuple(float(value) for value in record.get("raw_schedule", ())),
        main_warped_schedule=tuple(
            float(value) for value in record.get("main_warped_schedule", ())
        ),
        mcp_warped_schedule=tuple(
            float(value) for value in record.get("mcp_warped_schedule", ())
        ),
    )
    validate_step_sweep_schedule(schedule, expected_step_count=expected_step_count)
    if str(record.get("schedule_fingerprint_sha256")) != schedule_fingerprint(schedule):
        raise RuntimeError("step sweep schedule fingerprint mismatch")


def _validate_predicted_branch(
    branch: Mapping[str, Any],
    *,
    branch_name: str,
    expected_step_count: int,
) -> None:
    if branch.get("mode") != branch_name:
        raise RuntimeError(f"{branch_name} mode mismatch")
    if branch.get("mcp_depths_used") != [1]:
        raise RuntimeError(f"{branch_name} must use MCP depth1 only")
    if branch.get("used_model_flow_for_transition") is not True:
        raise RuntimeError(f"{branch_name} must use model flow for transition")
    _validate_steps(
        branch.get("steps"),
        expected_step_count=expected_step_count,
        branch_name=branch_name,
        predicted=True,
    )
    _require_finite_number(
        branch.get("final_chunk2_mse_to_teacher"),
        label=f"{branch_name} final MSE",
    )
    if not _is_sha256(branch.get("final_chunk2_sha256")):
        raise RuntimeError(f"{branch_name} final chunk2 SHA invalid")
    if not _is_sha256(branch.get("history_kv_fingerprint_sha256")):
        raise RuntimeError(f"{branch_name} history KV fingerprint invalid")
    if not _is_sha256(branch.get("crossattn_cache_fingerprint_sha256")):
        raise RuntimeError(f"{branch_name} cross-attn cache fingerprint invalid")
    if branch_name == TEACHER_CURRENT_PREDICTED_MCP:
        if branch.get("current_state_depends_on_previous_main_x0") is not False:
            raise RuntimeError("teacher-current branch must not depend on previous Main x0")
        if branch.get("future_state_depends_on_previous_mcp_x0") is not True:
            raise RuntimeError("teacher-current branch must keep serial MCP future state")


def _validate_oracle_branch(
    branch: Mapping[str, Any],
    *,
    expected_step_count: int,
) -> None:
    if branch.get("mode") != ORACLE_FLOW:
        raise RuntimeError("oracle branch mode mismatch")
    if branch.get("mcp_depths_used") != [1]:
        raise RuntimeError("oracle branch must use MCP depth1 only")
    if branch.get("used_exact_flow_for_transition") is not True:
        raise RuntimeError("oracle branch must use exact flow for transition")
    if branch.get("main_trajectory_matches_predicted") is not True:
        raise RuntimeError("oracle Main trajectory must match live predicted")
    _validate_steps(
        branch.get("steps"),
        expected_step_count=expected_step_count,
        branch_name=ORACLE_FLOW,
        predicted=False,
    )
    _require_finite_number(
        branch.get("final_oracle_chunk2_mse_to_teacher"),
        label="oracle final MSE",
    )
    if branch.get("oracle_gate_pass") is not True:
        raise RuntimeError("oracle gate failure")
    if not _is_sha256(branch.get("final_chunk2_sha256")):
        raise RuntimeError("oracle final chunk2 SHA invalid")
    if not _is_sha256(branch.get("history_kv_fingerprint_sha256")):
        raise RuntimeError("oracle history KV fingerprint invalid")
    if not _is_sha256(branch.get("crossattn_cache_fingerprint_sha256")):
        raise RuntimeError("oracle cross-attn cache fingerprint invalid")


def _validate_steps(
    steps: Any,
    *,
    expected_step_count: int,
    branch_name: str,
    predicted: bool,
) -> None:
    if not isinstance(steps, Sequence) or isinstance(steps, (str, bytes, bytearray)):
        raise RuntimeError(f"{branch_name} steps missing")
    if len(steps) != int(expected_step_count):
        raise RuntimeError(f"{branch_name} step count mismatch")
    for index, step in enumerate(steps):
        if not isinstance(step, Mapping):
            raise RuntimeError(f"{branch_name} step must be a mapping")
        if int(step.get("raw_index", -1)) != index:
            raise RuntimeError(f"{branch_name} raw index mismatch")
        for field in ("raw_timestep", "main_warped_timestep", "mcp_warped_timestep"):
            _require_finite_number(step.get(field), label=f"{branch_name} {field}")
        if predicted:
            for field in (
                "resolved_sigma",
                "predicted_flow_vs_teacher_directed_flow_mse",
                "predicted_x0_vs_teacher_mse",
            ):
                _require_finite_number(step.get(field), label=f"{branch_name} {field}")
            if not _is_sha256(step.get("future_state", {}).get("sha256")):
                raise RuntimeError(f"{branch_name} future state SHA invalid")
        else:
            for field in (
                "resolved_sigma",
                "model_predicted_flow_vs_exact_training_target_mse",
                "model_predicted_x0_vs_teacher_mse",
                "oracle_x0_vs_teacher_mse",
            ):
                _require_finite_number(step.get(field), label=f"{branch_name} {field}")
        flow_audit._validate_joint_forward_rng_guard(
            step.get("joint_forward_rng"),
            branch=branch_name,
            step_index=index,
        )


def _preregistered_decision_rule() -> dict[str, Any]:
    return {
        "primary_comparison": (
            "live_joint_predicted final_chunk2_mse_to_teacher, 32-step vs 4-step"
        ),
        "improvement_pct_formula": "(mse4 - mseN) / mse4 * 100",
        "support_few_step_inference_gap": {
            "live_32_vs_4_minimum_improvement_pct": 30.0,
            "requires_all_oracle_gates_pass": True,
            "teacher_current_32_vs_4_must_be_positive": True,
            "trend_preference": "8/16/32 generally improve, strict monotonicity not required",
        },
        "no_support": {
            "requires_all_oracle_gates_pass": True,
            "live_32_vs_4_maximum_improvement_pct_exclusive": 10.0,
            "live_any_8_16_32_improvement_at_least_10pct_forbidden": True,
            "teacher_current_32_vs_4_maximum_improvement_pct_exclusive": 10.0,
            "teacher_current_any_8_16_32_improvement_at_least_10pct_forbidden": True,
        },
        "inconclusive_range_pct": [10.0, 30.0],
        "intermediate_step_losses_are_not_primary": True,
        "timing_is_not_a_speed_benchmark": True,
    }


def _normalize_step_counts(values: Sequence[int]) -> tuple[int, ...]:
    counts = tuple(int(value) for value in values)
    if len(counts) != len(set(counts)):
        raise ValueError("duplicate step counts are not allowed")
    if counts != tuple(sorted(counts)):
        raise ValueError("step counts must be sorted ascending")
    if counts != DEFAULT_STEP_COUNTS:
        raise ValueError("step sweep supports exactly (4, 8, 16, 32)")
    for count in counts:
        _validate_step_count(count)
    return counts


def _validate_step_count(value: int) -> int:
    count = int(value)
    if count not in DEFAULT_STEP_COUNTS:
        raise ValueError("step_count must be one of 4, 8, 16, 32")
    return count


def _float_tuple(tensor: torch.Tensor) -> tuple[float, ...]:
    return tuple(float(value) for value in tensor.detach().cpu().tolist())


def _require_finite_descending(values: Sequence[float], *, label: str) -> None:
    previous = None
    for value in values:
        _require_finite_number(value, label=label)
        if previous is not None and not float(value) < float(previous):
            raise RuntimeError(f"{label} must be strictly descending")
        previous = float(value)


def _require_close_tuple(
    actual: Sequence[float],
    expected: Sequence[float],
    label: str,
    *,
    tolerance: float = 1.0e-4,
) -> None:
    if len(actual) != len(expected):
        raise RuntimeError(f"{label} length mismatch")
    for left, right in zip(actual, expected):
        if abs(float(left) - float(right)) > float(tolerance):
            raise RuntimeError(f"{label} mismatch")


def _require_finite_number(value: Any, *, label: str) -> None:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{label} must be numeric") from exc
    if not bool(torch.isfinite(torch.tensor(numeric, dtype=torch.float64)).item()):
        raise RuntimeError(f"{label} must be finite")


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    text = value.lower()
    return len(text) == 64 and all(char in "0123456789abcdef" for char in text)


__all__ = [
    "DEFAULT_STEP_COUNTS",
    "FIRST_MCP_STEP_SWEEP_SCHEMA",
    "FIRST_MCP_STEP_SWEEP_TENSOR_SCHEMA",
    "INCONCLUSIVE",
    "INVALID_ORACLE_GATE",
    "LIVE_JOINT_PREDICTED",
    "NO_SUPPORT",
    "ORACLE_FLOW",
    "SUPPORT_FEW_STEP_INFERENCE_GAP",
    "TEACHER_CURRENT_PREDICTED_MCP",
    "FirstMCPStepSweepResult",
    "build_step_sweep_rng_plan",
    "build_step_sweep_schedule",
    "build_step_sweep_scheduler",
    "evaluate_few_step_decision",
    "run_first_mcp_step_sweep",
    "schedule_fingerprint",
    "validate_first_mcp_step_sweep_manifest",
    "validate_step_sweep_schedule",
]
