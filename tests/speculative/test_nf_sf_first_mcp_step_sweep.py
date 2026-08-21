from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import scripts.run_nf_sf_first_mcp_step_sweep as runner
import utils.nf_sf_first_mcp_flow_audit as flow_audit
import utils.nf_sf_first_mcp_step_sweep as sweep
import utils.nf_sf_full_sequence_eval as ev
from utils.nf_sf_tensors import DEFAULT_S_MAIN, DEFAULT_S_MCP


RUNTIME_GIT_SHA = "d" * 40
TRAINING_GIT_SHA = "c3f89888bf6da31b48650f0a680dd6534943f56f"
TEST_SHA = "a" * 64
FRAME_SEQ_LENGTH = 2


class BaseSweepGenerator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[dict] = []
        self.mcp_call_count = 0

    def _record_and_update_kv(self, kwargs) -> tuple[torch.Tensor, bool]:
        current = kwargs["noisy_image_or_video"]
        current_start = int(kwargs["current_start"])
        mcp_requested = kwargs.get("mcp_future_noises") is not None
        if mcp_requested:
            self.mcp_call_count += 1
        token_count = int(current.shape[1]) * FRAME_SEQ_LENGTH
        token_end = current_start + token_count
        self.calls.append(
            {
                "mcp_requested": mcp_requested,
                "mcp_future_count": (
                    None
                    if kwargs.get("mcp_future_noises") is None
                    else len(kwargs["mcp_future_noises"])
                ),
                "current_start": current_start,
                "token_end": token_end,
            }
        )
        for layer in kwargs["kv_cache"]:
            layer["k"][:, current_start:token_end] = 1.0 + len(self.calls)
            layer["v"][:, current_start:token_end] = 2.0 + len(self.calls)
            layer["global_end_index"].fill_(token_end)
            layer["local_end_index"].fill_(token_end)
        return current, mcp_requested


class RegressionGenerator(BaseSweepGenerator):
    def forward(self, **kwargs):
        current, mcp_requested = self._record_and_update_kv(kwargs)
        main_flow = torch.zeros_like(current)
        main_x0 = current
        if not mcp_requested:
            return main_flow, main_x0
        future = kwargs["mcp_future_noises"][0]
        return main_flow, main_x0, [future * 0.125 + 0.25]


class ExactVectorFieldGenerator(BaseSweepGenerator):
    def __init__(self, teacher_target: torch.Tensor) -> None:
        super().__init__()
        self.teacher_chunk1 = flow_audit._chunk(
            teacher_target,
            flow_audit.CURRENT_CHUNK_INDEX,
        )
        self.teacher_chunk2 = flow_audit._chunk(
            teacher_target,
            flow_audit.FUTURE_CHUNK_INDEX,
        )

    def forward(self, **kwargs):
        current, mcp_requested = self._record_and_update_kv(kwargs)
        if not mcp_requested:
            return torch.zeros_like(current), current
        main_timestep = kwargs["timestep"]
        future = kwargs["mcp_future_noises"][0]
        mcp_timestep = kwargs["mcp_timesteps"][0]
        main_x0 = self.teacher_chunk1.to(device=current.device, dtype=current.dtype)
        future_x0 = self.teacher_chunk2.to(device=future.device, dtype=future.dtype)
        main_flow = _exact_flow_from_state(current, main_x0, main_timestep)
        mcp_flow = _exact_flow_from_state(future, future_x0, mcp_timestep)
        return main_flow, main_x0, [mcp_flow]


class BadConstantGenerator(BaseSweepGenerator):
    def forward(self, **kwargs):
        current, mcp_requested = self._record_and_update_kv(kwargs)
        main_flow = torch.zeros_like(current)
        main_x0 = current
        if not mcp_requested:
            return main_flow, main_x0
        future = kwargs["mcp_future_noises"][0]
        return main_flow, main_x0, [torch.full_like(future, 100.0)]


class MainDriftGenerator(BaseSweepGenerator):
    def forward(self, **kwargs):
        current, mcp_requested = self._record_and_update_kv(kwargs)
        main_flow = torch.zeros_like(current)
        main_x0 = current + 99.0
        if not mcp_requested:
            return main_flow, main_x0
        future = kwargs["mcp_future_noises"][0]
        return main_flow, main_x0, [torch.zeros_like(future)]


def make_runtime(generator: nn.Module) -> ev.DeploymentRuntime:
    capacity = ev.FULL_SEQUENCE_FRAME_COUNT * FRAME_SEQ_LENGTH
    kv_cache = [
        {
            "k": torch.zeros((1, capacity, 1, 1), dtype=torch.float32),
            "v": torch.zeros((1, capacity, 1, 1), dtype=torch.float32),
            "global_end_index": torch.tensor([0], dtype=torch.long),
            "local_end_index": torch.tensor([0], dtype=torch.long),
        }
        for _ in range(2)
    ]
    return ev.DeploymentRuntime(
        generator=generator,
        scheduler=flow_audit.build_flow_match_scheduler(
            shift=DEFAULT_S_MAIN,
            device=torch.device("cpu"),
        ),
        kv_cache=kv_cache,
        crossattn_cache=[{"is_init": False}],
        frame_seq_length=FRAME_SEQ_LENGTH,
        num_frame_per_block=ev.FULL_SEQUENCE_CHUNK_FRAMES,
        context_noise=0,
    )


def make_source_noise() -> torch.Tensor:
    return torch.linspace(
        -1.0,
        1.0,
        ev.FULL_SEQUENCE_FRAME_COUNT,
        dtype=torch.float32,
    ).reshape(1, ev.FULL_SEQUENCE_FRAME_COUNT, 1, 1, 1)


def make_teacher_target() -> torch.Tensor:
    return torch.linspace(
        0.25,
        2.25,
        ev.FULL_SEQUENCE_FRAME_COUNT,
        dtype=torch.float32,
    ).reshape(1, ev.FULL_SEQUENCE_FRAME_COUNT, 1, 1, 1)


def make_common(source_noise: torch.Tensor, teacher_target: torch.Tensor) -> tuple[dict, str]:
    common = {
        "schema": ev.EVAL_COMMON_INPUTS_SCHEMA,
        "sample_identity": "validation-0",
        "fixed_identity_contract": True,
        "fixed_decode_validation_identity": "validation-0",
        "runtime_git_sha": RUNTIME_GIT_SHA,
        "training_checkpoint_git_sha": TRAINING_GIT_SHA,
        "source_noise_sha256": ev.tensor_sha256(source_noise),
        "conditioning_sha256": ev.conditioning_json_summary(
            {"prompt_embeds": torch.zeros((1, 2, 3))}
        )["sha256"],
        "teacher_target_sha256": ev.tensor_sha256(teacher_target),
        "sample_plan_sha256": "b" * 64,
        "teacher_manifest_sha256": "c" * 64,
    }
    return common, ev.canonical_json_sha256(common)


def run_fake_sweep(generator: nn.Module) -> sweep.FirstMCPStepSweepResult:
    source = make_source_noise()
    teacher = make_teacher_target()
    common, fingerprint = make_common(source, teacher)
    return sweep.run_first_mcp_step_sweep(
        runtime_factory=lambda: make_runtime(generator),
        source_noise=source,
        teacher_target=teacher,
        teacher_payload={"rollout_seed": 123, "prompt": "prompt"},
        conditional_dict={"prompt_embeds": torch.zeros((1, 2, 3))},
        checkpoint_summary={
            "type": "full_sequence_step6500",
            "sha256": TEST_SHA,
            "global_step": 6500,
            "load_mode": "DIAGNOSTIC_INTERMEDIATE_STRICT",
        },
        common_inputs=common,
        common_inputs_fingerprint_sha256=fingerprint,
        runtime_git_sha=RUNTIME_GIT_SHA,
        training_checkpoint_git_sha=TRAINING_GIT_SHA,
    )


def _exact_flow_from_state(
    state: torch.Tensor,
    teacher: torch.Tensor,
    timestep: torch.Tensor,
) -> torch.Tensor:
    sigma = (timestep.detach().float() / 1000.0).reshape(
        *timestep.shape,
        1,
        1,
        1,
    )
    return ((state.float() - teacher.float()) / sigma).to(dtype=state.dtype)


def make_decision_runs(
    *,
    live_improvements: dict[int, float],
    teacher_current_improvements: dict[int, float],
    oracle_fail_count: int | None = None,
) -> dict[str, dict]:
    runs: dict[str, dict] = {}
    for count in sweep.DEFAULT_STEP_COUNTS:
        live_mse = 100.0
        teacher_mse = 100.0
        if count != 4:
            live_mse = 100.0 * (1.0 - float(live_improvements[count]) / 100.0)
            teacher_mse = 100.0 * (
                1.0 - float(teacher_current_improvements[count]) / 100.0
            )
        runs[str(count)] = {
            "branches": {
                sweep.LIVE_JOINT_PREDICTED: {
                    "final_chunk2_mse_to_teacher": live_mse,
                },
                sweep.TEACHER_CURRENT_PREDICTED_MCP: {
                    "final_chunk2_mse_to_teacher": teacher_mse,
                },
                sweep.ORACLE_FLOW: {
                    "oracle_gate_pass": count != oracle_fail_count,
                },
            }
        }
    return runs


def test_step_sweep_schedules_are_generated_by_flow_match_scheduler() -> None:
    for count in sweep.DEFAULT_STEP_COUNTS:
        schedule = sweep.build_step_sweep_schedule(step_count=count)
        assert len(schedule.raw_schedule) == count
        assert len(schedule.main_warped_schedule) == count
        assert len(schedule.mcp_warped_schedule) == count
        assert schedule.raw_schedule[0] == 1000.0
        assert schedule.main_warped_schedule[0] == 1000.0
        assert schedule.mcp_warped_schedule[0] == 1000.0
        assert all(
            left > right
            for left, right in zip(schedule.raw_schedule, schedule.raw_schedule[1:])
        )
    schedule4 = sweep.build_step_sweep_schedule(step_count=4)
    assert schedule4.raw_schedule == ev.RAW_DEPLOYMENT_SCHEDULE
    assert schedule4.main_warped_schedule == ev.MAIN_DEPLOYMENT_SCHEDULE
    assert schedule4.mcp_warped_schedule == ev.MCP_DEPLOYMENT_SCHEDULE


def test_diagnostic_only_no_training_guards_are_recorded_and_source_clean() -> None:
    result = run_fake_sweep(RegressionGenerator())
    manifest = result.manifest
    assert manifest["diagnostic_only"] is True
    assert manifest["training_eligible"] is False
    assert manifest["canonical_training_eligible"] is False
    assert manifest["canonical_deployment_eligible"] is False
    assert manifest["forbidden_features"]["training"] is False
    repo = Path(__file__).resolve().parents[2]
    text = "\n".join(
        [
            (repo / "utils" / "nf_sf_first_mcp_step_sweep.py").read_text(
                encoding="utf-8"
            ),
            (repo / "scripts" / "run_nf_sf_first_mcp_step_sweep.py").read_text(
                encoding="utf-8"
            ),
        ]
    )
    for forbidden in ("optimizer.step", ".backward(", "requires_grad_("):
        assert forbidden not in text


def test_runner_fixed_identity_contract_default_identity_passes() -> None:
    sample_plan = {"fixed_decode_validation_identity": "validation-0"}
    selected = runner.select_eval_identity(
        sample_plan,
        sample_identity=None,
        num_samples=1,
    )
    contract = runner.enforce_fixed_validation_identity_contract(
        sample_plan,
        selected_identity=selected,
    )
    assert contract == {
        "fixed_identity_contract": True,
        "fixed_decode_validation_identity": "validation-0",
    }


def test_runner_fixed_identity_contract_rejects_override() -> None:
    sample_plan = {"fixed_decode_validation_identity": "validation-0"}
    with pytest.raises(RuntimeError, match="fixed_decode_validation_identity"):
        runner.enforce_fixed_validation_identity_contract(
            sample_plan,
            selected_identity="validation-1",
        )


def test_runner_requires_context_noise_zero() -> None:
    runner.validate_step_sweep_config(SimpleNamespace(context_noise=0))
    with pytest.raises(RuntimeError, match="context_noise=0"):
        runner.validate_step_sweep_config(SimpleNamespace(context_noise=1))


def test_synthetic_perfect_vector_field_recovers_teacher_for_all_step_counts() -> None:
    teacher = make_teacher_target()
    result = run_fake_sweep(ExactVectorFieldGenerator(teacher))
    for run in result.manifest["runs"].values():
        branches = run["branches"]
        assert branches[sweep.LIVE_JOINT_PREDICTED]["final_chunk2_mse_to_teacher"] < 1e-10
        assert (
            branches[sweep.TEACHER_CURRENT_PREDICTED_MCP][
                "final_chunk2_mse_to_teacher"
            ]
            < 1e-10
        )
        assert branches[sweep.ORACLE_FLOW]["final_oracle_chunk2_mse_to_teacher"] < 1e-10
        assert branches[sweep.ORACLE_FLOW]["oracle_gate_pass"] is True


def test_synthetic_bad_constant_vector_field_does_not_support_step_gap() -> None:
    result = run_fake_sweep(BadConstantGenerator())
    decision = result.manifest["primary_decision"]
    assert decision["status"] == sweep.NO_SUPPORT
    assert decision["live_32_vs_4_improvement_pct"] < 10.0


def test_decision_live_support_teacher_current_worse_is_inconclusive() -> None:
    runs = make_decision_runs(
        live_improvements={8: 0.0, 16: 0.0, 32: 35.0},
        teacher_current_improvements={8: 0.0, 16: 0.0, 32: -20.0},
    )
    decision = sweep.evaluate_few_step_decision(runs)
    assert decision["status"] == sweep.INCONCLUSIVE
    assert decision["teacher_current_direction_consistent"] is False


def test_decision_middle_step_large_improvement_is_not_no_support() -> None:
    runs = make_decision_runs(
        live_improvements={8: 35.0, 16: 40.0, 32: 5.0},
        teacher_current_improvements={8: 0.0, 16: 0.0, 32: 0.0},
    )
    decision = sweep.evaluate_few_step_decision(runs)
    assert decision["status"] == sweep.INCONCLUSIVE
    assert decision["live_32_vs_4_improvement_pct"] < 10.0


def test_decision_all_branches_same_error_scale_is_no_support() -> None:
    runs = make_decision_runs(
        live_improvements={8: 1.0, 16: 2.0, 32: 3.0},
        teacher_current_improvements={8: -1.0, 16: 0.0, 32: 2.0},
    )
    decision = sweep.evaluate_few_step_decision(runs)
    assert decision["status"] == sweep.NO_SUPPORT


def test_decision_live_support_and_teacher_direction_match_is_support() -> None:
    runs = make_decision_runs(
        live_improvements={8: 5.0, 16: 20.0, 32: 30.0},
        teacher_current_improvements={8: -2.0, 16: 0.0, 32: 0.1},
    )
    decision = sweep.evaluate_few_step_decision(runs)
    assert decision["status"] == sweep.SUPPORT_FEW_STEP_INFERENCE_GAP
    assert decision["teacher_current_direction_consistent"] is True


def test_decision_oracle_failure_is_invalid_gate() -> None:
    runs = make_decision_runs(
        live_improvements={8: 5.0, 16: 20.0, 32: 30.0},
        teacher_current_improvements={8: 1.0, 16: 1.0, 32: 1.0},
        oracle_fail_count=16,
    )
    decision = sweep.evaluate_few_step_decision(runs)
    assert decision["status"] == sweep.INVALID_ORACLE_GATE


def test_new_sweep_4step_live_branch_matches_existing_flow_audit_rollout() -> None:
    source = make_source_noise()
    teacher = make_teacher_target()
    conditional = {"prompt_embeds": torch.zeros((1, 2, 3))}
    rng_plan = ev.build_absolute_chunk_rng_plan(
        source_noise=source,
        rollout_seed=123,
        num_denoising_steps=4,
        chunk_frames=ev.FULL_SEQUENCE_CHUNK_FRAMES,
    )
    expected = flow_audit._run_predicted_rollout(
        runtime=make_runtime(RegressionGenerator()),
        mcp_scheduler=flow_audit.build_flow_match_scheduler(
            shift=DEFAULT_S_MCP,
            device=torch.device("cpu"),
        ),
        source_noise=source,
        teacher_target=teacher,
        conditional_dict=conditional,
        schedule=ev.resolve_deployment_schedule(),
        rng_plan=rng_plan,
    )
    result = run_fake_sweep(RegressionGenerator())
    actual = result.tensors["runs"]["4"][sweep.LIVE_JOINT_PREDICTED]["final_chunk2"]
    assert torch.equal(actual, expected["final_chunk2"].detach().cpu())
    actual_step = result.manifest["runs"]["4"]["branches"][
        sweep.LIVE_JOINT_PREDICTED
    ]["steps"][-1]
    expected_step = expected["trace"]["steps"][-1]
    assert actual_step["predicted_x0_vs_teacher_mse"] == expected_step[
        "predicted_x0_vs_teacher_mse"
    ]


def test_fairness_provenance_and_cpu_determinism() -> None:
    first = run_fake_sweep(RegressionGenerator())
    second = run_fake_sweep(RegressionGenerator())
    input_tensors = first.manifest["input_tensors"]
    assert input_tensors["source_noise_sha256"] == ev.tensor_sha256(make_source_noise())
    assert input_tensors["teacher_chunk1_sha256"]
    assert input_tensors["teacher_chunk2_sha256"]
    for count in sweep.DEFAULT_STEP_COUNTS:
        key = str(count)
        assert first.manifest["rng_plan_by_step_count"][key][
            "source_noise_sha256"
        ] == input_tensors["source_noise_sha256"]
        assert first.manifest["rng_plan_by_step_count"][key][
            "rng_plan_fingerprint_sha256"
        ] == second.manifest["rng_plan_by_step_count"][key][
            "rng_plan_fingerprint_sha256"
        ]
        assert first.manifest["runs"][key]["branches"][
            sweep.LIVE_JOINT_PREDICTED
        ]["final_chunk2_sha256"] == second.manifest["runs"][key]["branches"][
            sweep.LIVE_JOINT_PREDICTED
        ]["final_chunk2_sha256"]


def test_teacher_current_branch_uses_teacher_corrupted_current_not_main_drift() -> None:
    result = run_fake_sweep(MainDriftGenerator())
    branch = result.manifest["runs"]["4"]["branches"][
        sweep.TEACHER_CURRENT_PREDICTED_MCP
    ]
    assert branch["current_state_depends_on_previous_main_x0"] is False
    assert branch["future_state_depends_on_previous_mcp_x0"] is True
    assert branch["steps"][1]["main_predicted_x0_ignored_for_next_current"] is True

    source = make_source_noise()
    teacher = make_teacher_target()
    schedule = sweep.build_step_sweep_schedule(step_count=4)
    main_scheduler = sweep.build_step_sweep_scheduler(
        step_count=4,
        shift=DEFAULT_S_MAIN,
        device=torch.device("cpu"),
    )
    rng_plan = ev.build_absolute_chunk_rng_plan(
        source_noise=source,
        rollout_seed=123,
        num_denoising_steps=4,
        chunk_frames=ev.FULL_SEQUENCE_CHUNK_FRAMES,
    )
    teacher_chunk1 = flow_audit._chunk(teacher, flow_audit.CURRENT_CHUNK_INDEX)
    expected_noise = flow_audit._state_noise_for_step(
        rng_plan,
        source_noise=source,
        chunk_index=flow_audit.CURRENT_CHUNK_INDEX,
        step_index=1,
        template=teacher_chunk1,
    )
    expected_current = flow_audit._add_noise_chunk(
        main_scheduler,
        clean=teacher_chunk1,
        noise=expected_noise,
        timestep=flow_audit._timestep(
            schedule.main_warped_schedule[1],
            teacher_chunk1,
        ),
    )
    assert branch["steps"][1]["current_state"]["sha256"] == flow_audit._tensor_record(
        expected_current
    )["sha256"]
    assert branch["steps"][1]["future_state"]["sha256"] == branch["steps"][0][
        "transition_state"
    ]["sha256"]


def test_manifest_validation_rejects_fail_closed_cases() -> None:
    manifest = run_fake_sweep(RegressionGenerator()).manifest

    missing = copy.deepcopy(manifest)
    del missing["runs"]["32"]
    with pytest.raises(RuntimeError, match="run set"):
        sweep.validate_first_mcp_step_sweep_manifest(missing)

    duplicate = copy.deepcopy(manifest)
    duplicate["step_counts_requested"] = [4, 4, 8, 16, 32]
    with pytest.raises(RuntimeError, match="duplicate"):
        sweep.validate_first_mcp_step_sweep_manifest(duplicate)

    requested = copy.deepcopy(manifest)
    requested["step_counts_requested"] = [4, 8, 32, 16]
    with pytest.raises(RuntimeError, match="requested counts"):
        sweep.validate_first_mcp_step_sweep_manifest(requested)

    wrong_schedule = copy.deepcopy(manifest)
    wrong_schedule["schedule_by_step_count"]["4"]["raw_schedule"][0] = 999.0
    with pytest.raises(RuntimeError, match="start at timestep 1000|raw 4-step"):
        sweep.validate_first_mcp_step_sweep_manifest(wrong_schedule)

    schedule_fingerprint = copy.deepcopy(manifest)
    schedule_fingerprint["runs"]["8"]["schedule_fingerprint_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="schedule fingerprint"):
        sweep.validate_first_mcp_step_sweep_manifest(schedule_fingerprint)

    rng_fingerprint = copy.deepcopy(manifest)
    rng_fingerprint["rng_plan_by_step_count"]["8"][
        "rng_plan_fingerprint_sha256"
    ] = "0" * 64
    with pytest.raises(RuntimeError, match="RNG fingerprint"):
        sweep.validate_first_mcp_step_sweep_manifest(rng_fingerprint)

    rng_source = copy.deepcopy(manifest)
    rng_source["rng_plan_by_step_count"]["16"]["source_noise_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="RNG source noise SHA"):
        sweep.validate_first_mcp_step_sweep_manifest(rng_source)

    common_fingerprint = copy.deepcopy(manifest)
    common_fingerprint["common_inputs_fingerprint_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="common inputs fingerprint"):
        sweep.validate_first_mcp_step_sweep_manifest(common_fingerprint)

    decision = copy.deepcopy(manifest)
    decision["primary_decision"] = dict(decision["primary_decision"])
    decision["primary_decision"]["status"] = sweep.INCONCLUSIVE
    with pytest.raises(RuntimeError, match="primary decision"):
        sweep.validate_first_mcp_step_sweep_manifest(decision)

    canonical_training = copy.deepcopy(manifest)
    canonical_training["canonical_training_eligible"] = True
    with pytest.raises(RuntimeError, match="canonical training"):
        sweep.validate_first_mcp_step_sweep_manifest(canonical_training)

    oracle_failure = copy.deepcopy(manifest)
    oracle_failure["runs"]["8"]["branches"][sweep.ORACLE_FLOW][
        "oracle_gate_pass"
    ] = False
    with pytest.raises(RuntimeError, match="oracle gate"):
        sweep.validate_first_mcp_step_sweep_manifest(oracle_failure)

    nonfinite = copy.deepcopy(manifest)
    nonfinite["runs"]["16"]["branches"][sweep.LIVE_JOINT_PREDICTED][
        "final_chunk2_mse_to_teacher"
    ] = float("nan")
    with pytest.raises(RuntimeError, match="finite"):
        sweep.validate_first_mcp_step_sweep_manifest(nonfinite)


def test_step0_single_variable_fairness_gate_passes_and_rejects_tamper() -> None:
    manifest = run_fake_sweep(RegressionGenerator()).manifest
    fairness = manifest["step0_single_variable_fairness"]
    assert fairness["step0_single_variable_fairness_gate_pass"] is True
    assert fairness["cross_count_live_exact"] is True
    assert fairness["cross_count_teacher_current_exact"] is True
    assert fairness["within_count_live_vs_teacher_current_exact"] is True

    live_cross_count = copy.deepcopy(manifest)
    live_cross_count["runs"]["8"]["branches"][sweep.LIVE_JOINT_PREDICTED][
        "steps"
    ][0]["predicted_flow"]["sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="step0 fairness|step0 predicted"):
        sweep.validate_first_mcp_step_sweep_manifest(live_cross_count)

    within_count = copy.deepcopy(manifest)
    within_count["runs"]["16"]["branches"][sweep.TEACHER_CURRENT_PREDICTED_MCP][
        "steps"
    ][0]["current_state"]["sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="step0"):
        sweep.validate_first_mcp_step_sweep_manifest(within_count)
