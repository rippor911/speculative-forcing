import pytest
import torch
from torch import nn

from scripts import diagnose_nf_sf_m3_trajectory as diag
from utils.scheduler import FlowMatchScheduler


SHA_A = "a" * 40
SHA_B = "b" * 40


def _scheduler(shift: float, *, steps: int = 4) -> FlowMatchScheduler:
    scheduler = FlowMatchScheduler(shift=shift, sigma_min=0.0, extra_one_step=True)
    scheduler.set_timesteps(steps, denoising_strength=1.0)
    return scheduler


def _chunk(values, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.tensor(values, dtype=dtype).reshape(1, 3, 1, 1, 1)


def _inputs(dtype: torch.dtype = torch.float32):
    clean_history = _chunk([-1.0, -1.0, -1.0], dtype=dtype)
    current_target = _chunk([0.0, 1.0, 2.0], dtype=dtype)
    next1_target = _chunk([10.0, 11.0, 12.0], dtype=dtype)
    epsilon_main = _chunk([3.0, 4.0, 5.0], dtype=dtype)
    epsilon_future = _chunk([-2.0, -3.0, -4.0], dtype=dtype)
    conditional_dict = {"prompt_embeds": torch.zeros((1, 1, 1), dtype=dtype)}
    return {
        "clean_history": clean_history,
        "current_target": current_target,
        "next1_target": next1_target,
        "epsilon_main": epsilon_main,
        "epsilon_future": epsilon_future,
        "conditional_dict": conditional_dict,
    }


class RecordingGenerator(nn.Module):
    def __init__(
        self,
        *,
        scheduler=None,
        main_flow: torch.Tensor | None = None,
        mcp_flow: torch.Tensor | None = None,
        flow_bias: float = 0.0,
        assert_dtype: torch.dtype | None = None,
        nonfinite: bool = False,
    ) -> None:
        super().__init__()
        self.scheduler = scheduler
        self.main_flow = main_flow
        self.mcp_flow = mcp_flow
        self.flow_bias = float(flow_bias)
        self.assert_dtype = assert_dtype
        self.nonfinite = bool(nonfinite)
        self.calls = []

    def get_scheduler(self):
        if self.scheduler is None:
            raise RuntimeError("test generator has no scheduler")
        return self.scheduler

    def forward(self, **kwargs):
        current = kwargs["noisy_image_or_video"]
        if self.assert_dtype is not None:
            assert current.dtype == self.assert_dtype
        call = {
            "noisy_image_or_video": current.detach().clone(),
            "timestep": kwargs["timestep"].detach().clone(),
            "mcp_timesteps": None,
            "mcp_future_noises": None,
            "mcp_future_start_frames": kwargs.get("mcp_future_start_frames"),
        }
        if "mcp_future_noises" in kwargs:
            future = kwargs["mcp_future_noises"][0]
            if self.assert_dtype is not None:
                assert future.dtype == self.assert_dtype
            call["mcp_future_noises"] = [future.detach().clone()]
            call["mcp_timesteps"] = [
                timestep.detach().clone() for timestep in kwargs["mcp_timesteps"]
            ]
            self.calls.append(call)
            flow = self.mcp_flow.to(device=future.device, dtype=future.dtype)
            if self.nonfinite:
                flow = torch.full_like(future, float("inf"))
            else:
                flow = flow + self.flow_bias
            return (
                torch.zeros_like(current),
                torch.zeros_like(current),
                [flow, torch.zeros_like(future), torch.zeros_like(future)],
            )

        self.calls.append(call)
        flow = self.main_flow.to(device=current.device, dtype=current.dtype)
        if self.nonfinite:
            flow = torch.full_like(current, float("inf"))
        else:
            flow = flow + self.flow_bias
        return (flow, torch.zeros_like(current), [])


def test_oracle_state_construction_zero_error_flow_reaches_next_oracle() -> None:
    tensors = _inputs()
    scheduler = _scheduler(5.0)
    target_flow = tensors["epsilon_main"] - tensors["current_target"]
    generator = RecordingGenerator(main_flow=target_flow)

    records = diag.run_main_trajectory(
        generator,
        conditional_dict=tensors["conditional_dict"],
        clean_history=tensors["clean_history"],
        current_target=tensors["current_target"],
        epsilon_main=tensors["epsilon_main"],
        scheduler=scheduler,
        timesteps=scheduler.timesteps,
        mode="teacher_forced",
    )

    assert len(records) == 4
    assert all(record["pre_state_mse_vs_oracle"] == pytest.approx(0.0) for record in records)
    assert all(record["post_state_mse_vs_oracle_next"] < 1.0e-12 for record in records)


def test_teacher_forced_main_uses_each_oracle_state_independently() -> None:
    tensors = _inputs()
    scheduler = _scheduler(5.0)
    target_flow = tensors["epsilon_main"] - tensors["current_target"]
    generator = RecordingGenerator(main_flow=target_flow)

    diag.run_main_trajectory(
        generator,
        conditional_dict=tensors["conditional_dict"],
        clean_history=tensors["clean_history"],
        current_target=tensors["current_target"],
        epsilon_main=tensors["epsilon_main"],
        scheduler=scheduler,
        timesteps=scheduler.timesteps,
        mode="teacher_forced",
    )

    for index, call in enumerate(generator.calls):
        timestep = diag._timestep_chunk(scheduler.timesteps[index], tensors["current_target"])
        expected = diag.oracle_state_for_step(
            scheduler,
            target=tensors["current_target"],
            epsilon=tensors["epsilon_main"],
            timestep=timestep,
        )
        assert torch.equal(call["noisy_image_or_video"], expected)


def test_free_running_main_drift_accumulates_with_biased_flow() -> None:
    tensors = _inputs()
    scheduler = _scheduler(5.0)
    target_flow = tensors["epsilon_main"] - tensors["current_target"]
    generator = RecordingGenerator(main_flow=target_flow, flow_bias=0.5)

    records = diag.run_main_trajectory(
        generator,
        conditional_dict=tensors["conditional_dict"],
        clean_history=tensors["clean_history"],
        current_target=tensors["current_target"],
        epsilon_main=tensors["epsilon_main"],
        scheduler=scheduler,
        timesteps=scheduler.timesteps,
        mode="free_running",
    )

    assert records[0]["pre_state_mse_vs_oracle"] == pytest.approx(0.0)
    assert records[1]["pre_state_mse_vs_oracle"] > records[0]["pre_state_mse_vs_oracle"]
    assert (
        records[-1]["post_state_mse_vs_oracle_next"]
        > records[0]["post_state_mse_vs_oracle_next"]
    )


def test_mcp_baseline_and_shift10_use_distinct_future_schedules_and_timesteps() -> None:
    tensors = _inputs()
    main_scheduler = _scheduler(5.0)
    mcp_scheduler = _scheduler(10.0)
    main_flow = tensors["epsilon_main"] - tensors["current_target"]
    mcp_flow = tensors["epsilon_future"] - tensors["next1_target"]

    baseline_generator = RecordingGenerator(main_flow=main_flow, mcp_flow=mcp_flow)
    diag.run_mcp1_trajectory(
        baseline_generator,
        conditional_dict=tensors["conditional_dict"],
        clean_history=tensors["clean_history"],
        current_target=tensors["current_target"],
        next1_target=tensors["next1_target"],
        epsilon_main=tensors["epsilon_main"],
        epsilon_future=tensors["epsilon_future"],
        current_condition_scheduler=main_scheduler,
        future_scheduler=main_scheduler,
        main_timesteps=main_scheduler.timesteps,
        mcp_timesteps=main_scheduler.timesteps,
        mode="teacher_forced",
    )

    aligned_generator = RecordingGenerator(main_flow=main_flow, mcp_flow=mcp_flow)
    diag.run_mcp1_trajectory(
        aligned_generator,
        conditional_dict=tensors["conditional_dict"],
        clean_history=tensors["clean_history"],
        current_target=tensors["current_target"],
        next1_target=tensors["next1_target"],
        epsilon_main=tensors["epsilon_main"],
        epsilon_future=tensors["epsilon_future"],
        current_condition_scheduler=main_scheduler,
        future_scheduler=mcp_scheduler,
        main_timesteps=main_scheduler.timesteps,
        mcp_timesteps=mcp_scheduler.timesteps,
        mode="teacher_forced",
    )

    assert not torch.allclose(main_scheduler.timesteps, mcp_scheduler.timesteps)
    baseline_mcp_timesteps = [
        call["mcp_timesteps"][0][0, 0].item() for call in baseline_generator.calls
    ]
    aligned_mcp_timesteps = [
        call["mcp_timesteps"][0][0, 0].item() for call in aligned_generator.calls
    ]
    assert baseline_mcp_timesteps == pytest.approx(main_scheduler.timesteps.tolist())
    assert aligned_mcp_timesteps == pytest.approx(mcp_scheduler.timesteps.tolist())
    assert aligned_generator.calls[1]["timestep"][0, 0].item() != pytest.approx(
        aligned_generator.calls[1]["mcp_timesteps"][0][0, 0].item()
    )


def test_mcp_teacher_forced_current_condition_and_future_start_frame_are_fixed() -> None:
    tensors = _inputs()
    main_scheduler = _scheduler(5.0)
    mcp_scheduler = _scheduler(10.0)
    mcp_flow = tensors["epsilon_future"] - tensors["next1_target"]
    generator = RecordingGenerator(mcp_flow=mcp_flow)

    diag.run_mcp1_trajectory(
        generator,
        conditional_dict=tensors["conditional_dict"],
        clean_history=tensors["clean_history"],
        current_target=tensors["current_target"],
        next1_target=tensors["next1_target"],
        epsilon_main=tensors["epsilon_main"],
        epsilon_future=tensors["epsilon_future"],
        current_condition_scheduler=main_scheduler,
        future_scheduler=mcp_scheduler,
        main_timesteps=main_scheduler.timesteps,
        mcp_timesteps=mcp_scheduler.timesteps,
        mode="teacher_forced",
    )

    first_timestep = diag._timestep_chunk(main_scheduler.timesteps[0], tensors["current_target"])
    expected_current_condition = diag.oracle_state_for_step(
        main_scheduler,
        target=tensors["current_target"],
        epsilon=tensors["epsilon_main"],
        timestep=first_timestep,
    )
    assert torch.equal(generator.calls[0]["noisy_image_or_video"], expected_current_condition)
    assert all(
        call["mcp_future_start_frames"] == [diag.MCP1_FUTURE_START_FRAME]
        for call in generator.calls
    )
    with pytest.raises(RuntimeError, match="future_start_frame changed"):
        diag.run_mcp1_trajectory(
            generator,
            conditional_dict=tensors["conditional_dict"],
            clean_history=tensors["clean_history"],
            current_target=tensors["current_target"],
            next1_target=tensors["next1_target"],
            epsilon_main=tensors["epsilon_main"],
            epsilon_future=tensors["epsilon_future"],
            current_condition_scheduler=main_scheduler,
            future_scheduler=mcp_scheduler,
            main_timesteps=main_scheduler.timesteps,
            mcp_timesteps=mcp_scheduler.timesteps,
            mode="teacher_forced",
            future_start_frame=9,
        )


def test_bf16_main_and_mcp_states_remain_bf16_after_multistep_updates() -> None:
    tensors = _inputs(dtype=torch.bfloat16)
    main_scheduler = _scheduler(5.0)
    mcp_scheduler = _scheduler(10.0)
    main_flow = tensors["epsilon_main"] - tensors["current_target"]
    mcp_flow = tensors["epsilon_future"] - tensors["next1_target"]

    main_generator = RecordingGenerator(
        main_flow=main_flow,
        assert_dtype=torch.bfloat16,
    )
    main_records = diag.run_main_trajectory(
        main_generator,
        conditional_dict=tensors["conditional_dict"],
        clean_history=tensors["clean_history"],
        current_target=tensors["current_target"],
        epsilon_main=tensors["epsilon_main"],
        scheduler=main_scheduler,
        timesteps=main_scheduler.timesteps,
        mode="free_running",
    )
    assert len(main_generator.calls) == 4
    assert all(record["state_dtype"] == "torch.bfloat16" for record in main_records)

    mcp_generator = RecordingGenerator(
        mcp_flow=mcp_flow,
        assert_dtype=torch.bfloat16,
    )
    mcp_records = diag.run_mcp1_trajectory(
        mcp_generator,
        conditional_dict=tensors["conditional_dict"],
        clean_history=tensors["clean_history"],
        current_target=tensors["current_target"],
        next1_target=tensors["next1_target"],
        epsilon_main=tensors["epsilon_main"],
        epsilon_future=tensors["epsilon_future"],
        current_condition_scheduler=main_scheduler,
        future_scheduler=mcp_scheduler,
        main_timesteps=main_scheduler.timesteps,
        mcp_timesteps=mcp_scheduler.timesteps,
        mode="free_running",
    )
    assert len(mcp_generator.calls) == 4
    assert all(record["state_dtype"] == "torch.bfloat16" for record in mcp_records)


def test_nonfinite_flow_is_rejected() -> None:
    tensors = _inputs()
    scheduler = _scheduler(5.0)
    target_flow = tensors["epsilon_main"] - tensors["current_target"]
    generator = RecordingGenerator(main_flow=target_flow, nonfinite=True)

    with pytest.raises(RuntimeError, match="non-finite|not finite"):
        diag.run_main_trajectory(
            generator,
            conditional_dict=tensors["conditional_dict"],
            clean_history=tensors["clean_history"],
            current_target=tensors["current_target"],
            epsilon_main=tensors["epsilon_main"],
            scheduler=scheduler,
            timesteps=scheduler.timesteps,
            mode="teacher_forced",
        )


def test_trajectory_requires_exactly_four_records() -> None:
    tensors = _inputs()
    scheduler = _scheduler(5.0, steps=3)
    target_flow = tensors["epsilon_main"] - tensors["current_target"]
    generator = RecordingGenerator(main_flow=target_flow)

    with pytest.raises(RuntimeError, match="exactly 4 steps"):
        diag.run_main_trajectory(
            generator,
            conditional_dict=tensors["conditional_dict"],
            clean_history=tensors["clean_history"],
            current_target=tensors["current_target"],
            epsilon_main=tensors["epsilon_main"],
            scheduler=scheduler,
            timesteps=scheduler.timesteps,
            mode="teacher_forced",
        )

    with pytest.raises(RuntimeError, match="record count"):
        diag.summarize_trajectory([{"step_index": 0, "finite": True}])


def _payload(git_sha: str = SHA_A) -> dict:
    return {"git_sha": git_sha}


def _fake_git(
    *,
    status_text: str = "",
    diff_text: str = "",
    current_sha: str = SHA_B,
    ancestor: bool = True,
):
    def git_text(command):
        command = list(command)
        if command == ["git", "rev-parse", "HEAD"]:
            return current_sha
        if command == ["git", "branch", "--show-current"]:
            return "next-forcing"
        if command == ["git", "status", "--short"]:
            return status_text
        if command[:3] == ["git", "diff", "--name-status"]:
            return diff_text
        raise AssertionError(f"unexpected git command: {command}")

    def git_success(command):
        command = list(command)
        if command[:3] == ["git", "merge-base", "--is-ancestor"]:
            return ancestor
        raise AssertionError(f"unexpected git success command: {command}")

    return git_text, git_success


def test_provenance_gate_accepts_only_two_allowed_added_files() -> None:
    diff_text = "\n".join(
        [
            "A\tscripts/diagnose_nf_sf_m3_trajectory.py",
            "A\ttests/speculative/test_nf_sf_m3_trajectory.py",
        ]
    )
    git_text, git_success = _fake_git(diff_text=diff_text)

    report = diag.diagnostic_provenance_gate(
        initial_payload=_payload(),
        final_payload=_payload(),
        git_text=git_text,
        git_success=git_success,
    )

    assert report["status"] == "PASS"
    assert report["checkpoint_git_sha"] == SHA_A
    assert len(report["git_diff_entries"]) == 2


@pytest.mark.parametrize(
    "diff_text",
    [
        "A\tscripts/diagnose_nf_sf_m3_trajectory.py",
        "A\ttests/speculative/test_nf_sf_m3_trajectory.py",
        "",
    ],
)
def test_provenance_gate_rejects_incomplete_or_empty_diff(diff_text: str) -> None:
    git_text, git_success = _fake_git(diff_text=diff_text)

    with pytest.raises(RuntimeError, match="expected=.*actual="):
        diag.diagnostic_provenance_gate(
            initial_payload=_payload(),
            final_payload=_payload(),
            git_text=git_text,
            git_success=git_success,
        )


def test_provenance_gate_rejects_existing_file_modification() -> None:
    git_text, git_success = _fake_git(diff_text="M\tutils/nf_sf_m3.py")

    with pytest.raises(RuntimeError, match="expected=.*actual="):
        diag.diagnostic_provenance_gate(
            initial_payload=_payload(),
            final_payload=_payload(),
            git_text=git_text,
            git_success=git_success,
        )


def test_provenance_gate_rejects_dirty_worktree() -> None:
    git_text, git_success = _fake_git(status_text=" M utils/nf_sf_m3.py")

    with pytest.raises(RuntimeError, match="worktree is dirty"):
        diag.diagnostic_provenance_gate(
            initial_payload=_payload(),
            final_payload=_payload(),
            git_text=git_text,
            git_success=git_success,
        )


def test_trajectory_checkpoint_steps_accepts_zero_to_one_hundred() -> None:
    diag.validate_trajectory_checkpoint_steps(
        {"initial_global_step": 0, "final_global_step": 100}
    )


@pytest.mark.parametrize(
    "checkpoint_pair",
    [
        {"initial_global_step": 0, "final_global_step": 10},
        {"initial_global_step": 1, "final_global_step": 100},
    ],
)
def test_trajectory_checkpoint_steps_rejects_non_overfit100_pair(checkpoint_pair) -> None:
    with pytest.raises(RuntimeError, match="0 -> 100"):
        diag.validate_trajectory_checkpoint_steps(checkpoint_pair)


def test_provenance_gate_rejects_non_ancestor_checkpoint_commit() -> None:
    git_text, git_success = _fake_git(ancestor=False)

    with pytest.raises(RuntimeError, match="not an ancestor"):
        diag.diagnostic_provenance_gate(
            initial_payload=_payload(),
            final_payload=_payload(),
            git_text=git_text,
            git_success=git_success,
        )
