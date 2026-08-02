import pytest
import torch
from torch import nn

from utils.nf_sf_tensors import make_cpu_generator
from utils.nf_sf_training import (
    NFSFSelectedState,
    compute_nf_sf_losses,
    configure_nf_sf_optimizer_plan,
    prepare_nf_sf_noisy_batch,
    run_nf_sf_mcp1_grid_point_loss,
    run_nf_sf_forward_loss,
)
from utils.scheduler import FlowMatchScheduler


class FakeGenerator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = []

    def forward(self, **kwargs):
        self.calls.append(kwargs)
        noisy_current = kwargs["noisy_image_or_video"]
        mcp_noises = kwargs["mcp_future_noises"]
        return (
            torch.zeros_like(noisy_current),
            torch.zeros_like(noisy_current),
            [torch.zeros_like(noise) for noise in mcp_noises],
        )


class FakeGridGenerator(nn.Module):
    def __init__(
        self,
        *,
        top_level_count: int = 3,
        mcp_count: int = 1,
        nonfinite: bool = False,
    ) -> None:
        super().__init__()
        self.top_level_count = int(top_level_count)
        self.mcp_count = int(mcp_count)
        self.nonfinite = bool(nonfinite)
        self.calls = []

    def forward(self, **kwargs):
        future = kwargs["mcp_future_noises"][0]
        self.calls.append(
            {
                "noisy_current": kwargs["noisy_image_or_video"].detach().clone(),
                "future": future.detach().clone(),
                "timestep": kwargs["timestep"].detach().clone(),
                "mcp_timestep": kwargs["mcp_timesteps"][0].detach().clone(),
                "future_start": list(kwargs["mcp_future_start_frames"]),
            }
        )
        if self.nonfinite:
            flow = torch.full_like(future, float("inf"))
        else:
            flow = torch.zeros_like(future)
        flows = [flow]
        flows.extend(torch.zeros_like(future) for _ in range(max(0, self.mcp_count - 1)))
        flows = flows[: max(0, self.mcp_count)]
        outputs = (torch.zeros_like(kwargs["noisy_image_or_video"]), torch.zeros_like(future), flows)
        return outputs[: self.top_level_count]


class FakeBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.patch_embedding = nn.Conv3d(1, 1, kernel_size=1)
        self.block = nn.Linear(2, 2)


class FakeMCP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fusion = nn.Sequential(nn.Linear(2, 2))
        self.mcp_modules = nn.ModuleList([nn.Linear(2, 2) for _ in range(3)])


class FakeWanWrapper(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = FakeBackbone()
        self.mcp = FakeMCP()


def _scheduler(shift: float) -> FlowMatchScheduler:
    scheduler = FlowMatchScheduler(shift=shift, sigma_min=0.0, extra_one_step=True)
    scheduler.set_timesteps(1000, training=True)
    return scheduler


def _state() -> NFSFSelectedState:
    history = torch.full((1, 3, 1, 1, 1), -1.0)
    current = torch.arange(3, dtype=torch.float32).reshape(1, 3, 1, 1, 1)
    futures = tuple(current + offset for offset in (10.0, 20.0, 30.0))
    masks = (
        torch.tensor([True, True, True]),
        torch.tensor([True, False, True]),
        torch.tensor([False, False, False]),
    )
    return NFSFSelectedState(
        clean_history=history,
        current_target=current,
        future_targets=futures,
        future_valid_masks=masks,
        current_start_frame=3,
    )


def test_prepare_noisy_batch_uses_selected_state_granularity_and_scheduler_targets() -> None:
    state = _state()
    scheduler_main = _scheduler(5.0)
    scheduler_mcp = _scheduler(10.0)

    batch = prepare_nf_sf_noisy_batch(
        state,
        scheduler_main=scheduler_main,
        scheduler_mcp=scheduler_mcp,
        rng=make_cpu_generator(11),
    )

    assert batch.noisy_current.shape == state.current_target.shape
    assert len(batch.noisy_futures) == 3
    assert batch.timestep_main.shape == (1, 3)
    assert batch.timestep_depths[0].shape == (1, 3)
    assert torch.equal(batch.timestep_main[:, :1].expand(-1, 3), batch.timestep_main)
    assert torch.equal(batch.timestep_depths[0][:, :1].expand(-1, 3), batch.timestep_depths[0])
    assert batch.future_start_frames == (6, 9, 12)
    assert torch.equal(
        batch.target_flow_main,
        scheduler_main.training_target(
            state.current_target.flatten(0, 1),
            batch.epsilon_main.flatten(0, 1),
            batch.timestep_main.flatten(0, 1),
        ).unflatten(0, state.current_target.shape[:2]),
    )


def test_prepare_noisy_batch_rejects_multi_chunk_history_for_m2_harness() -> None:
    state = _state()
    invalid = NFSFSelectedState(
        clean_history=torch.cat([state.clean_history, state.clean_history], dim=1),
        current_target=state.current_target,
        future_targets=state.future_targets,
        future_valid_masks=state.future_valid_masks,
        current_start_frame=6,
    )

    with pytest.raises(ValueError, match="exactly one clean history chunk"):
        prepare_nf_sf_noisy_batch(
            invalid,
            scheduler_main=_scheduler(5.0),
            scheduler_mcp=_scheduler(10.0),
            rng=make_cpu_generator(13),
        )


def test_forward_loss_passes_per_depth_timesteps_and_applies_masks() -> None:
    state = _state()
    batch = prepare_nf_sf_noisy_batch(
        state,
        scheduler_main=_scheduler(5.0),
        scheduler_mcp=_scheduler(10.0),
        rng=make_cpu_generator(12),
    )
    generator = FakeGenerator()

    result = run_nf_sf_forward_loss(
        generator,
        conditional_dict={"prompt_embeds": torch.zeros((1, 1, 1))},
        noisy_batch=batch,
    )
    call = generator.calls[-1]

    assert call["clean_x"] is state.clean_history
    assert torch.equal(call["mcp_timesteps"][0], batch.timestep_depths[0])
    assert call["mcp_future_start_frames"] == [6, 9, 12]
    assert len(result.mcp_flow_preds) == 3

    expected = compute_nf_sf_losses(
        main_flow_pred=torch.zeros_like(batch.target_flow_main),
        mcp_flow_preds=tuple(torch.zeros_like(target) for target in batch.target_flow_depths),
        noisy_batch=batch,
    )
    assert torch.equal(result.losses.main_loss, expected.main_loss)
    assert torch.equal(result.losses.mcp_depth_losses[0], expected.mcp_depth_losses[0])
    assert torch.equal(result.losses.mcp_depth_losses[1], expected.mcp_depth_losses[1])
    assert torch.equal(result.losses.mcp_depth_losses[2], expected.mcp_depth_losses[2])
    assert result.losses.mcp_depth_losses[2].item() == 0.0


def test_mcp1_grid_point_loss_uses_oracle_inputs_and_single_future_contract() -> None:
    state = _state()
    scheduler = _scheduler(5.0)
    generator = FakeGridGenerator()
    epsilon_main = torch.full_like(state.current_target, 3.0)
    epsilon_future = torch.full_like(state.future_targets[0], 4.0)
    timestep = torch.tensor(float(scheduler.timesteps[7].item()))

    point = run_nf_sf_mcp1_grid_point_loss(
        generator,
        conditional_dict={"prompt_embeds": torch.zeros((1, 1, 1))},
        state=state,
        scheduler=scheduler,
        epsilon_main=epsilon_main,
        epsilon_future=epsilon_future,
        timestep=timestep,
    )
    call = generator.calls[-1]
    timestep_chunk = torch.full(state.current_target.shape[:2], float(timestep.item()))
    expected_current = scheduler.add_noise(
        state.current_target.flatten(0, 1),
        epsilon_main.flatten(0, 1),
        timestep_chunk.flatten(0, 1),
    ).unflatten(0, state.current_target.shape[:2])
    expected_future = scheduler.add_noise(
        state.future_targets[0].flatten(0, 1),
        epsilon_future.flatten(0, 1),
        timestep_chunk.flatten(0, 1),
    ).unflatten(0, state.current_target.shape[:2])
    expected_target = scheduler.training_target(
        state.future_targets[0].flatten(0, 1),
        epsilon_future.flatten(0, 1),
        timestep_chunk.flatten(0, 1),
    ).unflatten(0, state.current_target.shape[:2])

    assert torch.allclose(call["noisy_current"], expected_current)
    assert torch.allclose(call["future"], expected_future)
    assert torch.equal(call["timestep"], call["mcp_timestep"])
    assert call["future_start"] == [6]
    assert point.loss.item() == pytest.approx(expected_target.float().square().mean().item())
    assert point.metadata["future_start_frame"] == 6


@pytest.mark.parametrize(
    ("top_level_count", "mcp_count", "message"),
    [
        (2, 1, "output triple"),
        (3, 0, "exactly one MCP"),
        (3, 2, "exactly one MCP"),
    ],
)
def test_mcp1_grid_point_loss_rejects_output_contract_errors(
    top_level_count,
    mcp_count,
    message,
) -> None:
    state = _state()
    scheduler = _scheduler(5.0)

    with pytest.raises(RuntimeError, match=message):
        run_nf_sf_mcp1_grid_point_loss(
            FakeGridGenerator(top_level_count=top_level_count, mcp_count=mcp_count),
            conditional_dict={},
            state=state,
            scheduler=scheduler,
            epsilon_main=torch.ones_like(state.current_target),
            epsilon_future=torch.ones_like(state.future_targets[0]),
            timestep=torch.tensor(float(scheduler.timesteps[0].item())),
        )


def test_mcp1_grid_point_loss_rejects_nonfinite_flow_and_wrong_start() -> None:
    state = _state()
    scheduler = _scheduler(5.0)

    with pytest.raises(RuntimeError, match="non-finite"):
        run_nf_sf_mcp1_grid_point_loss(
            FakeGridGenerator(nonfinite=True),
            conditional_dict={},
            state=state,
            scheduler=scheduler,
            epsilon_main=torch.ones_like(state.current_target),
            epsilon_future=torch.ones_like(state.future_targets[0]),
            timestep=torch.tensor(float(scheduler.timesteps[0].item())),
        )

    bad_state = NFSFSelectedState(
        clean_history=state.clean_history,
        current_target=state.current_target,
        future_targets=state.future_targets,
        current_start_frame=0,
    )
    with pytest.raises(ValueError, match="future_start_frame=6"):
        run_nf_sf_mcp1_grid_point_loss(
            FakeGridGenerator(),
            conditional_dict={},
            state=bad_state,
            scheduler=scheduler,
            epsilon_main=torch.ones_like(state.current_target),
            epsilon_future=torch.ones_like(state.future_targets[0]),
            timestep=torch.tensor(float(scheduler.timesteps[0].item())),
        )


def _audit_by_name(plan):
    return {audit.name: audit for audit in plan.audits}


def test_frozen_optimizer_plan_trains_only_fusion_and_mcp_depths() -> None:
    wrapper = FakeWanWrapper()

    plan = configure_nf_sf_optimizer_plan(wrapper, mode="frozen", lr=1.0e-4)
    audits = _audit_by_name(plan)

    assert audits["backbone"].requires_grad is False
    assert audits["patch_embedding"].requires_grad is False
    assert audits["mcp_fusion"].requires_grad is True
    assert audits["mcp_depth1"].requires_grad is True
    assert audits["mcp_depth2"].requires_grad is True
    assert audits["mcp_depth3"].requires_grad is True
    assert audits["backbone"].in_optimizer is False
    assert audits["patch_embedding"].in_optimizer is False
    assert audits["mcp_fusion"].in_optimizer is True


def test_joint_optimizer_plan_trains_backbone_patch_embedding_fusion_and_mcp_depths() -> None:
    wrapper = FakeWanWrapper()

    plan = configure_nf_sf_optimizer_plan(wrapper, mode="joint", lr=1.0e-4)
    audits = _audit_by_name(plan)

    assert audits["backbone"].requires_grad is True
    assert audits["patch_embedding"].requires_grad is True
    assert audits["mcp_fusion"].requires_grad is True
    assert audits["mcp_depth1"].requires_grad is True
    assert audits["mcp_depth2"].requires_grad is True
    assert audits["mcp_depth3"].requires_grad is True
    assert audits["backbone"].in_optimizer is True
    assert audits["patch_embedding"].in_optimizer is True
