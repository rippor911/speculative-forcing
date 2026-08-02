import pytest
import torch
from torch import nn

from scripts import run_nf_sf_m3_fixed_grid_probe as probe
from utils.scheduler import FlowMatchScheduler


SHA_BASE = probe.BASE_M3_CHECKPOINT_GIT_SHA
SHA_HEAD = "b" * 40


@pytest.fixture
def tmp_path():
    path = probe.ROOT / "videos" / "fixed_grid_probe_pytest_tmp_path"
    path.mkdir(parents=True, exist_ok=True)
    yield path


def _scheduler() -> FlowMatchScheduler:
    scheduler = FlowMatchScheduler(shift=5.0, sigma_min=0.0, extra_one_step=True)
    scheduler.set_timesteps(4, denoising_strength=1.0)
    return scheduler


def _chunk(values, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.tensor(values, dtype=dtype).reshape(1, len(values), 1, 1, 1)


def _inputs(*, zero_target_flow: bool = False):
    clean_history = _chunk([-1.0, -1.0, -1.0])
    current_target = _chunk([0.0, 1.0, 2.0])
    next1_target = _chunk([1.0, -2.0, 0.5])
    epsilon_main = _chunk([3.0, 4.0, 5.0])
    epsilon_future = next1_target.clone() if zero_target_flow else _chunk([2.0, -1.0, 1.5])
    return {
        "conditional_dict": {"prompt_embeds": torch.zeros((1, 1, 1))},
        "clean_history": clean_history,
        "current_target": current_target,
        "next1_target": next1_target,
        "epsilon_main": epsilon_main,
        "epsilon_future": epsilon_future,
    }


class FlowParam(nn.Module):
    def __init__(self, shape=(1, 3, 1, 1, 1)) -> None:
        super().__init__()
        self.flow = nn.Parameter(torch.zeros(shape))


class FusionParam(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(()))


class FakeBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.patch_embedding = nn.Linear(1, 1)
        self.block = nn.Linear(1, 1)


class FakeMCP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fusion = FusionParam()
        self.mcp_modules = nn.ModuleList([FlowParam(), FlowParam(), FlowParam()])


class NanGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        return value.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return torch.full_like(grad_output, float("nan"))


class SingleLiveGraph(torch.autograd.Function):
    active = 0
    backward_calls = 0

    @classmethod
    def reset(cls) -> None:
        cls.active = 0
        cls.backward_calls = 0

    @staticmethod
    def forward(ctx, value):
        if SingleLiveGraph.active != 0:
            raise RuntimeError("previous fixed-grid point was not backwarded")
        SingleLiveGraph.active += 1
        return value.clone()

    @staticmethod
    def backward(ctx, grad_output):
        SingleLiveGraph.active -= 1
        SingleLiveGraph.backward_calls += 1
        return grad_output


class FakeGenerator(nn.Module):
    def __init__(
        self,
        *,
        scheduler=None,
        fixed_outputs=None,
        mcp_output_count: int = 1,
        top_level_output_count: int = 3,
        nonfinite_flow: bool = False,
        nan_gradient: bool = False,
        guard_single_live_graph: bool = False,
        use_fusion: bool = True,
        use_mcp1: bool = True,
        use_mcp2: bool = False,
    ) -> None:
        super().__init__()
        self.model = FakeBackbone()
        self.mcp = FakeMCP()
        self.scheduler = _scheduler() if scheduler is None else scheduler
        self.fixed_outputs = fixed_outputs
        self.mcp_output_count = int(mcp_output_count)
        self.top_level_output_count = int(top_level_output_count)
        self.nonfinite_flow = bool(nonfinite_flow)
        self.nan_gradient = bool(nan_gradient)
        self.guard_single_live_graph = bool(guard_single_live_graph)
        self.use_fusion = bool(use_fusion)
        self.use_mcp1 = bool(use_mcp1)
        self.use_mcp2 = bool(use_mcp2)
        self.calls = []

    def get_scheduler(self):
        return self.scheduler

    def forward(self, **kwargs):
        current = kwargs["noisy_image_or_video"]
        future = kwargs["mcp_future_noises"][0]
        self.calls.append(
            {
                "current": current.detach().clone(),
                "future": future.detach().clone(),
                "timestep": kwargs["timestep"].detach().clone(),
                "mcp_timestep": kwargs["mcp_timesteps"][0].detach().clone(),
                "future_start": list(kwargs["mcp_future_start_frames"]),
            }
        )
        if self.nonfinite_flow:
            flow = torch.full_like(future, float("inf"))
        elif self.fixed_outputs is not None:
            value = float(self.fixed_outputs[len(self.calls) - 1])
            flow = torch.full_like(future, value)
        else:
            flow = torch.zeros_like(future)
            if self.use_mcp1:
                base = self.mcp.mcp_modules[0].flow
                if self.nan_gradient:
                    base = NanGradient.apply(base)
                flow = flow + base.to(device=future.device, dtype=future.dtype)
            if self.use_fusion:
                flow = flow + self.mcp.fusion.bias
            if self.use_mcp2:
                flow = flow + self.mcp.mcp_modules[1].flow.to(
                    device=future.device,
                    dtype=future.dtype,
                )
            if self.guard_single_live_graph:
                flow = SingleLiveGraph.apply(flow)
        outputs = [flow]
        outputs.extend(torch.zeros_like(future) for _ in range(max(0, self.mcp_output_count - 1)))
        outputs = outputs[: max(0, self.mcp_output_count)]
        top = (torch.zeros_like(current), torch.zeros_like(current), outputs)
        return top[: self.top_level_output_count]


class CountingSGD(torch.optim.SGD):
    def __init__(self, params, **kwargs) -> None:
        super().__init__(params, **kwargs)
        self.step_calls = 0
        self.zero_grad_calls = 0

    def step(self, closure=None):
        self.step_calls += 1
        return super().step(closure=closure)

    def zero_grad(self, set_to_none: bool = True):
        self.zero_grad_calls += 1
        return super().zero_grad(set_to_none=set_to_none)


def _teacher_payload():
    scheduler = _scheduler()
    return {
        "raw_denoising_steps": [0.0, 1.0, 2.0, 3.0],
        "warped_denoising_steps": [
            float(value) for value in scheduler.timesteps.detach().float().tolist()
        ],
    }


def _loss_kwargs(generator):
    tensors = _inputs()
    return {
        **tensors,
        "generator": generator,
        "scheduler": generator.get_scheduler(),
        "timesteps": generator.get_scheduler().timesteps,
    }


def _step_kwargs(generator):
    kwargs = dict(_loss_kwargs(generator))
    kwargs.pop("generator")
    return kwargs


def _counting_optimizer(generator: nn.Module) -> CountingSGD:
    params = [
        parameter
        for name, parameter in generator.named_parameters()
        if probe.parameter_group_for_name(name) in probe.TRAINABLE_GROUPS
    ]
    return CountingSGD(params, lr=0.2)


def test_uses_exactly_four_fixed_timesteps() -> None:
    generator = FakeGenerator()
    scheduler, timesteps, report = probe.resolve_fixed_grid_schedule(
        generator,
        teacher_payload=_teacher_payload(),
        device="cpu",
    )

    assert scheduler is generator.get_scheduler()
    assert timesteps.tolist() == pytest.approx(list(probe.EXPECTED_FIXED_TIMESTEPS))
    assert report["generated_timesteps"] == pytest.approx(list(probe.EXPECTED_FIXED_TIMESTEPS))

    loss, records, summary = probe.fixed_grid_loss_and_records(**_loss_kwargs(generator))
    assert loss.item() >= 0.0
    assert len(records) == 4
    assert len(generator.calls) == 4
    assert summary["max_flow_mse"] >= summary["mean_flow_mse"]


def test_optimizer_step_called_once_and_four_point_loss_is_mean() -> None:
    generator = FakeGenerator(fixed_outputs=[1.0, 2.0, 3.0, 4.0])
    probe.configure_fixed_grid_trainable_parameters(generator)
    optimizer = _counting_optimizer(generator)
    tensors = _inputs(zero_target_flow=True)

    loss, records, summary = probe.fixed_grid_loss_and_records(
        generator,
        conditional_dict=tensors["conditional_dict"],
        clean_history=tensors["clean_history"],
        current_target=tensors["current_target"],
        next1_target=tensors["next1_target"],
        epsilon_main=tensors["epsilon_main"],
        epsilon_future=tensors["epsilon_future"],
        scheduler=generator.get_scheduler(),
        timesteps=generator.get_scheduler().timesteps,
    )
    assert loss.item() == pytest.approx((1.0 + 4.0 + 9.0 + 16.0) / 4.0)
    assert summary["mean_flow_mse"] == pytest.approx(loss.item())
    assert [record["flow_mse"] for record in records] == pytest.approx([1.0, 4.0, 9.0, 16.0])

    trainable_generator = FakeGenerator()
    probe.configure_fixed_grid_trainable_parameters(trainable_generator)
    counting_optimizer = _counting_optimizer(trainable_generator)
    probe.run_fixed_grid_optimizer_step(
        trainable_generator,
        optimizer=counting_optimizer,
        **_step_kwargs(trainable_generator),
    )
    assert counting_optimizer.step_calls == 1
    assert counting_optimizer.zero_grad_calls == 1


def test_training_step_uses_sequential_backward_without_retaining_four_graphs() -> None:
    SingleLiveGraph.reset()
    generator = FakeGenerator(guard_single_live_graph=True)
    probe.configure_fixed_grid_trainable_parameters(generator)
    optimizer = _counting_optimizer(generator)

    result = probe.run_fixed_grid_optimizer_step(
        generator,
        optimizer=optimizer,
        **_step_kwargs(generator),
    )

    assert SingleLiveGraph.backward_calls == 4
    assert SingleLiveGraph.active == 0
    assert optimizer.step_calls == 1
    assert optimizer.zero_grad_calls == 1
    assert result["training_objective_before_update"] == pytest.approx(
        result["pre_update_summary"]["mean_flow_mse"]
    )


def test_current_and_future_use_oracle_add_noise_and_future_start_six() -> None:
    generator = FakeGenerator()
    probe.fixed_grid_loss_and_records(**_loss_kwargs(generator))
    tensors = _inputs()
    scheduler = generator.get_scheduler()

    for index, call in enumerate(generator.calls):
        timestep = torch.full((1, 3), float(scheduler.timesteps[index].item()))
        expected_current = probe.add_noise_chunk(
            scheduler,
            tensors["current_target"],
            tensors["epsilon_main"],
            timestep,
        )
        expected_future = probe.add_noise_chunk(
            scheduler,
            tensors["next1_target"],
            tensors["epsilon_future"],
            timestep,
        )
        assert torch.equal(call["current"], expected_current)
        assert torch.equal(call["future"], expected_future)
        assert call["future_start"] == [6]
        assert torch.equal(call["timestep"], call["mcp_timestep"])


def test_only_fusion_and_mcp1_are_trainable_and_optimized() -> None:
    generator = FakeGenerator()
    setup = probe.configure_fixed_grid_trainable_parameters(generator)
    optimizer = probe.make_fixed_grid_optimizer(generator, lr=1.0e-3)
    optimizer_ids = probe.optimizer_parameter_ids(optimizer)

    assert setup["mcp_fusion"]["trainable_tensor_count"] > 0
    assert setup["mcp_depth1"]["trainable_tensor_count"] > 0
    for name, parameter in generator.named_parameters():
        group = probe.parameter_group_for_name(name)
        assert parameter.requires_grad is (group in probe.TRAINABLE_GROUPS)
        assert (id(parameter) in optimizer_ids) is (group in probe.TRAINABLE_GROUPS)


def test_non_target_parameters_do_not_change() -> None:
    generator = FakeGenerator()
    probe.configure_fixed_grid_trainable_parameters(generator)
    optimizer = _counting_optimizer(generator)
    result = probe.run_fixed_grid_optimizer_step(
        generator,
        optimizer=optimizer,
        **_step_kwargs(generator),
    )

    for group_name, group in result["parameter_audit"].items():
        if group_name not in probe.TRAINABLE_GROUPS:
            assert group["parameter_changed"] is False


def test_target_gradient_missing_fails() -> None:
    no_fusion = FakeGenerator(use_fusion=False)
    probe.configure_fixed_grid_trainable_parameters(no_fusion)
    optimizer = _counting_optimizer(no_fusion)
    with pytest.raises(RuntimeError, match="mcp_fusion"):
        probe.run_fixed_grid_optimizer_step(
            no_fusion,
            optimizer=optimizer,
            **_step_kwargs(no_fusion),
        )

    no_mcp1 = FakeGenerator(use_mcp1=False)
    probe.configure_fixed_grid_trainable_parameters(no_mcp1)
    optimizer = _counting_optimizer(no_mcp1)
    with pytest.raises(RuntimeError, match="mcp_depth1"):
        probe.run_fixed_grid_optimizer_step(
            no_mcp1,
            optimizer=optimizer,
            **_step_kwargs(no_mcp1),
        )


def test_non_target_gradient_fails() -> None:
    generator = FakeGenerator(use_mcp2=True)
    probe.configure_fixed_grid_trainable_parameters(generator)
    generator.mcp.mcp_modules[1].flow.requires_grad_(True)
    optimizer = _counting_optimizer(generator)

    with pytest.raises(RuntimeError, match="non-target"):
        probe.run_fixed_grid_optimizer_step(
            generator,
            optimizer=optimizer,
            **_step_kwargs(generator),
        )


def test_fake_mcp_overfits_fixed_four_points() -> None:
    generator = FakeGenerator()
    probe.configure_fixed_grid_trainable_parameters(generator)
    optimizer = _counting_optimizer(generator)
    initial_loss, _, _ = probe.fixed_grid_loss_and_records(**_loss_kwargs(generator))

    final = None
    for _ in range(40):
        final = probe.run_fixed_grid_optimizer_step(
            generator,
            optimizer=optimizer,
            evaluate_post_update=True,
            **_step_kwargs(generator),
        )
    assert final is not None
    assert final["post_update_loss"] < initial_loss.item() * 0.05


def test_post_update_metrics_match_independent_no_grad_eval() -> None:
    generator = FakeGenerator()
    probe.configure_fixed_grid_trainable_parameters(generator)
    optimizer = _counting_optimizer(generator)

    result = probe.run_fixed_grid_optimizer_step(
        generator,
        optimizer=optimizer,
        evaluate_post_update=True,
        **_step_kwargs(generator),
    )
    with torch.no_grad():
        eval_loss, eval_records, eval_summary = probe.fixed_grid_loss_and_records(
            **_loss_kwargs(generator)
        )

    assert result["post_update_loss"] == pytest.approx(eval_loss.item())
    assert result["post_update_summary"] == eval_summary
    assert result["post_update_records"] == eval_records
    assert result["training_objective_before_update"] != pytest.approx(
        result["post_update_loss"]
    )


def test_cosine_and_rms_ratio_are_correct() -> None:
    pred = torch.tensor([1.0, 0.0]).reshape(1, 2, 1, 1, 1)
    target = torch.tensor([2.0, 0.0]).reshape(1, 2, 1, 1, 1)

    metrics = probe.flow_metrics(pred, target)

    assert metrics["flow_cosine_similarity"] == pytest.approx(1.0)
    assert metrics["predicted_to_target_rms_ratio"] == pytest.approx(0.5)


def test_nonfinite_flow_and_gradient_fail() -> None:
    flow_generator = FakeGenerator(nonfinite_flow=True)
    with pytest.raises(RuntimeError, match="non-finite"):
        probe.fixed_grid_loss_and_records(**_loss_kwargs(flow_generator))

    grad_generator = FakeGenerator(nan_gradient=True)
    probe.configure_fixed_grid_trainable_parameters(grad_generator)
    optimizer = _counting_optimizer(grad_generator)
    with pytest.raises(RuntimeError, match="non-finite|not finite"):
        probe.run_fixed_grid_optimizer_step(
            grad_generator,
            optimizer=optimizer,
            **_step_kwargs(grad_generator),
        )


def test_final_checkpoint_metrics_align_with_saved_parameters() -> None:
    generator = FakeGenerator()
    probe.configure_fixed_grid_trainable_parameters(generator)
    optimizer = _counting_optimizer(generator)
    result = probe.run_fixed_grid_optimizer_step(
        generator,
        optimizer=optimizer,
        evaluate_post_update=True,
        **_step_kwargs(generator),
    )
    payload = probe.make_probe_checkpoint_payload(
        generator=generator,
        base_m3_checkpoint_path=probe.ROOT / "checkpoints" / "base.pt",
        base_m3_checkpoint_sha256="c" * 64,
        checkpoint_git_sha=SHA_BASE,
        current_git_sha=SHA_HEAD,
        sample_metadata={"sample_index": 0, "target_latent": {"sha256": "d" * 64}},
        probe_seed=123,
        timesteps=generator.get_scheduler().timesteps,
        optimizer_step=1,
        resolved_cli={"mode": "train"},
        final_metrics={
            "loss": result["post_update_loss"],
            "summary": result["post_update_summary"],
            "timestep_records": result["post_update_records"],
        },
    )
    restored = FakeGenerator()
    probe.load_probe_modules_into_generator(restored, payload)
    with torch.no_grad():
        restored_loss, _, restored_summary = probe.fixed_grid_loss_and_records(
            **_loss_kwargs(restored)
        )

    assert payload["final_metrics"]["loss"] == pytest.approx(restored_loss.item())
    assert payload["final_metrics"]["summary"] == restored_summary


def test_lightweight_checkpoint_save_load_and_restore_exact(tmp_path) -> None:
    generator = FakeGenerator()
    with torch.no_grad():
        generator.mcp.fusion.bias.fill_(1.25)
        generator.mcp.mcp_modules[0].flow.fill_(2.5)
    payload = probe.make_probe_checkpoint_payload(
        generator=generator,
        base_m3_checkpoint_path=probe.ROOT / "checkpoints" / "base.pt",
        base_m3_checkpoint_sha256="c" * 64,
        checkpoint_git_sha=SHA_BASE,
        current_git_sha=SHA_HEAD,
        sample_metadata={"sample_index": 0, "target_latent": {"sha256": "d" * 64}},
        probe_seed=123,
        timesteps=generator.get_scheduler().timesteps,
        optimizer_step=10,
        resolved_cli={"mode": "train"},
        final_metrics={"mean_flow_mse": 0.0},
    )
    path = tmp_path / "probe_checkpoint.pt"
    probe.save_probe_checkpoint(payload, path)
    loaded = probe.load_probe_checkpoint(path)
    restored = FakeGenerator()
    report = probe.load_probe_modules_into_generator(restored, loaded)

    assert loaded["format"] == "nf_sf_m3_fixed_grid_probe_checkpoint_v1"
    assert set(loaded.keys()) >= {"fusion", "mcp_depth1", "base_m3_checkpoint"}
    assert "generator" not in loaded
    assert "optimizer" not in loaded
    assert "backbone" not in loaded
    assert report["exact"] is True
    assert torch.equal(restored.mcp.fusion.bias, generator.mcp.fusion.bias)
    assert torch.equal(restored.mcp.mcp_modules[0].flow, generator.mcp.mcp_modules[0].flow)


def test_global_step_must_be_one_hundred() -> None:
    probe.validate_step100_checkpoint({"global_step": 100})
    with pytest.raises(RuntimeError, match="global_step=100"):
        probe.validate_step100_checkpoint({"global_step": 10})


def _fake_git(*, diff_text: str, status_text: str = "", ancestor: bool = True):
    def git_text(command):
        command = list(command)
        if command == ["git", "rev-parse", "HEAD"]:
            return SHA_HEAD
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


def test_provenance_accepts_exact_four_allowed_added_files() -> None:
    diff_text = "\n".join(
        f"A\t{path}" for path in probe.ALLOWED_FIXED_GRID_PROVENANCE_FILES
    )
    git_text, git_success = _fake_git(diff_text=diff_text)

    report = probe.fixed_grid_provenance_gate(
        checkpoint_payload={"git_sha": SHA_BASE},
        git_text=git_text,
        git_success=git_success,
    )

    assert report["status"] == "PASS"
    assert len(report["git_diff_entries"]) == 4


@pytest.mark.parametrize(
    "diff_text",
    [
        "",
        "A\tscripts/run_nf_sf_m3_fixed_grid_probe.py",
        "M\tutils/nf_sf_m3.py",
    ],
)
def test_provenance_rejects_non_exact_diff(diff_text: str) -> None:
    git_text, git_success = _fake_git(diff_text=diff_text)

    with pytest.raises(RuntimeError, match="expected=.*actual="):
        probe.fixed_grid_provenance_gate(
            checkpoint_payload={"git_sha": SHA_BASE},
            git_text=git_text,
            git_success=git_success,
        )


def test_provenance_rejects_dirty_or_non_ancestor() -> None:
    git_text, git_success = _fake_git(diff_text="", status_text=" M utils/nf_sf_m3.py")
    with pytest.raises(RuntimeError, match="worktree is dirty"):
        probe.fixed_grid_provenance_gate(
            checkpoint_payload={"git_sha": SHA_BASE},
            git_text=git_text,
            git_success=git_success,
        )

    git_text, git_success = _fake_git(diff_text="", ancestor=False)
    with pytest.raises(RuntimeError, match="not an ancestor"):
        probe.fixed_grid_provenance_gate(
            checkpoint_payload={"git_sha": SHA_BASE},
            git_text=git_text,
            git_success=git_success,
        )


def test_top_level_triple_and_single_mcp_flow_contract() -> None:
    valid = FakeGenerator()
    loss, records, _ = probe.fixed_grid_loss_and_records(**_loss_kwargs(valid))
    assert loss.item() >= 0.0
    assert len(records) == 4

    bad_top = FakeGenerator(top_level_output_count=2)
    with pytest.raises(RuntimeError, match="exactly three generator outputs"):
        probe.fixed_grid_loss_and_records(**_loss_kwargs(bad_top))

    bad_mcp = FakeGenerator(mcp_output_count=2)
    with pytest.raises(RuntimeError, match="exactly one MCP flow output"):
        probe.fixed_grid_loss_and_records(**_loss_kwargs(bad_mcp))
