from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

import utils.nf_sf_first_mcp_flow_audit as audit
import utils.nf_sf_full_sequence_eval as ev
from utils.nf_sf_tensors import DEFAULT_S_MAIN, DEFAULT_S_MCP
from utils.scheduler import FlowMatchScheduler


RUNTIME_GIT_SHA = "d" * 40
TRAINING_GIT_SHA = ev.TRAINING_CHECKPOINT_GIT_SHA
TEST_SHA = "a" * 64
FRAME_SEQ_LENGTH = 2


class CountingFlowMatchScheduler(FlowMatchScheduler):
    def __init__(self, *, shift: float) -> None:
        super().__init__(shift=shift, sigma_min=0.0, extra_one_step=True)
        self.set_timesteps(1000, training=True)
        self.training_target_calls = 0

    def training_target(self, sample, noise, timestep):
        self.training_target_calls += 1
        return super().training_target(sample, noise, timestep)


class FakeGenerator(nn.Module):
    def __init__(
        self,
        *,
        consume_rng_in_mcp_forward: bool = False,
        mcp_nonfinite: str | None = None,
    ) -> None:
        super().__init__()
        self.calls: list[dict] = []
        self.mcp_call_count = 0
        self.consume_rng_in_mcp_forward = bool(consume_rng_in_mcp_forward)
        self.mcp_nonfinite = mcp_nonfinite

    def forward(self, **kwargs):
        current = kwargs["noisy_image_or_video"]
        timestep = kwargs["timestep"]
        kv_cache = kwargs["kv_cache"]
        current_start = int(kwargs["current_start"])
        mcp_requested = kwargs.get("mcp_future_noises") is not None
        if mcp_requested:
            self.mcp_call_count += 1
            if self.consume_rng_in_mcp_forward:
                torch.randn((), device=current.device)
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
                "is_context": bool((timestep.detach().float() == 0).all().item()),
                "current_start": current_start,
                "token_end": token_end,
            }
        )
        for layer in kv_cache:
            layer["k"][:, current_start:token_end] = 1.0 + len(self.calls)
            layer["v"][:, current_start:token_end] = 2.0 + len(self.calls)
            layer["global_end_index"].fill_(token_end)
            layer["local_end_index"].fill_(token_end)
        main_flow = torch.zeros_like(current)
        main_x0 = current
        if mcp_requested:
            future = kwargs["mcp_future_noises"][0]
            if self.mcp_nonfinite == "nan":
                mcp_flow = torch.full_like(future, float("nan"))
            elif self.mcp_nonfinite == "inf":
                mcp_flow = torch.full_like(future, float("inf"))
            else:
                mcp_flow = future * 0.125 + 0.25
            return main_flow, main_x0, [mcp_flow]
        return main_flow, main_x0


def make_runtime(generator: FakeGenerator) -> ev.DeploymentRuntime:
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
        scheduler=audit.build_flow_match_scheduler(
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


def make_audit_result(
    *,
    mcp_scheduler: FlowMatchScheduler | None = None,
    generator: FakeGenerator | None = None,
) -> audit.FirstMCPFlowAuditResult:
    generator = generator or FakeGenerator()
    source = make_source_noise()
    teacher = make_teacher_target()
    common, fingerprint = make_common(source, teacher)
    scheduler = mcp_scheduler or audit.build_flow_match_scheduler(
        shift=DEFAULT_S_MCP,
        device=torch.device("cpu"),
    )
    return audit.run_first_mcp_flow_audit(
        runtime_factory=lambda: make_runtime(generator),
        mcp_scheduler=scheduler,
        source_noise=source,
        teacher_target=teacher,
        teacher_payload={"rollout_seed": 123, "prompt": "prompt"},
        conditional_dict={"prompt_embeds": torch.zeros((1, 2, 3))},
        checkpoint_summary={"type": "full_sequence_step5000", "sha256": TEST_SHA},
        common_inputs=common,
        common_inputs_fingerprint_sha256=fingerprint,
        runtime_git_sha=RUNTIME_GIT_SHA,
        training_checkpoint_git_sha=TRAINING_GIT_SHA,
    )


def test_flow_match_exact_algebra_returns_clean() -> None:
    scheduler = audit.build_flow_match_scheduler(
        shift=DEFAULT_S_MCP,
        device=torch.device("cpu"),
    )
    clean = torch.randn((3, 1, 2, 2), dtype=torch.float32)
    noise = torch.randn_like(clean)
    timestep = torch.full((3,), ev.MCP_DEPLOYMENT_SCHEDULE[1])
    xt = scheduler.add_noise(clean, noise, timestep)
    target = scheduler.training_target(clean, noise, timestep)
    x0 = scheduler.step(target, timestep, xt, to_final=True)
    assert torch.allclose(x0, clean, atol=1e-6)


def test_step0_predicted_and_teacher_probe_mcp_flow_sha_match() -> None:
    result = make_audit_result()
    step0 = result.manifest["step0_decisive_metrics"]
    assert step0["predicted_vs_teacher_probe_input_exact"] is True
    assert step0["predicted_vs_teacher_probe_flow_sha_exact"] is True


def test_predicted_oracle_and_probe_have_four_raw_steps() -> None:
    result = make_audit_result()
    branches = result.manifest["branches"]
    for key in (
        audit.PREDICTED_ROLLOUT,
        audit.ORACLE_FLOW_ROLLOUT,
        audit.TEACHER_STATE_PROBE,
    ):
        steps = branches[key]["steps"]
        assert [step["raw_index"] for step in steps] == [0, 1, 2, 3]
        assert [step["raw_timestep"] for step in steps] == list(ev.RAW_DEPLOYMENT_SCHEDULE)


def test_oracle_main_trajectory_matches_predicted_exactly() -> None:
    result = make_audit_result()
    assert result.manifest["branches"][audit.ORACLE_FLOW_ROLLOUT][
        "main_trajectory_matches_predicted"
    ] is True


def test_oracle_uses_exact_flow_and_predicted_uses_model_flow() -> None:
    result = make_audit_result()
    branches = result.manifest["branches"]
    assert branches[audit.ORACLE_FLOW_ROLLOUT]["used_exact_flow_for_transition"] is True
    assert branches[audit.PREDICTED_ROLLOUT]["used_model_flow_for_transition"] is True
    oracle_mse = branches[audit.ORACLE_FLOW_ROLLOUT]["steps"][-1][
        "oracle_x0_vs_teacher_mse"
    ]
    predicted_mse = branches[audit.PREDICTED_ROLLOUT]["steps"][-1][
        "predicted_x0_vs_teacher_mse"
    ]
    assert oracle_mse < 1.0e-10
    assert predicted_mse > oracle_mse


def test_predicted_teacher_directed_step1_uses_actual_state_implied_noise() -> None:
    result = make_audit_result()
    source = make_source_noise()
    teacher = make_teacher_target()
    scheduler = audit.build_flow_match_scheduler(
        shift=DEFAULT_S_MCP,
        device=torch.device("cpu"),
    )
    rng_plan = ev.build_absolute_chunk_rng_plan(source_noise=source, rollout_seed=123)
    planned_noise = rng_plan["transition_noises"][(audit.FUTURE_CHUNK_INDEX, 0)].unflatten(
        0,
        (1, ev.FULL_SEQUENCE_CHUNK_FRAMES),
    )
    teacher_chunk2 = teacher[:, 6:9]
    wrong_planned_target = scheduler.training_target(
        teacher_chunk2.flatten(0, 1),
        planned_noise.flatten(0, 1),
        torch.full(
            (ev.FULL_SEQUENCE_CHUNK_FRAMES,),
            ev.MCP_DEPLOYMENT_SCHEDULE[1],
            dtype=torch.float32,
        ),
    ).unflatten(0, teacher_chunk2.shape[:2])
    recorded = result.tensors["predicted_teacher_directed_flows"][1]
    assert result.manifest["branches"][audit.PREDICTED_ROLLOUT]["steps"][1][
        "teacher_directed_target_source"
    ] == "actual_state_implied_noise"
    assert not torch.equal(recorded, wrong_planned_target)


def test_teacher_state_probe_targets_call_scheduler_training_target() -> None:
    scheduler = CountingFlowMatchScheduler(shift=DEFAULT_S_MCP)
    _ = make_audit_result(mcp_scheduler=scheduler)
    assert scheduler.training_target_calls >= 12


def test_depth_and_chunk_contract_are_exact() -> None:
    result = make_audit_result()
    contract = result.manifest["chunk_contract"]
    assert contract["history_chunks"] == [0]
    assert contract["current_chunk"] == 1
    assert contract["future_chunk"] == 2
    assert contract["depths_used"] == [1]
    for branch in result.manifest["branches"].values():
        assert branch["mcp_depths_used"] == [1]
        assert branch["history_chunks"] == [0]
        assert branch["current_chunk"] == 1
        assert branch["future_chunk"] == 2


def test_source_rng_and_common_fingerprints_are_recorded() -> None:
    result = make_audit_result()
    manifest = result.manifest
    assert manifest["common_inputs_fingerprint_sha256"]
    assert manifest["rng_plan_fingerprint_sha256"]
    assert manifest["input_tensors"]["source_noise_chunk1_sha256"]
    assert manifest["input_tensors"]["source_noise_chunk2_sha256"]


def test_joint_forward_rng_guards_are_recorded_and_unchanged() -> None:
    result = make_audit_result()
    for branch in result.manifest["branches"].values():
        for step in branch["steps"]:
            guard = step["joint_forward_rng"]
            assert guard["state_before_hash"] == guard["state_after_hash"]
            assert guard["unchanged"] is True


def test_joint_forward_rng_consumption_rejects() -> None:
    generator = FakeGenerator(consume_rng_in_mcp_forward=True)
    with pytest.raises(RuntimeError, match="first_mcp_joint_forward changed active global RNG state"):
        make_audit_result(generator=generator)


@pytest.mark.parametrize("kind", ["nan", "inf"])
def test_nonfinite_mcp_flow_rejects(kind: str) -> None:
    generator = FakeGenerator(mcp_nonfinite=kind)
    with pytest.raises(RuntimeError, match="nonfinite"):
        make_audit_result(generator=generator)


def test_hybrid_latents_only_replace_chunk2() -> None:
    teacher = make_teacher_target()
    replacement = torch.full_like(teacher[:, 6:9], 99.0)
    hybrid = audit.build_chunk2_hybrid_latent(teacher, replacement_chunk=replacement)
    assert torch.equal(hybrid[:, :6], teacher[:, :6])
    assert torch.equal(hybrid[:, 6:9], replacement)
    assert torch.equal(hybrid[:, 9:], teacher[:, 9:])


def test_manifest_hypothesis_fields_are_null() -> None:
    result = make_audit_result()
    contract = result.manifest["interpretation_contract"]
    assert contract["model_flow_failure_supported"] is None
    assert contract["solver_semantics_failure_supported"] is None
    assert contract["solver_state_distribution_drift_supported"] is None
    assert contract["training_like_state_failure_supported"] is None


def test_manifest_validator_rejects_step0_flow_mismatch() -> None:
    result = make_audit_result()
    manifest = dict(result.manifest)
    manifest["step0_decisive_metrics"] = dict(manifest["step0_decisive_metrics"])
    manifest["step0_decisive_metrics"][
        "predicted_vs_teacher_probe_flow_sha_exact"
    ] = False
    try:
        audit.validate_first_mcp_flow_audit_manifest(manifest)
    except RuntimeError as exc:
        assert "flow SHA" in str(exc)
    else:
        raise AssertionError("validator accepted step0 flow mismatch")


def test_manifest_validator_rejects_changed_joint_forward_rng_guard() -> None:
    result = make_audit_result()
    manifest = dict(result.manifest)
    manifest["branches"] = {
        key: dict(value) for key, value in result.manifest["branches"].items()
    }
    manifest["branches"][audit.PREDICTED_ROLLOUT]["steps"] = [
        dict(step)
        for step in result.manifest["branches"][audit.PREDICTED_ROLLOUT]["steps"]
    ]
    manifest["branches"][audit.PREDICTED_ROLLOUT]["steps"][0]["joint_forward_rng"] = dict(
        manifest["branches"][audit.PREDICTED_ROLLOUT]["steps"][0]["joint_forward_rng"]
    )
    manifest["branches"][audit.PREDICTED_ROLLOUT]["steps"][0]["joint_forward_rng"][
        "unchanged"
    ] = False
    with pytest.raises(RuntimeError, match="joint-forward RNG guard"):
        audit.validate_first_mcp_flow_audit_manifest(manifest)


def test_source_guard_no_forbidden_old_oracle_imports() -> None:
    repo = Path(__file__).resolve().parents[2]
    sources = [
        repo / "utils" / "nf_sf_first_mcp_flow_audit.py",
        repo / "scripts" / "diagnose_nf_sf_first_mcp_flow.py",
    ]
    forbidden = (
        "import inference_next_forcing",
        "from inference_next_forcing",
        "import inference_mcp",
        "from inference_mcp",
        "nf_sf_m6",
    )
    for path in sources:
        text = path.read_text(encoding="utf-8")
        for pattern in forbidden:
            assert pattern not in text
