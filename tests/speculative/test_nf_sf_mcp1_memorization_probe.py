from __future__ import annotations

import argparse

import pytest
import torch
from torch import nn

import scripts.run_nf_sf_mcp1_memorization_probe as runner
import utils.nf_sf_first_mcp_flow_audit as flow_audit
import utils.nf_sf_full_sequence_eval as ev
import utils.nf_sf_mcp1_memorization_probe as probe
from utils.nf_sf_tensors import DEFAULT_S_MAIN, DEFAULT_S_MCP


RUNTIME_GIT_SHA = "d" * 40
TRAINING_GIT_SHA = "c" * 40
TEST_SHA = "a" * 64
FRAME_SEQ_LENGTH = 1


class FakeDepth(nn.Module):
    def __init__(self, *, proj: nn.Linear | None = None) -> None:
        super().__init__()
        self.proj = proj or nn.Linear(2, 1)
        self.offset = nn.Parameter(torch.zeros(()))

    def forward(self, future: torch.Tensor, fused: torch.Tensor) -> torch.Tensor:
        fused_expanded = fused.expand_as(future)
        stacked = torch.stack((future, fused_expanded), dim=-1)
        return self.proj(stacked).squeeze(-1) + self.offset


class FakeMCP(nn.Module):
    def __init__(self, *, alias_allowed_param: bool = False) -> None:
        super().__init__()
        if alias_allowed_param:
            shared = nn.Linear(2, 1)
            self.fusion = shared
            self.mcp_modules = nn.ModuleList(
                [FakeDepth(proj=shared), FakeDepth(), FakeDepth()]
            )
        else:
            self.fusion = nn.Linear(1, 1)
            self.mcp_modules = nn.ModuleList([FakeDepth(), FakeDepth(), FakeDepth()])


class FakeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.patch_embedding = nn.Linear(1, 1, bias=False)
        self.backbone = nn.Linear(1, 1, bias=False)
        self.gradient_checkpointing = False


class FakeGenerator(nn.Module):
    def __init__(
        self,
        *,
        call_depth2: bool = False,
        alias_allowed_param: bool = False,
    ) -> None:
        super().__init__()
        self.model = FakeModel()
        self.mcp = FakeMCP(alias_allowed_param=alias_allowed_param)
        self.call_depth2 = bool(call_depth2)
        self.forward_calls = []

    def forward(self, **kwargs):
        current = kwargs["noisy_image_or_video"]
        timestep = kwargs["timestep"]
        kv_cache = kwargs.get("kv_cache")
        current_start = int(kwargs.get("current_start") or 0)
        run_mcp = kwargs.get("mcp_future_noises") is not None
        self.forward_calls.append(
            {
                "run_mcp": run_mcp,
                "mcp_future_count": (
                    None
                    if kwargs.get("mcp_future_noises") is None
                    else len(kwargs["mcp_future_noises"])
                ),
                "timestep_sha256": ev.tensor_sha256(timestep.detach().cpu()),
            }
        )
        if kv_cache is not None:
            token_count = int(current.shape[1]) * FRAME_SEQ_LENGTH
            token_end = current_start + token_count
            fill = float(current.detach().float().mean().item()) + len(self.forward_calls)
            for layer in kv_cache:
                layer["k"][:, current_start:token_end] = fill
                layer["v"][:, current_start:token_end] = fill + 1.0
                layer["global_end_index"].fill_(token_end)
                layer["local_end_index"].fill_(token_end)
        main_flow = torch.zeros_like(current)
        main_x0 = current
        if not run_mcp:
            return main_flow, main_x0
        future = kwargs["mcp_future_noises"][0]
        fused = self.mcp.fusion(current.float().mean(dim=(1, 2, 3, 4), keepdim=False).reshape(-1, 1))
        fused = fused.to(device=future.device, dtype=future.dtype).reshape(-1, 1, 1, 1, 1)
        if self.call_depth2:
            _ = self.mcp.mcp_modules[1](future, fused)
        mcp_flow = self.mcp.mcp_modules[0](future, fused)
        return main_flow, main_x0, [mcp_flow]


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
    crossattn_cache = [
        {
            "is_init": True,
            "k": torch.zeros((1, 2, 1, 1), dtype=torch.float32),
            "v": torch.ones((1, 2, 1, 1), dtype=torch.float32),
        }
    ]
    return ev.DeploymentRuntime(
        generator=generator,
        scheduler=probe.build_memorization_flow_scheduler(
            shift=DEFAULT_S_MAIN,
            device=torch.device("cpu"),
        ),
        kv_cache=kv_cache,
        crossattn_cache=crossattn_cache,
        frame_seq_length=FRAME_SEQ_LENGTH,
        num_frame_per_block=ev.FULL_SEQUENCE_CHUNK_FRAMES,
        context_noise=0,
    )


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


def make_probe_result(
    *,
    generator: FakeGenerator | None = None,
    optimizer_steps: int = 1,
) -> probe.MCP1MemorizationProbeResult:
    generator = generator or FakeGenerator()
    source = make_source_noise()
    teacher = make_teacher_target()
    common, fingerprint = make_common(source, teacher)
    return probe.run_mcp1_memorization_probe(
        runtime_factory=lambda: make_runtime(generator),
        source_noise=source,
        teacher_target=teacher,
        teacher_payload={"rollout_seed": 123, "prompt": "prompt"},
        conditional_dict={"prompt_embeds": torch.zeros((1, 2, 3))},
        checkpoint_summary={"type": "full_sequence_step6500", "sha256": TEST_SHA},
        common_inputs=common,
        common_inputs_fingerprint_sha256=fingerprint,
        runtime_git_sha=RUNTIME_GIT_SHA,
        training_checkpoint_git_sha=TRAINING_GIT_SHA,
        optimizer_steps=optimizer_steps,
        optimizer_lr=1.0e-2,
        log_interval=1,
        noise_seed=77,
    )


def build_recache_snapshot_and_states(
    generator: FakeGenerator | None = None,
) -> tuple[
    ev.DeploymentRuntime,
    tuple[probe.MCP1MemorizationState, ...],
    probe.MCP1ProbePristineCacheSnapshot,
]:
    generator = generator or FakeGenerator()
    source = make_source_noise()
    teacher = make_teacher_target()
    runtime = make_runtime(generator)
    rng_plan = ev.build_absolute_chunk_rng_plan(
        source_noise=source,
        rollout_seed=123,
        num_denoising_steps=len(ev.RAW_DEPLOYMENT_SCHEDULE),
        chunk_frames=ev.FULL_SEQUENCE_CHUNK_FRAMES,
    )
    with torch.no_grad():
        history_recache = flow_audit._recache_teacher_history0(
            runtime=runtime,
            source_noise=source,
            teacher_target=teacher,
            conditional_dict={"prompt_embeds": torch.zeros((1, 2, 3))},
            rng_plan=rng_plan,
        )
    states, _ = probe.build_mcp1_memorization_states(
        source_noise=source,
        teacher_target=teacher,
        main_scheduler=probe.build_memorization_flow_scheduler(
            shift=DEFAULT_S_MAIN,
            device="cpu",
        ),
        mcp_scheduler=probe.build_memorization_flow_scheduler(
            shift=DEFAULT_S_MCP,
            device="cpu",
        ),
        noise_seed=77,
    )
    snapshot = probe.build_pristine_cache_snapshot(
        runtime=runtime,
        history_recache=history_recache,
    )
    return runtime, states, snapshot


def test_exact_state_construction_is_deterministic() -> None:
    source = make_source_noise()
    teacher = make_teacher_target()
    main_scheduler = probe.build_memorization_flow_scheduler(
        shift=DEFAULT_S_MAIN,
        device="cpu",
    )
    mcp_scheduler = probe.build_memorization_flow_scheduler(
        shift=DEFAULT_S_MCP,
        device="cpu",
    )
    states_a, provenance_a = probe.build_mcp1_memorization_states(
        source_noise=source,
        teacher_target=teacher,
        main_scheduler=main_scheduler,
        mcp_scheduler=mcp_scheduler,
        noise_seed=11,
    )
    states_b, provenance_b = probe.build_mcp1_memorization_states(
        source_noise=source,
        teacher_target=teacher,
        main_scheduler=main_scheduler,
        mcp_scheduler=mcp_scheduler,
        noise_seed=11,
    )
    assert provenance_a["state_collection_fingerprint_sha256"] == provenance_b[
        "state_collection_fingerprint_sha256"
    ]
    for left, right in zip(states_a, states_b):
        assert left.state_id == right.state_id
        assert torch.equal(left.current_state, right.current_state)
        assert torch.equal(left.future_state, right.future_state)
        assert torch.equal(left.mcp_target, right.mcp_target)


def test_sixteen_states_identity_and_fingerprint_are_stable() -> None:
    states, provenance = probe.build_mcp1_memorization_states(
        source_noise=make_source_noise(),
        teacher_target=make_teacher_target(),
        main_scheduler=probe.build_memorization_flow_scheduler(
            shift=DEFAULT_S_MAIN,
            device="cpu",
        ),
        mcp_scheduler=probe.build_memorization_flow_scheduler(
            shift=DEFAULT_S_MCP,
            device="cpu",
        ),
    )
    assert len(states) == 16
    assert [state.raw_timestep for state in states] == [
        raw for raw in probe.RAW_TIMESTEPS for _ in range(4)
    ]
    assert [state.noise_index for state in states] == [0, 1, 2, 3] * 4
    assert provenance["state_count"] == 16
    assert len(provenance["state_collection_fingerprint_sha256"]) == 64


def test_only_mcp1_and_fusion_parameters_are_trainable() -> None:
    generator = FakeGenerator()
    selection = probe.configure_stage_a_trainable_parameters(generator, lr=1.0e-3)
    names = selection.summary["trainable_parameter_names"]
    assert names
    assert any(name.startswith("mcp.fusion.") for name in names)
    assert any(name.startswith("mcp.mcp_modules.0.proj.") for name in names)
    assert all(
        name.startswith("mcp.fusion.") or name.startswith("mcp.mcp_modules.0.")
        for name in names
    )
    assert not any("mcp_modules.1" in name or "mcp_modules.2" in name for name in names)
    assert selection.summary["component_contract"]["patch_embedding"].startswith("frozen")
    optimizer = torch.optim.AdamW(list(selection.optimizer_param_groups), lr=1.0e-3)
    allowed_ids = probe.allowed_stage_a_param_ids(selection)
    optimizer_ids = {
        id(param)
        for group in optimizer.param_groups
        for param in group.get("params", ())
    }
    assert optimizer_ids == allowed_ids
    probe.validate_optimizer_excludes_main(
        generator,
        optimizer,
        allowed_param_ids=allowed_ids,
    )
    probe.assert_stage_a_fail_closed(generator)


def test_run_keeps_main_parameters_exactly_unchanged() -> None:
    result = make_probe_result(optimizer_steps=1)
    proof = result.manifest["main_parameters_exact_unchanged_proof"]
    assert proof["all_sha256_exact_match"] is True
    assert proof["mismatch_parameter_names"] == []
    assert result.manifest["parameter_delta_norm"]["aggregate_l2"] > 0.0


def test_pristine_kv_fingerprint_is_fixed_after_history_recache() -> None:
    runtime, _states, snapshot = build_recache_snapshot_and_states()
    snapshot.assert_current_matches_reference(runtime, label="after_capture")
    assert len(snapshot.reference_kv_fingerprint_sha256) == 64
    assert len(snapshot.reference_crossattn_fingerprint_sha256) == 64
    write_set = snapshot.kv_write_set
    assert write_set["history_end_tokens"] == 3
    assert write_set["current_start_tokens"] == 3
    assert write_set["current_tokens"] == 3
    for layer in write_set["layers"]:
        assert layer["write_windows"][-1] == {
            "role": "current_chunk1_main_kv_write_window",
            "start": 3,
            "end": 6,
        }
        assert layer["mcp_future_writes_runtime_kv"] is False


def test_state_forward_write_window_restores_exactly() -> None:
    runtime, states, snapshot = build_recache_snapshot_and_states()
    with torch.no_grad():
        probe.call_isolated_mcp1_joint_forward(
            runtime=runtime,
            conditional_dict={"prompt_embeds": torch.zeros((1, 2, 3))},
            state=states[0],
            pristine_cache_snapshot=snapshot,
            pre_forward_phase="test_state_a_pre_forward",
            restore_after_forward=True,
            post_forward_phase="test_state_a_post_forward",
        )
    snapshot.assert_current_matches_reference(runtime, label="after_state_a")


def test_state_b_starts_from_reference_after_state_a() -> None:
    runtime, states, snapshot = build_recache_snapshot_and_states()
    with torch.no_grad():
        probe.call_isolated_mcp1_joint_forward(
            runtime=runtime,
            conditional_dict={"prompt_embeds": torch.zeros((1, 2, 3))},
            state=states[0],
            pristine_cache_snapshot=snapshot,
            pre_forward_phase="test_state_a_pre",
            restore_after_forward=True,
            post_forward_phase="test_state_a_post",
        )
        snapshot.assert_current_matches_reference(runtime, label="before_state_b")
        probe.call_isolated_mcp1_joint_forward(
            runtime=runtime,
            conditional_dict={"prompt_embeds": torch.zeros((1, 2, 3))},
            state=states[1],
            pristine_cache_snapshot=snapshot,
            pre_forward_phase="test_state_b_pre",
            restore_after_forward=True,
            post_forward_phase="test_state_b_post",
        )
    snapshot.assert_current_matches_reference(runtime, label="after_state_b")


def test_multiple_states_do_not_accumulate_kv() -> None:
    runtime, states, snapshot = build_recache_snapshot_and_states()
    with torch.no_grad():
        for state in states[:4]:
            probe.call_isolated_mcp1_joint_forward(
                runtime=runtime,
                conditional_dict={"prompt_embeds": torch.zeros((1, 2, 3))},
                state=state,
                pristine_cache_snapshot=snapshot,
                pre_forward_phase="test_multi_state_pre",
                restore_after_forward=True,
                post_forward_phase="test_multi_state_post",
            )
            snapshot.assert_current_matches_reference(
                runtime,
                label=f"after_{state.state_id}",
            )
    counts = snapshot.manifest_record()["phase_counts"]
    assert counts["test_multi_state_pre"] == 4
    assert counts["test_multi_state_post"] == 4


def test_multiple_optimizer_steps_do_not_accumulate_kv() -> None:
    result = make_probe_result(optimizer_steps=2)
    counts = result.manifest["kv_isolation"]["phase_counts"]
    assert counts["train_pre_forward"] == 32
    assert counts["train_post_backward"] == 32
    assert counts["post_optimizer_step"] == 2
    assert counts["eval_pre_forward"] == 32
    assert counts["eval_post_forward"] == 32


def test_write_window_mutation_is_fail_closed() -> None:
    runtime, _states, snapshot = build_recache_snapshot_and_states()
    window = snapshot.kv_write_set["layers"][0]["write_windows"][-1]
    start = int(window["start"])
    end = int(window["end"])
    runtime.kv_cache[0]["k"][:, start:end].fill_(99.0)
    with pytest.raises(RuntimeError, match="pristine KV fingerprint mismatch"):
        snapshot.assert_current_matches_reference(runtime, label="tampered_write_window")
    snapshot.restore_and_verify(runtime, phase="restore_after_tamper")


def test_crossattn_mutation_is_fail_closed() -> None:
    runtime, _states, snapshot = build_recache_snapshot_and_states()
    runtime.crossattn_cache[0]["k"].add_(1.0)
    with pytest.raises(RuntimeError, match="pristine cross-attn fingerprint mismatch"):
        snapshot.assert_current_matches_reference(runtime, label="tampered_crossattn")
    snapshot.restore_and_verify(runtime, phase="restore_after_crossattn_tamper")


def test_training_step_has_allowed_gradients_only() -> None:
    result = make_probe_result(optimizer_steps=1)
    audit = result.manifest["last_gradient_audit"]
    assert audit["allowed_finite_nonzero_gradient_count"] > 0
    assert audit["main_and_mcp23_gradients_absent_or_zero"] is True


def test_duplicate_parameter_alias_is_fail_closed() -> None:
    generator = FakeGenerator(alias_allowed_param=True)
    with pytest.raises(RuntimeError, match="appears more than once"):
        probe.configure_stage_a_trainable_parameters(generator, lr=1.0e-3)


def test_gradient_checkpointing_enabled_is_rejected() -> None:
    generator = FakeGenerator()
    generator.model.gradient_checkpointing = True
    with pytest.raises(RuntimeError, match="refuses gradient_checkpointing=True"):
        make_probe_result(generator=generator, optimizer_steps=0)


def test_mcp2_and_mcp3_forward_calls_are_forbidden() -> None:
    generator = FakeGenerator(call_depth2=True)
    with pytest.raises(RuntimeError, match="forbids depth2"):
        make_probe_result(generator=generator, optimizer_steps=0)


def test_flow_targets_are_exact_scheduler_targets() -> None:
    source = make_source_noise()
    teacher = make_teacher_target()
    main_scheduler = probe.build_memorization_flow_scheduler(
        shift=DEFAULT_S_MAIN,
        device="cpu",
    )
    mcp_scheduler = probe.build_memorization_flow_scheduler(
        shift=DEFAULT_S_MCP,
        device="cpu",
    )
    states, _ = probe.build_mcp1_memorization_states(
        source_noise=source,
        teacher_target=teacher,
        main_scheduler=main_scheduler,
        mcp_scheduler=mcp_scheduler,
    )
    state = states[5]
    teacher_chunk1 = teacher[:, 3:6]
    teacher_chunk2 = teacher[:, 6:9]
    expected_mcp = mcp_scheduler.training_target(
        teacher_chunk2.flatten(0, 1),
        state.future_noise.flatten(0, 1),
        state.mcp_timestep.flatten(0, 1),
    ).unflatten(0, teacher_chunk2.shape[:2])
    expected_current = main_scheduler.training_target(
        teacher_chunk1.flatten(0, 1),
        state.current_noise.flatten(0, 1),
        state.main_timestep.flatten(0, 1),
    ).unflatten(0, teacher_chunk1.shape[:2])
    assert torch.equal(state.mcp_target, expected_mcp)
    assert torch.equal(state.main_target, expected_current)


def test_metric_status_thresholds_are_pre_registered() -> None:
    strong = probe.evaluate_memorization_status(
        initial_mean_mse=10.0,
        initial_max_mse=20.0,
        final_mean_mse=0.5,
        final_max_mse=2.0,
    )
    weak = probe.evaluate_memorization_status(
        initial_mean_mse=10.0,
        initial_max_mse=20.0,
        final_mean_mse=0.51,
        final_max_mse=2.0,
    )
    assert strong["status"] == probe.STRONG_MEMORIZATION_SUPPORT
    assert weak["status"] == probe.INSUFFICIENT_MEMORIZATION
    assert strong["relative_reduction"]["mean"] == pytest.approx(0.95)
    assert strong["thresholds"]["final_mean_mse_max"] == pytest.approx(0.5)


def test_resume_and_probe_checkpoint_options_are_not_allowed() -> None:
    with pytest.raises(SystemExit):
        runner.parse_args(
            [
                "--full_sequence_checkpoint",
                "c.pt",
                "--sample_plan",
                "p.json",
                "--teacher_manifest",
                "t.json",
                "--dataset_root",
                "data",
                "--output_dir",
                "out",
                "--expected_runtime_git_sha",
                RUNTIME_GIT_SHA,
                "--resume",
                "bad.pt",
            ]
        )
    args = argparse.Namespace(resume=True)
    with pytest.raises(RuntimeError, match="forbids resume/checkpoint"):
        runner.validate_no_resume_checkpoint_contract(args)


def test_diagnostic_output_is_non_deployable_and_non_canonical() -> None:
    result = make_probe_result(optimizer_steps=0)
    manifest = result.manifest
    assert manifest["diagnostic_only"] is True
    assert manifest["non_deployable"] is True
    assert manifest["non_canonical"] is True
    assert manifest["canonical_training_eligible"] is False
    assert manifest["canonical_deployment_eligible"] is False
    assert manifest["checkpoint_output"]["written"] is False
    assert manifest["kv_isolation_mode"] == (
        "probe_local_pristine_history_plus_joint_write_window"
    )
    assert manifest["kv_isolation_verified_every_state"] is True
    assert manifest["kv_isolation_verified_every_optimizer_step"] is True
    assert len(manifest["pristine_history_kv_fingerprint_sha256"]) == 64
    assert len(manifest["pristine_crossattn_cache_fingerprint_sha256"]) == 64
    probe.validate_mcp1_memorization_manifest(manifest)


def test_optimizer_step_contract_rejects_main_parameters() -> None:
    generator = FakeGenerator()
    probe.configure_stage_a_trainable_parameters(generator, lr=1.0e-3)
    bad_optimizer = torch.optim.AdamW(generator.parameters(), lr=1.0e-3)
    with pytest.raises(RuntimeError, match="includes Main parameters"):
        probe.validate_optimizer_excludes_main(generator, bad_optimizer)
    generator.model.backbone.weight.requires_grad_(True)
    with pytest.raises(RuntimeError, match="Main parameter requires_grad=True"):
        probe.assert_stage_a_fail_closed(generator)
