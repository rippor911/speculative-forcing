import io
import json
from pathlib import Path

import pytest
import torch
from torch import nn

import utils.nf_sf_m3 as m3
from utils.nf_sf_m3 import (
    M3_CHECKPOINT_FORMAT,
    M3_REFERENCE_CHECKPOINT_SHA256,
    audit_parameter_changes,
    compare_probe_outputs,
    compare_serialized_probe_tensors,
    gradient_group_audit,
    load_m3_teacher_sample,
    make_m3_checkpoint_payload,
    make_m3_probe,
    prompt_sha256,
    reconstruct_main_current,
    reconstruct_mcp1_next,
    resolve_m3_solver_schedule,
    select_m3_selected_state,
    serialize_noisy_batch,
    solver_timesteps_from_scheduler,
    tensor_summary,
    validate_m3_checkpoint_pair,
    validate_m3_checkpoint_git_sha,
    validate_m3_mode,
)
from utils.nf_sf_tensors import make_cpu_generator
from utils.nf_sf_training import (
    configure_nf_sf_optimizer_plan,
    prepare_nf_sf_noisy_batch,
)
from utils.scheduler import FlowMatchScheduler


TEST_GIT_SHA = "a" * 40


def _scheduler(shift: float) -> FlowMatchScheduler:
    scheduler = FlowMatchScheduler(shift=shift, sigma_min=0.0, extra_one_step=True)
    scheduler.set_timesteps(1000, training=True)
    return scheduler


def _latent(num_frames: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.arange(num_frames, dtype=dtype).reshape(1, num_frames, 1, 1, 1)


def _state():
    return select_m3_selected_state(_latent(15))


def _teacher_manifest_and_payload(*, num_frames: int = 15):
    target = torch.arange(num_frames, dtype=torch.float32).reshape(
        1, num_frames, 1, 1, 1
    ).to(torch.bfloat16)
    source = (target.float() + 100.0).to(torch.bfloat16)
    prompt = "a test prompt"
    prompt_hash = prompt_sha256(prompt)
    payload = {
        "format": "self_forcing_teacher_v1",
        "sample_index": 0,
        "sample_id": "sample-train-0",
        "split": "train",
        "split_index": 0,
        "source_line_index": 7,
        "shard_id": 0,
        "plan_index": 0,
        "prompt": prompt,
        "prompt_sha256": prompt_hash,
        "seed": 1000000,
        "noise_seed": 1000000,
        "rollout_seed": 1000000,
        "source_noise": source,
        "target_latent": target,
        "backbone_sha256": M3_REFERENCE_CHECKPOINT_SHA256,
        "num_frames": num_frames,
        "num_frame_per_block": 3,
        "mcp_depth": 3,
        "raw_denoising_steps": [1000, 750, 500, 250],
        "warped_denoising_steps": [1000.0, 750.0, 500.0, 250.0],
        "writer_git_head": "abc",
    }
    manifest = {
        "status": "PASS",
        "experiment": "E0208C_teacher_rollout_formal",
        "format": "self_forcing_teacher_manifest_v2",
        "writer_format": "e0208_teacher_writer_v1",
        "checkpoint": {
            "path": "checkpoints/self_forcing_dmd.pt",
            "sha256": M3_REFERENCE_CHECKPOINT_SHA256,
        },
        "generation": {
            "num_samples": 2304,
            "num_train": 2048,
            "num_validation": 256,
            "num_frames": num_frames,
            "num_frame_per_block": 3,
            "num_blocks": num_frames // 3,
            "mcp_depth": 3,
            "mcp_num_modules": 0,
            "mcp_accel_depths": 0,
            "last_step_only": True,
        },
        "samples": [
            {
                "sample_index": 0,
                "sample_id": "sample-train-0",
                "split": "train",
                "split_index": 0,
                "source_line_index": 7,
                "shard_id": 0,
                "plan_index": 0,
                "file": "/dataset/teacher_train_000000.pt",
                "file_sha256": "payload-file-sha",
                "prompt": prompt,
                "prompt_sha256": prompt_hash,
                "target_latent": tensor_summary(target),
                "source_noise": tensor_summary(source),
            }
        ],
    }
    return manifest, payload


def _load_teacher_with_monkeypatch(monkeypatch, manifest, payload):
    manifest_path = Path("formal_manifest.json")
    payload_path = Path("teacher_train_000000.pt")
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda self, encoding=None: json.dumps(manifest),
    )
    monkeypatch.setattr(
        m3,
        "resolve_payload_path",
        lambda *, manifest_path, record, dataset_root=None: payload_path,
    )
    monkeypatch.setattr(
        m3,
        "file_sha256",
        lambda path: "payload-file-sha" if Path(path) == payload_path else "manifest-sha",
    )
    monkeypatch.setattr(
        torch,
        "load",
        lambda path, map_location=None, weights_only=False: payload,
    )
    return load_m3_teacher_sample(manifest_path=manifest_path, sample_index=0)


def _solver_payload(num_steps: int = 4, shift: float = 5.0) -> dict:
    scheduler = FlowMatchScheduler(shift=shift, sigma_min=0.0, extra_one_step=True)
    timesteps = solver_timesteps_from_scheduler(
        scheduler,
        num_inference_steps=num_steps,
        device="cpu",
    )
    return {
        "raw_denoising_steps": [float(index) for index in range(num_steps)],
        "warped_denoising_steps": [
            float(value)
            for value in timesteps.detach().cpu().tolist()
        ],
    }


def _probe_outputs_for(probe):
    return {
        "main_flow_pred": torch.zeros_like(probe.noisy_batch.target_flow_main),
        "mcp_depth1_flow_pred": torch.zeros_like(
            probe.noisy_batch.target_flow_depths[0]
        ),
        "mcp_depth2_flow_pred": torch.zeros_like(
            probe.noisy_batch.target_flow_depths[1]
        ),
        "mcp_depth3_flow_pred": torch.zeros_like(
            probe.noisy_batch.target_flow_depths[2]
        ),
    }


def _checkpoint_for_pair(*, global_step: int, probe_seed: int = 44) -> dict:
    state = _state()
    probe = make_m3_probe(
        state,
        scheduler_main=_scheduler(5.0),
        scheduler_mcp=_scheduler(10.0),
        seed=probe_seed,
    )
    model = nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(
        [{"name": "linear", "params": list(model.parameters()), "lr": 1.0e-4}],
        weight_decay=0.01,
    )
    return make_m3_checkpoint_payload(
        generator=model,
        optimizer=optimizer,
        global_step=global_step,
        train_rng=make_cpu_generator(55),
        probe=probe,
        probe_summary={
            "probe_losses": {
                "main_loss": 1,
                "mcp_depth1_loss": 2,
                "mcp_depth2_loss": 3,
                "mcp_depth3_loss": 4,
                "total_loss": 5,
            }
        },
        probe_outputs=_probe_outputs_for(probe),
        selected_sample_metadata={
            "prompt": "a test prompt",
            "target_latent": {"sha256": "target-sha"},
        },
        resolved_config={
            "model_config": {"num_frame_per_block": 3},
            "m3": {"dtype": "float32"},
        },
        git_sha=TEST_GIT_SHA,
        reference_checkpoint_path=Path("ref.pt"),
        reference_checkpoint_sha256="ref-sha",
        train_seed=55,
        probe_seed=probe_seed,
        prompt_embedding={"prompt_embeds": torch.zeros((1, 1, 1))},
    )


def test_selected_sample_splits_c0_to_c4_exactly() -> None:
    state = select_m3_selected_state(_latent(15))

    assert torch.equal(state.clean_history.flatten(), torch.tensor([0.0, 1.0, 2.0]))
    assert torch.equal(state.current_target.flatten(), torch.tensor([3.0, 4.0, 5.0]))
    assert torch.equal(state.future_targets[0].flatten(), torch.tensor([6.0, 7.0, 8.0]))
    assert torch.equal(state.future_targets[1].flatten(), torch.tensor([9.0, 10.0, 11.0]))
    assert torch.equal(state.future_targets[2].flatten(), torch.tensor([12.0, 13.0, 14.0]))
    assert state.current_start_frame == 3


def test_selected_sample_rejects_too_few_frames() -> None:
    with pytest.raises(ValueError, match="at least 15"):
        select_m3_selected_state(torch.zeros((1, 14, 1, 1, 1)))


def test_selected_sample_rejects_non_chunk_aligned_frames() -> None:
    with pytest.raises(ValueError, match="chunk-aligned"):
        select_m3_selected_state(torch.zeros((1, 16, 1, 1, 1)))


def test_teacher_sample_audit_reads_manifest_payload_and_metadata(monkeypatch) -> None:
    manifest, payload = _teacher_manifest_and_payload()
    sample = _load_teacher_with_monkeypatch(monkeypatch, manifest, payload)

    assert sample.metadata["split"] == "train"
    assert sample.metadata["sample_index"] == 0
    assert sample.metadata["sample_id"] == "sample-train-0"
    assert sample.metadata["prompt"] == "a test prompt"
    assert sample.metadata["actual_prompt_sha256"] == prompt_sha256("a test prompt")
    assert sample.metadata["latent_layout"] == "[B, F, C, H, W]"
    assert sample.metadata["latent_dtype"] == "torch.bfloat16"
    assert sample.metadata["latent_frame_count"] == 15
    assert sample.metadata["chunk_aligned"] is True
    assert sample.metadata["has_minimum_15_latent_frames"] is True


def test_teacher_sample_rejects_payload_num_frames_manifest_mismatch(monkeypatch) -> None:
    manifest, payload = _teacher_manifest_and_payload()
    manifest["generation"]["num_frames"] = 18

    with pytest.raises(RuntimeError, match="num_frames differs"):
        _load_teacher_with_monkeypatch(monkeypatch, manifest, payload)


def test_teacher_sample_rejects_source_target_shape_mismatch(monkeypatch) -> None:
    manifest, payload = _teacher_manifest_and_payload()
    payload["source_noise"] = torch.zeros((1, 15, 1, 1, 2), dtype=torch.bfloat16)
    manifest["samples"][0]["source_noise"] = tensor_summary(payload["source_noise"])

    with pytest.raises(RuntimeError, match="source_noise and target_latent shapes"):
        _load_teacher_with_monkeypatch(monkeypatch, manifest, payload)


def test_teacher_sample_rejects_prompt_sha_mismatch(monkeypatch) -> None:
    manifest, payload = _teacher_manifest_and_payload()
    payload["prompt_sha256"] = "bad"
    manifest["samples"][0]["prompt_sha256"] = "bad"

    with pytest.raises(RuntimeError, match="prompt_sha256"):
        _load_teacher_with_monkeypatch(monkeypatch, manifest, payload)


def test_teacher_sample_rejects_manifest_tensor_sha_mismatch(monkeypatch) -> None:
    manifest, payload = _teacher_manifest_and_payload()
    manifest["samples"][0]["target_latent"]["sha256"] = "bad"

    with pytest.raises(RuntimeError, match="target_latent sha256"):
        _load_teacher_with_monkeypatch(monkeypatch, manifest, payload)


def test_teacher_sample_rejects_manifest_tensor_shape_mismatch(monkeypatch) -> None:
    manifest, payload = _teacher_manifest_and_payload()
    manifest["samples"][0]["source_noise"]["shape"] = [1, 99, 1, 1, 1]

    with pytest.raises(RuntimeError, match="source_noise shape"):
        _load_teacher_with_monkeypatch(monkeypatch, manifest, payload)


def test_fixed_probe_is_reproducible() -> None:
    state = _state()
    first = make_m3_probe(
        state,
        scheduler_main=_scheduler(5.0),
        scheduler_mcp=_scheduler(10.0),
        seed=123,
    )
    second = make_m3_probe(
        state,
        scheduler_main=_scheduler(5.0),
        scheduler_mcp=_scheduler(10.0),
        seed=123,
    )

    comparison = compare_serialized_probe_tensors(
        serialize_noisy_batch(first.noisy_batch),
        serialize_noisy_batch(second.noisy_batch),
    )
    assert comparison["exact"] is True
    assert all(
        entry["dtype_match"]
        for entry in comparison["tensors"].values()
    )


def test_probe_tensor_comparison_rejects_same_values_different_dtype() -> None:
    probe = make_m3_probe(
        _state(),
        scheduler_main=_scheduler(5.0),
        scheduler_mcp=_scheduler(10.0),
        seed=123,
    )
    left = serialize_noisy_batch(probe.noisy_batch)
    right = serialize_noisy_batch(probe.noisy_batch)
    right["noisy_current"] = right["noisy_current"].double()

    with pytest.raises(ValueError, match="dtype mismatch"):
        compare_serialized_probe_tensors(left, right)


def test_probe_output_comparison_rejects_same_values_different_dtype() -> None:
    probe = make_m3_probe(
        _state(),
        scheduler_main=_scheduler(5.0),
        scheduler_mcp=_scheduler(10.0),
        seed=123,
    )
    left = _probe_outputs_for(probe)
    right = _probe_outputs_for(probe)
    right["main_flow_pred"] = right["main_flow_pred"].double()

    with pytest.raises(ValueError, match="dtype mismatch"):
        compare_probe_outputs(left, right)


def test_train_rng_and_probe_rng_are_independent() -> None:
    state = _state()
    train_rng = make_cpu_generator(999)
    probe_before = make_m3_probe(
        state,
        scheduler_main=_scheduler(5.0),
        scheduler_mcp=_scheduler(10.0),
        seed=321,
    )
    _ = prepare_nf_sf_noisy_batch(
        state,
        scheduler_main=_scheduler(5.0),
        scheduler_mcp=_scheduler(10.0),
        rng=train_rng,
    )
    probe_after = make_m3_probe(
        state,
        scheduler_main=_scheduler(5.0),
        scheduler_mcp=_scheduler(10.0),
        seed=321,
    )

    assert compare_serialized_probe_tensors(
        serialize_noisy_batch(probe_before.noisy_batch),
        serialize_noisy_batch(probe_after.noisy_batch),
    )["exact"]


def test_checkpoint_contains_required_fields_and_round_trips() -> None:
    state = _state()
    probe = make_m3_probe(
        state,
        scheduler_main=_scheduler(5.0),
        scheduler_mcp=_scheduler(10.0),
        seed=44,
    )
    model = nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(
        [{"name": "linear", "params": list(model.parameters()), "lr": 1.0e-4}],
        weight_decay=0.01,
    )
    train_rng = make_cpu_generator(55)
    probe_outputs = _probe_outputs_for(probe)
    payload = make_m3_checkpoint_payload(
        generator=model,
        optimizer=optimizer,
        global_step=10,
        train_rng=train_rng,
        probe=probe,
        probe_summary={"probe_losses": {"main_loss": 1, "mcp_depth1_loss": 2, "mcp_depth2_loss": 3, "mcp_depth3_loss": 4, "total_loss": 5}},
        probe_outputs=probe_outputs,
        selected_sample_metadata={"sample_index": 0},
        resolved_config={"m3": {"dtype": "float32"}},
        git_sha=TEST_GIT_SHA,
        reference_checkpoint_path=Path("ref.pt"),
        reference_checkpoint_sha256="sha",
        train_seed=55,
        probe_seed=44,
        prompt_embedding={"prompt_embeds": torch.zeros((1, 1, 1))},
    )
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    buffer.seek(0)
    loaded = torch.load(buffer, map_location="cpu", weights_only=False)
    m3.validate_m3_checkpoint_payload(loaded)

    assert loaded["format"] == M3_CHECKPOINT_FORMAT
    assert loaded["global_step"] == 10
    assert "generator" in loaded
    assert "optimizer" in loaded
    assert "train_rng_state" in loaded
    assert "probe_rng_state" in loaded
    assert "probe_tensors" in loaded
    assert "probe_outputs" in loaded
    assert "probe_prompt_embedding" in loaded
    assert compare_serialized_probe_tensors(
        payload["probe_tensors"],
        loaded["probe_tensors"],
    )["exact"]
    assert compare_probe_outputs(
        payload["probe_outputs"],
        loaded["probe_outputs"],
    )["max_abs_diff"] == 0.0


def test_checkpoint_requires_probe_outputs_and_prompt_embedding() -> None:
    payload = {
        "format": M3_CHECKPOINT_FORMAT,
        "generator": {},
        "optimizer": {},
        "global_step": 0,
        "train_rng_state": torch.zeros(1),
        "probe_rng_state": torch.zeros(1),
        "probe_tensors": {},
        "probe_summary": {},
        "selected_sample_metadata": {},
        "resolved_config": {},
        "git_sha": TEST_GIT_SHA,
        "reference_checkpoint": {},
        "train_seed": 1,
        "probe_seed": 2,
        "optimizer_group_lrs": [],
    }

    with pytest.raises(KeyError, match="probe_outputs"):
        m3.validate_m3_checkpoint_payload(payload)

    payload["probe_outputs"] = {}
    payload["probe_prompt_embedding"] = None
    with pytest.raises(RuntimeError, match="prompt embedding"):
        m3.validate_m3_checkpoint_payload(payload)


def test_initial_final_checkpoint_pair_accepts_matching_pair() -> None:
    report = validate_m3_checkpoint_pair(
        initial_payload=_checkpoint_for_pair(global_step=0),
        final_payload=_checkpoint_for_pair(global_step=3),
        current_model_config={"num_frame_per_block": 3},
        current_git_sha=TEST_GIT_SHA,
    )

    assert report["status"] == "PASS"
    assert report["initial_global_step"] == 0
    assert report["final_global_step"] == 3


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda payload: payload.update({"global_step": 0}), "greater than 0"),
        (
            lambda payload: payload["selected_sample_metadata"].update(
                {"prompt": "different"}
            ),
            "prompt differs",
        ),
        (
            lambda payload: payload["selected_sample_metadata"]["target_latent"].update(
                {"sha256": "different"}
            ),
            "target_latent SHA",
        ),
        (
            lambda payload: payload.update({"probe_seed": 99}),
            "probe seed",
        ),
        (
            lambda payload: payload["reference_checkpoint"].update(
                {"sha256": "different"}
            ),
            "reference checkpoint SHA",
        ),
        (
            lambda payload: payload["resolved_config"].update(
                {"model_config": {"num_frame_per_block": 99}}
            ),
            "current eval config",
        ),
        (
            lambda payload: payload.update({"git_sha": "b" * 40}),
            "git_sha differs",
        ),
    ],
)
def test_initial_final_checkpoint_pair_rejects_mismatch(mutation, message) -> None:
    initial = _checkpoint_for_pair(global_step=0)
    final = _checkpoint_for_pair(global_step=3)
    mutation(final)

    with pytest.raises(RuntimeError, match=message):
        validate_m3_checkpoint_pair(
            initial_payload=initial,
            final_payload=final,
            current_model_config={"num_frame_per_block": 3},
        )


def test_checkpoint_git_sha_rejects_current_head_mismatch() -> None:
    payload = _checkpoint_for_pair(global_step=3)

    with pytest.raises(RuntimeError, match="current HEAD"):
        validate_m3_checkpoint_git_sha(
            payload,
            current_git_sha="b" * 40,
        )


def _optimizer_audit_for_m3_groups() -> dict:
    return {
        "param_audit": [
            {
                "name": group,
                "parameter_names": [
                    f"{group}.weight",
                    f"{group}.bias",
                ],
            }
            for group in m3.M3_PARAMETER_GROUP_NAMES
        ]
    }


def _state_dict_for_parameter_audit(*, changed_groups: set[str]) -> tuple[dict, dict]:
    initial = {}
    final = {}
    for group in m3.M3_PARAMETER_GROUP_NAMES:
        for suffix in ("weight", "bias"):
            name = f"{group}.{suffix}"
            initial[name] = torch.zeros((2,), dtype=torch.float32)
            final[name] = torch.zeros((2,), dtype=torch.float32)
        if group in changed_groups:
            final[f"{group}.weight"] = torch.ones((2,), dtype=torch.float32)
    return initial, final


def test_parameter_change_audit_reports_all_six_groups_changed() -> None:
    initial, final = _state_dict_for_parameter_audit(
        changed_groups=set(m3.M3_PARAMETER_GROUP_NAMES),
    )
    report = audit_parameter_changes(
        initial_state_dict=initial,
        final_state_dict=final,
        optimizer_audit=_optimizer_audit_for_m3_groups(),
    )

    assert report["status"] == "PASS"
    assert report["all_groups_parameter_changed"] is True
    assert set(report["groups"]) == set(m3.M3_PARAMETER_GROUP_NAMES)
    for group in report["groups"].values():
        assert group["tensor_count"] == 2
        assert group["changed_tensor_count"] == 1
        assert group["unchanged_tensor_count"] == 1
        assert group["max_abs_parameter_diff"] == 1.0
        assert group["parameter_changed"] is True


def test_parameter_change_audit_flags_unchanged_group() -> None:
    changed = set(m3.M3_PARAMETER_GROUP_NAMES) - {"mcp_fusion"}
    initial, final = _state_dict_for_parameter_audit(changed_groups=changed)
    report = audit_parameter_changes(
        initial_state_dict=initial,
        final_state_dict=final,
        optimizer_audit=_optimizer_audit_for_m3_groups(),
    )

    assert report["status"] == "FAIL"
    assert report["all_groups_parameter_changed"] is False
    assert report["groups"]["mcp_fusion"]["parameter_changed"] is False


def test_gradient_group_audit_reports_required_format() -> None:
    params = {}
    groups = []
    for group in m3.M3_PARAMETER_GROUP_NAMES:
        with_grad = nn.Parameter(torch.ones(()))
        without_grad = nn.Parameter(torch.ones(()))
        params[f"{group}.with_grad"] = with_grad
        params[f"{group}.without_grad"] = without_grad
        groups.append(
            {
                "name": group,
                "params": [with_grad, without_grad],
                "lr": 1.0e-3,
            }
        )
    loss = sum(
        parameter
        for name, parameter in params.items()
        if name.endswith(".with_grad")
    )
    loss.backward()
    optimizer = torch.optim.AdamW(groups)
    report = gradient_group_audit(optimizer)

    assert set(report) == set(m3.M3_PARAMETER_GROUP_NAMES)
    for group in report.values():
        assert group["tensor_count_with_grad"] == 1
        assert group["tensor_count_without_grad"] == 1
        assert group["grad_norm"] == 1.0
        assert group["finite"] is True


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


def test_optimizer_group_lr_values_are_separate() -> None:
    plan = configure_nf_sf_optimizer_plan(
        FakeWanWrapper(),
        mode="joint",
        group_lrs={
            "backbone": 1.0e-5,
            "patch_embedding": 2.0e-5,
            "mcp": 3.0e-5,
        },
    )
    by_name = {group["name"]: group["lr"] for group in plan.optimizer_param_groups}

    assert by_name["backbone"] == 1.0e-5
    assert by_name["patch_embedding"] == 2.0e-5
    assert by_name["mcp_fusion"] == 3.0e-5
    assert by_name["mcp_depth1"] == 3.0e-5
    assert by_name["mcp_depth2"] == 3.0e-5
    assert by_name["mcp_depth3"] == 3.0e-5


def test_m3_rejects_frozen_mode() -> None:
    with pytest.raises(ValueError, match="joint"):
        validate_m3_mode("frozen")


class ZeroMainFlowGenerator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scheduler = FlowMatchScheduler(shift=5.0, sigma_min=0.0, extra_one_step=True)
        self.calls = []

    def get_scheduler(self):
        return self.scheduler

    def forward(self, **kwargs):
        timestep = kwargs["timestep"]
        self.calls.append(float(timestep[0, 0].item()))
        noisy = kwargs["noisy_image_or_video"]
        return torch.zeros_like(noisy), torch.zeros_like(noisy), []


class ZeroMCPFlowGenerator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scheduler = FlowMatchScheduler(shift=5.0, sigma_min=0.0, extra_one_step=True)
        self.calls = []

    def get_scheduler(self):
        return self.scheduler

    def forward(self, **kwargs):
        timestep = kwargs["timestep"]
        future = kwargs["mcp_future_noises"][0]
        self.calls.append(
            {
                "main_timestep": float(timestep[0, 0].item()),
                "mcp_timestep": float(kwargs["mcp_timesteps"][0][0, 0].item()),
                "future_start_frame": int(kwargs["mcp_future_start_frames"][0]),
                "noisy_current": kwargs["noisy_image_or_video"].detach().clone(),
            }
        )
        noisy = kwargs["noisy_image_or_video"]
        return torch.zeros_like(noisy), torch.zeros_like(noisy), [torch.zeros_like(future)]


def test_main_reconstruction_uses_payload_solver_timestep_order() -> None:
    state = _state()
    generator = ZeroMainFlowGenerator()
    payload = _solver_payload(num_steps=4)
    generated = list(payload["warped_denoising_steps"])
    payload["warped_denoising_steps"] = [
        value + 5.0e-5
        for value in generated
    ]
    result = reconstruct_main_current(
        generator,
        conditional_dict={"prompt_embeds": torch.zeros((1, 1, 1))},
        state=state,
        initial_noise=torch.ones_like(state.current_target),
        teacher_payload=payload,
    )

    assert generator.calls == pytest.approx(generated)
    assert abs(generator.calls[0] - payload["warped_denoising_steps"][0]) > 1.0e-6
    assert result.solver_schedule.source == "teacher_payload"
    assert result.solver_schedule.generated_timesteps == pytest.approx(generated)
    assert result.solver_schedule.warped_denoising_steps == pytest.approx(
        payload["warped_denoising_steps"],
    )


def test_solver_schedule_mismatch_rejects() -> None:
    scheduler = FlowMatchScheduler(shift=5.0, sigma_min=0.0, extra_one_step=True)
    payload = _solver_payload(num_steps=4)
    payload["warped_denoising_steps"][0] += 1.0

    with pytest.raises(RuntimeError, match="timesteps differ"):
        resolve_m3_solver_schedule(
            scheduler,
            teacher_payload=payload,
            device="cpu",
        )


def test_solver_override_requires_explicit_flag() -> None:
    scheduler = FlowMatchScheduler(shift=5.0, sigma_min=0.0, extra_one_step=True)

    with pytest.raises(ValueError, match="allow_solver_override"):
        resolve_m3_solver_schedule(
            scheduler,
            teacher_payload=_solver_payload(num_steps=4),
            device="cpu",
            solver_steps_override=2,
        )


def test_standalone_mcp1_reconstruction_uses_payload_schedule_and_teacher_forcing() -> None:
    state = _state()
    generator = ZeroMCPFlowGenerator()
    payload = _solver_payload(num_steps=4)
    generated = list(payload["warped_denoising_steps"])
    payload["warped_denoising_steps"] = [
        value + 5.0e-5
        for value in generated
    ]
    condition_noise = torch.full_like(state.current_target, 2.0)
    _ = reconstruct_mcp1_next(
        generator,
        conditional_dict={"prompt_embeds": torch.zeros((1, 1, 1))},
        state=state,
        next_initial_noise=torch.ones_like(state.future_targets[0]),
        current_condition_noise=condition_noise,
        teacher_payload=payload,
    )

    expected_scheduler = FlowMatchScheduler(shift=5.0, sigma_min=0.0, extra_one_step=True)
    expected = generated
    wrong_scheduler = FlowMatchScheduler(shift=10.0, sigma_min=0.0, extra_one_step=True)
    wrong = [
        float(value.item())
        for value in solver_timesteps_from_scheduler(
            wrong_scheduler,
            num_inference_steps=4,
            device="cpu",
        )
    ]

    assert [call["main_timestep"] for call in generator.calls] == pytest.approx(expected)
    assert [call["mcp_timestep"] for call in generator.calls] == pytest.approx(expected)
    assert abs(generator.calls[0]["main_timestep"] - payload["warped_denoising_steps"][0]) > 1.0e-6
    assert [call["future_start_frame"] for call in generator.calls] == [6, 6, 6, 6]
    assert expected != wrong

    expected_scheduler.set_timesteps(4, denoising_strength=1.0)
    for call, timestep_value in zip(generator.calls, expected):
        timestep = torch.full(
            state.current_target.shape[:2],
            float(timestep_value),
            dtype=torch.float32,
        )
        expected_noisy = expected_scheduler.add_noise(
            state.current_target.flatten(0, 1),
            condition_noise.flatten(0, 1),
            timestep.flatten(0, 1),
        ).unflatten(0, state.current_target.shape[:2])
        assert torch.allclose(call["noisy_current"], expected_noisy)
