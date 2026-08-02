import hashlib
import io
import json
from pathlib import Path

import pytest
import torch
from torch import nn

import scripts.train_nf_sf_m3_overfit as train_m3
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
    NFSFSelectedState,
    configure_nf_sf_optimizer_plan,
    run_nf_sf_forward_loss,
    prepare_nf_sf_noisy_batch,
)
from utils.scheduler import FlowMatchScheduler


TEST_GIT_SHA = "a" * 40
NORMALIZED_PROMPT_SHA256 = "b" * 64


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
        "prompt_sha256": NORMALIZED_PROMPT_SHA256,
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
        "writer_git_head": TEST_GIT_SHA,
    }
    manifest = {
        "status": "PASS",
        "experiment": "E0208C_teacher_rollout_formal",
        "format": "self_forcing_teacher_manifest_v2",
        "writer_format": "e0208_teacher_writer_v1",
        "writer_git_head": TEST_GIT_SHA,
        "checkpoint": {
            "path": "checkpoints/self_forcing_dmd.pt",
            "sha256": M3_REFERENCE_CHECKPOINT_SHA256,
        },
        "generation": {
            "num_samples": 1,
            "num_completed": 1,
            "num_train": 1,
            "num_validation": 0,
            "num_reserve": 0,
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
                "status": "GENERATED",
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
                "prompt_sha256": NORMALIZED_PROMPT_SHA256,
                "target_latent": tensor_summary(target),
                "source_noise": tensor_summary(source),
            }
        ],
    }
    return manifest, payload


def _merged_manifest_from(manifest: dict) -> dict:
    merged = json.loads(json.dumps(manifest))
    merged["format"] = "self_forcing_teacher_manifest_v2_merged"
    merged.pop("writer_format", None)
    merged.pop("writer_git_head", None)
    merged["shards"] = [
        {
            "shard_id": 0,
            "split": "train",
            "count": len(merged["samples"]),
            "path": "/dataset/shard_000_manifest.json",
            "sha256": "b" * 64,
        }
    ]
    return merged


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
    assert sample.metadata["normalized_prompt_sha256"] == NORMALIZED_PROMPT_SHA256
    assert hashlib.sha256(b"a test prompt").hexdigest() != NORMALIZED_PROMPT_SHA256
    assert sample.metadata["latent_layout"] == "[B, F, C, H, W]"
    assert sample.metadata["latent_dtype"] == "torch.bfloat16"
    assert sample.metadata["latent_frame_count"] == 15
    assert sample.metadata["chunk_aligned"] is True
    assert sample.metadata["has_minimum_15_latent_frames"] is True


def test_teacher_sample_accepts_shard_v2_manifest(monkeypatch) -> None:
    manifest, payload = _teacher_manifest_and_payload()
    sample = _load_teacher_with_monkeypatch(monkeypatch, manifest, payload)

    assert sample.metadata["manifest_format"] == "self_forcing_teacher_manifest_v2"


def test_teacher_sample_accepts_merged_v2_manifest(monkeypatch) -> None:
    manifest, payload = _teacher_manifest_and_payload()
    merged = _merged_manifest_from(manifest)
    sample = _load_teacher_with_monkeypatch(monkeypatch, merged, payload)

    assert sample.metadata["manifest_format"] == "self_forcing_teacher_manifest_v2_merged"


def test_teacher_sample_rejects_unknown_manifest_format(monkeypatch) -> None:
    manifest, payload = _teacher_manifest_and_payload()
    manifest["format"] = "self_forcing_teacher_manifest_v2_future"

    with pytest.raises(RuntimeError, match="self_forcing_teacher_manifest_v2_future"):
        _load_teacher_with_monkeypatch(monkeypatch, manifest, payload)


def test_teacher_sample_rejects_merged_split_count_mismatch(monkeypatch) -> None:
    manifest, payload = _teacher_manifest_and_payload()
    merged = _merged_manifest_from(manifest)
    merged["generation"]["num_train"] = 0

    with pytest.raises(RuntimeError, match="num_train"):
        _load_teacher_with_monkeypatch(monkeypatch, merged, payload)


def test_teacher_sample_rejects_merged_sample_count_mismatch(monkeypatch) -> None:
    manifest, payload = _teacher_manifest_and_payload()
    merged = _merged_manifest_from(manifest)
    merged["generation"]["num_samples"] = 2

    with pytest.raises(RuntimeError, match="num_samples"):
        _load_teacher_with_monkeypatch(monkeypatch, merged, payload)


def test_teacher_sample_rejects_duplicate_sample_index(monkeypatch) -> None:
    manifest, payload = _teacher_manifest_and_payload()
    duplicate = dict(manifest["samples"][0])
    duplicate["split_index"] = 1
    manifest["samples"].append(duplicate)
    manifest["generation"]["num_samples"] = 2
    manifest["generation"]["num_completed"] = 2
    manifest["generation"]["num_train"] = 2

    with pytest.raises(RuntimeError, match="duplicate sample_index"):
        _load_teacher_with_monkeypatch(monkeypatch, manifest, payload)


def test_teacher_sample_rejects_duplicate_split_index(monkeypatch) -> None:
    manifest, payload = _teacher_manifest_and_payload()
    duplicate = dict(manifest["samples"][0])
    duplicate["sample_index"] = 1
    manifest["samples"].append(duplicate)
    manifest["generation"]["num_samples"] = 2
    manifest["generation"]["num_completed"] = 2
    manifest["generation"]["num_train"] = 2

    with pytest.raises(RuntimeError, match="duplicate split/split_index"):
        _load_teacher_with_monkeypatch(monkeypatch, manifest, payload)


def test_teacher_sample_rejects_shard_writer_format_mismatch(monkeypatch) -> None:
    manifest, payload = _teacher_manifest_and_payload()
    manifest["writer_format"] = "wrong_writer"

    with pytest.raises(RuntimeError, match="writer_format"):
        _load_teacher_with_monkeypatch(monkeypatch, manifest, payload)


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


def test_teacher_sample_rejects_payload_manifest_prompt_mismatch(monkeypatch) -> None:
    manifest, payload = _teacher_manifest_and_payload()
    payload["prompt"] = "a different prompt"

    with pytest.raises(RuntimeError, match="payload prompt differs from manifest prompt"):
        _load_teacher_with_monkeypatch(monkeypatch, manifest, payload)


def test_teacher_sample_rejects_payload_manifest_prompt_sha_mismatch(monkeypatch) -> None:
    manifest, payload = _teacher_manifest_and_payload()
    payload["prompt_sha256"] = "c" * 64

    with pytest.raises(
        RuntimeError,
        match="prompt_sha256 differs from manifest prompt_sha256",
    ):
        _load_teacher_with_monkeypatch(monkeypatch, manifest, payload)


def test_teacher_sample_rejects_invalid_prompt_sha(monkeypatch) -> None:
    manifest, payload = _teacher_manifest_and_payload()
    payload["prompt_sha256"] = "not-a-sha256"
    manifest["samples"][0]["prompt_sha256"] = "not-a-sha256"

    with pytest.raises(RuntimeError, match="prompt_sha256 is not a valid SHA256"):
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


class BF16MainFlowGenerator(ZeroMainFlowGenerator):
    def forward(self, **kwargs):
        noisy = kwargs["noisy_image_or_video"]
        assert noisy.dtype == torch.bfloat16
        self.calls.append(float(kwargs["timestep"][0, 0].item()))
        return torch.zeros_like(noisy), torch.zeros_like(noisy), []


class BF16MCPFlowGenerator(ZeroMCPFlowGenerator):
    def forward(self, **kwargs):
        noisy = kwargs["noisy_image_or_video"]
        future = kwargs["mcp_future_noises"][0]
        assert noisy.dtype == torch.bfloat16
        assert future.dtype == torch.bfloat16
        self.calls.append(
            {
                "main_timestep": float(kwargs["timestep"][0, 0].item()),
                "mcp_timestep": float(kwargs["mcp_timesteps"][0][0, 0].item()),
                "future_start_frame": int(kwargs["mcp_future_start_frames"][0]),
            }
        )
        return torch.zeros_like(noisy), torch.zeros_like(noisy), [torch.zeros_like(future)]


class BadShapeScheduler(FlowMatchScheduler):
    def step(self, model_output, timestep, sample, to_final=False):
        _ = super().step(model_output, timestep, sample, to_final=to_final)
        return sample[:1]


class BadShapeMainFlowGenerator(ZeroMainFlowGenerator):
    def __init__(self) -> None:
        super().__init__()
        self.scheduler = BadShapeScheduler(shift=5.0, sigma_min=0.0, extra_one_step=True)


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


def test_main_reconstruction_preserves_bf16_state_across_solver_steps() -> None:
    state = select_m3_selected_state(_latent(15, dtype=torch.bfloat16))
    generator = BF16MainFlowGenerator()
    result = reconstruct_main_current(
        generator,
        conditional_dict={"prompt_embeds": torch.zeros((1, 1, 1))},
        state=state,
        initial_noise=torch.ones_like(state.current_target),
        teacher_payload=_solver_payload(num_steps=4),
    )

    assert len(generator.calls) == 4
    assert result.latent.dtype == torch.bfloat16


def test_scheduler_step_shape_mismatch_is_rejected() -> None:
    state = _state()

    with pytest.raises(RuntimeError, match="shape mismatch"):
        reconstruct_main_current(
            BadShapeMainFlowGenerator(),
            conditional_dict={"prompt_embeds": torch.zeros((1, 1, 1))},
            state=state,
            initial_noise=torch.ones_like(state.current_target),
            teacher_payload=_solver_payload(num_steps=4),
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


def test_standalone_mcp1_reconstruction_preserves_bf16_state_across_solver_steps() -> None:
    state = select_m3_selected_state(_latent(15, dtype=torch.bfloat16))
    generator = BF16MCPFlowGenerator()
    result = reconstruct_mcp1_next(
        generator,
        conditional_dict={"prompt_embeds": torch.zeros((1, 1, 1))},
        state=state,
        next_initial_noise=torch.ones_like(state.future_targets[0]),
        current_condition_noise=torch.full_like(state.current_target, 2.0),
        teacher_payload=_solver_payload(num_steps=4),
    )

    assert len(generator.calls) == 4
    assert result.latent.dtype == torch.bfloat16


class AuxScalar(nn.Module):
    def __init__(self, value: float = 0.25) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(float(value)))


class AuxFakeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.patch_embedding = AuxScalar(0.1)
        self.backbone = AuxScalar(0.2)


class AuxFakeMCP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fusion = AuxScalar(0.3)
        self.mcp_modules = nn.ModuleList(
            [AuxScalar(0.4), AuxScalar(0.5), AuxScalar(0.6)]
        )


class SingleLiveGraph(torch.autograd.Function):
    active = 0
    backward_count = 0

    @classmethod
    def reset(cls) -> None:
        cls.active = 0
        cls.backward_count = 0

    @staticmethod
    def forward(ctx, value):
        if SingleLiveGraph.active != 0:
            raise RuntimeError("previous graph is still live")
        SingleLiveGraph.active += 1
        return value.clone()

    @staticmethod
    def backward(ctx, grad_output):
        SingleLiveGraph.active -= 1
        SingleLiveGraph.backward_count += 1
        return grad_output


class AuxFakeWan(nn.Module):
    def __init__(
        self,
        *,
        no_fusion_aux: bool = False,
        no_mcp1_aux: bool = False,
        leak_mcp2_aux: bool = False,
        single_live_aux: bool = False,
    ) -> None:
        super().__init__()
        self.model = AuxFakeModel()
        self.mcp = AuxFakeMCP()
        self.no_fusion_aux = no_fusion_aux
        self.no_mcp1_aux = no_mcp1_aux
        self.leak_mcp2_aux = leak_mcp2_aux
        self.single_live_aux = single_live_aux
        self.calls = []

    def forward(self, **kwargs):
        noisy = kwargs["noisy_image_or_video"]
        futures = kwargs.get("mcp_future_noises", [])
        self.calls.append(
            {
                "future_count": len(futures),
                "grad_enabled": torch.is_grad_enabled(),
                "future_start": list(kwargs.get("mcp_future_start_frames", [])),
                "timestep": kwargs["timestep"].detach().clone(),
                "mcp_timestep": None
                if not kwargs.get("mcp_timesteps")
                else kwargs["mcp_timesteps"][0].detach().clone(),
                "noisy_current": noisy.detach().clone(),
                "future": None if not futures else futures[0].detach().clone(),
            }
        )
        main_scale = self.model.backbone.weight + self.model.patch_embedding.weight
        main_flow = noisy * main_scale.to(device=noisy.device, dtype=noisy.dtype)
        mcp_flows = []
        for index, future in enumerate(futures):
            scale = self.mcp.fusion.weight + self.mcp.mcp_modules[index].weight
            if len(futures) == 1:
                terms = []
                if not self.no_fusion_aux:
                    terms.append(self.mcp.fusion.weight)
                if not self.no_mcp1_aux:
                    terms.append(self.mcp.mcp_modules[0].weight)
                if self.leak_mcp2_aux:
                    terms.append(self.mcp.mcp_modules[1].weight)
                scale = sum(terms, torch.zeros_like(self.mcp.fusion.weight))
            flow = future * scale.to(device=future.device, dtype=future.dtype)
            if len(futures) == 1 and self.single_live_aux:
                flow = SingleLiveGraph.apply(flow)
            mcp_flows.append(flow)
        return main_flow, torch.zeros_like(noisy), mcp_flows


class CountingSGD(torch.optim.SGD):
    def __init__(self, params, **kwargs) -> None:
        super().__init__(params, **kwargs)
        self.zero_grad_calls = 0
        self.step_calls = 0

    def zero_grad(self, set_to_none: bool = True):
        self.zero_grad_calls += 1
        return super().zero_grad(set_to_none=set_to_none)

    def step(self, closure=None):
        self.step_calls += 1
        return super().step(closure=closure)


def _aux_state() -> NFSFSelectedState:
    current = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float32).reshape(1, 3, 1, 1, 1)
    return NFSFSelectedState(
        clean_history=torch.full_like(current, -1.0),
        current_target=current,
        future_targets=(current + 1.0, current + 2.0, current + 3.0),
        current_start_frame=3,
    )


def _aux_optimizer(generator: AuxFakeWan) -> CountingSGD:
    plan = configure_nf_sf_optimizer_plan(
        generator,
        mode="joint",
        group_lrs={"backbone": 0.1, "patch_embedding": 0.1, "mcp": 0.1},
    )
    return CountingSGD(plan.optimizer_param_groups, lr=0.1)


def _run_random_backward(generator, optimizer, state, train_rng):
    scheduler_main = _scheduler(5.0)
    scheduler_mcp = _scheduler(10.0)
    batch = prepare_nf_sf_noisy_batch(
        state,
        scheduler_main=scheduler_main,
        scheduler_mcp=scheduler_mcp,
        rng=train_rng,
    )
    result = run_nf_sf_forward_loss(
        generator,
        conditional_dict={"prompt_embeds": torch.zeros((1, 1, 1))},
        noisy_batch=batch,
    )
    result.losses.total_loss.backward()
    return result, batch


def test_mcp1_grid_aux_schedule_is_four_point_main_shift_and_independent() -> None:
    random_scheduler = _scheduler(5.0)
    aux_scheduler, timesteps, report = train_m3.resolve_mcp1_grid_aux_schedule(
        teacher_payload=_solver_payload(num_steps=4, shift=5.0),
        device=torch.device("cpu"),
    )

    assert aux_scheduler is not random_scheduler
    assert timesteps.tolist() == pytest.approx(list(train_m3.MCP1_GRID_EXPECTED_TIMESTEPS))
    assert report["generated_timesteps"] == pytest.approx(
        list(train_m3.MCP1_GRID_EXPECTED_TIMESTEPS)
    )
    with pytest.raises(RuntimeError, match="exactly four"):
        train_m3.resolve_mcp1_grid_aux_schedule(
            teacher_payload=_solver_payload(num_steps=3, shift=5.0),
            device=torch.device("cpu"),
        )
    with pytest.raises(RuntimeError, match="timesteps differ|inference grid"):
        train_m3.resolve_mcp1_grid_aux_schedule(
            teacher_payload=_solver_payload(num_steps=4, shift=10.0),
            device=torch.device("cpu"),
        )


def test_mcp1_grid_aux_weight_zero_does_not_run_grid_forward() -> None:
    generator = AuxFakeWan()
    report = train_m3.accumulate_mcp1_grid_aux_gradients(
        generator,
        conditional_dict={},
        state=_aux_state(),
        scheduler=None,
        timesteps=None,
        epsilon_main=torch.zeros((1, 3, 1, 1, 1)),
        epsilon_future=torch.zeros((1, 3, 1, 1, 1)),
        weight=0.0,
    )

    assert report["enabled"] is False
    assert generator.calls == []


def test_mcp1_grid_aux_accumulates_four_autograd_grads_and_one_optimizer_step(monkeypatch) -> None:
    SingleLiveGraph.reset()
    generator = AuxFakeWan(single_live_aux=True)
    state = _aux_state()
    optimizer = _aux_optimizer(generator)
    train_rng = make_cpu_generator(101)
    optimizer.zero_grad(set_to_none=True)
    _, batch = _run_random_backward(generator, optimizer, state, train_rng)
    aux_scheduler, timesteps, _ = train_m3.resolve_mcp1_grid_aux_schedule(
        teacher_payload=_solver_payload(num_steps=4, shift=5.0),
        device=torch.device("cpu"),
    )
    grad_calls = []
    real_grad = torch.autograd.grad

    def counting_grad(*args, **kwargs):
        grad_calls.append(
            {
                "retain_graph": kwargs.get("retain_graph", None),
                "allow_unused": kwargs.get("allow_unused", None),
            }
        )
        return real_grad(*args, **kwargs)

    monkeypatch.setattr(train_m3.torch.autograd, "grad", counting_grad)
    report = train_m3.accumulate_mcp1_grid_aux_gradients(
        generator,
        conditional_dict={"prompt_embeds": torch.zeros((1, 1, 1))},
        state=state,
        scheduler=aux_scheduler,
        timesteps=timesteps,
        epsilon_main=batch.epsilon_main,
        epsilon_future=batch.epsilon_depths[0],
        weight=1.0,
    )
    optimizer.step()

    assert len(grad_calls) == 4
    assert all(call["retain_graph"] is False for call in grad_calls)
    assert all(call["allow_unused"] is False for call in grad_calls)
    assert SingleLiveGraph.backward_count == 4
    assert SingleLiveGraph.active == 0
    assert optimizer.zero_grad_calls == 1
    assert optimizer.step_calls == 1
    assert len([call for call in generator.calls if call["future_count"] == 1]) == 4
    assert report["mcp1_grid_aux_mean_loss"] == pytest.approx(
        sum(report["point_losses"]) / 4.0
    )
    assert report["mcp1_grid_aux_weighted_loss"] == pytest.approx(
        report["mcp1_grid_aux_mean_loss"]
    )


def test_mcp1_grid_aux_gradient_isolation_and_missing_target_grad_failures(monkeypatch) -> None:
    state = _aux_state()
    aux_scheduler, timesteps, _ = train_m3.resolve_mcp1_grid_aux_schedule(
        teacher_payload=_solver_payload(num_steps=4, shift=5.0),
        device=torch.device("cpu"),
    )
    generator = AuxFakeWan()
    optimizer = _aux_optimizer(generator)
    optimizer.zero_grad(set_to_none=True)
    _, batch = _run_random_backward(generator, optimizer, state, make_cpu_generator(102))
    before = train_m3.clone_gradients(train_m3.named_parameter_groups(generator))
    report = train_m3.accumulate_mcp1_grid_aux_gradients(
        generator,
        conditional_dict={},
        state=state,
        scheduler=aux_scheduler,
        timesteps=timesteps,
        epsilon_main=batch.epsilon_main,
        epsilon_future=batch.epsilon_depths[0],
        weight=0.7,
    )
    after = train_m3.clone_gradients(train_m3.named_parameter_groups(generator))
    for group_name in ("backbone", "patch_embedding", "mcp_depth2", "mcp_depth3"):
        for name in before[group_name]:
            assert torch.equal(before[group_name][name], after[group_name][name])
    assert report["gradient_isolation"]["mcp_fusion"]["aux_grad_changed"] is True
    assert report["gradient_isolation"]["mcp_depth1"]["aux_grad_changed"] is True

    for kwargs in [
        {"no_fusion_aux": True},
        {"no_mcp1_aux": True},
    ]:
        bad = AuxFakeWan(**kwargs)
        optimizer = _aux_optimizer(bad)
        optimizer.zero_grad(set_to_none=True)
        _, batch = _run_random_backward(bad, optimizer, state, make_cpu_generator(103))
        with pytest.raises(RuntimeError, match="mcp_fusion/mcp_depth1"):
            train_m3.accumulate_mcp1_grid_aux_gradients(
                bad,
                conditional_dict={},
                state=state,
                scheduler=aux_scheduler,
                timesteps=timesteps,
                epsilon_main=batch.epsilon_main,
                epsilon_future=batch.epsilon_depths[0],
                weight=1.0,
            )

    leaking = AuxFakeWan()
    optimizer = _aux_optimizer(leaking)
    optimizer.zero_grad(set_to_none=True)
    _, batch = _run_random_backward(leaking, optimizer, state, make_cpu_generator(104))
    real_grad = train_m3.torch.autograd.grad

    def leaking_grad(*args, **kwargs):
        grads = real_grad(*args, **kwargs)
        leaking.mcp.mcp_modules[1].weight.grad = (
            leaking.mcp.mcp_modules[1].weight.grad + torch.ones_like(leaking.mcp.mcp_modules[1].weight)
        )
        return grads

    monkeypatch.setattr(train_m3.torch.autograd, "grad", leaking_grad)
    with pytest.raises(RuntimeError, match="mcp_depth2"):
        train_m3.accumulate_mcp1_grid_aux_gradients(
            leaking,
            conditional_dict={},
            state=state,
            scheduler=aux_scheduler,
            timesteps=timesteps,
            epsilon_main=batch.epsilon_main,
            epsilon_future=batch.epsilon_depths[0],
            weight=1.0,
        )


def test_mcp1_grid_aux_combined_objective_math() -> None:
    random_total = 2.5
    aux = {
        "mcp1_grid_aux_mean_loss": 3.0,
        "mcp1_grid_aux_weighted_loss": 1.5,
    }
    combined = random_total + aux["mcp1_grid_aux_weighted_loss"]

    assert aux["mcp1_grid_aux_weighted_loss"] == pytest.approx(0.5 * aux["mcp1_grid_aux_mean_loss"])
    assert combined == pytest.approx(4.0)


def test_mcp1_grid_stable_probe_uses_fixed_noise_no_grad_and_preserves_rng() -> None:
    generator = AuxFakeWan()
    state = _aux_state()
    aux_scheduler, timesteps, _ = train_m3.resolve_mcp1_grid_aux_schedule(
        teacher_payload=_solver_payload(num_steps=4, shift=5.0),
        device=torch.device("cpu"),
    )
    train_rng = make_cpu_generator(999)
    before_rng = train_rng.get_state().clone()
    epsilon_main = torch.full_like(state.current_target, 5.0)
    epsilon_future = torch.full_like(state.future_targets[0], 6.0)
    probe = train_m3.run_mcp1_grid_stable_probe(
        generator,
        conditional_dict={},
        state=state,
        scheduler=aux_scheduler,
        timesteps=timesteps,
        epsilon_main=epsilon_main,
        epsilon_future=epsilon_future,
    )
    after_rng = train_rng.get_state()

    assert torch.equal(before_rng, after_rng)
    assert probe["mcp1_grid_probe_mean_loss"] == pytest.approx(
        sum(probe["point_losses"]) / 4.0
    )
    assert probe["all_finite"] is True
    assert len(probe["records"]) == 4
    assert all(call["grad_enabled"] is False for call in generator.calls)
    for call in generator.calls:
        timestep = call["timestep"]
        expected_current = aux_scheduler.add_noise(
            state.current_target.flatten(0, 1),
            epsilon_main.flatten(0, 1),
            timestep.flatten(0, 1),
        ).unflatten(0, state.current_target.shape[:2])
        assert torch.allclose(call["noisy_current"], expected_current)


def test_m3_checkpoint_format_unchanged_with_mcp1_grid_resolved_config() -> None:
    payload = _checkpoint_for_pair(global_step=3)
    payload["resolved_config"]["m3"].update(
        {
            "mcp1_grid_aux_weight": 1.0,
            "mcp1_grid_aux_enabled": True,
            "mcp1_grid_timesteps": list(train_m3.MCP1_GRID_EXPECTED_TIMESTEPS),
            "mcp1_grid_schedule": {"source": "teacher_payload"},
        }
    )
    m3.validate_m3_checkpoint_payload(payload)

    assert payload["format"] == M3_CHECKPOINT_FORMAT
    assert payload["resolved_config"]["m3"]["mcp1_grid_aux_enabled"] is True
