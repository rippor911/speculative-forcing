from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import pytest
import torch

import utils.nf_sf_m3 as m3
import utils.nf_sf_m4 as m4
import utils.nf_sf_m5_formal_plan as formal_plan
import utils.nf_sf_m5_samples as samples

TEST_GIT_SHA = "0123456789abcdef0123456789abcdef01234567"


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _formal_record(split: str, split_index: int, sample_index: int) -> dict[str, Any]:
    prompt = f"{split} prompt {split_index}"
    file_name = f"payloads/{split}_{split_index:06d}.pt"
    target = torch.zeros((1, 15, 1, 1, 1), dtype=torch.bfloat16)
    source = torch.ones((1, 15, 1, 1, 1), dtype=torch.bfloat16)
    return {
        "status": "GENERATED",
        "sample_index": sample_index,
        "sample_id": f"{split}-{split_index:06d}",
        "split": split,
        "split_index": split_index,
        "source_line_index": sample_index,
        "shard_id": 0,
        "plan_index": sample_index,
        "file": file_name,
        "file_sha256": _sha256_text(f"{file_name}:payload"),
        "prompt": prompt,
        "prompt_sha256": _sha256_text(prompt),
        "target_latent": m3.tensor_summary(target),
        "source_noise": m3.tensor_summary(source),
    }


def _formal_records() -> list[dict[str, Any]]:
    records = []
    for split_index in range(formal_plan.M5_FORMAL_TRAIN_SAMPLE_COUNT):
        records.append(_formal_record("train", split_index, split_index))
    for split_index in range(formal_plan.M5_FORMAL_VALIDATION_SAMPLE_COUNT):
        records.append(
            _formal_record("validation", split_index, 100_000 + split_index)
        )
    return records


def _write_manifest(
    directory: Path,
    *,
    records: list[dict[str, Any]] | None = None,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    samples_list = list(reversed(_formal_records())) if records is None else records
    manifest = {
        "status": "PASS",
        "experiment": "E0208C_teacher_rollout_formal",
        "format": "self_forcing_teacher_manifest_v2",
        "writer_format": "e0208_teacher_writer_v1",
        "writer_git_head": TEST_GIT_SHA,
        "checkpoint": {
            "path": "checkpoints/self_forcing_dmd.pt",
            "sha256": m3.M3_REFERENCE_CHECKPOINT_SHA256,
        },
        "generation": {
            "num_samples": len(samples_list),
            "num_completed": len(samples_list),
            "num_train": formal_plan.M5_FORMAL_TRAIN_SAMPLE_COUNT,
            "num_validation": formal_plan.M5_FORMAL_VALIDATION_SAMPLE_COUNT,
            "num_reserve": 0,
            "num_frames": 15,
            "num_frame_per_block": 3,
            "num_blocks": 5,
            "mcp_depth": 3,
            "mcp_num_modules": 0,
            "mcp_accel_depths": 0,
            "last_step_only": True,
        },
        "samples": samples_list,
    }
    path = directory / "manifest.json"
    path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    return path


def _case(tmp_path: Path) -> tuple[Path, Path, dict[str, Any]]:
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    manifest_path = _write_manifest(tmp_path)
    plan = formal_plan.build_m5_formal_sample_plan(
        manifest_path=manifest_path,
        dataset_root=dataset_root,
    )
    return manifest_path, dataset_root, plan


def _store(
    manifest_path: Path,
    dataset_root: Path,
    plan: dict[str, Any],
) -> samples.M5TeacherSampleStore:
    return samples.M5TeacherSampleStore(
        sample_plan=plan,
        manifest_path=manifest_path,
        dataset_root=dataset_root,
    )


def _refresh_plan_sha(plan: dict[str, Any]) -> dict[str, Any]:
    plan["sample_plan_sha256"] = m4.m4_sample_plan_sha256(plan)
    return plan


def _sample_for_entry(
    entry: samples.M5TeacherSampleEntry,
    *,
    manifest_path: Path,
    manifest_sha256: str,
    dataset_root: Path,
    overrides: dict[str, Any] | None = None,
) -> m3.M3TeacherSample:
    metadata = {
        "sample_index": entry.sample_index,
        "sample_id": entry.sample_id,
        "split": entry.split,
        "split_index": entry.split_index,
        "prompt_sha256": entry.prompt_sha256,
        "latent_file_sha256": entry.file_sha256,
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": manifest_sha256,
        "dataset_root": str(dataset_root.resolve()),
    }
    if overrides:
        metadata.update(overrides)
    tensor = torch.zeros((1, 15, 1, 1, 1), dtype=torch.bfloat16)
    return m3.M3TeacherSample(
        payload={},
        target_latent=tensor,
        source_noise=tensor,
        selected_state=object(),
        metadata=metadata,
    )


def _patch_fake_loader(
    monkeypatch: pytest.MonkeyPatch,
    store: samples.M5TeacherSampleStore,
    calls: list[dict[str, Any]],
    *,
    overrides_by_sample_index: dict[int, dict[str, Any]] | None = None,
) -> None:
    def fake_loader(**kwargs: Any) -> m3.M3TeacherSample:
        calls.append(dict(kwargs))
        sample_index = kwargs["sample_index"]
        matches = [
            store.entry(identity)
            for identity in (*store.train_identities, *store.validation_identities)
            if store.entry(identity).sample_index == sample_index
        ]
        if len(matches) != 1:
            raise AssertionError(f"unexpected sample_index={sample_index}")
        return _sample_for_entry(
            matches[0],
            manifest_path=store.manifest_path,
            manifest_sha256=store.manifest_sha256,
            dataset_root=store.dataset_root,
            overrides=(overrides_by_sample_index or {}).get(sample_index),
        )

    monkeypatch.setattr(samples, "load_m3_teacher_sample", fake_loader)


def test_store_initialization_is_lazy_and_preserves_plan_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path, dataset_root, plan = _case(tmp_path)
    torch_load_calls = []

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("unexpected eager payload or CUDA access")

    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: torch_load_calls.append(args))
    monkeypatch.setattr(torch.cuda, "is_available", forbidden)
    monkeypatch.setattr(torch.cuda, "device_count", forbidden)
    monkeypatch.setattr(m4, "load_m4_teacher_samples", forbidden)

    store = _store(manifest_path, dataset_root, plan)

    assert torch_load_calls == []
    assert store.sample_plan_sha256 == plan["sample_plan_sha256"]
    assert store.manifest_path == manifest_path.resolve()
    assert store.manifest_sha256 == plan["manifest_sha256"]
    assert store.dataset_root == dataset_root.resolve()
    assert store.train_identities == tuple(plan["train_sample_identities"])
    assert store.validation_identities == tuple(plan["validation_sample_identities"])
    assert (
        store.fixed_decode_validation_identity
        == plan["fixed_decode_validation_identity"]
    )
    assert store.live_sample_count == 0
    assert store.max_live_sample_count == 0
    assert store.total_load_count == 0
    assert store.load_attempt_count == 0


@pytest.mark.parametrize(
    "property_name",
    [
        "sample_plan_sha256",
        "manifest_path",
        "manifest_sha256",
        "dataset_root",
        "fixed_decode_validation_identity",
        "train_identities",
        "validation_identities",
    ],
)
def test_store_provenance_properties_are_read_only(
    tmp_path: Path,
    property_name: str,
) -> None:
    manifest_path, dataset_root, plan = _case(tmp_path)
    store = _store(manifest_path, dataset_root, plan)
    assert store.sample_plan_sha256 == plan["sample_plan_sha256"]
    assert store.manifest_path == manifest_path.resolve()
    assert store.manifest_sha256 == plan["manifest_sha256"]
    assert store.dataset_root == dataset_root.resolve()
    assert (
        store.fixed_decode_validation_identity
        == plan["fixed_decode_validation_identity"]
    )
    assert store.train_identities == tuple(plan["train_sample_identities"])
    assert store.validation_identities == tuple(plan["validation_sample_identities"])

    with pytest.raises(AttributeError):
        setattr(store, property_name, "mutated")


def test_missing_payload_files_do_not_block_store_initialization(tmp_path: Path) -> None:
    manifest_path, dataset_root, plan = _case(tmp_path)
    assert not (dataset_root / plan["samples"]["train"][0]["file"]).exists()
    store = _store(manifest_path, dataset_root, plan)
    assert len(store.train_identities) == 2048
    assert len(store.validation_identities) == 256


def test_store_copies_plan_and_returns_immutable_entries(tmp_path: Path) -> None:
    manifest_path, dataset_root, plan = _case(tmp_path)
    store = _store(manifest_path, dataset_root, plan)
    first_identity = store.train_identities[0]
    first_entry = store.entry(first_identity)

    plan["train_sample_identities"][0] = "mutated"
    plan["samples"]["train"][0]["sample_index"] = 999

    assert store.train_identities[0] == first_identity
    assert store.entry(first_identity) == first_entry
    with pytest.raises(FrozenInstanceError):
        first_entry.sample_index = 999


def test_unknown_identity_rejects_before_loader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path, dataset_root, plan = _case(tmp_path)
    store = _store(manifest_path, dataset_root, plan)

    def forbidden(**kwargs: Any) -> None:
        raise AssertionError("loader should not be called")

    monkeypatch.setattr(samples, "load_m3_teacher_sample", forbidden)
    with pytest.raises(RuntimeError, match="unknown"), store.acquire("not-in-plan"):
        pass
    assert store.load_attempt_count == 0
    assert store.total_load_count == 0


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (
            lambda entry: entry.pop("file"),
            "payload location",
        ),
        (
            lambda entry: entry.__setitem__("file", ""),
            "non-empty",
        ),
        (
            lambda entry: entry.__setitem__("file", 123),
            "string",
        ),
    ],
)
def test_store_rejects_missing_or_invalid_location(
    tmp_path: Path,
    mutator,
    message: str,
) -> None:
    manifest_path, dataset_root, plan = _case(tmp_path)
    tampered = copy.deepcopy(plan)
    mutator(tampered["samples"]["train"][0])
    _refresh_plan_sha(tampered)

    with pytest.raises((RuntimeError, TypeError), match=message):
        _store(manifest_path, dataset_root, tampered)


@pytest.mark.parametrize("same_value", [False, True])
def test_store_rejects_multiple_location_fields(
    tmp_path: Path,
    same_value: bool,
) -> None:
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    records = _formal_records()
    records[0]["path"] = records[0]["file"] if same_value else "alternate.pt"
    manifest_path = _write_manifest(tmp_path, records=list(reversed(records)))
    plan = formal_plan.build_m5_formal_sample_plan(
        manifest_path=manifest_path,
        dataset_root=dataset_root,
    )

    with pytest.raises(RuntimeError, match="exactly one payload location"):
        _store(manifest_path, dataset_root, plan)


def test_store_rejects_tmp_payload_location(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    records = _formal_records()
    records[0]["file"] = "payloads/train_000000.pt.tmp"
    manifest_path = _write_manifest(tmp_path, records=list(reversed(records)))
    plan = formal_plan.build_m5_formal_sample_plan(
        manifest_path=manifest_path,
        dataset_root=dataset_root,
    )

    with pytest.raises(RuntimeError, match=".tmp"):
        _store(manifest_path, dataset_root, plan)


def test_train_step_mapping_uses_m4_absolute_one_based_rule(tmp_path: Path) -> None:
    manifest_path, dataset_root, plan = _case(tmp_path)
    store = _store(manifest_path, dataset_root, plan)

    assert store.train_identity_for_step(1) == plan["train_sample_identities"][0]
    assert store.train_identity_for_step(2048) == plan["train_sample_identities"][2047]
    assert store.train_identity_for_step(2049) == plan["train_sample_identities"][0]
    assert store.train_identity_for_step(5000) == str(
        m4.m4_train_entry_for_step(plan, 5000)["identity"]
    )
    assert store.train_entry_for_step(1).identity == store.train_identity_for_step(1)


@pytest.mark.parametrize("bad_step", [0, -1])
def test_train_step_mapping_rejects_non_positive_steps(
    tmp_path: Path,
    bad_step: int,
) -> None:
    manifest_path, dataset_root, plan = _case(tmp_path)
    store = _store(manifest_path, dataset_root, plan)
    with pytest.raises(ValueError, match="step"):
        store.train_identity_for_step(bad_step)


@pytest.mark.parametrize("bad_step", [True, 1.0, "1"])
def test_train_step_mapping_rejects_non_python_int_steps(
    tmp_path: Path,
    bad_step: object,
) -> None:
    manifest_path, dataset_root, plan = _case(tmp_path)
    store = _store(manifest_path, dataset_root, plan)
    with pytest.raises(TypeError, match="Python int"):
        store.train_identity_for_step(bad_step)


def test_acquire_loads_only_requested_identity_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path, dataset_root, plan = _case(tmp_path)
    store = _store(manifest_path, dataset_root, plan)
    calls: list[dict[str, Any]] = []
    _patch_fake_loader(monkeypatch, store, calls)

    identity = store.train_identities[0]
    entry = store.entry(identity)
    with store.acquire(identity) as sample:
        assert isinstance(sample, m3.M3TeacherSample)
        assert store.live_sample_count == 1
        assert sample.metadata["sample_index"] == entry.sample_index

    assert len(calls) == 1
    assert calls[0]["sample_index"] == entry.sample_index
    assert calls[0]["manifest_path"] == store.manifest_path
    assert calls[0]["dataset_root"] == store.dataset_root
    assert calls[0]["reference_checkpoint_path"] is None
    assert calls[0]["expected_reference_sha256"] == m3.M3_REFERENCE_CHECKPOINT_SHA256
    assert store.live_sample_count == 0
    assert store.max_live_sample_count == 1
    assert store.total_load_count == 1
    assert store.load_attempt_count == 1


def test_loader_reentrant_acquire_rejects_before_second_loader_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path, dataset_root, plan = _case(tmp_path)
    store = _store(manifest_path, dataset_root, plan)
    calls: list[dict[str, Any]] = []
    nested_errors: list[str] = []

    def reentrant_loader(**kwargs: Any) -> m3.M3TeacherSample:
        calls.append(dict(kwargs))
        with pytest.raises(RuntimeError, match="active") as exc_info, store.acquire(
            store.train_identities[1]
        ):
            pass
        nested_errors.append(str(exc_info.value))
        return _sample_for_entry(
            store.entry(store.train_identities[0]),
            manifest_path=store.manifest_path,
            manifest_sha256=store.manifest_sha256,
            dataset_root=store.dataset_root,
        )

    monkeypatch.setattr(samples, "load_m3_teacher_sample", reentrant_loader)

    with store.acquire(store.train_identities[0]) as sample:
        assert isinstance(sample, m3.M3TeacherSample)
        assert store.live_sample_count == 1

    assert len(calls) == 1
    assert len(nested_errors) == 1
    assert store.live_sample_count == 0
    assert store.max_live_sample_count == 1
    assert store.load_attempt_count == 1
    assert store.total_load_count == 1


def test_real_tiny_payload_integration_uses_existing_loader(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    payload_dir = dataset_root / "payloads"
    payload_dir.mkdir(parents=True)
    records = _formal_records()
    record = records[0]
    target = torch.arange(15, dtype=torch.float32).reshape(1, 15, 1, 1, 1).to(
        torch.bfloat16
    )
    source = (target.float() + 100.0).to(torch.bfloat16)
    record["target_latent"] = m3.tensor_summary(target)
    record["source_noise"] = m3.tensor_summary(source)
    payload = {
        "format": "self_forcing_teacher_v1",
        "sample_index": record["sample_index"],
        "sample_id": record["sample_id"],
        "split": record["split"],
        "split_index": record["split_index"],
        "source_line_index": record["source_line_index"],
        "shard_id": record["shard_id"],
        "plan_index": record["plan_index"],
        "prompt": record["prompt"],
        "prompt_sha256": record["prompt_sha256"],
        "seed": 1000000,
        "noise_seed": 1000001,
        "rollout_seed": 1000002,
        "source_noise": source,
        "target_latent": target,
        "backbone_sha256": m3.M3_REFERENCE_CHECKPOINT_SHA256,
        "num_frames": 15,
        "num_frame_per_block": 3,
        "mcp_depth": 3,
        "raw_denoising_steps": [1000.0, 750.0, 500.0, 250.0],
        "warped_denoising_steps": [1000.0, 750.0, 500.0, 250.0],
        "writer_git_head": TEST_GIT_SHA,
    }
    payload_path = dataset_root / record["file"]
    torch.save(payload, payload_path)
    record["file_sha256"] = m3.file_sha256(payload_path)
    manifest_path = _write_manifest(tmp_path, records=list(reversed(records)))
    plan = formal_plan.build_m5_formal_sample_plan(
        manifest_path=manifest_path,
        dataset_root=dataset_root,
    )
    store = _store(manifest_path, dataset_root, plan)

    with store.acquire(store.train_identities[0]) as sample:
        assert isinstance(sample, m3.M3TeacherSample)
        assert m4.m4_sample_identity_from_metadata(sample.metadata) == store.train_identities[0]
        assert sample.selected_state is not None

    assert store.live_sample_count == 0
    assert store.total_load_count == 1


def test_acquire_missing_file_fails_closed(tmp_path: Path) -> None:
    manifest_path, dataset_root, plan = _case(tmp_path)
    store = _store(manifest_path, dataset_root, plan)
    with pytest.raises(FileNotFoundError), store.acquire(store.train_identities[0]):
        pass
    assert store.live_sample_count == 0
    assert store.max_live_sample_count == 0
    assert store.load_attempt_count == 1
    assert store.total_load_count == 0


def test_acquire_file_sha_mismatch_fails_closed(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    payload_dir = dataset_root / "payloads"
    payload_dir.mkdir(parents=True)
    records = _formal_records()
    record = records[0]
    payload_path = dataset_root / record["file"]
    torch.save({"not": "a valid payload"}, payload_path)
    record["file_sha256"] = "0" * 64
    manifest_path = _write_manifest(tmp_path, records=list(reversed(records)))
    plan = formal_plan.build_m5_formal_sample_plan(
        manifest_path=manifest_path,
        dataset_root=dataset_root,
    )
    store = _store(manifest_path, dataset_root, plan)

    with pytest.raises(RuntimeError, match="SHA256 mismatch"), store.acquire(
        store.train_identities[0]
    ):
        pass
    assert store.live_sample_count == 0
    assert store.total_load_count == 0


def test_real_loader_payload_manifest_metadata_mismatch_fails_closed(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "dataset"
    payload_dir = dataset_root / "payloads"
    payload_dir.mkdir(parents=True)
    records = _formal_records()
    record = records[0]
    target = torch.zeros((1, 15, 1, 1, 1), dtype=torch.bfloat16)
    source = torch.ones((1, 15, 1, 1, 1), dtype=torch.bfloat16)
    payload = {
        "format": "self_forcing_teacher_v1",
        "sample_index": record["sample_index"],
        "sample_id": record["sample_id"],
        "split": "validation",
        "split_index": record["split_index"],
        "source_line_index": record["source_line_index"],
        "shard_id": record["shard_id"],
        "plan_index": record["plan_index"],
        "prompt": record["prompt"],
        "prompt_sha256": record["prompt_sha256"],
        "seed": 1000000,
        "noise_seed": 1000001,
        "rollout_seed": 1000002,
        "source_noise": source,
        "target_latent": target,
        "backbone_sha256": m3.M3_REFERENCE_CHECKPOINT_SHA256,
        "num_frames": 15,
        "num_frame_per_block": 3,
        "mcp_depth": 3,
        "raw_denoising_steps": [1000.0],
        "warped_denoising_steps": [1000.0],
        "writer_git_head": TEST_GIT_SHA,
    }
    payload_path = dataset_root / record["file"]
    torch.save(payload, payload_path)
    record["file_sha256"] = m3.file_sha256(payload_path)
    manifest_path = _write_manifest(tmp_path, records=list(reversed(records)))
    plan = formal_plan.build_m5_formal_sample_plan(
        manifest_path=manifest_path,
        dataset_root=dataset_root,
    )
    store = _store(manifest_path, dataset_root, plan)

    with pytest.raises(RuntimeError, match="split"), store.acquire(
        store.train_identities[0]
    ):
        pass
    assert store.live_sample_count == 0
    assert store.total_load_count == 0


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"split": "validation"}, "metadata.split"),
        ({"split_index": 99}, "metadata.split_index"),
        ({"prompt_sha256": "0" * 64}, "metadata.prompt_sha256"),
        ({"latent_file_sha256": "1" * 64}, "metadata.latent_file_sha256"),
        ({"manifest_sha256": "2" * 64}, "metadata.manifest_sha256"),
    ],
)
def test_store_extra_metadata_checks_reject_mismatches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    override: dict[str, Any],
    message: str,
) -> None:
    manifest_path, dataset_root, plan = _case(tmp_path)
    store = _store(manifest_path, dataset_root, plan)
    entry = store.entry(store.train_identities[0])
    calls: list[dict[str, Any]] = []
    _patch_fake_loader(
        monkeypatch,
        store,
        calls,
        overrides_by_sample_index={entry.sample_index: override},
    )

    with pytest.raises(RuntimeError, match=message), store.acquire(entry.identity):
        pass
    assert calls
    assert store.live_sample_count == 0
    assert store.total_load_count == 0


def test_store_rejects_loader_returned_wrong_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path, dataset_root, plan = _case(tmp_path)
    store = _store(manifest_path, dataset_root, plan)
    calls: list[dict[str, Any]] = []
    _patch_fake_loader(monkeypatch, store, calls)
    monkeypatch.setattr(samples, "m4_sample_identity_from_metadata", lambda metadata: "wrong")

    with pytest.raises(RuntimeError, match="identity mismatch"), store.acquire(
        store.train_identities[0]
    ):
        pass
    assert store.live_sample_count == 0
    assert store.total_load_count == 0


def test_lifecycle_sequential_and_nested_acquire(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path, dataset_root, plan = _case(tmp_path)
    store = _store(manifest_path, dataset_root, plan)
    calls: list[dict[str, Any]] = []
    _patch_fake_loader(monkeypatch, store, calls)

    first = store.train_identities[0]
    second = store.train_identities[1]
    with store.acquire(first) as sample:
        assert store.live_sample_count == 1
        assert store._active_sample is sample
        with pytest.raises(RuntimeError, match="active"), store.acquire(second):
            pass
    assert store.live_sample_count == 0
    assert store._active_sample is None

    with store.acquire(second):
        assert store.live_sample_count == 1
    assert store.live_sample_count == 0
    assert store.max_live_sample_count == 1
    assert store.total_load_count == 2
    assert store.load_attempt_count == 2
    assert len(calls) == 2


def test_loader_exception_restores_lifecycle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path, dataset_root, plan = _case(tmp_path)
    store = _store(manifest_path, dataset_root, plan)

    def fail_loader(**kwargs: Any) -> None:
        raise RuntimeError("loader failed")

    monkeypatch.setattr(samples, "load_m3_teacher_sample", fail_loader)
    with pytest.raises(RuntimeError, match="loader failed"), store.acquire(
        store.train_identities[0]
    ):
        pass
    assert store.live_sample_count == 0
    assert store.max_live_sample_count == 0
    assert store.load_attempt_count == 1
    assert store.total_load_count == 0
    assert store._active_sample is None
    assert store._acquire_in_progress is False

    calls: list[dict[str, Any]] = []
    _patch_fake_loader(monkeypatch, store, calls)
    with store.acquire(store.train_identities[0]):
        assert store.live_sample_count == 1
    assert len(calls) == 1
    assert store.live_sample_count == 0
    assert store.load_attempt_count == 2
    assert store.total_load_count == 1


def test_metadata_validation_failure_restores_lifecycle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path, dataset_root, plan = _case(tmp_path)
    store = _store(manifest_path, dataset_root, plan)
    entry = store.entry(store.train_identities[0])
    calls: list[dict[str, Any]] = []
    _patch_fake_loader(
        monkeypatch,
        store,
        calls,
        overrides_by_sample_index={entry.sample_index: {"sample_index": 999}},
    )

    with pytest.raises(RuntimeError, match="metadata.sample_index"), store.acquire(
        entry.identity
    ):
        pass
    assert store.live_sample_count == 0
    assert store.max_live_sample_count == 0
    assert store.load_attempt_count == 1
    assert store.total_load_count == 0
    assert store._active_sample is None
    assert store._acquire_in_progress is False

    calls.clear()
    _patch_fake_loader(monkeypatch, store, calls)
    with store.acquire(entry.identity):
        assert store.live_sample_count == 1
    assert len(calls) == 1
    assert store.live_sample_count == 0
    assert store.load_attempt_count == 2
    assert store.total_load_count == 1


def test_with_body_exception_restores_lifecycle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path, dataset_root, plan = _case(tmp_path)
    store = _store(manifest_path, dataset_root, plan)
    calls: list[dict[str, Any]] = []
    _patch_fake_loader(monkeypatch, store, calls)

    with pytest.raises(ValueError, match="body failed"), store.acquire(
        store.train_identities[0]
    ):
        assert store.live_sample_count == 1
        raise ValueError("body failed")
    assert store.live_sample_count == 0
    assert store.max_live_sample_count == 1
    assert store.total_load_count == 1
    assert store._active_sample is None
    assert store._acquire_in_progress is False

    with store.acquire(store.train_identities[1]):
        assert store.live_sample_count == 1
    assert store.live_sample_count == 0
    assert store.max_live_sample_count == 1
    assert store.total_load_count == 2
    assert store.load_attempt_count == 2
    assert len(calls) == 2


def test_store_does_not_call_cuda_text_encoder_or_eager_loader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path, dataset_root, plan = _case(tmp_path)

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("forbidden side effect")

    monkeypatch.setattr(torch.cuda, "is_available", forbidden)
    monkeypatch.setattr(torch.cuda, "device_count", forbidden)
    monkeypatch.setattr(m4, "load_m4_teacher_samples", forbidden)
    store = _store(manifest_path, dataset_root, plan)
    calls: list[dict[str, Any]] = []
    _patch_fake_loader(monkeypatch, store, calls)

    with store.acquire(store.validation_identities[0]):
        pass

    assert len(calls) == 1
    assert store.max_live_sample_count == 1
