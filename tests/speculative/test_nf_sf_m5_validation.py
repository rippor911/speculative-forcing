from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
import torch

import utils.nf_sf_m4 as m4
import utils.nf_sf_m5_conditionals as conditionals
import utils.nf_sf_m5_formal_plan as formal_plan
import utils.nf_sf_m5_validation as validation
from utils.nf_sf_m3 import M3_REFERENCE_CHECKPOINT_SHA256

TEST_GIT_SHA = "0123456789abcdef0123456789abcdef01234567"
TEST_ARTIFACT_SHA = "2" * 64


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _formal_record(split: str, split_index: int, sample_index: int) -> dict[str, Any]:
    prompt = f"{split} prompt {split_index}"
    file_name = f"payloads/{split}_{split_index:06d}.pt"
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
    }


def _records() -> list[dict[str, Any]]:
    values = []
    for split_index in range(formal_plan.M5_FORMAL_TRAIN_SAMPLE_COUNT):
        values.append(_formal_record("train", split_index, split_index))
    for split_index in range(formal_plan.M5_FORMAL_VALIDATION_SAMPLE_COUNT):
        values.append(
            _formal_record("validation", split_index, 100_000 + split_index)
        )
    return values


def _write_manifest(directory: Path) -> Path:
    samples = list(reversed(_records()))
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
            "num_samples": len(samples),
            "num_completed": len(samples),
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
        "samples": samples,
    }
    path = directory / "manifest.json"
    path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    return path


@pytest.fixture
def formal_case(tmp_path: Path) -> dict[str, Any]:
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    manifest_path = _write_manifest(tmp_path)
    return formal_plan.build_m5_formal_sample_plan(
        manifest_path=manifest_path,
        dataset_root=dataset_root,
    )


class FakeSample:
    def __init__(self, identity: str) -> None:
        self.identity = identity


class FakeTeacherStore:
    def __init__(
        self,
        plan: Mapping[str, Any],
        *,
        history_attempt_count: int = 0,
        history_total_load_count: int = 0,
        history_max_live_count: int = 0,
    ) -> None:
        self.sample_plan_sha256 = str(plan["sample_plan_sha256"])
        self.manifest_sha256 = str(plan["manifest_sha256"])
        self.train_identities = tuple(str(value) for value in plan["train_sample_identities"])
        self.validation_identities = tuple(
            str(value) for value in plan["validation_sample_identities"]
        )
        self.live_sample_count = 0
        self.max_live_sample_count = history_max_live_count
        self.load_attempt_count = history_attempt_count
        self.total_load_count = history_total_load_count
        self.acquired: list[str] = []
        self.train_acquire_count = 0
        self._active = False

    @contextmanager
    def acquire(self, identity: str) -> Iterator[FakeSample]:
        if self._active:
            raise RuntimeError("teacher store already active")
        if identity in self.train_identities:
            self.train_acquire_count += 1
        if identity not in self.validation_identities:
            raise RuntimeError(f"unexpected teacher identity: {identity}")
        self.load_attempt_count += 1
        self._active = True
        self.live_sample_count = 1
        self.max_live_sample_count = max(self.max_live_sample_count, 1)
        self.total_load_count += 1
        self.acquired.append(identity)
        try:
            yield FakeSample(identity)
        finally:
            self._active = False
            self.live_sample_count = 0


class FakeConditionalStore:
    def __init__(
        self,
        plan: Mapping[str, Any],
        *,
        history_attempt_count: int = 0,
        history_total_load_count: int = 0,
        history_max_live_count: int = 0,
    ) -> None:
        self.sample_plan_sha256 = str(plan["sample_plan_sha256"])
        self.teacher_manifest_sha256 = str(plan["manifest_sha256"])
        self.artifact_sha256 = TEST_ARTIFACT_SHA
        self.train_identities = tuple(str(value) for value in plan["train_sample_identities"])
        self.validation_identities = tuple(
            str(value) for value in plan["validation_sample_identities"]
        )
        self.live_conditional_count = 0
        self.max_live_conditional_count = history_max_live_count
        self.load_attempt_count = history_attempt_count
        self.total_load_count = history_total_load_count
        self.acquired: list[str] = []
        self.train_acquire_count = 0
        self._active = False

    @contextmanager
    def acquire(self, identity: str) -> Iterator[dict[str, torch.Tensor]]:
        if self._active:
            raise RuntimeError("conditional store already active")
        if identity in self.train_identities:
            self.train_acquire_count += 1
        if identity not in self.validation_identities:
            raise RuntimeError(f"unexpected conditional identity: {identity}")
        self.load_attempt_count += 1
        self._active = True
        self.live_conditional_count = 1
        self.max_live_conditional_count = max(self.max_live_conditional_count, 1)
        self.total_load_count += 1
        self.acquired.append(identity)
        try:
            yield {"prompt_embeds": torch.ones((1, 2), dtype=torch.float32)}
        finally:
            self._active = False
            self.live_conditional_count = 0


def _losses(position: int, *, nonfinite: bool = False) -> dict[str, float | None]:
    base = float(position + 1)
    return {
        "main_loss": float("nan") if nonfinite else base,
        "mcp_depth1_loss": base + 1.0,
        "mcp_depth2_loss": base + 2.0,
        "mcp_depth3_loss": base + 3.0,
        "weighted_mcp_loss": base + 4.0,
        "total_validation_loss": base + 5.0,
    }


def _child_report(
    *,
    identity: str,
    position: int,
    global_step: int,
    status: str = "PASS",
    nonfinite: bool = False,
) -> dict[str, Any]:
    losses = _losses(position, nonfinite=nonfinite)
    return {
        "schema": m4.M4_VALIDATION_SCHEMA,
        "status": "FAIL" if nonfinite else status,
        "global_step": global_step,
        "validation_sample_identities": [identity],
        "sample_count": 1,
        "per_sample_losses": [
            {
                "sample_identity": identity,
                "sample_position": 0,
                "validation_seed": 9000 + position,
                "losses": losses,
                "tensor_contract": {"digest": f"digest-{position}"},
            }
        ],
        "aggregate_losses": losses,
        "validation_loss_finite_contract": not nonfinite,
        "nonfinite_validation_losses": []
        if not nonfinite
        else [{"scope": "sample", "sample_identity": identity, "fields": ["main_loss"]}],
        "gradients_unchanged_contract": True,
        "requires_grad_unchanged_contract": True,
        "train_rng_unchanged_contract": True,
        "probe_rng_unchanged_contract": True,
        "global_cpu_rng_unchanged_contract": True,
        "global_cuda_rng_unchanged_contract": True,
    }


def _patch_run_m4_validation(
    monkeypatch: pytest.MonkeyPatch,
    calls: list[dict[str, Any]],
    *,
    fail_at: int | None = None,
    nonfinite_at: int | None = None,
    raise_at: int | None = None,
) -> None:
    def fake_run_m4_validation(**kwargs: Any) -> dict[str, Any]:
        position = len(calls)
        identity = next(iter(kwargs["conditional_dicts"]))
        snapshot = dict(kwargs)
        snapshot["samples"] = list(kwargs["samples"])
        snapshot["conditional_dicts"] = {
            str(key): dict(value)
            for key, value in kwargs["conditional_dicts"].items()
        }
        calls.append(snapshot)
        if raise_at == position:
            raise RuntimeError("child validation exploded")
        return _child_report(
            identity=identity,
            position=position,
            global_step=kwargs["global_step"],
            status="FAIL" if fail_at == position else "PASS",
            nonfinite=nonfinite_at == position,
        )

    monkeypatch.setattr(validation, "run_m4_validation", fake_run_m4_validation)


def _run(
    plan: Mapping[str, Any],
    teacher_store: FakeTeacherStore,
    conditional_store: FakeConditionalStore,
) -> dict[str, Any]:
    return validation.run_m5_streaming_validation(
        generator=object(),
        teacher_store=teacher_store,
        conditional_store=conditional_store,
        sample_plan=plan,
        scheduler_main=object(),
        scheduler_mcp=object(),
        device=torch.device("cpu"),
        dtype=torch.float64,
        mode="joint",
        global_step=500,
        validation_seed=1234,
        train_rng=None,
        probe_rng_state=None,
        model_identity={"model": "tiny"},
    )


def _assert_no_tensor(value: Any) -> None:
    if isinstance(value, torch.Tensor):
        pytest.fail("report contains torch.Tensor")
    if isinstance(value, Mapping):
        for item in value.values():
            _assert_no_tensor(item)
    elif isinstance(value, list | tuple):
        for item in value:
            _assert_no_tensor(item)


def _locals_for_frame(exc: BaseException, frame_name: str) -> dict[str, Any]:
    traceback = exc.__traceback__
    while traceback is not None:
        frame = traceback.tb_frame
        if frame.f_code.co_name == frame_name:
            return dict(frame.f_locals)
        traceback = traceback.tb_next
    raise AssertionError(f"missing traceback frame: {frame_name}")


def test_streaming_validation_order_lifecycle_and_aggregation(
    formal_case: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    teacher_store = FakeTeacherStore(formal_case)
    conditional_store = FakeConditionalStore(formal_case)
    calls: list[dict[str, Any]] = []

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("unexpected eager validation helper")

    monkeypatch.setattr(m4, "load_m4_teacher_samples", forbidden)
    monkeypatch.setattr(conditionals, "encode_m5_prompt_condition", forbidden)
    _patch_run_m4_validation(monkeypatch, calls)

    report = _run(formal_case, teacher_store, conditional_store)
    expected_identities = list(formal_case["validation_sample_identities"])

    assert report["schema"] == validation.M5_STREAMING_VALIDATION_SCHEMA
    assert report["status"] == "PASS"
    assert report["validation_sample_identities"] == expected_identities
    assert [
        item["sample_position"]
        for item in report["per_sample_losses"]
    ] == list(range(256))
    assert teacher_store.acquired == expected_identities
    assert conditional_store.acquired == expected_identities
    assert teacher_store.train_acquire_count == 0
    assert conditional_store.train_acquire_count == 0
    assert len(calls) == formal_plan.M5_FORMAL_VALIDATION_SAMPLE_COUNT
    for position, call in enumerate(calls):
        identity = expected_identities[position]
        assert len(call["samples"]) == 1
        assert set(call["conditional_dicts"]) == {identity}
        conditional = call["conditional_dicts"][identity]
        assert set(conditional) == {"prompt_embeds"}
        assert conditional["prompt_embeds"].device.type == "cpu"
        assert conditional["prompt_embeds"].dtype is torch.float64
    assert teacher_store.max_live_sample_count == 1
    assert conditional_store.max_live_conditional_count == 1
    assert teacher_store.live_sample_count == 0
    assert conditional_store.live_conditional_count == 0
    assert report["max_live_teacher_samples"] == 1
    assert report["max_live_conditionals"] == 1
    assert report["teacher_store_telemetry_delta"]["successful_load_count_delta"] == 256
    assert (
        report["conditional_store_telemetry_delta"]["successful_load_count_delta"]
        == 256
    )
    assert report["sample_count"] == 256
    assert report["aggregate_losses"]["main_loss"] == pytest.approx(128.5)
    assert report["aggregate_losses"]["total_validation_loss"] == pytest.approx(133.5)
    assert report["all_child_m4_contracts_pass"] is True
    json.dumps(report, allow_nan=False)
    _assert_no_tensor(report)


def test_historical_telemetry_does_not_affect_run_peak_or_pass_status(
    formal_case: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    teacher_store = FakeTeacherStore(
        formal_case,
        history_attempt_count=11,
        history_total_load_count=13,
        history_max_live_count=7,
    )
    conditional_store = FakeConditionalStore(
        formal_case,
        history_attempt_count=17,
        history_total_load_count=19,
        history_max_live_count=9,
    )
    calls: list[dict[str, Any]] = []
    _patch_run_m4_validation(monkeypatch, calls)

    report = _run(formal_case, teacher_store, conditional_store)

    assert report["status"] == "PASS"
    assert report["teacher_store_telemetry_delta"]["load_attempt_count_delta"] == 256
    assert (
        report["teacher_store_telemetry_delta"]["successful_load_count_delta"]
        == 256
    )
    assert (
        report["conditional_store_telemetry_delta"]["load_attempt_count_delta"]
        == 256
    )
    assert (
        report["conditional_store_telemetry_delta"]["successful_load_count_delta"]
        == 256
    )
    assert "max_live_count_before" not in report["teacher_store_telemetry_delta"]
    assert "max_live_count_after" not in report["teacher_store_telemetry_delta"]
    assert "max_live_count_before" not in report["conditional_store_telemetry_delta"]
    assert "max_live_count_after" not in report["conditional_store_telemetry_delta"]
    assert report["max_live_teacher_samples"] == 1
    assert report["max_live_conditionals"] == 1
    assert teacher_store.live_sample_count == 0
    assert conditional_store.live_conditional_count == 0
    assert teacher_store.max_live_sample_count == 7
    assert conditional_store.max_live_conditional_count == 9


def test_sample_plan_store_sha_mismatch_rejects(
    formal_case: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    teacher_store = FakeTeacherStore(formal_case)
    conditional_store = FakeConditionalStore(formal_case)
    teacher_store.sample_plan_sha256 = "0" * 64
    _patch_run_m4_validation(monkeypatch, [])

    with pytest.raises(RuntimeError, match="teacher_store.sample_plan_sha256"):
        _run(formal_case, teacher_store, conditional_store)


def test_tampered_sample_plan_self_sha_rejects(
    formal_case: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    teacher_store = FakeTeacherStore(formal_case)
    conditional_store = FakeConditionalStore(formal_case)
    tampered_plan = copy.deepcopy(formal_case)
    tampered_plan["samples"]["validation"][0]["prompt_sha256"] = "0" * 64
    _patch_run_m4_validation(monkeypatch, [])

    with pytest.raises(RuntimeError, match="sample plan SHA256"):
        _run(tampered_plan, teacher_store, conditional_store)


def test_teacher_manifest_sha_mismatch_rejects(
    formal_case: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    teacher_store = FakeTeacherStore(formal_case)
    conditional_store = FakeConditionalStore(formal_case)
    conditional_store.teacher_manifest_sha256 = "0" * 64
    _patch_run_m4_validation(monkeypatch, [])

    with pytest.raises(RuntimeError, match="conditional_store.teacher_manifest_sha256"):
        _run(formal_case, teacher_store, conditional_store)


def test_validation_identity_order_mismatch_rejects(
    formal_case: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    teacher_store = FakeTeacherStore(formal_case)
    conditional_store = FakeConditionalStore(formal_case)
    teacher_store.validation_identities = tuple(reversed(teacher_store.validation_identities))
    _patch_run_m4_validation(monkeypatch, [])

    with pytest.raises(RuntimeError, match="teacher_store.validation_identities"):
        _run(formal_case, teacher_store, conditional_store)


def test_child_exception_cleans_store_lifecycle(
    formal_case: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    teacher_store = FakeTeacherStore(formal_case)
    conditional_store = FakeConditionalStore(formal_case)
    calls: list[dict[str, Any]] = []
    _patch_run_m4_validation(monkeypatch, calls, raise_at=17)

    with pytest.raises(RuntimeError, match="child validation exploded") as exc_info:
        _run(formal_case, teacher_store, conditional_store)

    assert len(calls) == 18
    assert teacher_store.live_sample_count == 0
    assert conditional_store.live_conditional_count == 0
    assert teacher_store.max_live_sample_count == 1
    assert conditional_store.max_live_conditional_count == 1
    wrapper_locals = _locals_for_frame(
        exc_info.value,
        "run_m5_streaming_validation",
    )
    assert wrapper_locals["sample"] is None
    assert wrapper_locals["cpu_conditional"] is None
    assert wrapper_locals["device_conditional"] is None
    assert wrapper_locals["child_samples"] == []
    assert wrapper_locals["child_conditionals"] == {}
    child_locals = _locals_for_frame(exc_info.value, "fake_run_m4_validation")
    assert child_locals["kwargs"]["samples"] == []
    assert child_locals["kwargs"]["conditional_dicts"] == {}


def test_child_fail_report_makes_final_report_fail(
    formal_case: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    teacher_store = FakeTeacherStore(formal_case)
    conditional_store = FakeConditionalStore(formal_case)
    calls: list[dict[str, Any]] = []
    _patch_run_m4_validation(monkeypatch, calls, fail_at=31)

    report = _run(formal_case, teacher_store, conditional_store)

    assert len(calls) == 256
    assert report["status"] == "FAIL"
    assert report["all_child_m4_contracts_pass"] is False
    assert report["validation_loss_finite_contract"] is True
    assert report["child_m4_failures"][0]["sample_position"] == 31
    assert teacher_store.live_sample_count == 0
    assert conditional_store.live_conditional_count == 0


def test_nonfinite_loss_makes_final_report_fail(
    formal_case: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    teacher_store = FakeTeacherStore(formal_case)
    conditional_store = FakeConditionalStore(formal_case)
    calls: list[dict[str, Any]] = []
    _patch_run_m4_validation(monkeypatch, calls, nonfinite_at=5)

    report = _run(formal_case, teacher_store, conditional_store)

    assert len(calls) == 256
    assert report["status"] == "FAIL"
    assert report["validation_loss_finite_contract"] is False
    assert report["aggregate_losses"]["main_loss"] is None
    assert report["nonfinite_validation_losses"] == [
        {
            "scope": "sample",
            "sample_identity": formal_case["validation_sample_identities"][5],
            "fields": ["main_loss"],
        }
    ]
    assert teacher_store.live_sample_count == 0
    assert conditional_store.live_conditional_count == 0
