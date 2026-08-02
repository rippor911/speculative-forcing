from __future__ import annotations

import argparse
import copy
import json
import subprocess
from pathlib import Path

import pytest
import torch
import torch.nn as nn

import scripts.train_nf_sf_m3_overfit as train_m3
import scripts.run_nf_sf_m4_compare as m4_compare
import utils.nf_sf_m4 as m4
from utils.nf_sf_m3 import (
    M3_REFERENCE_CHECKPOINT_SHA256,
    M3TeacherSample,
    select_m3_selected_state,
    tensor_summary,
)
from utils.nf_sf_tensors import make_generator
from utils.scheduler import FlowMatchScheduler


TEST_GIT_SHA = "0123456789abcdef0123456789abcdef01234567"
PROMPT_SHA_A = "a" * 64
PROMPT_SHA_B = "b" * 64


@pytest.fixture
def work_tmp(tmp_path: Path) -> Path:
    return tmp_path


def test_work_tmp_fixture_uses_system_temp(work_tmp: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    resolved = work_tmp.resolve()
    assert repo_root.resolve() not in (resolved, *resolved.parents)
    assert not (repo_root / "outputs" / "nf_sf_m4_pytest_tmp").exists()
    outputs_dir = repo_root / "outputs"
    assert not outputs_dir.exists() or not list(outputs_dir.glob("m4_plan_only_smoke_*"))


def _m4_temp_files(path: Path) -> list[Path]:
    return sorted(path.parent.glob(f".{path.name}.*.tmp"))


def _probe_outputs(value: float = 0.0) -> dict[str, torch.Tensor]:
    tensor = torch.tensor([value, value + 1.0], dtype=torch.float32)
    return {
        "main_flow_pred": tensor,
        "mcp_depth1_flow_pred": tensor + 1.0,
        "mcp_depth2_flow_pred": tensor + 2.0,
        "mcp_depth3_flow_pred": tensor + 3.0,
    }


def test_m4_json_writer_is_strict_atomic_and_calls_fsync(
    work_tmp: Path,
    monkeypatch,
) -> None:
    target = work_tmp / "record.json"
    fsync_calls = []
    monkeypatch.setattr(m4.os, "fsync", lambda fd: fsync_calls.append(fd))

    assert m4.write_m4_json({"b": 2, "a": 1}, target) == target
    assert json.loads(target.read_text(encoding="utf-8")) == {"a": 1, "b": 2}
    assert fsync_calls
    assert not _m4_temp_files(target)

    m4.write_m4_json({"a": 3}, target)
    assert json.loads(target.read_text(encoding="utf-8")) == {"a": 3}


def test_m4_json_writer_rejects_nan_before_final_file(work_tmp: Path) -> None:
    target = work_tmp / "nan.json"
    with pytest.raises(ValueError):
        m4.write_m4_json({"bad": float("nan")}, target)
    assert not target.exists()
    assert not _m4_temp_files(target)


def test_m4_json_writer_preserves_target_when_replace_fails(
    work_tmp: Path,
    monkeypatch,
) -> None:
    target = work_tmp / "status.json"
    target.write_text('{"old": true}\n', encoding="utf-8")

    def fail_replace(src, dst):
        raise PermissionError("replace denied")

    monkeypatch.setattr(m4.os, "replace", fail_replace)
    with pytest.raises(PermissionError, match="replace denied"):
        m4.write_m4_json({"new": True}, target)

    assert target.read_text(encoding="utf-8") == '{"old": true}\n'
    assert not _m4_temp_files(target)


def test_m4_json_writer_serialization_failure_leaves_no_file(
    work_tmp: Path,
    monkeypatch,
) -> None:
    target = work_tmp / "bad.json"

    def fail_dumps(*args, **kwargs):
        raise TypeError("cannot serialize")

    monkeypatch.setattr(m4.json, "dumps", fail_dumps)
    with pytest.raises(TypeError, match="cannot serialize"):
        m4.write_m4_json({"value": object()}, target)
    assert not target.exists()
    assert not _m4_temp_files(target)


def test_m4_json_writer_has_no_nonatomic_env_fallback() -> None:
    env_name = "NF_SF_M4_" + "NONATOMIC_JSON_FALLBACK"
    source = Path("utils/nf_sf_m4.py").read_text(encoding="utf-8")
    tests = Path("tests/speculative/test_nf_sf_m4.py").read_text(encoding="utf-8")
    assert env_name not in source
    assert env_name not in tests


def test_m4_probe_report_uses_strict_writer_and_reads_json(
    work_tmp: Path,
    monkeypatch,
) -> None:
    calls = []
    real_writer = train_m3.write_m4_json

    def capture_writer(payload, path):
        calls.append((payload, Path(path)))
        return real_writer(payload, path)

    monkeypatch.setattr(train_m3, "write_m4_json", capture_writer)
    report = train_m3.write_probe_report(
        work_tmp,
        0,
        {"total_loss": 1.0, "main_loss": 0.5},
        _probe_outputs(),
        strict=True,
    )

    path = work_tmp / "probe_step000000.json"
    assert calls and calls[0][1] == path
    assert json.loads(path.read_text(encoding="utf-8"))["probe_losses"] == report[
        "probe_losses"
    ]


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf")])
def test_m4_probe_report_rejects_nonfinite_without_writing(
    work_tmp: Path,
    bad_value: float,
) -> None:
    target = work_tmp / "probe_step000001.json"
    with pytest.raises(RuntimeError, match="non-finite"):
        train_m3.write_probe_report(
            work_tmp,
            1,
            {"total_loss": bad_value},
            _probe_outputs(),
            strict=True,
        )
    assert not target.exists()


def test_m4_probe_report_nonfinite_preserves_existing_target(work_tmp: Path) -> None:
    target = work_tmp / "probe_step000002.json"
    target.write_text('{"old": true}\n', encoding="utf-8")

    with pytest.raises(RuntimeError, match="non-finite"):
        train_m3.write_probe_report(
            work_tmp,
            2,
            {"total_loss": float("nan")},
            _probe_outputs(),
            strict=True,
        )

    assert target.read_text(encoding="utf-8") == '{"old": true}\n'


def test_m3_probe_report_keeps_legacy_writer(
    work_tmp: Path,
    monkeypatch,
) -> None:
    calls = []

    def capture_legacy_writer(payload, path):
        calls.append((payload, Path(path)))

    def fail_m4_writer(*args, **kwargs):
        raise AssertionError("M3 probe path must not use M4 strict writer")

    monkeypatch.setattr(train_m3, "atomic_json_write", capture_legacy_writer)
    monkeypatch.setattr(train_m3, "write_m4_json", fail_m4_writer)
    train_m3.write_probe_report(
        work_tmp,
        3,
        {"total_loss": float("nan")},
        _probe_outputs(),
        strict=False,
    )

    assert calls and calls[0][1] == work_tmp / "probe_step000003.json"


def _scheduler(shift: float = 5.0) -> FlowMatchScheduler:
    scheduler = FlowMatchScheduler(shift=shift, sigma_min=0.0, extra_one_step=True)
    scheduler.set_timesteps(1000, training=True)
    return scheduler


def _manifest(tmp_path: Path, *, train_count: int, validation_count: int) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    records = []
    for split, count, offset in (
        ("train", train_count, 0),
        ("validation", validation_count, 100),
    ):
        for split_index in range(count):
            sample_index = offset + split_index
            records.append(
                {
                    "status": "GENERATED",
                    "sample_index": sample_index,
                    "sample_id": f"{split}-{split_index:03d}",
                    "split": split,
                    "split_index": split_index,
                    "source_line_index": sample_index,
                    "shard_id": 0,
                    "plan_index": sample_index,
                    "file": f"{split}_{split_index:06d}.pt",
                    "prompt": f"{split} prompt {split_index}",
                    "prompt_sha256": PROMPT_SHA_A
                    if split == "train"
                    else PROMPT_SHA_B,
                    "target_latent": {"shape": [1, 15, 1, 1, 1], "dtype": "torch.bfloat16"},
                    "source_noise": {"shape": [1, 15, 1, 1, 1], "dtype": "torch.bfloat16"},
                }
            )
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
            "num_samples": train_count + validation_count,
            "num_completed": train_count + validation_count,
            "num_train": train_count,
            "num_validation": validation_count,
            "num_reserve": 0,
            "num_frames": 15,
            "num_frame_per_block": 3,
            "num_blocks": 5,
            "mcp_depth": 3,
            "mcp_num_modules": 0,
            "mcp_accel_depths": 0,
            "last_step_only": True,
        },
        "samples": list(reversed(records)),
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def _plan(tmp_path: Path, *, train_count: int = 16, validation_count: int = 8) -> dict:
    return m4.build_m4_sample_plan(
        manifest_path=_manifest(
            tmp_path,
            train_count=train_count,
            validation_count=validation_count,
        ),
        train_subset_size=train_count,
        validation_subset_size=validation_count,
    )


def _identity(split: str, split_index: int, sample_index: int) -> str:
    prompt_sha = PROMPT_SHA_A if split == "train" else PROMPT_SHA_B
    return (
        f"sample_index={sample_index}|sample_id={split}-{split_index:03d}|"
        f"split={split}|split_index={split_index}|prompt_sha256={prompt_sha}"
    )


def test_sample_plan_selects_16_train_and_8_validation_deterministically(work_tmp) -> None:
    manifest = _manifest(work_tmp, train_count=20, validation_count=10)
    first = m4.build_m4_sample_plan(
        manifest_path=manifest,
        train_subset_size=16,
        validation_subset_size=8,
    )
    shuffled_manifest = json.loads(manifest.read_text(encoding="utf-8"))
    shuffled_manifest["samples"] = list(reversed(shuffled_manifest["samples"]))
    manifest.write_text(json.dumps(shuffled_manifest), encoding="utf-8")
    second = m4.build_m4_sample_plan(
        manifest_path=manifest,
        train_subset_size=16,
        validation_subset_size=8,
    )
    assert first["train_sample_identities"] == second["train_sample_identities"]
    assert first["validation_sample_identities"] == second["validation_sample_identities"]
    assert first["train_sample_identities"][0] == _identity("train", 0, 0)
    assert first["validation_sample_identities"][0] == _identity("validation", 0, 100)
    assert first["fixed_decode_validation_identity"] == first["validation_sample_identities"][0]


def test_sample_plan_rejects_overlap_shortage_duplicate_and_tampering(work_tmp) -> None:
    manifest = _manifest(work_tmp, train_count=1, validation_count=1)
    with pytest.raises(RuntimeError, match="requested 2 train samples"):
        m4.build_m4_sample_plan(
            manifest_path=manifest,
            train_subset_size=2,
            validation_subset_size=1,
        )

    duplicate = json.loads(manifest.read_text(encoding="utf-8"))
    duplicate["samples"][1]["sample_index"] = duplicate["samples"][0]["sample_index"]
    duplicate["samples"][1]["split_index"] = duplicate["samples"][0]["split_index"]
    manifest.write_text(json.dumps(duplicate), encoding="utf-8")
    with pytest.raises(RuntimeError, match="duplicate"):
        m4.build_m4_sample_plan(
            manifest_path=manifest,
            train_subset_size=1,
            validation_subset_size=1,
        )

    plan = _plan(work_tmp / "clean", train_count=2, validation_count=2)
    assert m4.validate_m4_sample_plan(plan)["status"] == "PASS"
    tampered = copy.deepcopy(plan)
    tampered["train_sample_identities"][0] = "tampered"
    with pytest.raises(RuntimeError, match="identity list"):
        m4.validate_m4_sample_plan(tampered)
    tampered_sha = copy.deepcopy(plan)
    tampered_sha["sample_plan_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="SHA256"):
        m4.validate_m4_sample_plan(tampered_sha)


def test_round_robin_sequence_and_resume_are_deterministic(work_tmp) -> None:
    plan = _plan(work_tmp, train_count=2, validation_count=2)
    train_ids = plan["train_sample_identities"]
    assert m4.m4_train_identity_sequence(plan, 3) == [
        train_ids[0],
        train_ids[1],
        train_ids[0],
    ]
    assert m4.m4_next_train_entry_after_global_step(plan, 2)["identity"] == train_ids[0]
    assert m4.m4_train_identity_sequence(plan, 5) == m4.m4_train_identity_sequence(plan, 5)


class ValidationGenerator(nn.Module):
    def __init__(
        self,
        *,
        fail: bool = False,
        main_value: float = 0.0,
        mcp_values: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.0))
        self.fail = fail
        self.main_value = float(main_value)
        self.mcp_values = tuple(float(value) for value in mcp_values)

    def forward(
        self,
        *,
        noisy_image_or_video,
        conditional_dict,
        timestep,
        mcp_future_noises,
        mcp_future_start_frames,
        mcp_timesteps,
        **kwargs,
    ):
        if self.fail:
            raise RuntimeError("forced validation failure")
        main = torch.full_like(noisy_image_or_video, self.main_value) + self.weight * 0.0
        mcp = [
            torch.full_like(value, self.mcp_values[index]) + self.weight * 0.0
            for index, value in enumerate(mcp_future_noises)
        ]
        return main, noisy_image_or_video, mcp


def _validation_report(
    *,
    status: str = "PASS",
    step: int = 0,
    sample_identity: str = "validation-sample",
    failed_fields: list[str] | None = None,
) -> dict:
    aggregate = {key: 1.0 for key in m4.M4_VALIDATION_LOSS_KEYS}
    nonfinite = []
    if status != "PASS":
        failed_fields = failed_fields or ["main_loss", "total_validation_loss"]
        aggregate = {key: None for key in m4.M4_VALIDATION_LOSS_KEYS}
        nonfinite = [
            {
                "scope": "sample",
                "sample_identity": sample_identity,
                "fields": list(failed_fields),
            }
        ]
    return {
        "schema": m4.M4_VALIDATION_SCHEMA,
        "status": status,
        "global_step": int(step),
        "mode": "joint",
        "model_identity": {},
        "sample_plan_sha256": "a" * 64,
        "validation_seed_contract": {"base_seed": 1},
        "validation_sample_identities": [sample_identity],
        "sample_count": 1,
        "per_sample_losses": [
            {
                "sample_identity": sample_identity,
                "sample_position": 0,
                "validation_seed": 1,
                "losses": dict(aggregate),
                "tensor_contract": {},
            }
        ],
        "aggregate_losses": aggregate,
        "validation_loss_finite_contract": status == "PASS",
        "nonfinite_validation_loss_count": len(nonfinite),
        "nonfinite_validation_losses": nonfinite,
    }


def test_m4_validation_fail_gate_blocks_step0_optimizer_and_summary(
    work_tmp: Path,
) -> None:
    metrics_path = work_tmp / "metrics.jsonl"
    reports: list[dict] = []
    calls = {"optimizer_step": 0, "entered_loop": 0}
    report = _validation_report(
        status="FAIL",
        step=0,
        sample_identity="val-A",
        failed_fields=["main_loss"],
    )

    with pytest.raises(RuntimeError, match="step 0.*val-A.*main_loss"):
        train_m3.run_m4_validation_stage(
            global_step=0,
            run_validation=lambda: report,
            handle_report=lambda *, report, global_step: train_m3.handle_m4_validation_report(
                output_dir=work_tmp,
                metrics_path=metrics_path,
                validation_reports=reports,
                report=report,
                global_step=global_step,
            ),
            after_pass=lambda: calls.__setitem__("entered_loop", 1),
        )
        calls["optimizer_step"] += 1

    assert calls == {"optimizer_step": 0, "entered_loop": 0}
    assert reports == []
    assert (work_tmp / "validation_step000000.json").is_file()
    assert not (work_tmp / "training_summary.json").exists()
    assert '"validation_status": "FAIL"' in metrics_path.read_text(encoding="utf-8")


def test_m4_validation_fail_gate_blocks_in_loop_checkpoint(
    work_tmp: Path,
) -> None:
    metrics_path = work_tmp / "metrics.jsonl"
    reports: list[dict] = []
    calls = {
        "optimizer_step": 1,
        "probe": 0,
        "checkpoint": 0,
        "next_optimizer_step": 0,
        "pass_summary": 0,
    }
    report = _validation_report(
        status="FAIL",
        step=3,
        sample_identity="val-B",
        failed_fields=["mcp_depth1_loss"],
    )

    def after_pass() -> None:
        calls["probe"] += 1
        calls["checkpoint"] += 1
        calls["next_optimizer_step"] += 1
        calls["pass_summary"] += 1

    with pytest.raises(RuntimeError, match="step 3.*val-B.*mcp_depth1_loss"):
        train_m3.run_m4_validation_stage(
            global_step=3,
            run_validation=lambda: report,
            handle_report=lambda *, report, global_step: train_m3.handle_m4_validation_report(
                output_dir=work_tmp,
                metrics_path=metrics_path,
                validation_reports=reports,
                report=report,
                global_step=global_step,
            ),
            after_pass=after_pass,
        )

    assert calls == {
        "optimizer_step": 1,
        "probe": 0,
        "checkpoint": 0,
        "next_optimizer_step": 0,
        "pass_summary": 0,
    }
    assert reports == []
    assert (work_tmp / "validation_step000003.json").is_file()
    assert not (work_tmp / "checkpoint_step000003.pt").exists()
    assert not (work_tmp / "training_summary.json").exists()


def test_m4_validation_pass_gate_appends_report(work_tmp: Path) -> None:
    metrics_path = work_tmp / "metrics.jsonl"
    reports: list[dict] = []
    calls = {"probe": 0, "checkpoint": 0}
    report = _validation_report(status="PASS", step=2, sample_identity="val-C")

    train_m3.run_m4_validation_stage(
        global_step=2,
        run_validation=lambda: report,
        handle_report=lambda *, report, global_step: train_m3.handle_m4_validation_report(
            output_dir=work_tmp,
            metrics_path=metrics_path,
            validation_reports=reports,
            report=report,
            global_step=global_step,
        ),
        after_pass=lambda: (
            calls.__setitem__("probe", calls["probe"] + 1),
            calls.__setitem__("checkpoint", calls["checkpoint"] + 1),
        ),
    )

    assert reports == [report]
    assert calls == {"probe": 1, "checkpoint": 1}
    assert (work_tmp / "validation_step000002.json").is_file()
    assert '"validation_status": "PASS"' in metrics_path.read_text(encoding="utf-8")


@pytest.mark.parametrize("status", [None, "FAIL", "UNKNOWN"])
def test_m4_validation_gate_rejects_missing_fail_and_unknown_status(status) -> None:
    report = _validation_report(status="FAIL")
    if status is None:
        report.pop("status")
    else:
        report["status"] = status
    with pytest.raises(RuntimeError, match="M4 validation contract failed"):
        train_m3.require_m4_validation_pass(report, global_step=4)


def _teacher_sample(split: str, split_index: int, sample_index: int) -> M3TeacherSample:
    target = (
        torch.arange(15, dtype=torch.float32).reshape(1, 15, 1, 1, 1)
        + float(sample_index)
    ).to(torch.bfloat16)
    source = (target.float() + 100.0).to(torch.bfloat16)
    state = select_m3_selected_state(target)
    metadata = {
        "sample_index": sample_index,
        "sample_id": f"{split}-{split_index:03d}",
        "split": split,
        "split_index": split_index,
        "prompt": f"{split} prompt {split_index}",
        "prompt_sha256": PROMPT_SHA_A if split == "train" else PROMPT_SHA_B,
        "target_latent": tensor_summary(target),
        "latent_file_sha256": "f" * 64,
    }
    payload = {
        "target_latent": target,
        "source_noise": source,
        "raw_denoising_steps": [1000, 750, 500, 250],
        "warped_denoising_steps": [1000.0, 750.0, 500.0, 250.0],
    }
    return M3TeacherSample(
        payload=payload,
        target_latent=target,
        source_noise=source,
        selected_state=state,
        metadata=metadata,
    )


def _validation_plan_for_samples(tmp_path: Path, samples: list[M3TeacherSample]) -> dict:
    manifest = _manifest(tmp_path, train_count=2, validation_count=len(samples))
    plan = m4.build_m4_sample_plan(
        manifest_path=manifest,
        train_subset_size=2,
        validation_subset_size=len(samples),
    )
    validation_entries = []
    for sample in samples:
        identity = m4.m4_sample_identity_from_metadata(sample.metadata)
        validation_entries.append(
            {
                "identity": identity,
                "sample_index": int(sample.metadata["sample_index"]),
                "sample_id": sample.metadata["sample_id"],
                "split": "validation",
                "split_index": int(sample.metadata["split_index"]),
                "prompt_sha256": sample.metadata["prompt_sha256"],
            }
        )
    plan["samples"]["validation"] = validation_entries
    plan["validation_sample_identities"] = [entry["identity"] for entry in validation_entries]
    plan["validation_subset_size"] = len(validation_entries)
    plan["fixed_decode_validation_identity"] = validation_entries[0]["identity"]
    plan["sample_plan_sha256"] = m4.m4_sample_plan_sha256(plan)
    m4.validate_m4_sample_plan(plan)
    return plan


def test_validation_contract_is_deterministic_json_safe_and_isolated(work_tmp) -> None:
    samples = [
        _teacher_sample("validation", 0, 100),
        _teacher_sample("validation", 1, 101),
    ]
    plan = _validation_plan_for_samples(work_tmp, samples)
    generator = ValidationGenerator()
    generator.train()
    generator.weight.grad = torch.ones_like(generator.weight)
    train_rng = make_generator(123, "cpu")
    probe_rng = make_generator(456, "cpu").get_state()
    conditionals = {
        m4.m4_sample_identity_from_metadata(sample.metadata): {}
        for sample in samples
    }
    report = m4.run_m4_validation(
        generator=generator,
        samples=samples,
        conditional_dicts=conditionals,
        scheduler_main=_scheduler(5.0),
        scheduler_mcp=_scheduler(10.0),
        device=torch.device("cpu"),
        dtype=torch.float32,
        mode="joint",
        global_step=3,
        sample_plan=plan,
        validation_seed=999,
        train_rng=train_rng,
        probe_rng_state=probe_rng,
        model_identity={"run": "test"},
    )
    assert report["status"] == "PASS"
    assert report["model_mode_before"] == "train"
    assert report["model_mode_after"] == "train"
    assert report["train_rng_before_digest"] == report["train_rng_after_digest"]
    assert report["probe_rng_before_digest"] == report["probe_rng_after_digest"]
    assert report["gradients_unchanged_contract"] is True
    assert report["requires_grad_unchanged_contract"] is True
    assert torch.equal(generator.weight.grad, torch.ones_like(generator.weight))
    json.dumps(report, allow_nan=False)

    repeat = m4.run_m4_validation(
        generator=generator,
        samples=samples,
        conditional_dicts=conditionals,
        scheduler_main=_scheduler(5.0),
        scheduler_mcp=_scheduler(10.0),
        device=torch.device("cpu"),
        dtype=torch.float32,
        mode="joint",
        global_step=3,
        sample_plan=plan,
        validation_seed=999,
        train_rng=train_rng,
        probe_rng_state=probe_rng,
        model_identity={"run": "test"},
    )
    assert repeat == report


def test_validation_uses_validation_split_and_aggregates_means(work_tmp) -> None:
    samples = [_teacher_sample("validation", 0, 100)]
    plan = _validation_plan_for_samples(work_tmp, samples)
    train_sample = _teacher_sample("train", 0, 0)
    with pytest.raises(RuntimeError, match="not from validation plan"):
        m4.run_m4_validation(
            generator=ValidationGenerator(),
            samples=[train_sample],
            conditional_dicts={m4.m4_sample_identity_from_metadata(train_sample.metadata): {}},
            scheduler_main=_scheduler(5.0),
            scheduler_mcp=_scheduler(10.0),
            device=torch.device("cpu"),
            dtype=torch.float32,
            mode="frozen",
            global_step=0,
            sample_plan=plan,
            validation_seed=1,
            train_rng=make_generator(1, "cpu"),
            probe_rng_state=make_generator(2, "cpu").get_state(),
            model_identity={},
        )
    report = m4.run_m4_validation(
        generator=ValidationGenerator(),
        samples=samples,
        conditional_dicts={m4.m4_sample_identity_from_metadata(samples[0].metadata): {}},
        scheduler_main=_scheduler(5.0),
        scheduler_mcp=_scheduler(10.0),
        device=torch.device("cpu"),
        dtype=torch.float32,
        mode="frozen",
        global_step=0,
        sample_plan=plan,
        validation_seed=1,
        train_rng=make_generator(1, "cpu"),
        probe_rng_state=make_generator(2, "cpu").get_state(),
        model_identity={},
    )
    only_losses = report["per_sample_losses"][0]["losses"]
    assert report["aggregate_losses"]["total_validation_loss"] == pytest.approx(
        only_losses["total_validation_loss"]
    )


@pytest.mark.parametrize(
    ("generator", "expected_fields"),
    [
        (ValidationGenerator(main_value=float("nan")), {"main_loss", "total_validation_loss"}),
        (
            ValidationGenerator(mcp_values=(float("inf"), 0.0, 0.0)),
            {"mcp_depth1_loss", "weighted_mcp_loss", "total_validation_loss"},
        ),
    ],
)
def test_validation_rejects_nonfinite_sample_losses(
    work_tmp,
    generator: ValidationGenerator,
    expected_fields: set[str],
) -> None:
    samples = [_teacher_sample("validation", 0, 100)]
    plan = _validation_plan_for_samples(work_tmp, samples)
    report = m4.run_m4_validation(
        generator=generator,
        samples=samples,
        conditional_dicts={m4.m4_sample_identity_from_metadata(samples[0].metadata): {}},
        scheduler_main=_scheduler(5.0),
        scheduler_mcp=_scheduler(10.0),
        device=torch.device("cpu"),
        dtype=torch.float32,
        mode="joint",
        global_step=0,
        sample_plan=plan,
        validation_seed=1,
        train_rng=make_generator(1, "cpu"),
        probe_rng_state=make_generator(2, "cpu").get_state(),
        model_identity={},
    )
    assert report["status"] == "FAIL"
    assert report["validation_loss_finite_contract"] is False
    sample_failure = next(
        item for item in report["nonfinite_validation_losses"] if item["scope"] == "sample"
    )
    assert sample_failure["sample_identity"] == plan["validation_sample_identities"][0]
    assert expected_fields.issubset(set(sample_failure["fields"]))
    json.dumps(report, allow_nan=False)


def test_validation_rejects_nonfinite_aggregate_loss() -> None:
    loss_record = {key: 1.0e308 for key in m4.M4_VALIDATION_LOSS_KEYS}
    aggregate = m4._aggregate_validation_loss_values([loss_record, loss_record])
    report = m4.validation_loss_finite_report(
        per_sample_losses=[
            {"sample_identity": "A", "losses": loss_record},
            {"sample_identity": "B", "losses": loss_record},
        ],
        aggregate_losses=aggregate,
    )
    assert report["contract_pass"] is False
    aggregate_failure = next(
        item for item in report["nonfinite_validation_losses"] if item["scope"] == "aggregate"
    )
    assert "total_validation_loss" in aggregate_failure["fields"]


def test_validation_exception_restores_model_state(work_tmp) -> None:
    samples = [_teacher_sample("validation", 0, 100)]
    plan = _validation_plan_for_samples(work_tmp, samples)
    generator = ValidationGenerator(fail=True)
    generator.train(False)
    generator.weight.requires_grad_(False)
    generator.weight.grad = None
    with pytest.raises(RuntimeError, match="forced validation failure"):
        m4.run_m4_validation(
            generator=generator,
            samples=samples,
            conditional_dicts={m4.m4_sample_identity_from_metadata(samples[0].metadata): {}},
            scheduler_main=_scheduler(5.0),
            scheduler_mcp=_scheduler(10.0),
            device=torch.device("cpu"),
            dtype=torch.float32,
            mode="joint",
            global_step=0,
            sample_plan=plan,
            validation_seed=1,
            train_rng=make_generator(1, "cpu"),
            probe_rng_state=make_generator(2, "cpu").get_state(),
            model_identity={},
        )
    assert generator.training is False
    assert generator.weight.requires_grad is False
    assert generator.weight.grad is None


def test_validation_seed_derivation_does_not_use_python_hash() -> None:
    first = m4.derive_m4_validation_seed(
        base_seed=1,
        sample_identity="sample-A",
        tensor_slot="slot",
    )
    second = m4.derive_m4_validation_seed(
        base_seed=1,
        sample_identity="sample-A",
        tensor_slot="slot",
    )
    assert first == second
    assert "hash(" not in Path("utils/nf_sf_m4.py").read_text(encoding="utf-8")


def _wrapper_args(work_tmp: Path, *, plan_only: bool = True) -> argparse.Namespace:
    work_tmp.mkdir(parents=True, exist_ok=True)
    config = work_tmp / "config.yaml"
    checkpoint = work_tmp / "self_forcing_dmd.pt"
    config.write_text("dummy: true\n", encoding="utf-8")
    checkpoint.write_bytes(b"checkpoint")
    return argparse.Namespace(
        config=config,
        checkpoint=checkpoint,
        manifest=_manifest(work_tmp, train_count=2, validation_count=2),
        dataset_root=None,
        output_dir=work_tmp / "pair",
        sample_plan=None,
        train_subset_size=2,
        validation_subset_size=2,
        optimizer_steps=3,
        train_seed=2026080101,
        probe_seed=2026080199,
        validation_seed=2026080188,
        timing_warmup_steps=0,
        validation_steps=None,
        checkpoint_steps=None,
        backbone_lr=1.0e-6,
        patch_embedding_lr=1.0e-6,
        mcp_lr=1.0e-5,
        weight_decay=0.01,
        mcp1_grid_aux_weight=1.0,
        dtype="bf16",
        device="cuda:0",
        log_interval=1,
        python="python",
        plan_only=plan_only,
    )


def _pair_plan_for_args(args: argparse.Namespace) -> dict:
    validation_steps, checkpoint_steps = m4_compare.validate_wrapper_args(args)
    sample_plan = m4.build_m4_sample_plan(
        manifest_path=args.manifest,
        train_subset_size=args.train_subset_size,
        validation_subset_size=args.validation_subset_size,
    )
    return m4_compare.build_pair_plan(
        args=args,
        sample_plan=sample_plan,
        sample_plan_path=args.output_dir / "m4_sample_plan.json",
        validation_steps=validation_steps,
        checkpoint_steps=checkpoint_steps,
        current_git_sha=TEST_GIT_SHA,
    )


def test_pair_wrapper_plan_only_writes_shared_plan_without_subprocess(
    work_tmp,
    monkeypatch,
) -> None:
    args = _wrapper_args(work_tmp, plan_only=True)
    monkeypatch.setattr(m4_compare, "git_head", lambda: TEST_GIT_SHA)

    def fail_run(*args, **kwargs):
        raise AssertionError("plan_only must not start subprocess")

    monkeypatch.setattr(subprocess, "run", fail_run)
    assert m4_compare.run_pair(args) == 0
    pair_plan = json.loads((args.output_dir / "m4_pair_plan.json").read_text(encoding="utf-8"))
    commands = json.loads((args.output_dir / "m4_pair_commands.json").read_text(encoding="utf-8"))
    status = json.loads((args.output_dir / "m4_pair_status.json").read_text(encoding="utf-8"))
    assert status["status"] == "PLAN_ONLY"
    assert commands["schema"] == m4.M4_PAIR_COMMANDS_SCHEMA
    assert m4.validate_m4_pair_contract(pair_plan)["status"] == "PASS"
    assert (args.output_dir / "m4_sample_plan.json").is_file()
    assert not (args.output_dir / "frozen").exists()
    assert not (args.output_dir / "joint").exists()
    assert not (args.output_dir / "m4_pair_status.json.tmp").exists()
    frozen_args = pair_plan["runs"]["frozen"]["argv"]
    joint_args = pair_plan["runs"]["joint"]["argv"]
    assert frozen_args[frozen_args.index("--m4_sample_plan") + 1] == joint_args[
        joint_args.index("--m4_sample_plan") + 1
    ]


def test_pair_contract_rejects_shared_argument_drift(work_tmp, monkeypatch) -> None:
    args = _wrapper_args(work_tmp, plan_only=True)
    monkeypatch.setattr(m4_compare, "git_head", lambda: TEST_GIT_SHA)
    pair_plan = _pair_plan_for_args(args)
    pair_plan["runs"]["joint"]["train_seed"] += 1
    with pytest.raises(RuntimeError, match="locked field differs"):
        m4.validate_m4_pair_contract(pair_plan)


@pytest.mark.parametrize(
    ("checkpoint_steps", "passes"),
    [
        ("0,3", True),
        ("0,2,3", True),
        ("0", False),
        ("3", False),
        ("0,2", False),
        ("-1,3", False),
        ("0,4", False),
    ],
)
def test_wrapper_checkpoint_steps_require_zero_and_final(
    work_tmp,
    checkpoint_steps: str,
    passes: bool,
) -> None:
    args = _wrapper_args(work_tmp, plan_only=True)
    args.checkpoint_steps = checkpoint_steps
    if passes:
        assert m4_compare.validate_wrapper_args(args)[1] == tuple(
            sorted(int(value) for value in checkpoint_steps.split(","))
        )
    else:
        with pytest.raises(ValueError):
            m4_compare.validate_wrapper_args(args)


def test_pair_contract_rejects_locked_field_and_argv_drift(work_tmp) -> None:
    args = _wrapper_args(work_tmp, plan_only=True)
    pair_plan = _pair_plan_for_args(args)
    assert m4.validate_m4_pair_contract(pair_plan)["status"] == "PASS"

    missing = copy.deepcopy(pair_plan)
    del missing["shared_arguments"]["python_executable"]
    with pytest.raises(RuntimeError, match="missing locked field"):
        m4.validate_m4_pair_contract(missing)

    python_changed = copy.deepcopy(pair_plan)
    python_changed["runs"]["joint"]["argv"][0] = "different-python"
    with pytest.raises(RuntimeError, match="argv differs"):
        m4.validate_m4_pair_contract(python_changed)

    script_changed = copy.deepcopy(pair_plan)
    script_changed["runs"]["joint"]["argv"][2] = "different-script.py"
    with pytest.raises(RuntimeError, match="argv differs"):
        m4.validate_m4_pair_contract(script_changed)

    extra_arg = copy.deepcopy(pair_plan)
    extra_arg["runs"]["joint"]["argv"].append("--extra")
    with pytest.raises(RuntimeError, match="argv differs"):
        m4.validate_m4_pair_contract(extra_arg)

    reordered = copy.deepcopy(pair_plan)
    argv = reordered["runs"]["joint"]["argv"]
    argv[3], argv[4] = argv[4], argv[3]
    with pytest.raises(RuntimeError, match="argv differs"):
        m4.validate_m4_pair_contract(reordered)

    cwd_changed = copy.deepcopy(pair_plan)
    cwd_changed["runs"]["joint"]["subprocess_cwd"] = str(work_tmp / "other")
    with pytest.raises(RuntimeError, match="locked field differs"):
        m4.validate_m4_pair_contract(cwd_changed)


def test_pair_contract_allows_only_mode_and_output_dir_argv_differences(work_tmp) -> None:
    args = _wrapper_args(work_tmp, plan_only=True)
    pair_plan = _pair_plan_for_args(args)
    frozen_args = pair_plan["runs"]["frozen"]["argv"]
    joint_args = pair_plan["runs"]["joint"]["argv"]
    differing_indices = [
        index
        for index, (left, right) in enumerate(zip(frozen_args, joint_args))
        if left != right
    ]
    assert {
        frozen_args[index - 1] for index in differing_indices
    } == {"--mode", "--output_dir"}
    assert m4.validate_m4_pair_contract(pair_plan)["status"] == "PASS"


def test_pair_wrapper_rejects_bad_existing_output_and_records_frozen_failure(
    work_tmp,
    monkeypatch,
) -> None:
    args = _wrapper_args(work_tmp, plan_only=True)
    args.output_dir.mkdir()
    with pytest.raises(FileExistsError):
        m4_compare.run_pair(args)

    args = _wrapper_args(work_tmp / "fail", plan_only=False)
    monkeypatch.setattr(m4_compare, "git_head", lambda: TEST_GIT_SHA)

    class Result:
        returncode = 7

    calls = []

    def fake_run(argv, *, shell, check, cwd):
        calls.append((argv, shell, check, cwd))
        return Result()

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert m4_compare.run_pair(args) == 7
    assert calls and calls[0][1] is False
    assert calls[0][3] == str(m4_compare.ROOT.resolve())
    status = json.loads((args.output_dir / "m4_pair_status.json").read_text(encoding="utf-8"))
    assert status["status"] == "FROZEN_FAILED"
    assert status["runs"]["frozen"]["exit_code"] == 7
    assert status["runs"]["joint"]["exit_code"] is None


def test_plan_only_reuse_allows_later_matching_formal_run(work_tmp, monkeypatch) -> None:
    args = _wrapper_args(work_tmp, plan_only=True)
    monkeypatch.setattr(m4_compare, "git_head", lambda: TEST_GIT_SHA)
    assert m4_compare.run_pair(args) == 0
    assert not (args.output_dir / "frozen").exists()
    assert not (args.output_dir / "joint").exists()
    assert m4_compare.run_pair(args) == 0

    run_args = _wrapper_args(work_tmp, plan_only=False)
    calls = []

    class Result:
        returncode = 0

    def fake_run(argv, *, shell, check, cwd):
        calls.append((argv, shell, check, cwd))
        status = json.loads((run_args.output_dir / "m4_pair_status.json").read_text())
        assert status["status"] == "RUNNING"
        return Result()

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert m4_compare.run_pair(run_args) == 0
    assert len(calls) == 2
    assert all(call[1] is False and call[3] == str(m4_compare.ROOT.resolve()) for call in calls)
    status = json.loads((run_args.output_dir / "m4_pair_status.json").read_text())
    assert status["status"] == "COMPLETED"


def test_plan_only_reuse_rejects_different_args_and_unknown_files(
    work_tmp,
    monkeypatch,
) -> None:
    args = _wrapper_args(work_tmp, plan_only=True)
    monkeypatch.setattr(m4_compare, "git_head", lambda: TEST_GIT_SHA)
    assert m4_compare.run_pair(args) == 0

    changed = _wrapper_args(work_tmp, plan_only=True)
    changed.train_seed += 1
    with pytest.raises(RuntimeError, match="pair plan differs"):
        m4_compare.run_pair(changed)

    (args.output_dir / "unexpected.txt").write_text("x", encoding="utf-8")
    formal = _wrapper_args(work_tmp, plan_only=False)
    with pytest.raises(FileExistsError, match="outside"):
        m4_compare.run_pair(formal)


def test_wrapper_status_for_joint_failure_interrupt_and_exception(
    work_tmp,
    monkeypatch,
) -> None:
    monkeypatch.setattr(m4_compare, "git_head", lambda: TEST_GIT_SHA)

    joint_fail_args = _wrapper_args(work_tmp / "joint_fail", plan_only=False)
    returncodes = iter([0, 9])

    class Result:
        def __init__(self, returncode: int) -> None:
            self.returncode = returncode

    monkeypatch.setattr(
        subprocess,
        "run",
        lambda argv, *, shell, check, cwd: Result(next(returncodes)),
    )
    assert m4_compare.run_pair(joint_fail_args) == 9
    status = json.loads((joint_fail_args.output_dir / "m4_pair_status.json").read_text())
    assert status["status"] == "JOINT_FAILED"
    assert status["runs"]["frozen"]["exit_code"] == 0
    assert status["runs"]["joint"]["exit_code"] == 9

    interrupt_args = _wrapper_args(work_tmp / "interrupt", plan_only=False)

    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt()

    monkeypatch.setattr(subprocess, "run", interrupt)
    assert m4_compare.run_pair(interrupt_args) == 130
    status = json.loads((interrupt_args.output_dir / "m4_pair_status.json").read_text())
    assert status["status"] == "INTERRUPTED"

    exception_args = _wrapper_args(work_tmp / "exception", plan_only=False)

    def explode(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(subprocess, "run", explode)
    with pytest.raises(RuntimeError, match="boom"):
        m4_compare.run_pair(exception_args)
    status = json.loads((exception_args.output_dir / "m4_pair_status.json").read_text())
    assert status["status"] == "FAILED"
    assert status["exception_type"] == "RuntimeError"
    assert not (exception_args.output_dir / "m4_pair_status.json.tmp").exists()


def _checkpoint_payload_for_plan(
    plan: dict,
    plan_path: Path,
    *,
    validation_seed: int = 2026080188,
    validation_steps: tuple[int, ...] = (0, 3),
) -> dict:
    return {
        "resolved_config": {
            "m4": {
                "sample_plan_path": str(plan_path.resolve()),
                "sample_plan_sha256": plan["sample_plan_sha256"],
                "train_sample_identities": list(plan["train_sample_identities"]),
                "validation_sample_identities": list(plan["validation_sample_identities"]),
                "validation_seed": int(validation_seed),
                "validation_steps": list(validation_steps),
                "fixed_decode_validation_identity": plan["fixed_decode_validation_identity"],
                "ordering_rule": m4.M4_SAMPLE_ORDERING_RULE,
            }
        }
    }


def test_decode_identity_and_checkpoint_sample_plan_contract(work_tmp) -> None:
    plan = _plan(work_tmp, train_count=2, validation_count=2)
    plan_path = work_tmp / "m4_sample_plan.json"
    m4.write_m4_sample_plan(plan, plan_path)
    valid_identity = plan["validation_sample_identities"][0]
    train_identity = plan["train_sample_identities"][0]
    assert m4.validate_m4_decode_identity(
        sample_plan=plan,
        identity=valid_identity,
    )["status"] == "PASS"
    with pytest.raises(RuntimeError, match="validation split"):
        m4.validate_m4_decode_identity(sample_plan=plan, identity=train_identity)

    payload = _checkpoint_payload_for_plan(plan, plan_path)
    assert m4.validate_m4_checkpoint_sample_plan(
        payload,
        plan,
        sample_plan_path=plan_path,
        expected_validation_seed=2026080188,
        expected_validation_steps=(0, 3),
    )["status"] == "PASS"
    assert m4.validate_m4_checkpoint_sample_plan({"resolved_config": {}}, plan)[
        "status"
    ] == "LEGACY_M3"
    bad_payload = {"resolved_config": {"m4": {"sample_plan_sha256": "0" * 64}}}
    with pytest.raises(RuntimeError, match="missing"):
        m4.validate_m4_checkpoint_sample_plan(bad_payload, plan)


@pytest.mark.parametrize("missing_field", m4.M4_CHECKPOINT_REQUIRED_FIELDS)
def test_checkpoint_metadata_required_fields_are_enforced(
    work_tmp,
    missing_field: str,
) -> None:
    plan = _plan(work_tmp, train_count=2, validation_count=2)
    plan_path = work_tmp / "m4_sample_plan.json"
    payload = _checkpoint_payload_for_plan(plan, plan_path)
    payload["resolved_config"]["m4"].pop(missing_field)
    with pytest.raises(RuntimeError, match="missing"):
        m4.validate_m4_checkpoint_sample_plan(
            payload,
            plan,
            sample_plan_path=plan_path,
        )


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (
            lambda payload, plan: payload["resolved_config"]["m4"][
                "train_sample_identities"
            ].reverse(),
            "train identities",
        ),
        (
            lambda payload, plan: payload["resolved_config"]["m4"][
                "validation_sample_identities"
            ].reverse(),
            "validation identities",
        ),
        (
            lambda payload, plan: payload["resolved_config"]["m4"].__setitem__(
                "validation_seed", 1
            ),
            "validation seed",
        ),
        (
            lambda payload, plan: payload["resolved_config"]["m4"].__setitem__(
                "validation_steps", [0, 2]
            ),
            "validation steps",
        ),
        (
            lambda payload, plan: payload["resolved_config"]["m4"].__setitem__(
                "fixed_decode_validation_identity", plan["validation_sample_identities"][1]
            ),
            "fixed decode",
        ),
        (
            lambda payload, plan: payload["resolved_config"]["m4"].__setitem__(
                "ordering_rule", "different"
            ),
            "ordering rule",
        ),
        (
            lambda payload, plan: payload["resolved_config"]["m4"].__setitem__(
                "sample_plan_sha256", "0" * 64
            ),
            "SHA256",
        ),
    ],
)
def test_checkpoint_metadata_mismatches_fail(work_tmp, mutator, message: str) -> None:
    plan = _plan(work_tmp, train_count=2, validation_count=2)
    plan_path = work_tmp / "m4_sample_plan.json"
    payload = _checkpoint_payload_for_plan(plan, plan_path)
    mutator(payload, plan)
    with pytest.raises(RuntimeError, match=message):
        m4.validate_m4_checkpoint_sample_plan(
            payload,
            plan,
            sample_plan_path=plan_path,
            expected_validation_seed=2026080188,
            expected_validation_steps=(0, 3),
        )


def test_checkpoint_metadata_sample_plan_path_mismatch_fails(work_tmp) -> None:
    plan = _plan(work_tmp, train_count=2, validation_count=2)
    plan_path = work_tmp / "m4_sample_plan.json"
    payload = _checkpoint_payload_for_plan(plan, plan_path)
    with pytest.raises(RuntimeError, match="path differs"):
        m4.validate_m4_checkpoint_sample_plan(
            payload,
            plan,
            sample_plan_path=work_tmp / "different_plan.json",
        )


def test_m3_decode_default_arguments_remain_unset(monkeypatch) -> None:
    import scripts.eval_nf_sf_m3_overfit as eval_m3

    monkeypatch.setattr(
        "sys.argv",
        [
            "eval",
            "--config",
            "config.yaml",
            "--m3_checkpoint",
            "checkpoint.pt",
            "--output_dir",
            "out",
        ],
    )
    args = eval_m3.parse_args()
    assert args.m4_sample_plan is None
    assert args.m4_decode_sample_identity is None


def test_decode_video_names_are_unchanged() -> None:
    source = Path("scripts/eval_nf_sf_m3_overfit.py").read_text(encoding="utf-8")
    for name in (
        "target_current.mp4",
        "target_next1.mp4",
        "initial_main.mp4",
        "final_main.mp4",
        "initial_mcp1.mp4",
        "final_mcp1.mp4",
    ):
        assert name in source
