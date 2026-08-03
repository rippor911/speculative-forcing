from __future__ import annotations

import copy
import hashlib
import json
import shutil
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import torch

import utils.nf_sf_m4 as m4
import utils.nf_sf_m5_formal_plan as formal_plan
from utils.nf_sf_m3 import M3_REFERENCE_CHECKPOINT_SHA256

TEST_GIT_SHA = "0123456789abcdef0123456789abcdef01234567"


@pytest.fixture
def work_tmp() -> Iterator[Path]:
    root = Path(".m5b2_formal_plan_tests")
    root.mkdir(exist_ok=True)
    case_dir = root / "case"
    if case_dir.exists():
        shutil.rmtree(case_dir)
    case_dir.mkdir()
    try:
        yield case_dir
    finally:
        if case_dir.exists():
            shutil.rmtree(case_dir)
        if root.exists() and not any(root.iterdir()):
            root.rmdir()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _formal_record(split: str, split_index: int, sample_index: int) -> dict[str, Any]:
    prompt_sha = _sha256_text(f"{split} prompt {split_index}")
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
        "prompt": f"{split} prompt {split_index}",
        "prompt_sha256": prompt_sha,
        "target_latent": {"shape": [1, 15, 1, 1, 1], "dtype": "torch.bfloat16"},
        "source_noise": {"shape": [1, 15, 1, 1, 1], "dtype": "torch.bfloat16"},
    }


def _formal_records(
    *,
    train_count: int = formal_plan.M5_FORMAL_TRAIN_SAMPLE_COUNT,
    validation_count: int = formal_plan.M5_FORMAL_VALIDATION_SAMPLE_COUNT,
) -> list[dict[str, Any]]:
    records = []
    for split_index in range(train_count):
        records.append(_formal_record("train", split_index, split_index))
    for split_index in range(validation_count):
        records.append(
            _formal_record("validation", split_index, 100_000 + split_index)
        )
    return records


def _write_manifest(
    tmp_path: Path,
    *,
    train_count: int = formal_plan.M5_FORMAL_TRAIN_SAMPLE_COUNT,
    validation_count: int = formal_plan.M5_FORMAL_VALIDATION_SAMPLE_COUNT,
    records: list[dict[str, Any]] | None = None,
    name: str = "manifest.json",
) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    samples = (
        list(reversed(_formal_records(
            train_count=train_count,
            validation_count=validation_count,
        )))
        if records is None
        else list(records)
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
            "num_samples": len(samples),
            "num_completed": len(samples),
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
        "samples": samples,
    }
    path = tmp_path / name
    path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    return path


def _build_case(tmp_path: Path) -> tuple[Path, Path, dict[str, Any]]:
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    manifest = _write_manifest(tmp_path)
    plan = formal_plan.build_m5_formal_sample_plan(
        manifest_path=manifest,
        dataset_root=dataset_root,
    )
    return manifest, dataset_root, plan


def _refresh_plan_sha(plan: dict[str, Any]) -> dict[str, Any]:
    plan["sample_plan_sha256"] = m4.m4_sample_plan_sha256(plan)
    return plan


def _refresh_identities(plan: dict[str, Any]) -> None:
    for split in ("train", "validation"):
        identities = [entry["identity"] for entry in plan["samples"][split]]
        plan[f"{split}_sample_identities"] = identities
    plan["fixed_decode_validation_identity"] = plan["validation_sample_identities"][0]


def test_builds_exact_formal_plan_and_m4_loader_can_read_it(work_tmp: Path) -> None:
    manifest, dataset_root, plan = _build_case(work_tmp)
    assert plan["schema"] == m4.M4_SAMPLE_PLAN_SCHEMA
    assert len(plan["samples"]["train"]) == 2048
    assert len(plan["samples"]["validation"]) == 256
    assert plan["train_subset_size"] == 2048
    assert plan["validation_subset_size"] == 256
    assert plan["fixed_decode_validation_identity"] == plan["validation_sample_identities"][0]

    output = work_tmp / "formal_plan.json"
    formal_plan.write_m5_formal_sample_plan(plan, output)
    loaded = m4.load_m4_sample_plan(output)
    assert loaded["sample_plan_sha256"] == plan["sample_plan_sha256"]
    audit = formal_plan.validate_m5_formal_sample_plan(
        loaded,
        manifest_path=manifest,
        dataset_root=dataset_root,
    )
    assert audit == {
        "schema": formal_plan.M5_FORMAL_SAMPLE_PLAN_AUDIT_SCHEMA,
        "status": "PASS",
        "sample_plan_schema": m4.M4_SAMPLE_PLAN_SCHEMA,
        "sample_plan_sha256": plan["sample_plan_sha256"],
        "manifest_sha256": plan["manifest_sha256"],
        "train_sample_count": 2048,
        "validation_sample_count": 256,
        "fixed_decode_validation_identity": plan["fixed_decode_validation_identity"],
        "ordering_rule": m4.M4_SAMPLE_ORDERING_RULE,
        "all_entries_have_payload_location": True,
        "all_entries_have_file_sha256": True,
        "all_entries_have_prompt_sha256": True,
    }


def test_repeated_build_is_deterministic_and_output_path_not_hashed(
    work_tmp: Path,
) -> None:
    manifest, dataset_root, first = _build_case(work_tmp)
    second = formal_plan.build_m5_formal_sample_plan(
        manifest_path=manifest,
        dataset_root=str(dataset_root),
    )
    assert second == first
    assert second["sample_plan_sha256"] == first["sample_plan_sha256"]

    output_a = work_tmp / "a.json"
    output_b = work_tmp / "b.json"
    formal_plan.write_m5_formal_sample_plan(first, output_a)
    formal_plan.write_m5_formal_sample_plan(second, output_b)
    assert m4.load_m4_sample_plan(output_a)["sample_plan_sha256"] == m4.load_m4_sample_plan(
        output_b
    )["sample_plan_sha256"]


def test_shuffled_manifest_records_keep_stable_identity_order(work_tmp: Path) -> None:
    records = _formal_records()
    dataset_root = work_tmp / "dataset"
    dataset_root.mkdir()
    ordered = _write_manifest(work_tmp / "ordered", records=records)
    shuffled = _write_manifest(work_tmp / "shuffled", records=list(reversed(records)))
    ordered_plan = formal_plan.build_m5_formal_sample_plan(
        manifest_path=ordered,
        dataset_root=dataset_root,
    )
    shuffled_plan = formal_plan.build_m5_formal_sample_plan(
        manifest_path=shuffled,
        dataset_root=dataset_root,
    )
    assert shuffled_plan["manifest_sha256"] != ordered_plan["manifest_sha256"]
    assert shuffled_plan["sample_plan_sha256"] != ordered_plan["sample_plan_sha256"]
    assert shuffled_plan["train_sample_identities"] == ordered_plan[
        "train_sample_identities"
    ]
    assert shuffled_plan["validation_sample_identities"] == ordered_plan[
        "validation_sample_identities"
    ]


def test_path_and_string_inputs_are_supported(work_tmp: Path) -> None:
    manifest, dataset_root, path_plan = _build_case(work_tmp)
    string_plan = formal_plan.build_m5_formal_sample_plan(
        manifest_path=str(manifest),
        dataset_root=str(dataset_root),
    )
    assert string_plan == path_plan


def test_cli_builds_plan_and_prints_pass_marker(work_tmp: Path) -> None:
    manifest, dataset_root, _plan = _build_case(work_tmp)
    output = work_tmp / "cli_plan.json"
    script = Path("scripts/build_nf_sf_m5_formal_sample_plan.py")
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--manifest",
            str(manifest),
            "--dataset_root",
            str(dataset_root),
            "--output",
            str(output),
        ],
        cwd=Path(__file__).resolve().parents[2],
        check=True,
        capture_output=True,
        text=True,
    )
    assert output.exists()
    assert "M5_FORMAL_SAMPLE_PLAN_BUILD_PASS" in result.stdout
    audit_text = result.stdout.split("M5_FORMAL_SAMPLE_PLAN_BUILD_PASS", 1)[0]
    assert json.loads(audit_text)["status"] == "PASS"


@pytest.mark.parametrize(
    ("train_count", "validation_count", "message"),
    [
        (2047, 256, "train_sample_count"),
        (2049, 256, "train_sample_count"),
        (2048, 255, "validation_sample_count"),
        (2048, 257, "validation_sample_count"),
    ],
)
def test_build_rejects_non_exact_split_counts(
    work_tmp: Path,
    train_count: int,
    validation_count: int,
    message: str,
) -> None:
    dataset_root = work_tmp / "dataset"
    dataset_root.mkdir()
    manifest = _write_manifest(
        work_tmp,
        train_count=train_count,
        validation_count=validation_count,
    )
    with pytest.raises(RuntimeError, match=message):
        formal_plan.build_m5_formal_sample_plan(
            manifest_path=manifest,
            dataset_root=dataset_root,
        )


def test_validate_rejects_m4_truncated_plan_from_extra_train_manifest(
    work_tmp: Path,
) -> None:
    dataset_root = work_tmp / "dataset"
    dataset_root.mkdir()
    manifest = _write_manifest(work_tmp, train_count=2049, validation_count=256)
    plan = m4.build_m4_sample_plan(
        manifest_path=manifest,
        train_subset_size=2048,
        validation_subset_size=256,
        dataset_root=dataset_root,
    )
    assert len(plan["samples"]["train"]) == 2048
    assert len(plan["samples"]["validation"]) == 256

    with pytest.raises(RuntimeError, match="train_sample_count"):
        formal_plan.validate_m5_formal_sample_plan(
            plan,
            manifest_path=manifest,
            dataset_root=dataset_root,
        )


def test_validate_rejects_m4_truncated_plan_from_extra_validation_manifest(
    work_tmp: Path,
) -> None:
    dataset_root = work_tmp / "dataset"
    dataset_root.mkdir()
    manifest = _write_manifest(work_tmp, train_count=2048, validation_count=257)
    plan = m4.build_m4_sample_plan(
        manifest_path=manifest,
        train_subset_size=2048,
        validation_subset_size=256,
        dataset_root=dataset_root,
    )
    assert len(plan["samples"]["train"]) == 2048
    assert len(plan["samples"]["validation"]) == 256

    with pytest.raises(RuntimeError, match="validation_sample_count"):
        formal_plan.validate_m5_formal_sample_plan(
            plan,
            manifest_path=manifest,
            dataset_root=dataset_root,
        )


@pytest.mark.parametrize(
    ("train_count", "validation_count", "message"),
    [
        (2047, 256, "requested 2048 train samples"),
        (2048, 255, "requested 256 validation samples"),
    ],
)
def test_short_manifest_fails_before_a_formal_surface_plan_exists(
    work_tmp: Path,
    train_count: int,
    validation_count: int,
    message: str,
) -> None:
    dataset_root = work_tmp / "dataset"
    dataset_root.mkdir()
    manifest = _write_manifest(
        work_tmp,
        train_count=train_count,
        validation_count=validation_count,
    )

    with pytest.raises(RuntimeError, match=message):
        m4.build_m4_sample_plan(
            manifest_path=manifest,
            train_subset_size=2048,
            validation_subset_size=256,
            dataset_root=dataset_root,
        )


def test_build_rejects_bad_manifest_path(work_tmp: Path) -> None:
    dataset_root = work_tmp / "dataset"
    dataset_root.mkdir()
    with pytest.raises(FileNotFoundError, match="manifest_path"):
        formal_plan.build_m5_formal_sample_plan(
            manifest_path=work_tmp / "missing.json",
            dataset_root=dataset_root,
        )
    with pytest.raises(ValueError, match="regular file"):
        formal_plan.build_m5_formal_sample_plan(
            manifest_path=dataset_root,
            dataset_root=dataset_root,
        )
    tmp_manifest = _write_manifest(work_tmp, name="manifest.json.tmp")
    with pytest.raises(ValueError, match="manifest_path"):
        formal_plan.build_m5_formal_sample_plan(
            manifest_path=tmp_manifest,
            dataset_root=dataset_root,
        )


def test_build_rejects_bad_dataset_root(work_tmp: Path) -> None:
    manifest = _write_manifest(work_tmp)
    with pytest.raises(FileNotFoundError, match="dataset_root"):
        formal_plan.build_m5_formal_sample_plan(
            manifest_path=manifest,
            dataset_root=work_tmp / "missing-dataset",
        )
    not_dir = work_tmp / "not-dir"
    not_dir.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="dataset_root"):
        formal_plan.build_m5_formal_sample_plan(
            manifest_path=manifest,
            dataset_root=not_dir,
        )


def test_write_rejects_bad_output_paths_and_leaves_no_tmp(work_tmp: Path) -> None:
    _manifest, _dataset_root, plan = _build_case(work_tmp)
    existing = work_tmp / "existing.json"
    existing.write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError, match="output_path"):
        formal_plan.write_m5_formal_sample_plan(plan, existing)

    tmp_output = work_tmp / "formal.json.tmp"
    with pytest.raises(ValueError, match="output_path"):
        formal_plan.write_m5_formal_sample_plan(plan, tmp_output)

    missing_parent = work_tmp / "missing-parent" / "formal.json"
    with pytest.raises(FileNotFoundError, match="parent"):
        formal_plan.write_m5_formal_sample_plan(plan, missing_parent)

    assert not list(work_tmp.glob("*.tmp"))
    assert not list(work_tmp.glob(".*.tmp"))


def test_validate_rejects_duplicate_identity(work_tmp: Path) -> None:
    _manifest, _dataset_root, plan = _build_case(work_tmp)
    tampered = copy.deepcopy(plan)
    tampered["samples"]["train"][1] = copy.deepcopy(tampered["samples"]["train"][0])
    _refresh_identities(tampered)
    _refresh_plan_sha(tampered)
    with pytest.raises(RuntimeError, match="duplicates"):
        formal_plan.validate_m5_formal_sample_plan(tampered)


def test_validate_rejects_train_validation_sample_index_overlap(
    work_tmp: Path,
) -> None:
    _manifest, _dataset_root, plan = _build_case(work_tmp)
    tampered = copy.deepcopy(plan)
    tampered["samples"]["validation"][0]["sample_index"] = tampered["samples"]["train"][0][
        "sample_index"
    ]
    tampered["samples"]["validation"][0]["identity"] = m4.m4_sample_identity_from_record(
        tampered["samples"]["validation"][0]
    )
    _refresh_identities(tampered)
    _refresh_plan_sha(tampered)
    with pytest.raises(RuntimeError, match="overlap"):
        formal_plan.validate_m5_formal_sample_plan(tampered)


def test_validate_rejects_missing_prompt_sha256(work_tmp: Path) -> None:
    _manifest, _dataset_root, plan = _build_case(work_tmp)
    tampered = copy.deepcopy(plan)
    tampered["samples"]["train"][0].pop("prompt_sha256")
    _refresh_plan_sha(tampered)
    with pytest.raises(RuntimeError, match="prompt_sha256"):
        formal_plan.validate_m5_formal_sample_plan(tampered)


def test_validate_rejects_bad_prompt_sha256_format(work_tmp: Path) -> None:
    _manifest, _dataset_root, plan = _build_case(work_tmp)
    tampered = copy.deepcopy(plan)
    entry = tampered["samples"]["train"][0]
    entry["prompt_sha256"] = "A" * 64
    entry["identity"] = m4.m4_sample_identity_from_record(entry)
    _refresh_identities(tampered)
    _refresh_plan_sha(tampered)
    with pytest.raises(RuntimeError, match="prompt_sha256"):
        formal_plan.validate_m5_formal_sample_plan(tampered)


def test_validate_rejects_missing_file_sha256(work_tmp: Path) -> None:
    _manifest, _dataset_root, plan = _build_case(work_tmp)
    tampered = copy.deepcopy(plan)
    tampered["samples"]["train"][0].pop("file_sha256")
    _refresh_plan_sha(tampered)
    with pytest.raises(RuntimeError, match="file_sha256"):
        formal_plan.validate_m5_formal_sample_plan(tampered)


def test_validate_rejects_bad_file_sha256_format(work_tmp: Path) -> None:
    _manifest, _dataset_root, plan = _build_case(work_tmp)
    tampered = copy.deepcopy(plan)
    tampered["samples"]["train"][0]["file_sha256"] = "not-a-sha"
    _refresh_plan_sha(tampered)
    with pytest.raises(RuntimeError, match="file_sha256"):
        formal_plan.validate_m5_formal_sample_plan(tampered)


def test_validate_rejects_missing_payload_location(work_tmp: Path) -> None:
    _manifest, _dataset_root, plan = _build_case(work_tmp)
    tampered = copy.deepcopy(plan)
    entry = tampered["samples"]["train"][0]
    for key in ("file", "path", "payload_path", "artifact_path"):
        entry.pop(key, None)
    _refresh_plan_sha(tampered)
    with pytest.raises(RuntimeError, match="payload location"):
        formal_plan.validate_m5_formal_sample_plan(tampered)


def test_validate_rejects_empty_payload_location(work_tmp: Path) -> None:
    _manifest, _dataset_root, plan = _build_case(work_tmp)
    tampered = copy.deepcopy(plan)
    tampered["samples"]["train"][0]["file"] = ""
    _refresh_plan_sha(tampered)
    with pytest.raises(RuntimeError, match="file"):
        formal_plan.validate_m5_formal_sample_plan(tampered)


def test_validate_rejects_entry_split_mismatch(work_tmp: Path) -> None:
    _manifest, _dataset_root, plan = _build_case(work_tmp)
    tampered = copy.deepcopy(plan)
    tampered["samples"]["train"][0]["split"] = "validation"
    tampered["samples"]["train"][0]["identity"] = m4.m4_sample_identity_from_record(
        tampered["samples"]["train"][0]
    )
    _refresh_identities(tampered)
    _refresh_plan_sha(tampered)
    with pytest.raises(RuntimeError, match="split"):
        formal_plan.validate_m5_formal_sample_plan(tampered)


def test_validate_rejects_entry_identity_mismatch(work_tmp: Path) -> None:
    _manifest, _dataset_root, plan = _build_case(work_tmp)
    tampered = copy.deepcopy(plan)
    tampered["samples"]["train"][0]["identity"] = "wrong"
    _refresh_identities(tampered)
    _refresh_plan_sha(tampered)
    with pytest.raises(RuntimeError, match="identity"):
        formal_plan.validate_m5_formal_sample_plan(tampered)


def test_validate_rejects_plan_entry_order_tampering(work_tmp: Path) -> None:
    _manifest, _dataset_root, plan = _build_case(work_tmp)
    tampered = copy.deepcopy(plan)
    train_entries = tampered["samples"]["train"]
    train_entries[0], train_entries[1] = train_entries[1], train_entries[0]
    _refresh_identities(tampered)
    _refresh_plan_sha(tampered)
    with pytest.raises(RuntimeError, match="ordering"):
        formal_plan.validate_m5_formal_sample_plan(tampered)


def test_validate_rejects_plan_sha_tampering(work_tmp: Path) -> None:
    _manifest, _dataset_root, plan = _build_case(work_tmp)
    tampered = copy.deepcopy(plan)
    tampered["sample_plan_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="SHA256"):
        formal_plan.validate_m5_formal_sample_plan(tampered)


def test_validate_rejects_manifest_sha_tampering(work_tmp: Path) -> None:
    manifest, _dataset_root, plan = _build_case(work_tmp)
    tampered = copy.deepcopy(plan)
    tampered["manifest_sha256"] = "0" * 64
    _refresh_plan_sha(tampered)
    with pytest.raises(RuntimeError, match="manifest SHA256"):
        formal_plan.validate_m5_formal_sample_plan(
            tampered,
            manifest_path=manifest,
        )


def test_validate_rejects_dataset_root_tampering(work_tmp: Path) -> None:
    _manifest, dataset_root, plan = _build_case(work_tmp)
    other_root = work_tmp / "other-dataset"
    other_root.mkdir()
    tampered = copy.deepcopy(plan)
    tampered["dataset_root"] = str(other_root.resolve())
    _refresh_plan_sha(tampered)
    with pytest.raises(RuntimeError, match="dataset_root"):
        formal_plan.validate_m5_formal_sample_plan(
            tampered,
            dataset_root=dataset_root,
        )


def test_validate_rejects_fixed_decode_tampering(work_tmp: Path) -> None:
    _manifest, _dataset_root, plan = _build_case(work_tmp)
    tampered = copy.deepcopy(plan)
    tampered["fixed_decode_validation_identity"] = tampered[
        "validation_sample_identities"
    ][1]
    _refresh_plan_sha(tampered)
    with pytest.raises(RuntimeError, match="fixed decode"):
        formal_plan.validate_m5_formal_sample_plan(tampered)


def test_expected_sha256_is_strict(work_tmp: Path) -> None:
    _manifest, _dataset_root, plan = _build_case(work_tmp)
    with pytest.raises(RuntimeError, match="expected"):
        formal_plan.validate_m5_formal_sample_plan(plan, expected_sha256="0" * 64)


def test_build_does_not_call_torch_load_or_cuda(work_tmp: Path, monkeypatch) -> None:
    dataset_root = work_tmp / "dataset"
    dataset_root.mkdir()
    manifest = _write_manifest(work_tmp)

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("forbidden runtime payload or CUDA access")

    monkeypatch.setattr(torch, "load", forbidden)
    monkeypatch.setattr(torch.cuda, "is_available", forbidden)
    monkeypatch.setattr(torch.cuda, "device_count", forbidden)
    monkeypatch.setattr(torch.cuda, "current_device", forbidden)

    plan = formal_plan.build_m5_formal_sample_plan(
        manifest_path=manifest,
        dataset_root=dataset_root,
    )
    assert plan["sample_plan_sha256"]


def test_build_does_not_open_teacher_payload_files(work_tmp: Path) -> None:
    dataset_root = work_tmp / "dataset"
    dataset_root.mkdir()
    records = _formal_records()
    for record in records:
        record["file"] = f"missing_payloads/{record['file']}"
    manifest = _write_manifest(work_tmp, records=records)
    plan = formal_plan.build_m5_formal_sample_plan(
        manifest_path=manifest,
        dataset_root=dataset_root,
    )
    assert plan["samples"]["train"][0]["file"].startswith("missing_payloads/")
