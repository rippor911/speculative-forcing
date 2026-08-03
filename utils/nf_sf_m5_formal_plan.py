from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from os import PathLike
from pathlib import Path
from typing import Any

from utils.nf_sf_m4 import (
    M4_SAMPLE_ORDERING_RULE,
    M4_SAMPLE_PLAN_SCHEMA,
    build_m4_sample_plan,
    load_m4_sample_plan,
    load_teacher_manifest,
    m4_sample_identity_from_record,
    validate_m4_sample_plan,
    write_m4_json,
)

M5_FORMAL_TRAIN_SAMPLE_COUNT = 2048
M5_FORMAL_VALIDATION_SAMPLE_COUNT = 256
M5_FORMAL_SAMPLE_PLAN_AUDIT_SCHEMA = "nf_sf_m5_formal_sample_plan_audit_v1"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_PAYLOAD_LOCATION_FIELDS = ("file", "path", "payload_path", "artifact_path")


def build_m5_formal_sample_plan(
    *,
    manifest_path: Path | str,
    dataset_root: Path | str,
) -> dict[str, Any]:
    manifest_file = _require_existing_file(manifest_path, "manifest_path")
    root = _require_existing_directory(dataset_root, "dataset_root")
    manifest = load_teacher_manifest(manifest_file)
    _require_manifest_split_counts(manifest)

    plan = build_m4_sample_plan(
        manifest_path=manifest_file,
        train_subset_size=M5_FORMAL_TRAIN_SAMPLE_COUNT,
        validation_subset_size=M5_FORMAL_VALIDATION_SAMPLE_COUNT,
        dataset_root=root,
    )
    validate_m5_formal_sample_plan(
        plan,
        manifest_path=manifest_file,
        dataset_root=root,
    )
    return plan


def validate_m5_formal_sample_plan(
    plan: Mapping[str, Any],
    *,
    manifest_path: Path | str | None = None,
    dataset_root: Path | str | None = None,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    if not isinstance(plan, Mapping):
        raise TypeError(
            "plan must be a mapping, "
            f"actual={type(plan).__name__}"
        )

    manifest_file = (
        None
        if manifest_path is None
        else _require_existing_file(manifest_path, "manifest_path")
    )
    root = (
        None
        if dataset_root is None
        else _require_existing_directory(dataset_root, "dataset_root")
    )
    if manifest_file is not None:
        manifest = load_teacher_manifest(manifest_file)
        _require_manifest_split_counts(manifest)

    m4_report = validate_m4_sample_plan(
        plan,
        manifest_path=manifest_file,
        expected_sha256=expected_sha256,
    )

    if plan.get("schema") != M4_SAMPLE_PLAN_SCHEMA:
        raise RuntimeError("schema must remain nf_sf_m4_sample_plan_v1")
    if plan.get("ordering_rule") != M4_SAMPLE_ORDERING_RULE:
        raise RuntimeError("ordering_rule mismatch for formal sample plan")

    samples = _require_samples_mapping(plan)
    train_entries = _require_entry_sequence(samples, "train")
    validation_entries = _require_entry_sequence(samples, "validation")
    _require_exact_count(
        train_entries,
        M5_FORMAL_TRAIN_SAMPLE_COUNT,
        "samples.train",
    )
    _require_exact_count(
        validation_entries,
        M5_FORMAL_VALIDATION_SAMPLE_COUNT,
        "samples.validation",
    )
    _require_plan_count_field(
        plan,
        "train_subset_size",
        M5_FORMAL_TRAIN_SAMPLE_COUNT,
    )
    _require_plan_count_field(
        plan,
        "validation_subset_size",
        M5_FORMAL_VALIDATION_SAMPLE_COUNT,
    )

    _validate_entry_set(train_entries, split="train", field_path="samples.train")
    _validate_entry_set(
        validation_entries,
        split="validation",
        field_path="samples.validation",
    )
    _validate_stable_order(train_entries, "samples.train")
    _validate_stable_order(validation_entries, "samples.validation")
    _validate_no_sample_index_overlap(train_entries, validation_entries)

    fixed_decode = str(plan["fixed_decode_validation_identity"])
    expected_fixed_decode = str(validation_entries[0]["identity"])
    if fixed_decode != expected_fixed_decode:
        raise RuntimeError(
            "fixed_decode_validation_identity mismatch: "
            f"expected={expected_fixed_decode}, actual={fixed_decode}"
        )

    if root is not None:
        expected_root = str(root.resolve())
        actual_root = plan.get("dataset_root")
        if not isinstance(actual_root, str):
            raise RuntimeError("dataset_root must be a string in formal sample plan")
        if actual_root != expected_root:
            raise RuntimeError(
                "dataset_root mismatch: "
                f"expected={expected_root}, actual={actual_root}"
            )

    if manifest_file is not None:
        canonical_root = root
        if canonical_root is None and plan.get("dataset_root") is not None:
            canonical_root = Path(str(plan["dataset_root"]))
        canonical = build_m4_sample_plan(
            manifest_path=manifest_file,
            train_subset_size=M5_FORMAL_TRAIN_SAMPLE_COUNT,
            validation_subset_size=M5_FORMAL_VALIDATION_SAMPLE_COUNT,
            dataset_root=canonical_root,
        )
        _require_equal(
            plan["train_sample_identities"],
            canonical["train_sample_identities"],
            "train_sample_identities",
        )
        _require_equal(
            plan["validation_sample_identities"],
            canonical["validation_sample_identities"],
            "validation_sample_identities",
        )
        _require_equal(
            train_entries,
            canonical["samples"]["train"],
            "samples.train",
        )
        _require_equal(
            validation_entries,
            canonical["samples"]["validation"],
            "samples.validation",
        )

    actual_sha256 = str(m4_report["sample_plan_sha256"])
    return {
        "schema": M5_FORMAL_SAMPLE_PLAN_AUDIT_SCHEMA,
        "status": "PASS",
        "sample_plan_schema": str(plan["schema"]),
        "sample_plan_sha256": actual_sha256,
        "manifest_sha256": str(plan["manifest_sha256"]),
        "train_sample_count": len(train_entries),
        "validation_sample_count": len(validation_entries),
        "fixed_decode_validation_identity": fixed_decode,
        "ordering_rule": str(plan["ordering_rule"]),
        "all_entries_have_payload_location": True,
        "all_entries_have_file_sha256": True,
        "all_entries_have_prompt_sha256": True,
    }


def write_m5_formal_sample_plan(
    plan: Mapping[str, Any],
    output_path: Path | str,
) -> Path:
    validate_m5_formal_sample_plan(plan)
    path = _require_output_path(output_path)
    write_m4_json(plan, path)
    leftovers = sorted(path.parent.glob(f".{path.name}.*.tmp"))
    if leftovers:
        raise RuntimeError(
            "write_m5_formal_sample_plan left temporary files: "
            f"{[str(item) for item in leftovers]}"
        )
    loaded = load_m4_sample_plan(path, expected_sha256=str(plan["sample_plan_sha256"]))
    validate_m5_formal_sample_plan(
        loaded,
        expected_sha256=str(plan["sample_plan_sha256"]),
    )
    return path


def _require_pathlike(value: Path | str | PathLike[str], field_path: str) -> Path:
    if not isinstance(value, (str, PathLike)):
        raise TypeError(
            f"{field_path} must be a string or Path, "
            f"actual={type(value).__name__}"
        )
    text = str(value)
    if text.strip() == "":
        raise ValueError(f"{field_path} must be non-empty")
    return Path(value)


def _is_tmp_path(path: Path) -> bool:
    return path.name.lower().endswith(".tmp")


def _require_existing_file(value: Path | str, field_path: str) -> Path:
    path = _require_pathlike(value, field_path)
    if _is_tmp_path(path):
        raise ValueError(f"{field_path} must not end with .tmp")
    if not path.exists():
        raise FileNotFoundError(f"{field_path} does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"{field_path} must be a regular file: {path}")
    return path


def _require_existing_directory(value: Path | str, field_path: str) -> Path:
    path = _require_pathlike(value, field_path)
    if not path.exists():
        raise FileNotFoundError(f"{field_path} does not exist: {path}")
    if not path.is_dir():
        raise ValueError(f"{field_path} must be a directory: {path}")
    return path


def _require_output_path(value: Path | str) -> Path:
    path = _require_pathlike(value, "output_path")
    if _is_tmp_path(path):
        raise ValueError("output_path must not end with .tmp")
    if path.exists():
        raise FileExistsError(f"output_path already exists: {path}")
    if not path.parent.exists():
        raise FileNotFoundError(f"output_path parent does not exist: {path.parent}")
    if not path.parent.is_dir():
        raise ValueError(f"output_path parent must be a directory: {path.parent}")
    return path


def _require_manifest_split_counts(manifest: Mapping[str, Any]) -> None:
    samples = manifest["samples"]
    train_count = sum(1 for sample in samples if str(sample.get("split")) == "train")
    validation_count = sum(
        1 for sample in samples if str(sample.get("split")) == "validation"
    )
    if train_count != M5_FORMAL_TRAIN_SAMPLE_COUNT:
        raise RuntimeError(
            "train_sample_count mismatch: "
            f"expected={M5_FORMAL_TRAIN_SAMPLE_COUNT}, actual={train_count}"
        )
    if validation_count != M5_FORMAL_VALIDATION_SAMPLE_COUNT:
        raise RuntimeError(
            "validation_sample_count mismatch: "
            f"expected={M5_FORMAL_VALIDATION_SAMPLE_COUNT}, actual={validation_count}"
        )


def _require_samples_mapping(plan: Mapping[str, Any]) -> Mapping[str, Any]:
    samples = plan.get("samples")
    if not isinstance(samples, Mapping):
        raise TypeError("samples must be a mapping")
    return samples


def _require_entry_sequence(
    samples: Mapping[str, Any],
    split: str,
) -> list[Mapping[str, Any]]:
    entries = samples.get(split)
    if not isinstance(entries, list):
        raise TypeError(f"samples.{split} must be a list")
    for index, entry in enumerate(entries):
        if not isinstance(entry, Mapping):
            raise TypeError(f"samples.{split}[{index}] must be a mapping")
    return list(entries)


def _require_exact_count(
    entries: Sequence[Mapping[str, Any]],
    expected: int,
    field_path: str,
) -> None:
    actual = len(entries)
    if actual != expected:
        raise RuntimeError(
            f"{field_path} count mismatch: expected={expected}, actual={actual}"
        )


def _require_plan_count_field(
    plan: Mapping[str, Any],
    field_path: str,
    expected: int,
) -> None:
    value = plan.get(field_path)
    if type(value) is not int:
        raise TypeError(
            f"{field_path} must be an int, actual={type(value).__name__}"
        )
    if value != expected:
        raise RuntimeError(
            f"{field_path} mismatch: expected={expected}, actual={value}"
        )


def _validate_entry_set(
    entries: Sequence[Mapping[str, Any]],
    *,
    split: str,
    field_path: str,
) -> None:
    for index, entry in enumerate(entries):
        _validate_formal_entry(entry, split=split, field_path=f"{field_path}[{index}]")


def _validate_formal_entry(
    entry: Mapping[str, Any],
    *,
    split: str,
    field_path: str,
) -> None:
    required = {
        "identity",
        "sample_index",
        "sample_id",
        "split",
        "split_index",
        "prompt_sha256",
        "file_sha256",
    }
    missing = sorted(required - set(entry.keys()))
    if missing:
        raise RuntimeError(f"{field_path} missing fields: {missing}")
    _require_entry_int(entry, "sample_index", field_path)
    _require_entry_int(entry, "split_index", field_path)
    if str(entry["split"]) != split:
        raise RuntimeError(
            f"{field_path}.split mismatch: expected={split}, actual={entry['split']}"
        )
    if entry["sample_id"] is not None and not isinstance(entry["sample_id"], str):
        raise TypeError(
            f"{field_path}.sample_id must be a string or null, "
            f"actual={type(entry['sample_id']).__name__}"
        )
    _require_lower_sha256(entry["prompt_sha256"], f"{field_path}.prompt_sha256")
    _require_lower_sha256(entry["file_sha256"], f"{field_path}.file_sha256")
    _require_payload_location(entry, field_path)
    expected_identity = m4_sample_identity_from_record(entry)
    actual_identity = entry["identity"]
    if not isinstance(actual_identity, str):
        raise TypeError(
            f"{field_path}.identity must be a string, "
            f"actual={type(actual_identity).__name__}"
        )
    if actual_identity != expected_identity:
        raise RuntimeError(
            f"{field_path}.identity mismatch: "
            f"expected={expected_identity}, actual={actual_identity}"
        )


def _require_entry_int(
    entry: Mapping[str, Any],
    key: str,
    field_path: str,
) -> int:
    value = entry[key]
    if type(value) is not int:
        raise TypeError(
            f"{field_path}.{key} must be an int, "
            f"actual={type(value).__name__}"
        )
    return value


def _require_lower_sha256(value: Any, field_path: str) -> None:
    if not isinstance(value, str):
        raise TypeError(
            f"{field_path} must be a string, actual={type(value).__name__}"
        )
    if _SHA256_RE.fullmatch(value) is None:
        raise RuntimeError(f"{field_path} must be a 64-char lowercase hex SHA256")


def _require_payload_location(entry: Mapping[str, Any], field_path: str) -> None:
    found = False
    for key in _PAYLOAD_LOCATION_FIELDS:
        if key not in entry:
            continue
        value = entry[key]
        if not isinstance(value, str):
            raise TypeError(
                f"{field_path}.{key} must be a string, "
                f"actual={type(value).__name__}"
            )
        if value.strip() == "":
            raise RuntimeError(f"{field_path}.{key} must be non-empty")
        found = True
    if not found:
        raise RuntimeError(f"{field_path} missing payload location")


def _stable_entry_sort_key(entry: Mapping[str, Any]) -> tuple[int, int, str, str]:
    return (
        int(entry["split_index"]),
        int(entry["sample_index"]),
        "" if entry.get("sample_id") is None else str(entry.get("sample_id")),
        str(entry["prompt_sha256"]),
    )


def _validate_stable_order(
    entries: Sequence[Mapping[str, Any]],
    field_path: str,
) -> None:
    expected = sorted(entries, key=_stable_entry_sort_key)
    if list(entries) != expected:
        raise RuntimeError(f"{field_path} entries are not in stable M4 ordering")


def _validate_no_sample_index_overlap(
    train_entries: Sequence[Mapping[str, Any]],
    validation_entries: Sequence[Mapping[str, Any]],
) -> None:
    train_indices = {int(entry["sample_index"]) for entry in train_entries}
    validation_indices = {int(entry["sample_index"]) for entry in validation_entries}
    overlap = sorted(train_indices & validation_indices)
    if overlap:
        raise RuntimeError(f"train/validation sample_index overlap: {overlap[:5]}")


def _require_equal(actual: Any, expected: Any, field_path: str) -> None:
    if actual != expected:
        raise RuntimeError(f"{field_path} differs from manifest reconstruction")


__all__ = [
    "M5_FORMAL_SAMPLE_PLAN_AUDIT_SCHEMA",
    "M5_FORMAL_TRAIN_SAMPLE_COUNT",
    "M5_FORMAL_VALIDATION_SAMPLE_COUNT",
    "build_m5_formal_sample_plan",
    "validate_m5_formal_sample_plan",
    "write_m5_formal_sample_plan",
]
