from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from utils.nf_sf_m3 import (
    M3_REFERENCE_CHECKPOINT_SHA256,
    M3TeacherSample,
    load_m3_teacher_sample,
)
from utils.nf_sf_m4 import m4_sample_identity_from_metadata, m4_train_entry_for_step
from utils.nf_sf_m5_formal_plan import validate_m5_formal_sample_plan

_PAYLOAD_LOCATION_FIELDS = ("file", "path", "payload_path", "artifact_path")


@dataclass(frozen=True, slots=True)
class M5TeacherSampleEntry:
    identity: str
    sample_index: int
    sample_id: str | None
    split: str
    split_index: int
    prompt_sha256: str
    payload_location_field: str
    payload_location: str
    file_sha256: str


class M5TeacherSampleStore:
    def __init__(
        self,
        *,
        sample_plan: Mapping[str, Any],
        manifest_path: Path | str,
        dataset_root: Path | str,
        reference_checkpoint_path: Path | str | None = None,
        expected_reference_sha256: str = M3_REFERENCE_CHECKPOINT_SHA256,
    ) -> None:
        if not isinstance(sample_plan, Mapping):
            raise TypeError(
                "sample_plan must be a mapping, "
                f"actual={type(sample_plan).__name__}"
            )
        saved_sha256 = sample_plan.get("sample_plan_sha256")
        if not isinstance(saved_sha256, str) or saved_sha256.strip() == "":
            raise RuntimeError("sample_plan.sample_plan_sha256 must be a non-empty string")

        manifest_resolved = Path(manifest_path).resolve()
        dataset_root_resolved = Path(dataset_root).resolve()
        audit = validate_m5_formal_sample_plan(
            sample_plan,
            manifest_path=manifest_resolved,
            dataset_root=dataset_root_resolved,
            expected_sha256=saved_sha256,
        )

        train_entries = tuple(
            _entry_from_plan_entry(entry, field_path=f"samples.train[{index}]")
            for index, entry in enumerate(sample_plan["samples"]["train"])
        )
        validation_entries = tuple(
            _entry_from_plan_entry(
                entry,
                field_path=f"samples.validation[{index}]",
            )
            for index, entry in enumerate(sample_plan["samples"]["validation"])
        )
        entries: dict[str, M5TeacherSampleEntry] = {}
        for entry in (*train_entries, *validation_entries):
            if entry.identity in entries:
                raise RuntimeError(f"duplicate sample identity: {entry.identity}")
            entries[entry.identity] = entry

        self._sample_plan_sha256 = str(audit["sample_plan_sha256"])
        self._manifest_path = manifest_resolved
        self._manifest_sha256 = str(audit["manifest_sha256"])
        self._dataset_root = dataset_root_resolved
        self._fixed_decode_validation_identity = str(
            sample_plan["fixed_decode_validation_identity"]
        )
        self._train_identities = tuple(entry.identity for entry in train_entries)
        self._validation_identities = tuple(
            entry.identity for entry in validation_entries
        )
        self._entries_by_identity: Mapping[str, M5TeacherSampleEntry] = MappingProxyType(
            entries
        )
        self._train_step_plan = {
            "samples": {
                "train": [
                    _plan_entry_dict(entry)
                    for entry in train_entries
                ],
            },
        }
        self._reference_checkpoint_path = (
            None
            if reference_checkpoint_path is None
            else Path(reference_checkpoint_path).resolve()
        )
        self._expected_reference_sha256 = str(expected_reference_sha256)
        self._active_sample: M3TeacherSample | None = None
        self._acquire_in_progress = False
        self._live_sample_count = 0
        self._max_live_sample_count = 0
        self._load_attempt_count = 0
        self._successful_load_count = 0

    @property
    def sample_plan_sha256(self) -> str:
        return self._sample_plan_sha256

    @property
    def manifest_path(self) -> Path:
        return self._manifest_path

    @property
    def manifest_sha256(self) -> str:
        return self._manifest_sha256

    @property
    def dataset_root(self) -> Path:
        return self._dataset_root

    @property
    def fixed_decode_validation_identity(self) -> str:
        return self._fixed_decode_validation_identity

    @property
    def train_identities(self) -> tuple[str, ...]:
        return self._train_identities

    @property
    def validation_identities(self) -> tuple[str, ...]:
        return self._validation_identities

    @property
    def live_sample_count(self) -> int:
        return self._live_sample_count

    @property
    def max_live_sample_count(self) -> int:
        return self._max_live_sample_count

    @property
    def load_attempt_count(self) -> int:
        return self._load_attempt_count

    @property
    def successful_load_count(self) -> int:
        return self._successful_load_count

    @property
    def total_load_count(self) -> int:
        return self._successful_load_count

    def entry(self, identity: str) -> M5TeacherSampleEntry:
        if not isinstance(identity, str):
            raise TypeError(
                "identity must be a string, "
                f"actual={type(identity).__name__}"
            )
        try:
            return self._entries_by_identity[identity]
        except KeyError as exc:
            raise RuntimeError(f"unknown teacher sample identity: {identity}") from exc

    def train_entry_for_step(self, step: int) -> M5TeacherSampleEntry:
        step_value = _require_positive_python_int(step, "step")
        entry = m4_train_entry_for_step(self._train_step_plan, step_value)
        return self.entry(str(entry["identity"]))

    def train_identity_for_step(self, step: int) -> str:
        return self.train_entry_for_step(step).identity

    @contextmanager
    def acquire(self, identity: str) -> Iterator[M3TeacherSample]:
        entry = self.entry(identity)
        if self._acquire_in_progress or self._active_sample is not None:
            raise RuntimeError("M5TeacherSampleStore already has an active sample")

        self._acquire_in_progress = True
        try:
            self._load_attempt_count += 1
            sample = load_m3_teacher_sample(
                manifest_path=self._manifest_path,
                dataset_root=self._dataset_root,
                sample_index=entry.sample_index,
                reference_checkpoint_path=self._reference_checkpoint_path,
                expected_reference_sha256=self._expected_reference_sha256,
            )
            if not isinstance(sample, M3TeacherSample):
                raise TypeError(
                    "load_m3_teacher_sample returned "
                    f"{type(sample).__name__}, expected M3TeacherSample"
                )
            _validate_loaded_sample(
                sample,
                entry=entry,
                manifest_path=self._manifest_path,
                manifest_sha256=self._manifest_sha256,
                dataset_root=self._dataset_root,
            )

            self._active_sample = sample
            self._live_sample_count = 1
            self._max_live_sample_count = max(self._max_live_sample_count, 1)
            self._successful_load_count += 1
            yield sample
        finally:
            self._active_sample = None
            self._live_sample_count = 0
            self._acquire_in_progress = False


def _entry_from_plan_entry(
    entry: Mapping[str, Any],
    *,
    field_path: str,
) -> M5TeacherSampleEntry:
    payload_field, payload_location = _single_payload_location(entry, field_path)
    return M5TeacherSampleEntry(
        identity=str(entry["identity"]),
        sample_index=_require_python_int(entry["sample_index"], f"{field_path}.sample_index"),
        sample_id=None if entry["sample_id"] is None else str(entry["sample_id"]),
        split=str(entry["split"]),
        split_index=_require_python_int(entry["split_index"], f"{field_path}.split_index"),
        prompt_sha256=str(entry["prompt_sha256"]),
        payload_location_field=payload_field,
        payload_location=payload_location,
        file_sha256=str(entry["file_sha256"]),
    )


def _single_payload_location(
    entry: Mapping[str, Any],
    field_path: str,
) -> tuple[str, str]:
    locations = []
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
        locations.append((key, value))
    if len(locations) != 1:
        raise RuntimeError(
            f"{field_path} must have exactly one payload location field, "
            f"actual={len(locations)}"
        )
    key, value = locations[0]
    if Path(value).name.lower().endswith(".tmp"):
        raise RuntimeError(f"{field_path}.{key} must not point to a .tmp payload")
    return key, value


def _require_python_int(value: Any, field_path: str) -> int:
    if type(value) is not int:
        raise TypeError(
            f"{field_path} must be a Python int, actual={type(value).__name__}"
        )
    return value


def _require_positive_python_int(value: Any, field_path: str) -> int:
    result = _require_python_int(value, field_path)
    if result <= 0:
        raise ValueError(f"{field_path} must be positive, actual={result}")
    return result


def _plan_entry_dict(entry: M5TeacherSampleEntry) -> dict[str, Any]:
    return {
        "identity": entry.identity,
        "sample_index": entry.sample_index,
        "sample_id": entry.sample_id,
        "split": entry.split,
        "split_index": entry.split_index,
        "prompt_sha256": entry.prompt_sha256,
        entry.payload_location_field: entry.payload_location,
        "file_sha256": entry.file_sha256,
    }


def _validate_loaded_sample(
    sample: M3TeacherSample,
    *,
    entry: M5TeacherSampleEntry,
    manifest_path: Path,
    manifest_sha256: str,
    dataset_root: Path,
) -> None:
    metadata = sample.metadata
    if not isinstance(metadata, Mapping):
        raise TypeError("loaded sample metadata must be a mapping")
    for key in (
        "sample_index",
        "sample_id",
        "split",
        "split_index",
        "prompt_sha256",
        "latent_file_sha256",
        "manifest_path",
        "manifest_sha256",
        "dataset_root",
    ):
        if key not in metadata:
            raise RuntimeError(f"loaded sample metadata missing {key}")

    _require_metadata_equal(metadata, "sample_index", entry.sample_index)
    _require_metadata_equal(metadata, "sample_id", entry.sample_id)
    _require_metadata_equal(metadata, "split", entry.split)
    _require_metadata_equal(metadata, "split_index", entry.split_index)
    _require_metadata_equal(metadata, "prompt_sha256", entry.prompt_sha256)
    _require_metadata_equal(metadata, "latent_file_sha256", entry.file_sha256)
    _require_metadata_equal(metadata, "manifest_path", str(manifest_path.resolve()))
    _require_metadata_equal(metadata, "manifest_sha256", manifest_sha256)
    _require_metadata_equal(metadata, "dataset_root", str(dataset_root.resolve()))

    actual_identity = m4_sample_identity_from_metadata(metadata)
    if actual_identity != entry.identity:
        raise RuntimeError(
            "loaded sample identity mismatch: "
            f"expected={entry.identity}, actual={actual_identity}"
        )


def _require_metadata_equal(
    metadata: Mapping[str, Any],
    key: str,
    expected: Any,
) -> None:
    actual = metadata[key]
    if actual != expected:
        raise RuntimeError(
            f"loaded sample metadata.{key} mismatch: "
            f"expected={expected}, actual={actual}"
        )


__all__ = [
    "M5TeacherSampleEntry",
    "M5TeacherSampleStore",
]
