from __future__ import annotations

import json
import os
import re
import uuid
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from types import MappingProxyType
from typing import Any

import torch

from utils.nf_sf_m3 import file_sha256
from utils.nf_sf_m4 import (
    canonical_json_sha256,
    load_teacher_manifest,
    m4_sample_identity_from_record,
    validate_m4_sample_plan,
    write_m4_json,
)
from utils.nf_sf_m5_formal_plan import validate_m5_formal_sample_plan

M5_CONDITIONAL_ARTIFACT_SCHEMA = "nf_sf_m5_conditional_artifact_v1"
M5_CONDITIONAL_ITEM_SCHEMA = "nf_sf_m5_conditional_item_v1"
M5_CONDITIONAL_ARTIFACT_MANIFEST_NAME = "manifest.json"
M5_CONDITIONAL_ARTIFACT_ITEMS_DIR = "items"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REQUIRED_ENCODER_PROVENANCE_FIELDS = (
    "encoder_class",
    "model_checkpoint_path",
    "model_checkpoint_sha256",
    "tokenizer_path",
    "dtype",
)


def m5_conditional_item_name(identity: str) -> str:
    digest = canonical_json_sha256({"identity": str(identity)})
    return f"{digest}.pt"


def encode_m5_prompt_condition(
    text_encoder: Callable[[list[str]], Mapping[str, torch.Tensor]],
    prompt: str,
) -> dict[str, torch.Tensor]:
    with torch.no_grad():
        conditional = text_encoder([prompt])
    return _cpu_conditional_tensors(conditional, field_path="text_encoder output")


def build_m5_conditional_artifact(
    *,
    sample_plan: Mapping[str, Any],
    manifest_path: Path | str,
    dataset_root: Path | str,
    output_dir: Path | str,
    text_encoder: Callable[[list[str]], Mapping[str, torch.Tensor]],
    encoder_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    output_path = _require_new_artifact_dir(output_dir)
    manifest_file = _require_existing_manifest(manifest_path)
    plan_sha256 = _require_nonempty_string(
        sample_plan.get("sample_plan_sha256"),
        "sample_plan.sample_plan_sha256",
    )
    validate_m5_formal_sample_plan(
        sample_plan,
        manifest_path=manifest_file,
        dataset_root=dataset_root,
        expected_sha256=plan_sha256,
    )
    provenance = _validate_encoder_provenance(encoder_provenance)
    teacher_manifest = load_teacher_manifest(manifest_file)
    prompts = _prompt_index_for_plan(sample_plan, teacher_manifest)

    output_path.mkdir()
    items_dir = output_path / M5_CONDITIONAL_ARTIFACT_ITEMS_DIR
    items_dir.mkdir()

    items: dict[str, dict[str, Any]] = {}
    max_live_conditionals = 0
    encoded_count = 0
    all_identities = _all_plan_identities(sample_plan)
    for identity in all_identities:
        prompt, prompt_sha256 = prompts[identity]
        conditional = encode_m5_prompt_condition(text_encoder, prompt)
        max_live_conditionals = max(max_live_conditionals, 1)
        encoded_count += 1
        item_rel = (
            f"{M5_CONDITIONAL_ARTIFACT_ITEMS_DIR}/"
            f"{m5_conditional_item_name(identity)}"
        )
        item_path = output_path / item_rel
        item_payload = {
            "schema": M5_CONDITIONAL_ITEM_SCHEMA,
            "identity": identity,
            "prompt_sha256": prompt_sha256,
            "tensors": conditional,
        }
        _atomic_torch_save(item_payload, item_path)
        items[identity] = {
            "prompt_sha256": prompt_sha256,
            "item_relative_path": item_rel,
            "item_file_sha256": file_sha256(item_path),
            "tensors": _tensor_metadata(conditional),
        }
        del conditional
        del item_payload

    artifact_manifest = {
        "schema": M5_CONDITIONAL_ARTIFACT_SCHEMA,
        "status": "PASS",
        "sample_plan_sha256": plan_sha256,
        "teacher_manifest_sha256": str(sample_plan["manifest_sha256"]),
        "encoder_provenance": provenance,
        "train_identities": [
            str(value) for value in sample_plan["train_sample_identities"]
        ],
        "validation_identities": [
            str(value) for value in sample_plan["validation_sample_identities"]
        ],
        "items": items,
    }
    artifact_manifest["artifact_sha256"] = m5_conditional_artifact_sha256(
        artifact_manifest
    )
    validate_m5_conditional_artifact_manifest(
        artifact_manifest,
        artifact_dir=output_path,
        sample_plan=sample_plan,
        check_item_files=True,
    )
    manifest_path_out = output_path / M5_CONDITIONAL_ARTIFACT_MANIFEST_NAME
    write_m4_json(artifact_manifest, manifest_path_out)
    loaded = load_m5_conditional_artifact_manifest(manifest_path_out)
    validate_m5_conditional_artifact_manifest(
        loaded,
        artifact_dir=output_path,
        sample_plan=sample_plan,
        expected_artifact_sha256=str(artifact_manifest["artifact_sha256"]),
        check_item_files=True,
    )

    return {
        "status": "PASS",
        "artifact_dir": str(output_path.resolve()),
        "artifact_sha256": str(artifact_manifest["artifact_sha256"]),
        "encoded_count": encoded_count,
        "max_live_conditionals": max_live_conditionals,
        "manifest_path": str(manifest_path_out.resolve()),
    }


def load_m5_conditional_artifact_manifest(path: Path | str) -> dict[str, Any]:
    manifest_path = Path(path)
    if manifest_path.name.lower().endswith(".tmp"):
        raise ValueError("conditional artifact manifest path must not end with .tmp")
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("conditional artifact manifest must be a JSON object")
    return value


def m5_conditional_artifact_sha256(manifest: Mapping[str, Any]) -> str:
    return canonical_json_sha256(_manifest_without_artifact_sha256(manifest))


def validate_m5_conditional_artifact_manifest(
    manifest: Mapping[str, Any],
    *,
    artifact_dir: Path | str,
    sample_plan: Mapping[str, Any] | None = None,
    expected_artifact_sha256: str | None = None,
    check_item_files: bool = False,
) -> dict[str, Any]:
    if not isinstance(manifest, Mapping):
        raise TypeError("conditional artifact manifest must be a mapping")
    if manifest.get("schema") != M5_CONDITIONAL_ARTIFACT_SCHEMA:
        raise RuntimeError("conditional artifact schema mismatch")
    if manifest.get("status") != "PASS":
        raise RuntimeError("conditional artifact status is not PASS")
    artifact_path = _require_existing_artifact_dir(artifact_dir)
    sample_plan_sha256 = _require_nonempty_string(
        manifest.get("sample_plan_sha256"),
        "sample_plan_sha256",
    )
    teacher_manifest_sha256 = _require_nonempty_string(
        manifest.get("teacher_manifest_sha256"),
        "teacher_manifest_sha256",
    )
    provenance = manifest.get("encoder_provenance")
    if not isinstance(provenance, Mapping):
        raise TypeError("encoder_provenance must be a mapping")
    _validate_encoder_provenance(provenance)
    train_identities = _string_list(manifest.get("train_identities"), "train_identities")
    validation_identities = _string_list(
        manifest.get("validation_identities"),
        "validation_identities",
    )
    items = manifest.get("items")
    if not isinstance(items, Mapping):
        raise TypeError("items must be a mapping")
    if set(items) != {*train_identities, *validation_identities}:
        raise RuntimeError("conditional artifact item identities differ from order lists")
    if set(train_identities) & set(validation_identities):
        raise RuntimeError("conditional artifact train/validation identities overlap")
    if len(train_identities) != len(set(train_identities)):
        raise RuntimeError("conditional artifact train identities contain duplicates")
    if len(validation_identities) != len(set(validation_identities)):
        raise RuntimeError("conditional artifact validation identities contain duplicates")

    if sample_plan is not None:
        plan_sha256 = _require_nonempty_string(
            sample_plan.get("sample_plan_sha256"),
            "sample_plan.sample_plan_sha256",
        )
        plan_report = validate_m4_sample_plan(
            sample_plan,
            expected_sha256=plan_sha256,
        )
        if str(plan_report["sample_plan_sha256"]) != sample_plan_sha256:
            raise RuntimeError("conditional artifact sample_plan_sha256 mismatch")
        if str(sample_plan.get("manifest_sha256")) != teacher_manifest_sha256:
            raise RuntimeError("conditional artifact teacher_manifest_sha256 mismatch")
        if train_identities != [str(value) for value in sample_plan["train_sample_identities"]]:
            raise RuntimeError("conditional artifact train identity order mismatch")
        if validation_identities != [
            str(value) for value in sample_plan["validation_sample_identities"]
        ]:
            raise RuntimeError("conditional artifact validation identity order mismatch")
        sample_plan_entries = _sample_plan_entry_index(sample_plan)
        artifact_identities = (*train_identities, *validation_identities)
        if set(sample_plan_entries) != set(artifact_identities):
            raise RuntimeError("conditional artifact identities differ from sample plan")
        for identity in artifact_identities:
            item_metadata = items[identity]
            if not isinstance(item_metadata, Mapping):
                raise TypeError(f"items[{identity!r}] must be a mapping")
            item_prompt_sha256 = _require_sha256(
                item_metadata.get("prompt_sha256"),
                f"items[{identity!r}].prompt_sha256",
            )
            plan_entry = sample_plan_entries[identity]
            plan_prompt_sha256 = _require_sha256(
                plan_entry.get("prompt_sha256"),
                f"sample_plan entry {identity}.prompt_sha256",
            )
            if item_prompt_sha256 != plan_prompt_sha256:
                raise RuntimeError(
                    "conditional artifact prompt_sha256 differs from sample plan: "
                    f"identity={identity}, expected={plan_prompt_sha256}, "
                    f"actual={item_prompt_sha256}"
                )

    for identity in (*train_identities, *validation_identities):
        _validate_item_metadata(
            identity,
            items[identity],
            artifact_dir=artifact_path,
            check_item_file=check_item_files,
        )

    actual_sha256 = m5_conditional_artifact_sha256(manifest)
    saved_sha256 = manifest.get("artifact_sha256")
    if not isinstance(saved_sha256, str):
        raise TypeError("artifact_sha256 must be a string")
    if saved_sha256 != actual_sha256:
        raise RuntimeError("conditional artifact SHA256 does not match contents")
    if expected_artifact_sha256 is not None and expected_artifact_sha256 != actual_sha256:
        raise RuntimeError("conditional artifact SHA256 differs from expected value")
    return {
        "status": "PASS",
        "artifact_sha256": actual_sha256,
        "sample_plan_sha256": sample_plan_sha256,
        "teacher_manifest_sha256": teacher_manifest_sha256,
        "train_count": len(train_identities),
        "validation_count": len(validation_identities),
    }


class M5ConditionalArtifactStore:
    def __init__(
        self,
        *,
        artifact_dir: Path | str,
        sample_plan: Mapping[str, Any] | None = None,
        expected_artifact_sha256: str | None = None,
    ) -> None:
        artifact_path = _require_existing_artifact_dir(artifact_dir)
        manifest_path = artifact_path / M5_CONDITIONAL_ARTIFACT_MANIFEST_NAME
        manifest = load_m5_conditional_artifact_manifest(manifest_path)
        report = validate_m5_conditional_artifact_manifest(
            manifest,
            artifact_dir=artifact_path,
            sample_plan=sample_plan,
            expected_artifact_sha256=expected_artifact_sha256,
            check_item_files=False,
        )
        items = {
            str(identity): MappingProxyType(dict(metadata))
            for identity, metadata in manifest["items"].items()
        }
        self._artifact_dir = artifact_path
        self._manifest_path = manifest_path
        self._artifact_sha256 = str(report["artifact_sha256"])
        self._sample_plan_sha256 = str(report["sample_plan_sha256"])
        self._teacher_manifest_sha256 = str(report["teacher_manifest_sha256"])
        self._encoder_provenance = MappingProxyType(dict(manifest["encoder_provenance"]))
        self._train_identities = tuple(str(value) for value in manifest["train_identities"])
        self._validation_identities = tuple(
            str(value) for value in manifest["validation_identities"]
        )
        self._items = MappingProxyType(items)
        self._active_conditional: dict[str, torch.Tensor] | None = None
        self._acquire_in_progress = False
        self._live_conditional_count = 0
        self._max_live_conditional_count = 0
        self._load_attempt_count = 0
        self._successful_load_count = 0

    @property
    def artifact_sha256(self) -> str:
        return self._artifact_sha256

    @property
    def sample_plan_sha256(self) -> str:
        return self._sample_plan_sha256

    @property
    def teacher_manifest_sha256(self) -> str:
        return self._teacher_manifest_sha256

    @property
    def train_identities(self) -> tuple[str, ...]:
        return self._train_identities

    @property
    def validation_identities(self) -> tuple[str, ...]:
        return self._validation_identities

    @property
    def live_conditional_count(self) -> int:
        return self._live_conditional_count

    @property
    def max_live_conditional_count(self) -> int:
        return self._max_live_conditional_count

    @property
    def load_attempt_count(self) -> int:
        return self._load_attempt_count

    @property
    def total_load_count(self) -> int:
        return self._successful_load_count

    @contextmanager
    def acquire(self, identity: str) -> Iterator[dict[str, torch.Tensor]]:
        if not isinstance(identity, str):
            raise TypeError("identity must be a string")
        if identity not in self._items:
            raise RuntimeError(f"unknown conditional identity: {identity}")
        if self._acquire_in_progress or self._active_conditional is not None:
            raise RuntimeError("M5ConditionalArtifactStore already has an active item")

        metadata = self._items[identity]
        self._acquire_in_progress = True
        try:
            self._load_attempt_count += 1
            item_path = _item_path(self._artifact_dir, metadata["item_relative_path"])
            actual_file_sha256 = file_sha256(item_path)
            expected_file_sha256 = str(metadata["item_file_sha256"])
            if actual_file_sha256 != expected_file_sha256:
                raise RuntimeError(
                    "conditional item SHA256 mismatch: "
                    f"expected={expected_file_sha256}, actual={actual_file_sha256}"
                )
            item_payload = torch.load(
                item_path,
                map_location="cpu",
                weights_only=False,
            )
            conditional = _validate_item_payload(
                item_payload,
                identity=identity,
                item_metadata=metadata,
            )
            self._active_conditional = conditional
            self._live_conditional_count = 1
            self._max_live_conditional_count = max(
                self._max_live_conditional_count,
                1,
            )
            self._successful_load_count += 1
            yield conditional
        finally:
            self._active_conditional = None
            self._live_conditional_count = 0
            self._acquire_in_progress = False


def _manifest_without_artifact_sha256(manifest: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(manifest)
    value.pop("artifact_sha256", None)
    return value


def _require_new_artifact_dir(value: Path | str) -> Path:
    path = Path(value)
    if str(path).strip() == "":
        raise ValueError("output_dir must be non-empty")
    if path.name.lower().endswith(".tmp"):
        raise ValueError("output_dir must not end with .tmp")
    if path.exists():
        raise FileExistsError(f"output_dir already exists: {path}")
    if not path.parent.exists():
        raise FileNotFoundError(f"output_dir parent does not exist: {path.parent}")
    if not path.parent.is_dir():
        raise ValueError(f"output_dir parent must be a directory: {path.parent}")
    return path


def _require_existing_artifact_dir(value: Path | str) -> Path:
    path = Path(value)
    if path.name.lower().endswith(".tmp"):
        raise ValueError("artifact_dir must not end with .tmp")
    if not path.is_dir():
        raise FileNotFoundError(f"artifact_dir must be an existing directory: {path}")
    return path


def _require_existing_manifest(value: Path | str) -> Path:
    path = Path(value)
    if path.name.lower().endswith(".tmp"):
        raise ValueError("manifest_path must not end with .tmp")
    if not path.is_file():
        raise FileNotFoundError(f"manifest_path must be an existing file: {path}")
    return path


def _require_nonempty_string(value: Any, field_path: str) -> str:
    if not isinstance(value, str) or value.strip() == "":
        raise RuntimeError(f"{field_path} must be a non-empty string")
    return value


def _json_safe_mapping(value: Mapping[str, Any], field_path: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_path} must be a mapping")
    json.dumps(value, sort_keys=True, allow_nan=False)
    return dict(value)


def _require_sha256(value: Any, field_path: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_path} must be a string")
    if _SHA256_RE.fullmatch(value) is None:
        raise RuntimeError(f"{field_path} must be a 64-character lowercase SHA256")
    return value


def _validate_encoder_provenance(value: Mapping[str, Any]) -> dict[str, Any]:
    provenance = _json_safe_mapping(value, "encoder_provenance")
    for field in _REQUIRED_ENCODER_PROVENANCE_FIELDS:
        _require_nonempty_string(
            provenance.get(field),
            f"encoder_provenance.{field}",
        )
    _require_sha256(
        provenance["model_checkpoint_sha256"],
        "encoder_provenance.model_checkpoint_sha256",
    )
    return provenance


def _sample_plan_entry_index(
    sample_plan: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    entries: dict[str, Mapping[str, Any]] = {}
    for split in ("train", "validation"):
        for entry in sample_plan["samples"][split]:
            if not isinstance(entry, Mapping):
                raise TypeError(f"sample_plan.samples.{split} entries must be mappings")
            identity = _require_nonempty_string(
                entry.get("identity"),
                f"sample_plan.samples.{split}.identity",
            )
            if identity in entries:
                raise RuntimeError(f"duplicate sample plan identity: {identity}")
            entries[identity] = entry
    return entries


def _all_plan_identities(sample_plan: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(
        str(identity)
        for identity in (
            *sample_plan["train_sample_identities"],
            *sample_plan["validation_sample_identities"],
        )
    )


def _prompt_index_for_plan(
    sample_plan: Mapping[str, Any],
    teacher_manifest: Mapping[str, Any],
) -> dict[str, tuple[str, str]]:
    prompts: dict[str, tuple[str, str]] = {}
    for record in teacher_manifest["samples"]:
        identity = m4_sample_identity_from_record(record)
        prompt = _require_nonempty_string(record.get("prompt"), f"manifest sample {identity}.prompt")
        prompt_sha256 = _require_nonempty_string(
            record.get("prompt_sha256"),
            f"manifest sample {identity}.prompt_sha256",
        )
        if identity in prompts:
            raise RuntimeError(f"duplicate teacher manifest identity: {identity}")
        prompts[identity] = (prompt, prompt_sha256)
    required = _all_plan_identities(sample_plan)
    missing = [identity for identity in required if identity not in prompts]
    if missing:
        raise RuntimeError(f"teacher manifest missing prompts for identities: {missing[:3]}")
    for entry in (*sample_plan["samples"]["train"], *sample_plan["samples"]["validation"]):
        identity = str(entry["identity"])
        if prompts[identity][1] != str(entry["prompt_sha256"]):
            raise RuntimeError(f"prompt_sha256 mismatch for identity: {identity}")
    return prompts


def _cpu_conditional_tensors(
    value: Mapping[str, torch.Tensor],
    *,
    field_path: str,
) -> dict[str, torch.Tensor]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_path} must be a mapping")
    tensors: dict[str, torch.Tensor] = {}
    for key in sorted(value):
        if not isinstance(key, str) or key.strip() == "":
            raise TypeError(f"{field_path} keys must be non-empty strings")
        tensor = value[key]
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{field_path}.{key} must be a torch.Tensor")
        tensors[key] = tensor.detach().contiguous().cpu()
        if tensors[key].device.type != "cpu":
            raise RuntimeError(f"{field_path}.{key} must be saved on CPU")
    if not tensors:
        raise RuntimeError(f"{field_path} must contain at least one tensor")
    return tensors


def _tensor_metadata(tensors: Mapping[str, torch.Tensor]) -> dict[str, dict[str, Any]]:
    return {
        key: {
            "shape": [int(dim) for dim in tensor.shape],
            "dtype": str(tensor.dtype),
        }
        for key, tensor in sorted(tensors.items())
    }


def _atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    if path.name.lower().endswith(".tmp"):
        raise ValueError("conditional item path must not end with .tmp")
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            pass


def _string_list(value: Any, field_path: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise TypeError(f"{field_path} must be a string list")
    return list(value)


def _validate_item_metadata(
    identity: str,
    metadata: Any,
    *,
    artifact_dir: Path,
    check_item_file: bool,
) -> None:
    if not isinstance(metadata, Mapping):
        raise TypeError(f"items[{identity!r}] must be a mapping")
    prompt_sha256 = _require_nonempty_string(
        metadata.get("prompt_sha256"),
        f"items[{identity!r}].prompt_sha256",
    )
    _require_sha256(prompt_sha256, f"items[{identity!r}].prompt_sha256")
    item_relative_path = _require_nonempty_string(
        metadata.get("item_relative_path"),
        f"items[{identity!r}].item_relative_path",
    )
    _item_path(artifact_dir, item_relative_path)
    item_file_sha256 = _require_nonempty_string(
        metadata.get("item_file_sha256"),
        f"items[{identity!r}].item_file_sha256",
    )
    _require_sha256(item_file_sha256, f"items[{identity!r}].item_file_sha256")
    tensors = metadata.get("tensors")
    if not isinstance(tensors, Mapping) or not tensors:
        raise TypeError(f"items[{identity!r}].tensors must be a non-empty mapping")
    for key, tensor_metadata in tensors.items():
        if not isinstance(key, str) or key.strip() == "":
            raise TypeError(f"items[{identity!r}].tensors keys must be strings")
        if not isinstance(tensor_metadata, Mapping):
            raise TypeError(f"items[{identity!r}].tensors[{key!r}] must be a mapping")
        shape = tensor_metadata.get("shape")
        dtype = tensor_metadata.get("dtype")
        if not isinstance(shape, list) or not all(type(dim) is int for dim in shape):
            raise TypeError(f"items[{identity!r}].tensors[{key!r}].shape must be ints")
        if not isinstance(dtype, str) or dtype.strip() == "":
            raise TypeError(f"items[{identity!r}].tensors[{key!r}].dtype must be a string")
    if check_item_file:
        actual = file_sha256(_item_path(artifact_dir, item_relative_path))
        if actual != item_file_sha256:
            raise RuntimeError(f"conditional item file SHA256 mismatch for {identity}")
    if prompt_sha256.strip() == "":
        raise RuntimeError(f"items[{identity!r}].prompt_sha256 must be non-empty")


def _item_path(artifact_dir: Path, relative_path: Any) -> Path:
    relative = _require_nonempty_string(relative_path, "item_relative_path")
    raw = Path(relative)
    if raw.is_absolute():
        raise RuntimeError("conditional item path must be relative")
    if raw.name.lower().endswith(".tmp"):
        raise RuntimeError("conditional item path must not end with .tmp")
    resolved = (artifact_dir / raw).resolve()
    artifact_resolved = artifact_dir.resolve()
    if artifact_resolved not in (resolved.parent, *resolved.parents):
        raise RuntimeError("conditional item path escapes artifact directory")
    return resolved


def _validate_item_payload(
    payload: Any,
    *,
    identity: str,
    item_metadata: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    if not isinstance(payload, Mapping):
        raise TypeError("conditional item payload must be a mapping")
    if payload.get("schema") != M5_CONDITIONAL_ITEM_SCHEMA:
        raise RuntimeError("conditional item schema mismatch")
    if payload.get("identity") != identity:
        raise RuntimeError("conditional item identity mismatch")
    if payload.get("prompt_sha256") != item_metadata["prompt_sha256"]:
        raise RuntimeError("conditional item prompt_sha256 mismatch")
    tensors = payload.get("tensors")
    if not isinstance(tensors, Mapping):
        raise TypeError("conditional item tensors must be a mapping")
    expected = item_metadata["tensors"]
    if set(tensors) != set(expected):
        raise RuntimeError("conditional item tensor keys mismatch")
    conditional: dict[str, torch.Tensor] = {}
    for key, tensor in tensors.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"conditional item tensor {key!r} must be a torch.Tensor")
        if tensor.device.type != "cpu":
            raise RuntimeError(f"conditional item tensor {key!r} must be on CPU")
        metadata = expected[key]
        shape = [int(dim) for dim in tensor.shape]
        dtype = str(tensor.dtype)
        if shape != metadata["shape"]:
            raise RuntimeError(f"conditional item tensor {key!r} shape mismatch")
        if dtype != metadata["dtype"]:
            raise RuntimeError(f"conditional item tensor {key!r} dtype mismatch")
        conditional[str(key)] = tensor.detach().contiguous()
    return conditional


__all__ = [
    "M5_CONDITIONAL_ARTIFACT_ITEMS_DIR",
    "M5_CONDITIONAL_ARTIFACT_MANIFEST_NAME",
    "M5_CONDITIONAL_ARTIFACT_SCHEMA",
    "M5_CONDITIONAL_ITEM_SCHEMA",
    "M5ConditionalArtifactStore",
    "build_m5_conditional_artifact",
    "encode_m5_prompt_condition",
    "load_m5_conditional_artifact_manifest",
    "m5_conditional_artifact_sha256",
    "m5_conditional_item_name",
    "validate_m5_conditional_artifact_manifest",
]
