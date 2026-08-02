from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from utils.nf_sf_m3 import (
    M3_CHUNK_FRAMES,
    M3_DEPTHS,
    M3_DEPTH_WEIGHTS,
    M3_PARAMETER_GROUP_NAMES,
    M3TeacherSample,
    _validate_teacher_manifest,
    file_sha256,
    load_m3_teacher_sample,
    loss_dict_to_floats,
    selected_state_to_device,
    tensor_sha256,
)
from utils.nf_sf_tensors import DEFAULT_S_MAIN, DEFAULT_S_MCP, make_generator
from utils.nf_sf_training import prepare_nf_sf_noisy_batch, run_nf_sf_forward_loss


M4_SAMPLE_PLAN_SCHEMA = "nf_sf_m4_sample_plan_v1"
M4_VALIDATION_SCHEMA = "nf_sf_m4_validation_v1"
M4_PAIR_PLAN_SCHEMA = "nf_sf_m4_pair_plan_v1"
M4_PAIR_COMMANDS_SCHEMA = "nf_sf_m4_pair_commands_v1"
M4_PAIR_STATUS_SCHEMA = "nf_sf_m4_pair_status_v1"
M4_SAMPLE_ORDERING_RULE = (
    "split-specific stable sort by split_index, sample_index, sample_id, prompt_sha256"
)
M4_DEFAULT_TRAIN_SUBSET_SIZE = 16
M4_DEFAULT_VALIDATION_SUBSET_SIZE = 8
M4_DEFAULT_OPTIMIZER_STEPS = 100
M4_DEFAULT_TRAIN_SEED = 2026080101
M4_DEFAULT_PROBE_SEED = 2026080199
M4_DEFAULT_VALIDATION_SEED = 2026080188
M4_DEFAULT_TIMING_WARMUP_STEPS = 5
M4_DEFAULT_GRID_AUX_WEIGHT = 1.0
M4_VALIDATION_LOSS_KEYS = (
    "main_loss",
    "mcp_depth1_loss",
    "mcp_depth2_loss",
    "mcp_depth3_loss",
    "weighted_mcp_loss",
    "total_validation_loss",
)
M4_CHECKPOINT_REQUIRED_FIELDS = (
    "sample_plan_path",
    "sample_plan_sha256",
    "train_sample_identities",
    "validation_sample_identities",
    "validation_seed",
    "validation_steps",
    "fixed_decode_validation_identity",
    "ordering_rule",
)
M4_PAIR_LOCKED_FIELDS = (
    "python_executable",
    "train_script_path",
    "repository_root",
    "subprocess_cwd",
    "reference_checkpoint_path",
    "reference_checkpoint_sha256",
    "manifest_path",
    "manifest_sha256",
    "sample_plan_path",
    "sample_plan_sha256",
    "train_sample_identities",
    "validation_sample_identities",
    "fixed_decode_validation_identity",
    "optimizer_steps",
    "train_seed",
    "probe_seed",
    "validation_seed",
    "timestep_noise_contract",
    "depth_weights",
    "mcp1_grid_aux_weight",
    "mcp1_grid_timesteps",
    "optimizer_type",
    "backbone_lr",
    "patch_embedding_lr",
    "mcp_lr",
    "weight_decay",
    "dtype",
    "checkpoint_steps",
    "validation_steps",
    "timing_warmup_steps",
    "output_schema",
)


def canonical_json_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def write_m4_json(payload: Mapping[str, Any], path: Path | str) -> Path:
    path = Path(path)
    text = (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    )
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            pass
        raise
    return path


def load_teacher_manifest(manifest_path: Path | str) -> dict[str, Any]:
    path = Path(manifest_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise TypeError("teacher manifest must be a JSON object")
    _validate_teacher_manifest(manifest)
    return manifest


def m4_sample_identity_from_record(record: Mapping[str, Any]) -> str:
    return "|".join(
        (
            f"sample_index={int(record['sample_index'])}",
            f"sample_id={'' if record.get('sample_id') is None else str(record.get('sample_id'))}",
            f"split={str(record['split'])}",
            f"split_index={int(record['split_index'])}",
            f"prompt_sha256={str(record.get('prompt_sha256', ''))}",
        )
    )


def m4_sample_identity_from_metadata(metadata: Mapping[str, Any]) -> str:
    return "|".join(
        (
            f"sample_index={int(metadata['sample_index'])}",
            "sample_id="
            f"{'' if metadata.get('sample_id') is None else str(metadata.get('sample_id'))}",
            f"split={str(metadata['split'])}",
            f"split_index={int(metadata['split_index'])}",
            f"prompt_sha256={str(metadata.get('prompt_sha256', ''))}",
        )
    )


def _stable_record_sort_key(record: Mapping[str, Any]) -> tuple[int, int, str, str]:
    return (
        int(record["split_index"]),
        int(record["sample_index"]),
        "" if record.get("sample_id") is None else str(record.get("sample_id")),
        str(record.get("prompt_sha256", "")),
    )


def _record_plan_entry(record: Mapping[str, Any]) -> dict[str, Any]:
    entry = {
        "identity": m4_sample_identity_from_record(record),
        "sample_index": int(record["sample_index"]),
        "sample_id": None if record.get("sample_id") is None else str(record.get("sample_id")),
        "split": str(record["split"]),
        "split_index": int(record["split_index"]),
        "prompt_sha256": str(record.get("prompt_sha256", "")),
    }
    for key in ("file", "path", "payload_path", "artifact_path", "file_sha256"):
        if record.get(key) is not None:
            entry[key] = str(record[key])
    return entry


def _select_split_records(
    manifest: Mapping[str, Any],
    *,
    split: str,
    subset_size: int,
    explicit_identities: Sequence[str] | None,
) -> list[Mapping[str, Any]]:
    if subset_size <= 0:
        raise ValueError("M4 subset sizes must be positive")
    records = [
        dict(record)
        for record in manifest["samples"]
        if str(record.get("split")) == split
    ]
    if len(records) < subset_size and explicit_identities is None:
        raise RuntimeError(
            f"M4 sample plan requested {subset_size} {split} samples, "
            f"but manifest only has {len(records)}"
        )
    records = sorted(records, key=_stable_record_sort_key)
    identities = [m4_sample_identity_from_record(record) for record in records]
    if len(identities) != len(set(identities)):
        raise RuntimeError(f"M4 manifest has duplicate {split} sample identity")
    if explicit_identities is None:
        return records[:subset_size]

    requested = [str(identity) for identity in explicit_identities]
    if len(requested) != len(set(requested)):
        raise RuntimeError(f"M4 explicit {split} identity list has duplicates")
    selected = []
    for requested_identity in requested:
        matches = [
            record
            for record in records
            if requested_identity
            in {
                m4_sample_identity_from_record(record),
                "" if record.get("sample_id") is None else str(record.get("sample_id")),
                str(int(record["sample_index"])),
            }
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"M4 explicit {split} identity {requested_identity!r} "
                f"matched {len(matches)} records"
            )
        selected.append(matches[0])
    if len(selected) != subset_size:
        raise RuntimeError(
            f"M4 explicit {split} identity list has {len(selected)} records, "
            f"expected subset size {subset_size}"
        )
    return selected


def build_m4_sample_plan(
    *,
    manifest_path: Path | str,
    train_subset_size: int = M4_DEFAULT_TRAIN_SUBSET_SIZE,
    validation_subset_size: int = M4_DEFAULT_VALIDATION_SUBSET_SIZE,
    dataset_root: Path | str | None = None,
    explicit_train_identities: Sequence[str] | None = None,
    explicit_validation_identities: Sequence[str] | None = None,
) -> dict[str, Any]:
    manifest_path = Path(manifest_path)
    manifest = load_teacher_manifest(manifest_path)
    train_records = _select_split_records(
        manifest,
        split="train",
        subset_size=int(train_subset_size),
        explicit_identities=explicit_train_identities,
    )
    validation_records = _select_split_records(
        manifest,
        split="validation",
        subset_size=int(validation_subset_size),
        explicit_identities=explicit_validation_identities,
    )
    train_entries = [_record_plan_entry(record) for record in train_records]
    validation_entries = [_record_plan_entry(record) for record in validation_records]
    train_identities = [entry["identity"] for entry in train_entries]
    validation_identities = [entry["identity"] for entry in validation_entries]
    _validate_identity_sets(
        train_entries=train_entries,
        validation_entries=validation_entries,
    )
    plan = {
        "schema": M4_SAMPLE_PLAN_SCHEMA,
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": file_sha256(manifest_path),
        "dataset_root": None if dataset_root is None else str(Path(dataset_root).resolve()),
        "ordering_rule": M4_SAMPLE_ORDERING_RULE,
        "train_subset_size": int(train_subset_size),
        "validation_subset_size": int(validation_subset_size),
        "train_sample_identities": train_identities,
        "validation_sample_identities": validation_identities,
        "fixed_decode_validation_identity": validation_identities[0],
        "samples": {
            "train": train_entries,
            "validation": validation_entries,
        },
    }
    plan["sample_plan_sha256"] = m4_sample_plan_sha256(plan)
    validate_m4_sample_plan(plan)
    return plan


def _validate_identity_sets(
    *,
    train_entries: Sequence[Mapping[str, Any]],
    validation_entries: Sequence[Mapping[str, Any]],
) -> None:
    train_identities = [str(entry["identity"]) for entry in train_entries]
    validation_identities = [str(entry["identity"]) for entry in validation_entries]
    if len(train_identities) != len(set(train_identities)):
        raise RuntimeError("M4 train sample identities contain duplicates")
    if len(validation_identities) != len(set(validation_identities)):
        raise RuntimeError("M4 validation sample identities contain duplicates")
    if set(train_identities) & set(validation_identities):
        raise RuntimeError("M4 train and validation sample identities overlap")
    train_indices = {int(entry["sample_index"]) for entry in train_entries}
    validation_indices = {int(entry["sample_index"]) for entry in validation_entries}
    if train_indices & validation_indices:
        raise RuntimeError("M4 train and validation sample indices overlap")


def _plan_without_sha(plan: Mapping[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(dict(plan))
    value.pop("sample_plan_sha256", None)
    return value


def m4_sample_plan_sha256(plan: Mapping[str, Any]) -> str:
    return canonical_json_sha256(_plan_without_sha(plan))


def validate_m4_sample_plan(
    plan: Mapping[str, Any],
    *,
    manifest_path: Path | str | None = None,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    if plan.get("schema") != M4_SAMPLE_PLAN_SCHEMA:
        raise RuntimeError("M4 sample plan schema mismatch")
    for key in (
        "manifest_path",
        "manifest_sha256",
        "train_sample_identities",
        "validation_sample_identities",
        "fixed_decode_validation_identity",
        "samples",
    ):
        if key not in plan:
            raise RuntimeError(f"M4 sample plan missing {key!r}")
    train_entries = plan["samples"].get("train")
    validation_entries = plan["samples"].get("validation")
    if not isinstance(train_entries, list) or not train_entries:
        raise RuntimeError("M4 sample plan has no train samples")
    if not isinstance(validation_entries, list) or not validation_entries:
        raise RuntimeError("M4 sample plan has no validation samples")
    train_identities = [str(entry["identity"]) for entry in train_entries]
    validation_identities = [str(entry["identity"]) for entry in validation_entries]
    if train_identities != [str(value) for value in plan["train_sample_identities"]]:
        raise RuntimeError("M4 train identity list does not match sample entries")
    if validation_identities != [str(value) for value in plan["validation_sample_identities"]]:
        raise RuntimeError("M4 validation identity list does not match sample entries")
    _validate_identity_sets(
        train_entries=train_entries,
        validation_entries=validation_entries,
    )
    fixed_decode_identity = str(plan["fixed_decode_validation_identity"])
    if fixed_decode_identity != validation_identities[0]:
        raise RuntimeError("M4 fixed decode identity must be validation identity at index 0")
    if manifest_path is not None:
        manifest_path = Path(manifest_path).resolve()
        if str(manifest_path) != str(plan["manifest_path"]):
            raise RuntimeError("M4 sample plan manifest path differs")
        if file_sha256(manifest_path) != str(plan["manifest_sha256"]):
            raise RuntimeError("M4 sample plan manifest SHA256 differs")
    actual_sha = m4_sample_plan_sha256(plan)
    saved_sha = plan.get("sample_plan_sha256")
    if saved_sha is not None and str(saved_sha) != actual_sha:
        raise RuntimeError("M4 sample plan SHA256 does not match contents")
    if expected_sha256 is not None and str(expected_sha256) != actual_sha:
        raise RuntimeError("M4 sample plan SHA256 differs from expected value")
    return {
        "status": "PASS",
        "sample_plan_sha256": actual_sha,
        "train_subset_size": len(train_entries),
        "validation_subset_size": len(validation_entries),
        "fixed_decode_validation_identity": fixed_decode_identity,
    }


def write_m4_sample_plan(plan: Mapping[str, Any], path: Path | str) -> None:
    validate_m4_sample_plan(plan)
    write_m4_json(dict(plan), path)


def load_m4_sample_plan(
    path: Path | str,
    *,
    manifest_path: Path | str | None = None,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    plan_path = Path(path)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if not isinstance(plan, dict):
        raise TypeError("M4 sample plan must be a JSON object")
    validate_m4_sample_plan(
        plan,
        manifest_path=manifest_path,
        expected_sha256=expected_sha256,
    )
    return plan


def m4_sample_entry(plan: Mapping[str, Any], identity: str) -> dict[str, Any]:
    identity = str(identity)
    for split in ("train", "validation"):
        for entry in plan["samples"][split]:
            if str(entry["identity"]) == identity:
                return dict(entry)
    raise RuntimeError(f"M4 sample identity not in sample plan: {identity}")


def m4_validation_entry(plan: Mapping[str, Any], identity: str) -> dict[str, Any]:
    identity = str(identity)
    for entry in plan["samples"]["validation"]:
        if str(entry["identity"]) == identity:
            return dict(entry)
    raise RuntimeError(f"M4 decode identity must come from validation split: {identity}")


def m4_train_entry_for_step(plan: Mapping[str, Any], step: int) -> dict[str, Any]:
    step = int(step)
    if step <= 0:
        raise ValueError("M4 train step must be 1-based and positive")
    train_entries = plan["samples"]["train"]
    index = (step - 1) % len(train_entries)
    entry = dict(train_entries[index])
    entry["train_sample_position"] = int(index)
    entry["train_cycle_index"] = int((step - 1) // len(train_entries))
    return entry


def m4_next_train_entry_after_global_step(
    plan: Mapping[str, Any],
    global_step: int,
) -> dict[str, Any]:
    return m4_train_entry_for_step(plan, int(global_step) + 1)


def m4_train_identity_sequence(plan: Mapping[str, Any], steps: int) -> list[str]:
    return [
        str(m4_train_entry_for_step(plan, step)["identity"])
        for step in range(1, int(steps) + 1)
    ]


def load_m4_teacher_samples(
    plan: Mapping[str, Any],
    *,
    split: str,
    manifest_path: Path | str,
    dataset_root: Path | str | None,
    reference_checkpoint_path: Path | str | None,
) -> dict[str, M3TeacherSample]:
    samples = {}
    for entry in plan["samples"][split]:
        sample = load_m3_teacher_sample(
            manifest_path=manifest_path,
            dataset_root=dataset_root,
            sample_index=int(entry["sample_index"]),
            reference_checkpoint_path=reference_checkpoint_path,
        )
        identity = m4_sample_identity_from_metadata(sample.metadata)
        if identity != str(entry["identity"]):
            raise RuntimeError(
                f"M4 loaded sample identity mismatch: {identity} != {entry['identity']}"
            )
        samples[identity] = sample
    return samples


def parse_m4_step_list(
    value: str | Sequence[int] | None,
    *,
    optimizer_steps: int,
    name: str,
    require_zero: bool = False,
    require_final: bool = False,
) -> tuple[int, ...] | None:
    if value is None:
        return None
    if isinstance(value, str):
        raw_values = [] if value.strip() == "" else value.split(",")
    else:
        raw_values = list(value)
    steps = []
    for raw in raw_values:
        try:
            step = int(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a comma-separated list of integers") from exc
        if step < 0 or step > int(optimizer_steps):
            raise ValueError(f"{name} entries must satisfy 0 <= step <= optimizer_steps")
        steps.append(step)
    if len(steps) != len(set(steps)):
        raise ValueError(f"{name} must not contain duplicate steps")
    normalized = tuple(sorted(steps))
    if require_zero and 0 not in normalized:
        raise ValueError(f"{name} must include step 0")
    if require_final and int(optimizer_steps) not in normalized:
        raise ValueError(f"{name} must include final optimizer step {int(optimizer_steps)}")
    return normalized


def default_m4_validation_steps(optimizer_steps: int) -> tuple[int, ...]:
    optimizer_steps = int(optimizer_steps)
    if optimizer_steps == M4_DEFAULT_OPTIMIZER_STEPS:
        return (0, 20, 40, 60, 80, optimizer_steps)
    return (0, optimizer_steps)


def default_m4_checkpoint_steps(optimizer_steps: int) -> tuple[int, ...]:
    return (0, int(optimizer_steps))


def rng_state_digest(rng: torch.Generator | None) -> str | None:
    if rng is None:
        return None
    return tensor_sha256(rng.get_state().detach().cpu())


def tensor_state_digest(tensor: torch.Tensor | None) -> str | None:
    if tensor is None:
        return None
    return tensor_sha256(tensor.detach().cpu())


def global_cuda_rng_digest(device: torch.device) -> str | None:
    if device.type != "cuda" or not torch.cuda.is_available():
        return None
    return tensor_sha256(torch.cuda.get_rng_state(device).detach().cpu())


def derive_m4_validation_seed(
    *,
    base_seed: int,
    sample_identity: str,
    tensor_slot: str,
) -> int:
    payload = canonical_json_bytes(
        {
            "base_seed": int(base_seed),
            "sample_identity": str(sample_identity),
            "tensor_slot": str(tensor_slot),
        }
    )
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], byteorder="big")
    return int(value % (2**63 - 1))


def _requires_grad_snapshot(generator) -> list[dict[str, Any]]:
    return [
        {"name": str(name), "requires_grad": bool(parameter.requires_grad)}
        for name, parameter in generator.named_parameters()
    ]


def _grad_snapshot(generator) -> list[dict[str, Any]]:
    records = []
    for name, parameter in generator.named_parameters():
        grad = parameter.grad
        record = {
            "name": str(name),
            "has_grad": grad is not None,
            "shape": None,
            "dtype": None,
            "sha256": None,
        }
        if grad is not None:
            record.update(
                {
                    "shape": [int(dim) for dim in grad.shape],
                    "dtype": str(grad.dtype),
                    "sha256": tensor_sha256(grad.detach().cpu().reshape(-1)),
                }
            )
        records.append(record)
    return records


def _validation_tensor_contract(noisy_batch) -> dict[str, Any]:
    return {
        "timestep_main_sha256": tensor_sha256(noisy_batch.timestep_main.detach().cpu()),
        "timestep_depth_sha256s": [
            tensor_sha256(timestep.detach().cpu())
            for timestep in noisy_batch.timestep_depths
        ],
        "epsilon_main_sha256": tensor_sha256(noisy_batch.epsilon_main.detach().cpu()),
        "epsilon_depth_sha256s": [
            tensor_sha256(epsilon.detach().cpu())
            for epsilon in noisy_batch.epsilon_depths
        ],
        "target_flow_main_sha256": tensor_sha256(noisy_batch.target_flow_main.detach().cpu()),
        "target_flow_depth_sha256s": [
            tensor_sha256(target.detach().cpu())
            for target in noisy_batch.target_flow_depths
        ],
    }


def _validation_loss_record(losses: Mapping[str, float]) -> dict[str, float]:
    weighted_mcp = sum(
        float(M3_DEPTH_WEIGHTS[index]) * float(losses[f"mcp_depth{index + 1}_loss"])
        for index in range(len(M3_DEPTH_WEIGHTS))
    )
    total = float(losses["main_loss"]) + float(weighted_mcp)
    return {
        "main_loss": float(losses["main_loss"]),
        "mcp_depth1_loss": float(losses["mcp_depth1_loss"]),
        "mcp_depth2_loss": float(losses["mcp_depth2_loss"]),
        "mcp_depth3_loss": float(losses["mcp_depth3_loss"]),
        "weighted_mcp_loss": float(weighted_mcp),
        "total_validation_loss": float(total),
    }


def _json_safe_loss_record(losses: Mapping[str, float]) -> dict[str, float | None]:
    return {
        key: float(value) if math.isfinite(float(value)) else None
        for key, value in losses.items()
    }


def _nonfinite_loss_fields(losses: Mapping[str, Any]) -> list[str]:
    fields = []
    for key in M4_VALIDATION_LOSS_KEYS:
        try:
            value = float(losses[key])
        except (KeyError, TypeError, ValueError):
            fields.append(key)
            continue
        if not math.isfinite(value):
            fields.append(key)
    return fields


def _aggregate_validation_loss_values(
    loss_records: Sequence[Mapping[str, Any]],
) -> dict[str, float]:
    return {
        key: float(sum(float(losses[key]) for losses in loss_records) / len(loss_records))
        for key in M4_VALIDATION_LOSS_KEYS
    }


def _aggregate_validation_losses(per_sample: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    return _aggregate_validation_loss_values([sample["losses"] for sample in per_sample])


def validation_loss_finite_report(
    *,
    per_sample_losses: Sequence[Mapping[str, Any]],
    aggregate_losses: Mapping[str, Any] | None,
) -> dict[str, Any]:
    nonfinite = []
    for sample in per_sample_losses:
        fields = _nonfinite_loss_fields(sample["losses"])
        if fields:
            nonfinite.append(
                {
                    "scope": "sample",
                    "sample_identity": str(sample.get("sample_identity", "")),
                    "fields": fields,
                }
            )
    if aggregate_losses is None:
        nonfinite.append(
            {
                "scope": "aggregate",
                "sample_identity": None,
                "fields": list(M4_VALIDATION_LOSS_KEYS),
            }
        )
    else:
        fields = _nonfinite_loss_fields(aggregate_losses)
        if fields:
            nonfinite.append(
                {
                    "scope": "aggregate",
                    "sample_identity": None,
                    "fields": fields,
                }
            )
    return {
        "contract_pass": not nonfinite,
        "nonfinite_validation_loss_count": int(len(nonfinite)),
        "nonfinite_validation_losses": nonfinite,
    }


def run_m4_validation(
    *,
    generator,
    samples: Sequence[M3TeacherSample],
    conditional_dicts: Mapping[str, Mapping[str, Any]],
    scheduler_main,
    scheduler_mcp,
    device: torch.device,
    dtype: torch.dtype,
    mode: str,
    global_step: int,
    sample_plan: Mapping[str, Any],
    validation_seed: int,
    train_rng: torch.Generator | None,
    probe_rng_state: torch.Tensor | None,
    model_identity: Mapping[str, Any],
    depth_weights: Sequence[float] = M3_DEPTH_WEIGHTS,
) -> dict[str, Any]:
    if not samples:
        raise ValueError("M4 validation requires at least one sample")
    plan_sha = m4_sample_plan_sha256(sample_plan)
    expected_validation_identities = set(
        str(identity) for identity in sample_plan["validation_sample_identities"]
    )
    before_training = bool(generator.training)
    requires_grad_before = _requires_grad_snapshot(generator)
    grad_before = _grad_snapshot(generator)
    train_rng_before = rng_state_digest(train_rng)
    probe_rng_before = tensor_state_digest(probe_rng_state)
    global_cpu_rng_before = tensor_sha256(torch.random.get_rng_state())
    global_cuda_rng_before = global_cuda_rng_digest(device)
    per_sample = []
    raw_loss_records = []
    nonfinite_loss_records = []
    try:
        generator.eval()
        with torch.no_grad():
            for position, sample in enumerate(samples):
                identity = m4_sample_identity_from_metadata(sample.metadata)
                if identity not in expected_validation_identities:
                    raise RuntimeError(
                        f"M4 validation sample is not from validation plan: {identity}"
                    )
                state = selected_state_to_device(sample.selected_state, device=device, dtype=dtype)
                seed = derive_m4_validation_seed(
                    base_seed=validation_seed,
                    sample_identity=identity,
                    tensor_slot="nf_sf_random_joint_loss",
                )
                validation_rng = make_generator(seed, device)
                noisy_batch = prepare_nf_sf_noisy_batch(
                    state,
                    scheduler_main=scheduler_main,
                    scheduler_mcp=scheduler_mcp,
                    rng=validation_rng,
                    chunk_frames=M3_CHUNK_FRAMES,
                    depths=M3_DEPTHS,
                    s_main=DEFAULT_S_MAIN,
                    s_mcp=DEFAULT_S_MCP,
                )
                result = run_nf_sf_forward_loss(
                    generator,
                    conditional_dict=dict(conditional_dicts[identity]),
                    noisy_batch=noisy_batch,
                    depth_weights=depth_weights,
                )
                raw_losses = _validation_loss_record(loss_dict_to_floats(result.losses))
                raw_loss_records.append(raw_losses)
                loss_fields = _nonfinite_loss_fields(raw_losses)
                if loss_fields:
                    nonfinite_loss_records.append(
                        {
                            "scope": "sample",
                            "sample_identity": identity,
                            "fields": loss_fields,
                        }
                    )
                losses = _json_safe_loss_record(raw_losses)
                per_sample.append(
                    {
                        "sample_identity": identity,
                        "sample_position": int(position),
                        "validation_seed": int(seed),
                        "losses": losses,
                        "tensor_contract": _validation_tensor_contract(noisy_batch),
                    }
                )
    finally:
        generator.train(before_training)
    train_rng_after = rng_state_digest(train_rng)
    probe_rng_after = tensor_state_digest(probe_rng_state)
    global_cpu_rng_after = tensor_sha256(torch.random.get_rng_state())
    global_cuda_rng_after = global_cuda_rng_digest(device)
    requires_grad_after = _requires_grad_snapshot(generator)
    grad_after = _grad_snapshot(generator)
    gradients_unchanged = grad_before == grad_after
    requires_grad_unchanged = requires_grad_before == requires_grad_after
    model_mode_after = bool(generator.training)
    aggregate_losses: dict[str, float | None]
    if nonfinite_loss_records:
        aggregate_losses = {key: None for key in M4_VALIDATION_LOSS_KEYS}
        nonfinite_loss_records.append(
            {
                "scope": "aggregate",
                "sample_identity": None,
                "fields": list(M4_VALIDATION_LOSS_KEYS),
            }
        )
    else:
        raw_aggregate_losses = _aggregate_validation_loss_values(raw_loss_records)
        aggregate_fields = _nonfinite_loss_fields(raw_aggregate_losses)
        if aggregate_fields:
            nonfinite_loss_records.append(
                {
                    "scope": "aggregate",
                    "sample_identity": None,
                    "fields": aggregate_fields,
                }
            )
        aggregate_losses = _json_safe_loss_record(raw_aggregate_losses)
    loss_finite_contract = not nonfinite_loss_records
    status_pass = (
        gradients_unchanged
        and requires_grad_unchanged
        and model_mode_after == before_training
        and train_rng_before == train_rng_after
        and probe_rng_before == probe_rng_after
        and global_cpu_rng_before == global_cpu_rng_after
        and global_cuda_rng_before == global_cuda_rng_after
        and loss_finite_contract
    )
    return {
        "schema": M4_VALIDATION_SCHEMA,
        "status": "PASS" if status_pass else "FAIL",
        "global_step": int(global_step),
        "mode": str(mode),
        "model_identity": dict(model_identity),
        "sample_plan_sha256": plan_sha,
        "validation_seed_contract": {
            "base_seed": int(validation_seed),
            "derivation": "sha256(base_seed, sample_identity, tensor_slot)",
            "tensor_slot": "nf_sf_random_joint_loss",
        },
        "validation_sample_identities": [sample["sample_identity"] for sample in per_sample],
        "sample_count": int(len(per_sample)),
        "per_sample_losses": per_sample,
        "aggregate_losses": aggregate_losses,
        "validation_loss_finite_contract": bool(loss_finite_contract),
        "nonfinite_validation_loss_count": int(len(nonfinite_loss_records)),
        "nonfinite_validation_losses": nonfinite_loss_records,
        "train_rng_before_digest": train_rng_before,
        "train_rng_after_digest": train_rng_after,
        "probe_rng_before_digest": probe_rng_before,
        "probe_rng_after_digest": probe_rng_after,
        "global_cpu_rng_before_digest": global_cpu_rng_before,
        "global_cpu_rng_after_digest": global_cpu_rng_after,
        "global_cuda_rng_before_digest": global_cuda_rng_before,
        "global_cuda_rng_after_digest": global_cuda_rng_after,
        "model_mode_before": "train" if before_training else "eval",
        "model_mode_after": "train" if model_mode_after else "eval",
        "gradients_unchanged_contract": bool(gradients_unchanged),
        "requires_grad_unchanged_contract": bool(requires_grad_unchanged),
        "train_rng_unchanged_contract": bool(train_rng_before == train_rng_after),
        "probe_rng_unchanged_contract": bool(probe_rng_before == probe_rng_after),
        "global_cpu_rng_unchanged_contract": bool(
            global_cpu_rng_before == global_cpu_rng_after
        ),
        "global_cuda_rng_unchanged_contract": bool(
            global_cuda_rng_before == global_cuda_rng_after
        ),
    }


def validate_m4_checkpoint_sample_plan(
    checkpoint_payload: Mapping[str, Any],
    sample_plan: Mapping[str, Any],
    *,
    sample_plan_path: Path | str | None = None,
    expected_validation_seed: int | None = None,
    expected_validation_steps: Sequence[int] | None = None,
) -> dict[str, Any]:
    resolved_config = checkpoint_payload.get("resolved_config", {})
    if not isinstance(resolved_config, Mapping):
        raise RuntimeError("M4 checkpoint resolved_config must be a mapping")
    if "m4" not in resolved_config:
        return {"status": "LEGACY_M3", "is_m4": False}
    m4_config = resolved_config["m4"]
    if not isinstance(m4_config, Mapping):
        raise RuntimeError("M4 checkpoint m4 config must be a mapping")
    for key in M4_CHECKPOINT_REQUIRED_FIELDS:
        if key not in m4_config:
            raise RuntimeError(f"M4 checkpoint m4 config missing {key!r}")
    plan_sha = m4_sample_plan_sha256(sample_plan)
    checkpoint_sha = m4_config["sample_plan_sha256"]
    if not isinstance(checkpoint_sha, str):
        raise RuntimeError("M4 checkpoint sample_plan_sha256 must be a string")
    if checkpoint_sha != plan_sha:
        raise RuntimeError("M4 checkpoint sample plan SHA256 differs")
    if sample_plan_path is not None:
        checkpoint_plan_path = Path(str(m4_config["sample_plan_path"])).resolve()
        expected_plan_path = Path(sample_plan_path).resolve()
        if checkpoint_plan_path != expected_plan_path:
            raise RuntimeError("M4 checkpoint sample plan path differs")
    if not isinstance(m4_config["sample_plan_path"], str):
        raise RuntimeError("M4 checkpoint sample_plan_path must be a string")
    train_identities = m4_config["train_sample_identities"]
    validation_identities = m4_config["validation_sample_identities"]
    if not isinstance(train_identities, list) or not all(
        isinstance(value, str) for value in train_identities
    ):
        raise RuntimeError("M4 checkpoint train identities must be a string list")
    if not isinstance(validation_identities, list) or not all(
        isinstance(value, str) for value in validation_identities
    ):
        raise RuntimeError("M4 checkpoint validation identities must be a string list")
    if train_identities != list(sample_plan["train_sample_identities"]):
        raise RuntimeError("M4 checkpoint train identities differ from sample plan")
    if validation_identities != list(sample_plan["validation_sample_identities"]):
        raise RuntimeError("M4 checkpoint validation identities differ from sample plan")
    if not isinstance(m4_config["validation_seed"], int):
        raise RuntimeError("M4 checkpoint validation_seed must be an integer")
    if (
        expected_validation_seed is not None
        and int(m4_config["validation_seed"]) != int(expected_validation_seed)
    ):
        raise RuntimeError("M4 checkpoint validation seed differs")
    validation_steps = m4_config["validation_steps"]
    if not isinstance(validation_steps, list) or not all(
        isinstance(value, int) for value in validation_steps
    ):
        raise RuntimeError("M4 checkpoint validation_steps must be an integer list")
    if (
        expected_validation_steps is not None
        and validation_steps != [int(value) for value in expected_validation_steps]
    ):
        raise RuntimeError("M4 checkpoint validation steps differ")
    fixed_decode_identity = m4_config["fixed_decode_validation_identity"]
    if not isinstance(fixed_decode_identity, str):
        raise RuntimeError("M4 checkpoint fixed decode identity must be a string")
    if fixed_decode_identity != str(sample_plan["validation_sample_identities"][0]):
        raise RuntimeError("M4 checkpoint fixed decode identity differs from sample plan")
    ordering_rule = m4_config["ordering_rule"]
    if not isinstance(ordering_rule, str):
        raise RuntimeError("M4 checkpoint ordering_rule must be a string")
    if ordering_rule != M4_SAMPLE_ORDERING_RULE:
        raise RuntimeError("M4 checkpoint ordering rule differs")
    return {
        "status": "PASS",
        "is_m4": True,
        "sample_plan_sha256": plan_sha,
        "sample_plan_path": str(m4_config["sample_plan_path"]),
        "fixed_decode_validation_identity": fixed_decode_identity,
        "train_sample_identities": list(train_identities),
        "validation_sample_identities": list(validation_identities),
        "validation_seed": int(m4_config["validation_seed"]),
        "validation_steps": list(validation_steps),
        "ordering_rule": ordering_rule,
    }


def validate_m4_decode_identity(
    *,
    sample_plan: Mapping[str, Any],
    identity: str,
) -> dict[str, Any]:
    entry = m4_validation_entry(sample_plan, identity)
    return {
        "status": "PASS",
        "sample_identity": str(identity),
        "sample_index": int(entry["sample_index"]),
        "split": str(entry["split"]),
        "split_index": int(entry["split_index"]),
        "sample_id": entry.get("sample_id"),
    }


def validate_m4_pair_contract(pair_plan: Mapping[str, Any]) -> dict[str, Any]:
    if pair_plan.get("schema") != M4_PAIR_PLAN_SCHEMA:
        raise RuntimeError("M4 pair plan schema mismatch")
    shared = pair_plan.get("shared_arguments")
    if not isinstance(shared, Mapping):
        raise RuntimeError("M4 pair plan requires shared_arguments")
    for field in M4_PAIR_LOCKED_FIELDS:
        if field not in shared:
            raise RuntimeError(f"M4 pair contract missing locked field {field!r}")
    if Path(str(shared["subprocess_cwd"])).resolve() != Path(
        str(shared["repository_root"])
    ).resolve():
        raise RuntimeError("M4 pair subprocess cwd must match repository root")
    runs = pair_plan.get("runs")
    if not isinstance(runs, Mapping) or set(runs.keys()) != {"frozen", "joint"}:
        raise RuntimeError("M4 pair plan requires frozen and joint runs")
    frozen = runs["frozen"]
    joint = runs["joint"]
    if frozen.get("mode") != "frozen" or joint.get("mode") != "joint":
        raise RuntimeError("M4 pair plan run modes are invalid")
    allowed = {"mode", "output_dir", "run_label", "argv"}
    differences = []
    for run_name, run in (("frozen", frozen), ("joint", joint)):
        if not isinstance(run, Mapping):
            raise RuntimeError(f"M4 pair {run_name} run must be a mapping")
        for field in M4_PAIR_LOCKED_FIELDS:
            if field not in run:
                raise RuntimeError(
                    f"M4 pair {run_name} run missing locked field {field!r}"
                )
            if run[field] != shared[field]:
                raise RuntimeError(
                    f"M4 pair {run_name} run locked field differs: {field}"
                )
    for key in sorted(set(frozen.keys()) | set(joint.keys())):
        if key in allowed:
            continue
        if key not in frozen or key not in joint or frozen[key] != joint[key]:
            differences.append(key)
    if not isinstance(frozen.get("argv"), list) or not isinstance(joint.get("argv"), list):
        raise RuntimeError("M4 pair commands must be argv lists")
    argv_difference = _first_disallowed_argv_difference(frozen["argv"], joint["argv"])
    if differences:
        raise RuntimeError(
            "M4 pair contract differs outside allowed fields: "
            f"fields={differences}"
        )
    if argv_difference is not None:
        raise RuntimeError(
            "M4 pair argv differs outside allowed fields: "
            f"index={argv_difference['index']}, frozen={argv_difference['frozen']!r}, "
            f"joint={argv_difference['joint']!r}, reason={argv_difference['reason']}"
        )
    return {
        "status": "PASS",
        "allowed_differences": sorted(allowed),
        "allowed_argv_value_options": ["--mode", "--output_dir"],
        "shared_argument_count": int(len(pair_plan.get("shared_arguments", {}))),
        "subprocess_cwd": str(shared["subprocess_cwd"]),
        "python_executable": str(shared["python_executable"]),
    }


def _first_disallowed_argv_difference(
    frozen_argv: Sequence[Any],
    joint_argv: Sequence[Any],
) -> dict[str, Any] | None:
    frozen_tokens = [str(value) for value in frozen_argv]
    joint_tokens = [str(value) for value in joint_argv]
    if len(frozen_tokens) != len(joint_tokens):
        return {
            "index": min(len(frozen_tokens), len(joint_tokens)),
            "frozen": None if len(frozen_tokens) <= len(joint_tokens) else frozen_tokens[
                len(joint_tokens)
            ],
            "joint": None if len(joint_tokens) <= len(frozen_tokens) else joint_tokens[
                len(frozen_tokens)
            ],
            "reason": "argv length differs",
        }
    allowed_value_options = {"--mode", "--output_dir"}
    for index, (frozen_token, joint_token) in enumerate(zip(frozen_tokens, joint_tokens)):
        if frozen_token == joint_token:
            continue
        previous_frozen = frozen_tokens[index - 1] if index > 0 else None
        previous_joint = joint_tokens[index - 1] if index > 0 else None
        if previous_frozen == previous_joint and previous_frozen in allowed_value_options:
            continue
        return {
            "index": int(index),
            "frozen": frozen_token,
            "joint": joint_token,
            "reason": "token differs",
        }
    return None
