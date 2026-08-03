from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch

from utils.nf_sf_m3 import M3_DEPTH_WEIGHTS
from utils.nf_sf_m4 import (
    M4_VALIDATION_LOSS_KEYS,
    M4_VALIDATION_SCHEMA,
    run_m4_validation,
)
from utils.nf_sf_m5_conditionals import M5ConditionalArtifactStore
from utils.nf_sf_m5_formal_plan import (
    M5_FORMAL_VALIDATION_SAMPLE_COUNT,
    validate_m5_formal_sample_plan,
)
from utils.nf_sf_m5_samples import M5TeacherSampleStore

M5_STREAMING_VALIDATION_SCHEMA = "nf_sf_m5_streaming_validation_v1"

_CHILD_CONTRACT_KEYS = (
    "gradients_unchanged_contract",
    "requires_grad_unchanged_contract",
    "train_rng_unchanged_contract",
    "probe_rng_unchanged_contract",
    "global_cpu_rng_unchanged_contract",
    "global_cuda_rng_unchanged_contract",
)


def run_m5_streaming_validation(
    *,
    generator: Any,
    teacher_store: M5TeacherSampleStore,
    conditional_store: M5ConditionalArtifactStore,
    sample_plan: Mapping[str, Any],
    scheduler_main: Any,
    scheduler_mcp: Any,
    device: torch.device | str,
    dtype: torch.dtype,
    mode: str,
    global_step: int,
    validation_seed: int,
    train_rng: torch.Generator | None,
    probe_rng_state: torch.Tensor | None,
    model_identity: Mapping[str, Any],
    depth_weights: Sequence[float] = M3_DEPTH_WEIGHTS,
) -> dict[str, Any]:
    contract = _validate_streaming_contract(
        sample_plan=sample_plan,
        teacher_store=teacher_store,
        conditional_store=conditional_store,
    )
    target_device = torch.device(device)
    if not isinstance(dtype, torch.dtype):
        raise TypeError(f"dtype must be a torch.dtype, actual={type(dtype).__name__}")

    validation_identities = tuple(contract["validation_identities"])
    teacher_before = _teacher_telemetry(teacher_store)
    conditional_before = _conditional_telemetry(conditional_store)
    per_sample_losses: list[dict[str, Any]] = []
    loss_records: list[dict[str, float | None]] = []
    nonfinite_records: list[dict[str, Any]] = []
    child_failures: list[dict[str, Any]] = []
    all_child_contracts_pass = True
    max_live_teacher_samples = 0
    max_live_conditionals = 0

    for position, identity in enumerate(validation_identities):
        sample = None
        cpu_conditional = None
        device_conditional: dict[str, torch.Tensor] | None = None
        child_samples: list[Any] = []
        child_conditionals: dict[str, dict[str, torch.Tensor]] = {}
        try:
            with teacher_store.acquire(identity) as sample:
                max_live_teacher_samples = max(
                    max_live_teacher_samples,
                    _teacher_telemetry(teacher_store)["live_count"],
                )
                with conditional_store.acquire(identity) as cpu_conditional:
                    max_live_conditionals = max(
                        max_live_conditionals,
                        _conditional_telemetry(conditional_store)["live_count"],
                    )
                    device_conditional = _conditional_to_device(
                        cpu_conditional,
                        device=target_device,
                        dtype=dtype,
                        identity=identity,
                    )
                    child_samples.append(sample)
                    child_conditionals[identity] = device_conditional
                    child_report = run_m4_validation(
                        generator=generator,
                        samples=child_samples,
                        conditional_dicts=child_conditionals,
                        scheduler_main=scheduler_main,
                        scheduler_mcp=scheduler_mcp,
                        device=target_device,
                        dtype=dtype,
                        mode=mode,
                        global_step=global_step,
                        sample_plan=sample_plan,
                        validation_seed=validation_seed,
                        train_rng=train_rng,
                        probe_rng_state=probe_rng_state,
                        model_identity=model_identity,
                        depth_weights=depth_weights,
                    )

            child_contract_pass = _child_contract_pass(
                child_report=child_report,
                identity=identity,
                position=position,
                global_step=global_step,
            )
            if not child_contract_pass:
                all_child_contracts_pass = False
                child_failures.append(
                    {
                        "sample_identity": identity,
                        "sample_position": position,
                        "status": str(child_report.get("status")),
                        "failed_contracts": _failed_child_contracts(child_report),
                    }
                )
            sample_record, loss_record, nonfinite_fields = _extract_child_sample_loss(
                child_report,
                identity=identity,
                position=position,
            )
            per_sample_losses.append(sample_record)
            loss_records.append(loss_record)
            if nonfinite_fields:
                nonfinite_records.append(
                    {
                        "scope": "sample",
                        "sample_identity": identity,
                        "fields": nonfinite_fields,
                    }
                )
        finally:
            child_conditionals.clear()
            child_samples.clear()
            if device_conditional is not None:
                device_conditional.clear()
            device_conditional = None
            cpu_conditional = None
            sample = None
            child_report = None

    teacher_after = _teacher_telemetry(teacher_store)
    conditional_after = _conditional_telemetry(conditional_store)
    teacher_delta = _telemetry_delta(teacher_before, teacher_after)
    conditional_delta = _telemetry_delta(conditional_before, conditional_after)
    aggregate_losses = _aggregate_losses(loss_records, nonfinite=bool(nonfinite_records))
    validation_loss_finite_contract = not nonfinite_records
    sample_count = len(per_sample_losses)
    expected_count = M5_FORMAL_VALIDATION_SAMPLE_COUNT
    store_contract_pass = (
        teacher_delta["successful_load_count_delta"] == expected_count
        and conditional_delta["successful_load_count_delta"] == expected_count
        and teacher_after["live_count"] == 0
        and conditional_after["live_count"] == 0
        and max_live_teacher_samples <= 1
        and max_live_conditionals <= 1
    )
    status_pass = (
        sample_count == expected_count
        and validation_loss_finite_contract
        and all_child_contracts_pass
        and store_contract_pass
        and [item["sample_identity"] for item in per_sample_losses]
        == list(validation_identities)
    )
    report = {
        "schema": M5_STREAMING_VALIDATION_SCHEMA,
        "status": "PASS" if status_pass else "FAIL",
        "global_step": _require_python_int(global_step, "global_step"),
        "mode": str(mode),
        "model_identity": _json_safe_value(dict(model_identity), "model_identity"),
        "sample_plan_sha256": str(contract["sample_plan_sha256"]),
        "teacher_manifest_sha256": str(contract["teacher_manifest_sha256"]),
        "conditional_artifact_sha256": str(conditional_store.artifact_sha256),
        "validation_seed": _require_python_int(validation_seed, "validation_seed"),
        "validation_sample_identities": list(validation_identities),
        "sample_count": sample_count,
        "per_sample_losses": per_sample_losses,
        "aggregate_losses": aggregate_losses,
        "validation_loss_finite_contract": bool(validation_loss_finite_contract),
        "nonfinite_validation_losses": nonfinite_records,
        "teacher_store_telemetry_delta": teacher_delta,
        "conditional_store_telemetry_delta": conditional_delta,
        "max_live_teacher_samples": max_live_teacher_samples,
        "max_live_conditionals": max_live_conditionals,
        "all_child_m4_contracts_pass": bool(all_child_contracts_pass),
        "child_m4_failures": child_failures,
    }
    _assert_no_tensors(report, "report")
    return report


def _validate_streaming_contract(
    *,
    sample_plan: Mapping[str, Any],
    teacher_store: M5TeacherSampleStore,
    conditional_store: M5ConditionalArtifactStore,
) -> dict[str, Any]:
    plan_sha = _require_nonempty_string(
        sample_plan.get("sample_plan_sha256"),
        "sample_plan.sample_plan_sha256",
    )
    formal_report = validate_m5_formal_sample_plan(
        sample_plan,
        expected_sha256=plan_sha,
    )
    train_identities = _string_tuple(
        sample_plan["train_sample_identities"],
        "sample_plan.train_sample_identities",
    )
    validation_identities = _string_tuple(
        sample_plan["validation_sample_identities"],
        "sample_plan.validation_sample_identities",
    )
    if len(validation_identities) != M5_FORMAL_VALIDATION_SAMPLE_COUNT:
        raise RuntimeError(
            "validation_sample_identities count mismatch: "
            f"expected={M5_FORMAL_VALIDATION_SAMPLE_COUNT}, "
            f"actual={len(validation_identities)}"
        )
    if len(validation_identities) != len(set(validation_identities)):
        raise RuntimeError("validation_sample_identities contain duplicates")
    if set(validation_identities) & set(train_identities):
        raise RuntimeError("validation identities overlap train identities")
    if teacher_store.sample_plan_sha256 != plan_sha:
        raise RuntimeError(
            "teacher_store.sample_plan_sha256 mismatch: "
            f"expected={plan_sha}, actual={teacher_store.sample_plan_sha256}"
        )
    if conditional_store.sample_plan_sha256 != plan_sha:
        raise RuntimeError(
            "conditional_store.sample_plan_sha256 mismatch: "
            f"expected={plan_sha}, actual={conditional_store.sample_plan_sha256}"
        )
    teacher_manifest_sha = str(formal_report["manifest_sha256"])
    if teacher_store.manifest_sha256 != teacher_manifest_sha:
        raise RuntimeError(
            "teacher_store.manifest_sha256 mismatch: "
            f"expected={teacher_manifest_sha}, actual={teacher_store.manifest_sha256}"
        )
    if conditional_store.teacher_manifest_sha256 != teacher_manifest_sha:
        raise RuntimeError(
            "conditional_store.teacher_manifest_sha256 mismatch: "
            f"expected={teacher_manifest_sha}, "
            f"actual={conditional_store.teacher_manifest_sha256}"
        )
    _require_identity_order(
        teacher_store.validation_identities,
        validation_identities,
        "teacher_store.validation_identities",
    )
    _require_identity_order(
        conditional_store.validation_identities,
        validation_identities,
        "conditional_store.validation_identities",
    )
    return {
        "sample_plan_sha256": plan_sha,
        "teacher_manifest_sha256": teacher_manifest_sha,
        "validation_identities": validation_identities,
    }


def _conditional_to_device(
    conditional: Mapping[str, torch.Tensor],
    *,
    device: torch.device,
    dtype: torch.dtype,
    identity: str,
) -> dict[str, torch.Tensor]:
    if not isinstance(conditional, Mapping):
        raise TypeError(f"conditional for {identity} must be a mapping")
    moved: dict[str, torch.Tensor] = {}
    for key, tensor in conditional.items():
        if not isinstance(key, str) or key.strip() == "":
            raise TypeError(f"conditional for {identity} has a non-string tensor key")
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"conditional {identity}.{key} must be a torch.Tensor")
        moved[key] = tensor.to(device=device, dtype=dtype)
    if not moved:
        raise RuntimeError(f"conditional for {identity} must contain at least one tensor")
    return moved


def _child_contract_pass(
    *,
    child_report: Mapping[str, Any],
    identity: str,
    position: int,
    global_step: int,
) -> bool:
    if not isinstance(child_report, Mapping):
        raise TypeError("run_m4_validation must return a mapping")
    if child_report.get("schema") != M4_VALIDATION_SCHEMA:
        raise RuntimeError(f"child M4 validation schema mismatch for {identity}")
    if child_report.get("global_step") != global_step:
        raise RuntimeError(f"child M4 validation global_step mismatch for {identity}")
    if child_report.get("sample_count") != 1:
        raise RuntimeError(f"child M4 validation sample_count mismatch for {identity}")
    child_identities = _string_tuple(
        child_report.get("validation_sample_identities"),
        "child.validation_sample_identities",
    )
    if child_identities != (identity,):
        raise RuntimeError(f"child M4 validation identity mismatch for {identity}")
    per_sample = child_report.get("per_sample_losses")
    if not isinstance(per_sample, list) or len(per_sample) != 1:
        raise RuntimeError(f"child M4 validation per_sample_losses mismatch for {identity}")
    sample_record = per_sample[0]
    if not isinstance(sample_record, Mapping):
        raise TypeError(f"child per_sample_losses[{position}] must be a mapping")
    if sample_record.get("sample_identity") != identity:
        raise RuntimeError(f"child per-sample identity mismatch for {identity}")
    return child_report.get("status") == "PASS" and all(
        child_report.get(key) is True for key in _CHILD_CONTRACT_KEYS
    )


def _failed_child_contracts(child_report: Mapping[str, Any]) -> list[str]:
    failures = []
    if child_report.get("status") != "PASS":
        failures.append("status")
    for key in _CHILD_CONTRACT_KEYS:
        if child_report.get(key) is not True:
            failures.append(key)
    return failures


def _extract_child_sample_loss(
    child_report: Mapping[str, Any],
    *,
    identity: str,
    position: int,
) -> tuple[dict[str, Any], dict[str, float | None], list[str]]:
    per_sample = child_report["per_sample_losses"]
    if not isinstance(per_sample, list) or len(per_sample) != 1:
        raise RuntimeError(f"child per_sample_losses malformed for {identity}")
    sample_record = per_sample[0]
    if not isinstance(sample_record, Mapping):
        raise TypeError(f"child per_sample_losses[{position}] must be a mapping")
    losses = sample_record.get("losses")
    if not isinstance(losses, Mapping):
        raise TypeError(f"child losses for {identity} must be a mapping")
    loss_record, nonfinite_fields = _loss_record(losses, identity=identity)
    sample_record_copy = dict(sample_record)
    sample_record_copy["losses"] = loss_record
    safe_record = _json_safe_value(
        sample_record_copy,
        f"per_sample_losses[{position}]",
    )
    if not isinstance(safe_record, dict):
        raise TypeError(f"per_sample_losses[{position}] must serialize as an object")
    safe_record["sample_identity"] = identity
    safe_record["sample_position"] = position
    return safe_record, loss_record, nonfinite_fields


def _loss_record(
    losses: Mapping[str, Any],
    *,
    identity: str,
) -> tuple[dict[str, float | None], list[str]]:
    record: dict[str, float | None] = {}
    nonfinite_fields: list[str] = []
    for key in M4_VALIDATION_LOSS_KEYS:
        if key not in losses:
            raise RuntimeError(f"child losses for {identity} missing {key}")
        value = losses[key]
        if isinstance(value, bool):
            record[key] = None
            nonfinite_fields.append(key)
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            record[key] = None
            nonfinite_fields.append(key)
            continue
        if not math.isfinite(number):
            record[key] = None
            nonfinite_fields.append(key)
            continue
        record[key] = number
    return record, nonfinite_fields


def _aggregate_losses(
    loss_records: Sequence[Mapping[str, float | None]],
    *,
    nonfinite: bool,
) -> dict[str, float | None]:
    if nonfinite:
        return {key: None for key in M4_VALIDATION_LOSS_KEYS}
    if not loss_records:
        raise RuntimeError("cannot aggregate zero validation losses")
    return {
        key: float(sum(float(record[key]) for record in loss_records) / len(loss_records))
        for key in M4_VALIDATION_LOSS_KEYS
    }


def _teacher_telemetry(store: M5TeacherSampleStore) -> dict[str, int]:
    return {
        "live_count": _require_int(store.live_sample_count, "teacher.live_sample_count"),
        "max_live_count": _require_int(
            store.max_live_sample_count,
            "teacher.max_live_sample_count",
        ),
        "load_attempt_count": _require_int(
            store.load_attempt_count,
            "teacher.load_attempt_count",
        ),
        "successful_load_count": _require_int(
            store.total_load_count,
            "teacher.total_load_count",
        ),
    }


def _conditional_telemetry(store: M5ConditionalArtifactStore) -> dict[str, int]:
    return {
        "live_count": _require_int(
            store.live_conditional_count,
            "conditional.live_conditional_count",
        ),
        "max_live_count": _require_int(
            store.max_live_conditional_count,
            "conditional.max_live_conditional_count",
        ),
        "load_attempt_count": _require_int(
            store.load_attempt_count,
            "conditional.load_attempt_count",
        ),
        "successful_load_count": _require_int(
            store.total_load_count,
            "conditional.total_load_count",
        ),
    }


def _telemetry_delta(
    before: Mapping[str, int],
    after: Mapping[str, int],
) -> dict[str, int]:
    return {
        "load_attempt_count_delta": after["load_attempt_count"]
        - before["load_attempt_count"],
        "successful_load_count_delta": after["successful_load_count"]
        - before["successful_load_count"],
        "live_count_before": before["live_count"],
        "live_count_after": after["live_count"],
    }


def _require_identity_order(
    actual_values: Sequence[str],
    expected_values: Sequence[str],
    field_path: str,
) -> None:
    actual = _string_tuple(actual_values, field_path)
    expected = tuple(expected_values)
    if len(actual) != len(set(actual)):
        raise RuntimeError(f"{field_path} contains duplicates")
    if len(actual) != len(expected):
        raise RuntimeError(
            f"{field_path} count mismatch: expected={len(expected)}, actual={len(actual)}"
        )
    if actual != expected:
        raise RuntimeError(f"{field_path} order mismatch")


def _string_tuple(value: Any, field_path: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise TypeError(f"{field_path} must be a string sequence")
    values = tuple(value)
    if not all(isinstance(item, str) for item in values):
        raise TypeError(f"{field_path} must contain only strings")
    return values


def _require_nonempty_string(value: Any, field_path: str) -> str:
    if not isinstance(value, str) or value.strip() == "":
        raise RuntimeError(f"{field_path} must be a non-empty string")
    return value


def _require_python_int(value: Any, field_path: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{field_path} must be a Python int")
    return value


def _require_int(value: Any, field_path: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{field_path} must be a Python int")
    return value


def _json_safe_value(value: Any, field_path: str) -> Any:
    if isinstance(value, torch.Tensor):
        raise TypeError(f"{field_path} must not contain torch.Tensor")
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise RuntimeError(f"{field_path} must be finite")
        return value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{field_path} keys must be strings")
            result[key] = _json_safe_value(item, f"{field_path}.{key}")
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [
            _json_safe_value(item, f"{field_path}[{index}]")
            for index, item in enumerate(value)
        ]
    raise TypeError(f"{field_path} is not JSON serializable: {type(value).__name__}")


def _assert_no_tensors(value: Any, field_path: str) -> None:
    if isinstance(value, torch.Tensor):
        raise TypeError(f"{field_path} must not contain torch.Tensor")
    if isinstance(value, Mapping):
        for key, item in value.items():
            _assert_no_tensors(item, f"{field_path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            _assert_no_tensors(item, f"{field_path}[{index}]")


__all__ = [
    "M5_STREAMING_VALIDATION_SCHEMA",
    "run_m5_streaming_validation",
]
