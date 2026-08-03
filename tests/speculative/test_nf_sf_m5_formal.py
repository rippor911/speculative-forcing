from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import pytest

from utils.nf_sf_m5_formal import (
    M5_FORMAL_STAGE_CONTRACTS,
    M5_FORMAL_TARGET_STEPS,
    FormalStageContract,
    resolve_m5_formal_stage_contract,
    validate_m5_formal_stage_request,
)


def _valid_request_kwargs(target_global_step: int = 500) -> dict[str, Any]:
    contract = resolve_m5_formal_stage_contract(target_global_step)
    return {
        "mode": "joint",
        "target_global_step": target_global_step,
        "validation_steps": contract.validation_steps,
        "checkpoint_steps": contract.checkpoint_steps,
        "sample_plan_path": "formal_sample_plan.json",
        "conditionals_artifact_path": "formal_conditionals/manifest.json",
        "device": "cuda:0",
        "expected_cuda_device_count": 1,
        "resume_checkpoint_path": None
        if contract.parent_global_step is None
        else "parent_checkpoint.pt",
        "parent_global_step": contract.parent_global_step,
    }


def test_formal_stage_constants_are_immutable() -> None:
    assert M5_FORMAL_TARGET_STEPS == (500, 2000, 5000)
    assert M5_FORMAL_STAGE_CONTRACTS[500]["validation_steps"] == (0, 500)
    with pytest.raises(TypeError):
        M5_FORMAL_STAGE_CONTRACTS[500]["validation_steps"] = (0,)


def test_resolve_target_500_fresh_contract() -> None:
    contract = resolve_m5_formal_stage_contract(500)

    assert contract == FormalStageContract(
        target_global_step=500,
        parent_global_step=None,
        validation_steps=(0, 500),
        checkpoint_steps=(0, 500),
        is_resume_stage=False,
    )


def test_resolve_target_2000_resume_contract() -> None:
    contract = resolve_m5_formal_stage_contract(2000)

    assert contract.parent_global_step == 500
    assert contract.validation_steps == (0, 500, 2000)
    assert contract.checkpoint_steps == (0, 500, 2000)
    assert contract.is_resume_stage is True


def test_resolve_target_5000_resume_contract() -> None:
    contract = resolve_m5_formal_stage_contract(5000)

    assert contract.parent_global_step == 2000
    assert contract.validation_steps == (0, 500, 2000, 5000)
    assert contract.checkpoint_steps == (0, 500, 2000, 5000)
    assert contract.is_resume_stage is True


@pytest.mark.parametrize("target", [301, 501, 0, -1])
def test_resolve_rejects_unsupported_target(target: int) -> None:
    with pytest.raises(ValueError, match="target_global_step"):
        resolve_m5_formal_stage_contract(target)


@pytest.mark.parametrize("target", [True, 500.0, "500"])
def test_resolve_rejects_non_python_int_target(target: object) -> None:
    with pytest.raises(TypeError, match="target_global_step"):
        resolve_m5_formal_stage_contract(target)


@pytest.mark.parametrize(
    ("target", "resume_path"),
    [
        (500, None),
        (2000, "stage_a/checkpoint_step000500.pt"),
        (5000, Path("stage_b/checkpoint_step002000.pt")),
    ],
)
def test_validate_accepts_formal_stage_requests(
    target: int,
    resume_path: str | Path | None,
) -> None:
    kwargs = _valid_request_kwargs(target)
    kwargs["resume_checkpoint_path"] = resume_path

    contract = validate_m5_formal_stage_request(**kwargs)

    assert contract == resolve_m5_formal_stage_contract(target)


def test_validate_rejects_frozen_mode() -> None:
    kwargs = _valid_request_kwargs()
    kwargs["mode"] = "frozen"

    with pytest.raises(ValueError, match="mode"):
        validate_m5_formal_stage_request(**kwargs)


def test_validate_rejects_non_string_mode() -> None:
    kwargs = _valid_request_kwargs()
    kwargs["mode"] = 123

    with pytest.raises(TypeError, match="mode"):
        validate_m5_formal_stage_request(**kwargs)


@pytest.mark.parametrize("value", [None, ""])
def test_validate_rejects_missing_or_empty_sample_plan_path(value: object) -> None:
    kwargs = _valid_request_kwargs()
    kwargs["sample_plan_path"] = value

    with pytest.raises((TypeError, ValueError), match="sample_plan_path"):
        validate_m5_formal_stage_request(**kwargs)


@pytest.mark.parametrize("value", [None, ""])
def test_validate_rejects_missing_or_empty_conditionals_artifact_path(
    value: object,
) -> None:
    kwargs = _valid_request_kwargs()
    kwargs["conditionals_artifact_path"] = value

    with pytest.raises((TypeError, ValueError), match="conditionals_artifact_path"):
        validate_m5_formal_stage_request(**kwargs)


@pytest.mark.parametrize("device", ["cpu", "cuda", "cuda:1"])
def test_validate_rejects_non_cuda0_device(device: str) -> None:
    kwargs = _valid_request_kwargs()
    kwargs["device"] = device

    with pytest.raises(ValueError, match="device"):
        validate_m5_formal_stage_request(**kwargs)


def test_validate_rejects_non_string_device() -> None:
    kwargs = _valid_request_kwargs()
    kwargs["device"] = Path("cuda:0")

    with pytest.raises(TypeError, match="device"):
        validate_m5_formal_stage_request(**kwargs)


@pytest.mark.parametrize("count", [0, 2])
def test_validate_rejects_wrong_cuda_device_count(count: int) -> None:
    kwargs = _valid_request_kwargs()
    kwargs["expected_cuda_device_count"] = count

    with pytest.raises(ValueError, match="expected_cuda_device_count"):
        validate_m5_formal_stage_request(**kwargs)


def test_validate_rejects_bool_cuda_device_count() -> None:
    kwargs = _valid_request_kwargs()
    kwargs["expected_cuda_device_count"] = True

    with pytest.raises(TypeError, match="expected_cuda_device_count"):
        validate_m5_formal_stage_request(**kwargs)


def test_validate_rejects_stage_a_resume_checkpoint() -> None:
    kwargs = _valid_request_kwargs(500)
    kwargs["resume_checkpoint_path"] = "checkpoint_step000500.pt"

    with pytest.raises(ValueError, match="resume_checkpoint_path"):
        validate_m5_formal_stage_request(**kwargs)


def test_validate_rejects_stage_a_parent_step() -> None:
    kwargs = _valid_request_kwargs(500)
    kwargs["parent_global_step"] = 0

    with pytest.raises(ValueError, match="parent_global_step"):
        validate_m5_formal_stage_request(**kwargs)


def test_validate_rejects_stage_b_missing_resume_checkpoint() -> None:
    kwargs = _valid_request_kwargs(2000)
    kwargs["resume_checkpoint_path"] = None

    with pytest.raises(ValueError, match="resume_checkpoint_path"):
        validate_m5_formal_stage_request(**kwargs)


def test_validate_rejects_stage_b_wrong_parent_step() -> None:
    kwargs = _valid_request_kwargs(2000)
    kwargs["parent_global_step"] = 499

    with pytest.raises(ValueError, match="parent_global_step"):
        validate_m5_formal_stage_request(**kwargs)


def test_validate_rejects_stage_c_wrong_parent_step() -> None:
    kwargs = _valid_request_kwargs(5000)
    kwargs["parent_global_step"] = 500

    with pytest.raises(ValueError, match="parent_global_step"):
        validate_m5_formal_stage_request(**kwargs)


@pytest.mark.parametrize(
    "resume_path",
    ["checkpoint.tmp", "checkpoint.pt.tmp"],
)
def test_validate_rejects_tmp_resume_path(resume_path: str) -> None:
    kwargs = _valid_request_kwargs(2000)
    kwargs["resume_checkpoint_path"] = resume_path

    with pytest.raises(ValueError, match="resume_checkpoint_path"):
        validate_m5_formal_stage_request(**kwargs)


@pytest.mark.parametrize(
    "validation_steps",
    [
        (0,),
        (0, 500, 501),
        (500, 0),
    ],
)
def test_validate_rejects_validation_schedule_mismatch(
    validation_steps: tuple[int, ...],
) -> None:
    kwargs = _valid_request_kwargs(500)
    kwargs["validation_steps"] = validation_steps

    with pytest.raises(ValueError, match="validation_steps"):
        validate_m5_formal_stage_request(**kwargs)


def test_validate_rejects_duplicate_validation_schedule() -> None:
    kwargs = _valid_request_kwargs(500)
    kwargs["validation_steps"] = (0, 500, 500)

    with pytest.raises(ValueError, match="validation_steps"):
        validate_m5_formal_stage_request(**kwargs)


def test_validate_rejects_checkpoint_schedule_mismatch() -> None:
    kwargs = _valid_request_kwargs(500)
    kwargs["checkpoint_steps"] = (0,)

    with pytest.raises(ValueError, match="checkpoint_steps"):
        validate_m5_formal_stage_request(**kwargs)


@pytest.mark.parametrize("bad_step", [True, 500.0, "500"])
def test_validate_rejects_non_python_int_schedule_item(bad_step: object) -> None:
    kwargs = _valid_request_kwargs(500)
    kwargs["validation_steps"] = (0, bad_step)

    with pytest.raises(TypeError, match=r"validation_steps\[1\]"):
        validate_m5_formal_stage_request(**kwargs)


def test_validate_rejects_string_schedule() -> None:
    kwargs = _valid_request_kwargs(500)
    kwargs["validation_steps"] = "0,500"

    with pytest.raises(TypeError, match="validation_steps"):
        validate_m5_formal_stage_request(**kwargs)


def test_multiple_resolve_results_do_not_pollute_module_contract() -> None:
    first = resolve_m5_formal_stage_contract(500)
    with pytest.raises(FrozenInstanceError):
        first.validation_steps = (0,)

    second = resolve_m5_formal_stage_contract(500)

    assert first is not second
    assert second.validation_steps == (0, 500)
    assert M5_FORMAL_STAGE_CONTRACTS[500]["validation_steps"] == (0, 500)
