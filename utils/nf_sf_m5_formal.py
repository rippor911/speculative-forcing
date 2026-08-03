from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

M5_FORMAL_TARGET_STEPS = (500, 2000, 5000)

M5_FORMAL_STAGE_CONTRACTS: Mapping[int, Mapping[str, Any]] = MappingProxyType(
    {
        500: MappingProxyType(
            {
                "parent_global_step": None,
                "validation_steps": (0, 500),
                "checkpoint_steps": (0, 500),
            }
        ),
        2000: MappingProxyType(
            {
                "parent_global_step": 500,
                "validation_steps": (0, 500, 2000),
                "checkpoint_steps": (0, 500, 2000),
            }
        ),
        5000: MappingProxyType(
            {
                "parent_global_step": 2000,
                "validation_steps": (0, 500, 2000, 5000),
                "checkpoint_steps": (0, 500, 2000, 5000),
            }
        ),
    }
)


@dataclass(frozen=True, slots=True)
class FormalStageContract:
    target_global_step: int
    parent_global_step: int | None
    validation_steps: tuple[int, ...]
    checkpoint_steps: tuple[int, ...]
    is_resume_stage: bool


def resolve_m5_formal_stage_contract(target_global_step: object) -> FormalStageContract:
    target = _require_python_int(target_global_step, "target_global_step")
    if target not in M5_FORMAL_STAGE_CONTRACTS:
        raise ValueError(
            "target_global_step must be one of "
            f"{M5_FORMAL_TARGET_STEPS}, actual={target}"
        )
    raw = M5_FORMAL_STAGE_CONTRACTS[target]
    parent = cast(int | None, raw["parent_global_step"])
    validation_steps = tuple(cast(Sequence[int], raw["validation_steps"]))
    checkpoint_steps = tuple(cast(Sequence[int], raw["checkpoint_steps"]))
    return FormalStageContract(
        target_global_step=target,
        parent_global_step=parent,
        validation_steps=validation_steps,
        checkpoint_steps=checkpoint_steps,
        is_resume_stage=parent is not None,
    )


def validate_m5_formal_stage_request(
    *,
    mode: object,
    target_global_step: object,
    validation_steps: object,
    checkpoint_steps: object,
    sample_plan_path: object,
    conditionals_artifact_path: object,
    device: object,
    expected_cuda_device_count: object,
    resume_checkpoint_path: object,
    parent_global_step: object,
) -> FormalStageContract:
    _require_joint_mode(mode)
    contract = resolve_m5_formal_stage_contract(target_global_step)
    _require_nonempty_path(sample_plan_path, "sample_plan_path")
    _require_nonempty_path(
        conditionals_artifact_path,
        "conditionals_artifact_path",
    )
    _require_cuda0_device(device)
    cuda_count = _require_python_int(
        expected_cuda_device_count,
        "expected_cuda_device_count",
    )
    if cuda_count != 1:
        raise ValueError(
            "expected_cuda_device_count must equal 1, "
            f"actual={cuda_count}"
        )

    actual_validation_steps = _require_step_sequence(
        validation_steps,
        "validation_steps",
    )
    actual_checkpoint_steps = _require_step_sequence(
        checkpoint_steps,
        "checkpoint_steps",
    )
    _require_exact_schedule(
        actual_validation_steps,
        contract.validation_steps,
        "validation_steps",
    )
    _require_exact_schedule(
        actual_checkpoint_steps,
        contract.checkpoint_steps,
        "checkpoint_steps",
    )

    if contract.parent_global_step is None:
        if resume_checkpoint_path is not None:
            raise ValueError(
                "resume_checkpoint_path must be None for target_global_step=500"
            )
        if parent_global_step is not None:
            raise ValueError(
                "parent_global_step must be None for target_global_step=500"
            )
        return contract

    _require_resume_checkpoint_path(resume_checkpoint_path)
    parent = _require_python_int(parent_global_step, "parent_global_step")
    if parent != contract.parent_global_step:
        raise ValueError(
            "parent_global_step mismatch: "
            f"expected={contract.parent_global_step}, actual={parent}"
        )
    return contract


def _require_joint_mode(mode: object) -> None:
    if not isinstance(mode, str):
        raise TypeError(f"mode must be a string, actual={type(mode).__name__}")
    if mode != "joint":
        raise ValueError(f"mode must be 'joint', actual={mode!r}")


def _require_python_int(value: object, field_name: str) -> int:
    if type(value) is not int:
        raise TypeError(
            f"{field_name} must be a Python int, actual={type(value).__name__}"
        )
    return value


def _require_nonempty_path(value: object, field_name: str) -> str:
    if value is None:
        raise ValueError(f"{field_name} is required")
    if not isinstance(value, (str, PathLike)):
        raise TypeError(
            f"{field_name} must be a string or Path, actual={type(value).__name__}"
        )
    text = str(value)
    if text.strip() == "":
        raise ValueError(f"{field_name} must be non-empty")
    return text


def _require_resume_checkpoint_path(value: object) -> str:
    text = _require_nonempty_path(value, "resume_checkpoint_path")
    name = Path(text).name.lower()
    suffix = Path(text).suffix.lower()
    if suffix == ".tmp" or name.endswith(".tmp"):
        raise ValueError(
            "resume_checkpoint_path must not point to a .tmp checkpoint, "
            f"actual={text!r}"
        )
    return text


def _require_cuda0_device(device: object) -> None:
    if not isinstance(device, str):
        raise TypeError(f"device must be a string, actual={type(device).__name__}")
    if device != "cuda:0":
        raise ValueError(f"device must be cuda:0, actual={device!r}")


def _require_step_sequence(value: object, field_name: str) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(
            f"{field_name} must be a sequence of Python int values, "
            f"actual={type(value).__name__}"
        )
    steps = []
    for index, item in enumerate(value):
        steps.append(_require_python_int(item, f"{field_name}[{index}]"))
    if len(steps) != len(set(steps)):
        raise ValueError(f"{field_name} must not contain duplicate steps")
    return tuple(steps)


def _require_exact_schedule(
    actual: tuple[int, ...],
    expected: tuple[int, ...],
    field_name: str,
) -> None:
    if actual != expected:
        raise ValueError(
            f"{field_name} mismatch: expected={expected}, actual={actual}"
        )
