from __future__ import annotations

import copy
import importlib
import math
import random
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from pathlib import Path
from typing import Any, cast

import torch

_m3 = importlib.import_module("utils.nf_sf_m3")
_m4 = importlib.import_module("utils.nf_sf_m4")
_nf_sf_tensors = importlib.import_module("utils.nf_sf_tensors")

M3_CHECKPOINT_FORMAT = cast(str, _m3.M3_CHECKPOINT_FORMAT)
M3_CHUNK_FRAMES = cast(int, _m3.M3_CHUNK_FRAMES)
M3_DEPTHS = tuple(int(depth) for depth in cast(Sequence[int], _m3.M3_DEPTHS))
M3_DEPTH_WEIGHTS = tuple(
    float(weight) for weight in cast(Sequence[float], _m3.M3_DEPTH_WEIGHTS)
)
M3_PARAMETER_GROUP_NAMES = tuple(
    str(name) for name in cast(Sequence[str], _m3.M3_PARAMETER_GROUP_NAMES)
)
DEFAULT_S_MAIN = float(_nf_sf_tensors.DEFAULT_S_MAIN)
DEFAULT_S_MCP = float(_nf_sf_tensors.DEFAULT_S_MCP)

file_sha256 = cast(Callable[[Path | str], str], _m3.file_sha256)
tensor_sha256 = cast(Callable[[torch.Tensor], str], _m3.tensor_sha256)
validate_git_sha = cast(Callable[..., str], _m3.validate_git_sha)
validate_m3_checkpoint_payload = cast(
    Callable[[Mapping[str, Any]], None],
    _m3.validate_m3_checkpoint_payload,
)
m4_next_train_entry_after_global_step = cast(
    Callable[[Mapping[str, Any], int], dict[str, Any]],
    _m4.m4_next_train_entry_after_global_step,
)
m4_sample_plan_sha256 = cast(
    Callable[[Mapping[str, Any]], str],
    _m4.m4_sample_plan_sha256,
)
m4_train_entry_for_step = cast(
    Callable[[Mapping[str, Any], int], dict[str, Any]],
    _m4.m4_train_entry_for_step,
)
validate_m4_sample_plan = cast(
    Callable[..., dict[str, Any]],
    _m4.validate_m4_sample_plan,
)


M5_RESUME_SCHEMA = "nf_sf_m5_resume_contract_v1"
M5_RNG_EXTENSION_FIELD = "m5_rng_state"
M5_RNG_EXTENSION_SCHEMA = "nf_sf_m5_rng_state_v1"

M5_ALLOWED_OVERRIDE_FIELDS = (
    "output_dir",
    "target_global_step",
    "target_validation_steps",
    "target_checkpoint_steps",
)

M5_ALLOWED_CONFIG_OVERRIDE_FIELDS = (
    "m3.optimizer_steps",
    "m4.validation_steps",
    "m4.checkpoint_steps",
)

M5_GLOBAL_RNG_FIELDS = (
    "python_random_state",
    "torch_cpu_rng_state",
    "torch_cuda_rng_states",
)

M5_TIMESTEP_NOISE_CONTRACT = {
    "train_rng": "checkpoint.train_rng_state local torch.Generator",
    "probe_rng": "checkpoint.probe_rng_state fixed probe torch.Generator",
    "s_main": float(DEFAULT_S_MAIN),
    "s_mcp": float(DEFAULT_S_MCP),
    "chunk_frames": int(M3_CHUNK_FRAMES),
    "mcp_depths": [int(depth) for depth in M3_DEPTHS],
}

M5_RNG_EXTENSION_DESIGN = {
    "field": M5_RNG_EXTENSION_FIELD,
    "schema": M5_RNG_EXTENSION_SCHEMA,
    "purpose": (
        "backward-compatible top-level checkpoint field for global RNG states "
        "not present in M3 checkpoint v1"
    ),
    "fields": {
        "python_random_state": "random.getstate() payload",
        "torch_cpu_rng_state": "torch.random.get_rng_state() uint8 tensor",
        "torch_cuda_rng_states": "torch.cuda.get_rng_state_all() uint8 tensors",
        "cuda_rng_captured": "whether non-empty CUDA RNG state tensors were captured",
        "cuda_device_count": "number of captured CUDA RNG state tensors",
        "train_generator_device": "device type used for train torch.Generator",
        "probe_generator_device": "device type used for fixed-probe torch.Generator",
    },
}

_MISSING = object()
_PATH_LOCKED_FIELDS = frozenset(
    {
        "reference_checkpoint.path",
        "manifest.path",
    }
)
_GIT_SHA_LOCKED_FIELDS = frozenset({"git_sha"})
_SHA256_LOCKED_FIELDS = frozenset(
    {
        "reference_checkpoint.sha256",
        "manifest.sha256",
        "m4.sample_plan_sha256",
    }
)
_IDENTITY_LIST_FIELDS = frozenset(
    {
        "m4.train_sample_identities",
        "m4.validation_sample_identities",
    }
)
_FLOAT_FIELDS = frozenset(
    {
        "m3.backbone_lr",
        "m3.patch_embedding_lr",
        "m3.mcp_lr",
        "m3.weight_decay",
        "m3.mcp1_grid_aux_weight",
        "optimizer.eps",
        "optimizer.weight_decay",
    }
)
_FLOAT_LIST_FIELDS = frozenset(
    {
        "optimizer.betas",
        "mcp.depth_weights",
    }
)
_INT_FIELDS = frozenset(
    {
        "m3.train_seed",
        "m3.probe_seed",
        "m4.validation_seed",
        "chunk_frames",
    }
)
_INT_LIST_FIELDS = frozenset({"mcp.depths"})


class ResumeContractError(RuntimeError):
    def __init__(
        self,
        field_path: str,
        expected: Any,
        actual: Any,
        *,
        reason: str = "resume contract mismatch",
        report: Mapping[str, Any] | None = None,
    ) -> None:
        self.field_path = str(field_path)
        self.expected = expected
        self.actual = actual
        self.report = None if report is None else dict(report)
        super().__init__(
            f"{reason}: field_path={self.field_path}, "
            f"expected={_short_repr(expected)}, actual={_short_repr(actual)}"
        )


def checkpoint_source_record(
    path: Path | str | None,
    *,
    sha256: str | None = None,
) -> dict[str, str | bool | None]:
    resolved_path = None if path is None else _normalize_path(path)
    resolved_sha = str(sha256) if sha256 is not None else None
    source_verified = False
    if path is not None:
        path_obj = Path(path)
        if path_obj.is_file():
            actual_sha = file_sha256(path_obj)
            if resolved_sha is not None and actual_sha != resolved_sha:
                raise ResumeContractError(
                    "parent_checkpoint_sha256",
                    actual_sha,
                    resolved_sha,
                    reason="parent checkpoint SHA256 mismatch",
                )
            resolved_sha = actual_sha
            source_verified = True
        elif path_obj.exists():
            raise ResumeContractError(
                "parent_checkpoint_path",
                "checkpoint file",
                resolved_path,
                reason="invalid parent checkpoint provenance",
            )
    if resolved_sha is not None:
        _require_sha256(resolved_sha, "parent_checkpoint_sha256")
    return {
        "path": resolved_path,
        "sha256": resolved_sha,
        "source_verified": source_verified,
    }


def load_resume_checkpoint_payload(path: Path | str) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ResumeContractError(
            "checkpoint_payload",
            "dict",
            type(payload).__name__,
            reason="invalid resume checkpoint payload",
        )
    parse_resume_checkpoint_payload(
        payload,
        parent_checkpoint_path=path,
        parent_checkpoint_sha256=file_sha256(path),
    )
    return payload


def parse_resume_checkpoint_payload(
    payload: Mapping[str, Any],
    *,
    parent_checkpoint_path: Path | str | None = None,
    parent_checkpoint_sha256: str | None = None,
    sample_plan: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    _validate_m3_payload(payload)
    global_step = _require_int(_require_key(payload, "global_step", "global_step"))
    if global_step < 0:
        raise ResumeContractError(
            "global_step",
            "non-negative integer",
            global_step,
            reason="invalid resume checkpoint payload",
        )
    resolved_config = _require_mapping_field(
        payload,
        "resolved_config",
        "resolved_config",
    )
    selected_metadata = _require_mapping_field(
        payload,
        "selected_sample_metadata",
        "selected_sample_metadata",
    )
    reference_checkpoint = _require_mapping_field(
        payload,
        "reference_checkpoint",
        "reference_checkpoint",
    )
    locked_fields = build_resume_run_fields(
        resolved_config=resolved_config,
        reference_checkpoint=reference_checkpoint,
        git_sha=_require_key(payload, "git_sha", "git_sha"),
        optimizer_state_dict=_require_mapping_field(payload, "optimizer", "optimizer"),
        optimizer_group_lrs=_require_sequence(
            _require_key(
                payload,
                "optimizer_group_lrs",
                "optimizer_group_lrs",
            ),
            "optimizer_group_lrs",
        ),
        selected_sample_metadata=selected_metadata,
        sample_plan=sample_plan,
    )
    _validate_top_level_seed_consistency(payload, locked_fields)
    rng_states = extract_resume_rng_states(payload)
    source = checkpoint_source_record(
        parent_checkpoint_path,
        sha256=parent_checkpoint_sha256,
    )
    return {
        "schema": M5_RESUME_SCHEMA,
        "status": "PARSED",
        "parent_checkpoint_path": source["path"],
        "parent_checkpoint_sha256": source["sha256"],
        "source_verified": source["source_verified"],
        "checkpoint_format": str(payload["format"]),
        "resumed_global_step": int(global_step),
        "first_resumed_step": first_resumed_global_step(global_step),
        "locked_fields": locked_fields,
        "optimizer_restore": optimizer_restore_report(
            payload,
            resolved_config=resolved_config,
        ),
        "rng_inventory": validate_resume_rng_states(rng_states),
        "lr_scheduler": None,
        "lr_scheduler_restore": "not_applicable",
    }


def validate_resume_checkpoint_payload(
    payload: Mapping[str, Any],
    *,
    parent_checkpoint_path: Path | str | None = None,
    parent_checkpoint_sha256: str | None = None,
    sample_plan: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return parse_resume_checkpoint_payload(
        payload,
        parent_checkpoint_path=parent_checkpoint_path,
        parent_checkpoint_sha256=parent_checkpoint_sha256,
        sample_plan=sample_plan,
    )


def build_resume_contract(
    checkpoint_payload: Mapping[str, Any],
    *,
    parent_checkpoint_path: Path | str | None = None,
    parent_checkpoint_sha256: str | None = None,
    sample_plan: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    parsed = parse_resume_checkpoint_payload(
        checkpoint_payload,
        parent_checkpoint_path=parent_checkpoint_path,
        parent_checkpoint_sha256=parent_checkpoint_sha256,
        sample_plan=sample_plan,
    )
    if parsed["parent_checkpoint_path"] is None:
        raise ResumeContractError(
            "parent_checkpoint_path",
            "present",
            "missing",
            reason="missing parent checkpoint provenance",
        )
    if parsed["parent_checkpoint_sha256"] is None:
        raise ResumeContractError(
            "parent_checkpoint_sha256",
            "present",
            "missing",
            reason="missing parent checkpoint provenance",
        )
    parsed["status"] = "READY"
    return parsed


def build_resume_run_fields(
    *,
    resolved_config: Mapping[str, Any],
    reference_checkpoint: Mapping[str, Any],
    git_sha: Any,
    optimizer_state_dict: Mapping[str, Any] | None = None,
    optimizer_group_lrs: Sequence[Any] | None = None,
    selected_sample_metadata: Mapping[str, Any] | None = None,
    sample_plan: Mapping[str, Any] | None = None,
    manifest_path: Path | str | None = None,
    manifest_sha256: str | None = None,
    mcp_depths: Sequence[int] = M3_DEPTHS,
    timestep_noise_contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    model_config = _require_mapping_field(
        resolved_config,
        "model_config",
        "resolved_config.model_config",
    )
    m3_config = _require_mapping_field(
        resolved_config,
        "m3",
        "resolved_config.m3",
    )
    m4_config = _require_mapping_field(
        resolved_config,
        "m4",
        "resolved_config.m4",
    )
    optimizer_config = _require_mapping_field(
        m3_config,
        "optimizer_config",
        "resolved_config.m3.optimizer_config",
    )
    selected_metadata = (
        {} if selected_sample_metadata is None else selected_sample_metadata
    )
    resolved_manifest_path = _manifest_path(
        m3_config,
        selected_metadata,
        manifest_path,
    )
    resolved_manifest_sha = _manifest_sha256(
        m3_config,
        selected_metadata,
        manifest_sha256,
    )
    plan_fields = _m4_plan_fields(m4_config, sample_plan)
    chunk_frames = _require_int(
        _require_key(
            model_config,
            "num_frame_per_block",
            "resolved_config.model_config.num_frame_per_block",
        )
    )
    if selected_sample_metadata is not None and "chunk_frames" in selected_metadata:
        metadata_chunk_frames = _require_int(
            selected_metadata["chunk_frames"],
            "selected_sample_metadata.chunk_frames",
        )
        if metadata_chunk_frames != chunk_frames:
            raise ResumeContractError(
                "selected_sample_metadata.chunk_frames",
                chunk_frames,
                metadata_chunk_frames,
                reason="resume checkpoint internal mismatch",
            )
    contract = (
        M5_TIMESTEP_NOISE_CONTRACT
        if timestep_noise_contract is None
        else dict(timestep_noise_contract)
    )
    optimizer_contract = None
    if optimizer_state_dict is not None:
        if optimizer_group_lrs is None:
            raise ResumeContractError(
                "optimizer_group_lrs",
                "present",
                "missing",
                reason="missing optimizer resume contract field",
            )
        optimizer_contract = validate_optimizer_state_contract(
            optimizer_state_dict,
            resolved_config=resolved_config,
            optimizer_group_lrs=optimizer_group_lrs,
        )
    fields = {
        "format": M3_CHECKPOINT_FORMAT,
        "git_sha": _require_git_sha(git_sha, "git_sha"),
        "resolved_config.locked": canonical_locked_resolved_config(resolved_config),
        "m3.mode": str(_require_key(m3_config, "mode", "resolved_config.m3.mode")),
        "reference_checkpoint.path": _normalize_path(
            _require_key(
                reference_checkpoint,
                "path",
                "reference_checkpoint.path",
            )
        ),
        "reference_checkpoint.sha256": _require_sha256(
            _require_key(
                reference_checkpoint,
                "sha256",
                "reference_checkpoint.sha256",
            ),
            "reference_checkpoint.sha256",
        ),
        "manifest.path": resolved_manifest_path,
        "manifest.sha256": resolved_manifest_sha,
        "m4.sample_plan_sha256": plan_fields["sample_plan_sha256"],
        "m4.train_sample_identities": plan_fields["train_sample_identities"],
        "m4.validation_sample_identities": plan_fields[
            "validation_sample_identities"
        ],
        "m3.train_seed": _require_int(
            _require_key(m3_config, "train_seed", "resolved_config.m3.train_seed")
        ),
        "m3.probe_seed": _require_int(
            _require_key(m3_config, "probe_seed", "resolved_config.m3.probe_seed")
        ),
        "m4.validation_seed": _require_int(
            _require_key(
                m4_config,
                "validation_seed",
                "resolved_config.m4.validation_seed",
            )
        ),
        "m3.backbone_lr": _require_float(
            _require_key(
                m3_config,
                "backbone_lr",
                "resolved_config.m3.backbone_lr",
            ),
            "resolved_config.m3.backbone_lr",
        ),
        "m3.patch_embedding_lr": _require_float(
            _require_key(
                m3_config,
                "patch_embedding_lr",
                "resolved_config.m3.patch_embedding_lr",
            ),
            "resolved_config.m3.patch_embedding_lr",
        ),
        "m3.mcp_lr": _require_float(
            _require_key(m3_config, "mcp_lr", "resolved_config.m3.mcp_lr"),
            "resolved_config.m3.mcp_lr",
        ),
        "m3.weight_decay": _require_float(
            _require_key(
                m3_config,
                "weight_decay",
                "resolved_config.m3.weight_decay",
            ),
            "resolved_config.m3.weight_decay",
        ),
        "m3.mcp1_grid_aux_weight": _require_float(
            _require_key(
                m3_config,
                "mcp1_grid_aux_weight",
                "resolved_config.m3.mcp1_grid_aux_weight",
            ),
            "resolved_config.m3.mcp1_grid_aux_weight",
        ),
        "m3.dtype": str(_require_key(m3_config, "dtype", "resolved_config.m3.dtype")),
        "model_config": _json_safe(model_config),
        "optimizer.type": str(_require_key(
            optimizer_config,
            "optimizer",
            "resolved_config.m3.optimizer_config.optimizer",
        )),
        "optimizer.betas": _float_list(
            _require_key(
                optimizer_config,
                "betas",
                "resolved_config.m3.optimizer_config.betas",
            ),
            "resolved_config.m3.optimizer_config.betas",
        ),
        "optimizer.eps": _require_float(
            _require_key(
                optimizer_config,
                "eps",
                "resolved_config.m3.optimizer_config.eps",
            ),
            "resolved_config.m3.optimizer_config.eps",
        ),
        "optimizer.weight_decay": _require_float(
            _require_key(
                optimizer_config,
                "weight_decay",
                "resolved_config.m3.optimizer_config.weight_decay",
            ),
            "resolved_config.m3.optimizer_config.weight_decay",
        ),
        "chunk_frames": chunk_frames,
        "mcp.depths": [int(depth) for depth in mcp_depths],
        "mcp.depth_weights": _float_list(
            _require_key(
                model_config,
                "mcp_depth_weights",
                "resolved_config.model_config.mcp_depth_weights",
            ),
            "resolved_config.model_config.mcp_depth_weights",
        ),
        "timestep_noise_contract": _json_safe(contract),
        "m4.fixed_decode_validation_identity": str(plan_fields[
            "fixed_decode_validation_identity"
        ]),
    }
    if optimizer_contract is not None:
        fields["optimizer.param_groups"] = optimizer_contract["param_groups"]
        fields["optimizer.optimizer_group_lrs"] = optimizer_contract[
            "optimizer_group_lrs"
        ]
    return {
        field_path: _normalize_locked_field_value(field_path, value)
        for field_path, value in fields.items()
    }


def validate_resume_contract(
    contract: Mapping[str, Any],
    current_run_fields: Mapping[str, Any],
    *,
    target_global_step: int,
    sample_plan: Mapping[str, Any],
    output_dir: Path | str | None = None,
    target_validation_steps: Sequence[int] | None = None,
    target_checkpoint_steps: Sequence[int] | None = None,
    allow_legacy_missing_global_rng: bool = False,
    global_rng_independence_evidence: bool = False,
    expected_cuda_device_count: int | None = None,
) -> dict[str, Any]:
    if contract.get("schema") != M5_RESUME_SCHEMA:
        raise ResumeContractError(
            "schema",
            M5_RESUME_SCHEMA,
            contract.get("schema"),
            reason="invalid resume contract",
        )
    expected_fields = _contract_locked_fields(contract)
    actual_fields = _current_locked_fields(current_run_fields)
    resumed_step = _require_int(
        _require_key(contract, "resumed_global_step", "resumed_global_step")
    )
    target_step = _require_int(target_global_step, "target_global_step")
    incompatibilities: list[dict[str, Any]] = []
    parent_path = contract.get("parent_checkpoint_path")
    parent_sha = contract.get("parent_checkpoint_sha256")
    if parent_path is None:
        incompatibilities.append(
            _incompatibility("parent_checkpoint_path", "present", "missing")
        )
    else:
        _normalize_path(parent_path)
    if parent_sha is None:
        incompatibilities.append(
            _incompatibility("parent_checkpoint_sha256", "present", "missing")
        )
    else:
        _require_sha256(parent_sha, "parent_checkpoint_sha256")
    for field_path in sorted(expected_fields):
        expected = _normalize_locked_field_value(
            field_path,
            expected_fields[field_path],
        )
        if field_path not in actual_fields:
            incompatibilities.append(
                _incompatibility(field_path, expected, "missing")
            )
            continue
        actual = _normalize_locked_field_value(field_path, actual_fields[field_path])
        if actual != expected:
            _append_locked_field_incompatibilities(
                field_path,
                expected,
                actual,
                incompatibilities,
            )
    if target_step <= resumed_step:
        incompatibilities.append(
            _incompatibility(
                "target_global_step",
                f"> resumed_global_step ({resumed_step})",
                target_step,
            )
        )
    sample_plan_report = _validate_m4_resume_plan(
        sample_plan,
        expected_fields,
        incompatibilities,
    )
    first_identity = None
    next_after_target_identity = None
    if sample_plan_report["status"] == "PASS":
        first_identity = first_resumed_sample_identity(sample_plan, resumed_step)
        next_after_target_identity = next_sample_after_target_identity(
            sample_plan,
            target_step,
        )
    rng_restore = validate_rng_restore_contract(
        _require_mapping_field(contract, "rng_inventory", "rng_inventory"),
        allow_legacy_missing_global_rng=allow_legacy_missing_global_rng,
        global_rng_independence_evidence=global_rng_independence_evidence,
        expected_cuda_device_count=expected_cuda_device_count,
    )
    if str(rng_restore["status"]).startswith("FAIL"):
        cuda_mismatch = rng_restore.get("cuda_rng_mismatch")
        if isinstance(cuda_mismatch, Mapping):
            incompatibilities.append(
                _incompatibility(
                    str(cuda_mismatch["field_path"]),
                    cuda_mismatch["expected"],
                    cuda_mismatch["actual"],
                )
            )
        else:
            incompatibilities.append(
                _incompatibility(
                    "rng_restore.missing_global_rng_fields",
                    "present or explicit training-path independence evidence",
                    rng_restore["missing_global_rng_fields"],
                )
            )
    report = {
        "schema": M5_RESUME_SCHEMA,
        "status": "FAIL" if incompatibilities else "PASS",
        "parent_checkpoint_path": contract.get("parent_checkpoint_path"),
        "parent_checkpoint_sha256": contract.get("parent_checkpoint_sha256"),
        "source_verified": bool(contract.get("source_verified", False)),
        "resumed_global_step": int(resumed_step),
        "target_global_step": int(target_step),
        "first_resumed_step": first_resumed_global_step(resumed_step),
        "first_resumed_sample_identity": first_identity,
        "next_sample_after_target_identity": next_after_target_identity,
        "locked_fields_checked": sorted(expected_fields.keys()),
        "allowed_overrides": {
            "output_dir": None if output_dir is None else _normalize_path(output_dir),
            "target_global_step": int(target_step),
            "target_validation_steps": None
            if target_validation_steps is None
            else [int(step) for step in target_validation_steps],
            "target_checkpoint_steps": None
            if target_checkpoint_steps is None
            else [int(step) for step in target_checkpoint_steps],
            "fields": list(M5_ALLOWED_OVERRIDE_FIELDS),
        },
        "optimizer_restore": _json_safe(contract.get("optimizer_restore")),
        "rng_restore": rng_restore,
        "lr_scheduler": None,
        "lr_scheduler_restore": "not_applicable",
        "incompatibilities": incompatibilities,
    }
    if incompatibilities:
        first = incompatibilities[0]
        raise ResumeContractError(
            str(first["field_path"]),
            first["expected"],
            first["actual"],
            report=report,
        )
    return report


def move_optimizer_state_to_device(
    optimizer_state: Mapping[str, Any],
    *,
    device: torch.device | str,
) -> dict[str, Any]:
    moved, _ = _optimizer_value_to_device_copy(
        optimizer_state,
        device=torch.device(device),
        field_path="optimizer_state",
    )
    if not isinstance(moved, dict):
        raise TypeError("optimizer state migration must return a dict")
    return moved


def move_loaded_optimizer_state_to_device(
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device | str,
) -> dict[str, Any]:
    target_device = torch.device(device)
    tensor_count = 0
    floating_tensor_count = 0
    for state_index, state in enumerate(optimizer.state.values()):
        moved_state, moved_tensors, moved_floating = (
            _optimizer_value_to_device_in_place(
                state,
                device=target_device,
                field_path=f"optimizer.state[{state_index}]",
            )
        )
        if moved_state is not state:
            raise ResumeContractError(
                f"optimizer.state[{state_index}]",
                "mutable optimizer state mapping",
                type(state).__name__,
                reason="invalid loaded optimizer state",
            )
        tensor_count += moved_tensors
        floating_tensor_count += moved_floating
    return {
        "status": "moved",
        "device": str(target_device),
        "state_entry_count": len(optimizer.state),
        "state_tensor_count": tensor_count,
        "floating_state_tensor_count": floating_tensor_count,
    }


def extract_resume_rng_states(payload: Mapping[str, Any]) -> dict[str, Any]:
    train_state = _clone_rng_state_tensor(
        _require_key(payload, "train_rng_state", "train_rng_state"),
        "train_rng_state",
    )
    probe_state = _clone_rng_state_tensor(
        _require_key(payload, "probe_rng_state", "probe_rng_state"),
        "probe_rng_state",
    )
    extension = payload.get(M5_RNG_EXTENSION_FIELD)
    python_state: Any | None = None
    torch_cpu_state: torch.Tensor | None = None
    torch_cuda_states: tuple[torch.Tensor, ...] | None = None
    cuda_rng_captured: bool | None = None
    cuda_device_count: int | None = None
    train_generator_device: str | None = None
    probe_generator_device: str | None = None
    missing_global_fields = list(M5_GLOBAL_RNG_FIELDS)
    if extension is not None:
        extension_mapping = _require_mapping(extension, M5_RNG_EXTENSION_FIELD)
        schema = _require_key(
            extension_mapping,
            "schema",
            f"{M5_RNG_EXTENSION_FIELD}.schema",
        )
        if schema != M5_RNG_EXTENSION_SCHEMA:
            raise ResumeContractError(
                f"{M5_RNG_EXTENSION_FIELD}.schema",
                M5_RNG_EXTENSION_SCHEMA,
                schema,
                reason="invalid RNG extension",
            )
        cuda_rng_captured = _require_bool(
            _require_key(
                extension_mapping,
                "cuda_rng_captured",
                f"{M5_RNG_EXTENSION_FIELD}.cuda_rng_captured",
            ),
            f"{M5_RNG_EXTENSION_FIELD}.cuda_rng_captured",
        )
        cuda_device_count = _require_int(
            _require_key(
                extension_mapping,
                "cuda_device_count",
                f"{M5_RNG_EXTENSION_FIELD}.cuda_device_count",
            ),
            f"{M5_RNG_EXTENSION_FIELD}.cuda_device_count",
        )
        train_generator_device = str(_require_key(
            extension_mapping,
            "train_generator_device",
            f"{M5_RNG_EXTENSION_FIELD}.train_generator_device",
        ))
        probe_generator_device = str(_require_key(
            extension_mapping,
            "probe_generator_device",
            f"{M5_RNG_EXTENSION_FIELD}.probe_generator_device",
        ))
        if "python_random_state" in extension_mapping:
            python_state = copy.deepcopy(extension_mapping["python_random_state"])
            _validate_python_random_state(
                python_state,
                f"{M5_RNG_EXTENSION_FIELD}.python_random_state",
            )
            missing_global_fields.remove("python_random_state")
        if "torch_cpu_rng_state" in extension_mapping:
            torch_cpu_state = _clone_rng_state_tensor(
                extension_mapping["torch_cpu_rng_state"],
                f"{M5_RNG_EXTENSION_FIELD}.torch_cpu_rng_state",
            )
            _validate_torch_cpu_rng_state(
                torch_cpu_state,
                f"{M5_RNG_EXTENSION_FIELD}.torch_cpu_rng_state",
            )
            missing_global_fields.remove("torch_cpu_rng_state")
        if "torch_cuda_rng_states" in extension_mapping:
            torch_cuda_states = _clone_cuda_rng_states(
                extension_mapping["torch_cuda_rng_states"],
                f"{M5_RNG_EXTENSION_FIELD}.torch_cuda_rng_states",
            )
            if len(torch_cuda_states) != cuda_device_count:
                raise ResumeContractError(
                    f"{M5_RNG_EXTENSION_FIELD}.cuda_device_count",
                    len(torch_cuda_states),
                    cuda_device_count,
                    reason="invalid RNG extension",
                )
            if cuda_rng_captured != (len(torch_cuda_states) > 0):
                raise ResumeContractError(
                    f"{M5_RNG_EXTENSION_FIELD}.cuda_rng_captured",
                    len(torch_cuda_states) > 0,
                    cuda_rng_captured,
                    reason="invalid RNG extension",
                )
            missing_global_fields.remove("torch_cuda_rng_states")
    return {
        "train_generator_state": train_state,
        "probe_generator_state": probe_state,
        "python_random_state": python_state,
        "torch_cpu_rng_state": torch_cpu_state,
        "torch_cuda_rng_states": torch_cuda_states,
        "cuda_rng_captured": cuda_rng_captured,
        "cuda_device_count": cuda_device_count,
        "train_generator_device": train_generator_device,
        "probe_generator_device": probe_generator_device,
        "missing_global_fields": tuple(missing_global_fields),
    }


def validate_resume_rng_states(rng_states: Mapping[str, Any]) -> dict[str, Any]:
    train_state = _clone_rng_state_tensor(
        _require_key(
            rng_states,
            "train_generator_state",
            "rng_states.train_generator_state",
        ),
        "rng_states.train_generator_state",
    )
    probe_state = _clone_rng_state_tensor(
        _require_key(
            rng_states,
            "probe_generator_state",
            "rng_states.probe_generator_state",
        ),
        "rng_states.probe_generator_state",
    )
    python_state = rng_states.get("python_random_state")
    if python_state is not None:
        _validate_python_random_state(
            python_state,
            "rng_states.python_random_state",
        )
    torch_cpu_state = rng_states.get("torch_cpu_rng_state")
    if torch_cpu_state is not None:
        torch_cpu_state = _clone_rng_state_tensor(
            torch_cpu_state,
            "rng_states.torch_cpu_rng_state",
        )
        _validate_torch_cpu_rng_state(
            torch_cpu_state,
            "rng_states.torch_cpu_rng_state",
        )
    torch_cuda_states = rng_states.get("torch_cuda_rng_states")
    cuda_rng_captured = rng_states.get("cuda_rng_captured")
    if cuda_rng_captured is not None:
        cuda_rng_captured = _require_bool(
            cuda_rng_captured,
            "rng_states.cuda_rng_captured",
        )
    cuda_device_count = rng_states.get("cuda_device_count")
    if cuda_device_count is not None:
        cuda_device_count = _require_int(
            cuda_device_count,
            "rng_states.cuda_device_count",
        )
    train_generator_device = rng_states.get("train_generator_device")
    probe_generator_device = rng_states.get("probe_generator_device")
    cuda_summary: dict[str, Any] = {
        "available": False,
        "captured": False,
        "count": None,
        "sha256": [],
    }
    if torch_cuda_states is not None:
        cuda_states = _clone_cuda_rng_states(
            torch_cuda_states,
            "rng_states.torch_cuda_rng_states",
        )
        actual_cuda_count = len(cuda_states)
        if cuda_device_count is not None and cuda_device_count != actual_cuda_count:
            raise ResumeContractError(
                "rng_states.cuda_device_count",
                actual_cuda_count,
                cuda_device_count,
                reason="invalid RNG state",
            )
        actual_cuda_captured = actual_cuda_count > 0
        if (
            cuda_rng_captured is not None
            and cuda_rng_captured != actual_cuda_captured
        ):
            raise ResumeContractError(
                "rng_states.cuda_rng_captured",
                actual_cuda_captured,
                cuda_rng_captured,
                reason="invalid RNG state",
            )
        cuda_summary = {
            "available": True,
            "captured": actual_cuda_captured,
            "count": actual_cuda_count,
            "sha256": [tensor_sha256(state) for state in cuda_states],
        }
    missing = [
        str(field)
        for field in rng_states.get("missing_global_fields", ())
    ]
    return {
        "schema": M5_RNG_EXTENSION_SCHEMA,
        "train_generator": _rng_tensor_summary(train_state, "train_rng_state"),
        "probe_generator": _rng_tensor_summary(probe_state, "probe_rng_state"),
        "python_random_state": {
            "available": python_state is not None,
            "field": f"{M5_RNG_EXTENSION_FIELD}.python_random_state",
        },
        "torch_cpu_rng_state": None
        if torch_cpu_state is None
        else _rng_tensor_summary(
            torch_cpu_state,
            f"{M5_RNG_EXTENSION_FIELD}.torch_cpu_rng_state",
        ),
        "torch_cuda_rng_states": cuda_summary,
        "cuda_rng_captured": None
        if cuda_rng_captured is None
        else bool(cuda_rng_captured),
        "cuda_device_count": cuda_device_count,
        "train_generator_device": None
        if train_generator_device is None
        else str(train_generator_device),
        "probe_generator_device": None
        if probe_generator_device is None
        else str(probe_generator_device),
        "missing_global_rng_fields": missing,
        "m5_extension_design": copy.deepcopy(M5_RNG_EXTENSION_DESIGN),
    }


def validate_rng_restore_contract(
    rng_inventory: Mapping[str, Any],
    *,
    allow_legacy_missing_global_rng: bool,
    global_rng_independence_evidence: bool,
    expected_cuda_device_count: int | None = None,
) -> dict[str, Any]:
    missing = [str(field) for field in rng_inventory.get(
        "missing_global_rng_fields",
        [],
    )]
    status = "PASS"
    cuda_mismatch: dict[str, Any] | None = None
    if missing:
        if allow_legacy_missing_global_rng and global_rng_independence_evidence:
            status = "PASS_LEGACY_MISSING_GLOBAL_RNG"
        else:
            status = "FAIL_MISSING_GLOBAL_RNG"
    if expected_cuda_device_count is not None:
        expected_cuda_count = _require_int(
            expected_cuda_device_count,
            "expected_cuda_device_count",
        )
        actual_cuda_count = rng_inventory.get("cuda_device_count")
        if actual_cuda_count is None:
            actual_cuda_count = _require_mapping_field(
                rng_inventory,
                "torch_cuda_rng_states",
                "rng_inventory.torch_cuda_rng_states",
            ).get("count")
        actual_cuda_count = _require_int(
            actual_cuda_count,
            "rng_inventory.cuda_device_count",
        )
        cuda_captured = bool(rng_inventory.get("cuda_rng_captured", False))
        if actual_cuda_count != expected_cuda_count or (
            expected_cuda_count > 0 and not cuda_captured
        ):
            status = "FAIL_CUDA_RNG_STATE_COUNT"
            cuda_mismatch = {
                "field_path": "rng_restore.torch_cuda_rng_states",
                "expected": expected_cuda_count,
                "actual": actual_cuda_count,
                "cuda_rng_captured": cuda_captured,
            }
    report = dict(rng_inventory)
    report.update(
        {
            "status": status,
            "legacy_missing_global_rng_allowed": bool(
                allow_legacy_missing_global_rng
            ),
            "global_rng_independence_evidence": bool(
                global_rng_independence_evidence
            ),
            "expected_cuda_device_count": expected_cuda_device_count,
            "cuda_rng_mismatch": cuda_mismatch,
            "missing_global_rng_fields": missing,
        }
    )
    return _json_safe(report)


def restore_torch_generator_from_state(
    state: torch.Tensor,
    *,
    device: torch.device | str = "cpu",
) -> torch.Generator:
    state_tensor = _clone_rng_state_tensor(state, "generator_state")
    generator = torch.Generator(device=torch.device(device))
    try:
        generator.set_state(state_tensor)
    except RuntimeError as exc:
        raise ResumeContractError(
            "generator_state",
            f"valid torch.Generator state for device {torch.device(device)}",
            _rng_tensor_summary(state_tensor, "generator_state"),
            reason="invalid RNG state",
        ) from exc
    return generator


def restore_resume_generators(
    payload: Mapping[str, Any],
    *,
    device: torch.device | str = "cpu",
) -> dict[str, torch.Generator]:
    states = extract_resume_rng_states(payload)
    return {
        "train": restore_torch_generator_from_state(
            cast(torch.Tensor, states["train_generator_state"]),
            device=device,
        ),
        "probe": restore_torch_generator_from_state(
            cast(torch.Tensor, states["probe_generator_state"]),
            device=device,
        ),
    }


def restore_global_rng_states(rng_states: Mapping[str, Any]) -> None:
    python_state = rng_states.get("python_random_state")
    if python_state is not None:
        python_state = copy.deepcopy(python_state)
        _validate_python_random_state(
            python_state,
            "rng_states.python_random_state",
        )
    torch_cpu_state = rng_states.get("torch_cpu_rng_state")
    if torch_cpu_state is not None:
        torch_cpu_state = _clone_rng_state_tensor(
            torch_cpu_state,
            "rng_states.torch_cpu_rng_state",
        )
        _validate_torch_cpu_rng_state(
            torch_cpu_state,
            "rng_states.torch_cpu_rng_state",
        )
    torch_cuda_states = rng_states.get("torch_cuda_rng_states")
    cuda_states: tuple[torch.Tensor, ...] | None = None
    if torch_cuda_states is not None:
        cuda_states = _clone_cuda_rng_states(
            torch_cuda_states,
            "rng_states.torch_cuda_rng_states",
        )
        if cuda_states and not torch.cuda.is_available():
            raise ResumeContractError(
                "rng_states.torch_cuda_rng_states",
                "CUDA available",
                "CUDA unavailable",
                reason="cannot restore CUDA RNG states",
            )
    if python_state is not None:
        random.setstate(cast(Any, python_state))
    if torch_cpu_state is not None:
        torch.random.set_rng_state(torch_cpu_state)
    if cuda_states:
        torch.cuda.set_rng_state_all(list(cuda_states))


def capture_m5_global_rng_extension(
    *,
    include_cuda: bool = True,
    train_generator_device: torch.device | str = "cpu",
    probe_generator_device: torch.device | str = "cpu",
) -> dict[str, Any]:
    cuda_states: list[torch.Tensor] = []
    if include_cuda and torch.cuda.is_available():
        cuda_states = [
            state.detach().cpu().clone()
            for state in torch.cuda.get_rng_state_all()
        ]
    return {
        "schema": M5_RNG_EXTENSION_SCHEMA,
        "python_random_state": copy.deepcopy(random.getstate()),
        "torch_cpu_rng_state": torch.random.get_rng_state().detach().cpu().clone(),
        "torch_cuda_rng_states": cuda_states,
        "cuda_rng_captured": len(cuda_states) > 0,
        "cuda_device_count": len(cuda_states),
        "train_generator_device": str(torch.device(train_generator_device)),
        "probe_generator_device": str(torch.device(probe_generator_device)),
    }


def get_resumed_global_step(payload_or_contract: Mapping[str, Any]) -> int:
    if "resumed_global_step" in payload_or_contract:
        return _require_int(
            payload_or_contract["resumed_global_step"],
            "resumed_global_step",
        )
    return _require_int(_require_key(
        payload_or_contract,
        "global_step",
        "global_step",
    ))


def first_resumed_global_step(resumed_global_step: int) -> int:
    step = _require_int(resumed_global_step, "resumed_global_step")
    if step < 0:
        raise ResumeContractError(
            "resumed_global_step",
            "non-negative integer",
            step,
            reason="invalid resume step",
        )
    return int(step) + 1


def validate_target_global_step(
    *,
    resumed_global_step: int,
    target_global_step: int,
) -> int:
    resumed = _require_int(resumed_global_step, "resumed_global_step")
    target = _require_int(target_global_step, "target_global_step")
    if target <= resumed:
        raise ResumeContractError(
            "target_global_step",
            f"> resumed_global_step ({resumed})",
            target,
        )
    return target


def first_resumed_sample_identity(
    sample_plan: Mapping[str, Any],
    resumed_global_step: int,
) -> str:
    step = first_resumed_global_step(resumed_global_step)
    return str(m4_train_entry_for_step(sample_plan, step)["identity"])


def next_sample_after_target_identity(
    sample_plan: Mapping[str, Any],
    target_global_step: int,
) -> str:
    return str(
        m4_next_train_entry_after_global_step(sample_plan, target_global_step)[
            "identity"
        ]
    )


def optimizer_restore_report(
    payload: Mapping[str, Any],
    *,
    resolved_config: Mapping[str, Any],
) -> dict[str, Any]:
    optimizer_state = _require_mapping_field(payload, "optimizer", "optimizer")
    optimizer_contract = validate_optimizer_state_contract(
        optimizer_state,
        resolved_config=resolved_config,
        optimizer_group_lrs=_require_sequence(
            _require_key(
                payload,
                "optimizer_group_lrs",
                "optimizer_group_lrs",
            ),
            "optimizer_group_lrs",
        ),
    )
    return {
        "status": "ready",
        "restore_helper": "move_optimizer_state_to_device",
        "loaded_optimizer_state_restore_helper": (
            "move_loaded_optimizer_state_to_device"
        ),
        "state_reference_contract": "PASS",
        "param_group_contract": "PASS",
        "floating_state_finite": True,
        "state_entry_count": optimizer_contract["state_entry_count"],
        "param_group_count": optimizer_contract["param_group_count"],
        "state_tensor_count": optimizer_contract["state_tensor_count"],
        "param_group_tensor_count": optimizer_contract[
            "param_group_tensor_count"
        ],
        "param_groups": optimizer_contract["param_groups"],
    }


def canonical_locked_resolved_config(
    resolved_config: Mapping[str, Any],
) -> dict[str, Any]:
    canonical = _canonical_config_value(resolved_config, "resolved_config")
    if not isinstance(canonical, dict):
        raise ResumeContractError(
            "resolved_config",
            "mapping",
            type(canonical).__name__,
            reason="invalid resolved_config contract",
        )
    return canonical


def validate_optimizer_state_contract(
    optimizer_state_dict: Mapping[str, Any],
    *,
    resolved_config: Mapping[str, Any],
    optimizer_group_lrs: Sequence[Any],
) -> dict[str, Any]:
    m3_config = _require_mapping_field(
        resolved_config,
        "m3",
        "resolved_config.m3",
    )
    optimizer_config = _require_mapping_field(
        m3_config,
        "optimizer_config",
        "resolved_config.m3.optimizer_config",
    )
    optimizer_type = str(_require_key(
        optimizer_config,
        "optimizer",
        "resolved_config.m3.optimizer_config.optimizer",
    ))
    if optimizer_type != "AdamW":
        raise ResumeContractError(
            "resolved_config.m3.optimizer_config.optimizer",
            "AdamW",
            optimizer_type,
            reason="unsupported optimizer type",
        )
    expected_betas = _float_list(
        _require_key(
            optimizer_config,
            "betas",
            "resolved_config.m3.optimizer_config.betas",
        ),
        "resolved_config.m3.optimizer_config.betas",
    )
    if len(expected_betas) != 2:
        raise ResumeContractError(
            "resolved_config.m3.optimizer_config.betas",
            "two finite floats",
            expected_betas,
            reason="invalid optimizer contract",
        )
    expected_eps = _require_float(
        _require_key(
            optimizer_config,
            "eps",
            "resolved_config.m3.optimizer_config.eps",
        ),
        "resolved_config.m3.optimizer_config.eps",
    )
    expected_weight_decay = _require_float(
        _require_key(
            optimizer_config,
            "weight_decay",
            "resolved_config.m3.optimizer_config.weight_decay",
        ),
        "resolved_config.m3.optimizer_config.weight_decay",
    )
    expected_lrs = _expected_optimizer_group_lrs(m3_config)
    param_groups = _require_sequence(
        _require_key(
            optimizer_state_dict,
            "param_groups",
            "optimizer.param_groups",
        ),
        "optimizer.param_groups",
    )
    if len(param_groups) != len(M3_PARAMETER_GROUP_NAMES):
        raise ResumeContractError(
            "optimizer.param_groups",
            len(M3_PARAMETER_GROUP_NAMES),
            len(param_groups),
            reason="invalid optimizer param group contract",
        )
    summary_groups = _require_sequence(
        optimizer_group_lrs,
        "optimizer_group_lrs",
    )
    if len(summary_groups) != len(param_groups):
        raise ResumeContractError(
            "optimizer_group_lrs",
            len(param_groups),
            len(summary_groups),
            reason="optimizer group summary mismatch",
        )
    param_ids: set[int] = set()
    group_reports: list[dict[str, Any]] = []
    for index, raw_group in enumerate(param_groups):
        field_path = f"optimizer.param_groups[{index}]"
        group = _require_mapping(raw_group, field_path)
        expected_name = M3_PARAMETER_GROUP_NAMES[index]
        actual_name = str(_require_key(group, "name", f"{field_path}.name"))
        if actual_name != expected_name:
            raise ResumeContractError(
                f"{field_path}.name",
                expected_name,
                actual_name,
                reason="optimizer param group order/name mismatch",
            )
        expected_lr = expected_lrs[expected_name]
        actual_lr = _require_float(
            _require_key(group, "lr", f"{field_path}.lr"),
            f"{field_path}.lr",
        )
        if actual_lr != expected_lr:
            raise ResumeContractError(
                f"{field_path}.lr",
                expected_lr,
                actual_lr,
                reason="optimizer param group LR mismatch",
            )
        actual_betas = _float_list(
            _require_key(group, "betas", f"{field_path}.betas"),
            f"{field_path}.betas",
        )
        if actual_betas != expected_betas:
            raise ResumeContractError(
                f"{field_path}.betas",
                expected_betas,
                actual_betas,
                reason="optimizer param group betas mismatch",
            )
        actual_eps = _require_float(
            _require_key(group, "eps", f"{field_path}.eps"),
            f"{field_path}.eps",
        )
        if actual_eps != expected_eps:
            raise ResumeContractError(
                f"{field_path}.eps",
                expected_eps,
                actual_eps,
                reason="optimizer param group eps mismatch",
            )
        actual_weight_decay = _require_float(
            _require_key(group, "weight_decay", f"{field_path}.weight_decay"),
            f"{field_path}.weight_decay",
        )
        if actual_weight_decay != expected_weight_decay:
            raise ResumeContractError(
                f"{field_path}.weight_decay",
                expected_weight_decay,
                actual_weight_decay,
                reason="optimizer param group weight_decay mismatch",
            )
        group_param_ids = [
            _require_int(param_id, f"{field_path}.params[{param_index}]")
            for param_index, param_id in enumerate(_require_sequence(
                _require_key(group, "params", f"{field_path}.params"),
                f"{field_path}.params",
            ))
        ]
        duplicate_ids = param_ids.intersection(group_param_ids)
        if duplicate_ids:
            raise ResumeContractError(
                f"{field_path}.params",
                "unique parameter ids",
                sorted(duplicate_ids),
                reason="optimizer param id appears in multiple groups",
            )
        param_ids.update(group_param_ids)
        _validate_optimizer_group_summary(
            summary_groups[index],
            index=index,
            expected_name=expected_name,
            expected_lr=actual_lr,
            expected_weight_decay=actual_weight_decay,
            expected_tensor_count=len(group_param_ids),
        )
        group_reports.append(
            {
                "index": index,
                "name": expected_name,
                "lr": actual_lr,
                "betas": actual_betas,
                "eps": actual_eps,
                "weight_decay": actual_weight_decay,
                "param_ids": group_param_ids,
                "tensor_count": len(group_param_ids),
            }
        )
    state = _require_mapping_field(
        optimizer_state_dict,
        "state",
        "optimizer.state",
    )
    for raw_state_id, state_value in state.items():
        state_id = _require_int(raw_state_id, "optimizer.state.<param_id>")
        if state_id not in param_ids:
            raise ResumeContractError(
                f"optimizer.state[{state_id}]",
                "parameter id from optimizer.param_groups",
                state_id,
                reason="optimizer state references unknown parameter id",
            )
        _validate_optimizer_state_values(
            state_value,
            f"optimizer.state[{state_id}]",
        )
    return {
        "status": "PASS",
        "param_group_count": len(group_reports),
        "state_entry_count": len(state),
        "state_tensor_count": _count_tensors(state),
        "param_group_tensor_count": _count_tensors(param_groups),
        "param_groups": group_reports,
        "optimizer_group_lrs": [
            _canonical_optimizer_group_summary(group)
            for group in summary_groups
        ],
    }


def _validate_m3_payload(payload: Mapping[str, Any]) -> None:
    try:
        validate_m3_checkpoint_payload(payload)
    except (KeyError, RuntimeError, TypeError, ValueError) as exc:
        raise ResumeContractError(
            "checkpoint_payload",
            M3_CHECKPOINT_FORMAT,
            f"{type(exc).__name__}: {exc}",
            reason="invalid resume checkpoint payload",
        ) from exc
    if payload.get("format") != M3_CHECKPOINT_FORMAT:
        raise ResumeContractError(
            "format",
            M3_CHECKPOINT_FORMAT,
            payload.get("format"),
            reason="invalid resume checkpoint payload",
        )


def _validate_top_level_seed_consistency(
    payload: Mapping[str, Any],
    locked_fields: Mapping[str, Any],
) -> None:
    top_train_seed = _require_int(
        _require_key(payload, "train_seed", "train_seed")
    )
    top_probe_seed = _require_int(
        _require_key(payload, "probe_seed", "probe_seed")
    )
    if top_train_seed != locked_fields["m3.train_seed"]:
        raise ResumeContractError(
            "train_seed",
            locked_fields["m3.train_seed"],
            top_train_seed,
            reason="resume checkpoint internal mismatch",
        )
    if top_probe_seed != locked_fields["m3.probe_seed"]:
        raise ResumeContractError(
            "probe_seed",
            locked_fields["m3.probe_seed"],
            top_probe_seed,
            reason="resume checkpoint internal mismatch",
        )


def _manifest_path(
    m3_config: Mapping[str, Any],
    selected_metadata: Mapping[str, Any],
    manifest_path: Path | str | None,
) -> str:
    configured = manifest_path
    if configured is None:
        configured = _require_key(
            m3_config,
            "manifest",
            "resolved_config.m3.manifest",
        )
    normalized = _normalize_path(configured)
    metadata_path = selected_metadata.get("manifest_path")
    if metadata_path is not None and _normalize_path(metadata_path) != normalized:
        raise ResumeContractError(
            "selected_sample_metadata.manifest_path",
            normalized,
            _normalize_path(metadata_path),
            reason="resume checkpoint internal mismatch",
        )
    return normalized


def _manifest_sha256(
    m3_config: Mapping[str, Any],
    selected_metadata: Mapping[str, Any],
    manifest_sha256: str | None,
) -> str:
    value = manifest_sha256
    if value is None:
        raw = m3_config.get("manifest_sha256", _MISSING)
        if raw is not _MISSING:
            value = str(raw)
    if value is None:
        raw = selected_metadata.get("manifest_sha256", _MISSING)
        if raw is not _MISSING:
            value = str(raw)
    if value is None:
        raise ResumeContractError(
            "manifest.sha256",
            "present",
            "missing",
            reason="missing resume checkpoint field",
        )
    return _require_sha256(value, "manifest.sha256")


def _m4_plan_fields(
    m4_config: Mapping[str, Any],
    sample_plan: Mapping[str, Any] | None,
) -> dict[str, Any]:
    sample_plan_sha = _require_sha256(
        _require_key(
            m4_config,
            "sample_plan_sha256",
            "resolved_config.m4.sample_plan_sha256",
        ),
        "resolved_config.m4.sample_plan_sha256",
    )
    train_identities = _string_list(
        _require_key(
            m4_config,
            "train_sample_identities",
            "resolved_config.m4.train_sample_identities",
        ),
        "resolved_config.m4.train_sample_identities",
    )
    validation_identities = _string_list(
        _require_key(
            m4_config,
            "validation_sample_identities",
            "resolved_config.m4.validation_sample_identities",
        ),
        "resolved_config.m4.validation_sample_identities",
    )
    fixed_decode_identity = str(_require_key(
        m4_config,
        "fixed_decode_validation_identity",
        "resolved_config.m4.fixed_decode_validation_identity",
    ))
    if sample_plan is not None:
        validate_m4_sample_plan(sample_plan, expected_sha256=sample_plan_sha)
        actual_sha = m4_sample_plan_sha256(sample_plan)
        actual_train = [str(value) for value in sample_plan[
            "train_sample_identities"
        ]]
        actual_validation = [str(value) for value in sample_plan[
            "validation_sample_identities"
        ]]
        actual_fixed = str(sample_plan["fixed_decode_validation_identity"])
        if actual_train != train_identities:
            raise ResumeContractError(
                "resolved_config.m4.train_sample_identities",
                actual_train,
                train_identities,
                reason="resume checkpoint internal mismatch",
            )
        if actual_validation != validation_identities:
            raise ResumeContractError(
                "resolved_config.m4.validation_sample_identities",
                actual_validation,
                validation_identities,
                reason="resume checkpoint internal mismatch",
            )
        if actual_fixed != fixed_decode_identity:
            raise ResumeContractError(
                "resolved_config.m4.fixed_decode_validation_identity",
                actual_fixed,
                fixed_decode_identity,
                reason="resume checkpoint internal mismatch",
            )
        sample_plan_sha = actual_sha
    return {
        "sample_plan_sha256": sample_plan_sha,
        "train_sample_identities": train_identities,
        "validation_sample_identities": validation_identities,
        "fixed_decode_validation_identity": fixed_decode_identity,
    }


def _validate_m4_resume_plan(
    sample_plan: Mapping[str, Any],
    expected_fields: Mapping[str, Any],
    incompatibilities: list[dict[str, Any]],
) -> dict[str, Any]:
    try:
        report = validate_m4_sample_plan(
            sample_plan,
            expected_sha256=str(expected_fields["m4.sample_plan_sha256"]),
        )
    except (RuntimeError, TypeError, ValueError) as exc:
        incompatibilities.append(
            _incompatibility(
                "m4.sample_plan_sha256",
                expected_fields.get("m4.sample_plan_sha256"),
                f"{type(exc).__name__}: {exc}",
            )
        )
        return {"status": "FAIL"}
    return report


def _append_locked_field_incompatibilities(
    field_path: str,
    expected: Any,
    actual: Any,
    incompatibilities: list[dict[str, Any]],
) -> None:
    if field_path in {"resolved_config.locked", "optimizer.param_groups"}:
        _append_nested_incompatibilities(
            field_path,
            expected,
            actual,
            incompatibilities,
        )
        return
    incompatibilities.append(_incompatibility(field_path, expected, actual))


def _append_nested_incompatibilities(
    field_path: str,
    expected: Any,
    actual: Any,
    incompatibilities: list[dict[str, Any]],
) -> None:
    if isinstance(expected, Mapping) and isinstance(actual, Mapping):
        expected_keys = {str(key) for key in expected}
        actual_keys = {str(key) for key in actual}
        for key in sorted(expected_keys - actual_keys):
            incompatibilities.append(
                _incompatibility(f"{field_path}.{key}", expected[key], "missing")
            )
        for key in sorted(actual_keys - expected_keys):
            incompatibilities.append(
                _incompatibility(f"{field_path}.{key}", "absent", actual[key])
            )
        for key in sorted(expected_keys & actual_keys):
            _append_nested_incompatibilities(
                f"{field_path}.{key}",
                expected[key],
                actual[key],
                incompatibilities,
            )
        return
    if isinstance(expected, list) and isinstance(actual, list):
        limit = min(len(expected), len(actual))
        for index in range(limit):
            _append_nested_incompatibilities(
                f"{field_path}[{index}]",
                expected[index],
                actual[index],
                incompatibilities,
            )
        if len(expected) != len(actual):
            incompatibilities.append(
                _incompatibility(
                    f"{field_path}.length",
                    len(expected),
                    len(actual),
                )
            )
        return
    if expected != actual:
        incompatibilities.append(_incompatibility(field_path, expected, actual))


def _contract_locked_fields(contract: Mapping[str, Any]) -> dict[str, Any]:
    fields = _require_mapping_field(contract, "locked_fields", "locked_fields")
    return {str(key): _json_safe(value) for key, value in fields.items()}


def _current_locked_fields(current_run_fields: Mapping[str, Any]) -> dict[str, Any]:
    if "locked_fields" in current_run_fields:
        fields = _require_mapping_field(
            current_run_fields,
            "locked_fields",
            "current_run_fields.locked_fields",
        )
    else:
        fields = current_run_fields
    return {str(key): _json_safe(value) for key, value in fields.items()}


def _normalize_locked_field_value(field_path: str, value: Any) -> Any:
    if field_path in _PATH_LOCKED_FIELDS:
        return _normalize_path(value)
    if field_path in _GIT_SHA_LOCKED_FIELDS:
        return _require_git_sha(value, field_path)
    if field_path in _SHA256_LOCKED_FIELDS:
        return _require_sha256(value, field_path)
    if field_path in _IDENTITY_LIST_FIELDS:
        return _string_list(value, field_path)
    if field_path in _FLOAT_FIELDS:
        return _require_float(value, field_path)
    if field_path in _FLOAT_LIST_FIELDS:
        return _float_list(value, field_path)
    if field_path in _INT_FIELDS:
        return _require_int(value, field_path)
    if field_path in _INT_LIST_FIELDS:
        return _int_list(value, field_path)
    return _json_safe(value)


def _incompatibility(field_path: str, expected: Any, actual: Any) -> dict[str, Any]:
    return {
        "field_path": str(field_path),
        "expected": _json_safe(expected),
        "actual": _json_safe(actual),
    }


def _require_key(
    mapping: Mapping[str, Any],
    key: str,
    field_path: str,
) -> Any:
    if key not in mapping:
        raise ResumeContractError(
            field_path,
            "present",
            "missing",
            reason="missing resume checkpoint field",
        )
    return mapping[key]


def _require_mapping_field(
    mapping: Mapping[str, Any],
    key: str,
    field_path: str,
) -> Mapping[str, Any]:
    return _require_mapping(_require_key(mapping, key, field_path), field_path)


def _require_mapping(value: Any, field_path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ResumeContractError(
            field_path,
            "mapping",
            type(value).__name__,
            reason="invalid resume checkpoint field",
        )
    return value


def _require_sequence(value: Any, field_path: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ResumeContractError(
            field_path,
            "sequence",
            type(value).__name__,
            reason="invalid resume checkpoint field",
        )
    return value


def _require_int(value: Any, field_path: str = "value") -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ResumeContractError(
            field_path,
            "integer",
            value,
            reason="invalid resume checkpoint field",
        )
    return int(value)


def _require_bool(value: Any, field_path: str) -> bool:
    if not isinstance(value, bool):
        raise ResumeContractError(
            field_path,
            "bool",
            value,
            reason="invalid resume checkpoint field",
        )
    return bool(value)


def _require_float(value: Any, field_path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ResumeContractError(
            field_path,
            "finite float",
            value,
            reason="invalid resume checkpoint field",
        )
    result = float(value)
    if not math.isfinite(result):
        raise ResumeContractError(
            field_path,
            "finite float",
            value,
            reason="invalid resume checkpoint field",
        )
    return result


def _require_git_sha(value: Any, field_path: str) -> str:
    if not isinstance(value, str):
        raise ResumeContractError(
            field_path,
            "40-character lowercase hex Git SHA",
            value,
            reason="invalid git SHA field",
        )
    try:
        return validate_git_sha(value, name=field_path)
    except RuntimeError as exc:
        raise ResumeContractError(
            field_path,
            "40-character lowercase hex Git SHA",
            value,
            reason="invalid git SHA field",
        ) from exc


def _require_sha256(value: Any, field_path: str) -> str:
    text = str(value)
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise ResumeContractError(
            field_path,
            "64-character lowercase hex SHA256",
            value,
            reason="invalid resume checkpoint field",
        )
    return text


def _string_list(value: Any, field_path: str) -> list[str]:
    sequence = _require_sequence(value, field_path)
    if not all(isinstance(item, str) for item in sequence):
        raise ResumeContractError(
            field_path,
            "list[str]",
            value,
            reason="invalid resume checkpoint field",
        )
    return [str(item) for item in sequence]


def _float_list(value: Any, field_path: str) -> list[float]:
    sequence = _require_sequence(value, field_path)
    return [
        _require_float(item, f"{field_path}[{index}]")
        for index, item in enumerate(sequence)
    ]


def _int_list(value: Any, field_path: str) -> list[int]:
    sequence = _require_sequence(value, field_path)
    return [
        _require_int(item, f"{field_path}[{index}]")
        for index, item in enumerate(sequence)
    ]


def _normalize_path(value: Any) -> str:
    return str(Path(str(value)).resolve())


def _canonical_config_value(value: Any, field_path: str) -> Any:
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key in sorted(value.keys(), key=str):
            if not isinstance(key, str):
                raise ResumeContractError(
                    f"{field_path}.<key>",
                    "string key",
                    key,
                    reason="unsupported resolved_config type",
                )
            child_path = f"{field_path}.{key}"
            relative_path = child_path.removeprefix("resolved_config.")
            if relative_path in M5_ALLOWED_CONFIG_OVERRIDE_FIELDS:
                continue
            result[key] = _canonical_config_value(value[key], child_path)
        return result
    if isinstance(value, tuple):
        return [
            _canonical_config_value(item, f"{field_path}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, list):
        return [
            _canonical_config_value(item, f"{field_path}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ResumeContractError(
                field_path,
                "finite float",
                value,
                reason="invalid resolved_config field",
            )
        return float(value)
    raise ResumeContractError(
        field_path,
        "JSON scalar/list/mapping",
        type(value).__name__,
        reason="unsupported resolved_config type",
    )


def _expected_optimizer_group_lrs(
    m3_config: Mapping[str, Any],
) -> dict[str, float]:
    backbone_lr = _require_float(
        _require_key(m3_config, "backbone_lr", "resolved_config.m3.backbone_lr"),
        "resolved_config.m3.backbone_lr",
    )
    patch_embedding_lr = _require_float(
        _require_key(
            m3_config,
            "patch_embedding_lr",
            "resolved_config.m3.patch_embedding_lr",
        ),
        "resolved_config.m3.patch_embedding_lr",
    )
    mcp_lr = _require_float(
        _require_key(m3_config, "mcp_lr", "resolved_config.m3.mcp_lr"),
        "resolved_config.m3.mcp_lr",
    )
    result = {
        "backbone": backbone_lr,
        "patch_embedding": patch_embedding_lr,
    }
    for group_name in M3_PARAMETER_GROUP_NAMES:
        if group_name.startswith("mcp_"):
            result[group_name] = mcp_lr
    return result


def _validate_optimizer_group_summary(
    raw_summary: Any,
    *,
    index: int,
    expected_name: str,
    expected_lr: float,
    expected_weight_decay: float,
    expected_tensor_count: int,
) -> None:
    field_path = f"optimizer_group_lrs[{index}]"
    summary = _require_mapping(raw_summary, field_path)
    actual_index = _require_int(
        _require_key(summary, "index", f"{field_path}.index"),
        f"{field_path}.index",
    )
    if actual_index != index:
        raise ResumeContractError(
            f"{field_path}.index",
            index,
            actual_index,
            reason="optimizer group summary mismatch",
        )
    actual_name = str(_require_key(summary, "name", f"{field_path}.name"))
    if actual_name != expected_name:
        raise ResumeContractError(
            f"{field_path}.name",
            expected_name,
            actual_name,
            reason="optimizer group summary mismatch",
        )
    actual_lr = _require_float(
        _require_key(summary, "lr", f"{field_path}.lr"),
        f"{field_path}.lr",
    )
    if actual_lr != expected_lr:
        raise ResumeContractError(
            f"{field_path}.lr",
            expected_lr,
            actual_lr,
            reason="optimizer group summary mismatch",
        )
    actual_weight_decay = _require_float(
        _require_key(summary, "weight_decay", f"{field_path}.weight_decay"),
        f"{field_path}.weight_decay",
    )
    if actual_weight_decay != expected_weight_decay:
        raise ResumeContractError(
            f"{field_path}.weight_decay",
            expected_weight_decay,
            actual_weight_decay,
            reason="optimizer group summary mismatch",
        )
    actual_tensor_count = _require_int(
        _require_key(summary, "tensor_count", f"{field_path}.tensor_count"),
        f"{field_path}.tensor_count",
    )
    if actual_tensor_count != expected_tensor_count:
        raise ResumeContractError(
            f"{field_path}.tensor_count",
            expected_tensor_count,
            actual_tensor_count,
            reason="optimizer group summary mismatch",
        )


def _canonical_optimizer_group_summary(raw_summary: Any) -> dict[str, Any]:
    summary = _require_mapping(raw_summary, "optimizer_group_lrs[]")
    return {
        "index": _require_int(
            _require_key(summary, "index", "optimizer_group_lrs[].index"),
            "optimizer_group_lrs[].index",
        ),
        "name": str(_require_key(summary, "name", "optimizer_group_lrs[].name")),
        "lr": _require_float(
            _require_key(summary, "lr", "optimizer_group_lrs[].lr"),
            "optimizer_group_lrs[].lr",
        ),
        "weight_decay": _require_float(
            _require_key(
                summary,
                "weight_decay",
                "optimizer_group_lrs[].weight_decay",
            ),
            "optimizer_group_lrs[].weight_decay",
        ),
        "tensor_count": _require_int(
            _require_key(
                summary,
                "tensor_count",
                "optimizer_group_lrs[].tensor_count",
            ),
            "optimizer_group_lrs[].tensor_count",
        ),
    }


def _validate_optimizer_state_values(value: Any, field_path: str) -> None:
    if isinstance(value, torch.Tensor):
        if torch.is_floating_point(value) and not bool(torch.isfinite(value).all()):
            raise ResumeContractError(
                field_path,
                "finite optimizer state tensor",
                _rng_tensor_summary(value.detach().cpu(), field_path),
                reason="non-finite optimizer state",
            )
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            _validate_optimizer_state_values(item, f"{field_path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_optimizer_state_values(item, f"{field_path}[{index}]")
        return
    if isinstance(value, float):
        _require_float(value, field_path)
        return
    if isinstance(value, bool) or value is None or isinstance(value, (int, str)):
        return
    raise ResumeContractError(
        field_path,
        "optimizer state scalar/container/tensor",
        type(value).__name__,
        reason="unsupported optimizer state type",
    )


def _optimizer_value_to_device_copy(
    value: Any,
    *,
    device: torch.device,
    field_path: str,
) -> tuple[Any, int]:
    if isinstance(value, torch.Tensor):
        _validate_optimizer_state_values(value, field_path)
        return value.detach().clone().to(device), 1
    if isinstance(value, Mapping):
        result: dict[Any, Any] = {}
        tensor_count = 0
        for key, item in value.items():
            moved, count = _optimizer_value_to_device_copy(
                item,
                device=device,
                field_path=f"{field_path}.{key}",
            )
            result[key] = moved
            tensor_count += count
        return result, tensor_count
    if isinstance(value, tuple):
        moved_items = []
        tensor_count = 0
        for index, item in enumerate(value):
            moved, count = _optimizer_value_to_device_copy(
                item,
                device=device,
                field_path=f"{field_path}[{index}]",
            )
            moved_items.append(moved)
            tensor_count += count
        return tuple(moved_items), tensor_count
    if isinstance(value, list):
        moved_items = []
        tensor_count = 0
        for index, item in enumerate(value):
            moved, count = _optimizer_value_to_device_copy(
                item,
                device=device,
                field_path=f"{field_path}[{index}]",
            )
            moved_items.append(moved)
            tensor_count += count
        return moved_items, tensor_count
    _validate_optimizer_state_values(value, field_path)
    return value, 0


def _optimizer_value_to_device_in_place(
    value: Any,
    *,
    device: torch.device,
    field_path: str,
) -> tuple[Any, int, int]:
    if isinstance(value, torch.Tensor):
        _validate_optimizer_state_values(value, field_path)
        return value.to(device), 1, int(torch.is_floating_point(value))
    if isinstance(value, MutableMapping):
        tensor_count = 0
        floating_tensor_count = 0
        for key in list(value.keys()):
            moved, moved_tensors, moved_floating = (
                _optimizer_value_to_device_in_place(
                    value[key],
                    device=device,
                    field_path=f"{field_path}.{key}",
                )
            )
            value[key] = moved
            tensor_count += moved_tensors
            floating_tensor_count += moved_floating
        return value, tensor_count, floating_tensor_count
    if isinstance(value, list):
        tensor_count = 0
        floating_tensor_count = 0
        for index, item in enumerate(value):
            moved, moved_tensors, moved_floating = (
                _optimizer_value_to_device_in_place(
                    item,
                    device=device,
                    field_path=f"{field_path}[{index}]",
                )
            )
            value[index] = moved
            tensor_count += moved_tensors
            floating_tensor_count += moved_floating
        return value, tensor_count, floating_tensor_count
    if isinstance(value, tuple):
        moved_items = []
        tensor_count = 0
        floating_tensor_count = 0
        for index, item in enumerate(value):
            moved, moved_tensors, moved_floating = (
                _optimizer_value_to_device_in_place(
                    item,
                    device=device,
                    field_path=f"{field_path}[{index}]",
                )
            )
            moved_items.append(moved)
            tensor_count += moved_tensors
            floating_tensor_count += moved_floating
        return tuple(moved_items), tensor_count, floating_tensor_count
    _validate_optimizer_state_values(value, field_path)
    return value, 0, 0


def _clone_rng_state_tensor(value: Any, field_path: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise ResumeContractError(
            field_path,
            "torch.uint8 RNG state tensor",
            type(value).__name__,
            reason="invalid RNG state",
        )
    if value.dtype != torch.uint8:
        raise ResumeContractError(
            field_path,
            "torch.uint8 RNG state tensor",
            str(value.dtype),
            reason="invalid RNG state",
        )
    if value.numel() <= 0:
        raise ResumeContractError(
            field_path,
            "non-empty RNG state tensor",
            int(value.numel()),
            reason="invalid RNG state",
        )
    return value.detach().cpu().clone()


def _validate_torch_cpu_rng_state(state: torch.Tensor, field_path: str) -> None:
    generator = torch.Generator(device="cpu")
    try:
        generator.set_state(state)
    except RuntimeError as exc:
        raise ResumeContractError(
            field_path,
            "valid torch CPU RNG state",
            _rng_tensor_summary(state, field_path),
            reason="invalid RNG state",
        ) from exc


def _validate_python_random_state(state: Any, field_path: str) -> None:
    rng = random.Random()
    try:
        rng.setstate(cast(Any, state))
    except (TypeError, ValueError) as exc:
        raise ResumeContractError(
            field_path,
            "valid Python random state",
            type(state).__name__,
            reason="invalid RNG state",
        ) from exc


def _clone_cuda_rng_states(value: Any, field_path: str) -> tuple[torch.Tensor, ...]:
    sequence = _require_sequence(value, field_path)
    return tuple(
        _clone_rng_state_tensor(item, f"{field_path}[{index}]")
        for index, item in enumerate(sequence)
    )


def _rng_tensor_summary(state: torch.Tensor, field_path: str) -> dict[str, Any]:
    return {
        "available": True,
        "field": field_path,
        "dtype": str(state.dtype),
        "shape": [int(dim) for dim in state.shape],
        "numel": int(state.numel()),
        "sha256": tensor_sha256(state),
    }


def _count_tensors(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return 1
    if isinstance(value, Mapping):
        return sum(_count_tensors(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_count_tensors(item) for item in value)
    return 0


def _json_safe(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return {
            "tensor": True,
            "dtype": str(value.dtype),
            "shape": [int(dim) for dim in value.shape],
            "sha256": tensor_sha256(value.detach().cpu()),
        }
    if isinstance(value, Path):
        return _normalize_path(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ResumeContractError(
                "json_value",
                "finite float",
                value,
                reason="invalid JSON-serializable resume value",
            )
        return float(value)
    raise ResumeContractError(
        "json_value",
        "JSON-serializable value",
        type(value).__name__,
        reason="unsupported resume value type",
    )


def _short_repr(value: Any) -> str:
    text = repr(value)
    if len(text) <= 240:
        return text
    return text[:237] + "..."
