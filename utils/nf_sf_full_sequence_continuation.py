from __future__ import annotations

import copy
import hashlib
import json
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from utils.checkpoint import is_mcp_state_key
from utils.nf_sf_m3 import file_sha256
from utils.nf_sf_tensors import (
    DEFAULT_S_MAIN,
    DEFAULT_S_MCP,
    FULL_SEQUENCE_RNG_DRAW_ORDER_VERSION,
)
from utils.nf_sf_training import (
    FULL_SEQUENCE_CHECKPOINT_STEPS,
    FULL_SEQUENCE_DEPTH_WEIGHTS,
    FULL_SEQUENCE_OBJECTIVE_VERSION,
    FULL_SEQUENCE_RUN_KIND,
    FULL_SEQUENCE_TARGET_GLOBAL_STEP,
    FULL_SEQUENCE_TRAINER_SCHEMA,
    OFFICIAL_SELF_FORCING_CHECKPOINT_SHA256,
    nf_sf_full_sequence_train_cursor,
)


CONTINUATION_SCHEMA = "nf_sf_full_sequence_continuation_v1"
BASE_TRAINING_GIT_SHA = "2ab9b3a7c08b09140b6cbae23df21107817fe3be"
CONTINUATION_STAGE_PAIRS = ((5000, 6500), (6500, 8000))
CONTINUATION_TARGET_STEPS = (6500, 8000)
CONTINUATION_OBJECTIVE_MODE = "next_forcing_full"
CONTINUATION_ALLOWED_PARENT_STEPS = (5000, 6500)
CHECKPOINT_VALIDATION_SCHEMA = "nf_sf_full_sequence_checkpoint_validation_v1"
TAP_LAYERS = (3, 11, 19, 29)
MCP_NUM_LAYERS = 3
ADAMW_BETAS = (0.0, 0.999)
ADAMW_EPS = 1.0e-8
CANONICAL_OPTIMIZER_GROUP_NAMES = (
    "backbone",
    "patch_embedding",
    "mcp_fusion",
    "mcp_depth1",
    "mcp_depth2",
    "mcp_depth3",
)


@dataclass(frozen=True)
class ContinuationParentCheckpoint:
    path: Path
    sha256: str
    validation_sidecar: Mapping[str, Any]
    payload: Mapping[str, Any]
    parent_global_step: int
    parent_git_sha: str
    semantic_lock_fingerprint: str


def validate_continuation_stage_pair(parent_step: int, target_step: int) -> tuple[int, int]:
    pair = (int(parent_step), int(target_step))
    if pair not in CONTINUATION_STAGE_PAIRS:
        raise ValueError("continuation stage must be exactly 5000->6500 or 6500->8000")
    return pair


def continuation_checkpoint_steps(parent_step: int, target_step: int) -> tuple[int, ...]:
    validate_continuation_stage_pair(parent_step, target_step)
    return (int(target_step),)


def continuation_validation_steps(parent_step: int, target_step: int) -> tuple[int, ...]:
    validate_continuation_stage_pair(parent_step, target_step)
    return (int(target_step),)


def continuation_start_step(parent_step: int, target_step: int) -> int:
    validate_continuation_stage_pair(parent_step, target_step)
    return int(parent_step) + 1


def validate_sha256(value: str, *, name: str) -> str:
    text = str(value).strip().lower()
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise ValueError(f"{name} must be a lowercase SHA256 hex string")
    return text


def validate_git_sha(value: str, *, name: str) -> str:
    text = str(value).strip().lower()
    if len(text) != 40 or any(char not in "0123456789abcdef" for char in text):
        raise ValueError(f"{name} must be a lowercase git SHA")
    return text


def checkpoint_sidecar_paths(path: Path | str) -> dict[str, Path]:
    checkpoint_path = Path(path)
    stem = checkpoint_path.with_suffix("")
    return {
        "sha256": stem.with_suffix(".sha256.txt"),
        "validation": stem.with_suffix(".validation.json"),
    }


def validate_checkpoint_sidecars(path: Path | str) -> dict[str, Any]:
    path = Path(path)
    sidecars = checkpoint_sidecar_paths(path)
    if not sidecars["sha256"].is_file():
        raise RuntimeError("checkpoint SHA256 sidecar is missing")
    if not sidecars["validation"].is_file():
        raise RuntimeError("checkpoint validation sidecar is missing")
    actual_sha = file_sha256(path)
    sha_text = sidecars["sha256"].read_text(encoding="utf-8").strip().split()
    if not sha_text or sha_text[0] != actual_sha:
        raise RuntimeError("checkpoint SHA256 sidecar mismatch")
    validation = json.loads(sidecars["validation"].read_text(encoding="utf-8"))
    if validation.get("status") != "PASS":
        raise RuntimeError("checkpoint validation sidecar is not PASS")
    if validation.get("schema") != CHECKPOINT_VALIDATION_SCHEMA:
        raise RuntimeError("checkpoint validation sidecar schema mismatch")
    if validation.get("sha256") != actual_sha:
        raise RuntimeError("checkpoint validation sidecar SHA mismatch")
    if int(validation.get("size_bytes", -1)) != int(path.stat().st_size):
        raise RuntimeError("checkpoint validation sidecar size mismatch")
    return validation


def validate_checkpoint_payload(payload: Mapping[str, Any]) -> None:
    required = {
        "schema",
        "run_kind",
        "objective_version",
        "objective_mode",
        "global_step",
        "generator",
        "optimizer",
        "train_rng_state",
        "validation_seed",
        "validation_base_rng_state",
        "python_random_state",
        "torch_cpu_global_rng_state",
        "torch_cuda_global_rng_state",
        "sample_cursor",
        "sample_plan_sha256",
        "manifest_sha256",
        "conditionals_artifact_sha256",
        "resolved_config",
        "provenance",
        "reference_checkpoint",
        "optimizer_contract",
    }
    missing = required - set(payload.keys())
    if missing:
        raise RuntimeError(f"full-sequence checkpoint missing fields: {sorted(missing)}")
    if payload["schema"] != FULL_SEQUENCE_TRAINER_SCHEMA:
        raise RuntimeError("full-sequence checkpoint schema mismatch")
    if payload["run_kind"] != FULL_SEQUENCE_RUN_KIND:
        raise RuntimeError("full-sequence checkpoint run_kind mismatch")
    if payload["objective_version"] != FULL_SEQUENCE_OBJECTIVE_VERSION:
        raise RuntimeError("full-sequence checkpoint objective_version mismatch")
    if payload["objective_mode"] != CONTINUATION_OBJECTIVE_MODE:
        raise RuntimeError("full-sequence checkpoint objective_mode mismatch")
    reference = payload["reference_checkpoint"]
    if not isinstance(reference, Mapping):
        raise RuntimeError("full-sequence checkpoint reference_checkpoint missing")
    if reference.get("sha256") != OFFICIAL_SELF_FORCING_CHECKPOINT_SHA256:
        raise RuntimeError("full-sequence checkpoint official SHA mismatch")
    if payload["sample_cursor"] != nf_sf_full_sequence_train_cursor(
        int(payload["global_step"])
    ):
        raise RuntimeError("full-sequence checkpoint sample_cursor mismatch")
    for key in ("train_rng_state", "validation_base_rng_state", "torch_cpu_global_rng_state"):
        if not torch.is_tensor(payload.get(key)):
            raise RuntimeError(f"full-sequence checkpoint {key} must be a tensor")


def move_optimizer_state_to_device(
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device=device)


def load_continuation_parent_checkpoint(
    path: Path | str,
    *,
    expected_parent_checkpoint_sha256: str,
    expected_parent_global_step: int,
    expected_parent_checkpoint_git_sha: str,
    sample_plan_sha256: str,
    manifest_sha256: str,
    conditionals_artifact_sha256: str,
) -> ContinuationParentCheckpoint:
    parent_step = int(expected_parent_global_step)
    if parent_step not in CONTINUATION_ALLOWED_PARENT_STEPS:
        raise ValueError("continuation parent must be step5000 or step6500")
    path = Path(path)
    expected_name = f"checkpoint_step{parent_step:06d}.pt"
    if path.name != expected_name:
        raise RuntimeError(f"parent checkpoint filename must be {expected_name}")
    if not path.is_file():
        raise FileNotFoundError(f"parent checkpoint not found: {path}")
    expected_sha = validate_sha256(
        expected_parent_checkpoint_sha256,
        name="--expected_parent_checkpoint_sha256",
    )
    actual_sha = file_sha256(path)
    if actual_sha != expected_sha:
        raise RuntimeError("parent checkpoint SHA256 mismatch")
    _require_sidecar_names(path)
    validation = validate_checkpoint_sidecars(path=path)
    _validate_parent_validation_sidecar(
        validation,
        path=path,
        expected_checkpoint_step=parent_step,
        expected_sha256=actual_sha,
    )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise TypeError("parent checkpoint payload must be a mapping")
    if int(payload.get("global_step", -1)) != parent_step:
        raise RuntimeError("parent checkpoint global_step mismatch")
    validate_checkpoint_payload(payload)
    _validate_sidecar_payload_cross_checks(validation, payload=payload, path=path)
    expected_git = validate_git_sha(
        expected_parent_checkpoint_git_sha,
        name="--expected_parent_checkpoint_git_sha",
    )
    _validate_parent_payload_contract(
        payload,
        expected_parent_global_step=parent_step,
        expected_parent_checkpoint_git_sha=expected_git,
        sample_plan_sha256=sample_plan_sha256,
        manifest_sha256=manifest_sha256,
        conditionals_artifact_sha256=conditionals_artifact_sha256,
    )
    semantic_fingerprint = semantic_lock_fingerprint(payload["resolved_config"])
    return ContinuationParentCheckpoint(
        path=path.resolve(),
        sha256=actual_sha,
        validation_sidecar=validation,
        payload=payload,
        parent_global_step=parent_step,
        parent_git_sha=str(payload["git_sha"]),
        semantic_lock_fingerprint=semantic_fingerprint,
    )


def validate_semantic_lock(resolved_config: Mapping[str, Any]) -> dict[str, Any]:
    required_exact = {
        "schema": FULL_SEQUENCE_TRAINER_SCHEMA,
        "run_kind": FULL_SEQUENCE_RUN_KIND,
        "objective_version": FULL_SEQUENCE_OBJECTIVE_VERSION,
        "objective_mode": CONTINUATION_OBJECTIVE_MODE,
        "checkpoint_sha256": OFFICIAL_SELF_FORCING_CHECKPOINT_SHA256,
        "dtype": "bf16",
        "device": "cuda:0",
        "num_frame_per_block": 3,
        "gradient_checkpointing": True,
        "full_teacher_frames": 21,
        "chunk_frames": 3,
        "num_chunks": 7,
        "main_shift": DEFAULT_S_MAIN,
        "mcp_shift": DEFAULT_S_MCP,
        "depth_weights": list(FULL_SEQUENCE_DEPTH_WEIGHTS),
        "tap_layers": list(TAP_LAYERS),
        "mcp_blocks_per_depth": MCP_NUM_LAYERS,
        "rng_draw_order_version": FULL_SEQUENCE_RNG_DRAW_ORDER_VERSION,
        "validation_tensor_slot": "nf_sf_full_sequence_next_forcing_v1",
        "validation_seed_derivation": "derive_m4_validation_seed",
        "validation_identity_noise_is_paired_across_steps": True,
        "anchor_micro_loop": True,
        "main_backbone_forward_count_per_train_sample": 1,
        "paper_exact_reproduction": False,
    }
    for key, expected in required_exact.items():
        actual = resolved_config.get(key)
        if actual != expected:
            raise RuntimeError(f"semantic lock mismatch for {key}")
    for key in (
        "train_seed",
        "validation_seed",
        "global_seed",
        "backbone_lr",
        "patch_embedding_lr",
        "mcp_lr",
        "weight_decay",
        "sample_plan_sha256",
        "manifest_sha256",
        "conditionals_artifact_sha256",
    ):
        if key not in resolved_config:
            raise RuntimeError(f"semantic lock missing {key}")
    if [float(value) for value in resolved_config.get("adam_betas", [])] != list(ADAMW_BETAS):
        raise RuntimeError("semantic lock mismatch for adam_betas")
    if float(resolved_config.get("adam_eps", -1.0)) != ADAMW_EPS:
        raise RuntimeError("semantic lock mismatch for adam_eps")
    if int(resolved_config.get("production_target_global_step", -1)) != FULL_SEQUENCE_TARGET_GLOBAL_STEP:
        raise RuntimeError("base production target must remain 5000")
    if tuple(resolved_config.get("production_checkpoint_steps", ())) != FULL_SEQUENCE_CHECKPOINT_STEPS:
        raise RuntimeError("base production checkpoint steps must remain canonical")
    if tuple(resolved_config.get("production_validation_steps", ())) != FULL_SEQUENCE_CHECKPOINT_STEPS:
        raise RuntimeError("base production validation steps must remain canonical")
    return {key: copy.deepcopy(resolved_config[key]) for key in sorted(required_exact)}


def semantic_lock_fingerprint(resolved_config: Mapping[str, Any]) -> str:
    validate_semantic_lock(resolved_config)
    locked = {
        key: copy.deepcopy(resolved_config[key])
        for key in (
            "objective_mode",
            "sample_plan_sha256",
            "manifest_sha256",
            "conditionals_artifact_sha256",
            "checkpoint_sha256",
            "train_seed",
            "validation_seed",
            "global_seed",
            "backbone_lr",
            "patch_embedding_lr",
            "mcp_lr",
            "weight_decay",
            "adam_betas",
            "adam_eps",
            "dtype",
            "device",
            "num_frame_per_block",
            "gradient_checkpointing",
            "full_teacher_frames",
            "chunk_frames",
            "num_chunks",
            "main_shift",
            "mcp_shift",
            "depth_weights",
            "tap_layers",
            "mcp_blocks_per_depth",
            "rng_draw_order_version",
            "validation_tensor_slot",
            "validation_seed_derivation",
            "validation_identity_noise_is_paired_across_steps",
            "anchor_micro_loop",
            "main_backbone_forward_count_per_train_sample",
            "paper_exact_reproduction",
            "production_target_global_step",
            "production_checkpoint_steps",
            "production_validation_steps",
        )
    }
    return _json_sha256(locked)


def validate_optimizer_contract_for_continuation(
    payload: Mapping[str, Any],
    active_optimizer_contract: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    contract = payload.get("optimizer_contract")
    if not isinstance(contract, Mapping):
        raise RuntimeError("parent optimizer_contract missing")
    if contract.get("class") != "AdamW":
        raise RuntimeError("continuation requires AdamW optimizer")
    if [float(value) for value in contract.get("betas", [])] != list(ADAMW_BETAS):
        raise RuntimeError("parent optimizer betas mismatch")
    if float(contract.get("eps", -1.0)) != ADAMW_EPS:
        raise RuntimeError("parent optimizer eps mismatch")
    groups = contract.get("param_groups")
    if not isinstance(groups, Sequence):
        raise RuntimeError("parent optimizer param_groups missing")
    contract_names = tuple(
        str(group.get("name")) for group in groups if isinstance(group, Mapping)
    )
    if contract_names != CANONICAL_OPTIMIZER_GROUP_NAMES:
        raise RuntimeError("parent optimizer group names/order mismatch")
    if "mcp" in contract_names:
        raise RuntimeError("parent optimizer group name mcp is not allowed")
    state = payload.get("optimizer")
    if not isinstance(state, Mapping) or not isinstance(state.get("param_groups"), Sequence):
        raise RuntimeError("parent optimizer state missing param_groups")
    if len(groups) != len(state["param_groups"]):
        raise RuntimeError("parent optimizer contract/state group count mismatch")
    state_names = tuple(
        str(group.get("name")) for group in state["param_groups"] if isinstance(group, Mapping)
    )
    if state_names != CANONICAL_OPTIMIZER_GROUP_NAMES:
        raise RuntimeError("parent optimizer state group names/order mismatch")
    for contract_group, state_group in zip(groups, state["param_groups"]):
        if contract_group.get("name") != state_group.get("name"):
            raise RuntimeError("parent optimizer group name mismatch")
        if float(contract_group.get("lr")) != float(state_group.get("lr")):
            raise RuntimeError("parent optimizer group LR mismatch")
        if float(contract_group.get("weight_decay")) != float(state_group.get("weight_decay", 0.0)):
            raise RuntimeError("parent optimizer group weight_decay mismatch")
        if int(contract_group.get("param_count", -1)) != len(state_group.get("params", ())):
            raise RuntimeError("parent optimizer group param_count mismatch")
    if active_optimizer_contract is not None and dict(contract) != dict(active_optimizer_contract):
        raise RuntimeError("active continuation optimizer contract mismatch")
    return contract


def restore_continuation_state(
    *,
    generator: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_rng: torch.Generator,
    validation_base_rng: torch.Generator,
    payload: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    missing, unexpected = generator.load_state_dict(payload["generator"], strict=True)
    if missing or unexpected:
        raise RuntimeError(f"parent generator strict load failed: {missing}, {unexpected}")
    if _count_mcp_tensors(generator.state_dict()) <= 0:
        raise RuntimeError("restored continuation generator missing MCP tensors")
    optimizer.load_state_dict(payload["optimizer"])
    move_optimizer_state_to_device(optimizer, device=device)
    train_rng.set_state(payload["train_rng_state"])
    validation_base_rng.set_state(payload["validation_base_rng_state"])
    random.setstate(payload["python_random_state"])
    torch.set_rng_state(payload["torch_cpu_global_rng_state"])
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA must be available to restore CUDA RNG state")
        if payload["torch_cuda_global_rng_state"] is None:
            raise RuntimeError("parent checkpoint missing CUDA global RNG state")
        torch.cuda.set_rng_state(payload["torch_cuda_global_rng_state"], device)
    return {
        "status": "PASS",
        "generator_key_count": len(generator.state_dict()),
        "optimizer_state_entry_count": len(optimizer.state),
        "rng_fingerprint": rng_fingerprint(
            train_rng=train_rng,
            validation_base_rng=validation_base_rng,
            device=device,
        ),
    }


def rng_fingerprint(
    *,
    train_rng: torch.Generator,
    validation_base_rng: torch.Generator,
    device: torch.device,
) -> dict[str, Any]:
    record = {
        "train_rng_state_sha256": _tensor_sha256(train_rng.get_state()),
        "validation_base_rng_state_sha256": _tensor_sha256(validation_base_rng.get_state()),
        "python_random_state_sha256": hashlib.sha256(repr(random.getstate()).encode("utf-8")).hexdigest(),
        "torch_cpu_global_rng_state_sha256": _tensor_sha256(torch.get_rng_state()),
    }
    if device.type == "cuda" and torch.cuda.is_available():
        record["torch_cuda_global_rng_state_sha256"] = _tensor_sha256(
            torch.cuda.get_rng_state(device)
        )
    else:
        record["torch_cuda_global_rng_state_sha256"] = None
    return record


def first_continuation_step_contract(
    *,
    parent_step: int,
    target_step: int,
    train_identity: str,
    sample_cursor: Mapping[str, Any],
) -> dict[str, Any]:
    start_step = continuation_start_step(parent_step, target_step)
    expected_cursor = nf_sf_full_sequence_train_cursor(start_step)
    if dict(sample_cursor) != dict(expected_cursor):
        raise RuntimeError("first continuation sample_cursor mismatch")
    return {
        "status": "PASS",
        "parent_global_step": int(parent_step),
        "target_global_step": int(target_step),
        "first_global_step": int(start_step),
        "first_sample_identity": str(train_identity),
        "first_sample_cursor": dict(sample_cursor),
    }


def build_continuation_resolved_config(
    parent_resolved_config: Mapping[str, Any],
    *,
    runtime_git_sha: str,
    parent: ContinuationParentCheckpoint,
    target_global_step: int,
) -> dict[str, Any]:
    validate_continuation_stage_pair(parent.parent_global_step, target_global_step)
    validate_semantic_lock(parent_resolved_config)
    resolved = copy.deepcopy(dict(parent_resolved_config))
    resolved["expected_git_sha"] = str(runtime_git_sha)
    resolved["current_git_sha"] = str(runtime_git_sha)
    resolved.update(
        {
            "continuation_schema": CONTINUATION_SCHEMA,
            "continuation_parent_global_step": int(parent.parent_global_step),
            "continuation_target_global_step": int(target_global_step),
            "continuation_checkpoint_steps": [int(target_global_step)],
            "continuation_validation_steps": [int(target_global_step)],
            "base_training_git_sha": str(
                resolved.get("base_training_git_sha", BASE_TRAINING_GIT_SHA)
            ),
            "parent_checkpoint_git_sha": str(parent.parent_git_sha),
            "parent_checkpoint_sha256": str(parent.sha256),
            "parent_checkpoint_path": str(parent.path),
            "semantic_lock_fingerprint": semantic_lock_fingerprint(parent_resolved_config),
        }
    )
    return resolved


def build_continuation_provenance(
    parent_provenance: Mapping[str, Any],
    *,
    runtime_git_sha: str,
    parent: ContinuationParentCheckpoint,
    target_global_step: int,
) -> dict[str, Any]:
    validate_continuation_stage_pair(parent.parent_global_step, target_global_step)
    if parent_provenance.get("paper_exact_reproduction") is not False:
        raise RuntimeError("continuation parent provenance must not be paper-exact")
    provenance = copy.deepcopy(dict(parent_provenance))
    provenance["continuation"] = {
        "schema": CONTINUATION_SCHEMA,
        "continuation_only_training_horizon_extension": True,
        "objective_changed": False,
        "architecture_changed": False,
        "optimizer_hparams_changed": False,
        "data_contract_changed": False,
        "rng_semantics_changed": False,
        "freeze_policy_changed": False,
        "parent_global_step": int(parent.parent_global_step),
        "target_global_step": int(target_global_step),
        "runtime_git_sha": str(runtime_git_sha),
        "base_training_git_sha": BASE_TRAINING_GIT_SHA,
        "parent_checkpoint_git_sha": str(parent.parent_git_sha),
        "parent_checkpoint_sha256": str(parent.sha256),
    }
    return provenance


def build_continuation_summary(
    *,
    parent: ContinuationParentCheckpoint,
    runtime_git_sha: str,
    target_global_step: int,
    metrics_path: Path,
    train_record_count: int,
    final_train_record: Mapping[str, Any] | None,
    checkpoint_records: Sequence[Mapping[str, Any]],
    validation_summaries: Sequence[Mapping[str, Any]],
    semantic_lock_fingerprint_value: str,
    restored_rng_fingerprint: Mapping[str, Any],
    first_step_contract: Mapping[str, Any],
    reference_checkpoint_immutability: Mapping[str, Any],
    memory_maxima: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": CONTINUATION_SCHEMA,
        "status": "DONE",
        "parent_checkpoint_path": str(parent.path),
        "parent_checkpoint_sha256": str(parent.sha256),
        "parent_checkpoint_git_sha": str(parent.parent_git_sha),
        "parent_global_step": int(parent.parent_global_step),
        "runtime_git_sha": str(runtime_git_sha),
        "target_global_step": int(target_global_step),
        "start_step": int(parent.parent_global_step) + 1,
        "end_step": int(target_global_step),
        "train_record_count": int(train_record_count),
        "metrics_jsonl": str(metrics_path.resolve()),
        "final_train_record": None if final_train_record is None else dict(final_train_record),
        "checkpoint_records": [dict(record) for record in checkpoint_records],
        "validation_reports": [dict(record) for record in validation_summaries],
        "semantic_lock_fingerprint": str(semantic_lock_fingerprint_value),
        "restored_rng_fingerprint": dict(restored_rng_fingerprint),
        "first_continuation_step": dict(first_step_contract),
        "reference_checkpoint_immutability": dict(reference_checkpoint_immutability),
        "memory_maxima": dict(memory_maxima),
        "no_direct_5000_to_8000": True,
    }


def _validate_parent_validation_sidecar(
    validation: Mapping[str, Any],
    *,
    path: Path,
    expected_checkpoint_step: int,
    expected_sha256: str,
) -> None:
    if validation.get("status") != "PASS":
        raise RuntimeError("parent checkpoint validation sidecar must be PASS")
    if validation.get("schema") != CHECKPOINT_VALIDATION_SCHEMA:
        raise RuntimeError("parent checkpoint validation schema mismatch")
    if validation.get("sha256") != expected_sha256:
        raise RuntimeError("parent checkpoint validation SHA mismatch")
    if str(validation.get("path")) != str(path.resolve()):
        raise RuntimeError("parent checkpoint validation path mismatch")
    if int(validation.get("size_bytes", -1)) != int(path.stat().st_size):
        raise RuntimeError("parent checkpoint validation size mismatch")
    if validation.get("run_kind") != FULL_SEQUENCE_RUN_KIND:
        raise RuntimeError("parent checkpoint validation run_kind mismatch")
    if validation.get("objective_version") != FULL_SEQUENCE_OBJECTIVE_VERSION:
        raise RuntimeError("parent checkpoint validation objective_version mismatch")
    if validation.get("objective_mode") != CONTINUATION_OBJECTIVE_MODE:
        raise RuntimeError("parent checkpoint validation objective_mode mismatch")
    if int(validation.get("global_step", -1)) != int(expected_checkpoint_step):
        raise RuntimeError("parent checkpoint validation global_step mismatch")


def _validate_parent_payload_contract(
    payload: Mapping[str, Any],
    *,
    expected_parent_global_step: int,
    expected_parent_checkpoint_git_sha: str,
    sample_plan_sha256: str,
    manifest_sha256: str,
    conditionals_artifact_sha256: str,
) -> None:
    if payload.get("status") != "PRODUCTION":
        raise RuntimeError("parent checkpoint status must be PRODUCTION")
    if int(payload.get("global_step", -1)) != int(expected_parent_global_step):
        raise RuntimeError("parent checkpoint global_step mismatch")
    if str(payload.get("git_sha")) != str(expected_parent_checkpoint_git_sha):
        raise RuntimeError("parent checkpoint git_sha mismatch")
    if int(expected_parent_global_step) == 5000 and str(payload.get("git_sha")) != BASE_TRAINING_GIT_SHA:
        raise RuntimeError("canonical step5000 parent must use base training git")
    for key, expected in (
        ("sample_plan_sha256", sample_plan_sha256),
        ("manifest_sha256", manifest_sha256),
        ("conditionals_artifact_sha256", conditionals_artifact_sha256),
    ):
        expected_sha = validate_sha256(str(expected), name=key)
        if payload.get(key) != expected_sha:
            raise RuntimeError(f"parent checkpoint {key} mismatch")
        resolved_value = payload["resolved_config"].get(key)
        if resolved_value != payload.get(key) or resolved_value != expected_sha:
            raise RuntimeError(f"parent resolved_config {key} mismatch")
    if payload.get("sample_cursor") != nf_sf_full_sequence_train_cursor(
        int(expected_parent_global_step)
    ):
        raise RuntimeError("parent checkpoint sample_cursor mismatch")
    if _count_mcp_tensors(payload["generator"]) <= 0:
        raise RuntimeError("parent generator state missing MCP tensors")
    validate_semantic_lock(payload["resolved_config"])
    if int(expected_parent_global_step) == 5000:
        resolved = payload["resolved_config"]
        if int(resolved.get("production_target_global_step", -1)) != 5000:
            raise RuntimeError("canonical parent production target must remain 5000")
        if tuple(resolved.get("production_checkpoint_steps", ())) != FULL_SEQUENCE_CHECKPOINT_STEPS:
            raise RuntimeError("canonical parent checkpoint schedule mismatch")
    optimizer_contract_record = validate_optimizer_contract_for_continuation(payload)
    _validate_resolved_optimizer_semantics(payload["resolved_config"], optimizer_contract_record)
    provenance = payload["provenance"]
    if provenance.get("schema") != FULL_SEQUENCE_TRAINER_SCHEMA:
        raise RuntimeError("parent provenance schema mismatch")
    if provenance.get("run_kind") != FULL_SEQUENCE_RUN_KIND:
        raise RuntimeError("parent provenance run_kind mismatch")
    if provenance.get("objective_version") != FULL_SEQUENCE_OBJECTIVE_VERSION:
        raise RuntimeError("parent provenance objective_version mismatch")
    if provenance.get("paper_exact_reproduction") is not False:
        raise RuntimeError("parent provenance paper_exact_reproduction mismatch")
    _validate_provenance_semantics(provenance)
    if int(expected_parent_global_step) in CONTINUATION_TARGET_STEPS:
        validate_continuation_lineage(
            payload,
            expected_parent_step=5000 if int(expected_parent_global_step) == 6500 else 6500,
            expected_target_step=int(expected_parent_global_step),
        )


def validate_continuation_lineage(
    payload: Mapping[str, Any],
    *,
    expected_target_step: int,
    expected_parent_step: int | None = None,
) -> None:
    target_step = int(expected_target_step)
    if expected_parent_step is None:
        if target_step == 6500:
            expected_parent_step = 5000
        elif target_step == 8000:
            expected_parent_step = 6500
        else:
            raise RuntimeError("continuation target step must be 6500 or 8000")
    expected_parent_step = int(expected_parent_step)
    validate_continuation_stage_pair(expected_parent_step, target_step)
    resolved = payload.get("resolved_config")
    provenance = payload.get("provenance")
    if not isinstance(resolved, Mapping) or not isinstance(provenance, Mapping):
        raise RuntimeError("continuation lineage requires resolved_config/provenance")
    if resolved.get("continuation_schema") != CONTINUATION_SCHEMA:
        raise RuntimeError("continuation resolved_config schema missing")
    if int(resolved.get("continuation_parent_global_step", -1)) != expected_parent_step:
        raise RuntimeError("continuation resolved_config parent mismatch")
    if int(resolved.get("continuation_target_global_step", -1)) != target_step:
        raise RuntimeError("continuation resolved_config target mismatch")
    if list(resolved.get("continuation_checkpoint_steps", ())) != [target_step]:
        raise RuntimeError("continuation resolved_config checkpoint schedule mismatch")
    if list(resolved.get("continuation_validation_steps", ())) != [target_step]:
        raise RuntimeError("continuation resolved_config validation schedule mismatch")
    if resolved.get("base_training_git_sha") != BASE_TRAINING_GIT_SHA:
        raise RuntimeError("continuation base training git mismatch")
    continuation = provenance.get("continuation")
    if not isinstance(continuation, Mapping):
        raise RuntimeError("continuation provenance missing")
    if continuation.get("schema") != CONTINUATION_SCHEMA:
        raise RuntimeError("continuation provenance schema mismatch")
    if int(continuation.get("parent_global_step", -1)) != expected_parent_step:
        raise RuntimeError("continuation provenance parent mismatch")
    if int(continuation.get("target_global_step", -1)) != target_step:
        raise RuntimeError("continuation provenance target mismatch")
    if continuation.get("base_training_git_sha") != BASE_TRAINING_GIT_SHA:
        raise RuntimeError("continuation provenance base training git mismatch")
    payload_git = str(payload.get("git_sha"))
    if continuation.get("runtime_git_sha") != payload_git:
        raise RuntimeError("continuation provenance runtime git mismatch")
    if resolved.get("current_git_sha") != payload_git:
        raise RuntimeError("continuation resolved_config current git mismatch")
    if resolved.get("expected_git_sha") != payload_git:
        raise RuntimeError("continuation resolved_config expected git mismatch")
    if resolved.get("base_training_git_sha") != continuation.get("base_training_git_sha"):
        raise RuntimeError("continuation resolved/provenance base git mismatch")
    if resolved.get("parent_checkpoint_git_sha") != continuation.get("parent_checkpoint_git_sha"):
        raise RuntimeError("continuation parent checkpoint git mismatch")
    if resolved.get("parent_checkpoint_sha256") != continuation.get("parent_checkpoint_sha256"):
        raise RuntimeError("continuation parent checkpoint SHA mismatch")
    flags = {
        "objective_changed": False,
        "architecture_changed": False,
        "optimizer_hparams_changed": False,
        "data_contract_changed": False,
        "rng_semantics_changed": False,
        "freeze_policy_changed": False,
    }
    if continuation.get("continuation_only_training_horizon_extension") is not True:
        raise RuntimeError("continuation horizon extension flag missing")
    for key, expected in flags.items():
        if continuation.get(key) is not expected:
            raise RuntimeError(f"continuation lineage flag mismatch: {key}")


def _validate_resolved_optimizer_semantics(
    resolved: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> None:
    groups_sequence = [
        group for group in contract.get("param_groups", ()) if isinstance(group, Mapping)
    ]
    names = tuple(str(group.get("name")) for group in groups_sequence)
    if names != CANONICAL_OPTIMIZER_GROUP_NAMES:
        raise RuntimeError("semantic lock optimizer group names/order mismatch")
    if "mcp" in names:
        raise RuntimeError("semantic lock optimizer group mcp is not allowed")
    expected = {
        "backbone": float(resolved["backbone_lr"]),
        "patch_embedding": float(resolved["patch_embedding_lr"]),
        "mcp_fusion": float(resolved["mcp_lr"]),
        "mcp_depth1": float(resolved["mcp_lr"]),
        "mcp_depth2": float(resolved["mcp_lr"]),
        "mcp_depth3": float(resolved["mcp_lr"]),
    }
    for group in groups_sequence:
        name = str(group["name"])
        expected_lr = expected[name]
        if float(group.get("lr")) != expected_lr:
            raise RuntimeError(f"semantic lock optimizer LR mismatch: {name}")
        if float(group.get("weight_decay")) != float(resolved["weight_decay"]):
            raise RuntimeError(f"semantic lock optimizer weight_decay mismatch: {name}")


def _validate_sidecar_payload_cross_checks(
    validation: Mapping[str, Any],
    *,
    payload: Mapping[str, Any],
    path: Path,
) -> None:
    if str(validation.get("path")) != str(path.resolve()):
        raise RuntimeError("parent checkpoint validation path mismatch")
    if int(validation.get("generator_key_count", -1)) != len(payload["generator"]):
        raise RuntimeError("parent checkpoint validation generator_key_count mismatch")
    optimizer = payload.get("optimizer")
    if not isinstance(optimizer, Mapping) or not isinstance(optimizer.get("state"), Mapping):
        raise RuntimeError("parent checkpoint optimizer state missing")
    if int(validation.get("optimizer_state_entry_count", -1)) != len(optimizer["state"]):
        raise RuntimeError(
            "parent checkpoint validation optimizer_state_entry_count mismatch"
        )


def _validate_provenance_semantics(provenance: Mapping[str, Any]) -> None:
    if provenance.get("future_embedding_order") != "depth_major":
        raise RuntimeError("parent provenance future_embedding_order mismatch")
    if provenance.get("rng_draw_order_version") != FULL_SEQUENCE_RNG_DRAW_ORDER_VERSION:
        raise RuntimeError("parent provenance rng_draw_order_version mismatch")
    objective = provenance.get("objective")
    if not isinstance(objective, Mapping):
        raise RuntimeError("parent provenance objective missing")
    required_bools = {
        "joint_backbone": True,
        "shared_patch_embedding": True,
        "self_rollout": False,
        "dmd": False,
        "generated_history": False,
        "noisy_history_augmentation": False,
    }
    for key, expected in required_bools.items():
        if objective.get(key) is not expected:
            raise RuntimeError(f"parent provenance objective {key} mismatch")


def _require_sidecar_names(path: Path) -> None:
    sidecars = checkpoint_sidecar_paths(path)
    stem = path.with_suffix("")
    if sidecars["sha256"] != stem.with_suffix(".sha256.txt"):
        raise RuntimeError("checkpoint SHA sidecar name mismatch")
    if sidecars["validation"] != stem.with_suffix(".validation.json"):
        raise RuntimeError("checkpoint validation sidecar name mismatch")


def _count_mcp_tensors(state_dict: Mapping[str, Any]) -> int:
    return sum(
        1
        for key, value in state_dict.items()
        if is_mcp_state_key(str(key)) and torch.is_tensor(value)
    )


def _tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tuple(value.shape)).encode("utf-8"))
    digest.update(str(value.dtype).encode("utf-8"))
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _json_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
