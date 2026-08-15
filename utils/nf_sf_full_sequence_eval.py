from __future__ import annotations

import hashlib
import json
import math
import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch

from utils.checkpoint import extract_generator_state_dict, is_mcp_state_key
from utils.nf_sf_m3 import (
    M3_REFERENCE_CHECKPOINT_SHA256,
    atomic_json_write,
    atomic_torch_save,
    file_sha256,
    tensor_sha256,
    tensor_summary,
)
from utils.nf_sf_tensors import (
    DEFAULT_NUM_TRAIN_TIMESTEPS,
    DEFAULT_S_MAIN,
    DEFAULT_S_MCP,
    flow_match_shift_timesteps,
)
from utils.nf_sf_training import (
    FULL_SEQUENCE_OBJECTIVE_VERSION,
    FULL_SEQUENCE_RUN_KIND,
    FULL_SEQUENCE_TRAINER_SCHEMA,
)

EVAL_SCHEMA = "nf_sf_full_sequence_deployment_eval_v1"
EVAL_COMMON_INPUTS_SCHEMA = "nf_sf_full_sequence_deployment_common_inputs_v1"
EVAL_MODE_OUTPUT_SCHEMA = "nf_sf_full_sequence_deployment_mode_output_v1"
EVAL_COMPARISON_SCHEMA = "nf_sf_full_sequence_deployment_comparison_v1"
EVAL_RNG_PLAN_SCHEMA = "nf_sf_full_sequence_deployment_rng_plan_v1"
CHECKPOINT_VALIDATION_SCHEMA = "nf_sf_full_sequence_checkpoint_validation_v1"
TRAINING_CHECKPOINT_GIT_SHA = "2ab9b3a7c08b09140b6cbae23df21107817fe3be"
EXPECTED_CANONICAL_GIT_SHA = TRAINING_CHECKPOINT_GIT_SHA
OFFICIAL_SELF_FORCING_CHECKPOINT_SHA256 = M3_REFERENCE_CHECKPOINT_SHA256
FULL_SEQUENCE_OBJECTIVE_MODE = "next_forcing_full"
FULL_SEQUENCE_GLOBAL_STEP = 5000
FULL_SEQUENCE_FRAME_COUNT = 21
FULL_SEQUENCE_CHUNK_FRAMES = 3
FULL_SEQUENCE_NUM_CHUNKS = 7
FULL_SEQUENCE_FRAME_SEQ_LENGTH = 1560
FULL_SEQUENCE_TAP_LAYERS = (3, 11, 19, 29)
FULL_SEQUENCE_MCP_MODULES = 3
FULL_SEQUENCE_MCP_LAYERS = 3
RAW_DEPLOYMENT_SCHEDULE = (1000.0, 750.0, 500.0, 250.0)
MAIN_DEPLOYMENT_SCHEDULE = (
    1000.0,
    937.5,
    833.3333129882812,
    625.0,
)
MCP_DEPLOYMENT_SCHEDULE = (
    1000.0,
    967.7418823242188,
    909.0909423828125,
    769.2307739257812,
)
MODE_OFFICIAL_MAIN = "official_main"
MODE_TRAINED_MAIN = "trained_main"
MODE_TRAINED_MCP1 = "trained_mcp1"
EvalMode = Literal["official_main", "trained_main", "trained_mcp1"]
REPO_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_MEASUREMENT_STATUS = "SANITY_ONLY_NOT_BENCHMARK"


@dataclass(frozen=True)
class DeploymentSchedule:
    raw_schedule: tuple[float, ...]
    main_warped_schedule: tuple[float, ...]
    mcp_warped_schedule: tuple[float, ...]
    main_shift: float = DEFAULT_S_MAIN
    mcp_shift: float = DEFAULT_S_MCP
    num_train_timesteps: int = DEFAULT_NUM_TRAIN_TIMESTEPS

    def to_json(self, *, include_mcp: bool) -> dict[str, Any]:
        return {
            "raw_schedule": list(self.raw_schedule),
            "main_warped_schedule": list(self.main_warped_schedule),
            "mcp_warped_schedule": (
                list(self.mcp_warped_schedule) if include_mcp else None
            ),
            "main_shift": float(self.main_shift),
            "mcp_shift": float(self.mcp_shift) if include_mcp else None,
            "num_train_timesteps": int(self.num_train_timesteps),
            "warp_formula": "utils.nf_sf_tensors.flow_match_shift_timesteps",
            "raw_index_alignment": bool(include_mcp),
        }


@dataclass(frozen=True)
class DeploymentCheckpointRecord:
    path: str
    sha256: str
    checkpoint_type: str
    load_mode: str
    generator_state_dict: Mapping[str, Any]
    global_step: int | None = None
    training_git_sha: str | None = None
    payload: Mapping[str, Any] | None = None
    validation_sidecar: Mapping[str, Any] | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "type": self.checkpoint_type,
            "load_mode": self.load_mode,
            "global_step": self.global_step,
            "training_checkpoint_git_sha": self.training_git_sha,
            "generator_key_count": len(self.generator_state_dict),
            "mcp_tensor_count": count_mcp_tensors(self.generator_state_dict),
        }


@dataclass(frozen=True)
class DeploymentRuntime:
    generator: Any
    scheduler: Any
    kv_cache: list[dict[str, Any]]
    crossattn_cache: list[dict[str, Any]]
    frame_seq_length: int
    num_frame_per_block: int = FULL_SEQUENCE_CHUNK_FRAMES
    context_noise: int = 0


@dataclass(frozen=True)
class DeploymentResult:
    latent: torch.Tensor
    trace: dict[str, Any]
    summary: dict[str, Any]


class KVSnapshot:
    def __init__(self, layers: list[dict[str, Any]]) -> None:
        self._layers = layers

    @classmethod
    def capture(cls, kv_cache: Sequence[Mapping[str, Any]]) -> KVSnapshot:
        layers: list[dict[str, Any]] = []
        for layer in kv_cache:
            local_end = _cache_index_value(layer, "local_end_index")
            state = {
                "global_end_index": _cache_index_value(layer, "global_end_index"),
                "local_end_index": local_end,
            }
            for name in ("k", "v"):
                tensor = layer.get(name)
                if torch.is_tensor(tensor):
                    state[name] = tensor[:, :local_end].detach().clone()
            layers.append(state)
        return cls(layers)

    def restore(self, kv_cache: Sequence[Mapping[str, Any]]) -> bool:
        if len(kv_cache) != len(self._layers):
            raise RuntimeError("KV snapshot layer count differs from cache")
        restored = True
        for layer, state in zip(kv_cache, self._layers):
            local_end = int(state["local_end_index"])
            for name in ("k", "v"):
                saved = state.get(name)
                current = layer.get(name)
                if torch.is_tensor(saved) and torch.is_tensor(current):
                    current[:, :local_end].copy_(saved.to(device=current.device))
                    restored = restored and bool(
                        torch.equal(
                            current[:, :local_end],
                            saved.to(device=current.device),
                        )
                    )
            _set_cache_index(layer, "global_end_index", int(state["global_end_index"]))
            _set_cache_index(layer, "local_end_index", local_end)
        return restored


def current_git_head() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    value = result.stdout.strip()
    if len(value) != 40:
        raise RuntimeError("git rev-parse HEAD did not return a 40-char SHA")
    return value


def git_top_level() -> Path:
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return Path(result.stdout.strip()).resolve()


def git_changed_paths(*, cached: bool) -> tuple[str, ...]:
    command = ["git", "diff", "--name-only"]
    if cached:
        command.insert(2, "--cached")
    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return tuple(line.strip() for line in result.stdout.splitlines() if line.strip())


def repo_preflight_facts(
    *,
    expected_runtime_git_sha: str,
    output_dir: Path | str,
    repo_root: Path | str = REPO_ROOT,
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    out = Path(output_dir).resolve()
    return {
        "repo_root": str(root),
        "git_top_level": str(git_top_level()),
        "current_runtime_git_sha": current_git_head(),
        "expected_runtime_git_sha": str(expected_runtime_git_sha),
        "tracked_worktree_dirty_paths": list(git_changed_paths(cached=False)),
        "staged_index_dirty_paths": list(git_changed_paths(cached=True)),
        "output_dir": str(out),
        "output_dir_inside_repo": out.is_relative_to(root),
    }


def validate_repo_preflight_facts(facts: Mapping[str, Any]) -> dict[str, Any]:
    repo_root = Path(str(facts["repo_root"])).resolve()
    git_root = Path(str(facts["git_top_level"])).resolve()
    if git_root != repo_root:
        raise RuntimeError("git top-level does not match evaluator repo root")
    if str(facts["current_runtime_git_sha"]) != str(facts["expected_runtime_git_sha"]):
        raise RuntimeError("current runtime git SHA differs from expected_runtime_git_sha")
    tracked_dirty = tuple(str(path) for path in facts.get("tracked_worktree_dirty_paths", ()))
    if tracked_dirty:
        raise RuntimeError(f"tracked worktree is dirty: {tracked_dirty}")
    staged_dirty = tuple(str(path) for path in facts.get("staged_index_dirty_paths", ()))
    if staged_dirty:
        raise RuntimeError(f"staged index is dirty: {staged_dirty}")
    if bool(facts.get("output_dir_inside_repo")):
        raise RuntimeError("deployment eval output_dir must be outside the repo")
    return {
        "status": "PASS",
        "repo_root": str(repo_root),
        "git_top_level": str(git_root),
        "runtime_git_sha": str(facts["current_runtime_git_sha"]),
        "output_dir": str(facts["output_dir"]),
    }


def validate_repo_preflight(
    *,
    expected_runtime_git_sha: str,
    output_dir: Path | str,
    repo_root: Path | str = REPO_ROOT,
) -> dict[str, Any]:
    return validate_repo_preflight_facts(
        repo_preflight_facts(
            expected_runtime_git_sha=expected_runtime_git_sha,
            output_dir=output_dir,
            repo_root=repo_root,
        )
    )


def canonical_json_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            _json_safe(payload),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()


def resolve_deployment_schedule() -> DeploymentSchedule:
    raw = torch.tensor(RAW_DEPLOYMENT_SCHEDULE, dtype=torch.float32)
    main = tuple(
        float(value)
        for value in flow_match_shift_timesteps(
            raw,
            shift=DEFAULT_S_MAIN,
            num_train_timesteps=DEFAULT_NUM_TRAIN_TIMESTEPS,
        ).tolist()
    )
    mcp = tuple(
        float(value)
        for value in flow_match_shift_timesteps(
            raw,
            shift=DEFAULT_S_MCP,
            num_train_timesteps=DEFAULT_NUM_TRAIN_TIMESTEPS,
        ).tolist()
    )
    _require_schedule_close(main, MAIN_DEPLOYMENT_SCHEDULE, "main")
    _require_schedule_close(mcp, MCP_DEPLOYMENT_SCHEDULE, "mcp")
    return DeploymentSchedule(
        raw_schedule=RAW_DEPLOYMENT_SCHEDULE,
        main_warped_schedule=main,
        mcp_warped_schedule=mcp,
    )


def full_sequence_role_map() -> dict[str, list[int]]:
    return {
        "bootstrap": [0],
        "main_current": [1, 3, 5],
        "mcp_next": [2, 4, 6],
    }


def build_mcp1_execution_plan() -> list[dict[str, Any]]:
    return [
        {
            "phase": "bootstrap",
            "chunk_indices": [0],
            "main_chunk_index": 0,
            "next_chunk_index": None,
            "cursor_before": 0,
            "cursor_after": 1,
            "commit_order": [0],
        },
        {
            "phase": "paired_round",
            "round_index": 0,
            "chunk_indices": [1, 2],
            "main_chunk_index": 1,
            "next_chunk_index": 2,
            "cursor_before": 1,
            "cursor_after": 3,
            "commit_order": [1, 2],
            "clean_recache_order": [1, 2],
        },
        {
            "phase": "paired_round",
            "round_index": 1,
            "chunk_indices": [3, 4],
            "main_chunk_index": 3,
            "next_chunk_index": 4,
            "cursor_before": 3,
            "cursor_after": 5,
            "commit_order": [3, 4],
            "clean_recache_order": [3, 4],
        },
        {
            "phase": "paired_round",
            "round_index": 2,
            "chunk_indices": [5, 6],
            "main_chunk_index": 5,
            "next_chunk_index": 6,
            "cursor_before": 5,
            "cursor_after": 7,
            "commit_order": [5, 6],
            "clean_recache_order": [5, 6],
        },
    ]


def checkpoint_sidecar_paths(path: Path | str) -> dict[str, Path]:
    checkpoint_path = Path(path)
    stem = checkpoint_path.with_suffix("")
    return {
        "sha256": stem.with_suffix(".sha256.txt"),
        "validation": stem.with_suffix(".validation.json"),
    }


def load_official_checkpoint_record(
    path: Path | str,
    *,
    expected_sha256: str = OFFICIAL_SELF_FORCING_CHECKPOINT_SHA256,
) -> DeploymentCheckpointRecord:
    path = Path(path)
    actual_sha = file_sha256(path)
    if actual_sha != str(expected_sha256):
        raise RuntimeError("official Self-Forcing checkpoint SHA256 mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = extract_generator_state_dict(payload)
    if any(is_mcp_state_key(str(key)) for key in state_dict.keys()):
        raise RuntimeError("official Self-Forcing checkpoint must not contain MCP keys")
    return DeploymentCheckpointRecord(
        path=str(path.resolve()),
        sha256=actual_sha,
        checkpoint_type="official_self_forcing",
        load_mode="OFFICIAL_BACKBONE_STRICT_NO_MCP",
        generator_state_dict=state_dict,
        global_step=None,
        payload=payload if isinstance(payload, Mapping) else None,
        validation_sidecar=None,
    )


def load_full_sequence_checkpoint_record(
    path: Path | str,
    *,
    expected_training_git_sha: str = TRAINING_CHECKPOINT_GIT_SHA,
    expected_official_sha256: str = OFFICIAL_SELF_FORCING_CHECKPOINT_SHA256,
) -> DeploymentCheckpointRecord:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"full-sequence checkpoint not found: {path}")
    actual_sha = file_sha256(path)
    validation = validate_full_sequence_checkpoint_sidecars(
        path,
        expected_sha256=actual_sha,
    )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    validate_full_sequence_checkpoint_payload(
        payload,
        checkpoint_sha256=actual_sha,
        expected_training_git_sha=expected_training_git_sha,
        expected_official_sha256=expected_official_sha256,
    )
    return DeploymentCheckpointRecord(
        path=str(path.resolve()),
        sha256=actual_sha,
        checkpoint_type="full_sequence_step5000",
        load_mode="FULL_SEQUENCE_GENERATOR_STRICT_WITH_MCP",
        generator_state_dict=payload["generator"],
        global_step=int(payload["global_step"]),
        training_git_sha=str(payload["git_sha"]),
        payload=payload,
        validation_sidecar=validation,
    )


def validate_full_sequence_checkpoint_sidecars(
    path: Path | str,
    *,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    path = Path(path)
    sidecars = checkpoint_sidecar_paths(path)
    if not sidecars["sha256"].is_file():
        raise RuntimeError("full-sequence checkpoint SHA256 sidecar is missing")
    if not sidecars["validation"].is_file():
        raise RuntimeError("full-sequence checkpoint validation sidecar is missing")
    actual_sha = file_sha256(path)
    if expected_sha256 is not None and actual_sha != str(expected_sha256):
        raise RuntimeError("full-sequence checkpoint SHA256 mismatch")
    sha_tokens = sidecars["sha256"].read_text(encoding="utf-8").strip().split()
    if not sha_tokens or sha_tokens[0] != actual_sha:
        raise RuntimeError("full-sequence checkpoint SHA256 sidecar mismatch")
    validation = json.loads(sidecars["validation"].read_text(encoding="utf-8"))
    if validation.get("status") != "PASS":
        raise RuntimeError("full-sequence checkpoint validation sidecar is not PASS")
    if validation.get("schema") != CHECKPOINT_VALIDATION_SCHEMA:
        raise RuntimeError("full-sequence checkpoint validation schema mismatch")
    if validation.get("sha256") != actual_sha:
        raise RuntimeError("full-sequence checkpoint validation SHA mismatch")
    if int(validation.get("size_bytes", -1)) != int(path.stat().st_size):
        raise RuntimeError("full-sequence checkpoint validation size mismatch")
    if validation.get("run_kind") != FULL_SEQUENCE_RUN_KIND:
        raise RuntimeError("full-sequence checkpoint validation run_kind mismatch")
    if validation.get("objective_version") != FULL_SEQUENCE_OBJECTIVE_VERSION:
        raise RuntimeError("full-sequence checkpoint validation objective mismatch")
    if validation.get("objective_mode") != FULL_SEQUENCE_OBJECTIVE_MODE:
        raise RuntimeError("full-sequence checkpoint validation objective_mode mismatch")
    if int(validation.get("global_step", -1)) != FULL_SEQUENCE_GLOBAL_STEP:
        raise RuntimeError("full-sequence checkpoint validation global_step mismatch")
    return validation


def validate_full_sequence_checkpoint_payload(
    payload: Any,
    *,
    checkpoint_sha256: str,
    expected_training_git_sha: str,
    expected_official_sha256: str,
) -> None:
    if not isinstance(payload, Mapping):
        raise TypeError("full-sequence checkpoint payload must be a mapping")
    required = {
        "schema",
        "run_kind",
        "objective_version",
        "objective_mode",
        "status",
        "global_step",
        "git_sha",
        "generator",
        "reference_checkpoint",
        "resolved_config",
        "provenance",
    }
    missing = required - set(payload.keys())
    if missing:
        raise RuntimeError(
            f"full-sequence checkpoint missing fields: {sorted(missing)}"
        )
    if payload["schema"] != FULL_SEQUENCE_TRAINER_SCHEMA:
        raise RuntimeError("full-sequence checkpoint schema mismatch")
    if payload["run_kind"] != FULL_SEQUENCE_RUN_KIND:
        raise RuntimeError("full-sequence checkpoint run_kind mismatch")
    if payload["objective_version"] != FULL_SEQUENCE_OBJECTIVE_VERSION:
        raise RuntimeError("full-sequence checkpoint objective_version mismatch")
    if payload["objective_mode"] != FULL_SEQUENCE_OBJECTIVE_MODE:
        raise RuntimeError("full-sequence checkpoint objective_mode mismatch")
    if payload["status"] != "PRODUCTION":
        raise RuntimeError("full-sequence checkpoint status must be PRODUCTION")
    if int(payload["global_step"]) != FULL_SEQUENCE_GLOBAL_STEP:
        raise RuntimeError("full-sequence checkpoint global_step must be 5000")
    if str(payload["git_sha"]) != str(expected_training_git_sha):
        raise RuntimeError("full-sequence checkpoint training git_sha mismatch")
    reference = payload["reference_checkpoint"]
    if not isinstance(reference, Mapping):
        raise TypeError("full-sequence checkpoint reference_checkpoint must be mapping")
    if reference.get("sha256") != str(expected_official_sha256):
        raise RuntimeError("full-sequence checkpoint official parent SHA mismatch")
    state_dict = payload["generator"]
    if not isinstance(state_dict, Mapping):
        raise TypeError("full-sequence checkpoint generator must be a state dict")
    if count_mcp_tensors(state_dict) <= 0:
        raise RuntimeError("full-sequence checkpoint generator missing MCP tensors")
    for key in ("sample_plan_sha256", "manifest_sha256"):
        value = payload.get(key)
        if not isinstance(value, str) or len(value) != 64:
            raise RuntimeError(f"full-sequence checkpoint {key} missing or invalid")
    provenance = payload["provenance"]
    if not isinstance(provenance, Mapping):
        raise TypeError("full-sequence checkpoint provenance must be mapping")
    if provenance.get("paper_exact_reproduction") is not False:
        raise RuntimeError("full-sequence checkpoint must record paper_exact_reproduction=false")
    resolved = payload["resolved_config"]
    if not isinstance(resolved, Mapping):
        raise TypeError("full-sequence checkpoint resolved_config must be mapping")
    if int(resolved.get("num_frame_per_block", -1)) != FULL_SEQUENCE_CHUNK_FRAMES:
        raise RuntimeError("full-sequence checkpoint resolved_config nfpb mismatch")
    if resolved.get("gradient_checkpointing") is not True:
        raise RuntimeError("full-sequence checkpoint resolved_config gradient checkpointing mismatch")
    if not isinstance(checkpoint_sha256, str) or len(checkpoint_sha256) != 64:
        raise RuntimeError("checkpoint_sha256 must be a SHA256 hex string")


def count_mcp_tensors(state_dict: Mapping[str, Any]) -> int:
    return sum(
        1
        for key, value in state_dict.items()
        if is_mcp_state_key(str(key)) and torch.is_tensor(value)
    )


def sample_plan_validation_identities(sample_plan: Mapping[str, Any]) -> tuple[str, ...]:
    if "validation_sample_identities" in sample_plan:
        return tuple(str(value) for value in sample_plan["validation_sample_identities"])
    samples = sample_plan.get("samples")
    if isinstance(samples, Mapping):
        validation = samples.get("validation")
        if isinstance(validation, Sequence) and not isinstance(
            validation,
            (str, bytes, bytearray),
        ):
            return tuple(str(entry["identity"]) for entry in validation)
    raise RuntimeError("sample plan validation identities are missing")


def selected_validation_position(
    sample_plan: Mapping[str, Any],
    selected_identity: str,
) -> int:
    identities = sample_plan_validation_identities(sample_plan)
    try:
        return identities.index(str(selected_identity))
    except ValueError as exc:
        raise RuntimeError(
            "deployment evaluator selected identity must be from validation split"
        ) from exc


def validate_eval_artifact_identity(
    *,
    sample_plan: Mapping[str, Any],
    teacher_manifest_sha256: str,
    checkpoint_payload: Mapping[str, Any],
    selected_identity: str,
) -> dict[str, Any]:
    sample_plan_sha = str(sample_plan.get("sample_plan_sha256", ""))
    if sample_plan_sha != str(checkpoint_payload.get("sample_plan_sha256")):
        raise RuntimeError("current sample_plan SHA differs from training checkpoint")
    if str(teacher_manifest_sha256) != str(checkpoint_payload.get("manifest_sha256")):
        raise RuntimeError("current teacher manifest SHA differs from training checkpoint")
    validation_position = selected_validation_position(sample_plan, selected_identity)
    return {
        "status": "PASS",
        "sample_plan_sha256": sample_plan_sha,
        "teacher_manifest_sha256": str(teacher_manifest_sha256),
        "selected_identity": str(selected_identity),
        "selected_validation_position": int(validation_position),
        "default_fixed_decode_validation_identity": str(
            sample_plan.get("fixed_decode_validation_identity")
        ),
    }


def build_common_inputs_record(
    *,
    sample_identity: str,
    teacher_metadata: Mapping[str, Any],
    teacher_payload: Mapping[str, Any],
    source_noise: torch.Tensor,
    conditioning: Mapping[str, Any],
    runtime_git_sha: str,
    training_checkpoint_git_sha: str,
    fps: int,
    sample_plan_sha256: str | None = None,
    teacher_manifest_sha256: str | None = None,
    selected_validation_position: int | None = None,
) -> tuple[dict[str, Any], str]:
    _validate_source_noise(source_noise, teacher_payload=teacher_payload)
    schedule = resolve_deployment_schedule()
    conditioning_summary = conditioning_json_summary(conditioning)
    prompt_sha = str(teacher_payload["prompt_sha256"])
    source_summary = tensor_json_summary(source_noise)
    record = {
        "schema": EVAL_COMMON_INPUTS_SCHEMA,
        "sample_identity": str(sample_identity),
        "teacher_identity": teacher_identity_json(teacher_metadata),
        "prompt": {
            "text": str(teacher_payload["prompt"]),
            "prompt_sha256": prompt_sha,
        },
        "source_noise": source_summary,
        "source_noise_sha256": source_summary["sha256"],
        "conditioning": conditioning_summary,
        "conditioning_sha256": conditioning_summary["sha256"],
        "noise_seed": int(teacher_payload["noise_seed"]),
        "rollout_seed": int(teacher_payload["rollout_seed"]),
        "raw_teacher_schedule": [
            float(value) for value in teacher_payload.get("raw_denoising_steps", ())
        ],
        "warped_teacher_schedule": [
            float(value) for value in teacher_payload.get("warped_denoising_steps", ())
        ],
        "deployment_schedule": schedule.to_json(include_mcp=True),
        "latent_frames": FULL_SEQUENCE_FRAME_COUNT,
        "num_chunks": FULL_SEQUENCE_NUM_CHUNKS,
        "chunk_frames": FULL_SEQUENCE_CHUNK_FRAMES,
        "fps": int(fps),
        "runtime_git_sha": str(runtime_git_sha),
        "training_checkpoint_git_sha": str(training_checkpoint_git_sha),
        "sample_plan_sha256": sample_plan_sha256,
        "teacher_manifest_sha256": teacher_manifest_sha256,
        "selected_validation_position": selected_validation_position,
    }
    _require_teacher_schedule(record)
    fingerprint = canonical_json_sha256(record)
    return record, fingerprint


def assert_common_input_fingerprints(mode_records: Mapping[str, Mapping[str, Any]]) -> str:
    fingerprints: dict[str, str] = {}
    for mode, record in mode_records.items():
        value = str(record.get("common_inputs_fingerprint_sha256", ""))
        if not value:
            raise RuntimeError(f"{mode} common input fingerprint missing")
        fingerprints[str(mode)] = value
    unique = set(fingerprints.values())
    if len(unique) != 1:
        raise RuntimeError(f"common input fingerprint differs across modes: {fingerprints}")
    return next(iter(unique))


def rng_plan_fingerprint(rng_trace: Mapping[str, Any]) -> str:
    compatibility = rng_trace.get("compatibility_draw")
    if not isinstance(compatibility, Mapping):
        raise RuntimeError("RNG trace missing compatibility draw")
    draws = rng_trace.get("draws")
    if not isinstance(draws, Sequence) or isinstance(draws, (str, bytes, bytearray)):
        raise RuntimeError("RNG trace draws missing")
    payload = {
        "schema": EVAL_RNG_PLAN_SCHEMA,
        "rollout_seed": int(rng_trace["rollout_seed"]),
        "compatibility_draw": {
            "draw_order": int(compatibility["draw_order"]),
            "purpose": str(compatibility["purpose"]),
            "operation": str(compatibility["operation"]),
            "low": int(compatibility["low"]),
            "high": int(compatibility["high"]),
            "size": list(compatibility["size"]),
            "dtype": str(compatibility["dtype"]),
            "values_sha256": str(compatibility["values_sha256"]),
            "values_discarded": bool(compatibility["values_discarded"]),
        },
        "draws": [
            {
                "draw_order": int(record["draw_order"]),
                "purpose": str(record["purpose"]),
                "chunk_index": int(record["chunk_index"]),
                "absolute_chunk_index": int(record["absolute_chunk_index"]),
                "solver_step_index": record["solver_step_index"],
                "noise_shape": list(record["noise"]["shape"]),
                "noise_dtype": str(record["noise"]["dtype"]),
                "noise_sha256": str(record["noise"]["sha256"]),
            }
            for record in draws
        ],
    }
    return canonical_json_sha256(payload)


def assert_rng_plan_fingerprints(mode_records: Mapping[str, Mapping[str, Any]]) -> str:
    fingerprints: dict[str, str] = {}
    for mode, record in mode_records.items():
        value = str(record.get("rng_plan_fingerprint_sha256", ""))
        if not value:
            raise RuntimeError(f"{mode} RNG plan fingerprint missing")
        fingerprints[str(mode)] = value
    unique = set(fingerprints.values())
    if len(unique) != 1:
        raise RuntimeError(f"RNG plan fingerprint differs across modes: {fingerprints}")
    return next(iter(unique))


def assert_mode_inputs_match_common(
    mode_records: Mapping[str, Mapping[str, Any]],
    common_inputs: Mapping[str, Any],
) -> None:
    source_sha = str(common_inputs.get("source_noise_sha256"))
    conditioning_sha = str(common_inputs.get("conditioning_sha256"))
    for mode, record in mode_records.items():
        if str(record.get("source_noise_sha256")) != source_sha:
            raise RuntimeError(f"{mode} source_noise SHA differs from common inputs")
        if str(record.get("conditioning_sha256")) != conditioning_sha:
            raise RuntimeError(f"{mode} conditioning SHA differs from common inputs")


def build_absolute_chunk_rng_plan(
    *,
    source_noise: torch.Tensor,
    rollout_seed: int,
    num_denoising_steps: int = len(RAW_DEPLOYMENT_SCHEDULE),
    chunk_frames: int = FULL_SEQUENCE_CHUNK_FRAMES,
) -> dict[str, Any]:
    if source_noise.ndim != 5:
        raise ValueError("source_noise must have layout [B, F, C, H, W]")
    if int(source_noise.shape[1]) % int(chunk_frames) != 0:
        raise ValueError("source_noise frame count must be chunk-aligned")
    if int(num_denoising_steps) <= 1:
        raise ValueError("num_denoising_steps must be greater than one")
    device = source_noise.device
    num_chunks = int(source_noise.shape[1]) // int(chunk_frames)
    active_before = global_rng_state_hash(device)
    cuda_devices: list[int] = []
    if device.type == "cuda" and torch.cuda.is_available():
        cuda_devices = [
            torch.cuda.current_device() if device.index is None else int(device.index)
        ]
    transitions: dict[tuple[int, int], torch.Tensor] = {}
    contexts: dict[int, torch.Tensor] = {}
    draws: list[dict[str, Any]] = []
    with torch.random.fork_rng(devices=cuda_devices):
        reset_torch_rollout_rng(int(rollout_seed), device)
        post_reset_hash = global_rng_state_hash(device)
        compatibility = consume_teacher_compatibility_draw(
            num_chunks=num_chunks,
            num_denoising_steps=int(num_denoising_steps),
            device=device,
            draw_order=0,
        )
        draw_order = 1
        for chunk_index in range(num_chunks):
            start = int(chunk_index) * int(chunk_frames)
            template = source_noise[:, start:start + int(chunk_frames)].flatten(0, 1)
            for step_index in range(int(num_denoising_steps) - 1):
                noise, record = randn_like_with_trace(
                    template,
                    device=device,
                    purpose="transition_re_noise",
                    draw_order=draw_order,
                    chunk_index=chunk_index,
                    solver_step_index=step_index,
                )
                record["absolute_chunk_index"] = int(chunk_index)
                transitions[(int(chunk_index), int(step_index))] = noise.detach().clone()
                draws.append(record)
                draw_order += 1
            noise, record = randn_like_with_trace(
                template,
                device=device,
                purpose="context_clean_recache_noise",
                draw_order=draw_order,
                chunk_index=chunk_index,
                solver_step_index=None,
            )
            record["absolute_chunk_index"] = int(chunk_index)
            contexts[int(chunk_index)] = noise.detach().clone()
            draws.append(record)
            draw_order += 1
    active_after = global_rng_state_hash(device)
    if active_after != active_before:
        raise RuntimeError("deployment RNG plan generation changed active RNG state")
    trace = {
        "schema": EVAL_RNG_PLAN_SCHEMA,
        "rollout_seed": int(rollout_seed),
        "post_reset_global_rng_state_hash": post_reset_hash,
        "active_rng_unchanged": True,
        "compatibility_draw": compatibility,
        "draws": draws,
        "draw_count": len(draws),
    }
    trace["rng_plan_fingerprint_sha256"] = rng_plan_fingerprint(trace)
    return {
        "schema": EVAL_RNG_PLAN_SCHEMA,
        "rollout_seed": int(rollout_seed),
        "num_chunks": int(num_chunks),
        "chunk_frames": int(chunk_frames),
        "num_denoising_steps": int(num_denoising_steps),
        "source_noise_sha256": tensor_sha256(source_noise.detach().cpu()),
        "transition_noises": transitions,
        "context_noises": contexts,
        "trace": trace,
    }


def run_main_only_deployment(
    *,
    mode: EvalMode,
    runtime: DeploymentRuntime,
    source_noise: torch.Tensor,
    teacher_payload: Mapping[str, Any],
    teacher_metadata: Mapping[str, Any],
    conditional_dict: Mapping[str, Any],
    checkpoint: DeploymentCheckpointRecord,
    git_sha: str,
    common_inputs: Mapping[str, Any],
    common_inputs_fingerprint_sha256: str,
) -> DeploymentResult:
    if mode not in (MODE_OFFICIAL_MAIN, MODE_TRAINED_MAIN):
        raise ValueError("main-only deployment requires official_main or trained_main mode")
    elapsed_start = time.perf_counter()
    _validate_runtime(runtime)
    _validate_source_noise(source_noise, teacher_payload=teacher_payload)
    schedule = resolve_deployment_schedule()
    rng_plan = build_absolute_chunk_rng_plan(
        source_noise=source_noise,
        rollout_seed=int(teacher_payload["rollout_seed"]),
        num_denoising_steps=len(schedule.raw_schedule),
        chunk_frames=int(runtime.num_frame_per_block),
    )
    output = torch.empty_like(source_noise)
    chunks: list[dict[str, Any]] = []
    counts = {
        "main_solver_forward_count": 0,
        "mcp_call_count": 0,
        "clean_recache_forward_count": 0,
    }
    runtime.generator.eval()
    with torch.no_grad():
        for chunk_index in range(FULL_SEQUENCE_NUM_CHUNKS):
            chunk = _run_main_chunk(
                runtime=runtime,
                source_noise=source_noise,
                output=output,
                conditional_dict=conditional_dict,
                schedule=schedule,
                rng_plan=rng_plan,
                counts=counts,
                chunk_index=chunk_index,
                role="main",
                cursor_before=chunk_index,
                cursor_after=chunk_index + 1,
            )
            chunks.append(chunk)
    if _runtime_mcp_call_count(runtime.generator) != 0:
        raise RuntimeError(f"{mode} must not call MCP during rollout")
    _ensure_finite_tensor(output, name=f"{mode}.output_latent")
    return _deployment_result(
        mode=mode,
        output=output,
        checkpoint=checkpoint,
        schedule=schedule,
        common_inputs=common_inputs,
        common_inputs_fingerprint_sha256=common_inputs_fingerprint_sha256,
        actual_source_noise_sha256=tensor_sha256(source_noise.detach().cpu()),
        actual_conditioning_sha256=conditioning_json_summary(conditional_dict)["sha256"],
        git_sha=git_sha,
        chunks=chunks,
        parallel_rounds=[],
        execution_plan=[
            {
                "phase": "main_only",
                "chunk_indices": [index],
                "cursor_before": index,
                "cursor_after": index + 1,
            }
            for index in range(FULL_SEQUENCE_NUM_CHUNKS)
        ],
        counts=counts,
        rng_trace=rng_plan["trace"],
        generation_elapsed_ms=(time.perf_counter() - elapsed_start) * 1000.0,
    )


def run_mcp1_deployment(
    *,
    runtime: DeploymentRuntime,
    mcp_scheduler: Any,
    source_noise: torch.Tensor,
    teacher_payload: Mapping[str, Any],
    teacher_metadata: Mapping[str, Any],
    conditional_dict: Mapping[str, Any],
    checkpoint: DeploymentCheckpointRecord,
    git_sha: str,
    common_inputs: Mapping[str, Any],
    common_inputs_fingerprint_sha256: str,
) -> DeploymentResult:
    elapsed_start = time.perf_counter()
    _validate_runtime(runtime)
    _validate_source_noise(source_noise, teacher_payload=teacher_payload)
    if not hasattr(mcp_scheduler, "add_noise") or not hasattr(mcp_scheduler, "step"):
        raise TypeError("mcp_scheduler must provide add_noise and step")
    schedule = resolve_deployment_schedule()
    rng_plan = build_absolute_chunk_rng_plan(
        source_noise=source_noise,
        rollout_seed=int(teacher_payload["rollout_seed"]),
        num_denoising_steps=len(schedule.raw_schedule),
        chunk_frames=int(runtime.num_frame_per_block),
    )
    output = torch.empty_like(source_noise)
    chunks: list[dict[str, Any]] = []
    rounds: list[dict[str, Any]] = []
    counts = {
        "main_solver_forward_count": 0,
        "mcp_call_count": 0,
        "mcp_depth1_call_count": 0,
        "mcp_depth2_call_count": 0,
        "mcp_depth3_call_count": 0,
        "clean_recache_forward_count": 0,
        "returned_mcp_output_count": 0,
    }
    plan = build_mcp1_execution_plan()
    runtime.generator.eval()
    with torch.no_grad():
        for item in plan:
            phase = str(item["phase"])
            if phase == "bootstrap":
                chunks.append(
                    _run_main_chunk(
                        runtime=runtime,
                        source_noise=source_noise,
                        output=output,
                        conditional_dict=conditional_dict,
                        schedule=schedule,
                        rng_plan=rng_plan,
                        counts=counts,
                        chunk_index=int(item["main_chunk_index"]),
                        role="bootstrap",
                        cursor_before=int(item["cursor_before"]),
                        cursor_after=int(item["cursor_after"]),
                    )
                )
            elif phase == "paired_round":
                current, next_chunk, round_record = _run_mcp1_pair(
                    runtime=runtime,
                    mcp_scheduler=mcp_scheduler,
                    source_noise=source_noise,
                    output=output,
                    conditional_dict=conditional_dict,
                    schedule=schedule,
                    rng_plan=rng_plan,
                    counts=counts,
                    current_chunk_index=int(item["main_chunk_index"]),
                    next_chunk_index=int(item["next_chunk_index"]),
                    round_index=int(item["round_index"]),
                    cursor_before=int(item["cursor_before"]),
                    cursor_after=int(item["cursor_after"]),
                )
                chunks.extend([current, next_chunk])
                rounds.append(round_record)
            else:
                raise RuntimeError(f"unsupported deployment plan phase: {phase}")
    _ensure_finite_tensor(output, name="trained_mcp1.output_latent")
    return _deployment_result(
        mode=MODE_TRAINED_MCP1,
        output=output,
        checkpoint=checkpoint,
        schedule=schedule,
        common_inputs=common_inputs,
        common_inputs_fingerprint_sha256=common_inputs_fingerprint_sha256,
        actual_source_noise_sha256=tensor_sha256(source_noise.detach().cpu()),
        actual_conditioning_sha256=conditioning_json_summary(conditional_dict)["sha256"],
        git_sha=git_sha,
        chunks=chunks,
        parallel_rounds=rounds,
        execution_plan=plan,
        counts=counts,
        rng_trace=rng_plan["trace"],
        generation_elapsed_ms=(time.perf_counter() - elapsed_start) * 1000.0,
    )


def compare_latents(
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    chunk_frames: int = FULL_SEQUENCE_CHUNK_FRAMES,
) -> dict[str, Any]:
    if tuple(left.shape) != tuple(right.shape):
        raise RuntimeError("latent shape mismatch")
    diff = left.detach().float().cpu() - right.detach().float().cpu()
    per_chunk = []
    num_chunks = int(left.shape[1]) // int(chunk_frames)
    for index in range(num_chunks):
        start = index * int(chunk_frames)
        chunk_diff = diff[:, start:start + int(chunk_frames)]
        per_chunk.append(float(chunk_diff.square().mean().item()))
    return {
        "schema": EVAL_COMPARISON_SCHEMA,
        "kind": "latent",
        "shape": [int(dim) for dim in left.shape],
        "left_dtype": str(left.dtype),
        "right_dtype": str(right.dtype),
        "left_sha256": tensor_sha256(left.detach().cpu()),
        "right_sha256": tensor_sha256(right.detach().cpu()),
        "max_abs": float(diff.abs().max().item()),
        "mean_abs": float(diff.abs().mean().item()),
        "mse": float(diff.square().mean().item()),
        "per_chunk_mse": per_chunk,
    }


def compare_pixel_frames(
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    chunk_frames: int = FULL_SEQUENCE_CHUNK_FRAMES,
) -> dict[str, Any]:
    if tuple(left.shape) != tuple(right.shape):
        raise RuntimeError("pixel frame shape mismatch")
    diff = left.detach().float().cpu() - right.detach().float().cpu()
    mse = float(diff.square().mean().item())
    per_frame_mse = [
        float(diff[index].square().mean().item())
        for index in range(int(diff.shape[0]))
    ]
    psnr = math.inf if mse == 0.0 else 20.0 * math.log10(255.0) - 10.0 * math.log10(mse)
    mapping = latent_chunk_to_pixel_frame_ranges(
        decoded_frame_count=int(diff.shape[0]),
        latent_frame_count=FULL_SEQUENCE_FRAME_COUNT,
        chunk_frames=chunk_frames,
    )
    return {
        "schema": EVAL_COMPARISON_SCHEMA,
        "kind": "pixel",
        "shape": [int(dim) for dim in left.shape],
        "left_dtype": str(left.dtype),
        "right_dtype": str(right.dtype),
        "max_abs": float(diff.abs().max().item()),
        "mean_abs": float(diff.abs().mean().item()),
        "mae": float(diff.abs().mean().item()),
        "mse": mse,
        "psnr": psnr,
        "per_frame_mse": per_frame_mse,
        "pixel_chunk_mapping_status": mapping["status"],
        "pixel_chunk_mapping": mapping,
        "per_latent_chunk_pixel_mse": None,
    }


def latent_chunk_to_pixel_frame_ranges(
    *,
    decoded_frame_count: int,
    latent_frame_count: int = FULL_SEQUENCE_FRAME_COUNT,
    chunk_frames: int = FULL_SEQUENCE_CHUNK_FRAMES,
) -> dict[str, Any]:
    return {
        "status": "UNAVAILABLE",
        "reason": (
            "Wan VAE latent-to-pixel temporal mapping is not asserted in this "
            "evaluator; per-frame pixel metrics are reported, and latent "
            "per-chunk metrics remain the formal chunk comparison."
        ),
        "decoded_frame_count": int(decoded_frame_count),
        "latent_frame_count": int(latent_frame_count),
        "chunk_frames": int(chunk_frames),
        "latent_chunk_count": int(latent_frame_count) // int(chunk_frames),
        "ranges": None,
    }


def role_aware_latent_metrics(
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    role_map: Mapping[str, Sequence[int]] | None = None,
    chunk_frames: int = FULL_SEQUENCE_CHUNK_FRAMES,
) -> dict[str, Any]:
    role_map = full_sequence_role_map() if role_map is None else role_map
    diff = left.detach().float().cpu() - right.detach().float().cpu()
    metrics: dict[str, Any] = {}
    for role, chunks in role_map.items():
        parts = []
        for chunk_index in chunks:
            start = int(chunk_index) * int(chunk_frames)
            parts.append(diff[:, start:start + int(chunk_frames)])
        if not parts:
            continue
        value = torch.cat(parts, dim=1)
        metrics[str(role)] = {
            "chunks": [int(chunk) for chunk in chunks],
            "max_abs": float(value.abs().max().item()),
            "mean_abs": float(value.abs().mean().item()),
            "mse": float(value.square().mean().item()),
        }
    return metrics


def build_comparison_report(
    *,
    name: str,
    left_mode: str,
    right_mode: str,
    latent_left: torch.Tensor,
    latent_right: torch.Tensor,
    pixel_left: torch.Tensor | None = None,
    pixel_right: torch.Tensor | None = None,
    role_map: Mapping[str, Sequence[int]] | None = None,
) -> dict[str, Any]:
    report = {
        "schema": EVAL_COMPARISON_SCHEMA,
        "name": str(name),
        "left_mode": str(left_mode),
        "right_mode": str(right_mode),
        "latent": compare_latents(latent_left, latent_right),
        "pixel": (
            None
            if pixel_left is None or pixel_right is None
            else compare_pixel_frames(pixel_left, pixel_right)
        ),
        "role_aware_latent": (
            None
            if role_map is None
            else role_aware_latent_metrics(
                latent_left,
                latent_right,
                role_map=role_map,
            )
        ),
        "visual_review_status": "PENDING",
        "visual_quality_pass": None,
    }
    return report


def write_mode_outputs(
    *,
    mode_dir: Path | str,
    result: DeploymentResult,
    video_path: Path | str | None,
    fps: int,
) -> dict[str, Any]:
    mode_dir = Path(mode_dir)
    mode_dir.mkdir(parents=True, exist_ok=True)
    latent_path = mode_dir / "output_latent.pt"
    atomic_torch_save(
        {
            "schema": EVAL_MODE_OUTPUT_SCHEMA,
            "mode": result.trace["mode"],
            "latent": result.latent.detach().cpu(),
            "latent_sha256": tensor_sha256(result.latent.detach().cpu()),
            "common_inputs_fingerprint_sha256": result.trace[
                "common_inputs_fingerprint_sha256"
            ],
        },
        latent_path,
    )
    atomic_json_write(result.trace, mode_dir / "trace.json")
    summary = dict(result.summary)
    summary["output_latent"] = {
        "path": str(latent_path.resolve()),
        "sha256": file_sha256(latent_path),
        "tensor_sha256": tensor_sha256(result.latent.detach().cpu()),
    }
    if video_path is not None:
        video_path = Path(video_path)
        if not video_path.is_file() or video_path.stat().st_size <= 0:
            raise RuntimeError(f"mode video is missing or empty: {video_path}")
        summary["video"] = {
            "path": str(video_path.resolve()),
            "sha256": file_sha256(video_path),
            "size_bytes": int(video_path.stat().st_size),
            "fps": int(fps),
        }
    atomic_json_write(summary, mode_dir / "summary.json")
    return summary


def build_eval_manifest(
    *,
    common_inputs: Mapping[str, Any],
    common_inputs_fingerprint_sha256: str,
    mode_summaries: Mapping[str, Mapping[str, Any]],
    comparisons: Mapping[str, Mapping[str, Any]],
    output_dir: Path | str,
    git_sha: str,
) -> dict[str, Any]:
    fingerprint = assert_common_input_fingerprints(mode_summaries)
    if fingerprint != common_inputs_fingerprint_sha256:
        raise RuntimeError("mode common input fingerprint differs from common_inputs.json")
    rng_fingerprint = assert_rng_plan_fingerprints(mode_summaries)
    assert_mode_inputs_match_common(mode_summaries, common_inputs)
    manifest = {
        "schema": EVAL_SCHEMA,
        "status": "PASS",
        "runtime_git_sha": str(git_sha),
        "training_checkpoint_git_sha": str(
            common_inputs.get("training_checkpoint_git_sha")
        ),
        "output_dir": str(Path(output_dir).resolve()),
        "common_inputs": dict(common_inputs),
        "common_inputs_fingerprint_sha256": str(common_inputs_fingerprint_sha256),
        "rng_plan_fingerprint_sha256": rng_fingerprint,
        "modes": {
            key: {
                "status": value.get("status"),
                "latent_sha256": value.get("latent_sha256"),
                "source_noise_sha256": value.get("source_noise_sha256"),
                "conditioning_sha256": value.get("conditioning_sha256"),
                "rng_plan_fingerprint_sha256": value.get(
                    "rng_plan_fingerprint_sha256"
                ),
                "generation_elapsed_ms": value.get("generation_elapsed_ms"),
                "runtime_measurement_status": value.get(
                    "runtime_measurement_status"
                ),
                "video_sha256": (
                    value.get("video", {}).get("sha256")
                    if isinstance(value.get("video"), Mapping)
                    else None
                ),
                "mcp_call_count": value.get("mcp_call_count"),
            }
            for key, value in mode_summaries.items()
        },
        "comparisons": {
            key: {
                "visual_review_status": value.get("visual_review_status"),
                "visual_quality_pass": value.get("visual_quality_pass"),
            }
            for key, value in comparisons.items()
        },
        "engineering_acceptance": {
            "all_3_runs_complete": set(mode_summaries.keys())
            == {MODE_OFFICIAL_MAIN, MODE_TRAINED_MAIN, MODE_TRAINED_MCP1},
            "common_input_fingerprints_exact": True,
            "rng_plan_fingerprints_exact": True,
            "trained_main_zero_mcp_calls": int(
                mode_summaries[MODE_TRAINED_MAIN]["mcp_call_count"]
            )
            == 0,
            "trained_mcp1_role_map_exact": mode_summaries[MODE_TRAINED_MCP1].get(
                "role_map"
            )
            == full_sequence_role_map(),
            "videos_present_non_empty": all(
                isinstance(summary.get("video"), Mapping)
                and int(summary["video"].get("size_bytes", 1)) != 0
                for summary in mode_summaries.values()
            ),
            "no_auto_visual_threshold": True,
        },
        "visual_review_status": "PENDING",
        "visual_quality_pass": None,
    }
    if not all(bool(value) for value in manifest["engineering_acceptance"].values()):
        manifest["status"] = "FAIL"
    return manifest


def validate_trained_main_trace(trace: Mapping[str, Any]) -> None:
    if trace.get("mode") != MODE_TRAINED_MAIN:
        raise RuntimeError("trace is not trained_main")
    if int(trace.get("mcp_call_count", -1)) != 0:
        raise RuntimeError("trained_main must have zero MCP calls")
    for chunk in trace.get("chunks", []):
        _require_four_solver_steps(chunk, schedule_key="warped_timestep")


def validate_trained_mcp1_trace(trace: Mapping[str, Any]) -> None:
    if trace.get("mode") != MODE_TRAINED_MCP1:
        raise RuntimeError("trace is not trained_mcp1")
    if trace.get("role_map") != full_sequence_role_map():
        raise RuntimeError("trained_mcp1 role map mismatch")
    rounds = trace.get("parallel_rounds")
    if not isinstance(rounds, Sequence) or len(rounds) != 3:
        raise RuntimeError("trained_mcp1 must contain exactly three paired rounds")
    if trace.get("mcp_depths_used") != [1]:
        raise RuntimeError("trained_mcp1 must use MCP depth1 only")
    if int(trace.get("per_depth_call_counts", {}).get("2", -1)) != 0:
        raise RuntimeError("trained_mcp1 must not call MCP depth2")
    if int(trace.get("per_depth_call_counts", {}).get("3", -1)) != 0:
        raise RuntimeError("trained_mcp1 must not call MCP depth3")
    for round_index, round_record in enumerate(rounds):
        expected_current = 1 + 2 * int(round_index)
        expected_next = expected_current + 1
        if round_record.get("current_chunk_index") != expected_current:
            raise RuntimeError("trained_mcp1 current chunk mapping mismatch")
        if round_record.get("next_chunk_index") != expected_next:
            raise RuntimeError("trained_mcp1 next chunk mapping mismatch")
        if round_record.get("clean_recache_order") != [expected_current, expected_next]:
            raise RuntimeError("trained_mcp1 clean recache order mismatch")
        joint_steps = round_record.get("joint_solver_steps")
        if not isinstance(joint_steps, Sequence) or len(joint_steps) != len(RAW_DEPLOYMENT_SCHEDULE):
            raise RuntimeError("trained_mcp1 paired round must have four solver steps")


def tensor_json_summary(tensor: torch.Tensor) -> dict[str, Any]:
    summary = tensor_summary(tensor.detach().cpu())
    return {
        "shape": summary["shape"],
        "dtype": summary["dtype"],
        "finite": summary["finite"],
        "sha256": summary["sha256"],
    }


def conditioning_json_summary(conditional_dict: Mapping[str, Any]) -> dict[str, Any]:
    safe = _json_safe_conditioning_summary(conditional_dict)
    return {
        "sha256": hashlib.sha256(
            json.dumps(
                safe,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ).encode("utf-8")
        ).hexdigest(),
        "summary": safe,
    }


def teacher_identity_json(metadata: Mapping[str, Any]) -> dict[str, Any]:
    fields = (
        "identity",
        "sample_index",
        "sample_id",
        "split",
        "split_index",
        "source_line_index",
        "shard_id",
        "plan_index",
        "latent_path",
        "latent_file_sha256",
        "prompt_sha256",
    )
    return {field: metadata.get(field) for field in fields if field in metadata}


def global_rng_state_hash(device: torch.device | str) -> str:
    device = torch.device(device)
    cpu_state = torch.get_rng_state()
    payload = bytearray(cpu_state.detach().cpu().numpy().tobytes())
    if device.type == "cuda" and torch.cuda.is_available():
        cuda_state = torch.cuda.get_rng_state(device)
        payload.extend(cuda_state.detach().cpu().numpy().tobytes())
    return hashlib.sha256(bytes(payload)).hexdigest()


def reset_torch_rollout_rng(seed: int, device: torch.device | str) -> None:
    torch.manual_seed(int(seed))
    device = torch.device(device)
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def consume_teacher_compatibility_draw(
    *,
    num_chunks: int,
    num_denoising_steps: int,
    device: torch.device | str,
    draw_order: int,
) -> dict[str, Any]:
    state_before = global_rng_state_hash(device)
    values = torch.randint(
        low=0,
        high=int(num_denoising_steps),
        size=(int(num_chunks),),
        device=device,
        dtype=torch.long,
    )
    state_after = global_rng_state_hash(device)
    return {
        "draw_order": int(draw_order),
        "purpose": "teacher_exit_flag_randint_compatibility",
        "operation": "torch.randint",
        "low": 0,
        "high": int(num_denoising_steps),
        "size": [int(num_chunks)],
        "dtype": str(values.dtype),
        "device": str(values.device),
        "state_before_hash": state_before,
        "state_after_hash": state_after,
        "values_sha256": tensor_sha256(values.detach().cpu()),
        "values_discarded": True,
    }


def randn_like_with_trace(
    tensor: torch.Tensor,
    *,
    device: torch.device | str,
    purpose: str,
    draw_order: int,
    chunk_index: int,
    solver_step_index: int | None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    state_before = global_rng_state_hash(device)
    noise = torch.randn_like(tensor)
    state_after = global_rng_state_hash(device)
    return noise, {
        "draw_order": int(draw_order),
        "purpose": str(purpose),
        "chunk_index": int(chunk_index),
        "solver_step_index": (
            None if solver_step_index is None else int(solver_step_index)
        ),
        "state_before_hash": state_before,
        "state_after_hash": state_after,
        "noise": tensor_json_summary(noise),
    }


def _deployment_result(
    *,
    mode: EvalMode,
    output: torch.Tensor,
    checkpoint: DeploymentCheckpointRecord,
    schedule: DeploymentSchedule,
    common_inputs: Mapping[str, Any],
    common_inputs_fingerprint_sha256: str,
    actual_source_noise_sha256: str,
    actual_conditioning_sha256: str,
    git_sha: str,
    chunks: list[dict[str, Any]],
    parallel_rounds: list[dict[str, Any]],
    execution_plan: list[dict[str, Any]],
    counts: Mapping[str, int],
    rng_trace: Mapping[str, Any],
    generation_elapsed_ms: float,
) -> DeploymentResult:
    if str(actual_source_noise_sha256) != str(common_inputs["source_noise_sha256"]):
        raise RuntimeError("mode source_noise SHA differs from common inputs")
    if str(actual_conditioning_sha256) != str(common_inputs["conditioning_sha256"]):
        raise RuntimeError("mode conditioning SHA differs from common inputs")
    include_mcp = mode == MODE_TRAINED_MCP1
    role_map = (
        {"main_only": list(range(FULL_SEQUENCE_NUM_CHUNKS))}
        if not include_mcp
        else full_sequence_role_map()
    )
    mcp_call_count = int(counts.get("mcp_call_count", 0))
    per_depth = {
        "1": int(counts.get("mcp_depth1_call_count", 0)),
        "2": int(counts.get("mcp_depth2_call_count", 0)),
        "3": int(counts.get("mcp_depth3_call_count", 0)),
    }
    trace = {
        "schema": EVAL_MODE_OUTPUT_SCHEMA,
        "mode": str(mode),
        "git_sha": str(git_sha),
        "runtime_git_sha": str(git_sha),
        "training_checkpoint_git_sha": common_inputs.get(
            "training_checkpoint_git_sha"
        ),
        "checkpoint": checkpoint.to_json(),
        "schedule": schedule.to_json(include_mcp=include_mcp),
        "source_noise_sha256": str(common_inputs["source_noise_sha256"]),
        "conditioning_sha256": str(common_inputs["conditioning_sha256"]),
        "prompt_sha256": str(common_inputs["prompt"]["prompt_sha256"]),
        "common_inputs": dict(common_inputs),
        "common_inputs_fingerprint_sha256": str(common_inputs_fingerprint_sha256),
        "role_map": role_map,
        "execution_plan": execution_plan,
        "chunks": chunks,
        "parallel_rounds": parallel_rounds,
        "mcp_enabled": include_mcp,
        "mcp_depths_used": [1] if include_mcp else [],
        "mcp_call_count": mcp_call_count,
        "per_depth_call_counts": per_depth,
        "static_runtime_counts": dict(counts),
        "rng": rng_trace,
        "rng_plan_fingerprint_sha256": str(rng_trace["rng_plan_fingerprint_sha256"]),
        "runtime_measurement_status": RUNTIME_MEASUREMENT_STATUS,
        "generation_elapsed_ms": float(generation_elapsed_ms),
        "forbidden_features": {
            "target_refinement": False,
            "verifier": False,
            "rejection_routing": False,
            "hybrid_refinement": False,
            "dmd": False,
            "self_rollout": False,
            "mcp_depth2_emitted": False,
            "mcp_depth3_emitted": False,
        },
        "finite_checks": {"output_latent": True},
        "latent_sha256": tensor_sha256(output.detach().cpu()),
        "status": "PASS",
    }
    summary = {
        "schema": EVAL_MODE_OUTPUT_SCHEMA,
        "mode": str(mode),
        "status": "PASS",
        "git_sha": str(git_sha),
        "runtime_git_sha": str(git_sha),
        "training_checkpoint_git_sha": common_inputs.get(
            "training_checkpoint_git_sha"
        ),
        "checkpoint": checkpoint.to_json(),
        "source_noise_sha256": str(common_inputs["source_noise_sha256"]),
        "conditioning_sha256": str(common_inputs["conditioning_sha256"]),
        "prompt_sha256": str(common_inputs["prompt"]["prompt_sha256"]),
        "common_inputs_fingerprint_sha256": str(common_inputs_fingerprint_sha256),
        "latent_sha256": trace["latent_sha256"],
        "schedule": trace["schedule"],
        "role_map": role_map,
        "mcp_call_count": mcp_call_count,
        "rng_plan_fingerprint_sha256": str(rng_trace["rng_plan_fingerprint_sha256"]),
        "runtime_measurement_status": RUNTIME_MEASUREMENT_STATUS,
        "generation_elapsed_ms": float(generation_elapsed_ms),
        "per_depth_call_counts": per_depth,
        "static_runtime_counts": dict(counts),
        "chunk_role_map": role_map,
        "visual_review_status": "PENDING",
        "visual_quality_pass": None,
    }
    if mode == MODE_TRAINED_MAIN:
        validate_trained_main_trace(trace)
    if mode == MODE_TRAINED_MCP1:
        validate_trained_mcp1_trace(trace)
    return DeploymentResult(latent=output.detach().cpu(), trace=trace, summary=summary)


def _run_main_chunk(
    *,
    runtime: DeploymentRuntime,
    source_noise: torch.Tensor,
    output: torch.Tensor,
    conditional_dict: Mapping[str, Any],
    schedule: DeploymentSchedule,
    rng_plan: Mapping[str, Any],
    counts: dict[str, int],
    chunk_index: int,
    role: str,
    cursor_before: int,
    cursor_after: int,
) -> dict[str, Any]:
    chunk_frames = int(runtime.num_frame_per_block)
    start_frame = int(chunk_index) * chunk_frames
    current = source_noise[:, start_frame:start_frame + chunk_frames].detach().clone()
    step_records: list[dict[str, Any]] = []
    last_rollback = None
    for step_index, warped_timestep in enumerate(schedule.main_warped_schedule):
        forward_input = current.detach()
        snapshot = KVSnapshot.capture(runtime.kv_cache)
        kv_before = kv_boundary_summary(runtime.kv_cache)
        _require_kv_boundary_consistent(kv_before, label="main before forward")
        timestep = _timestep_chunk(float(warped_timestep), current)

        def call_main(
            current_chunk: torch.Tensor = current,
            current_timestep: torch.Tensor = timestep,
        ):
            return runtime.generator(
                noisy_image_or_video=current_chunk,
                conditional_dict=dict(conditional_dict),
                timestep=current_timestep,
                kv_cache=runtime.kv_cache,
                crossattn_cache=runtime.crossattn_cache,
                current_start=start_frame * int(runtime.frame_seq_length),
            )

        outputs, rng_guard = _call_with_rng_guard(
            device=current.device,
            label="main_solver_forward",
            fn=call_main,
        )
        counts["main_solver_forward_count"] = int(
            counts.get("main_solver_forward_count", 0)
        ) + 1
        flow_pred, clean_pred = _unpack_main_outputs(outputs)
        _ensure_finite_tensor(flow_pred, name="main_flow_pred")
        _ensure_finite_tensor(clean_pred, name="main_clean_pred")
        kv_temp = kv_boundary_summary(runtime.kv_cache)
        _require_kv_boundary_consistent(kv_temp, label="main temporary forward")
        restored = snapshot.restore(runtime.kv_cache)
        if not restored:
            raise RuntimeError("KV snapshot restore failed")
        kv_rollback = kv_boundary_summary(runtime.kv_cache)
        _require_kv_rollback_matches(kv_before, kv_rollback)
        last_rollback = kv_rollback
        transition = None
        if step_index < len(schedule.main_warped_schedule) - 1:
            next_t = float(schedule.main_warped_schedule[step_index + 1])
            noise, noise_record = _plan_transition_noise(
                rng_plan,
                chunk_index=chunk_index,
                step_index=step_index,
                template=clean_pred.flatten(0, 1),
            )
            current = runtime.scheduler.add_noise(
                clean_pred.flatten(0, 1),
                noise,
                torch.full(
                    (clean_pred.flatten(0, 1).shape[0],),
                    next_t,
                    device=clean_pred.device,
                    dtype=torch.float32,
                ),
            ).unflatten(0, clean_pred.shape[:2])
            _ensure_finite_tensor(current, name="main_re_noised_state")
            transition = {
                "next_warped_timestep": next_t,
                "rng_plan_record": noise_record,
                "re_noised_tensor": tensor_json_summary(current),
            }
        else:
            output[:, start_frame:start_frame + chunk_frames] = clean_pred
        step_records.append(
            {
                "raw_index": int(step_index),
                "raw_timestep": float(schedule.raw_schedule[step_index]),
                "warped_timestep": float(warped_timestep),
                "input_tensor": tensor_json_summary(forward_input),
                "flow_tensor": tensor_json_summary(flow_pred),
                "output_x0_tensor": tensor_json_summary(clean_pred),
                "forward_rng": rng_guard,
                "kv": {
                    "before": kv_before,
                    "temporary_after_forward": kv_temp,
                    "rollback_after_forward": kv_rollback,
                    "visible_data_restored": True,
                },
                "transition": transition,
            }
        )
    clean_chunk = output[:, start_frame:start_frame + chunk_frames]
    clean_recache = _clean_recache(
        runtime=runtime,
        conditional_dict=conditional_dict,
        rng_plan=rng_plan,
        counts=counts,
        clean_chunk=clean_chunk,
        chunk_index=chunk_index,
        start_frame=start_frame,
        expected_before=last_rollback,
    )
    produced_by = "Main"
    role_name = {
        "bootstrap": "bootstrap",
        "main": "main_only",
    }.get(role, role)
    return {
        "chunk_index": int(chunk_index),
        "role": role_name,
        "produced_by": produced_by,
        "start_frame": int(start_frame),
        "num_frames": int(chunk_frames),
        "solver_steps": step_records,
        "clean_recache": clean_recache,
        "commit": {
            "main_only": True,
            "next_commit": None,
            "commit_order": [int(chunk_index)],
            "cursor_before": int(cursor_before),
            "cursor_after": int(cursor_after),
        },
    }


def _run_mcp1_pair(
    *,
    runtime: DeploymentRuntime,
    mcp_scheduler: Any,
    source_noise: torch.Tensor,
    output: torch.Tensor,
    conditional_dict: Mapping[str, Any],
    schedule: DeploymentSchedule,
    rng_plan: Mapping[str, Any],
    counts: dict[str, int],
    current_chunk_index: int,
    next_chunk_index: int,
    round_index: int,
    cursor_before: int,
    cursor_after: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    chunk_frames = int(runtime.num_frame_per_block)
    current_start = int(current_chunk_index) * chunk_frames
    next_start = int(next_chunk_index) * chunk_frames
    current_state = source_noise[:, current_start:current_start + chunk_frames].detach().clone()
    next_state = source_noise[:, next_start:next_start + chunk_frames].detach().clone()
    current_steps: list[dict[str, Any]] = []
    next_steps: list[dict[str, Any]] = []
    joint_steps: list[dict[str, Any]] = []
    last_rollback = None
    for step_index, (raw_timestep, main_t, mcp_t) in enumerate(
        zip(
            schedule.raw_schedule,
            schedule.main_warped_schedule,
            schedule.mcp_warped_schedule,
        )
    ):
        snapshot = KVSnapshot.capture(runtime.kv_cache)
        kv_before = kv_boundary_summary(runtime.kv_cache)
        _require_kv_boundary_consistent(kv_before, label="mcp1 pair before forward")
        main_timestep = _timestep_chunk(float(main_t), current_state)
        mcp_timestep = _timestep_chunk(float(mcp_t), next_state)

        def call_joint(
            current_chunk: torch.Tensor = current_state,
            current_timestep: torch.Tensor = main_timestep,
            future_chunk: torch.Tensor = next_state,
            future_timestep: torch.Tensor = mcp_timestep,
        ):
            return runtime.generator(
                noisy_image_or_video=current_chunk,
                conditional_dict=dict(conditional_dict),
                timestep=current_timestep,
                kv_cache=runtime.kv_cache,
                crossattn_cache=runtime.crossattn_cache,
                current_start=current_start * int(runtime.frame_seq_length),
                mcp_future_noises=[future_chunk],
                mcp_future_start_frames=[next_start],
                mcp_timesteps=[future_timestep],
            )

        outputs, rng_guard = _call_with_rng_guard(
            device=current_state.device,
            label="mcp1_joint_solver_forward",
            fn=call_joint,
        )
        counts["main_solver_forward_count"] = int(
            counts.get("main_solver_forward_count", 0)
        ) + 1
        counts["mcp_call_count"] = int(counts.get("mcp_call_count", 0)) + 1
        counts["mcp_depth1_call_count"] = int(counts.get("mcp_depth1_call_count", 0)) + 1
        if not isinstance(outputs, (tuple, list)) or len(outputs) != 3:
            raise RuntimeError("MCP1 deployment forward must return main outputs and MCP outputs")
        main_flow, main_clean = _unpack_main_outputs(outputs)
        mcp_outputs = outputs[2]
        if not isinstance(mcp_outputs, (tuple, list)) or len(mcp_outputs) != 1:
            raise RuntimeError("MCP1 deployment must request and return depth1 only")
        counts["returned_mcp_output_count"] = int(
            counts.get("returned_mcp_output_count", 0)
        ) + 1
        mcp_flow = mcp_outputs[0]
        if not torch.is_tensor(mcp_flow):
            raise TypeError("MCP1 flow output must be a tensor")
        mcp_clean = mcp_flow_to_x0(
            mcp_scheduler,
            mcp_flow=mcp_flow,
            next_state=next_state,
            mcp_timestep=mcp_timestep,
        )
        for name, tensor in (
            ("mcp1_main_flow", main_flow),
            ("mcp1_main_clean", main_clean),
            ("mcp1_flow", mcp_flow),
            ("mcp1_clean", mcp_clean),
        ):
            _ensure_finite_tensor(tensor, name=name)
        kv_temp = kv_boundary_summary(runtime.kv_cache)
        _require_kv_boundary_consistent(kv_temp, label="mcp1 pair temporary forward")
        restored = snapshot.restore(runtime.kv_cache)
        if not restored:
            raise RuntimeError("KV snapshot restore failed")
        kv_rollback = kv_boundary_summary(runtime.kv_cache)
        _require_kv_rollback_matches(kv_before, kv_rollback)
        last_rollback = kv_rollback
        current_transition = None
        next_transition = None
        if step_index < len(schedule.raw_schedule) - 1:
            next_main_t = float(schedule.main_warped_schedule[step_index + 1])
            current_noise, current_noise_record = _plan_transition_noise(
                rng_plan,
                chunk_index=current_chunk_index,
                step_index=step_index,
                template=main_clean.flatten(0, 1),
            )
            current_state = runtime.scheduler.add_noise(
                main_clean.flatten(0, 1),
                current_noise,
                torch.full(
                    (main_clean.flatten(0, 1).shape[0],),
                    next_main_t,
                    device=main_clean.device,
                    dtype=torch.float32,
                ),
            ).unflatten(0, main_clean.shape[:2])
            current_transition = {
                "next_warped_timestep": next_main_t,
                "rng_plan_record": current_noise_record,
                "re_noised_tensor": tensor_json_summary(current_state),
            }
            next_mcp_t = float(schedule.mcp_warped_schedule[step_index + 1])
            next_noise, next_noise_record = _plan_transition_noise(
                rng_plan,
                chunk_index=next_chunk_index,
                step_index=step_index,
                template=mcp_clean.flatten(0, 1),
            )
            next_state = mcp_scheduler.add_noise(
                mcp_clean.flatten(0, 1),
                next_noise,
                torch.full(
                    (mcp_clean.flatten(0, 1).shape[0],),
                    next_mcp_t,
                    device=mcp_clean.device,
                    dtype=torch.float32,
                ),
            ).unflatten(0, mcp_clean.shape[:2])
            next_transition = {
                "next_warped_timestep": next_mcp_t,
                "rng_plan_record": next_noise_record,
                "re_noised_tensor": tensor_json_summary(next_state),
            }
        else:
            output[:, current_start:current_start + chunk_frames] = main_clean
            output[:, next_start:next_start + chunk_frames] = mcp_clean
        joint_kv = {
            "before": kv_before,
            "temporary_after_forward": kv_temp,
            "rollback_after_forward": kv_rollback,
            "visible_data_restored": True,
        }
        current_steps.append(
            {
                "raw_index": int(step_index),
                "raw_timestep": float(raw_timestep),
                "warped_timestep": float(main_t),
                "flow_tensor": tensor_json_summary(main_flow),
                "output_x0_tensor": tensor_json_summary(main_clean),
                "forward_rng": rng_guard,
                "kv": joint_kv,
                "transition": current_transition,
            }
        )
        next_steps.append(
            {
                "raw_index": int(step_index),
                "raw_timestep": float(raw_timestep),
                "mcp_warped_timestep": float(mcp_t),
                "flow_tensor": tensor_json_summary(mcp_flow),
                "output_x0_tensor": tensor_json_summary(mcp_clean),
                "returned_mcp_output_count": 1,
                "mcp_depths_requested": [1],
                "transition": next_transition,
            }
        )
        joint_steps.append(
            {
                "raw_index": int(step_index),
                "raw_timestep": float(raw_timestep),
                "main_warped_timestep": float(main_t),
                "mcp_warped_timestep": float(mcp_t),
                "raw_index_aligned": True,
                "returned_mcp_output_count": 1,
                "forward_rng": rng_guard,
                "kv": joint_kv,
            }
        )
    current_clean = output[:, current_start:current_start + chunk_frames]
    next_clean = output[:, next_start:next_start + chunk_frames]
    current_recache = _clean_recache(
        runtime=runtime,
        conditional_dict=conditional_dict,
        rng_plan=rng_plan,
        counts=counts,
        clean_chunk=current_clean,
        chunk_index=current_chunk_index,
        start_frame=current_start,
        expected_before=last_rollback,
    )
    next_recache = _clean_recache(
        runtime=runtime,
        conditional_dict=conditional_dict,
        rng_plan=rng_plan,
        counts=counts,
        clean_chunk=next_clean,
        chunk_index=next_chunk_index,
        start_frame=next_start,
        expected_before=None,
    )
    current_record = {
        "chunk_index": int(current_chunk_index),
        "role": "main_current",
        "produced_by": "Main",
        "start_frame": int(current_start),
        "num_frames": int(chunk_frames),
        "solver_steps": current_steps,
        "clean_recache": current_recache,
        "commit": {
            "main_only": False,
            "next_commit": int(next_chunk_index),
            "commit_order": [int(current_chunk_index), int(next_chunk_index)],
            "cursor_before": int(cursor_before),
            "cursor_after": int(cursor_after),
        },
    }
    next_record = {
        "chunk_index": int(next_chunk_index),
        "role": "mcp_next",
        "produced_by": "MCP1",
        "start_frame": int(next_start),
        "num_frames": int(chunk_frames),
        "solver_steps": next_steps,
        "clean_recache": next_recache,
        "commit": {
            "main_only": False,
            "accepted_next": True,
            "recomputed_by_main": False,
            "commit_order": [int(current_chunk_index), int(next_chunk_index)],
            "cursor_before": int(cursor_before),
            "cursor_after": int(cursor_after),
        },
    }
    round_record = {
        "round_index": int(round_index),
        "current_chunk_index": int(current_chunk_index),
        "next_chunk_index": int(next_chunk_index),
        "cursor_before": int(cursor_before),
        "cursor_after": int(cursor_after),
        "commit_order": [int(current_chunk_index), int(next_chunk_index)],
        "joint_solver_steps": joint_steps,
        "clean_recache_order": [int(current_chunk_index), int(next_chunk_index)],
        "temporary_kv_writes_rolled_back": True,
    }
    return current_record, next_record, round_record


def _clean_recache(
    *,
    runtime: DeploymentRuntime,
    conditional_dict: Mapping[str, Any],
    rng_plan: Mapping[str, Any],
    counts: dict[str, int],
    clean_chunk: torch.Tensor,
    chunk_index: int,
    start_frame: int,
    expected_before: Mapping[str, Any] | None,
) -> dict[str, Any]:
    before = kv_boundary_summary(runtime.kv_cache)
    _require_kv_boundary_consistent(before, label="clean recache before")
    if expected_before is not None:
        _require_kv_rollback_matches(expected_before, before)
    context_timestep = torch.full(
        clean_chunk.shape[:2],
        int(runtime.context_noise),
        device=clean_chunk.device,
        dtype=torch.int64,
    )
    context_noise, noise_record = _plan_context_noise(
        rng_plan,
        chunk_index=chunk_index,
        template=clean_chunk.flatten(0, 1),
    )
    context_latent = runtime.scheduler.add_noise(
        clean_chunk.flatten(0, 1),
        context_noise,
        context_timestep.flatten(0, 1),
    ).unflatten(0, clean_chunk.shape[:2])
    _, rng_guard = _call_with_rng_guard(
        device=clean_chunk.device,
        label="clean_recache_forward",
        fn=lambda: runtime.generator(
            noisy_image_or_video=context_latent,
            conditional_dict=dict(conditional_dict),
            timestep=context_timestep,
            kv_cache=runtime.kv_cache,
            crossattn_cache=runtime.crossattn_cache,
            current_start=int(start_frame) * int(runtime.frame_seq_length),
        ),
    )
    counts["clean_recache_forward_count"] = int(
        counts.get("clean_recache_forward_count", 0)
    ) + 1
    after = kv_boundary_summary(runtime.kv_cache)
    _require_clean_recache_transition(
        before,
        after,
        start_frame=int(start_frame),
        chunk_frames=int(runtime.num_frame_per_block),
        frame_seq_length=int(runtime.frame_seq_length),
    )
    return {
        "context_noise": int(runtime.context_noise),
        "before": before,
        "after": after,
        "rng_plan_record": noise_record,
        "forward_rng": rng_guard,
        "context_latent": tensor_json_summary(context_latent),
    }


def mcp_flow_to_x0(
    mcp_scheduler: Any,
    *,
    mcp_flow: torch.Tensor,
    next_state: torch.Tensor,
    mcp_timestep: torch.Tensor,
) -> torch.Tensor:
    original_shape = next_state.shape
    x0 = mcp_scheduler.step(
        mcp_flow.flatten(0, 1),
        mcp_timestep.flatten(0, 1),
        next_state.flatten(0, 1),
        to_final=True,
    ).unflatten(0, original_shape[:2])
    if tuple(x0.shape) != tuple(original_shape):
        raise RuntimeError("MCP flow-to-x0 shape mismatch")
    return x0.to(device=next_state.device, dtype=next_state.dtype)


def _plan_transition_noise(
    rng_plan: Mapping[str, Any],
    *,
    chunk_index: int,
    step_index: int,
    template: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, Any]]:
    transitions = rng_plan.get("transition_noises")
    if not isinstance(transitions, Mapping) or (int(chunk_index), int(step_index)) not in transitions:
        raise RuntimeError("deployment RNG plan missing transition noise")
    return _checked_plan_noise(
        transitions[(int(chunk_index), int(step_index))],
        rng_plan,
        chunk_index=chunk_index,
        purpose="transition_re_noise",
        solver_step_index=step_index,
        template=template,
    )


def _plan_context_noise(
    rng_plan: Mapping[str, Any],
    *,
    chunk_index: int,
    template: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, Any]]:
    contexts = rng_plan.get("context_noises")
    if not isinstance(contexts, Mapping) or int(chunk_index) not in contexts:
        raise RuntimeError("deployment RNG plan missing context noise")
    return _checked_plan_noise(
        contexts[int(chunk_index)],
        rng_plan,
        chunk_index=chunk_index,
        purpose="context_clean_recache_noise",
        solver_step_index=None,
        template=template,
    )


def _checked_plan_noise(
    noise: Any,
    rng_plan: Mapping[str, Any],
    *,
    chunk_index: int,
    purpose: str,
    solver_step_index: int | None,
    template: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if not torch.is_tensor(noise):
        raise TypeError("deployment RNG plan noise must be a tensor")
    if tuple(noise.shape) != tuple(template.shape):
        raise RuntimeError("deployment RNG plan noise shape mismatch")
    if noise.dtype != template.dtype:
        raise RuntimeError("deployment RNG plan noise dtype mismatch")
    if noise.device != template.device:
        noise = noise.to(device=template.device)
    _ensure_finite_tensor(noise, name="deployment_rng_plan_noise")
    matches = []
    for record in rng_plan.get("trace", {}).get("draws", []):
        if (
            isinstance(record, Mapping)
            and int(record.get("absolute_chunk_index", -1)) == int(chunk_index)
            and record.get("purpose") == purpose
            and record.get("solver_step_index")
            == (None if solver_step_index is None else int(solver_step_index))
        ):
            matches.append(record)
    if len(matches) != 1:
        raise RuntimeError("deployment RNG plan record lookup failed")
    record = dict(matches[0])
    if record.get("noise", {}).get("sha256") != tensor_sha256(noise.detach().cpu()):
        raise RuntimeError("deployment RNG plan noise SHA mismatch")
    return noise, record


def _call_with_rng_guard(
    *,
    device: torch.device | str,
    label: str,
    fn,
) -> tuple[Any, dict[str, Any]]:
    before = global_rng_state_hash(device)
    result = fn()
    after = global_rng_state_hash(device)
    if before != after:
        raise RuntimeError(f"{label} changed active global RNG state")
    return result, {
        "label": str(label),
        "state_before_hash": before,
        "state_after_hash": after,
        "unchanged": True,
    }


def _timestep_chunk(value: float, target: torch.Tensor) -> torch.Tensor:
    return torch.full(
        target.shape[:2],
        float(value),
        device=target.device,
        dtype=torch.float32,
    )


def kv_boundary_summary(kv_cache: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    layers = []
    for index, layer in enumerate(kv_cache):
        layers.append(
            {
                "layer": int(index),
                "global_end_index": _cache_index_value(layer, "global_end_index"),
                "local_end_index": _cache_index_value(layer, "local_end_index"),
            }
        )
    global_values = [layer["global_end_index"] for layer in layers]
    local_values = [layer["local_end_index"] for layer in layers]
    return {
        "layers": layers,
        "global_end_index": None if not global_values else global_values[0],
        "local_end_index": None if not local_values else local_values[0],
        "global_boundary_consistent": len(set(global_values)) <= 1,
        "local_boundary_consistent": len(set(local_values)) <= 1,
        "layer_count": len(layers),
    }


def _require_kv_boundary_consistent(summary: Mapping[str, Any], *, label: str) -> None:
    if int(summary.get("layer_count", 0)) <= 0:
        raise RuntimeError(f"KV {label} layer_count must be greater than 0")
    if summary.get("global_boundary_consistent") is not True:
        raise RuntimeError(f"KV {label} global boundaries are inconsistent")
    if summary.get("local_boundary_consistent") is not True:
        raise RuntimeError(f"KV {label} local boundaries are inconsistent")


def _require_kv_rollback_matches(before: Mapping[str, Any], rollback: Mapping[str, Any]) -> None:
    before_layers = before.get("layers")
    rollback_layers = rollback.get("layers")
    if not isinstance(before_layers, Sequence) or not isinstance(rollback_layers, Sequence):
        raise TypeError("KV rollback summary missing layers")
    if len(before_layers) != len(rollback_layers):
        raise RuntimeError("KV rollback layer count differs from before")
    for before_layer, rollback_layer in zip(before_layers, rollback_layers):
        if not isinstance(before_layer, Mapping) or not isinstance(rollback_layer, Mapping):
            raise TypeError("KV rollback layer entry invalid")
        for field in ("global_end_index", "local_end_index"):
            if rollback_layer.get(field) != before_layer.get(field):
                raise RuntimeError("KV rollback boundary mismatch")


def _require_clean_recache_transition(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    *,
    start_frame: int,
    chunk_frames: int,
    frame_seq_length: int,
) -> None:
    _require_kv_boundary_consistent(after, label="clean recache after")
    expected_before = int(start_frame) * int(frame_seq_length)
    expected_after = (int(start_frame) + int(chunk_frames)) * int(frame_seq_length)
    if int(before["local_end_index"]) != expected_before:
        raise RuntimeError("clean recache before boundary mismatch")
    if int(after["local_end_index"]) != expected_after:
        raise RuntimeError("clean recache after boundary mismatch")


def _validate_runtime(runtime: DeploymentRuntime) -> None:
    if int(runtime.num_frame_per_block) != FULL_SEQUENCE_CHUNK_FRAMES:
        raise RuntimeError("deployment runtime requires chunk_frames=3")
    if int(runtime.frame_seq_length) <= 0:
        raise RuntimeError("deployment runtime frame_seq_length must be positive")
    if not runtime.kv_cache:
        raise RuntimeError("deployment runtime requires an initialized KV cache")


def _validate_source_noise(
    source_noise: torch.Tensor,
    *,
    teacher_payload: Mapping[str, Any],
) -> None:
    if not torch.is_tensor(source_noise):
        raise TypeError("source_noise must be a tensor")
    if source_noise.ndim != 5:
        raise ValueError("source_noise must have layout [B, F, C, H, W]")
    if int(source_noise.shape[1]) != FULL_SEQUENCE_FRAME_COUNT:
        raise RuntimeError("deployment evaluator requires 21 latent frames")
    if int(source_noise.shape[1]) % FULL_SEQUENCE_CHUNK_FRAMES != 0:
        raise RuntimeError("source_noise frame count must be chunk-aligned")
    expected = teacher_payload.get("source_noise")
    if torch.is_tensor(expected):
        if tensor_sha256(source_noise.detach().cpu()) != tensor_sha256(expected.detach().cpu()):
            raise RuntimeError("source_noise differs from teacher payload")
    _ensure_finite_tensor(source_noise, name="source_noise")


def _require_teacher_schedule(common_record: Mapping[str, Any]) -> None:
    raw = tuple(float(value) for value in common_record.get("raw_teacher_schedule", ()))
    warped = tuple(float(value) for value in common_record.get("warped_teacher_schedule", ()))
    if raw and raw != RAW_DEPLOYMENT_SCHEDULE:
        raise RuntimeError("teacher raw schedule differs from deployment schedule")
    if warped:
        _require_schedule_close(warped, MAIN_DEPLOYMENT_SCHEDULE, "teacher main")


def _require_schedule_close(
    actual: Sequence[float],
    expected: Sequence[float],
    label: str,
    *,
    tolerance: float = 1.0e-4,
) -> None:
    if len(actual) != len(expected):
        raise RuntimeError(f"{label} schedule length mismatch")
    for left, right in zip(actual, expected):
        if abs(float(left) - float(right)) > float(tolerance):
            raise RuntimeError(f"{label} schedule mismatch")


def _unpack_main_outputs(outputs: Any) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(outputs, (tuple, list)) or len(outputs) < 2:
        raise RuntimeError("generator must return at least (flow, clean)")
    flow, clean = outputs[0], outputs[1]
    if not torch.is_tensor(flow) or not torch.is_tensor(clean):
        raise TypeError("generator flow and clean outputs must be tensors")
    return flow, clean


def _ensure_finite_tensor(tensor: torch.Tensor, *, name: str) -> None:
    if not torch.is_tensor(tensor):
        raise TypeError(f"{name} must be a tensor")
    if not bool(torch.isfinite(tensor.detach().float()).all().item()):
        raise RuntimeError(f"{name} contains nonfinite values")


def _runtime_mcp_call_count(generator: Any) -> int:
    if hasattr(generator, "mcp_call_count"):
        return int(getattr(generator, "mcp_call_count"))
    if hasattr(generator, "calls"):
        calls = getattr(generator, "calls")
        if isinstance(calls, Sequence):
            return sum(
                1
                for call in calls
                if isinstance(call, Mapping) and bool(call.get("mcp_requested"))
            )
    return 0


def _require_four_solver_steps(chunk: Mapping[str, Any], *, schedule_key: str) -> None:
    steps = chunk.get("solver_steps")
    if not isinstance(steps, Sequence) or len(steps) != len(RAW_DEPLOYMENT_SCHEDULE):
        raise RuntimeError("chunk must have four solver steps")
    for step_index, step in enumerate(steps):
        if not isinstance(step, Mapping):
            raise RuntimeError("solver step must be mapping")
        if int(step.get("raw_index", -1)) != step_index:
            raise RuntimeError("solver raw_index mismatch")
        if abs(float(step.get("raw_timestep")) - RAW_DEPLOYMENT_SCHEDULE[step_index]) > 1e-4:
            raise RuntimeError("solver raw_timestep mismatch")
        if schedule_key in step and not isinstance(step[schedule_key], (int, float)):
            raise RuntimeError("solver warped timestep missing")


def _json_safe_conditioning_summary(value: Any) -> Any:
    if torch.is_tensor(value):
        return tensor_json_summary(value.detach().cpu())
    if isinstance(value, Mapping):
        return {
            str(key): _json_safe_conditioning_summary(child)
            for key, child in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe_conditioning_summary(child) for child in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _json_safe(value: Any) -> Any:
    if torch.is_tensor(value):
        return tensor_json_summary(value)
    if isinstance(value, Mapping):
        return {
            str(key): _json_safe(child)
            for key, child in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe(child) for child in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _cache_index_value(layer: Mapping[str, Any], key: str) -> int:
    value = layer.get(key)
    if torch.is_tensor(value):
        return int(value.detach().cpu().reshape(-1)[0].item())
    return int(value)


def _set_cache_index(layer: Mapping[str, Any], key: str, value: int) -> None:
    target = layer.get(key)
    if torch.is_tensor(target):
        target.fill_(int(value))
    else:
        layer[key] = int(value)  # type: ignore[index]


__all__ = [
    "CHECKPOINT_VALIDATION_SCHEMA",
    "EVAL_SCHEMA",
    "EXPECTED_CANONICAL_GIT_SHA",
    "FULL_SEQUENCE_CHUNK_FRAMES",
    "FULL_SEQUENCE_GLOBAL_STEP",
    "FULL_SEQUENCE_NUM_CHUNKS",
    "FULL_SEQUENCE_OBJECTIVE_MODE",
    "MODE_OFFICIAL_MAIN",
    "MODE_TRAINED_MAIN",
    "MODE_TRAINED_MCP1",
    "OFFICIAL_SELF_FORCING_CHECKPOINT_SHA256",
    "RAW_DEPLOYMENT_SCHEDULE",
    "RUNTIME_MEASUREMENT_STATUS",
    "TRAINING_CHECKPOINT_GIT_SHA",
    "DeploymentCheckpointRecord",
    "DeploymentResult",
    "DeploymentRuntime",
    "assert_common_input_fingerprints",
    "assert_mode_inputs_match_common",
    "assert_rng_plan_fingerprints",
    "build_absolute_chunk_rng_plan",
    "build_common_inputs_record",
    "build_comparison_report",
    "build_eval_manifest",
    "build_mcp1_execution_plan",
    "checkpoint_sidecar_paths",
    "compare_latents",
    "compare_pixel_frames",
    "count_mcp_tensors",
    "current_git_head",
    "file_sha256",
    "full_sequence_role_map",
    "latent_chunk_to_pixel_frame_ranges",
    "load_full_sequence_checkpoint_record",
    "load_official_checkpoint_record",
    "resolve_deployment_schedule",
    "rng_plan_fingerprint",
    "role_aware_latent_metrics",
    "run_main_only_deployment",
    "run_mcp1_deployment",
    "sample_plan_validation_identities",
    "selected_validation_position",
    "validate_eval_artifact_identity",
    "validate_full_sequence_checkpoint_payload",
    "validate_full_sequence_checkpoint_sidecars",
    "validate_repo_preflight_facts",
    "validate_trained_main_trace",
    "validate_trained_mcp1_trace",
    "write_mode_outputs",
]
