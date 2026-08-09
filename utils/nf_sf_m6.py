from __future__ import annotations

import copy
import hashlib
import json
import math
import random
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from utils.checkpoint import extract_generator_state_dict, is_mcp_state_key
from utils.nf_sf_m3 import (
    M3_CHECKPOINT_FORMAT,
    M3_CHUNK_FRAMES,
    M3_REFERENCE_CHECKPOINT_SHA256,
    atomic_json_write,
    atomic_torch_save,
    file_sha256,
    load_m3_checkpoint,
    load_m3_teacher_sample,
    tensor_sha256,
    tensor_summary,
    validate_git_sha,
)
from utils.nf_sf_m5_formal import resolve_m5_formal_stage_contract
from utils.nf_sf_m5_validation import M5_STREAMING_VALIDATION_SCHEMA
from utils.nf_sf_tensors import (
    DEFAULT_NUM_TRAIN_TIMESTEPS,
    DEFAULT_S_MAIN,
    flow_match_shift_timesteps,
)

M6_ORACLE_SCHEMA = "nf_sf_m6_oracle_v1"
M6_COMPARISON_SCHEMA = "nf_sf_m6_oracle_comparison_v1"
M6_PIXEL_COMPARISON_SCHEMA = "nf_sf_m6_pixel_comparison_v1"
M6_COMMON_INPUTS_SCHEMA = "nf_sf_m6_common_inputs_v1"
M6_LOCKED_RAW_SCHEDULE = (1000.0, 750.0, 500.0, 250.0)
M6_RNG_DRAW_CONTRACT_VERSION = "m6_ab_teacher_compatible_rng_draw_contract_v2"
M6_ORACLE_C_MAIN_QUALITY_CONTRACT_VERSION = (
    "m6_oracle_c_main_quality_contract_v1"
)
M6_TEACHER_COMPATIBILITY_PURPOSE = "teacher_exit_flag_randint_compatibility"
M6_TEACHER_COMPATIBILITY_REASON = (
    "match formal Teacher generate_and_sync_list(last_step_only=True) RNG consumption"
)
M6_WAN_FRAME_SEQ_LENGTH = 1560
M6_CHECKPOINT_OFFICIAL = "official_reference"
M6_CHECKPOINT_FORMAL_STEP0 = "formal_step0"
M6_CHECKPOINT_FORMAL_STEP500 = "formal_step500"
M5_FORMAL_TRAINER_SCHEMA = "nf_sf_m5_formal_trainer_v1"

OracleKind = Literal["A", "B", "C"]


@dataclass(frozen=True)
class M6ResolvedSchedule:
    raw_schedule: tuple[float, ...]
    main_warped_schedule: tuple[float, ...]
    main_shift: float
    num_train_timesteps: int
    mcp_enabled: bool
    mcp_warped_schedule: tuple[float, ...] | None

    def to_json(self) -> dict[str, Any]:
        return {
            "raw_schedule": list(self.raw_schedule),
            "main_warped_schedule": list(self.main_warped_schedule),
            "main_shift": float(self.main_shift),
            "num_train_timesteps": int(self.num_train_timesteps),
            "warp_formula": "utils.nf_sf_tensors.flow_match_shift_timesteps",
            "mcp_enabled": bool(self.mcp_enabled),
            "mcp_warped_schedule": (
                None
                if self.mcp_warped_schedule is None
                else list(self.mcp_warped_schedule)
            ),
        }


@dataclass(frozen=True)
class M6CheckpointRecord:
    path: str
    sha256: str
    checkpoint_type: str
    load_mode: str
    generator_state_dict: Mapping[str, Any]
    global_step: int | None = None
    mcp_tensor_count: int = 0
    payload_format: str | None = None
    formal_metadata: Mapping[str, Any] | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "type": self.checkpoint_type,
            "load_mode": self.load_mode,
            "global_step": self.global_step,
            "mcp_tensor_count": int(self.mcp_tensor_count),
            "payload_format": self.payload_format,
            "formal_metadata": (
                None if self.formal_metadata is None else dict(self.formal_metadata)
            ),
        }


@dataclass(frozen=True)
class M6OracleRuntime:
    generator: Any
    scheduler: Any
    kv_cache: list[dict[str, Any]]
    crossattn_cache: list[dict[str, Any]]
    frame_seq_length: int
    num_frame_per_block: int = M3_CHUNK_FRAMES
    context_noise: int = 0


@dataclass(frozen=True)
class M6OracleResult:
    latent: torch.Tensor
    trace: dict[str, Any]
    summary: dict[str, Any]


@dataclass(frozen=True)
class M6OracleAArtifactRecord:
    trace: Mapping[str, Any]
    summary: Mapping[str, Any]
    latent_payload: Mapping[str, Any]
    latent: torch.Tensor
    common_inputs: Mapping[str, Any]
    common_inputs_fingerprint_sha256: str
    latent_sha256: str


@dataclass(frozen=True)
class M6OracleBArtifactRecord:
    artifact_dir: str
    trace: Mapping[str, Any]
    summary: Mapping[str, Any]
    latent_payload: Mapping[str, Any]
    latent: torch.Tensor
    common_inputs: Mapping[str, Any]
    common_inputs_fingerprint_sha256: str
    latent_sha256: str
    artifact_hashes: Mapping[str, str]
    checkpoint: Mapping[str, Any]


class M6KVSnapshot:
    def __init__(self, layers: list[dict[str, Any]]) -> None:
        self._states = layers

    @classmethod
    def capture(cls, kv_cache: Sequence[Mapping[str, Any]]) -> M6KVSnapshot:
        layers = []
        for layer in kv_cache:
            global_end = _cache_index_value(layer, "global_end_index")
            local_end = _cache_index_value(layer, "local_end_index")
            state = {
                "global_end_index": global_end,
                "local_end_index": local_end,
            }
            for name in ("k", "v"):
                tensor = layer.get(name)
                if torch.is_tensor(tensor):
                    state[name] = tensor[:, :local_end].detach().clone()
            layers.append(state)
        return cls(layers)

    def restore(self, kv_cache: Sequence[Mapping[str, Any]]) -> bool:
        if len(kv_cache) != len(self._states):
            raise RuntimeError("KV snapshot layer count differs from cache")
        restored = True
        for layer, state in zip(kv_cache, self._states):
            local_end = int(state["local_end_index"])
            for name in ("k", "v"):
                saved = state.get(name)
                current = layer.get(name)
                if torch.is_tensor(saved) and torch.is_tensor(current):
                    current[:, :local_end].copy_(saved.to(device=current.device))
                    restored = restored and bool(
                        torch.equal(current[:, :local_end], saved.to(device=current.device))
                    )
            _set_cache_index(layer, "global_end_index", int(state["global_end_index"]))
            _set_cache_index(layer, "local_end_index", local_end)
        return restored

    def visible_data_matches(self, kv_cache: Sequence[Mapping[str, Any]]) -> bool:
        if len(kv_cache) != len(self._states):
            return False
        for layer, state in zip(kv_cache, self._states):
            local_end = int(state["local_end_index"])
            for name in ("k", "v"):
                saved = state.get(name)
                current = layer.get(name)
                if torch.is_tensor(saved) and torch.is_tensor(current) and not torch.equal(
                    current[:, :local_end].detach().cpu(),
                    saved.detach().cpu(),
                ):
                    return False
        return True


def current_git_head() -> str:
    return validate_git_sha(
        subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip(),
        name="current_git_sha",
    )


def resolve_m6_schedule(
    config: Any,
    *,
    main_shift: float = DEFAULT_S_MAIN,
    num_train_timesteps: int = DEFAULT_NUM_TRAIN_TIMESTEPS,
) -> M6ResolvedSchedule:
    raw_schedule = tuple(_raw_schedule_from_config(config))
    if raw_schedule != M6_LOCKED_RAW_SCHEDULE:
        raise ValueError(
            "NF-SF M6.0 A/B/C requires raw denoising schedule "
            f"{list(M6_LOCKED_RAW_SCHEDULE)}, got {list(raw_schedule)}"
        )
    if len(raw_schedule) <= 1:
        raise ValueError("NF-SF M6 schedule must not degenerate to one step")

    configured_shift = _main_shift_from_config(config, default=main_shift)
    if not math.isclose(float(configured_shift), float(main_shift), rel_tol=0.0, abs_tol=0.0):
        raise ValueError(
            f"NF-SF M6.0 A/B/C requires main timestep shift {main_shift}, "
            f"got {configured_shift}"
        )

    warped = flow_match_shift_timesteps(
        torch.tensor(raw_schedule, dtype=torch.float32),
        shift=float(main_shift),
        num_train_timesteps=int(num_train_timesteps),
    )
    return M6ResolvedSchedule(
        raw_schedule=raw_schedule,
        main_warped_schedule=tuple(float(value) for value in warped.tolist()),
        main_shift=float(main_shift),
        num_train_timesteps=int(num_train_timesteps),
        mcp_enabled=False,
        mcp_warped_schedule=None,
    )


def select_m6_teacher_sample(
    *,
    teacher_manifest: Path | str,
    dataset_root: Path | str | None,
    sample_index: int | None = None,
    sample_id: str | None = None,
    split: str | None = None,
    split_index: int | None = None,
    reference_checkpoint_path: Path | str | None = None,
) -> Any:
    return load_m3_teacher_sample(
        manifest_path=teacher_manifest,
        dataset_root=dataset_root,
        sample_index=sample_index,
        sample_id=sample_id,
        split=split,
        split_index=split_index,
        reference_checkpoint_path=reference_checkpoint_path,
        expected_reference_sha256=M3_REFERENCE_CHECKPOINT_SHA256,
    )


def build_common_inputs(
    *,
    teacher_metadata: Mapping[str, Any],
    teacher_payload: Mapping[str, Any],
    source_noise: torch.Tensor,
    conditioning_summary: Mapping[str, Any],
    schedule: M6ResolvedSchedule,
    rollout_seed: int,
    context_noise: int,
    chunk_frames: int,
    frame_seq_length: int,
    device_runtime_contract: Mapping[str, Any],
    resolved_config_canonical_sha256: str,
    runtime_git_sha: str,
) -> tuple[dict[str, Any], str]:
    source_summary = tensor_json_summary(source_noise)
    frame_count = int(source_summary["shape"][1])
    if frame_count % int(chunk_frames) != 0:
        raise ValueError("source_noise frame count must be chunk-aligned")
    num_chunks = frame_count // int(chunk_frames)
    rng_compatibility_contract = {
        "operation": "torch.randint",
        "purpose": M6_TEACHER_COMPATIBILITY_PURPOSE,
        "low": 0,
        "high": len(schedule.main_warped_schedule),
        "size": [num_chunks],
        "dtype": str(torch.long),
        "values_discarded": True,
    }
    common_inputs = {
        "schema": M6_COMMON_INPUTS_SCHEMA,
        "teacher_identity": teacher_identity_json(teacher_metadata),
        "teacher_payload_sha256": str(teacher_metadata.get("latent_file_sha256")),
        "source_noise_sha256": source_summary["sha256"],
        "prompt_sha256": str(teacher_payload["prompt_sha256"]),
        "conditioning_sha256": str(conditioning_summary["sha256"]),
        "raw_schedule": list(schedule.raw_schedule),
        "main_warped_schedule": list(schedule.main_warped_schedule),
        "rollout_seed": int(rollout_seed),
        "rng_draw_contract_version": M6_RNG_DRAW_CONTRACT_VERSION,
        "rng_compatibility_contract": rng_compatibility_contract,
        "context_noise": int(context_noise),
        "latent_shape": source_summary["shape"],
        "latent_dtype": source_summary["dtype"],
        "chunk_frames": int(chunk_frames),
        "frame_seq_length": int(frame_seq_length),
        "device_runtime_contract": dict(device_runtime_contract),
        "resolved_config_canonical_sha256": _require_sha256(
            resolved_config_canonical_sha256,
            "resolved_config_canonical_sha256",
        ),
        "runtime_git_sha": validate_git_sha(str(runtime_git_sha), name="runtime_git_sha"),
    }
    validate_json_payload(common_inputs)
    return common_inputs, canonical_json_sha256(common_inputs)


def canonical_json_sha256(value: Mapping[str, Any]) -> str:
    validate_json_payload(value)
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def oracle_c_main_quality_contract() -> dict[str, Any]:
    contract = {
        "version": M6_ORACLE_C_MAIN_QUALITY_CONTRACT_VERSION,
        "oracle": "C",
        "checkpoint": "formal Stage A global_step=500",
        "runtime": "Main-only four-step rollout with MCP disabled",
        "automatic_quality_threshold": None,
        "evidence": {
            "latent_space": [
                "C step500 output latent vs strict PASS B step0 output latent",
                "shape",
                "dtype",
                "exact_equality",
                "max_abs_diff",
                "mean_abs_diff",
                "mse",
                "per_chunk",
                "sha256",
            ],
            "video_space": [
                "decode B and C latents with the same VAE runtime/config/device/dtype",
                "save step0_reference.mp4",
                "save step500_main.mp4",
                "record video SHA256",
                "record MAE/MSE/PSNR and per-frame MAE/MSE",
            ],
            "manual_review": (
                "main_quality_pass may be set only after reviewing numeric "
                "evidence and both videos"
            ),
        },
        "fail_criteria": [
            "subject, scene recognizability, or prompt-consistent content is newly lost",
            "new blur or detail collapse is obvious",
            "new temporal flicker or discontinuity is obvious",
            "progressive/autoregressive collapse or late-chunk instability is obvious",
        ],
        "initial_main_quality_pass": None,
        "initial_status_when_protocol_passes": "REPORT_ONLY",
        "initial_review_status": "PENDING",
    }
    validate_json_payload(contract)
    return contract


def load_oracle_checkpoint(
    *,
    path: Path | str,
    oracle_kind: OracleKind,
    expected_official_sha256: str | None = None,
) -> M6CheckpointRecord:
    path = Path(path)
    checkpoint_sha = file_sha256(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if oracle_kind == "A":
        return _official_checkpoint_record(
            path=path,
            payload=payload,
            checkpoint_sha=checkpoint_sha,
            expected_sha256=expected_official_sha256,
        )
    if oracle_kind == "B":
        return _formal_step0_checkpoint_record(
            path=path,
            payload=payload,
            checkpoint_sha=checkpoint_sha,
        )
    if oracle_kind == "C":
        return _formal_step500_checkpoint_record(
            path=path,
            payload=payload,
            checkpoint_sha=checkpoint_sha,
        )
    raise ValueError(f"unsupported oracle_kind: {oracle_kind!r}")


def run_main_only_oracle(
    *,
    oracle_kind: OracleKind,
    runtime: M6OracleRuntime,
    source_noise: torch.Tensor,
    teacher_payload: Mapping[str, Any],
    teacher_metadata: Mapping[str, Any],
    conditional_dict: Mapping[str, Any],
    schedule: M6ResolvedSchedule,
    checkpoint: M6CheckpointRecord,
    git_sha: str,
    resolved_config_canonical_sha256: str,
    device_runtime_contract: Mapping[str, Any],
    expected_common_inputs: Mapping[str, Any] | None = None,
    tolerance: float | None = None,
) -> M6OracleResult:
    if oracle_kind not in ("A", "B", "C"):
        raise ValueError("M6.0 supports only Oracle A, B, and C")
    if schedule.mcp_enabled or schedule.mcp_warped_schedule is not None:
        raise ValueError("M6.0 A/B/C must run with MCP disabled")
    validate_schedule_matches_teacher(schedule, teacher_payload)
    _validate_source_noise(source_noise, teacher_payload=teacher_payload)
    _validate_runtime(runtime)

    batch_size, num_frames = int(source_noise.shape[0]), int(source_noise.shape[1])
    chunk_frames = int(runtime.num_frame_per_block)
    num_chunks = num_frames // chunk_frames
    rollout_seed = int(teacher_payload["rollout_seed"])
    noise_seed = int(teacher_payload["noise_seed"])
    reset_rollout_rng(rollout_seed, source_noise.device)
    post_reset_rng_hash = global_rng_state_hash(source_noise.device)
    rng_draws: list[dict[str, Any]] = [
        consume_teacher_exit_flag_rng_compatibility_draw(
            num_chunks=num_chunks,
            num_denoising_steps=len(schedule.main_warped_schedule),
            device=source_noise.device,
            draw_order=0,
        )
    ]
    pre_solver_rng_hash = global_rng_state_hash(source_noise.device)
    output = torch.empty_like(source_noise)
    chunks = []

    runtime.generator.eval()
    with torch.no_grad():
        cursor = 0
        for chunk_index in range(num_chunks):
            start_frame = cursor
            current = source_noise[:, start_frame:start_frame + chunk_frames].detach().clone()
            step_records = []
            last_rollback_boundary = None

            for step_index, warped_timestep in enumerate(schedule.main_warped_schedule):
                raw_timestep = schedule.raw_schedule[step_index]
                forward_input = current.detach()
                step_snapshot = M6KVSnapshot.capture(runtime.kv_cache)
                kv_before = kv_boundary_summary(runtime.kv_cache)
                _require_kv_boundary_consistent(kv_before, label="before denoising forward")
                timestep = torch.full(
                    current.shape[:2],
                    float(warped_timestep),
                    device=current.device,
                    dtype=torch.float32,
                )
                generator_outputs = runtime.generator(
                    noisy_image_or_video=current,
                    conditional_dict=dict(conditional_dict),
                    timestep=timestep,
                    kv_cache=runtime.kv_cache,
                    crossattn_cache=runtime.crossattn_cache,
                    current_start=start_frame * int(runtime.frame_seq_length),
                )
                flow_pred, clean_pred = _unpack_main_outputs(generator_outputs)
                _ensure_finite_tensor(flow_pred, name="main_flow_pred")
                _ensure_finite_tensor(clean_pred, name="main_clean_pred")
                kv_temp = kv_boundary_summary(runtime.kv_cache)
                _require_kv_boundary_consistent(kv_temp, label="temporary denoising forward")
                restored_data = step_snapshot.restore(runtime.kv_cache)
                visible_data_restored = step_snapshot.visible_data_matches(runtime.kv_cache)
                if not restored_data or not visible_data_restored:
                    raise RuntimeError("KV visible data restore failed")
                kv_rollback = kv_boundary_summary(runtime.kv_cache)
                _require_kv_boundary_consistent(kv_rollback, label="rollback denoising forward")
                _require_kv_rollback_matches(kv_before, kv_rollback)
                last_rollback_boundary = kv_rollback

                transition_record = None
                if step_index < len(schedule.main_warped_schedule) - 1:
                    next_timestep = float(schedule.main_warped_schedule[step_index + 1])
                    transition_noise, transition_draw = randn_like_with_trace(
                        clean_pred.flatten(0, 1),
                        device=source_noise.device,
                        purpose="transition_re_noise",
                        draw_order=len(rng_draws),
                        chunk_index=chunk_index,
                        solver_step_index=step_index,
                    )
                    rng_draws.append(transition_draw)
                    next_timestep_tensor = torch.full(
                        (batch_size * chunk_frames,),
                        next_timestep,
                        device=current.device,
                        dtype=torch.float32,
                    )
                    current = runtime.scheduler.add_noise(
                        clean_pred.flatten(0, 1),
                        transition_noise,
                        next_timestep_tensor,
                    ).unflatten(0, clean_pred.shape[:2])
                    _ensure_finite_tensor(current, name="re_noised_current")
                    transition_record = {
                        "next_warped_timestep": next_timestep,
                        "transition_noise": transition_draw,
                        "re_noised_tensor": tensor_json_summary(current),
                    }
                else:
                    output[:, start_frame:start_frame + chunk_frames] = clean_pred

                step_records.append(
                    {
                        "raw_index": int(step_index),
                        "raw_timestep": _json_number(raw_timestep),
                        "warped_timestep": float(warped_timestep),
                        "input_tensor": tensor_json_summary(forward_input),
                        "flow_tensor": tensor_json_summary(flow_pred),
                        "output_x0_tensor": tensor_json_summary(clean_pred),
                        "kv": {
                            "before": kv_before,
                            "temporary_after_forward": kv_temp,
                            "rollback_after_forward": kv_rollback,
                            "visible_data_restored": bool(visible_data_restored),
                        },
                        "transition": transition_record,
                    }
                )

            clean_chunk = output[:, start_frame:start_frame + chunk_frames]
            recache_snapshot_before = kv_boundary_summary(runtime.kv_cache)
            _require_kv_boundary_consistent(recache_snapshot_before, label="clean recache before")
            if last_rollback_boundary is None:
                raise RuntimeError("chunk has no denoising rollback boundary")
            _require_kv_rollback_matches(last_rollback_boundary, recache_snapshot_before)
            context_timestep = torch.full(
                clean_chunk.shape[:2],
                int(runtime.context_noise),
                device=clean_chunk.device,
                dtype=torch.int64,
            )
            context_noise, context_draw = randn_like_with_trace(
                clean_chunk.flatten(0, 1),
                device=source_noise.device,
                purpose="context_clean_recache_noise",
                draw_order=len(rng_draws),
                chunk_index=chunk_index,
                solver_step_index=None,
            )
            rng_draws.append(context_draw)
            context_latent = runtime.scheduler.add_noise(
                clean_chunk.flatten(0, 1),
                context_noise,
                context_timestep.flatten(0, 1),
            ).unflatten(0, clean_chunk.shape[:2])
            runtime.generator(
                noisy_image_or_video=context_latent,
                conditional_dict=dict(conditional_dict),
                timestep=context_timestep,
                kv_cache=runtime.kv_cache,
                crossattn_cache=runtime.crossattn_cache,
                current_start=start_frame * int(runtime.frame_seq_length),
            )
            recache_snapshot_after = kv_boundary_summary(runtime.kv_cache)
            _require_clean_recache_transition(
                recache_snapshot_before,
                recache_snapshot_after,
                start_frame=start_frame,
                chunk_frames=chunk_frames,
                frame_seq_length=int(runtime.frame_seq_length),
            )
            cursor_before = cursor
            cursor += chunk_frames
            _require_commit_contract(
                cursor_before=cursor_before,
                cursor_after=cursor,
                start_frame=start_frame,
                chunk_frames=chunk_frames,
            )
            chunks.append(
                {
                    "chunk_index": int(chunk_index),
                    "start_frame": int(start_frame),
                    "num_frames": int(chunk_frames),
                    "solver_steps": step_records,
                    "clean_recache": {
                        "context_noise": int(runtime.context_noise),
                        "before": recache_snapshot_before,
                        "after": recache_snapshot_after,
                        "rng_draw": context_draw,
                        "context_latent": tensor_json_summary(context_latent),
                    },
                    "commit": {
                        "main_only": True,
                        "next_commit": None,
                        "final_committed_chunk_start": int(start_frame),
                        "cursor_before": int(cursor_before),
                        "cursor_after": int(cursor),
                    },
                }
            )

    mcp_call_count = _runtime_mcp_call_count(runtime.generator)
    if mcp_call_count != 0:
        raise RuntimeError(f"M6.0 A/B/C requires mcp_call_count=0, actual={mcp_call_count}")
    _ensure_finite_tensor(output, name="output_latent")
    conditioning_summary = conditioning_json_summary(conditional_dict)
    common_inputs, common_fingerprint = build_common_inputs(
        teacher_metadata=teacher_metadata,
        teacher_payload=teacher_payload,
        source_noise=source_noise,
        conditioning_summary=conditioning_summary,
        schedule=schedule,
        rollout_seed=rollout_seed,
        context_noise=int(runtime.context_noise),
        chunk_frames=chunk_frames,
        frame_seq_length=int(runtime.frame_seq_length),
        device_runtime_contract=device_runtime_contract,
        resolved_config_canonical_sha256=resolved_config_canonical_sha256,
        runtime_git_sha=git_sha,
    )
    if expected_common_inputs is not None:
        expected_payload = dict(expected_common_inputs)
        expected_fingerprint = canonical_json_sha256(expected_payload)
        if expected_payload != common_inputs or expected_fingerprint != common_fingerprint:
            raise RuntimeError("precomputed common inputs differ from rollout common inputs")
    target_comparison = compare_latents(
        output.detach().cpu(),
        teacher_payload["target_latent"].detach().cpu(),
        chunk_frames=chunk_frames,
        tolerance=tolerance,
    )
    oracle_c_quality_contract = (
        oracle_c_main_quality_contract() if oracle_kind == "C" else None
    )
    trace = {
        "schema": M6_ORACLE_SCHEMA,
        "oracle_kind": oracle_kind,
        "git_sha": git_sha,
        "checkpoint": checkpoint.to_json(),
        "teacher_identity": teacher_identity_json(teacher_metadata),
        "teacher_payload_hash": str(teacher_metadata.get("latent_file_sha256")),
        "source_noise": tensor_json_summary(source_noise),
        "teacher_payload_noise_seed": noise_seed,
        "teacher_payload_rollout_seed": rollout_seed,
        "prompt": {
            "text": str(teacher_payload["prompt"]),
            "prompt_sha256": str(teacher_payload["prompt_sha256"]),
        },
        "prompt_conditioning": conditioning_summary,
        "schedule": schedule.to_json(),
        "mcp_enabled": False,
        "mcp_warped_schedule": None,
        "mcp_call_count": int(mcp_call_count),
        "rng": {
            "noise_seed": noise_seed,
            "rollout_seed": rollout_seed,
            "initial_global_rng_state_hash": post_reset_rng_hash,
            "initial_global_rng_state_hash_semantics": (
                "post reset, before teacher compatibility draw"
            ),
            "post_reset_global_rng_state_hash": post_reset_rng_hash,
            "pre_solver_global_rng_state_hash": pre_solver_rng_hash,
            "draw_contract": {
                "source_noise": "exact tensor loaded from teacher payload",
                "rollout_rng": (
                    "global torch RNG reset to teacher payload rollout_seed; "
                    "consume one teacher-compatible exit-flag torch.randint draw; "
                    "then start solver transition/context draws"
                ),
                M6_TEACHER_COMPATIBILITY_PURPOSE: (
                    "one torch.randint(low=0, high=num_denoising_steps, "
                    "size=[num_chunks], dtype=torch.long, device=source_noise.device) "
                    "matching formal Teacher generate_and_sync_list(last_step_only=True); "
                    "values discarded"
                ),
                "transition_re_noise": (
                    "one torch.randn_like(flattened clean x0) after each "
                    "non-final forward"
                ),
                "context_clean_recache_noise": (
                    "one torch.randn_like(flattened clean chunk) before clean "
                    "recache for each chunk"
                ),
            },
            "draws": rng_draws,
        },
        "chunks": chunks,
        "finite_checks": {
            "output_latent": True,
            "all_solver_outputs": True,
        },
        "oracle_c_main_quality_contract": oracle_c_quality_contract,
        "target_latent_comparison": target_comparison,
        "artifact_hashes": {
            "output_latent_tensor_sha256": tensor_sha256(output.detach().cpu()),
        },
        "common_inputs": common_inputs,
        "common_inputs_fingerprint_sha256": common_fingerprint,
    }
    summary = {
        "schema": M6_ORACLE_SCHEMA,
        "oracle_kind": oracle_kind,
        "git_sha": git_sha,
        "checkpoint": checkpoint.to_json(),
        "teacher_identity": trace["teacher_identity"],
        "source_noise_sha256": trace["source_noise"]["sha256"],
        "prompt_conditioning_sha256": conditioning_summary["sha256"],
        "raw_schedule": list(schedule.raw_schedule),
        "main_warped_schedule": list(schedule.main_warped_schedule),
        "mcp_enabled": False,
        "mcp_call_count": int(mcp_call_count),
        "oracle_c_main_quality_contract": oracle_c_quality_contract,
        "target_latent_comparison": target_comparison,
        "output_latent": tensor_json_summary(output),
        "common_inputs": common_inputs,
        "common_inputs_fingerprint_sha256": common_fingerprint,
    }
    _apply_oracle_gate_fields(
        trace,
        summary,
        oracle_a_comparison=None,
        oracle_b_comparison=None,
        oracle_b_artifact=None,
    )
    validate_json_payload(trace)
    validate_json_payload(summary)
    return M6OracleResult(
        latent=output.detach().cpu(),
        trace=trace,
        summary=summary,
    )


def write_oracle_artifacts(
    *,
    output_dir: Path | str,
    resolved_config: Mapping[str, Any],
    result: M6OracleResult,
    oracle_comparison: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    output_dir = Path(output_dir)
    prepare_output_dir(output_dir)
    resolved_payload = dict(resolved_config)
    if result.trace.get("common_inputs") != result.summary.get("common_inputs"):
        raise RuntimeError("trace and summary common_inputs differ")
    if (
        result.trace.get("common_inputs_fingerprint_sha256")
        != result.summary.get("common_inputs_fingerprint_sha256")
    ):
        raise RuntimeError("trace and summary common input fingerprints differ")
    validate_json_payload(resolved_payload)
    atomic_json_write(resolved_payload, output_dir / "resolved_config.json")
    atomic_json_write(result.trace, output_dir / "oracle_trace.json")
    atomic_json_write(result.summary, output_dir / "oracle_summary.json")
    atomic_torch_save(
        {
            "schema": M6_ORACLE_SCHEMA,
            "latent": result.latent,
            "latent_sha256": tensor_sha256(result.latent),
            "oracle_kind": result.trace["oracle_kind"],
            "common_inputs": result.trace["common_inputs"],
            "common_inputs_fingerprint_sha256": result.trace[
                "common_inputs_fingerprint_sha256"
            ],
        },
        output_dir / "output_latent.pt",
    )
    hashes = {
        "resolved_config_json_sha256": file_sha256(output_dir / "resolved_config.json"),
        "oracle_trace_json_sha256": file_sha256(output_dir / "oracle_trace.json"),
        "oracle_summary_json_sha256": file_sha256(output_dir / "oracle_summary.json"),
        "output_latent_pt_sha256": file_sha256(output_dir / "output_latent.pt"),
    }
    if oracle_comparison is not None:
        comparison_payload = dict(oracle_comparison)
        validate_json_payload(comparison_payload)
        atomic_json_write(comparison_payload, output_dir / "oracle_comparison.json")
        hashes["oracle_comparison_json_sha256"] = file_sha256(
            output_dir / "oracle_comparison.json"
        )
    return hashes


def load_output_latent_artifact(path: Path | str) -> torch.Tensor:
    payload = load_output_latent_payload(path)
    return payload["latent"].detach().cpu()


def load_output_latent_payload(path: Path | str) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise TypeError("output_latent.pt must contain a mapping")
    if payload.get("schema") != M6_ORACLE_SCHEMA:
        raise RuntimeError("output_latent.pt schema mismatch")
    latent = payload.get("latent")
    if not torch.is_tensor(latent):
        raise TypeError("output_latent.pt missing latent tensor")
    expected = payload.get("latent_sha256")
    actual = tensor_sha256(latent)
    if expected is not None and str(expected) != actual:
        raise RuntimeError("output_latent.pt latent SHA256 mismatch")
    result = dict(payload)
    result["latent"] = latent.detach().cpu()
    result["latent_sha256"] = actual
    return result


def validate_oracle_a_artifact_dir(
    oracle_a_dir: Path | str,
    *,
    expected_common_inputs_fingerprint_sha256: str,
) -> M6OracleAArtifactRecord:
    record = _validate_oracle_artifact_dir(
        oracle_a_dir,
        expected_common_inputs_fingerprint_sha256=(
            expected_common_inputs_fingerprint_sha256
        ),
        expected_oracle_kind="A",
        expected_checkpoint_type=M6_CHECKPOINT_OFFICIAL,
    )
    return M6OracleAArtifactRecord(
        trace=record["trace"],
        summary=record["summary"],
        latent_payload=record["latent_payload"],
        latent=record["latent"],
        common_inputs=record["common_inputs"],
        common_inputs_fingerprint_sha256=record[
            "common_inputs_fingerprint_sha256"
        ],
        latent_sha256=record["latent_sha256"],
    )


def validate_oracle_b_artifact_dir(
    oracle_b_dir: Path | str,
    *,
    expected_common_inputs_fingerprint_sha256: str,
) -> M6OracleBArtifactRecord:
    record = _validate_oracle_artifact_dir(
        oracle_b_dir,
        expected_common_inputs_fingerprint_sha256=(
            expected_common_inputs_fingerprint_sha256
        ),
        expected_oracle_kind="B",
        expected_checkpoint_type=M6_CHECKPOINT_FORMAL_STEP0,
    )
    checkpoint = record["checkpoint"]
    if checkpoint.get("global_step") != 0:
        raise RuntimeError("Oracle B artifact checkpoint global_step must be 0")
    return M6OracleBArtifactRecord(
        artifact_dir=record["artifact_dir"],
        trace=record["trace"],
        summary=record["summary"],
        latent_payload=record["latent_payload"],
        latent=record["latent"],
        common_inputs=record["common_inputs"],
        common_inputs_fingerprint_sha256=record[
            "common_inputs_fingerprint_sha256"
        ],
        latent_sha256=record["latent_sha256"],
        artifact_hashes=record["artifact_hashes"],
        checkpoint=checkpoint,
    )


def oracle_b_artifact_identity(record: M6OracleBArtifactRecord) -> dict[str, Any]:
    identity = {
        "oracle_kind": "B",
        "artifact_dir": record.artifact_dir,
        "status": str(record.summary.get("status")),
        "protocol_pass": record.summary.get("protocol_pass"),
        "oracle_gate_pass": record.summary.get("oracle_gate_pass"),
        "checkpoint": dict(record.checkpoint),
        "common_inputs_fingerprint_sha256": record.common_inputs_fingerprint_sha256,
        "latent_sha256": record.latent_sha256,
        "artifact_hashes": dict(record.artifact_hashes),
    }
    validate_json_payload(identity)
    return identity


def _validate_oracle_artifact_dir(
    oracle_dir: Path | str,
    *,
    expected_common_inputs_fingerprint_sha256: str,
    expected_oracle_kind: str,
    expected_checkpoint_type: str,
) -> dict[str, Any]:
    oracle_dir = Path(oracle_dir)
    label = f"Oracle {expected_oracle_kind}"
    trace_path = oracle_dir / "oracle_trace.json"
    summary_path = oracle_dir / "oracle_summary.json"
    latent_path = oracle_dir / "output_latent.pt"
    for path in (trace_path, summary_path, latent_path):
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(f"{label} artifact missing or empty: {path}")

    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    latent_payload = load_output_latent_payload(latent_path)
    for name, payload in (
        ("oracle_trace.json", trace),
        ("oracle_summary.json", summary),
        ("output_latent.pt", latent_payload),
    ):
        if payload.get("schema") != M6_ORACLE_SCHEMA:
            raise RuntimeError(f"{name} schema mismatch")
        if payload.get("oracle_kind") != expected_oracle_kind:
            raise RuntimeError(f"{name} is not an {label} artifact")

    if trace.get("status") != "PASS" or summary.get("status") != "PASS":
        raise RuntimeError(f"{label} artifact status must be PASS")
    if trace.get("protocol_pass") is not True or summary.get("protocol_pass") is not True:
        raise RuntimeError(f"{label} artifact protocol_pass must be True")
    if trace.get("oracle_gate_pass") is not True or summary.get("oracle_gate_pass") is not True:
        raise RuntimeError(f"{label} artifact oracle_gate_pass must be True")
    checkpoint = trace.get("checkpoint")
    if not isinstance(checkpoint, Mapping) or checkpoint.get("type") != expected_checkpoint_type:
        raise RuntimeError(
            f"{label} artifact checkpoint type must be {expected_checkpoint_type}"
        )

    expected_fingerprint = _require_sha256(
        expected_common_inputs_fingerprint_sha256,
        "expected_common_inputs_fingerprint_sha256",
    )
    fingerprints = [
        trace.get("common_inputs_fingerprint_sha256"),
        summary.get("common_inputs_fingerprint_sha256"),
        latent_payload.get("common_inputs_fingerprint_sha256"),
    ]
    if any(item != expected_fingerprint for item in fingerprints):
        raise RuntimeError(f"{label} common inputs fingerprint mismatch")
    common_inputs = trace.get("common_inputs")
    if (
        not isinstance(common_inputs, Mapping)
        or summary.get("common_inputs") != common_inputs
        or latent_payload.get("common_inputs") != common_inputs
    ):
        raise RuntimeError(f"{label} common inputs differ across artifacts")
    if canonical_json_sha256(dict(common_inputs)) != expected_fingerprint:
        raise RuntimeError(
            f"{label} common inputs fingerprint does not match payload"
        )

    latent_sha = str(latent_payload["latent_sha256"])
    trace_latent_sha = (
        trace.get("artifact_hashes", {}).get("output_latent_tensor_sha256")
        if isinstance(trace.get("artifact_hashes"), Mapping)
        else None
    )
    summary_output = summary.get("output_latent")
    summary_latent_sha = (
        summary_output.get("sha256")
        if isinstance(summary_output, Mapping)
        else None
    )
    if trace_latent_sha != latent_sha or summary_latent_sha != latent_sha:
        raise RuntimeError(f"{label} latent SHA differs across artifacts")
    if tensor_sha256(latent_payload["latent"]) != latent_sha:
        raise RuntimeError(f"{label} actual latent tensor SHA mismatch")
    return {
        "artifact_dir": str(oracle_dir.resolve()),
        "trace": trace,
        "summary": summary,
        "latent_payload": latent_payload,
        "latent": latent_payload["latent"],
        "common_inputs": common_inputs,
        "common_inputs_fingerprint_sha256": expected_fingerprint,
        "latent_sha256": latent_sha,
        "checkpoint": checkpoint,
        "artifact_hashes": {
            "oracle_trace_json_sha256": file_sha256(trace_path),
            "oracle_summary_json_sha256": file_sha256(summary_path),
            "output_latent_pt_sha256": file_sha256(latent_path),
        },
    }


def validate_schedule_matches_teacher(
    schedule: M6ResolvedSchedule,
    teacher_payload: Mapping[str, Any],
    *,
    tolerance: float = 1.0e-4,
) -> dict[str, Any]:
    teacher_raw = tuple(float(value) for value in teacher_payload.get("raw_denoising_steps", ()))
    teacher_warped = tuple(
        float(value) for value in teacher_payload.get("warped_denoising_steps", ())
    )
    if teacher_raw != schedule.raw_schedule:
        raise RuntimeError("config raw schedule differs from teacher payload")
    if len(teacher_warped) != len(schedule.main_warped_schedule):
        raise RuntimeError("teacher warped schedule length differs from resolved config")
    diffs = [
        abs(actual - expected)
        for actual, expected in zip(schedule.main_warped_schedule, teacher_warped)
    ]
    max_abs_diff = max(diffs) if diffs else math.inf
    mean_abs_diff = sum(diffs) / len(diffs) if diffs else math.inf
    if max_abs_diff > float(tolerance):
        raise RuntimeError(
            "config warped schedule differs from teacher payload: "
            f"max_abs_diff={max_abs_diff}, tolerance={tolerance}"
        )
    return {
        "raw_match": True,
        "warped_match": True,
        "max_abs_diff": float(max_abs_diff),
        "mean_abs_diff": float(mean_abs_diff),
        "tolerance": float(tolerance),
    }


def compare_latents(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    chunk_frames: int = M3_CHUNK_FRAMES,
    tolerance: float | None = None,
) -> dict[str, Any]:
    if tolerance is not None and float(tolerance) < 0:
        raise ValueError("tolerance must be non-negative")
    shape_match = tuple(actual.shape) == tuple(expected.shape)
    dtype_match = actual.dtype == expected.dtype
    exact_equality = shape_match and dtype_match and bool(torch.equal(actual, expected))
    max_abs_diff = mean_abs_diff = mse = None
    per_chunk = []
    if shape_match:
        diff = actual.detach().float().cpu() - expected.detach().float().cpu()
        abs_diff = diff.abs()
        max_abs_diff = float(abs_diff.max().item())
        mean_abs_diff = float(abs_diff.mean().item())
        mse = float(diff.square().mean().item())
        if actual.ndim >= 2:
            frame_count = int(actual.shape[1])
            for chunk_index, start in enumerate(range(0, frame_count, int(chunk_frames))):
                chunk_diff = diff[:, start:start + int(chunk_frames)]
                if chunk_diff.numel() == 0:
                    continue
                per_chunk.append(
                    {
                        "chunk_index": int(chunk_index),
                        "start_frame": int(start),
                        "max_abs_diff": float(chunk_diff.abs().max().item()),
                        "mean_abs_diff": float(chunk_diff.abs().mean().item()),
                        "mse": float(chunk_diff.square().mean().item()),
                    }
                )
    reproduction_pass = None
    if tolerance is not None:
        reproduction_pass = bool(
            shape_match
            and dtype_match
            and max_abs_diff is not None
            and max_abs_diff <= float(tolerance)
        )
    return {
        "schema": M6_COMPARISON_SCHEMA,
        "shape_match": bool(shape_match),
        "dtype_match": bool(dtype_match),
        "exact_equality": bool(exact_equality),
        "actual_shape": [int(dim) for dim in actual.shape],
        "expected_shape": [int(dim) for dim in expected.shape],
        "actual_dtype": str(actual.dtype),
        "expected_dtype": str(expected.dtype),
        "actual_sha256": tensor_sha256(actual),
        "expected_sha256": tensor_sha256(expected),
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
        "mse": mse,
        "per_chunk": per_chunk,
        "tolerance": None if tolerance is None else float(tolerance),
        "reproduction_pass": reproduction_pass,
    }


def compare_pixel_frames(
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> dict[str, Any]:
    shape_match = tuple(actual.shape) == tuple(expected.shape)
    dtype_match = actual.dtype == expected.dtype
    exact_equal = shape_match and dtype_match and bool(torch.equal(actual, expected))
    mae = mse = psnr = None
    per_frame = []
    if shape_match:
        diff = actual.detach().float().cpu() - expected.detach().float().cpu()
        abs_diff = diff.abs()
        mae = float(abs_diff.mean().item())
        mse = float(diff.square().mean().item())
        if mse > 0.0:
            psnr = float(20.0 * math.log10(255.0) - 10.0 * math.log10(mse))
        if actual.ndim >= 1:
            for frame_index in range(int(actual.shape[0])):
                frame_diff = diff[frame_index]
                if frame_diff.numel() == 0:
                    continue
                per_frame.append(
                    {
                        "frame_index": int(frame_index),
                        "mae": float(frame_diff.abs().mean().item()),
                        "mse": float(frame_diff.square().mean().item()),
                    }
                )
    comparison = {
        "schema": M6_PIXEL_COMPARISON_SCHEMA,
        "shape_match": bool(shape_match),
        "dtype_match": bool(dtype_match),
        "exact_equal": bool(exact_equal),
        "actual_shape": [int(dim) for dim in actual.shape],
        "expected_shape": [int(dim) for dim in expected.shape],
        "actual_dtype": str(actual.dtype),
        "expected_dtype": str(expected.dtype),
        "mae": mae,
        "mse": mse,
        "psnr": psnr,
        "per_frame": per_frame,
    }
    validate_json_payload(comparison)
    return comparison


def write_oracle_comparison(
    *,
    output_dir: Path | str,
    oracle_b_latent: torch.Tensor,
    oracle_a_latent_path: Path | str,
    tolerance: float | None,
    chunk_frames: int = M3_CHUNK_FRAMES,
) -> dict[str, Any]:
    oracle_a_latent = load_output_latent_artifact(oracle_a_latent_path)
    comparison = compare_latents(
        oracle_b_latent.detach().cpu(),
        oracle_a_latent,
        chunk_frames=chunk_frames,
        tolerance=tolerance,
    )
    validate_json_payload(comparison)
    atomic_json_write(comparison, Path(output_dir) / "oracle_comparison.json")
    return comparison


def compare_with_oracle_a_artifact(
    *,
    oracle_b_latent: torch.Tensor,
    oracle_a_latent_path: Path | str,
    tolerance: float | None,
    chunk_frames: int = M3_CHUNK_FRAMES,
) -> dict[str, Any]:
    oracle_a_latent = load_output_latent_artifact(oracle_a_latent_path)
    comparison = compare_latents(
        oracle_b_latent.detach().cpu(),
        oracle_a_latent,
        chunk_frames=chunk_frames,
        tolerance=tolerance,
    )
    validate_json_payload(comparison)
    return comparison


def finalize_oracle_gate(
    result: M6OracleResult,
    *,
    oracle_a_comparison: Mapping[str, Any] | None = None,
    oracle_b_comparison: Mapping[str, Any] | None = None,
    oracle_b_artifact: Mapping[str, Any] | None = None,
) -> M6OracleResult:
    trace = copy.deepcopy(result.trace)
    summary = copy.deepcopy(result.summary)
    _apply_oracle_gate_fields(
        trace,
        summary,
        oracle_a_comparison=oracle_a_comparison,
        oracle_b_comparison=oracle_b_comparison,
        oracle_b_artifact=oracle_b_artifact,
    )
    validate_json_payload(trace)
    validate_json_payload(summary)
    return M6OracleResult(
        latent=result.latent,
        trace=trace,
        summary=summary,
    )


def oracle_stdout_payload(
    *,
    result: M6OracleResult,
    output_dir: Path | str,
    artifact_hashes: Mapping[str, str],
    comparison: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    summary = result.summary
    payload = {
        "schema": summary["schema"],
        "oracle": summary["oracle_kind"],
        "output_dir": str(Path(output_dir).resolve()),
        "artifact_hashes": dict(artifact_hashes),
        "comparison": None if comparison is None else dict(comparison),
        "execution_status": summary["execution_status"],
        "protocol_pass": summary["protocol_pass"],
        "target_reproduction_pass": summary["target_reproduction_pass"],
        "oracle_a_reproduction_pass": summary["oracle_a_reproduction_pass"],
        "main_quality_pass": summary["main_quality_pass"],
        "review_status": summary["review_status"],
        "oracle_gate_pass": summary["oracle_gate_pass"],
        "status": summary["status"],
        "gate_reasons": list(summary["gate_reasons"]),
        "mcp_call_count": summary["mcp_call_count"],
    }
    validate_json_payload(payload)
    return payload


def prepare_output_dir(output_dir: Path | str) -> None:
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output_dir must be absent or empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)


def validate_json_payload(value: Any, *, path: str = "root") -> None:
    if torch.is_tensor(value):
        raise TypeError(f"JSON payload must not contain tensors at {path}")
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"JSON object key must be a string at {path}")
            validate_json_payload(item, path=f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            validate_json_payload(item, path=f"{path}[{index}]")
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"JSON payload contains non-finite float at {path}")
        return
    if isinstance(value, (str, int, bool)) or value is None:
        return
    json.dumps(value)


def reset_rollout_rng(seed: int, device: torch.device | str) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    device = torch.device(device)
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def global_rng_state_hash(device: torch.device | str) -> str:
    device = torch.device(device)
    payload = {
        "torch_cpu": _bytes_sha256(torch.random.get_rng_state().cpu().numpy().tobytes()),
        "torch_cuda": None,
    }
    if device.type == "cuda" and torch.cuda.is_available():
        payload["torch_cuda"] = _bytes_sha256(
            torch.cuda.get_rng_state(device).cpu().numpy().tobytes()
        )
    return _bytes_sha256(json.dumps(payload, sort_keys=True).encode("utf-8"))


def consume_teacher_exit_flag_rng_compatibility_draw(
    *,
    num_chunks: int,
    num_denoising_steps: int,
    device: torch.device | str,
    draw_order: int,
) -> dict[str, Any]:
    if int(num_chunks) <= 0:
        raise ValueError("num_chunks must be positive")
    if int(num_denoising_steps) <= 0:
        raise ValueError("num_denoising_steps must be positive")
    device = torch.device(device)
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
        "purpose": M6_TEACHER_COMPATIBILITY_PURPOSE,
        "operation": "torch.randint",
        "low": 0,
        "high": int(num_denoising_steps),
        "size": [int(num_chunks)],
        "dtype": str(values.dtype),
        "device": str(values.device),
        "state_before_hash": state_before,
        "state_after_hash": state_after,
        "values": [int(value) for value in values.detach().cpu().tolist()],
        "values_discarded": True,
        "reason": M6_TEACHER_COMPATIBILITY_REASON,
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
    record = {
        "draw_order": int(draw_order),
        "purpose": purpose,
        "chunk_index": int(chunk_index),
        "solver_step_index": None if solver_step_index is None else int(solver_step_index),
        "state_before_hash": state_before,
        "state_after_hash": state_after,
        "noise": tensor_json_summary(noise),
    }
    return noise, record


def tensor_json_summary(tensor: torch.Tensor) -> dict[str, Any]:
    summary = tensor_summary(tensor.detach().cpu())
    return {
        "shape": summary["shape"],
        "dtype": summary["dtype"],
        "finite": summary["finite"],
        "sha256": summary["sha256"],
    }


def conditioning_json_summary(conditional_dict: Mapping[str, Any]) -> dict[str, Any]:
    summary = _json_safe_conditioning_summary(conditional_dict)
    digest = _bytes_sha256(
        json.dumps(summary, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    return {
        "sha256": digest,
        "summary": summary,
    }


def kv_boundary_summary(kv_cache: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    layers = []
    for index, layer in enumerate(kv_cache):
        global_end = _cache_index_value(layer, "global_end_index")
        local_end = _cache_index_value(layer, "local_end_index")
        layers.append(
            {
                "layer": int(index),
                "global_end_index": int(global_end),
                "local_end_index": int(local_end),
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


def _require_kv_rollback_matches(
    before: Mapping[str, Any],
    rollback: Mapping[str, Any],
) -> None:
    before_layers = before.get("layers")
    rollback_layers = rollback.get("layers")
    if not isinstance(before_layers, Sequence) or not isinstance(rollback_layers, Sequence):
        raise TypeError("KV rollback summary missing layers")
    if len(before_layers) != len(rollback_layers):
        raise RuntimeError("KV rollback layer count differs from before")
    for before_layer, rollback_layer in zip(before_layers, rollback_layers):
        if not isinstance(before_layer, Mapping) or not isinstance(rollback_layer, Mapping):
            raise TypeError("KV rollback layer entry is invalid")
        layer_id = before_layer.get("layer")
        if rollback_layer.get("layer") != layer_id:
            raise RuntimeError("KV rollback layer order differs from before")
        for field in ("global_end_index", "local_end_index"):
            if rollback_layer.get(field) != before_layer.get(field):
                raise RuntimeError(
                    "KV rollback boundary mismatch: "
                    f"layer={layer_id}, field={field}, "
                    f"before={before_layer.get(field)}, rollback={rollback_layer.get(field)}"
                )


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
    expected_delta = int(chunk_frames) * int(frame_seq_length)
    actual_before = int(before["local_end_index"])
    actual_after = int(after["local_end_index"])
    if actual_before != expected_before:
        raise RuntimeError(
            "clean recache before boundary mismatch: "
            f"expected={expected_before}, actual={actual_before}"
        )
    if actual_after != expected_after:
        raise RuntimeError(
            "clean recache after boundary mismatch: "
            f"expected={expected_after}, actual={actual_after}"
        )
    if actual_after - actual_before != expected_delta:
        raise RuntimeError(
            "clean recache advancement mismatch: "
            f"expected={expected_delta}, actual={actual_after - actual_before}"
        )


def _require_commit_contract(
    *,
    cursor_before: int,
    cursor_after: int,
    start_frame: int,
    chunk_frames: int,
) -> None:
    if int(cursor_before) != int(start_frame):
        raise RuntimeError(
            "cursor_before must equal chunk start_frame: "
            f"cursor_before={cursor_before}, start_frame={start_frame}"
        )
    expected_after = int(start_frame) + int(chunk_frames)
    if int(cursor_after) != expected_after:
        raise RuntimeError(
            "cursor_after mismatch: "
            f"expected={expected_after}, actual={cursor_after}"
        )


def _runtime_mcp_call_count(generator: Any) -> int:
    return int(getattr(generator, "mcp_call_count", 0))


def teacher_identity_json(metadata: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "manifest_path",
        "manifest_sha256",
        "dataset_root",
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
    return {key: metadata.get(key) for key in keys if key in metadata}


def _apply_oracle_gate_fields(
    trace: dict[str, Any],
    summary: dict[str, Any],
    *,
    oracle_a_comparison: Mapping[str, Any] | None,
    oracle_b_comparison: Mapping[str, Any] | None,
    oracle_b_artifact: Mapping[str, Any] | None,
) -> None:
    oracle_kind = str(trace.get("oracle_kind"))
    if oracle_a_comparison is not None:
        trace["oracle_a_comparison"] = dict(oracle_a_comparison)
        summary["oracle_a_comparison"] = dict(oracle_a_comparison)
    elif oracle_kind == "B":
        trace["oracle_a_comparison"] = None
        summary["oracle_a_comparison"] = None
    if oracle_b_comparison is not None:
        trace["oracle_b_comparison"] = dict(oracle_b_comparison)
        summary["oracle_b_comparison"] = dict(oracle_b_comparison)
    elif oracle_kind == "C":
        trace["oracle_b_comparison"] = None
        summary["oracle_b_comparison"] = None
    if oracle_b_artifact is not None:
        trace["oracle_b_artifact"] = dict(oracle_b_artifact)
        summary["oracle_b_artifact"] = dict(oracle_b_artifact)
    elif oracle_kind == "C":
        trace["oracle_b_artifact"] = None
        summary["oracle_b_artifact"] = None

    protocol_reasons = _protocol_failure_reasons(trace)
    protocol_pass = len(protocol_reasons) == 0
    target_pass, target_reason = _nullable_comparison_gate_value(
        trace.get("target_latent_comparison"),
        report_only_reason="TARGET_REPRODUCTION_TOLERANCE_NOT_PROVIDED",
        failed_reason="TARGET_REPRODUCTION_FAILED",
    )

    oracle_a_pass: bool | None
    oracle_a_reason: str | None = None
    if oracle_kind == "B":
        if oracle_a_comparison is None:
            oracle_a_pass = False
            oracle_a_reason = "ORACLE_A_COMPARISON_MISSING"
        else:
            oracle_a_pass, oracle_a_reason = _nullable_comparison_gate_value(
                oracle_a_comparison,
                report_only_reason="ORACLE_A_REPRODUCTION_TOLERANCE_NOT_PROVIDED",
                failed_reason="ORACLE_A_REPRODUCTION_FAILED",
            )
    else:
        oracle_a_pass = None

    main_quality_pass: bool | None = None
    review_status: str | None = None
    main_quality_reason: str | None = None
    if oracle_kind == "C":
        review_status = "PENDING"
        if protocol_pass:
            main_quality_reason = "MAIN_QUALITY_REVIEW_PENDING"

    gate_reasons = list(protocol_reasons)
    if oracle_kind == "A" and target_reason is not None:
        gate_reasons.append(target_reason)
    if oracle_a_reason is not None:
        gate_reasons.append(oracle_a_reason)
    if main_quality_reason is not None:
        gate_reasons.append(main_quality_reason)

    if oracle_kind == "A":
        oracle_gate_pass = _and_nullable(protocol_pass, target_pass)
    elif oracle_kind == "B":
        oracle_gate_pass = _and_nullable(protocol_pass, oracle_a_pass)
    elif oracle_kind == "C":
        oracle_gate_pass = _and_nullable(protocol_pass, main_quality_pass)
    else:
        oracle_gate_pass = False
        gate_reasons.append("UNSUPPORTED_ORACLE_KIND")
    status = _status_from_gate(oracle_gate_pass)
    gate_fields = {
        "execution_status": "COMPLETED",
        "protocol_pass": bool(protocol_pass),
        "target_reproduction_pass": target_pass,
        "oracle_a_reproduction_pass": oracle_a_pass,
        "main_quality_pass": main_quality_pass,
        "review_status": review_status,
        "oracle_gate_pass": oracle_gate_pass,
        "status": status,
        "gate_reasons": gate_reasons,
    }
    if status == "PASS" and gate_reasons:
        raise RuntimeError("PASS oracle gate must not contain gate_reasons")
    trace.update(gate_fields)
    summary.update(gate_fields)


def _nullable_comparison_gate_value(
    comparison: Any,
    *,
    report_only_reason: str,
    failed_reason: str,
) -> tuple[bool, str | None]:
    if not isinstance(comparison, Mapping):
        return False, failed_reason
    reproduction_pass = comparison.get("reproduction_pass")
    if reproduction_pass is True:
        return True, None
    if reproduction_pass is None:
        return None, report_only_reason
    return False, failed_reason


def _and_nullable(left: bool | None, right: bool | None) -> bool | None:
    if left is False or right is False:
        return False
    if left is None or right is None:
        return None
    return True


def _status_from_gate(oracle_gate_pass: bool | None) -> str:
    if oracle_gate_pass is True:
        return "PASS"
    if oracle_gate_pass is None:
        return "REPORT_ONLY"
    return "FAIL"


def _protocol_failure_reasons(trace: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []
    oracle_kind = str(trace.get("oracle_kind"))
    checkpoint = trace.get("checkpoint")
    if not isinstance(checkpoint, Mapping):
        reasons.append("CHECKPOINT_METADATA_MISSING")
    elif oracle_kind == "A" and checkpoint.get("type") != M6_CHECKPOINT_OFFICIAL:
        reasons.append("CHECKPOINT_TYPE_NOT_OFFICIAL")
    elif oracle_kind == "B":
        if checkpoint.get("type") != M6_CHECKPOINT_FORMAL_STEP0:
            reasons.append("CHECKPOINT_TYPE_NOT_FORMAL_STEP0")
        if checkpoint.get("global_step") != 0:
            reasons.append("CHECKPOINT_GLOBAL_STEP_NOT_ZERO")
        if int(checkpoint.get("mcp_tensor_count", 0)) <= 0:
            reasons.append("CHECKPOINT_MCP_TENSORS_MISSING")
    elif oracle_kind == "C":
        if checkpoint.get("type") != M6_CHECKPOINT_FORMAL_STEP500:
            reasons.append("CHECKPOINT_TYPE_NOT_FORMAL_STEP500")
        if checkpoint.get("global_step") != 500:
            reasons.append("CHECKPOINT_GLOBAL_STEP_NOT_500")
        if int(checkpoint.get("mcp_tensor_count", 0)) <= 0:
            reasons.append("CHECKPOINT_MCP_TENSORS_MISSING")

    schedule = trace.get("schedule")
    if not isinstance(schedule, Mapping):
        reasons.append("SCHEDULE_METADATA_MISSING")
    else:
        raw_schedule = tuple(
            float(value) for value in schedule.get("raw_schedule", ())
        )
        if raw_schedule != M6_LOCKED_RAW_SCHEDULE:
            reasons.append("RAW_SCHEDULE_MISMATCH")
        if len(schedule.get("main_warped_schedule", ())) != len(M6_LOCKED_RAW_SCHEDULE):
            reasons.append("MAIN_SCHEDULE_NOT_FOUR_STEP")
        if (
            schedule.get("mcp_enabled") is not False
            or schedule.get("mcp_warped_schedule") is not None
        ):
            reasons.append("MCP_SCHEDULE_NOT_DISABLED")

    if trace.get("mcp_enabled") is not False:
        reasons.append("MCP_NOT_DISABLED")
    if int(trace.get("mcp_call_count", -1)) != 0:
        reasons.append("MCP_CALL_COUNT_NONZERO")

    common_inputs = trace.get("common_inputs")
    common_fingerprint = trace.get("common_inputs_fingerprint_sha256")
    if not isinstance(common_inputs, Mapping):
        reasons.append("COMMON_INPUTS_MISSING")
    else:
        if common_inputs.get("rng_draw_contract_version") != M6_RNG_DRAW_CONTRACT_VERSION:
            reasons.append("RNG_DRAW_CONTRACT_NOT_V2")
        if canonical_json_sha256(dict(common_inputs)) != common_fingerprint:
            reasons.append("COMMON_INPUTS_FINGERPRINT_MISMATCH")

    finite_checks = trace.get("finite_checks")
    if not isinstance(finite_checks, Mapping) or not all(
        bool(value) for value in finite_checks.values()
    ):
        reasons.append("FINITE_CHECKS_FAILED")

    chunks = trace.get("chunks")
    if not isinstance(chunks, Sequence) or isinstance(chunks, (str, bytes, bytearray)):
        reasons.append("CHUNKS_TRACE_MISSING")
    else:
        for chunk_index, chunk in enumerate(chunks):
            if not isinstance(chunk, Mapping):
                reasons.append(f"CHUNK_{chunk_index}_TRACE_INVALID")
                continue
            steps = chunk.get("solver_steps")
            if not isinstance(steps, Sequence) or isinstance(steps, (str, bytes, bytearray)):
                reasons.append(f"CHUNK_{chunk_index}_SOLVER_STEPS_MISSING")
                continue
            if len(steps) != len(M6_LOCKED_RAW_SCHEDULE):
                reasons.append(f"CHUNK_{chunk_index}_SOLVER_STEP_COUNT_INVALID")
            for step_index, step in enumerate(steps):
                if not isinstance(step, Mapping):
                    reasons.append(f"CHUNK_{chunk_index}_STEP_{step_index}_INVALID")
                    continue
                kv = step.get("kv")
                if not isinstance(kv, Mapping):
                    reasons.append(f"CHUNK_{chunk_index}_STEP_{step_index}_KV_MISSING")
                    continue
                before = kv.get("before")
                rollback = kv.get("rollback_after_forward")
                if not isinstance(before, Mapping) or not isinstance(rollback, Mapping):
                    reasons.append(f"CHUNK_{chunk_index}_STEP_{step_index}_KV_BOUNDARY_MISSING")
                elif (
                    before.get("global_end_index") != rollback.get("global_end_index")
                    or before.get("local_end_index") != rollback.get("local_end_index")
                ):
                    reasons.append(f"CHUNK_{chunk_index}_STEP_{step_index}_KV_ROLLBACK_MISMATCH")
                if kv.get("visible_data_restored") is not True:
                    reasons.append(f"CHUNK_{chunk_index}_STEP_{step_index}_KV_DATA_NOT_RESTORED")
            recache = chunk.get("clean_recache")
            if not isinstance(recache, Mapping):
                reasons.append(f"CHUNK_{chunk_index}_CLEAN_RECACHE_MISSING")
            else:
                before = recache.get("before")
                after = recache.get("after")
                if not isinstance(before, Mapping) or not isinstance(after, Mapping):
                    reasons.append(f"CHUNK_{chunk_index}_CLEAN_RECACHE_BOUNDARY_MISSING")
                elif int(after.get("local_end_index", -1)) <= int(
                    before.get("local_end_index", -1)
                ):
                    reasons.append(f"CHUNK_{chunk_index}_CLEAN_RECACHE_DID_NOT_ADVANCE")
            commit = chunk.get("commit")
            if not isinstance(commit, Mapping):
                reasons.append(f"CHUNK_{chunk_index}_COMMIT_MISSING")
            else:
                start_frame = int(chunk.get("start_frame", -1))
                chunk_frames = int(chunk.get("num_frames", -1))
                if commit.get("main_only") is not True or commit.get("next_commit") is not None:
                    reasons.append(f"CHUNK_{chunk_index}_COMMIT_NOT_MAIN_ONLY")
                if int(commit.get("cursor_before", -1)) != start_frame:
                    reasons.append(f"CHUNK_{chunk_index}_CURSOR_BEFORE_MISMATCH")
                if int(commit.get("cursor_after", -1)) != start_frame + chunk_frames:
                    reasons.append(f"CHUNK_{chunk_index}_CURSOR_AFTER_MISMATCH")
    if oracle_kind == "C":
        oracle_b_artifact = trace.get("oracle_b_artifact")
        oracle_b_comparison = trace.get("oracle_b_comparison")
        if not isinstance(oracle_b_artifact, Mapping):
            reasons.append("ORACLE_B_ARTIFACT_MISSING")
        else:
            if oracle_b_artifact.get("oracle_kind") != "B":
                reasons.append("ORACLE_B_ARTIFACT_KIND_INVALID")
            if oracle_b_artifact.get("status") != "PASS":
                reasons.append("ORACLE_B_ARTIFACT_STATUS_NOT_PASS")
            if oracle_b_artifact.get("protocol_pass") is not True:
                reasons.append("ORACLE_B_ARTIFACT_PROTOCOL_NOT_PASS")
            if oracle_b_artifact.get("oracle_gate_pass") is not True:
                reasons.append("ORACLE_B_ARTIFACT_GATE_NOT_PASS")
            if oracle_b_artifact.get("common_inputs_fingerprint_sha256") != common_fingerprint:
                reasons.append("ORACLE_B_COMMON_INPUTS_FINGERPRINT_MISMATCH")
        if not isinstance(oracle_b_comparison, Mapping):
            reasons.append("ORACLE_B_COMPARISON_MISSING")
        else:
            output_sha = (
                trace.get("artifact_hashes", {}).get("output_latent_tensor_sha256")
                if isinstance(trace.get("artifact_hashes"), Mapping)
                else None
            )
            if oracle_b_comparison.get("actual_sha256") != output_sha:
                reasons.append("ORACLE_B_COMPARISON_ACTUAL_SHA_MISMATCH")
            if (
                isinstance(oracle_b_artifact, Mapping)
                and oracle_b_comparison.get("expected_sha256")
                != oracle_b_artifact.get("latent_sha256")
            ):
                reasons.append("ORACLE_B_COMPARISON_EXPECTED_SHA_MISMATCH")
    return reasons


def _official_checkpoint_record(
    *,
    path: Path,
    payload: Any,
    checkpoint_sha: str,
    expected_sha256: str | None,
) -> M6CheckpointRecord:
    if isinstance(payload, Mapping) and payload.get("format") == M3_CHECKPOINT_FORMAT:
        raise RuntimeError("Oracle A requires official checkpoint, got formal checkpoint")
    if expected_sha256 is not None and checkpoint_sha != str(expected_sha256):
        raise RuntimeError("Oracle A official checkpoint SHA256 mismatch")
    state_dict = extract_generator_state_dict(payload)
    mcp_tensor_count = _count_mcp_tensors(state_dict)
    if mcp_tensor_count != 0:
        raise RuntimeError("Oracle A official checkpoint must not contain MCP tensors")
    return M6CheckpointRecord(
        path=str(path.resolve()),
        sha256=checkpoint_sha,
        checkpoint_type=M6_CHECKPOINT_OFFICIAL,
        load_mode="OFFICIAL_BACKBONE_STRICT",
        generator_state_dict=state_dict,
        global_step=None,
        mcp_tensor_count=mcp_tensor_count,
        payload_format=(
            None
            if not isinstance(payload, Mapping)
            else _string_or_none(payload.get("format"))
        ),
    )


def _formal_step0_checkpoint_record(
    *,
    path: Path,
    payload: Any,
    checkpoint_sha: str,
) -> M6CheckpointRecord:
    return _formal_stage_a_checkpoint_record(
        path=path,
        payload=payload,
        checkpoint_sha=checkpoint_sha,
        oracle_kind="B",
        expected_global_step=0,
        checkpoint_type=M6_CHECKPOINT_FORMAL_STEP0,
        load_mode="FORMAL_STEP0_FULL_GENERATOR_STRICT",
    )


def _formal_step500_checkpoint_record(
    *,
    path: Path,
    payload: Any,
    checkpoint_sha: str,
) -> M6CheckpointRecord:
    return _formal_stage_a_checkpoint_record(
        path=path,
        payload=payload,
        checkpoint_sha=checkpoint_sha,
        oracle_kind="C",
        expected_global_step=500,
        checkpoint_type=M6_CHECKPOINT_FORMAL_STEP500,
        load_mode="FORMAL_STEP500_FULL_GENERATOR_STRICT",
    )


def _formal_stage_a_checkpoint_record(
    *,
    path: Path,
    payload: Any,
    checkpoint_sha: str,
    oracle_kind: OracleKind,
    expected_global_step: int,
    checkpoint_type: str,
    load_mode: str,
) -> M6CheckpointRecord:
    descriptor = _formal_checkpoint_descriptor(oracle_kind, expected_global_step)
    if str(path).lower().endswith(".tmp"):
        raise RuntimeError(f"{descriptor} checkpoint path must not end with .tmp")
    if not isinstance(payload, Mapping) or payload.get("format") != M3_CHECKPOINT_FORMAT:
        raise RuntimeError(f"{descriptor} requires formal M3 checkpoint")
    formal_payload = load_m3_checkpoint(path)
    global_step = int(formal_payload["global_step"])
    if global_step != int(expected_global_step):
        raise RuntimeError(
            f"{descriptor} requires formal global_step={int(expected_global_step)} checkpoint"
        )
    state_dict = formal_payload["generator"]
    mcp_tensor_count = _count_mcp_tensors(state_dict)
    if mcp_tensor_count <= 0:
        raise RuntimeError(f"{descriptor} checkpoint must contain MCP tensors")
    formal_metadata = _validate_m5_formal_stage_a_checkpoint(
        formal_payload,
        oracle_kind=oracle_kind,
        expected_global_step=int(expected_global_step),
    )
    return M6CheckpointRecord(
        path=str(path.resolve()),
        sha256=checkpoint_sha,
        checkpoint_type=checkpoint_type,
        load_mode=load_mode,
        generator_state_dict=state_dict,
        global_step=global_step,
        mcp_tensor_count=mcp_tensor_count,
        payload_format=str(formal_payload["format"]),
        formal_metadata=formal_metadata,
    )


def _validate_m5_formal_step0_checkpoint(payload: Mapping[str, Any]) -> dict[str, Any]:
    return _validate_m5_formal_stage_a_checkpoint(
        payload,
        oracle_kind="B",
        expected_global_step=0,
    )


def _validate_m5_formal_stage_a_checkpoint(
    payload: Mapping[str, Any],
    *,
    oracle_kind: OracleKind,
    expected_global_step: int,
) -> dict[str, Any]:
    descriptor = _formal_checkpoint_descriptor(oracle_kind, expected_global_step)
    if "m5_formal_smoke" in payload:
        raise RuntimeError(f"{descriptor} rejects M5 formal smoke checkpoint markers")
    validate_git_sha(str(payload.get("git_sha", "")), name="formal_checkpoint.git_sha")
    metadata = payload.get("m5_formal_trainer")
    if not isinstance(metadata, Mapping):
        raise TypeError(f"{descriptor} checkpoint missing m5_formal_trainer metadata")
    if metadata.get("schema") != M5_FORMAL_TRAINER_SCHEMA:
        raise RuntimeError(f"{descriptor} checkpoint schema mismatch")
    if metadata.get("status") != "PASS":
        raise RuntimeError(f"{descriptor} checkpoint status mismatch")
    if metadata.get("formal_enabled") is not True:
        raise RuntimeError(f"{descriptor} checkpoint marker is not enabled")
    if metadata.get("smoke_enabled") is True:
        raise RuntimeError(f"{descriptor} checkpoint must not enable smoke")
    if metadata.get("run_kind") == "short_smoke":
        raise RuntimeError(f"{descriptor} checkpoint must not be short_smoke")

    stage_contract = resolve_m5_formal_stage_contract(500)
    expected_stage_contract = {
        "target_global_step": int(stage_contract.target_global_step),
        "parent_global_step": stage_contract.parent_global_step,
        "validation_steps": list(stage_contract.validation_steps),
        "checkpoint_steps": list(stage_contract.checkpoint_steps),
        "is_resume_stage": bool(stage_contract.is_resume_stage),
    }
    expected_stage = "stage_a"
    if metadata.get("stage") != expected_stage:
        raise RuntimeError(
            f"{descriptor} checkpoint stage mismatch: "
            f"expected={expected_stage}, actual={metadata.get('stage')}"
        )
    actual_stage_contract = _strict_formal_stage_contract(
        metadata.get("stage_contract"),
        field_path="m5_formal_trainer.stage_contract",
    )
    if actual_stage_contract != expected_stage_contract:
        raise RuntimeError(
            f"{descriptor} checkpoint stage_contract mismatch: "
            f"expected={expected_stage_contract}, actual={actual_stage_contract}"
        )

    resolved_config = payload.get("resolved_config")
    if not isinstance(resolved_config, Mapping):
        raise TypeError(f"{descriptor} checkpoint missing resolved_config")
    formal_config = resolved_config.get("m5_formal")
    if not isinstance(formal_config, Mapping):
        raise TypeError(f"{descriptor} checkpoint resolved_config missing m5_formal")

    matched_fields = {
        "sample_plan_sha256": metadata.get("sample_plan_sha256"),
        "teacher_manifest_sha256": metadata.get("teacher_manifest_sha256"),
        "conditional_artifact_sha256": metadata.get("conditional_artifact_sha256"),
        "validation_implementation_schema": M5_STREAMING_VALIDATION_SCHEMA,
    }
    for field in (
        "sample_plan_sha256",
        "teacher_manifest_sha256",
        "conditional_artifact_sha256",
    ):
        _require_sha256(metadata.get(field), f"m5_formal_trainer.{field}")
    if metadata.get("validation_implementation_schema") != M5_STREAMING_VALIDATION_SCHEMA:
        raise RuntimeError(
            f"{descriptor} validation implementation schema mismatch"
        )
    for field, expected in matched_fields.items():
        actual = formal_config.get(field)
        if actual != expected:
            raise RuntimeError(
                f"{descriptor} checkpoint provenance mismatch: "
                f"field=m5_formal.{field}, expected={expected}, actual={actual}"
            )
    if formal_config.get("schema") != M5_FORMAL_TRAINER_SCHEMA:
        raise RuntimeError(f"{descriptor} resolved_config schema mismatch")
    if formal_config.get("enabled") is not True:
        raise RuntimeError(f"{descriptor} resolved_config marker is not enabled")
    return {
        "schema": str(metadata["schema"]),
        "status": str(metadata["status"]),
        "formal_enabled": True,
        "stage": str(metadata["stage"]),
        "stage_contract": actual_stage_contract,
        "sample_plan_sha256": str(metadata["sample_plan_sha256"]),
        "teacher_manifest_sha256": str(metadata["teacher_manifest_sha256"]),
        "conditional_artifact_sha256": str(metadata["conditional_artifact_sha256"]),
        "validation_implementation_schema": str(metadata["validation_implementation_schema"]),
    }


def _formal_checkpoint_descriptor(
    oracle_kind: OracleKind,
    expected_global_step: int,
) -> str:
    return f"Oracle {oracle_kind} formal step{int(expected_global_step)}"


def _strict_formal_stage_contract(value: Any, *, field_path: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_path} must be a mapping")
    expected_keys = {
        "target_global_step",
        "parent_global_step",
        "validation_steps",
        "checkpoint_steps",
        "is_resume_stage",
    }
    actual_keys = set(value.keys())
    if actual_keys != expected_keys:
        raise RuntimeError(
            f"{field_path} keys mismatch: expected={sorted(expected_keys)}, "
            f"actual={sorted(actual_keys, key=str)}"
        )
    return {
        "target_global_step": _require_python_int(
            value["target_global_step"],
            f"{field_path}.target_global_step",
        ),
        "parent_global_step": _require_optional_python_int(
            value["parent_global_step"],
            f"{field_path}.parent_global_step",
        ),
        "validation_steps": _require_int_list(
            value["validation_steps"],
            f"{field_path}.validation_steps",
        ),
        "checkpoint_steps": _require_int_list(
            value["checkpoint_steps"],
            f"{field_path}.checkpoint_steps",
        ),
        "is_resume_stage": _require_bool(
            value["is_resume_stage"],
            f"{field_path}.is_resume_stage",
        ),
    }


def _raw_schedule_from_config(config: Any) -> list[float]:
    if not hasattr(config, "denoising_step_list"):
        raise ValueError("config must define denoising_step_list")
    values = list(config.denoising_step_list)
    if not values:
        raise ValueError("config.denoising_step_list must not be empty")
    raw = []
    for index, value in enumerate(values):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"denoising_step_list[{index}] is non-finite")
        if number < 0 or number > DEFAULT_NUM_TRAIN_TIMESTEPS:
            raise ValueError(f"denoising_step_list[{index}] out of range")
        raw.append(number)
    return raw


def _main_shift_from_config(config: Any, *, default: float) -> float:
    model_kwargs = getattr(config, "model_kwargs", None)
    if isinstance(model_kwargs, Mapping) and "timestep_shift" in model_kwargs:
        return float(model_kwargs["timestep_shift"])
    if hasattr(config, "timestep_shift"):
        return float(config.timestep_shift)
    return float(default)


def _validate_source_noise(
    source_noise: torch.Tensor,
    *,
    teacher_payload: Mapping[str, Any],
) -> None:
    if not torch.is_tensor(source_noise):
        raise TypeError("source_noise must be a torch.Tensor")
    if source_noise.ndim != 5:
        raise ValueError("source_noise must have layout [B, F, C, H, W]")
    if not source_noise.is_floating_point():
        raise ValueError("source_noise must be floating point")
    if int(source_noise.shape[1]) % M3_CHUNK_FRAMES != 0:
        raise ValueError("source_noise frame count must be chunk-aligned")
    expected = teacher_payload.get("source_noise")
    if not torch.is_tensor(expected):
        raise TypeError("teacher_payload.source_noise must be a tensor")
    if tensor_sha256(source_noise.detach().cpu()) != tensor_sha256(expected.detach().cpu()):
        raise RuntimeError("stored source_noise tensor does not match teacher payload")
    _ensure_finite_tensor(source_noise, name="source_noise")


def _validate_runtime(runtime: M6OracleRuntime) -> None:
    if int(runtime.num_frame_per_block) != M3_CHUNK_FRAMES:
        raise ValueError("M6.0 A/B/C requires chunk_frames=3")
    if not runtime.kv_cache:
        raise ValueError("runtime.kv_cache must not be empty")
    if not hasattr(runtime.scheduler, "add_noise"):
        raise TypeError("runtime.scheduler must provide add_noise")


def _unpack_main_outputs(outputs: Any) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(outputs, (tuple, list)) or len(outputs) < 2:
        raise RuntimeError("generator must return at least (flow_pred, pred_x0)")
    flow_pred, clean_pred = outputs[0], outputs[1]
    if not torch.is_tensor(flow_pred) or not torch.is_tensor(clean_pred):
        raise TypeError("generator flow and x0 outputs must be tensors")
    return flow_pred, clean_pred


def _ensure_finite_tensor(tensor: torch.Tensor, *, name: str) -> None:
    if not torch.is_tensor(tensor):
        raise TypeError(f"{name} must be a tensor")
    if not bool(torch.isfinite(tensor.detach().float()).all().item()):
        raise RuntimeError(f"{name} contains non-finite values")


def _cache_index_value(layer: Mapping[str, Any], key: str) -> int:
    value = layer.get(key)
    if torch.is_tensor(value):
        return int(value.item())
    return int(value)


def _set_cache_index(layer: Mapping[str, Any], key: str, value: int) -> None:
    target = layer[key]
    if torch.is_tensor(target):
        target.fill_(int(value))
    else:
        layer[key] = int(value)


def _count_mcp_tensors(state_dict: Mapping[str, Any]) -> int:
    return sum(
        1
        for key, value in state_dict.items()
        if is_mcp_state_key(str(key)) and torch.is_tensor(value)
    )


def _json_safe_conditioning_summary(value: Any) -> Any:
    if torch.is_tensor(value):
        return tensor_json_summary(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe_conditioning_summary(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe_conditioning_summary(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return copy.deepcopy(value)
    return repr(value)


def _bytes_sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _json_number(value: float) -> int | float:
    number = float(value)
    if number.is_integer():
        return int(number)
    return number


def _string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _require_python_int(value: Any, field_path: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{field_path} must be a Python int")
    return value


def _require_optional_python_int(value: Any, field_path: str) -> int | None:
    if value is None:
        return None
    return _require_python_int(value, field_path)


def _require_bool(value: Any, field_path: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{field_path} must be a bool")
    return value


def _require_int_list(value: Any, field_path: str) -> list[int]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise TypeError(f"{field_path} must be an int sequence")
    return [
        _require_python_int(item, f"{field_path}[{index}]")
        for index, item in enumerate(value)
    ]


def _require_sha256(value: Any, field_path: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_path} must be a lowercase SHA256 string")
    if len(value) != 64 or value.lower() != value:
        raise RuntimeError(f"{field_path} must be a lowercase SHA256 string")
    allowed = set("0123456789abcdef")
    if any(char not in allowed for char in value):
        raise RuntimeError(f"{field_path} must be a lowercase SHA256 string")
    return value
