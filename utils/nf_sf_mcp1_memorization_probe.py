from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

import utils.nf_sf_first_mcp_flow_audit as flow_audit
import utils.nf_sf_full_sequence_eval as deployment
from utils.nf_sf_m3 import tensor_sha256, tensor_summary
from utils.nf_sf_tensors import (
    DEFAULT_NUM_TRAIN_TIMESTEPS,
    DEFAULT_S_MAIN,
    DEFAULT_S_MCP,
    flow_match_shift_timesteps,
)
from utils.nf_sf_training import collect_nf_sf_parameter_groups
from utils.scheduler import FlowMatchScheduler


MCP1_MEMORIZATION_PROBE_SCHEMA = "nf_sf_mcp1_tiny_memorization_probe_v1"
MCP1_MEMORIZATION_TENSOR_SCHEMA = "nf_sf_mcp1_tiny_memorization_probe_tensors_v1"
MCP1_MEMORIZATION_STATE_SCHEMA = "nf_sf_mcp1_tiny_memorization_state_v1"
MCP1_MEMORIZATION_NOISE_SCHEMA = "nf_sf_mcp1_tiny_memorization_noise_v1"
RAW_TIMESTEPS = (999, 750, 500, 250)
NOISE_REALIZATIONS_PER_RAW_TIMESTEP = 4
DEFAULT_OPTIMIZER_STEPS = 1000
DEFAULT_OPTIMIZER_LR = 3.0e-4
DEFAULT_LOG_INTERVAL = 50
DEFAULT_NOISE_SEED = 6500
STRONG_MEMORIZATION_SUPPORT = "STRONG_MEMORIZATION_SUPPORT"
INSUFFICIENT_MEMORIZATION = "INSUFFICIENT_MEMORIZATION"
ALLOWED_STAGE_A_GROUPS = ("mcp_fusion", "mcp_depth1")
MAIN_GROUPS = ("backbone", "patch_embedding")
FORBIDDEN_MCP_GROUPS = ("mcp_depth2", "mcp_depth3")
HISTORY_CHUNK_INDEX = flow_audit.HISTORY_CHUNK_INDEX
CURRENT_CHUNK_INDEX = flow_audit.CURRENT_CHUNK_INDEX
FUTURE_CHUNK_INDEX = flow_audit.FUTURE_CHUNK_INDEX


@dataclass(frozen=True)
class MCP1MemorizationState:
    state_id: str
    raw_timestep: int
    noise_index: int
    main_warped_timestep: float
    mcp_warped_timestep: float
    current_state: torch.Tensor
    future_state: torch.Tensor
    current_noise: torch.Tensor
    future_noise: torch.Tensor
    main_target: torch.Tensor
    mcp_target: torch.Tensor
    main_timestep: torch.Tensor
    mcp_timestep: torch.Tensor
    provenance: dict[str, Any]


@dataclass(frozen=True)
class MCP1MemorizationProbeResult:
    manifest: dict[str, Any]
    tensors: dict[str, Any]


@dataclass(frozen=True)
class StageAParamSelection:
    trainable_named_parameters: tuple[tuple[str, torch.nn.Parameter], ...]
    optimizer_param_groups: tuple[dict[str, Any], ...]
    summary: dict[str, Any]


class MCP1ProbePristineCacheSnapshot:
    def __init__(
        self,
        *,
        kv_layers: tuple[dict[str, Any], ...],
        crossattn_layers: tuple[dict[str, Any], ...],
        kv_write_set: Mapping[str, Any],
    ) -> None:
        self._kv_layers = kv_layers
        self._crossattn_layers = crossattn_layers
        self.kv_write_set = dict(kv_write_set)
        self.reference_kv_payload = self._saved_kv_payload()
        self.reference_kv_fingerprint_sha256 = deployment.canonical_json_sha256(
            self.reference_kv_payload
        )
        self.reference_crossattn_payload = self._saved_crossattn_payload()
        self.reference_crossattn_fingerprint_sha256 = deployment.canonical_json_sha256(
            self.reference_crossattn_payload
        )
        self._phase_counts: dict[str, int] = {}

    @classmethod
    def capture(
        cls,
        *,
        runtime: deployment.DeploymentRuntime,
        history_recache: Mapping[str, Any],
    ) -> "MCP1ProbePristineCacheSnapshot":
        history_kv = flow_audit.kv_cache_content_fingerprint(runtime.kv_cache)
        if history_kv["fingerprint_sha256"] != history_recache.get(
            "history_kv_fingerprint_sha256"
        ):
            raise RuntimeError("pristine KV capture does not match history recache")
        crossattn = flow_audit.crossattn_cache_content_fingerprint(
            runtime.crossattn_cache
        )
        if crossattn["fingerprint_sha256"] != history_recache.get(
            "crossattn_cache_fingerprint_sha256"
        ):
            raise RuntimeError("pristine cross-attn capture does not match recache")

        kv_write_set = compute_mcp1_probe_kv_write_set(runtime)
        write_layers = kv_write_set["layers"]
        if len(write_layers) != len(runtime.kv_cache):
            raise RuntimeError("MCP1 probe KV write-set layer count mismatch")
        kv_layers = []
        for layer, write_layer in zip(runtime.kv_cache, write_layers):
            covered_windows = _merge_windows(
                (
                    (0, int(write_layer["history_active_prefix_end"])),
                    *(
                        (int(window["start"]), int(window["end"]))
                        for window in write_layer["write_windows"]
                    ),
                )
            )
            saved = {
                "layer": int(write_layer["layer"]),
                "global_end_index": _cache_index_int(layer.get("global_end_index")),
                "local_end_index": _cache_index_int(layer.get("local_end_index")),
                "semantic_windows": [
                    {
                        "role": "history_active_prefix",
                        "start": 0,
                        "end": int(write_layer["history_active_prefix_end"]),
                    },
                    *[
                        {
                            "role": str(window["role"]),
                            "start": int(window["start"]),
                            "end": int(window["end"]),
                        }
                        for window in write_layer["write_windows"]
                    ],
                ],
                "covered_windows": [
                    {"start": int(start), "end": int(end)}
                    for start, end in covered_windows
                ],
                "tensors": {},
            }
            for name in ("k", "v"):
                tensor = layer.get(name)
                if not torch.is_tensor(tensor):
                    raise RuntimeError(f"KV layer missing tensor {name!r}")
                saved["tensors"][name] = [
                    tensor[:, start:end].detach().cpu().clone()
                    for start, end in covered_windows
                ]
            kv_layers.append(saved)

        crossattn_layers = []
        for layer_index, layer in enumerate(runtime.crossattn_cache):
            saved = {
                "layer": int(layer_index),
                "is_init": bool(layer.get("is_init", False)),
                "tensors": {},
            }
            for name in ("k", "v"):
                tensor = layer.get(name)
                if torch.is_tensor(tensor):
                    saved["tensors"][name] = tensor.detach().cpu().clone()
            crossattn_layers.append(saved)

        snapshot = cls(
            kv_layers=tuple(kv_layers),
            crossattn_layers=tuple(crossattn_layers),
            kv_write_set=kv_write_set,
        )
        snapshot.assert_current_matches_reference(runtime, label="capture")
        return snapshot

    def restore(self, runtime: deployment.DeploymentRuntime) -> None:
        if len(runtime.kv_cache) != len(self._kv_layers):
            raise RuntimeError("pristine KV restore layer count differs")
        if len(runtime.crossattn_cache) != len(self._crossattn_layers):
            raise RuntimeError("pristine cross-attn restore layer count differs")
        with torch.no_grad():
            for layer, saved in zip(runtime.kv_cache, self._kv_layers):
                covered_windows = saved["covered_windows"]
                for name in ("k", "v"):
                    current = layer.get(name)
                    if not torch.is_tensor(current):
                        raise RuntimeError(f"KV restore missing tensor {name!r}")
                    saved_slices = saved["tensors"][name]
                    if len(saved_slices) != len(covered_windows):
                        raise RuntimeError("KV restore saved slice count mismatch")
                    for window, saved_slice in zip(covered_windows, saved_slices):
                        start = int(window["start"])
                        end = int(window["end"])
                        current[:, start:end].copy_(saved_slice.to(device=current.device))
                _set_cache_index(layer, "global_end_index", int(saved["global_end_index"]))
                _set_cache_index(layer, "local_end_index", int(saved["local_end_index"]))

            for layer, saved in zip(runtime.crossattn_cache, self._crossattn_layers):
                current_is_init = layer.get("is_init")
                if torch.is_tensor(current_is_init):
                    current_is_init.fill_(bool(saved["is_init"]))
                else:
                    layer["is_init"] = bool(saved["is_init"])
                for name, saved_tensor in saved["tensors"].items():
                    current = layer.get(name)
                    restored = saved_tensor.to(
                        device=current.device if torch.is_tensor(current) else saved_tensor.device
                    )
                    if torch.is_tensor(current) and tuple(current.shape) == tuple(restored.shape):
                        current.copy_(restored)
                    else:
                        layer[name] = restored.clone()

    def restore_and_verify(
        self,
        runtime: deployment.DeploymentRuntime,
        *,
        phase: str,
        state_id: str | None = None,
        optimizer_step: int | None = None,
    ) -> None:
        self.restore(runtime)
        self.assert_current_matches_reference(runtime, label=str(phase))
        self._phase_counts[str(phase)] = self._phase_counts.get(str(phase), 0) + 1
        _ = state_id, optimizer_step

    def assert_current_matches_reference(
        self,
        runtime: deployment.DeploymentRuntime,
        *,
        label: str,
    ) -> None:
        kv_payload = self.current_kv_payload(runtime)
        kv_fingerprint = deployment.canonical_json_sha256(kv_payload)
        if kv_fingerprint != self.reference_kv_fingerprint_sha256:
            raise RuntimeError(f"pristine KV fingerprint mismatch at {label}")
        crossattn_payload = self.current_crossattn_payload(runtime)
        crossattn_fingerprint = deployment.canonical_json_sha256(crossattn_payload)
        if crossattn_fingerprint != self.reference_crossattn_fingerprint_sha256:
            raise RuntimeError(f"pristine cross-attn fingerprint mismatch at {label}")

    def current_kv_payload(self, runtime: deployment.DeploymentRuntime) -> dict[str, Any]:
        if len(runtime.kv_cache) != len(self._kv_layers):
            raise RuntimeError("KV fingerprint layer count differs")
        layers = []
        for layer, saved in zip(runtime.kv_cache, self._kv_layers):
            entry: dict[str, Any] = {
                "layer": int(saved["layer"]),
                "global_end_index": _cache_index_int(layer.get("global_end_index")),
                "local_end_index": _cache_index_int(layer.get("local_end_index")),
                "semantic_windows": saved["semantic_windows"],
                "covered_windows": saved["covered_windows"],
                "tensors": {},
            }
            for name in ("k", "v"):
                tensor = layer.get(name)
                if not torch.is_tensor(tensor):
                    raise RuntimeError(f"KV fingerprint missing tensor {name!r}")
                records = []
                for window in saved["covered_windows"]:
                    start = int(window["start"])
                    end = int(window["end"])
                    active = tensor[:, start:end].detach().cpu()
                    records.append(
                        {
                            "start": start,
                            "end": end,
                            "shape": [int(dim) for dim in active.shape],
                            "dtype": str(active.dtype),
                            "sha256": tensor_sha256(active),
                        }
                    )
                entry["tensors"][name] = records
            layers.append(entry)
        return {
            "schema": "nf_sf_mcp1_pristine_kv_window_fingerprint_v1",
            "layers": layers,
            "kv_write_set_fingerprint_sha256": self.kv_write_set[
                "fingerprint_sha256"
            ],
        }

    def current_crossattn_payload(
        self,
        runtime: deployment.DeploymentRuntime,
    ) -> dict[str, Any]:
        if len(runtime.crossattn_cache) != len(self._crossattn_layers):
            raise RuntimeError("cross-attn fingerprint layer count differs")
        layers = []
        for layer, saved in zip(runtime.crossattn_cache, self._crossattn_layers):
            entry: dict[str, Any] = {
                "layer": int(saved["layer"]),
                "is_init": bool(layer.get("is_init", False)),
                "tensors": {},
            }
            for name in ("k", "v"):
                tensor = layer.get(name)
                if torch.is_tensor(tensor):
                    entry["tensors"][name] = {
                        "shape": [int(dim) for dim in tensor.shape],
                        "dtype": str(tensor.dtype),
                        "sha256": tensor_sha256(tensor.detach().cpu()),
                    }
            layers.append(entry)
        return {
            "schema": "nf_sf_mcp1_pristine_crossattn_fingerprint_v1",
            "layers": layers,
        }

    def manifest_record(self) -> dict[str, Any]:
        return {
            "mode": "probe_local_pristine_history_plus_joint_write_window",
            "kv_write_set": self.kv_write_set,
            "pristine_history_kv_fingerprint_sha256": (
                self.reference_kv_fingerprint_sha256
            ),
            "pristine_crossattn_cache_fingerprint_sha256": (
                self.reference_crossattn_fingerprint_sha256
            ),
            "phase_counts": dict(sorted(self._phase_counts.items())),
            "verified_every_state": True,
            "verified_every_optimizer_step": True,
        }

    def _saved_kv_payload(self) -> dict[str, Any]:
        layers = []
        for saved in self._kv_layers:
            entry: dict[str, Any] = {
                "layer": int(saved["layer"]),
                "global_end_index": int(saved["global_end_index"]),
                "local_end_index": int(saved["local_end_index"]),
                "semantic_windows": saved["semantic_windows"],
                "covered_windows": saved["covered_windows"],
                "tensors": {},
            }
            for name, saved_slices in saved["tensors"].items():
                records = []
                for window, tensor in zip(saved["covered_windows"], saved_slices):
                    records.append(
                        {
                            "start": int(window["start"]),
                            "end": int(window["end"]),
                            "shape": [int(dim) for dim in tensor.shape],
                            "dtype": str(tensor.dtype),
                            "sha256": tensor_sha256(tensor),
                        }
                    )
                entry["tensors"][name] = records
            layers.append(entry)
        return {
            "schema": "nf_sf_mcp1_pristine_kv_window_fingerprint_v1",
            "layers": layers,
            "kv_write_set_fingerprint_sha256": self.kv_write_set[
                "fingerprint_sha256"
            ],
        }

    def _saved_crossattn_payload(self) -> dict[str, Any]:
        layers = []
        for saved in self._crossattn_layers:
            entry: dict[str, Any] = {
                "layer": int(saved["layer"]),
                "is_init": bool(saved["is_init"]),
                "tensors": {},
            }
            for name, tensor in saved["tensors"].items():
                entry["tensors"][name] = {
                    "shape": [int(dim) for dim in tensor.shape],
                    "dtype": str(tensor.dtype),
                    "sha256": tensor_sha256(tensor),
                }
            layers.append(entry)
        return {
            "schema": "nf_sf_mcp1_pristine_crossattn_fingerprint_v1",
            "layers": layers,
        }


def build_pristine_cache_snapshot(
    *,
    runtime: deployment.DeploymentRuntime,
    history_recache: Mapping[str, Any],
) -> MCP1ProbePristineCacheSnapshot:
    return MCP1ProbePristineCacheSnapshot.capture(
        runtime=runtime,
        history_recache=history_recache,
    )


def compute_mcp1_probe_kv_write_set(
    runtime: deployment.DeploymentRuntime,
) -> dict[str, Any]:
    frame_seq_length = int(runtime.frame_seq_length)
    chunk_frames = int(runtime.num_frame_per_block)
    if frame_seq_length <= 0 or chunk_frames <= 0:
        raise RuntimeError("MCP1 probe requires positive frame/token geometry")
    history_end = (HISTORY_CHUNK_INDEX + 1) * chunk_frames * frame_seq_length
    current_start = CURRENT_CHUNK_INDEX * chunk_frames * frame_seq_length
    current_tokens = chunk_frames * frame_seq_length
    cache_start = current_start
    cache_end = cache_start + current_tokens
    future_start = FUTURE_CHUNK_INDEX * chunk_frames * frame_seq_length
    if current_start != history_end:
        raise RuntimeError("MCP1 probe current_start must equal recached history end")

    layers = []
    for layer_index, layer in enumerate(runtime.kv_cache):
        global_before = _cache_index_int(layer.get("global_end_index"))
        local_before = _cache_index_int(layer.get("local_end_index"))
        if global_before != history_end or local_before != history_end:
            raise RuntimeError(
                "MCP1 probe KV write-set requires recached history chunk0 "
                f"index={history_end}, got layer {layer_index} "
                f"global={global_before} local={local_before}"
            )
        k_tensor = layer.get("k")
        v_tensor = layer.get("v")
        if not torch.is_tensor(k_tensor) or not torch.is_tensor(v_tensor):
            raise RuntimeError("MCP1 probe KV write-set requires k/v tensors")
        if int(k_tensor.shape[1]) != int(v_tensor.shape[1]):
            raise RuntimeError("MCP1 probe KV k/v capacities differ")
        capacity = int(k_tensor.shape[1])
        local_attn_size, sink_size = _attention_cache_config(runtime, layer_index)
        eviction_possible = (
            local_attn_size is not None
            and int(local_attn_size) != -1
            and cache_end > global_before
            and current_tokens + local_before > capacity
        )
        if local_attn_size is None and current_tokens + local_before > capacity:
            raise RuntimeError(
                "MCP1 probe cannot prove KV write-set when cache capacity is "
                "insufficient and attention local_attn_size is unknown"
            )

        write_windows: list[dict[str, Any]] = []
        if eviction_possible:
            sink_tokens = int(sink_size or 0) * frame_seq_length
            num_evicted = current_tokens + local_before - capacity
            num_rolled = local_before - num_evicted - sink_tokens
            if num_rolled < 0:
                raise RuntimeError("MCP1 probe local-attention roll window is invalid")
            if num_rolled:
                write_windows.append(
                    {
                        "role": "local_attention_roll_window",
                        "start": sink_tokens,
                        "end": sink_tokens + num_rolled,
                    }
                )
            local_end = local_before + cache_end - global_before - num_evicted
            local_start = local_end - current_tokens
        else:
            local_end = local_before + cache_end - global_before
            local_start = local_end - current_tokens

        write_windows.append(
            {
                "role": "current_chunk1_main_kv_write_window",
                "start": int(local_start),
                "end": int(local_end),
            }
        )
        for window in write_windows:
            start = int(window["start"])
            end = int(window["end"])
            if not (0 <= start <= end <= capacity):
                raise RuntimeError("MCP1 probe KV write window is out of capacity")
        layers.append(
            {
                "layer": int(layer_index),
                "capacity_tokens": capacity,
                "global_end_index_before": global_before,
                "local_end_index_before": local_before,
                "history_active_prefix_end": history_end,
                "current_start_tokens": current_start,
                "current_tokens": current_tokens,
                "cache_start_tokens": cache_start,
                "cache_end_tokens": cache_end,
                "future_start_tokens": future_start,
                "mcp_future_writes_runtime_kv": False,
                "local_attn_size": (
                    None if local_attn_size is None else int(local_attn_size)
                ),
                "sink_size_frames": None if sink_size is None else int(sink_size),
                "eviction_possible": bool(eviction_possible),
                "write_windows": write_windows,
                "expected_global_end_index_after_forward": int(cache_end),
                "expected_local_end_index_after_forward": int(local_end),
            }
        )

    payload = {
        "schema": "nf_sf_mcp1_probe_joint_forward_kv_write_set_v1",
        "history_chunk_index": HISTORY_CHUNK_INDEX,
        "current_chunk_index": CURRENT_CHUNK_INDEX,
        "future_chunk_index": FUTURE_CHUNK_INDEX,
        "chunk_frames": chunk_frames,
        "frame_seq_length": frame_seq_length,
        "history_end_tokens": history_end,
        "current_start_tokens": current_start,
        "current_tokens": current_tokens,
        "future_start_tokens": future_start,
        "causal_attention_write_formula": {
            "source": "wan.modules.causal_model.CausalWanSelfAttention.forward",
            "cache_start": "current_start because _call_joint_depth1 omits cache_start",
            "cache_end": "cache_start + roped_query.shape[1]",
            "non_eviction_write": (
                "local_start_index:local_end_index where local_end_index = "
                "local_before + cache_end - global_before"
            ),
            "mcp_future_runtime_kv_write": False,
        },
        "layers": layers,
    }
    return {
        **payload,
        "fingerprint_sha256": deployment.canonical_json_sha256(payload),
    }


def _attention_cache_config(
    runtime: deployment.DeploymentRuntime,
    layer_index: int,
) -> tuple[int | None, int | None]:
    blocks = getattr(getattr(runtime.generator, "model", None), "blocks", None)
    if blocks is None or int(layer_index) >= len(blocks):
        return None, None
    self_attn = getattr(blocks[int(layer_index)], "self_attn", None)
    if self_attn is None:
        return None, None
    local_attn_size = getattr(self_attn, "local_attn_size", None)
    sink_size = getattr(self_attn, "sink_size", 0)
    return (
        None if local_attn_size is None else int(local_attn_size),
        None if sink_size is None else int(sink_size),
    )


def _merge_windows(windows: Sequence[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    normalized = sorted(
        (int(start), int(end))
        for start, end in windows
        if int(end) > int(start)
    )
    if not normalized:
        return ()
    merged: list[tuple[int, int]] = [normalized[0]]
    for start, end in normalized[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return tuple(merged)


def _set_cache_index(layer: Mapping[str, Any], key: str, value: int) -> None:
    target = layer.get(key)
    if torch.is_tensor(target):
        target.fill_(int(value))
    else:
        layer[key] = int(value)  # type: ignore[index]


def _cache_index_int(value: Any) -> int:
    if torch.is_tensor(value):
        return int(value.detach().cpu().reshape(-1)[0].item())
    return int(value)


def build_memorization_flow_scheduler(
    *,
    shift: float,
    device: torch.device | str,
) -> FlowMatchScheduler:
    scheduler = FlowMatchScheduler(
        shift=float(shift),
        sigma_min=0.0,
        extra_one_step=True,
    )
    scheduler.set_timesteps(DEFAULT_NUM_TRAIN_TIMESTEPS, training=True)
    scheduler.sigmas = scheduler.sigmas.to(device)
    scheduler.timesteps = scheduler.timesteps.to(device)
    return scheduler


def build_mcp1_memorization_states(
    *,
    source_noise: torch.Tensor,
    teacher_target: torch.Tensor,
    main_scheduler: Any,
    mcp_scheduler: Any,
    noise_seed: int = DEFAULT_NOISE_SEED,
    raw_timesteps: Sequence[int] = RAW_TIMESTEPS,
    noise_realizations_per_raw: int = NOISE_REALIZATIONS_PER_RAW_TIMESTEP,
) -> tuple[tuple[MCP1MemorizationState, ...], dict[str, Any]]:
    flow_audit._validate_source_and_teacher(source_noise, teacher_target)
    if int(noise_realizations_per_raw) != NOISE_REALIZATIONS_PER_RAW_TIMESTEP:
        raise ValueError("MCP1 memorization probe requires exactly four noise realizations")
    raw_values = tuple(int(value) for value in raw_timesteps)
    if raw_values != RAW_TIMESTEPS:
        raise ValueError("MCP1 memorization probe raw timestep grid is locked")
    teacher_chunk1 = flow_audit._chunk(teacher_target, CURRENT_CHUNK_INDEX)
    teacher_chunk2 = flow_audit._chunk(teacher_target, FUTURE_CHUNK_INDEX)
    source_chunk1 = flow_audit._chunk(source_noise, CURRENT_CHUNK_INDEX)
    source_chunk2 = flow_audit._chunk(source_noise, FUTURE_CHUNK_INDEX)
    fixed_context = {
        "schema": MCP1_MEMORIZATION_STATE_SCHEMA,
        "source_noise_sha256": tensor_sha256(source_noise.detach().cpu()),
        "teacher_target_sha256": tensor_sha256(teacher_target.detach().cpu()),
        "teacher_chunk0_sha256": tensor_sha256(
            flow_audit._chunk(teacher_target, HISTORY_CHUNK_INDEX).detach().cpu()
        ),
        "teacher_chunk1_sha256": tensor_sha256(teacher_chunk1.detach().cpu()),
        "teacher_chunk2_sha256": tensor_sha256(teacher_chunk2.detach().cpu()),
        "source_noise_chunk1_sha256": tensor_sha256(source_chunk1.detach().cpu()),
        "source_noise_chunk2_sha256": tensor_sha256(source_chunk2.detach().cpu()),
        "noise_seed": int(noise_seed),
        "raw_timesteps": [int(value) for value in raw_values],
        "noise_realizations_per_raw": int(noise_realizations_per_raw),
        "main_shift": DEFAULT_S_MAIN,
        "mcp_shift": DEFAULT_S_MCP,
        "num_train_timesteps": DEFAULT_NUM_TRAIN_TIMESTEPS,
        "scheduler_semantics": {
            "class": "utils.scheduler.FlowMatchScheduler",
            "sigma_min": 0.0,
            "extra_one_step": True,
            "raw_to_warped": "utils.nf_sf_tensors.flow_match_shift_timesteps",
            "state": "FlowMatchScheduler.add_noise",
            "target": "FlowMatchScheduler.training_target",
        },
        "noise_realization_contract": {
            "index0": "stored fixed validation source_noise chunks",
            "index1_to_3": MCP1_MEMORIZATION_NOISE_SCHEMA,
            "global_rng_mutation_allowed": False,
        },
    }
    states: list[MCP1MemorizationState] = []
    for raw_timestep in raw_values:
        main_t = _warp_raw_timestep(raw_timestep, shift=DEFAULT_S_MAIN)
        mcp_t = _warp_raw_timestep(raw_timestep, shift=DEFAULT_S_MCP)
        for noise_index in range(int(noise_realizations_per_raw)):
            current_noise, current_noise_record = _noise_for_realization(
                template=source_chunk1,
                source_noise=source_noise,
                teacher_target=teacher_target,
                raw_timestep=raw_timestep,
                noise_index=noise_index,
                role="current_chunk1",
                base_seed=int(noise_seed),
            )
            future_noise, future_noise_record = _noise_for_realization(
                template=source_chunk2,
                source_noise=source_noise,
                teacher_target=teacher_target,
                raw_timestep=raw_timestep,
                noise_index=noise_index,
                role="future_chunk2",
                base_seed=int(noise_seed),
            )
            main_timestep = flow_audit._timestep(main_t, teacher_chunk1)
            mcp_timestep = flow_audit._timestep(mcp_t, teacher_chunk2)
            current_state = flow_audit._add_noise_chunk(
                main_scheduler,
                clean=teacher_chunk1,
                noise=current_noise,
                timestep=main_timestep,
            )
            future_state = flow_audit._add_noise_chunk(
                mcp_scheduler,
                clean=teacher_chunk2,
                noise=future_noise,
                timestep=mcp_timestep,
            )
            main_target = flow_audit._training_target_chunk(
                main_scheduler,
                clean=teacher_chunk1,
                noise=current_noise,
                timestep=main_timestep,
            )
            mcp_target = flow_audit._training_target_chunk(
                mcp_scheduler,
                clean=teacher_chunk2,
                noise=future_noise,
                timestep=mcp_timestep,
            )
            state_id = f"raw{int(raw_timestep):03d}_noise{int(noise_index)}"
            provenance = {
                "state_id": state_id,
                "raw_timestep": int(raw_timestep),
                "noise_index": int(noise_index),
                "main_warped_timestep": float(main_t),
                "mcp_warped_timestep": float(mcp_t),
                "current_noise": current_noise_record,
                "future_noise": future_noise_record,
                "current_state": _tensor_record(current_state),
                "future_state": _tensor_record(future_state),
                "main_target": _tensor_record(main_target),
                "mcp_target": _tensor_record(mcp_target),
                "main_timestep_sha256": tensor_sha256(main_timestep.detach().cpu()),
                "mcp_timestep_sha256": tensor_sha256(mcp_timestep.detach().cpu()),
            }
            states.append(
                MCP1MemorizationState(
                    state_id=state_id,
                    raw_timestep=int(raw_timestep),
                    noise_index=int(noise_index),
                    main_warped_timestep=float(main_t),
                    mcp_warped_timestep=float(mcp_t),
                    current_state=current_state.detach().clone(),
                    future_state=future_state.detach().clone(),
                    current_noise=current_noise.detach().clone(),
                    future_noise=future_noise.detach().clone(),
                    main_target=main_target.detach().clone(),
                    mcp_target=mcp_target.detach().clone(),
                    main_timestep=main_timestep.detach().clone(),
                    mcp_timestep=mcp_timestep.detach().clone(),
                    provenance=provenance,
                )
            )
    provenance_records = [state.provenance for state in states]
    collection = {
        **fixed_context,
        "state_count": len(states),
        "state_ids": [state.state_id for state in states],
        "states": provenance_records,
    }
    collection["state_collection_fingerprint_sha256"] = deployment.canonical_json_sha256(
        collection
    )
    if len(states) != len(RAW_TIMESTEPS) * NOISE_REALIZATIONS_PER_RAW_TIMESTEP:
        raise RuntimeError("MCP1 memorization probe state count must be 16")
    return tuple(states), collection


def configure_stage_a_trainable_parameters(
    generator: Any,
    *,
    lr: float = DEFAULT_OPTIMIZER_LR,
) -> StageAParamSelection:
    if getattr(generator, "mcp", None) is None:
        raise RuntimeError("Stage A requires attached MCP modules")
    groups = collect_nf_sf_parameter_groups(generator)
    required = set(MAIN_GROUPS + ALLOWED_STAGE_A_GROUPS + FORBIDDEN_MCP_GROUPS)
    missing = sorted(required.difference(groups.keys()))
    if missing:
        raise RuntimeError(f"Stage A parameter group(s) missing: {missing}")
    group_param_ids = {
        group_name: tuple(id(param) for _, param in named_params)
        for group_name, named_params in groups.items()
    }
    allowed_param_ids_list = [
        param_id
        for group_name in ALLOWED_STAGE_A_GROUPS
        for param_id in group_param_ids[group_name]
    ]
    if len(set(allowed_param_ids_list)) != len(allowed_param_ids_list):
        raise RuntimeError("Stage A allowed parameter object id appears more than once")
    allowed_ids = set(allowed_param_ids_list)
    main_ids = {
        param_id
        for group_name in MAIN_GROUPS
        for param_id in group_param_ids[group_name]
    }
    forbidden_ids = {
        param_id
        for group_name in FORBIDDEN_MCP_GROUPS
        for param_id in group_param_ids[group_name]
    }
    if allowed_ids & main_ids:
        raise RuntimeError("Stage A allowed parameters alias Main/shared patch embedding")
    if allowed_ids & forbidden_ids:
        raise RuntimeError("Stage A allowed parameters alias MCP depth2/3")
    generator.requires_grad_(False)
    trainable_named: list[tuple[str, torch.nn.Parameter]] = []
    optimizer_groups: list[dict[str, Any]] = []
    group_records: dict[str, Any] = {}
    for group_name, named_params in groups.items():
        allow = group_name in ALLOWED_STAGE_A_GROUPS
        params = []
        for name, param in named_params:
            param.requires_grad_(allow)
            if allow:
                trainable_named.append((name, param))
                params.append(param)
        if allow:
            if not params:
                raise RuntimeError(f"Stage A allowed group {group_name} has no parameters")
            optimizer_groups.append(
                {
                    "name": group_name,
                    "params": params,
                    "lr": float(lr),
                }
            )
        group_records[group_name] = {
            "parameter_names": [name for name, _ in named_params],
            "tensor_count": len(named_params),
            "parameter_count": int(sum(param.numel() for _, param in named_params)),
            "trainable_tensor_count": int(
                sum(1 for _, param in named_params if param.requires_grad)
            ),
            "trainable_parameter_count": int(
                sum(param.numel() for _, param in named_params if param.requires_grad)
            ),
            "requires_grad": bool(named_params)
            and all(param.requires_grad for _, param in named_params),
            "in_optimizer": bool(allow),
        }
    projection_names = [
        name
        for name, _ in groups["mcp_depth1"]
        if ".proj." in name or name.endswith(".proj.weight") or name.endswith(".proj.bias")
    ]
    if not projection_names:
        raise RuntimeError("Stage A could not identify MCP depth1 projection parameters")
    assert_stage_a_fail_closed(generator)
    optimizer_param_ids_list = [
        id(param) for group in optimizer_groups for param in group["params"]
    ]
    if len(set(optimizer_param_ids_list)) != len(optimizer_param_ids_list):
        raise RuntimeError("Stage A optimizer parameter object id appears more than once")
    optimizer_param_ids = set(optimizer_param_ids_list)
    expected_ids = {id(param) for _, param in trainable_named}
    if optimizer_param_ids != expected_ids:
        raise RuntimeError("Stage A optimizer parameter set mismatch")
    summary = {
        "schema": "nf_sf_mcp1_stage_a_trainable_parameter_contract_v1",
        "stage": "A",
        "main_backbone_frozen": True,
        "trainable_groups": list(ALLOWED_STAGE_A_GROUPS),
        "frozen_main_groups": list(MAIN_GROUPS),
        "frozen_forbidden_mcp_groups": list(FORBIDDEN_MCP_GROUPS),
        "component_contract": {
            "fusion": "mcp.fusion.*",
            "projection": "mcp.mcp_modules.0.proj.*",
            "mcp_depth1_module": "mcp.mcp_modules.0.*",
            "patch_embedding": "frozen shared Main projection; not trainable in Stage A",
        },
        "projection_parameter_names": projection_names,
        "trainable_parameter_names": [name for name, _ in trainable_named],
        "trainable_tensor_count": len(trainable_named),
        "trainable_parameter_count": int(sum(param.numel() for _, param in trainable_named)),
        "parameter_identity_contract": {
            "allowed_unique_id_count": len(allowed_ids),
            "allowed_named_parameter_count": len(allowed_param_ids_list),
            "optimizer_unique_id_count": len(optimizer_param_ids),
            "optimizer_named_parameter_count": len(optimizer_param_ids_list),
            "allowed_ids_equal_optimizer_ids": True,
            "allowed_ids_intersect_main_or_patch_embedding": False,
            "allowed_ids_intersect_mcp_depth2_or_depth3": False,
        },
        "group_records": group_records,
    }
    return StageAParamSelection(
        trainable_named_parameters=tuple(trainable_named),
        optimizer_param_groups=tuple(optimizer_groups),
        summary=summary,
    )


def run_mcp1_memorization_probe(
    *,
    runtime_factory: Callable[[], deployment.DeploymentRuntime],
    source_noise: torch.Tensor,
    teacher_target: torch.Tensor,
    teacher_payload: Mapping[str, Any],
    conditional_dict: Mapping[str, Any],
    checkpoint_summary: Mapping[str, Any],
    common_inputs: Mapping[str, Any],
    common_inputs_fingerprint_sha256: str,
    runtime_git_sha: str,
    training_checkpoint_git_sha: str,
    main_scheduler: Any | None = None,
    mcp_scheduler: Any | None = None,
    optimizer_steps: int = DEFAULT_OPTIMIZER_STEPS,
    optimizer_lr: float = DEFAULT_OPTIMIZER_LR,
    log_interval: int = DEFAULT_LOG_INTERVAL,
    noise_seed: int = DEFAULT_NOISE_SEED,
) -> MCP1MemorizationProbeResult:
    _validate_optimizer_contract_inputs(
        optimizer_steps=optimizer_steps,
        optimizer_lr=optimizer_lr,
        log_interval=log_interval,
    )
    flow_audit._validate_source_and_teacher(source_noise, teacher_target)
    device = teacher_target.device
    main_scheduler = main_scheduler or build_memorization_flow_scheduler(
        shift=DEFAULT_S_MAIN,
        device=device,
    )
    mcp_scheduler = mcp_scheduler or build_memorization_flow_scheduler(
        shift=DEFAULT_S_MCP,
        device=device,
    )
    states, state_provenance = build_mcp1_memorization_states(
        source_noise=source_noise,
        teacher_target=teacher_target,
        main_scheduler=main_scheduler,
        mcp_scheduler=mcp_scheduler,
        noise_seed=int(noise_seed),
    )
    runtime = runtime_factory()
    generator = runtime.generator
    assert_gradient_checkpointing_disabled(generator)
    selection = configure_stage_a_trainable_parameters(
        generator,
        lr=float(optimizer_lr),
    )
    allowed_ids = allowed_stage_a_param_ids(selection)
    main_before = parameter_sha256_report(generator, groups=MAIN_GROUPS)
    trainable_before = _trainable_parameter_snapshot(selection.trainable_named_parameters)
    optimizer = torch.optim.AdamW(
        list(selection.optimizer_param_groups),
        lr=float(optimizer_lr),
        weight_decay=0.0,
    )
    validate_optimizer_excludes_main(
        generator,
        optimizer,
        allowed_param_ids=allowed_ids,
    )
    rng_plan = deployment.build_absolute_chunk_rng_plan(
        source_noise=source_noise,
        rollout_seed=int(teacher_payload["rollout_seed"]),
        num_denoising_steps=len(deployment.RAW_DEPLOYMENT_SCHEDULE),
        chunk_frames=deployment.FULL_SEQUENCE_CHUNK_FRAMES,
    )
    generator.eval()
    with torch.no_grad():
        history_recache = flow_audit._recache_teacher_history0(
            runtime=runtime,
            source_noise=source_noise,
            teacher_target=teacher_target,
            conditional_dict=conditional_dict,
            rng_plan=rng_plan,
        )
    assert_stage_a_fail_closed(generator)
    pristine_cache = build_pristine_cache_snapshot(
        runtime=runtime,
        history_recache=history_recache,
    )
    with forbid_mcp_depth23_calls(generator) as depth_guard:
        generator.eval()
        initial_records, initial_flows = evaluate_probe_states(
            runtime=runtime,
            conditional_dict=conditional_dict,
            states=states,
            no_grad=True,
            pristine_cache_snapshot=pristine_cache,
        )
        initial_stats = summarize_flow_mse(initial_records)
        loss_curve: list[dict[str, Any]] = []
        logged_losses: list[dict[str, Any]] = []
        last_gradient_audit: dict[str, Any] | None = None
        for step in range(1, int(optimizer_steps) + 1):
            generator.train()
            optimizer.zero_grad(set_to_none=True)
            step_loss, last_gradient_audit = _training_step_backward(
                runtime=runtime,
                conditional_dict=conditional_dict,
                states=states,
                selection=selection,
                pristine_cache_snapshot=pristine_cache,
            )
            assert_stage_a_fail_closed(generator)
            assert_no_frozen_stage_a_gradients(generator)
            validate_optimizer_excludes_main(
                generator,
                optimizer,
                allowed_param_ids=allowed_ids,
            )
            optimizer.step()
            pristine_cache.restore_and_verify(
                runtime,
                phase="post_optimizer_step",
                optimizer_step=int(step),
            )
            loss_value = float(step_loss.detach().float().item())
            record = {"step": int(step), "loss": loss_value}
            loss_curve.append(record)
            if step == 1 or step % int(log_interval) == 0 or step == int(optimizer_steps):
                logged_losses.append(record)
        optimizer.zero_grad(set_to_none=True)
        generator.eval()
        final_records, final_flows = evaluate_probe_states(
            runtime=runtime,
            conditional_dict=conditional_dict,
            states=states,
            no_grad=True,
            pristine_cache_snapshot=pristine_cache,
        )
    final_stats = summarize_flow_mse(final_records)
    status = evaluate_memorization_status(
        initial_mean_mse=initial_stats["mean_mse"],
        initial_max_mse=initial_stats["max_mse"],
        final_mean_mse=final_stats["mean_mse"],
        final_max_mse=final_stats["max_mse"],
    )
    trainable_delta = trainable_parameter_delta_report(
        selection.trainable_named_parameters,
        trainable_before,
    )
    main_after = parameter_sha256_report(generator, groups=MAIN_GROUPS)
    unchanged_proof = compare_parameter_sha256_reports(main_before, main_after)
    if not bool(unchanged_proof["all_sha256_exact_match"]):
        raise RuntimeError("Stage A Main parameter exact-unchanged proof failed")
    manifest = {
        "schema": MCP1_MEMORIZATION_PROBE_SCHEMA,
        "status": "PASS",
        "diagnostic_result": status["status"],
        "diagnostic_only": True,
        "non_deployable": True,
        "non_canonical": True,
        "training_eligible": False,
        "canonical_training_eligible": False,
        "canonical_deployment_eligible": False,
        "checkpoint_output": {
            "written": False,
            "resume_allowed": False,
            "probe_checkpointing_allowed": False,
        },
        "runtime_git_sha": str(runtime_git_sha),
        "training_checkpoint_git_sha": str(training_checkpoint_git_sha),
        "checkpoint": dict(checkpoint_summary),
        "common_inputs": dict(common_inputs),
        "common_inputs_fingerprint_sha256": str(common_inputs_fingerprint_sha256),
        "fixed_input_target_provenance": state_provenance,
        "history_recache": history_recache,
        "history_kv_content_fingerprint_sha256": history_recache[
            "history_kv_fingerprint_sha256"
        ],
        "crossattn_cache_fingerprint_sha256": history_recache[
            "crossattn_cache_fingerprint_sha256"
        ],
        "pristine_history_kv_fingerprint_sha256": (
            pristine_cache.reference_kv_fingerprint_sha256
        ),
        "pristine_crossattn_cache_fingerprint_sha256": (
            pristine_cache.reference_crossattn_fingerprint_sha256
        ),
        "kv_isolation_mode": "probe_local_pristine_history_plus_joint_write_window",
        "kv_isolation_verified_every_state": True,
        "kv_isolation_verified_every_optimizer_step": True,
        "kv_isolation": pristine_cache.manifest_record(),
        "stage_a_trainable_parameter_selection": selection.summary,
        "optimizer_contract": {
            "class": optimizer.__class__.__name__,
            "steps": int(optimizer_steps),
            "lr": float(optimizer_lr),
            "weight_decay": 0.0,
            "betas": [float(value) for value in optimizer.defaults.get("betas", ())],
            "eps": float(optimizer.defaults.get("eps", 0.0)),
            "log_interval": int(log_interval),
            "auto_lr_change_allowed": False,
            "resume_allowed": False,
            "checkpointing_allowed": False,
            "loss": "MCP1 exact flow MSE on the 16 fixed states only",
        },
        "optimization_scope": {
            "stage": "A",
            "main_backbone_frozen": True,
            "mcp_depths_used": [1],
            "mcp_depth2_called": False,
            "mcp_depth3_called": False,
            "mcp_depth_call_guard": dict(depth_guard),
            "serial_rollout": False,
            "verifier": False,
            "dmd": False,
            "self_rollout": False,
            "new_loss_added": False,
            "architecture_changed": False,
        },
        "initial_per_state_flow_mse": initial_records,
        "initial_mse": initial_stats,
        "loss_curve": loss_curve,
        "logged_losses": logged_losses,
        "final_per_state_flow_mse": final_records,
        "final_mse": final_stats,
        "relative_reduction": status["relative_reduction"],
        "memorization_thresholds": status["thresholds"],
        "parameter_delta_norm": trainable_delta,
        "last_gradient_audit": last_gradient_audit,
        "main_parameters_exact_unchanged_proof": unchanged_proof,
        "interpretation_contract": {
            "strong_memorization_support": (
                "final mean MSE <= 5% of initial mean and final max MSE <= "
                "10% of initial max"
            ),
            "otherwise": INSUFFICIENT_MEMORIZATION,
            "does_not_claim_model_fixed": True,
            "stage_b_not_run": True,
        },
    }
    validate_mcp1_memorization_manifest(manifest)
    tensors = {
        "schema": MCP1_MEMORIZATION_TENSOR_SCHEMA,
        "states": [
            {
                "state_id": state.state_id,
                "raw_timestep": int(state.raw_timestep),
                "noise_index": int(state.noise_index),
                "current_state": state.current_state.detach().cpu(),
                "future_state": state.future_state.detach().cpu(),
                "current_noise": state.current_noise.detach().cpu(),
                "future_noise": state.future_noise.detach().cpu(),
                "main_target": state.main_target.detach().cpu(),
                "mcp_target": state.mcp_target.detach().cpu(),
                "main_timestep": state.main_timestep.detach().cpu(),
                "mcp_timestep": state.mcp_timestep.detach().cpu(),
            }
            for state in states
        ],
        "initial_mcp_predicted_flows": initial_flows,
        "final_mcp_predicted_flows": final_flows,
    }
    return MCP1MemorizationProbeResult(manifest=manifest, tensors=tensors)


def evaluate_probe_states(
    *,
    runtime: deployment.DeploymentRuntime,
    conditional_dict: Mapping[str, Any],
    states: Sequence[MCP1MemorizationState],
    no_grad: bool,
    pristine_cache_snapshot: MCP1ProbePristineCacheSnapshot | None = None,
) -> tuple[list[dict[str, Any]], list[torch.Tensor]]:
    if pristine_cache_snapshot is not None and not bool(no_grad):
        raise RuntimeError("use _training_step_backward for grad-enabled isolated forwards")
    records: list[dict[str, Any]] = []
    flows: list[torch.Tensor] = []
    context = torch.no_grad() if no_grad else _null_context()
    with context:
        for state in states:
            if pristine_cache_snapshot is None:
                _, _, mcp_flow, call_record = flow_audit._call_joint_depth1(
                    runtime=runtime,
                    conditional_dict=conditional_dict,
                    current_state=state.current_state,
                    future_state=state.future_state,
                    current_start_frame=(
                        CURRENT_CHUNK_INDEX * deployment.FULL_SEQUENCE_CHUNK_FRAMES
                    ),
                    future_start_frame=(
                        FUTURE_CHUNK_INDEX * deployment.FULL_SEQUENCE_CHUNK_FRAMES
                    ),
                    main_timestep_value=float(state.main_warped_timestep),
                    mcp_timestep_value=float(state.mcp_warped_timestep),
                )
            else:
                _, _, mcp_flow, call_record = call_isolated_mcp1_joint_forward(
                    runtime=runtime,
                    conditional_dict=conditional_dict,
                    state=state,
                    pristine_cache_snapshot=pristine_cache_snapshot,
                    pre_forward_phase="eval_pre_forward",
                    restore_after_forward=True,
                    post_forward_phase="eval_post_forward",
                )
            loss = F.mse_loss(mcp_flow.float(), state.mcp_target.float(), reduction="mean")
            if not bool(torch.isfinite(loss.detach()).all().item()):
                raise RuntimeError("MCP1 memorization flow MSE is non-finite")
            records.append(
                {
                    "state_id": state.state_id,
                    "raw_timestep": int(state.raw_timestep),
                    "noise_index": int(state.noise_index),
                    "mse": float(loss.detach().float().item()),
                    "predicted_flow_sha256": tensor_sha256(mcp_flow.detach().cpu()),
                    "target_flow_sha256": tensor_sha256(state.mcp_target.detach().cpu()),
                    "current_state_sha256": tensor_sha256(state.current_state.detach().cpu()),
                    "future_state_sha256": tensor_sha256(state.future_state.detach().cpu()),
                    "joint_forward_rng": call_record["joint_forward_rng"],
                }
            )
            flows.append(mcp_flow.detach().cpu())
    return records, flows


def call_isolated_mcp1_joint_forward(
    *,
    runtime: deployment.DeploymentRuntime,
    conditional_dict: Mapping[str, Any],
    state: MCP1MemorizationState,
    pristine_cache_snapshot: MCP1ProbePristineCacheSnapshot,
    pre_forward_phase: str,
    restore_after_forward: bool,
    post_forward_phase: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    pristine_cache_snapshot.restore_and_verify(
        runtime,
        phase=pre_forward_phase,
        state_id=state.state_id,
    )
    main_timestep = flow_audit._timestep(
        float(state.main_warped_timestep),
        state.current_state,
    )
    mcp_timestep = flow_audit._timestep(
        float(state.mcp_warped_timestep),
        state.future_state,
    )
    current_start_frame = CURRENT_CHUNK_INDEX * deployment.FULL_SEQUENCE_CHUNK_FRAMES
    future_start_frame = FUTURE_CHUNK_INDEX * deployment.FULL_SEQUENCE_CHUNK_FRAMES

    def call_joint():
        return runtime.generator(
            noisy_image_or_video=state.current_state,
            conditional_dict=dict(conditional_dict),
            timestep=main_timestep,
            kv_cache=runtime.kv_cache,
            crossattn_cache=runtime.crossattn_cache,
            current_start=int(current_start_frame) * int(runtime.frame_seq_length),
            mcp_future_noises=[state.future_state],
            mcp_future_start_frames=[int(future_start_frame)],
            mcp_timesteps=[mcp_timestep],
        )

    outputs, rng_guard = deployment._call_with_rng_guard(
        device=state.current_state.device,
        label="mcp1_memorization_joint_forward",
        fn=call_joint,
    )
    if not isinstance(outputs, (tuple, list)) or len(outputs) != 3:
        raise RuntimeError("MCP1 memorization joint forward must return MCP output")
    main_flow, main_x0 = deployment._unpack_main_outputs(outputs)
    _require_finite_tensor(main_flow, name="main_flow")
    _require_finite_tensor(main_x0, name="main_x0")
    mcp_outputs = outputs[2]
    if not isinstance(mcp_outputs, (tuple, list)) or len(mcp_outputs) != 1:
        raise RuntimeError("MCP1 memorization must request MCP depth1 only")
    mcp_flow = mcp_outputs[0]
    if not torch.is_tensor(mcp_flow):
        raise TypeError("MCP1 memorization MCP flow must be a tensor")
    _require_finite_tensor(mcp_flow, name="mcp_flow")
    if restore_after_forward:
        pristine_cache_snapshot.restore_and_verify(
            runtime,
            phase=post_forward_phase,
            state_id=state.state_id,
        )
    return main_flow, main_x0, mcp_flow, {
        "current_input": state.current_state.detach().clone(),
        "future_input": state.future_state.detach().clone(),
        "joint_forward_rng": rng_guard,
    }


def summarize_flow_mse(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(records) != len(RAW_TIMESTEPS) * NOISE_REALIZATIONS_PER_RAW_TIMESTEP:
        raise RuntimeError("MCP1 memorization MSE summary requires 16 states")
    values = [float(record["mse"]) for record in records]
    if not all(math.isfinite(value) for value in values):
        raise RuntimeError("MCP1 memorization MSE summary found non-finite value")
    return {
        "mean_mse": float(sum(values) / len(values)),
        "max_mse": float(max(values)),
        "min_mse": float(min(values)),
        "state_count": len(values),
    }


def evaluate_memorization_status(
    *,
    initial_mean_mse: float,
    initial_max_mse: float,
    final_mean_mse: float,
    final_max_mse: float,
) -> dict[str, Any]:
    for name, value in (
        ("initial_mean_mse", initial_mean_mse),
        ("initial_max_mse", initial_max_mse),
        ("final_mean_mse", final_mean_mse),
        ("final_max_mse", final_max_mse),
    ):
        if not math.isfinite(float(value)) or float(value) < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")
    mean_threshold = 0.05 * float(initial_mean_mse)
    max_threshold = 0.10 * float(initial_max_mse)
    strong = (
        float(final_mean_mse) <= mean_threshold
        and float(final_max_mse) <= max_threshold
    )
    return {
        "status": STRONG_MEMORIZATION_SUPPORT if strong else INSUFFICIENT_MEMORIZATION,
        "thresholds": {
            "final_mean_mse_max": mean_threshold,
            "final_max_mse_max": max_threshold,
            "mean_fraction": 0.05,
            "max_fraction": 0.10,
        },
        "relative_reduction": {
            "mean": _relative_reduction(initial_mean_mse, final_mean_mse),
            "max": _relative_reduction(initial_max_mse, final_max_mse),
            "final_mean_fraction_of_initial": _fraction(final_mean_mse, initial_mean_mse),
            "final_max_fraction_of_initial": _fraction(final_max_mse, initial_max_mse),
        },
    }


def assert_stage_a_fail_closed(generator: Any) -> None:
    groups = collect_nf_sf_parameter_groups(generator)
    for group_name in MAIN_GROUPS:
        for name, param in groups.get(group_name, ()):
            if param.requires_grad:
                raise RuntimeError(
                    f"Stage A fail-closed: Main parameter requires_grad=True: {name}"
                )
    for group_name in FORBIDDEN_MCP_GROUPS:
        for name, param in groups.get(group_name, ()):
            if param.requires_grad:
                raise RuntimeError(
                    f"Stage A fail-closed: forbidden MCP parameter trainable: {name}"
                )
    allowed_ids = {
        id(param)
        for group_name in ALLOWED_STAGE_A_GROUPS
        for _, param in groups.get(group_name, ())
    }
    for name, param in generator.named_parameters():
        if param.requires_grad and id(param) not in allowed_ids:
            raise RuntimeError(
                "Stage A fail-closed: unexpected trainable parameter "
                f"{name}"
            )


def assert_gradient_checkpointing_disabled(generator: Any) -> None:
    candidates = (
        ("generator", generator),
        ("generator.model", getattr(generator, "model", None)),
        ("generator.backbone", getattr(generator, "backbone", None)),
    )
    for label, module in candidates:
        if module is not None and bool(getattr(module, "gradient_checkpointing", False)):
            raise RuntimeError(
                "MCP1 memorization probe refuses gradient_checkpointing=True "
                f"on {label}; KV side effects must not be recomputed during backward"
            )


def assert_no_main_gradients(generator: Any) -> None:
    groups = collect_nf_sf_parameter_groups(generator)
    for group_name in MAIN_GROUPS:
        for name, param in groups[group_name]:
            _assert_no_parameter_gradient(name, param, owner="Main")


def assert_no_frozen_stage_a_gradients(generator: Any) -> None:
    groups = collect_nf_sf_parameter_groups(generator)
    for group_name in MAIN_GROUPS + FORBIDDEN_MCP_GROUPS:
        owner = "Main" if group_name in MAIN_GROUPS else "MCP depth2/3"
        for name, param in groups[group_name]:
            _assert_no_parameter_gradient(name, param, owner=owner)


def audit_stage_a_allowed_gradients(
    selection: StageAParamSelection,
    generator: Any,
) -> dict[str, Any]:
    assert_no_frozen_stage_a_gradients(generator)
    records = []
    finite_nonzero = 0
    for name, param in selection.trainable_named_parameters:
        grad = param.grad
        if grad is None:
            records.append(
                {
                    "name": name,
                    "grad_present": False,
                    "finite": True,
                    "max_abs": 0.0,
                    "l2": 0.0,
                }
            )
            continue
        detached = grad.detach()
        finite = bool(torch.isfinite(detached).all().item())
        if not finite:
            raise RuntimeError(f"Stage A allowed gradient is non-finite: {name}")
        max_abs = float(detached.float().abs().max().item())
        l2 = float(detached.float().square().sum().item()) ** 0.5
        if max_abs > 0.0:
            finite_nonzero += 1
        records.append(
            {
                "name": name,
                "grad_present": True,
                "finite": True,
                "max_abs": max_abs,
                "l2": l2,
            }
        )
    if finite_nonzero <= 0:
        raise RuntimeError("Stage A fusion/MCP1 parameters received no nonzero gradient")
    return {
        "schema": "nf_sf_mcp1_stage_a_gradient_audit_v1",
        "allowed_parameter_count": len(records),
        "allowed_finite_nonzero_gradient_count": int(finite_nonzero),
        "main_and_mcp23_gradients_absent_or_zero": True,
        "records": records,
    }


def _assert_no_parameter_gradient(
    name: str,
    param: torch.nn.Parameter,
    *,
    owner: str,
) -> None:
    if param.grad is None:
        return
    grad = param.grad.detach()
    if bool(torch.isfinite(grad).all().item()) and float(
        grad.float().abs().max().item()
    ) == 0.0:
        return
    raise RuntimeError(f"Stage A optimizer step blocked: {owner} gradient present on {name}")


def allowed_stage_a_param_ids(selection: StageAParamSelection) -> set[int]:
    ids = [id(param) for _, param in selection.trainable_named_parameters]
    if len(set(ids)) != len(ids):
        raise RuntimeError("Stage A allowed parameter id list contains duplicates")
    return set(ids)


def validate_optimizer_excludes_main(
    generator: Any,
    optimizer: torch.optim.Optimizer,
    *,
    allowed_param_ids: set[int] | None = None,
) -> None:
    groups = collect_nf_sf_parameter_groups(generator)
    main_ids = {
        id(param)
        for group_name in MAIN_GROUPS
        for _, param in groups[group_name]
    }
    forbidden_mcp_ids = {
        id(param)
        for group_name in FORBIDDEN_MCP_GROUPS
        for _, param in groups[group_name]
    }
    optimizer_ids_list = [
        id(param)
        for group in optimizer.param_groups
        for param in group.get("params", ())
    ]
    if len(set(optimizer_ids_list)) != len(optimizer_ids_list):
        raise RuntimeError("Stage A optimizer contains duplicate parameter ids")
    optimizer_ids = set(optimizer_ids_list)
    if optimizer_ids & main_ids:
        raise RuntimeError("Stage A optimizer includes Main parameters")
    if optimizer_ids & forbidden_mcp_ids:
        raise RuntimeError("Stage A optimizer includes MCP depth2/3 parameters")
    if allowed_param_ids is not None and optimizer_ids != set(allowed_param_ids):
        raise RuntimeError("Stage A optimizer parameter ids differ from allowed ids")


@contextmanager
def forbid_mcp_depth23_calls(generator: Any):
    modules = getattr(getattr(generator, "mcp", None), "mcp_modules", None)
    if modules is None or len(modules) < 3:
        raise RuntimeError("Stage A requires MCP depth1/2/3 modules")
    call_counts = {"depth2": 0, "depth3": 0}
    handles = []

    def make_hook(depth_name: str):
        def hook(_module, _args):
            call_counts[depth_name] += 1
            raise RuntimeError(f"Stage A forbids {depth_name} forward calls")

        return hook

    for module_index, depth_name in ((1, "depth2"), (2, "depth3")):
        handles.append(modules[module_index].register_forward_pre_hook(make_hook(depth_name)))
    try:
        yield call_counts
    finally:
        for handle in handles:
            handle.remove()


def parameter_sha256_report(
    generator: Any,
    *,
    groups: Sequence[str],
) -> dict[str, Any]:
    all_groups = collect_nf_sf_parameter_groups(generator)
    records: dict[str, dict[str, Any]] = {}
    for group_name in groups:
        if group_name not in all_groups:
            raise RuntimeError(f"parameter group {group_name!r} missing")
        group_records = {}
        for name, param in all_groups[group_name]:
            tensor = param.detach().cpu()
            group_records[name] = {
                "sha256": tensor_sha256(tensor),
                "shape": [int(dim) for dim in tensor.shape],
                "dtype": str(tensor.dtype),
                "requires_grad": bool(param.requires_grad),
            }
        records[group_name] = group_records
    payload = {
        "groups": list(groups),
        "parameter_count": int(
            sum(
                len(records[group_name])
                for group_name in records
            )
        ),
        "parameters": records,
    }
    return {
        **payload,
        "fingerprint_sha256": deployment.canonical_json_sha256(payload),
    }


def compare_parameter_sha256_reports(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
) -> dict[str, Any]:
    mismatches: list[str] = []
    before_params = before.get("parameters", {})
    after_params = after.get("parameters", {})
    for group_name, group_records in before_params.items():
        after_group = after_params.get(group_name, {})
        for name, record in group_records.items():
            after_record = after_group.get(name)
            if not isinstance(after_record, Mapping):
                mismatches.append(name)
                continue
            if str(record.get("sha256")) != str(after_record.get("sha256")):
                mismatches.append(name)
    return {
        "checked_groups": list(before.get("groups", ())),
        "before_fingerprint_sha256": str(before.get("fingerprint_sha256")),
        "after_fingerprint_sha256": str(after.get("fingerprint_sha256")),
        "parameter_count": int(before.get("parameter_count", 0)),
        "all_sha256_exact_match": len(mismatches) == 0,
        "mismatch_parameter_names": mismatches,
        "proof_method": "per-parameter tensor SHA256 before/after plus optimizer exclusion",
    }


def trainable_parameter_delta_report(
    named_parameters: Sequence[tuple[str, torch.nn.Parameter]],
    before: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    total_sq = 0.0
    by_parameter = {}
    for name, param in named_parameters:
        if name not in before:
            raise RuntimeError(f"missing trainable snapshot for {name}")
        delta = param.detach().cpu().float() - before[name].float()
        norm = float(delta.square().sum().item()) ** 0.5
        total_sq += norm * norm
        by_parameter[name] = norm
    return {
        "aggregate_l2": float(total_sq ** 0.5),
        "by_parameter_l2": by_parameter,
        "parameter_count": len(by_parameter),
    }


def validate_mcp1_memorization_manifest(manifest: Mapping[str, Any]) -> None:
    if manifest.get("schema") != MCP1_MEMORIZATION_PROBE_SCHEMA:
        raise RuntimeError("MCP1 memorization manifest schema mismatch")
    for field in (
        "diagnostic_only",
        "non_deployable",
        "non_canonical",
    ):
        if manifest.get(field) is not True:
            raise RuntimeError(f"MCP1 memorization manifest must mark {field}=True")
    if manifest.get("canonical_training_eligible") is not False:
        raise RuntimeError("MCP1 memorization output must not be canonical training eligible")
    if manifest.get("canonical_deployment_eligible") is not False:
        raise RuntimeError("MCP1 memorization output must not be canonical deployment eligible")
    if manifest.get("kv_isolation_mode") != (
        "probe_local_pristine_history_plus_joint_write_window"
    ):
        raise RuntimeError("MCP1 memorization KV isolation mode missing")
    if manifest.get("kv_isolation_verified_every_state") is not True:
        raise RuntimeError("MCP1 memorization must verify KV isolation every state")
    if manifest.get("kv_isolation_verified_every_optimizer_step") is not True:
        raise RuntimeError(
            "MCP1 memorization must verify KV isolation every optimizer step"
        )
    for field in (
        "pristine_history_kv_fingerprint_sha256",
        "pristine_crossattn_cache_fingerprint_sha256",
    ):
        value = manifest.get(field)
        if not isinstance(value, str) or len(value) != 64:
            raise RuntimeError(f"MCP1 memorization manifest missing {field}")
    kv_isolation = manifest.get("kv_isolation")
    if not isinstance(kv_isolation, Mapping):
        raise RuntimeError("MCP1 memorization KV isolation evidence missing")
    checkpoint_output = manifest.get("checkpoint_output")
    if not isinstance(checkpoint_output, Mapping):
        raise RuntimeError("MCP1 memorization checkpoint output contract missing")
    if checkpoint_output.get("written") is not False:
        raise RuntimeError("MCP1 memorization probe must not write checkpoints")
    if checkpoint_output.get("resume_allowed") is not False:
        raise RuntimeError("MCP1 memorization probe must not allow resume")
    scope = manifest.get("optimization_scope")
    if not isinstance(scope, Mapping):
        raise RuntimeError("MCP1 memorization optimization scope missing")
    if scope.get("mcp_depths_used") != [1]:
        raise RuntimeError("MCP1 memorization must use depth1 only")
    if scope.get("mcp_depth2_called") is not False or scope.get("mcp_depth3_called") is not False:
        raise RuntimeError("MCP1 memorization must not call MCP2/3")
    fixed = manifest.get("fixed_input_target_provenance")
    if not isinstance(fixed, Mapping):
        raise RuntimeError("MCP1 memorization fixed provenance missing")
    if int(fixed.get("state_count", -1)) != 16:
        raise RuntimeError("MCP1 memorization fixed state count must be 16")
    initial = manifest.get("initial_per_state_flow_mse")
    final = manifest.get("final_per_state_flow_mse")
    if not isinstance(initial, Sequence) or len(initial) != 16:
        raise RuntimeError("MCP1 memorization initial per-state MSE must contain 16 records")
    if not isinstance(final, Sequence) or len(final) != 16:
        raise RuntimeError("MCP1 memorization final per-state MSE must contain 16 records")
    proof = manifest.get("main_parameters_exact_unchanged_proof")
    if not isinstance(proof, Mapping) or proof.get("all_sha256_exact_match") is not True:
        raise RuntimeError("MCP1 memorization Main unchanged proof missing or failed")
    status = manifest.get("diagnostic_result")
    if status not in (STRONG_MEMORIZATION_SUPPORT, INSUFFICIENT_MEMORIZATION):
        raise RuntimeError("MCP1 memorization diagnostic_result invalid")


def _training_step_backward(
    *,
    runtime: deployment.DeploymentRuntime,
    conditional_dict: Mapping[str, Any],
    states: Sequence[MCP1MemorizationState],
    selection: StageAParamSelection,
    pristine_cache_snapshot: MCP1ProbePristineCacheSnapshot,
) -> tuple[torch.Tensor, dict[str, Any]]:
    losses = []
    state_count = len(states)
    if state_count != len(RAW_TIMESTEPS) * NOISE_REALIZATIONS_PER_RAW_TIMESTEP:
        raise RuntimeError("MCP1 memorization training loss requires 16 fixed states")
    for state in states:
        _, _, mcp_flow, _ = call_isolated_mcp1_joint_forward(
            runtime=runtime,
            conditional_dict=conditional_dict,
            state=state,
            pristine_cache_snapshot=pristine_cache_snapshot,
            pre_forward_phase="train_pre_forward",
            restore_after_forward=False,
            post_forward_phase="unused_train_post_forward",
        )
        state_loss = F.mse_loss(
            mcp_flow.float(),
            state.mcp_target.float(),
            reduction="mean",
        )
        if not bool(torch.isfinite(state_loss.detach()).all().item()):
            raise RuntimeError("MCP1 memorization training state loss is non-finite")
        (state_loss / float(state_count)).backward()
        pristine_cache_snapshot.restore_and_verify(
            runtime,
            phase="train_post_backward",
            state_id=state.state_id,
        )
        losses.append(state_loss.detach())
    loss = torch.stack([value.float() for value in losses]).mean()
    if not bool(torch.isfinite(loss.detach()).all().item()):
        raise RuntimeError("MCP1 memorization training loss is non-finite")
    gradient_audit = audit_stage_a_allowed_gradients(selection, runtime.generator)
    return loss, gradient_audit


def _validate_optimizer_contract_inputs(
    *,
    optimizer_steps: int,
    optimizer_lr: float,
    log_interval: int,
) -> None:
    if int(optimizer_steps) < 0:
        raise ValueError("optimizer_steps must be non-negative")
    if not math.isfinite(float(optimizer_lr)) or float(optimizer_lr) <= 0.0:
        raise ValueError("optimizer_lr must be positive and finite")
    if int(log_interval) <= 0:
        raise ValueError("log_interval must be positive")


def _require_finite_tensor(tensor: torch.Tensor, *, name: str) -> None:
    if not torch.is_tensor(tensor):
        raise TypeError(f"{name} must be a tensor")
    if not bool(torch.isfinite(tensor.detach()).all().item()):
        raise RuntimeError(f"{name} contains non-finite values")


def _trainable_parameter_snapshot(
    named_parameters: Sequence[tuple[str, torch.nn.Parameter]],
) -> dict[str, torch.Tensor]:
    return {
        name: param.detach().cpu().clone()
        for name, param in named_parameters
    }


def _warp_raw_timestep(raw_timestep: int, *, shift: float) -> float:
    raw = torch.tensor([float(raw_timestep)], dtype=torch.float32)
    warped = flow_match_shift_timesteps(
        raw,
        shift=float(shift),
        num_train_timesteps=DEFAULT_NUM_TRAIN_TIMESTEPS,
    )
    return float(warped.item())


def _noise_for_realization(
    *,
    template: torch.Tensor,
    source_noise: torch.Tensor,
    teacher_target: torch.Tensor,
    raw_timestep: int,
    noise_index: int,
    role: str,
    base_seed: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if int(noise_index) == 0:
        noise = template.detach().clone()
        record = {
            "schema": MCP1_MEMORIZATION_NOISE_SCHEMA,
            "role": str(role),
            "raw_timestep": int(raw_timestep),
            "noise_index": 0,
            "source": "stored_fixed_validation_source_noise",
            "sha256": tensor_sha256(noise.detach().cpu()),
            "shape": [int(dim) for dim in noise.shape],
            "dtype": str(noise.dtype),
        }
        return noise, record
    key = {
        "schema": MCP1_MEMORIZATION_NOISE_SCHEMA,
        "role": str(role),
        "raw_timestep": int(raw_timestep),
        "noise_index": int(noise_index),
        "base_seed": int(base_seed),
        "source_noise_sha256": tensor_sha256(source_noise.detach().cpu()),
        "teacher_target_sha256": tensor_sha256(teacher_target.detach().cpu()),
    }
    key_sha = deployment.canonical_json_sha256(key)
    seed = (int(key_sha[:16], 16) + int(base_seed)) % (2**63 - 1)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    cpu_noise = torch.randn(
        tuple(int(dim) for dim in template.shape),
        generator=generator,
        dtype=torch.float32,
        device="cpu",
    )
    noise = cpu_noise.to(device=template.device, dtype=template.dtype)
    record = {
        **key,
        "source": "semantic_cpu_generator",
        "semantic_key_sha256": key_sha,
        "torch_generator_seed": int(seed),
        "sha256": tensor_sha256(noise.detach().cpu()),
        "shape": [int(dim) for dim in noise.shape],
        "dtype": str(noise.dtype),
    }
    return noise, record


def _tensor_record(tensor: torch.Tensor) -> dict[str, Any]:
    summary = tensor_summary(tensor.detach().cpu())
    return {
        "shape": summary["shape"],
        "dtype": summary["dtype"],
        "finite": summary["finite"],
        "sha256": summary["sha256"],
    }


def _relative_reduction(initial: float, final: float) -> float | None:
    initial = float(initial)
    final = float(final)
    if initial == 0.0:
        return None
    return float((initial - final) / initial)


def _fraction(numerator: float, denominator: float) -> float | None:
    denominator = float(denominator)
    if denominator == 0.0:
        return None
    return float(float(numerator) / denominator)


@contextmanager
def _null_context():
    yield


__all__ = [
    "ALLOWED_STAGE_A_GROUPS",
    "CURRENT_CHUNK_INDEX",
    "DEFAULT_LOG_INTERVAL",
    "DEFAULT_NOISE_SEED",
    "DEFAULT_OPTIMIZER_LR",
    "DEFAULT_OPTIMIZER_STEPS",
    "FORBIDDEN_MCP_GROUPS",
    "FUTURE_CHUNK_INDEX",
    "HISTORY_CHUNK_INDEX",
    "INSUFFICIENT_MEMORIZATION",
    "MCP1MemorizationProbeResult",
    "MCP1MemorizationState",
    "MCP1ProbePristineCacheSnapshot",
    "MCP1_MEMORIZATION_PROBE_SCHEMA",
    "MCP1_MEMORIZATION_TENSOR_SCHEMA",
    "NOISE_REALIZATIONS_PER_RAW_TIMESTEP",
    "RAW_TIMESTEPS",
    "STRONG_MEMORIZATION_SUPPORT",
    "StageAParamSelection",
    "allowed_stage_a_param_ids",
    "assert_gradient_checkpointing_disabled",
    "assert_no_frozen_stage_a_gradients",
    "assert_no_main_gradients",
    "assert_stage_a_fail_closed",
    "audit_stage_a_allowed_gradients",
    "build_mcp1_memorization_states",
    "build_memorization_flow_scheduler",
    "build_pristine_cache_snapshot",
    "call_isolated_mcp1_joint_forward",
    "compare_parameter_sha256_reports",
    "compute_mcp1_probe_kv_write_set",
    "configure_stage_a_trainable_parameters",
    "evaluate_memorization_status",
    "evaluate_probe_states",
    "forbid_mcp_depth23_calls",
    "parameter_sha256_report",
    "run_mcp1_memorization_probe",
    "summarize_flow_mse",
    "trainable_parameter_delta_report",
    "validate_mcp1_memorization_manifest",
    "validate_optimizer_excludes_main",
]
