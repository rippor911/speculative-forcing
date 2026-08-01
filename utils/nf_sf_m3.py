from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from utils.nf_sf_tensors import DEFAULT_S_MAIN, DEFAULT_S_MCP, make_generator
from utils.nf_sf_training import (
    NFSFNoisyBatch,
    NFSFSelectedState,
    prepare_nf_sf_noisy_batch,
    run_nf_sf_forward_loss,
)


M3_CHUNK_FRAMES = 3
M3_DEPTHS = (1, 2, 3)
M3_DEPTH_WEIGHTS = (0.5, 0.2, 0.1)
M3_MIN_LATENT_FRAMES = 15

M3_TEACHER_MANIFEST_FORMAT = "self_forcing_teacher_manifest_v2"
M3_TEACHER_PAYLOAD_FORMAT = "self_forcing_teacher_v1"
M3_REFERENCE_CHECKPOINT_SHA256 = (
    "a0413986d9734e02c09504e1520f5697"
    "ba6df731bb2f0f35577485e9cc8f56a3"
)
M3_CHECKPOINT_FORMAT = "nf_sf_m3_overfit_checkpoint_v1"

M3_REQUIRED_CHECKPOINT_KEYS = (
    "format",
    "generator",
    "optimizer",
    "global_step",
    "train_rng_state",
    "probe_rng_state",
    "probe_tensors",
    "probe_outputs",
    "probe_summary",
    "probe_prompt_embedding",
    "selected_sample_metadata",
    "resolved_config",
    "git_sha",
    "reference_checkpoint",
    "train_seed",
    "probe_seed",
    "optimizer_group_lrs",
)

M3_PROBE_OUTPUT_KEYS = (
    "main_flow_pred",
    "mcp_depth1_flow_pred",
    "mcp_depth2_flow_pred",
    "mcp_depth3_flow_pred",
)

M3_PARAMETER_GROUP_NAMES = (
    "backbone",
    "patch_embedding",
    "mcp_fusion",
    "mcp_depth1",
    "mcp_depth2",
    "mcp_depth3",
)

_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


@dataclass(frozen=True)
class M3TeacherSample:
    payload: dict[str, Any]
    target_latent: torch.Tensor
    source_noise: torch.Tensor
    selected_state: NFSFSelectedState
    metadata: dict[str, Any]


@dataclass(frozen=True)
class M3Probe:
    seed: int
    rng_state: torch.Tensor
    noisy_batch: NFSFNoisyBatch


@dataclass(frozen=True)
class M3ProbeForward:
    losses: dict[str, float]
    outputs: dict[str, torch.Tensor]


@dataclass(frozen=True)
class M3SolverSchedule:
    source: str
    raw_denoising_steps: tuple[float, ...]
    warped_denoising_steps: tuple[float, ...]
    generated_timesteps: tuple[float, ...]
    timesteps: torch.Tensor
    max_abs_diff: float | None
    mean_abs_diff: float | None
    tolerance: float
    override_steps: int | None


@dataclass(frozen=True)
class M3ReconstructionResult:
    latent: torch.Tensor
    solver_schedule: M3SolverSchedule


def validate_m3_mode(mode: str) -> None:
    if mode != "joint":
        raise ValueError("NF-SF M3 only accepts joint mode")


def validate_git_sha(value: str, *, name: str = "git_sha") -> str:
    value = str(value)
    if not _GIT_SHA_RE.fullmatch(value):
        raise RuntimeError(f"{name} must be a 40-character lowercase hex Git SHA")
    return value


def file_sha256(path: Path | str) -> str:
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    return hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()


def prompt_sha256(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def tensor_summary(tensor: torch.Tensor) -> dict[str, Any]:
    value = tensor.detach().float()
    return {
        "shape": [int(dim) for dim in tensor.shape],
        "dtype": str(tensor.dtype),
        "finite": bool(torch.isfinite(value).all().item()),
        "mean": float(value.mean().item()),
        "std": float(value.std().item()),
        "rms": float(value.square().mean().sqrt().item()),
        "sha256": tensor_sha256(tensor),
    }


def atomic_json_write(payload: Mapping[str, Any], path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_torch_save(payload: Mapping[str, Any], path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    temporary.replace(path)


def move_tensors_to_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, Mapping):
        return {key: move_tensors_to_cpu(child) for key, child in value.items()}
    if isinstance(value, tuple):
        return tuple(move_tensors_to_cpu(child) for child in value)
    if isinstance(value, list):
        return [move_tensors_to_cpu(child) for child in value]
    return value


def move_tensors_to_device(
    value: Any,
    *,
    device: torch.device | str,
    floating_dtype: torch.dtype | None = None,
) -> Any:
    device = torch.device(device)
    if isinstance(value, torch.Tensor):
        result = value.to(device=device)
        if floating_dtype is not None and result.is_floating_point():
            result = result.to(dtype=floating_dtype)
        return result
    if isinstance(value, Mapping):
        return {
            key: move_tensors_to_device(
                child,
                device=device,
                floating_dtype=floating_dtype,
            )
            for key, child in value.items()
        }
    if isinstance(value, tuple):
        return tuple(
            move_tensors_to_device(
                child,
                device=device,
                floating_dtype=floating_dtype,
            )
            for child in value
        )
    if isinstance(value, list):
        return [
            move_tensors_to_device(
                child,
                device=device,
                floating_dtype=floating_dtype,
            )
            for child in value
        ]
    return value


def load_m3_teacher_sample(
    *,
    manifest_path: Path | str,
    dataset_root: Path | str | None = None,
    sample_index: int | None = None,
    sample_id: str | None = None,
    split: str | None = None,
    split_index: int | None = None,
    reference_checkpoint_path: Path | str | None = None,
    expected_reference_sha256: str = M3_REFERENCE_CHECKPOINT_SHA256,
) -> M3TeacherSample:
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _validate_teacher_manifest(manifest)
    record = select_manifest_record(
        manifest,
        sample_index=sample_index,
        sample_id=sample_id,
        split=split,
        split_index=split_index,
    )
    payload_path = resolve_payload_path(
        manifest_path=manifest_path,
        record=record,
        dataset_root=None if dataset_root is None else Path(dataset_root),
    )
    payload_file_sha256 = file_sha256(payload_path)
    expected_file_sha256 = record.get("file_sha256")
    if expected_file_sha256 is not None and payload_file_sha256 != expected_file_sha256:
        raise RuntimeError(
            "teacher latent payload SHA256 mismatch: "
            f"{payload_file_sha256} != {expected_file_sha256}"
        )

    payload = torch.load(payload_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError("teacher payload must be a dict")
    _validate_teacher_payload(payload, record, manifest)

    target_latent = payload["target_latent"].detach().cpu()
    source_noise = payload["source_noise"].detach().cpu()
    selected_state = select_m3_selected_state(target_latent)
    raw_steps, warped_steps = extract_m3_solver_steps(payload)

    manifest_checkpoint = manifest.get("checkpoint", {})
    manifest_checkpoint_sha = str(manifest_checkpoint.get("sha256", ""))
    if manifest_checkpoint_sha != expected_reference_sha256:
        raise RuntimeError(
            "teacher manifest reference checkpoint SHA256 differs from M3 reference"
        )

    reference_checkpoint_sha256 = None
    reference_checkpoint_resolved = None
    if reference_checkpoint_path is not None:
        reference_checkpoint_resolved = str(Path(reference_checkpoint_path).resolve())
        reference_checkpoint_sha256 = file_sha256(reference_checkpoint_path)
        if reference_checkpoint_sha256 != manifest_checkpoint_sha:
            raise RuntimeError(
                "provided reference checkpoint SHA256 does not match teacher data"
            )

    target_summary = tensor_summary(target_latent)
    source_summary = tensor_summary(source_noise)
    true_sample_id = payload.get("sample_id", record.get("sample_id"))
    metadata = {
        "dataset_root": None if dataset_root is None else str(Path(dataset_root).resolve()),
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": file_sha256(manifest_path),
        "manifest_format": manifest.get("format"),
        "manifest_status": manifest.get("status"),
        "manifest_or_index_file": str(manifest_path.resolve()),
        "sample_index": int(payload["sample_index"]),
        "sample_id": None if true_sample_id is None else str(true_sample_id),
        "split": str(payload["split"]),
        "split_index": int(payload["split_index"]),
        "source_line_index": int(payload["source_line_index"]),
        "shard_id": int(payload["shard_id"]),
        "plan_index": int(payload["plan_index"]),
        "prompt": str(payload["prompt"]),
        "prompt_field": "prompt",
        "prompt_sha256": str(payload["prompt_sha256"]),
        "actual_prompt_sha256": prompt_sha256(str(payload["prompt"])),
        "seed": int(payload["seed"]),
        "noise_seed": int(payload["noise_seed"]),
        "rollout_seed": int(payload["rollout_seed"]),
        "latent_path": str(payload_path.resolve()),
        "latent_file_sha256": payload_file_sha256,
        "latent_file_format": "torch.save dict payload",
        "latent_tensor_key": "target_latent",
        "latent_layout": "[B, F, C, H, W]",
        "latent_shape": target_summary["shape"],
        "latent_dtype": target_summary["dtype"],
        "latent_frame_count": int(target_latent.shape[1]),
        "source_noise": source_summary,
        "target_latent": target_summary,
        "chunk_frames": M3_CHUNK_FRAMES,
        "chunk_aligned": bool(target_latent.shape[1] % M3_CHUNK_FRAMES == 0),
        "has_minimum_15_latent_frames": bool(target_latent.shape[1] >= M3_MIN_LATENT_FRAMES),
        "selected_slices": selected_state_slices(),
        "solver_schedule": {
            "source": "teacher_payload",
            "raw_denoising_steps": raw_steps,
            "warped_denoising_steps": warped_steps,
        },
        "generation_source": {
            "experiment": manifest.get("experiment"),
            "writer_format": manifest.get("writer_format"),
            "payload_format": payload.get("format"),
            "writer_git_head": payload.get("writer_git_head"),
            "checkpoint_path": manifest_checkpoint.get("path"),
            "checkpoint_sha256": manifest_checkpoint_sha,
            "reference_checkpoint_path": reference_checkpoint_resolved,
            "reference_checkpoint_sha256": reference_checkpoint_sha256,
            "raw_denoising_steps": raw_steps,
            "warped_denoising_steps": warped_steps,
            "mcp_num_modules": manifest.get("generation", {}).get("mcp_num_modules"),
            "mcp_accel_depths": manifest.get("generation", {}).get("mcp_accel_depths"),
            "last_step_only": manifest.get("generation", {}).get("last_step_only"),
        },
    }
    return M3TeacherSample(
        payload=payload,
        target_latent=target_latent,
        source_noise=source_noise,
        selected_state=selected_state,
        metadata=metadata,
    )


def _validate_teacher_manifest(manifest: Mapping[str, Any]) -> None:
    if manifest.get("status") != "PASS":
        raise RuntimeError("teacher manifest is not PASS")
    if manifest.get("format") != M3_TEACHER_MANIFEST_FORMAT:
        raise RuntimeError("teacher manifest format is not the formal v2 format")
    generation = manifest.get("generation")
    if not isinstance(generation, Mapping):
        raise RuntimeError("teacher manifest has no generation block")
    if int(generation.get("num_train", -1)) != 2048:
        raise RuntimeError("teacher manifest train sample count is not 2048")
    if int(generation.get("num_validation", -1)) != 256:
        raise RuntimeError("teacher manifest validation sample count is not 256")
    if int(generation.get("num_frames", -1)) < M3_MIN_LATENT_FRAMES:
        raise RuntimeError("teacher manifest has fewer than 15 latent frames")
    if int(generation.get("num_frame_per_block", -1)) != M3_CHUNK_FRAMES:
        raise RuntimeError("teacher manifest chunk_frames is not 3")
    if int(generation.get("mcp_depth", -1)) != len(M3_DEPTHS):
        raise RuntimeError("teacher manifest mcp_depth is not 3")
    samples = manifest.get("samples")
    if not isinstance(samples, list) or not samples:
        raise RuntimeError("teacher manifest has no samples list")


def select_manifest_record(
    manifest: Mapping[str, Any],
    *,
    sample_index: int | None = None,
    sample_id: str | None = None,
    split: str | None = None,
    split_index: int | None = None,
) -> dict[str, Any]:
    if sum(value is not None for value in (sample_index, sample_id, split_index)) > 1:
        raise ValueError("select only one of sample_index, sample_id, or split_index")

    records = list(manifest["samples"])
    if split is not None:
        records = [record for record in records if str(record.get("split")) == split]
    if sample_index is not None:
        records = [
            record
            for record in records
            if int(record.get("sample_index", -1)) == int(sample_index)
        ]
    elif sample_id is not None:
        records = [
            record
            for record in records
            if record.get("sample_id") is not None
            and str(record.get("sample_id")) == str(sample_id)
        ]
    elif split_index is not None:
        if split is None:
            raise ValueError("split_index selection also requires split")
        records = [
            record
            for record in records
            if int(record.get("split_index", -1)) == int(split_index)
        ]
    else:
        default_split = split or "train"
        records = [
            record
            for record in records
            if str(record.get("split")) == default_split
        ]
        records = sorted(records, key=lambda item: int(item["sample_index"]))[:1]

    if len(records) != 1:
        raise RuntimeError(f"expected exactly one teacher sample, found {len(records)}")
    return dict(records[0])


def resolve_payload_path(
    *,
    manifest_path: Path,
    record: Mapping[str, Any],
    dataset_root: Path | None = None,
) -> Path:
    value = None
    for key in ("file", "path", "payload_path", "artifact_path"):
        if record.get(key):
            value = str(record[key])
            break
    if value is None:
        raise RuntimeError("manifest record has no payload path field")

    raw = Path(value)
    candidates: list[Path] = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.extend([manifest_path.parent / raw, Path.cwd() / raw])
    if dataset_root is not None:
        candidates.insert(0, dataset_root / raw.name)
        if not raw.is_absolute():
            candidates.insert(0, dataset_root / raw)

    seen = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.is_file():
            return resolved
    raise FileNotFoundError(
        "could not resolve teacher payload path; tried "
        + ", ".join(str(path) for path in candidates)
    )


def _validate_teacher_payload(
    payload: Mapping[str, Any],
    record: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> None:
    required = {
        "format",
        "sample_index",
        "split",
        "split_index",
        "source_line_index",
        "shard_id",
        "plan_index",
        "prompt",
        "prompt_sha256",
        "seed",
        "noise_seed",
        "rollout_seed",
        "source_noise",
        "target_latent",
        "backbone_sha256",
        "num_frames",
        "num_frame_per_block",
        "mcp_depth",
        "raw_denoising_steps",
        "warped_denoising_steps",
    }
    missing = required - payload.keys()
    if missing:
        raise KeyError(f"teacher payload missing fields: {sorted(missing)}")
    if payload["format"] != M3_TEACHER_PAYLOAD_FORMAT:
        raise RuntimeError("teacher payload format is not self_forcing_teacher_v1")
    for key in ("sample_index", "split", "split_index", "prompt", "prompt_sha256"):
        if payload.get(key) != record.get(key):
            raise RuntimeError(f"teacher payload field {key!r} differs from manifest")
    generation = manifest.get("generation", {})
    if int(payload["num_frames"]) != int(generation.get("num_frames", -1)):
        raise RuntimeError("teacher payload num_frames differs from manifest")
    actual_prompt_sha = prompt_sha256(str(payload["prompt"]))
    if actual_prompt_sha != str(payload["prompt_sha256"]):
        raise RuntimeError("teacher payload prompt_sha256 differs from prompt text")
    if actual_prompt_sha != str(record.get("prompt_sha256")):
        raise RuntimeError("teacher manifest prompt_sha256 differs from prompt text")
    manifest_checkpoint_sha = manifest.get("checkpoint", {}).get("sha256")
    if payload.get("backbone_sha256") != manifest_checkpoint_sha:
        raise RuntimeError("teacher payload backbone SHA differs from manifest")
    for key in ("source_noise", "target_latent"):
        tensor = payload[key]
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{key} must be a torch.Tensor")
        _validate_latent_tensor(tensor, name=key, expected_frames=int(payload["num_frames"]))
        if tensor.dtype != torch.bfloat16:
            raise RuntimeError(f"{key} dtype is {tensor.dtype}, expected torch.bfloat16")
        _compare_manifest_tensor_summary(
            key,
            tensor=tensor,
            record=record,
        )
    if tuple(payload["source_noise"].shape) != tuple(payload["target_latent"].shape):
        raise RuntimeError("teacher source_noise and target_latent shapes differ")
    if int(payload["num_frame_per_block"]) != M3_CHUNK_FRAMES:
        raise RuntimeError("teacher payload chunk_frames is not 3")
    if int(payload["mcp_depth"]) != len(M3_DEPTHS):
        raise RuntimeError("teacher payload mcp_depth is not 3")
    extract_m3_solver_steps(payload)


def _compare_manifest_tensor_summary(
    name: str,
    *,
    tensor: torch.Tensor,
    record: Mapping[str, Any],
) -> None:
    expected = record.get(name)
    if not isinstance(expected, Mapping):
        raise RuntimeError(f"teacher manifest record has no {name} summary")
    actual = tensor_summary(tensor)
    for key in ("shape", "dtype", "sha256"):
        if actual[key] != expected.get(key):
            raise RuntimeError(
                f"teacher manifest {name} {key} differs from actual tensor"
            )


def _float_schedule_values(value: Any, *, name: str) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise RuntimeError(f"teacher payload {name} must be a non-empty sequence")
    values = tuple(float(item) for item in value)
    if not all(torch.isfinite(torch.tensor(values, dtype=torch.float64)).tolist()):
        raise RuntimeError(f"teacher payload {name} contains non-finite values")
    return values


def extract_m3_solver_steps(payload: Mapping[str, Any]) -> tuple[list[float], list[float]]:
    raw_steps = _float_schedule_values(
        payload.get("raw_denoising_steps"),
        name="raw_denoising_steps",
    )
    warped_steps = _float_schedule_values(
        payload.get("warped_denoising_steps"),
        name="warped_denoising_steps",
    )
    if len(raw_steps) != len(warped_steps):
        raise RuntimeError("teacher payload raw/warped denoising steps length mismatch")
    return list(raw_steps), list(warped_steps)


def _validate_latent_tensor(
    latent: torch.Tensor,
    *,
    name: str,
    expected_frames: int | None = None,
) -> None:
    if latent.ndim != 5:
        raise ValueError(f"{name} must have layout [B, F, C, H, W]")
    if expected_frames is not None and latent.shape[1] != expected_frames:
        raise ValueError(f"{name} frame count differs from metadata")
    if latent.shape[1] < M3_MIN_LATENT_FRAMES:
        raise ValueError(f"{name} must contain at least 15 latent frames")
    if latent.shape[1] % M3_CHUNK_FRAMES != 0:
        raise ValueError(f"{name} frame count must be chunk-aligned")
    if not latent.is_floating_point():
        raise ValueError(f"{name} must use a floating dtype")
    if not bool(torch.isfinite(latent.float()).all().item()):
        raise RuntimeError(f"{name} contains NaN or Inf")


def select_m3_selected_state(
    latent: torch.Tensor,
    *,
    chunk_frames: int = M3_CHUNK_FRAMES,
) -> NFSFSelectedState:
    if chunk_frames != M3_CHUNK_FRAMES:
        raise ValueError("NF-SF M3 selected-state audit requires chunk_frames=3")
    _validate_latent_tensor(latent, name="target_latent")
    history = latent[:, 0:3]
    current = latent[:, 3:6]
    next1 = latent[:, 6:9]
    next2 = latent[:, 9:12]
    next3 = latent[:, 12:15]
    return NFSFSelectedState(
        clean_history=history,
        current_target=current,
        future_targets=(next1, next2, next3),
        current_start_frame=chunk_frames,
    )


def selected_state_slices() -> dict[str, list[int]]:
    return {
        "history": [0, 3],
        "current": [3, 6],
        "next1": [6, 9],
        "next2": [9, 12],
        "next3": [12, 15],
    }


def selected_state_to_device(
    state: NFSFSelectedState,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
) -> NFSFSelectedState:
    return NFSFSelectedState(
        clean_history=None
        if state.clean_history is None
        else state.clean_history.to(device=device, dtype=dtype),
        current_target=state.current_target.to(device=device, dtype=dtype),
        future_targets=tuple(
            target.to(device=device, dtype=dtype)
            for target in state.future_targets
        ),
        future_valid_masks=None
        if state.future_valid_masks is None
        else tuple(mask.to(device=device) for mask in state.future_valid_masks),
        current_start_frame=state.current_start_frame,
    )


def make_m3_probe(
    state: NFSFSelectedState,
    *,
    scheduler_main,
    scheduler_mcp,
    seed: int,
) -> M3Probe:
    rng = make_generator(seed, state.current_target.device)
    rng_state = rng.get_state()
    noisy_batch = prepare_nf_sf_noisy_batch(
        state,
        scheduler_main=scheduler_main,
        scheduler_mcp=scheduler_mcp,
        rng=rng,
        chunk_frames=M3_CHUNK_FRAMES,
        depths=M3_DEPTHS,
        s_main=DEFAULT_S_MAIN,
        s_mcp=DEFAULT_S_MCP,
    )
    return M3Probe(seed=int(seed), rng_state=rng_state, noisy_batch=noisy_batch)


def serialize_noisy_batch(noisy_batch: NFSFNoisyBatch) -> dict[str, Any]:
    return move_tensors_to_cpu(
        {
            "noisy_current": noisy_batch.noisy_current,
            "noisy_futures": list(noisy_batch.noisy_futures),
            "timestep_main": noisy_batch.timestep_main,
            "timestep_depths": list(noisy_batch.timestep_depths),
            "epsilon_main": noisy_batch.epsilon_main,
            "epsilon_depths": list(noisy_batch.epsilon_depths),
            "target_flow_main": noisy_batch.target_flow_main,
            "target_flow_depths": list(noisy_batch.target_flow_depths),
            "future_valid_masks": list(noisy_batch.future_valid_masks),
            "future_start_frames": list(noisy_batch.future_start_frames),
        }
    )


def deserialize_noisy_batch(
    payload: Mapping[str, Any],
    *,
    state: NFSFSelectedState,
    device: torch.device | str,
    dtype: torch.dtype,
) -> NFSFNoisyBatch:
    value = move_tensors_to_device(payload, device=device, floating_dtype=None)
    return NFSFNoisyBatch(
        state=state,
        noisy_current=value["noisy_current"].to(dtype=dtype),
        noisy_futures=tuple(tensor.to(dtype=dtype) for tensor in value["noisy_futures"]),
        timestep_main=value["timestep_main"].float(),
        timestep_depths=tuple(timestep.float() for timestep in value["timestep_depths"]),
        epsilon_main=value["epsilon_main"].to(dtype=dtype),
        epsilon_depths=tuple(tensor.to(dtype=dtype) for tensor in value["epsilon_depths"]),
        target_flow_main=value["target_flow_main"].to(dtype=dtype),
        target_flow_depths=tuple(
            tensor.to(dtype=dtype) for tensor in value["target_flow_depths"]
        ),
        future_valid_masks=tuple(mask.bool() for mask in value["future_valid_masks"]),
        future_start_frames=tuple(int(frame) for frame in value["future_start_frames"]),
    )


def compare_serialized_probe_tensors(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> dict[str, Any]:
    max_abs = 0.0
    mean_values = []
    compared = 0
    tensor_reports: dict[str, Any] = {}

    def visit(a: Any, b: Any, path: str) -> None:
        nonlocal max_abs, compared
        if isinstance(a, torch.Tensor):
            if not isinstance(b, torch.Tensor):
                raise TypeError("probe tensor structure mismatch")
            if a.shape != b.shape:
                raise ValueError(
                    "probe tensor shape mismatch "
                    f"at {path}: {tuple(a.shape)} != {tuple(b.shape)}"
                )
            left_dtype = str(a.dtype)
            right_dtype = str(b.dtype)
            if a.dtype != b.dtype:
                raise ValueError(
                    "probe tensor dtype mismatch "
                    f"at {path}: left_dtype={left_dtype}, right_dtype={right_dtype}"
                )
            diff = a.detach().float().cpu() - b.detach().float().cpu()
            tensor_max_abs = float(diff.abs().max().item())
            tensor_mean_abs = float(diff.abs().mean().item())
            max_abs = max(max_abs, tensor_max_abs)
            mean_values.append(tensor_mean_abs)
            tensor_reports[path] = {
                "left_shape": [int(dim) for dim in a.shape],
                "right_shape": [int(dim) for dim in b.shape],
                "shape_match": True,
                "left_dtype": left_dtype,
                "right_dtype": right_dtype,
                "dtype_match": True,
                "max_abs_diff": tensor_max_abs,
                "mean_abs_diff": tensor_mean_abs,
            }
            compared += 1
            return
        if isinstance(a, Mapping):
            if not isinstance(b, Mapping):
                raise TypeError("probe mapping structure mismatch")
            if set(a.keys()) != set(b.keys()):
                raise ValueError("probe mapping keys mismatch")
            for key in a:
                visit(a[key], b[key], f"{path}.{key}" if path else str(key))
            return
        if isinstance(a, (list, tuple)):
            if not isinstance(b, (list, tuple)):
                raise TypeError("probe sequence structure mismatch")
            if len(a) != len(b):
                raise ValueError("probe sequence length mismatch")
            for index, (child_a, child_b) in enumerate(zip(a, b)):
                visit(child_a, child_b, f"{path}[{index}]")
            return
        if a != b:
            raise ValueError(f"probe scalar mismatch {a!r} != {b!r}")

    visit(left, right, "")
    return {
        "tensor_count": compared,
        "max_abs_diff": max_abs,
        "mean_abs_diff": 0.0 if not mean_values else sum(mean_values) / len(mean_values),
        "exact": max_abs == 0.0,
        "tensors": tensor_reports,
    }


def _probe_outputs_from_result(result) -> dict[str, torch.Tensor]:
    if len(result.mcp_flow_preds) != len(M3_DEPTHS):
        raise RuntimeError("M3 probe expected exactly three MCP flow outputs")
    outputs = {
        "main_flow_pred": result.main_flow_pred,
    }
    for index, tensor in enumerate(result.mcp_flow_preds, start=1):
        outputs[f"mcp_depth{index}_flow_pred"] = tensor
    return outputs


def serialize_probe_outputs(outputs: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    missing = set(M3_PROBE_OUTPUT_KEYS) - set(outputs.keys())
    if missing:
        raise KeyError(f"probe outputs missing keys: {sorted(missing)}")
    return {
        key: move_tensors_to_cpu(outputs[key])
        for key in M3_PROBE_OUTPUT_KEYS
    }


def probe_output_summaries(outputs: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    return {
        key: tensor_summary(outputs[key])
        for key in M3_PROBE_OUTPUT_KEYS
    }


def compare_probe_outputs(
    actual: Mapping[str, torch.Tensor],
    expected: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    actual_outputs = serialize_probe_outputs(actual)
    expected_outputs = serialize_probe_outputs(expected)
    per_branch = {}
    max_values = []
    mean_values = []
    for key in M3_PROBE_OUTPUT_KEYS:
        left = actual_outputs[key]
        right = expected_outputs[key]
        if left.shape != right.shape:
            raise ValueError(
                f"probe output {key} shape mismatch: "
                f"{tuple(left.shape)} != {tuple(right.shape)}"
            )
        left_dtype = str(left.dtype)
        right_dtype = str(right.dtype)
        if left.dtype != right.dtype:
            raise ValueError(
                f"probe output {key} dtype mismatch: "
                f"left_dtype={left_dtype}, right_dtype={right_dtype}"
            )
        diff = left.float() - right.float()
        branch = {
            "left_shape": [int(dim) for dim in left.shape],
            "right_shape": [int(dim) for dim in right.shape],
            "shape_match": True,
            "left_dtype": left_dtype,
            "right_dtype": right_dtype,
            "dtype_match": True,
            "max_abs_diff": float(diff.abs().max().item()),
            "mean_abs_diff": float(diff.abs().mean().item()),
        }
        per_branch[key] = branch
        max_values.append(branch["max_abs_diff"])
        mean_values.append(branch["mean_abs_diff"])
    return {
        "branches": per_branch,
        "max_abs_diff": max(max_values),
        "mean_abs_diff": sum(mean_values) / len(mean_values),
    }


def run_m3_probe_forward(
    generator,
    *,
    conditional_dict: dict[str, Any],
    noisy_batch: NFSFNoisyBatch,
) -> M3ProbeForward:
    was_training = bool(generator.training)
    generator.eval()
    try:
        with torch.no_grad():
            result = run_nf_sf_forward_loss(
                generator,
                conditional_dict=conditional_dict,
                noisy_batch=noisy_batch,
                depth_weights=M3_DEPTH_WEIGHTS,
            )
    finally:
        generator.train(was_training)
    return M3ProbeForward(
        losses=loss_dict_to_floats(result.losses),
        outputs=serialize_probe_outputs(_probe_outputs_from_result(result)),
    )


def run_m3_probe_loss(
    generator,
    *,
    conditional_dict: dict[str, Any],
    noisy_batch: NFSFNoisyBatch,
) -> dict[str, float]:
    return run_m3_probe_forward(
        generator,
        conditional_dict=conditional_dict,
        noisy_batch=noisy_batch,
    ).losses


def loss_dict_to_floats(losses) -> dict[str, float]:
    return {
        "main_loss": float(losses.main_loss.detach().float().item()),
        "mcp_depth1_loss": float(losses.mcp_depth_losses[0].detach().float().item()),
        "mcp_depth2_loss": float(losses.mcp_depth_losses[1].detach().float().item()),
        "mcp_depth3_loss": float(losses.mcp_depth_losses[2].detach().float().item()),
        "total_loss": float(losses.total_loss.detach().float().item()),
    }


def prefix_metrics(prefix: str, losses: Mapping[str, float]) -> dict[str, float]:
    return {f"{prefix}/{key}": float(value) for key, value in losses.items()}


def optimizer_group_lr_summary(optimizer: torch.optim.Optimizer) -> list[dict[str, Any]]:
    groups = []
    for index, group in enumerate(optimizer.param_groups):
        groups.append(
            {
                "index": index,
                "name": group.get("name"),
                "lr": float(group["lr"]),
                "weight_decay": float(group.get("weight_decay", 0.0)),
                "tensor_count": len(group.get("params", [])),
            }
        )
    return groups


def optimizer_config_summary(optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    defaults = optimizer.defaults
    betas = defaults.get("betas", (None, None))
    return {
        "optimizer": optimizer.__class__.__name__,
        "betas": [float(value) for value in betas],
        "eps": float(defaults.get("eps")),
        "weight_decay": float(defaults.get("weight_decay", 0.0)),
    }


def gradient_group_audit(optimizer: torch.optim.Optimizer) -> dict[str, dict[str, Any]]:
    report = {}
    for group in optimizer.param_groups:
        name = str(group.get("name"))
        if name not in M3_PARAMETER_GROUP_NAMES:
            continue
        grad_sq_sum = 0.0
        with_grad = 0
        without_grad = 0
        finite = True
        for parameter in group.get("params", []):
            grad = parameter.grad
            if grad is None:
                without_grad += 1
                continue
            with_grad += 1
            grad_value = grad.detach().float()
            finite = finite and bool(torch.isfinite(grad_value).all().item())
            grad_sq_sum += float(grad_value.square().sum().item())
        grad_norm = float(grad_sq_sum ** 0.5)
        finite = finite and bool(torch.isfinite(torch.tensor(grad_norm)).item())
        report[name] = {
            "tensor_count_with_grad": int(with_grad),
            "tensor_count_without_grad": int(without_grad),
            "grad_norm": grad_norm,
            "finite": bool(finite),
        }
    missing = set(M3_PARAMETER_GROUP_NAMES) - set(report.keys())
    if missing:
        raise RuntimeError(f"missing M3 optimizer groups for grad audit: {sorted(missing)}")
    return report


def parameter_names_by_group_from_optimizer_audit(
    optimizer_audit: Mapping[str, Any],
) -> dict[str, tuple[str, ...]]:
    param_audit = optimizer_audit.get("param_audit")
    if not isinstance(param_audit, list):
        raise RuntimeError("optimizer audit has no param_audit list")
    groups: dict[str, tuple[str, ...]] = {}
    for entry in param_audit:
        if not isinstance(entry, Mapping):
            raise RuntimeError("optimizer audit param_audit entry must be a mapping")
        name = str(entry.get("name"))
        if name in M3_PARAMETER_GROUP_NAMES:
            names = entry.get("parameter_names")
            if not isinstance(names, list) or not names:
                raise RuntimeError(f"optimizer audit group {name} has no parameter names")
            groups[name] = tuple(str(value) for value in names)
    missing = set(M3_PARAMETER_GROUP_NAMES) - set(groups.keys())
    if missing:
        raise RuntimeError(f"optimizer audit missing M3 groups: {sorted(missing)}")
    return groups


def audit_parameter_changes(
    *,
    initial_state_dict: Mapping[str, torch.Tensor],
    final_state_dict: Mapping[str, torch.Tensor],
    optimizer_audit: Mapping[str, Any],
) -> dict[str, Any]:
    groups = parameter_names_by_group_from_optimizer_audit(optimizer_audit)
    group_reports = {}
    all_changed = True
    for group_name in M3_PARAMETER_GROUP_NAMES:
        names = groups[group_name]
        changed_count = 0
        unchanged_count = 0
        max_abs = 0.0
        for name in names:
            if name not in initial_state_dict or name not in final_state_dict:
                raise RuntimeError(f"parameter {name!r} missing from checkpoint state_dict")
            initial = initial_state_dict[name]
            final = final_state_dict[name]
            if not torch.is_tensor(initial) or not torch.is_tensor(final):
                raise TypeError(f"parameter {name!r} is not a tensor")
            if tuple(initial.shape) != tuple(final.shape):
                raise RuntimeError(f"parameter {name!r} shape differs between checkpoints")
            if initial.dtype != final.dtype:
                raise RuntimeError(f"parameter {name!r} dtype differs between checkpoints")
            diff = (final.detach().float().cpu() - initial.detach().float().cpu()).abs()
            tensor_max_abs = float(diff.max().item()) if diff.numel() else 0.0
            max_abs = max(max_abs, tensor_max_abs)
            if tensor_max_abs > 0.0:
                changed_count += 1
            else:
                unchanged_count += 1
        parameter_changed = changed_count > 0
        all_changed = all_changed and parameter_changed
        group_reports[group_name] = {
            "tensor_count": int(len(names)),
            "changed_tensor_count": int(changed_count),
            "unchanged_tensor_count": int(unchanged_count),
            "max_abs_parameter_diff": max_abs,
            "parameter_changed": bool(parameter_changed),
        }
    return {
        "status": "PASS" if all_changed else "FAIL",
        "groups": group_reports,
        "all_groups_parameter_changed": bool(all_changed),
    }


def make_m3_checkpoint_payload(
    *,
    generator,
    optimizer: torch.optim.Optimizer,
    global_step: int,
    train_rng: torch.Generator,
    probe: M3Probe,
    probe_summary: Mapping[str, Any],
    probe_outputs: Mapping[str, torch.Tensor],
    selected_sample_metadata: Mapping[str, Any],
    resolved_config: Mapping[str, Any],
    git_sha: str,
    reference_checkpoint_path: Path | str,
    reference_checkpoint_sha256: str,
    train_seed: int,
    probe_seed: int,
    prompt_embedding: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if prompt_embedding is None:
        raise ValueError("M3 checkpoint requires fixed probe prompt embedding")
    payload = {
        "format": M3_CHECKPOINT_FORMAT,
        "generator": move_tensors_to_cpu(generator.state_dict()),
        "optimizer": move_tensors_to_cpu(optimizer.state_dict()),
        "global_step": int(global_step),
        "train_rng_state": train_rng.get_state().detach().cpu().clone(),
        "probe_rng_state": probe.rng_state.detach().cpu().clone(),
        "probe_tensors": serialize_noisy_batch(probe.noisy_batch),
        "probe_outputs": serialize_probe_outputs(probe_outputs),
        "probe_summary": dict(probe_summary),
        "probe_prompt_embedding": move_tensors_to_cpu(prompt_embedding),
        "selected_sample_metadata": dict(selected_sample_metadata),
        "resolved_config": dict(resolved_config),
        "git_sha": validate_git_sha(str(git_sha)),
        "reference_checkpoint": {
            "path": str(Path(reference_checkpoint_path).resolve()),
            "sha256": str(reference_checkpoint_sha256),
        },
        "train_seed": int(train_seed),
        "probe_seed": int(probe_seed),
        "optimizer_group_lrs": optimizer_group_lr_summary(optimizer),
    }
    validate_m3_checkpoint_payload(payload)
    return payload


def validate_m3_checkpoint_payload(payload: Mapping[str, Any]) -> None:
    missing = set(M3_REQUIRED_CHECKPOINT_KEYS) - payload.keys()
    if missing:
        raise KeyError(f"M3 checkpoint missing fields: {sorted(missing)}")
    if payload.get("format") != M3_CHECKPOINT_FORMAT:
        raise RuntimeError("M3 checkpoint format mismatch")
    if payload.get("probe_prompt_embedding") is None:
        raise RuntimeError("M3 checkpoint missing fixed probe prompt embedding")
    validate_git_sha(str(payload.get("git_sha", "")))
    serialize_probe_outputs(payload["probe_outputs"])


def save_m3_checkpoint(payload: Mapping[str, Any], path: Path | str) -> None:
    validate_m3_checkpoint_payload(payload)
    atomic_torch_save(payload, path)


def load_m3_checkpoint(path: Path | str) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError("M3 checkpoint must contain a dict payload")
    validate_m3_checkpoint_payload(payload)
    return payload


def solver_timesteps_from_scheduler(
    scheduler,
    *,
    num_inference_steps: int,
    device: torch.device | str,
) -> torch.Tensor:
    if num_inference_steps <= 0:
        raise ValueError("num_inference_steps must be positive")
    scheduler.set_timesteps(num_inference_steps, denoising_strength=1.0)
    scheduler.sigmas = scheduler.sigmas.to(device)
    scheduler.timesteps = scheduler.timesteps.to(device)
    return scheduler.timesteps


def resolve_m3_solver_schedule(
    scheduler,
    *,
    teacher_payload: Mapping[str, Any],
    device: torch.device | str,
    solver_steps_override: int | None = None,
    allow_solver_override: bool = False,
    tolerance: float = 1.0e-4,
) -> M3SolverSchedule:
    raw_steps, warped_steps = extract_m3_solver_steps(teacher_payload)
    if solver_steps_override is not None:
        if not allow_solver_override:
            raise ValueError(
                "--solver_steps requires explicit --allow_solver_override"
            )
        timesteps = solver_timesteps_from_scheduler(
            scheduler,
            num_inference_steps=int(solver_steps_override),
            device=device,
        )
        generated = tuple(float(value) for value in timesteps.detach().cpu().tolist())
        return M3SolverSchedule(
            source="cli_override",
            raw_denoising_steps=tuple(raw_steps),
            warped_denoising_steps=tuple(warped_steps),
            generated_timesteps=generated,
            timesteps=timesteps,
            max_abs_diff=None,
            mean_abs_diff=None,
            tolerance=float(tolerance),
            override_steps=int(solver_steps_override),
        )

    timesteps = solver_timesteps_from_scheduler(
        scheduler,
        num_inference_steps=len(warped_steps),
        device=device,
    )
    generated_tensor = timesteps.detach().float().cpu()
    expected_tensor = torch.tensor(warped_steps, dtype=torch.float32)
    if generated_tensor.shape != expected_tensor.shape:
        raise RuntimeError("scheduler timestep count differs from teacher payload")
    diff = (generated_tensor - expected_tensor).abs()
    max_abs = float(diff.max().item())
    mean_abs = float(diff.mean().item())
    if max_abs > float(tolerance):
        raise RuntimeError(
            "scheduler timesteps differ from teacher payload: "
            f"max_abs_diff={max_abs}, tolerance={tolerance}"
        )
    return M3SolverSchedule(
        source="teacher_payload",
        raw_denoising_steps=tuple(raw_steps),
        warped_denoising_steps=tuple(warped_steps),
        generated_timesteps=tuple(float(value) for value in generated_tensor.tolist()),
        timesteps=timesteps,
        max_abs_diff=max_abs,
        mean_abs_diff=mean_abs,
        tolerance=float(tolerance),
        override_steps=None,
    )


def solver_schedule_to_json(schedule: M3SolverSchedule) -> dict[str, Any]:
    return {
        "source": schedule.source,
        "raw_denoising_steps": list(schedule.raw_denoising_steps),
        "warped_denoising_steps": list(schedule.warped_denoising_steps),
        "generated_timesteps": list(schedule.generated_timesteps),
        "max_abs_diff": schedule.max_abs_diff,
        "mean_abs_diff": schedule.mean_abs_diff,
        "tolerance": schedule.tolerance,
        "override_steps": schedule.override_steps,
    }


def reconstruct_main_current(
    generator,
    *,
    conditional_dict: Mapping[str, Any],
    state: NFSFSelectedState,
    initial_noise: torch.Tensor,
    teacher_payload: Mapping[str, Any],
    solver_steps_override: int | None = None,
    allow_solver_override: bool = False,
) -> M3ReconstructionResult:
    scheduler = generator.get_scheduler()
    schedule = resolve_m3_solver_schedule(
        scheduler,
        teacher_payload=teacher_payload,
        device=state.current_target.device,
        solver_steps_override=solver_steps_override,
        allow_solver_override=allow_solver_override,
    )
    timesteps = schedule.timesteps
    current = initial_noise.detach().clone()
    clean_history = state.clean_history
    if clean_history is None:
        raise ValueError("M3 reconstruction requires clean history")

    generator.eval()
    with torch.no_grad():
        for timestep_value in timesteps:
            timestep = torch.full(
                state.current_target.shape[:2],
                float(timestep_value.item()),
                device=state.current_target.device,
                dtype=torch.float32,
            )
            outputs = generator(
                noisy_image_or_video=current,
                conditional_dict=dict(conditional_dict),
                timestep=timestep,
                clean_x=clean_history,
                aug_t=torch.zeros_like(timestep),
            )
            flow_pred = outputs[0]
            current = scheduler.step(
                flow_pred.flatten(0, 1),
                timestep.flatten(0, 1),
                current.flatten(0, 1),
            ).unflatten(0, current.shape[:2])
    return M3ReconstructionResult(latent=current.detach(), solver_schedule=schedule)


def reconstruct_mcp1_next(
    generator,
    *,
    conditional_dict: Mapping[str, Any],
    state: NFSFSelectedState,
    next_initial_noise: torch.Tensor,
    current_condition_noise: torch.Tensor,
    teacher_payload: Mapping[str, Any],
    solver_steps_override: int | None = None,
    allow_solver_override: bool = False,
) -> M3ReconstructionResult:
    scheduler = generator.get_scheduler()
    schedule = resolve_m3_solver_schedule(
        scheduler,
        teacher_payload=teacher_payload,
        device=state.current_target.device,
        solver_steps_override=solver_steps_override,
        allow_solver_override=allow_solver_override,
    )
    timesteps = schedule.timesteps
    next_state = next_initial_noise.detach().clone()
    clean_history = state.clean_history
    if clean_history is None:
        raise ValueError("M3 reconstruction requires clean history")
    future_start = int(state.current_start_frame or M3_CHUNK_FRAMES) + M3_CHUNK_FRAMES

    generator.eval()
    with torch.no_grad():
        for timestep_value in timesteps:
            timestep = torch.full(
                state.current_target.shape[:2],
                float(timestep_value.item()),
                device=state.current_target.device,
                dtype=torch.float32,
            )
            noisy_current = scheduler.add_noise(
                state.current_target.flatten(0, 1),
                current_condition_noise.flatten(0, 1),
                timestep.flatten(0, 1),
            ).unflatten(0, state.current_target.shape[:2])
            outputs = generator(
                noisy_image_or_video=noisy_current,
                conditional_dict=dict(conditional_dict),
                timestep=timestep,
                clean_x=clean_history,
                aug_t=torch.zeros_like(timestep),
                mcp_future_noises=[next_state],
                mcp_future_start_frames=[future_start],
                mcp_timesteps=[timestep],
            )
            mcp_flow = outputs[2][0]
            next_state = scheduler.step(
                mcp_flow.flatten(0, 1),
                timestep.flatten(0, 1),
                next_state.flatten(0, 1),
            ).unflatten(0, next_state.shape[:2])
    return M3ReconstructionResult(latent=next_state.detach(), solver_schedule=schedule)


def compare_loss_dicts(
    actual: Mapping[str, float],
    expected: Mapping[str, float],
) -> dict[str, Any]:
    keys = ("main_loss", "mcp_depth1_loss", "mcp_depth2_loss", "mcp_depth3_loss", "total_loss")
    diffs = {
        key: abs(float(actual[key]) - float(expected[key]))
        for key in keys
    }
    return {
        "keys": list(keys),
        "per_key_abs_diff": diffs,
        "max_abs_diff": max(diffs.values()),
        "mean_abs_diff": sum(diffs.values()) / len(diffs),
    }


def validate_m3_eval_config_matches_checkpoint(
    checkpoint_payload: Mapping[str, Any],
    current_model_config: Mapping[str, Any],
) -> None:
    saved_model_config = checkpoint_payload.get("resolved_config", {}).get("model_config")
    if saved_model_config != current_model_config:
        raise RuntimeError("current eval config differs from checkpoint resolved config")


def validate_m3_checkpoint_git_sha(
    checkpoint_payload: Mapping[str, Any],
    *,
    current_git_sha: str,
) -> None:
    checkpoint_git_sha = validate_git_sha(str(checkpoint_payload.get("git_sha", "")))
    current_git_sha = validate_git_sha(current_git_sha, name="current_git_sha")
    if checkpoint_git_sha != current_git_sha:
        raise RuntimeError("checkpoint git_sha differs from current HEAD")


def validate_m3_checkpoint_pair(
    *,
    initial_payload: Mapping[str, Any],
    final_payload: Mapping[str, Any],
    current_model_config: Mapping[str, Any],
    current_git_sha: str | None = None,
) -> dict[str, Any]:
    validate_m3_checkpoint_payload(initial_payload)
    validate_m3_checkpoint_payload(final_payload)
    if int(initial_payload["global_step"]) != 0:
        raise RuntimeError("initial M3 checkpoint global_step must be 0")
    if int(final_payload["global_step"]) <= 0:
        raise RuntimeError("final M3 checkpoint global_step must be greater than 0")
    validate_m3_eval_config_matches_checkpoint(initial_payload, current_model_config)
    validate_m3_eval_config_matches_checkpoint(final_payload, current_model_config)
    if initial_payload.get("resolved_config") != final_payload.get("resolved_config"):
        raise RuntimeError("initial/final resolved config differs")
    initial_git_sha = validate_git_sha(str(initial_payload["git_sha"]), name="initial.git_sha")
    final_git_sha = validate_git_sha(str(final_payload["git_sha"]), name="final.git_sha")
    if initial_git_sha != final_git_sha:
        raise RuntimeError("initial/final git_sha differs")
    if current_git_sha is not None:
        validate_m3_checkpoint_git_sha(initial_payload, current_git_sha=current_git_sha)
        validate_m3_checkpoint_git_sha(final_payload, current_git_sha=current_git_sha)

    initial_meta = initial_payload["selected_sample_metadata"]
    final_meta = final_payload["selected_sample_metadata"]
    if initial_meta.get("prompt") != final_meta.get("prompt"):
        raise RuntimeError("initial/final prompt differs")
    initial_target_sha = initial_meta.get("target_latent", {}).get("sha256")
    final_target_sha = final_meta.get("target_latent", {}).get("sha256")
    if initial_target_sha != final_target_sha:
        raise RuntimeError("initial/final target_latent SHA differs")

    initial_reference_sha = initial_payload.get("reference_checkpoint", {}).get("sha256")
    final_reference_sha = final_payload.get("reference_checkpoint", {}).get("sha256")
    if initial_reference_sha != final_reference_sha:
        raise RuntimeError("initial/final reference checkpoint SHA differs")
    if int(initial_payload["probe_seed"]) != int(final_payload["probe_seed"]):
        raise RuntimeError("initial/final probe seed differs")

    probe_tensor_comparison = compare_serialized_probe_tensors(
        initial_payload["probe_tensors"],
        final_payload["probe_tensors"],
    )
    if not probe_tensor_comparison["exact"]:
        raise RuntimeError("initial/final probe tensors differ")
    prompt_embedding_comparison = compare_serialized_probe_tensors(
        initial_payload["probe_prompt_embedding"],
        final_payload["probe_prompt_embedding"],
    )
    if not prompt_embedding_comparison["exact"]:
        raise RuntimeError("initial/final probe prompt embedding differs")

    return {
        "status": "PASS",
        "initial_global_step": int(initial_payload["global_step"]),
        "final_global_step": int(final_payload["global_step"]),
        "target_latent_sha256": initial_target_sha,
        "prompt": initial_meta.get("prompt"),
        "reference_checkpoint_sha256": initial_reference_sha,
        "git_sha": final_git_sha,
        "probe_seed": int(final_payload["probe_seed"]),
        "probe_tensor_comparison": probe_tensor_comparison,
        "prompt_embedding_comparison": prompt_embedding_comparison,
    }


def latent_mse_rmse(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    if tuple(prediction.shape) != tuple(target.shape):
        raise ValueError("latent reconstruction shape differs from target")
    diff = prediction.detach().float().cpu() - target.detach().float().cpu()
    mse = float(diff.square().mean().item())
    return {
        "mse": mse,
        "rmse": float(mse ** 0.5),
    }


def reconstruction_metrics(
    *,
    initial_main: torch.Tensor,
    final_main: torch.Tensor,
    initial_mcp1: torch.Tensor,
    final_mcp1: torch.Tensor,
    state: NFSFSelectedState,
) -> dict[str, Any]:
    current_target = state.current_target.detach().cpu()
    next1_target = state.future_targets[0].detach().cpu()
    return {
        "metrics": {
            "initial_main_vs_target_current": latent_mse_rmse(
                initial_main,
                current_target,
            ),
            "final_main_vs_target_current": latent_mse_rmse(
                final_main,
                current_target,
            ),
            "initial_mcp1_vs_target_next1": latent_mse_rmse(
                initial_mcp1,
                next1_target,
            ),
            "final_mcp1_vs_target_next1": latent_mse_rmse(
                final_mcp1,
                next1_target,
            ),
        },
        "tensor_summaries": {
            "target_current": tensor_summary(current_target),
            "target_next1": tensor_summary(next1_target),
            "initial_main": tensor_summary(initial_main),
            "final_main": tensor_summary(final_main),
            "initial_mcp1": tensor_summary(initial_mcp1),
            "final_mcp1": tensor_summary(final_mcp1),
        },
    }
