from __future__ import annotations

import argparse
import gc
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.nf_sf_m3 import (
    M3_CHUNK_FRAMES,
    atomic_json_write,
    load_m3_checkpoint,
    load_m3_teacher_sample,
    move_tensors_to_device,
    resolve_m3_solver_schedule,
    selected_state_to_device,
    solver_schedule_to_json,
    validate_git_sha,
    validate_m3_checkpoint_pair,
    validate_m3_eval_config_matches_checkpoint,
)
from utils.scheduler import FlowMatchScheduler


ALLOWED_DIAGNOSTIC_ONLY_FILES = (
    "scripts/diagnose_nf_sf_m3_trajectory.py",
    "tests/speculative/test_nf_sf_m3_trajectory.py",
)
M3_TRAJECTORY_STEPS = 4
MCP1_FUTURE_START_FRAME = 6
COSINE_EPS = 1.0e-12

GitTextRunner = Callable[[Sequence[str]], str]
GitSuccessRunner = Callable[[Sequence[str]], bool]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="NF-SF v1 M3 read-only trajectory diagnostic."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--m3_checkpoint", required=True, type=Path)
    parser.add_argument("--initial_m3_checkpoint", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--dataset_root", type=Path, default=None)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bf16", "float32"), default="bf16")
    return parser.parse_args()


def dtype_from_arg(value: str) -> torch.dtype:
    if value == "bf16":
        return torch.bfloat16
    if value == "float32":
        return torch.float32
    raise ValueError(f"unsupported dtype: {value}")


def _run_git_text(command: Sequence[str]) -> str:
    return subprocess.check_output(
        list(command),
        cwd=ROOT,
        stderr=subprocess.STDOUT,
        text=True,
    ).strip()


def _run_git_success(command: Sequence[str]) -> bool:
    return (
        subprocess.run(
            list(command),
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        ).returncode
        == 0
    )


def _parse_name_status(diff_text: str) -> list[dict[str, str]]:
    entries = []
    for raw in diff_text.splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            raise RuntimeError(f"invalid git diff --name-status line: {raw!r}")
        entries.append(
            {
                "status": parts[0],
                "path": parts[-1].replace("\\", "/"),
                "raw": raw,
            }
        )
    return entries


def diagnostic_provenance_gate(
    *,
    initial_payload: Mapping[str, Any],
    final_payload: Mapping[str, Any],
    git_text: GitTextRunner | None = None,
    git_success: GitSuccessRunner | None = None,
) -> dict[str, Any]:
    git_text = _run_git_text if git_text is None else git_text
    git_success = _run_git_success if git_success is None else git_success

    initial_git_sha = validate_git_sha(
        str(initial_payload.get("git_sha", "")),
        name="initial.git_sha",
    )
    final_git_sha = validate_git_sha(
        str(final_payload.get("git_sha", "")),
        name="final.git_sha",
    )
    if initial_git_sha != final_git_sha:
        raise RuntimeError("diagnostic provenance failed: initial/final git_sha differs")
    checkpoint_git_sha = final_git_sha
    current_git_sha = validate_git_sha(
        git_text(["git", "rev-parse", "HEAD"]),
        name="current_git_sha",
    )
    branch = git_text(["git", "branch", "--show-current"])
    status_text = git_text(["git", "status", "--short"])
    if status_text.strip():
        raise RuntimeError("diagnostic provenance failed: worktree is dirty")
    if not git_success(["git", "merge-base", "--is-ancestor", checkpoint_git_sha, "HEAD"]):
        raise RuntimeError(
            "diagnostic provenance failed: checkpoint commit is not an ancestor of HEAD"
        )

    diff_text = git_text(["git", "diff", "--name-status", f"{checkpoint_git_sha}..HEAD"])
    entries = _parse_name_status(diff_text)
    expected = {("A", path) for path in ALLOWED_DIAGNOSTIC_ONLY_FILES}
    actual = {(entry["status"], entry["path"]) for entry in entries}
    if actual != expected:
        raise RuntimeError(
            "diagnostic provenance failed: git diff entries do not match "
            f"expected={sorted(expected)!r}, actual={sorted(actual)!r}"
        )

    return {
        "status": "PASS",
        "checkpoint_git_sha": checkpoint_git_sha,
        "current_git_sha": current_git_sha,
        "branch": branch,
        "git_status": status_text,
        "git_diff_name_status": diff_text,
        "git_diff_entries": entries,
        "allowed_diagnostic_only_files": list(ALLOWED_DIAGNOSTIC_ONLY_FILES),
    }


def _require_four_timesteps(name: str, timesteps: torch.Tensor) -> torch.Tensor:
    values = timesteps.detach().float().flatten()
    if values.numel() != M3_TRAJECTORY_STEPS:
        raise RuntimeError(
            f"{name} trajectory must have exactly {M3_TRAJECTORY_STEPS} steps, "
            f"got {int(values.numel())}"
        )
    return values


def _timestep_chunk(value: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.full(
        target.shape[:2],
        float(value.detach().float().item()),
        device=target.device,
        dtype=torch.float32,
    )


def _add_noise_chunk(
    scheduler: Any,
    target: torch.Tensor,
    epsilon: torch.Tensor,
    timestep: torch.Tensor,
) -> torch.Tensor:
    if tuple(target.shape) != tuple(epsilon.shape):
        raise RuntimeError("oracle target/noise shape mismatch")
    noisy = scheduler.add_noise(
        target.flatten(0, 1),
        epsilon.flatten(0, 1),
        timestep.flatten(0, 1),
    ).unflatten(0, target.shape[:2])
    if tuple(noisy.shape) != tuple(target.shape):
        raise RuntimeError(
            "oracle add_noise shape mismatch: "
            f"{tuple(noisy.shape)} != {tuple(target.shape)}"
        )
    return noisy.to(device=target.device, dtype=epsilon.dtype)


def oracle_state_for_step(
    scheduler: Any,
    *,
    target: torch.Tensor,
    epsilon: torch.Tensor,
    timestep: torch.Tensor,
) -> torch.Tensor:
    return _add_noise_chunk(scheduler, target, epsilon, timestep)


def oracle_next_state_for_step(
    scheduler: Any,
    *,
    target: torch.Tensor,
    epsilon: torch.Tensor,
    timesteps: torch.Tensor,
    step_index: int,
) -> torch.Tensor:
    if step_index + 1 >= timesteps.numel():
        return target.detach().clone()
    return _add_noise_chunk(
        scheduler,
        target,
        epsilon,
        _timestep_chunk(timesteps[step_index + 1], target),
    )


def _scheduler_step_preserve_sample_dtype(
    scheduler: Any,
    *,
    model_output: torch.Tensor,
    timestep: torch.Tensor,
    sample: torch.Tensor,
) -> torch.Tensor:
    sample_shape = tuple(sample.shape)
    sample_dtype = sample.dtype
    sample_device = sample.device
    next_sample = scheduler.step(
        model_output,
        timestep,
        sample,
    )
    if tuple(next_sample.shape) != sample_shape:
        raise RuntimeError(
            "trajectory scheduler step shape mismatch: "
            f"{tuple(next_sample.shape)} != {sample_shape}"
        )
    return next_sample.to(device=sample_device, dtype=sample_dtype)


def _require_tensor_contract(
    name: str,
    tensor: torch.Tensor,
    *,
    shape: torch.Size | tuple[int, ...],
    dtype: torch.dtype,
) -> None:
    if tuple(tensor.shape) != tuple(shape):
        raise RuntimeError(f"{name} shape changed: {tuple(tensor.shape)} != {tuple(shape)}")
    if tensor.dtype != dtype:
        raise RuntimeError(f"{name} dtype changed: {tensor.dtype} != {dtype}")
    if not bool(torch.isfinite(tensor.detach().float()).all().item()):
        raise RuntimeError(f"{name} contains non-finite values")


def _finite_float(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise RuntimeError(f"{name} is not finite")
    return value


def _tensor_stats(prefix: str, tensor: torch.Tensor) -> dict[str, float]:
    value = tensor.detach().float()
    finite = bool(torch.isfinite(value).all().item())
    if not finite:
        raise RuntimeError(f"{prefix} tensor contains non-finite values")
    mean = _finite_float(f"{prefix}_mean", float(value.mean().item()))
    std = _finite_float(f"{prefix}_std", float(value.std(unbiased=False).item()))
    rms = _finite_float(f"{prefix}_rms", float(value.square().mean().sqrt().item()))
    return {
        f"{prefix}_mean": mean,
        f"{prefix}_std": std,
        f"{prefix}_rms": rms,
    }


def _mse_rmse(name: str, left: torch.Tensor, right: torch.Tensor) -> tuple[float, float]:
    if tuple(left.shape) != tuple(right.shape):
        raise RuntimeError(
            f"{name} shape mismatch: {tuple(left.shape)} != {tuple(right.shape)}"
        )
    diff = left.detach().float() - right.detach().float()
    mse = _finite_float(f"{name}_mse", float(diff.square().mean().item()))
    return mse, _finite_float(f"{name}_rmse", mse**0.5)


def _flow_cosine_similarity(
    predicted: torch.Tensor,
    target: torch.Tensor,
    *,
    eps: float = COSINE_EPS,
) -> float:
    if tuple(predicted.shape) != tuple(target.shape):
        raise RuntimeError(
            "flow cosine shape mismatch: "
            f"{tuple(predicted.shape)} != {tuple(target.shape)}"
        )
    pred = predicted.detach().float().reshape(-1)
    tgt = target.detach().float().reshape(-1)
    numerator = torch.dot(pred, tgt)
    denominator = pred.square().sum().sqrt() * tgt.square().sum().sqrt() + float(eps)
    cosine = float((numerator / denominator).item())
    return _finite_float("flow_cosine_similarity", cosine)


def _target_flow(target: torch.Tensor, epsilon: torch.Tensor) -> torch.Tensor:
    if tuple(target.shape) != tuple(epsilon.shape):
        raise RuntimeError("target flow shape mismatch")
    return (epsilon - target).to(device=target.device, dtype=epsilon.dtype)


def _step_record(
    *,
    step_index: int,
    main_timestep: torch.Tensor,
    mcp_timestep: torch.Tensor | None,
    state: torch.Tensor,
    oracle_state: torch.Tensor,
    predicted_flow: torch.Tensor,
    target_flow: torch.Tensor,
    post_state: torch.Tensor,
    oracle_next: torch.Tensor,
) -> dict[str, Any]:
    if tuple(predicted_flow.shape) != tuple(target_flow.shape):
        raise RuntimeError(
            "predicted flow shape mismatch: "
            f"{tuple(predicted_flow.shape)} != {tuple(target_flow.shape)}"
        )
    if predicted_flow.dtype != state.dtype:
        raise RuntimeError(
            f"predicted flow dtype changed: {predicted_flow.dtype} != {state.dtype}"
        )
    _require_tensor_contract(
        "trajectory state",
        state,
        shape=oracle_state.shape,
        dtype=oracle_state.dtype,
    )
    _require_tensor_contract(
        "trajectory post state",
        post_state,
        shape=oracle_next.shape,
        dtype=state.dtype,
    )

    flow_mse, flow_rmse = _mse_rmse("flow", predicted_flow, target_flow)
    pre_mse, pre_rmse = _mse_rmse("pre_state_vs_oracle", state, oracle_state)
    post_mse, post_rmse = _mse_rmse(
        "post_state_vs_oracle_next",
        post_state,
        oracle_next,
    )
    record = {
        "step_index": int(step_index),
        "main_timestep": float(main_timestep.detach().float().reshape(-1)[0].item()),
        "mcp_timestep": None
        if mcp_timestep is None
        else float(mcp_timestep.detach().float().reshape(-1)[0].item()),
        "state_shape": [int(dim) for dim in state.shape],
        "state_dtype": str(state.dtype),
        "state_device": str(state.device),
        "finite": True,
        **_tensor_stats("state", state),
        **_tensor_stats("oracle_state", oracle_state),
        "flow_mse": flow_mse,
        "flow_rmse": flow_rmse,
        "flow_cosine_similarity": _flow_cosine_similarity(predicted_flow, target_flow),
        **_tensor_stats("predicted_flow", predicted_flow),
        **_tensor_stats("target_flow", target_flow),
        "pre_state_mse_vs_oracle": pre_mse,
        "pre_state_rmse_vs_oracle": pre_rmse,
        "post_state_mse_vs_oracle_next": post_mse,
        "post_state_rmse_vs_oracle_next": post_rmse,
    }
    for key, value in record.items():
        if isinstance(value, float):
            _finite_float(key, value)
    return record


def _validate_records(records: Sequence[Mapping[str, Any]], *, name: str) -> None:
    if len(records) != M3_TRAJECTORY_STEPS:
        raise RuntimeError(
            f"{name} trajectory record count must be {M3_TRAJECTORY_STEPS}, "
            f"got {len(records)}"
        )
    for index, record in enumerate(records):
        if int(record.get("step_index", -1)) != index:
            raise RuntimeError(f"{name} trajectory step index mismatch at {index}")
        if not bool(record.get("finite")):
            raise RuntimeError(f"{name} trajectory has non-finite step {index}")


def summarize_trajectory(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    _validate_records(records, name="summary")
    flow_mse = [float(record["flow_mse"]) for record in records]
    post_mse = [float(record["post_state_mse_vs_oracle_next"]) for record in records]
    pre_mse = [float(record["pre_state_mse_vs_oracle"]) for record in records]
    summary = {
        "first_flow_mse": flow_mse[0],
        "last_flow_mse": flow_mse[-1],
        "mean_flow_mse": sum(flow_mse) / len(flow_mse),
        "first_post_state_mse": post_mse[0],
        "final_post_state_mse": post_mse[-1],
        "max_post_state_mse": max(post_mse),
        "final_pre_state_mse": pre_mse[-1],
        "all_finite": all(bool(record["finite"]) for record in records),
    }
    for key, value in summary.items():
        if isinstance(value, float):
            _finite_float(key, value)
    return summary


def _trajectory_block(
    teacher_forced: list[dict[str, Any]],
    free_running: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "teacher_forced": teacher_forced,
        "free_running": free_running,
        "summary": {
            "teacher_forced": summarize_trajectory(teacher_forced),
            "free_running": summarize_trajectory(free_running),
        },
    }


def _call_main_generator(
    generator: Any,
    *,
    current_state: torch.Tensor,
    conditional_dict: Mapping[str, Any],
    main_timestep: torch.Tensor,
    clean_history: torch.Tensor,
) -> torch.Tensor:
    outputs = generator(
        noisy_image_or_video=current_state,
        conditional_dict=dict(conditional_dict),
        timestep=main_timestep,
        clean_x=clean_history,
        aug_t=torch.zeros_like(main_timestep),
    )
    if not isinstance(outputs, (tuple, list)) or len(outputs) < 1:
        raise RuntimeError("main diagnostic expected generator outputs with main flow")
    return outputs[0]


def run_main_trajectory(
    generator: Any,
    *,
    conditional_dict: Mapping[str, Any],
    clean_history: torch.Tensor,
    current_target: torch.Tensor,
    epsilon_main: torch.Tensor,
    scheduler: Any,
    timesteps: torch.Tensor,
    mode: str,
) -> list[dict[str, Any]]:
    timesteps = _require_four_timesteps("main", timesteps)
    if mode not in {"teacher_forced", "free_running"}:
        raise ValueError(f"unsupported main trajectory mode: {mode}")
    _require_tensor_contract(
        "main epsilon",
        epsilon_main,
        shape=current_target.shape,
        dtype=current_target.dtype,
    )
    target_flow = _target_flow(current_target, epsilon_main)
    running_state = epsilon_main.detach().clone()
    records: list[dict[str, Any]] = []
    with torch.no_grad():
        for step_index, timestep_value in enumerate(timesteps):
            main_timestep = _timestep_chunk(timestep_value, current_target)
            oracle_state = oracle_state_for_step(
                scheduler,
                target=current_target,
                epsilon=epsilon_main,
                timestep=main_timestep,
            )
            oracle_next = oracle_next_state_for_step(
                scheduler,
                target=current_target,
                epsilon=epsilon_main,
                timesteps=timesteps,
                step_index=step_index,
            )
            current_state = (
                oracle_state.detach().clone()
                if mode == "teacher_forced"
                else running_state
            )
            _require_tensor_contract(
                "main reconstruction state",
                current_state,
                shape=current_target.shape,
                dtype=current_target.dtype,
            )
            predicted_flow = _call_main_generator(
                generator,
                current_state=current_state,
                conditional_dict=conditional_dict,
                main_timestep=main_timestep,
                clean_history=clean_history,
            )
            post_state = _scheduler_step_preserve_sample_dtype(
                scheduler,
                model_output=predicted_flow.flatten(0, 1),
                timestep=main_timestep.flatten(0, 1),
                sample=current_state.flatten(0, 1),
            ).unflatten(0, current_state.shape[:2])
            records.append(
                _step_record(
                    step_index=step_index,
                    main_timestep=main_timestep,
                    mcp_timestep=None,
                    state=current_state,
                    oracle_state=oracle_state,
                    predicted_flow=predicted_flow,
                    target_flow=target_flow,
                    post_state=post_state,
                    oracle_next=oracle_next,
                )
            )
            if mode == "free_running":
                running_state = post_state.detach()
    _validate_records(records, name=f"main/{mode}")
    return records


def _call_mcp1_generator(
    generator: Any,
    *,
    noisy_current_condition: torch.Tensor,
    future_state: torch.Tensor,
    conditional_dict: Mapping[str, Any],
    main_timestep: torch.Tensor,
    mcp_timestep: torch.Tensor,
    clean_history: torch.Tensor,
    future_start_frame: int,
) -> torch.Tensor:
    if int(future_start_frame) != MCP1_FUTURE_START_FRAME:
        raise RuntimeError("MCP-1 diagnostic future_start_frame changed")
    outputs = generator(
        noisy_image_or_video=noisy_current_condition,
        conditional_dict=dict(conditional_dict),
        timestep=main_timestep,
        clean_x=clean_history,
        aug_t=torch.zeros_like(main_timestep),
        mcp_future_noises=[future_state],
        mcp_future_start_frames=[future_start_frame],
        mcp_timesteps=[mcp_timestep],
    )
    if not isinstance(outputs, (tuple, list)) or len(outputs) != 3:
        raise RuntimeError("MCP-1 diagnostic expected exactly three generator outputs")
    mcp_outputs = outputs[2]
    if not isinstance(mcp_outputs, (tuple, list)) or len(mcp_outputs) != 3:
        raise RuntimeError("MCP-1 diagnostic expected exactly three MCP flow outputs")
    return mcp_outputs[0]


def run_mcp1_trajectory(
    generator: Any,
    *,
    conditional_dict: Mapping[str, Any],
    clean_history: torch.Tensor,
    current_target: torch.Tensor,
    next1_target: torch.Tensor,
    epsilon_main: torch.Tensor,
    epsilon_future: torch.Tensor,
    current_condition_scheduler: Any,
    future_scheduler: Any,
    main_timesteps: torch.Tensor,
    mcp_timesteps: torch.Tensor,
    mode: str,
    future_start_frame: int = MCP1_FUTURE_START_FRAME,
) -> list[dict[str, Any]]:
    main_timesteps = _require_four_timesteps("MCP-1 main", main_timesteps)
    mcp_timesteps = _require_four_timesteps("MCP-1 future", mcp_timesteps)
    if mode not in {"teacher_forced", "free_running"}:
        raise ValueError(f"unsupported MCP-1 trajectory mode: {mode}")
    if int(future_start_frame) != MCP1_FUTURE_START_FRAME:
        raise RuntimeError("MCP-1 diagnostic future_start_frame changed")
    _require_tensor_contract(
        "MCP-1 future epsilon",
        epsilon_future,
        shape=next1_target.shape,
        dtype=next1_target.dtype,
    )
    _require_tensor_contract(
        "MCP-1 current condition epsilon",
        epsilon_main,
        shape=current_target.shape,
        dtype=current_target.dtype,
    )
    target_flow = _target_flow(next1_target, epsilon_future)
    running_state = epsilon_future.detach().clone()
    records: list[dict[str, Any]] = []
    with torch.no_grad():
        for step_index, (main_value, mcp_value) in enumerate(
            zip(main_timesteps, mcp_timesteps)
        ):
            main_timestep = _timestep_chunk(main_value, current_target)
            mcp_timestep = _timestep_chunk(mcp_value, next1_target)
            noisy_current_condition = _add_noise_chunk(
                current_condition_scheduler,
                current_target,
                epsilon_main,
                main_timestep,
            )
            oracle_state = oracle_state_for_step(
                future_scheduler,
                target=next1_target,
                epsilon=epsilon_future,
                timestep=mcp_timestep,
            )
            oracle_next = oracle_next_state_for_step(
                future_scheduler,
                target=next1_target,
                epsilon=epsilon_future,
                timesteps=mcp_timesteps,
                step_index=step_index,
            )
            future_state = (
                oracle_state.detach().clone()
                if mode == "teacher_forced"
                else running_state
            )
            _require_tensor_contract(
                "MCP-1 future state",
                future_state,
                shape=next1_target.shape,
                dtype=next1_target.dtype,
            )
            predicted_flow = _call_mcp1_generator(
                generator,
                noisy_current_condition=noisy_current_condition,
                future_state=future_state,
                conditional_dict=conditional_dict,
                main_timestep=main_timestep,
                mcp_timestep=mcp_timestep,
                clean_history=clean_history,
                future_start_frame=future_start_frame,
            )
            post_state = _scheduler_step_preserve_sample_dtype(
                future_scheduler,
                model_output=predicted_flow.flatten(0, 1),
                timestep=mcp_timestep.flatten(0, 1),
                sample=future_state.flatten(0, 1),
            ).unflatten(0, future_state.shape[:2])
            records.append(
                _step_record(
                    step_index=step_index,
                    main_timestep=main_timestep,
                    mcp_timestep=mcp_timestep,
                    state=future_state,
                    oracle_state=oracle_state,
                    predicted_flow=predicted_flow,
                    target_flow=target_flow,
                    post_state=post_state,
                    oracle_next=oracle_next,
                )
            )
            if mode == "free_running":
                running_state = post_state.detach()
    _validate_records(records, name=f"MCP-1/{mode}")
    return records


def scheduler_timesteps_json(scheduler: Any, *, source: str) -> dict[str, Any]:
    timesteps = scheduler.timesteps.detach().float().cpu()
    sigmas = scheduler.sigmas.detach().float().cpu()
    return {
        "source": source,
        "shift": float(getattr(scheduler, "shift", float("nan"))),
        "sigma_min": float(getattr(scheduler, "sigma_min", float("nan"))),
        "extra_one_step": bool(getattr(scheduler, "extra_one_step", False)),
        "generated_timesteps": [float(value) for value in timesteps.tolist()],
        "generated_sigmas": [float(value) for value in sigmas.tolist()],
    }


def build_mcp_shift10_diagnostic_scheduler(
    *,
    num_steps: int,
    device: torch.device | str,
) -> FlowMatchScheduler:
    scheduler = FlowMatchScheduler(shift=10.0, sigma_min=0.0, extra_one_step=True)
    scheduler.set_timesteps(num_steps, denoising_strength=1.0)
    scheduler.sigmas = scheduler.sigmas.to(device)
    scheduler.timesteps = scheduler.timesteps.to(device)
    return scheduler


def run_all_trajectories_for_generator(
    generator: Any,
    *,
    conditional_dict: Mapping[str, Any],
    clean_history: torch.Tensor,
    current_target: torch.Tensor,
    next1_target: torch.Tensor,
    epsilon_main: torch.Tensor,
    epsilon_future: torch.Tensor,
    teacher_payload: Mapping[str, Any],
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any]]:
    generator.eval().requires_grad_(False)
    main_scheduler = generator.get_scheduler()
    main_schedule = resolve_m3_solver_schedule(
        main_scheduler,
        teacher_payload=teacher_payload,
        device=device,
        solver_steps_override=None,
        allow_solver_override=False,
    )
    main_timesteps = _require_four_timesteps("main payload", main_schedule.timesteps)
    mcp_scheduler = build_mcp_shift10_diagnostic_scheduler(
        num_steps=main_timesteps.numel(),
        device=device,
    )
    mcp_timesteps = _require_four_timesteps(
        "MCP shift10 diagnostic",
        mcp_scheduler.timesteps,
    )

    with torch.no_grad():
        main_teacher = run_main_trajectory(
            generator,
            conditional_dict=conditional_dict,
            clean_history=clean_history,
            current_target=current_target,
            epsilon_main=epsilon_main,
            scheduler=main_scheduler,
            timesteps=main_timesteps,
            mode="teacher_forced",
        )
        main_free = run_main_trajectory(
            generator,
            conditional_dict=conditional_dict,
            clean_history=clean_history,
            current_target=current_target,
            epsilon_main=epsilon_main,
            scheduler=main_scheduler,
            timesteps=main_timesteps,
            mode="free_running",
        )
        baseline_teacher = run_mcp1_trajectory(
            generator,
            conditional_dict=conditional_dict,
            clean_history=clean_history,
            current_target=current_target,
            next1_target=next1_target,
            epsilon_main=epsilon_main,
            epsilon_future=epsilon_future,
            current_condition_scheduler=main_scheduler,
            future_scheduler=main_scheduler,
            main_timesteps=main_timesteps,
            mcp_timesteps=main_timesteps,
            mode="teacher_forced",
            future_start_frame=MCP1_FUTURE_START_FRAME,
        )
        baseline_free = run_mcp1_trajectory(
            generator,
            conditional_dict=conditional_dict,
            clean_history=clean_history,
            current_target=current_target,
            next1_target=next1_target,
            epsilon_main=epsilon_main,
            epsilon_future=epsilon_future,
            current_condition_scheduler=main_scheduler,
            future_scheduler=main_scheduler,
            main_timesteps=main_timesteps,
            mcp_timesteps=main_timesteps,
            mode="free_running",
            future_start_frame=MCP1_FUTURE_START_FRAME,
        )
        aligned_teacher = run_mcp1_trajectory(
            generator,
            conditional_dict=conditional_dict,
            clean_history=clean_history,
            current_target=current_target,
            next1_target=next1_target,
            epsilon_main=epsilon_main,
            epsilon_future=epsilon_future,
            current_condition_scheduler=main_scheduler,
            future_scheduler=mcp_scheduler,
            main_timesteps=main_timesteps,
            mcp_timesteps=mcp_timesteps,
            mode="teacher_forced",
            future_start_frame=MCP1_FUTURE_START_FRAME,
        )
        aligned_free = run_mcp1_trajectory(
            generator,
            conditional_dict=conditional_dict,
            clean_history=clean_history,
            current_target=current_target,
            next1_target=next1_target,
            epsilon_main=epsilon_main,
            epsilon_future=epsilon_future,
            current_condition_scheduler=main_scheduler,
            future_scheduler=mcp_scheduler,
            main_timesteps=main_timesteps,
            mcp_timesteps=mcp_timesteps,
            mode="free_running",
            future_start_frame=MCP1_FUTURE_START_FRAME,
        )

    report = {
        "main": _trajectory_block(main_teacher, main_free),
        "mcp1_baseline_shared_main_schedule": _trajectory_block(
            baseline_teacher,
            baseline_free,
        ),
        "mcp1_training_aligned_mcp_schedule": {
            "diagnostic_alternative": True,
            **_trajectory_block(aligned_teacher, aligned_free),
        },
    }
    schedules = {
        "main_teacher_payload": solver_schedule_to_json(main_schedule),
        "mcp_shift10_diagnostic": scheduler_timesteps_json(
            mcp_scheduler,
            source="diagnostic_alternative_training_aligned_mcp_schedule",
        ),
    }
    return report, schedules


def validate_trajectory_checkpoint_steps(checkpoint_pair: Mapping[str, Any]) -> None:
    initial_step = int(checkpoint_pair.get("initial_global_step", -1))
    final_step = int(checkpoint_pair.get("final_global_step", -1))
    if initial_step != 0 or final_step != 100:
        raise RuntimeError(
            "trajectory diagnostic requires checkpoint steps 0 -> 100, "
            f"got {initial_step} -> {final_step}"
        )


def run_checkpoint_trajectories(
    *,
    label: str,
    config: Any,
    checkpoint_payload: dict[str, Any],
    conditional_dict: Mapping[str, Any],
    clean_history: torch.Tensor,
    current_target: torch.Tensor,
    next1_target: torch.Tensor,
    epsilon_main: torch.Tensor,
    epsilon_future: torch.Tensor,
    teacher_payload: Mapping[str, Any],
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[dict[str, Any], dict[str, Any]]:
    from scripts.eval_nf_sf_m3_overfit import load_generator_from_m3_checkpoint

    generator = load_generator_from_m3_checkpoint(
        config=config,
        checkpoint_payload=checkpoint_payload,
        device=device,
        dtype=dtype,
    )
    try:
        return run_all_trajectories_for_generator(
            generator,
            conditional_dict=conditional_dict,
            clean_history=clean_history,
            current_target=current_target,
            next1_target=next1_target,
            epsilon_main=epsilon_main,
            epsilon_future=epsilon_future,
            teacher_payload=teacher_payload,
            device=device,
        )
    except Exception as exc:
        raise RuntimeError(f"{label} checkpoint trajectory diagnostic failed: {exc}") from exc
    finally:
        generator.to("cpu")
        del generator
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()

    initial_payload = load_m3_checkpoint(args.initial_m3_checkpoint)
    final_payload = load_m3_checkpoint(args.m3_checkpoint)
    provenance = diagnostic_provenance_gate(
        initial_payload=initial_payload,
        final_payload=final_payload,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json_write(provenance, args.output_dir / "provenance.json")

    from inference_mcp import merge_config, require_single_gpu_runtime
    from scripts.eval_nf_sf_m3_overfit import (
        compare_sample_metadata,
        conditional_dict_for_checkpoint,
        resolved_config_dict,
        validate_config,
    )

    device = require_single_gpu_runtime(torch, args.device)
    dtype = dtype_from_arg(args.dtype)
    config = merge_config(str(args.config))
    validate_config(config)
    current_model_config = resolved_config_dict(config)
    validate_m3_eval_config_matches_checkpoint(final_payload, current_model_config)
    checkpoint_pair = validate_m3_checkpoint_pair(
        initial_payload=initial_payload,
        final_payload=final_payload,
        current_model_config=current_model_config,
    )
    validate_trajectory_checkpoint_steps(checkpoint_pair)

    metadata = final_payload["selected_sample_metadata"]
    sample = load_m3_teacher_sample(
        manifest_path=args.manifest,
        dataset_root=args.dataset_root,
        sample_index=int(metadata["sample_index"]),
        reference_checkpoint_path=final_payload["reference_checkpoint"]["path"],
    )
    compare_sample_metadata(metadata, sample.metadata)

    state = selected_state_to_device(sample.selected_state, device=device, dtype=dtype)
    if state.clean_history is None:
        raise RuntimeError("M3 trajectory diagnostic requires clean history")
    future_start = int(state.current_start_frame or M3_CHUNK_FRAMES) + M3_CHUNK_FRAMES
    if future_start != MCP1_FUTURE_START_FRAME:
        raise RuntimeError("MCP-1 diagnostic future_start_frame changed")

    probe_tensors = move_tensors_to_device(
        final_payload["probe_tensors"],
        device=device,
        floating_dtype=dtype,
    )
    epsilon_main = probe_tensors["epsilon_main"]
    epsilon_future = probe_tensors["epsilon_depths"][0]
    _require_tensor_contract(
        "probe epsilon_main",
        epsilon_main,
        shape=state.current_target.shape,
        dtype=state.current_target.dtype,
    )
    _require_tensor_contract(
        "probe epsilon_depths[0]",
        epsilon_future,
        shape=state.future_targets[0].shape,
        dtype=state.future_targets[0].dtype,
    )

    conditional_dict = conditional_dict_for_checkpoint(
        payload=final_payload,
        prompt=sample.metadata["prompt"],
        device=device,
        dtype=dtype,
    )

    initial_report, initial_schedules = run_checkpoint_trajectories(
        label="initial",
        config=config,
        checkpoint_payload=initial_payload,
        conditional_dict=conditional_dict,
        clean_history=state.clean_history,
        current_target=state.current_target,
        next1_target=state.future_targets[0],
        epsilon_main=epsilon_main,
        epsilon_future=epsilon_future,
        teacher_payload=sample.payload,
        device=device,
        dtype=dtype,
    )
    final_report, final_schedules = run_checkpoint_trajectories(
        label="final",
        config=config,
        checkpoint_payload=final_payload,
        conditional_dict=conditional_dict,
        clean_history=state.clean_history,
        current_target=state.current_target,
        next1_target=state.future_targets[0],
        epsilon_main=epsilon_main,
        epsilon_future=epsilon_future,
        teacher_payload=sample.payload,
        device=device,
        dtype=dtype,
    )
    if initial_schedules != final_schedules:
        raise RuntimeError("initial/final trajectory schedules differ")

    report = {
        "status": "PASS",
        "diagnostic_only": True,
        "checkpoint_pair": checkpoint_pair,
        "provenance": {
            "checkpoint_git_sha": provenance["checkpoint_git_sha"],
            "current_git_sha": provenance["current_git_sha"],
        },
        "sample": {
            "sample_index": int(sample.metadata["sample_index"]),
            "sample_id": sample.metadata.get("sample_id"),
            "split": sample.metadata["split"],
            "split_index": int(sample.metadata["split_index"]),
            "target_latent_sha256": sample.metadata["target_latent"]["sha256"],
            "probe_seed": int(final_payload["probe_seed"]),
        },
        "schedules": initial_schedules,
        "initial": initial_report,
        "final": final_report,
    }
    atomic_json_write(report, args.output_dir / "trajectory_diagnostic.json")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
