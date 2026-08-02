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

from scripts.diagnose_nf_sf_m3_trajectory import (
    MCP1_FUTURE_START_FRAME,
    run_mcp1_trajectory,
    summarize_trajectory,
)
from utils.nf_sf_m3 import (
    atomic_json_write,
    file_sha256,
    load_m3_checkpoint,
    load_m3_teacher_sample,
    move_tensors_to_device,
    reconstruct_mcp1_next,
    resolve_m3_solver_schedule,
    selected_state_to_device,
    tensor_summary,
    validate_git_sha,
    validate_m3_eval_config_matches_checkpoint,
)


BASE_M3_CHECKPOINT_GIT_SHA = "3c49a1c6cb7b3da19cbe0e77a8ee58e9f286b3bf"
EXPECTED_FIXED_TIMESTEPS = (
    1000.0,
    937.5,
    833.3333129882812,
    625.0,
)
ALLOWED_FIXED_GRID_PROVENANCE_FILES = (
    "scripts/diagnose_nf_sf_m3_trajectory.py",
    "tests/speculative/test_nf_sf_m3_trajectory.py",
    "scripts/run_nf_sf_m3_fixed_grid_probe.py",
    "tests/speculative/test_nf_sf_m3_fixed_grid_probe.py",
)
PARAMETER_GROUP_NAMES = (
    "backbone",
    "patch_embedding",
    "mcp_fusion",
    "mcp_depth1",
    "mcp_depth2",
    "mcp_depth3",
    "other",
)
TRAINABLE_GROUPS = ("mcp_fusion", "mcp_depth1")
COSINE_EPS = 1.0e-12
TIMESTEP_TOLERANCE = 1.0e-4

GitTextRunner = Callable[[Sequence[str]], str]
GitSuccessRunner = Callable[[Sequence[str]], bool]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="NF-SF M3.1 fixed inference-grid MCP-1 micro-overfit probe."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--m3_checkpoint", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--dataset_root", type=Path, default=None)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bf16", "float32"), default="bf16")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--log_interval", type=int, default=1)
    parser.add_argument("--mode", choices=("train", "eval"), default="train")
    parser.add_argument("--probe_checkpoint", type=Path, default=None)
    parser.add_argument("--decode", action="store_true")
    parser.add_argument("--decode_output_dir", type=Path, default=None)
    return parser.parse_args()


def dtype_from_arg(value: str) -> torch.dtype:
    if value == "bf16":
        return torch.bfloat16
    if value == "float32":
        return torch.float32
    raise ValueError(f"unsupported dtype: {value}")


def validate_cli_values(*, steps: int, lr: float, log_interval: int) -> None:
    if int(steps) < 0:
        raise ValueError("--steps must be >= 0")
    if float(lr) <= 0.0:
        raise ValueError("--lr must be > 0")
    if int(log_interval) <= 0:
        raise ValueError("--log_interval must be > 0")


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


def fixed_grid_provenance_gate(
    *,
    checkpoint_payload: Mapping[str, Any],
    git_text: GitTextRunner | None = None,
    git_success: GitSuccessRunner | None = None,
) -> dict[str, Any]:
    git_text = _run_git_text if git_text is None else git_text
    git_success = _run_git_success if git_success is None else git_success
    checkpoint_git_sha = validate_git_sha(str(checkpoint_payload.get("git_sha", "")))
    if checkpoint_git_sha != BASE_M3_CHECKPOINT_GIT_SHA:
        raise RuntimeError(
            "fixed-grid probe requires base checkpoint git_sha "
            f"{BASE_M3_CHECKPOINT_GIT_SHA}, got {checkpoint_git_sha}"
        )
    current_git_sha = validate_git_sha(
        git_text(["git", "rev-parse", "HEAD"]),
        name="current_git_sha",
    )
    branch = git_text(["git", "branch", "--show-current"])
    status_text = git_text(["git", "status", "--short"])
    if status_text.strip():
        raise RuntimeError("fixed-grid provenance failed: worktree is dirty")
    if not git_success(["git", "merge-base", "--is-ancestor", checkpoint_git_sha, "HEAD"]):
        raise RuntimeError(
            "fixed-grid provenance failed: checkpoint commit is not an ancestor of HEAD"
        )
    diff_text = git_text(["git", "diff", "--name-status", f"{checkpoint_git_sha}..HEAD"])
    entries = _parse_name_status(diff_text)
    expected = {("A", path) for path in ALLOWED_FIXED_GRID_PROVENANCE_FILES}
    actual = {(entry["status"], entry["path"]) for entry in entries}
    if actual != expected:
        raise RuntimeError(
            "fixed-grid provenance failed: git diff entries do not match "
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
        "allowed_fixed_grid_files": list(ALLOWED_FIXED_GRID_PROVENANCE_FILES),
    }


def validate_step100_checkpoint(checkpoint_payload: Mapping[str, Any]) -> None:
    global_step = int(checkpoint_payload.get("global_step", -1))
    if global_step != 100:
        raise RuntimeError(f"fixed-grid probe requires M3 global_step=100, got {global_step}")


def _timestep_chunk(value: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.full(
        target.shape[:2],
        float(value.detach().float().item()),
        device=target.device,
        dtype=torch.float32,
    )


def add_noise_chunk(
    scheduler: Any,
    target: torch.Tensor,
    epsilon: torch.Tensor,
    timestep: torch.Tensor,
) -> torch.Tensor:
    if tuple(target.shape) != tuple(epsilon.shape):
        raise RuntimeError("fixed-grid target/noise shape mismatch")
    noisy = scheduler.add_noise(
        target.flatten(0, 1),
        epsilon.flatten(0, 1),
        timestep.flatten(0, 1),
    ).unflatten(0, target.shape[:2])
    if tuple(noisy.shape) != tuple(target.shape):
        raise RuntimeError(
            "fixed-grid add_noise shape mismatch: "
            f"{tuple(noisy.shape)} != {tuple(target.shape)}"
        )
    return noisy.to(device=target.device, dtype=epsilon.dtype)


def _finite_float(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise RuntimeError(f"{name} is not finite")
    return value


def _stats(tensor: torch.Tensor, prefix: str) -> dict[str, float]:
    value = tensor.detach().float()
    if not bool(torch.isfinite(value).all().item()):
        raise RuntimeError(f"{prefix} contains non-finite values")
    return {
        f"{prefix}_mean": _finite_float(f"{prefix}_mean", float(value.mean().item())),
        f"{prefix}_std": _finite_float(
            f"{prefix}_std",
            float(value.std(unbiased=False).item()),
        ),
        f"{prefix}_rms": _finite_float(
            f"{prefix}_rms",
            float(value.square().mean().sqrt().item()),
        ),
    }


def flow_metrics(predicted_flow: torch.Tensor, target_flow: torch.Tensor) -> dict[str, Any]:
    if tuple(predicted_flow.shape) != tuple(target_flow.shape):
        raise RuntimeError(
            "fixed-grid flow shape mismatch: "
            f"{tuple(predicted_flow.shape)} != {tuple(target_flow.shape)}"
        )
    if not bool(torch.isfinite(predicted_flow.detach().float()).all().item()):
        raise RuntimeError("fixed-grid predicted flow contains non-finite values")
    if not bool(torch.isfinite(target_flow.detach().float()).all().item()):
        raise RuntimeError("fixed-grid target flow contains non-finite values")
    pred = predicted_flow.detach().float()
    target = target_flow.detach().float()
    diff = pred - target
    mse = _finite_float("flow_mse", float(diff.square().mean().item()))
    pred_rms = _finite_float("predicted_flow_rms", float(pred.square().mean().sqrt().item()))
    target_rms = _finite_float("target_flow_rms", float(target.square().mean().sqrt().item()))
    numerator = torch.dot(pred.reshape(-1), target.reshape(-1))
    denominator = pred.square().sum().sqrt() * target.square().sum().sqrt() + COSINE_EPS
    cosine = _finite_float("flow_cosine_similarity", float((numerator / denominator).item()))
    return {
        "flow_mse": mse,
        "flow_rmse": _finite_float("flow_rmse", mse**0.5),
        "flow_cosine_similarity": cosine,
        "predicted_flow_rms": pred_rms,
        "target_flow_rms": target_rms,
        "predicted_to_target_rms_ratio": _finite_float(
            "predicted_to_target_rms_ratio",
            pred_rms / (target_rms + COSINE_EPS),
        ),
        **_stats(predicted_flow, "predicted_flow"),
        **_stats(target_flow, "target_flow"),
        "finite": True,
    }


def aggregate_grid_metrics(records: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    if len(records) != len(EXPECTED_FIXED_TIMESTEPS):
        raise RuntimeError(f"fixed-grid metric record count must be 4, got {len(records)}")
    mses = [float(record["flow_mse"]) for record in records]
    cosines = [float(record["flow_cosine_similarity"]) for record in records]
    ratios = [float(record["predicted_to_target_rms_ratio"]) for record in records]
    summary = {
        "mean_flow_mse": sum(mses) / len(mses),
        "mean_cosine": sum(cosines) / len(cosines),
        "mean_rms_ratio": sum(ratios) / len(ratios),
        "max_flow_mse": max(mses),
        "min_cosine": min(cosines),
    }
    for key, value in summary.items():
        _finite_float(key, value)
    return summary


def extract_single_mcp_flow(outputs: Any) -> torch.Tensor:
    if not isinstance(outputs, (tuple, list)) or len(outputs) != 3:
        raise RuntimeError("fixed-grid probe expected exactly three generator outputs")
    mcp_outputs = outputs[2]
    if not isinstance(mcp_outputs, (tuple, list)) or len(mcp_outputs) != 1:
        actual_count = "non-sequence" if not isinstance(mcp_outputs, (tuple, list)) else len(mcp_outputs)
        raise RuntimeError(
            "fixed-grid probe expected exactly one MCP flow output "
            f"for one requested future chunk, got {actual_count}"
        )
    return mcp_outputs[0]


def validate_fixed_grid_timesteps(timesteps: torch.Tensor) -> torch.Tensor:
    values = timesteps.detach().float().flatten()
    if values.numel() != len(EXPECTED_FIXED_TIMESTEPS):
        raise RuntimeError(f"fixed-grid probe requires exactly four timesteps, got {values.numel()}")
    expected = torch.tensor(EXPECTED_FIXED_TIMESTEPS, dtype=torch.float32)
    diff = (values.detach().cpu() - expected).abs()
    max_abs = float(diff.max().item())
    if max_abs > TIMESTEP_TOLERANCE:
        raise RuntimeError(
            "fixed-grid scheduler timesteps differ from expected inference grid: "
            f"max_abs_diff={max_abs}, tolerance={TIMESTEP_TOLERANCE}"
        )
    return values


def resolve_fixed_grid_schedule(
    generator: Any,
    *,
    teacher_payload: Mapping[str, Any],
    device: torch.device | str,
) -> tuple[Any, torch.Tensor, dict[str, Any]]:
    scheduler = generator.get_scheduler()
    schedule = resolve_m3_solver_schedule(
        scheduler,
        teacher_payload=teacher_payload,
        device=device,
        solver_steps_override=None,
        allow_solver_override=False,
    )
    timesteps = validate_fixed_grid_timesteps(schedule.timesteps)
    return scheduler, timesteps, {
        "source": schedule.source,
        "raw_denoising_steps": list(schedule.raw_denoising_steps),
        "warped_denoising_steps": list(schedule.warped_denoising_steps),
        "generated_timesteps": list(schedule.generated_timesteps),
        "max_abs_diff": schedule.max_abs_diff,
        "mean_abs_diff": schedule.mean_abs_diff,
        "tolerance": schedule.tolerance,
    }


def fixed_grid_loss_and_records(
    generator: Any,
    *,
    conditional_dict: Mapping[str, Any],
    clean_history: torch.Tensor,
    current_target: torch.Tensor,
    next1_target: torch.Tensor,
    epsilon_main: torch.Tensor,
    epsilon_future: torch.Tensor,
    scheduler: Any,
    timesteps: torch.Tensor,
) -> tuple[torch.Tensor, list[dict[str, Any]], dict[str, float]]:
    timesteps = validate_fixed_grid_timesteps(timesteps)
    target_flow = (epsilon_future - next1_target).to(
        device=next1_target.device,
        dtype=epsilon_future.dtype,
    )
    losses = []
    records: list[dict[str, Any]] = []
    for index, timestep_value in enumerate(timesteps):
        timestep = _timestep_chunk(timestep_value, current_target)
        current_xt = add_noise_chunk(
            scheduler,
            current_target,
            epsilon_main,
            timestep,
        )
        future_xt = add_noise_chunk(
            scheduler,
            next1_target,
            epsilon_future,
            timestep,
        )
        outputs = generator(
            noisy_image_or_video=current_xt,
            conditional_dict=dict(conditional_dict),
            timestep=timestep,
            clean_x=clean_history,
            aug_t=torch.zeros_like(timestep),
            mcp_future_noises=[future_xt],
            mcp_future_start_frames=[MCP1_FUTURE_START_FRAME],
            mcp_timesteps=[timestep],
        )
        flow_pred = extract_single_mcp_flow(outputs)
        if not bool(torch.isfinite(flow_pred.detach().float()).all().item()):
            raise RuntimeError("fixed-grid predicted flow contains non-finite values")
        loss = (flow_pred.float() - target_flow.float()).square().mean()
        if not bool(torch.isfinite(loss.detach()).all().item()):
            raise RuntimeError("fixed-grid loss is not finite")
        metrics = flow_metrics(flow_pred, target_flow)
        records.append(
            {
                "grid_index": int(index),
                "timestep": float(timestep_value.detach().float().item()),
                **metrics,
            }
        )
        losses.append(loss)
    mean_loss = torch.stack(losses).mean()
    if not bool(torch.isfinite(mean_loss.detach()).all().item()):
        raise RuntimeError("fixed-grid mean loss is not finite")
    return mean_loss, records, aggregate_grid_metrics(records)


def fixed_grid_point_loss_and_record(
    generator: Any,
    *,
    conditional_dict: Mapping[str, Any],
    clean_history: torch.Tensor,
    current_target: torch.Tensor,
    next1_target: torch.Tensor,
    epsilon_main: torch.Tensor,
    epsilon_future: torch.Tensor,
    scheduler: Any,
    timestep_value: torch.Tensor,
    grid_index: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    target_flow = (epsilon_future - next1_target).to(
        device=next1_target.device,
        dtype=epsilon_future.dtype,
    )
    timestep = _timestep_chunk(timestep_value, current_target)
    current_xt = add_noise_chunk(
        scheduler,
        current_target,
        epsilon_main,
        timestep,
    )
    future_xt = add_noise_chunk(
        scheduler,
        next1_target,
        epsilon_future,
        timestep,
    )
    outputs = generator(
        noisy_image_or_video=current_xt,
        conditional_dict=dict(conditional_dict),
        timestep=timestep,
        clean_x=clean_history,
        aug_t=torch.zeros_like(timestep),
        mcp_future_noises=[future_xt],
        mcp_future_start_frames=[MCP1_FUTURE_START_FRAME],
        mcp_timesteps=[timestep],
    )
    flow_pred = extract_single_mcp_flow(outputs)
    if not bool(torch.isfinite(flow_pred.detach().float()).all().item()):
        raise RuntimeError("fixed-grid predicted flow contains non-finite values")
    point_loss = (flow_pred.float() - target_flow.float()).square().mean()
    if not bool(torch.isfinite(point_loss.detach()).all().item()):
        raise RuntimeError("fixed-grid point loss is not finite")
    metrics = flow_metrics(flow_pred.detach(), target_flow.detach())
    return point_loss, {
        "grid_index": int(grid_index),
        "timestep": float(timestep_value.detach().float().item()),
        **metrics,
    }


def parameter_group_for_name(name: str) -> str:
    normalized = name.replace("\\", "/")
    if normalized.startswith("mcp.fusion."):
        return "mcp_fusion"
    if normalized.startswith("mcp.mcp_modules.0."):
        return "mcp_depth1"
    if normalized.startswith("mcp.mcp_modules.1."):
        return "mcp_depth2"
    if normalized.startswith("mcp.mcp_modules.2."):
        return "mcp_depth3"
    if ".patch_embedding." in normalized or normalized.startswith("model.patch_embedding."):
        return "patch_embedding"
    if normalized.startswith("model.") or normalized.startswith("backbone."):
        return "backbone"
    return "other"


def configure_fixed_grid_trainable_parameters(generator: Any) -> dict[str, Any]:
    target_groups = set(TRAINABLE_GROUPS)
    group_counts = {name: 0 for name in PARAMETER_GROUP_NAMES}
    trainable_counts = {name: 0 for name in PARAMETER_GROUP_NAMES}
    for name, parameter in generator.named_parameters():
        group = parameter_group_for_name(name)
        group_counts[group] += 1
        should_train = group in target_groups
        parameter.requires_grad_(should_train)
        if should_train:
            trainable_counts[group] += 1
    missing = [group for group in TRAINABLE_GROUPS if group_counts[group] == 0]
    if missing:
        raise RuntimeError(f"fixed-grid probe missing trainable parameter groups: {missing}")
    return {
        group: {
            "tensor_count": int(group_counts[group]),
            "trainable_tensor_count": int(trainable_counts[group]),
            "target_trainable": group in target_groups,
        }
        for group in PARAMETER_GROUP_NAMES
    }


def make_fixed_grid_optimizer(generator: Any, *, lr: float) -> torch.optim.Optimizer:
    if float(lr) <= 0.0:
        raise ValueError("--lr must be > 0")
    fusion_params = [
        parameter
        for name, parameter in generator.named_parameters()
        if parameter_group_for_name(name) == "mcp_fusion" and parameter.requires_grad
    ]
    mcp1_params = [
        parameter
        for name, parameter in generator.named_parameters()
        if parameter_group_for_name(name) == "mcp_depth1" and parameter.requires_grad
    ]
    if not fusion_params or not mcp1_params:
        raise RuntimeError("fixed-grid optimizer requires fusion and MCP-1 parameters")
    return torch.optim.AdamW(
        [
            {"name": "mcp_fusion", "params": fusion_params, "lr": float(lr)},
            {"name": "mcp_depth1", "params": mcp1_params, "lr": float(lr)},
        ],
        lr=float(lr),
        weight_decay=0.0,
    )


def optimizer_parameter_ids(optimizer: torch.optim.Optimizer) -> set[int]:
    return {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group.get("params", [])
    }


def snapshot_named_parameters(generator: Any) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in generator.named_parameters()
    }


def parameter_group_audit(
    generator: Any,
    *,
    optimizer: torch.optim.Optimizer,
    before_snapshot: Mapping[str, torch.Tensor] | None = None,
    after_snapshot: Mapping[str, torch.Tensor] | None = None,
) -> dict[str, Any]:
    optimizer_ids = optimizer_parameter_ids(optimizer)
    groups: dict[str, dict[str, Any]] = {
        name: {
            "tensor_count": 0,
            "requires_grad": True,
            "in_optimizer": True,
            "has_gradient": False,
            "gradient_finite": True,
            "gradient_norm": 0.0,
            "requires_grad_tensor_count": 0,
            "optimizer_tensor_count": 0,
            "gradient_tensor_count": 0,
            "parameter_changed": False,
            "changed_tensor_count": 0,
            "max_abs_parameter_diff": 0.0,
        }
        for name in PARAMETER_GROUP_NAMES
    }
    for name, parameter in generator.named_parameters():
        group_name = parameter_group_for_name(name)
        group = groups[group_name]
        group["tensor_count"] += 1
        if parameter.requires_grad:
            group["requires_grad_tensor_count"] += 1
        if id(parameter) in optimizer_ids:
            group["optimizer_tensor_count"] += 1
        group["requires_grad"] = (
            int(group["requires_grad_tensor_count"]) == int(group["tensor_count"])
        )
        group["in_optimizer"] = (
            int(group["optimizer_tensor_count"]) == int(group["tensor_count"])
        )
        grad = parameter.grad
        if grad is not None:
            group["has_gradient"] = True
            group["gradient_tensor_count"] += 1
            grad_value = grad.detach().float()
            finite = bool(torch.isfinite(grad_value).all().item())
            group["gradient_finite"] = bool(group["gradient_finite"]) and finite
            group["gradient_norm"] += float(grad_value.square().sum().item())
        if before_snapshot is not None and after_snapshot is not None:
            before = before_snapshot[name]
            after = after_snapshot[name]
            if tuple(before.shape) != tuple(after.shape) or before.dtype != after.dtype:
                raise RuntimeError(f"parameter snapshot contract changed for {name}")
            diff = (after.float() - before.float()).abs()
            max_abs = float(diff.max().item()) if diff.numel() else 0.0
            group["max_abs_parameter_diff"] = max(
                float(group["max_abs_parameter_diff"]),
                max_abs,
            )
            if max_abs > 0.0:
                group["parameter_changed"] = True
                group["changed_tensor_count"] += 1
    for group in groups.values():
        if group["tensor_count"] == 0:
            group["requires_grad"] = False
            group["in_optimizer"] = False
        group["gradient_norm"] = _finite_float(
            "gradient_norm",
            float(group["gradient_norm"]) ** 0.5,
        )
        group["gradient_finite"] = bool(group["gradient_finite"]) and math.isfinite(
            group["gradient_norm"]
        )
    return groups


def assert_gradients_finite(audit: Mapping[str, Mapping[str, Any]]) -> None:
    for group_name, group in audit.items():
        if not bool(group.get("gradient_finite", False)):
            raise RuntimeError(f"fixed-grid gradient is non-finite for {group_name}")


def assert_fixed_grid_gradient_contract(audit: Mapping[str, Mapping[str, Any]]) -> None:
    target_groups = set(TRAINABLE_GROUPS)
    for group_name, group in audit.items():
        tensor_count = int(group.get("tensor_count", 0))
        requires_grad_count = int(group.get("requires_grad_tensor_count", 0))
        optimizer_count = int(group.get("optimizer_tensor_count", 0))
        gradient_count = int(group.get("gradient_tensor_count", 0))
        gradient_norm = float(group.get("gradient_norm", 0.0))
        if group_name in target_groups:
            if tensor_count <= 0:
                raise RuntimeError(f"fixed-grid target group has no parameters: {group_name}")
            if not bool(group.get("requires_grad", False)) or requires_grad_count != tensor_count:
                raise RuntimeError(f"fixed-grid target group is not trainable: {group_name}")
            if not bool(group.get("in_optimizer", False)) or optimizer_count != tensor_count:
                raise RuntimeError(f"fixed-grid target group is not in optimizer: {group_name}")
            if not bool(group.get("has_gradient", False)) or gradient_count <= 0:
                raise RuntimeError(f"fixed-grid target group has no gradient: {group_name}")
            if not bool(group.get("gradient_finite", False)) or not math.isfinite(gradient_norm):
                raise RuntimeError(f"fixed-grid target gradient is non-finite: {group_name}")
            continue
        if requires_grad_count != 0 or bool(group.get("requires_grad", False)):
            raise RuntimeError(f"fixed-grid non-target group requires grad: {group_name}")
        if optimizer_count != 0 or bool(group.get("in_optimizer", False)):
            raise RuntimeError(f"fixed-grid non-target group is in optimizer: {group_name}")
        if gradient_count != 0 or bool(group.get("has_gradient", False)):
            raise RuntimeError(f"fixed-grid non-target group has gradient: {group_name}")


def assert_only_target_parameters_changed(audit: Mapping[str, Mapping[str, Any]]) -> None:
    target_groups = set(TRAINABLE_GROUPS)
    for group_name, group in audit.items():
        if group_name not in target_groups and bool(group.get("parameter_changed", False)):
            raise RuntimeError(f"fixed-grid non-target parameter changed: {group_name}")


def run_fixed_grid_optimizer_step(
    generator: Any,
    *,
    optimizer: torch.optim.Optimizer,
    conditional_dict: Mapping[str, Any],
    clean_history: torch.Tensor,
    current_target: torch.Tensor,
    next1_target: torch.Tensor,
    epsilon_main: torch.Tensor,
    epsilon_future: torch.Tensor,
    scheduler: Any,
    timesteps: torch.Tensor,
    evaluate_post_update: bool = False,
) -> dict[str, Any]:
    before = snapshot_named_parameters(generator)
    optimizer.zero_grad(set_to_none=True)
    timesteps = validate_fixed_grid_timesteps(timesteps)
    records: list[dict[str, Any]] = []
    objective_value = 0.0
    for index, timestep_value in enumerate(timesteps):
        point_loss, record = fixed_grid_point_loss_and_record(
            generator,
            conditional_dict=conditional_dict,
            clean_history=clean_history,
            current_target=current_target,
            next1_target=next1_target,
            epsilon_main=epsilon_main,
            epsilon_future=epsilon_future,
            scheduler=scheduler,
            timestep_value=timestep_value,
            grid_index=index,
        )
        records.append(record)
        objective_value += float(point_loss.detach().float().item()) / len(EXPECTED_FIXED_TIMESTEPS)
        (point_loss / len(EXPECTED_FIXED_TIMESTEPS)).backward()
    summary = aggregate_grid_metrics(records)
    grad_audit = parameter_group_audit(
        generator,
        optimizer=optimizer,
        before_snapshot=before,
        after_snapshot=None,
    )
    assert_gradients_finite(grad_audit)
    assert_fixed_grid_gradient_contract(grad_audit)
    optimizer.step()
    after = snapshot_named_parameters(generator)
    step_audit = parameter_group_audit(
        generator,
        optimizer=optimizer,
        before_snapshot=before,
        after_snapshot=after,
    )
    assert_only_target_parameters_changed(step_audit)
    result = {
        "training_objective_before_update": _finite_float(
            "training_objective_before_update",
            objective_value,
        ),
        "pre_update_records": records,
        "pre_update_summary": summary,
        "parameter_audit": step_audit,
        "gradient_audit": grad_audit,
    }
    if evaluate_post_update:
        with torch.no_grad():
            post_loss, post_records, post_summary = fixed_grid_loss_and_records(
                generator,
                conditional_dict=conditional_dict,
                clean_history=clean_history,
                current_target=current_target,
                next1_target=next1_target,
                epsilon_main=epsilon_main,
                epsilon_future=epsilon_future,
                scheduler=scheduler,
                timesteps=timesteps,
            )
        result.update(
            {
                "post_update_loss": float(post_loss.detach().float().item()),
                "post_update_records": post_records,
                "post_update_summary": post_summary,
            }
        )
    return result


def append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(payload), ensure_ascii=False) + "\n")


def state_dict_to_cpu(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in module.state_dict().items()
    }


def compare_state_dicts_exact(
    left: Mapping[str, torch.Tensor],
    right: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    if set(left.keys()) != set(right.keys()):
        raise RuntimeError("state_dict keys differ")
    per_tensor = {}
    exact = True
    for key in left:
        a = left[key]
        b = right[key]
        if tuple(a.shape) != tuple(b.shape) or a.dtype != b.dtype:
            raise RuntimeError(f"state_dict tensor contract differs for {key}")
        match = bool(torch.equal(a.detach().cpu(), b.detach().cpu()))
        exact = exact and match
        diff = (a.detach().float().cpu() - b.detach().float().cpu()).abs()
        per_tensor[key] = {
            "exact": match,
            "shape": [int(dim) for dim in a.shape],
            "dtype": str(a.dtype),
            "max_abs_diff": float(diff.max().item()) if diff.numel() else 0.0,
        }
    return {"exact": bool(exact), "tensors": per_tensor}


def make_probe_checkpoint_payload(
    *,
    generator: Any,
    base_m3_checkpoint_path: Path,
    base_m3_checkpoint_sha256: str,
    checkpoint_git_sha: str,
    current_git_sha: str,
    sample_metadata: Mapping[str, Any],
    probe_seed: int,
    timesteps: torch.Tensor,
    optimizer_step: int,
    resolved_cli: Mapping[str, Any],
    final_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "format": "nf_sf_m3_fixed_grid_probe_checkpoint_v1",
        "fusion": state_dict_to_cpu(generator.mcp.fusion),
        "mcp_depth1": state_dict_to_cpu(generator.mcp.mcp_modules[0]),
        "base_m3_checkpoint": {
            "path": str(base_m3_checkpoint_path.resolve()),
            "sha256": str(base_m3_checkpoint_sha256),
        },
        "checkpoint_git_sha": validate_git_sha(str(checkpoint_git_sha)),
        "current_git_sha": validate_git_sha(str(current_git_sha), name="current_git_sha"),
        "sample_metadata": dict(sample_metadata),
        "probe_seed": int(probe_seed),
        "four_timesteps": [float(value) for value in timesteps.detach().float().cpu().tolist()],
        "optimizer_step": int(optimizer_step),
        "resolved_cli": dict(resolved_cli),
        "final_metrics": dict(final_metrics),
    }


def save_probe_checkpoint(payload: Mapping[str, Any], path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    try:
        temporary.replace(path)
    except PermissionError:
        torch.save(dict(payload), path)
        try:
            temporary.unlink(missing_ok=True)
        except PermissionError:
            pass


def load_probe_checkpoint(path: Path | str) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise RuntimeError("fixed-grid probe checkpoint must contain a dict")
    if payload.get("format") != "nf_sf_m3_fixed_grid_probe_checkpoint_v1":
        raise RuntimeError("fixed-grid probe checkpoint format mismatch")
    return payload


def load_probe_modules_into_generator(generator: Any, payload: Mapping[str, Any]) -> dict[str, Any]:
    result_fusion = generator.mcp.fusion.load_state_dict(payload["fusion"], strict=True)
    result_mcp1 = generator.mcp.mcp_modules[0].load_state_dict(
        payload["mcp_depth1"],
        strict=True,
    )
    if result_fusion.missing_keys or result_fusion.unexpected_keys:
        raise RuntimeError(f"fusion restore mismatch: {result_fusion}")
    if result_mcp1.missing_keys or result_mcp1.unexpected_keys:
        raise RuntimeError(f"MCP-1 restore mismatch: {result_mcp1}")
    fusion_compare = compare_state_dicts_exact(
        payload["fusion"],
        state_dict_to_cpu(generator.mcp.fusion),
    )
    mcp1_compare = compare_state_dicts_exact(
        payload["mcp_depth1"],
        state_dict_to_cpu(generator.mcp.mcp_modules[0]),
    )
    exact = bool(fusion_compare["exact"] and mcp1_compare["exact"])
    return {
        "exact": exact,
        "fusion": fusion_compare,
        "mcp_depth1": mcp1_compare,
    }


def _reconstruct_mcp1_free_running_latent(
    generator: Any,
    *,
    conditional_dict: Mapping[str, Any],
    state: Any,
    epsilon_main: torch.Tensor,
    epsilon_future: torch.Tensor,
    teacher_payload: Mapping[str, Any],
) -> torch.Tensor:
    result = reconstruct_mcp1_next(
        generator,
        conditional_dict=conditional_dict,
        state=state,
        next_initial_noise=epsilon_future,
        current_condition_noise=epsilon_main,
        teacher_payload=teacher_payload,
        solver_steps_override=None,
        allow_solver_override=False,
    )
    return result.latent.detach()


def run_decode_reconstructions(
    generator: Any,
    *,
    probe_payload: Mapping[str, Any],
    conditional_dict: Mapping[str, Any],
    state: Any,
    epsilon_main: torch.Tensor,
    epsilon_future: torch.Tensor,
    teacher_payload: Mapping[str, Any],
    restore_fn: Callable[[Any, Mapping[str, Any]], Mapping[str, Any]] | None = None,
    reconstruct_fn: Callable[..., torch.Tensor] | None = None,
) -> dict[str, Any]:
    restore_fn = load_probe_modules_into_generator if restore_fn is None else restore_fn
    reconstruct_fn = _reconstruct_mcp1_free_running_latent if reconstruct_fn is None else reconstruct_fn
    base_latent = reconstruct_fn(
        generator,
        conditional_dict=conditional_dict,
        state=state,
        epsilon_main=epsilon_main,
        epsilon_future=epsilon_future,
        teacher_payload=teacher_payload,
    )
    restore_report = dict(restore_fn(generator, probe_payload))
    if not bool(restore_report.get("exact", False)):
        raise RuntimeError("fixed-grid probe parameter restore is not exact")
    fixed_grid_latent = reconstruct_fn(
        generator,
        conditional_dict=conditional_dict,
        state=state,
        epsilon_main=epsilon_main,
        epsilon_future=epsilon_future,
        teacher_payload=teacher_payload,
    )
    return {
        "base_latent": base_latent.detach(),
        "fixed_grid_latent": fixed_grid_latent.detach(),
        "parameter_restore": restore_report,
    }


def latent_mse_report(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    if tuple(prediction.shape) != tuple(target.shape):
        raise RuntimeError(
            "decode latent MSE shape mismatch: "
            f"{tuple(prediction.shape)} != {tuple(target.shape)}"
        )
    diff = prediction.detach().float() - target.detach().float()
    mse = _finite_float("latent_mse", float(diff.square().mean().item()))
    return {"mse": mse, "rmse": _finite_float("latent_rmse", mse**0.5)}


def validate_decode_latent(name: str, latent: torch.Tensor) -> None:
    if not bool(torch.isfinite(latent.detach().float()).all().item()):
        raise RuntimeError(f"decode latent {name} contains non-finite values")


def decode_m3_fixed_grid_video(
    *,
    vae: Any,
    full_target_latent: torch.Tensor,
    chunk: torch.Tensor,
    block_index: int,
    output_path: Path,
    device: torch.device,
    dtype: torch.dtype,
    fps: int,
) -> dict[str, Any]:
    from scripts.eval_nf_sf_m3_overfit import (
        block_pixel_span,
        normalize_pixels,
        save_video,
        splice_chunk,
    )

    validate_decode_latent(output_path.stem, chunk)
    full_latent = splice_chunk(
        full_target_latent,
        chunk,
        start_frame=block_index * 3,
    )
    with torch.no_grad():
        decoded = vae.decode_to_pixel(
            full_latent.to(device=device, dtype=dtype),
            use_cache=False,
        )
    frames = normalize_pixels(decoded)
    start, end = block_pixel_span(block_index, frames.shape[0])
    cropped = frames[start:end]
    if cropped.ndim != 4 or int(cropped.shape[0]) <= 0:
        raise RuntimeError(f"decode produced zero frames for {output_path.name}")
    save_video(output_path, cropped, fps=fps)
    if not output_path.is_file() or output_path.stat().st_size <= 0:
        raise RuntimeError(f"decode output file missing or empty: {output_path}")
    return {
        "path": output_path.name,
        "frames": int(cropped.shape[0]),
        "height": int(cropped.shape[1]),
        "width": int(cropped.shape[2]),
        "fps": int(fps),
        "block_index": int(block_index),
        "decoded_pixel_frames": int(frames.shape[0]),
        "saved_pixel_start": int(start),
        "saved_pixel_end": int(end),
    }


def build_decode_manifest(
    *,
    decode_output_dir: Path,
    base_checkpoint_path: Path,
    base_checkpoint_sha256: str,
    probe_checkpoint_path: Path,
    probe_checkpoint_sha256: str,
    checkpoint_git_sha: str,
    current_git_sha: str,
    sample_metadata: Mapping[str, Any],
    timesteps: torch.Tensor,
    parameter_restore_exact: bool,
    target_next1: torch.Tensor,
    base_latent: torch.Tensor,
    fixed_grid_latent: torch.Tensor,
    video_records: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    latents = {
        "target_next1": target_next1.detach().cpu(),
        "base_m3_mcp1_free_running": base_latent.detach().cpu(),
        "fixed_grid_mcp1_free_running": fixed_grid_latent.detach().cpu(),
    }
    for name, latent in latents.items():
        validate_decode_latent(name, latent)
    required_videos = {
        "target_next1": "target_next1.mp4",
        "base_m3_mcp1_free_running": "base_m3_mcp1_free_running.mp4",
        "fixed_grid_mcp1_free_running": "fixed_grid_mcp1_free_running.mp4",
    }
    videos = {}
    for name, filename in required_videos.items():
        record = dict(video_records.get(name, {}))
        if record.get("path") != filename:
            raise RuntimeError(f"decode video path mismatch for {name}")
        if int(record.get("frames", 0)) <= 0:
            raise RuntimeError(f"decode video has zero frames for {name}")
        if int(record.get("height", 0)) <= 0 or int(record.get("width", 0)) <= 0:
            raise RuntimeError(f"decode video has invalid resolution for {name}")
        videos[name] = record
    base_metrics = latent_mse_report(base_latent, target_next1)
    probe_metrics = latent_mse_report(fixed_grid_latent, target_next1)
    improvement = (base_metrics["mse"] - probe_metrics["mse"]) / (
        base_metrics["mse"] + COSINE_EPS
    )
    return {
        "status": "PASS",
        "base_m3_checkpoint": {
            "path": str(base_checkpoint_path.resolve()),
            "sha256": str(base_checkpoint_sha256),
        },
        "probe_checkpoint": {
            "path": str(probe_checkpoint_path.resolve()),
            "sha256": str(probe_checkpoint_sha256),
        },
        "checkpoint_git_sha": str(checkpoint_git_sha),
        "current_git_sha": str(current_git_sha),
        "sample_metadata": dict(sample_metadata),
        "four_timesteps": [float(value) for value in timesteps.detach().float().cpu().tolist()],
        "parameter_restore_exact": bool(parameter_restore_exact),
        "latents": {name: tensor_summary(latent) for name, latent in latents.items()},
        "videos": videos,
        "latent_metrics": {
            "base_vs_target_next1": base_metrics,
            "fixed_grid_vs_target_next1": probe_metrics,
            "probe_vs_base_mse_improvement_ratio": _finite_float(
                "probe_vs_base_mse_improvement_ratio",
                improvement,
            ),
        },
        "decode_output_dir": str(decode_output_dir.resolve()),
    }


def run_decode_outputs(
    *,
    decode_output_dir: Path,
    base_checkpoint_path: Path,
    base_checkpoint_sha256: str,
    probe_checkpoint_path: Path,
    probe_checkpoint_sha256: str,
    checkpoint_git_sha: str,
    current_git_sha: str,
    sample_metadata: Mapping[str, Any],
    full_target_latent: torch.Tensor,
    target_next1: torch.Tensor,
    base_latent: torch.Tensor,
    fixed_grid_latent: torch.Tensor,
    timesteps: torch.Tensor,
    parameter_restore_exact: bool,
    device: torch.device,
    dtype: torch.dtype,
    fps: int = 16,
) -> dict[str, Any]:
    from utils.wan_wrapper import WanVAEWrapper

    decode_output_dir.mkdir(parents=True, exist_ok=True)
    vae = WanVAEWrapper().eval().requires_grad_(False)
    vae.to(device=device, dtype=dtype)
    try:
        video_records = {
            "target_next1": decode_m3_fixed_grid_video(
                vae=vae,
                full_target_latent=full_target_latent,
                chunk=target_next1,
                block_index=2,
                output_path=decode_output_dir / "target_next1.mp4",
                device=device,
                dtype=dtype,
                fps=fps,
            ),
            "base_m3_mcp1_free_running": decode_m3_fixed_grid_video(
                vae=vae,
                full_target_latent=full_target_latent,
                chunk=base_latent,
                block_index=2,
                output_path=decode_output_dir / "base_m3_mcp1_free_running.mp4",
                device=device,
                dtype=dtype,
                fps=fps,
            ),
            "fixed_grid_mcp1_free_running": decode_m3_fixed_grid_video(
                vae=vae,
                full_target_latent=full_target_latent,
                chunk=fixed_grid_latent,
                block_index=2,
                output_path=decode_output_dir / "fixed_grid_mcp1_free_running.mp4",
                device=device,
                dtype=dtype,
                fps=fps,
            ),
        }
    finally:
        vae.to("cpu")
        del vae
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    manifest = build_decode_manifest(
        decode_output_dir=decode_output_dir,
        base_checkpoint_path=base_checkpoint_path,
        base_checkpoint_sha256=base_checkpoint_sha256,
        probe_checkpoint_path=probe_checkpoint_path,
        probe_checkpoint_sha256=probe_checkpoint_sha256,
        checkpoint_git_sha=checkpoint_git_sha,
        current_git_sha=current_git_sha,
        sample_metadata=sample_metadata,
        timesteps=timesteps,
        parameter_restore_exact=parameter_restore_exact,
        target_next1=target_next1,
        base_latent=base_latent,
        fixed_grid_latent=fixed_grid_latent,
        video_records=video_records,
    )
    atomic_json_write(manifest, decode_output_dir / "decode_manifest.json")
    return manifest


def validate_probe_checkpoint_matches(
    *,
    probe_payload: Mapping[str, Any],
    base_checkpoint_payload: Mapping[str, Any],
    base_checkpoint_path: Path,
    base_checkpoint_sha256: str,
    sample_metadata: Mapping[str, Any],
    timesteps: torch.Tensor,
    current_git_sha: str,
) -> None:
    base = probe_payload.get("base_m3_checkpoint", {})
    if str(Path(base.get("path", "")).resolve()) != str(base_checkpoint_path.resolve()):
        raise RuntimeError("probe checkpoint base M3 checkpoint path differs")
    if str(base.get("sha256")) != str(base_checkpoint_sha256):
        raise RuntimeError("probe checkpoint base M3 checkpoint SHA256 differs")
    if str(probe_payload.get("checkpoint_git_sha")) != str(base_checkpoint_payload["git_sha"]):
        raise RuntimeError("probe checkpoint base git_sha differs")
    if str(probe_payload.get("current_git_sha")) != str(current_git_sha):
        raise RuntimeError("probe checkpoint current git_sha differs")
    if int(probe_payload.get("probe_seed", -1)) != int(base_checkpoint_payload["probe_seed"]):
        raise RuntimeError("probe checkpoint probe_seed differs")
    saved_sample = probe_payload.get("sample_metadata", {})
    for key in ("sample_index", "sample_id", "split", "split_index", "prompt", "latent_file_sha256"):
        if saved_sample.get(key) != sample_metadata.get(key):
            raise RuntimeError(f"probe checkpoint sample metadata differs for {key}")
    saved_target = saved_sample.get("target_latent", {})
    current_target = sample_metadata.get("target_latent", {})
    if saved_target.get("sha256") != current_target.get("sha256"):
        raise RuntimeError("probe checkpoint target_latent SHA256 differs")
    actual_timesteps = [float(value) for value in timesteps.detach().float().cpu().tolist()]
    saved_timesteps = [float(value) for value in probe_payload.get("four_timesteps", [])]
    if len(saved_timesteps) != len(actual_timesteps):
        raise RuntimeError("probe checkpoint timestep count differs")
    max_abs = max(abs(a - b) for a, b in zip(saved_timesteps, actual_timesteps))
    if max_abs > TIMESTEP_TOLERANCE:
        raise RuntimeError("probe checkpoint timesteps differ from current fixed grid")


def resolved_cli(args: argparse.Namespace, *, mode: str) -> dict[str, Any]:
    return {
        "config": str(args.config.resolve()),
        "m3_checkpoint": str(args.m3_checkpoint.resolve()),
        "manifest": str(args.manifest.resolve()),
        "dataset_root": None if args.dataset_root is None else str(args.dataset_root.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "device": str(args.device),
        "dtype": str(args.dtype),
        "steps": int(args.steps),
        "lr": float(args.lr),
        "log_interval": int(args.log_interval),
        "mode": str(mode),
        "probe_checkpoint": None
        if args.probe_checkpoint is None
        else str(args.probe_checkpoint.resolve()),
        "decode": bool(getattr(args, "decode", False)),
        "decode_output_dir": None
        if getattr(args, "decode_output_dir", None) is None
        else str(args.decode_output_dir.resolve()),
    }


def run_train(args: argparse.Namespace) -> None:
    validate_cli_values(steps=args.steps, lr=args.lr, log_interval=args.log_interval)
    checkpoint_payload = load_m3_checkpoint(args.m3_checkpoint)
    validate_step100_checkpoint(checkpoint_payload)
    provenance = fixed_grid_provenance_gate(checkpoint_payload=checkpoint_payload)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json_write(provenance, args.output_dir / "provenance.json")

    from inference_mcp import merge_config, require_single_gpu_runtime
    from scripts.eval_nf_sf_m3_overfit import (
        compare_sample_metadata,
        conditional_dict_for_checkpoint,
        load_generator_from_m3_checkpoint,
        resolved_config_dict,
        validate_config,
    )

    device = require_single_gpu_runtime(torch, args.device)
    dtype = dtype_from_arg(args.dtype)
    config = merge_config(str(args.config))
    validate_config(config)
    current_model_config = resolved_config_dict(config)
    validate_m3_eval_config_matches_checkpoint(checkpoint_payload, current_model_config)
    metadata = checkpoint_payload["selected_sample_metadata"]
    sample = load_m3_teacher_sample(
        manifest_path=args.manifest,
        dataset_root=args.dataset_root,
        sample_index=int(metadata["sample_index"]),
        reference_checkpoint_path=checkpoint_payload["reference_checkpoint"]["path"],
    )
    compare_sample_metadata(metadata, sample.metadata)
    state = selected_state_to_device(sample.selected_state, device=device, dtype=dtype)
    if state.clean_history is None:
        raise RuntimeError("fixed-grid probe requires clean history")
    future_start = int(state.current_start_frame or 3) + 3
    if future_start != MCP1_FUTURE_START_FRAME:
        raise RuntimeError("fixed-grid probe future_start_frame changed")
    probe_tensors = move_tensors_to_device(
        checkpoint_payload["probe_tensors"],
        device=device,
        floating_dtype=dtype,
    )
    epsilon_main = probe_tensors["epsilon_main"]
    epsilon_future = probe_tensors["epsilon_depths"][0]
    conditional_dict = conditional_dict_for_checkpoint(
        payload=checkpoint_payload,
        prompt=sample.metadata["prompt"],
        device=device,
        dtype=dtype,
    )

    generator = load_generator_from_m3_checkpoint(
        config=config,
        checkpoint_payload=checkpoint_payload,
        device=device,
        dtype=dtype,
    )
    metrics_path = args.output_dir / "train_metrics.jsonl"
    metrics_path.unlink(missing_ok=True)
    parameter_history = []
    try:
        generator.eval()
        scheduler, timesteps, schedule_report = resolve_fixed_grid_schedule(
            generator,
            teacher_payload=sample.payload,
            device=device,
        )
        trainable_setup = configure_fixed_grid_trainable_parameters(generator)
        optimizer = make_fixed_grid_optimizer(generator, lr=args.lr)

        with torch.no_grad():
            step0_loss, step0_records, step0_summary = fixed_grid_loss_and_records(
                generator,
                conditional_dict=conditional_dict,
                clean_history=state.clean_history,
                current_target=state.current_target,
                next1_target=state.future_targets[0],
                epsilon_main=epsilon_main,
                epsilon_future=epsilon_future,
                scheduler=scheduler,
                timesteps=timesteps,
            )
        step0_audit = parameter_group_audit(
            generator,
            optimizer=optimizer,
            before_snapshot=None,
            after_snapshot=None,
        )
        append_jsonl(
            metrics_path,
            {
                "optimizer_step": 0,
                "loss": float(step0_loss.detach().float().item()),
                "timestep_records": step0_records,
                "summary": step0_summary,
                "parameter_audit": step0_audit,
            },
        )
        parameter_history.append({"optimizer_step": 0, "parameter_audit": step0_audit})
        final_metrics = {
            "loss": float(step0_loss.detach().float().item()),
            "timestep_records": step0_records,
            "summary": step0_summary,
        }

        for step in range(1, int(args.steps) + 1):
            should_log = step % int(args.log_interval) == 0 or step == int(args.steps)
            result = run_fixed_grid_optimizer_step(
                generator,
                optimizer=optimizer,
                conditional_dict=conditional_dict,
                clean_history=state.clean_history,
                current_target=state.current_target,
                next1_target=state.future_targets[0],
                epsilon_main=epsilon_main,
                epsilon_future=epsilon_future,
                scheduler=scheduler,
                timesteps=timesteps,
                evaluate_post_update=should_log,
            )
            parameter_history.append(
                {"optimizer_step": step, "parameter_audit": result["parameter_audit"]}
            )
            if should_log:
                final_metrics = {
                    "loss": result["post_update_loss"],
                    "timestep_records": result["post_update_records"],
                    "summary": result["post_update_summary"],
                }
                append_jsonl(
                    metrics_path,
                    {
                        "optimizer_step": step,
                        "loss": result["post_update_loss"],
                        "timestep_records": result["post_update_records"],
                        "summary": result["post_update_summary"],
                        "post_update_loss": result["post_update_loss"],
                        "post_update_records": result["post_update_records"],
                        "post_update_summary": result["post_update_summary"],
                        "training_objective_before_update": result[
                            "training_objective_before_update"
                        ],
                        "pre_update_records": result["pre_update_records"],
                        "pre_update_summary": result["pre_update_summary"],
                        "parameter_audit": result["parameter_audit"],
                        "gradient_audit": result["gradient_audit"],
                    },
                )

        base_sha = file_sha256(args.m3_checkpoint)
        probe_path = args.probe_checkpoint or (args.output_dir / "probe_checkpoint.pt")
        probe_payload = make_probe_checkpoint_payload(
            generator=generator,
            base_m3_checkpoint_path=args.m3_checkpoint,
            base_m3_checkpoint_sha256=base_sha,
            checkpoint_git_sha=checkpoint_payload["git_sha"],
            current_git_sha=provenance["current_git_sha"],
            sample_metadata=sample.metadata,
            probe_seed=int(checkpoint_payload["probe_seed"]),
            timesteps=timesteps,
            optimizer_step=int(args.steps),
            resolved_cli=resolved_cli(args, mode="train"),
            final_metrics=final_metrics,
        )
        save_probe_checkpoint(probe_payload, probe_path)
        audit_report = {
            "status": "PASS",
            "trainable_setup": trainable_setup,
            "optimizer": {
                "name": "AdamW",
                "lr": float(args.lr),
                "weight_decay": 0.0,
                "param_groups": [
                    {"name": group.get("name"), "tensor_count": len(group["params"])}
                    for group in optimizer.param_groups
                ],
            },
            "history": parameter_history,
        }
        atomic_json_write(audit_report, args.output_dir / "parameter_audit.json")
        probe_report = {
            "status": "PASS",
            "mode": "train",
            "base_m3_checkpoint": str(args.m3_checkpoint.resolve()),
            "probe_checkpoint": str(probe_path.resolve()),
            "sample": {
                "sample_index": int(sample.metadata["sample_index"]),
                "sample_id": sample.metadata.get("sample_id"),
                "split": sample.metadata["split"],
                "split_index": int(sample.metadata["split_index"]),
                "target_latent_sha256": sample.metadata["target_latent"]["sha256"],
                "probe_seed": int(checkpoint_payload["probe_seed"]),
            },
            "schedule": schedule_report,
            "final_metrics": final_metrics,
        }
        atomic_json_write(probe_report, args.output_dir / "fixed_grid_probe.json")
        print(json.dumps(probe_report, indent=2, ensure_ascii=False), flush=True)
    finally:
        generator.to("cpu")
        del generator
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_eval(args: argparse.Namespace) -> None:
    validate_cli_values(steps=args.steps, lr=args.lr, log_interval=args.log_interval)
    if bool(getattr(args, "decode", False)) and args.dtype != "bf16":
        raise RuntimeError("--decode requires --dtype bf16")
    checkpoint_payload = load_m3_checkpoint(args.m3_checkpoint)
    validate_step100_checkpoint(checkpoint_payload)
    provenance = fixed_grid_provenance_gate(checkpoint_payload=checkpoint_payload)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json_write(provenance, args.output_dir / "provenance.json")
    probe_path = args.probe_checkpoint or (args.output_dir / "probe_checkpoint.pt")
    probe_payload = load_probe_checkpoint(probe_path)
    base_checkpoint_sha256 = file_sha256(args.m3_checkpoint)
    probe_checkpoint_sha256 = file_sha256(probe_path)

    from inference_mcp import merge_config, require_single_gpu_runtime
    from scripts.eval_nf_sf_m3_overfit import (
        compare_sample_metadata,
        conditional_dict_for_checkpoint,
        load_generator_from_m3_checkpoint,
        resolved_config_dict,
        validate_config,
    )

    device = require_single_gpu_runtime(torch, args.device)
    dtype = dtype_from_arg(args.dtype)
    config = merge_config(str(args.config))
    validate_config(config)
    current_model_config = resolved_config_dict(config)
    validate_m3_eval_config_matches_checkpoint(checkpoint_payload, current_model_config)
    metadata = checkpoint_payload["selected_sample_metadata"]
    sample = load_m3_teacher_sample(
        manifest_path=args.manifest,
        dataset_root=args.dataset_root,
        sample_index=int(metadata["sample_index"]),
        reference_checkpoint_path=checkpoint_payload["reference_checkpoint"]["path"],
    )
    compare_sample_metadata(metadata, sample.metadata)
    state = selected_state_to_device(sample.selected_state, device=device, dtype=dtype)
    if state.clean_history is None:
        raise RuntimeError("fixed-grid eval requires clean history")
    probe_tensors = move_tensors_to_device(
        checkpoint_payload["probe_tensors"],
        device=device,
        floating_dtype=dtype,
    )
    epsilon_main = probe_tensors["epsilon_main"]
    epsilon_future = probe_tensors["epsilon_depths"][0]
    conditional_dict = conditional_dict_for_checkpoint(
        payload=checkpoint_payload,
        prompt=sample.metadata["prompt"],
        device=device,
        dtype=dtype,
    )

    generator = load_generator_from_m3_checkpoint(
        config=config,
        checkpoint_payload=checkpoint_payload,
        device=device,
        dtype=dtype,
    )
    try:
        scheduler, timesteps, schedule_report = resolve_fixed_grid_schedule(
            generator,
            teacher_payload=sample.payload,
            device=device,
        )
        validate_probe_checkpoint_matches(
            probe_payload=probe_payload,
            base_checkpoint_payload=checkpoint_payload,
            base_checkpoint_path=args.m3_checkpoint,
            base_checkpoint_sha256=base_checkpoint_sha256,
            sample_metadata=sample.metadata,
            timesteps=timesteps,
            current_git_sha=provenance["current_git_sha"],
        )
        decode_reconstructions = None
        if bool(getattr(args, "decode", False)):
            decode_reconstructions = run_decode_reconstructions(
                generator,
                probe_payload=probe_payload,
                conditional_dict=conditional_dict,
                state=state,
                epsilon_main=epsilon_main,
                epsilon_future=epsilon_future,
                teacher_payload=sample.payload,
            )
            restore_comparison = decode_reconstructions["parameter_restore"]
        else:
            restore_comparison = load_probe_modules_into_generator(generator, probe_payload)
        if not bool(restore_comparison["exact"]):
            raise RuntimeError("fixed-grid probe parameter restore is not exact")
        generator.eval().requires_grad_(False)
        with torch.no_grad():
            teacher_forced = run_mcp1_trajectory(
                generator,
                conditional_dict=conditional_dict,
                clean_history=state.clean_history,
                current_target=state.current_target,
                next1_target=state.future_targets[0],
                epsilon_main=epsilon_main,
                epsilon_future=epsilon_future,
                current_condition_scheduler=scheduler,
                future_scheduler=scheduler,
                main_timesteps=timesteps,
                mcp_timesteps=timesteps,
                mode="teacher_forced",
                future_start_frame=MCP1_FUTURE_START_FRAME,
            )
            free_running = run_mcp1_trajectory(
                generator,
                conditional_dict=conditional_dict,
                clean_history=state.clean_history,
                current_target=state.current_target,
                next1_target=state.future_targets[0],
                epsilon_main=epsilon_main,
                epsilon_future=epsilon_future,
                current_condition_scheduler=scheduler,
                future_scheduler=scheduler,
                main_timesteps=timesteps,
                mcp_timesteps=timesteps,
                mode="free_running",
                future_start_frame=MCP1_FUTURE_START_FRAME,
            )
        teacher_summary = summarize_trajectory(teacher_forced)
        free_summary = summarize_trajectory(free_running)
        report = {
            "status": "PASS",
            "mode": "eval",
            "parameter_restore_exact": bool(restore_comparison["exact"]),
            "parameter_restore_comparison": restore_comparison,
            "schedule": schedule_report,
            "teacher_forced": {
                "records": teacher_forced,
                "summary": teacher_summary,
            },
            "free_running": {
                "records": free_running,
                "summary": free_summary,
            },
            "all_finite": bool(
                teacher_summary["all_finite"] and free_summary["all_finite"]
            ),
        }
        atomic_json_write(report, args.output_dir / "restore_eval.json")
        if bool(getattr(args, "decode", False)):
            assert decode_reconstructions is not None
            decode_output_dir = args.decode_output_dir or (args.output_dir / "decoded")
            run_decode_outputs(
                decode_output_dir=decode_output_dir,
                base_checkpoint_path=args.m3_checkpoint,
                base_checkpoint_sha256=base_checkpoint_sha256,
                probe_checkpoint_path=probe_path,
                probe_checkpoint_sha256=probe_checkpoint_sha256,
                checkpoint_git_sha=checkpoint_payload["git_sha"],
                current_git_sha=provenance["current_git_sha"],
                sample_metadata=sample.metadata,
                full_target_latent=sample.target_latent,
                target_next1=sample.selected_state.future_targets[0],
                base_latent=decode_reconstructions["base_latent"].detach().cpu(),
                fixed_grid_latent=decode_reconstructions["fixed_grid_latent"].detach().cpu(),
                timesteps=timesteps,
                parameter_restore_exact=bool(restore_comparison["exact"]),
                device=device,
                dtype=dtype,
            )
        print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    finally:
        generator.to("cpu")
        del generator
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    if args.mode == "train":
        run_train(args)
    elif args.mode == "eval":
        run_eval(args)
    else:
        raise ValueError(f"unsupported mode: {args.mode}")


if __name__ == "__main__":
    main()
