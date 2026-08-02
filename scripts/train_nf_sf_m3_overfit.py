from __future__ import annotations

import argparse
import gc
import json
import math
import random
import statistics
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inference_mcp import merge_config, require_single_gpu_runtime
from utils.checkpoint import (
    extract_generator_state_dict,
    is_mcp_state_key,
    load_state_dict_allowing_mcp_mismatch,
)
from utils.nf_sf_m3 import (
    M3_CHUNK_FRAMES,
    M3_DEPTHS,
    M3_DEPTH_WEIGHTS,
    M3_TRAIN_MODES,
    atomic_json_write,
    file_sha256,
    gradient_group_audit,
    load_m3_teacher_sample,
    loss_dict_to_floats,
    make_m3_checkpoint_payload,
    make_m3_probe,
    optimizer_config_summary,
    optimizer_group_lr_summary,
    probe_output_summaries,
    prefix_metrics,
    run_m3_probe_forward,
    save_m3_checkpoint,
    selected_state_to_device,
    resolve_m3_solver_schedule,
    solver_schedule_to_json,
    validate_git_sha,
    validate_m3_mode,
)
from utils.nf_sf_m4 import (
    default_m4_checkpoint_steps,
    load_m4_sample_plan,
    load_m4_teacher_samples,
    m4_next_train_entry_after_global_step,
    m4_sample_plan_sha256,
    m4_train_entry_for_step,
    m4_validation_entry,
    parse_m4_step_list,
    run_m4_validation,
    validate_m4_sample_plan,
    write_m4_json,
    write_m4_sample_plan,
)
from utils.nf_sf_tensors import DEFAULT_S_MAIN, DEFAULT_S_MCP, make_generator
from utils.nf_sf_training import (
    collect_nf_sf_parameter_groups,
    configure_nf_sf_optimizer_plan,
    prepare_nf_sf_noisy_batch,
    run_nf_sf_mcp1_grid_point_loss,
    run_nf_sf_forward_loss,
)
from utils.scheduler import FlowMatchScheduler


TAP_LAYERS = (3, 11, 19, 29)
ADAMW_BETAS = (0.0, 0.999)
ADAMW_EPS = 1.0e-8
MCP1_GRID_EXPECTED_TIMESTEPS = (
    1000.0,
    937.5,
    833.3333129882812,
    625.0,
)
MCP1_GRID_TIMESTEP_TOLERANCE = 1.0e-4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="NF-SF v1 M3 single-sample Joint overfit."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--dataset_root", type=Path, default=None)
    parser.add_argument("--sample_index", type=int, default=None)
    parser.add_argument("--sample_id", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--split_index", type=int, default=None)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--mode", choices=M3_TRAIN_MODES, default="joint")
    parser.add_argument("--train_seed", type=int, required=True)
    parser.add_argument("--probe_seed", type=int, required=True)
    parser.add_argument("--optimizer_steps", type=int, required=True)
    parser.add_argument("--timing_warmup_steps", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--checkpoint_interval", type=int, default=10)
    parser.add_argument("--backbone_lr", type=float, required=True)
    parser.add_argument("--patch_embedding_lr", type=float, required=True)
    parser.add_argument("--mcp_lr", type=float, required=True)
    parser.add_argument("--weight_decay", type=float, required=True)
    parser.add_argument("--mcp1_grid_aux_weight", type=float, default=0.0)
    parser.add_argument("--m4_sample_plan", type=Path, default=None)
    parser.add_argument("--validation_seed", type=int, default=None)
    parser.add_argument("--validation_steps", default=None)
    parser.add_argument("--checkpoint_steps", default=None)
    parser.add_argument("--dtype", choices=("bf16", "float32"), default="bf16")
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def git_head() -> str:
    return validate_git_sha(
        subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip(),
        name="current_git_sha",
    )


def reset_global_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def dtype_from_arg(value: str) -> torch.dtype:
    return torch.bfloat16 if value == "bf16" else torch.float32


def validate_timing_warmup_steps(timing_warmup_steps: int, optimizer_steps: int) -> int:
    timing_warmup_steps = int(timing_warmup_steps)
    optimizer_steps = int(optimizer_steps)
    if timing_warmup_steps < 0:
        raise ValueError("--timing_warmup_steps must be non-negative")
    if timing_warmup_steps > optimizer_steps:
        raise ValueError("--timing_warmup_steps must be <= --optimizer_steps")
    return timing_warmup_steps


def summarize_step_timings(
    all_step_elapsed_ms: list[float],
    *,
    timing_warmup_steps: int,
) -> dict[str, Any]:
    validate_timing_warmup_steps(timing_warmup_steps, len(all_step_elapsed_ms))
    all_steps = [float(value) for value in all_step_elapsed_ms]
    measured = all_steps[int(timing_warmup_steps) :]
    summary = {
        "all_step_elapsed_ms": all_steps,
        "measured_step_elapsed_ms": measured,
        "measured_step_count": int(len(measured)),
    }
    if measured:
        summary.update(
            {
                "mean_step_elapsed_ms": float(statistics.fmean(measured)),
                "median_step_elapsed_ms": float(statistics.median(measured)),
                "min_step_elapsed_ms": float(min(measured)),
                "max_step_elapsed_ms": float(max(measured)),
            }
        )
    else:
        summary.update(
            {
                "mean_step_elapsed_ms": None,
                "median_step_elapsed_ms": None,
                "min_step_elapsed_ms": None,
                "max_step_elapsed_ms": None,
            }
        )
    return summary


def cuda_synchronize_if_available(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def reset_peak_memory_stats_if_available(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)


def cuda_peak_memory_summary(device: torch.device) -> dict[str, int | None]:
    if device.type != "cuda" or not torch.cuda.is_available():
        return {
            "max_memory_allocated_bytes": None,
            "max_memory_reserved_bytes": None,
        }
    return {
        "max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }


def resolved_config_dict(config: Any) -> dict[str, Any]:
    from omegaconf import OmegaConf

    return OmegaConf.to_container(config, resolve=True)


def validate_config(config: Any, args: argparse.Namespace) -> None:
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if args.optimizer_steps <= 0:
        raise ValueError("--optimizer_steps must be positive")
    validate_timing_warmup_steps(args.timing_warmup_steps, args.optimizer_steps)
    if args.log_interval <= 0:
        raise ValueError("--log_interval must be positive")
    if args.checkpoint_interval <= 0:
        raise ValueError("--checkpoint_interval must be positive")
    if args.optimizer_steps > 300:
        raise ValueError("M3 gate refuses runs longer than 300 optimizer steps")
    if args.backbone_lr <= 0:
        raise ValueError("--backbone_lr must be > 0")
    if args.patch_embedding_lr <= 0:
        raise ValueError("--patch_embedding_lr must be > 0")
    if args.mcp_lr <= 0:
        raise ValueError("--mcp_lr must be > 0")
    if args.weight_decay < 0:
        raise ValueError("--weight_decay must be >= 0")
    if args.mcp1_grid_aux_weight < 0:
        raise ValueError("--mcp1_grid_aux_weight must be >= 0")
    if m4_enabled(args):
        if not args.m4_sample_plan.is_file():
            raise FileNotFoundError(args.m4_sample_plan)
        if args.validation_seed is None:
            raise ValueError("--validation_seed is required with --m4_sample_plan")
        resolved_m4_step_sets(args)
    elif any(
        value is not None
        for value in (
            args.validation_seed,
            args.validation_steps,
            args.checkpoint_steps,
        )
    ):
        raise ValueError("M4 validation/checkpoint step arguments require --m4_sample_plan")
    if bool(getattr(config, "i2v", False)):
        raise ValueError("NF-SF M3 supports T2V only")
    if int(getattr(config, "num_frame_per_block", 0)) != M3_CHUNK_FRAMES:
        raise ValueError("NF-SF M3 requires chunk_frames=3")
    if int(getattr(config, "mcp_num_modules", 0)) != len(M3_DEPTHS):
        raise ValueError("NF-SF M3 requires mcp_num_modules=3")
    if int(getattr(config, "mcp_num_layers", 0)) != 3:
        raise ValueError("NF-SF M3 requires mcp_num_layers=3")
    if tuple(int(x) for x in getattr(config, "mcp_tap_layers", ())) != TAP_LAYERS:
        raise ValueError("NF-SF M3 requires mcp_tap_layers=[3, 11, 19, 29]")
    if tuple(float(x) for x in getattr(config, "mcp_depth_weights", ())) != M3_DEPTH_WEIGHTS:
        raise ValueError("NF-SF M3 requires depth weights [0.5, 0.2, 0.1]")
    model_kwargs = getattr(config, "model_kwargs", {})
    if float(model_kwargs.get("timestep_shift", DEFAULT_S_MAIN)) != DEFAULT_S_MAIN:
        raise ValueError("NF-SF M3 requires main timestep shift s_main=5.0")


def load_generator(config: Any, checkpoint_path: Path) -> tuple[Any, str, int]:
    from utils.wan_wrapper import WanDiffusionWrapper

    model_kwargs = dict(getattr(config, "model_kwargs", {}))
    generator = WanDiffusionWrapper(**model_kwargs, is_causal=True)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = extract_generator_state_dict(checkpoint)
    checkpoint_has_mcp = any(is_mcp_state_key(key) for key in state_dict.keys())

    if checkpoint_has_mcp:
        generator.add_mcp_modules(
            num_modules=len(M3_DEPTHS),
            num_layers=3,
            tap_layers=TAP_LAYERS,
        )
        load_state_dict_allowing_mcp_mismatch(generator, state_dict)
        load_mode = str(load_state_dict_allowing_mcp_mismatch.last_load_mode)
    else:
        missing, unexpected = generator.load_state_dict(state_dict, strict=True)
        if missing or unexpected:
            raise RuntimeError(
                f"Backbone checkpoint mismatch: missing={missing}, unexpected={unexpected}"
            )
        generator.add_mcp_modules(
            num_modules=len(M3_DEPTHS),
            num_layers=3,
            tap_layers=TAP_LAYERS,
        )
        load_mode = "BACKBONE_STRICT_THEN_INITIALIZE_MCP"

    mcp_tensor_count = sum(
        1
        for key, value in generator.state_dict().items()
        if is_mcp_state_key(key) and torch.is_tensor(value)
    )
    if mcp_tensor_count <= 0:
        raise RuntimeError("MCP modules were not attached")
    return generator, load_mode, mcp_tensor_count


def make_mcp_scheduler(device: torch.device) -> FlowMatchScheduler:
    scheduler = FlowMatchScheduler(
        shift=DEFAULT_S_MCP,
        sigma_min=0.0,
        extra_one_step=True,
    )
    scheduler.set_timesteps(1000, training=True)
    scheduler.sigmas = scheduler.sigmas.to(device)
    scheduler.timesteps = scheduler.timesteps.to(device)
    return scheduler


def make_mcp1_grid_aux_scheduler(device: torch.device) -> FlowMatchScheduler:
    scheduler = FlowMatchScheduler(
        shift=DEFAULT_S_MAIN,
        sigma_min=0.0,
        extra_one_step=True,
    )
    scheduler.sigmas = scheduler.sigmas.to(device)
    scheduler.timesteps = scheduler.timesteps.to(device)
    return scheduler


def validate_mcp1_grid_timesteps(timesteps: torch.Tensor) -> torch.Tensor:
    values = timesteps.detach().float().flatten()
    if values.numel() != len(MCP1_GRID_EXPECTED_TIMESTEPS):
        raise RuntimeError(
            "MCP-1 grid auxiliary schedule must contain exactly four timesteps"
        )
    expected = torch.tensor(MCP1_GRID_EXPECTED_TIMESTEPS, dtype=torch.float32)
    diff = (values.detach().cpu() - expected).abs()
    max_abs = float(diff.max().item())
    if max_abs > MCP1_GRID_TIMESTEP_TOLERANCE:
        raise RuntimeError(
            "MCP-1 grid auxiliary timesteps differ from expected inference grid: "
            f"max_abs_diff={max_abs}, tolerance={MCP1_GRID_TIMESTEP_TOLERANCE}"
        )
    return values


def resolve_mcp1_grid_aux_schedule(
    *,
    teacher_payload: dict[str, Any],
    device: torch.device,
) -> tuple[FlowMatchScheduler, torch.Tensor, dict[str, Any]]:
    scheduler = make_mcp1_grid_aux_scheduler(device)
    schedule = resolve_m3_solver_schedule(
        scheduler,
        teacher_payload=teacher_payload,
        device=device,
        solver_steps_override=None,
        allow_solver_override=False,
        tolerance=MCP1_GRID_TIMESTEP_TOLERANCE,
    )
    timesteps = validate_mcp1_grid_timesteps(schedule.timesteps)
    return scheduler, timesteps, solver_schedule_to_json(schedule)


def audit_to_json(audit) -> dict[str, Any]:
    return {
        "name": audit.name,
        "parameter_names": list(audit.parameter_names),
        "tensor_count": audit.tensor_count,
        "trainable_parameter_count": audit.trainable_parameter_count,
        "requires_grad": audit.requires_grad,
        "in_optimizer": audit.in_optimizer,
    }


def has_nonfinite_grad(generator) -> bool:
    for _, parameter in generator.named_parameters():
        if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all().item()):
            return True
    return False


def write_probe_report(
    output_dir: Path,
    step: int,
    losses: dict[str, float],
    outputs: dict[str, torch.Tensor],
    mcp1_grid_probe: dict[str, Any] | None = None,
    *,
    strict: bool = False,
) -> dict[str, Any]:
    report = {
        "step": int(step),
        **prefix_metrics("probe", losses),
        "probe_losses": losses,
        "probe_output_summaries": probe_output_summaries(outputs),
    }
    if mcp1_grid_probe is not None:
        report["mcp1_grid_probe"] = mcp1_grid_probe
        report["probe/mcp1_grid_probe_mean_loss"] = float(
            mcp1_grid_probe["mcp1_grid_probe_mean_loss"]
        )
    if strict:
        require_finite_json_numbers(report, path=f"probe_step{step:06d}")
        write_m4_json(report, output_dir / f"probe_step{step:06d}.json")
    else:
        atomic_json_write(report, output_dir / f"probe_step{step:06d}.json")
    return report


def require_finite_json_numbers(value: Any, *, path: str) -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise RuntimeError(f"M4 JSON numeric field is non-finite: {path}")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            require_finite_json_numbers(item, path=f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            require_finite_json_numbers(item, path=f"{path}[{index}]")
        return


def append_metrics(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def atomic_strict_json_write(payload: dict[str, Any], path: Path) -> None:
    write_m4_json(payload, path)


def append_strict_metrics(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")


def append_run_metrics(path: Path, record: dict[str, Any], *, strict: bool) -> None:
    if strict:
        append_strict_metrics(path, record)
    else:
        append_metrics(path, record)


def write_run_json(payload: dict[str, Any], path: Path, *, strict: bool) -> None:
    if strict:
        write_m4_json(payload, path)
    else:
        atomic_json_write(payload, path)


def named_parameter_groups(generator) -> dict[str, tuple[tuple[str, torch.nn.Parameter], ...]]:
    return collect_nf_sf_parameter_groups(generator)


def configure_m3_optimizer_plan(generator, *, mode: str, group_lrs: dict[str, float]):
    return configure_nf_sf_optimizer_plan(
        generator,
        mode=validate_m3_mode(mode),
        group_lrs=group_lrs,
    )


def m4_enabled(args: argparse.Namespace) -> bool:
    return args.m4_sample_plan is not None


def resolved_m4_step_sets(
    args: argparse.Namespace,
) -> tuple[tuple[int, ...] | None, tuple[int, ...] | None]:
    if not m4_enabled(args):
        return None, None
    validation_steps = parse_m4_step_list(
        args.validation_steps,
        optimizer_steps=args.optimizer_steps,
        name="--validation_steps",
        require_zero=True,
    )
    if validation_steps is None:
        raise ValueError("--validation_steps is required with --m4_sample_plan")
    checkpoint_steps = parse_m4_step_list(
        args.checkpoint_steps,
        optimizer_steps=args.optimizer_steps,
        name="--checkpoint_steps",
        require_zero=True,
        require_final=True,
    )
    if checkpoint_steps is None:
        checkpoint_steps = default_m4_checkpoint_steps(args.optimizer_steps)
    return validation_steps, checkpoint_steps


def conditional_dict_for_identity(
    *,
    text_encoder,
    sample,
    identity: str,
    cache: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if identity not in cache:
        with torch.no_grad():
            cache[identity] = text_encoder([sample.metadata["prompt"]])
    return cache[identity]


def write_m4_validation_report(
    *,
    output_dir: Path,
    metrics_path: Path,
    report: dict[str, Any],
) -> None:
    step = int(report["global_step"])
    atomic_strict_json_write(report, output_dir / f"validation_step{step:06d}.json")
    append_strict_metrics(
        metrics_path,
        {
            "event": "validation",
            "step": step,
            "mode": report["mode"],
            "sample_plan_sha256": report["sample_plan_sha256"],
            "validation_status": report["status"],
            "validation_sample_count": report["sample_count"],
            "validation_loss_finite_contract": report["validation_loss_finite_contract"],
            "nonfinite_validation_loss_count": report["nonfinite_validation_loss_count"],
            **{
                f"validation/{key}": value
                for key, value in report["aggregate_losses"].items()
            },
        },
    )


def require_m4_validation_pass(report: Any, *, global_step: int) -> None:
    if not isinstance(report, Mapping):
        raise RuntimeError(
            "M4 validation contract failed "
            f"at step {int(global_step)}: report is not a mapping"
        )
    status = report.get("status")
    if status == "PASS":
        return
    nonfinite = report.get("nonfinite_validation_losses", [])
    sample_identities = []
    failed_fields = []
    if isinstance(nonfinite, list):
        for item in nonfinite:
            if not isinstance(item, Mapping):
                continue
            identity = item.get("sample_identity")
            if identity is not None:
                sample_identities.append(str(identity))
            fields = item.get("fields")
            if isinstance(fields, list):
                failed_fields.extend(str(field) for field in fields)
    raise RuntimeError(
        "M4 validation contract failed "
        f"at step {int(global_step)}: status={status!r}; "
        f"sample_identities={sorted(set(sample_identities))}; "
        f"failed_fields={sorted(set(failed_fields))}; "
        f"nonfinite_diagnostics={nonfinite!r}"
    )


def handle_m4_validation_report(
    *,
    output_dir: Path,
    metrics_path: Path,
    validation_reports: list[dict[str, Any]],
    report: dict[str, Any],
    global_step: int,
) -> None:
    write_m4_validation_report(
        output_dir=output_dir,
        metrics_path=metrics_path,
        report=report,
    )
    require_m4_validation_pass(report, global_step=global_step)
    validation_reports.append(report)


def run_m4_validation_stage(
    *,
    global_step: int,
    run_validation,
    handle_report,
    after_pass=None,
) -> dict[str, Any]:
    report = run_validation()
    handle_report(report=report, global_step=global_step)
    if after_pass is not None:
        after_pass()
    return report


def gradient_audit_contract_pass(report: dict[str, Any]) -> bool:
    optimizer_contract = report.get("optimizer_contract")
    if isinstance(optimizer_contract, dict):
        return bool(optimizer_contract.get("all_contract_pass"))
    return all(
        isinstance(entry, dict)
        and bool(entry.get("finite"))
        and bool(entry.get("contract_pass"))
        for entry in report.values()
    )


def gradient_metadata_snapshot(
    groups: dict[str, tuple[tuple[str, torch.nn.Parameter], ...]],
) -> dict[str, dict[str, dict[str, Any]]]:
    snapshot = {}
    for group_name, named_params in groups.items():
        group_snapshot = {}
        for name, parameter in named_params:
            grad = parameter.grad
            if grad is None:
                group_snapshot[name] = {
                    "grad_is_none": True,
                    "grad_object_id": None,
                    "grad_data_ptr": None,
                    "grad_version": None,
                    "shape": None,
                    "dtype": None,
                    "device": None,
                }
            else:
                group_snapshot[name] = {
                    "grad_is_none": False,
                    "grad_object_id": int(id(grad)),
                    "grad_data_ptr": int(grad.data_ptr()),
                    "grad_version": int(grad._version),
                    "shape": [int(dim) for dim in grad.shape],
                    "dtype": str(grad.dtype),
                    "device": str(grad.device),
                }
        snapshot[group_name] = group_snapshot
    return snapshot


def validate_mcp1_grid_aux_gradient_isolation(
    before: dict[str, dict[str, dict[str, Any]]],
    after: dict[str, dict[str, dict[str, Any]]],
    groups: dict[str, tuple[tuple[str, torch.nn.Parameter], ...]],
) -> dict[str, Any]:
    target_groups = {"mcp_fusion", "mcp_depth1"}
    reports = {}
    for group_name, tensors in before.items():
        changed_count = 0
        object_changed_count = 0
        data_ptr_changed_count = 0
        version_changed_count = 0
        metadata_changed_count = 0
        version_increased_count = 0
        for name, before_grad in tensors.items():
            after_grad = after[group_name][name]
            identity_keys = (
                "grad_is_none",
                "grad_object_id",
                "grad_data_ptr",
                "shape",
                "dtype",
                "device",
            )
            identity_changed = any(
                before_grad[key] != after_grad[key] for key in identity_keys
            )
            object_changed = (
                before_grad["grad_object_id"] != after_grad["grad_object_id"]
            )
            data_ptr_changed = (
                before_grad["grad_data_ptr"] != after_grad["grad_data_ptr"]
            )
            version_changed = (
                before_grad["grad_version"] != after_grad["grad_version"]
            )
            if identity_changed or version_changed:
                changed_count += 1
            if object_changed:
                object_changed_count += 1
            if data_ptr_changed:
                data_ptr_changed_count += 1
            if version_changed:
                version_changed_count += 1
            if identity_changed:
                metadata_changed_count += 1
            if (
                before_grad["grad_version"] is not None
                and after_grad["grad_version"] is not None
                and after_grad["grad_version"] > before_grad["grad_version"]
            ):
                version_increased_count += 1
        reports[group_name] = {
            "tensor_count": int(len(tensors)),
            "changed_tensor_count": int(changed_count),
            "object_id_changed_count": int(object_changed_count),
            "data_ptr_changed_count": int(data_ptr_changed_count),
            "version_changed_count": int(version_changed_count),
            "metadata_changed_count": int(metadata_changed_count),
            "version_increased_count": int(version_increased_count),
            "aux_grad_changed": changed_count > 0,
        }
    for group_name, report in reports.items():
        if group_name not in target_groups and report["changed_tensor_count"] != 0:
            raise RuntimeError(
                f"MCP-1 grid auxiliary gradient leaked into {group_name}"
            )
    for group_name in target_groups:
        for name, before_grad in before[group_name].items():
            after_grad = after[group_name][name]
            if before_grad["grad_is_none"] or after_grad["grad_is_none"]:
                raise RuntimeError(
                    f"MCP-1 grid auxiliary missing random gradient for {group_name}"
                )
            if before_grad["grad_object_id"] != after_grad["grad_object_id"]:
                raise RuntimeError(
                    f"MCP-1 grid auxiliary replaced gradient object for {group_name}"
                )
            if before_grad["grad_data_ptr"] != after_grad["grad_data_ptr"]:
                raise RuntimeError(
                    f"MCP-1 grid auxiliary replaced gradient storage for {group_name}"
                )
            if before_grad["shape"] != after_grad["shape"]:
                raise RuntimeError(
                    f"MCP-1 grid auxiliary changed gradient shape for {group_name}"
                )
            if before_grad["dtype"] != after_grad["dtype"]:
                raise RuntimeError(
                    f"MCP-1 grid auxiliary changed gradient dtype for {group_name}"
                )
            if before_grad["device"] != after_grad["device"]:
                raise RuntimeError(
                    f"MCP-1 grid auxiliary changed gradient device for {group_name}"
                )
            if after_grad["grad_version"] <= before_grad["grad_version"]:
                raise RuntimeError(
                    f"MCP-1 grid auxiliary did not update gradients for {group_name}"
                )
        if reports[group_name]["version_increased_count"] <= 0:
            raise RuntimeError(
                f"MCP-1 grid auxiliary did not change gradients for {group_name}"
            )
        for _, parameter in groups[group_name]:
            grad = parameter.grad
            if grad is None:
                raise RuntimeError(
                    f"MCP-1 grid auxiliary missing final gradient for {group_name}"
                )
            if not bool(torch.isfinite(grad.detach().float()).all().item()):
                raise RuntimeError(
                    f"MCP-1 grid auxiliary final gradient is non-finite for {group_name}"
                )
    return reports


def mcp1_grid_aux_named_parameters(
    groups: dict[str, tuple[tuple[str, torch.nn.Parameter], ...]],
) -> list[tuple[str, torch.nn.Parameter]]:
    named_params = [
        (name, parameter)
        for group_name in ("mcp_fusion", "mcp_depth1")
        for name, parameter in groups[group_name]
    ]
    if not named_params:
        raise RuntimeError("MCP-1 grid auxiliary parameter set is empty")
    return named_params


def mcp1_grid_aux_parameters(
    groups: dict[str, tuple[tuple[str, torch.nn.Parameter], ...]],
) -> list[torch.nn.Parameter]:
    return [parameter for _, parameter in mcp1_grid_aux_named_parameters(groups)]


def set_mcp1_grid_aux_requires_grad(
    generator,
    groups: dict[str, tuple[tuple[str, torch.nn.Parameter], ...]],
) -> tuple[tuple[torch.nn.Parameter, bool], ...]:
    target_ids = {id(parameter) for parameter in mcp1_grid_aux_parameters(groups)}
    originals = []
    seen = set()
    for _, parameter in generator.named_parameters():
        if id(parameter) in seen:
            continue
        seen.add(id(parameter))
        originals.append((parameter, bool(parameter.requires_grad)))
        parameter.requires_grad_(id(parameter) in target_ids)
    validate_mcp1_grid_aux_requires_grad_scope(generator, target_ids)
    return tuple(originals)


def restore_requires_grad(
    originals: tuple[tuple[torch.nn.Parameter, bool], ...],
) -> None:
    for parameter, requires_grad in originals:
        parameter.requires_grad_(requires_grad)


def validate_mcp1_grid_aux_requires_grad_scope(generator, target_ids: set[int]) -> None:
    for name, parameter in generator.named_parameters():
        expected = id(parameter) in target_ids
        if bool(parameter.requires_grad) != expected:
            group = "auxiliary target" if expected else "non-auxiliary"
            raise RuntimeError(
                "MCP-1 grid auxiliary requires_grad scope mismatch for "
                f"{group} parameter {name}"
            )


def validate_initial_target_gradients(
    before: dict[str, dict[str, dict[str, Any]]],
) -> None:
    for group_name in ("mcp_fusion", "mcp_depth1"):
        for metadata in before[group_name].values():
            if metadata["grad_is_none"]:
                raise RuntimeError(
                    f"MCP-1 grid auxiliary missing random gradient for {group_name}"
                )


def validate_aux_grads(
    params: list[torch.nn.Parameter],
    grads: tuple[torch.Tensor, ...],
) -> None:
    if len(grads) != len(params):
        raise RuntimeError("MCP-1 grid auxiliary grad count mismatch")
    for parameter, grad in zip(params, grads):
        if grad is None:
            raise RuntimeError("MCP-1 grid auxiliary missing gradient")
        if tuple(grad.shape) != tuple(parameter.shape):
            raise RuntimeError("MCP-1 grid auxiliary gradient shape mismatch")
        if not bool(torch.isfinite(grad.detach().float()).all().item()):
            raise RuntimeError("MCP-1 grid auxiliary gradient is non-finite")


def accumulate_mcp1_grid_aux_gradients(
    generator,
    *,
    conditional_dict: dict[str, Any],
    state,
    scheduler,
    timesteps: torch.Tensor,
    epsilon_main: torch.Tensor,
    epsilon_future: torch.Tensor,
    weight: float,
) -> dict[str, Any]:
    if weight <= 0.0:
        return {
            "enabled": False,
            "mcp1_grid_aux_mean_loss": 0.0,
            "mcp1_grid_aux_weighted_loss": 0.0,
            "point_losses": [],
            "timesteps": [],
            "gradient_isolation": {},
        }
    timesteps = validate_mcp1_grid_timesteps(timesteps)
    groups = named_parameter_groups(generator)
    before = gradient_metadata_snapshot(groups)
    params = mcp1_grid_aux_parameters(groups)
    point_losses = []
    point_metadata = []
    requires_grad_originals = set_mcp1_grid_aux_requires_grad(generator, groups)
    try:
        validate_initial_target_gradients(before)
        for timestep in timesteps:
            validate_mcp1_grid_aux_requires_grad_scope(
                generator,
                {id(parameter) for parameter in params},
            )
            point = run_nf_sf_mcp1_grid_point_loss(
                generator,
                conditional_dict=conditional_dict,
                state=state,
                scheduler=scheduler,
                epsilon_main=epsilon_main,
                epsilon_future=epsilon_future,
                timestep=timestep,
                chunk_frames=M3_CHUNK_FRAMES,
            )
            scaled_loss = point.loss * (float(weight) / float(len(timesteps)))
            try:
                grads = torch.autograd.grad(
                    scaled_loss,
                    params,
                    allow_unused=False,
                    retain_graph=False,
                )
            except RuntimeError as exc:
                if "not have been used in the graph" in str(exc):
                    raise RuntimeError(
                        "MCP-1 grid auxiliary missing gradient for target groups "
                        "mcp_fusion/mcp_depth1"
                    ) from exc
                raise
            validate_aux_grads(params, grads)
            with torch.no_grad():
                for parameter, grad in zip(params, grads):
                    if parameter.grad is None:
                        raise RuntimeError(
                            "MCP-1 grid auxiliary expected existing "
                            "random-training gradient"
                        )
                    parameter.grad.add_(
                        grad.to(
                            device=parameter.grad.device,
                            dtype=parameter.grad.dtype,
                        )
                    )
            point_losses.append(float(point.loss.detach().float().item()))
            point_metadata.append(point.metadata)
            del grads, scaled_loss, point
    finally:
        restore_requires_grad(requires_grad_originals)
    after = gradient_metadata_snapshot(groups)
    isolation = validate_mcp1_grid_aux_gradient_isolation(
        before,
        after,
        groups,
    )
    mean_loss = sum(point_losses) / len(point_losses)
    return {
        "enabled": True,
        "mcp1_grid_aux_mean_loss": float(mean_loss),
        "mcp1_grid_aux_weighted_loss": float(float(weight) * mean_loss),
        "point_losses": point_losses,
        "point_metadata": point_metadata,
        "timesteps": [float(value.detach().float().item()) for value in timesteps],
        "gradient_isolation": isolation,
    }


def run_mcp1_grid_stable_probe(
    generator,
    *,
    conditional_dict: dict[str, Any],
    state,
    scheduler,
    timesteps: torch.Tensor,
    epsilon_main: torch.Tensor,
    epsilon_future: torch.Tensor,
) -> dict[str, Any]:
    timesteps = validate_mcp1_grid_timesteps(timesteps)
    was_training = bool(generator.training)
    generator.eval()
    try:
        point_losses = []
        records = []
        with torch.no_grad():
            for timestep in timesteps:
                point = run_nf_sf_mcp1_grid_point_loss(
                    generator,
                    conditional_dict=conditional_dict,
                    state=state,
                    scheduler=scheduler,
                    epsilon_main=epsilon_main,
                    epsilon_future=epsilon_future,
                    timestep=timestep,
                    chunk_frames=M3_CHUNK_FRAMES,
                )
                point_losses.append(float(point.loss.detach().float().item()))
                records.append(point.metadata)
    finally:
        generator.train(was_training)
    mean_loss = sum(point_losses) / len(point_losses)
    return {
        "mcp1_grid_probe_mean_loss": float(mean_loss),
        "point_losses": point_losses,
        "records": records,
        "timesteps": [float(value.detach().float().item()) for value in timesteps],
        "all_finite": bool(
            all(torch.isfinite(torch.tensor(point_losses, dtype=torch.float32)).tolist())
        ),
    }


def save_checkpoint_at_step(
    *,
    output_dir: Path,
    generator,
    optimizer: torch.optim.Optimizer,
    step: int,
    train_rng: torch.Generator,
    probe,
    probe_summary: dict[str, Any],
    probe_outputs: dict[str, torch.Tensor],
    sample_metadata: dict[str, Any],
    resolved_config: dict[str, Any],
    git_sha: str,
    reference_checkpoint_path: Path,
    reference_checkpoint_sha256: str,
    train_seed: int,
    probe_seed: int,
    prompt_embedding: dict[str, Any],
) -> Path:
    path = output_dir / f"checkpoint_step{step:06d}.pt"
    payload = make_m3_checkpoint_payload(
        generator=generator,
        optimizer=optimizer,
        global_step=step,
        train_rng=train_rng,
        probe=probe,
        probe_summary=probe_summary,
        probe_outputs=probe_outputs,
        selected_sample_metadata=sample_metadata,
        resolved_config=resolved_config,
        git_sha=git_sha,
        reference_checkpoint_path=reference_checkpoint_path,
        reference_checkpoint_sha256=reference_checkpoint_sha256,
        train_seed=train_seed,
        probe_seed=probe_seed,
        prompt_embedding=prompt_embedding,
    )
    save_m3_checkpoint(payload, path)
    return path


def main() -> None:
    args = parse_args()
    args.mode = validate_m3_mode(args.mode)
    dtype = dtype_from_arg(args.dtype)
    device = require_single_gpu_runtime(torch, args.device)
    reset_global_seed(args.train_seed)

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"--output_dir must be empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    reference_checkpoint_sha256 = file_sha256(args.checkpoint)
    m4_plan = None
    m4_plan_sha = None
    m4_validation_steps = None
    m4_checkpoint_steps = None
    train_samples_by_identity = None
    validation_samples_by_identity = None
    if m4_enabled(args):
        m4_plan = load_m4_sample_plan(args.m4_sample_plan, manifest_path=args.manifest)
        validate_m4_sample_plan(m4_plan)
        m4_plan_sha = m4_sample_plan_sha256(m4_plan)
        write_m4_sample_plan(m4_plan, args.output_dir / "m4_sample_plan.json")
        train_samples_by_identity = load_m4_teacher_samples(
            m4_plan,
            split="train",
            manifest_path=args.manifest,
            dataset_root=args.dataset_root,
            reference_checkpoint_path=args.checkpoint,
        )
        validation_samples_by_identity = load_m4_teacher_samples(
            m4_plan,
            split="validation",
            manifest_path=args.manifest,
            dataset_root=args.dataset_root,
            reference_checkpoint_path=args.checkpoint,
        )
        fixed_decode_entry = m4_validation_entry(
            m4_plan,
            str(m4_plan["fixed_decode_validation_identity"]),
        )
        sample = validation_samples_by_identity[str(fixed_decode_entry["identity"])]
    else:
        sample = load_m3_teacher_sample(
            manifest_path=args.manifest,
            dataset_root=args.dataset_root,
            sample_index=args.sample_index,
            sample_id=args.sample_id,
            split=args.split,
            split_index=args.split_index,
            reference_checkpoint_path=args.checkpoint,
        )
    write_run_json(
        sample.metadata,
        args.output_dir / "sample_metadata.json",
        strict=m4_enabled(args),
    )
    (args.output_dir / "reference_checkpoint_sha256.txt").write_text(
        reference_checkpoint_sha256 + "\n",
        encoding="utf-8",
    )

    config = merge_config(str(args.config))
    validate_config(config, args)
    if m4_enabled(args):
        m4_validation_steps, m4_checkpoint_steps = resolved_m4_step_sets(args)
    mcp1_grid_aux_enabled = float(args.mcp1_grid_aux_weight) > 0.0
    mcp1_grid_aux_scheduler = None
    mcp1_grid_aux_timesteps = None
    mcp1_grid_aux_schedule = None
    if mcp1_grid_aux_enabled:
        (
            mcp1_grid_aux_scheduler,
            mcp1_grid_aux_timesteps,
            mcp1_grid_aux_schedule,
        ) = resolve_mcp1_grid_aux_schedule(
            teacher_payload=sample.payload,
            device=device,
        )
    optimizer_config = {
        "optimizer": "AdamW",
        "betas": [float(value) for value in ADAMW_BETAS],
        "eps": ADAMW_EPS,
        "weight_decay": args.weight_decay,
    }
    resolved_config = {
        "model_config": resolved_config_dict(config),
        "m3": {
            "mode": args.mode,
            "manifest": str(args.manifest.resolve()),
            "dataset_root": None
            if args.dataset_root is None
            else str(args.dataset_root.resolve()),
            "sample_index": args.sample_index,
            "sample_id": args.sample_id,
            "split": args.split,
            "split_index": args.split_index,
            "train_seed": args.train_seed,
            "probe_seed": args.probe_seed,
            "optimizer_steps": args.optimizer_steps,
            "timing_warmup_steps": args.timing_warmup_steps,
            "log_interval": args.log_interval,
            "checkpoint_interval": args.checkpoint_interval,
            "backbone_lr": args.backbone_lr,
            "patch_embedding_lr": args.patch_embedding_lr,
            "mcp_lr": args.mcp_lr,
            "weight_decay": args.weight_decay,
            "mcp1_grid_aux_weight": float(args.mcp1_grid_aux_weight),
            "mcp1_grid_aux_enabled": bool(mcp1_grid_aux_enabled),
            "mcp1_grid_timesteps": []
            if mcp1_grid_aux_timesteps is None
            else [
                float(value)
                for value in mcp1_grid_aux_timesteps.detach().float().cpu().tolist()
            ],
            "mcp1_grid_schedule": mcp1_grid_aux_schedule,
            "optimizer_config": optimizer_config,
            "dtype": args.dtype,
            "device": str(device),
        },
    }
    if m4_enabled(args):
        assert m4_plan is not None
        assert m4_plan_sha is not None
        assert m4_validation_steps is not None
        assert m4_checkpoint_steps is not None
        resolved_config["m4"] = {
            "enabled": True,
            "sample_plan_path": str(Path(args.m4_sample_plan).resolve()),
            "sample_plan_sha256": m4_plan_sha,
            "train_sample_identities": list(m4_plan["train_sample_identities"]),
            "validation_sample_identities": list(m4_plan["validation_sample_identities"]),
            "train_subset_size": int(m4_plan["train_subset_size"]),
            "validation_subset_size": int(m4_plan["validation_subset_size"]),
            "validation_seed": int(args.validation_seed),
            "validation_steps": list(m4_validation_steps),
            "checkpoint_steps": list(m4_checkpoint_steps),
            "fixed_decode_validation_identity": str(
                m4_plan["fixed_decode_validation_identity"]
            ),
            "sample_ordering_rule": str(m4_plan["ordering_rule"]),
            "ordering_rule": str(m4_plan["ordering_rule"]),
        }
    write_run_json(
        resolved_config,
        args.output_dir / "resolved_config.json",
        strict=m4_enabled(args),
    )
    current_git_sha = git_head()
    (args.output_dir / "git_sha.txt").write_text(current_git_sha + "\n", encoding="utf-8")

    generator = None
    text_encoder = None
    train_rng = make_generator(args.train_seed, device)
    metrics_path = args.output_dir / "metrics.jsonl"
    try:
        generator, load_mode, mcp_tensor_count = load_generator(config, args.checkpoint)
        generator.to(device=device, dtype=dtype)
        generator.train()

        from utils.wan_wrapper import WanTextEncoder

        text_encoder = WanTextEncoder().to(device=device, dtype=dtype).eval().requires_grad_(False)
        conditional_cache: dict[str, dict[str, Any]] = {}
        if m4_enabled(args):
            assert m4_plan is not None
            fixed_decode_identity = str(m4_plan["fixed_decode_validation_identity"])
            conditional_dict = conditional_dict_for_identity(
                text_encoder=text_encoder,
                sample=sample,
                identity=fixed_decode_identity,
                cache=conditional_cache,
            )
        else:
            with torch.no_grad():
                conditional_dict = text_encoder([sample.metadata["prompt"]])

        state = selected_state_to_device(sample.selected_state, device=device, dtype=dtype)

        scheduler_main = generator.get_scheduler()
        scheduler_main.sigmas = scheduler_main.sigmas.to(device)
        scheduler_main.timesteps = scheduler_main.timesteps.to(device)
        scheduler_mcp = make_mcp_scheduler(device)
        if mcp1_grid_aux_enabled and mcp1_grid_aux_scheduler is scheduler_main:
            raise RuntimeError("MCP-1 grid auxiliary scheduler must be independent")

        probe = make_m3_probe(
            state,
            scheduler_main=scheduler_main,
            scheduler_mcp=scheduler_mcp,
            seed=args.probe_seed,
        )
        initial_probe = run_m3_probe_forward(
            generator,
            conditional_dict=conditional_dict,
            noisy_batch=probe.noisy_batch,
        )
        initial_grid_probe = None
        if mcp1_grid_aux_enabled:
            initial_grid_probe = run_mcp1_grid_stable_probe(
                generator,
                conditional_dict=conditional_dict,
                state=state,
                scheduler=mcp1_grid_aux_scheduler,
                timesteps=mcp1_grid_aux_timesteps,
                epsilon_main=probe.noisy_batch.epsilon_main,
                epsilon_future=probe.noisy_batch.epsilon_depths[0],
            )
        initial_probe_report = write_probe_report(
            args.output_dir,
            0,
            initial_probe.losses,
            initial_probe.outputs,
            initial_grid_probe,
            strict=m4_enabled(args),
        )

        group_lrs = {
            "backbone": args.backbone_lr,
            "patch_embedding": args.patch_embedding_lr,
            "mcp": args.mcp_lr,
        }
        plan = configure_m3_optimizer_plan(
            generator,
            mode=args.mode,
            group_lrs=group_lrs,
        )
        optimizer = torch.optim.AdamW(
            plan.optimizer_param_groups,
            betas=ADAMW_BETAS,
            eps=ADAMW_EPS,
            weight_decay=args.weight_decay,
        )
        write_run_json(
            {
                "mode": plan.mode,
                "optimizer_config": optimizer_config_summary(optimizer),
                "param_audit": [audit_to_json(audit) for audit in plan.audits],
                "optimizer_group_lrs": optimizer_group_lr_summary(optimizer),
                "checkpoint_load_mode": load_mode,
                "mcp_tensor_count": mcp_tensor_count,
            },
            args.output_dir / "optimizer_audit.json",
            strict=m4_enabled(args),
        )
        save_checkpoint_at_step(
            output_dir=args.output_dir,
            generator=generator,
            optimizer=optimizer,
            step=0,
            train_rng=train_rng,
            probe=probe,
            probe_summary=initial_probe_report,
            probe_outputs=initial_probe.outputs,
            sample_metadata=sample.metadata,
            resolved_config=resolved_config,
            git_sha=current_git_sha,
            reference_checkpoint_path=args.checkpoint,
            reference_checkpoint_sha256=reference_checkpoint_sha256,
            train_seed=args.train_seed,
            probe_seed=args.probe_seed,
            prompt_embedding=conditional_dict,
        )
        append_run_metrics(
            metrics_path,
            {
                "step": 0,
                "elapsed_ms": 0.0,
                **prefix_metrics("probe", initial_probe.losses),
                **(
                    {}
                    if initial_grid_probe is None
                    else {
                        "probe/mcp1_grid_probe_mean_loss": initial_grid_probe[
                            "mcp1_grid_probe_mean_loss"
                        ],
                        "probe/mcp1_grid_probe_point_losses": initial_grid_probe[
                            "point_losses"
                        ],
                        "probe/mcp1_grid_probe_all_finite": initial_grid_probe[
                            "all_finite"
                        ],
                    }
                ),
            },
            strict=m4_enabled(args),
        )
        validation_reports: list[dict[str, Any]] = []
        if m4_enabled(args):
            assert m4_plan is not None
            assert validation_samples_by_identity is not None
            assert m4_validation_steps is not None
            if 0 in m4_validation_steps:
                validation_samples = [
                    validation_samples_by_identity[str(identity)]
                    for identity in m4_plan["validation_sample_identities"]
                ]
                validation_conditionals = {
                    str(identity): conditional_dict_for_identity(
                        text_encoder=text_encoder,
                        sample=validation_samples_by_identity[str(identity)],
                        identity=str(identity),
                        cache=conditional_cache,
                    )
                    for identity in m4_plan["validation_sample_identities"]
                }
                def run_step0_validation() -> dict[str, Any]:
                    return run_m4_validation(
                        generator=generator,
                        samples=validation_samples,
                        conditional_dicts=validation_conditionals,
                        scheduler_main=scheduler_main,
                        scheduler_mcp=scheduler_mcp,
                        device=device,
                        dtype=dtype,
                        mode=args.mode,
                        global_step=0,
                        sample_plan=m4_plan,
                        validation_seed=int(args.validation_seed),
                        train_rng=train_rng,
                        probe_rng_state=probe.rng_state,
                        model_identity={
                            "output_dir": str(args.output_dir.resolve()),
                            "git_sha": current_git_sha,
                            "reference_checkpoint_sha256": reference_checkpoint_sha256,
                        },
                    )

                run_m4_validation_stage(
                    global_step=0,
                    run_validation=run_step0_validation,
                    handle_report=lambda *, report, global_step: handle_m4_validation_report(
                        output_dir=args.output_dir,
                        metrics_path=metrics_path,
                        validation_reports=validation_reports,
                        report=report,
                        global_step=global_step,
                    ),
                )

        all_step_elapsed_ms: list[float] = []
        reset_peak_memory_stats_if_available(device)
        for step in range(1, args.optimizer_steps + 1):
            cuda_synchronize_if_available(device)
            started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            train_sample_identity = None
            train_sample_position = None
            train_cycle_index = None
            if m4_enabled(args):
                assert m4_plan is not None
                assert train_samples_by_identity is not None
                train_entry = m4_train_entry_for_step(m4_plan, step)
                train_sample_identity = str(train_entry["identity"])
                train_sample_position = int(train_entry["train_sample_position"])
                train_cycle_index = int(train_entry["train_cycle_index"])
                train_sample = train_samples_by_identity[train_sample_identity]
                step_state = selected_state_to_device(
                    train_sample.selected_state,
                    device=device,
                    dtype=dtype,
                )
                step_conditional_dict = conditional_dict_for_identity(
                    text_encoder=text_encoder,
                    sample=train_sample,
                    identity=train_sample_identity,
                    cache=conditional_cache,
                )
            else:
                step_state = state
                step_conditional_dict = conditional_dict
            noisy_batch = prepare_nf_sf_noisy_batch(
                step_state,
                scheduler_main=scheduler_main,
                scheduler_mcp=scheduler_mcp,
                rng=train_rng,
                chunk_frames=M3_CHUNK_FRAMES,
                depths=M3_DEPTHS,
                s_main=DEFAULT_S_MAIN,
                s_mcp=DEFAULT_S_MCP,
            )
            result = run_nf_sf_forward_loss(
                generator,
                conditional_dict=step_conditional_dict,
                noisy_batch=noisy_batch,
                depth_weights=M3_DEPTH_WEIGHTS,
            )
            result.losses.total_loss.backward()
            parameter_groups = named_parameter_groups(generator)
            random_grad_audit = gradient_group_audit(
                optimizer,
                mode=args.mode,
                parameter_groups=parameter_groups,
            )
            if not gradient_audit_contract_pass(random_grad_audit):
                raise RuntimeError(f"random gradient audit contract failed at step {step}")
            aux_report = accumulate_mcp1_grid_aux_gradients(
                generator,
                conditional_dict=step_conditional_dict,
                state=step_state,
                scheduler=mcp1_grid_aux_scheduler,
                timesteps=mcp1_grid_aux_timesteps,
                epsilon_main=noisy_batch.epsilon_main,
                epsilon_future=noisy_batch.epsilon_depths[0],
                weight=float(args.mcp1_grid_aux_weight),
            )
            grad_audit = gradient_group_audit(
                optimizer,
                mode=args.mode,
                parameter_groups=parameter_groups,
            )
            if not gradient_audit_contract_pass(grad_audit):
                raise RuntimeError(f"gradient audit contract failed at step {step}")
            if has_nonfinite_grad(generator):
                raise RuntimeError(f"non-finite gradient at step {step}")
            optimizer.step()
            cuda_synchronize_if_available(device)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            all_step_elapsed_ms.append(elapsed_ms)
            train_losses = loss_dict_to_floats(result.losses)
            combined_objective = (
                train_losses["total_loss"]
                + aux_report["mcp1_grid_aux_weighted_loss"]
            )

            should_log = step % args.log_interval == 0 or step == args.optimizer_steps
            if m4_enabled(args):
                assert m4_checkpoint_steps is not None
                should_checkpoint = step in m4_checkpoint_steps
            else:
                should_checkpoint = (
                    step % args.checkpoint_interval == 0
                    or step == args.optimizer_steps
                )
            metric_record = {
                "step": step,
                "elapsed_ms": elapsed_ms,
                **prefix_metrics("train", train_losses),
                "train/random_total_loss": train_losses["total_loss"],
                "train/mcp1_grid_aux_mean_loss": aux_report[
                    "mcp1_grid_aux_mean_loss"
                ],
                "train/mcp1_grid_aux_weighted_loss": aux_report[
                    "mcp1_grid_aux_weighted_loss"
                ],
                "train/combined_objective": combined_objective,
                "mcp1_grid_aux": aux_report,
                "random_grad_audit": random_grad_audit,
                "grad_audit": grad_audit,
            }
            if m4_enabled(args):
                metric_record.update(
                    {
                        "train_sample_identity": train_sample_identity,
                        "train_sample_position": train_sample_position,
                        "train_cycle_index": train_cycle_index,
                    }
                )
            def write_step_artifacts_after_validation() -> None:
                if should_log or should_checkpoint:
                    grid_probe = None
                    if mcp1_grid_aux_enabled:
                        grid_probe = run_mcp1_grid_stable_probe(
                            generator,
                            conditional_dict=conditional_dict,
                            state=state,
                            scheduler=mcp1_grid_aux_scheduler,
                            timesteps=mcp1_grid_aux_timesteps,
                            epsilon_main=probe.noisy_batch.epsilon_main,
                            epsilon_future=probe.noisy_batch.epsilon_depths[0],
                        )
                    probe_forward = run_m3_probe_forward(
                        generator,
                        conditional_dict=conditional_dict,
                        noisy_batch=probe.noisy_batch,
                    )
                    probe_report = write_probe_report(
                        args.output_dir,
                        step,
                        probe_forward.losses,
                        probe_forward.outputs,
                        grid_probe,
                        strict=m4_enabled(args),
                    )
                    metric_record.update(prefix_metrics("probe", probe_forward.losses))
                    if grid_probe is not None:
                        metric_record.update(
                            {
                                "probe/mcp1_grid_probe_mean_loss": grid_probe[
                                    "mcp1_grid_probe_mean_loss"
                                ],
                                "probe/mcp1_grid_probe_point_losses": grid_probe[
                                    "point_losses"
                                ],
                                "probe/mcp1_grid_probe_all_finite": grid_probe[
                                    "all_finite"
                                ],
                            }
                        )
                    if should_checkpoint:
                        save_checkpoint_at_step(
                            output_dir=args.output_dir,
                            generator=generator,
                            optimizer=optimizer,
                            step=step,
                            train_rng=train_rng,
                            probe=probe,
                            probe_summary=probe_report,
                            probe_outputs=probe_forward.outputs,
                            sample_metadata=sample.metadata,
                            resolved_config=resolved_config,
                            git_sha=current_git_sha,
                            reference_checkpoint_path=args.checkpoint,
                            reference_checkpoint_sha256=reference_checkpoint_sha256,
                            train_seed=args.train_seed,
                            probe_seed=args.probe_seed,
                            prompt_embedding=conditional_dict,
                        )
                    append_run_metrics(
                        metrics_path,
                        metric_record,
                        strict=m4_enabled(args),
                    )
                    print(
                        json.dumps(
                            {
                                "step": step,
                                **prefix_metrics("train", train_losses),
                                "train/mcp1_grid_aux_mean_loss": aux_report[
                                    "mcp1_grid_aux_mean_loss"
                                ],
                                "train/mcp1_grid_aux_weighted_loss": aux_report[
                                    "mcp1_grid_aux_weighted_loss"
                                ],
                                "train/combined_objective": combined_objective,
                                **prefix_metrics("probe", probe_forward.losses),
                                "grad_audit": grad_audit,
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
                else:
                    append_run_metrics(
                        metrics_path,
                        metric_record,
                        strict=m4_enabled(args),
                    )

            if m4_enabled(args):
                assert m4_plan is not None
                assert validation_samples_by_identity is not None
                assert m4_validation_steps is not None
                if step in m4_validation_steps:
                    validation_samples = [
                        validation_samples_by_identity[str(identity)]
                        for identity in m4_plan["validation_sample_identities"]
                    ]
                    validation_conditionals = {
                        str(identity): conditional_dict_for_identity(
                            text_encoder=text_encoder,
                            sample=validation_samples_by_identity[str(identity)],
                            identity=str(identity),
                            cache=conditional_cache,
                        )
                        for identity in m4_plan["validation_sample_identities"]
                    }

                    def run_step_validation() -> dict[str, Any]:
                        return run_m4_validation(
                            generator=generator,
                            samples=validation_samples,
                            conditional_dicts=validation_conditionals,
                            scheduler_main=scheduler_main,
                            scheduler_mcp=scheduler_mcp,
                            device=device,
                            dtype=dtype,
                            mode=args.mode,
                            global_step=step,
                            sample_plan=m4_plan,
                            validation_seed=int(args.validation_seed),
                            train_rng=train_rng,
                            probe_rng_state=probe.rng_state,
                            model_identity={
                                "output_dir": str(args.output_dir.resolve()),
                                "git_sha": current_git_sha,
                                "reference_checkpoint_sha256": reference_checkpoint_sha256,
                            },
                        )

                    run_m4_validation_stage(
                        global_step=step,
                        run_validation=run_step_validation,
                        handle_report=lambda *, report, global_step: handle_m4_validation_report(
                            output_dir=args.output_dir,
                            metrics_path=metrics_path,
                            validation_reports=validation_reports,
                            report=report,
                            global_step=global_step,
                        ),
                        after_pass=write_step_artifacts_after_validation,
                    )
                else:
                    write_step_artifacts_after_validation()
            else:
                write_step_artifacts_after_validation()

        summary = {
            "status": "PASS",
            "output_dir": str(args.output_dir.resolve()),
            "mode": args.mode,
            "optimizer_steps": int(args.optimizer_steps),
            "timing_warmup_steps": int(args.timing_warmup_steps),
            **summarize_step_timings(
                all_step_elapsed_ms,
                timing_warmup_steps=args.timing_warmup_steps,
            ),
            **cuda_peak_memory_summary(device),
        }
        if m4_enabled(args):
            assert m4_plan is not None
            assert m4_plan_sha is not None
            assert m4_validation_steps is not None
            assert m4_checkpoint_steps is not None
            summary["m4"] = {
                "enabled": True,
                "sample_plan_sha256": m4_plan_sha,
                "train_sample_identities": list(m4_plan["train_sample_identities"]),
                "validation_sample_identities": list(
                    m4_plan["validation_sample_identities"]
                ),
                "validation_steps": list(m4_validation_steps),
                "checkpoint_steps": list(m4_checkpoint_steps),
                "validation_reports": [
                    f"validation_step{int(report['global_step']):06d}.json"
                    for report in validation_reports
                ],
                "fixed_decode_validation_identity": str(
                    m4_plan["fixed_decode_validation_identity"]
                ),
                "next_train_sample_after_final_step": m4_next_train_entry_after_global_step(
                    m4_plan,
                    args.optimizer_steps,
                ),
            }
        write_run_json(
            summary,
            args.output_dir / "training_summary.json",
            strict=m4_enabled(args),
        )
        print(
            json.dumps(summary, indent=2),
            flush=True,
        )
    finally:
        try:
            if generator is not None:
                generator.to("cpu")
            if text_encoder is not None:
                text_encoder.to("cpu")
        finally:
            del generator
            del text_encoder
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
