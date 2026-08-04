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
from collections.abc import Mapping, Sequence
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
    M3_DEPTH_WEIGHTS,
    M3_DEPTHS,
    M3_TRAIN_MODES,
    M3Probe,
    atomic_json_write,
    compare_probe_outputs,
    compare_serialized_probe_tensors,
    deserialize_noisy_batch,
    file_sha256,
    gradient_group_audit,
    load_m3_checkpoint,
    load_m3_teacher_sample,
    loss_dict_to_floats,
    make_m3_checkpoint_payload,
    make_m3_probe,
    move_tensors_to_device,
    optimizer_config_summary,
    optimizer_group_lr_summary,
    prefix_metrics,
    probe_output_summaries,
    resolve_m3_solver_schedule,
    run_m3_probe_forward,
    save_m3_checkpoint,
    selected_state_to_device,
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
from utils.nf_sf_m5 import (
    M5_RNG_EXTENSION_FIELD,
    ResumeContractError,
    build_resume_contract,
    build_resume_run_fields,
    capture_m5_global_rng_extension,
    extract_resume_rng_states,
    first_resumed_global_step,
    move_loaded_optimizer_state_to_device,
    restore_global_rng_states,
    restore_torch_generator_from_state,
    validate_resume_contract,
)
from utils.nf_sf_m5_conditionals import (
    M5_CONDITIONAL_ARTIFACT_MANIFEST_NAME,
    M5ConditionalArtifactStore,
    load_m5_conditional_artifact_manifest,
    validate_m5_conditional_artifact_manifest,
)
from utils.nf_sf_m5_formal import (
    FormalStageContract,
    resolve_m5_formal_stage_contract,
    validate_m5_formal_stage_request,
)
from utils.nf_sf_m5_formal_plan import (
    M5_FORMAL_TRAIN_SAMPLE_COUNT,
    M5_FORMAL_VALIDATION_SAMPLE_COUNT,
    validate_m5_formal_sample_plan,
)
from utils.nf_sf_m5_samples import M5TeacherSampleStore
from utils.nf_sf_m5_validation import (
    M5_STREAMING_VALIDATION_SCHEMA,
    run_m5_streaming_validation,
)
from utils.nf_sf_tensors import DEFAULT_S_MAIN, DEFAULT_S_MCP, make_generator
from utils.nf_sf_training import (
    collect_nf_sf_parameter_groups,
    configure_nf_sf_optimizer_plan,
    prepare_nf_sf_noisy_batch,
    run_nf_sf_forward_loss,
    run_nf_sf_mcp1_grid_point_loss,
)
from utils.scheduler import FlowMatchScheduler

TAP_LAYERS = (3, 11, 19, 29)
ADAMW_BETAS = (0.0, 0.999)
ADAMW_EPS = 1.0e-8
M5_FORMAL_TRAINER_SCHEMA = "nf_sf_m5_formal_trainer_v1"
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
    parser.add_argument("--resume_checkpoint", type=Path, default=None)
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
    parser.add_argument("--m5_formal_long_train", action="store_true")
    parser.add_argument("--m5_conditionals_artifact", type=Path, default=None)
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
        "measured_step_count": len(measured),
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


def m5_absolute_training_steps(
    *,
    resumed_global_step: int | None,
    target_global_step: int,
) -> range:
    target = int(target_global_step)
    start = 1 if resumed_global_step is None else first_resumed_global_step(resumed_global_step)
    if target < start:
        raise ValueError(
            "target_global_step must be >= first training step: "
            f"target_global_step={target}, first_training_step={start}"
        )
    return range(start, target + 1)


def _step_in_schedule(global_step: int, steps: Sequence[int] | None) -> bool:
    if steps is None:
        return False
    return int(global_step) in {int(step) for step in steps}


def m5_training_step_orchestration(
    *,
    global_step: int,
    target_global_step: int,
    sample_plan: Mapping[str, Any] | None,
    validation_steps: Sequence[int] | None,
    checkpoint_steps: Sequence[int] | None,
    checkpoint_interval: int,
    log_interval: int,
) -> dict[str, Any]:
    step = int(global_step)
    target = int(target_global_step)
    if step <= 0:
        raise ValueError(f"global_step must be positive: {step}")
    if target < step:
        raise ValueError(
            "target_global_step must be >= global_step: "
            f"target_global_step={target}, global_step={step}"
        )
    if int(log_interval) <= 0:
        raise ValueError("log_interval must be positive")
    if int(checkpoint_interval) <= 0:
        raise ValueError("checkpoint_interval must be positive")

    record: dict[str, Any] = {
        "global_step": step,
        "step": step,
        "should_log": step % int(log_interval) == 0 or step == target,
        "should_validate": False,
        "should_checkpoint": False,
        "train_sample_identity": None,
        "train_sample_position": None,
        "train_cycle_index": None,
    }
    if sample_plan is None:
        record["should_checkpoint"] = step % int(checkpoint_interval) == 0 or step == target
        return record

    train_entry = m4_train_entry_for_step(sample_plan, step)
    record.update(
        {
            "should_validate": _step_in_schedule(step, validation_steps),
            "should_checkpoint": _step_in_schedule(step, checkpoint_steps),
            "train_sample_identity": str(train_entry["identity"]),
            "train_sample_position": int(train_entry["train_sample_position"]),
            "train_cycle_index": int(train_entry["train_cycle_index"]),
        }
    )
    return record


def m5_timing_record(
    *,
    global_step: int,
    elapsed_ms: float,
    timing_warmup_steps: int,
) -> dict[str, Any]:
    step = int(global_step)
    warmup = int(timing_warmup_steps)
    if step <= 0:
        raise ValueError(f"global_step must be positive: {step}")
    if warmup < 0:
        raise ValueError("timing_warmup_steps must be non-negative")
    elapsed = float(elapsed_ms)
    if not math.isfinite(elapsed):
        raise ValueError(f"elapsed_ms must be finite: {elapsed}")
    return {
        "global_step": step,
        "elapsed_ms": elapsed,
        "measured": step > warmup,
    }


def m5_metrics_step_record(
    *,
    step_plan: Mapping[str, Any],
    elapsed_ms: float,
    timing_warmup_steps: int,
) -> dict[str, Any]:
    timing = m5_timing_record(
        global_step=int(step_plan["global_step"]),
        elapsed_ms=elapsed_ms,
        timing_warmup_steps=timing_warmup_steps,
    )
    return {
        "step": int(step_plan["global_step"]),
        "elapsed_ms": float(elapsed_ms),
        "timing": timing,
    }


def summarize_m5_step_timing_records(
    timing_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    executed_global_steps = [int(record["global_step"]) for record in timing_records]
    all_steps = [float(record["elapsed_ms"]) for record in timing_records]
    measured_records = [
        record
        for record in timing_records
        if bool(record.get("measured"))
    ]
    measured_global_steps = [
        int(record["global_step"])
        for record in measured_records
    ]
    measured = [float(record["elapsed_ms"]) for record in measured_records]
    summary = {
        "executed_global_steps": executed_global_steps,
        "executed_step_count": len(executed_global_steps),
        "measured_global_steps": measured_global_steps,
        "all_step_elapsed_ms": all_steps,
        "measured_step_elapsed_ms": measured,
        "measured_step_count": len(measured),
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


def m5_formal_enabled(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "m5_formal_long_train", False))


def parse_m5_formal_step_list(value: Any, *, name: str) -> tuple[int, ...]:
    if value is None:
        raise ValueError(f"{name} is required in M5 formal mode")
    if isinstance(value, str):
        raw_values: list[Any] = [] if value.strip() == "" else value.split(",")
    elif isinstance(value, Sequence):
        raw_values = list(value)
    else:
        raise TypeError(f"{name} must be a sequence or comma-separated string")

    steps: list[int] = []
    for index, raw in enumerate(raw_values):
        if type(raw) is int:
            steps.append(raw)
            continue
        if isinstance(raw, str):
            text = raw.strip()
            if text == "":
                raise ValueError(f"{name}[{index}] must be non-empty")
            try:
                steps.append(int(text))
            except ValueError as exc:
                raise ValueError(f"{name}[{index}] must be a Python int") from exc
            continue
        raise TypeError(
            f"{name}[{index}] must be a Python int, actual={type(raw).__name__}"
        )
    return tuple(steps)


def require_m5_formal_conditionals_manifest_path(value: Any) -> Path:
    if value is None:
        raise ValueError("--m5_conditionals_artifact is required in M5 formal mode")
    path = Path(value)
    if not path.is_absolute():
        raise ValueError("--m5_conditionals_artifact must be an absolute manifest.json path")
    if path.name != M5_CONDITIONAL_ARTIFACT_MANIFEST_NAME:
        raise ValueError(
            "--m5_conditionals_artifact must point to "
            f"{M5_CONDITIONAL_ARTIFACT_MANIFEST_NAME}"
        )
    if path.name.lower().endswith(".tmp"):
        raise ValueError("--m5_conditionals_artifact must not end with .tmp")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def validate_m5_formal_cli_contract(
    args: argparse.Namespace,
    *,
    cuda_device_count: int,
) -> FormalStageContract:
    if float(args.mcp1_grid_aux_weight) != 0.0:
        raise ValueError("M5 formal mode requires --mcp1_grid_aux_weight 0")
    if args.dataset_root is None:
        raise ValueError("M5 formal mode requires --dataset_root")
    validation_steps = parse_m5_formal_step_list(
        args.validation_steps,
        name="--validation_steps",
    )
    checkpoint_steps = parse_m5_formal_step_list(
        args.checkpoint_steps,
        name="--checkpoint_steps",
    )
    stage_hint = resolve_m5_formal_stage_contract(args.optimizer_steps)
    parent_hint = (
        None
        if args.resume_checkpoint is None
        else stage_hint.parent_global_step
    )
    conditionals_manifest_path = require_m5_formal_conditionals_manifest_path(
        args.m5_conditionals_artifact
    )
    return validate_m5_formal_stage_request(
        mode=args.mode,
        target_global_step=args.optimizer_steps,
        validation_steps=validation_steps,
        checkpoint_steps=checkpoint_steps,
        sample_plan_path=args.m4_sample_plan,
        conditionals_artifact_path=conditionals_manifest_path,
        device=str(args.device),
        expected_cuda_device_count=cuda_device_count,
        resume_checkpoint_path=args.resume_checkpoint,
        parent_global_step=parent_hint,
    )


def validate_config(config: Any, args: argparse.Namespace) -> None:
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if m5_formal_enabled(args):
        validate_m5_formal_cli_contract(
            args,
            cuda_device_count=torch.cuda.device_count(),
        )
    elif getattr(args, "m5_conditionals_artifact", None) is not None:
        raise ValueError("--m5_conditionals_artifact requires --m5_formal_long_train")
    if args.resume_checkpoint is not None:
        if args.mode != "joint":
            raise ValueError("M5 resume requires --mode joint")
        if not m4_enabled(args):
            raise ValueError("M5 resume requires --m4_sample_plan")
    if args.optimizer_steps <= 0:
        raise ValueError("--optimizer_steps must be positive")
    validate_timing_warmup_steps(args.timing_warmup_steps, args.optimizer_steps)
    if args.log_interval <= 0:
        raise ValueError("--log_interval must be positive")
    if args.checkpoint_interval <= 0:
        raise ValueError("--checkpoint_interval must be positive")
    if args.optimizer_steps > 300 and not m5_formal_enabled(args):
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
        if not m5_formal_enabled(args):
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
    checkpoint_has_mcp = any(is_mcp_state_key(key) for key in state_dict)

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


def require_output_dir_empty_for_resume(output_dir: Path) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"--output_dir must be empty: {output_dir}")


def load_parent_resume_checkpoint(path: Path) -> tuple[dict[str, Any], str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    parent_sha256 = file_sha256(path)
    payload = load_m3_checkpoint(path)
    return payload, parent_sha256


def current_m5_resume_run_fields(
    *,
    resolved_config: dict[str, Any],
    reference_checkpoint: Mapping[str, Any],
    selected_sample_metadata: Mapping[str, Any],
    optimizer: torch.optim.Optimizer,
    current_git_sha: str,
    sample_plan: Mapping[str, Any],
) -> dict[str, Any]:
    return build_resume_run_fields(
        resolved_config=resolved_config,
        reference_checkpoint=reference_checkpoint,
        git_sha=current_git_sha,
        optimizer_state_dict=optimizer.state_dict(),
        optimizer_group_lrs=optimizer_group_lr_summary(optimizer),
        selected_sample_metadata=selected_sample_metadata,
        sample_plan=sample_plan,
    )


def require_m5_resume_devices(
    resume_report: Mapping[str, Any],
    *,
    train_device: torch.device,
    probe_device: torch.device,
) -> None:
    rng_restore = resume_report.get("rng_restore")
    if not isinstance(rng_restore, Mapping):
        raise TypeError("M5 resume report missing rng_restore")
    expected_train = str(torch.device(train_device))
    expected_probe = str(torch.device(probe_device))
    actual_train = rng_restore.get("train_generator_device")
    actual_probe = rng_restore.get("probe_generator_device")
    if actual_train != expected_train:
        raise RuntimeError(
            "M5 train generator device mismatch: "
            f"expected={expected_train}, actual={actual_train}"
        )
    if actual_probe != expected_probe:
        raise RuntimeError(
            "M5 probe generator device mismatch: "
            f"expected={expected_probe}, actual={actual_probe}"
        )


def build_and_validate_m5_resume_report(
    *,
    parent_payload: Mapping[str, Any],
    parent_checkpoint_path: Path,
    parent_checkpoint_sha256: str,
    current_run_fields: Mapping[str, Any],
    target_global_step: int,
    sample_plan: Mapping[str, Any],
    output_dir: Path,
    target_validation_steps: tuple[int, ...] | None,
    target_checkpoint_steps: tuple[int, ...] | None,
    expected_cuda_device_count: int | None,
) -> dict[str, Any]:
    contract = build_resume_contract(
        parent_payload,
        parent_checkpoint_path=parent_checkpoint_path,
        parent_checkpoint_sha256=parent_checkpoint_sha256,
        sample_plan=sample_plan,
    )
    if not bool(contract.get("source_verified")):
        raise RuntimeError("M5 resume parent checkpoint source was not verified")
    locked_fields = contract.get("locked_fields")
    if not isinstance(locked_fields, Mapping):
        raise TypeError("M5 resume contract missing locked fields")
    if locked_fields.get("m3.mode") != "joint":
        raise ResumeContractError(
            "m3.mode",
            "joint",
            locked_fields.get("m3.mode"),
            reason="M5 resume requires joint mode",
        )
    return validate_resume_contract(
        contract,
        current_run_fields,
        target_global_step=target_global_step,
        sample_plan=sample_plan,
        output_dir=output_dir,
        target_validation_steps=target_validation_steps,
        target_checkpoint_steps=target_checkpoint_steps,
        expected_cuda_device_count=expected_cuda_device_count,
    )


def restore_m5_probe_from_checkpoint(
    *,
    parent_payload: Mapping[str, Any],
    selected_state,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[M3Probe, dict[str, Any]]:
    probe_tensors = parent_payload.get("probe_tensors")
    if not isinstance(probe_tensors, Mapping):
        raise TypeError("M5 resume checkpoint missing probe_tensors")
    noisy_batch = deserialize_noisy_batch(
        probe_tensors,
        state=selected_state,
        device=device,
        dtype=dtype,
    )
    probe_rng_state = parent_payload.get("probe_rng_state")
    if not isinstance(probe_rng_state, torch.Tensor):
        raise TypeError("M5 resume checkpoint missing probe_rng_state")
    prompt_embedding = parent_payload.get("probe_prompt_embedding")
    if not isinstance(prompt_embedding, Mapping):
        raise TypeError("M5 resume checkpoint missing probe_prompt_embedding")
    return (
        M3Probe(
            seed=int(parent_payload["probe_seed"]),
            rng_state=probe_rng_state.detach().cpu().clone(),
            noisy_batch=noisy_batch,
        ),
        move_tensors_to_device(
            prompt_embedding,
            device=device,
            floating_dtype=dtype,
        ),
    )


M5_RESTORED_PROBE_LOSS_KEYS = frozenset(
    {
        "main_loss",
        "mcp_depth1_loss",
        "mcp_depth2_loss",
        "mcp_depth3_loss",
        "total_loss",
    }
)


def _strict_restored_probe_losses(value: Any, *, field_path: str) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_path} must be a Mapping[str, float]")
    actual_keys = set(value.keys())
    if actual_keys != M5_RESTORED_PROBE_LOSS_KEYS:
        missing = sorted(M5_RESTORED_PROBE_LOSS_KEYS - actual_keys)
        extra = sorted(actual_keys - M5_RESTORED_PROBE_LOSS_KEYS, key=str)
        raise ValueError(
            f"{field_path} keys mismatch: missing={missing}, extra={extra}"
        )
    losses: dict[str, float] = {}
    for key in sorted(M5_RESTORED_PROBE_LOSS_KEYS):
        item = value[key]
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise TypeError(
                f"{field_path}.{key} must be a finite float, actual={type(item).__name__}"
            )
        number = float(item)
        if not math.isfinite(number):
            raise ValueError(f"{field_path}.{key} must be finite, actual={number}")
        losses[key] = number
    return losses


def require_restored_probe_matches_checkpoint(
    *,
    parent_payload: Mapping[str, Any],
    restored_prompt_embedding: Mapping[str, Any],
    probe_forward,
) -> dict[str, Any]:
    output_comparison = compare_probe_outputs(
        probe_forward.outputs,
        parent_payload["probe_outputs"],
    )
    if output_comparison["max_abs_diff"] != 0.0:
        raise RuntimeError("M5 restored probe outputs differ from checkpoint")
    actual_losses = _strict_restored_probe_losses(
        probe_forward.losses,
        field_path="probe_forward.losses",
    )
    expected_summary = parent_payload.get("probe_summary")
    if not isinstance(expected_summary, Mapping):
        raise TypeError("M5 resume checkpoint missing probe_summary")
    expected_losses = _strict_restored_probe_losses(
        expected_summary.get("probe_losses"),
        field_path="checkpoint.probe_summary.probe_losses",
    )
    for key, actual in actual_losses.items():
        expected = expected_losses[key]
        if actual != expected:
            raise RuntimeError(
                "M5 restored probe loss differs from checkpoint: "
                f"field={key}, expected={expected}, actual={actual}"
            )
    expected_prompt_embedding = parent_payload.get("probe_prompt_embedding")
    if not isinstance(expected_prompt_embedding, Mapping):
        raise TypeError("M5 resume checkpoint missing probe_prompt_embedding")
    try:
        prompt_comparison = compare_serialized_probe_tensors(
            restored_prompt_embedding,
            expected_prompt_embedding,
        )
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            "M5 restored probe prompt embedding structure differs from checkpoint"
        ) from exc
    if prompt_comparison["max_abs_diff"] != 0.0:
        raise RuntimeError("M5 restored probe prompt embedding differs from checkpoint")
    return {
        "status": "PASS",
        "probe_output_comparison": output_comparison,
        "probe_prompt_embedding_comparison": prompt_comparison,
        "probe_losses": actual_losses,
    }


def strict_load_m5_generator_state(
    generator,
    state_dict: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        missing, unexpected = generator.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(f"M5 generator strict load failed: {exc}") from exc
    if missing or unexpected:
        raise RuntimeError(
            "M5 generator strict load failed: "
            f"missing={missing}, unexpected={unexpected}"
        )
    return {
        "status": "PASS",
        "strict": True,
        "missing_keys": [],
        "unexpected_keys": [],
    }


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


def precompute_m5_resume_conditionals(
    *,
    text_encoder,
    sample_plan: Mapping[str, Any],
    train_samples_by_identity: Mapping[str, Any],
    validation_samples_by_identity: Mapping[str, Any],
    fixed_decode_prompt_embedding: dict[str, Any],
    conditional_cache: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    fixed_decode_identity = str(sample_plan["fixed_decode_validation_identity"])
    train_identities = [
        str(identity)
        for identity in sample_plan["train_sample_identities"]
    ]
    validation_identities = [
        str(identity)
        for identity in sample_plan["validation_sample_identities"]
    ]
    conditional_cache[fixed_decode_identity] = fixed_decode_prompt_embedding
    for identity in train_identities:
        conditional_dict_for_identity(
            text_encoder=text_encoder,
            sample=train_samples_by_identity[identity],
            identity=identity,
            cache=conditional_cache,
        )
    for identity in validation_identities:
        conditional_dict_for_identity(
            text_encoder=text_encoder,
            sample=validation_samples_by_identity[identity],
            identity=identity,
            cache=conditional_cache,
        )
    return {
        "fixed_decode_identity": fixed_decode_identity,
        "train_identities": train_identities,
        "validation_identities": validation_identities,
        "cached_identities": sorted(conditional_cache),
    }


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
        raise TypeError(
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


def run_m5_m4_validation_for_step(
    *,
    generator,
    text_encoder,
    validation_samples_by_identity: Mapping[str, Any],
    conditional_cache: dict[str, dict[str, Any]],
    scheduler_main,
    scheduler_mcp,
    device: torch.device,
    dtype: torch.dtype,
    mode: str,
    global_step: int,
    sample_plan: Mapping[str, Any],
    validation_seed: int,
    train_rng: torch.Generator,
    probe: M3Probe,
    output_dir: Path,
    current_git_sha: str,
    reference_checkpoint_sha256: str,
) -> dict[str, Any]:
    validation_samples = [
        validation_samples_by_identity[str(identity)]
        for identity in sample_plan["validation_sample_identities"]
    ]
    validation_conditionals = {
        str(identity): conditional_dict_for_identity(
            text_encoder=text_encoder,
            sample=validation_samples_by_identity[str(identity)],
            identity=str(identity),
            cache=conditional_cache,
        )
        for identity in sample_plan["validation_sample_identities"]
    }
    return run_m4_validation(
        generator=generator,
        samples=validation_samples,
        conditional_dicts=validation_conditionals,
        scheduler_main=scheduler_main,
        scheduler_mcp=scheduler_mcp,
        device=device,
        dtype=dtype,
        mode=mode,
        global_step=global_step,
        sample_plan=sample_plan,
        validation_seed=int(validation_seed),
        train_rng=train_rng,
        probe_rng_state=probe.rng_state,
        model_identity={
            "output_dir": str(output_dir.resolve()),
            "git_sha": current_git_sha,
            "reference_checkpoint_sha256": reference_checkpoint_sha256,
        },
    )


def write_m5_step_artifacts(
    *,
    output_dir: Path,
    metrics_path: Path,
    generator,
    optimizer: torch.optim.Optimizer,
    global_step: int,
    metric_record: dict[str, Any],
    should_log: bool,
    should_checkpoint: bool,
    train_rng: torch.Generator,
    probe: M3Probe,
    sample_metadata: Mapping[str, Any],
    resolved_config: dict[str, Any],
    current_git_sha: str,
    reference_checkpoint_path: Path,
    reference_checkpoint_sha256: str,
    train_seed: int,
    probe_seed: int,
    prompt_embedding: dict[str, Any],
    device: torch.device,
    strict: bool,
    mcp1_grid_aux_enabled: bool,
    mcp1_grid_aux_scheduler,
    mcp1_grid_aux_timesteps,
    state,
    train_losses: Mapping[str, float],
    aux_report: Mapping[str, Any],
    grad_audit: Mapping[str, Any],
    extra_checkpoint_payload_fields: Mapping[str, Any] | None = None,
) -> None:
    if not should_log and not should_checkpoint:
        append_run_metrics(metrics_path, metric_record, strict=strict)
        return

    grid_probe = None
    if mcp1_grid_aux_enabled:
        grid_probe = run_mcp1_grid_stable_probe(
            generator,
            conditional_dict=prompt_embedding,
            state=state,
            scheduler=mcp1_grid_aux_scheduler,
            timesteps=mcp1_grid_aux_timesteps,
            epsilon_main=probe.noisy_batch.epsilon_main,
            epsilon_future=probe.noisy_batch.epsilon_depths[0],
        )
    probe_forward = run_m3_probe_forward(
        generator,
        conditional_dict=prompt_embedding,
        noisy_batch=probe.noisy_batch,
    )
    probe_report = write_probe_report(
        output_dir,
        global_step,
        probe_forward.losses,
        probe_forward.outputs,
        grid_probe,
        strict=strict,
    )
    metric_record.update(prefix_metrics("probe", probe_forward.losses))
    if grid_probe is not None:
        metric_record.update(
            {
                "probe/mcp1_grid_probe_mean_loss": grid_probe[
                    "mcp1_grid_probe_mean_loss"
                ],
                "probe/mcp1_grid_probe_point_losses": grid_probe["point_losses"],
                "probe/mcp1_grid_probe_all_finite": grid_probe["all_finite"],
            }
        )
    if should_checkpoint:
        save_checkpoint_at_step(
            output_dir=output_dir,
            generator=generator,
            optimizer=optimizer,
            step=global_step,
            train_rng=train_rng,
            probe=probe,
            probe_summary=probe_report,
            probe_outputs=probe_forward.outputs,
            sample_metadata=dict(sample_metadata),
            resolved_config=resolved_config,
            git_sha=current_git_sha,
            reference_checkpoint_path=reference_checkpoint_path,
            reference_checkpoint_sha256=reference_checkpoint_sha256,
            train_seed=train_seed,
            probe_seed=probe_seed,
            prompt_embedding=prompt_embedding,
            device=device,
            extra_payload_fields=extra_checkpoint_payload_fields,
        )
    append_run_metrics(metrics_path, metric_record, strict=strict)
    print(
        json.dumps(
            {
                "step": global_step,
                **prefix_metrics("train", train_losses),
                "train/mcp1_grid_aux_mean_loss": aux_report[
                    "mcp1_grid_aux_mean_loss"
                ],
                "train/mcp1_grid_aux_weighted_loss": aux_report[
                    "mcp1_grid_aux_weighted_loss"
                ],
                "train/combined_objective": metric_record["train/combined_objective"],
                **prefix_metrics("probe", probe_forward.losses),
                "grad_audit": grad_audit,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


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
                    "grad_object_id": id(grad),
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
            "tensor_count": len(tensors),
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
    device: torch.device,
    extra_payload_fields: Mapping[str, Any] | None = None,
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
    payload[M5_RNG_EXTENSION_FIELD] = capture_m5_global_rng_extension(
        include_cuda=device.type == "cuda",
        train_generator_device=device,
        probe_generator_device=device,
    )
    if extra_payload_fields is not None:
        for key, value in extra_payload_fields.items():
            payload[str(key)] = value
    save_m3_checkpoint(payload, path)
    return path


def m5_formal_step_sets(
    args: argparse.Namespace,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    return (
        parse_m5_formal_step_list(
            args.validation_steps,
            name="--validation_steps",
        ),
        parse_m5_formal_step_list(
            args.checkpoint_steps,
            name="--checkpoint_steps",
        ),
    )


def m5_formal_artifact_dir_from_manifest(path: Path) -> Path:
    path = require_m5_formal_conditionals_manifest_path(path)
    return path.parent


def m5_formal_checkpoint_metadata(
    *,
    stage_contract: FormalStageContract,
    resolved_config: Mapping[str, Any],
    conditionals_manifest_path: Path,
    conditional_artifact_sha256: str,
) -> dict[str, Any]:
    formal_config = resolved_config.get("m5_formal")
    if not isinstance(formal_config, Mapping):
        raise TypeError("resolved_config missing m5_formal block")
    return {
        "schema": M5_FORMAL_TRAINER_SCHEMA,
        "status": "PASS",
        "formal_enabled": True,
        "stage": m5_formal_stage_name(stage_contract),
        "stage_contract": {
            "target_global_step": int(stage_contract.target_global_step),
            "parent_global_step": stage_contract.parent_global_step,
            "validation_steps": list(stage_contract.validation_steps),
            "checkpoint_steps": list(stage_contract.checkpoint_steps),
            "is_resume_stage": bool(stage_contract.is_resume_stage),
        },
        "sample_plan_sha256": str(formal_config["sample_plan_sha256"]),
        "teacher_manifest_sha256": str(formal_config["teacher_manifest_sha256"]),
        "conditional_manifest_path": str(conditionals_manifest_path.resolve()),
        "conditional_artifact_sha256": str(conditional_artifact_sha256),
        "validation_implementation_schema": M5_STREAMING_VALIDATION_SCHEMA,
    }


def m5_formal_stage_name(stage_contract: FormalStageContract) -> str:
    names = {
        500: "stage_a",
        2000: "stage_b",
        5000: "stage_c",
    }
    return names[int(stage_contract.target_global_step)]


def m5_formal_stage_contract_json(
    stage_contract: FormalStageContract,
) -> dict[str, Any]:
    return {
        "target_global_step": int(stage_contract.target_global_step),
        "parent_global_step": stage_contract.parent_global_step,
        "validation_steps": list(stage_contract.validation_steps),
        "checkpoint_steps": list(stage_contract.checkpoint_steps),
        "is_resume_stage": bool(stage_contract.is_resume_stage),
    }


def _require_m5_formal_int(value: Any, field_path: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{field_path} must be a Python int")
    return value


def _require_m5_formal_optional_int(value: Any, field_path: str) -> int | None:
    if value is None:
        return None
    return _require_m5_formal_int(value, field_path)


def _require_m5_formal_bool(value: Any, field_path: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{field_path} must be a bool")
    return value


def _require_m5_formal_schedule(value: Any, field_path: str) -> list[int]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise TypeError(f"{field_path} must be an int sequence")
    return [
        _require_m5_formal_int(item, f"{field_path}[{index}]")
        for index, item in enumerate(value)
    ]


def _require_m5_formal_stage_contract_mapping(
    value: Any,
    field_path: str,
) -> dict[str, Any]:
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
        "target_global_step": _require_m5_formal_int(
            value["target_global_step"],
            f"{field_path}.target_global_step",
        ),
        "parent_global_step": _require_m5_formal_optional_int(
            value["parent_global_step"],
            f"{field_path}.parent_global_step",
        ),
        "validation_steps": _require_m5_formal_schedule(
            value["validation_steps"],
            f"{field_path}.validation_steps",
        ),
        "checkpoint_steps": _require_m5_formal_schedule(
            value["checkpoint_steps"],
            f"{field_path}.checkpoint_steps",
        ),
        "is_resume_stage": _require_m5_formal_bool(
            value["is_resume_stage"],
            f"{field_path}.is_resume_stage",
        ),
    }


def require_m5_formal_parent_checkpoint(
    *,
    parent_payload: Mapping[str, Any],
    stage_contract: FormalStageContract,
    sample_plan_sha256: str,
    teacher_manifest_sha256: str,
    conditional_artifact_sha256: str,
    current_git_sha: str,
) -> dict[str, Any]:
    expected_parent = stage_contract.parent_global_step
    if expected_parent is None:
        raise RuntimeError("fresh formal stage must not validate a parent checkpoint")
    actual_step = _require_m5_formal_int(
        parent_payload.get("global_step"),
        "parent_payload.global_step",
    )
    if actual_step != int(expected_parent):
        raise RuntimeError(
            "M5 formal parent global_step mismatch: "
            f"expected={expected_parent}, actual={actual_step}"
        )
    metadata = parent_payload.get("m5_formal_trainer")
    if not isinstance(metadata, Mapping):
        raise TypeError("M5 formal parent checkpoint missing formal metadata")
    if metadata.get("schema") != M5_FORMAL_TRAINER_SCHEMA:
        raise RuntimeError("M5 formal parent checkpoint schema mismatch")
    if metadata.get("status") != "PASS":
        raise RuntimeError(
            "M5 formal parent checkpoint status mismatch: "
            f"expected=PASS, actual={metadata.get('status')}"
        )
    if metadata.get("formal_enabled") is not True:
        raise RuntimeError("M5 formal parent checkpoint marker is not enabled")
    expected_parent_contract = resolve_m5_formal_stage_contract(expected_parent)
    expected_stage = m5_formal_stage_name(expected_parent_contract)
    actual_stage = metadata.get("stage")
    if actual_stage != expected_stage:
        raise RuntimeError(
            "M5 formal parent checkpoint stage mismatch: "
            f"expected={expected_stage}, actual={actual_stage}"
        )
    expected_stage_contract = m5_formal_stage_contract_json(expected_parent_contract)
    actual_stage_contract = _require_m5_formal_stage_contract_mapping(
        metadata.get("stage_contract"),
        "m5_formal_trainer.stage_contract",
    )
    if actual_stage_contract != expected_stage_contract:
        raise RuntimeError(
            "M5 formal parent checkpoint stage_contract mismatch: "
            f"expected={expected_stage_contract}, actual={actual_stage_contract}"
        )
    expected_git_sha = validate_git_sha(str(current_git_sha))
    parent_git_sha = parent_payload.get("git_sha")
    if not isinstance(parent_git_sha, str):
        raise TypeError("M5 formal parent checkpoint git_sha must be a string")
    actual_git_sha = validate_git_sha(parent_git_sha)
    if actual_git_sha != expected_git_sha:
        raise RuntimeError(
            "M5 formal parent checkpoint git_sha mismatch: "
            f"expected={expected_git_sha}, actual={actual_git_sha}"
        )
    checks = {
        "sample_plan_sha256": sample_plan_sha256,
        "teacher_manifest_sha256": teacher_manifest_sha256,
        "conditional_artifact_sha256": conditional_artifact_sha256,
        "validation_implementation_schema": M5_STREAMING_VALIDATION_SCHEMA,
    }
    for field, expected in checks.items():
        actual = metadata.get(field)
        if actual != expected:
            raise RuntimeError(
                "M5 formal parent checkpoint provenance mismatch: "
                f"field={field}, expected={expected}, actual={actual}"
            )
    resolved_config = parent_payload.get("resolved_config")
    if not isinstance(resolved_config, Mapping):
        raise TypeError("M5 formal parent checkpoint missing resolved_config")
    formal_config = resolved_config.get("m5_formal")
    if not isinstance(formal_config, Mapping):
        raise TypeError("M5 formal parent resolved_config missing m5_formal")
    config_checks = {
        "schema": M5_FORMAL_TRAINER_SCHEMA,
        "enabled": True,
        "sample_plan_sha256": sample_plan_sha256,
        "teacher_manifest_sha256": teacher_manifest_sha256,
        "conditional_artifact_sha256": conditional_artifact_sha256,
        "validation_implementation_schema": M5_STREAMING_VALIDATION_SCHEMA,
    }
    for field, expected in config_checks.items():
        actual = formal_config.get(field)
        if actual != expected:
            raise RuntimeError(
                "M5 formal parent resolved_config provenance mismatch: "
                f"field=m5_formal.{field}, expected={expected}, actual={actual}"
            )
    return dict(metadata)


def write_m5_formal_validation_report(
    *,
    output_dir: Path,
    metrics_path: Path,
    report: dict[str, Any],
) -> None:
    step = int(report["global_step"])
    atomic_strict_json_write(report, output_dir / f"validation_step{step:06d}.json")
    nonfinite_count = len(report.get("nonfinite_validation_losses", []))
    append_strict_metrics(
        metrics_path,
        {
            "event": "validation",
            "step": step,
            "mode": report["mode"],
            "sample_plan_sha256": report["sample_plan_sha256"],
            "conditional_artifact_sha256": report["conditional_artifact_sha256"],
            "validation_status": report["status"],
            "validation_sample_count": report["sample_count"],
            "validation_loss_finite_contract": report[
                "validation_loss_finite_contract"
            ],
            "nonfinite_validation_loss_count": int(nonfinite_count),
            **{
                f"validation/{key}": value
                for key, value in report["aggregate_losses"].items()
            },
        },
    )


def handle_m5_formal_validation_report(
    *,
    output_dir: Path,
    metrics_path: Path,
    validation_reports: list[dict[str, Any]],
    report: dict[str, Any],
    global_step: int,
) -> None:
    if not bool(report.get("validation_loss_finite_contract", False)):
        write_m5_formal_validation_report(
            output_dir=output_dir,
            metrics_path=metrics_path,
            report=report,
        )
        raise RuntimeError(f"M5 formal validation loss contract failed at step {global_step}")
    write_m5_formal_validation_report(
        output_dir=output_dir,
        metrics_path=metrics_path,
        report=report,
    )
    require_m4_validation_pass(report, global_step=global_step)
    validation_reports.append(report)


def run_m5_formal_validation_for_step(
    *,
    generator,
    teacher_store: M5TeacherSampleStore,
    conditional_store: M5ConditionalArtifactStore,
    scheduler_main,
    scheduler_mcp,
    device: torch.device,
    dtype: torch.dtype,
    mode: str,
    global_step: int,
    sample_plan: Mapping[str, Any],
    validation_seed: int,
    train_rng: torch.Generator,
    probe: M3Probe,
    output_dir: Path,
    current_git_sha: str,
    reference_checkpoint_sha256: str,
) -> dict[str, Any]:
    return run_m5_streaming_validation(
        generator=generator,
        teacher_store=teacher_store,
        conditional_store=conditional_store,
        sample_plan=sample_plan,
        scheduler_main=scheduler_main,
        scheduler_mcp=scheduler_mcp,
        device=device,
        dtype=dtype,
        mode=mode,
        global_step=global_step,
        validation_seed=int(validation_seed),
        train_rng=train_rng,
        probe_rng_state=probe.rng_state,
        model_identity={
            "output_dir": str(output_dir.resolve()),
            "git_sha": current_git_sha,
            "reference_checkpoint_sha256": reference_checkpoint_sha256,
        },
    )


def m5_formal_conditional_to_device(
    conditional: Mapping[str, Any],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    value = move_tensors_to_device(
        conditional,
        device=device,
        floating_dtype=dtype,
    )
    if not isinstance(value, dict):
        raise TypeError("formal conditional must materialize as a dict")
    return value


def acquire_m5_formal_fixed_probe_inputs(
    *,
    teacher_store: M5TeacherSampleStore,
    conditional_store: M5ConditionalArtifactStore,
    sample_plan: Mapping[str, Any],
    scheduler_main,
    scheduler_mcp,
    device: torch.device,
    dtype: torch.dtype,
    probe_seed: int,
) -> tuple[dict[str, Any], M3Probe, dict[str, Any]]:
    fixed_identity = str(sample_plan["fixed_decode_validation_identity"])
    sample = None
    cpu_conditional = None
    fixed_state = None
    fixed_prompt_embedding = None
    try:
        with (
            teacher_store.acquire(fixed_identity) as sample,
            conditional_store.acquire(fixed_identity) as cpu_conditional,
        ):
            fixed_state = selected_state_to_device(
                sample.selected_state,
                device=device,
                dtype=dtype,
            )
            fixed_prompt_embedding = m5_formal_conditional_to_device(
                cpu_conditional,
                device=device,
                dtype=dtype,
            )
            probe = make_m3_probe(
                fixed_state,
                scheduler_main=scheduler_main,
                scheduler_mcp=scheduler_mcp,
                seed=probe_seed,
            )
            return dict(sample.metadata), probe, fixed_prompt_embedding
    finally:
        fixed_state = None
        fixed_prompt_embedding = None
        cpu_conditional = None
        sample = None


def acquire_m5_formal_fixed_sample_state(
    *,
    teacher_store: M5TeacherSampleStore,
    sample_plan: Mapping[str, Any],
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[dict[str, Any], Any]:
    fixed_identity = str(sample_plan["fixed_decode_validation_identity"])
    sample = None
    fixed_state = None
    try:
        with teacher_store.acquire(fixed_identity) as sample:
            fixed_state = selected_state_to_device(
                sample.selected_state,
                device=device,
                dtype=dtype,
            )
            return dict(sample.metadata), fixed_state
    finally:
        fixed_state = None
        sample = None


def write_m5_formal_probe_checkpoint_node(
    *,
    output_dir: Path,
    metrics_path: Path,
    generator,
    optimizer: torch.optim.Optimizer,
    global_step: int,
    train_rng: torch.Generator,
    probe: M3Probe,
    fixed_sample_metadata: Mapping[str, Any],
    resolved_config: dict[str, Any],
    current_git_sha: str,
    reference_checkpoint_path: Path,
    reference_checkpoint_sha256: str,
    train_seed: int,
    probe_seed: int,
    fixed_prompt_embedding: dict[str, Any],
    device: torch.device,
    formal_checkpoint_metadata: Mapping[str, Any],
) -> Path:
    probe_forward = run_m3_probe_forward(
        generator,
        conditional_dict=fixed_prompt_embedding,
        noisy_batch=probe.noisy_batch,
    )
    probe_report = write_probe_report(
        output_dir,
        global_step,
        probe_forward.losses,
        probe_forward.outputs,
        None,
        strict=True,
    )
    checkpoint_path = save_checkpoint_at_step(
        output_dir=output_dir,
        generator=generator,
        optimizer=optimizer,
        step=global_step,
        train_rng=train_rng,
        probe=probe,
        probe_summary=probe_report,
        probe_outputs=probe_forward.outputs,
        sample_metadata=dict(fixed_sample_metadata),
        resolved_config=resolved_config,
        git_sha=current_git_sha,
        reference_checkpoint_path=reference_checkpoint_path,
        reference_checkpoint_sha256=reference_checkpoint_sha256,
        train_seed=train_seed,
        probe_seed=probe_seed,
        prompt_embedding=fixed_prompt_embedding,
        device=device,
        extra_payload_fields={"m5_formal_trainer": dict(formal_checkpoint_metadata)},
    )
    append_strict_metrics(
        metrics_path,
        {
            "event": "probe_checkpoint",
            "step": int(global_step),
            **prefix_metrics("probe", probe_forward.losses),
            "checkpoint_path": str(checkpoint_path.resolve()),
        },
    )
    return checkpoint_path


def require_m5_formal_finite_train_losses(
    losses: Any,
    *,
    global_step: int,
) -> dict[str, float]:
    train_losses = loss_dict_to_floats(losses)
    nonfinite_fields = [
        field for field, value in train_losses.items()
        if not math.isfinite(float(value))
    ]
    if nonfinite_fields:
        raise RuntimeError(
            "M5 formal non-finite train loss before backward: "
            f"global_step={global_step}, fields={nonfinite_fields}"
        )
    return train_losses


def run_m5_formal_train_step(
    *,
    step: int,
    target_global_step: int,
    teacher_store: M5TeacherSampleStore,
    conditional_store: M5ConditionalArtifactStore,
    sample_plan: Mapping[str, Any],
    generator,
    optimizer: torch.optim.Optimizer,
    scheduler_main,
    scheduler_mcp,
    train_rng: torch.Generator,
    device: torch.device,
    dtype: torch.dtype,
    mode: str,
    validation_steps: Sequence[int],
    checkpoint_steps: Sequence[int],
    checkpoint_interval: int,
    log_interval: int,
    timing_warmup_steps: int,
) -> tuple[dict[str, Any], dict[str, float], dict[str, Any], dict[str, Any]]:
    train_sample = None
    cpu_conditional = None
    device_conditional: dict[str, Any] = {}
    step_state = None
    noisy_batch = None
    result = None
    try:
        step_plan = m5_training_step_orchestration(
            global_step=step,
            target_global_step=target_global_step,
            sample_plan=sample_plan,
            validation_steps=validation_steps,
            checkpoint_steps=checkpoint_steps,
            checkpoint_interval=checkpoint_interval,
            log_interval=log_interval,
        )
        train_identity = teacher_store.train_identity_for_step(step)
        if train_identity != str(step_plan["train_sample_identity"]):
            raise RuntimeError(
                "formal train identity mismatch: "
                f"step_plan={step_plan['train_sample_identity']}, "
                f"teacher_store={train_identity}"
            )
        optimizer.zero_grad(set_to_none=True)
        with (
            teacher_store.acquire(train_identity) as train_sample,
            conditional_store.acquire(train_identity) as cpu_conditional,
        ):
            step_state = selected_state_to_device(
                train_sample.selected_state,
                device=device,
                dtype=dtype,
            )
            device_conditional = m5_formal_conditional_to_device(
                cpu_conditional,
                device=device,
                dtype=dtype,
            )
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
                conditional_dict=device_conditional,
                noisy_batch=noisy_batch,
                depth_weights=M3_DEPTH_WEIGHTS,
            )
            train_losses = require_m5_formal_finite_train_losses(
                result.losses,
                global_step=step,
            )
            result.losses.total_loss.backward()
            parameter_groups = named_parameter_groups(generator)
            random_grad_audit = gradient_group_audit(
                optimizer,
                mode=mode,
                parameter_groups=parameter_groups,
            )
            if not gradient_audit_contract_pass(random_grad_audit):
                raise RuntimeError(
                    f"random gradient audit contract failed at step {step}"
                )
            aux_report = accumulate_mcp1_grid_aux_gradients(
                generator,
                conditional_dict=device_conditional,
                state=step_state,
                scheduler=None,
                timesteps=None,
                epsilon_main=noisy_batch.epsilon_main,
                epsilon_future=noisy_batch.epsilon_depths[0],
                weight=0.0,
            )
            grad_audit = gradient_group_audit(
                optimizer,
                mode=mode,
                parameter_groups=parameter_groups,
            )
            if not gradient_audit_contract_pass(grad_audit):
                raise RuntimeError(f"gradient audit contract failed at step {step}")
            if has_nonfinite_grad(generator):
                raise RuntimeError(f"non-finite gradient at step {step}")
            optimizer.step()

        metric_step_record = m5_metrics_step_record(
            step_plan=step_plan,
            elapsed_ms=0.0,
            timing_warmup_steps=timing_warmup_steps,
        )
        metric_record = {
            **metric_step_record,
            **prefix_metrics("train", train_losses),
            "train/random_total_loss": train_losses["total_loss"],
            "train/mcp1_grid_aux_mean_loss": aux_report[
                "mcp1_grid_aux_mean_loss"
            ],
            "train/mcp1_grid_aux_weighted_loss": aux_report[
                "mcp1_grid_aux_weighted_loss"
            ],
            "train/combined_objective": train_losses["total_loss"],
            "mcp1_grid_aux": aux_report,
            "random_grad_audit": random_grad_audit,
            "grad_audit": grad_audit,
            "should_log": bool(step_plan["should_log"]),
            "train_sample_identity": step_plan["train_sample_identity"],
            "train_sample_position": step_plan["train_sample_position"],
            "train_cycle_index": step_plan["train_cycle_index"],
            "should_validate": bool(step_plan["should_validate"]),
            "should_checkpoint": bool(step_plan["should_checkpoint"]),
        }
        return metric_record, train_losses, aux_report, grad_audit
    finally:
        device_conditional.clear()
        step_state = None
        noisy_batch = None
        result = None
        cpu_conditional = None
        train_sample = None


def build_m5_formal_resolved_config(
    *,
    config: Any,
    args: argparse.Namespace,
    device: torch.device,
    sample_plan: Mapping[str, Any],
    sample_plan_sha256: str,
    formal_plan_audit: Mapping[str, Any],
    conditionals_manifest_path: Path,
    conditionals_manifest: Mapping[str, Any],
    conditional_artifact_sha256: str,
    optimizer_config: Mapping[str, Any],
) -> dict[str, Any]:
    validation_steps, checkpoint_steps = m5_formal_step_sets(args)
    stage_topology = {}
    for target in (500, 2000, 5000):
        contract = resolve_m5_formal_stage_contract(target)
        stage_topology[str(target)] = {
            "parent_global_step": contract.parent_global_step,
            "validation_steps": list(contract.validation_steps),
            "checkpoint_steps": list(contract.checkpoint_steps),
        }
    return {
        "model_config": resolved_config_dict(config),
        "m3": {
            "mode": args.mode,
            "manifest": str(args.manifest.resolve()),
            "dataset_root": str(args.dataset_root.resolve()),
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
            "mcp1_grid_aux_weight": 0.0,
            "mcp1_grid_aux_enabled": False,
            "mcp1_grid_timesteps": [],
            "mcp1_grid_schedule": None,
            "optimizer_config": dict(optimizer_config),
            "dtype": args.dtype,
            "device": str(device),
        },
        "m4": {
            "enabled": True,
            "sample_plan_path": str(Path(args.m4_sample_plan).resolve()),
            "sample_plan_sha256": sample_plan_sha256,
            "train_sample_identities": list(sample_plan["train_sample_identities"]),
            "validation_sample_identities": list(
                sample_plan["validation_sample_identities"]
            ),
            "train_subset_size": int(sample_plan["train_subset_size"]),
            "validation_subset_size": int(sample_plan["validation_subset_size"]),
            "validation_seed": int(args.validation_seed),
            "validation_steps": list(validation_steps),
            "checkpoint_steps": list(checkpoint_steps),
            "fixed_decode_validation_identity": str(
                sample_plan["fixed_decode_validation_identity"]
            ),
            "sample_ordering_rule": str(sample_plan["ordering_rule"]),
            "ordering_rule": str(sample_plan["ordering_rule"]),
        },
        "m5_formal": {
            "schema": M5_FORMAL_TRAINER_SCHEMA,
            "enabled": True,
            "run_kind": "formal_long_train",
            "stage_topology": stage_topology,
            "mode": "joint",
            "device": "cuda:0",
            "expected_cuda_device_count": 1,
            "mcp1_grid_aux_weight": 0.0,
            "sample_plan_schema": str(sample_plan["schema"]),
            "sample_plan_sha256": sample_plan_sha256,
            "train_sample_count": int(formal_plan_audit["train_sample_count"]),
            "validation_sample_count": int(
                formal_plan_audit["validation_sample_count"]
            ),
            "teacher_manifest_path": str(args.manifest.resolve()),
            "teacher_manifest_sha256": str(formal_plan_audit["manifest_sha256"]),
            "dataset_root": str(args.dataset_root.resolve()),
            "conditional_manifest_path": str(conditionals_manifest_path.resolve()),
            "conditional_schema": str(conditionals_manifest["schema"]),
            "conditional_artifact_sha256": conditional_artifact_sha256,
            "conditional_encoder_provenance": dict(
                conditionals_manifest["encoder_provenance"]
            ),
            "validation_implementation_schema": M5_STREAMING_VALIDATION_SCHEMA,
        },
    }


def run_m5_formal_training(
    *,
    args: argparse.Namespace,
    config: Any,
    dtype: torch.dtype,
    device: torch.device,
    current_git_sha: str,
    reference_checkpoint_sha256: str,
) -> None:
    require_output_dir_empty_for_resume(args.output_dir)
    stage_contract = validate_m5_formal_cli_contract(
        args,
        cuda_device_count=torch.cuda.device_count(),
    )
    validation_steps, checkpoint_steps = m5_formal_step_sets(args)
    conditionals_manifest_path = require_m5_formal_conditionals_manifest_path(
        args.m5_conditionals_artifact
    )
    m4_plan = load_m4_sample_plan(args.m4_sample_plan, manifest_path=args.manifest)
    saved_plan_sha = str(m4_plan["sample_plan_sha256"])
    formal_plan_audit = validate_m5_formal_sample_plan(
        m4_plan,
        manifest_path=args.manifest,
        dataset_root=args.dataset_root,
        expected_sha256=saved_plan_sha,
    )
    if int(formal_plan_audit["train_sample_count"]) != M5_FORMAL_TRAIN_SAMPLE_COUNT:
        raise RuntimeError("M5 formal train sample count mismatch")
    if (
        int(formal_plan_audit["validation_sample_count"])
        != M5_FORMAL_VALIDATION_SAMPLE_COUNT
    ):
        raise RuntimeError("M5 formal validation sample count mismatch")

    conditionals_manifest = load_m5_conditional_artifact_manifest(
        conditionals_manifest_path
    )
    conditional_artifact_dir = m5_formal_artifact_dir_from_manifest(
        conditionals_manifest_path
    )
    conditional_audit = validate_m5_conditional_artifact_manifest(
        conditionals_manifest,
        artifact_dir=conditional_artifact_dir,
        sample_plan=m4_plan,
    )
    conditional_artifact_sha256 = str(conditional_audit["artifact_sha256"])
    teacher_store = M5TeacherSampleStore(
        sample_plan=m4_plan,
        manifest_path=args.manifest,
        dataset_root=args.dataset_root,
        reference_checkpoint_path=args.checkpoint,
    )
    conditional_store = M5ConditionalArtifactStore(
        artifact_dir=conditional_artifact_dir,
        sample_plan=m4_plan,
        expected_artifact_sha256=conditional_artifact_sha256,
    )
    if teacher_store.sample_plan_sha256 != saved_plan_sha:
        raise RuntimeError("M5 formal teacher store sample plan SHA mismatch")
    if conditional_store.sample_plan_sha256 != saved_plan_sha:
        raise RuntimeError("M5 formal conditional store sample plan SHA mismatch")
    if teacher_store.manifest_sha256 != str(formal_plan_audit["manifest_sha256"]):
        raise RuntimeError("M5 formal teacher store manifest SHA mismatch")
    if conditional_store.teacher_manifest_sha256 != str(
        formal_plan_audit["manifest_sha256"]
    ):
        raise RuntimeError("M5 formal conditional store manifest SHA mismatch")

    parent_resume_payload = None
    parent_resume_sha256 = None
    parent_formal_metadata = None
    if stage_contract.is_resume_stage:
        assert args.resume_checkpoint is not None
        parent_resume_payload, parent_resume_sha256 = load_parent_resume_checkpoint(
            args.resume_checkpoint
        )
        parent_formal_metadata = require_m5_formal_parent_checkpoint(
            parent_payload=parent_resume_payload,
            stage_contract=stage_contract,
            sample_plan_sha256=saved_plan_sha,
            teacher_manifest_sha256=str(formal_plan_audit["manifest_sha256"]),
            conditional_artifact_sha256=conditional_artifact_sha256,
            current_git_sha=current_git_sha,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_m4_sample_plan(m4_plan, args.output_dir / "m4_sample_plan.json")
    (args.output_dir / "reference_checkpoint_sha256.txt").write_text(
        reference_checkpoint_sha256 + "\n",
        encoding="utf-8",
    )
    if parent_resume_sha256 is not None:
        (args.output_dir / "parent_checkpoint_sha256.txt").write_text(
            parent_resume_sha256 + "\n",
            encoding="utf-8",
        )
    (args.output_dir / "git_sha.txt").write_text(current_git_sha + "\n", encoding="utf-8")

    optimizer_config = {
        "optimizer": "AdamW",
        "betas": [float(value) for value in ADAMW_BETAS],
        "eps": ADAMW_EPS,
        "weight_decay": args.weight_decay,
    }
    resolved_config = build_m5_formal_resolved_config(
        config=config,
        args=args,
        device=device,
        sample_plan=m4_plan,
        sample_plan_sha256=saved_plan_sha,
        formal_plan_audit=formal_plan_audit,
        conditionals_manifest_path=conditionals_manifest_path,
        conditionals_manifest=conditionals_manifest,
        conditional_artifact_sha256=conditional_artifact_sha256,
        optimizer_config=optimizer_config,
    )
    write_run_json(
        resolved_config,
        args.output_dir / "resolved_config.json",
        strict=True,
    )
    formal_checkpoint_metadata = m5_formal_checkpoint_metadata(
        stage_contract=stage_contract,
        resolved_config=resolved_config,
        conditionals_manifest_path=conditionals_manifest_path,
        conditional_artifact_sha256=conditional_artifact_sha256,
    )

    generator = None
    text_encoder = None
    train_rng = None
    metrics_path = args.output_dir / "metrics.jsonl"
    try:
        if not stage_contract.is_resume_stage:
            reset_global_seed(args.train_seed)
        generator, load_mode, mcp_tensor_count = load_generator(config, args.checkpoint)
        generator.to(device=device, dtype=dtype)
        generator.train()
        scheduler_main = generator.get_scheduler()
        scheduler_main.sigmas = scheduler_main.sigmas.to(device)
        scheduler_main.timesteps = scheduler_main.timesteps.to(device)
        scheduler_mcp = make_mcp_scheduler(device)

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
            strict=True,
        )

        fixed_state = None
        if stage_contract.is_resume_stage:
            fixed_sample_metadata, fixed_state = acquire_m5_formal_fixed_sample_state(
                teacher_store=teacher_store,
                sample_plan=m4_plan,
                device=device,
                dtype=dtype,
            )
            probe = None
            fixed_prompt_embedding = {}
        else:
            fixed_sample_metadata, probe, fixed_prompt_embedding = (
                acquire_m5_formal_fixed_probe_inputs(
                    teacher_store=teacher_store,
                    conditional_store=conditional_store,
                    sample_plan=m4_plan,
                    scheduler_main=scheduler_main,
                    scheduler_mcp=scheduler_mcp,
                    device=device,
                    dtype=dtype,
                    probe_seed=args.probe_seed,
                )
            )
        write_run_json(
            fixed_sample_metadata,
            args.output_dir / "sample_metadata.json",
            strict=True,
        )

        resumed_global_step = None
        resume_report = None
        if stage_contract.is_resume_stage:
            assert parent_resume_payload is not None
            assert parent_resume_sha256 is not None
            assert args.resume_checkpoint is not None
            current_run_fields = current_m5_resume_run_fields(
                resolved_config=resolved_config,
                reference_checkpoint={
                    "path": args.checkpoint,
                    "sha256": reference_checkpoint_sha256,
                },
                selected_sample_metadata=fixed_sample_metadata,
                optimizer=optimizer,
                current_git_sha=current_git_sha,
                sample_plan=m4_plan,
            )
            resume_report = build_and_validate_m5_resume_report(
                parent_payload=parent_resume_payload,
                parent_checkpoint_path=args.resume_checkpoint,
                parent_checkpoint_sha256=parent_resume_sha256,
                current_run_fields=current_run_fields,
                target_global_step=int(args.optimizer_steps),
                sample_plan=m4_plan,
                output_dir=args.output_dir,
                target_validation_steps=validation_steps,
                target_checkpoint_steps=checkpoint_steps,
                expected_cuda_device_count=1,
            )
            require_m5_resume_devices(
                resume_report,
                train_device=device,
                probe_device=device,
            )
            generator_restore = strict_load_m5_generator_state(
                generator,
                parent_resume_payload["generator"],
            )
            optimizer.load_state_dict(parent_resume_payload["optimizer"])
            optimizer_device_report = move_loaded_optimizer_state_to_device(
                optimizer,
                device=device,
            )
            rng_states = extract_resume_rng_states(parent_resume_payload)
            train_rng = restore_torch_generator_from_state(
                rng_states["train_generator_state"],
                device=device,
            )
            restored_probe, restored_prompt_embedding = restore_m5_probe_from_checkpoint(
                parent_payload=parent_resume_payload,
                selected_state=fixed_state,
                device=device,
                dtype=dtype,
            )
            restored_probe_forward = run_m3_probe_forward(
                generator,
                conditional_dict=restored_prompt_embedding,
                noisy_batch=restored_probe.noisy_batch,
            )
            probe_restore = require_restored_probe_matches_checkpoint(
                parent_payload=parent_resume_payload,
                restored_prompt_embedding=restored_prompt_embedding,
                probe_forward=restored_probe_forward,
            )
            probe = restored_probe
            fixed_prompt_embedding = restored_prompt_embedding
            fixed_state = None
            resumed_global_step = int(parent_resume_payload["global_step"])
            resume_report.update(
                {
                    "formal_parent_metadata": parent_formal_metadata,
                    "generator_restore": generator_restore,
                    "optimizer_device_restore": optimizer_device_report,
                    "probe_restore": probe_restore,
                    "conditional_artifact_restore": {
                        "status": "PASS",
                        "artifact_sha256": conditional_artifact_sha256,
                        "manifest_path": str(conditionals_manifest_path.resolve()),
                    },
                }
            )
            write_run_json(
                resume_report,
                args.output_dir / "resume_report.json",
                strict=True,
            )
            parent_resume_payload = None
            restored_probe_forward = None
            gc.collect()
            restore_global_rng_states(rng_states)
        else:
            train_rng = make_generator(args.train_seed, device)
            if 0 in validation_steps:
                report = run_m5_formal_validation_for_step(
                    generator=generator,
                    teacher_store=teacher_store,
                    conditional_store=conditional_store,
                    scheduler_main=scheduler_main,
                    scheduler_mcp=scheduler_mcp,
                    device=device,
                    dtype=dtype,
                    mode=args.mode,
                    global_step=0,
                    sample_plan=m4_plan,
                    validation_seed=int(args.validation_seed),
                    train_rng=train_rng,
                    probe=probe,
                    output_dir=args.output_dir,
                    current_git_sha=current_git_sha,
                    reference_checkpoint_sha256=reference_checkpoint_sha256,
                )
                validation_reports: list[dict[str, Any]] = []
                handle_m5_formal_validation_report(
                    output_dir=args.output_dir,
                    metrics_path=metrics_path,
                    validation_reports=validation_reports,
                    report=report,
                    global_step=0,
                )
            else:
                validation_reports = []
            if 0 in checkpoint_steps:
                write_m5_formal_probe_checkpoint_node(
                    output_dir=args.output_dir,
                    metrics_path=metrics_path,
                    generator=generator,
                    optimizer=optimizer,
                    global_step=0,
                    train_rng=train_rng,
                    probe=probe,
                    fixed_sample_metadata=fixed_sample_metadata,
                    resolved_config=resolved_config,
                    current_git_sha=current_git_sha,
                    reference_checkpoint_path=args.checkpoint,
                    reference_checkpoint_sha256=reference_checkpoint_sha256,
                    train_seed=args.train_seed,
                    probe_seed=args.probe_seed,
                    fixed_prompt_embedding=fixed_prompt_embedding,
                    device=device,
                    formal_checkpoint_metadata=formal_checkpoint_metadata,
                )

        if stage_contract.is_resume_stage:
            validation_reports = []
        step_timing_records: list[dict[str, Any]] = []
        reset_peak_memory_stats_if_available(device)
        assert train_rng is not None
        for step in m5_absolute_training_steps(
            resumed_global_step=resumed_global_step,
            target_global_step=args.optimizer_steps,
        ):
            cuda_synchronize_if_available(device)
            started = time.perf_counter()
            metric_record, train_losses, aux_report, grad_audit = (
                run_m5_formal_train_step(
                    step=step,
                    target_global_step=args.optimizer_steps,
                    teacher_store=teacher_store,
                    conditional_store=conditional_store,
                    sample_plan=m4_plan,
                    generator=generator,
                    optimizer=optimizer,
                    scheduler_main=scheduler_main,
                    scheduler_mcp=scheduler_mcp,
                    train_rng=train_rng,
                    device=device,
                    dtype=dtype,
                    mode=args.mode,
                    validation_steps=validation_steps,
                    checkpoint_steps=checkpoint_steps,
                    checkpoint_interval=args.checkpoint_interval,
                    log_interval=args.log_interval,
                    timing_warmup_steps=args.timing_warmup_steps,
                )
            )
            cuda_synchronize_if_available(device)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            metric_record["elapsed_ms"] = elapsed_ms
            metric_record["timing"] = m5_timing_record(
                global_step=step,
                elapsed_ms=elapsed_ms,
                timing_warmup_steps=args.timing_warmup_steps,
            )
            step_timing_records.append(metric_record["timing"])

            if bool(metric_record["should_validate"]):
                report = run_m5_formal_validation_for_step(
                    generator=generator,
                    teacher_store=teacher_store,
                    conditional_store=conditional_store,
                    scheduler_main=scheduler_main,
                    scheduler_mcp=scheduler_mcp,
                    device=device,
                    dtype=dtype,
                    mode=args.mode,
                    global_step=step,
                    sample_plan=m4_plan,
                    validation_seed=int(args.validation_seed),
                    train_rng=train_rng,
                    probe=probe,
                    output_dir=args.output_dir,
                    current_git_sha=current_git_sha,
                    reference_checkpoint_sha256=reference_checkpoint_sha256,
                )
                handle_m5_formal_validation_report(
                    output_dir=args.output_dir,
                    metrics_path=metrics_path,
                    validation_reports=validation_reports,
                    report=report,
                    global_step=step,
                )
            write_m5_step_artifacts(
                output_dir=args.output_dir,
                metrics_path=metrics_path,
                generator=generator,
                optimizer=optimizer,
                global_step=step,
                metric_record=metric_record,
                should_log=bool(metric_record["should_log"]),
                should_checkpoint=bool(metric_record["should_checkpoint"]),
                train_rng=train_rng,
                probe=probe,
                sample_metadata=fixed_sample_metadata,
                resolved_config=resolved_config,
                current_git_sha=current_git_sha,
                reference_checkpoint_path=args.checkpoint,
                reference_checkpoint_sha256=reference_checkpoint_sha256,
                train_seed=args.train_seed,
                probe_seed=args.probe_seed,
                prompt_embedding=fixed_prompt_embedding,
                device=device,
                strict=True,
                mcp1_grid_aux_enabled=False,
                mcp1_grid_aux_scheduler=None,
                mcp1_grid_aux_timesteps=None,
                state=probe.noisy_batch.state,
                train_losses=train_losses,
                aux_report=aux_report,
                grad_audit=grad_audit,
                extra_checkpoint_payload_fields={
                    "m5_formal_trainer": dict(formal_checkpoint_metadata),
                },
            )

        summary = {
            "status": "PASS",
            "schema": M5_FORMAL_TRAINER_SCHEMA,
            "output_dir": str(args.output_dir.resolve()),
            "mode": args.mode,
            "optimizer_steps": int(args.optimizer_steps),
            "timing_warmup_steps": int(args.timing_warmup_steps),
            "sample_plan_sha256": saved_plan_sha,
            "conditional_artifact_sha256": conditional_artifact_sha256,
            "stage_contract": {
                "target_global_step": int(stage_contract.target_global_step),
                "parent_global_step": stage_contract.parent_global_step,
                "validation_steps": list(stage_contract.validation_steps),
                "checkpoint_steps": list(stage_contract.checkpoint_steps),
            },
            "validation_reports": [
                f"validation_step{int(report['global_step']):06d}.json"
                for report in validation_reports
            ],
            "next_train_sample_after_final_step": m4_next_train_entry_after_global_step(
                m4_plan,
                args.optimizer_steps,
            ),
            **summarize_m5_step_timing_records(step_timing_records),
            **cuda_peak_memory_summary(device),
        }
        write_run_json(
            summary,
            args.output_dir / "training_summary.json",
            strict=True,
        )
        print(json.dumps(summary, indent=2), flush=True)
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


def main() -> None:
    args = parse_args()
    args.mode = validate_m3_mode(args.mode)
    resume_enabled = args.resume_checkpoint is not None
    dtype = dtype_from_arg(args.dtype)
    device = require_single_gpu_runtime(torch, args.device)

    require_output_dir_empty_for_resume(args.output_dir)

    config = merge_config(str(args.config))
    validate_config(config, args)
    current_git_sha = git_head()
    reference_checkpoint_sha256 = file_sha256(args.checkpoint)
    if m5_formal_enabled(args):
        run_m5_formal_training(
            args=args,
            config=config,
            dtype=dtype,
            device=device,
            current_git_sha=current_git_sha,
            reference_checkpoint_sha256=reference_checkpoint_sha256,
        )
        return
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
    parent_resume_payload = None
    parent_resume_sha256 = None
    if resume_enabled:
        assert args.resume_checkpoint is not None
        parent_resume_payload, parent_resume_sha256 = load_parent_resume_checkpoint(
            args.resume_checkpoint
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if m4_enabled(args):
        assert m4_plan is not None
        write_m4_sample_plan(m4_plan, args.output_dir / "m4_sample_plan.json")
    write_run_json(
        sample.metadata,
        args.output_dir / "sample_metadata.json",
        strict=m4_enabled(args),
    )
    (args.output_dir / "reference_checkpoint_sha256.txt").write_text(
        reference_checkpoint_sha256 + "\n",
        encoding="utf-8",
    )
    if resume_enabled:
        assert parent_resume_sha256 is not None
        (args.output_dir / "parent_checkpoint_sha256.txt").write_text(
            parent_resume_sha256 + "\n",
            encoding="utf-8",
        )
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
    (args.output_dir / "git_sha.txt").write_text(current_git_sha + "\n", encoding="utf-8")

    generator = None
    text_encoder = None
    reset_global_seed(args.train_seed)
    train_rng = None
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
            if resume_enabled:
                conditional_dict = {}
            else:
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
        resumed_global_step = None
        resume_report = None
        if resume_enabled:
            assert parent_resume_payload is not None
            assert parent_resume_sha256 is not None
            assert args.resume_checkpoint is not None
            assert m4_plan is not None
            current_run_fields = current_m5_resume_run_fields(
                resolved_config=resolved_config,
                reference_checkpoint={
                    "path": args.checkpoint,
                    "sha256": reference_checkpoint_sha256,
                },
                selected_sample_metadata=sample.metadata,
                optimizer=optimizer,
                current_git_sha=current_git_sha,
                sample_plan=m4_plan,
            )
            resume_report = build_and_validate_m5_resume_report(
                parent_payload=parent_resume_payload,
                parent_checkpoint_path=args.resume_checkpoint,
                parent_checkpoint_sha256=parent_resume_sha256,
                current_run_fields=current_run_fields,
                target_global_step=int(args.optimizer_steps),
                sample_plan=m4_plan,
                output_dir=args.output_dir,
                target_validation_steps=m4_validation_steps,
                target_checkpoint_steps=m4_checkpoint_steps,
                expected_cuda_device_count=1 if device.type == "cuda" else None,
            )
            require_m5_resume_devices(
                resume_report,
                train_device=device,
                probe_device=device,
            )
            generator_restore = strict_load_m5_generator_state(
                generator,
                parent_resume_payload["generator"],
            )
            optimizer.load_state_dict(parent_resume_payload["optimizer"])
            optimizer_device_report = move_loaded_optimizer_state_to_device(
                optimizer,
                device=device,
            )
            rng_states = extract_resume_rng_states(parent_resume_payload)
            train_rng = restore_torch_generator_from_state(
                rng_states["train_generator_state"],
                device=device,
            )
            probe, restored_prompt_embedding = restore_m5_probe_from_checkpoint(
                parent_payload=parent_resume_payload,
                selected_state=state,
                device=device,
                dtype=dtype,
            )
            conditional_dict = restored_prompt_embedding
            restored_probe = run_m3_probe_forward(
                generator,
                conditional_dict=conditional_dict,
                noisy_batch=probe.noisy_batch,
            )
            probe_restore = require_restored_probe_matches_checkpoint(
                parent_payload=parent_resume_payload,
                restored_prompt_embedding=restored_prompt_embedding,
                probe_forward=restored_probe,
            )
            assert train_samples_by_identity is not None
            assert validation_samples_by_identity is not None
            conditional_restore = precompute_m5_resume_conditionals(
                text_encoder=text_encoder,
                sample_plan=m4_plan,
                train_samples_by_identity=train_samples_by_identity,
                validation_samples_by_identity=validation_samples_by_identity,
                fixed_decode_prompt_embedding=restored_prompt_embedding,
                conditional_cache=conditional_cache,
            )
            resume_report.update(
                {
                    "generator_restore": generator_restore,
                    "optimizer_device_restore": optimizer_device_report,
                    "probe_restore": probe_restore,
                    "conditional_cache_restore": conditional_restore,
                }
            )
            write_run_json(
                resume_report,
                args.output_dir / "resume_report.json",
                strict=True,
            )
            restore_global_rng_states(rng_states)
            resumed_global_step = int(parent_resume_payload["global_step"])
        else:
            train_rng = make_generator(args.train_seed, device)
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
                device=device,
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
        if m4_enabled(args) and not resume_enabled:
            assert m4_plan is not None
            assert validation_samples_by_identity is not None
            assert m4_validation_steps is not None
            assert train_rng is not None
            if 0 in m4_validation_steps:
                report = run_m5_m4_validation_for_step(
                    generator=generator,
                    text_encoder=text_encoder,
                    validation_samples_by_identity=validation_samples_by_identity,
                    conditional_cache=conditional_cache,
                    scheduler_main=scheduler_main,
                    scheduler_mcp=scheduler_mcp,
                    device=device,
                    dtype=dtype,
                    mode=args.mode,
                    global_step=0,
                    sample_plan=m4_plan,
                    validation_seed=int(args.validation_seed),
                    train_rng=train_rng,
                    probe=probe,
                    output_dir=args.output_dir,
                    current_git_sha=current_git_sha,
                    reference_checkpoint_sha256=reference_checkpoint_sha256,
                )
                handle_m4_validation_report(
                    output_dir=args.output_dir,
                    metrics_path=metrics_path,
                    validation_reports=validation_reports,
                    report=report,
                    global_step=0,
                )

        step_timing_records: list[dict[str, Any]] = []
        reset_peak_memory_stats_if_available(device)
        assert train_rng is not None
        for step in m5_absolute_training_steps(
            resumed_global_step=resumed_global_step,
            target_global_step=args.optimizer_steps,
        ):
            cuda_synchronize_if_available(device)
            started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            step_plan = m5_training_step_orchestration(
                global_step=step,
                target_global_step=args.optimizer_steps,
                sample_plan=m4_plan if m4_enabled(args) else None,
                validation_steps=m4_validation_steps,
                checkpoint_steps=m4_checkpoint_steps,
                checkpoint_interval=args.checkpoint_interval,
                log_interval=args.log_interval,
            )
            if m4_enabled(args):
                assert m4_plan is not None
                assert train_samples_by_identity is not None
                train_sample_identity = str(step_plan["train_sample_identity"])
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
            metric_step_record = m5_metrics_step_record(
                step_plan=step_plan,
                elapsed_ms=elapsed_ms,
                timing_warmup_steps=args.timing_warmup_steps,
            )
            step_timing_records.append(metric_step_record["timing"])
            train_losses = loss_dict_to_floats(result.losses)
            combined_objective = (
                train_losses["total_loss"]
                + aux_report["mcp1_grid_aux_weighted_loss"]
            )

            should_log = bool(step_plan["should_log"])
            should_checkpoint = bool(step_plan["should_checkpoint"])
            should_validate = bool(step_plan["should_validate"])
            metric_record = {
                **metric_step_record,
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
                        "train_sample_identity": step_plan["train_sample_identity"],
                        "train_sample_position": step_plan["train_sample_position"],
                        "train_cycle_index": step_plan["train_cycle_index"],
                        "should_validate": should_validate,
                        "should_checkpoint": should_checkpoint,
                    }
                )

            if m4_enabled(args):
                assert m4_plan is not None
                assert validation_samples_by_identity is not None
                assert m4_validation_steps is not None
                if should_validate:
                    report = run_m5_m4_validation_for_step(
                        generator=generator,
                        text_encoder=text_encoder,
                        validation_samples_by_identity=validation_samples_by_identity,
                        conditional_cache=conditional_cache,
                        scheduler_main=scheduler_main,
                        scheduler_mcp=scheduler_mcp,
                        device=device,
                        dtype=dtype,
                        mode=args.mode,
                        global_step=step,
                        sample_plan=m4_plan,
                        validation_seed=int(args.validation_seed),
                        train_rng=train_rng,
                        probe=probe,
                        output_dir=args.output_dir,
                        current_git_sha=current_git_sha,
                        reference_checkpoint_sha256=reference_checkpoint_sha256,
                    )
                    handle_m4_validation_report(
                        output_dir=args.output_dir,
                        metrics_path=metrics_path,
                        validation_reports=validation_reports,
                        report=report,
                        global_step=step,
                    )
            write_m5_step_artifacts(
                output_dir=args.output_dir,
                metrics_path=metrics_path,
                generator=generator,
                optimizer=optimizer,
                global_step=step,
                metric_record=metric_record,
                should_log=should_log,
                should_checkpoint=should_checkpoint,
                train_rng=train_rng,
                probe=probe,
                sample_metadata=sample.metadata,
                resolved_config=resolved_config,
                current_git_sha=current_git_sha,
                reference_checkpoint_path=args.checkpoint,
                reference_checkpoint_sha256=reference_checkpoint_sha256,
                train_seed=args.train_seed,
                probe_seed=args.probe_seed,
                prompt_embedding=conditional_dict,
                device=device,
                strict=m4_enabled(args),
                mcp1_grid_aux_enabled=mcp1_grid_aux_enabled,
                mcp1_grid_aux_scheduler=mcp1_grid_aux_scheduler,
                mcp1_grid_aux_timesteps=mcp1_grid_aux_timesteps,
                state=state,
                train_losses=train_losses,
                aux_report=aux_report,
                grad_audit=grad_audit,
            )

        summary = {
            "status": "PASS",
            "output_dir": str(args.output_dir.resolve()),
            "mode": args.mode,
            "optimizer_steps": int(args.optimizer_steps),
            "timing_warmup_steps": int(args.timing_warmup_steps),
            **summarize_m5_step_timing_records(step_timing_records),
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
