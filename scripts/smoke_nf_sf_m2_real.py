from __future__ import annotations

import argparse
import gc
import json
import random
import subprocess
import time
from pathlib import Path
from typing import Any

import torch

from inference_mcp import merge_config, require_single_gpu_runtime
from utils.checkpoint import (
    extract_generator_state_dict,
    is_mcp_state_key,
    load_state_dict_allowing_mcp_mismatch,
)
from utils.nf_sf_tensors import DEFAULT_S_MAIN, DEFAULT_S_MCP, make_generator
from utils.nf_sf_training import (
    NFSFSelectedState,
    collect_nf_sf_parameter_groups,
    configure_nf_sf_optimizer_plan,
    prepare_nf_sf_noisy_batch,
    run_nf_sf_forward_loss,
)
from utils.scheduler import FlowMatchScheduler
from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder


CHUNK_FRAMES = 3
DEPTHS = (1, 2, 3)
DEPTH_WEIGHTS = (0.5, 0.2, 0.1)
TAP_LAYERS = (3, 11, 19, 29)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="NF-SF v1 M2 real-Wan 3-step Flow Matching gradient smoke."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--mode", choices=("frozen", "joint"), required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--dtype", choices=("bf16", "float32"), default="bf16")
    return parser.parse_args()


def git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "UNKNOWN"


def reset_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def dtype_from_arg(value: str) -> torch.dtype:
    return torch.bfloat16 if value == "bf16" else torch.float32


def validate_config(config: Any, args: argparse.Namespace) -> None:
    if not Path(args.checkpoint).is_file():
        raise ValueError(f"--checkpoint must exist: {args.checkpoint}")
    if args.steps < 3:
        raise ValueError("M2 smoke must run at least 3 optimizer steps")
    if bool(getattr(config, "i2v", False)):
        raise ValueError("NF-SF M2 smoke supports T2V only")
    if int(getattr(config, "num_frame_per_block", 0)) != CHUNK_FRAMES:
        raise ValueError("NF-SF M2 real smoke requires chunk_frames=3")
    if int(getattr(config, "mcp_num_modules", 0)) != len(DEPTHS):
        raise ValueError("NF-SF M2 real smoke requires mcp_num_modules=3")
    if int(getattr(config, "mcp_num_layers", 0)) != 3:
        raise ValueError("NF-SF M2 real smoke requires mcp_num_layers=3")
    if tuple(int(x) for x in getattr(config, "mcp_tap_layers", ())) != TAP_LAYERS:
        raise ValueError("NF-SF M2 real smoke requires tap_layers=[3, 11, 19, 29]")
    if tuple(float(x) for x in getattr(config, "mcp_depth_weights", ())) != DEPTH_WEIGHTS:
        raise ValueError("NF-SF M2 real smoke requires depth_weights=[0.5, 0.2, 0.1]")
    model_kwargs = getattr(config, "model_kwargs", {})
    if float(model_kwargs.get("timestep_shift", DEFAULT_S_MAIN)) != DEFAULT_S_MAIN:
        raise ValueError("NF-SF M2 real smoke requires s_main=5.0")


def load_generator(config: Any, checkpoint_path: str) -> tuple[WanDiffusionWrapper, str, int]:
    model_kwargs = dict(getattr(config, "model_kwargs", {}))
    generator = WanDiffusionWrapper(**model_kwargs, is_causal=True)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = extract_generator_state_dict(checkpoint)
    checkpoint_has_mcp = any(is_mcp_state_key(key) for key in state_dict.keys())

    if checkpoint_has_mcp:
        generator.add_mcp_modules(
            num_modules=len(DEPTHS),
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
            num_modules=len(DEPTHS),
            num_layers=3,
            tap_layers=TAP_LAYERS,
        )
        load_mode = "BACKBONE_STRICT_THEN_INITIALIZE_MCP"

    mcp_tensor_count = sum(
        1 for key, value in generator.state_dict().items()
        if is_mcp_state_key(key) and torch.is_tensor(value)
    )
    if mcp_tensor_count <= 0:
        raise RuntimeError("MCP modules were not attached")
    return generator, load_mode, mcp_tensor_count


def make_mcp_scheduler(device: torch.device) -> FlowMatchScheduler:
    scheduler = FlowMatchScheduler(shift=DEFAULT_S_MCP, sigma_min=0.0, extra_one_step=True)
    scheduler.set_timesteps(1000, training=True)
    scheduler.sigmas = scheduler.sigmas.to(device)
    scheduler.timesteps = scheduler.timesteps.to(device)
    return scheduler


def make_selected_state(
    *,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
) -> NFSFSelectedState:
    rng = make_generator(seed, device)
    chunks = torch.randn(
        (1, (1 + 1 + len(DEPTHS)) * CHUNK_FRAMES, 16, 60, 104),
        device=device,
        dtype=dtype,
        generator=rng,
    )
    history = chunks[:, 0:CHUNK_FRAMES]
    current = chunks[:, CHUNK_FRAMES:2 * CHUNK_FRAMES]
    futures = tuple(
        chunks[:, (2 + index) * CHUNK_FRAMES:(3 + index) * CHUNK_FRAMES]
        for index in range(len(DEPTHS))
    )
    return NFSFSelectedState(
        clean_history=history,
        current_target=current,
        future_targets=futures,
        current_start_frame=CHUNK_FRAMES,
    )


def clone_parameter_groups(generator) -> dict[str, list[torch.Tensor]]:
    return {
        name: [param.detach().clone() for _, param in params]
        for name, params in collect_nf_sf_parameter_groups(generator).items()
    }


def changed_groups(generator, before: dict[str, list[torch.Tensor]]) -> dict[str, bool]:
    result = {}
    groups = collect_nf_sf_parameter_groups(generator)
    for name, params in groups.items():
        result[name] = any(
            not torch.equal(param.detach(), previous)
            for (_, param), previous in zip(params, before[name])
        )
    return result


def grad_norms(generator) -> dict[str, float]:
    norms = {}
    for name, params in collect_nf_sf_parameter_groups(generator).items():
        total = 0.0
        for _, param in params:
            if param.grad is None:
                continue
            grad = param.grad.detach().float()
            total += float(grad.square().sum().item())
        norms[name] = total ** 0.5
    return norms


def has_nonfinite_grad(generator) -> bool:
    for _, param in generator.named_parameters():
        if param.grad is not None and not bool(torch.isfinite(param.grad).all().item()):
            return True
    return False


def audit_to_json(audit) -> dict[str, Any]:
    return {
        "name": audit.name,
        "parameter_names": list(audit.parameter_names),
        "tensor_count": audit.tensor_count,
        "trainable_parameter_count": audit.trainable_parameter_count,
        "requires_grad": audit.requires_grad,
        "in_optimizer": audit.in_optimizer,
    }


def validate_parameter_changes(mode: str, changes: dict[str, bool]) -> dict[str, bool]:
    required_changed = {"mcp_fusion", "mcp_depth1", "mcp_depth2", "mcp_depth3"}
    required_unchanged = {"backbone", "patch_embedding"}
    if mode == "joint":
        required_changed.update({"backbone", "patch_embedding"})
        required_unchanged.clear()

    failures = []
    for name in sorted(required_changed):
        if not changes.get(name, False):
            failures.append(f"{name} did not change")
    for name in sorted(required_unchanged):
        if changes.get(name, False):
            failures.append(f"{name} changed unexpectedly")
    if failures:
        raise RuntimeError("; ".join(failures))
    return {
        "required_changed_pass": True,
        "required_unchanged_pass": True,
    }


def main() -> None:
    args = parse_args()
    device = require_single_gpu_runtime(torch, args.device)
    torch.cuda.reset_peak_memory_stats(device)
    reset_seed(args.seed)
    dtype = dtype_from_arg(args.dtype)
    config = merge_config(args.config)
    validate_config(config, args)

    generator = None
    text_encoder = None
    report: dict[str, Any] = {
        "git_head": git_head(),
        "mode": args.mode,
        "steps_requested": args.steps,
        "device": str(device),
        "dtype": str(dtype),
        "s_main": DEFAULT_S_MAIN,
        "s_mcp": DEFAULT_S_MCP,
        "depth_weights": list(DEPTH_WEIGHTS),
    }
    try:
        generator, load_mode, mcp_tensor_count = load_generator(config, args.checkpoint)
        generator.to(device=device, dtype=dtype)
        generator.train()

        text_encoder = WanTextEncoder().to(device=device, dtype=dtype).eval().requires_grad_(False)
        with torch.no_grad():
            conditional_dict = text_encoder([args.prompt])

        scheduler_main = generator.get_scheduler()
        scheduler_main.sigmas = scheduler_main.sigmas.to(device)
        scheduler_main.timesteps = scheduler_main.timesteps.to(device)
        scheduler_mcp = make_mcp_scheduler(device)

        state = make_selected_state(device=device, dtype=dtype, seed=args.seed + 17)
        plan = configure_nf_sf_optimizer_plan(generator, mode=args.mode, lr=args.lr)
        optimizer = torch.optim.AdamW(
            plan.optimizer_param_groups,
            lr=args.lr,
            betas=(0.0, 0.999),
            weight_decay=args.weight_decay,
        )
        before = clone_parameter_groups(generator)

        step_reports = []
        for step in range(1, args.steps + 1):
            step_start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            rng = make_generator(args.seed + 1000 + step, device)
            noisy_batch = prepare_nf_sf_noisy_batch(
                state,
                scheduler_main=scheduler_main,
                scheduler_mcp=scheduler_mcp,
                rng=rng,
                chunk_frames=CHUNK_FRAMES,
                depths=DEPTHS,
                s_main=DEFAULT_S_MAIN,
                s_mcp=DEFAULT_S_MCP,
            )
            result = run_nf_sf_forward_loss(
                generator,
                conditional_dict=conditional_dict,
                noisy_batch=noisy_batch,
                depth_weights=DEPTH_WEIGHTS,
            )
            result.losses.total_loss.backward()
            nonfinite_grad = has_nonfinite_grad(generator)
            optimizer.step()
            torch.cuda.synchronize(device)
            elapsed_ms = (time.perf_counter() - step_start) * 1000.0
            if nonfinite_grad:
                raise RuntimeError(f"non-finite gradient at step {step}")
            losses = result.losses
            step_reports.append(
                {
                    "step": step,
                    "total_loss": float(losses.total_loss.detach().float().item()),
                    "main_loss": float(losses.main_loss.detach().float().item()),
                    "mcp_depth1_loss": float(losses.mcp_depth_losses[0].detach().float().item()),
                    "mcp_depth2_loss": float(losses.mcp_depth_losses[1].detach().float().item()),
                    "mcp_depth3_loss": float(losses.mcp_depth_losses[2].detach().float().item()),
                    "grad_norms": grad_norms(generator),
                    "nonfinite_grad": nonfinite_grad,
                    "peak_memory_gib": torch.cuda.max_memory_allocated(device) / (1024 ** 3),
                    "elapsed_ms": elapsed_ms,
                    "main_timestep_unique": sorted(
                        float(x) for x in noisy_batch.timestep_main.detach().flatten().unique().cpu()
                    ),
                    "mcp_timestep_unique": [
                        sorted(float(x) for x in t.detach().flatten().unique().cpu())
                        for t in noisy_batch.timestep_depths
                    ],
                }
            )

        changes = changed_groups(generator, before)
        change_validation = validate_parameter_changes(args.mode, changes)
        report.update(
            {
                "checkpoint_load_mode": load_mode,
                "mcp_tensor_count": mcp_tensor_count,
                "param_audit": [audit_to_json(audit) for audit in plan.audits],
                "step_reports": step_reports,
                "parameter_changed": changes,
                "change_validation": change_validation,
                "exit_code": 0,
            }
        )
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2), flush=True)
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
