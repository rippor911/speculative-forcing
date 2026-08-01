from __future__ import annotations

import argparse
import gc
import json
import random
import subprocess
import sys
import time
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
    validate_git_sha,
    validate_m3_mode,
)
from utils.nf_sf_tensors import DEFAULT_S_MAIN, DEFAULT_S_MCP, make_generator
from utils.nf_sf_training import (
    configure_nf_sf_optimizer_plan,
    prepare_nf_sf_noisy_batch,
    run_nf_sf_forward_loss,
)
from utils.scheduler import FlowMatchScheduler


TAP_LAYERS = (3, 11, 19, 29)
ADAMW_BETAS = (0.0, 0.999)
ADAMW_EPS = 1.0e-8


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
    parser.add_argument("--mode", default="joint")
    parser.add_argument("--train_seed", type=int, required=True)
    parser.add_argument("--probe_seed", type=int, required=True)
    parser.add_argument("--optimizer_steps", type=int, required=True)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--checkpoint_interval", type=int, default=10)
    parser.add_argument("--backbone_lr", type=float, required=True)
    parser.add_argument("--patch_embedding_lr", type=float, required=True)
    parser.add_argument("--mcp_lr", type=float, required=True)
    parser.add_argument("--weight_decay", type=float, required=True)
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


def resolved_config_dict(config: Any) -> dict[str, Any]:
    from omegaconf import OmegaConf

    return OmegaConf.to_container(config, resolve=True)


def validate_config(config: Any, args: argparse.Namespace) -> None:
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if args.optimizer_steps <= 0:
        raise ValueError("--optimizer_steps must be positive")
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
) -> dict[str, Any]:
    report = {
        "step": int(step),
        **prefix_metrics("probe", losses),
        "probe_losses": losses,
        "probe_output_summaries": probe_output_summaries(outputs),
    }
    atomic_json_write(report, output_dir / f"probe_step{step:06d}.json")
    return report


def append_metrics(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


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
    validate_m3_mode(args.mode)
    dtype = dtype_from_arg(args.dtype)
    device = require_single_gpu_runtime(torch, args.device)
    reset_global_seed(args.train_seed)

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"--output_dir must be empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    reference_checkpoint_sha256 = file_sha256(args.checkpoint)
    sample = load_m3_teacher_sample(
        manifest_path=args.manifest,
        dataset_root=args.dataset_root,
        sample_index=args.sample_index,
        sample_id=args.sample_id,
        split=args.split,
        split_index=args.split_index,
        reference_checkpoint_path=args.checkpoint,
    )
    atomic_json_write(sample.metadata, args.output_dir / "sample_metadata.json")
    (args.output_dir / "reference_checkpoint_sha256.txt").write_text(
        reference_checkpoint_sha256 + "\n",
        encoding="utf-8",
    )

    config = merge_config(str(args.config))
    validate_config(config, args)
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
            "log_interval": args.log_interval,
            "checkpoint_interval": args.checkpoint_interval,
            "backbone_lr": args.backbone_lr,
            "patch_embedding_lr": args.patch_embedding_lr,
            "mcp_lr": args.mcp_lr,
            "weight_decay": args.weight_decay,
            "optimizer_config": optimizer_config,
            "dtype": args.dtype,
            "device": str(device),
        },
    }
    atomic_json_write(resolved_config, args.output_dir / "resolved_config.json")
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
        with torch.no_grad():
            conditional_dict = text_encoder([sample.metadata["prompt"]])

        state = selected_state_to_device(sample.selected_state, device=device, dtype=dtype)

        scheduler_main = generator.get_scheduler()
        scheduler_main.sigmas = scheduler_main.sigmas.to(device)
        scheduler_main.timesteps = scheduler_main.timesteps.to(device)
        scheduler_mcp = make_mcp_scheduler(device)

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
        initial_probe_report = write_probe_report(
            args.output_dir,
            0,
            initial_probe.losses,
            initial_probe.outputs,
        )

        group_lrs = {
            "backbone": args.backbone_lr,
            "patch_embedding": args.patch_embedding_lr,
            "mcp": args.mcp_lr,
        }
        plan = configure_nf_sf_optimizer_plan(
            generator,
            mode="joint",
            group_lrs=group_lrs,
        )
        optimizer = torch.optim.AdamW(
            plan.optimizer_param_groups,
            betas=ADAMW_BETAS,
            eps=ADAMW_EPS,
            weight_decay=args.weight_decay,
        )
        atomic_json_write(
            {
                "mode": plan.mode,
                "optimizer_config": optimizer_config_summary(optimizer),
                "param_audit": [audit_to_json(audit) for audit in plan.audits],
                "optimizer_group_lrs": optimizer_group_lr_summary(optimizer),
                "checkpoint_load_mode": load_mode,
                "mcp_tensor_count": mcp_tensor_count,
            },
            args.output_dir / "optimizer_audit.json",
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
        append_metrics(
            metrics_path,
            {
                "step": 0,
                "elapsed_ms": 0.0,
                **prefix_metrics("probe", initial_probe.losses),
            },
        )

        for step in range(1, args.optimizer_steps + 1):
            started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            noisy_batch = prepare_nf_sf_noisy_batch(
                state,
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
                conditional_dict=conditional_dict,
                noisy_batch=noisy_batch,
                depth_weights=M3_DEPTH_WEIGHTS,
            )
            result.losses.total_loss.backward()
            grad_audit = gradient_group_audit(optimizer)
            if not all(entry["finite"] for entry in grad_audit.values()):
                raise RuntimeError(f"non-finite gradient audit at step {step}")
            if has_nonfinite_grad(generator):
                raise RuntimeError(f"non-finite gradient at step {step}")
            optimizer.step()
            torch.cuda.synchronize(device)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            train_losses = loss_dict_to_floats(result.losses)

            should_log = step % args.log_interval == 0 or step == args.optimizer_steps
            should_checkpoint = (
                step % args.checkpoint_interval == 0
                or step == args.optimizer_steps
            )
            metric_record = {
                "step": step,
                "elapsed_ms": elapsed_ms,
                **prefix_metrics("train", train_losses),
                "grad_audit": grad_audit,
            }
            if should_log or should_checkpoint:
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
                )
                metric_record.update(prefix_metrics("probe", probe_forward.losses))
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
                append_metrics(metrics_path, metric_record)
                print(
                    json.dumps(
                        {
                            "step": step,
                            **prefix_metrics("train", train_losses),
                            **prefix_metrics("probe", probe_forward.losses),
                            "grad_audit": grad_audit,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
            else:
                append_metrics(metrics_path, metric_record)

        print(
            json.dumps(
                {
                    "status": "PASS",
                    "output_dir": str(args.output_dir.resolve()),
                    "optimizer_steps": args.optimizer_steps,
                },
                indent=2,
            ),
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
