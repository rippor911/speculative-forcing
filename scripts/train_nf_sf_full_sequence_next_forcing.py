from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
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

from utils.checkpoint import extract_generator_state_dict, is_mcp_state_key
from utils.nf_sf_m3 import file_sha256, move_tensors_to_device
from utils.nf_sf_m4 import (
    derive_m4_validation_seed,
    load_m4_sample_plan,
    write_m4_json,
)
from utils.nf_sf_m5_conditionals import M5ConditionalArtifactStore
from utils.nf_sf_m5_samples import M5TeacherSampleStore
from utils.nf_sf_tensors import (
    DEFAULT_S_MAIN,
    DEFAULT_S_MCP,
    FULL_SEQUENCE_CHUNK_FRAMES,
    FULL_SEQUENCE_DEPTHS,
    FULL_SEQUENCE_FRAME_COUNT,
    FULL_SEQUENCE_NUM_CHUNKS,
    FULL_SEQUENCE_RNG_DRAW_ORDER_VERSION,
    make_generator,
)
from utils.nf_sf_training import (
    FULL_SEQUENCE_CHECKPOINT_STEPS,
    FULL_SEQUENCE_CHUNK_TOKENS,
    FULL_SEQUENCE_DEPTH_WEIGHTS,
    FULL_SEQUENCE_FRAME_SEQ_LENGTH,
    FULL_SEQUENCE_OBJECTIVE_VERSION,
    FULL_SEQUENCE_RUN_KIND,
    FULL_SEQUENCE_TARGET_GLOBAL_STEP,
    FULL_SEQUENCE_TRAINER_SCHEMA,
    OFFICIAL_SELF_FORCING_CHECKPOINT_SHA256,
    audit_nf_sf_full_sequence_gradients,
    build_nf_sf_full_sequence_provenance,
    configure_nf_sf_full_sequence_optimizer_plan,
    nf_sf_full_sequence_train_cursor,
    prepare_nf_sf_full_sequence_noisy_batch,
    require_nf_sf_full_sequence_runtime,
    run_nf_sf_full_sequence_forward_loss,
    validate_nf_sf_full_sequence_objective_mode,
)
from utils.scheduler import FlowMatchScheduler
from utils.wan_wrapper import WanDiffusionWrapper


TAP_LAYERS = (3, 11, 19, 29)
MCP_NUM_MODULES = 3
MCP_NUM_LAYERS = 3
ADAMW_BETAS = (0.0, 0.999)
ADAMW_EPS = 1.0e-8
TRAIN_SAMPLE_COUNT = 2048
VALIDATION_SAMPLE_COUNT = 256
NON_PRODUCTION_SMOKE_TAG = "NON_PRODUCTION_SMOKE"
CHECKPOINT_VALIDATION_SCHEMA = "nf_sf_full_sequence_checkpoint_validation_v1"
VALIDATION_TENSOR_SLOT = "nf_sf_full_sequence_next_forcing_v1"
CANONICAL_CONFIG_PATH = (ROOT / "configs" / "self_forcing_dmd_mcp.yaml").resolve()
SMOKE_CLEANUP_TOLERANCE_BYTES = 2 * 1024**3
MCP_ZERO_HEAD_BOOTSTRAP_REASON = "zero_initialized_mcp_output_heads"


def atomic_torch_save(payload: Mapping[str, Any], path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp"
    )
    try:
        with temporary.open("wb") as handle:
            torch.save(dict(payload), handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="NF-SF Full-Sequence Next-Forcing v1 trainer."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/self_forcing_dmd_mcp.yaml"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/self_forcing_dmd.pt"),
    )
    parser.add_argument("--resume_checkpoint", type=Path, default=None)
    parser.add_argument("--expected_git_sha", required=True)
    parser.add_argument("--expected_sample_plan_sha256", required=True)
    parser.add_argument("--expected_manifest_sha256", required=True)
    parser.add_argument("--expected_conditionals_artifact_sha256", required=True)
    parser.add_argument("--expected_resume_checkpoint_sha256", default=None)
    parser.add_argument("--sample_plan", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--dataset_root", required=True, type=Path)
    parser.add_argument("--conditionals_artifact", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument(
        "--objective_mode",
        choices=("next_forcing_full", "main_only_full_control"),
        default="next_forcing_full",
    )
    parser.add_argument("--engineering_smoke_one_step", action="store_true")
    parser.add_argument("--train_seed", type=int, required=True)
    parser.add_argument("--validation_seed", type=int, required=True)
    parser.add_argument("--global_seed", type=int, required=True)
    parser.add_argument("--backbone_lr", type=float, required=True)
    parser.add_argument("--patch_embedding_lr", type=float, required=True)
    parser.add_argument("--mcp_lr", type=float, required=True)
    parser.add_argument("--weight_decay", type=float, required=True)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--memory_log_interval", type=int, default=100)
    parser.add_argument("--dtype", choices=("bf16", "float32"), default="bf16")
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args(argv)


def merge_config(config_path: Path | str):
    from omegaconf import OmegaConf

    default_config = OmegaConf.load(ROOT / "configs" / "default_config.yaml")
    run_config = OmegaConf.load(config_path)
    return OmegaConf.merge(default_config, run_config)


def git_head() -> str:
    value = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        stderr=subprocess.DEVNULL,
        text=True,
    ).strip()
    if len(value) != 40 or any(char not in "0123456789abcdef" for char in value):
        raise RuntimeError(f"invalid git HEAD: {value!r}")
    return value


def git_output(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=ROOT,
        stderr=subprocess.DEVNULL,
        text=True,
    ).strip()


def validate_sha256(value: str, *, name: str) -> str:
    text = str(value).strip().lower()
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise ValueError(f"{name} must be a lowercase SHA256 hex string")
    return text


def validate_git_sha(value: str, *, name: str) -> str:
    text = str(value).strip().lower()
    if len(text) != 40 or any(char not in "0123456789abcdef" for char in text):
        raise ValueError(f"{name} must be a lowercase git SHA")
    return text


def path_is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def collect_repo_preflight_facts(
    *,
    args: argparse.Namespace,
    sample_plan: Mapping[str, Any],
    conditional_store: M5ConditionalArtifactStore,
) -> dict[str, Any]:
    return {
        "git_top_level": str(Path(git_output("rev-parse", "--show-toplevel")).resolve()),
        "root": str(ROOT.resolve()),
        "current_git_sha": git_head(),
        "expected_git_sha": validate_git_sha(
            args.expected_git_sha,
            name="--expected_git_sha",
        ),
        "tracked_dirty": bool(git_output("diff", "--name-only")),
        "staged_dirty": bool(git_output("diff", "--cached", "--name-only")),
        "output_dir": str(args.output_dir.resolve()),
        "config_path": str(args.config.resolve()),
        "checkpoint_path": str(args.checkpoint.resolve()),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "checkpoint_size_bytes": int(args.checkpoint.stat().st_size),
        "sample_plan_path": str(args.sample_plan.resolve()),
        "sample_plan_sha256": str(sample_plan["sample_plan_sha256"]),
        "expected_sample_plan_sha256": validate_sha256(
            args.expected_sample_plan_sha256,
            name="--expected_sample_plan_sha256",
        ),
        "manifest_path": str(args.manifest.resolve()),
        "manifest_sha256": file_sha256(args.manifest),
        "expected_manifest_sha256": validate_sha256(
            args.expected_manifest_sha256,
            name="--expected_manifest_sha256",
        ),
        "conditionals_artifact_path": str(args.conditionals_artifact.resolve()),
        "conditionals_artifact_sha256": str(conditional_store.artifact_sha256),
        "expected_conditionals_artifact_sha256": validate_sha256(
            args.expected_conditionals_artifact_sha256,
            name="--expected_conditionals_artifact_sha256",
        ),
        "device": str(args.device),
        "dtype": str(args.dtype),
        "cuda_available": bool(torch.cuda.is_available()),
    }


def validate_repo_preflight_facts(facts: Mapping[str, Any]) -> dict[str, Any]:
    root = Path(str(facts["root"])).resolve()
    git_top_level = Path(str(facts["git_top_level"])).resolve()
    if git_top_level != root:
        raise RuntimeError("git top-level does not match trainer ROOT")
    current_git_sha = validate_git_sha(
        str(facts["current_git_sha"]),
        name="current_git_sha",
    )
    expected_git_sha = validate_git_sha(
        str(facts["expected_git_sha"]),
        name="expected_git_sha",
    )
    if current_git_sha != expected_git_sha:
        raise RuntimeError("git HEAD does not match --expected_git_sha")
    if bool(facts["tracked_dirty"]):
        raise RuntimeError("tracked worktree must be clean before real run")
    if bool(facts["staged_dirty"]):
        raise RuntimeError("staged index must be clean before real run")
    if path_is_within(Path(str(facts["output_dir"])), root):
        raise RuntimeError("output_dir must resolve outside the repository")
    if Path(str(facts["config_path"])).resolve() != CANONICAL_CONFIG_PATH:
        raise RuntimeError("config path must be configs/self_forcing_dmd_mcp.yaml")
    if str(facts["device"]) != "cuda:0":
        raise RuntimeError("full-sequence real trainer requires --device cuda:0")
    if str(facts["dtype"]) != "bf16":
        raise RuntimeError("full-sequence real trainer requires --dtype bf16")
    if not bool(facts["cuda_available"]):
        raise RuntimeError("CUDA must be available for real trainer execution")
    if str(facts["checkpoint_sha256"]).lower() != OFFICIAL_SELF_FORCING_CHECKPOINT_SHA256:
        raise RuntimeError("official Self-Forcing checkpoint SHA256 mismatch")
    for key in ("sample_plan", "manifest", "conditionals_artifact"):
        actual = validate_sha256(str(facts[f"{key}_sha256"]), name=f"{key}_sha256")
        expected = validate_sha256(
            str(facts[f"expected_{key}_sha256"]),
            name=f"expected_{key}_sha256",
        )
        if actual != expected:
            raise RuntimeError(f"{key} SHA256 does not match expected value")
    return {
        "status": "PASS",
        "current_git_sha": current_git_sha,
        "expected_git_sha": expected_git_sha,
        "sample_plan_sha256": str(facts["sample_plan_sha256"]),
        "manifest_sha256": str(facts["manifest_sha256"]).lower(),
        "conditionals_artifact_sha256": str(
            facts["conditionals_artifact_sha256"]
        ).lower(),
        "checkpoint_sha256": str(facts["checkpoint_sha256"]).lower(),
        "checkpoint_size_bytes": int(facts["checkpoint_size_bytes"]),
    }


def validate_store_identity_order(
    *,
    sample_plan: Mapping[str, Any],
    teacher_store: M5TeacherSampleStore,
    conditional_store: M5ConditionalArtifactStore,
) -> None:
    train_ids = tuple(str(value) for value in sample_plan["train_sample_identities"])
    validation_ids = tuple(
        str(value) for value in sample_plan["validation_sample_identities"]
    )
    if teacher_store.train_identities != train_ids:
        raise RuntimeError("teacher store train identity order differs from sample plan")
    if teacher_store.validation_identities != validation_ids:
        raise RuntimeError(
            "teacher store validation identity order differs from sample plan"
        )
    if conditional_store.train_identities != train_ids:
        raise RuntimeError(
            "conditional artifact train identity order differs from sample plan"
        )
    if conditional_store.validation_identities != validation_ids:
        raise RuntimeError(
            "conditional artifact validation identity order differs from sample plan"
        )


def reset_global_seed(seed: int) -> None:
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.benchmark = False


def dtype_from_arg(value: str) -> torch.dtype:
    return torch.bfloat16 if value == "bf16" else torch.float32


def make_flow_scheduler(shift: float) -> FlowMatchScheduler:
    scheduler = FlowMatchScheduler(shift=shift, sigma_min=0.0, extra_one_step=True)
    scheduler.set_timesteps(1000, training=True)
    return scheduler


def validate_cli_contract(args: argparse.Namespace, config: Any) -> dict[str, Any]:
    objective_mode = validate_nf_sf_full_sequence_objective_mode(args.objective_mode)
    if args.resume_checkpoint is not None and args.expected_resume_checkpoint_sha256 is None:
        raise ValueError("--expected_resume_checkpoint_sha256 is required for resume")
    if args.resume_checkpoint is None and args.expected_resume_checkpoint_sha256 is not None:
        raise ValueError("--expected_resume_checkpoint_sha256 requires --resume_checkpoint")
    if str(args.device) != "cuda:0":
        raise ValueError("full-sequence real trainer requires --device cuda:0")
    if str(args.dtype) != "bf16":
        raise ValueError("full-sequence real trainer requires --dtype bf16")
    for name in ("train_seed", "validation_seed", "global_seed"):
        value = getattr(args, name)
        if not isinstance(value, int):
            raise ValueError(f"--{name} must be a Python int")
    for name in ("backbone_lr", "patch_embedding_lr", "mcp_lr"):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"--{name} must be positive and finite")
    weight_decay = float(args.weight_decay)
    if not math.isfinite(weight_decay) or weight_decay < 0.0:
        raise ValueError("--weight_decay must be non-negative and finite")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if args.config.resolve() != CANONICAL_CONFIG_PATH:
        raise RuntimeError("config path must be configs/self_forcing_dmd_mcp.yaml")
    if int(getattr(config, "num_frame_per_block", 0)) != FULL_SEQUENCE_CHUNK_FRAMES:
        raise RuntimeError("config.num_frame_per_block must be 3")
    if not bool(getattr(config, "gradient_checkpointing", False)):
        raise RuntimeError("config.gradient_checkpointing must be true")
    if bool(getattr(config, "i2v", False)):
        raise RuntimeError("full-sequence v1 trainer supports T2V only")
    if int(args.log_interval) <= 0:
        raise ValueError("--log_interval must be positive")
    if int(args.memory_log_interval) <= 0:
        raise ValueError("--memory_log_interval must be positive")
    if args.engineering_smoke_one_step and args.resume_checkpoint is not None:
        raise ValueError("engineering smoke must start fresh")
    return {
        "objective_mode": objective_mode,
        "target_global_step": (
            1
            if args.engineering_smoke_one_step
            else FULL_SEQUENCE_TARGET_GLOBAL_STEP
        ),
        "checkpoint_steps": (
            ()
            if args.engineering_smoke_one_step
            else FULL_SEQUENCE_CHECKPOINT_STEPS
        ),
        "validation_steps": (
            ()
            if args.engineering_smoke_one_step
            else FULL_SEQUENCE_CHECKPOINT_STEPS
        ),
    }


def validate_sample_plan_contract(sample_plan: Mapping[str, Any]) -> None:
    train_ids = tuple(str(value) for value in sample_plan["train_sample_identities"])
    validation_ids = tuple(
        str(value) for value in sample_plan["validation_sample_identities"]
    )
    if len(train_ids) != TRAIN_SAMPLE_COUNT:
        raise RuntimeError("full-sequence train split must contain 2048 identities")
    if len(validation_ids) != VALIDATION_SAMPLE_COUNT:
        raise RuntimeError("full-sequence validation split must contain 256 identities")
    if set(train_ids) & set(validation_ids):
        raise RuntimeError("train and validation identities must be disjoint")


def build_fresh_generator(
    *,
    config: Any,
    checkpoint_path: Path,
    device: torch.device,
    dtype: torch.dtype,
) -> WanDiffusionWrapper:
    model_kwargs = dict(getattr(config, "model_kwargs", {}) or {})
    reset_before = bool(model_kwargs.pop("reset_before_init", False))
    if reset_before:
        raise RuntimeError("reset_before_init is not supported by this trainer")
    generator = WanDiffusionWrapper(
        **model_kwargs,
        is_causal=True,
        local_attn_size=int(getattr(config, "local_attn_size", -1)),
        sink_size=int(getattr(config, "sink_size", 0)),
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = extract_generator_state_dict(checkpoint)
    if any(is_mcp_state_key(key) for key in state_dict.keys()):
        raise RuntimeError("fresh full-sequence run must not load an MCP checkpoint")
    missing, unexpected = generator.load_state_dict(state_dict, strict=True)
    if missing or unexpected:
        raise RuntimeError(
            "official Self-Forcing backbone did not restore strictly: "
            f"missing={missing}, unexpected={unexpected}"
        )
    generator.add_mcp_modules(
        num_modules=MCP_NUM_MODULES,
        num_layers=MCP_NUM_LAYERS,
        tap_layers=TAP_LAYERS,
    )
    if not bool(getattr(generator, "mcp_initialized_from_backbone", False)):
        raise RuntimeError("MCP init_from_backbone flag was not set")
    generator.model.num_frame_per_block = FULL_SEQUENCE_CHUNK_FRAMES
    generator.enable_gradient_checkpointing()
    generator.to(device=device, dtype=dtype)
    generator.train()
    require_nf_sf_full_sequence_runtime(
        config=config,
        generator=generator,
        objective_mode="next_forcing_full",
    )
    return generator


def build_optimizer(
    generator: WanDiffusionWrapper,
    *,
    objective_mode: str,
    backbone_lr: float,
    patch_embedding_lr: float,
    mcp_lr: float,
    weight_decay: float,
) -> tuple[torch.optim.AdamW, dict[str, Any]]:
    plan = configure_nf_sf_full_sequence_optimizer_plan(
        generator,
        objective_mode=objective_mode,
        group_lrs={
            "backbone": float(backbone_lr),
            "patch_embedding": float(patch_embedding_lr),
            "mcp": float(mcp_lr),
            "mcp_fusion": float(mcp_lr),
        },
    )
    optimizer = torch.optim.AdamW(
        plan.optimizer_param_groups,
        betas=ADAMW_BETAS,
        eps=ADAMW_EPS,
        weight_decay=float(weight_decay),
    )
    return optimizer, optimizer_plan_summary(plan)


def optimizer_plan_summary(plan) -> dict[str, Any]:
    return {
        "mode": plan.mode,
        "groups": [
            {
                "name": audit.name,
                "tensor_count": audit.tensor_count,
                "trainable_parameter_count": audit.trainable_parameter_count,
                "requires_grad": audit.requires_grad,
                "in_optimizer": audit.in_optimizer,
            }
            for audit in plan.audits
        ],
    }


def full_sequence_resolved_config(
    config: Any,
    args: argparse.Namespace,
    *,
    preflight_report: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": FULL_SEQUENCE_TRAINER_SCHEMA,
        "run_kind": FULL_SEQUENCE_RUN_KIND,
        "objective_version": FULL_SEQUENCE_OBJECTIVE_VERSION,
        "objective_mode": args.objective_mode,
        "expected_git_sha": str(preflight_report["expected_git_sha"]),
        "current_git_sha": str(preflight_report["current_git_sha"]),
        "config_path": str(args.config.resolve()),
        "checkpoint_path": str(args.checkpoint.resolve()),
        "checkpoint_sha256": str(preflight_report["checkpoint_sha256"]),
        "checkpoint_size_bytes": int(preflight_report["checkpoint_size_bytes"]),
        "sample_plan_path": str(args.sample_plan.resolve()),
        "sample_plan_sha256": str(preflight_report["sample_plan_sha256"]),
        "manifest_path": str(args.manifest.resolve()),
        "manifest_sha256": str(preflight_report["manifest_sha256"]),
        "dataset_root": str(args.dataset_root.resolve()),
        "conditionals_artifact_path": str(args.conditionals_artifact.resolve()),
        "conditionals_artifact_sha256": str(
            preflight_report["conditionals_artifact_sha256"]
        ),
        "train_seed": int(args.train_seed),
        "validation_seed": int(args.validation_seed),
        "global_seed": int(args.global_seed),
        "backbone_lr": float(args.backbone_lr),
        "patch_embedding_lr": float(args.patch_embedding_lr),
        "mcp_lr": float(args.mcp_lr),
        "weight_decay": float(args.weight_decay),
        "adam_betas": [float(value) for value in ADAMW_BETAS],
        "adam_eps": float(ADAMW_EPS),
        "dtype": str(args.dtype),
        "device": str(args.device),
        "num_frame_per_block": int(getattr(config, "num_frame_per_block", 0)),
        "gradient_checkpointing": bool(
            getattr(config, "gradient_checkpointing", False)
        ),
        "full_teacher_frames": FULL_SEQUENCE_FRAME_COUNT,
        "chunk_frames": FULL_SEQUENCE_CHUNK_FRAMES,
        "num_chunks": FULL_SEQUENCE_NUM_CHUNKS,
        "main_shift": DEFAULT_S_MAIN,
        "mcp_shift": DEFAULT_S_MCP,
        "depth_weights": list(FULL_SEQUENCE_DEPTH_WEIGHTS),
        "tap_layers": list(TAP_LAYERS),
        "mcp_blocks_per_depth": MCP_NUM_LAYERS,
        "rng_draw_order_version": FULL_SEQUENCE_RNG_DRAW_ORDER_VERSION,
        "validation_tensor_slot": VALIDATION_TENSOR_SLOT,
        "validation_seed_derivation": "derive_m4_validation_seed",
        "validation_identity_noise_is_paired_across_steps": True,
        "production_target_global_step": FULL_SEQUENCE_TARGET_GLOBAL_STEP,
        "production_checkpoint_steps": list(FULL_SEQUENCE_CHECKPOINT_STEPS),
        "production_validation_steps": list(FULL_SEQUENCE_CHECKPOINT_STEPS),
        "main_backbone_forward_count_per_train_sample": 1,
        "anchor_micro_loop": True,
        "paper_exact_reproduction": False,
        "no_125_step_pilot": True,
    }


def save_full_sequence_checkpoint(
    *,
    output_dir: Path,
    generator: WanDiffusionWrapper,
    optimizer: torch.optim.Optimizer,
    global_step: int,
    train_rng: torch.Generator,
    validation_base_rng: torch.Generator,
    validation_seed: int,
    sample_plan: Mapping[str, Any],
    resolved_config: Mapping[str, Any],
    provenance: Mapping[str, Any],
    git_sha: str,
    reference_checkpoint_path: Path,
    objective_mode: str,
    smoke: bool,
    extra_metadata: Mapping[str, Any] | None = None,
) -> Path:
    payload = {
        "schema": FULL_SEQUENCE_TRAINER_SCHEMA,
        "run_kind": FULL_SEQUENCE_RUN_KIND,
        "objective_version": FULL_SEQUENCE_OBJECTIVE_VERSION,
        "objective_mode": objective_mode,
        "status": NON_PRODUCTION_SMOKE_TAG if smoke else "PRODUCTION",
        "global_step": int(global_step),
        "generator": move_tensors_to_cpu(generator.state_dict()),
        "optimizer": move_tensors_to_cpu(optimizer.state_dict()),
        "train_rng_state": train_rng.get_state().detach().cpu().clone(),
        "validation_seed": int(validation_seed),
        "validation_base_rng_state": validation_base_rng.get_state().detach().cpu().clone(),
        "python_random_state": random.getstate(),
        "torch_cpu_global_rng_state": torch.get_rng_state().detach().cpu().clone(),
        "torch_cuda_global_rng_state": capture_cuda_rng_state_for_checkpoint(
            torch.device(str(resolved_config["device"]))
        ),
        "sample_cursor": nf_sf_full_sequence_train_cursor(global_step),
        "sample_plan_sha256": str(sample_plan["sample_plan_sha256"]),
        "manifest_sha256": str(resolved_config["manifest_sha256"]),
        "conditionals_artifact_sha256": str(
            resolved_config["conditionals_artifact_sha256"]
        ),
        "resolved_config": dict(resolved_config),
        "provenance": dict(provenance),
        "git_sha": str(git_sha),
        "reference_checkpoint": {
            "path": str(reference_checkpoint_path.resolve()),
            "sha256": OFFICIAL_SELF_FORCING_CHECKPOINT_SHA256,
            "size_bytes": int(resolved_config.get("checkpoint_size_bytes", 0)),
        },
        "optimizer_contract": optimizer_contract(optimizer),
        "extra_metadata": dict(extra_metadata or {}),
    }
    validate_checkpoint_payload(payload)
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "smoke" if smoke else f"step{int(global_step):06d}"
    path = output_dir / f"checkpoint_{suffix}.pt"
    if path.exists():
        raise FileExistsError(f"refusing to overwrite checkpoint: {path}")
    sidecar_paths = checkpoint_sidecar_paths(path)
    for sidecar in sidecar_paths.values():
        if sidecar.exists():
            raise FileExistsError(f"refusing to overwrite checkpoint sidecar: {sidecar}")
    atomic_torch_save(payload, path)
    validation = write_checkpoint_sidecars(
        path=path,
        payload=payload,
        optimizer=optimizer,
    )
    validate_checkpoint_sidecars(path=path, expected_validation=validation)
    return path


def capture_cuda_rng_state_for_checkpoint(device: torch.device) -> torch.Tensor | None:
    if device.type != "cuda" or not torch.cuda.is_available():
        return None
    return torch.cuda.get_rng_state(device).detach().cpu().clone()


def optimizer_contract(optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    betas = optimizer.defaults.get("betas", ADAMW_BETAS)
    return {
        "class": optimizer.__class__.__name__,
        "betas": [float(value) for value in betas],
        "eps": float(optimizer.defaults.get("eps", ADAMW_EPS)),
        "weight_decay": float(optimizer.defaults.get("weight_decay", 0.0)),
        "param_groups": [
            {
                "name": str(group.get("name")),
                "lr": float(group["lr"]),
                "weight_decay": float(group.get("weight_decay", optimizer.defaults.get("weight_decay", 0.0))),
                "param_count": len(group.get("params", ())),
            }
            for group in optimizer.param_groups
        ],
    }


def checkpoint_sidecar_paths(path: Path) -> dict[str, Path]:
    checkpoint_path = Path(path)
    stem = checkpoint_path.with_suffix("")
    return {
        "sha256": stem.with_suffix(".sha256.txt"),
        "validation": stem.with_suffix(".validation.json"),
    }


def write_checkpoint_sidecars(
    *,
    path: Path,
    payload: Mapping[str, Any],
    optimizer: torch.optim.Optimizer,
) -> dict[str, Any]:
    checkpoint_sha = file_sha256(path)
    size_bytes = int(Path(path).stat().st_size)
    sidecars = checkpoint_sidecar_paths(path)
    write_atomic_text(
        sidecars["sha256"],
        f"{checkpoint_sha}  {Path(path).name}\n",
    )
    validation = {
        "status": "PASS",
        "path": str(Path(path).resolve()),
        "sha256": checkpoint_sha,
        "size_bytes": size_bytes,
        "schema": CHECKPOINT_VALIDATION_SCHEMA,
        "run_kind": FULL_SEQUENCE_RUN_KIND,
        "objective_version": FULL_SEQUENCE_OBJECTIVE_VERSION,
        "objective_mode": str(payload["objective_mode"]),
        "global_step": int(payload["global_step"]),
        "generator_key_count": len(payload["generator"]),
        "optimizer_state_entry_count": len(optimizer.state),
    }
    write_m4_json(validation, sidecars["validation"])
    return validation


def write_atomic_text(path: Path, text: str) -> None:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8", newline="\n")
        temporary.replace(path)
    except Exception:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def validate_checkpoint_sidecars(
    *,
    path: Path,
    expected_validation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    sidecars = checkpoint_sidecar_paths(path)
    if not sidecars["sha256"].is_file():
        raise RuntimeError("checkpoint SHA256 sidecar is missing")
    if not sidecars["validation"].is_file():
        raise RuntimeError("checkpoint validation sidecar is missing")
    actual_sha = file_sha256(path)
    sha_text = sidecars["sha256"].read_text(encoding="utf-8").strip().split()
    if not sha_text or sha_text[0] != actual_sha:
        raise RuntimeError("checkpoint SHA256 sidecar mismatch")
    validation = json.loads(sidecars["validation"].read_text(encoding="utf-8"))
    if validation.get("status") != "PASS":
        raise RuntimeError("checkpoint validation sidecar is not PASS")
    if validation.get("schema") != CHECKPOINT_VALIDATION_SCHEMA:
        raise RuntimeError("checkpoint validation sidecar schema mismatch")
    if validation.get("sha256") != actual_sha:
        raise RuntimeError("checkpoint validation sidecar SHA mismatch")
    if int(validation.get("size_bytes", -1)) != int(Path(path).stat().st_size):
        raise RuntimeError("checkpoint validation sidecar size mismatch")
    if expected_validation is not None:
        for key in (
            "schema",
            "run_kind",
            "objective_version",
            "objective_mode",
            "global_step",
            "generator_key_count",
            "optimizer_state_entry_count",
        ):
            if validation.get(key) != expected_validation.get(key):
                raise RuntimeError(f"checkpoint validation sidecar {key} mismatch")
    return validation


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


def validate_checkpoint_payload(payload: Mapping[str, Any]) -> None:
    required = {
        "schema",
        "run_kind",
        "objective_version",
        "objective_mode",
        "global_step",
        "generator",
        "optimizer",
        "train_rng_state",
        "validation_seed",
        "validation_base_rng_state",
        "python_random_state",
        "torch_cpu_global_rng_state",
        "torch_cuda_global_rng_state",
        "sample_cursor",
        "sample_plan_sha256",
        "manifest_sha256",
        "conditionals_artifact_sha256",
        "resolved_config",
        "provenance",
        "reference_checkpoint",
        "optimizer_contract",
    }
    missing = required - payload.keys()
    if missing:
        raise RuntimeError(f"full-sequence checkpoint missing fields: {sorted(missing)}")
    if payload["schema"] != FULL_SEQUENCE_TRAINER_SCHEMA:
        raise RuntimeError("full-sequence checkpoint schema mismatch")
    if payload["run_kind"] != FULL_SEQUENCE_RUN_KIND:
        raise RuntimeError("full-sequence checkpoint run_kind mismatch")
    if payload["objective_version"] != FULL_SEQUENCE_OBJECTIVE_VERSION:
        raise RuntimeError("full-sequence checkpoint objective_version mismatch")
    validate_nf_sf_full_sequence_objective_mode(str(payload["objective_mode"]))
    reference = payload["reference_checkpoint"]
    if reference["sha256"] != OFFICIAL_SELF_FORCING_CHECKPOINT_SHA256:
        raise RuntimeError("full-sequence checkpoint official SHA mismatch")
    if payload["sample_cursor"] != nf_sf_full_sequence_train_cursor(
        int(payload["global_step"])
    ):
        raise RuntimeError("full-sequence checkpoint sample_cursor mismatch")
    if not isinstance(payload["train_rng_state"], torch.Tensor):
        raise RuntimeError("full-sequence checkpoint missing train RNG tensor")
    if not isinstance(payload["validation_base_rng_state"], torch.Tensor):
        raise RuntimeError("full-sequence checkpoint missing validation RNG tensor")
    if not isinstance(payload["torch_cpu_global_rng_state"], torch.Tensor):
        raise RuntimeError("full-sequence checkpoint missing CPU global RNG tensor")


def load_full_sequence_checkpoint(
    path: Path,
    *,
    generator: WanDiffusionWrapper,
    optimizer: torch.optim.Optimizer,
    train_rng: torch.Generator,
    validation_base_rng: torch.Generator,
    objective_mode: str,
    expected_resume_checkpoint_sha256: str,
    expected_git_sha: str,
    expected_resolved_config: Mapping[str, Any],
    expected_provenance: Mapping[str, Any],
    expected_optimizer_contract: Mapping[str, Any],
) -> dict[str, Any]:
    expected_resume_sha = validate_sha256(
        expected_resume_checkpoint_sha256,
        name="--expected_resume_checkpoint_sha256",
    )
    actual_resume_sha = file_sha256(path)
    if actual_resume_sha != expected_resume_sha:
        raise RuntimeError("resume checkpoint SHA256 does not match expected value")
    validate_checkpoint_sidecars(path=path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError("full-sequence checkpoint must contain a dict")
    validate_checkpoint_payload(payload)
    if str(payload.get("status")) != "PRODUCTION":
        raise RuntimeError("resume checkpoint must be a PRODUCTION checkpoint")
    if str(payload["objective_mode"]) != objective_mode:
        raise RuntimeError("resume objective_mode mismatch")
    if int(payload["global_step"]) not in (0, 500, 2000):
        raise RuntimeError("resume is only allowed from step0, step500, or step2000")
    expected_git = validate_git_sha(expected_git_sha, name="expected_git_sha")
    if str(payload["git_sha"]) != expected_git:
        raise RuntimeError("resume checkpoint git_sha mismatch")
    generator.load_state_dict(payload["generator"], strict=True)
    if not any(str(key).startswith("mcp.") for key in generator.state_dict().keys()):
        raise RuntimeError("resume generator state does not contain MCP keys")
    optimizer.load_state_dict(payload["optimizer"])
    move_optimizer_state_to_device(optimizer, device=next(generator.parameters()).device)
    if payload["sample_plan_sha256"] != expected_resolved_config["sample_plan_sha256"]:
        raise RuntimeError("resume sample plan SHA mismatch")
    if payload["manifest_sha256"] != expected_resolved_config["manifest_sha256"]:
        raise RuntimeError("resume manifest SHA mismatch")
    if (
        payload["conditionals_artifact_sha256"]
        != expected_resolved_config["conditionals_artifact_sha256"]
    ):
        raise RuntimeError("resume conditional artifact SHA mismatch")
    if payload["resolved_config"] != dict(expected_resolved_config):
        raise RuntimeError("resume resolved_config semantic mismatch")
    if payload["provenance"] != dict(expected_provenance):
        raise RuntimeError("resume provenance mismatch")
    if payload["optimizer_contract"] != dict(expected_optimizer_contract):
        raise RuntimeError("resume optimizer contract mismatch")
    train_rng.set_state(payload["train_rng_state"])
    validation_base_rng.set_state(payload["validation_base_rng_state"])
    random.setstate(payload["python_random_state"])
    torch.set_rng_state(payload["torch_cpu_global_rng_state"])
    device = next(generator.parameters()).device
    if device.type == "cuda":
        if payload["torch_cuda_global_rng_state"] is None:
            raise RuntimeError("resume checkpoint missing CUDA global RNG state")
        torch.cuda.set_rng_state(payload["torch_cuda_global_rng_state"], device)
    payload["restore_contract"] = {"status": "PASS"}
    return payload


def move_optimizer_state_to_device(
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device=device)


def target_latent_from_sample(sample: Any) -> torch.Tensor:
    target = sample.target_latent
    if not isinstance(target, torch.Tensor):
        raise TypeError("teacher sample target_latent must be a tensor")
    if target.ndim != 5:
        raise ValueError("target_latent must have shape [B, 21, C, H, W]")
    if int(target.shape[1]) != FULL_SEQUENCE_FRAME_COUNT:
        raise ValueError("target_latent must contain exactly 21 latent frames")
    return target


def run_full_sequence_train_step(
    *,
    generator: WanDiffusionWrapper,
    optimizer: torch.optim.Optimizer,
    scheduler_main: FlowMatchScheduler,
    scheduler_mcp: FlowMatchScheduler,
    teacher_store: M5TeacherSampleStore,
    conditional_store: M5ConditionalArtifactStore,
    train_rng: torch.Generator,
    global_step: int,
    objective_mode: str,
    device: torch.device,
    dtype: torch.dtype,
    smoke: bool = False,
    full_gradient_audit: bool = False,
    structural_gate: bool = False,
    run_gc: bool = False,
    capture_memory: bool = False,
) -> dict[str, Any]:
    cursor = nf_sf_full_sequence_train_cursor(global_step)
    identity = teacher_store.train_identity_for_step(global_step)
    if smoke and device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    memory = {} if capture_memory or smoke else None
    optimizer.zero_grad(set_to_none=True)
    if memory is not None:
        memory["before_sample"] = memory_snapshot("before_sample", device)

    with teacher_store.acquire(identity) as sample:
        with conditional_store.acquire(identity) as conditional_cpu:
            clean_target = target_latent_from_sample(sample).to(
                device=device,
                dtype=dtype,
            )
            conditional = move_tensors_to_device(
                conditional_cpu,
                device=device,
                floating_dtype=dtype,
            )
            noisy_batch = prepare_nf_sf_full_sequence_noisy_batch(
                clean_target,
                scheduler_main=scheduler_main,
                scheduler_mcp=scheduler_mcp,
                rng=train_rng,
            )
            before_global_rng = capture_global_rng_state(device)
            result = run_nf_sf_full_sequence_forward_loss(
                generator,
                conditional_dict=conditional,
                noisy_batch=noisy_batch,
                objective_mode=objective_mode,
            )
            after_forward_global_rng = capture_global_rng_state(device)
            assert_global_rng_equal(before_global_rng, after_forward_global_rng)
            assert_finite_loss(result.losses.total_loss, name="total_loss")
            structural_report = None
            if structural_gate or smoke:
                structural_report = require_full_sequence_structural_gate(
                    result=result,
                    noisy_batch=noisy_batch,
                    objective_mode=objective_mode,
                )
            if memory is not None:
                memory["after_forward"] = memory_snapshot("after_forward", device)
            result.losses.total_loss.backward()
            after_backward_global_rng = capture_global_rng_state(device)
            assert_global_rng_equal(before_global_rng, after_backward_global_rng)
            if memory is not None:
                memory["after_backward"] = memory_snapshot("after_backward", device)
            gradient_report = None
            if full_gradient_audit or smoke:
                gradient_report = validate_full_sequence_gradient_audit(
                    audit_nf_sf_full_sequence_gradients(
                        generator,
                        objective_mode=objective_mode,
                    ),
                    objective_mode=objective_mode,
                    global_step=global_step,
                )
                nonfinite_grad = gradient_report_has_nonfinite(gradient_report)
            else:
                nonfinite_grad = has_nonfinite_grad(generator)
            if nonfinite_grad:
                raise RuntimeError("non-finite gradient detected")
            mcp_head_bootstrap_before = None
            if is_mcp_zero_head_bootstrap_step(
                global_step=global_step,
                objective_mode=objective_mode,
            ):
                mcp_head_bootstrap_before = (
                    capture_mcp_output_head_bootstrap_before_step(generator)
                )
            optimizer.step()
            mcp_head_bootstrap_report = None
            if mcp_head_bootstrap_before is not None:
                mcp_head_bootstrap_report = (
                    validate_mcp_output_heads_left_zero_init_after_step1(
                        generator,
                        before_report=mcp_head_bootstrap_before,
                    )
                )
            after_step_global_rng = capture_global_rng_state(device)
            assert_global_rng_equal(before_global_rng, after_step_global_rng)
            if memory is not None:
                memory["after_optimizer_step"] = memory_snapshot(
                    "after_optimizer_step",
                    device,
                )
            optimizer_state_size = len(optimizer.state)
            if smoke and optimizer_state_size <= 0:
                raise RuntimeError("Adam state was not allocated by smoke step")

            metrics = loss_breakdown_to_floats(result.losses)
            metrics.update(
                {
                    "global_step": int(global_step),
                    "sample_identity": identity,
                    "sample_cursor": cursor,
                    "target_latent_shape": [int(dim) for dim in clean_target.shape],
                    "main_pred_shape": [
                        int(dim) for dim in result.main_flow_pred.shape
                    ],
                    "mcp_pred_shapes": [
                        [int(dim) for dim in pred.shape]
                        for pred in result.mcp_flow_preds_by_depth
                    ],
                    "gradient_report": gradient_report,
                    "mcp_output_head_bootstrap_report": mcp_head_bootstrap_report,
                    "optimizer_state_entries": int(optimizer_state_size),
                    "memory": memory,
                    "structural_report": structural_report,
                    "smoke": bool(smoke),
                }
            )
            del result
            del noisy_batch
            del conditional
            del clean_target

    optimizer.zero_grad(set_to_none=True)
    if run_gc or smoke:
        gc.collect()
    if memory is not None:
        memory["after_cleanup"] = memory_snapshot("after_cleanup", device)
        if smoke:
            metrics["smoke_memory_gate"] = validate_smoke_memory_gate(memory)
    return metrics


@torch.no_grad()
def run_full_sequence_validation(
    *,
    generator: WanDiffusionWrapper,
    scheduler_main: FlowMatchScheduler,
    scheduler_mcp: FlowMatchScheduler,
    teacher_store: M5TeacherSampleStore,
    conditional_store: M5ConditionalArtifactStore,
    validation_seed: int,
    train_rng: torch.Generator,
    objective_mode: str,
    device: torch.device,
    dtype: torch.dtype,
    global_step: int,
) -> dict[str, Any]:
    train_rng_before = train_rng.get_state().detach().cpu().clone()
    global_rng_before = capture_global_rng_state(device)
    totals = []
    main_totals = []
    per_sample = []
    main_chunks = [[] for _ in range(FULL_SEQUENCE_NUM_CHUNKS)]
    mcp_depths = [[] for _ in FULL_SEQUENCE_DEPTHS]
    mcp_anchors = [
        [[] for _ in range(FULL_SEQUENCE_NUM_CHUNKS - depth)]
        for depth in FULL_SEQUENCE_DEPTHS
    ]
    was_training = generator.training
    generator.eval()
    try:
        for identity in teacher_store.validation_identities:
            derived_seed = derive_m4_validation_seed(
                base_seed=int(validation_seed),
                sample_identity=identity,
                tensor_slot=VALIDATION_TENSOR_SLOT,
            )
            local_validation_rng = make_generator(derived_seed, device)
            with teacher_store.acquire(identity) as sample:
                with conditional_store.acquire(identity) as conditional_cpu:
                    clean_target = target_latent_from_sample(sample).to(
                        device=device,
                        dtype=dtype,
                    )
                    conditional = move_tensors_to_device(
                        conditional_cpu,
                        device=device,
                        floating_dtype=dtype,
                    )
                    noisy_batch = prepare_nf_sf_full_sequence_noisy_batch(
                        clean_target,
                        scheduler_main=scheduler_main,
                        scheduler_mcp=scheduler_mcp,
                        rng=local_validation_rng,
                    )
                    result = run_nf_sf_full_sequence_forward_loss(
                        generator,
                        conditional_dict=conditional,
                        noisy_batch=noisy_batch,
                        objective_mode=objective_mode,
                    )
                    losses = result.losses
                    totals.append(float(losses.total_loss.detach().float().item()))
                    main_totals.append(float(losses.main_loss.detach().float().item()))
                    for index, loss in enumerate(losses.main_chunk_losses):
                        main_chunks[index].append(float(loss.detach().float().item()))
                    for index, loss in enumerate(losses.mcp_depth_losses):
                        mcp_depths[index].append(float(loss.detach().float().item()))
                    for depth_index, anchor_losses in enumerate(losses.mcp_anchor_losses):
                        for anchor_index, loss in enumerate(anchor_losses):
                            mcp_anchors[depth_index][anchor_index].append(
                                float(loss.detach().float().item())
                            )
                    per_sample.append(
                        {
                            "identity": str(identity),
                            "derived_seed": int(derived_seed),
                            "total_loss": float(
                                losses.total_loss.detach().float().item()
                            ),
                            "main_loss": float(
                                losses.main_loss.detach().float().item()
                            ),
                        }
                    )
                    del result
                    del noisy_batch
                    del conditional
                    del clean_target
    finally:
        generator.train(was_training)
    if not torch.equal(train_rng_before, train_rng.get_state().cpu()):
        raise RuntimeError("validation consumed train_rng")
    assert_global_rng_equal(global_rng_before, capture_global_rng_state(device))
    return {
        "schema": f"{FULL_SEQUENCE_TRAINER_SCHEMA}_validation_v1",
        "global_step": int(global_step),
        "identity_count": len(teacher_store.validation_identities),
        "objective_mode": objective_mode,
        "validation_seed": int(validation_seed),
        "seed_derivation": "derive_m4_validation_seed",
        "tensor_slot": VALIDATION_TENSOR_SLOT,
        "paired_identity_noise_across_steps": True,
        "per_sample": per_sample,
        "weighted_total": mean_or_none(totals),
        "main_total": mean_or_none(main_totals),
        "main_per_chunk": [mean_or_none(values) for values in main_chunks],
        "mcp_depth_means": [mean_or_none(values) for values in mcp_depths],
        "mcp_per_anchor": [
            [mean_or_none(values) for values in anchors]
            for anchors in mcp_anchors
        ],
    }


def loss_breakdown_to_floats(losses) -> dict[str, Any]:
    return {
        "total_loss": float(losses.total_loss.detach().float().item()),
        "main_loss": float(losses.main_loss.detach().float().item()),
        "main_chunk_losses": [
            float(loss.detach().float().item()) for loss in losses.main_chunk_losses
        ],
        "mcp_depth_losses": [
            float(loss.detach().float().item()) for loss in losses.mcp_depth_losses
        ],
        "mcp_anchor_losses": [
            [float(loss.detach().float().item()) for loss in anchor_losses]
            for anchor_losses in losses.mcp_anchor_losses
        ],
    }


def assert_finite_loss(loss: torch.Tensor, *, name: str) -> None:
    if not bool(torch.isfinite(loss.detach()).all().item()):
        raise RuntimeError(f"{name} is non-finite")


def has_nonfinite_grad(module: torch.nn.Module) -> bool:
    for param in module.parameters():
        if param.grad is None:
            continue
        if not bool(torch.isfinite(param.grad.detach().float()).all().item()):
            return True
    return False


def capture_global_rng_state(device: torch.device) -> dict[str, Any]:
    state = {"cpu": torch.get_rng_state().detach().cpu().clone()}
    if device.type == "cuda" and torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state(device).detach().cpu().clone()
    return state


def assert_global_rng_equal(before: Mapping[str, Any], after: Mapping[str, Any]) -> None:
    if before.keys() != after.keys():
        raise RuntimeError("global RNG state device set changed")
    for key in before:
        if not torch.equal(before[key], after[key]):
            raise RuntimeError(f"global RNG state changed during {key} execution")


def memory_snapshot(label: str, device: torch.device) -> dict[str, Any]:
    if device.type != "cuda" or not torch.cuda.is_available():
        return {"label": label, "cuda": False}
    return {
        "label": label,
        "cuda": True,
        "allocated": int(torch.cuda.memory_allocated(device)),
        "reserved": int(torch.cuda.memory_reserved(device)),
        "max_allocated": int(torch.cuda.max_memory_allocated(device)),
        "max_reserved": int(torch.cuda.max_memory_reserved(device)),
        "total": int(torch.cuda.get_device_properties(device).total_memory),
    }


def expected_anchor_token_slices() -> tuple[tuple[int, int], ...]:
    return tuple(
        (
            anchor_index * FULL_SEQUENCE_CHUNK_TOKENS,
            (anchor_index + 1) * FULL_SEQUENCE_CHUNK_TOKENS,
        )
        for anchor_index in range(FULL_SEQUENCE_NUM_CHUNKS)
    )


def gradient_report_has_nonfinite(report: Mapping[str, Mapping[str, Any]]) -> bool:
    return any(
        int(item.get("grad_tensors", 0)) > 0
        and not bool(item.get("all_finite", False))
        for item in report.values()
    )


def is_mcp_zero_head_bootstrap_step(
    *,
    global_step: int,
    objective_mode: str,
) -> bool:
    return int(global_step) == 1 and objective_mode == "next_forcing_full"


def validate_full_sequence_gradient_audit(
    report: Mapping[str, Mapping[str, Any]],
    *,
    objective_mode: str,
    global_step: int,
) -> dict[str, dict[str, Any]]:
    objective_mode = validate_nf_sf_full_sequence_objective_mode(objective_mode)
    augmented = {name: dict(item) for name, item in report.items()}
    required = ["backbone", "patch_embedding"]
    if objective_mode == "next_forcing_full":
        required.extend(["mcp_fusion", "mcp_depth1", "mcp_depth2", "mcp_depth3"])
    missing_groups = [name for name in required if name not in augmented]
    if missing_groups:
        raise RuntimeError(f"full-sequence gradient audit missing groups: {missing_groups}")

    bootstrap_step = is_mcp_zero_head_bootstrap_step(
        global_step=global_step,
        objective_mode=objective_mode,
    )
    if bootstrap_step:
        fusion = augmented["mcp_fusion"]
        fusion["bootstrap_zero_grad_allowed"] = True
        fusion["bootstrap_reason"] = MCP_ZERO_HEAD_BOOTSTRAP_REASON
        fusion["pass"] = (
            bool(fusion.get("expected_trainable", False))
            and int(fusion.get("trainable_tensors", 0)) > 0
            and int(fusion.get("missing_grad_tensors", -1)) == 0
            and bool(fusion.get("all_finite", False))
        )

    failures = [name for name, item in augmented.items() if not bool(item.get("pass"))]
    if failures:
        raise RuntimeError(
            "full-sequence gradient audit failed: "
            + ", ".join(str(name) for name in failures)
        )
    return augmented


def mcp_output_head_weight_tensors(
    generator: WanDiffusionWrapper,
) -> tuple[torch.Tensor, ...]:
    mcp = getattr(generator, "mcp", None)
    modules = getattr(mcp, "mcp_modules", None)
    if modules is None:
        raise RuntimeError("MCP output head bootstrap check requires mcp_modules")
    module_list = list(modules)
    if len(module_list) < 3:
        raise RuntimeError("MCP output head bootstrap check requires three depths")
    weights = []
    for depth, module in enumerate(module_list[:3], start=1):
        head = getattr(getattr(module, "head", None), "head", None)
        weight = getattr(head, "weight", None)
        if not isinstance(weight, torch.Tensor):
            raise RuntimeError(
                f"MCP depth {depth} missing output head.head.weight tensor"
            )
        weights.append(weight)
    return tuple(weights)


def capture_mcp_output_head_bootstrap_before_step(
    generator: WanDiffusionWrapper,
) -> dict[str, Any]:
    per_depth = []
    for depth, weight in enumerate(mcp_output_head_weight_tensors(generator), start=1):
        detached = weight.detach().float()
        finite = bool(torch.isfinite(detached).all().item())
        norm = float(detached.norm().item())
        all_zero = finite and norm == 0.0
        per_depth.append(
            {
                "depth": depth,
                "finite": finite,
                "all_zero_before_step": all_zero,
                "norm_before_step": norm,
            }
        )
    if not all(item["all_zero_before_step"] for item in per_depth):
        raise RuntimeError("MCP output heads were not all zero before step1")
    return {
        "status": "PASS",
        "bootstrap_reason": MCP_ZERO_HEAD_BOOTSTRAP_REASON,
        "per_depth": per_depth,
    }


def validate_mcp_output_heads_left_zero_init_after_step1(
    generator: WanDiffusionWrapper,
    *,
    before_report: Mapping[str, Any],
) -> dict[str, Any]:
    before_by_depth = {
        int(item["depth"]): dict(item)
        for item in before_report.get("per_depth", ())
    }
    per_depth = []
    for depth, weight in enumerate(mcp_output_head_weight_tensors(generator), start=1):
        detached = weight.detach().float()
        finite = bool(torch.isfinite(detached).all().item())
        norm = float(detached.norm().item())
        before_zero = bool(before_by_depth.get(depth, {}).get("all_zero_before_step"))
        left_zero = before_zero and finite and norm > 0.0
        per_depth.append(
            {
                "depth": depth,
                "finite": finite,
                "all_zero_before_step": before_zero,
                "norm_after_step": norm,
                "left_zero_init_after_step1": left_zero,
            }
        )
    passed = all(item["left_zero_init_after_step1"] for item in per_depth)
    report = {
        "status": "PASS" if passed else "FAIL",
        "bootstrap_reason": MCP_ZERO_HEAD_BOOTSTRAP_REASON,
        "mcp_output_heads_left_zero_init_after_step1": passed,
        "per_depth": per_depth,
    }
    if not passed:
        raise RuntimeError("MCP output heads did not leave zero init after step1")
    return report


def require_full_sequence_structural_gate(
    *,
    result,
    noisy_batch,
    objective_mode: str,
) -> dict[str, Any]:
    objective_mode = validate_nf_sf_full_sequence_objective_mode(objective_mode)
    batch_size = int(noisy_batch.clean_target.shape[0])
    channel_tail = tuple(int(dim) for dim in noisy_batch.clean_target.shape[2:])
    expected_main_shape = (
        batch_size,
        FULL_SEQUENCE_FRAME_COUNT,
        *channel_tail,
    )
    if tuple(int(dim) for dim in result.main_flow_pred.shape) != expected_main_shape:
        raise RuntimeError("full-sequence structural gate failed: main pred shape")
    if tuple(int(dim) for dim in noisy_batch.raw_timestep_main.shape) != (
        batch_size,
        FULL_SEQUENCE_NUM_CHUNKS,
    ):
        raise RuntimeError("full-sequence structural gate failed: raw main shape")
    if tuple(int(dim) for dim in noisy_batch.timestep_main.shape) != (
        batch_size,
        FULL_SEQUENCE_FRAME_COUNT,
    ):
        raise RuntimeError("full-sequence structural gate failed: timestep main shape")
    if len(result.losses.main_chunk_losses) != FULL_SEQUENCE_NUM_CHUNKS:
        raise RuntimeError("full-sequence structural gate failed: main loss count")
    if str(noisy_batch.rng_draw_order_version) != FULL_SEQUENCE_RNG_DRAW_ORDER_VERSION:
        raise RuntimeError("full-sequence structural gate failed: RNG order version")
    if not bool(torch.isfinite(result.main_flow_pred.detach()).all().item()):
        raise RuntimeError("full-sequence structural gate failed: main pred non-finite")

    formula = result.losses.main_loss.detach().float()
    expected_mcp_shapes: list[list[int]] = []
    if objective_mode == "next_forcing_full":
        if int(result.main_backbone_forward_count) != 1:
            raise RuntimeError(
                "full-sequence structural gate failed: backbone forward count"
            )
        if tuple(result.anchor_token_slices) != expected_anchor_token_slices():
            raise RuntimeError("full-sequence structural gate failed: anchor slices")
        if len(result.tap_shapes) != len(TAP_LAYERS):
            raise RuntimeError("full-sequence structural gate failed: tap count")
        hidden_dim = None
        token_count = FULL_SEQUENCE_FRAME_COUNT * FULL_SEQUENCE_FRAME_SEQ_LENGTH
        for shape in result.tap_shapes:
            shape_tuple = tuple(int(dim) for dim in shape)
            if len(shape_tuple) != 3:
                raise RuntimeError("full-sequence structural gate failed: tap rank")
            if shape_tuple[0] != batch_size or shape_tuple[1] != token_count:
                raise RuntimeError("full-sequence structural gate failed: tap shape")
            if shape_tuple[2] <= 0:
                raise RuntimeError("full-sequence structural gate failed: hidden dim")
            if hidden_dim is None:
                hidden_dim = shape_tuple[2]
            elif hidden_dim != shape_tuple[2]:
                raise RuntimeError(
                    "full-sequence structural gate failed: tap hidden mismatch"
                )
        if result.future_embedding_order != "depth_major":
            raise RuntimeError(
                "full-sequence structural gate failed: future embedding order"
            )
        if len(result.mcp_flow_preds_by_depth) != len(FULL_SEQUENCE_DEPTHS):
            raise RuntimeError("full-sequence structural gate failed: MCP depth count")
        if len(result.losses.mcp_depth_losses) != len(FULL_SEQUENCE_DEPTHS):
            raise RuntimeError(
                "full-sequence structural gate failed: MCP depth loss count"
            )
        if len(result.losses.mcp_anchor_losses) != len(FULL_SEQUENCE_DEPTHS):
            raise RuntimeError(
                "full-sequence structural gate failed: MCP anchor loss depths"
            )
        for index, depth in enumerate(FULL_SEQUENCE_DEPTHS):
            anchor_count = FULL_SEQUENCE_NUM_CHUNKS - int(depth)
            expected_shape = (
                batch_size,
                anchor_count,
                FULL_SEQUENCE_CHUNK_FRAMES,
                *channel_tail,
            )
            actual_shape = tuple(int(dim) for dim in result.mcp_flow_preds_by_depth[index].shape)
            if actual_shape != expected_shape:
                raise RuntimeError(
                    f"full-sequence structural gate failed: MCP depth {depth} shape"
                )
            if len(result.losses.mcp_anchor_losses[index]) != anchor_count:
                raise RuntimeError(
                    f"full-sequence structural gate failed: MCP depth {depth} anchors"
                )
            if not bool(
                torch.isfinite(result.mcp_flow_preds_by_depth[index].detach()).all().item()
            ):
                raise RuntimeError(
                    f"full-sequence structural gate failed: MCP depth {depth} non-finite"
                )
            formula = formula + float(FULL_SEQUENCE_DEPTH_WEIGHTS[index]) * (
                result.losses.mcp_depth_losses[index].detach().float()
            )
            expected_mcp_shapes.append(list(expected_shape))
    else:
        if result.mcp_flow_preds_by_depth:
            raise RuntimeError("main_only_full_control must not return MCP predictions")
        if result.losses.mcp_depth_losses or result.losses.mcp_anchor_losses:
            raise RuntimeError("main_only_full_control must not return MCP losses")

    total_loss = result.losses.total_loss.detach().float()
    if not bool(torch.isfinite(total_loss).all().item()):
        raise RuntimeError("full-sequence structural gate failed: total non-finite")
    if not torch.allclose(total_loss, formula, rtol=1.0e-6, atol=1.0e-6):
        raise RuntimeError("full-sequence structural gate failed: total formula")
    return {
        "status": "PASS",
        "objective_mode": objective_mode,
        "main_backbone_forward_count": int(result.main_backbone_forward_count),
        "tap_shapes": [list(map(int, shape)) for shape in result.tap_shapes],
        "anchor_token_slices": [list(item) for item in result.anchor_token_slices],
        "main_pred_shape": list(expected_main_shape),
        "raw_main_shape": [batch_size, FULL_SEQUENCE_NUM_CHUNKS],
        "timestep_main_shape": [batch_size, FULL_SEQUENCE_FRAME_COUNT],
        "mcp_pred_shapes": expected_mcp_shapes,
        "main_chunk_loss_count": len(result.losses.main_chunk_losses),
        "mcp_anchor_loss_counts": [
            len(anchor_losses) for anchor_losses in result.losses.mcp_anchor_losses
        ],
        "total_formula_checked": True,
        "tail_loss_excluded": True,
        "rng_draw_order_version": FULL_SEQUENCE_RNG_DRAW_ORDER_VERSION,
        "future_embedding_order": result.future_embedding_order,
    }


def validate_smoke_memory_gate(memory: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    required = (
        "before_sample",
        "after_forward",
        "after_backward",
        "after_optimizer_step",
        "after_cleanup",
    )
    missing = [label for label in required if label not in memory]
    if missing:
        raise RuntimeError(f"smoke memory gate missing snapshots: {missing}")
    if not bool(memory["after_cleanup"].get("cuda", False)):
        return {
            "status": "PASS",
            "cuda": False,
            "checked_cuda_thresholds": False,
            "snapshots": list(required),
        }
    total = int(memory["after_cleanup"]["total"])
    max_allocated = max(int(memory[label]["max_allocated"]) for label in required)
    if total <= 0 or max_allocated >= total:
        raise RuntimeError("smoke memory gate failed: peak allocation exceeds device")
    baseline = int(memory["after_optimizer_step"]["allocated"])
    after_cleanup = int(memory["after_cleanup"]["allocated"])
    if after_cleanup > baseline + SMOKE_CLEANUP_TOLERANCE_BYTES:
        raise RuntimeError("smoke memory gate failed: cleanup allocation too high")
    return {
        "status": "PASS",
        "cuda": True,
        "baseline_label": "after_optimizer_step",
        "cleanup_tolerance_bytes": SMOKE_CLEANUP_TOLERANCE_BYTES,
        "baseline_allocated": baseline,
        "after_cleanup_allocated": after_cleanup,
        "max_allocated": max_allocated,
        "total": total,
    }


def should_run_step0_validation(
    *,
    resume_checkpoint: Path | None,
    validation_steps: Sequence[int],
) -> bool:
    return resume_checkpoint is None and 0 in tuple(int(step) for step in validation_steps)


def should_run_full_gradient_audit(
    global_step: int,
    *,
    smoke: bool,
    checkpoint_steps: Sequence[int],
    validation_steps: Sequence[int],
    memory_log_interval: int,
) -> bool:
    step = int(global_step)
    interval = int(memory_log_interval)
    if interval <= 0:
        raise ValueError("memory_log_interval must be positive")
    return (
        bool(smoke)
        or step in (1, 2)
        or step % interval == 0
        or step in set(int(value) for value in checkpoint_steps)
        or step in set(int(value) for value in validation_steps)
    )


def should_run_cleanup_gc(
    global_step: int,
    *,
    smoke: bool,
    checkpoint_steps: Sequence[int],
    validation_steps: Sequence[int],
    memory_log_interval: int,
) -> bool:
    return should_run_full_gradient_audit(
        global_step,
        smoke=smoke,
        checkpoint_steps=checkpoint_steps,
        validation_steps=validation_steps,
        memory_log_interval=memory_log_interval,
    )


def should_capture_memory(
    global_step: int,
    *,
    smoke: bool,
    checkpoint_steps: Sequence[int],
    validation_steps: Sequence[int],
    memory_log_interval: int,
) -> bool:
    return should_run_full_gradient_audit(
        global_step,
        smoke=smoke,
        checkpoint_steps=checkpoint_steps,
        validation_steps=validation_steps,
        memory_log_interval=memory_log_interval,
    )


def memory_metrics_for_record(
    memory: Mapping[str, Any] | None,
    *,
    device: torch.device,
) -> dict[str, Any] | None:
    if memory is not None:
        return dict(memory)
    if device.type != "cuda" or not torch.cuda.is_available():
        return None
    return {
        "post_step_allocated": {
            "label": "post_step_allocated",
            "cuda": True,
            "allocated": int(torch.cuda.memory_allocated(device)),
            "reserved": None,
            "max_allocated": None,
            "max_reserved": None,
            "total": int(torch.cuda.get_device_properties(device).total_memory),
        }
    }


def update_memory_maxima(
    maxima: dict[str, int],
    memory: Mapping[str, Any] | None,
) -> dict[str, int]:
    if not memory:
        return maxima
    for snapshot in memory.values():
        if not isinstance(snapshot, Mapping) or not bool(snapshot.get("cuda", False)):
            continue
        for key in ("allocated", "reserved", "max_allocated", "max_reserved"):
            value = snapshot.get(key)
            if value is None:
                continue
            maxima[key] = max(int(maxima.get(key, 0)), int(value))
        total = snapshot.get("total")
        if total is not None:
            maxima["total"] = int(total)
    return maxima


def compact_train_record(
    record: Mapping[str, Any],
    *,
    device: torch.device,
) -> dict[str, Any]:
    compact = {
        "global_step": int(record["global_step"]),
        "sample_identity": str(record["sample_identity"]),
        "sample_cursor": record["sample_cursor"],
        "total_loss": float(record["total_loss"]),
        "main_loss": float(record["main_loss"]),
        "mcp_depth_losses": record.get("mcp_depth_losses", []),
        "main_chunk_losses": record.get("main_chunk_losses", []),
        "mcp_anchor_losses": record.get("mcp_anchor_losses", []),
        "elapsed_ms": float(record.get("elapsed_ms", 0.0)),
        "memory": memory_metrics_for_record(record.get("memory"), device=device),
        "gradient_report": record.get("gradient_report"),
        "mcp_output_head_bootstrap_report": record.get(
            "mcp_output_head_bootstrap_report"
        ),
        "structural_report": record.get("structural_report"),
        "smoke_memory_gate": record.get("smoke_memory_gate"),
    }
    return compact


def append_jsonl(path: Path, record: Mapping[str, Any], *, fsync: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()
        if fsync:
            os.fsync(handle.fileno())


def compact_validation_summary(report: Mapping[str, Any], *, path: Path) -> dict[str, Any]:
    return {
        "global_step": int(report["global_step"]),
        "path": str(path.resolve()),
        "identity_count": int(report["identity_count"]),
        "weighted_total": report.get("weighted_total"),
        "main_total": report.get("main_total"),
        "mcp_depth_means": report.get("mcp_depth_means"),
        "tensor_slot": report.get("tensor_slot"),
        "paired_identity_noise_across_steps": bool(
            report.get("paired_identity_noise_across_steps")
        ),
    }


def checkpoint_record_from_path(path: Path) -> dict[str, Any]:
    validation = validate_checkpoint_sidecars(path=path)
    return {
        "path": str(Path(path).resolve()),
        "sha256": validation["sha256"],
        "size_bytes": int(validation["size_bytes"]),
        "global_step": int(validation["global_step"]),
        "objective_mode": validation["objective_mode"],
        "optimizer_state_entry_count": int(validation["optimizer_state_entry_count"]),
    }


def verify_reference_checkpoint_immutability(
    *,
    reference_checkpoint_path: Path,
    preflight_report: Mapping[str, Any],
) -> dict[str, Any]:
    before = {
        "path": str(reference_checkpoint_path.resolve()),
        "sha256": str(preflight_report["checkpoint_sha256"]),
        "size_bytes": int(preflight_report["checkpoint_size_bytes"]),
    }
    after = {
        "path": str(reference_checkpoint_path.resolve()),
        "sha256": file_sha256(reference_checkpoint_path),
        "size_bytes": int(reference_checkpoint_path.stat().st_size),
    }
    if before["sha256"] != after["sha256"] or before["size_bytes"] != after["size_bytes"]:
        raise RuntimeError("official Self-Forcing checkpoint changed during run")
    return {
        "status": "PASS",
        "before": before,
        "after": after,
    }


def build_training_summary(
    *,
    objective_mode: str,
    status: str,
    target_global_step: int,
    completed_global_step: int,
    checkpoint_steps: Sequence[int],
    validation_steps: Sequence[int],
    metrics_path: Path,
    train_record_count: int,
    final_train_record: Mapping[str, Any] | None,
    checkpoint_records: Sequence[Mapping[str, Any]],
    validation_summaries: Sequence[Mapping[str, Any]],
    memory_maxima: Mapping[str, int],
    resume_payload: Mapping[str, Any] | None,
    smoke: bool,
    reference_checkpoint_immutability: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema": FULL_SEQUENCE_TRAINER_SCHEMA,
        "run_kind": FULL_SEQUENCE_RUN_KIND,
        "objective_version": FULL_SEQUENCE_OBJECTIVE_VERSION,
        "objective_mode": objective_mode,
        "status": status,
        "target_global_step": int(target_global_step),
        "completed_global_step": int(completed_global_step),
        "checkpoint_steps": [int(step) for step in checkpoint_steps],
        "validation_steps": [int(step) for step in validation_steps],
        "train_record_count": int(train_record_count),
        "metrics_jsonl": str(metrics_path.resolve()),
        "final_train_record": dict(final_train_record) if final_train_record else None,
        "checkpoint_records": [dict(record) for record in checkpoint_records],
        "validation_reports": [dict(record) for record in validation_summaries],
        "memory_maxima": dict(memory_maxima),
        "resume_from": None if resume_payload is None else int(resume_payload["global_step"]),
        "resume_info": None
        if resume_payload is None
        else dict(resume_payload.get("restore_contract", {})),
        "reference_checkpoint_immutability": (
            None
            if reference_checkpoint_immutability is None
            else dict(reference_checkpoint_immutability)
        ),
        "next_action_after_smoke_pass": (
            "production next_forcing_full fresh 0->5000; no 125-step pilot"
            if smoke
            else None
        ),
    }


def mean_or_none(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return float(sum(float(value) for value in values) / len(values))


def build_step0_metadata(
    *,
    args: argparse.Namespace,
    config: Any,
    sample_plan: Mapping[str, Any],
    git_sha: str,
    preflight_report: Mapping[str, Any],
) -> dict[str, Any]:
    provenance = build_nf_sf_full_sequence_provenance(
        objective_mode=args.objective_mode,
        reference_checkpoint_sha256=OFFICIAL_SELF_FORCING_CHECKPOINT_SHA256,
        git_sha=git_sha,
    )
    provenance["objective"]["mcp_zero_head_bootstrap"] = args.objective_mode == "next_forcing_full"
    provenance["objective"]["first_step_fusion_zero_grad_expected"] = (
        args.objective_mode == "next_forcing_full"
    )
    return {
        "resolved_config": full_sequence_resolved_config(
            config,
            args,
            preflight_report=preflight_report,
        ),
        "provenance": provenance,
        "sample_plan_sha256": str(sample_plan["sample_plan_sha256"]),
        "preflight": dict(preflight_report),
        "fresh_parent": "official_self_forcing_checkpoint",
        "old_fixed_window_parent": None,
        "no_125_step_pilot": True,
    }


def prepare_output_dir(output_dir: Path, *, resume: bool) -> None:
    if output_dir.exists():
        if not output_dir.is_dir():
            raise ValueError(f"output_dir is not a directory: {output_dir}")
        if not resume and any(output_dir.iterdir()):
            raise FileExistsError(
                "fresh full-sequence run requires a new or empty output_dir"
            )
    else:
        output_dir.mkdir(parents=True)


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    config = merge_config(args.config)
    cli = validate_cli_contract(args, config)
    sample_plan = load_m4_sample_plan(args.sample_plan, manifest_path=args.manifest)
    validate_sample_plan_contract(sample_plan)
    objective_mode = str(cli["objective_mode"])
    device = torch.device(args.device)
    dtype = dtype_from_arg(args.dtype)

    conditional_store = M5ConditionalArtifactStore(
        artifact_dir=args.conditionals_artifact,
        sample_plan=sample_plan,
    )
    preflight_report = validate_repo_preflight_facts(
        collect_repo_preflight_facts(
            args=args,
            sample_plan=sample_plan,
            conditional_store=conditional_store,
        )
    )
    current_git_sha = str(preflight_report["current_git_sha"])
    reset_global_seed(args.global_seed)

    teacher_store = M5TeacherSampleStore(
        sample_plan=sample_plan,
        manifest_path=args.manifest,
        dataset_root=args.dataset_root,
        # The official 11GB parent is hashed once in preflight. Per-sample loading
        # still validates the manifest checkpoint SHA via expected_reference_sha256.
        reference_checkpoint_path=None,
        expected_reference_sha256=OFFICIAL_SELF_FORCING_CHECKPOINT_SHA256,
    )
    validate_store_identity_order(
        sample_plan=sample_plan,
        teacher_store=teacher_store,
        conditional_store=conditional_store,
    )
    generator = build_fresh_generator(
        config=config,
        checkpoint_path=args.checkpoint,
        device=device,
        dtype=dtype,
    )
    require_nf_sf_full_sequence_runtime(
        config=config,
        generator=generator,
        objective_mode=objective_mode,
    )
    optimizer, optimizer_summary = build_optimizer(
        generator,
        objective_mode=objective_mode,
        backbone_lr=args.backbone_lr,
        patch_embedding_lr=args.patch_embedding_lr,
        mcp_lr=args.mcp_lr,
        weight_decay=args.weight_decay,
    )
    train_rng = make_generator(args.train_seed, device)
    validation_base_rng = make_generator(args.validation_seed, device)
    metadata = build_step0_metadata(
        args=args,
        config=config,
        sample_plan=sample_plan,
        git_sha=current_git_sha,
        preflight_report=preflight_report,
    )
    metadata["optimizer"] = optimizer_summary

    scheduler_main = make_flow_scheduler(DEFAULT_S_MAIN)
    scheduler_mcp = make_flow_scheduler(DEFAULT_S_MCP)
    prepare_output_dir(args.output_dir, resume=args.resume_checkpoint is not None)
    write_m4_json(metadata, args.output_dir / "run_metadata.json")

    resume_payload = None
    start_step = 1
    if args.resume_checkpoint is not None:
        resume_payload = load_full_sequence_checkpoint(
            args.resume_checkpoint,
            generator=generator,
            optimizer=optimizer,
            train_rng=train_rng,
            validation_base_rng=validation_base_rng,
            objective_mode=objective_mode,
            expected_resume_checkpoint_sha256=args.expected_resume_checkpoint_sha256,
            expected_git_sha=current_git_sha,
            expected_resolved_config=metadata["resolved_config"],
            expected_provenance=metadata["provenance"],
            expected_optimizer_contract=optimizer_contract(optimizer),
        )
        start_step = int(resume_payload["global_step"]) + 1
    elif not args.engineering_smoke_one_step:
        step0_checkpoint = save_full_sequence_checkpoint(
            output_dir=args.output_dir,
            generator=generator,
            optimizer=optimizer,
            global_step=0,
            train_rng=train_rng,
            validation_base_rng=validation_base_rng,
            validation_seed=int(args.validation_seed),
            sample_plan=sample_plan,
            resolved_config=metadata["resolved_config"],
            provenance=metadata["provenance"],
            git_sha=current_git_sha,
            reference_checkpoint_path=args.checkpoint,
            objective_mode=objective_mode,
            smoke=bool(args.engineering_smoke_one_step),
            extra_metadata={"step0": True},
        )

    target_global_step = int(cli["target_global_step"])
    checkpoint_records = []
    if args.resume_checkpoint is None and not args.engineering_smoke_one_step:
        checkpoint_records.append(checkpoint_record_from_path(step0_checkpoint))
    validation_summaries = []
    metrics_path = args.output_dir / "metrics.jsonl"
    train_record_count = 0
    final_train_record = None
    memory_maxima: dict[str, int] = {}
    checkpoint_steps = tuple(int(step) for step in cli["checkpoint_steps"])
    validation_steps = tuple(int(step) for step in cli["validation_steps"])
    memory_log_interval = int(args.memory_log_interval)
    if should_run_step0_validation(
        resume_checkpoint=args.resume_checkpoint,
        validation_steps=validation_steps,
    ):
        validation0 = run_full_sequence_validation(
            generator=generator,
            scheduler_main=scheduler_main,
            scheduler_mcp=scheduler_mcp,
            teacher_store=teacher_store,
            conditional_store=conditional_store,
            validation_seed=int(args.validation_seed),
            train_rng=train_rng,
            objective_mode=objective_mode,
            device=device,
            dtype=dtype,
            global_step=0,
        )
        validation0_path = args.output_dir / "validation_step000000.json"
        write_m4_json(validation0, validation0_path)
        validation_summaries.append(
            compact_validation_summary(validation0, path=validation0_path)
        )
    for step in range(start_step, target_global_step + 1):
        started = time.perf_counter()
        full_gradient_audit = should_run_full_gradient_audit(
            step,
            smoke=bool(args.engineering_smoke_one_step),
            checkpoint_steps=checkpoint_steps,
            validation_steps=validation_steps,
            memory_log_interval=memory_log_interval,
        )
        record = run_full_sequence_train_step(
            generator=generator,
            optimizer=optimizer,
            scheduler_main=scheduler_main,
            scheduler_mcp=scheduler_mcp,
            teacher_store=teacher_store,
            conditional_store=conditional_store,
            train_rng=train_rng,
            global_step=step,
            objective_mode=objective_mode,
            device=device,
            dtype=dtype,
            smoke=bool(args.engineering_smoke_one_step),
            full_gradient_audit=full_gradient_audit,
            structural_gate=bool(args.engineering_smoke_one_step) or step == 1,
            run_gc=should_run_cleanup_gc(
                step,
                smoke=bool(args.engineering_smoke_one_step),
                checkpoint_steps=checkpoint_steps,
                validation_steps=validation_steps,
                memory_log_interval=memory_log_interval,
            ),
            capture_memory=should_capture_memory(
                step,
                smoke=bool(args.engineering_smoke_one_step),
                checkpoint_steps=checkpoint_steps,
                validation_steps=validation_steps,
                memory_log_interval=memory_log_interval,
            ),
        )
        record["elapsed_ms"] = float((time.perf_counter() - started) * 1000.0)
        compact_record = compact_train_record(record, device=device)
        update_memory_maxima(memory_maxima, compact_record.get("memory"))
        append_jsonl(
            metrics_path,
            compact_record,
            fsync=(step % int(args.log_interval) == 0 or step == target_global_step),
        )
        train_record_count += 1
        final_train_record = compact_record
        if step % int(args.log_interval) == 0 or step == target_global_step:
            write_m4_json(compact_record, args.output_dir / f"train_step{step:06d}.json")
        if (
            not args.engineering_smoke_one_step
            and step in validation_steps
        ):
            validation = run_full_sequence_validation(
                generator=generator,
                scheduler_main=scheduler_main,
                scheduler_mcp=scheduler_mcp,
                teacher_store=teacher_store,
                conditional_store=conditional_store,
                validation_seed=int(args.validation_seed),
                train_rng=train_rng,
                objective_mode=objective_mode,
                device=device,
                dtype=dtype,
                global_step=step,
            )
            validation_path = args.output_dir / f"validation_step{step:06d}.json"
            write_m4_json(validation, validation_path)
            validation_summaries.append(
                compact_validation_summary(validation, path=validation_path)
            )
        if (not args.engineering_smoke_one_step) and step in checkpoint_steps:
            checkpoint_path = save_full_sequence_checkpoint(
                output_dir=args.output_dir,
                generator=generator,
                optimizer=optimizer,
                global_step=step,
                train_rng=train_rng,
                validation_base_rng=validation_base_rng,
                validation_seed=int(args.validation_seed),
                sample_plan=sample_plan,
                resolved_config=metadata["resolved_config"],
                provenance=metadata["provenance"],
                git_sha=current_git_sha,
                reference_checkpoint_path=args.checkpoint,
                objective_mode=objective_mode,
                smoke=False,
            )
            checkpoint_records.append(checkpoint_record_from_path(checkpoint_path))
        if args.engineering_smoke_one_step:
            smoke_checkpoint = save_full_sequence_checkpoint(
                output_dir=args.output_dir,
                generator=generator,
                optimizer=optimizer,
                global_step=step,
                train_rng=train_rng,
                validation_base_rng=validation_base_rng,
                validation_seed=int(args.validation_seed),
                sample_plan=sample_plan,
                resolved_config=metadata["resolved_config"],
                provenance=metadata["provenance"],
                git_sha=current_git_sha,
                reference_checkpoint_path=args.checkpoint,
                objective_mode=objective_mode,
                smoke=True,
                extra_metadata={"post_smoke_optimizer_step": True},
            )
            checkpoint_records.append(checkpoint_record_from_path(smoke_checkpoint))
            break

    reference_checkpoint_immutability = verify_reference_checkpoint_immutability(
        reference_checkpoint_path=args.checkpoint,
        preflight_report=preflight_report,
    )
    summary = build_training_summary(
        objective_mode=objective_mode,
        status=NON_PRODUCTION_SMOKE_TAG
        if args.engineering_smoke_one_step
        else "DONE",
        target_global_step=target_global_step,
        completed_global_step=int(final_train_record["global_step"])
        if final_train_record
        else 0,
        checkpoint_steps=checkpoint_steps,
        validation_steps=validation_steps,
        metrics_path=metrics_path,
        train_record_count=train_record_count,
        final_train_record=final_train_record,
        checkpoint_records=checkpoint_records,
        validation_summaries=validation_summaries,
        memory_maxima=memory_maxima,
        resume_payload=resume_payload,
        smoke=bool(args.engineering_smoke_one_step),
        reference_checkpoint_immutability=reference_checkpoint_immutability,
    )
    write_m4_json(summary, args.output_dir / "training_summary.json")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run_training(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
