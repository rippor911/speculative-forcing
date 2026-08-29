from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import math
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.nf_sf_full_sequence_continuation import (
    CONTINUATION_OBJECTIVE_MODE,
    load_continuation_parent_checkpoint,
    restore_continuation_state,
    rng_fingerprint,
    semantic_lock_fingerprint,
    validate_git_sha,
    validate_optimizer_contract_for_continuation,
)
from utils.nf_sf_m3 import file_sha256, tensor_sha256
from utils.nf_sf_m4 import load_m4_sample_plan
from utils.nf_sf_tensors import (
    DEFAULT_NUM_TRAIN_TIMESTEPS,
    DEFAULT_S_MAIN,
    DEFAULT_S_MCP,
    FULL_SEQUENCE_CHUNK_FRAMES,
    FULL_SEQUENCE_DEPTHS,
    FULL_SEQUENCE_FRAME_COUNT,
    FULL_SEQUENCE_NUM_CHUNKS,
    FULL_SEQUENCE_RNG_DRAW_ORDER_VERSION,
    expand_raw_chunk_timesteps,
)
from utils.nf_sf_training import (
    FULL_SEQUENCE_CHUNK_TOKENS,
    FULL_SEQUENCE_DEPTH_WEIGHTS,
    NFSFFullSequenceNoisyBatch,
    build_full_sequence_mcp_anchor_inputs,
    build_full_sequence_mcp_anchor_specs,
    collect_nf_sf_parameter_groups,
)


TINY_AB_SCHEMA = "nf_sf_tiny_mcp1_paper_fidelity_ab_v1"
TINY_AB_PLAN_SCHEMA = f"{TINY_AB_SCHEMA}_plan_v1"
TINY_AB_ARM_SCHEMA = f"{TINY_AB_SCHEMA}_arm_v1"
TINY_AB_EVAL_SCHEMA = f"{TINY_AB_SCHEMA}_evaluation_v1"
TINY_AB_SUMMARY_SCHEMA = f"{TINY_AB_SCHEMA}_summary_v1"
SUPPORT_PAPER_FIDELITY_MCP1 = "SUPPORT_PAPER_FIDELITY_MCP1"
NO_SUPPORT = "NO_SUPPORT"
INCONCLUSIVE = "INCONCLUSIVE"

PARENT_STEP = 6500
TARGET_TINY_STEP = 200
TRAIN_IDENTITY_COUNT = 8
VALIDATION_IDENTITY_COUNT = 8
RAW_TIMESTEPS = (999, 750, 500, 250)
NOISE_INDEX = 0
ANCHOR_INDEX = 1
CURRENT_CHUNK_INDEX = 1
FUTURE_CHUNK_INDEX = 2
MCP_DEPTH = 1
TAP_LAYERS = (3, 11, 19, 29)
MCP_BLOCKS_PER_DEPTH = 3
CANONICAL_CONFIG_PATH = (ROOT / "configs" / "self_forcing_dmd_mcp.yaml").resolve()
PARENT_CHECKPOINT = Path(
    "/home/dataset-assist-0/luojy/efficiency/rippor/experiment_outputs/"
    "nf_sf_full_sequence_continuation/c3f8988/"
    "continuation_5000_6500_20260817_165916/"
    "checkpoint_step006500.pt"
)
PARENT_CHECKPOINT_SHA256 = (
    "9ef57cb2d3e5f20b244129317af4a0e1d2b1c810ba65ec970892e60ccbd34f4f"
)
PARENT_GIT_SHA = "c3f89888bf6da31b48650f0a680dd6534943f56f"

TRAINABLE_EXPECTED_NONZERO_GRAD = (
    "backbone",
    "patch_embedding",
    "mcp_fusion",
    "mcp_depth1",
)
EXPECTED_NO_GRAD = ("mcp_depth2", "mcp_depth3", "main_final_head")


@dataclass(frozen=True)
class TinyForwardResult:
    outputs: Any
    loss: torch.Tensor
    mcp1_anchor1_mse: torch.Tensor
    state_proof: dict[str, Any]


def _trainer_helpers() -> Mapping[str, Any]:
    from scripts import train_nf_sf_full_sequence_next_forcing as trainer

    return {
        "append_jsonl": trainer.append_jsonl,
        "assert_global_rng_equal": trainer.assert_global_rng_equal,
        "build_fresh_generator": trainer.build_fresh_generator,
        "build_optimizer": trainer.build_optimizer,
        "capture_global_rng_state": trainer.capture_global_rng_state,
        "dtype_from_arg": trainer.dtype_from_arg,
        "make_flow_scheduler": trainer.make_flow_scheduler,
        "merge_config": trainer.merge_config,
        "move_tensors_to_cpu": trainer.move_tensors_to_cpu,
        "optimizer_contract": trainer.optimizer_contract,
        "prepare_output_dir": trainer.prepare_output_dir,
        "target_latent_from_sample": trainer.target_latent_from_sample,
        "validate_sample_plan_contract": trainer.validate_sample_plan_contract,
        "validate_store_identity_order": trainer.validate_store_identity_order,
        "write_m4_json": trainer.write_m4_json,
    }


def current_git_head() -> str:
    value = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        stderr=subprocess.DEVNULL,
        text=True,
    ).strip()
    return validate_git_sha(value, name="runtime_git_sha")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="NF-SF tiny MCP1 paper-fidelity A/B runner."
    )
    parser.add_argument("--execute_real_run", action="store_true")
    parser.add_argument("--config", type=Path, default=Path("configs/self_forcing_dmd_mcp.yaml"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/self_forcing_dmd.pt"))
    parser.add_argument("--parent_checkpoint", type=Path, default=PARENT_CHECKPOINT)
    parser.add_argument(
        "--expected_parent_checkpoint_sha256",
        default=PARENT_CHECKPOINT_SHA256,
    )
    parser.add_argument(
        "--expected_parent_global_step",
        type=int,
        choices=(PARENT_STEP,),
        default=PARENT_STEP,
    )
    parser.add_argument(
        "--expected_parent_checkpoint_git_sha",
        default=PARENT_GIT_SHA,
    )
    parser.add_argument("--expected_runtime_git_sha")
    parser.add_argument("--sample_plan", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--dataset_root", type=Path)
    parser.add_argument("--conditionals_artifact", type=Path)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bf16",), default="bf16")
    parser.add_argument("--log_interval", type=int, default=25)
    return parser.parse_args(argv)


def arm_specs() -> tuple[dict[str, Any], ...]:
    return (
        {
            "name": "canonical",
            "path": "canonical",
            "mcp_path_kind": "canonical_target_only",
            "paper_fidelity_mcp1_mask": False,
        },
        {
            "name": "paper_fidelity",
            "path": "paper_fidelity",
            "mcp_path_kind": "paper_fidelity_clean_residual_mask",
            "paper_fidelity_mcp1_mask": True,
        },
    )


def decision_thresholds() -> dict[str, Any]:
    return {
        "schema": f"{TINY_AB_SCHEMA}_decision_thresholds_v1",
        "support": {
            "validation_relative_improvement_min": 0.10,
            "identity_win_rate_min": 0.75,
            "identity_win_count_min": 6,
            "raw_win_count_min": 3,
            "all_loss_and_grad_finite": True,
        },
        "no_support": {
            "low_improvement_max": 0.05,
            "low_identity_win_rate_max_exclusive": 0.60,
            "treatment_degradation_min": 0.05,
        },
    }


def _json_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _identity_entries(sample_plan: Mapping[str, Any], split: str) -> tuple[dict[str, Any], ...]:
    samples = sample_plan.get("samples")
    if isinstance(samples, Mapping) and isinstance(samples.get(split), Sequence):
        return tuple(
            {
                "identity": str(entry["identity"]),
                "split": str(entry.get("split", split)),
                "split_index": int(entry.get("split_index", index)),
                "sample_index": int(entry["sample_index"]) if entry.get("sample_index") is not None else None,
                "sample_id": None if entry.get("sample_id") is None else str(entry.get("sample_id")),
            }
            for index, entry in enumerate(samples[split])
        )
    key = f"{split}_sample_identities"
    identities = tuple(str(value) for value in sample_plan[key])
    return tuple(
        {
            "identity": identity,
            "split": split,
            "split_index": index,
            "sample_index": None,
            "sample_id": None,
        }
        for index, identity in enumerate(identities)
    )


def evenly_spaced_positions(total_count: int, selected_count: int) -> tuple[int, ...]:
    total = int(total_count)
    count = int(selected_count)
    if count <= 0:
        raise ValueError("selected_count must be positive")
    if total < count:
        raise ValueError("not enough identities for fixed tiny selection")
    if count == 1:
        return (0,)
    positions = tuple(round(index * (total - 1) / (count - 1)) for index in range(count))
    if len(set(positions)) != len(positions):
        raise RuntimeError("fixed tiny identity selection produced duplicate positions")
    return positions


def _state_specs_for_split(
    sample_plan: Mapping[str, Any],
    *,
    split: str,
    selected_count: int,
) -> tuple[dict[str, Any], ...]:
    entries = _identity_entries(sample_plan, split)
    positions = evenly_spaced_positions(len(entries), selected_count)
    states: list[dict[str, Any]] = []
    for selected_order, position in enumerate(positions):
        entry = entries[position]
        for raw_order, raw_timestep in enumerate(RAW_TIMESTEPS):
            states.append(
                {
                    "state_index": len(states),
                    "state_id": (
                        f"{split}_sel{selected_order:02d}_pos{position:04d}_"
                        f"raw{int(raw_timestep):03d}_noise{NOISE_INDEX}"
                    ),
                    "split": split,
                    "selection_policy": "uniform_even_split_positions_identity_major_raw_order",
                    "selected_identity_order": selected_order,
                    "selected_split_position": int(position),
                    "identity": str(entry["identity"]),
                    "split_index": int(entry["split_index"]),
                    "sample_index": entry["sample_index"],
                    "sample_id": entry["sample_id"],
                    "raw_order": raw_order,
                    "raw_timestep": int(raw_timestep),
                    "noise_index": NOISE_INDEX,
                    "anchor_index": ANCHOR_INDEX,
                    "current_chunk_index": CURRENT_CHUNK_INDEX,
                    "future_chunk_index": FUTURE_CHUNK_INDEX,
                    "mcp_depth": MCP_DEPTH,
                }
            )
    return tuple(states)


def build_tiny_state_plan(sample_plan: Mapping[str, Any]) -> dict[str, Any]:
    train_states = _state_specs_for_split(
        sample_plan,
        split="train",
        selected_count=TRAIN_IDENTITY_COUNT,
    )
    validation_states = _state_specs_for_split(
        sample_plan,
        split="validation",
        selected_count=VALIDATION_IDENTITY_COUNT,
    )
    train_identities = {state["identity"] for state in train_states}
    validation_identities = {state["identity"] for state in validation_states}
    if train_identities & validation_identities:
        raise RuntimeError("tiny A/B train and validation identities overlap")
    plan = {
        "schema": f"{TINY_AB_SCHEMA}_state_plan_v1",
        "selection_policy": "uniform_even_split_positions_identity_major_raw_order",
        "raw_timesteps": list(RAW_TIMESTEPS),
        "noise_index": NOISE_INDEX,
        "anchor_index": ANCHOR_INDEX,
        "current_chunk_index": CURRENT_CHUNK_INDEX,
        "future_chunk_index": FUTURE_CHUNK_INDEX,
        "mcp_depth": MCP_DEPTH,
        "train_identity_count": TRAIN_IDENTITY_COUNT,
        "validation_identity_count": VALIDATION_IDENTITY_COUNT,
        "train_state_count": len(train_states),
        "validation_state_count": len(validation_states),
        "train_states": [dict(state) for state in train_states],
        "validation_states": [dict(state) for state in validation_states],
        "train_validation_identity_disjoint": True,
    }
    plan["state_plan_fingerprint_sha256"] = _json_sha256(plan)
    validate_tiny_state_plan(plan)
    return plan


def tiny_update_schedule(state_plan: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    train_states = tuple(state_plan["train_states"])
    if len(train_states) != 32:
        raise RuntimeError("tiny A/B update schedule requires 32 train states")
    return tuple(
        {
            "tiny_step": step,
            "state_index": int(train_states[(step - 1) % len(train_states)]["state_index"]),
            "state_id": str(train_states[(step - 1) % len(train_states)]["state_id"]),
            "identity": str(train_states[(step - 1) % len(train_states)]["identity"]),
            "raw_timestep": int(train_states[(step - 1) % len(train_states)]["raw_timestep"]),
            "noise_index": int(train_states[(step - 1) % len(train_states)]["noise_index"]),
        }
        for step in range(1, TARGET_TINY_STEP + 1)
    )


def build_tiny_ab_plan(sample_plan: Mapping[str, Any]) -> dict[str, Any]:
    state_plan = build_tiny_state_plan(sample_plan)
    schedule = tiny_update_schedule(state_plan)
    plan = {
        "schema": TINY_AB_PLAN_SCHEMA,
        "diagnostic": "tiny_mcp1_paper_fidelity_ab",
        "diagnostic_only": True,
        "parent_checkpoint": {
            "path": str(PARENT_CHECKPOINT),
            "sha256": PARENT_CHECKPOINT_SHA256,
            "global_step": PARENT_STEP,
            "git_sha": PARENT_GIT_SHA,
        },
        "sample_data_provenance": {
            "sample_plan_sha256": sample_plan.get("sample_plan_sha256"),
            "train_split_count": len(_identity_entries(sample_plan, "train")),
            "validation_split_count": len(_identity_entries(sample_plan, "validation")),
            "state_selection_policy": state_plan["selection_policy"],
            "noise_source": "formal_sample_source_noise_index0",
        },
        "arms": list(arm_specs()),
        "only_variable": "paper_fidelity_mcp1_mask",
        "objective": {
            "loss": "MCP1 exact Flow Matching MSE for anchor1 only",
            "anchor_index": ANCHOR_INDEX,
            "current_chunk_index": CURRENT_CHUNK_INDEX,
            "future_chunk_index": FUTURE_CHUNK_INDEX,
            "main_loss": False,
            "mcp2_loss": False,
            "mcp3_loss": False,
            "auxiliary_loss": False,
        },
        "canonical_locked_variables": {
            "main_shift": DEFAULT_S_MAIN,
            "mcp_shift": DEFAULT_S_MCP,
            "taps": list(TAP_LAYERS),
            "shared_patch_embedding": True,
            "depth_weights_not_used_for_backward": list(FULL_SEQUENCE_DEPTH_WEIGHTS),
            "chunk_tokens": FULL_SEQUENCE_CHUNK_TOKENS,
            "chunk_frames": FULL_SEQUENCE_CHUNK_FRAMES,
            "num_chunks": FULL_SEQUENCE_NUM_CHUNKS,
            "rng_draw_order_version": FULL_SEQUENCE_RNG_DRAW_ORDER_VERSION,
        },
        "state_plan": state_plan,
        "update_count": TARGET_TINY_STEP,
        "update_schedule": [dict(item) for item in schedule],
        "early_stop_enabled": False,
        "evaluation_steps": [0, TARGET_TINY_STEP],
        "decision_thresholds": decision_thresholds(),
    }
    plan["plan_fingerprint_sha256"] = _json_sha256(plan)
    validate_tiny_ab_plan(plan)
    return plan


def validate_tiny_state_plan(state_plan: Mapping[str, Any]) -> dict[str, Any]:
    train_states = tuple(state_plan.get("train_states", ()))
    validation_states = tuple(state_plan.get("validation_states", ()))
    if len(train_states) != TRAIN_IDENTITY_COUNT * len(RAW_TIMESTEPS):
        raise RuntimeError("tiny A/B train state count must be 32")
    if len(validation_states) != VALIDATION_IDENTITY_COUNT * len(RAW_TIMESTEPS):
        raise RuntimeError("tiny A/B validation state count must be 32")
    train_identities = {str(state["identity"]) for state in train_states}
    validation_identities = {str(state["identity"]) for state in validation_states}
    if len(train_identities) != TRAIN_IDENTITY_COUNT:
        raise RuntimeError("tiny A/B train identity count must be 8")
    if len(validation_identities) != VALIDATION_IDENTITY_COUNT:
        raise RuntimeError("tiny A/B validation identity count must be 8")
    if train_identities & validation_identities:
        raise RuntimeError("tiny A/B train and validation identities must be disjoint")
    for states, split in ((train_states, "train"), (validation_states, "validation")):
        for index, state in enumerate(states):
            if int(state["state_index"]) != index:
                raise RuntimeError(f"{split} state_index must match state order")
            if int(state["noise_index"]) != NOISE_INDEX:
                raise RuntimeError("tiny A/B only supports noise_index=0")
            if int(state["anchor_index"]) != ANCHOR_INDEX:
                raise RuntimeError("tiny A/B only supports MCP1 anchor1")
    return {"status": "PASS"}


def validate_tiny_ab_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    if plan.get("schema") != TINY_AB_PLAN_SCHEMA:
        raise RuntimeError("tiny A/B plan schema mismatch")
    if int(plan.get("update_count", -1)) != TARGET_TINY_STEP:
        raise RuntimeError("tiny A/B must run exactly 200 updates")
    if bool(plan.get("early_stop_enabled", True)):
        raise RuntimeError("tiny A/B early stop must be disabled")
    if plan.get("only_variable") != "paper_fidelity_mcp1_mask":
        raise RuntimeError("tiny A/B only variable must be paper_fidelity_mcp1_mask")
    arms = tuple(plan.get("arms", ()))
    if tuple(arm["paper_fidelity_mcp1_mask"] for arm in arms) != (False, True):
        raise RuntimeError("tiny A/B arm flags must be false/true")
    validate_tiny_state_plan(plan["state_plan"])
    schedule = tuple(plan.get("update_schedule", ()))
    if len(schedule) != TARGET_TINY_STEP:
        raise RuntimeError("tiny A/B update schedule length must be 200")
    train_states = tuple(plan["state_plan"]["train_states"])
    for step, item in enumerate(schedule, start=1):
        expected = train_states[(step - 1) % len(train_states)]
        if int(item["tiny_step"]) != step:
            raise RuntimeError("tiny A/B update schedule step mismatch")
        if str(item["state_id"]) != str(expected["state_id"]):
            raise RuntimeError("tiny A/B update schedule state mismatch")
    return {"status": "PASS"}


def _video_add_noise(scheduler: Any, clean: torch.Tensor, noise: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
    flat = scheduler.add_noise(
        clean.flatten(0, 1),
        noise.flatten(0, 1),
        timestep.flatten(0, 1),
    )
    return flat.unflatten(0, clean.shape[:2])


def _video_training_target(scheduler: Any, clean: torch.Tensor, noise: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
    flat = scheduler.training_target(
        clean.flatten(0, 1),
        noise.flatten(0, 1),
        timestep.flatten(0, 1),
    )
    return flat.unflatten(0, clean.shape[:2])


def _anchor_add_noise(scheduler: Any, clean: torch.Tensor, noise: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
    b, count, frames = clean.shape[:3]
    flat = scheduler.add_noise(
        clean.flatten(0, 2),
        noise.flatten(0, 2),
        timestep.flatten(0, 2),
    )
    return flat.unflatten(0, (b, count, frames))


def _anchor_training_target(scheduler: Any, clean: torch.Tensor, noise: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
    b, count, frames = clean.shape[:3]
    flat = scheduler.training_target(
        clean.flatten(0, 2),
        noise.flatten(0, 2),
        timestep.flatten(0, 2),
    )
    return flat.unflatten(0, (b, count, frames))


def _depth_chunks(latent: torch.Tensor, depth: int) -> torch.Tensor:
    chunks = latent.unflatten(1, (FULL_SEQUENCE_NUM_CHUNKS, FULL_SEQUENCE_CHUNK_FRAMES))
    return chunks[:, int(depth) :]


def build_fixed_raw_noisy_batch(
    *,
    clean_target: torch.Tensor,
    source_noise: torch.Tensor,
    raw_timestep: int,
    scheduler_main: Any,
    scheduler_mcp: Any,
) -> NFSFFullSequenceNoisyBatch:
    if tuple(clean_target.shape) != tuple(source_noise.shape):
        raise ValueError("clean_target and source_noise must have identical shapes")
    if clean_target.ndim != 5 or int(clean_target.shape[1]) != FULL_SEQUENCE_FRAME_COUNT:
        raise ValueError("fixed tiny A/B states require [B, 21, C, H, W]")
    raw = int(raw_timestep)
    if raw not in RAW_TIMESTEPS:
        raise ValueError("raw_timestep is not in the preregistered tiny grid")
    batch = int(clean_target.shape[0])
    raw_main = torch.full(
        (batch, FULL_SEQUENCE_NUM_CHUNKS),
        raw,
        device=clean_target.device,
        dtype=torch.int64,
    )
    timestep_main = expand_raw_chunk_timesteps(
        raw_main,
        chunk_frames=FULL_SEQUENCE_CHUNK_FRAMES,
        shift=DEFAULT_S_MAIN,
        num_train_timesteps=DEFAULT_NUM_TRAIN_TIMESTEPS,
    ).flatten(1, 2)
    noisy_main = _video_add_noise(
        scheduler_main,
        clean_target,
        source_noise,
        timestep_main,
    )
    target_flow_main = _video_training_target(
        scheduler_main,
        clean_target,
        source_noise,
        timestep_main,
    )

    noisy_depths = []
    target_flow_depths = []
    epsilon_depths = []
    raw_timestep_depths = []
    timestep_depths = []
    for depth in FULL_SEQUENCE_DEPTHS:
        target = _depth_chunks(clean_target, depth)
        epsilon = _depth_chunks(source_noise, depth)
        count = int(target.shape[1])
        raw_depth = torch.full(
            (batch, count),
            raw,
            device=clean_target.device,
            dtype=torch.int64,
        )
        timestep_depth = expand_raw_chunk_timesteps(
            raw_depth,
            chunk_frames=FULL_SEQUENCE_CHUNK_FRAMES,
            shift=DEFAULT_S_MCP,
            num_train_timesteps=DEFAULT_NUM_TRAIN_TIMESTEPS,
        )
        noisy_depths.append(
            _anchor_add_noise(scheduler_mcp, target, epsilon, timestep_depth)
        )
        target_flow_depths.append(
            _anchor_training_target(scheduler_mcp, target, epsilon, timestep_depth)
        )
        epsilon_depths.append(epsilon)
        raw_timestep_depths.append(raw_depth)
        timestep_depths.append(timestep_depth)

    return NFSFFullSequenceNoisyBatch(
        clean_target=clean_target,
        noisy_main=noisy_main,
        target_flow_main=target_flow_main,
        epsilon_main=source_noise,
        raw_timestep_main=raw_main,
        timestep_main=timestep_main,
        noisy_mcp_depths=tuple(noisy_depths),
        target_flow_mcp_depths=tuple(target_flow_depths),
        epsilon_mcp_depths=tuple(epsilon_depths),
        raw_timestep_mcp_depths=tuple(raw_timestep_depths),
        timestep_mcp_depths=tuple(timestep_depths),
        anchor_specs=build_full_sequence_mcp_anchor_specs(),
    )


def _tensor_record(tensor: torch.Tensor) -> dict[str, Any]:
    detached = tensor.detach()
    return {
        "shape": [int(dim) for dim in detached.shape],
        "dtype": str(detached.dtype),
        "finite": bool(torch.isfinite(detached.float()).all().item()),
        "sha256": tensor_sha256(detached.cpu()),
    }


def anchor1_state_proof(
    noisy_batch: NFSFFullSequenceNoisyBatch,
    *,
    state_spec: Mapping[str, Any],
) -> dict[str, Any]:
    anchor_target = noisy_batch.clean_target[
        :,
        FUTURE_CHUNK_INDEX * FULL_SEQUENCE_CHUNK_FRAMES : (FUTURE_CHUNK_INDEX + 1)
        * FULL_SEQUENCE_CHUNK_FRAMES,
    ]
    return {
        "state_id": str(state_spec["state_id"]),
        "identity": str(state_spec["identity"]),
        "raw_timestep": int(state_spec["raw_timestep"]),
        "noise_index": int(state_spec["noise_index"]),
        "anchor_index": ANCHOR_INDEX,
        "current_chunk_index": CURRENT_CHUNK_INDEX,
        "future_chunk_index": FUTURE_CHUNK_INDEX,
        "main_warped_timestep_sha256": tensor_sha256(noisy_batch.timestep_main.detach().cpu()),
        "mcp_warped_timestep_sha256": tensor_sha256(
            noisy_batch.timestep_mcp_depths[0][:, ANCHOR_INDEX].detach().cpu()
        ),
        "current_noise_sha256": tensor_sha256(
            noisy_batch.epsilon_main[
                :,
                CURRENT_CHUNK_INDEX
                * FULL_SEQUENCE_CHUNK_FRAMES : (CURRENT_CHUNK_INDEX + 1)
                * FULL_SEQUENCE_CHUNK_FRAMES,
            ]
            .detach()
            .cpu()
        ),
        "future_noise_sha256": tensor_sha256(
            noisy_batch.epsilon_mcp_depths[0][:, ANCHOR_INDEX].detach().cpu()
        ),
        "future_target_sha256": tensor_sha256(anchor_target.detach().cpu()),
        "noisy_future_sha256": tensor_sha256(
            noisy_batch.noisy_mcp_depths[0][:, ANCHOR_INDEX].detach().cpu()
        ),
        "exact_fm_target_sha256": tensor_sha256(
            noisy_batch.target_flow_mcp_depths[0][:, ANCHOR_INDEX].detach().cpu()
        ),
    }


def _output_field(outputs: Any, key: str) -> Any:
    if isinstance(outputs, Mapping):
        return outputs[key]
    return getattr(outputs, key)


def run_tiny_anchor1_forward_loss(
    generator: Any,
    *,
    conditional_dict: Mapping[str, Any],
    noisy_batch: NFSFFullSequenceNoisyBatch,
    state_spec: Mapping[str, Any],
    paper_fidelity_mcp1_mask: bool,
) -> TinyForwardResult:
    outputs = generator.forward_full_sequence_next_forcing(
        noisy_image_or_video=noisy_batch.noisy_main,
        clean_x=noisy_batch.clean_target,
        conditional_dict=dict(conditional_dict),
        timestep_main=noisy_batch.timestep_main,
        mcp_anchor_inputs=build_full_sequence_mcp_anchor_inputs(noisy_batch),
        paper_fidelity_mcp1_mask=bool(paper_fidelity_mcp1_mask),
    )
    mcp_flow_preds_by_depth = tuple(_output_field(outputs, "mcp_flow_preds_by_depth"))
    if len(mcp_flow_preds_by_depth) != 3:
        raise RuntimeError("tiny A/B forward must keep canonical MCP1/2/3 outputs")
    pred = mcp_flow_preds_by_depth[0][:, ANCHOR_INDEX]
    target = noisy_batch.target_flow_mcp_depths[0][:, ANCHOR_INDEX]
    if tuple(pred.shape) != tuple(target.shape):
        raise RuntimeError("MCP1 anchor1 prediction/target shape mismatch")
    loss = F.mse_loss(pred.float(), target.float(), reduction="mean")
    return TinyForwardResult(
        outputs=outputs,
        loss=loss,
        mcp1_anchor1_mse=loss,
        state_proof=anchor1_state_proof(noisy_batch, state_spec=state_spec),
    )


def _grad_group_summary(
    group_name: str,
    named_params: Sequence[tuple[str, torch.nn.Parameter]],
) -> dict[str, Any]:
    grad_tensors = 0
    finite = True
    sq_norm = 0.0
    max_abs = 0.0
    for _name, param in named_params:
        if param.grad is None:
            continue
        grad = param.grad.detach().float()
        grad_tensors += 1
        grad_finite = bool(torch.isfinite(grad).all().item())
        finite = finite and grad_finite
        sq_norm += float(grad.square().sum().item())
        max_abs = max(max_abs, float(grad.abs().max().item()))
    return {
        "group": group_name,
        "parameter_tensors": int(len(named_params)),
        "grad_tensors": int(grad_tensors),
        "all_finite": bool(finite),
        "aggregate_grad_norm": float(sq_norm ** 0.5),
        "max_abs_grad": float(max_abs),
    }


def _named_main_final_head_parameters(generator: Any) -> tuple[tuple[str, torch.nn.Parameter], ...]:
    head = getattr(getattr(generator, "model", None), "head", None)
    if head is None or not hasattr(head, "named_parameters"):
        return ()
    return tuple((f"model.head.{name}", param) for name, param in head.named_parameters())


def gradient_group_report(generator: Any) -> dict[str, dict[str, Any]]:
    groups = {
        name: tuple(named_params)
        for name, named_params in collect_nf_sf_parameter_groups(generator).items()
    }
    groups["main_final_head"] = _named_main_final_head_parameters(generator)
    return {
        name: _grad_group_summary(name, named_params)
        for name, named_params in groups.items()
    }


def validate_gradient_group_report(report: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    failures = []
    for name, item in report.items():
        if int(item.get("grad_tensors", 0)) > 0 and not bool(item.get("all_finite", False)):
            failures.append(f"{name}:nonfinite")
    for name in TRAINABLE_EXPECTED_NONZERO_GRAD:
        item = report.get(name)
        if item is None or float(item.get("aggregate_grad_norm", 0.0)) <= 0.0:
            failures.append(f"{name}:expected_nonzero_grad")
    for name in EXPECTED_NO_GRAD:
        item = report.get(name)
        if item is None:
            failures.append(f"{name}:missing")
        elif int(item.get("grad_tensors", 0)) != 0:
            failures.append(f"{name}:expected_no_grad")
    if failures:
        raise RuntimeError("tiny A/B gradient contract failed: " + ", ".join(failures))
    return {"status": "PASS"}


def _sha256_tensor_with_meta(tensor: torch.Tensor) -> str:
    detached = tensor.detach().cpu().contiguous()
    payload = hashlib.sha256()
    payload.update(str(detached.dtype).encode("utf-8"))
    payload.update(str([int(dim) for dim in detached.shape]).encode("utf-8"))
    payload.update(detached.reshape(-1).view(torch.uint8).numpy().tobytes())
    return payload.hexdigest()


def parameter_group_sha256_report(generator: Any) -> dict[str, Any]:
    groups = {
        name: tuple(named_params)
        for name, named_params in collect_nf_sf_parameter_groups(generator).items()
    }
    groups["main_final_head"] = _named_main_final_head_parameters(generator)
    records = {}
    for group_name, named_params in groups.items():
        entries = [
            {
                "name": str(name),
                "shape": [int(dim) for dim in param.shape],
                "dtype": str(param.dtype),
                "requires_grad": bool(param.requires_grad),
                "sha256": _sha256_tensor_with_meta(param),
            }
            for name, param in named_params
        ]
        group_sha = hashlib.sha256(
            "\n".join(f"{item['name']}:{item['sha256']}" for item in entries).encode("utf-8")
        ).hexdigest()
        records[group_name] = {
            "parameter_tensors": len(entries),
            "sha256": group_sha,
            "parameters": entries,
        }
    aggregate = hashlib.sha256(
        "\n".join(
            f"{name}:{records[name]['sha256']}" for name in sorted(records)
        ).encode("utf-8")
    ).hexdigest()
    return {
        "aggregate_sha256": aggregate,
        "groups": records,
    }


def updated_group_report(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
) -> dict[str, Any]:
    before_groups = before["groups"]
    after_groups = after["groups"]
    updated = []
    unchanged = []
    for group_name, before_item in before_groups.items():
        if str(before_item["sha256"]) == str(after_groups[group_name]["sha256"]):
            unchanged.append(group_name)
        else:
            updated.append(group_name)
    forbidden = [
        group for group in ("mcp_depth2", "mcp_depth3", "main_final_head")
        if group in updated
    ]
    required_missing = [
        group for group in ("backbone", "patch_embedding", "mcp_fusion", "mcp_depth1")
        if group not in updated
    ]
    return {
        "updated_groups": updated,
        "unchanged_groups": unchanged,
        "forbidden_updated_groups": forbidden,
        "required_update_groups_missing": required_missing,
        "pass": not forbidden and not required_missing,
    }


def finite_parameter_report(generator: Any) -> dict[str, Any]:
    nonfinite = []
    for name, param in generator.named_parameters():
        if not bool(torch.isfinite(param.detach().float()).all().item()):
            nonfinite.append(str(name))
    return {
        "status": "PASS" if not nonfinite else "FAIL_NONFINITE_PARAMETER",
        "nonfinite_parameter_names": nonfinite,
    }


def optimizer_fingerprint(optimizer: torch.optim.Optimizer) -> str:
    state = optimizer.state_dict()
    entries: list[str] = []
    for index, group in enumerate(state.get("param_groups", ())):
        entries.append(
            "group:"
            + json.dumps(
                {
                    "index": index,
                    "name": group.get("name"),
                    "lr": group.get("lr"),
                    "weight_decay": group.get("weight_decay"),
                    "params": list(group.get("params", ())),
                },
                sort_keys=True,
            )
        )
    for key, value in sorted(state.get("state", {}).items(), key=lambda item: str(item[0])):
        entries.append(f"state:{key}")
        if isinstance(value, Mapping):
            for state_key, state_value in sorted(value.items(), key=lambda item: str(item[0])):
                if torch.is_tensor(state_value):
                    entries.append(
                        f"{state_key}:tensor:{list(state_value.shape)}:"
                        f"{state_value.dtype}:{_sha256_tensor_with_meta(state_value)}"
                    )
                else:
                    entries.append(f"{state_key}:{state_value!r}")
    return hashlib.sha256("\n".join(entries).encode("utf-8")).hexdigest()


def run_tiny_update_on_batch(
    *,
    generator: Any,
    optimizer: torch.optim.Optimizer,
    conditional_dict: Mapping[str, Any],
    noisy_batch: NFSFFullSequenceNoisyBatch,
    state_spec: Mapping[str, Any],
    paper_fidelity_mcp1_mask: bool,
) -> dict[str, Any]:
    optimizer.zero_grad(set_to_none=True)
    forward = run_tiny_anchor1_forward_loss(
        generator,
        conditional_dict=conditional_dict,
        noisy_batch=noisy_batch,
        state_spec=state_spec,
        paper_fidelity_mcp1_mask=paper_fidelity_mcp1_mask,
    )
    if not bool(torch.isfinite(forward.loss.detach().float()).all().item()):
        raise RuntimeError("tiny A/B non-finite anchor1 loss")
    forward.loss.backward()
    gradient_report = gradient_group_report(generator)
    gradient_gate = validate_gradient_group_report(gradient_report)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return {
        "loss": float(forward.loss.detach().float().item()),
        "finite": True,
        "state_proof": forward.state_proof,
        "gradient_report": gradient_report,
        "gradient_gate": gradient_gate,
    }


@torch.no_grad()
def evaluate_tiny_states(
    *,
    generator: Any,
    sample_store: Any,
    conditional_store: Any,
    states: Sequence[Mapping[str, Any]],
    scheduler_main: Any,
    scheduler_mcp: Any,
    device: torch.device,
    dtype: torch.dtype,
    paper_fidelity_mcp1_mask: bool,
    target_latent_from_sample: Any,
) -> dict[str, Any]:
    was_training = bool(getattr(generator, "training", False))
    generator.eval()
    records = []
    try:
        for state in states:
            with sample_store.acquire(str(state["identity"])) as sample:
                with conditional_store.acquire(str(state["identity"])) as conditional_cpu:
                    clean_target = target_latent_from_sample(sample).to(
                        device=device,
                        dtype=dtype,
                    )
                    source_noise = sample.source_noise.to(device=device, dtype=dtype)
                    conditional = _move_tensors_to_device(
                        conditional_cpu,
                        device=device,
                        floating_dtype=dtype,
                    )
                    noisy_batch = build_fixed_raw_noisy_batch(
                        clean_target=clean_target,
                        source_noise=source_noise,
                        raw_timestep=int(state["raw_timestep"]),
                        scheduler_main=scheduler_main,
                        scheduler_mcp=scheduler_mcp,
                    )
                    forward = run_tiny_anchor1_forward_loss(
                        generator,
                        conditional_dict=conditional,
                        noisy_batch=noisy_batch,
                        state_spec=state,
                        paper_fidelity_mcp1_mask=paper_fidelity_mcp1_mask,
                    )
                    records.append(
                        {
                            "state_id": str(state["state_id"]),
                            "state_index": int(state["state_index"]),
                            "identity": str(state["identity"]),
                            "raw_timestep": int(state["raw_timestep"]),
                            "noise_index": int(state["noise_index"]),
                            "mcp1_anchor1_mse": float(
                                forward.loss.detach().float().item()
                            ),
                            "finite": bool(
                                torch.isfinite(forward.loss.detach().float()).all().item()
                            ),
                            "state_proof": forward.state_proof,
                        }
                    )
    finally:
        generator.train(was_training)
    aggregate = aggregate_eval_records(records)
    return {
        "schema": TINY_AB_EVAL_SCHEMA,
        "state_count": len(records),
        "records": records,
        "aggregate": aggregate,
    }


def _move_tensors_to_device(value: Any, *, device: torch.device, floating_dtype: torch.dtype) -> Any:
    if torch.is_tensor(value):
        if value.is_floating_point():
            return value.to(device=device, dtype=floating_dtype)
        return value.to(device=device)
    if isinstance(value, Mapping):
        return {key: _move_tensors_to_device(item, device=device, floating_dtype=floating_dtype) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_move_tensors_to_device(item, device=device, floating_dtype=floating_dtype) for item in value)
    if isinstance(value, list):
        return [_move_tensors_to_device(item, device=device, floating_dtype=floating_dtype) for item in value]
    return value


def aggregate_eval_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not records:
        raise RuntimeError("cannot aggregate empty tiny A/B evaluation")
    values = [float(record["mcp1_anchor1_mse"]) for record in records]
    finite = all(math.isfinite(value) for value in values) and all(
        bool(record.get("finite", False)) for record in records
    )
    by_raw = {}
    by_identity = {}
    for record in records:
        raw_key = str(int(record["raw_timestep"]))
        by_raw.setdefault(raw_key, []).append(float(record["mcp1_anchor1_mse"]))
        identity = str(record["identity"])
        by_identity.setdefault(identity, []).append(float(record["mcp1_anchor1_mse"]))
    return {
        "overall_mse": float(sum(values) / len(values)),
        "finite": bool(finite),
        "by_raw_timestep": {
            raw: {
                "mean_mse": float(sum(items) / len(items)),
                "state_count": len(items),
            }
            for raw, items in sorted(by_raw.items(), key=lambda item: int(item[0]))
        },
        "by_identity": {
            identity: {
                "mean_mse": float(sum(items) / len(items)),
                "state_count": len(items),
            }
            for identity, items in sorted(by_identity.items())
        },
    }


def compare_eval_aggregates(
    *,
    control: Mapping[str, Any],
    treatment: Mapping[str, Any],
) -> dict[str, Any]:
    control_agg = control["aggregate"]
    treatment_agg = treatment["aggregate"]
    control_mse = float(control_agg["overall_mse"])
    treatment_mse = float(treatment_agg["overall_mse"])
    if not math.isfinite(control_mse) or control_mse <= 0.0:
        raise RuntimeError("control validation MSE must be positive and finite")
    improvement = (control_mse - treatment_mse) / control_mse
    degradation = (treatment_mse - control_mse) / control_mse
    control_by_identity = control_agg["by_identity"]
    treatment_by_identity = treatment_agg["by_identity"]
    common_identities = sorted(set(control_by_identity) & set(treatment_by_identity))
    identity_wins = [
        identity
        for identity in common_identities
        if float(treatment_by_identity[identity]["mean_mse"])
        < float(control_by_identity[identity]["mean_mse"])
    ]
    control_by_raw = control_agg["by_raw_timestep"]
    treatment_by_raw = treatment_agg["by_raw_timestep"]
    raw_wins = [
        raw
        for raw in sorted(set(control_by_raw) & set(treatment_by_raw), key=int)
        if float(treatment_by_raw[raw]["mean_mse"])
        < float(control_by_raw[raw]["mean_mse"])
    ]
    return {
        "control_val_mse": control_mse,
        "treatment_val_mse": treatment_mse,
        "validation_relative_improvement": float(improvement),
        "treatment_relative_degradation": float(max(degradation, 0.0)),
        "identity_win_count": len(identity_wins),
        "identity_count": len(common_identities),
        "identity_win_rate": (
            float(len(identity_wins) / len(common_identities))
            if common_identities
            else 0.0
        ),
        "identity_wins": identity_wins,
        "raw_win_count": len(raw_wins),
        "raw_count": len(set(control_by_raw) & set(treatment_by_raw)),
        "raw_wins": raw_wins,
    }


def classify_tiny_ab_decision(
    *,
    comparison: Mapping[str, Any],
    all_loss_and_grad_finite: bool,
) -> dict[str, Any]:
    thresholds = decision_thresholds()
    improvement = float(comparison["validation_relative_improvement"])
    identity_win_rate = float(comparison["identity_win_rate"])
    identity_win_count = int(comparison["identity_win_count"])
    raw_win_count = int(comparison["raw_win_count"])
    degradation = float(comparison["treatment_relative_degradation"])
    support = thresholds["support"]
    no_support = thresholds["no_support"]
    if (
        all_loss_and_grad_finite
        and improvement >= float(support["validation_relative_improvement_min"])
        and identity_win_rate >= float(support["identity_win_rate_min"])
        and identity_win_count >= int(support["identity_win_count_min"])
        and raw_win_count >= int(support["raw_win_count_min"])
    ):
        decision = SUPPORT_PAPER_FIDELITY_MCP1
    elif (
        improvement < float(no_support["low_improvement_max"])
        and identity_win_rate < float(no_support["low_identity_win_rate_max_exclusive"])
    ) or degradation >= float(no_support["treatment_degradation_min"]):
        decision = NO_SUPPORT
    else:
        decision = INCONCLUSIVE
    return {
        "schema": f"{TINY_AB_SCHEMA}_decision_v1",
        "decision": decision,
        "thresholds": thresholds,
        "comparison": dict(comparison),
        "all_loss_and_grad_finite": bool(all_loss_and_grad_finite),
    }


def fairness_key_from_state_record(record: Mapping[str, Any]) -> dict[str, Any]:
    proof = record["state_proof"]
    return {
        "state_id": str(record["state_id"]),
        "identity": str(record["identity"]),
        "raw_timestep": int(record["raw_timestep"]),
        "noise_index": int(record["noise_index"]),
        "future_noise_sha256": str(proof["future_noise_sha256"]),
        "future_target_sha256": str(proof["future_target_sha256"]),
        "exact_fm_target_sha256": str(proof["exact_fm_target_sha256"]),
    }


def compare_arm_fairness(control: Mapping[str, Any], treatment: Mapping[str, Any]) -> dict[str, Any]:
    failures = []
    for key in (
        "parent_checkpoint_sha256",
        "parent_global_step",
        "initial_parameter_sha256",
        "initial_optimizer_fingerprint_sha256",
        "initial_rng_fingerprint_sha256",
        "state_plan_fingerprint_sha256",
    ):
        if str(control.get(key)) != str(treatment.get(key)):
            failures.append(f"{key}:mismatch")
    control_steps = tuple(control.get("train_state_order", ()))
    treatment_steps = tuple(treatment.get("train_state_order", ()))
    if len(control_steps) != len(treatment_steps) or len(control_steps) != TARGET_TINY_STEP:
        failures.append("train_state_order:length_mismatch")
    else:
        for index, (left, right) in enumerate(zip(control_steps, treatment_steps), start=1):
            if dict(left) != dict(right):
                failures.append(f"train_state_order:step{index}_mismatch")
                break
    for eval_key in (
        "validation_step0",
        "validation_step200",
        "train_eval_step0",
        "train_eval_step200",
    ):
        control_records = tuple(control.get(eval_key, {}).get("records", ()))
        treatment_records = tuple(treatment.get(eval_key, {}).get("records", ()))
        if len(control_records) != len(treatment_records):
            failures.append(f"{eval_key}:length_mismatch")
            continue
        for index, (left, right) in enumerate(
            zip(control_records, treatment_records)
        ):
            if fairness_key_from_state_record(left) != fairness_key_from_state_record(right):
                failures.append(f"{eval_key}:state{index}_paired_tensor_mismatch")
                break
    return {
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "same_parent_checkpoint_sha": "parent_checkpoint_sha256:mismatch" not in failures,
        "same_parent_global_step": "parent_global_step:mismatch" not in failures,
        "same_initial_parameter_sha": "initial_parameter_sha256:mismatch" not in failures,
        "same_optimizer_fingerprint": "initial_optimizer_fingerprint_sha256:mismatch" not in failures,
        "same_rng_fingerprint": "initial_rng_fingerprint_sha256:mismatch" not in failures,
        "same_state_plan": "state_plan_fingerprint_sha256:mismatch" not in failures,
        "same_sample_order_and_paired_state_tensors": (
            not any(
                item.startswith("train_state_order:")
                or item.startswith("validation_step")
                or item.startswith("train_eval_step")
                for item in failures
            )
        ),
        "only_allowed_difference": "paper_fidelity_mcp1_mask",
    }


def build_summary(
    *,
    plan: Mapping[str, Any],
    control: Mapping[str, Any],
    treatment: Mapping[str, Any],
) -> dict[str, Any]:
    comparison = compare_eval_aggregates(
        control=control["validation_step200"],
        treatment=treatment["validation_step200"],
    )
    all_finite = bool(control["all_loss_and_grad_finite"]) and bool(
        treatment["all_loss_and_grad_finite"]
    )
    decision = classify_tiny_ab_decision(
        comparison=comparison,
        all_loss_and_grad_finite=all_finite,
    )
    fairness = compare_arm_fairness(control, treatment)
    summary = {
        "schema": TINY_AB_SUMMARY_SCHEMA,
        "status": "PASS" if fairness["status"] == "PASS" else "FAIL_FAIRNESS",
        "diagnostic_only": True,
        "plan_fingerprint_sha256": str(plan["plan_fingerprint_sha256"]),
        "control_val_mse": comparison["control_val_mse"],
        "treatment_val_mse": comparison["treatment_val_mse"],
        "validation_relative_improvement": comparison[
            "validation_relative_improvement"
        ],
        "identity_win_rate": comparison["identity_win_rate"],
        "identity_win_count": comparison["identity_win_count"],
        "raw_win_count": comparison["raw_win_count"],
        "decision": decision["decision"],
        "decision_report": decision,
        "fairness": fairness,
        "control_final_state": control["final_state"],
        "treatment_final_state": treatment["final_state"],
    }
    validate_summary_schema(summary)
    return summary


def validate_summary_schema(summary: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "schema",
        "status",
        "control_val_mse",
        "treatment_val_mse",
        "validation_relative_improvement",
        "identity_win_rate",
        "raw_win_count",
        "decision",
        "fairness",
    }
    missing = required - set(summary)
    if missing:
        raise RuntimeError(f"tiny A/B summary missing fields: {sorted(missing)}")
    if summary["schema"] != TINY_AB_SUMMARY_SCHEMA:
        raise RuntimeError("tiny A/B summary schema mismatch")
    if summary["decision"] not in (
        SUPPORT_PAPER_FIDELITY_MCP1,
        NO_SUPPORT,
        INCONCLUSIVE,
    ):
        raise RuntimeError("tiny A/B invalid decision")
    return {"status": "PASS"}


def _require_real_paths(args: argparse.Namespace) -> None:
    missing = [
        name
        for name in ("sample_plan", "manifest", "dataset_root", "conditionals_artifact")
        if getattr(args, name) is None
    ]
    if missing:
        raise ValueError(
            "--execute_real_run requires: "
            + ", ".join(f"--{name}" for name in missing)
        )
    if args.expected_runtime_git_sha is None:
        raise ValueError("--execute_real_run requires --expected_runtime_git_sha")


def _validate_real_static_contract(args: argparse.Namespace, config: Any) -> None:
    if Path(args.config).resolve() != CANONICAL_CONFIG_PATH:
        raise RuntimeError("config path must be configs/self_forcing_dmd_mcp.yaml")
    if str(args.device) != "cuda:0":
        raise RuntimeError("real tiny A/B requires --device cuda:0")
    if str(args.dtype) != "bf16":
        raise RuntimeError("real tiny A/B requires --dtype bf16")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required only when --execute_real_run is set")
    if int(getattr(config, "num_frame_per_block", 0)) != FULL_SEQUENCE_CHUNK_FRAMES:
        raise RuntimeError("config.num_frame_per_block must be 3")
    if not bool(getattr(config, "gradient_checkpointing", False)):
        raise RuntimeError("config.gradient_checkpointing must be true")


def _path_is_outside_repo(path: Path, *, repo_root: Path) -> bool:
    try:
        Path(path).resolve().relative_to(Path(repo_root).resolve())
        return False
    except ValueError:
        return True


def _repo_dirty_flags() -> tuple[bool, bool]:
    proc = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    staged = False
    tracked = False
    for line in proc.stdout.splitlines():
        if not line:
            continue
        if line.startswith("??"):
            continue
        if line[0] != " " and line[0] != "?":
            staged = True
        if len(line) > 1 and line[1] != " ":
            tracked = True
    return staged, tracked


def _load_store_class(module_name: str, class_name_parts: Sequence[str]) -> Any:
    module = importlib.import_module(module_name)
    return getattr(module, "".join(class_name_parts))


def run_tiny_arm(
    *,
    helpers: Mapping[str, Any],
    arm: Mapping[str, Any],
    parent: Any,
    plan: Mapping[str, Any],
    sample_store: Any,
    conditional_store: Any,
    config: Any,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
    output_dir: Path,
    runtime_git_sha: str,
) -> dict[str, Any]:
    generator = helpers["build_fresh_generator"](
        config=config,
        checkpoint_path=args.checkpoint,
        device=device,
        dtype=dtype,
    )
    optimizer, optimizer_summary = helpers["build_optimizer"](
        generator,
        objective_mode=CONTINUATION_OBJECTIVE_MODE,
        backbone_lr=float(parent.payload["resolved_config"]["backbone_lr"]),
        patch_embedding_lr=float(parent.payload["resolved_config"]["patch_embedding_lr"]),
        mcp_lr=float(parent.payload["resolved_config"]["mcp_lr"]),
        weight_decay=float(parent.payload["resolved_config"]["weight_decay"]),
    )
    validate_optimizer_contract_for_continuation(
        parent.payload,
        active_optimizer_contract=helpers["optimizer_contract"](optimizer),
    )
    train_rng = torch.Generator(device=device)
    validation_base_rng = torch.Generator(device=device)
    restore_report = restore_continuation_state(
        generator=generator,
        optimizer=optimizer,
        train_rng=train_rng,
        validation_base_rng=validation_base_rng,
        payload=parent.payload,
        device=device,
    )
    initial_params = parameter_group_sha256_report(generator)
    initial_optimizer = optimizer_fingerprint(optimizer)
    initial_rng = rng_fingerprint(
        train_rng=train_rng,
        validation_base_rng=validation_base_rng,
        device=device,
    )
    initial_rng_sha = _json_sha256(initial_rng)
    scheduler_main = helpers["make_flow_scheduler"](DEFAULT_S_MAIN)
    scheduler_mcp = helpers["make_flow_scheduler"](DEFAULT_S_MCP)
    arm_dir = output_dir / str(arm["path"])
    arm_dir.mkdir(parents=True, exist_ok=False)

    state_plan = plan["state_plan"]
    paper_flag = bool(arm["paper_fidelity_mcp1_mask"])
    validation_step0 = evaluate_tiny_states(
        generator=generator,
        sample_store=sample_store,
        conditional_store=conditional_store,
        states=state_plan["validation_states"],
        scheduler_main=scheduler_main,
        scheduler_mcp=scheduler_mcp,
        device=device,
        dtype=dtype,
        paper_fidelity_mcp1_mask=paper_flag,
        target_latent_from_sample=helpers["target_latent_from_sample"],
    )
    train_eval_step0 = evaluate_tiny_states(
        generator=generator,
        sample_store=sample_store,
        conditional_store=conditional_store,
        states=state_plan["train_states"],
        scheduler_main=scheduler_main,
        scheduler_mcp=scheduler_mcp,
        device=device,
        dtype=dtype,
        paper_fidelity_mcp1_mask=paper_flag,
        target_latent_from_sample=helpers["target_latent_from_sample"],
    )
    helpers["write_m4_json"](validation_step0, arm_dir / "validation_step000.json")
    helpers["write_m4_json"](train_eval_step0, arm_dir / "train_eval_step000.json")

    train_state_order = []
    all_loss_and_grad_finite = True
    metrics_path = arm_dir / "metrics.jsonl"
    for schedule_item in plan["update_schedule"]:
        step = int(schedule_item["tiny_step"])
        state = state_plan["train_states"][int(schedule_item["state_index"])]
        started = time.perf_counter()
        with sample_store.acquire(str(state["identity"])) as sample:
            with conditional_store.acquire(str(state["identity"])) as conditional_cpu:
                clean_target = helpers["target_latent_from_sample"](sample).to(
                    device=device,
                    dtype=dtype,
                )
                source_noise = sample.source_noise.to(device=device, dtype=dtype)
                conditional = _move_tensors_to_device(
                    conditional_cpu,
                    device=device,
                    floating_dtype=dtype,
                )
                noisy_batch = build_fixed_raw_noisy_batch(
                    clean_target=clean_target,
                    source_noise=source_noise,
                    raw_timestep=int(state["raw_timestep"]),
                    scheduler_main=scheduler_main,
                    scheduler_mcp=scheduler_mcp,
                )
                global_rng_before = helpers["capture_global_rng_state"](device)
                record = run_tiny_update_on_batch(
                    generator=generator,
                    optimizer=optimizer,
                    conditional_dict=conditional,
                    noisy_batch=noisy_batch,
                    state_spec=state,
                    paper_fidelity_mcp1_mask=paper_flag,
                )
                helpers["assert_global_rng_equal"](
                    global_rng_before,
                    helpers["capture_global_rng_state"](device),
                )
        record.update(
            {
                "schema": TINY_AB_ARM_SCHEMA,
                "arm": str(arm["name"]),
                "mcp_path_kind": str(arm["mcp_path_kind"]),
                "paper_fidelity_mcp1_mask": paper_flag,
                "tiny_step": step,
                "elapsed_ms": float((time.perf_counter() - started) * 1000.0),
                "objective": "MCP1 anchor1 exact Flow Matching MSE only",
                "main_loss_in_backward": False,
                "mcp2_loss_in_backward": False,
                "mcp3_loss_in_backward": False,
            }
        )
        all_loss_and_grad_finite = all_loss_and_grad_finite and bool(record["finite"])
        train_state_order.append(fairness_key_from_state_record(record))
        helpers["append_jsonl"](
            metrics_path,
            record,
            fsync=(step % int(args.log_interval) == 0 or step == TARGET_TINY_STEP),
        )

    validation_step200 = evaluate_tiny_states(
        generator=generator,
        sample_store=sample_store,
        conditional_store=conditional_store,
        states=state_plan["validation_states"],
        scheduler_main=scheduler_main,
        scheduler_mcp=scheduler_mcp,
        device=device,
        dtype=dtype,
        paper_fidelity_mcp1_mask=paper_flag,
        target_latent_from_sample=helpers["target_latent_from_sample"],
    )
    train_eval_step200 = evaluate_tiny_states(
        generator=generator,
        sample_store=sample_store,
        conditional_store=conditional_store,
        states=state_plan["train_states"],
        scheduler_main=scheduler_main,
        scheduler_mcp=scheduler_mcp,
        device=device,
        dtype=dtype,
        paper_fidelity_mcp1_mask=paper_flag,
        target_latent_from_sample=helpers["target_latent_from_sample"],
    )
    helpers["write_m4_json"](validation_step200, arm_dir / "validation_step200.json")
    helpers["write_m4_json"](train_eval_step200, arm_dir / "train_eval_step200.json")

    final_params = parameter_group_sha256_report(generator)
    updates = updated_group_report(initial_params, final_params)
    finite_params = finite_parameter_report(generator)
    if not bool(updates["pass"]):
        raise RuntimeError("tiny A/B parameter update contract failed")
    if finite_params["status"] != "PASS":
        raise RuntimeError("tiny A/B non-finite parameter detected")
    final_state = {
        "schema": f"{TINY_AB_SCHEMA}_final_state_v1",
        "arm": str(arm["name"]),
        "runtime_git_sha": runtime_git_sha,
        "parent_checkpoint_sha256": str(parent.sha256),
        "parent_global_step": int(parent.parent_global_step),
        "initial_parameter_sha256": initial_params["aggregate_sha256"],
        "final_parameter_sha256": final_params["aggregate_sha256"],
        "initial_optimizer_fingerprint_sha256": initial_optimizer,
        "final_optimizer_fingerprint_sha256": optimizer_fingerprint(optimizer),
        "initial_rng_fingerprint_sha256": initial_rng_sha,
        "final_rng_fingerprint_sha256": _json_sha256(
            rng_fingerprint(
                train_rng=train_rng,
                validation_base_rng=validation_base_rng,
                device=device,
            )
        ),
        "state_plan_fingerprint_sha256": str(
            state_plan["state_plan_fingerprint_sha256"]
        ),
        "parameter_updates": updates,
        "finite_parameters": finite_params,
        "optimizer_state_valid": len(optimizer.state) > 0,
        "optimizer_summary": optimizer_summary,
        "restore_report": restore_report,
    }
    helpers["write_m4_json"](final_state, arm_dir / "final_state.json")
    return {
        "schema": TINY_AB_ARM_SCHEMA,
        "arm": str(arm["name"]),
        "mcp_path_kind": str(arm["mcp_path_kind"]),
        "paper_fidelity_mcp1_mask": paper_flag,
        "parent_checkpoint_sha256": str(parent.sha256),
        "parent_global_step": int(parent.parent_global_step),
        "initial_parameter_sha256": initial_params["aggregate_sha256"],
        "initial_optimizer_fingerprint_sha256": initial_optimizer,
        "initial_rng_fingerprint_sha256": initial_rng_sha,
        "state_plan_fingerprint_sha256": str(state_plan["state_plan_fingerprint_sha256"]),
        "train_state_order": train_state_order,
        "all_loss_and_grad_finite": bool(all_loss_and_grad_finite),
        "validation_step0": validation_step0,
        "validation_step200": validation_step200,
        "train_eval_step0": train_eval_step0,
        "train_eval_step200": train_eval_step200,
        "final_state": final_state,
    }


def run_tiny_ab(args: argparse.Namespace) -> dict[str, Any]:
    if int(args.log_interval) <= 0:
        raise ValueError("--log_interval must be positive")
    if not args.execute_real_run:
        sample_plan = _synthetic_plan_for_dry_run()
        plan = build_tiny_ab_plan(sample_plan)
        return {
            "schema": TINY_AB_SUMMARY_SCHEMA,
            "status": "DRY_RUN",
            "dry_run": True,
            "plan": plan,
            "decision_thresholds_preregistered": decision_thresholds(),
        }

    _require_real_paths(args)
    helpers = _trainer_helpers()
    config = helpers["merge_config"](args.config)
    _validate_real_static_contract(args, config)
    if not _path_is_outside_repo(args.output_dir, repo_root=ROOT):
        raise RuntimeError("real tiny A/B output_dir must be outside the repository")
    staged, tracked = _repo_dirty_flags()
    if staged or tracked:
        raise RuntimeError("real tiny A/B requires clean tracked worktree and index")
    runtime_git_sha = current_git_head()
    expected_runtime_git = validate_git_sha(
        str(args.expected_runtime_git_sha),
        name="--expected_runtime_git_sha",
    )
    if runtime_git_sha != expected_runtime_git:
        raise RuntimeError("runtime git SHA mismatch")
    helpers["prepare_output_dir"](args.output_dir, resume=False)

    sample_plan = load_m4_sample_plan(args.sample_plan, manifest_path=args.manifest)
    helpers["validate_sample_plan_contract"](sample_plan)
    plan = build_tiny_ab_plan(sample_plan)
    helpers["write_m4_json"](plan, args.output_dir / "tiny_ab_plan.json")

    ConditionalStore = _load_store_class(
        "utils.nf_sf_m5_conditionals",
        ("M5ConditionalArtifactStore",),
    )
    SampleStore = _load_store_class(
        "utils.nf_sf_m5_samples",
        ("M5", "Teach", "erSampleStore"),
    )
    conditional_store = ConditionalStore(
        artifact_dir=args.conditionals_artifact,
        sample_plan=sample_plan,
    )
    parent = load_continuation_parent_checkpoint(
        args.parent_checkpoint,
        expected_parent_checkpoint_sha256=args.expected_parent_checkpoint_sha256,
        expected_parent_global_step=int(args.expected_parent_global_step),
        expected_parent_checkpoint_git_sha=args.expected_parent_checkpoint_git_sha,
        sample_plan_sha256=str(sample_plan["sample_plan_sha256"]),
        manifest_sha256=file_sha256(args.manifest),
        conditionals_artifact_sha256=conditional_store.artifact_sha256,
    )
    if int(parent.parent_global_step) != PARENT_STEP:
        raise RuntimeError("tiny A/B parent must be canonical step6500")
    if str(parent.sha256) != PARENT_CHECKPOINT_SHA256:
        raise RuntimeError("tiny A/B parent checkpoint SHA mismatch")
    parent_semantic = semantic_lock_fingerprint(parent.payload["resolved_config"])
    if parent_semantic != parent.semantic_lock_fingerprint:
        raise RuntimeError("parent semantic lock fingerprint mismatch")
    sample_store = SampleStore(
        sample_plan=sample_plan,
        manifest_path=args.manifest,
        dataset_root=args.dataset_root,
        reference_checkpoint_path=None,
        expected_reference_sha256=str(parent.payload["reference_checkpoint"]["sha256"]),
    )
    helpers["validate_store_identity_order"](
        sample_plan=sample_plan,
        teacher_store=sample_store,
        conditional_store=conditional_store,
    )
    device = torch.device(args.device)
    dtype = helpers["dtype_from_arg"](args.dtype)
    arm_results = {}
    for arm in arm_specs():
        arm_results[str(arm["name"])] = run_tiny_arm(
            helpers=helpers,
            arm=arm,
            parent=parent,
            plan=plan,
            sample_store=sample_store,
            conditional_store=conditional_store,
            config=config,
            args=args,
            device=device,
            dtype=dtype,
            output_dir=args.output_dir,
            runtime_git_sha=runtime_git_sha,
        )
    summary = build_summary(
        plan=plan,
        control=arm_results["canonical"],
        treatment=arm_results["paper_fidelity"],
    )
    helpers["write_m4_json"](summary, args.output_dir / "tiny_ab_summary.json")
    return summary


def _synthetic_plan_for_dry_run() -> dict[str, Any]:
    return {
        "samples": {
            "train": [
                {
                    "identity": f"train_{index:04d}",
                    "split": "train",
                    "split_index": index,
                    "sample_index": index,
                    "sample_id": None,
                }
                for index in range(2048)
            ],
            "validation": [
                {
                    "identity": f"validation_{index:04d}",
                    "split": "validation",
                    "split_index": index,
                    "sample_index": 2048 + index,
                    "sample_id": None,
                }
                for index in range(256)
            ],
        }
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = run_tiny_ab(args)
    print(summary["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
