from __future__ import annotations

import argparse
import gc
import json
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

from utils.nf_sf_full_sequence_continuation import (
    CONTINUATION_OBJECTIVE_MODE,
    load_continuation_parent_checkpoint,
    restore_continuation_state,
    rng_fingerprint,
    semantic_lock_fingerprint,
    validate_git_sha,
    validate_optimizer_contract_for_continuation,
)
from utils.nf_sf_full_sequence_eval import current_git_head
from utils.nf_sf_m3 import file_sha256, move_tensors_to_device
from utils.nf_sf_m4 import derive_m4_validation_seed, load_m4_sample_plan
from utils.nf_sf_m5_conditionals import M5ConditionalArtifactStore
from utils.nf_sf_m5_samples import M5TeacherSampleStore
from utils.nf_sf_mcp_direct_context_ablation import (
    NF_SF_MCP_DIRECT_CLEAN_KV_ABLATION_PARENT_STEP,
    NF_SF_MCP_DIRECT_CLEAN_KV_ABLATION_SCHEMA,
    NF_SF_MCP_DIRECT_CLEAN_KV_ABLATION_TARGET_STEP,
    NF_SF_MCP_DIRECT_CLEAN_KV_FIXED_PROBE_FILENAME,
    NF_SF_MCP_DIRECT_CLEAN_KV_PARENT_CHECKPOINT_SHA256,
    NF_SF_MCP_DIRECT_CLEAN_KV_PARENT_GIT_SHA,
    ablation_step_numbers,
    build_ablation_provenance,
    build_ablation_run_plan,
    build_ablation_smoke_plan,
    build_checkpoint_metadata,
    direct_clean_context_kv_enabled,
    noisy_batch_draw_fingerprint,
    parameter_key_tuple,
    run_fixed_raw999_probe_for_ablation,
    run_nf_sf_full_sequence_forward_loss_for_ablation,
    train_rng_state_sha256,
    validate_ablation_arm,
    validate_ablation_real_run_guards,
)
from utils.nf_sf_tensors import DEFAULT_S_MAIN, DEFAULT_S_MCP
from utils.nf_sf_training import (
    OFFICIAL_SELF_FORCING_CHECKPOINT_SHA256,
    nf_sf_full_sequence_train_cursor,
    prepare_nf_sf_full_sequence_noisy_batch,
)


ABLATION_CHECKPOINT_VALIDATION_SCHEMA = (
    f"{NF_SF_MCP_DIRECT_CLEAN_KV_ABLATION_SCHEMA}_checkpoint_validation_v1"
)
CANONICAL_CONFIG_PATH = (ROOT / "configs" / "self_forcing_dmd_mcp.yaml").resolve()


def _trainer_helpers() -> Mapping[str, Any]:
    from scripts import train_nf_sf_full_sequence_next_forcing as trainer

    return {
        "append_jsonl": trainer.append_jsonl,
        "atomic_torch_save": trainer.atomic_torch_save,
        "build_fresh_generator": trainer.build_fresh_generator,
        "build_optimizer": trainer.build_optimizer,
        "capture_global_rng_state": trainer.capture_global_rng_state,
        "assert_global_rng_equal": trainer.assert_global_rng_equal,
        "assert_finite_loss": trainer.assert_finite_loss,
        "compact_train_record": trainer.compact_train_record,
        "compact_validation_summary": trainer.compact_validation_summary,
        "dtype_from_arg": trainer.dtype_from_arg,
        "has_nonfinite_grad": trainer.has_nonfinite_grad,
        "loss_breakdown_to_floats": trainer.loss_breakdown_to_floats,
        "make_flow_scheduler": trainer.make_flow_scheduler,
        "memory_snapshot": trainer.memory_snapshot,
        "merge_config": trainer.merge_config,
        "move_tensors_to_cpu": trainer.move_tensors_to_cpu,
        "optimizer_contract": trainer.optimizer_contract,
        "prepare_output_dir": trainer.prepare_output_dir,
        "run_full_sequence_train_step": trainer.run_full_sequence_train_step,
        "target_latent_from_sample": trainer.target_latent_from_sample,
        "validate_sample_plan_contract": trainer.validate_sample_plan_contract,
        "validate_full_sequence_gradient_audit": trainer.validate_full_sequence_gradient_audit,
        "audit_nf_sf_full_sequence_gradients": trainer.audit_nf_sf_full_sequence_gradients,
        "validate_smoke_memory_gate": trainer.validate_smoke_memory_gate,
        "validate_store_identity_order": trainer.validate_store_identity_order,
        "write_m4_json": trainer.write_m4_json,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="NF-SF MCP direct clean-context K/V diagnostic A/B runner."
    )
    parser.add_argument("--arm", choices=("control", "direct_clean_kv"), required=True)
    parser.add_argument("--execute_real_run", action="store_true")
    parser.add_argument("--engineering_smoke_one_step", action="store_true")
    parser.add_argument("--config", type=Path, default=Path("configs/self_forcing_dmd_mcp.yaml"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/self_forcing_dmd.pt"))
    parser.add_argument("--parent_checkpoint", required=True, type=Path)
    parser.add_argument(
        "--expected_parent_checkpoint_sha256",
        default=NF_SF_MCP_DIRECT_CLEAN_KV_PARENT_CHECKPOINT_SHA256,
    )
    parser.add_argument(
        "--expected_parent_global_step",
        type=int,
        choices=(NF_SF_MCP_DIRECT_CLEAN_KV_ABLATION_PARENT_STEP,),
        default=NF_SF_MCP_DIRECT_CLEAN_KV_ABLATION_PARENT_STEP,
    )
    parser.add_argument(
        "--expected_parent_checkpoint_git_sha",
        default=NF_SF_MCP_DIRECT_CLEAN_KV_PARENT_GIT_SHA,
    )
    parser.add_argument(
        "--target_global_step",
        type=int,
        choices=(NF_SF_MCP_DIRECT_CLEAN_KV_ABLATION_TARGET_STEP,),
        default=NF_SF_MCP_DIRECT_CLEAN_KV_ABLATION_TARGET_STEP,
    )
    parser.add_argument("--expected_runtime_git_sha", required=True)
    parser.add_argument("--sample_plan", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--dataset_root", type=Path)
    parser.add_argument("--conditionals_artifact", type=Path)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bf16",), default="bf16")
    parser.add_argument("--log_interval", type=int, default=100)
    return parser.parse_args(argv)


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
        if line[0] != " " and line[0] != "?":
            staged = True
        if len(line) > 1 and line[1] != " ":
            tracked = True
    return staged, tracked


def _require_real_run_paths(args: argparse.Namespace) -> None:
    required = (
        "sample_plan",
        "manifest",
        "dataset_root",
        "conditionals_artifact",
    )
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        raise ValueError(f"--execute_real_run requires: {', '.join('--' + name for name in missing)}")


def _plan_dict(plan: Any) -> dict[str, Any]:
    return dict(plan.__dict__) if hasattr(plan, "__dict__") else dict(plan)


def _plan_target_step(plan: Any) -> int:
    return int(getattr(plan, "target_step", _plan_dict(plan)["target_step"]))


def _audit_expected_draw_fingerprint_for_step(
    *,
    helpers: Mapping[str, Any],
    teacher_store,
    scheduler_main,
    scheduler_mcp,
    train_rng: torch.Generator,
    global_step: int,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    cloned_rng = torch.Generator(device=device)
    cloned_rng.set_state(train_rng.get_state())
    identity = teacher_store.train_identity_for_step(global_step)
    with teacher_store.acquire(identity) as sample:
        clean_target = helpers["target_latent_from_sample"](sample).to(
            device=device,
            dtype=dtype,
        )
        noisy_batch = prepare_nf_sf_full_sequence_noisy_batch(
            clean_target,
            scheduler_main=scheduler_main,
            scheduler_mcp=scheduler_mcp,
            rng=cloned_rng,
        )
    return noisy_batch_draw_fingerprint(noisy_batch)


def run_ablation_train_step(
    *,
    helpers: Mapping[str, Any],
    arm: str,
    generator,
    optimizer: torch.optim.Optimizer,
    scheduler_main,
    scheduler_mcp,
    teacher_store,
    conditional_store,
    train_rng: torch.Generator,
    global_step: int,
    device: torch.device,
    dtype: torch.dtype,
    engineering_smoke: bool = False,
) -> dict[str, Any]:
    if not direct_clean_context_kv_enabled(arm):
        train_rng_before = train_rng_state_sha256(train_rng)
        draw_fingerprint = (
            _audit_expected_draw_fingerprint_for_step(
                helpers=helpers,
                teacher_store=teacher_store,
                scheduler_main=scheduler_main,
                scheduler_mcp=scheduler_mcp,
                train_rng=train_rng,
                global_step=global_step,
                device=device,
                dtype=dtype,
            )
            if engineering_smoke
            else None
        )
        record = helpers["run_full_sequence_train_step"](
            generator=generator,
            optimizer=optimizer,
            scheduler_main=scheduler_main,
            scheduler_mcp=scheduler_mcp,
            teacher_store=teacher_store,
            conditional_store=conditional_store,
            train_rng=train_rng,
            global_step=global_step,
            objective_mode=CONTINUATION_OBJECTIVE_MODE,
            device=device,
            dtype=dtype,
            smoke=bool(engineering_smoke),
            full_gradient_audit=bool(engineering_smoke),
            structural_gate=bool(engineering_smoke),
            run_gc=bool(engineering_smoke),
            capture_memory=bool(engineering_smoke),
        )
        record.update(
            {
                "schema": NF_SF_MCP_DIRECT_CLEAN_KV_ABLATION_SCHEMA,
                "arm": validate_ablation_arm(arm),
                "canonical_control_path": True,
                "direct_clean_context_kv": False,
                "engineering_smoke": bool(engineering_smoke),
                "train_rng_before_sha256": train_rng_before,
                "train_rng_after_sha256": train_rng_state_sha256(train_rng),
            }
        )
        if draw_fingerprint is not None:
            record["draw_fingerprint"] = draw_fingerprint
            record["first_sample_identity"] = str(record.get("sample_identity"))
            record["first_sample_cursor"] = dict(record.get("sample_cursor", {}))
            record["formal_ablation_pass"] = False
        return record

    cursor = nf_sf_full_sequence_train_cursor(global_step)
    identity = teacher_store.train_identity_for_step(global_step)
    train_rng_before = train_rng_state_sha256(train_rng)
    memory = {} if engineering_smoke else None
    optimizer.zero_grad(set_to_none=True)
    if memory is not None:
        memory["before_sample"] = helpers["memory_snapshot"]("before_sample", device)

    with teacher_store.acquire(identity) as sample:
        with conditional_store.acquire(identity) as conditional_cpu:
            clean_target = helpers["target_latent_from_sample"](sample).to(
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
            draw_fingerprint = noisy_batch_draw_fingerprint(noisy_batch)
            before_global_rng = helpers["capture_global_rng_state"](device)
            result = run_nf_sf_full_sequence_forward_loss_for_ablation(
                generator,
                arm=arm,
                conditional_dict=conditional,
                noisy_batch=noisy_batch,
                objective_mode=CONTINUATION_OBJECTIVE_MODE,
            )
            helpers["assert_global_rng_equal"](
                before_global_rng,
                helpers["capture_global_rng_state"](device),
            )
            if memory is not None:
                memory["after_forward"] = helpers["memory_snapshot"]("after_forward", device)
            helpers["assert_finite_loss"](result.losses.total_loss, name="total_loss")
            result.losses.total_loss.backward()
            helpers["assert_global_rng_equal"](
                before_global_rng,
                helpers["capture_global_rng_state"](device),
            )
            if memory is not None:
                memory["after_backward"] = helpers["memory_snapshot"]("after_backward", device)
            gradient_report = None
            if engineering_smoke:
                gradient_report = helpers["validate_full_sequence_gradient_audit"](
                    helpers["audit_nf_sf_full_sequence_gradients"](
                        generator,
                        objective_mode=CONTINUATION_OBJECTIVE_MODE,
                    ),
                    objective_mode=CONTINUATION_OBJECTIVE_MODE,
                    global_step=global_step,
                )
                nonfinite_grad = any(
                    int(item.get("grad_tensors", 0)) > 0
                    and not bool(item.get("all_finite", False))
                    for item in gradient_report.values()
                )
            else:
                nonfinite_grad = helpers["has_nonfinite_grad"](generator)
            if nonfinite_grad:
                raise RuntimeError("non-finite gradient detected")
            optimizer.step()
            helpers["assert_global_rng_equal"](
                before_global_rng,
                helpers["capture_global_rng_state"](device),
            )
            if memory is not None:
                memory["after_optimizer_step"] = helpers["memory_snapshot"](
                    "after_optimizer_step",
                    device,
                )
            optimizer_state_size = len(optimizer.state)
            optimizer.zero_grad(set_to_none=True)
            if memory is not None:
                memory["after_cleanup"] = helpers["memory_snapshot"]("after_cleanup", device)
            record = {
                "schema": NF_SF_MCP_DIRECT_CLEAN_KV_ABLATION_SCHEMA,
                "arm": validate_ablation_arm(arm),
                "global_step": int(global_step),
                "sample_cursor": dict(cursor),
                "sample_identity": str(identity),
                "identity": str(identity),
                "canonical_control_path": False,
                "direct_clean_context_kv": direct_clean_context_kv_enabled(arm),
                "depth1_direct_context_only": True,
                "engineering_smoke": bool(engineering_smoke),
                "train_rng_before_sha256": train_rng_before,
                "train_rng_after_sha256": train_rng_state_sha256(train_rng),
                "draw_fingerprint": draw_fingerprint,
                "losses": helpers["loss_breakdown_to_floats"](result.losses),
                "tap_shapes": [list(shape) for shape in result.tap_shapes],
                "anchor_token_slices": [list(value) for value in result.anchor_token_slices],
                "main_backbone_forward_count": int(result.main_backbone_forward_count),
            }
            if engineering_smoke:
                record.update(
                    {
                        "first_sample_identity": str(identity),
                        "first_sample_cursor": dict(cursor),
                        "finite_loss": True,
                        "gradient_report": gradient_report,
                        "optimizer_state_entries": int(optimizer_state_size),
                        "memory": memory,
                        "smoke_memory_gate": helpers["validate_smoke_memory_gate"](memory),
                        "formal_ablation_pass": False,
                        "treatment_clean_context_gradient_path": {
                            "mcp_fusion_pass": bool(gradient_report["mcp_fusion"]["pass"]),
                            "mcp_depth1_pass": bool(gradient_report["mcp_depth1"]["pass"]),
                            "backbone_pass": bool(gradient_report["backbone"]["pass"]),
                            "patch_embedding_pass": bool(gradient_report["patch_embedding"]["pass"]),
                        },
                    }
                )
            return record


@torch.no_grad()
def run_ablation_validation(
    *,
    helpers: Mapping[str, Any],
    arm: str,
    generator,
    scheduler_main,
    scheduler_mcp,
    teacher_store,
    conditional_store,
    validation_seed: int,
    train_rng: torch.Generator,
    device: torch.device,
    dtype: torch.dtype,
    global_step: int,
) -> dict[str, Any]:
    train_rng_before = train_rng.get_state().detach().cpu().clone()
    global_rng_before = helpers["capture_global_rng_state"](device)
    totals = []
    main_totals = []
    mcp_depths = [[], [], []]
    was_training = generator.training
    generator.eval()
    try:
        for identity in teacher_store.validation_identities:
            derived_seed = derive_m4_validation_seed(
                base_seed=int(validation_seed),
                sample_identity=identity,
                tensor_slot="nf_sf_full_sequence_next_forcing_v1",
            )
            local_validation_rng = torch.Generator(device=device)
            local_validation_rng.manual_seed(int(derived_seed))
            with teacher_store.acquire(identity) as sample:
                with conditional_store.acquire(identity) as conditional_cpu:
                    clean_target = helpers["target_latent_from_sample"](sample).to(
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
                    result = run_nf_sf_full_sequence_forward_loss_for_ablation(
                        generator,
                        arm=arm,
                        conditional_dict=conditional,
                        noisy_batch=noisy_batch,
                        objective_mode=CONTINUATION_OBJECTIVE_MODE,
                    )
                    losses = result.losses
                    totals.append(float(losses.total_loss.detach().float().item()))
                    main_totals.append(float(losses.main_loss.detach().float().item()))
                    for index, loss in enumerate(losses.mcp_depth_losses):
                        mcp_depths[index].append(float(loss.detach().float().item()))
    finally:
        generator.train(was_training)
    if not torch.equal(train_rng_before, train_rng.get_state().cpu()):
        raise RuntimeError("ablation validation consumed train_rng")
    helpers["assert_global_rng_equal"](
        global_rng_before,
        helpers["capture_global_rng_state"](device),
    )
    return {
        "schema": f"{NF_SF_MCP_DIRECT_CLEAN_KV_ABLATION_SCHEMA}_validation_v1",
        "arm": validate_ablation_arm(arm),
        "global_step": int(global_step),
        "identity_count": len(teacher_store.validation_identities),
        "validation_seed": int(validation_seed),
        "paired_identity_noise_across_steps": True,
        "weighted_total": _mean_or_none(totals),
        "main_total": _mean_or_none(main_totals),
        "mcp_depth_means": [_mean_or_none(values) for values in mcp_depths],
    }


def _mean_or_none(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def save_ablation_checkpoint(
    *,
    helpers: Mapping[str, Any],
    output_dir: Path,
    generator,
    optimizer: torch.optim.Optimizer,
    global_step: int,
    train_rng: torch.Generator,
    validation_base_rng: torch.Generator,
    validation_seed: int,
    resolved_config: Mapping[str, Any],
    metadata: Mapping[str, Any],
    git_sha: str,
    device: torch.device,
) -> Path:
    checkpoint_path = output_dir / f"checkpoint_step{global_step:06d}.pt"
    payload = {
        "schema": NF_SF_MCP_DIRECT_CLEAN_KV_ABLATION_SCHEMA,
        "status": "DIAGNOSTIC_ABLATION",
        "canonical_training_eligible": False,
        "canonical_deployment_eligible": False,
        "global_step": int(global_step),
        "git_sha": str(git_sha),
        "generator": helpers["move_tensors_to_cpu"](generator.state_dict()),
        "optimizer": helpers["move_tensors_to_cpu"](optimizer.state_dict()),
        "train_rng_state": train_rng.get_state().detach().cpu().clone(),
        "validation_base_rng_state": validation_base_rng.get_state().detach().cpu().clone(),
        "validation_seed": int(validation_seed),
        "python_random_state": __import__("random").getstate(),
        "torch_cpu_global_rng_state": torch.get_rng_state().detach().cpu().clone(),
        "torch_cuda_global_rng_state": (
            torch.cuda.get_rng_state(device).detach().cpu().clone()
            if device.type == "cuda"
            else None
        ),
        "sample_cursor": nf_sf_full_sequence_train_cursor(global_step),
        "resolved_config": dict(resolved_config),
        "metadata": dict(metadata),
    }
    helpers["atomic_torch_save"](payload, checkpoint_path)
    checkpoint_sha = file_sha256(checkpoint_path)
    checkpoint_path.with_suffix(".sha256.txt").write_text(
        f"{checkpoint_sha}  {checkpoint_path.name}\n",
        encoding="utf-8",
        newline="\n",
    )
    checkpoint_path.with_suffix(".validation.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "schema": ABLATION_CHECKPOINT_VALIDATION_SCHEMA,
                "path": str(checkpoint_path.resolve()),
                "sha256": checkpoint_sha,
                "size_bytes": int(checkpoint_path.stat().st_size),
                "global_step": int(global_step),
                "canonical_training_eligible": False,
                "canonical_deployment_eligible": False,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
        newline="\n",
    )
    return checkpoint_path


def run_ablation(args: argparse.Namespace) -> dict[str, Any]:
    if int(args.log_interval) <= 0:
        raise ValueError("--log_interval must be positive")
    validate_ablation_arm(args.arm)
    plan = (
        build_ablation_smoke_plan(args.arm)
        if args.engineering_smoke_one_step
        else build_ablation_run_plan(args.arm)
    )
    if not args.execute_real_run:
        return {
            "schema": NF_SF_MCP_DIRECT_CLEAN_KV_ABLATION_SCHEMA,
            "dry_run": True,
            "engineering_smoke": bool(args.engineering_smoke_one_step),
            "run_plan": _plan_dict(plan),
            "checkpoint_metadata": build_checkpoint_metadata(
                arm=args.arm,
                runtime_git_sha=str(args.expected_runtime_git_sha),
                semantic_lock_fingerprint="DRY_RUN_NOT_LOADED",
            ),
        }

    _require_real_run_paths(args)
    if args.config.resolve() != CANONICAL_CONFIG_PATH:
        raise RuntimeError("config path must be configs/self_forcing_dmd_mcp.yaml")
    if str(args.device) != "cuda:0" or str(args.dtype) != "bf16":
        raise RuntimeError("real ablation requires --device cuda:0 --dtype bf16")
    runtime_git_sha = current_git_head()
    expected_runtime_git = validate_git_sha(
        args.expected_runtime_git_sha,
        name="--expected_runtime_git_sha",
    )
    if runtime_git_sha != expected_runtime_git:
        raise RuntimeError("runtime git SHA mismatch")
    staged, tracked = _repo_dirty_flags()
    validate_ablation_real_run_guards(
        arm=args.arm,
        parent_step=int(args.expected_parent_global_step),
        parent_checkpoint_sha256=str(args.expected_parent_checkpoint_sha256),
        output_dir=args.output_dir,
        repo_root=ROOT,
        staged_changes=staged,
        tracked_changes=tracked,
    )
    if file_sha256(args.checkpoint) != OFFICIAL_SELF_FORCING_CHECKPOINT_SHA256:
        raise RuntimeError("official Self-Forcing checkpoint SHA256 mismatch")

    helpers = _trainer_helpers()
    config = helpers["merge_config"](args.config)
    sample_plan = load_m4_sample_plan(args.sample_plan, manifest_path=args.manifest)
    helpers["validate_sample_plan_contract"](sample_plan)
    conditional_store = M5ConditionalArtifactStore(
        artifact_dir=args.conditionals_artifact,
        sample_plan=sample_plan,
    )
    parent = load_continuation_parent_checkpoint(
        args.parent_checkpoint,
        expected_parent_checkpoint_sha256=args.expected_parent_checkpoint_sha256,
        expected_parent_global_step=NF_SF_MCP_DIRECT_CLEAN_KV_ABLATION_PARENT_STEP,
        expected_parent_checkpoint_git_sha=args.expected_parent_checkpoint_git_sha,
        sample_plan_sha256=str(sample_plan["sample_plan_sha256"]),
        manifest_sha256=file_sha256(args.manifest),
        conditionals_artifact_sha256=str(conditional_store.artifact_sha256),
    )
    parent_resolved = parent.payload["resolved_config"]
    semantic_fingerprint = semantic_lock_fingerprint(parent_resolved)
    if semantic_fingerprint != parent.semantic_lock_fingerprint:
        raise RuntimeError("semantic lock fingerprint changed after parent load")

    teacher_store = M5TeacherSampleStore(
        sample_plan=sample_plan,
        manifest_path=args.manifest,
        dataset_root=args.dataset_root,
        reference_checkpoint_path=None,
        expected_reference_sha256=OFFICIAL_SELF_FORCING_CHECKPOINT_SHA256,
    )
    helpers["validate_store_identity_order"](
        sample_plan=sample_plan,
        teacher_store=teacher_store,
        conditional_store=conditional_store,
    )
    device = torch.device(args.device)
    dtype = helpers["dtype_from_arg"](args.dtype)
    generator = helpers["build_fresh_generator"](
        config=config,
        checkpoint_path=args.checkpoint,
        device=device,
        dtype=dtype,
    )
    parameter_keys_before = parameter_key_tuple(generator)
    optimizer, _optimizer_summary = helpers["build_optimizer"](
        generator,
        objective_mode=CONTINUATION_OBJECTIVE_MODE,
        backbone_lr=float(parent_resolved["backbone_lr"]),
        patch_embedding_lr=float(parent_resolved["patch_embedding_lr"]),
        mcp_lr=float(parent_resolved["mcp_lr"]),
        weight_decay=float(parent_resolved["weight_decay"]),
    )
    validate_optimizer_contract_for_continuation(
        parent.payload,
        active_optimizer_contract=helpers["optimizer_contract"](optimizer),
    )
    train_rng = torch.Generator(device=device)
    validation_base_rng = torch.Generator(device=device)
    restored = restore_continuation_state(
        generator=generator,
        optimizer=optimizer,
        train_rng=train_rng,
        validation_base_rng=validation_base_rng,
        payload=parent.payload,
        device=device,
    )
    if parameter_key_tuple(generator) != parameter_keys_before:
        raise RuntimeError("parameter keys changed across parent restore")

    scheduler_main = helpers["make_flow_scheduler"](DEFAULT_S_MAIN)
    scheduler_mcp = helpers["make_flow_scheduler"](DEFAULT_S_MCP)
    helpers["prepare_output_dir"](args.output_dir, resume=False)
    provenance = build_ablation_provenance(
        arm=args.arm,
        runtime_git_sha=runtime_git_sha,
        semantic_lock_fingerprint=semantic_fingerprint,
    )
    metadata = {
        "schema": NF_SF_MCP_DIRECT_CLEAN_KV_ABLATION_SCHEMA,
        "run_plan": _plan_dict(plan),
        "provenance": provenance,
        "restore_contract": restored,
        "rng_fingerprint_after_restore": rng_fingerprint(
            train_rng=train_rng,
            validation_base_rng=validation_base_rng,
            device=device,
        ),
    }
    helpers["write_m4_json"](metadata, args.output_dir / "run_metadata.json")

    metrics_path = args.output_dir / "metrics.jsonl"
    final_record = None
    steps = (
        (NF_SF_MCP_DIRECT_CLEAN_KV_ABLATION_PARENT_STEP + 1,)
        if args.engineering_smoke_one_step
        else ablation_step_numbers()
    )
    target_step = _plan_target_step(plan)
    for step in steps:
        started = time.perf_counter()
        record = run_ablation_train_step(
            helpers=helpers,
            arm=args.arm,
            generator=generator,
            optimizer=optimizer,
            scheduler_main=scheduler_main,
            scheduler_mcp=scheduler_mcp,
            teacher_store=teacher_store,
            conditional_store=conditional_store,
            train_rng=train_rng,
            global_step=step,
            device=device,
            dtype=dtype,
            engineering_smoke=bool(args.engineering_smoke_one_step),
        )
        record["elapsed_ms"] = float((time.perf_counter() - started) * 1000.0)
        helpers["append_jsonl"](
            metrics_path,
            record,
            fsync=(step % int(args.log_interval) == 0 or step == target_step),
        )
        if step % int(args.log_interval) == 0 or step == target_step:
            helpers["write_m4_json"](record, args.output_dir / f"train_step{step:06d}.json")
        final_record = record

    if args.engineering_smoke_one_step:
        summary = {
            "schema": NF_SF_MCP_DIRECT_CLEAN_KV_ABLATION_SCHEMA,
            "status": "ENGINEERING_SMOKE_COMPLETE",
            "formal_ablation_pass": False,
            "engineering_smoke": True,
            "arm": validate_ablation_arm(args.arm),
            "parent_step": NF_SF_MCP_DIRECT_CLEAN_KV_ABLATION_PARENT_STEP,
            "target_step": NF_SF_MCP_DIRECT_CLEAN_KV_ABLATION_PARENT_STEP + 1,
            "train_record_count": 1,
            "validation_steps": [],
            "checkpoint_steps": [],
            "fixed_probe": None,
            "final_train_record": final_record,
            "canonical_training_eligible": False,
            "canonical_deployment_eligible": False,
        }
        helpers["write_m4_json"](summary, args.output_dir / "smoke_summary.json")
        generator.to("cpu")
        del generator
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return summary

    validation = run_ablation_validation(
        helpers=helpers,
        arm=args.arm,
        generator=generator,
        scheduler_main=scheduler_main,
        scheduler_mcp=scheduler_mcp,
        teacher_store=teacher_store,
        conditional_store=conditional_store,
        validation_seed=int(parent_resolved["validation_seed"]),
        train_rng=train_rng,
        device=device,
        dtype=dtype,
        global_step=plan.target_step,
    )
    validation_path = args.output_dir / f"validation_step{plan.target_step:06d}.json"
    helpers["write_m4_json"](validation, validation_path)
    fixed_identity = str(sample_plan["fixed_decode_validation_identity"])
    with teacher_store.acquire(fixed_identity) as teacher_sample:
        source_noise = teacher_sample.source_noise.detach().cpu().to(
            device=device,
            dtype=dtype,
        )
        teacher_target = teacher_sample.target_latent.detach().cpu().to(
            device=device,
            dtype=dtype,
        )
    with conditional_store.acquire(fixed_identity) as conditional_cpu:
        conditional = move_tensors_to_device(
            conditional_cpu,
            device=device,
            floating_dtype=dtype,
        )
    fixed_probe = run_fixed_raw999_probe_for_ablation(
        generator,
        arm=args.arm,
        main_scheduler=scheduler_main,
        mcp_scheduler=scheduler_mcp,
        source_noise=source_noise,
        teacher_target=teacher_target,
        conditional_dict=conditional,
        sample_identity=fixed_identity,
        global_step=plan.target_step,
    )
    fixed_probe_path = args.output_dir / NF_SF_MCP_DIRECT_CLEAN_KV_FIXED_PROBE_FILENAME
    helpers["write_m4_json"](fixed_probe, fixed_probe_path)
    metadata = {
        **metadata,
        "fixed_probe": fixed_probe,
        "fixed_probe_path": str(fixed_probe_path.resolve()),
    }
    checkpoint_path = save_ablation_checkpoint(
        helpers=helpers,
        output_dir=args.output_dir,
        generator=generator,
        optimizer=optimizer,
        global_step=plan.target_step,
        train_rng=train_rng,
        validation_base_rng=validation_base_rng,
        validation_seed=int(parent_resolved["validation_seed"]),
        resolved_config={
            **dict(parent_resolved),
            "schema": NF_SF_MCP_DIRECT_CLEAN_KV_ABLATION_SCHEMA,
            "canonical_training_eligible": False,
            "canonical_deployment_eligible": False,
            "ablation_arm": validate_ablation_arm(args.arm),
            "ablation_parent_step": plan.parent_step,
            "ablation_target_step": plan.target_step,
            "ablation_update_count": plan.update_count,
        },
        metadata=metadata,
        git_sha=runtime_git_sha,
        device=device,
    )
    summary = {
        "schema": NF_SF_MCP_DIRECT_CLEAN_KV_ABLATION_SCHEMA,
        "status": "PASS",
        "arm": validate_ablation_arm(args.arm),
        "parent_step": plan.parent_step,
        "target_step": plan.target_step,
        "train_record_count": plan.update_count,
        "final_train_record": final_record,
        "validation": helpers["compact_validation_summary"](validation, path=validation_path),
        "fixed_probe": fixed_probe,
        "fixed_probe_path": str(fixed_probe_path.resolve()),
        "primary_fixed_probe_required": True,
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "canonical_training_eligible": False,
        "canonical_deployment_eligible": False,
    }
    helpers["write_m4_json"](summary, args.output_dir / "ablation_summary.json")
    generator.to("cpu")
    del generator
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    summary = run_ablation(parse_args(argv))
    print(
        json.dumps(
            {
                "schema": summary["schema"],
                "dry_run": bool(summary.get("dry_run", False)),
                "arm": summary.get("arm") or summary.get("run_plan", {}).get("arm"),
                "target_step": summary.get("target_step")
                or summary.get("run_plan", {}).get("target_step"),
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
