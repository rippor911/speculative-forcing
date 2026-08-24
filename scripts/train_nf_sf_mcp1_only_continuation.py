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
from utils.nf_sf_m4 import load_m4_sample_plan
from utils.nf_sf_m5_conditionals import M5ConditionalArtifactStore
from utils.nf_sf_m5_samples import M5TeacherSampleStore
from utils.nf_sf_mcp_direct_context_ablation import (
    NF_SF_MCP_DIRECT_CLEAN_KV_FIXED_PROBE_FILENAME,
    noisy_batch_draw_fingerprint,
    run_fixed_raw999_probe_for_ablation,
    train_rng_state_sha256,
)
from utils.nf_sf_mcp1_only_continuation import (
    MCP1_ONLY_FROZEN_GROUPS,
    MCP1_ONLY_TRAINABLE_GROUPS,
    NF_SF_MCP1_ONLY_CONTINUATION_PARENT_STEP,
    NF_SF_MCP1_ONLY_CONTINUATION_SCHEMA,
    NF_SF_MCP1_ONLY_CONTINUATION_TARGET_STEP,
    NF_SF_MCP1_ONLY_PARENT_CHECKPOINT_SHA256,
    NF_SF_MCP1_ONLY_PARENT_GIT_SHA,
    assert_no_forbidden_mcp1_only_gradients,
    audit_mcp1_only_gradients,
    build_mcp1_only_optimizer_from_canonical_state,
    build_mcp1_only_provenance,
    build_mcp1_only_run_plan,
    compare_parameter_sha256_reports,
    forbidden_feature_contract,
    has_nonfinite_trainable_grad,
    mcp1_only_first_step_contract,
    mcp1_only_loss_metrics,
    mcp1_only_step_numbers,
    parameter_sha256_report,
    run_mcp1_only_forward_loss,
    trainable_parameter_delta_report,
    trainable_parameter_snapshot,
    validate_matching_control_provenance,
    validate_mcp1_only_manifest,
    validate_mcp1_only_optimizer_isolation,
)
from utils.nf_sf_tensors import DEFAULT_S_MAIN, DEFAULT_S_MCP
from utils.nf_sf_training import (
    OFFICIAL_SELF_FORCING_CHECKPOINT_SHA256,
    nf_sf_full_sequence_train_cursor,
    prepare_nf_sf_full_sequence_noisy_batch,
)


CANONICAL_CONFIG_PATH = (ROOT / "configs" / "self_forcing_dmd_mcp.yaml").resolve()
MCP1_ONLY_CHECKPOINT_VALIDATION_SCHEMA = (
    f"{NF_SF_MCP1_ONLY_CONTINUATION_SCHEMA}_checkpoint_validation_v1"
)


def _trainer_helpers() -> Mapping[str, Any]:
    from scripts import train_nf_sf_full_sequence_next_forcing as trainer

    return {
        "append_jsonl": trainer.append_jsonl,
        "assert_finite_loss": trainer.assert_finite_loss,
        "assert_global_rng_equal": trainer.assert_global_rng_equal,
        "atomic_torch_save": trainer.atomic_torch_save,
        "build_fresh_generator": trainer.build_fresh_generator,
        "build_optimizer": trainer.build_optimizer,
        "capture_global_rng_state": trainer.capture_global_rng_state,
        "compact_validation_summary": trainer.compact_validation_summary,
        "dtype_from_arg": trainer.dtype_from_arg,
        "make_flow_scheduler": trainer.make_flow_scheduler,
        "merge_config": trainer.merge_config,
        "move_tensors_to_cpu": trainer.move_tensors_to_cpu,
        "optimizer_contract": trainer.optimizer_contract,
        "prepare_output_dir": trainer.prepare_output_dir,
        "run_full_sequence_validation": trainer.run_full_sequence_validation,
        "target_latent_from_sample": trainer.target_latent_from_sample,
        "validate_sample_plan_contract": trainer.validate_sample_plan_contract,
        "validate_store_identity_order": trainer.validate_store_identity_order,
        "write_m4_json": trainer.write_m4_json,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="NF-SF diagnostic full-data MCP1-only continuation."
    )
    parser.add_argument("--execute_real_run", action="store_true")
    parser.add_argument("--engineering_smoke_one_step", action="store_true")
    parser.add_argument("--config", type=Path, default=Path("configs/self_forcing_dmd_mcp.yaml"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/self_forcing_dmd.pt"))
    parser.add_argument("--parent_checkpoint", required=True, type=Path)
    parser.add_argument(
        "--expected_parent_checkpoint_sha256",
        default=NF_SF_MCP1_ONLY_PARENT_CHECKPOINT_SHA256,
    )
    parser.add_argument(
        "--expected_parent_global_step",
        type=int,
        choices=(NF_SF_MCP1_ONLY_CONTINUATION_PARENT_STEP,),
        default=NF_SF_MCP1_ONLY_CONTINUATION_PARENT_STEP,
    )
    parser.add_argument(
        "--expected_parent_checkpoint_git_sha",
        default=NF_SF_MCP1_ONLY_PARENT_GIT_SHA,
    )
    parser.add_argument(
        "--target_global_step",
        type=int,
        choices=(NF_SF_MCP1_ONLY_CONTINUATION_TARGET_STEP,),
        default=NF_SF_MCP1_ONLY_CONTINUATION_TARGET_STEP,
    )
    parser.add_argument("--expected_runtime_git_sha", required=True)
    parser.add_argument("--sample_plan", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--dataset_root", type=Path)
    parser.add_argument("--conditionals_artifact", type=Path)
    parser.add_argument("--matching_control_summary", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bf16",), default="bf16")
    parser.add_argument("--log_interval", type=int, default=100)
    return parser.parse_args(argv)


def run_mcp1_only_train_step(
    *,
    helpers: Mapping[str, Any],
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
    allowed_param_ids: set[int] | frozenset[int],
    capture_draw_fingerprint: bool = False,
) -> dict[str, Any]:
    cursor = nf_sf_full_sequence_train_cursor(global_step)
    identity = teacher_store.train_identity_for_step(global_step)
    train_rng_before = train_rng_state_sha256(train_rng)
    optimizer.zero_grad(set_to_none=True)
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
            draw_fingerprint = (
                noisy_batch_draw_fingerprint(noisy_batch)
                if capture_draw_fingerprint
                else None
            )
            before_global_rng = helpers["capture_global_rng_state"](device)
            result = run_mcp1_only_forward_loss(
                generator,
                conditional_dict=conditional,
                noisy_batch=noisy_batch,
            )
            helpers["assert_global_rng_equal"](
                before_global_rng,
                helpers["capture_global_rng_state"](device),
            )
            helpers["assert_finite_loss"](result.total_loss, name="mcp1_only_loss")
            result.total_loss.backward()
            helpers["assert_global_rng_equal"](
                before_global_rng,
                helpers["capture_global_rng_state"](device),
            )
            gradient_audit = audit_mcp1_only_gradients(generator)
            if has_nonfinite_trainable_grad(optimizer):
                raise RuntimeError("MCP1-only non-finite trainable gradient detected")
            optimizer_isolation = validate_mcp1_only_optimizer_isolation(
                generator,
                optimizer,
                allowed_param_ids=set(allowed_param_ids),
            )
            optimizer.step()
            helpers["assert_global_rng_equal"](
                before_global_rng,
                helpers["capture_global_rng_state"](device),
            )
            assert_no_forbidden_mcp1_only_gradients(generator)
            optimizer.zero_grad(set_to_none=True)
            record = {
                "schema": NF_SF_MCP1_ONLY_CONTINUATION_SCHEMA,
                "global_step": int(global_step),
                "sample_cursor": dict(cursor),
                "sample_identity": str(identity),
                "identity": str(identity),
                "diagnostic_only": True,
                "non_canonical": True,
                "canonical_training_eligible": False,
                "deployment_eligible": False,
                "objective_mode": CONTINUATION_OBJECTIVE_MODE,
                "loss_scope": "mcp_depth1_exact_flow_matching_only",
                "trainable_groups": list(MCP1_ONLY_TRAINABLE_GROUPS),
                "frozen_groups": list(MCP1_ONLY_FROZEN_GROUPS),
                "train_rng_before_sha256": train_rng_before,
                "train_rng_after_sha256": train_rng_state_sha256(train_rng),
                "losses": mcp1_only_loss_metrics(result),
                "tap_shapes": [list(shape) for shape in result.tap_shapes],
                "anchor_token_slices": [list(value) for value in result.anchor_token_slices],
                "main_backbone_forward_count": int(result.main_backbone_forward_count),
                "future_embedding_order": result.future_embedding_order,
                "gradient_audit": gradient_audit,
                "optimizer_isolation": optimizer_isolation,
            }
            if draw_fingerprint is not None:
                record["draw_fingerprint"] = draw_fingerprint
            return record


def save_mcp1_only_checkpoint(
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
        "schema": NF_SF_MCP1_ONLY_CONTINUATION_SCHEMA,
        "status": "DIAGNOSTIC_MCP1_ONLY",
        "diagnostic_only": True,
        "non_canonical": True,
        "canonical_training_eligible": False,
        "deployment_eligible": False,
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
                "schema": MCP1_ONLY_CHECKPOINT_VALIDATION_SCHEMA,
                "path": str(checkpoint_path.resolve()),
                "sha256": checkpoint_sha,
                "size_bytes": int(checkpoint_path.stat().st_size),
                "global_step": int(global_step),
                "diagnostic_only": True,
                "non_canonical": True,
                "canonical_training_eligible": False,
                "deployment_eligible": False,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
        newline="\n",
    )
    return checkpoint_path


def run_mcp1_only_continuation(args: argparse.Namespace) -> dict[str, Any]:
    if int(args.log_interval) <= 0:
        raise ValueError("--log_interval must be positive")
    plan = build_mcp1_only_run_plan()
    if args.engineering_smoke_one_step:
        plan_dict = {
            **dict(plan.__dict__),
            "engineering_smoke": True,
            "target_step": NF_SF_MCP1_ONLY_CONTINUATION_PARENT_STEP + 1,
            "update_count": 1,
            "checkpoint_steps": (),
            "validation_steps": (),
        }
    else:
        plan_dict = dict(plan.__dict__)
    if not args.execute_real_run:
        control_reuse = _load_or_missing_control_audit(args.matching_control_summary)
        return {
            "schema": NF_SF_MCP1_ONLY_CONTINUATION_SCHEMA,
            "dry_run": True,
            "diagnostic_only": True,
            "non_canonical": True,
            "canonical_training_eligible": False,
            "deployment_eligible": False,
            "run_plan": plan_dict,
            "control_reuse_audit": control_reuse,
            "provenance": build_mcp1_only_provenance(
                runtime_git_sha=str(args.expected_runtime_git_sha),
                semantic_lock_fingerprint="DRY_RUN_NOT_LOADED",
            ),
        }

    _require_real_run_paths(args)
    _validate_real_run_static_guards(args)
    runtime_git_sha = current_git_head()
    expected_runtime_git = validate_git_sha(
        args.expected_runtime_git_sha,
        name="--expected_runtime_git_sha",
    )
    if runtime_git_sha != expected_runtime_git:
        raise RuntimeError("runtime git SHA mismatch")
    staged, tracked = _repo_dirty_flags()
    validate_mcp1_only_real_run_guards(
        parent_step=int(args.expected_parent_global_step),
        target_step=int(args.target_global_step),
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
        expected_parent_global_step=NF_SF_MCP1_ONLY_CONTINUATION_PARENT_STEP,
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
    canonical_optimizer, _canonical_optimizer_summary = helpers["build_optimizer"](
        generator,
        objective_mode=CONTINUATION_OBJECTIVE_MODE,
        backbone_lr=float(parent_resolved["backbone_lr"]),
        patch_embedding_lr=float(parent_resolved["patch_embedding_lr"]),
        mcp_lr=float(parent_resolved["mcp_lr"]),
        weight_decay=float(parent_resolved["weight_decay"]),
    )
    validate_optimizer_contract_for_continuation(
        parent.payload,
        active_optimizer_contract=helpers["optimizer_contract"](canonical_optimizer),
    )
    train_rng = torch.Generator(device=device)
    validation_base_rng = torch.Generator(device=device)
    restored = restore_continuation_state(
        generator=generator,
        optimizer=canonical_optimizer,
        train_rng=train_rng,
        validation_base_rng=validation_base_rng,
        payload=parent.payload,
        device=device,
    )
    optimizer, selection, optimizer_state_report = (
        build_mcp1_only_optimizer_from_canonical_state(
            generator,
            canonical_optimizer,
            mcp_lr=float(parent_resolved["mcp_lr"]),
            weight_decay=float(parent_resolved["weight_decay"]),
            require_existing_state=True,
        )
    )
    del canonical_optimizer

    frozen_before = parameter_sha256_report(generator, groups=MCP1_ONLY_FROZEN_GROUPS)
    trainable_before = trainable_parameter_snapshot(selection.trainable_named_parameters)
    scheduler_main = helpers["make_flow_scheduler"](DEFAULT_S_MAIN)
    scheduler_mcp = helpers["make_flow_scheduler"](DEFAULT_S_MCP)
    helpers["prepare_output_dir"](args.output_dir, resume=False)
    first_identity = teacher_store.train_identity_for_step(
        NF_SF_MCP1_ONLY_CONTINUATION_PARENT_STEP + 1
    )
    first_contract = mcp1_only_first_step_contract(
        train_identity=first_identity,
        sample_cursor=nf_sf_full_sequence_train_cursor(
            NF_SF_MCP1_ONLY_CONTINUATION_PARENT_STEP + 1
        ),
    )
    provenance = build_mcp1_only_provenance(
        runtime_git_sha=runtime_git_sha,
        semantic_lock_fingerprint=semantic_fingerprint,
        parent_checkpoint_sha256=parent.sha256,
        parent_git_sha=parent.parent_git_sha,
    )
    control_reuse = _load_or_missing_control_audit(args.matching_control_summary)
    metadata = {
        "schema": NF_SF_MCP1_ONLY_CONTINUATION_SCHEMA,
        "diagnostic_only": True,
        "non_canonical": True,
        "canonical_training_eligible": False,
        "deployment_eligible": False,
        "run_plan": plan_dict,
        "provenance": provenance,
        "restore_contract": restored,
        "rng_fingerprint_after_restore": rng_fingerprint(
            train_rng=train_rng,
            validation_base_rng=validation_base_rng,
            device=device,
        ),
        "first_continuation_step": first_contract,
        "trainable_parameter_selection": selection.summary,
        "optimizer_state_report": optimizer_state_report,
        "control_reuse_audit": control_reuse,
        "forbidden_features": forbidden_feature_contract(),
    }
    helpers["write_m4_json"](metadata, args.output_dir / "run_metadata.json")

    metrics_path = args.output_dir / "metrics.jsonl"
    steps = (
        (NF_SF_MCP1_ONLY_CONTINUATION_PARENT_STEP + 1,)
        if args.engineering_smoke_one_step
        else mcp1_only_step_numbers()
    )
    target_step = int(plan_dict["target_step"])
    final_record = None
    for step in steps:
        started = time.perf_counter()
        capture_draw = (
            step == steps[0]
            or step == steps[-1]
            or step % int(args.log_interval) == 0
        )
        record = run_mcp1_only_train_step(
            helpers=helpers,
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
            allowed_param_ids=selection.allowed_param_ids,
            capture_draw_fingerprint=capture_draw,
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

    frozen_after = parameter_sha256_report(generator, groups=MCP1_ONLY_FROZEN_GROUPS)
    frozen_unchanged = compare_parameter_sha256_reports(frozen_before, frozen_after)
    if not bool(frozen_unchanged["all_sha256_exact_match"]):
        raise RuntimeError("MCP1-only frozen parameter exact SHA proof failed")
    trainable_delta = trainable_parameter_delta_report(
        selection.trainable_named_parameters,
        trainable_before,
    )
    if float(trainable_delta["aggregate_l2"]) <= 0.0:
        raise RuntimeError("MCP1-only fusion/MCP1 parameter delta is zero")

    if args.engineering_smoke_one_step:
        summary = {
            "schema": NF_SF_MCP1_ONLY_CONTINUATION_SCHEMA,
            "status": "ENGINEERING_SMOKE_COMPLETE",
            "diagnostic_only": True,
            "non_canonical": True,
            "canonical_training_eligible": False,
            "deployment_eligible": False,
            "parent_step": NF_SF_MCP1_ONLY_CONTINUATION_PARENT_STEP,
            "target_step": target_step,
            "train_record_count": 1,
            "final_train_record": final_record,
            "optimization_scope": {
                "trainable_groups": list(MCP1_ONLY_TRAINABLE_GROUPS),
                "frozen_groups": list(MCP1_ONLY_FROZEN_GROUPS),
                "loss": "MCP1 exact Flow Matching MSE only",
            },
            "parameter_immutability": frozen_unchanged,
            "trainable_parameter_delta": trainable_delta,
            "forbidden_features": forbidden_feature_contract(),
            "control_reuse_audit": control_reuse,
        }
        validate_mcp1_only_manifest(summary)
        helpers["write_m4_json"](summary, args.output_dir / "smoke_summary.json")
        _cleanup_generator(generator)
        return summary

    validation = helpers["run_full_sequence_validation"](
        generator=generator,
        scheduler_main=scheduler_main,
        scheduler_mcp=scheduler_mcp,
        teacher_store=teacher_store,
        conditional_store=conditional_store,
        validation_seed=int(parent_resolved["validation_seed"]),
        train_rng=train_rng,
        objective_mode=CONTINUATION_OBJECTIVE_MODE,
        device=device,
        dtype=dtype,
        global_step=target_step,
    )
    validation_path = args.output_dir / f"validation_step{target_step:06d}.json"
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
        arm="control",
        main_scheduler=scheduler_main,
        mcp_scheduler=scheduler_mcp,
        source_noise=source_noise,
        teacher_target=teacher_target,
        conditional_dict=conditional,
        sample_identity=fixed_identity,
        global_step=target_step,
    )
    fixed_probe_path = args.output_dir / NF_SF_MCP_DIRECT_CLEAN_KV_FIXED_PROBE_FILENAME
    helpers["write_m4_json"](fixed_probe, fixed_probe_path)
    metadata = {
        **metadata,
        "validation": validation,
        "fixed_probe": fixed_probe,
        "fixed_probe_path": str(fixed_probe_path.resolve()),
        "parameter_immutability": frozen_unchanged,
        "trainable_parameter_delta": trainable_delta,
    }
    checkpoint_path = save_mcp1_only_checkpoint(
        helpers=helpers,
        output_dir=args.output_dir,
        generator=generator,
        optimizer=optimizer,
        global_step=target_step,
        train_rng=train_rng,
        validation_base_rng=validation_base_rng,
        validation_seed=int(parent_resolved["validation_seed"]),
        resolved_config={
            **dict(parent_resolved),
            "schema": NF_SF_MCP1_ONLY_CONTINUATION_SCHEMA,
            "diagnostic_only": True,
            "non_canonical": True,
            "canonical_training_eligible": False,
            "deployment_eligible": False,
            "mcp1_only_parent_step": NF_SF_MCP1_ONLY_CONTINUATION_PARENT_STEP,
            "mcp1_only_target_step": target_step,
            "mcp1_only_update_count": len(steps),
        },
        metadata=metadata,
        git_sha=runtime_git_sha,
        device=device,
    )
    summary = {
        "schema": NF_SF_MCP1_ONLY_CONTINUATION_SCHEMA,
        "status": "PASS",
        "diagnostic_only": True,
        "non_canonical": True,
        "canonical_training_eligible": False,
        "deployment_eligible": False,
        "parent_step": NF_SF_MCP1_ONLY_CONTINUATION_PARENT_STEP,
        "target_step": target_step,
        "train_record_count": len(steps),
        "metrics_jsonl": str(metrics_path.resolve()),
        "final_train_record": final_record,
        "validation": helpers["compact_validation_summary"](validation, path=validation_path),
        "fixed_probe": fixed_probe,
        "fixed_probe_path": str(fixed_probe_path.resolve()),
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "optimization_scope": {
            "trainable_groups": list(MCP1_ONLY_TRAINABLE_GROUPS),
            "frozen_groups": list(MCP1_ONLY_FROZEN_GROUPS),
            "loss": "MCP1 exact Flow Matching MSE only",
            "main_loss_backward": False,
            "mcp2_loss_backward": False,
            "mcp3_loss_backward": False,
        },
        "parameter_immutability": frozen_unchanged,
        "trainable_parameter_delta": trainable_delta,
        "trainable_parameter_selection": selection.summary,
        "optimizer_state_report": optimizer_state_report,
        "restore_contract": restored,
        "first_continuation_step": first_contract,
        "control_reuse_audit": control_reuse,
        "forbidden_features": forbidden_feature_contract(),
    }
    validate_mcp1_only_manifest(summary)
    helpers["write_m4_json"](summary, args.output_dir / "mcp1_only_summary.json")
    _cleanup_generator(generator)
    return summary


def validate_mcp1_only_real_run_guards(
    *,
    parent_step: int,
    target_step: int,
    parent_checkpoint_sha256: str,
    output_dir: Path,
    repo_root: Path,
    staged_changes: bool,
    tracked_changes: bool,
) -> None:
    if int(parent_step) != NF_SF_MCP1_ONLY_CONTINUATION_PARENT_STEP:
        raise RuntimeError("MCP1-only continuation must fork only from step6500")
    if int(target_step) != NF_SF_MCP1_ONLY_CONTINUATION_TARGET_STEP:
        raise RuntimeError("MCP1-only continuation target must be step7000")
    if str(parent_checkpoint_sha256) != NF_SF_MCP1_ONLY_PARENT_CHECKPOINT_SHA256:
        raise RuntimeError("parent checkpoint SHA256 mismatch for MCP1-only continuation")
    if not path_is_outside_repo(Path(output_dir), repo_root=Path(repo_root)):
        raise RuntimeError("real MCP1-only output_dir must be outside the repo")
    if staged_changes:
        raise RuntimeError("real MCP1-only continuation requires no staged changes")
    if tracked_changes:
        raise RuntimeError("real MCP1-only continuation requires tracked worktree clean")


def path_is_outside_repo(path: Path, *, repo_root: Path) -> bool:
    resolved_path = Path(path).resolve()
    resolved_root = Path(repo_root).resolve()
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError:
        return True
    return False


def _load_or_missing_control_audit(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {
            "schema": f"{NF_SF_MCP1_ONLY_CONTINUATION_SCHEMA}_control_reuse_audit_v1",
            "CONTROL_REUSABLE": False,
            "failures": ["matching control summary was not provided"],
            "checked_items": [],
        }
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return validate_matching_control_provenance(payload)


def _require_real_run_paths(args: argparse.Namespace) -> None:
    required = (
        "sample_plan",
        "manifest",
        "dataset_root",
        "conditionals_artifact",
    )
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        required_flags = ", ".join("--" + name for name in missing)
        raise ValueError(f"--execute_real_run requires: {required_flags}")


def _validate_real_run_static_guards(args: argparse.Namespace) -> None:
    if args.config.resolve() != CANONICAL_CONFIG_PATH:
        raise RuntimeError("config path must be configs/self_forcing_dmd_mcp.yaml")
    if str(args.device) != "cuda:0" or str(args.dtype) != "bf16":
        raise RuntimeError("real MCP1-only continuation requires --device cuda:0 --dtype bf16")


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


def _cleanup_generator(generator) -> None:
    generator.to("cpu")
    del generator
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main(argv: Sequence[str] | None = None) -> int:
    summary = run_mcp1_only_continuation(parse_args(argv))
    print(
        json.dumps(
            {
                "schema": summary["schema"],
                "dry_run": bool(summary.get("dry_run", False)),
                "status": summary.get("status"),
                "target_step": summary.get("target_step")
                or summary.get("run_plan", {}).get("target_step"),
                "diagnostic_only": bool(summary.get("diagnostic_only", False)),
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
