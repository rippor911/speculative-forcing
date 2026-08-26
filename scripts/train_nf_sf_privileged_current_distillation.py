from __future__ import annotations

import argparse
import gc
import json
import random
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eval_nf_sf_full_sequence_deployment import (
    build_generator as build_eval_generator,
    initialize_runtime,
)
from utils.nf_sf_full_sequence_continuation import (
    load_continuation_parent_checkpoint,
    restore_continuation_state,
    semantic_lock_fingerprint,
    validate_git_sha,
    validate_optimizer_contract_for_continuation,
)
from utils.nf_sf_full_sequence_eval import (
    MODE_OFFICIAL_MAIN,
    current_git_head,
    load_official_checkpoint_record,
)
from utils.nf_sf_m3 import file_sha256, move_tensors_to_device
from utils.nf_sf_m4 import load_m4_sample_plan
from utils.nf_sf_m5_conditionals import M5ConditionalArtifactStore
from utils.nf_sf_m5_samples import M5TeacherSampleStore
from utils.nf_sf_mcp_direct_context_ablation import (
    NF_SF_MCP_DIRECT_CLEAN_KV_FIXED_PROBE_FILENAME,
    run_fixed_raw999_probe_for_ablation,
    train_rng_state_sha256,
)
from utils.nf_sf_privileged_current_distillation import (
    INCONCLUSIVE,
    NO_SUPPORT,
    PRIVILEGED_CURRENT_DISTILLATION_SCHEMA,
    PRIVILEGED_CURRENT_LAMBDA,
    PRIVILEGED_CURRENT_PARENT_CHECKPOINT_SHA256,
    PRIVILEGED_CURRENT_PARENT_GIT_SHA,
    PRIVILEGED_CURRENT_PARENT_STEP,
    PRIVILEGED_CURRENT_TARGET_STEP,
    PRIVILEGED_CURRENT_TEACHER_CHECKPOINT_SHA256,
    PRIVILEGED_CURRENT_UPDATE_COUNT,
    STRONG_SUPPORT,
    classify_privileged_current_ab,
    compare_gradient_reports,
    first_step_contract,
    gradient_group_report,
    optimizer_state_fingerprint,
    parameter_sha256_report,
    privileged_run_plan,
    provenance_contract,
    run_privileged_current_forward_loss,
    strip_gradient_flats,
    teacher_frozen_report,
    validate_control_reuse,
    validate_lambda_priv,
)
from utils.nf_sf_tensors import DEFAULT_S_MAIN, DEFAULT_S_MCP
from utils.nf_sf_training import (
    FULL_SEQUENCE_TRAINER_SCHEMA,
    nf_sf_full_sequence_train_cursor,
    prepare_nf_sf_full_sequence_noisy_batch,
    run_nf_sf_full_sequence_forward_loss,
)


CANONICAL_CONFIG_PATH = (ROOT / "configs" / "self_forcing_dmd_mcp.yaml").resolve()


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
        "memory_snapshot": trainer.memory_snapshot,
        "memory_metrics_for_record": trainer.memory_metrics_for_record,
        "update_memory_maxima": trainer.update_memory_maxima,
        "validate_full_sequence_gradient_audit": (
            trainer.validate_full_sequence_gradient_audit
        ),
        "audit_nf_sf_full_sequence_gradients": (
            trainer.audit_nf_sf_full_sequence_gradients
        ),
        "gradient_report_has_nonfinite": trainer.gradient_report_has_nonfinite,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="NF-SF privileged-current auxiliary distillation treatment."
    )
    parser.add_argument("--execute_real_run", action="store_true")
    parser.add_argument("--gradient_probe_only", action="store_true")
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
    parser.add_argument(
        "--teacher_checkpoint",
        type=Path,
        default=Path("checkpoints/self_forcing_dmd.pt"),
    )
    parser.add_argument("--parent_checkpoint", required=True, type=Path)
    parser.add_argument(
        "--expected_parent_checkpoint_sha256",
        default=PRIVILEGED_CURRENT_PARENT_CHECKPOINT_SHA256,
    )
    parser.add_argument(
        "--expected_parent_global_step",
        type=int,
        choices=(PRIVILEGED_CURRENT_PARENT_STEP,),
        default=PRIVILEGED_CURRENT_PARENT_STEP,
    )
    parser.add_argument(
        "--expected_parent_checkpoint_git_sha",
        default=PRIVILEGED_CURRENT_PARENT_GIT_SHA,
    )
    parser.add_argument(
        "--target_global_step",
        type=int,
        choices=(PRIVILEGED_CURRENT_TARGET_STEP,),
        default=PRIVILEGED_CURRENT_TARGET_STEP,
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
    parser.add_argument("--memory_log_interval", type=int, default=100)
    parser.add_argument(
        "--lambda_priv",
        type=float,
        default=PRIVILEGED_CURRENT_LAMBDA,
    )
    return parser.parse_args(argv)


def run_privileged_current_distillation(args: argparse.Namespace) -> dict[str, Any]:
    validate_lambda_priv(float(args.lambda_priv), formal=True)
    plan = privileged_run_plan()
    control_reuse = validate_control_reuse(args.matching_control_summary)
    if not args.execute_real_run:
        return {
            "schema": PRIVILEGED_CURRENT_DISTILLATION_SCHEMA,
            "status": "DRY_RUN",
            "dry_run": True,
            "diagnostic_only": True,
            "non_canonical": True,
            "canonical_training_eligible": False,
            "deployment_eligible": False,
            "run_plan": plan,
            "control_reuse_audit": control_reuse,
            "provenance": provenance_contract(),
        }
    if control_reuse.get("CONTROL_REUSABLE") is not True:
        raise RuntimeError("strict matching canonical control is not reusable")
    _require_real_run_paths(args)
    helpers = _trainer_helpers()
    config = helpers["merge_config"](args.config)
    _validate_real_run_static_contract(args, config)
    sample_plan = load_m4_sample_plan(args.sample_plan, manifest_path=args.manifest)
    helpers["validate_sample_plan_contract"](sample_plan)
    device = torch.device(args.device)
    dtype = helpers["dtype_from_arg"](args.dtype)
    runtime_git = current_git_head()
    expected_runtime_git = validate_git_sha(
        args.expected_runtime_git_sha,
        name="--expected_runtime_git_sha",
    )
    if runtime_git != expected_runtime_git:
        raise RuntimeError("runtime git SHA mismatch")
    conditional_store = M5ConditionalArtifactStore(
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
    teacher_store = M5TeacherSampleStore(
        sample_plan=sample_plan,
        manifest_path=args.manifest,
        dataset_root=args.dataset_root,
        reference_checkpoint_path=None,
        expected_reference_sha256=PRIVILEGED_CURRENT_TEACHER_CHECKPOINT_SHA256,
    )
    helpers["validate_store_identity_order"](
        sample_plan=sample_plan,
        teacher_store=teacher_store,
        conditional_store=conditional_store,
    )
    generator = helpers["build_fresh_generator"](
        config=config,
        checkpoint_path=args.checkpoint,
        device=device,
        dtype=dtype,
    )
    resolved = _resolved_config_from_parent(parent.payload, args=args)
    optimizer, optimizer_summary = helpers["build_optimizer"](
        generator,
        objective_mode="next_forcing_full",
        backbone_lr=float(resolved["backbone_lr"]),
        patch_embedding_lr=float(resolved["patch_embedding_lr"]),
        mcp_lr=float(resolved["mcp_lr"]),
        weight_decay=float(resolved["weight_decay"]),
    )
    active_optimizer_contract = helpers["optimizer_contract"](optimizer)
    validate_optimizer_contract_for_continuation(
        parent.payload,
        active_optimizer_contract=active_optimizer_contract,
    )
    train_rng = torch.Generator(device=device)
    validation_base_rng = torch.Generator(device=device)
    restore = restore_continuation_state(
        generator=generator,
        optimizer=optimizer,
        train_rng=train_rng,
        validation_base_rng=validation_base_rng,
        payload=parent.payload,
        device=device,
    )
    teacher_checkpoint = load_official_checkpoint_record(args.teacher_checkpoint)
    teacher_generator = build_eval_generator(
        config=config,
        checkpoint=teacher_checkpoint,
        mode=MODE_OFFICIAL_MAIN,
        device=device,
        dtype=dtype,
    )
    teacher_before = teacher_frozen_report(teacher_generator)
    scheduler_main = helpers["make_flow_scheduler"](DEFAULT_S_MAIN)
    scheduler_mcp = helpers["make_flow_scheduler"](DEFAULT_S_MCP)
    helpers["prepare_output_dir"](args.output_dir, resume=False)
    metadata = _run_metadata(
        args=args,
        parent=parent,
        runtime_git=runtime_git,
        resolved=resolved,
        optimizer_summary=optimizer_summary,
        restore=restore,
        control_reuse=control_reuse,
    )
    helpers["write_m4_json"](metadata, args.output_dir / "run_metadata.json")
    probe = run_gradient_safety_probe(
        helpers=helpers,
        generator=generator,
        teacher_generator=teacher_generator,
        optimizer=optimizer,
        scheduler_main=scheduler_main,
        scheduler_mcp=scheduler_mcp,
        teacher_store=teacher_store,
        conditional_store=conditional_store,
        train_rng=train_rng,
        global_step=PRIVILEGED_CURRENT_PARENT_STEP + 1,
        config=config,
        device=device,
        dtype=dtype,
        lambda_priv=float(args.lambda_priv),
    )
    helpers["write_m4_json"](probe, args.output_dir / "gradient_safety_probe.json")
    if probe["status"] != "PASS":
        raise RuntimeError("gradient safety probe failed")
    if args.gradient_probe_only:
        return _probe_only_summary(
            args=args,
            metadata=metadata,
            probe=probe,
            teacher_before=teacher_before,
            teacher_after=teacher_frozen_report(teacher_generator),
        )
    metrics_path = args.output_dir / "metrics.jsonl"
    checkpoint_records = []
    validation_summaries = []
    memory_maxima: dict[str, int] = {}
    final_record = None
    train_count = 0
    for step in range(PRIVILEGED_CURRENT_PARENT_STEP + 1, PRIVILEGED_CURRENT_TARGET_STEP + 1):
        started = time.perf_counter()
        record = run_privileged_train_step(
            helpers=helpers,
            generator=generator,
            teacher_generator=teacher_generator,
            optimizer=optimizer,
            scheduler_main=scheduler_main,
            scheduler_mcp=scheduler_mcp,
            teacher_store=teacher_store,
            conditional_store=conditional_store,
            train_rng=train_rng,
            global_step=step,
            config=config,
            device=device,
            dtype=dtype,
            lambda_priv=float(args.lambda_priv),
            capture_memory=(step % int(args.memory_log_interval) == 0),
        )
        record["elapsed_ms"] = float((time.perf_counter() - started) * 1000.0)
        helpers["update_memory_maxima"](memory_maxima, record.get("memory"))
        helpers["append_jsonl"](
            metrics_path,
            record,
            fsync=(step % int(args.log_interval) == 0 or step == PRIVILEGED_CURRENT_TARGET_STEP),
        )
        train_count += 1
        final_record = record
    validation = helpers["run_full_sequence_validation"](
        generator=generator,
        scheduler_main=scheduler_main,
        scheduler_mcp=scheduler_mcp,
        teacher_store=teacher_store,
        conditional_store=conditional_store,
        validation_seed=int(parent.payload["validation_seed"]),
        train_rng=train_rng,
        objective_mode="next_forcing_full",
        device=device,
        dtype=dtype,
        global_step=PRIVILEGED_CURRENT_TARGET_STEP,
    )
    validation_path = args.output_dir / "validation_step007000.json"
    helpers["write_m4_json"](validation, validation_path)
    validation_summaries.append(
        helpers["compact_validation_summary"](validation, path=validation_path)
    )
    fixed_probe = _run_fixed_probe(
        helpers=helpers,
        generator=generator,
        scheduler_main=scheduler_main,
        scheduler_mcp=scheduler_mcp,
        teacher_store=teacher_store,
        conditional_store=conditional_store,
        sample_plan=sample_plan,
        device=device,
        dtype=dtype,
    )
    fixed_probe_path = args.output_dir / NF_SF_MCP_DIRECT_CLEAN_KV_FIXED_PROBE_FILENAME
    helpers["write_m4_json"](fixed_probe, fixed_probe_path)
    teacher_after = teacher_frozen_report(teacher_generator)
    if teacher_before["parameter_sha256"]["aggregate_sha256"] != (
        teacher_after["parameter_sha256"]["aggregate_sha256"]
    ):
        raise RuntimeError("frozen Teacher parameters changed")
    checkpoint_path = save_privileged_checkpoint(
        helpers=helpers,
        output_dir=args.output_dir,
        generator=generator,
        optimizer=optimizer,
        train_rng=train_rng,
        validation_base_rng=validation_base_rng,
        validation_seed=int(parent.payload["validation_seed"]),
        metadata=metadata,
        final_record=final_record,
        validation=validation,
        fixed_probe=fixed_probe,
        teacher_before=teacher_before,
        teacher_after=teacher_after,
        git_sha=runtime_git,
        device=device,
    )
    checkpoint_records.append(_checkpoint_record(checkpoint_path))
    summary = build_training_summary(
        metadata=metadata,
        metrics_path=metrics_path,
        train_record_count=train_count,
        final_record=final_record,
        checkpoint_records=checkpoint_records,
        validation_summaries=validation_summaries,
        fixed_probe_path=fixed_probe_path,
        fixed_probe=fixed_probe,
        gradient_probe=probe,
        teacher_before=teacher_before,
        teacher_after=teacher_after,
        memory_maxima=memory_maxima,
    )
    helpers["write_m4_json"](summary, args.output_dir / "training_summary.json")
    return summary


def run_privileged_train_step(
    *,
    helpers: Mapping[str, Any],
    generator: Any,
    teacher_generator: Any,
    optimizer: torch.optim.Optimizer,
    scheduler_main: Any,
    scheduler_mcp: Any,
    teacher_store: M5TeacherSampleStore,
    conditional_store: M5ConditionalArtifactStore,
    train_rng: torch.Generator,
    global_step: int,
    config: Any,
    device: torch.device,
    dtype: torch.dtype,
    lambda_priv: float,
    capture_memory: bool = False,
) -> dict[str, Any]:
    cursor = nf_sf_full_sequence_train_cursor(global_step)
    identity = teacher_store.train_identity_for_step(global_step)
    train_rng_before = train_rng_state_sha256(train_rng)
    memory = {} if capture_memory else None
    optimizer.zero_grad(set_to_none=True)
    if memory is not None:
        memory["before_sample"] = helpers["memory_snapshot"]("before_sample", device)
    with teacher_store.acquire(identity) as sample:
        with conditional_store.acquire(identity) as conditional_cpu:
            clean_target = helpers["target_latent_from_sample"](sample).to(
                device=device,
                dtype=dtype,
            )
            source_noise = sample.source_noise.to(device=device, dtype=dtype)
            teacher_payload = dict(sample.payload)
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
            before_global_rng = helpers["capture_global_rng_state"](device)
            result = run_privileged_current_forward_loss(
                generator,
                teacher_runtime_factory=lambda: initialize_runtime(
                    generator=teacher_generator,
                    config=config,
                    source_noise=source_noise,
                ),
                conditional_dict=conditional,
                noisy_batch=noisy_batch,
                source_noise=source_noise,
                teacher_payload=teacher_payload,
                mcp_scheduler=scheduler_mcp,
                lambda_priv=float(lambda_priv),
            )
            helpers["assert_global_rng_equal"](
                before_global_rng,
                helpers["capture_global_rng_state"](device),
            )
            helpers["assert_finite_loss"](result.total_loss, name="total_loss")
            if memory is not None:
                memory["after_forward"] = helpers["memory_snapshot"](
                    "after_forward",
                    device,
                )
            result.total_loss.backward()
            helpers["assert_global_rng_equal"](
                before_global_rng,
                helpers["capture_global_rng_state"](device),
            )
            if memory is not None:
                memory["after_backward"] = helpers["memory_snapshot"](
                    "after_backward",
                    device,
                )
            gradient_report = helpers["validate_full_sequence_gradient_audit"](
                helpers["audit_nf_sf_full_sequence_gradients"](
                    generator,
                    objective_mode="next_forcing_full",
                ),
                objective_mode="next_forcing_full",
                global_step=global_step,
            )
            if helpers["gradient_report_has_nonfinite"](gradient_report):
                raise RuntimeError("non-finite privileged-current gradient detected")
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
            record = {
                "schema": PRIVILEGED_CURRENT_DISTILLATION_SCHEMA,
                "global_step": int(global_step),
                "sample_cursor": dict(cursor),
                "first_step_contract": (
                    first_step_contract(global_step, cursor)
                    if int(global_step) == PRIVILEGED_CURRENT_PARENT_STEP + 1
                    else None
                ),
                "sample_identity": str(identity),
                "train_rng_before_sha256": train_rng_before,
                "train_rng_after_sha256": train_rng_state_sha256(train_rng),
                "objective_mode": "next_forcing_full",
                "canonical_objective_unchanged": True,
                "lambda_priv": float(lambda_priv),
                "losses": dict(result.loss_record),
                "gradient_report": gradient_report,
                "teacher_rng_guard": dict(result.teacher_targets.rng_guard),
                "memory": memory,
            }
            del result
            del noisy_batch
            del conditional
            del clean_target
            del source_noise
    optimizer.zero_grad(set_to_none=True)
    if memory is not None:
        gc.collect()
        memory["after_cleanup"] = helpers["memory_snapshot"]("after_cleanup", device)
    return record


def run_gradient_safety_probe(
    *,
    helpers: Mapping[str, Any],
    generator: Any,
    teacher_generator: Any,
    optimizer: torch.optim.Optimizer,
    scheduler_main: Any,
    scheduler_mcp: Any,
    teacher_store: M5TeacherSampleStore,
    conditional_store: M5ConditionalArtifactStore,
    train_rng: torch.Generator,
    global_step: int,
    config: Any,
    device: torch.device,
    dtype: torch.dtype,
    lambda_priv: float,
) -> dict[str, Any]:
    validate_lambda_priv(lambda_priv, formal=True)
    cursor = nf_sf_full_sequence_train_cursor(global_step)
    identity = teacher_store.train_identity_for_step(global_step)
    parameter_before = parameter_sha256_report(generator)
    teacher_before = parameter_sha256_report(teacher_generator)
    optimizer_before = optimizer_state_fingerprint(optimizer)
    train_rng_before = train_rng.get_state().detach().cpu().clone()
    global_rng_before = helpers["capture_global_rng_state"](device)
    memory = {}
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    memory["before_probe"] = helpers["memory_snapshot"]("before_probe", device)
    canonical_grad = _probe_gradient(
        helpers=helpers,
        generator=generator,
        teacher_generator=teacher_generator,
        scheduler_main=scheduler_main,
        scheduler_mcp=scheduler_mcp,
        teacher_store=teacher_store,
        conditional_store=conditional_store,
        train_rng=train_rng,
        train_rng_state=train_rng_before,
        global_step=global_step,
        config=config,
        device=device,
        dtype=dtype,
        mode="canonical",
        lambda_priv=lambda_priv,
    )
    memory["after_canonical_grad"] = helpers["memory_snapshot"](
        "after_canonical_grad",
        device,
    )
    privileged_grad = _probe_gradient(
        helpers=helpers,
        generator=generator,
        teacher_generator=teacher_generator,
        scheduler_main=scheduler_main,
        scheduler_mcp=scheduler_mcp,
        teacher_store=teacher_store,
        conditional_store=conditional_store,
        train_rng=train_rng,
        train_rng_state=train_rng_before,
        global_step=global_step,
        config=config,
        device=device,
        dtype=dtype,
        mode="privileged",
        lambda_priv=lambda_priv,
    )
    memory["after_privileged_grad"] = helpers["memory_snapshot"](
        "after_privileged_grad",
        device,
    )
    optimizer.zero_grad(set_to_none=True)
    train_rng.set_state(train_rng_before)
    helpers["assert_global_rng_equal"](
        global_rng_before,
        helpers["capture_global_rng_state"](device),
    )
    parameter_after = parameter_sha256_report(generator)
    teacher_after = parameter_sha256_report(teacher_generator)
    optimizer_after = optimizer_state_fingerprint(optimizer)
    if parameter_after["aggregate_sha256"] != parameter_before["aggregate_sha256"]:
        raise RuntimeError("gradient probe changed model parameters")
    if teacher_after["aggregate_sha256"] != teacher_before["aggregate_sha256"]:
        raise RuntimeError("gradient probe changed Teacher parameters")
    if optimizer_after != optimizer_before:
        raise RuntimeError("gradient probe changed optimizer state")
    comparison = compare_gradient_reports(
        canonical_grad,
        privileged_grad,
        lambda_priv=lambda_priv,
    )
    if any(not item["canonical_finite"] for item in comparison.values()):
        raise RuntimeError("canonical gradient probe produced non-finite gradient")
    if any(not item["privileged_finite"] for item in comparison.values()):
        raise RuntimeError("privileged gradient probe produced non-finite gradient")
    memory["after_restore"] = helpers["memory_snapshot"]("after_restore", device)
    return {
        "schema": f"{PRIVILEGED_CURRENT_DISTILLATION_SCHEMA}_gradient_probe_v1",
        "status": "PASS",
        "global_step": int(global_step),
        "sample_identity": str(identity),
        "sample_cursor": dict(cursor),
        "first_step_contract": first_step_contract(global_step, cursor),
        "lambda_priv": float(lambda_priv),
        "optimizer_step_executed": False,
        "parameter_sha_restored_exact": True,
        "optimizer_state_restored_exact": True,
        "rng_restored_exact": True,
        "teacher_sha_unchanged": True,
        "canonical_gradients": strip_gradient_flats(canonical_grad),
        "privileged_gradients": strip_gradient_flats(privileged_grad),
        "gradient_comparison": comparison,
        "memory": memory,
    }


def _probe_gradient(
    *,
    helpers: Mapping[str, Any],
    generator: Any,
    teacher_generator: Any,
    scheduler_main: Any,
    scheduler_mcp: Any,
    teacher_store: M5TeacherSampleStore,
    conditional_store: M5ConditionalArtifactStore,
    train_rng: torch.Generator,
    train_rng_state: torch.Tensor,
    global_step: int,
    config: Any,
    device: torch.device,
    dtype: torch.dtype,
    mode: str,
    lambda_priv: float,
) -> dict[str, Any]:
    train_rng.set_state(train_rng_state)
    optimizer_like_zero_grad(generator)
    identity = teacher_store.train_identity_for_step(global_step)
    with teacher_store.acquire(identity) as sample:
        with conditional_store.acquire(identity) as conditional_cpu:
            clean_target = helpers["target_latent_from_sample"](sample).to(
                device=device,
                dtype=dtype,
            )
            source_noise = sample.source_noise.to(device=device, dtype=dtype)
            teacher_payload = dict(sample.payload)
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
            before_global = helpers["capture_global_rng_state"](device)
            if mode == "canonical":
                result = run_nf_sf_full_sequence_forward_loss(
                    generator,
                    conditional_dict=conditional,
                    noisy_batch=noisy_batch,
                    objective_mode="next_forcing_full",
                )
                loss = result.losses.total_loss
            elif mode == "privileged":
                result = run_privileged_current_forward_loss(
                    generator,
                    teacher_runtime_factory=lambda: initialize_runtime(
                        generator=teacher_generator,
                        config=config,
                        source_noise=source_noise,
                    ),
                    conditional_dict=conditional,
                    noisy_batch=noisy_batch,
                    source_noise=source_noise,
                    teacher_payload=teacher_payload,
                    mcp_scheduler=scheduler_mcp,
                    lambda_priv=lambda_priv,
                )
                loss = result.privileged_loss
            else:
                raise ValueError("probe mode must be canonical or privileged")
            helpers["assert_global_rng_equal"](
                before_global,
                helpers["capture_global_rng_state"](device),
            )
            loss.backward()
            helpers["assert_global_rng_equal"](
                before_global,
                helpers["capture_global_rng_state"](device),
            )
            report = gradient_group_report(generator)
            del result
            del noisy_batch
            del conditional
            del clean_target
            del source_noise
    return report


def optimizer_like_zero_grad(module: torch.nn.Module) -> None:
    for param in module.parameters():
        param.grad = None


def save_privileged_checkpoint(
    *,
    helpers: Mapping[str, Any],
    output_dir: Path,
    generator: Any,
    optimizer: torch.optim.Optimizer,
    train_rng: torch.Generator,
    validation_base_rng: torch.Generator,
    validation_seed: int,
    metadata: Mapping[str, Any],
    final_record: Mapping[str, Any] | None,
    validation: Mapping[str, Any],
    fixed_probe: Mapping[str, Any],
    teacher_before: Mapping[str, Any],
    teacher_after: Mapping[str, Any],
    git_sha: str,
    device: torch.device,
) -> Path:
    checkpoint_path = output_dir / "checkpoint_step007000.pt"
    payload = {
        "schema": PRIVILEGED_CURRENT_DISTILLATION_SCHEMA,
        "status": "DIAGNOSTIC_PRIVILEGED_CURRENT_DISTILLATION",
        "diagnostic_only": True,
        "non_canonical": True,
        "canonical_training_eligible": False,
        "deployment_eligible": False,
        "global_step": PRIVILEGED_CURRENT_TARGET_STEP,
        "git_sha": str(git_sha),
        "generator": helpers["move_tensors_to_cpu"](generator.state_dict()),
        "optimizer": helpers["move_tensors_to_cpu"](optimizer.state_dict()),
        "train_rng_state": train_rng.get_state().detach().cpu().clone(),
        "validation_base_rng_state": validation_base_rng.get_state().detach().cpu().clone(),
        "validation_seed": int(validation_seed),
        "python_random_state": random.getstate(),
        "torch_cpu_global_rng_state": torch.get_rng_state().detach().cpu().clone(),
        "torch_cuda_global_rng_state": (
            torch.cuda.get_rng_state(device).detach().cpu().clone()
            if device.type == "cuda"
            else None
        ),
        "sample_cursor": nf_sf_full_sequence_train_cursor(PRIVILEGED_CURRENT_TARGET_STEP),
        "metadata": dict(metadata),
        "final_train_record": None if final_record is None else dict(final_record),
        "validation": dict(validation),
        "fixed_probe": dict(fixed_probe),
        "teacher_parameter_sha256_before": dict(teacher_before["parameter_sha256"]),
        "teacher_parameter_sha256_after": dict(teacher_after["parameter_sha256"]),
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
                "schema": f"{PRIVILEGED_CURRENT_DISTILLATION_SCHEMA}_checkpoint_v1",
                "path": str(checkpoint_path.resolve()),
                "sha256": checkpoint_sha,
                "size_bytes": int(checkpoint_path.stat().st_size),
                "global_step": PRIVILEGED_CURRENT_TARGET_STEP,
                "diagnostic_only": True,
                "non_canonical": True,
                "canonical_training_eligible": False,
                "deployment_eligible": False,
                "optimizer_state_entry_count": len(optimizer.state),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
        newline="\n",
    )
    return checkpoint_path


def build_training_summary(
    *,
    metadata: Mapping[str, Any],
    metrics_path: Path,
    train_record_count: int,
    final_record: Mapping[str, Any] | None,
    checkpoint_records: Sequence[Mapping[str, Any]],
    validation_summaries: Sequence[Mapping[str, Any]],
    fixed_probe_path: Path,
    fixed_probe: Mapping[str, Any],
    gradient_probe: Mapping[str, Any],
    teacher_before: Mapping[str, Any],
    teacher_after: Mapping[str, Any],
    memory_maxima: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": PRIVILEGED_CURRENT_DISTILLATION_SCHEMA,
        "status": "DONE",
        "diagnostic_only": True,
        "non_canonical": True,
        "canonical_training_eligible": False,
        "deployment_eligible": False,
        "run_plan": dict(metadata["run_plan"]),
        "provenance": dict(metadata["provenance"]),
        "control_reuse_audit": dict(metadata["control_reuse_audit"]),
        "train_record_count": int(train_record_count),
        "metrics_jsonl": str(metrics_path.resolve()),
        "final_train_record": None if final_record is None else dict(final_record),
        "checkpoint_records": [dict(record) for record in checkpoint_records],
        "validation_reports": [dict(record) for record in validation_summaries],
        "fixed_probe_path": str(fixed_probe_path.resolve()),
        "fixed_probe": dict(fixed_probe),
        "gradient_safety_probe": dict(gradient_probe),
        "teacher_sha_unchanged": (
            teacher_before["parameter_sha256"]["aggregate_sha256"]
            == teacher_after["parameter_sha256"]["aggregate_sha256"]
        ),
        "teacher_before": dict(teacher_before),
        "teacher_after": dict(teacher_after),
        "memory_maxima": dict(memory_maxima),
    }


def _run_metadata(
    *,
    args: argparse.Namespace,
    parent: Any,
    runtime_git: str,
    resolved: Mapping[str, Any],
    optimizer_summary: Mapping[str, Any],
    restore: Mapping[str, Any],
    control_reuse: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": PRIVILEGED_CURRENT_DISTILLATION_SCHEMA,
        "status": "RUNNING",
        "run_plan": privileged_run_plan(),
        "provenance": provenance_contract(),
        "parent_checkpoint": {
            "path": str(parent.path),
            "sha256": str(parent.sha256),
            "git_sha": str(parent.parent_git_sha),
            "global_step": int(parent.parent_global_step),
            "semantic_lock_fingerprint": str(parent.semantic_lock_fingerprint),
        },
        "runtime_git_sha": str(runtime_git),
        "target_global_step": PRIVILEGED_CURRENT_TARGET_STEP,
        "lambda_priv": float(args.lambda_priv),
        "resolved_config": dict(resolved),
        "optimizer": dict(optimizer_summary),
        "restore_contract": dict(restore),
        "restored_rng_fingerprint": dict(restore["rng_fingerprint"]),
        "control_reuse_audit": dict(control_reuse),
    }


def _resolved_config_from_parent(
    parent_payload: Mapping[str, Any],
    *,
    args: argparse.Namespace,
) -> dict[str, Any]:
    resolved = dict(parent_payload["resolved_config"])
    resolved.update(
        {
            "schema": PRIVILEGED_CURRENT_DISTILLATION_SCHEMA,
            "base_schema": FULL_SEQUENCE_TRAINER_SCHEMA,
            "privileged_current_distillation": True,
            "canonical_objective_unchanged": True,
            "exact_fm_replaced": False,
            "lambda_priv": float(args.lambda_priv),
            "parent_checkpoint_sha256": PRIVILEGED_CURRENT_PARENT_CHECKPOINT_SHA256,
            "parent_checkpoint_git_sha": PRIVILEGED_CURRENT_PARENT_GIT_SHA,
            "parent_global_step": PRIVILEGED_CURRENT_PARENT_STEP,
            "target_global_step": PRIVILEGED_CURRENT_TARGET_STEP,
            "update_count": PRIVILEGED_CURRENT_UPDATE_COUNT,
            "teacher_checkpoint_sha256": PRIVILEGED_CURRENT_TEACHER_CHECKPOINT_SHA256,
        }
    )
    resolved["semantic_lock_fingerprint"] = semantic_lock_fingerprint(
        parent_payload["resolved_config"]
    )
    return resolved


def _probe_only_summary(
    *,
    args: argparse.Namespace,
    metadata: Mapping[str, Any],
    probe: Mapping[str, Any],
    teacher_before: Mapping[str, Any],
    teacher_after: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": PRIVILEGED_CURRENT_DISTILLATION_SCHEMA,
        "status": "GRADIENT_PROBE_ONLY_PASS",
        "diagnostic_only": True,
        "non_canonical": True,
        "run_plan": dict(metadata["run_plan"]),
        "gradient_safety_probe": dict(probe),
        "teacher_sha_unchanged": (
            teacher_before["parameter_sha256"]["aggregate_sha256"]
            == teacher_after["parameter_sha256"]["aggregate_sha256"]
        ),
        "output_dir": str(args.output_dir.resolve()),
    }


def _run_fixed_probe(
    *,
    helpers: Mapping[str, Any],
    generator: Any,
    scheduler_main: Any,
    scheduler_mcp: Any,
    teacher_store: M5TeacherSampleStore,
    conditional_store: M5ConditionalArtifactStore,
    sample_plan: Mapping[str, Any],
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    identity = str(sample_plan["fixed_decode_validation_identity"])
    with teacher_store.acquire(identity) as sample:
        with conditional_store.acquire(identity) as conditional_cpu:
            source_noise = sample.source_noise.to(device=device, dtype=dtype)
            teacher_target = helpers["target_latent_from_sample"](sample).to(
                device=device,
                dtype=dtype,
            )
            conditional = move_tensors_to_device(
                conditional_cpu,
                device=device,
                floating_dtype=dtype,
            )
            probe = run_fixed_raw999_probe_for_ablation(
                generator,
                arm="control",
                main_scheduler=scheduler_main,
                mcp_scheduler=scheduler_mcp,
                source_noise=source_noise,
                teacher_target=teacher_target,
                conditional_dict=conditional,
                sample_identity=identity,
                global_step=PRIVILEGED_CURRENT_TARGET_STEP,
            )
            probe["probe_route_arm"] = "control"
            probe["arm"] = "privileged_current_distillation"
            probe["experiment_schema"] = PRIVILEGED_CURRENT_DISTILLATION_SCHEMA
            probe["direct_clean_context_kv"] = False
            return probe


def _checkpoint_record(path: Path) -> dict[str, Any]:
    validation_path = path.with_suffix(".validation.json")
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    return {
        "path": str(path.resolve()),
        "sha256": str(validation["sha256"]),
        "size_bytes": int(validation["size_bytes"]),
        "global_step": int(validation["global_step"]),
        "schema": str(validation["schema"]),
    }


def _require_real_run_paths(args: argparse.Namespace) -> None:
    for name in ("sample_plan", "manifest", "dataset_root", "conditionals_artifact"):
        if getattr(args, name) is None:
            raise RuntimeError(f"--{name} is required with --execute_real_run")


def _validate_real_run_static_contract(args: argparse.Namespace, config: Any) -> None:
    if args.config.resolve() != CANONICAL_CONFIG_PATH:
        raise RuntimeError("privileged-current treatment requires canonical config")
    if str(args.device) != "cuda:0":
        raise RuntimeError("privileged-current treatment requires --device cuda:0")
    if str(args.dtype) != "bf16":
        raise RuntimeError("privileged-current treatment requires --dtype bf16")
    if int(args.log_interval) <= 0 or int(args.memory_log_interval) <= 0:
        raise RuntimeError("log intervals must be positive")
    if file_sha256(args.teacher_checkpoint) != PRIVILEGED_CURRENT_TEACHER_CHECKPOINT_SHA256:
        raise RuntimeError("Teacher checkpoint SHA mismatch")
    if file_sha256(args.checkpoint) != PRIVILEGED_CURRENT_TEACHER_CHECKPOINT_SHA256:
        raise RuntimeError("official initialization checkpoint SHA mismatch")
    if int(getattr(config, "num_frame_per_block", 0)) != 3:
        raise RuntimeError("privileged-current treatment requires chunk_frames=3")
    if int(getattr(config, "context_noise", -1)) != 0:
        raise RuntimeError("privileged-current treatment requires context_noise=0")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = run_privileged_current_distillation(args)
    print(
        json.dumps(
            {
                "schema": summary["schema"],
                "status": summary["status"],
                "diagnostic_only": summary.get("diagnostic_only", True),
                "run_plan": summary.get("run_plan"),
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


__all__ = [
    "INCONCLUSIVE",
    "NO_SUPPORT",
    "STRONG_SUPPORT",
    "build_training_summary",
    "classify_privileged_current_ab",
    "parse_args",
    "run_gradient_safety_probe",
    "run_privileged_current_distillation",
    "run_privileged_train_step",
    "save_privileged_checkpoint",
]


if __name__ == "__main__":
    raise SystemExit(main())
