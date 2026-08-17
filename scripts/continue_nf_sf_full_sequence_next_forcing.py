from __future__ import annotations

import argparse
import gc
import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from utils.nf_sf_full_sequence_continuation import (
    CONTINUATION_OBJECTIVE_MODE,
    CONTINUATION_SCHEMA,
    BASE_TRAINING_GIT_SHA,
    build_continuation_provenance,
    build_continuation_resolved_config,
    build_continuation_summary,
    continuation_checkpoint_steps,
    continuation_start_step,
    continuation_validation_steps,
    first_continuation_step_contract,
    load_continuation_parent_checkpoint,
    restore_continuation_state,
    rng_fingerprint,
    semantic_lock_fingerprint,
    validate_continuation_stage_pair,
    validate_optimizer_contract_for_continuation,
    validate_git_sha,
)
from utils.nf_sf_full_sequence_eval import (
    current_git_head,
    validate_repo_preflight,
)
from utils.nf_sf_m3 import file_sha256
from utils.nf_sf_m4 import load_m4_sample_plan
from utils.nf_sf_m5_conditionals import M5ConditionalArtifactStore
from utils.nf_sf_m5_samples import M5TeacherSampleStore
from utils.nf_sf_tensors import DEFAULT_S_MAIN, DEFAULT_S_MCP
from utils.nf_sf_training import (
    OFFICIAL_SELF_FORCING_CHECKPOINT_SHA256,
    nf_sf_full_sequence_train_cursor,
)


ROOT = Path(__file__).resolve().parents[1]
CANONICAL_CONFIG_PATH = (ROOT / "configs" / "self_forcing_dmd_mcp.yaml").resolve()


def _trainer_helpers() -> Mapping[str, Any]:
    from scripts import train_nf_sf_full_sequence_next_forcing as trainer

    return {
        "append_jsonl": trainer.append_jsonl,
        "build_fresh_generator": trainer.build_fresh_generator,
        "build_optimizer": trainer.build_optimizer,
        "checkpoint_record_from_path": trainer.checkpoint_record_from_path,
        "compact_train_record": trainer.compact_train_record,
        "compact_validation_summary": trainer.compact_validation_summary,
        "dtype_from_arg": trainer.dtype_from_arg,
        "make_flow_scheduler": trainer.make_flow_scheduler,
        "merge_config": trainer.merge_config,
        "optimizer_contract": trainer.optimizer_contract,
        "prepare_output_dir": trainer.prepare_output_dir,
        "run_full_sequence_train_step": trainer.run_full_sequence_train_step,
        "run_full_sequence_validation": trainer.run_full_sequence_validation,
        "save_full_sequence_checkpoint": trainer.save_full_sequence_checkpoint,
        "should_capture_memory": trainer.should_capture_memory,
        "should_run_cleanup_gc": trainer.should_run_cleanup_gc,
        "should_run_full_gradient_audit": trainer.should_run_full_gradient_audit,
        "update_memory_maxima": trainer.update_memory_maxima,
        "validate_sample_plan_contract": trainer.validate_sample_plan_contract,
        "validate_store_identity_order": trainer.validate_store_identity_order,
        "verify_reference_checkpoint_immutability": trainer.verify_reference_checkpoint_immutability,
        "write_m4_json": trainer.write_m4_json,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Strict NF-SF full-sequence continuation runner."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/self_forcing_dmd_mcp.yaml"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/self_forcing_dmd.pt"))
    parser.add_argument("--parent_checkpoint", required=True, type=Path)
    parser.add_argument("--expected_parent_checkpoint_sha256", required=True)
    parser.add_argument("--expected_parent_global_step", type=int, choices=(5000, 6500), required=True)
    parser.add_argument("--expected_parent_checkpoint_git_sha", required=True)
    parser.add_argument("--target_global_step", type=int, choices=(6500, 8000), required=True)
    parser.add_argument("--expected_runtime_git_sha", required=True)
    parser.add_argument("--sample_plan", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--dataset_root", required=True, type=Path)
    parser.add_argument("--conditionals_artifact", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bf16",), default="bf16")
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--memory_log_interval", type=int, default=100)
    return parser.parse_args(argv)


def run_continuation(args: argparse.Namespace) -> dict[str, Any]:
    helpers = _trainer_helpers()
    parent_step, target_step = validate_continuation_stage_pair(
        int(args.expected_parent_global_step),
        int(args.target_global_step),
    )
    if int(args.log_interval) <= 0 or int(args.memory_log_interval) <= 0:
        raise ValueError("log and memory intervals must be positive")
    runtime_git_sha = current_git_head()
    expected_runtime_git = validate_git_sha(
        args.expected_runtime_git_sha,
        name="--expected_runtime_git_sha",
    )
    if runtime_git_sha != expected_runtime_git:
        raise RuntimeError("current runtime git SHA mismatch")
    if parent_step == 5000 and str(args.expected_parent_checkpoint_git_sha) != BASE_TRAINING_GIT_SHA:
        raise RuntimeError("step5000 parent must use canonical base training git")
    if str(args.device) != "cuda:0" or str(args.dtype) != "bf16":
        raise RuntimeError("continuation runner requires --device cuda:0 --dtype bf16")
    if args.config.resolve() != CANONICAL_CONFIG_PATH:
        raise RuntimeError("config path must be configs/self_forcing_dmd_mcp.yaml")
    if file_sha256(args.checkpoint) != OFFICIAL_SELF_FORCING_CHECKPOINT_SHA256:
        raise RuntimeError("official Self-Forcing checkpoint SHA256 mismatch")

    config = helpers["merge_config"](args.config)
    sample_plan = load_m4_sample_plan(args.sample_plan, manifest_path=args.manifest)
    helpers["validate_sample_plan_contract"](sample_plan)
    conditional_store = M5ConditionalArtifactStore(
        artifact_dir=args.conditionals_artifact,
        sample_plan=sample_plan,
    )
    manifest_sha256 = file_sha256(args.manifest)
    parent = load_continuation_parent_checkpoint(
        args.parent_checkpoint,
        expected_parent_checkpoint_sha256=args.expected_parent_checkpoint_sha256,
        expected_parent_global_step=parent_step,
        expected_parent_checkpoint_git_sha=args.expected_parent_checkpoint_git_sha,
        sample_plan_sha256=str(sample_plan["sample_plan_sha256"]),
        manifest_sha256=manifest_sha256,
        conditionals_artifact_sha256=str(conditional_store.artifact_sha256),
    )
    repo_preflight = validate_repo_preflight(
        repo_root=ROOT,
        output_dir=args.output_dir,
        expected_runtime_git_sha=runtime_git_sha,
    )
    device = torch.device(args.device)
    dtype = helpers["dtype_from_arg"](args.dtype)
    parent_resolved = parent.payload["resolved_config"]
    if str(parent_resolved["device"]) != str(args.device) or str(parent_resolved["dtype"]) != str(args.dtype):
        raise RuntimeError("continuation device/dtype must match parent resolved_config")

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
    generator = helpers["build_fresh_generator"](
        config=config,
        checkpoint_path=args.checkpoint,
        device=device,
        dtype=dtype,
    )
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
    restored_rng = rng_fingerprint(
        train_rng=train_rng,
        validation_base_rng=validation_base_rng,
        device=device,
    )
    if restored["rng_fingerprint"] != restored_rng:
        raise RuntimeError("restored RNG fingerprint changed")

    resolved_config = build_continuation_resolved_config(
        parent_resolved,
        runtime_git_sha=runtime_git_sha,
        parent=parent,
        target_global_step=target_step,
    )
    provenance = build_continuation_provenance(
        parent.payload["provenance"],
        runtime_git_sha=runtime_git_sha,
        parent=parent,
        target_global_step=target_step,
    )
    semantic_fingerprint = semantic_lock_fingerprint(parent_resolved)
    if semantic_fingerprint != parent.semantic_lock_fingerprint:
        raise RuntimeError("semantic lock fingerprint changed after parent load")
    checkpoint_steps = continuation_checkpoint_steps(parent_step, target_step)
    validation_steps = continuation_validation_steps(parent_step, target_step)
    start_step = continuation_start_step(parent_step, target_step)
    first_identity = teacher_store.train_identity_for_step(start_step)
    first_contract = first_continuation_step_contract(
        parent_step=parent_step,
        target_step=target_step,
        train_identity=first_identity,
        sample_cursor=nf_sf_full_sequence_train_cursor(start_step),
    )
    scheduler_main = helpers["make_flow_scheduler"](DEFAULT_S_MAIN)
    scheduler_mcp = helpers["make_flow_scheduler"](DEFAULT_S_MCP)
    helpers["prepare_output_dir"](args.output_dir, resume=False)
    run_metadata = {
        "schema": CONTINUATION_SCHEMA,
        "runtime_git_sha": runtime_git_sha,
        "repo_preflight": repo_preflight,
        "parent_checkpoint": {
            "path": str(parent.path),
            "sha256": parent.sha256,
            "global_step": parent.parent_global_step,
            "git_sha": parent.parent_git_sha,
        },
        "target_global_step": int(target_step),
        "checkpoint_steps": list(checkpoint_steps),
        "validation_steps": list(validation_steps),
        "parent_checkpoint_resaved": False,
        "parent_validation_rerun": False,
        "semantic_lock_fingerprint": semantic_fingerprint,
        "restore_contract": restored,
        "first_continuation_step": first_contract,
    }
    helpers["write_m4_json"](run_metadata, args.output_dir / "run_metadata.json")

    metrics_path = args.output_dir / "metrics.jsonl"
    train_record_count = 0
    final_train_record = None
    checkpoint_records = []
    validation_summaries = []
    memory_maxima: dict[str, int] = {}
    for step in range(start_step, target_step + 1):
        started = time.perf_counter()
        record = helpers["run_full_sequence_train_step"](
            generator=generator,
            optimizer=optimizer,
            scheduler_main=scheduler_main,
            scheduler_mcp=scheduler_mcp,
            teacher_store=teacher_store,
            conditional_store=conditional_store,
            train_rng=train_rng,
            global_step=step,
            objective_mode=CONTINUATION_OBJECTIVE_MODE,
            device=device,
            dtype=dtype,
            smoke=False,
            full_gradient_audit=helpers["should_run_full_gradient_audit"](
                step,
                smoke=False,
                checkpoint_steps=checkpoint_steps,
                validation_steps=validation_steps,
                memory_log_interval=int(args.memory_log_interval),
            ),
            structural_gate=(step == start_step),
            run_gc=helpers["should_run_cleanup_gc"](
                step,
                smoke=False,
                checkpoint_steps=checkpoint_steps,
                validation_steps=validation_steps,
                memory_log_interval=int(args.memory_log_interval),
            ),
            capture_memory=helpers["should_capture_memory"](
                step,
                smoke=False,
                checkpoint_steps=checkpoint_steps,
                validation_steps=validation_steps,
                memory_log_interval=int(args.memory_log_interval),
            ),
        )
        record["elapsed_ms"] = float((time.perf_counter() - started) * 1000.0)
        compact = helpers["compact_train_record"](record, device=device)
        helpers["update_memory_maxima"](memory_maxima, compact.get("memory"))
        helpers["append_jsonl"](
            metrics_path,
            compact,
            fsync=(step % int(args.log_interval) == 0 or step == target_step),
        )
        train_record_count += 1
        final_train_record = compact
        if step % int(args.log_interval) == 0 or step == target_step:
            helpers["write_m4_json"](compact, args.output_dir / f"train_step{step:06d}.json")
        if step in validation_steps:
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
                global_step=step,
            )
            validation_path = args.output_dir / f"validation_step{step:06d}.json"
            helpers["write_m4_json"](validation, validation_path)
            validation_summaries.append(
                helpers["compact_validation_summary"](validation, path=validation_path)
            )
        if step in checkpoint_steps:
            checkpoint_path = helpers["save_full_sequence_checkpoint"](
                output_dir=args.output_dir,
                generator=generator,
                optimizer=optimizer,
                global_step=step,
                train_rng=train_rng,
                validation_base_rng=validation_base_rng,
                validation_seed=int(parent_resolved["validation_seed"]),
                sample_plan=sample_plan,
                resolved_config=resolved_config,
                provenance=provenance,
                git_sha=runtime_git_sha,
                reference_checkpoint_path=args.checkpoint,
                objective_mode=CONTINUATION_OBJECTIVE_MODE,
                smoke=False,
                extra_metadata={
                    "continuation_schema": CONTINUATION_SCHEMA,
                    "parent_global_step": parent_step,
                    "target_global_step": target_step,
                    "parent_checkpoint_sha256": parent.sha256,
                    "parent_checkpoint_git_sha": parent.parent_git_sha,
                },
            )
            checkpoint_records.append(helpers["checkpoint_record_from_path"](checkpoint_path))

    reference_immutability = helpers["verify_reference_checkpoint_immutability"](
        reference_checkpoint_path=args.checkpoint,
        preflight_report={
            "checkpoint_sha256": OFFICIAL_SELF_FORCING_CHECKPOINT_SHA256,
            "checkpoint_size_bytes": int(args.checkpoint.stat().st_size),
        },
    )
    summary = build_continuation_summary(
        parent=parent,
        runtime_git_sha=runtime_git_sha,
        target_global_step=target_step,
        metrics_path=metrics_path,
        train_record_count=train_record_count,
        final_train_record=final_train_record,
        checkpoint_records=checkpoint_records,
        validation_summaries=validation_summaries,
        semantic_lock_fingerprint_value=semantic_fingerprint,
        restored_rng_fingerprint=restored_rng,
        first_step_contract=first_contract,
        reference_checkpoint_immutability=reference_immutability,
        memory_maxima=memory_maxima,
    )
    helpers["write_m4_json"](summary, args.output_dir / "continuation_summary.json")
    generator.to("cpu")
    del generator
    gc.collect()
    torch.cuda.empty_cache()
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    manifest = run_continuation(parse_args(argv))
    print(
        json.dumps(
            {
                "schema": manifest["schema"],
                "status": manifest["status"],
                "parent_global_step": manifest["parent_global_step"],
                "target_global_step": manifest["target_global_step"],
                "train_record_count": manifest["train_record_count"],
            },
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
