from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.train_nf_sf_m3_overfit as train_m3
from inference_mcp import merge_config, require_single_gpu_runtime
from utils.nf_sf_m5_validation import M5_STREAMING_VALIDATION_SCHEMA

M5_FORMAL_SMOKE_SCHEMA = "nf_sf_m5_formal_smoke_v1"
M5_FORMAL_SMOKE_RUN_KIND = "short_smoke"
M5_FORMAL_SMOKE_FRESH_TARGET = 2
M5_FORMAL_SMOKE_RESUME_PARENT = 2
M5_FORMAL_SMOKE_RESUME_TARGET = 3


@dataclass(frozen=True, slots=True)
class SmokeContract:
    parent_global_step: int | None
    target_global_step: int
    validation_steps: tuple[int, ...]
    checkpoint_steps: tuple[int, ...]
    is_resume: bool


class SmokeLifecycleAudit:
    def __init__(self, *, run_kind: str) -> None:
        self.run_kind = run_kind
        self.scope = {"phase": "setup", "step": None}
        self.teacher_acquire_count = 0
        self.teacher_release_count = 0
        self.conditional_acquire_count = 0
        self.conditional_release_count = 0
        self.teacher_max_live_count = 0
        self.conditional_max_live_count = 0
        self.teacher_end_live_count = 0
        self.conditional_end_live_count = 0
        self.store_events: list[dict[str, Any]] = []
        self.train_step_store_counts: dict[str, dict[str, int]] = {}
        self.step_state_records: list[dict[str, Any]] = []
        self.device_conditional_records: list[dict[str, Any]] = []
        self.train_step_memory: list[dict[str, Any]] = []
        self.validation_memory: list[dict[str, Any]] = []
        self.checkpoint_memory: list[dict[str, Any]] = []
        self.markers: list[dict[str, Any]] = []

    @contextmanager
    def scoped(self, phase: str, step: int | None) -> Iterator[None]:
        previous = dict(self.scope)
        self.scope = {"phase": str(phase), "step": step}
        try:
            yield
        finally:
            self.scope = previous

    def mark(self, name: str, **fields: Any) -> None:
        marker = {
            "name": str(name),
            "index": len(self.markers),
            "phase": str(self.scope["phase"]),
            "step": self.scope["step"],
        }
        marker.update(_json_safe(fields, "marker"))
        self.markers.append(marker)

    def record_store_event(
        self,
        *,
        kind: str,
        action: str,
        identity: str,
        live_count: int,
        max_live_count: int,
    ) -> None:
        if kind == "teacher":
            if action == "acquire":
                self.teacher_acquire_count += 1
            elif action == "release":
                self.teacher_release_count += 1
            self.teacher_max_live_count = max(self.teacher_max_live_count, max_live_count)
            self.teacher_end_live_count = int(live_count)
        elif kind == "conditional":
            if action == "acquire":
                self.conditional_acquire_count += 1
            elif action == "release":
                self.conditional_release_count += 1
            self.conditional_max_live_count = max(
                self.conditional_max_live_count,
                max_live_count,
            )
            self.conditional_end_live_count = int(live_count)
        else:
            raise ValueError(f"unknown store kind: {kind}")

        step = self.scope["step"]
        if self.scope["phase"] == "train" and step is not None:
            counts = self.train_step_store_counts.setdefault(
                str(step),
                {
                    "teacher_acquire": 0,
                    "teacher_release": 0,
                    "conditional_acquire": 0,
                    "conditional_release": 0,
                },
            )
            counts[f"{kind}_{action}"] += 1

        self.store_events.append(
            {
                "index": len(self.store_events),
                "kind": kind,
                "action": action,
                "identity": str(identity),
                "phase": str(self.scope["phase"]),
                "step": step,
                "live_count": int(live_count),
                "max_live_count": int(max_live_count),
            }
        )

    def record_state(self, value: Any) -> None:
        self.step_state_records.append(
            {
                "index": len(self.step_state_records),
                "phase": str(self.scope["phase"]),
                "step": self.scope["step"],
                "state": _selected_state_metadata(value),
            }
        )

    def record_device_conditional(self, value: Mapping[str, Any]) -> None:
        self.device_conditional_records.append(
            {
                "index": len(self.device_conditional_records),
                "phase": str(self.scope["phase"]),
                "step": self.scope["step"],
                "conditional": _tensor_mapping_metadata(value),
            }
        )

    def record_train_memory(
        self,
        *,
        step: int,
        moment: str,
        device: torch.device,
    ) -> None:
        self.train_step_memory.append(
            {
                "step": int(step),
                "moment": str(moment),
                "cuda": cuda_memory_snapshot(device),
            }
        )

    def record_validation_memory(
        self,
        *,
        step: int,
        moment: str,
        device: torch.device,
    ) -> None:
        self.validation_memory.append(
            {
                "step": int(step),
                "moment": str(moment),
                "cuda": cuda_memory_snapshot(device),
            }
        )

    def record_checkpoint_memory(
        self,
        *,
        step: int,
        moment: str,
        device: torch.device,
    ) -> None:
        self.checkpoint_memory.append(
            {
                "step": int(step),
                "moment": str(moment),
                "cuda": cuda_memory_snapshot(device),
            }
        )

    def report(
        self,
        *,
        teacher_store: Any,
        conditional_store: Any,
        executed_global_steps: Sequence[int],
        validation_reports: Sequence[Mapping[str, Any]],
        train_loss_records: Sequence[Mapping[str, Any]],
        checkpoint_records: Sequence[Mapping[str, Any]],
        contract: SmokeContract,
        provenance: Mapping[str, Any],
    ) -> dict[str, Any]:
        self.teacher_end_live_count = int(getattr(teacher_store, "live_sample_count", 0))
        self.conditional_end_live_count = int(
            getattr(conditional_store, "live_conditional_count", 0)
        )
        self.teacher_max_live_count = max(
            self.teacher_max_live_count,
            int(getattr(teacher_store, "max_live_sample_count", 0)),
        )
        self.conditional_max_live_count = max(
            self.conditional_max_live_count,
            int(getattr(conditional_store, "max_live_conditional_count", 0)),
        )
        report = {
            "schema": M5_FORMAL_SMOKE_SCHEMA,
            "status": "PASS",
            "run_kind": self.run_kind,
            "target_global_step": int(contract.target_global_step),
            "parent_global_step": contract.parent_global_step,
            "executed_global_steps": [int(step) for step in executed_global_steps],
            "validation_steps": [
                int(report["global_step"]) for report in validation_reports
            ],
            "checkpoint_steps": [
                int(record["global_step"]) for record in checkpoint_records
            ],
            "teacher": {
                "acquire_count": int(self.teacher_acquire_count),
                "release_count": int(self.teacher_release_count),
                "max_live_count": int(self.teacher_max_live_count),
                "end_live_count": int(self.teacher_end_live_count),
            },
            "conditional": {
                "acquire_count": int(self.conditional_acquire_count),
                "release_count": int(self.conditional_release_count),
                "max_live_count": int(self.conditional_max_live_count),
                "end_live_count": int(self.conditional_end_live_count),
            },
            "train_step_store_counts": self.train_step_store_counts,
            "store_events": self.store_events,
            "step_state_records": self.step_state_records,
            "device_conditional_records": self.device_conditional_records,
            "cuda_memory": {
                "train_steps": self.train_step_memory,
                "validation": self.validation_memory,
                "checkpoint": self.checkpoint_memory,
            },
            "markers": self.markers,
            "validation_reports": [
                _validation_report_summary(report) for report in validation_reports
            ],
            "train_loss_records": [
                _json_safe(dict(record), "train_loss_records[]")
                for record in train_loss_records
            ],
            "checkpoint_records": [
                _json_safe(dict(record), "checkpoint_records[]")
                for record in checkpoint_records
            ],
            "provenance": _json_safe(dict(provenance), "provenance"),
        }
        _require_smoke_pass_contract(report)
        _assert_no_tensors(report, "smoke_audit_report")
        return report


class AuditedTeacherStore:
    def __init__(self, store: Any, audit: SmokeLifecycleAudit) -> None:
        self._store = store
        self._audit = audit

    def __getattr__(self, name: str) -> Any:
        return getattr(self._store, name)

    def train_identity_for_step(self, step: int) -> str:
        return str(self._store.train_identity_for_step(step))

    @contextmanager
    def acquire(self, identity: str) -> Iterator[Any]:
        entered = False
        try:
            with self._store.acquire(identity) as sample:
                entered = True
                self._audit.record_store_event(
                    kind="teacher",
                    action="acquire",
                    identity=identity,
                    live_count=int(getattr(self._store, "live_sample_count", 0)),
                    max_live_count=int(
                        getattr(self._store, "max_live_sample_count", 0)
                    ),
                )
                yield sample
        finally:
            if entered:
                self._audit.record_store_event(
                    kind="teacher",
                    action="release",
                    identity=identity,
                    live_count=int(getattr(self._store, "live_sample_count", 0)),
                    max_live_count=int(
                        getattr(self._store, "max_live_sample_count", 0)
                    ),
                )


class AuditedConditionalStore:
    def __init__(self, store: Any, audit: SmokeLifecycleAudit) -> None:
        self._store = store
        self._audit = audit

    def __getattr__(self, name: str) -> Any:
        return getattr(self._store, name)

    @contextmanager
    def acquire(self, identity: str) -> Iterator[dict[str, torch.Tensor]]:
        entered = False
        try:
            with self._store.acquire(identity) as conditional:
                entered = True
                self._audit.record_store_event(
                    kind="conditional",
                    action="acquire",
                    identity=identity,
                    live_count=int(getattr(self._store, "live_conditional_count", 0)),
                    max_live_count=int(
                        getattr(self._store, "max_live_conditional_count", 0)
                    ),
                )
                yield conditional
        finally:
            if entered:
                self._audit.record_store_event(
                    kind="conditional",
                    action="release",
                    identity=identity,
                    live_count=int(getattr(self._store, "live_conditional_count", 0)),
                    max_live_count=int(
                        getattr(self._store, "max_live_conditional_count", 0)
                    ),
                )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Isolated NF-SF M5 formal short smoke trainer."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--resume_checkpoint", type=Path, default=None)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--dataset_root", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--mode", default="joint")
    parser.add_argument("--train_seed", type=int, required=True)
    parser.add_argument("--probe_seed", type=int, required=True)
    parser.add_argument("--target_global_step", type=int, required=True)
    parser.add_argument("--parent_global_step", type=int, default=None)
    parser.add_argument("--timing_warmup_steps", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=1)
    parser.add_argument("--checkpoint_interval", type=int, default=1)
    parser.add_argument("--backbone_lr", type=float, required=True)
    parser.add_argument("--patch_embedding_lr", type=float, required=True)
    parser.add_argument("--mcp_lr", type=float, required=True)
    parser.add_argument("--weight_decay", type=float, required=True)
    parser.add_argument("--mcp1_grid_aux_weight", type=float, default=0.0)
    parser.add_argument("--m4_sample_plan", required=True, type=Path)
    parser.add_argument("--m5_conditionals_artifact", required=True, type=Path)
    parser.add_argument("--validation_seed", required=True, type=int)
    parser.add_argument("--dtype", choices=("bf16", "float32"), default="bf16")
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def validate_smoke_cli_contract(
    args: argparse.Namespace,
    *,
    cuda_device_count: int,
) -> SmokeContract:
    if str(args.mode) != "joint":
        raise ValueError("M5 formal smoke requires --mode joint")
    if str(args.device) != "cuda:0":
        raise ValueError("M5 formal smoke requires --device cuda:0")
    if int(cuda_device_count) != 1:
        raise ValueError(
            "M5 formal smoke requires exactly one CUDA device: "
            f"actual={cuda_device_count}"
        )
    if float(args.mcp1_grid_aux_weight) != 0.0:
        raise ValueError("M5 formal smoke requires --mcp1_grid_aux_weight 0")
    _require_existing_file(args.checkpoint, "--checkpoint")
    _require_existing_file(args.manifest, "--manifest")
    _require_existing_directory(args.dataset_root, "--dataset_root")
    _require_existing_file(args.m4_sample_plan, "--m4_sample_plan")
    train_m3.require_m5_formal_conditionals_manifest_path(
        args.m5_conditionals_artifact
    )
    if args.validation_seed is None:
        raise ValueError("M5 formal smoke requires --validation_seed")

    parent = args.parent_global_step
    target = int(args.target_global_step)
    if args.resume_checkpoint is None:
        if parent is not None:
            raise ValueError("fresh smoke requires parent_global_step=None")
        if target != M5_FORMAL_SMOKE_FRESH_TARGET:
            raise ValueError("fresh smoke requires target_global_step=2")
        return SmokeContract(
            parent_global_step=None,
            target_global_step=M5_FORMAL_SMOKE_FRESH_TARGET,
            validation_steps=(0, M5_FORMAL_SMOKE_FRESH_TARGET),
            checkpoint_steps=(0, M5_FORMAL_SMOKE_FRESH_TARGET),
            is_resume=False,
        )

    _require_existing_file(args.resume_checkpoint, "--resume_checkpoint")
    if parent != M5_FORMAL_SMOKE_RESUME_PARENT:
        raise ValueError("resume smoke requires parent_global_step=2")
    if target != M5_FORMAL_SMOKE_RESUME_TARGET:
        raise ValueError("resume smoke requires target_global_step=3")
    return SmokeContract(
        parent_global_step=M5_FORMAL_SMOKE_RESUME_PARENT,
        target_global_step=M5_FORMAL_SMOKE_RESUME_TARGET,
        validation_steps=(M5_FORMAL_SMOKE_RESUME_TARGET,),
        checkpoint_steps=(M5_FORMAL_SMOKE_RESUME_TARGET,),
        is_resume=True,
    )


def run_m5_formal_smoke(
    *,
    args: argparse.Namespace,
    config: Any,
    dtype: torch.dtype,
    device: torch.device,
    current_git_sha: str,
    reference_checkpoint_sha256: str,
) -> dict[str, Any]:
    contract = validate_smoke_cli_contract(
        args,
        cuda_device_count=torch.cuda.device_count(),
    )
    _require_output_dir_independent_empty(
        args.output_dir,
        resume_checkpoint=args.resume_checkpoint,
    )

    conditionals_manifest_path = train_m3.require_m5_formal_conditionals_manifest_path(
        args.m5_conditionals_artifact
    )
    m4_plan = train_m3.load_m4_sample_plan(
        args.m4_sample_plan,
        manifest_path=args.manifest,
    )
    saved_plan_sha = str(m4_plan["sample_plan_sha256"])
    formal_plan_audit = train_m3.validate_m5_formal_sample_plan(
        m4_plan,
        manifest_path=args.manifest,
        dataset_root=args.dataset_root,
        expected_sha256=saved_plan_sha,
    )
    conditionals_manifest = train_m3.load_m5_conditional_artifact_manifest(
        conditionals_manifest_path
    )
    conditional_artifact_dir = train_m3.m5_formal_artifact_dir_from_manifest(
        conditionals_manifest_path
    )
    conditional_audit = train_m3.validate_m5_conditional_artifact_manifest(
        conditionals_manifest,
        artifact_dir=conditional_artifact_dir,
        sample_plan=m4_plan,
    )
    conditional_artifact_sha256 = str(conditional_audit["artifact_sha256"])

    teacher_store_raw = train_m3.M5TeacherSampleStore(
        sample_plan=m4_plan,
        manifest_path=args.manifest,
        dataset_root=args.dataset_root,
        reference_checkpoint_path=args.checkpoint,
    )
    conditional_store_raw = train_m3.M5ConditionalArtifactStore(
        artifact_dir=conditional_artifact_dir,
        sample_plan=m4_plan,
        expected_artifact_sha256=conditional_artifact_sha256,
    )
    _require_store_provenance(
        teacher_store=teacher_store_raw,
        conditional_store=conditional_store_raw,
        sample_plan_sha256=saved_plan_sha,
        teacher_manifest_sha256=str(formal_plan_audit["manifest_sha256"]),
    )

    parent_resume_payload = None
    parent_resume_sha256 = None
    parent_smoke_metadata = None
    if contract.is_resume:
        parent_resume_payload, parent_resume_sha256 = train_m3.load_parent_resume_checkpoint(
            args.resume_checkpoint
        )
        parent_smoke_metadata = require_m5_smoke_parent_checkpoint(
            parent_payload=parent_resume_payload,
            sample_plan_sha256=saved_plan_sha,
            teacher_manifest_sha256=str(formal_plan_audit["manifest_sha256"]),
            conditional_artifact_sha256=conditional_artifact_sha256,
            current_git_sha=current_git_sha,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_m3.write_run_json(
        m4_plan,
        args.output_dir / "m4_sample_plan.json",
        strict=True,
    )
    (args.output_dir / "reference_checkpoint_sha256.txt").write_text(
        reference_checkpoint_sha256 + "\n",
        encoding="utf-8",
    )
    if parent_resume_sha256 is not None:
        (args.output_dir / "parent_checkpoint_sha256.txt").write_text(
            parent_resume_sha256 + "\n",
            encoding="utf-8",
        )
    (args.output_dir / "git_sha.txt").write_text(
        current_git_sha + "\n",
        encoding="utf-8",
    )

    optimizer_config = {
        "optimizer": "AdamW",
        "betas": [float(value) for value in train_m3.ADAMW_BETAS],
        "eps": train_m3.ADAMW_EPS,
        "weight_decay": float(args.weight_decay),
    }
    resolved_config = build_m5_smoke_resolved_config(
        config=config,
        args=args,
        device=device,
        contract=contract,
        sample_plan=m4_plan,
        sample_plan_sha256=saved_plan_sha,
        formal_plan_audit=formal_plan_audit,
        conditionals_manifest_path=conditionals_manifest_path,
        conditionals_manifest=conditionals_manifest,
        conditional_artifact_sha256=conditional_artifact_sha256,
        optimizer_config=optimizer_config,
    )
    train_m3.write_run_json(
        resolved_config,
        args.output_dir / "resolved_config.json",
        strict=True,
    )
    checkpoint_metadata_base = m5_smoke_checkpoint_metadata(
        contract=contract,
        sample_plan_sha256=saved_plan_sha,
        teacher_manifest_sha256=str(formal_plan_audit["manifest_sha256"]),
        conditional_artifact_sha256=conditional_artifact_sha256,
        current_git_sha=current_git_sha,
        parent_checkpoint_sha256=parent_resume_sha256,
    )

    audit = SmokeLifecycleAudit(run_kind=M5_FORMAL_SMOKE_RUN_KIND)
    teacher_store = AuditedTeacherStore(teacher_store_raw, audit)
    conditional_store = AuditedConditionalStore(conditional_store_raw, audit)
    metrics_path = args.output_dir / "metrics.jsonl"
    generator = None
    train_rng = None
    validation_reports: list[dict[str, Any]] = []
    train_loss_records: list[dict[str, Any]] = []
    checkpoint_records: list[dict[str, Any]] = []
    step_timing_records: list[dict[str, Any]] = []
    executed_global_steps: list[int] = []
    resume_report = None
    fixed_state = None
    restored_probe_forward = None
    try:
        with _audit_train_m3_device_transfers(audit):
            if not contract.is_resume:
                train_m3.reset_global_seed(args.train_seed)
            generator, load_mode, mcp_tensor_count = train_m3.load_generator(
                config,
                args.checkpoint,
            )
            generator.to(device=device, dtype=dtype)
            generator.train()
            scheduler_main = generator.get_scheduler()
            scheduler_main.sigmas = scheduler_main.sigmas.to(device)
            scheduler_main.timesteps = scheduler_main.timesteps.to(device)
            scheduler_mcp = train_m3.make_mcp_scheduler(device)

            group_lrs = {
                "backbone": float(args.backbone_lr),
                "patch_embedding": float(args.patch_embedding_lr),
                "mcp": float(args.mcp_lr),
            }
            optimizer_plan = train_m3.configure_m3_optimizer_plan(
                generator,
                mode=args.mode,
                group_lrs=group_lrs,
            )
            optimizer = torch.optim.AdamW(
                optimizer_plan.optimizer_param_groups,
                betas=train_m3.ADAMW_BETAS,
                eps=train_m3.ADAMW_EPS,
                weight_decay=float(args.weight_decay),
            )
            train_m3.write_run_json(
                {
                    "mode": optimizer_plan.mode,
                    "optimizer_config": train_m3.optimizer_config_summary(optimizer),
                    "param_audit": [
                        train_m3.audit_to_json(audit_item)
                        for audit_item in optimizer_plan.audits
                    ],
                    "optimizer_group_lrs": train_m3.optimizer_group_lr_summary(
                        optimizer
                    ),
                    "checkpoint_load_mode": load_mode,
                    "mcp_tensor_count": mcp_tensor_count,
                },
                args.output_dir / "optimizer_audit.json",
                strict=True,
            )

            if contract.is_resume:
                with audit.scoped("resume_fixed_sample", contract.parent_global_step):
                    fixed_sample_metadata, fixed_state = (
                        train_m3.acquire_m5_formal_fixed_sample_state(
                            teacher_store=teacher_store,
                            sample_plan=m4_plan,
                            device=device,
                            dtype=dtype,
                        )
                    )
                probe = None
                fixed_prompt_embedding: dict[str, Any] = {}
            else:
                with audit.scoped("fixed_probe", 0):
                    fixed_sample_metadata, probe, fixed_prompt_embedding = (
                        train_m3.acquire_m5_formal_fixed_probe_inputs(
                            teacher_store=teacher_store,
                            conditional_store=conditional_store,
                            sample_plan=m4_plan,
                            scheduler_main=scheduler_main,
                            scheduler_mcp=scheduler_mcp,
                            device=device,
                            dtype=dtype,
                            probe_seed=int(args.probe_seed),
                        )
                    )
            train_m3.write_run_json(
                fixed_sample_metadata,
                args.output_dir / "sample_metadata.json",
                strict=True,
            )

            resumed_global_step = None
            if contract.is_resume:
                assert parent_resume_payload is not None
                assert parent_resume_sha256 is not None
                assert args.resume_checkpoint is not None
                current_run_fields = train_m3.current_m5_resume_run_fields(
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
                resume_report = train_m3.build_and_validate_m5_resume_report(
                    parent_payload=parent_resume_payload,
                    parent_checkpoint_path=args.resume_checkpoint,
                    parent_checkpoint_sha256=parent_resume_sha256,
                    current_run_fields=current_run_fields,
                    target_global_step=int(contract.target_global_step),
                    sample_plan=m4_plan,
                    output_dir=args.output_dir,
                    target_validation_steps=contract.validation_steps,
                    target_checkpoint_steps=contract.checkpoint_steps,
                    expected_cuda_device_count=1,
                )
                train_m3.require_m5_resume_devices(
                    resume_report,
                    train_device=device,
                    probe_device=device,
                )
                generator_restore = train_m3.strict_load_m5_generator_state(
                    generator,
                    parent_resume_payload["generator"],
                )
                optimizer.load_state_dict(parent_resume_payload["optimizer"])
                optimizer_device_report = train_m3.move_loaded_optimizer_state_to_device(
                    optimizer,
                    device=device,
                )
                rng_states = train_m3.extract_resume_rng_states(parent_resume_payload)
                train_rng = train_m3.restore_torch_generator_from_state(
                    rng_states["train_generator_state"],
                    device=device,
                )
                with audit.scoped("resume_probe_restore", contract.parent_global_step):
                    restored_probe, restored_prompt_embedding = (
                        train_m3.restore_m5_probe_from_checkpoint(
                            parent_payload=parent_resume_payload,
                            selected_state=fixed_state,
                            device=device,
                            dtype=dtype,
                        )
                    )
                    restored_probe_forward = train_m3.run_m3_probe_forward(
                        generator,
                        conditional_dict=restored_prompt_embedding,
                        noisy_batch=restored_probe.noisy_batch,
                    )
                    probe_restore = train_m3.require_restored_probe_matches_checkpoint(
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
                        "smoke_parent_metadata": parent_smoke_metadata,
                        "generator_restore": generator_restore,
                        "optimizer_device_restore": optimizer_device_report,
                        "probe_restore": probe_restore,
                        "conditional_artifact_restore": {
                            "status": "PASS",
                            "artifact_sha256": conditional_artifact_sha256,
                            "manifest_path": str(
                                conditionals_manifest_path.resolve()
                            ),
                        },
                    }
                )
                train_m3.write_run_json(
                    resume_report,
                    args.output_dir / "resume_report.json",
                    strict=True,
                )
                parent_resume_payload = None
                audit.mark("parent_payload_released")
                restored_probe_forward = None
                audit.mark("restored_probe_temporary_released")
                gc.collect()
                audit.mark("gc_collect_after_parent_release")
                train_m3.restore_global_rng_states(rng_states)
                audit.mark("global_rng_restored")
            else:
                train_rng = train_m3.make_generator(args.train_seed, device)
                if 0 in contract.validation_steps:
                    report = _run_smoke_validation(
                        audit=audit,
                        step=0,
                        device=device,
                        generator=generator,
                        teacher_store=teacher_store,
                        conditional_store=conditional_store,
                        scheduler_main=scheduler_main,
                        scheduler_mcp=scheduler_mcp,
                        dtype=dtype,
                        mode=args.mode,
                        sample_plan=m4_plan,
                        validation_seed=int(args.validation_seed),
                        train_rng=train_rng,
                        probe=probe,
                        output_dir=args.output_dir,
                        metrics_path=metrics_path,
                        current_git_sha=current_git_sha,
                        reference_checkpoint_sha256=reference_checkpoint_sha256,
                        validation_reports=validation_reports,
                    )
                    _require_validation_pass(report, step=0)
                if 0 in contract.checkpoint_steps:
                    checkpoint_path = _write_smoke_probe_checkpoint_node(
                        audit=audit,
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
                        train_seed=int(args.train_seed),
                        probe_seed=int(args.probe_seed),
                        fixed_prompt_embedding=fixed_prompt_embedding,
                        device=device,
                        checkpoint_metadata={
                            **checkpoint_metadata_base,
                            "checkpoint_global_step": 0,
                        },
                    )
                    checkpoint_records.append(
                        _checkpoint_record(0, checkpoint_path, checkpoint_metadata_base)
                    )

            train_m3.reset_peak_memory_stats_if_available(device)
            assert train_rng is not None
            assert probe is not None
            for step in train_m3.m5_absolute_training_steps(
                resumed_global_step=resumed_global_step,
                target_global_step=contract.target_global_step,
            ):
                if contract.is_resume:
                    audit.mark("first_resumed_step_begin", step=int(step))
                audit.record_train_memory(step=step, moment="start", device=device)
                train_m3.cuda_synchronize_if_available(device)
                started = time.perf_counter()
                with audit.scoped("train", int(step)):
                    metric_record, train_losses, aux_report, grad_audit = (
                        train_m3.run_m5_formal_train_step(
                            step=step,
                            target_global_step=contract.target_global_step,
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
                            validation_steps=contract.validation_steps,
                            checkpoint_steps=contract.checkpoint_steps,
                            checkpoint_interval=int(args.checkpoint_interval),
                            log_interval=int(args.log_interval),
                            timing_warmup_steps=int(args.timing_warmup_steps),
                        )
                    )
                train_m3.cuda_synchronize_if_available(device)
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                metric_record["elapsed_ms"] = elapsed_ms
                metric_record["timing"] = train_m3.m5_timing_record(
                    global_step=step,
                    elapsed_ms=elapsed_ms,
                    timing_warmup_steps=int(args.timing_warmup_steps),
                )
                step_timing_records.append(metric_record["timing"])
                executed_global_steps.append(int(step))
                train_loss_records.append(
                    {
                        "global_step": int(step),
                        "losses": {
                            key: float(value) for key, value in train_losses.items()
                        },
                    }
                )
                audit.record_train_memory(step=step, moment="end", device=device)

                if bool(metric_record["should_validate"]):
                    report = _run_smoke_validation(
                        audit=audit,
                        step=int(step),
                        device=device,
                        generator=generator,
                        teacher_store=teacher_store,
                        conditional_store=conditional_store,
                        scheduler_main=scheduler_main,
                        scheduler_mcp=scheduler_mcp,
                        dtype=dtype,
                        mode=args.mode,
                        sample_plan=m4_plan,
                        validation_seed=int(args.validation_seed),
                        train_rng=train_rng,
                        probe=probe,
                        output_dir=args.output_dir,
                        metrics_path=metrics_path,
                        current_git_sha=current_git_sha,
                        reference_checkpoint_sha256=reference_checkpoint_sha256,
                        validation_reports=validation_reports,
                    )
                    _require_validation_pass(report, step=int(step))

                if bool(metric_record["should_checkpoint"]):
                    checkpoint_path = _write_smoke_step_artifacts(
                        audit=audit,
                        output_dir=args.output_dir,
                        metrics_path=metrics_path,
                        generator=generator,
                        optimizer=optimizer,
                        global_step=int(step),
                        metric_record=metric_record,
                        train_rng=train_rng,
                        probe=probe,
                        sample_metadata=fixed_sample_metadata,
                        resolved_config=resolved_config,
                        current_git_sha=current_git_sha,
                        reference_checkpoint_path=args.checkpoint,
                        reference_checkpoint_sha256=reference_checkpoint_sha256,
                        train_seed=int(args.train_seed),
                        probe_seed=int(args.probe_seed),
                        prompt_embedding=fixed_prompt_embedding,
                        device=device,
                        train_losses=train_losses,
                        aux_report=aux_report,
                        grad_audit=grad_audit,
                        checkpoint_metadata={
                            **checkpoint_metadata_base,
                            "checkpoint_global_step": int(step),
                        },
                    )
                    checkpoint_records.append(
                        _checkpoint_record(
                            int(step),
                            checkpoint_path,
                            checkpoint_metadata_base,
                        )
                    )

        provenance = {
            "sample_plan_sha256": saved_plan_sha,
            "teacher_manifest_sha256": str(formal_plan_audit["manifest_sha256"]),
            "conditional_artifact_sha256": conditional_artifact_sha256,
            "git_sha": current_git_sha,
            "reference_checkpoint_sha256": reference_checkpoint_sha256,
            "parent_checkpoint_sha256": parent_resume_sha256,
            "validation_implementation_schema": M5_STREAMING_VALIDATION_SCHEMA,
        }
        audit_report = audit.report(
            teacher_store=teacher_store,
            conditional_store=conditional_store,
            executed_global_steps=executed_global_steps,
            validation_reports=validation_reports,
            train_loss_records=train_loss_records,
            checkpoint_records=checkpoint_records,
            contract=contract,
            provenance=provenance,
        )
        summary = {
            "schema": M5_FORMAL_SMOKE_SCHEMA,
            "status": "PASS",
            "smoke_enabled": True,
            "formal_enabled": False,
            "run_kind": M5_FORMAL_SMOKE_RUN_KIND,
            "output_dir": str(args.output_dir.resolve()),
            "mode": args.mode,
            "target_global_step": int(contract.target_global_step),
            "parent_global_step": contract.parent_global_step,
            "executed_global_steps": executed_global_steps,
            "validation_reports": [
                f"validation_step{int(report['global_step']):06d}.json"
                for report in validation_reports
            ],
            "checkpoint_records": checkpoint_records,
            "sample_plan_sha256": saved_plan_sha,
            "teacher_manifest_sha256": str(formal_plan_audit["manifest_sha256"]),
            "conditional_artifact_sha256": conditional_artifact_sha256,
            "git_sha": current_git_sha,
            "validation_implementation_schema": M5_STREAMING_VALIDATION_SCHEMA,
            "resume_report": None if resume_report is None else "resume_report.json",
            **train_m3.summarize_m5_step_timing_records(step_timing_records),
        }
        _assert_no_tensors(summary, "smoke_summary")
        train_m3.write_run_json(
            audit_report,
            args.output_dir / "smoke_audit_report.json",
            strict=True,
        )
        train_m3.write_run_json(
            summary,
            args.output_dir / "smoke_summary.json",
            strict=True,
        )
        print(json.dumps(summary, indent=2), flush=True)
        return summary
    finally:
        try:
            if generator is not None:
                generator.to("cpu")
        finally:
            restored_probe_forward = None
            fixed_state = None
            parent_resume_payload = None
            gc.collect()


def build_m5_smoke_resolved_config(
    *,
    config: Any,
    args: argparse.Namespace,
    device: torch.device,
    contract: SmokeContract,
    sample_plan: Mapping[str, Any],
    sample_plan_sha256: str,
    formal_plan_audit: Mapping[str, Any],
    conditionals_manifest_path: Path,
    conditionals_manifest: Mapping[str, Any],
    conditional_artifact_sha256: str,
    optimizer_config: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "model_config": train_m3.resolved_config_dict(config),
        "m3": {
            "mode": args.mode,
            "manifest": str(args.manifest.resolve()),
            "dataset_root": str(args.dataset_root.resolve()),
            "sample_index": None,
            "sample_id": None,
            "split": None,
            "split_index": None,
            "train_seed": int(args.train_seed),
            "probe_seed": int(args.probe_seed),
            "optimizer_steps": int(contract.target_global_step),
            "timing_warmup_steps": int(args.timing_warmup_steps),
            "log_interval": int(args.log_interval),
            "checkpoint_interval": int(args.checkpoint_interval),
            "backbone_lr": float(args.backbone_lr),
            "patch_embedding_lr": float(args.patch_embedding_lr),
            "mcp_lr": float(args.mcp_lr),
            "weight_decay": float(args.weight_decay),
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
            "validation_steps": list(contract.validation_steps),
            "checkpoint_steps": list(contract.checkpoint_steps),
            "fixed_decode_validation_identity": str(
                sample_plan["fixed_decode_validation_identity"]
            ),
            "sample_ordering_rule": str(sample_plan["ordering_rule"]),
            "ordering_rule": str(sample_plan["ordering_rule"]),
        },
        "m5_formal_smoke": {
            "schema": M5_FORMAL_SMOKE_SCHEMA,
            "enabled": True,
            "smoke_enabled": True,
            "formal_enabled": False,
            "run_kind": M5_FORMAL_SMOKE_RUN_KIND,
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


def m5_smoke_checkpoint_metadata(
    *,
    contract: SmokeContract,
    sample_plan_sha256: str,
    teacher_manifest_sha256: str,
    conditional_artifact_sha256: str,
    current_git_sha: str,
    parent_checkpoint_sha256: str | None,
) -> dict[str, Any]:
    return {
        "schema": M5_FORMAL_SMOKE_SCHEMA,
        "status": "PASS",
        "smoke_enabled": True,
        "formal_enabled": False,
        "run_kind": M5_FORMAL_SMOKE_RUN_KIND,
        "target_global_step": int(contract.target_global_step),
        "parent_global_step": contract.parent_global_step,
        "sample_plan_sha256": str(sample_plan_sha256),
        "teacher_manifest_sha256": str(teacher_manifest_sha256),
        "conditional_artifact_sha256": str(conditional_artifact_sha256),
        "git_sha": str(current_git_sha),
        "validation_implementation_schema": M5_STREAMING_VALIDATION_SCHEMA,
        "parent_checkpoint_sha256": parent_checkpoint_sha256,
    }


def require_m5_smoke_parent_checkpoint(
    *,
    parent_payload: Mapping[str, Any],
    sample_plan_sha256: str,
    teacher_manifest_sha256: str,
    conditional_artifact_sha256: str,
    current_git_sha: str,
) -> dict[str, Any]:
    actual_step = _require_python_int(
        parent_payload.get("global_step"),
        "parent_payload.global_step",
    )
    if actual_step != M5_FORMAL_SMOKE_RESUME_PARENT:
        raise RuntimeError(
            "M5 formal smoke parent global_step mismatch: "
            f"expected=2, actual={actual_step}"
        )
    metadata = parent_payload.get("m5_formal_smoke")
    if not isinstance(metadata, Mapping):
        raise TypeError("M5 formal smoke parent missing smoke metadata")
    if metadata.get("schema") != M5_FORMAL_SMOKE_SCHEMA:
        raise RuntimeError("M5 formal smoke parent schema mismatch")
    if metadata.get("status") != "PASS":
        raise RuntimeError(
            "M5 formal smoke parent status mismatch: "
            f"expected=PASS, actual={metadata.get('status')}"
        )
    if metadata.get("smoke_enabled") is not True:
        raise RuntimeError("M5 formal smoke parent marker is not enabled")
    if metadata.get("formal_enabled") is not False:
        raise RuntimeError("M5 formal smoke parent formal marker is mixed")
    if metadata.get("run_kind") != M5_FORMAL_SMOKE_RUN_KIND:
        raise RuntimeError("M5 formal smoke parent run_kind mismatch")
    if metadata.get("target_global_step") != M5_FORMAL_SMOKE_FRESH_TARGET:
        raise RuntimeError("M5 formal smoke parent target_global_step mismatch")
    if metadata.get("parent_global_step") is not None:
        raise RuntimeError("M5 formal smoke parent parent_global_step mismatch")
    checks = {
        "sample_plan_sha256": sample_plan_sha256,
        "teacher_manifest_sha256": teacher_manifest_sha256,
        "conditional_artifact_sha256": conditional_artifact_sha256,
        "git_sha": current_git_sha,
        "validation_implementation_schema": M5_STREAMING_VALIDATION_SCHEMA,
    }
    for field, expected in checks.items():
        actual = metadata.get(field)
        if actual != expected:
            raise RuntimeError(
                "M5 formal smoke parent provenance mismatch: "
                f"field={field}, expected={expected}, actual={actual}"
            )
    resolved_config = parent_payload.get("resolved_config")
    if not isinstance(resolved_config, Mapping):
        raise TypeError("M5 formal smoke parent missing resolved_config")
    smoke_config = resolved_config.get("m5_formal_smoke")
    if not isinstance(smoke_config, Mapping):
        raise TypeError("M5 formal smoke parent resolved_config missing smoke block")
    config_checks = {
        "schema": M5_FORMAL_SMOKE_SCHEMA,
        "enabled": True,
        "smoke_enabled": True,
        "formal_enabled": False,
        "run_kind": M5_FORMAL_SMOKE_RUN_KIND,
        "sample_plan_sha256": sample_plan_sha256,
        "teacher_manifest_sha256": teacher_manifest_sha256,
        "conditional_artifact_sha256": conditional_artifact_sha256,
        "validation_implementation_schema": M5_STREAMING_VALIDATION_SCHEMA,
    }
    for field, expected in config_checks.items():
        actual = smoke_config.get(field)
        if actual != expected:
            raise RuntimeError(
                "M5 formal smoke parent resolved_config provenance mismatch: "
                f"field=m5_formal_smoke.{field}, expected={expected}, actual={actual}"
            )
    return dict(metadata)


def _run_smoke_validation(
    *,
    audit: SmokeLifecycleAudit,
    step: int,
    device: torch.device,
    generator: Any,
    teacher_store: Any,
    conditional_store: Any,
    scheduler_main: Any,
    scheduler_mcp: Any,
    dtype: torch.dtype,
    mode: str,
    sample_plan: Mapping[str, Any],
    validation_seed: int,
    train_rng: torch.Generator,
    probe: Any,
    output_dir: Path,
    metrics_path: Path,
    current_git_sha: str,
    reference_checkpoint_sha256: str,
    validation_reports: list[dict[str, Any]],
) -> dict[str, Any]:
    with audit.scoped("validation", int(step)):
        audit.record_validation_memory(step=step, moment="before", device=device)
        report = train_m3.run_m5_formal_validation_for_step(
            generator=generator,
            teacher_store=teacher_store,
            conditional_store=conditional_store,
            scheduler_main=scheduler_main,
            scheduler_mcp=scheduler_mcp,
            device=device,
            dtype=dtype,
            mode=mode,
            global_step=int(step),
            sample_plan=sample_plan,
            validation_seed=int(validation_seed),
            train_rng=train_rng,
            probe=probe,
            output_dir=output_dir,
            current_git_sha=current_git_sha,
            reference_checkpoint_sha256=reference_checkpoint_sha256,
        )
        audit.record_validation_memory(step=step, moment="after", device=device)
        train_m3.handle_m5_formal_validation_report(
            output_dir=output_dir,
            metrics_path=metrics_path,
            validation_reports=validation_reports,
            report=report,
            global_step=int(step),
        )
        return report


def _write_smoke_probe_checkpoint_node(
    *,
    audit: SmokeLifecycleAudit,
    output_dir: Path,
    metrics_path: Path,
    generator: Any,
    optimizer: torch.optim.Optimizer,
    global_step: int,
    train_rng: torch.Generator,
    probe: Any,
    fixed_sample_metadata: Mapping[str, Any],
    resolved_config: dict[str, Any],
    current_git_sha: str,
    reference_checkpoint_path: Path,
    reference_checkpoint_sha256: str,
    train_seed: int,
    probe_seed: int,
    fixed_prompt_embedding: dict[str, Any],
    device: torch.device,
    checkpoint_metadata: Mapping[str, Any],
) -> Path:
    with audit.scoped("checkpoint", int(global_step)):
        audit.record_checkpoint_memory(
            step=global_step,
            moment="before",
            device=device,
        )
        probe_forward = train_m3.run_m3_probe_forward(
            generator,
            conditional_dict=fixed_prompt_embedding,
            noisy_batch=probe.noisy_batch,
        )
        probe_report = train_m3.write_probe_report(
            output_dir,
            global_step,
            probe_forward.losses,
            probe_forward.outputs,
            None,
            strict=True,
        )
        checkpoint_path = train_m3.save_checkpoint_at_step(
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
            extra_payload_fields={"m5_formal_smoke": dict(checkpoint_metadata)},
        )
        _require_checkpoint_written(checkpoint_path)
        train_m3.append_strict_metrics(
            metrics_path,
            {
                "event": "smoke_probe_checkpoint",
                "step": int(global_step),
                **train_m3.prefix_metrics("probe", probe_forward.losses),
                "checkpoint_path": str(checkpoint_path.resolve()),
            },
        )
        audit.record_checkpoint_memory(
            step=global_step,
            moment="after",
            device=device,
        )
        return checkpoint_path


def _write_smoke_step_artifacts(
    *,
    audit: SmokeLifecycleAudit,
    output_dir: Path,
    metrics_path: Path,
    generator: Any,
    optimizer: torch.optim.Optimizer,
    global_step: int,
    metric_record: dict[str, Any],
    train_rng: torch.Generator,
    probe: Any,
    sample_metadata: Mapping[str, Any],
    resolved_config: dict[str, Any],
    current_git_sha: str,
    reference_checkpoint_path: Path,
    reference_checkpoint_sha256: str,
    train_seed: int,
    probe_seed: int,
    prompt_embedding: dict[str, Any],
    device: torch.device,
    train_losses: Mapping[str, float],
    aux_report: Mapping[str, Any],
    grad_audit: Mapping[str, Any],
    checkpoint_metadata: Mapping[str, Any],
) -> Path:
    checkpoint_path = output_dir / f"checkpoint_step{int(global_step):06d}.pt"
    with audit.scoped("checkpoint", int(global_step)):
        audit.record_checkpoint_memory(
            step=global_step,
            moment="before",
            device=device,
        )
        train_m3.write_m5_step_artifacts(
            output_dir=output_dir,
            metrics_path=metrics_path,
            generator=generator,
            optimizer=optimizer,
            global_step=global_step,
            metric_record=metric_record,
            should_log=True,
            should_checkpoint=True,
            train_rng=train_rng,
            probe=probe,
            sample_metadata=sample_metadata,
            resolved_config=resolved_config,
            current_git_sha=current_git_sha,
            reference_checkpoint_path=reference_checkpoint_path,
            reference_checkpoint_sha256=reference_checkpoint_sha256,
            train_seed=train_seed,
            probe_seed=probe_seed,
            prompt_embedding=prompt_embedding,
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
                "m5_formal_smoke": dict(checkpoint_metadata)
            },
        )
        _require_checkpoint_written(checkpoint_path)
        audit.record_checkpoint_memory(
            step=global_step,
            moment="after",
            device=device,
        )
    return checkpoint_path


@contextmanager
def _audit_train_m3_device_transfers(audit: SmokeLifecycleAudit) -> Iterator[None]:
    original_state = train_m3.selected_state_to_device
    original_conditional = train_m3.m5_formal_conditional_to_device
    original_forward = train_m3.run_nf_sf_forward_loss

    def selected_state_wrapper(*args: Any, **kwargs: Any) -> Any:
        value = original_state(*args, **kwargs)
        audit.record_state(value)
        return value

    def conditional_wrapper(*args: Any, **kwargs: Any) -> Any:
        value = original_conditional(*args, **kwargs)
        if isinstance(value, Mapping):
            audit.record_device_conditional(value)
        return value

    def forward_wrapper(*args: Any, **kwargs: Any) -> Any:
        if (
            audit.scope["phase"] == "train"
            and not any(marker["name"] == "first_resumed_forward" for marker in audit.markers)
            and any(marker["name"] == "global_rng_restored" for marker in audit.markers)
        ):
            audit.mark("first_resumed_forward")
        return original_forward(*args, **kwargs)

    train_m3.selected_state_to_device = selected_state_wrapper
    train_m3.m5_formal_conditional_to_device = conditional_wrapper
    train_m3.run_nf_sf_forward_loss = forward_wrapper
    try:
        yield
    finally:
        train_m3.selected_state_to_device = original_state
        train_m3.m5_formal_conditional_to_device = original_conditional
        train_m3.run_nf_sf_forward_loss = original_forward


def _require_smoke_pass_contract(report: Mapping[str, Any]) -> None:
    teacher = report["teacher"]
    conditional = report["conditional"]
    if teacher["max_live_count"] > 1:
        raise RuntimeError("teacher max live count exceeded 1")
    if conditional["max_live_count"] > 1:
        raise RuntimeError("conditional max live count exceeded 1")
    if teacher["end_live_count"] != 0:
        raise RuntimeError("teacher final live count is non-zero")
    if conditional["end_live_count"] != 0:
        raise RuntimeError("conditional final live count is non-zero")
    if teacher["acquire_count"] != teacher["release_count"]:
        raise RuntimeError("teacher acquire/release count mismatch")
    if conditional["acquire_count"] != conditional["release_count"]:
        raise RuntimeError("conditional acquire/release count mismatch")
    _require_acquire_release_pairs(report["store_events"])
    for step, counts in report["train_step_store_counts"].items():
        if counts["teacher_acquire"] != 1 or counts["teacher_release"] != 1:
            raise RuntimeError(f"teacher train-step acquire/release mismatch: {step}")
        if counts["conditional_acquire"] != 1 or counts["conditional_release"] != 1:
            raise RuntimeError(
                f"conditional train-step acquire/release mismatch: {step}"
            )
    for validation in report["validation_reports"]:
        if validation["status"] != "PASS":
            raise RuntimeError("smoke validation did not PASS")
        if validation["validation_loss_finite_contract"] is not True:
            raise RuntimeError("smoke validation loss finite contract failed")
    for record in report["train_loss_records"]:
        losses = record["losses"]
        for key, value in losses.items():
            if not math.isfinite(float(value)):
                raise RuntimeError(f"non-finite smoke train loss: {key}")
    if report["parent_global_step"] is None:
        if report["executed_global_steps"] != [1, 2]:
            raise RuntimeError("fresh smoke executed step range mismatch")
        if report["validation_steps"] != [0, 2]:
            raise RuntimeError("fresh smoke validation steps mismatch")
        if report["checkpoint_steps"] != [0, 2]:
            raise RuntimeError("fresh smoke checkpoint steps mismatch")
    else:
        if report["executed_global_steps"] != [3]:
            raise RuntimeError("resume smoke executed step range mismatch")
        if report["validation_steps"] != [3]:
            raise RuntimeError("resume smoke validation steps mismatch")
        if report["checkpoint_steps"] != [3]:
            raise RuntimeError("resume smoke checkpoint steps mismatch")
        marker_names = [marker["name"] for marker in report["markers"]]
        parent_release = marker_names.index("parent_payload_released")
        rng_restore = marker_names.index("global_rng_restored")
        first_forward = marker_names.index("first_resumed_forward")
        if not parent_release < rng_restore < first_forward:
            raise RuntimeError("resume release/RNG/forward marker order mismatch")
    for record in report["checkpoint_records"]:
        if record["write_success"] is not True:
            raise RuntimeError("smoke checkpoint write failed")


def _require_acquire_release_pairs(events: Sequence[Mapping[str, Any]]) -> None:
    stacks: dict[str, list[str]] = {"teacher": [], "conditional": []}
    for event in events:
        kind = str(event["kind"])
        identity = str(event["identity"])
        action = str(event["action"])
        if action == "acquire":
            stacks[kind].append(identity)
            continue
        if action != "release":
            raise RuntimeError(f"unknown store action: {action}")
        if not stacks[kind]:
            raise RuntimeError(f"{kind} release without acquire: {identity}")
        expected = stacks[kind].pop()
        if expected != identity:
            raise RuntimeError(
                f"{kind} acquire/release order mismatch: "
                f"expected={expected}, actual={identity}"
            )
    for kind, stack in stacks.items():
        if stack:
            raise RuntimeError(f"{kind} unreleased identities: {stack}")


def _require_validation_pass(report: Mapping[str, Any], *, step: int) -> None:
    if report.get("status") != "PASS":
        raise RuntimeError(f"M5 formal smoke validation failed at step {step}")
    if report.get("validation_loss_finite_contract") is not True:
        raise RuntimeError(
            f"M5 formal smoke validation loss contract failed at step {step}"
        )


def _require_checkpoint_written(path: Path) -> None:
    if not path.is_file():
        raise RuntimeError(f"smoke checkpoint was not written: {path}")


def _checkpoint_record(
    step: int,
    path: Path,
    checkpoint_metadata_base: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "global_step": int(step),
        "path": str(path.resolve()),
        "write_success": path.is_file(),
        "schema": checkpoint_metadata_base["schema"],
        "smoke_enabled": True,
        "formal_enabled": False,
    }


def _validation_report_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "global_step": int(report["global_step"]),
        "schema": str(report["schema"]),
        "status": str(report["status"]),
        "sample_plan_sha256": str(report["sample_plan_sha256"]),
        "conditional_artifact_sha256": str(report["conditional_artifact_sha256"]),
        "validation_loss_finite_contract": bool(
            report["validation_loss_finite_contract"]
        ),
        "sample_count": int(report["sample_count"]),
        "aggregate_losses": _json_safe(
            dict(report["aggregate_losses"]),
            "validation.aggregate_losses",
        ),
    }


def _require_store_provenance(
    *,
    teacher_store: Any,
    conditional_store: Any,
    sample_plan_sha256: str,
    teacher_manifest_sha256: str,
) -> None:
    if teacher_store.sample_plan_sha256 != sample_plan_sha256:
        raise RuntimeError("M5 formal smoke teacher store sample plan SHA mismatch")
    if conditional_store.sample_plan_sha256 != sample_plan_sha256:
        raise RuntimeError("M5 formal smoke conditional store sample plan SHA mismatch")
    if teacher_store.manifest_sha256 != teacher_manifest_sha256:
        raise RuntimeError("M5 formal smoke teacher store manifest SHA mismatch")
    if conditional_store.teacher_manifest_sha256 != teacher_manifest_sha256:
        raise RuntimeError("M5 formal smoke conditional store manifest SHA mismatch")


def _require_output_dir_independent_empty(
    output_dir: Path,
    *,
    resume_checkpoint: Path | None,
) -> None:
    path = Path(output_dir)
    if str(path).strip() == "":
        raise ValueError("--output_dir must be non-empty")
    if path.name.lower().endswith(".tmp"):
        raise ValueError("--output_dir must not end with .tmp")
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"--output_dir must be empty: {path}")
    if not path.parent.exists():
        raise FileNotFoundError(f"--output_dir parent does not exist: {path.parent}")
    if not path.parent.is_dir():
        raise ValueError(f"--output_dir parent must be a directory: {path.parent}")
    if resume_checkpoint is not None:
        output_resolved = path.resolve()
        parent_checkpoint = Path(resume_checkpoint).resolve()
        if output_resolved in (parent_checkpoint.parent, *parent_checkpoint.parents):
            raise ValueError("--output_dir must be independent from parent checkpoint")


def _require_existing_file(value: Path | str, name: str) -> Path:
    path = Path(value)
    if path.name.lower().endswith(".tmp"):
        raise ValueError(f"{name} must not end with .tmp")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _require_existing_directory(value: Path | str, name: str) -> Path:
    path = Path(value)
    if not path.is_dir():
        raise FileNotFoundError(path)
    return path


def _require_python_int(value: Any, field_path: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{field_path} must be a Python int")
    return value


def cuda_memory_snapshot(device: torch.device) -> dict[str, int | None]:
    if device.type != "cuda" or not torch.cuda.is_available():
        return {
            "memory_allocated": None,
            "memory_reserved": None,
            "max_memory_allocated": None,
            "max_memory_reserved": None,
        }
    return {
        "memory_allocated": int(torch.cuda.memory_allocated(device)),
        "memory_reserved": int(torch.cuda.memory_reserved(device)),
        "max_memory_allocated": int(torch.cuda.max_memory_allocated(device)),
        "max_memory_reserved": int(torch.cuda.max_memory_reserved(device)),
    }


def _selected_state_metadata(value: Any) -> dict[str, Any]:
    future_valid_masks = value.future_valid_masks
    return {
        "clean_history": _optional_tensor_metadata(
            getattr(value, "clean_history", None)
        ),
        "current_target": _tensor_metadata(value.current_target),
        "future_targets": [
            _tensor_metadata(tensor)
            for tensor in value.future_targets
        ],
        "future_valid_masks": None
        if future_valid_masks is None
        else [
            _tensor_metadata(tensor)
            for tensor in future_valid_masks
        ],
        "current_start_frame": getattr(value, "current_start_frame", None),
    }


def _tensor_mapping_metadata(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(key): _tensor_metadata(tensor)
        for key, tensor in sorted(value.items(), key=lambda item: str(item[0]))
        if isinstance(tensor, torch.Tensor)
    }


def _optional_tensor_metadata(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    return _tensor_metadata(value)


def _tensor_metadata(tensor: Any) -> dict[str, Any]:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"expected torch.Tensor metadata target, got {type(tensor).__name__}")
    return {
        "shape": [int(dim) for dim in tensor.shape],
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
    }


def _json_safe(value: Any, field_path: str) -> Any:
    if isinstance(value, torch.Tensor):
        raise TypeError(f"{field_path} must not contain torch.Tensor")
    if isinstance(value, Path):
        return str(value.resolve())
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise RuntimeError(f"{field_path} must be finite")
        return float(value)
    if isinstance(value, Mapping):
        return {
            str(key): _json_safe(item, f"{field_path}.{key}")
            for key, item in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [
            _json_safe(item, f"{field_path}[{index}]")
            for index, item in enumerate(value)
        ]
    raise TypeError(f"{field_path} is not JSON safe: {type(value).__name__}")


def _assert_no_tensors(value: Any, field_path: str) -> None:
    if isinstance(value, torch.Tensor):
        raise TypeError(f"{field_path} must not contain torch.Tensor")
    if isinstance(value, Mapping):
        for key, item in value.items():
            _assert_no_tensors(item, f"{field_path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            _assert_no_tensors(item, f"{field_path}[{index}]")


def main() -> None:
    args = parse_args()
    args.mode = train_m3.validate_m3_mode(args.mode)
    dtype = train_m3.dtype_from_arg(args.dtype)
    device = require_single_gpu_runtime(torch, args.device)
    _require_output_dir_independent_empty(
        args.output_dir,
        resume_checkpoint=args.resume_checkpoint,
    )
    config = merge_config(str(args.config))
    current_git_sha = train_m3.git_head()
    reference_checkpoint_sha256 = train_m3.file_sha256(args.checkpoint)
    run_m5_formal_smoke(
        args=args,
        config=config,
        dtype=dtype,
        device=device,
        current_git_sha=current_git_sha,
        reference_checkpoint_sha256=reference_checkpoint_sha256,
    )


if __name__ == "__main__":
    main()
