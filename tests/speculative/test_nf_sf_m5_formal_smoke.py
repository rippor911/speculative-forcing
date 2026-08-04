from __future__ import annotations

import argparse
import json
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn

import scripts.run_nf_sf_m5_formal_smoke as smoke
import scripts.train_nf_sf_m3_overfit as train_m3
import utils.nf_sf_m4 as m4
import utils.nf_sf_m5_validation as m5_validation
from utils.nf_sf_m3 import M3_PARAMETER_GROUP_NAMES
from utils.nf_sf_training import NFSFNoisyBatch, NFSFSelectedState

MANIFEST_SHA = "2" * 64
ARTIFACT_SHA = "3" * 64
REFERENCE_SHA = "4" * 64
GIT_SHA = "0123456789abcdef0123456789abcdef01234567"
ENCODER_SHA = "5" * 64


class FakeTensorOnAnyDevice:
    def to(self, *args: Any, **kwargs: Any) -> FakeTensorOnAnyDevice:
        return self


class TinyGenerator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Parameter(torch.tensor(1.0))
        self.patch_embedding = nn.Parameter(torch.tensor(1.0))
        self.mcp_fusion = nn.Parameter(torch.tensor(1.0))
        self.mcp_depth1 = nn.Parameter(torch.tensor(1.0))
        self.mcp_depth2 = nn.Parameter(torch.tensor(1.0))
        self.mcp_depth3 = nn.Parameter(torch.tensor(1.0))

    def to(self, *args: Any, **kwargs: Any) -> TinyGenerator:
        return self

    def get_scheduler(self) -> SimpleNamespace:
        return SimpleNamespace(
            sigmas=FakeTensorOnAnyDevice(),
            timesteps=FakeTensorOnAnyDevice(),
        )


class FakeStore:
    def __init__(
        self,
        *,
        plan: dict[str, Any],
        events: list[tuple[Any, ...]],
        kind: str,
        manifest_path: Path,
        dataset_root: Path,
    ) -> None:
        self.plan = plan
        self.events = events
        self.kind = kind
        self.manifest_path = manifest_path
        self.dataset_root = dataset_root
        self.sample_plan_sha256 = str(plan["sample_plan_sha256"])
        self.manifest_sha256 = MANIFEST_SHA
        self.teacher_manifest_sha256 = MANIFEST_SHA
        self.artifact_sha256 = ARTIFACT_SHA
        self.train_identities = tuple(plan["train_sample_identities"])
        self.validation_identities = tuple(plan["validation_sample_identities"])
        self.entries = {
            str(entry["identity"]): dict(entry)
            for split in ("train", "validation")
            for entry in plan["samples"][split]
        }
        self.fixed_decode_validation_identity = str(
            plan["fixed_decode_validation_identity"]
        )
        self.live_sample_count = 0
        self.live_conditional_count = 0
        self.max_live_sample_count = 0
        self.max_live_conditional_count = 0
        self.load_attempt_count = 0
        self.successful_load_count = 0
        self.train_identity_steps: list[int] = []

    @property
    def total_load_count(self) -> int:
        return self.successful_load_count

    def train_identity_for_step(self, step: int) -> str:
        self.train_identity_steps.append(int(step))
        return str(m4.m4_train_entry_for_step(self.plan, int(step))["identity"])

    @contextmanager
    def acquire(self, identity: str) -> Iterator[Any]:
        self.load_attempt_count += 1
        self.successful_load_count += 1
        if self.kind == "teacher":
            entry = self.entries[str(identity)]
            self.live_sample_count = 1
            self.max_live_sample_count = max(self.max_live_sample_count, 1)
            payload = SimpleNamespace(
                metadata=_sample_metadata(
                    identity=identity,
                    split=str(entry["split"]),
                    manifest_path=self.manifest_path,
                    dataset_root=self.dataset_root,
                ),
                selected_state=_selected_state(),
            )
        else:
            self.live_conditional_count = 1
            self.max_live_conditional_count = max(self.max_live_conditional_count, 1)
            payload = {"prompt_embeds": torch.ones(1, 1)}
        self.events.append((f"{self.kind}_acquire_raw", identity))
        try:
            yield payload
        finally:
            if self.kind == "teacher":
                self.live_sample_count = 0
            else:
                self.live_conditional_count = 0
            self.events.append((f"{self.kind}_release_raw", identity))


class Runtime:
    def __init__(self) -> None:
        self.events: list[tuple[Any, ...]] = []
        self.teacher_stores: list[FakeStore] = []
        self.conditional_stores: list[FakeStore] = []
        self.validation_status = "PASS"
        self.validation_finite = True
        self.nonfinite_train_loss: float | None = None
        self.forward_exception: Exception | None = None
        self.validation_seen_steps: set[int] = set()


def _selected_state() -> NFSFSelectedState:
    tensor = torch.ones(1, 3, 1, 1, 1)
    return NFSFSelectedState(
        clean_history=tensor,
        current_target=tensor + 1,
        future_targets=(tensor + 2, tensor + 3, tensor + 4),
        current_start_frame=3,
    )


def _noisy_batch(state: NFSFSelectedState | None = None) -> NFSFNoisyBatch:
    state = _selected_state() if state is None else state
    tensor = torch.ones(1)
    return NFSFNoisyBatch(
        state=state,
        noisy_current=tensor,
        noisy_futures=(tensor, tensor, tensor),
        timestep_main=tensor,
        timestep_depths=(tensor, tensor, tensor),
        epsilon_main=tensor,
        epsilon_depths=(tensor, tensor, tensor),
        target_flow_main=tensor,
        target_flow_depths=(tensor, tensor, tensor),
        future_valid_masks=(torch.ones(1, dtype=torch.bool),) * 3,
        future_start_frames=(6, 9, 12),
    )


def _sample_metadata(
    *,
    identity: str,
    split: str,
    manifest_path: Path,
    dataset_root: Path,
) -> dict[str, Any]:
    return {
        "identity": identity,
        "prompt": f"prompt {identity}",
        "sample_index": 0,
        "sample_id": identity,
        "split": split,
        "split_index": 0,
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": MANIFEST_SHA,
        "dataset_root": str(dataset_root.resolve()),
        "chunk_frames": 3,
    }


def _plan(manifest_path: Path) -> dict[str, Any]:
    train_entries = []
    for index in range(2048):
        entry = {
            "sample_index": index,
            "sample_id": f"train-{index:04d}",
            "split": "train",
            "split_index": index,
            "prompt_sha256": "a" * 64,
            "file_sha256": "c" * 64,
            "file": f"train-{index:04d}.pt",
        }
        entry["identity"] = m4.m4_sample_identity_from_record(entry)
        train_entries.append(entry)
    validation_entries = []
    for index in range(256):
        entry = {
            "sample_index": 100_000 + index,
            "sample_id": f"validation-{index:04d}",
            "split": "validation",
            "split_index": index,
            "prompt_sha256": "b" * 64,
            "file_sha256": "d" * 64,
            "file": f"validation-{index:04d}.pt",
        }
        entry["identity"] = m4.m4_sample_identity_from_record(entry)
        validation_entries.append(entry)
    plan = {
        "schema": "nf_sf_m4_sample_plan_v1",
        "status": "PASS",
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": MANIFEST_SHA,
        "dataset_root": str(manifest_path.parent.resolve()),
        "ordering_rule": m4.M4_SAMPLE_ORDERING_RULE,
        "train_subset_size": len(train_entries),
        "validation_subset_size": len(validation_entries),
        "train_sample_identities": [entry["identity"] for entry in train_entries],
        "validation_sample_identities": [
            entry["identity"] for entry in validation_entries
        ],
        "fixed_decode_validation_identity": validation_entries[0]["identity"],
        "samples": {"train": train_entries, "validation": validation_entries},
    }
    plan["sample_plan_sha256"] = m4.m4_sample_plan_sha256(plan)
    return plan


def _args(
    tmp_path: Path,
    *,
    target: int = 2,
    parent: int | None = None,
    resume_checkpoint: Path | None = None,
    output_name: str = "out",
    mode: str = "joint",
    device: str = "cuda:0",
    mcp1_grid_aux_weight: float = 0.0,
) -> argparse.Namespace:
    tmp_path.mkdir(parents=True, exist_ok=True)
    checkpoint = tmp_path / "reference.pt"
    checkpoint.write_bytes(b"reference")
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir(exist_ok=True)
    sample_plan = tmp_path / "sample_plan.json"
    sample_plan.write_text("{}", encoding="utf-8")
    artifact_dir = tmp_path / "conditionals"
    artifact_dir.mkdir(exist_ok=True)
    artifact_manifest = artifact_dir / "manifest.json"
    artifact_manifest.write_text("{}", encoding="utf-8")
    config = tmp_path / "config.yaml"
    config.write_text("{}", encoding="utf-8")
    return argparse.Namespace(
        config=config,
        checkpoint=checkpoint,
        resume_checkpoint=resume_checkpoint,
        manifest=manifest,
        dataset_root=dataset_root,
        output_dir=tmp_path / output_name,
        mode=mode,
        train_seed=123,
        probe_seed=456,
        target_global_step=target,
        parent_global_step=parent,
        timing_warmup_steps=0,
        log_interval=10_000,
        checkpoint_interval=10_000,
        backbone_lr=1.0e-4,
        patch_embedding_lr=2.0e-4,
        mcp_lr=3.0e-4,
        weight_decay=0.01,
        mcp1_grid_aux_weight=mcp1_grid_aux_weight,
        m4_sample_plan=sample_plan,
        m5_conditionals_artifact=artifact_manifest.resolve(),
        validation_seed=789,
        dtype="float32",
        device=device,
    )


def _config() -> SimpleNamespace:
    return SimpleNamespace(
        i2v=False,
        num_frame_per_block=3,
        mcp_num_modules=3,
        mcp_num_layers=3,
        mcp_tap_layers=[3, 11, 19, 29],
        mcp_depth_weights=[0.5, 0.2, 0.1],
        model_kwargs={"timestep_shift": 5.0},
    )


def _install_runtime(
    monkeypatch: pytest.MonkeyPatch,
    args: argparse.Namespace,
    runtime: Runtime,
) -> dict[str, Any]:
    plan = _plan(args.manifest)
    monkeypatch.setattr(smoke.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(smoke.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(smoke.torch.cuda, "manual_seed_all", lambda seed: None)
    monkeypatch.setattr(smoke.torch.cuda, "get_rng_state_all", _cuda_rng_state)
    monkeypatch.setattr(smoke.torch.cuda, "set_rng_state_all", lambda states: None)
    monkeypatch.setattr(smoke.torch.cuda, "synchronize", lambda device=None: None)
    monkeypatch.setattr(smoke.torch.cuda, "memory_allocated", lambda device=None: 11)
    monkeypatch.setattr(smoke.torch.cuda, "memory_reserved", lambda device=None: 22)
    monkeypatch.setattr(
        smoke.torch.cuda,
        "max_memory_allocated",
        lambda device=None: 33,
    )
    monkeypatch.setattr(
        smoke.torch.cuda,
        "max_memory_reserved",
        lambda device=None: 44,
    )
    monkeypatch.setattr(
        smoke.torch.cuda,
        "reset_peak_memory_stats",
        lambda device=None: None,
    )
    monkeypatch.setattr(
        smoke.torch.cuda,
        "is_current_stream_capturing",
        lambda: False,
    )
    monkeypatch.setattr(
        train_m3,
        "resolved_config_dict",
        lambda config: {
            "i2v": bool(config.i2v),
            "num_frame_per_block": int(config.num_frame_per_block),
            "mcp_num_modules": int(config.mcp_num_modules),
            "mcp_num_layers": int(config.mcp_num_layers),
            "mcp_tap_layers": list(config.mcp_tap_layers),
            "mcp_depth_weights": list(config.mcp_depth_weights),
            "model_kwargs": dict(config.model_kwargs),
        },
    )
    monkeypatch.setattr(train_m3, "load_m4_sample_plan", lambda *a, **k: plan)
    monkeypatch.setattr(
        train_m3,
        "validate_m5_formal_sample_plan",
        lambda *a, **k: {
            "status": "PASS",
            "sample_plan_sha256": plan["sample_plan_sha256"],
            "manifest_sha256": MANIFEST_SHA,
            "train_sample_count": 2048,
            "validation_sample_count": 256,
        },
    )
    monkeypatch.setattr(
        train_m3,
        "load_m5_conditional_artifact_manifest",
        lambda *a, **k: {
            "schema": "nf_sf_m5_conditional_artifact_v1",
            "encoder_provenance": {
                "encoder_class": "Tiny",
                "model_checkpoint_path": "model.pt",
                "model_checkpoint_sha256": ENCODER_SHA,
                "tokenizer_path": "tokenizer",
                "dtype": "torch.float32",
            },
        },
    )
    monkeypatch.setattr(
        train_m3,
        "validate_m5_conditional_artifact_manifest",
        lambda *a, **k: {"status": "PASS", "artifact_sha256": ARTIFACT_SHA},
    )

    def teacher_store_factory(*args_: Any, **kwargs: Any) -> FakeStore:
        store = FakeStore(
            plan=plan,
            events=runtime.events,
            kind="teacher",
            manifest_path=Path(kwargs["manifest_path"]),
            dataset_root=Path(kwargs["dataset_root"]),
        )
        runtime.teacher_stores.append(store)
        return store

    def conditional_store_factory(*args_: Any, **kwargs: Any) -> FakeStore:
        store = FakeStore(
            plan=plan,
            events=runtime.events,
            kind="conditional",
            manifest_path=args.manifest,
            dataset_root=args.dataset_root,
        )
        runtime.conditional_stores.append(store)
        return store

    monkeypatch.setattr(train_m3, "M5TeacherSampleStore", teacher_store_factory)
    monkeypatch.setattr(
        train_m3,
        "M5ConditionalArtifactStore",
        conditional_store_factory,
    )
    monkeypatch.setattr(
        train_m3,
        "load_m4_teacher_samples",
        lambda *a, **k: pytest.fail("eager teacher loader was called"),
    )

    def reset_seed(seed: int) -> None:
        runtime.events.append(("reset_seed", int(seed)))

    def load_generator(*args_: Any, **kwargs: Any) -> tuple[TinyGenerator, str, int]:
        runtime.events.append(("load_generator",))
        return TinyGenerator(), "FAKE", 4

    monkeypatch.setattr(train_m3, "reset_global_seed", reset_seed)
    monkeypatch.setattr(train_m3, "load_generator", load_generator)
    monkeypatch.setattr(
        train_m3,
        "make_mcp_scheduler",
        lambda device: runtime.events.append(("make_mcp_scheduler",))
        or TinyGenerator().get_scheduler(),
    )
    monkeypatch.setattr(
        train_m3,
        "selected_state_to_device",
        lambda state, **k: state,
    )
    monkeypatch.setattr(
        train_m3,
        "m5_formal_conditional_to_device",
        lambda conditional, **k: dict(conditional),
    )
    monkeypatch.setattr(
        train_m3,
        "make_generator",
        lambda *a, **k: runtime.events.append(("make_train_rng",))
        or torch.Generator(device="cpu").manual_seed(123),
    )
    monkeypatch.setattr(
        train_m3,
        "make_m3_probe",
        lambda state, **k: runtime.events.append(("make_probe",))
        or train_m3.M3Probe(
            seed=int(k["seed"]),
            rng_state=torch.Generator(device="cpu").manual_seed(1).get_state(),
            noisy_batch=_noisy_batch(state),
        ),
    )
    monkeypatch.setattr(
        train_m3,
        "configure_m3_optimizer_plan",
        lambda generator, **k: _optimizer_plan(generator),
    )
    monkeypatch.setattr(train_m3, "named_parameter_groups", _named_groups)
    monkeypatch.setattr(
        train_m3,
        "gradient_group_audit",
        lambda *a, **k: {"optimizer_contract": {"all_contract_pass": True}},
    )
    monkeypatch.setattr(train_m3, "has_nonfinite_grad", lambda generator: False)
    monkeypatch.setattr(
        train_m3,
        "prepare_nf_sf_noisy_batch",
        lambda state, **k: _noisy_batch(state),
    )
    monkeypatch.setattr(train_m3, "run_nf_sf_forward_loss", _forward(runtime))
    monkeypatch.setattr(train_m3, "loss_dict_to_floats", _loss_dict(runtime))
    monkeypatch.setattr(train_m3, "run_m3_probe_forward", _probe_forward)
    monkeypatch.setattr(
        train_m3,
        "restore_torch_generator_from_state",
        lambda *a, **k: torch.Generator(device="cpu").manual_seed(999),
    )
    monkeypatch.setattr(
        train_m3,
        "move_loaded_optimizer_state_to_device",
        lambda *a, **k: {"status": "moved", "device": str(k["device"])},
    )
    monkeypatch.setattr(
        train_m3,
        "restore_m5_probe_from_checkpoint",
        lambda parent_payload, selected_state, **k: (
            train_m3.M3Probe(
                seed=int(parent_payload["probe_seed"]),
                rng_state=parent_payload["probe_rng_state"].detach().cpu().clone(),
                noisy_batch=_noisy_batch(selected_state),
            ),
            parent_payload["probe_prompt_embedding"],
        ),
    )
    original_restore_global = train_m3.restore_global_rng_states

    def restore_global_rng_states(rng_states: Mapping[str, Any]) -> None:
        runtime.events.append(("restore_global_rng",))
        original_restore_global(rng_states)

    monkeypatch.setattr(train_m3, "restore_global_rng_states", restore_global_rng_states)
    original_save = train_m3.save_checkpoint_at_step

    def save_checkpoint_at_step(*args_: Any, **kwargs: Any) -> Path:
        runtime.events.append(("checkpoint", int(kwargs["step"])))
        return original_save(*args_, **kwargs)

    monkeypatch.setattr(train_m3, "save_checkpoint_at_step", save_checkpoint_at_step)
    monkeypatch.setattr(
        m5_validation,
        "_conditional_to_device",
        lambda conditional, **k: dict(conditional),
    )
    monkeypatch.setattr(m5_validation, "run_m4_validation", _m4_validation(runtime))
    return plan


def _cuda_rng_state() -> list[torch.Tensor]:
    return [torch.arange(8, dtype=torch.uint8)]


def _optimizer_plan(generator: TinyGenerator) -> SimpleNamespace:
    groups = []
    for name in M3_PARAMETER_GROUP_NAMES:
        parameter = getattr(generator, name)
        lr = 1.0e-4 if name == "backbone" else 2.0e-4
        if name.startswith("mcp_"):
            lr = 3.0e-4
        groups.append({"params": [parameter], "lr": lr, "name": name})
    return SimpleNamespace(mode="joint", optimizer_param_groups=groups, audits=[])


def _named_groups(generator: TinyGenerator) -> dict[str, tuple[tuple[str, nn.Parameter], ...]]:
    return {
        name: ((name, getattr(generator, name)),)
        for name in M3_PARAMETER_GROUP_NAMES
    }


def _forward(runtime: Runtime):
    def forward(generator: TinyGenerator, **kwargs: Any) -> SimpleNamespace:
        if runtime.forward_exception is not None:
            raise runtime.forward_exception
        runtime.events.append(("forward",))
        loss = sum(parameter.sum() * 0.0 for parameter in generator.parameters())
        loss = loss + torch.ones(())
        losses = SimpleNamespace(
            total_loss=loss,
            main_loss=loss,
            mcp_depth_losses=(loss, loss, loss),
        )
        return SimpleNamespace(losses=losses)

    return forward


def _loss_dict(runtime: Runtime):
    def loss_dict_to_floats(losses: Any) -> dict[str, float]:
        total = 1.0 if runtime.nonfinite_train_loss is None else runtime.nonfinite_train_loss
        return {
            "main_loss": 1.0,
            "mcp_depth1_loss": 1.0,
            "mcp_depth2_loss": 1.0,
            "mcp_depth3_loss": 1.0,
            "total_loss": float(total),
        }

    return loss_dict_to_floats


def _probe_forward(*args: Any, **kwargs: Any) -> SimpleNamespace:
    losses = {
        "main_loss": 1.0,
        "mcp_depth1_loss": 1.0,
        "mcp_depth2_loss": 1.0,
        "mcp_depth3_loss": 1.0,
        "total_loss": 1.0,
    }
    outputs = {
        "main_flow_pred": torch.ones(2),
        "mcp_depth1_flow_pred": torch.ones(2),
        "mcp_depth2_flow_pred": torch.ones(2),
        "mcp_depth3_flow_pred": torch.ones(2),
    }
    return SimpleNamespace(losses=losses, outputs=outputs)


def _m4_validation(runtime: Runtime):
    def run_m4_validation(**kwargs: Any) -> dict[str, Any]:
        step = int(kwargs["global_step"])
        if step not in runtime.validation_seen_steps:
            runtime.validation_seen_steps.add(step)
            runtime.events.append(("validation", step))
        identity = next(iter(kwargs["conditional_dicts"]))
        losses = {
            "main_loss": 1.0,
            "mcp_depth1_loss": 1.0,
            "mcp_depth2_loss": 1.0,
            "mcp_depth3_loss": 1.0,
            "weighted_mcp_loss": 1.0,
            "total_validation_loss": 1.0,
        }
        if not runtime.validation_finite:
            losses["total_validation_loss"] = float("nan")
        return {
            "schema": m4.M4_VALIDATION_SCHEMA,
            "status": runtime.validation_status,
            "global_step": step,
            "sample_count": 1,
            "validation_sample_identities": [identity],
            "per_sample_losses": [
                {
                    "sample_identity": identity,
                    "losses": losses,
                }
            ],
            "gradients_unchanged_contract": True,
            "requires_grad_unchanged_contract": True,
            "train_rng_unchanged_contract": True,
            "probe_rng_unchanged_contract": True,
            "global_cpu_rng_unchanged_contract": True,
            "global_cuda_rng_unchanged_contract": True,
        }

    return run_m4_validation


def _run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    runtime: Runtime | None = None,
    target: int = 2,
    parent: int | None = None,
    resume_checkpoint: Path | None = None,
    output_name: str = "out",
) -> tuple[dict[str, Any], argparse.Namespace, Runtime, dict[str, Any]]:
    runtime = Runtime() if runtime is None else runtime
    args = _args(
        tmp_path,
        target=target,
        parent=parent,
        resume_checkpoint=resume_checkpoint,
        output_name=output_name,
    )
    plan = _install_runtime(monkeypatch, args, runtime)
    summary = smoke.run_m5_formal_smoke(
        args=args,
        config=_config(),
        dtype=torch.float32,
        device=torch.device("cuda:0"),
        current_git_sha=GIT_SHA,
        reference_checkpoint_sha256=REFERENCE_SHA,
    )
    return summary, args, runtime, plan


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _assert_no_tensor(value: Any) -> None:
    assert not isinstance(value, torch.Tensor)
    if isinstance(value, Mapping):
        for item in value.values():
            _assert_no_tensor(item)
    elif isinstance(value, list):
        for item in value:
            _assert_no_tensor(item)


def test_fresh_smoke_contract_validation_checkpoint_and_lifecycle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    summary, args, runtime, _ = _run(tmp_path, monkeypatch)
    report = _load_json(args.output_dir / "smoke_audit_report.json")

    assert summary["executed_global_steps"] == [1, 2]
    assert report["validation_steps"] == [0, 2]
    assert report["checkpoint_steps"] == [0, 2]
    assert runtime.events.index(("reset_seed", 123)) < runtime.events.index(
        ("load_generator",)
    )
    assert runtime.events.index(("validation", 0)) < runtime.events.index(
        ("checkpoint", 0)
    )
    assert runtime.events.index(("validation", 2)) < runtime.events.index(
        ("checkpoint", 2)
    )
    assert report["train_step_store_counts"]["1"]["teacher_acquire"] == 1
    assert report["train_step_store_counts"]["1"]["conditional_acquire"] == 1
    assert report["train_step_store_counts"]["2"]["teacher_acquire"] == 1
    assert report["train_step_store_counts"]["2"]["conditional_acquire"] == 1
    assert report["teacher"]["max_live_count"] <= 1
    assert report["conditional"]["max_live_count"] <= 1
    assert report["teacher"]["end_live_count"] == 0
    assert report["conditional"]["end_live_count"] == 0
    assert (args.output_dir / "checkpoint_step000000.pt").is_file()
    assert (args.output_dir / "checkpoint_step000002.pt").is_file()
    assert report["cuda_memory"]["train_steps"][0]["cuda"] == {
        "memory_allocated": 11,
        "memory_reserved": 22,
        "max_memory_allocated": 33,
        "max_memory_reserved": 44,
    }
    assert report["device_conditional_records"]
    assert report["step_state_records"]
    assert not any(event[0] == "eager_teacher" for event in runtime.events)
    _assert_no_tensor(report)


def test_resume_smoke_only_step3_and_release_rng_forward_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fresh_summary, fresh_args, _, _ = _run(
        tmp_path,
        monkeypatch,
        output_name="fresh",
    )
    assert fresh_summary["executed_global_steps"] == [1, 2]
    monkeypatch.undo()
    runtime = Runtime()
    parent_checkpoint = fresh_args.output_dir / "checkpoint_step000002.pt"
    summary, args, runtime, _ = _run(
        tmp_path,
        monkeypatch,
        runtime=runtime,
        target=3,
        parent=2,
        resume_checkpoint=parent_checkpoint,
        output_name="resume",
    )
    report = _load_json(args.output_dir / "smoke_audit_report.json")

    assert summary["executed_global_steps"] == [3]
    assert report["validation_steps"] == [3]
    assert report["checkpoint_steps"] == [3]
    assert not any(event[0] == "reset_seed" for event in runtime.events)
    assert (args.output_dir / "checkpoint_step000003.pt").is_file()
    marker_names = [marker["name"] for marker in report["markers"]]
    assert marker_names.index("parent_payload_released") < marker_names.index(
        "global_rng_restored"
    )
    assert marker_names.index("restored_probe_temporary_released") < marker_names.index(
        "global_rng_restored"
    )
    assert marker_names.index("global_rng_restored") < marker_names.index(
        "first_resumed_forward"
    )
    restore_event = runtime.events.index(("restore_global_rng",))
    first_forward = runtime.events.index(("forward",))
    forbidden = {"reset_seed", "make_train_rng", "make_probe", "load_generator"}
    assert not any(event[0] in forbidden for event in runtime.events[restore_event + 1:first_forward])


def test_forward_exception_cleans_store_and_writes_no_pass_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = Runtime()
    runtime.forward_exception = RuntimeError("forward failed")
    args = _args(tmp_path)
    _install_runtime(monkeypatch, args, runtime)

    with pytest.raises(RuntimeError, match="forward failed"):
        smoke.run_m5_formal_smoke(
            args=args,
            config=_config(),
            dtype=torch.float32,
            device=torch.device("cuda:0"),
            current_git_sha=GIT_SHA,
            reference_checkpoint_sha256=REFERENCE_SHA,
        )

    assert all(store.live_sample_count == 0 for store in runtime.teacher_stores)
    assert all(store.live_conditional_count == 0 for store in runtime.conditional_stores)
    assert not (args.output_dir / "smoke_summary.json").exists()


def test_validation_fail_does_not_write_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = Runtime()
    runtime.validation_status = "FAIL"
    args = _args(tmp_path)
    _install_runtime(monkeypatch, args, runtime)

    with pytest.raises(RuntimeError, match="validation"):
        smoke.run_m5_formal_smoke(
            args=args,
            config=_config(),
            dtype=torch.float32,
            device=torch.device("cuda:0"),
            current_git_sha=GIT_SHA,
            reference_checkpoint_sha256=REFERENCE_SHA,
        )

    assert ("validation", 0) in runtime.events
    assert not (args.output_dir / "checkpoint_step000000.pt").exists()
    assert not (args.output_dir / "smoke_summary.json").exists()


@pytest.mark.parametrize("bad_loss", [float("nan"), float("inf")])
def test_nonfinite_train_loss_blocks_backward_step_and_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    bad_loss: float,
) -> None:
    runtime = Runtime()
    runtime.nonfinite_train_loss = bad_loss
    args = _args(tmp_path)
    _install_runtime(monkeypatch, args, runtime)
    backward_events: list[tuple[str]] = []

    class LossWithBackward:
        def backward(self) -> None:
            backward_events.append(("backward",))

    def forward(*args: Any, **kwargs: Any) -> SimpleNamespace:
        loss = LossWithBackward()
        losses = SimpleNamespace(
            total_loss=loss,
            main_loss=loss,
            mcp_depth_losses=(loss, loss, loss),
        )
        return SimpleNamespace(losses=losses)

    monkeypatch.setattr(train_m3, "run_nf_sf_forward_loss", forward)

    with pytest.raises(RuntimeError, match="non-finite train loss"):
        smoke.run_m5_formal_smoke(
            args=args,
            config=_config(),
            dtype=torch.float32,
            device=torch.device("cuda:0"),
            current_git_sha=GIT_SHA,
            reference_checkpoint_sha256=REFERENCE_SHA,
        )

    assert backward_events == []
    assert not (args.output_dir / "checkpoint_step000002.pt").exists()
    assert not (args.output_dir / "smoke_summary.json").exists()


def test_smoke_checkpoint_is_rejected_by_formal_parent_validator_and_resolver(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, args, _, plan = _run(tmp_path, monkeypatch)
    payload = torch.load(
        args.output_dir / "checkpoint_step000002.pt",
        map_location="cpu",
        weights_only=False,
    )

    formal_parent_shaped_payload = dict(payload)
    formal_parent_shaped_payload["global_step"] = 500
    with pytest.raises(TypeError, match="formal metadata"):
        train_m3.require_m5_formal_parent_checkpoint(
            parent_payload=formal_parent_shaped_payload,
            stage_contract=train_m3.resolve_m5_formal_stage_contract(2000),
            sample_plan_sha256=plan["sample_plan_sha256"],
            teacher_manifest_sha256=MANIFEST_SHA,
            conditional_artifact_sha256=ARTIFACT_SHA,
            current_git_sha=GIT_SHA,
        )
    with pytest.raises(ValueError, match="target_global_step"):
        train_m3.resolve_m5_formal_stage_contract(2)
    with pytest.raises(ValueError, match="target_global_step"):
        train_m3.resolve_m5_formal_stage_contract(3)
    for target in (500, 2000, 5000):
        bad_args = _args(tmp_path, target=target, output_name=f"bad-{target}")
        with pytest.raises(ValueError, match="smoke"):
            smoke.validate_smoke_cli_contract(bad_args, cuda_device_count=1)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("conditional_artifact_sha256", "0" * 64, "provenance"),
        ("status", "FAIL", "status"),
        ("schema", "wrong", "schema"),
        ("target_global_step", 3, "target_global_step"),
    ],
)
def test_resume_rejects_parent_provenance_status_schema_or_step(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: Any,
    match: str,
) -> None:
    _, fresh_args, _, _ = _run(tmp_path, monkeypatch, output_name="fresh")
    parent_checkpoint = fresh_args.output_dir / "checkpoint_step000002.pt"
    payload = torch.load(parent_checkpoint, map_location="cpu", weights_only=False)
    payload["m5_formal_smoke"][field] = value
    torch.save(payload, parent_checkpoint)
    monkeypatch.undo()
    runtime = Runtime()
    args = _args(
        tmp_path,
        target=3,
        parent=2,
        resume_checkpoint=parent_checkpoint,
        output_name=f"resume-{field}",
    )
    _install_runtime(monkeypatch, args, runtime)

    with pytest.raises(RuntimeError, match=match):
        smoke.run_m5_formal_smoke(
            args=args,
            config=_config(),
            dtype=torch.float32,
            device=torch.device("cuda:0"),
            current_git_sha=GIT_SHA,
            reference_checkpoint_sha256=REFERENCE_SHA,
        )


def test_resume_rejects_parent_global_step_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, fresh_args, _, _ = _run(tmp_path, monkeypatch, output_name="fresh")
    parent_checkpoint = fresh_args.output_dir / "checkpoint_step000002.pt"
    payload = torch.load(parent_checkpoint, map_location="cpu", weights_only=False)
    payload["global_step"] = 1
    torch.save(payload, parent_checkpoint)
    monkeypatch.undo()
    runtime = Runtime()
    args = _args(
        tmp_path,
        target=3,
        parent=2,
        resume_checkpoint=parent_checkpoint,
        output_name="resume-step-bad",
    )
    _install_runtime(monkeypatch, args, runtime)

    with pytest.raises(RuntimeError, match="global_step mismatch"):
        smoke.run_m5_formal_smoke(
            args=args,
            config=_config(),
            dtype=torch.float32,
            device=torch.device("cuda:0"),
            current_git_sha=GIT_SHA,
            reference_checkpoint_sha256=REFERENCE_SHA,
        )


def test_cli_rejects_mode_device_mcp_parent_and_nonempty_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cases = [
        {"mode": "frozen", "match": "mode"},
        {"device": "cpu", "match": "device"},
        {"mcp1_grid_aux_weight": 0.1, "match": "mcp1_grid_aux_weight"},
        {"target": 2, "parent": 2, "match": "parent_global_step"},
    ]
    for index, case in enumerate(cases):
        args = _args(
            tmp_path / f"case-{index}",
            target=case.get("target", 2),
            parent=case.get("parent"),
            mode=case.get("mode", "joint"),
            device=case.get("device", "cuda:0"),
            mcp1_grid_aux_weight=case.get("mcp1_grid_aux_weight", 0.0),
        )
        with pytest.raises((ValueError, TypeError), match=str(case["match"])):
            smoke.validate_smoke_cli_contract(args, cuda_device_count=1)

    args = _args(tmp_path / "nonempty")
    args.output_dir.mkdir()
    (args.output_dir / "file.txt").write_text("x", encoding="utf-8")
    _install_runtime(monkeypatch, args, Runtime())
    with pytest.raises(FileExistsError, match="empty"):
        smoke.run_m5_formal_smoke(
            args=args,
            config=_config(),
            dtype=torch.float32,
            device=torch.device("cuda:0"),
            current_git_sha=GIT_SHA,
            reference_checkpoint_sha256=REFERENCE_SHA,
        )


def test_report_acquire_release_order_and_cuda_fields_are_complete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, args, _, _ = _run(tmp_path, monkeypatch)
    report = _load_json(args.output_dir / "smoke_audit_report.json")
    events = report["store_events"]

    assert events[0]["action"] == "acquire"
    assert events[1]["action"] == "acquire"
    assert events[2]["action"] == "release"
    assert events[3]["action"] == "release"
    assert events[0]["kind"] == "teacher"
    assert events[1]["kind"] == "conditional"
    for section in ("train_steps", "validation", "checkpoint"):
        assert report["cuda_memory"][section]
        for record in report["cuda_memory"][section]:
            assert set(record["cuda"]) == {
                "memory_allocated",
                "memory_reserved",
                "max_memory_allocated",
                "max_memory_reserved",
            }
    _assert_no_tensor(report)
