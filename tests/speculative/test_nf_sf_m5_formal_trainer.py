from __future__ import annotations

import argparse
import inspect
import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn

import scripts.train_nf_sf_m3_overfit as train_m3
import utils.nf_sf_m4 as m4
from utils.nf_sf_m5_validation import M5_STREAMING_VALIDATION_SCHEMA

PLAN_SHA = "1" * 64
MANIFEST_SHA = "2" * 64
ARTIFACT_SHA = "3" * 64
REFERENCE_SHA = "4" * 64
GIT_SHA = "0123456789abcdef0123456789abcdef01234567"


class FakeTensorOnAnyDevice:
    def to(self, *args: Any, **kwargs: Any) -> FakeTensorOnAnyDevice:
        return self


class TinyGenerator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))

    def to(self, *args: Any, **kwargs: Any) -> TinyGenerator:
        return self

    def get_scheduler(self) -> SimpleNamespace:
        return SimpleNamespace(
            sigmas=FakeTensorOnAnyDevice(),
            timesteps=FakeTensorOnAnyDevice(),
        )


class FakeStore:
    def __init__(self, plan: dict[str, Any], events: list[tuple[Any, ...]], kind: str):
        self.plan = plan
        self.events = events
        self.kind = kind
        self.sample_plan_sha256 = PLAN_SHA
        self.manifest_sha256 = MANIFEST_SHA
        self.teacher_manifest_sha256 = MANIFEST_SHA
        self.artifact_sha256 = ARTIFACT_SHA
        self.train_identities = tuple(plan["train_sample_identities"])
        self.validation_identities = tuple(plan["validation_sample_identities"])
        self.fixed_decode_validation_identity = str(
            plan["fixed_decode_validation_identity"]
        )
        self.live_sample_count = 0
        self.live_conditional_count = 0
        self.max_live_sample_count = 0
        self.max_live_conditional_count = 0
        self.acquired: list[str] = []
        self.train_identity_steps: list[int] = []

    def train_identity_for_step(self, step: int) -> str:
        self.train_identity_steps.append(step)
        return str(m4.m4_train_entry_for_step(self.plan, step)["identity"])

    @contextmanager
    def acquire(self, identity: str) -> Iterator[Any]:
        self.acquired.append(identity)
        self.events.append((f"{self.kind}_acquire", identity))
        if self.kind == "teacher":
            self.live_sample_count = 1
            self.max_live_sample_count = max(self.max_live_sample_count, 1)
            payload = SimpleNamespace(
                metadata={
                    "identity": identity,
                    "prompt": f"prompt {identity}",
                    "sample_index": 0,
                    "split": "validation"
                    if identity.startswith("validation")
                    else "train",
                    "split_index": 0,
                    "manifest_path": "manifest.json",
                    "manifest_sha256": MANIFEST_SHA,
                    "dataset_root": "dataset",
                },
                selected_state=SimpleNamespace(identity=identity),
            )
        else:
            self.live_conditional_count = 1
            self.max_live_conditional_count = max(self.max_live_conditional_count, 1)
            payload = {"prompt_embeds": torch.ones(1)}
        try:
            yield payload
        finally:
            if self.kind == "teacher":
                self.live_sample_count = 0
            else:
                self.live_conditional_count = 0
            self.events.append((f"{self.kind}_release", identity))


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


def _plan(train_count: int = 2048, validation_count: int = 256) -> dict[str, Any]:
    train_entries = [
        {
            "identity": f"train-{index:04d}",
            "sample_index": index,
            "sample_id": f"train-{index:04d}",
            "split": "train",
            "split_index": index,
            "prompt_sha256": "a" * 64,
        }
        for index in range(train_count)
    ]
    validation_entries = [
        {
            "identity": f"validation-{index:04d}",
            "sample_index": 100_000 + index,
            "sample_id": f"validation-{index:04d}",
            "split": "validation",
            "split_index": index,
            "prompt_sha256": "b" * 64,
        }
        for index in range(validation_count)
    ]
    return {
        "schema": "nf_sf_m4_sample_plan_v1",
        "status": "PASS",
        "sample_plan_sha256": PLAN_SHA,
        "manifest_sha256": MANIFEST_SHA,
        "dataset_root": "dataset",
        "ordering_rule": "split_then_sample_index",
        "train_subset_size": train_count,
        "validation_subset_size": validation_count,
        "train_sample_identities": [entry["identity"] for entry in train_entries],
        "validation_sample_identities": [
            entry["identity"] for entry in validation_entries
        ],
        "fixed_decode_validation_identity": validation_entries[0]["identity"],
        "samples": {"train": train_entries, "validation": validation_entries},
    }


def _args(
    tmp_path: Path,
    *,
    target: int = 500,
    resume_checkpoint: Path | None = None,
    mode: str = "joint",
    device: str = "cuda:0",
    validation_steps: str = "0,500",
    checkpoint_steps: str = "0,500",
    mcp1_grid_aux_weight: float = 0.0,
) -> argparse.Namespace:
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
    return argparse.Namespace(
        config=tmp_path / "config.yaml",
        checkpoint=checkpoint,
        resume_checkpoint=resume_checkpoint,
        manifest=manifest,
        dataset_root=dataset_root,
        sample_index=None,
        sample_id=None,
        split=None,
        split_index=None,
        output_dir=tmp_path / f"out-{target}-{resume_checkpoint is not None}",
        mode=mode,
        train_seed=123,
        probe_seed=456,
        optimizer_steps=target,
        timing_warmup_steps=0,
        log_interval=10_000,
        checkpoint_interval=10_000,
        backbone_lr=1.0e-4,
        patch_embedding_lr=2.0e-4,
        mcp_lr=3.0e-4,
        weight_decay=0.01,
        mcp1_grid_aux_weight=mcp1_grid_aux_weight,
        m4_sample_plan=sample_plan,
        m5_formal_long_train=True,
        m5_conditionals_artifact=artifact_manifest.resolve(),
        validation_seed=789,
        validation_steps=validation_steps,
        checkpoint_steps=checkpoint_steps,
        dtype="float32",
        device=device,
    )


def _validation_report(step: int, *, status: str = "PASS", finite: bool = True) -> dict[str, Any]:
    aggregate = {
        "main_loss": 1.0,
        "mcp_depth1_loss": 2.0,
        "mcp_depth2_loss": 3.0,
        "mcp_depth3_loss": 4.0,
        "weighted_mcp_loss": 1.3,
        "total_validation_loss": 2.3,
    }
    return {
        "schema": M5_STREAMING_VALIDATION_SCHEMA,
        "status": status,
        "global_step": step,
        "mode": "joint",
        "sample_plan_sha256": PLAN_SHA,
        "conditional_artifact_sha256": ARTIFACT_SHA,
        "sample_count": 256,
        "validation_loss_finite_contract": finite,
        "nonfinite_validation_losses": []
        if finite
        else [{"scope": "sample", "sample_identity": "validation-0000"}],
        "aggregate_losses": aggregate,
    }


def _install_formal_runtime(
    monkeypatch: pytest.MonkeyPatch,
    plan: dict[str, Any],
    events: list[tuple[Any, ...]],
    *,
    validation_status: str = "PASS",
    validation_finite: bool = True,
    parent_payload: dict[str, Any] | None = None,
    inspect_resume_release: bool = False,
) -> None:
    monkeypatch.setattr(train_m3.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(train_m3.torch.cuda, "is_available", lambda: False)
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
            "sample_plan_sha256": PLAN_SHA,
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
                "model_checkpoint_sha256": "5" * 64,
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
    monkeypatch.setattr(train_m3, "write_m4_sample_plan", lambda *a, **k: None)
    monkeypatch.setattr(train_m3, "file_sha256", lambda *a, **k: REFERENCE_SHA)
    monkeypatch.setattr(
        train_m3,
        "M5TeacherSampleStore",
        lambda *a, **k: FakeStore(plan, events, "teacher"),
    )
    monkeypatch.setattr(
        train_m3,
        "M5ConditionalArtifactStore",
        lambda *a, **k: FakeStore(plan, events, "conditional"),
    )
    monkeypatch.setattr(
        train_m3,
        "load_m4_teacher_samples",
        lambda *a, **k: pytest.fail("eager teacher loader was called"),
    )
    monkeypatch.setattr(
        train_m3,
        "run_m5_m4_validation_for_step",
        lambda *a, **k: pytest.fail("eager validation wrapper was called"),
    )

    def fake_reset_global_seed(seed: int) -> None:
        events.append(("reset_seed", int(seed)))

    def fake_load_generator(*args: Any, **kwargs: Any) -> tuple[TinyGenerator, str, int]:
        events.append(("load_generator",))
        return TinyGenerator(), "FAKE", 1

    def fake_make_mcp_scheduler(*args: Any, **kwargs: Any) -> SimpleNamespace:
        events.append(("make_mcp_scheduler",))
        return TinyGenerator().get_scheduler()

    monkeypatch.setattr(train_m3, "reset_global_seed", fake_reset_global_seed)
    monkeypatch.setattr(train_m3, "load_generator", fake_load_generator)
    monkeypatch.setattr(train_m3, "make_mcp_scheduler", fake_make_mcp_scheduler)
    monkeypatch.setattr(train_m3, "selected_state_to_device", lambda state, **k: state)
    monkeypatch.setattr(
        train_m3,
        "m5_formal_conditional_to_device",
        lambda conditional, **k: dict(conditional),
    )
    monkeypatch.setattr(
        train_m3,
        "make_generator",
        lambda *a, **k: events.append(("make_train_rng",))
        or torch.Generator(device="cpu").manual_seed(123),
    )
    monkeypatch.setattr(
        train_m3,
        "make_m3_probe",
        lambda state, **k: events.append(("make_probe",))
        or SimpleNamespace(
            rng_state=torch.Generator(device="cpu").manual_seed(1).get_state(),
            noisy_batch=SimpleNamespace(state=state),
        ),
    )
    monkeypatch.setattr(
        train_m3,
        "configure_m3_optimizer_plan",
        lambda generator, **k: SimpleNamespace(
            mode="joint",
            optimizer_param_groups=[
                {
                    "params": [generator.weight],
                    "lr": 1.0e-4,
                    "name": "backbone",
                }
            ],
            audits=[],
        ),
    )
    monkeypatch.setattr(train_m3, "named_parameter_groups", lambda generator: {})
    monkeypatch.setattr(
        train_m3,
        "gradient_group_audit",
        lambda *a, **k: {"optimizer_contract": {"all_contract_pass": True}},
    )
    monkeypatch.setattr(train_m3, "has_nonfinite_grad", lambda generator: False)
    monkeypatch.setattr(
        train_m3,
        "prepare_nf_sf_noisy_batch",
        lambda state, **k: SimpleNamespace(
            state=state,
            epsilon_main=torch.ones(1),
            epsilon_depths=(torch.ones(1),),
        ),
    )

    def fake_forward(generator: TinyGenerator, **kwargs: Any) -> SimpleNamespace:
        loss = generator.weight * 0.0 + torch.ones(())
        losses = SimpleNamespace(
            total_loss=loss,
            main_loss=loss,
            mcp_depth_losses=(loss, loss, loss),
        )
        events.append(("forward", kwargs["conditional_dict"].copy()))
        return SimpleNamespace(losses=losses)

    monkeypatch.setattr(train_m3, "run_nf_sf_forward_loss", fake_forward)
    monkeypatch.setattr(
        train_m3,
        "loss_dict_to_floats",
        lambda losses: {
            "main_loss": 1.0,
            "mcp_depth1_loss": 1.0,
            "mcp_depth2_loss": 1.0,
            "mcp_depth3_loss": 1.0,
            "total_loss": 1.0,
        },
    )

    def fake_streaming_validation(**kwargs: Any) -> dict[str, Any]:
        step = int(kwargs["global_step"])
        events.append(("validation", step))
        return _validation_report(
            step,
            status=validation_status,
            finite=validation_finite,
        )

    monkeypatch.setattr(train_m3, "run_m5_streaming_validation", fake_streaming_validation)

    def fake_step_artifacts(**kwargs: Any) -> None:
        events.append(
            (
                "step_artifacts",
                int(kwargs["global_step"]),
                bool(kwargs["should_checkpoint"]),
                kwargs.get("extra_checkpoint_payload_fields"),
            )
        )

    monkeypatch.setattr(train_m3, "write_m5_step_artifacts", fake_step_artifacts)

    def fake_initial_checkpoint(**kwargs: Any) -> Path:
        events.append(("initial_checkpoint", int(kwargs["global_step"])))
        assert kwargs["formal_checkpoint_metadata"]["schema"] == (
            train_m3.M5_FORMAL_TRAINER_SCHEMA
        )
        return kwargs["output_dir"] / "checkpoint_step000000.pt"

    monkeypatch.setattr(
        train_m3,
        "write_m5_formal_probe_checkpoint_node",
        fake_initial_checkpoint,
    )
    monkeypatch.setattr(
        train_m3,
        "cuda_synchronize_if_available",
        lambda *a, **k: events.append(("cuda_sync",)),
    )
    monkeypatch.setattr(train_m3, "reset_peak_memory_stats_if_available", lambda *a, **k: None)
    monkeypatch.setattr(
        train_m3.gc,
        "collect",
        lambda: events.append(("gc_collect",)) or 0,
    )
    monkeypatch.setattr(
        train_m3,
        "cuda_peak_memory_summary",
        lambda *a, **k: {
            "max_memory_allocated_bytes": None,
            "max_memory_reserved_bytes": None,
        },
    )

    if parent_payload is not None:
        monkeypatch.setattr(
            train_m3,
            "load_parent_resume_checkpoint",
            lambda *a, **k: (parent_payload, "6" * 64),
        )
        monkeypatch.setattr(
            train_m3,
            "current_m5_resume_run_fields",
            lambda *a, **k: {"current": "fields"},
        )
        monkeypatch.setattr(
            train_m3,
            "build_and_validate_m5_resume_report",
            lambda *a, **k: {
                "status": "PASS",
                "resumed_global_step": parent_payload["global_step"],
                "rng_restore": {
                    "train_generator_device": "cuda:0",
                    "probe_generator_device": "cuda:0",
                },
            },
        )
        monkeypatch.setattr(
            train_m3,
            "strict_load_m5_generator_state",
            lambda *a, **k: {"status": "PASS"},
        )
        monkeypatch.setattr(
            train_m3,
            "move_loaded_optimizer_state_to_device",
            lambda *a, **k: events.append(("optimizer_move", str(k["device"])))
            or {"status": "moved"},
        )
        monkeypatch.setattr(
            train_m3,
            "extract_resume_rng_states",
            lambda *a, **k: {"train_generator_state": torch.Generator().get_state()},
        )
        monkeypatch.setattr(
            train_m3,
            "restore_torch_generator_from_state",
            lambda *a, **k: torch.Generator(device="cpu").manual_seed(999),
        )
        monkeypatch.setattr(
            train_m3,
            "restore_m5_probe_from_checkpoint",
            lambda *a, **k: (
                SimpleNamespace(
                    rng_state=torch.Generator(device="cpu").manual_seed(2).get_state(),
                    noisy_batch=SimpleNamespace(state=SimpleNamespace()),
                ),
                {"prompt_embeds": torch.ones(1)},
            ),
        )
        monkeypatch.setattr(
            train_m3,
            "run_m3_probe_forward",
            lambda *a, **k: SimpleNamespace(
                losses={
                    "main_loss": 1.0,
                    "mcp_depth1_loss": 1.0,
                    "mcp_depth2_loss": 1.0,
                    "mcp_depth3_loss": 1.0,
                    "total_loss": 1.0,
                },
                outputs={"main_flow_pred": torch.ones(1)},
            ),
        )
        monkeypatch.setattr(
            train_m3,
            "require_restored_probe_matches_checkpoint",
            lambda *a, **k: events.append(("probe_restore",))
            or {"status": "PASS"},
        )
        monkeypatch.setattr(
            train_m3,
            "restore_global_rng_states",
            lambda *a, **k: _record_restore_global_rng(
                events,
                inspect_resume_release=inspect_resume_release,
            ),
        )


def _parent_payload(step: int = 500) -> dict[str, Any]:
    contract = train_m3.resolve_m5_formal_stage_contract(step)
    formal_config = {
        "schema": train_m3.M5_FORMAL_TRAINER_SCHEMA,
        "enabled": True,
        "sample_plan_sha256": PLAN_SHA,
        "teacher_manifest_sha256": MANIFEST_SHA,
        "conditional_artifact_sha256": ARTIFACT_SHA,
        "validation_implementation_schema": M5_STREAMING_VALIDATION_SCHEMA,
    }
    return {
        "global_step": step,
        "git_sha": GIT_SHA,
        "resolved_config": {"m5_formal": dict(formal_config)},
        "generator": {},
        "optimizer": {
            "state": {},
            "param_groups": [
                {
                    "lr": 1.0e-4,
                    "betas": (0.0, 0.999),
                    "eps": 1.0e-8,
                    "weight_decay": 0.01,
                    "params": [0],
                    "name": "backbone",
                }
            ],
        },
        "m5_formal_trainer": {
            "schema": train_m3.M5_FORMAL_TRAINER_SCHEMA,
            "status": "PASS",
            "formal_enabled": True,
            "stage": train_m3.m5_formal_stage_name(contract),
            "sample_plan_sha256": PLAN_SHA,
            "teacher_manifest_sha256": MANIFEST_SHA,
            "conditional_artifact_sha256": ARTIFACT_SHA,
            "validation_implementation_schema": M5_STREAMING_VALIDATION_SCHEMA,
            "stage_contract": train_m3.m5_formal_stage_contract_json(contract),
        },
    }


def _record_restore_global_rng(
    events: list[tuple[Any, ...]],
    *,
    inspect_resume_release: bool,
) -> None:
    if inspect_resume_release:
        frame_locals: dict[str, Any] | None = None
        for frame_info in inspect.stack():
            if frame_info.function == "run_m5_formal_training":
                frame_locals = frame_info.frame.f_locals
                break
        assert frame_locals is not None
        assert frame_locals["parent_resume_payload"] is None
        assert frame_locals["restored_probe_forward"] is None
        assert ("gc_collect",) in events
    events.append(("restore_global_rng",))


def test_non_formal_301_gate_is_unchanged(tmp_path: Path) -> None:
    args = _args(tmp_path, target=301)
    args.m5_formal_long_train = False
    args.m5_conditionals_artifact = None
    args.m4_sample_plan = None
    args.validation_seed = None
    args.validation_steps = None
    args.checkpoint_steps = None

    with pytest.raises(ValueError, match="300 optimizer steps"):
        train_m3.validate_config(_config(), args)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("optimizer_steps", 301, "target_global_step"),
        ("optimizer_steps", 501, "target_global_step"),
        ("mode", "frozen", "mode"),
        ("device", "cpu", "device"),
        ("device", "cuda:1", "device"),
        ("mcp1_grid_aux_weight", 0.1, "mcp1_grid_aux_weight"),
        ("m4_sample_plan", None, "sample_plan_path"),
        ("m5_conditionals_artifact", None, "m5_conditionals_artifact"),
    ],
)
def test_formal_cli_rejects_invalid_requests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: Any,
    match: str,
) -> None:
    args = _args(tmp_path)
    setattr(args, field, value)
    monkeypatch.setattr(train_m3.torch.cuda, "device_count", lambda: 1)

    with pytest.raises((ValueError, TypeError), match=match):
        train_m3.validate_config(_config(), args)


def test_formal_cli_rejects_multi_gpu(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _args(tmp_path)
    monkeypatch.setattr(train_m3.torch.cuda, "device_count", lambda: 2)

    with pytest.raises(ValueError, match="expected_cuda_device_count"):
        train_m3.validate_config(_config(), args)


def test_preserving_order_schedule_parser_rejects_later() -> None:
    assert train_m3.parse_m5_formal_step_list("500,0,500", name="x") == (
        500,
        0,
        500,
    )

    with pytest.raises(ValueError, match="validation_steps"):
        train_m3.validate_m5_formal_stage_request(
            mode="joint",
            target_global_step=500,
            validation_steps=(500, 0),
            checkpoint_steps=(0, 500),
            sample_plan_path="plan.json",
            conditionals_artifact_path="manifest.json",
            device="cuda:0",
            expected_cuda_device_count=1,
            resume_checkpoint_path=None,
            parent_global_step=None,
        )


def test_stage_a_fresh_validation_before_checkpoint_and_lazy_loop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[Any, ...]] = []
    plan = _plan()
    args = _args(tmp_path)
    _install_formal_runtime(monkeypatch, plan, events)

    train_m3.run_m5_formal_training(
        args=args,
        config=_config(),
        dtype=torch.float32,
        device=torch.device("cuda:0"),
        current_git_sha=GIT_SHA,
        reference_checkpoint_sha256=REFERENCE_SHA,
    )

    assert events.index(("reset_seed", 123)) < events.index(("load_generator",))
    assert events.index(("validation", 0)) < events.index(("initial_checkpoint", 0))
    first_forward_index = next(
        index for index, event in enumerate(events) if event[0] == "forward"
    )
    sync_indices = [
        index for index, event in enumerate(events) if event[0] == "cuda_sync"
    ]
    assert max(index for index in sync_indices if index < first_forward_index)
    assert min(index for index in sync_indices if index > first_forward_index)
    final_checkpoint_events = [
        event for event in events if event[:3] == ("step_artifacts", 500, True)
    ]
    assert final_checkpoint_events
    assert final_checkpoint_events[0][3]["m5_formal_trainer"]["schema"] == (
        train_m3.M5_FORMAL_TRAINER_SCHEMA
    )
    train_acquires = [
        event[1]
        for event in events
        if event[0] == "teacher_acquire" and str(event[1]).startswith("train")
    ]
    assert train_acquires[0] == "train-0000"
    assert train_acquires[-1] == "train-0499"
    assert len(train_acquires) == 500
    assert not any(event[0] == "eager_teacher" for event in events)
    summary = json.loads((args.output_dir / "training_summary.json").read_text())
    assert summary["executed_global_steps"] == list(range(1, 501))
    assert summary["validation_reports"] == [
        "validation_step000000.json",
        "validation_step000500.json",
    ]


def test_stage_a_validation_fail_prevents_step0_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[Any, ...]] = []
    plan = _plan()
    args = _args(tmp_path)
    _install_formal_runtime(
        monkeypatch,
        plan,
        events,
        validation_status="FAIL",
        validation_finite=False,
    )

    with pytest.raises(RuntimeError, match="validation loss contract"):
        train_m3.run_m5_formal_training(
            args=args,
            config=_config(),
            dtype=torch.float32,
            device=torch.device("cuda:0"),
            current_git_sha=GIT_SHA,
            reference_checkpoint_sha256=REFERENCE_SHA,
        )

    assert ("validation", 0) in events
    assert not any(event[0] == "initial_checkpoint" for event in events)


@pytest.mark.parametrize(
    ("target", "parent", "expected_first", "expected_node"),
    [(2000, 500, 501, 2000), (5000, 2000, 2001, 5000)],
)
def test_resume_stages_execute_only_new_target_node(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target: int,
    parent: int,
    expected_first: int,
    expected_node: int,
) -> None:
    events: list[tuple[Any, ...]] = []
    plan = _plan()
    parent_path = tmp_path / f"checkpoint_step{parent:06d}.pt"
    parent_path.write_bytes(b"parent")
    schedules = "0,500,2000" if target == 2000 else "0,500,2000,5000"
    args = _args(
        tmp_path,
        target=target,
        resume_checkpoint=parent_path,
        validation_steps=schedules,
        checkpoint_steps=schedules,
    )
    _install_formal_runtime(
        monkeypatch,
        plan,
        events,
        parent_payload=_parent_payload(parent),
        inspect_resume_release=True,
    )

    train_m3.run_m5_formal_training(
        args=args,
        config=_config(),
        dtype=torch.float32,
        device=torch.device("cuda:0"),
        current_git_sha=GIT_SHA,
        reference_checkpoint_sha256=REFERENCE_SHA,
    )

    validation_steps = [event[1] for event in events if event[0] == "validation"]
    assert validation_steps == [expected_node]
    first_forward_index = next(
        index for index, event in enumerate(events) if event[0] == "forward"
    )
    first_forward = events[first_forward_index]
    assert tuple(first_forward[1]) == ("prompt_embeds",)
    train_acquires = [
        event[1]
        for event in events
        if event[0] == "teacher_acquire" and str(event[1]).startswith("train")
    ]
    assert train_acquires[0] == m4.m4_train_entry_for_step(plan, expected_first)[
        "identity"
    ]
    assert ("optimizer_move", "cuda:0") in events
    assert ("probe_restore",) in events
    assert ("restore_global_rng",) in events
    restore_index = events.index(("restore_global_rng",))
    assert events.index(("gc_collect",)) < restore_index
    assert restore_index < first_forward_index
    assert not any(event[0] == "reset_seed" for event in events)
    rng_init_events = {"make_train_rng", "make_probe", "load_generator"}
    assert not any(
        event[0] in rng_init_events
        for event in events[restore_index + 1:first_forward_index]
    )
    assert not any(event[0] == "initial_checkpoint" for event in events)


def test_formal_train_step_cleans_store_after_forward_exception(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[Any, ...]] = []
    plan = _plan()
    teacher_store = FakeStore(plan, events, "teacher")
    conditional_store = FakeStore(plan, events, "conditional")
    generator = TinyGenerator()
    optimizer = torch.optim.AdamW([{"params": [generator.weight], "name": "backbone"}])
    monkeypatch.setattr(train_m3, "selected_state_to_device", lambda state, **k: state)
    monkeypatch.setattr(
        train_m3,
        "m5_formal_conditional_to_device",
        lambda conditional, **k: dict(conditional),
    )
    monkeypatch.setattr(
        train_m3,
        "prepare_nf_sf_noisy_batch",
        lambda state, **k: SimpleNamespace(
            state=state,
            epsilon_main=torch.ones(1),
            epsilon_depths=(torch.ones(1),),
        ),
    )
    monkeypatch.setattr(
        train_m3,
        "run_nf_sf_forward_loss",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("forward failed")),
    )

    with pytest.raises(RuntimeError, match="forward failed"):
        train_m3.run_m5_formal_train_step(
            step=1,
            target_global_step=500,
            teacher_store=teacher_store,
            conditional_store=conditional_store,
            sample_plan=plan,
            generator=generator,
            optimizer=optimizer,
            scheduler_main=object(),
            scheduler_mcp=object(),
            train_rng=torch.Generator(device="cpu"),
            device=torch.device("cpu"),
            dtype=torch.float32,
            mode="joint",
            validation_steps=(0, 500),
            checkpoint_steps=(0, 500),
            checkpoint_interval=10,
            log_interval=10,
            timing_warmup_steps=0,
        )

    assert teacher_store.live_sample_count == 0
    assert conditional_store.live_conditional_count == 0
    assert events[-2:] == [
        ("conditional_release", "train-0000"),
        ("teacher_release", "train-0000"),
    ]


@pytest.mark.parametrize("bad_loss_value", [float("nan"), float("inf")])
def test_formal_train_step_rejects_nonfinite_loss_before_backward_or_step(
    monkeypatch: pytest.MonkeyPatch,
    bad_loss_value: float,
) -> None:
    events: list[tuple[Any, ...]] = []
    plan = _plan()
    teacher_store = FakeStore(plan, events, "teacher")
    conditional_store = FakeStore(plan, events, "conditional")
    generator = TinyGenerator()
    optimizer = torch.optim.AdamW([{"params": [generator.weight], "name": "backbone"}])
    optimizer.step = lambda *a, **k: events.append(("optimizer_step",))
    monkeypatch.setattr(train_m3, "selected_state_to_device", lambda state, **k: state)
    monkeypatch.setattr(
        train_m3,
        "m5_formal_conditional_to_device",
        lambda conditional, **k: dict(conditional),
    )
    monkeypatch.setattr(
        train_m3,
        "prepare_nf_sf_noisy_batch",
        lambda state, **k: SimpleNamespace(
            state=state,
            epsilon_main=torch.ones(1),
            epsilon_depths=(torch.ones(1),),
        ),
    )
    losses = SimpleNamespace(
        total_loss=SimpleNamespace(
            backward=lambda: events.append(("backward",)),
        ),
        main_loss=object(),
        mcp_depth_losses=(object(), object(), object()),
    )
    monkeypatch.setattr(
        train_m3,
        "run_nf_sf_forward_loss",
        lambda *a, **k: SimpleNamespace(losses=losses),
    )
    monkeypatch.setattr(
        train_m3,
        "loss_dict_to_floats",
        lambda value: {
            "main_loss": 1.0,
            "mcp_depth1_loss": 1.0,
            "mcp_depth2_loss": 1.0,
            "mcp_depth3_loss": 1.0,
            "total_loss": bad_loss_value,
        },
    )

    with pytest.raises(RuntimeError, match="non-finite train loss"):
        train_m3.run_m5_formal_train_step(
            step=1,
            target_global_step=500,
            teacher_store=teacher_store,
            conditional_store=conditional_store,
            sample_plan=plan,
            generator=generator,
            optimizer=optimizer,
            scheduler_main=object(),
            scheduler_mcp=object(),
            train_rng=torch.Generator(device="cpu"),
            device=torch.device("cpu"),
            dtype=torch.float32,
            mode="joint",
            validation_steps=(0, 500),
            checkpoint_steps=(0, 500),
            checkpoint_interval=10,
            log_interval=10,
            timing_warmup_steps=0,
        )

    assert ("backward",) not in events
    assert ("optimizer_step",) not in events
    assert teacher_store.live_sample_count == 0
    assert conditional_store.live_conditional_count == 0


def test_parent_formal_metadata_fail_closed() -> None:
    contract = train_m3.resolve_m5_formal_stage_contract(2000)
    with pytest.raises(TypeError, match="formal metadata"):
        train_m3.require_m5_formal_parent_checkpoint(
            parent_payload={"global_step": 500},
            stage_contract=contract,
            sample_plan_sha256=PLAN_SHA,
            teacher_manifest_sha256=MANIFEST_SHA,
            conditional_artifact_sha256=ARTIFACT_SHA,
            current_git_sha=GIT_SHA,
        )

    bad_payload = _parent_payload(500)
    bad_payload["m5_formal_trainer"]["conditional_artifact_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="provenance mismatch"):
        train_m3.require_m5_formal_parent_checkpoint(
            parent_payload=bad_payload,
            stage_contract=contract,
            sample_plan_sha256=PLAN_SHA,
            teacher_manifest_sha256=MANIFEST_SHA,
            conditional_artifact_sha256=ARTIFACT_SHA,
            current_git_sha=GIT_SHA,
        )


def test_parent_formal_checkpoint_status_must_pass() -> None:
    contract = train_m3.resolve_m5_formal_stage_contract(2000)
    payload = _parent_payload(500)
    payload["m5_formal_trainer"]["status"] = "FAIL"

    with pytest.raises(RuntimeError, match="status mismatch"):
        train_m3.require_m5_formal_parent_checkpoint(
            parent_payload=payload,
            stage_contract=contract,
            sample_plan_sha256=PLAN_SHA,
            teacher_manifest_sha256=MANIFEST_SHA,
            conditional_artifact_sha256=ARTIFACT_SHA,
            current_git_sha=GIT_SHA,
        )


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("target_global_step", 499),
        ("parent_global_step", 123),
        ("validation_steps", [500, 0]),
        ("checkpoint_steps", [0]),
        ("is_resume_stage", True),
    ],
)
def test_parent_stage_contract_mismatch_rejects(
    field: str,
    bad_value: Any,
) -> None:
    contract = train_m3.resolve_m5_formal_stage_contract(2000)
    payload = _parent_payload(500)
    payload["m5_formal_trainer"]["stage_contract"][field] = bad_value

    with pytest.raises(RuntimeError, match="stage_contract mismatch"):
        train_m3.require_m5_formal_parent_checkpoint(
            parent_payload=payload,
            stage_contract=contract,
            sample_plan_sha256=PLAN_SHA,
            teacher_manifest_sha256=MANIFEST_SHA,
            conditional_artifact_sha256=ARTIFACT_SHA,
            current_git_sha=GIT_SHA,
        )


@pytest.mark.parametrize(
    "field",
    [
        "target_global_step",
        "parent_global_step",
        "validation_steps",
        "checkpoint_steps",
        "is_resume_stage",
    ],
)
def test_parent_stage_contract_missing_field_rejects(field: str) -> None:
    contract = train_m3.resolve_m5_formal_stage_contract(2000)
    payload = _parent_payload(500)
    del payload["m5_formal_trainer"]["stage_contract"][field]

    with pytest.raises(RuntimeError, match="stage_contract.*keys mismatch"):
        train_m3.require_m5_formal_parent_checkpoint(
            parent_payload=payload,
            stage_contract=contract,
            sample_plan_sha256=PLAN_SHA,
            teacher_manifest_sha256=MANIFEST_SHA,
            conditional_artifact_sha256=ARTIFACT_SHA,
            current_git_sha=GIT_SHA,
        )


def test_parent_stage_name_mismatch_rejects() -> None:
    contract = train_m3.resolve_m5_formal_stage_contract(2000)
    payload = _parent_payload(500)
    payload["m5_formal_trainer"]["stage"] = "stage_b"

    with pytest.raises(RuntimeError, match="stage mismatch"):
        train_m3.require_m5_formal_parent_checkpoint(
            parent_payload=payload,
            stage_contract=contract,
            sample_plan_sha256=PLAN_SHA,
            teacher_manifest_sha256=MANIFEST_SHA,
            conditional_artifact_sha256=ARTIFACT_SHA,
            current_git_sha=GIT_SHA,
        )


def test_legal_stage_a_and_stage_b_parents_pass() -> None:
    stage_b_contract = train_m3.resolve_m5_formal_stage_contract(2000)
    stage_c_contract = train_m3.resolve_m5_formal_stage_contract(5000)

    stage_a_metadata = train_m3.require_m5_formal_parent_checkpoint(
        parent_payload=_parent_payload(500),
        stage_contract=stage_b_contract,
        sample_plan_sha256=PLAN_SHA,
        teacher_manifest_sha256=MANIFEST_SHA,
        conditional_artifact_sha256=ARTIFACT_SHA,
        current_git_sha=GIT_SHA,
    )
    stage_b_metadata = train_m3.require_m5_formal_parent_checkpoint(
        parent_payload=_parent_payload(2000),
        stage_contract=stage_c_contract,
        sample_plan_sha256=PLAN_SHA,
        teacher_manifest_sha256=MANIFEST_SHA,
        conditional_artifact_sha256=ARTIFACT_SHA,
        current_git_sha=GIT_SHA,
    )

    assert stage_a_metadata["stage"] == "stage_a"
    assert stage_b_metadata["stage"] == "stage_b"


def test_resolved_config_keeps_stage_out_of_locked_formal_invariants(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _plan()
    args = _args(tmp_path)
    manifest_path = args.m5_conditionals_artifact
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
    conditionals_manifest = {
        "schema": "nf_sf_m5_conditional_artifact_v1",
        "encoder_provenance": {
            "encoder_class": "Tiny",
            "model_checkpoint_path": "model.pt",
            "model_checkpoint_sha256": "5" * 64,
            "tokenizer_path": "tokenizer",
            "dtype": "torch.float32",
        },
    }
    config = train_m3.build_m5_formal_resolved_config(
        config=_config(),
        args=args,
        device=torch.device("cuda:0"),
        sample_plan=plan,
        sample_plan_sha256=PLAN_SHA,
        formal_plan_audit={
            "train_sample_count": 2048,
            "validation_sample_count": 256,
            "manifest_sha256": MANIFEST_SHA,
        },
        conditionals_manifest_path=manifest_path,
        conditionals_manifest=conditionals_manifest,
        conditional_artifact_sha256=ARTIFACT_SHA,
        optimizer_config={"optimizer": "AdamW", "betas": [0.0, 0.999], "eps": 1e-8},
    )
    metadata = train_m3.m5_formal_checkpoint_metadata(
        stage_contract=train_m3.resolve_m5_formal_stage_contract(500),
        resolved_config=config,
        conditionals_manifest_path=manifest_path,
        conditional_artifact_sha256=ARTIFACT_SHA,
    )

    assert "m5_formal_stage" not in config
    assert config["m5_formal"]["schema"] == train_m3.M5_FORMAL_TRAINER_SCHEMA
    assert config["m5_formal"]["conditional_artifact_sha256"] == ARTIFACT_SHA
    assert metadata["stage_contract"]["target_global_step"] == 500
