from __future__ import annotations

import copy
import random
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn

import scripts.train_nf_sf_m3_overfit as train_m3
import utils.nf_sf_m3 as m3
import utils.nf_sf_m4 as m4
import utils.nf_sf_m5 as m5
from utils.nf_sf_tensors import make_cpu_generator
from utils.nf_sf_training import NFSFLossBreakdown
from utils.scheduler import FlowMatchScheduler

TEST_GIT_SHA = "0123456789abcdef0123456789abcdef01234567"
BAD_GIT_SHA = "1111111111111111111111111111111111111111"
MANIFEST_SHA = "a" * 64
PROMPT_SHA_TRAIN = "b" * 64
PROMPT_SHA_VALIDATION = "c" * 64


class TinyResumeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.groups = nn.ModuleDict(
            {
                group_name: nn.Linear(1, 1, bias=False)
                for group_name in m3.M3_PARAMETER_GROUP_NAMES
            }
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        total = torch.zeros((), dtype=value.dtype, device=value.device)
        for module in self.groups.values():
            total = total + module(value).sum()
        return total


def _scheduler(shift: float) -> FlowMatchScheduler:
    scheduler = FlowMatchScheduler(shift=shift, sigma_min=0.0, extra_one_step=True)
    scheduler.set_timesteps(1000, training=True)
    return scheduler


def _selected_state() -> Any:
    latent = torch.arange(15, dtype=torch.float32).reshape(1, 15, 1, 1, 1)
    return m3.select_m3_selected_state(latent)


def _entry(split: str, split_index: int, sample_index: int) -> dict[str, Any]:
    prompt_sha = PROMPT_SHA_TRAIN if split == "train" else PROMPT_SHA_VALIDATION
    identity = (
        f"sample_index={sample_index}|sample_id={split}-{split_index:03d}|"
        f"split={split}|split_index={split_index}|prompt_sha256={prompt_sha}"
    )
    return {
        "identity": identity,
        "sample_index": sample_index,
        "sample_id": f"{split}-{split_index:03d}",
        "split": split,
        "split_index": split_index,
        "prompt": f"{split} prompt {split_index}",
        "prompt_sha256": prompt_sha,
    }


def _sample_plan(tmp_path: Path) -> dict[str, Any]:
    train_entries = [_entry("train", index, index) for index in range(3)]
    validation_entries = [
        _entry("validation", index, 100 + index)
        for index in range(2)
    ]
    plan = {
        "schema": m4.M4_SAMPLE_PLAN_SCHEMA,
        "manifest_path": str((tmp_path / "manifest.json").resolve()),
        "manifest_sha256": MANIFEST_SHA,
        "dataset_root": None,
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
    m4.validate_m4_sample_plan(plan)
    return plan


def _resolved_config(
    tmp_path: Path,
    plan: dict[str, Any],
    *,
    mode: str = "joint",
) -> dict[str, Any]:
    return {
        "model_config": {
            "num_frame_per_block": 3,
            "mcp_num_modules": 3,
            "mcp_num_layers": 3,
            "mcp_tap_layers": [3, 11, 19, 29],
            "mcp_depth_weights": [0.5, 0.2, 0.1],
            "model_kwargs": {"timestep_shift": 5.0},
        },
        "m3": {
            "mode": mode,
            "manifest": str((tmp_path / "manifest.json").resolve()),
            "dataset_root": None,
            "sample_index": 100,
            "sample_id": "validation-000",
            "split": "validation",
            "split_index": 0,
            "train_seed": 101,
            "probe_seed": 202,
            "optimizer_steps": 6,
            "timing_warmup_steps": 0,
            "log_interval": 1,
            "checkpoint_interval": 3,
            "backbone_lr": 1.0e-3,
            "patch_embedding_lr": 2.0e-3,
            "mcp_lr": 3.0e-3,
            "weight_decay": 0.01,
            "mcp1_grid_aux_weight": 1.0,
            "mcp1_grid_aux_enabled": True,
            "mcp1_grid_timesteps": [1000.0, 937.5, 833.3333129882812, 625.0],
            "mcp1_grid_schedule": {"source": "teacher_payload"},
            "optimizer_config": {
                "optimizer": "AdamW",
                "betas": [0.0, 0.999],
                "eps": 1.0e-8,
                "weight_decay": 0.01,
            },
            "dtype": "float32",
            "device": "cpu",
        },
        "m4": {
            "enabled": True,
            "sample_plan_path": str((tmp_path / "m4_sample_plan.json").resolve()),
            "sample_plan_sha256": str(plan["sample_plan_sha256"]),
            "train_sample_identities": list(plan["train_sample_identities"]),
            "validation_sample_identities": list(
                plan["validation_sample_identities"]
            ),
            "train_subset_size": int(plan["train_subset_size"]),
            "validation_subset_size": int(plan["validation_subset_size"]),
            "validation_seed": 303,
            "validation_steps": [0, 3, 6],
            "checkpoint_steps": [0, 3, 6],
            "fixed_decode_validation_identity": str(
                plan["fixed_decode_validation_identity"]
            ),
            "sample_ordering_rule": str(plan["ordering_rule"]),
            "ordering_rule": str(plan["ordering_rule"]),
        },
    }


def _sample_metadata(tmp_path: Path, plan: dict[str, Any]) -> dict[str, Any]:
    entry = plan["samples"]["validation"][0]
    metadata = dict(entry)
    metadata.update(
        {
            "manifest_path": str((tmp_path / "manifest.json").resolve()),
            "manifest_sha256": MANIFEST_SHA,
            "chunk_frames": 3,
            "target_latent": {"sha256": "target"},
        }
    )
    return metadata


def _model_and_optimizer(seed: int) -> tuple[TinyResumeModel, torch.optim.AdamW]:
    torch.manual_seed(seed)
    model = TinyResumeModel()
    group_lrs = {"backbone": 1.0e-3, "patch_embedding": 2.0e-3}
    groups = []
    for group_name in m3.M3_PARAMETER_GROUP_NAMES:
        groups.append(
            {
                "name": group_name,
                "params": list(model.groups[group_name].parameters()),
                "lr": group_lrs.get(group_name, 3.0e-3),
            }
        )
    optimizer = torch.optim.AdamW(
        groups,
        betas=train_m3.ADAMW_BETAS,
        eps=train_m3.ADAMW_EPS,
        weight_decay=0.01,
    )
    return model, optimizer


def _probe() -> m3.M3Probe:
    return m3.make_m3_probe(
        _selected_state(),
        scheduler_main=_scheduler(5.0),
        scheduler_mcp=_scheduler(10.0),
        seed=202,
    )


def _probe_forward(model: TinyResumeModel, probe: m3.M3Probe) -> Any:
    scale = model(torch.ones(1, 1)).detach().reshape(1, 1, 1, 1, 1)
    outputs = {
        "main_flow_pred": torch.zeros_like(probe.noisy_batch.target_flow_main) + scale,
        "mcp_depth1_flow_pred": torch.zeros_like(
            probe.noisy_batch.target_flow_depths[0]
        )
        + scale * 2.0,
        "mcp_depth2_flow_pred": torch.zeros_like(
            probe.noisy_batch.target_flow_depths[1]
        )
        + scale * 3.0,
        "mcp_depth3_flow_pred": torch.zeros_like(
            probe.noisy_batch.target_flow_depths[2]
        )
        + scale * 4.0,
    }
    main_loss = outputs["main_flow_pred"].float().mean()
    depth_losses = (
        outputs["mcp_depth1_flow_pred"].float().mean(),
        outputs["mcp_depth2_flow_pred"].float().mean(),
        outputs["mcp_depth3_flow_pred"].float().mean(),
    )
    total_loss = main_loss + sum(depth_losses)
    return SimpleNamespace(
        outputs=outputs,
        losses=NFSFLossBreakdown(
            total_loss=total_loss,
            main_loss=main_loss,
            mcp_depth_losses=depth_losses,
        ),
    )


def _prompt_embedding() -> dict[str, torch.Tensor]:
    return {"prompt_embeds": torch.tensor([[[1.0, 2.0, 3.0]]])}


class CountingTextEncoder:
    def __init__(self) -> None:
        self.calls = 0
        self.prompts: list[str] = []

    def __call__(self, prompts: list[str]) -> dict[str, torch.Tensor]:
        self.calls += 1
        self.prompts.extend(prompts)
        return {"prompt_embeds": torch.full((1, 1, 3), float(self.calls))}


def _sample_objects_by_identity(
    plan: dict[str, Any],
    split: str,
) -> dict[str, SimpleNamespace]:
    return {
        str(entry["identity"]): SimpleNamespace(metadata={"prompt": entry["prompt"]})
        for entry in plan["samples"][split]
    }


def _train_step(
    *,
    model: TinyResumeModel,
    optimizer: torch.optim.AdamW,
    train_rng: torch.Generator,
    plan: dict[str, Any],
    step: int,
    target_step: int = 6,
    validation_steps: tuple[int, ...] = (0, 3, 6),
    checkpoint_steps: tuple[int, ...] = (0, 3, 6),
    timing_warmup_steps: int = 0,
) -> dict[str, Any]:
    step_plan = train_m3.m5_training_step_orchestration(
        global_step=step,
        target_global_step=target_step,
        sample_plan=plan,
        validation_steps=validation_steps,
        checkpoint_steps=checkpoint_steps,
        checkpoint_interval=3,
        log_interval=1,
    )
    identity = str(step_plan["train_sample_identity"])
    dedicated_noise = torch.rand((1, 1), generator=train_rng)
    python_noise = random.random()
    cpu_noise = torch.rand((1, 1))
    sample_value = float(step_plan["train_sample_position"] + 1)
    x = dedicated_noise + cpu_noise + sample_value + python_noise
    target = torch.tensor(float(step) * 0.125)
    loss = (model(x).reshape(()) - target).square()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    record = train_m3.m5_metrics_step_record(
        step_plan=step_plan,
        elapsed_ms=float(step) * 10.0,
        timing_warmup_steps=timing_warmup_steps,
    )
    record.update(
        {
            "train_sample_identity": identity,
            "train_sample_position": int(step_plan["train_sample_position"]),
            "train_cycle_index": int(step_plan["train_cycle_index"]),
            "should_validate": bool(step_plan["should_validate"]),
            "should_checkpoint": bool(step_plan["should_checkpoint"]),
            "loss": float(loss.detach().item()),
        }
    )
    return record


def _save_checkpoint(
    *,
    output_dir: Path,
    model: TinyResumeModel,
    optimizer: torch.optim.AdamW,
    step: int,
    train_rng: torch.Generator,
    probe: m3.M3Probe,
    metadata: dict[str, Any],
    resolved_config: dict[str, Any],
    reference_path: Path,
    reference_sha: str,
) -> Path:
    probe_forward = _probe_forward(model, probe)
    return train_m3.save_checkpoint_at_step(
        output_dir=output_dir,
        generator=model,
        optimizer=optimizer,
        step=step,
        train_rng=train_rng,
        probe=probe,
        probe_summary={"probe_losses": m3.loss_dict_to_floats(probe_forward.losses)},
        probe_outputs=probe_forward.outputs,
        sample_metadata=metadata,
        resolved_config=resolved_config,
        git_sha=TEST_GIT_SHA,
        reference_checkpoint_path=reference_path,
        reference_checkpoint_sha256=reference_sha,
        train_seed=101,
        probe_seed=202,
        prompt_embedding=_prompt_embedding(),
        device=torch.device("cpu"),
    )


def _assert_nested_equal(left: Any, right: Any) -> None:
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        assert left.dtype == right.dtype
        assert left.shape == right.shape
        assert torch.equal(left, right)
        return
    if isinstance(left, dict):
        assert isinstance(right, dict)
        assert set(left.keys()) == set(right.keys())
        for key in left:
            _assert_nested_equal(left[key], right[key])
        return
    if isinstance(left, (list, tuple)):
        assert isinstance(right, type(left))
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right):
            _assert_nested_equal(left_item, right_item)
        return
    assert left == right


def _restore_from_checkpoint(
    *,
    parent_path: Path,
    tmp_path: Path,
    plan: dict[str, Any],
    resolved_config: dict[str, Any],
    reference_path: Path,
    reference_sha: str,
) -> tuple[
    TinyResumeModel,
    torch.optim.AdamW,
    torch.Generator,
    m3.M3Probe,
    dict[str, Any],
    dict[str, Any],
]:
    parent_payload, parent_sha = train_m3.load_parent_resume_checkpoint(parent_path)
    model, optimizer = _model_and_optimizer(seed=999)
    current_fields = train_m3.current_m5_resume_run_fields(
        resolved_config=resolved_config,
        reference_checkpoint={"path": reference_path, "sha256": reference_sha},
        selected_sample_metadata=_sample_metadata(tmp_path, plan),
        optimizer=optimizer,
        current_git_sha=TEST_GIT_SHA,
        sample_plan=plan,
    )
    report = train_m3.build_and_validate_m5_resume_report(
        parent_payload=parent_payload,
        parent_checkpoint_path=parent_path,
        parent_checkpoint_sha256=parent_sha,
        current_run_fields=current_fields,
        target_global_step=6,
        sample_plan=plan,
        output_dir=tmp_path / "resume",
        target_validation_steps=(0, 3, 6),
        target_checkpoint_steps=(0, 3, 6),
        expected_cuda_device_count=None,
    )
    train_m3.require_m5_resume_devices(
        report,
        train_device=torch.device("cpu"),
        probe_device=torch.device("cpu"),
    )
    train_m3.strict_load_m5_generator_state(model, parent_payload["generator"])
    optimizer.load_state_dict(parent_payload["optimizer"])
    train_m3.move_loaded_optimizer_state_to_device(optimizer, device="cpu")
    rng_states = m5.extract_resume_rng_states(parent_payload)
    train_rng = m5.restore_torch_generator_from_state(
        rng_states["train_generator_state"],
        device="cpu",
    )
    probe, restored_prompt_embedding = train_m3.restore_m5_probe_from_checkpoint(
        parent_payload=parent_payload,
        selected_state=_selected_state(),
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    restored_probe = _probe_forward(model, probe)
    train_m3.require_restored_probe_matches_checkpoint(
        parent_payload=parent_payload,
        restored_prompt_embedding=restored_prompt_embedding,
        probe_forward=restored_probe,
    )
    m5.restore_global_rng_states(rng_states)
    return model, optimizer, train_rng, probe, restored_prompt_embedding, report


def test_cpu_split_run_resume_matches_continuous_run(tmp_path: Path) -> None:
    plan = _sample_plan(tmp_path)
    resolved_config = _resolved_config(tmp_path, plan)
    metadata = _sample_metadata(tmp_path, plan)
    reference_path = tmp_path / "reference.pt"
    reference_path.write_bytes(b"reference")
    reference_sha = m3.file_sha256(reference_path)

    random.seed(700)
    torch.manual_seed(800)
    continuous_model, continuous_optimizer = _model_and_optimizer(seed=100)
    continuous_train_rng = make_cpu_generator(101)
    continuous_probe = _probe()
    continuous_metrics = [
        _train_step(
            model=continuous_model,
            optimizer=continuous_optimizer,
            train_rng=continuous_train_rng,
            plan=plan,
            step=step,
        )
        for step in range(1, 7)
    ]
    continuous_python_state = copy.deepcopy(random.getstate())
    continuous_torch_state = torch.random.get_rng_state()
    continuous_probe_forward = _probe_forward(continuous_model, continuous_probe)

    random.seed(700)
    torch.manual_seed(800)
    split_model, split_optimizer = _model_and_optimizer(seed=100)
    split_train_rng = make_cpu_generator(101)
    split_probe = _probe()
    first_metrics = [
        _train_step(
            model=split_model,
            optimizer=split_optimizer,
            train_rng=split_train_rng,
            plan=plan,
            step=step,
        )
        for step in range(1, 4)
    ]
    parent_path = _save_checkpoint(
        output_dir=tmp_path / "part1",
        model=split_model,
        optimizer=split_optimizer,
        step=3,
        train_rng=split_train_rng,
        probe=split_probe,
        metadata=metadata,
        resolved_config=resolved_config,
        reference_path=reference_path,
        reference_sha=reference_sha,
    )
    resumed_model, resumed_optimizer, resumed_rng, resumed_probe, _, report = (
        _restore_from_checkpoint(
            parent_path=parent_path,
            tmp_path=tmp_path,
            plan=plan,
            resolved_config=resolved_config,
            reference_path=reference_path,
            reference_sha=reference_sha,
        )
    )
    second_metrics = [
        _train_step(
            model=resumed_model,
            optimizer=resumed_optimizer,
            train_rng=resumed_rng,
            plan=plan,
            step=step,
            timing_warmup_steps=5,
        )
        for step in range(4, 7)
    ]
    final_checkpoint = _save_checkpoint(
        output_dir=tmp_path / "resume",
        model=resumed_model,
        optimizer=resumed_optimizer,
        step=6,
        train_rng=resumed_rng,
        probe=resumed_probe,
        metadata=metadata,
        resolved_config=resolved_config,
        reference_path=reference_path,
        reference_sha=reference_sha,
    )
    resumed_python_state = copy.deepcopy(random.getstate())
    resumed_torch_state = torch.random.get_rng_state()
    resumed_probe_forward = _probe_forward(resumed_model, resumed_probe)

    _assert_nested_equal(continuous_model.state_dict(), resumed_model.state_dict())
    _assert_nested_equal(
        continuous_optimizer.state_dict(),
        resumed_optimizer.state_dict(),
    )
    assert torch.equal(continuous_train_rng.get_state(), resumed_rng.get_state())
    assert torch.equal(continuous_probe.rng_state, resumed_probe.rng_state)
    assert continuous_python_state == resumed_python_state
    assert torch.equal(continuous_torch_state, resumed_torch_state)
    _assert_nested_equal(
        m3.serialize_noisy_batch(continuous_probe.noisy_batch),
        m3.serialize_noisy_batch(resumed_probe.noisy_batch),
    )
    _assert_nested_equal(
        continuous_probe_forward.outputs,
        resumed_probe_forward.outputs,
    )
    assert [record["step"] for record in first_metrics] == [1, 2, 3]
    assert [record["step"] for record in second_metrics] == [4, 5, 6]
    assert [record["step"] for record in first_metrics + second_metrics] == list(
        range(1, 7)
    )
    assert [record["should_validate"] for record in first_metrics] == [False, False, True]
    assert [record["should_checkpoint"] for record in first_metrics] == [False, False, True]
    assert [record["should_validate"] for record in second_metrics] == [
        False,
        False,
        True,
    ]
    assert [record["should_checkpoint"] for record in second_metrics] == [
        False,
        False,
        True,
    ]
    resume_timing_summary = train_m3.summarize_m5_step_timing_records(
        [record["timing"] for record in second_metrics]
    )
    assert resume_timing_summary["executed_global_steps"] == [4, 5, 6]
    assert resume_timing_summary["measured_global_steps"] == [6]
    assert final_checkpoint.name == "checkpoint_step000006.pt"
    assert [record["train_sample_identity"] for record in second_metrics] == [
        record["train_sample_identity"] for record in continuous_metrics[3:]
    ]
    assert [record["loss"] for record in second_metrics] == [
        record["loss"] for record in continuous_metrics[3:]
    ]
    assert int(report["resumed_global_step"]) == 3
    assert int(report["first_resumed_step"]) == 4
    assert report["first_resumed_sample_identity"] == plan["train_sample_identities"][0]
    assert report["target_global_step"] == 6
    assert report["next_sample_after_target_identity"] == (
        m5.next_sample_after_target_identity(plan, 6)
    )


def test_resume_precomputes_conditionals_before_global_rng_restore(
    tmp_path: Path,
) -> None:
    plan = _sample_plan(tmp_path)
    encoder = CountingTextEncoder()
    cache: dict[str, dict[str, Any]] = {}
    restored_prompt_embedding = _prompt_embedding()

    report = train_m3.precompute_m5_resume_conditionals(
        text_encoder=encoder,
        sample_plan=plan,
        train_samples_by_identity=_sample_objects_by_identity(plan, "train"),
        validation_samples_by_identity=_sample_objects_by_identity(
            plan,
            "validation",
        ),
        fixed_decode_prompt_embedding=restored_prompt_embedding,
        conditional_cache=cache,
    )
    calls_at_global_rng_restore = encoder.calls
    first_resumed_plan = train_m3.m5_training_step_orchestration(
        global_step=4,
        target_global_step=6,
        sample_plan=plan,
        validation_steps=(0, 3, 6),
        checkpoint_steps=(0, 3, 6),
        checkpoint_interval=3,
        log_interval=1,
    )
    first_resumed_identity = str(first_resumed_plan["train_sample_identity"])
    train_m3.conditional_dict_for_identity(
        text_encoder=encoder,
        sample=_sample_objects_by_identity(plan, "train")[first_resumed_identity],
        identity=first_resumed_identity,
        cache=cache,
    )

    assert cache[str(plan["fixed_decode_validation_identity"])] is restored_prompt_embedding
    assert sorted(report["cached_identities"]) == sorted(
        [
            *plan["train_sample_identities"],
            *plan["validation_sample_identities"],
        ]
    )
    assert calls_at_global_rng_restore == 4
    assert encoder.calls - calls_at_global_rng_restore == 0


def test_m5_timing_uses_absolute_global_step_warmup() -> None:
    fresh_steps = list(
        train_m3.m5_absolute_training_steps(
            resumed_global_step=None,
            target_global_step=6,
        )
    )
    fresh_summary = train_m3.summarize_m5_step_timing_records(
        [
            train_m3.m5_timing_record(
                global_step=step,
                elapsed_ms=float(step),
                timing_warmup_steps=2,
            )
            for step in fresh_steps
        ]
    )
    resume_from_3_steps = list(
        train_m3.m5_absolute_training_steps(
            resumed_global_step=3,
            target_global_step=6,
        )
    )
    resume_from_3_summary = train_m3.summarize_m5_step_timing_records(
        [
            train_m3.m5_timing_record(
                global_step=step,
                elapsed_ms=float(step),
                timing_warmup_steps=5,
            )
            for step in resume_from_3_steps
        ]
    )
    resume_from_6_steps = list(
        train_m3.m5_absolute_training_steps(
            resumed_global_step=6,
            target_global_step=8,
        )
    )
    resume_from_6_summary = train_m3.summarize_m5_step_timing_records(
        [
            train_m3.m5_timing_record(
                global_step=step,
                elapsed_ms=float(step),
                timing_warmup_steps=5,
            )
            for step in resume_from_6_steps
        ]
    )

    assert fresh_summary["executed_global_steps"] == [1, 2, 3, 4, 5, 6]
    assert fresh_summary["executed_step_count"] == 6
    assert fresh_summary["measured_global_steps"] == [3, 4, 5, 6]
    assert resume_from_3_summary["executed_global_steps"] == [4, 5, 6]
    assert resume_from_3_summary["executed_step_count"] == 3
    assert resume_from_3_summary["measured_global_steps"] == [6]
    assert resume_from_6_summary["executed_global_steps"] == [7, 8]
    assert resume_from_6_summary["executed_step_count"] == 2
    assert resume_from_6_summary["measured_global_steps"] == [7, 8]


def _valid_parent_payload_and_context(
    tmp_path: Path,
    *,
    mode: str = "joint",
    include_rng: bool = True,
) -> tuple[Path, dict[str, Any], dict[str, Any], dict[str, Any], Path, str]:
    plan = _sample_plan(tmp_path)
    resolved_config = _resolved_config(tmp_path, plan, mode=mode)
    metadata = _sample_metadata(tmp_path, plan)
    reference_path = tmp_path / "reference.pt"
    reference_path.write_bytes(b"reference")
    reference_sha = m3.file_sha256(reference_path)
    model, optimizer = _model_and_optimizer(seed=100)
    train_rng = make_cpu_generator(101)
    probe = _probe()
    path = _save_checkpoint(
        output_dir=tmp_path / "parent",
        model=model,
        optimizer=optimizer,
        step=3,
        train_rng=train_rng,
        probe=probe,
        metadata=metadata,
        resolved_config=resolved_config,
        reference_path=reference_path,
        reference_sha=reference_sha,
    )
    payload = m3.load_m3_checkpoint(path)
    if not include_rng:
        del payload[m5.M5_RNG_EXTENSION_FIELD]
        m3.save_m3_checkpoint(payload, path)
    return path, payload, plan, resolved_config, reference_path, reference_sha


def _current_fields_for(
    *,
    tmp_path: Path,
    plan: dict[str, Any],
    resolved_config: dict[str, Any],
    reference_path: Path,
    reference_sha: str,
    git_sha: str = TEST_GIT_SHA,
) -> dict[str, Any]:
    _, optimizer = _model_and_optimizer(seed=999)
    return train_m3.current_m5_resume_run_fields(
        resolved_config=resolved_config,
        reference_checkpoint={"path": reference_path, "sha256": reference_sha},
        selected_sample_metadata=_sample_metadata(tmp_path, plan),
        optimizer=optimizer,
        current_git_sha=git_sha,
        sample_plan=plan,
    )


def test_resume_rejects_non_joint_parent_mode(tmp_path: Path) -> None:
    path, payload, plan, config, reference_path, reference_sha = (
        _valid_parent_payload_and_context(tmp_path, mode="frozen")
    )
    with pytest.raises(m5.ResumeContractError, match="m3.mode"):
        train_m3.build_and_validate_m5_resume_report(
            parent_payload=payload,
            parent_checkpoint_path=path,
            parent_checkpoint_sha256=m3.file_sha256(path),
            current_run_fields=_current_fields_for(
                tmp_path=tmp_path,
                plan=plan,
                resolved_config=config,
                reference_path=reference_path,
                reference_sha=reference_sha,
            ),
            target_global_step=6,
            sample_plan=plan,
            output_dir=tmp_path / "resume",
            target_validation_steps=(0, 3, 6),
            target_checkpoint_steps=(0, 3, 6),
            expected_cuda_device_count=None,
        )


def test_resume_rejects_missing_parent_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        train_m3.load_parent_resume_checkpoint(tmp_path / "missing.pt")


def test_resume_rejects_parent_sha_mismatch(tmp_path: Path) -> None:
    path, payload, plan, config, reference_path, reference_sha = (
        _valid_parent_payload_and_context(tmp_path)
    )
    with pytest.raises(m5.ResumeContractError, match="parent_checkpoint_sha256"):
        train_m3.build_and_validate_m5_resume_report(
            parent_payload=payload,
            parent_checkpoint_path=path,
            parent_checkpoint_sha256="f" * 64,
            current_run_fields=_current_fields_for(
                tmp_path=tmp_path,
                plan=plan,
                resolved_config=config,
                reference_path=reference_path,
                reference_sha=reference_sha,
            ),
            target_global_step=6,
            sample_plan=plan,
            output_dir=tmp_path / "resume",
            target_validation_steps=(0, 3, 6),
            target_checkpoint_steps=(0, 3, 6),
            expected_cuda_device_count=None,
        )


def test_resume_rejects_current_git_mismatch(tmp_path: Path) -> None:
    path, payload, plan, config, reference_path, reference_sha = (
        _valid_parent_payload_and_context(tmp_path)
    )
    with pytest.raises(m5.ResumeContractError, match="git_sha"):
        train_m3.build_and_validate_m5_resume_report(
            parent_payload=payload,
            parent_checkpoint_path=path,
            parent_checkpoint_sha256=m3.file_sha256(path),
            current_run_fields=_current_fields_for(
                tmp_path=tmp_path,
                plan=plan,
                resolved_config=config,
                reference_path=reference_path,
                reference_sha=reference_sha,
                git_sha=BAD_GIT_SHA,
            ),
            target_global_step=6,
            sample_plan=plan,
            output_dir=tmp_path / "resume",
            target_validation_steps=(0, 3, 6),
            target_checkpoint_steps=(0, 3, 6),
            expected_cuda_device_count=None,
        )


def test_resume_rejects_config_mismatch(tmp_path: Path) -> None:
    path, payload, plan, config, reference_path, reference_sha = (
        _valid_parent_payload_and_context(tmp_path)
    )
    current_config = copy.deepcopy(config)
    current_config["m3"]["log_interval"] = 2
    with pytest.raises(m5.ResumeContractError, match="log_interval"):
        train_m3.build_and_validate_m5_resume_report(
            parent_payload=payload,
            parent_checkpoint_path=path,
            parent_checkpoint_sha256=m3.file_sha256(path),
            current_run_fields=_current_fields_for(
                tmp_path=tmp_path,
                plan=plan,
                resolved_config=current_config,
                reference_path=reference_path,
                reference_sha=reference_sha,
            ),
            target_global_step=6,
            sample_plan=plan,
            output_dir=tmp_path / "resume",
            target_validation_steps=(0, 3, 6),
            target_checkpoint_steps=(0, 3, 6),
            expected_cuda_device_count=None,
        )


def test_resume_rejects_sample_plan_mismatch(tmp_path: Path) -> None:
    path, payload, plan, config, reference_path, reference_sha = (
        _valid_parent_payload_and_context(tmp_path)
    )
    bad_plan = copy.deepcopy(plan)
    bad_plan["train_sample_identities"] = list(
        reversed(bad_plan["train_sample_identities"])
    )
    with pytest.raises(m5.ResumeContractError, match="m4.sample_plan_sha256"):
        train_m3.build_and_validate_m5_resume_report(
            parent_payload=payload,
            parent_checkpoint_path=path,
            parent_checkpoint_sha256=m3.file_sha256(path),
            current_run_fields=_current_fields_for(
                tmp_path=tmp_path,
                plan=plan,
                resolved_config=config,
                reference_path=reference_path,
                reference_sha=reference_sha,
            ),
            target_global_step=6,
            sample_plan=bad_plan,
            output_dir=tmp_path / "resume",
            target_validation_steps=(0, 3, 6),
            target_checkpoint_steps=(0, 3, 6),
            expected_cuda_device_count=None,
        )


def test_resume_rejects_optimizer_group_mismatch(tmp_path: Path) -> None:
    plan = _sample_plan(tmp_path)
    config = _resolved_config(tmp_path, plan)
    reference_path = tmp_path / "reference.pt"
    reference_path.write_bytes(b"reference")
    reference_sha = m3.file_sha256(reference_path)
    _, optimizer = _model_and_optimizer(seed=999)
    state = optimizer.state_dict()
    state["param_groups"][0]["name"] = "wrong"
    with pytest.raises(m5.ResumeContractError, match="param_groups"):
        m5.build_resume_run_fields(
            resolved_config=config,
            reference_checkpoint={"path": reference_path, "sha256": reference_sha},
            git_sha=TEST_GIT_SHA,
            optimizer_state_dict=state,
            optimizer_group_lrs=m3.optimizer_group_lr_summary(optimizer),
            selected_sample_metadata=_sample_metadata(tmp_path, plan),
            sample_plan=plan,
        )


def test_resume_rejects_checkpoint_missing_m5_rng_state(tmp_path: Path) -> None:
    path, payload, plan, config, reference_path, reference_sha = (
        _valid_parent_payload_and_context(tmp_path, include_rng=False)
    )
    payload = m3.load_m3_checkpoint(path)
    with pytest.raises(m5.ResumeContractError, match="missing_global_rng_fields"):
        train_m3.build_and_validate_m5_resume_report(
            parent_payload=payload,
            parent_checkpoint_path=path,
            parent_checkpoint_sha256=m3.file_sha256(path),
            current_run_fields=_current_fields_for(
                tmp_path=tmp_path,
                plan=plan,
                resolved_config=config,
                reference_path=reference_path,
                reference_sha=reference_sha,
            ),
            target_global_step=6,
            sample_plan=plan,
            output_dir=tmp_path / "resume",
            target_validation_steps=(0, 3, 6),
            target_checkpoint_steps=(0, 3, 6),
            expected_cuda_device_count=None,
        )


def test_resume_rejects_empty_cuda_state_when_cuda_expected(tmp_path: Path) -> None:
    path, payload, plan, config, reference_path, reference_sha = (
        _valid_parent_payload_and_context(tmp_path)
    )
    with pytest.raises(m5.ResumeContractError, match="torch_cuda_rng_states"):
        train_m3.build_and_validate_m5_resume_report(
            parent_payload=payload,
            parent_checkpoint_path=path,
            parent_checkpoint_sha256=m3.file_sha256(path),
            current_run_fields=_current_fields_for(
                tmp_path=tmp_path,
                plan=plan,
                resolved_config=config,
                reference_path=reference_path,
                reference_sha=reference_sha,
            ),
            target_global_step=6,
            sample_plan=plan,
            output_dir=tmp_path / "resume",
            target_validation_steps=(0, 3, 6),
            target_checkpoint_steps=(0, 3, 6),
            expected_cuda_device_count=1,
        )


def test_resume_rejects_generator_strict_load_mismatch() -> None:
    model, _ = _model_and_optimizer(seed=100)
    incompatible = nn.Linear(2, 1)
    with pytest.raises(RuntimeError, match="strict load failed"):
        train_m3.strict_load_m5_generator_state(incompatible, model.state_dict())


def test_resume_rejects_target_step_not_greater(tmp_path: Path) -> None:
    path, payload, plan, config, reference_path, reference_sha = (
        _valid_parent_payload_and_context(tmp_path)
    )
    with pytest.raises(m5.ResumeContractError, match="target_global_step"):
        train_m3.build_and_validate_m5_resume_report(
            parent_payload=payload,
            parent_checkpoint_path=path,
            parent_checkpoint_sha256=m3.file_sha256(path),
            current_run_fields=_current_fields_for(
                tmp_path=tmp_path,
                plan=plan,
                resolved_config=config,
                reference_path=reference_path,
                reference_sha=reference_sha,
            ),
            target_global_step=3,
            sample_plan=plan,
            output_dir=tmp_path / "resume",
            target_validation_steps=(0, 3),
            target_checkpoint_steps=(0, 3),
            expected_cuda_device_count=None,
        )


def test_resume_rejects_nonempty_output_directory(tmp_path: Path) -> None:
    output_dir = tmp_path / "resume"
    output_dir.mkdir()
    (output_dir / "existing.txt").write_text("x", encoding="utf-8")
    with pytest.raises(FileExistsError):
        train_m3.require_output_dir_empty_for_resume(output_dir)


def test_resume_rejects_restored_probe_mismatch(tmp_path: Path) -> None:
    path, payload, plan, config, reference_path, reference_sha = (
        _valid_parent_payload_and_context(tmp_path)
    )
    model, _, _, probe, restored_prompt_embedding, _ = _restore_from_checkpoint(
        parent_path=path,
        tmp_path=tmp_path,
        plan=plan,
        resolved_config=config,
        reference_path=reference_path,
        reference_sha=reference_sha,
    )
    bad_payload = copy.deepcopy(payload)
    bad_payload["probe_outputs"]["main_flow_pred"] = (
        bad_payload["probe_outputs"]["main_flow_pred"] + 1.0
    )
    with pytest.raises(RuntimeError, match="probe outputs differ"):
        train_m3.require_restored_probe_matches_checkpoint(
            parent_payload=bad_payload,
            restored_prompt_embedding=restored_prompt_embedding,
            probe_forward=_probe_forward(model, probe),
        )


def test_resume_rejects_restored_prompt_embedding_mismatch(tmp_path: Path) -> None:
    path, payload, plan, config, reference_path, reference_sha = (
        _valid_parent_payload_and_context(tmp_path)
    )
    model, _, _, probe, restored_prompt_embedding, _ = _restore_from_checkpoint(
        parent_path=path,
        tmp_path=tmp_path,
        plan=plan,
        resolved_config=config,
        reference_path=reference_path,
        reference_sha=reference_sha,
    )
    bad_prompt_embedding = copy.deepcopy(restored_prompt_embedding)
    bad_prompt_embedding["prompt_embeds"] = (
        bad_prompt_embedding["prompt_embeds"] + 1.0
    )
    with pytest.raises(RuntimeError, match="prompt embedding differs"):
        train_m3.require_restored_probe_matches_checkpoint(
            parent_payload=payload,
            restored_prompt_embedding=bad_prompt_embedding,
            probe_forward=_probe_forward(model, probe),
        )
