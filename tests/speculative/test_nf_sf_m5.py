from __future__ import annotations

import copy
import json
import random
import uuid
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import utils.nf_sf_m3 as m3
import utils.nf_sf_m4 as m4
import utils.nf_sf_m5 as m5
from utils.nf_sf_m3 import make_m3_checkpoint_payload, make_m3_probe
from utils.nf_sf_tensors import make_cpu_generator
from utils.scheduler import FlowMatchScheduler

TEST_GIT_SHA = "0123456789abcdef0123456789abcdef01234567"
REFERENCE_SHA = "a" * 64
MANIFEST_SHA = "b" * 64
PARENT_SHA = "c" * 64
PROMPT_SHA_A = "d" * 64
PROMPT_SHA_B = "e" * 64


@pytest.fixture
def work_dir() -> Path:
    return Path("m5_virtual_workspace") / uuid.uuid4().hex


def _scheduler(shift: float) -> FlowMatchScheduler:
    scheduler = FlowMatchScheduler(shift=shift, sigma_min=0.0, extra_one_step=True)
    scheduler.set_timesteps(1000, training=True)
    return scheduler


def _latent(num_frames: int = 15) -> torch.Tensor:
    return torch.arange(num_frames, dtype=torch.float32).reshape(
        1,
        num_frames,
        1,
        1,
        1,
    )


def _probe_outputs_for(probe: m3.M3Probe) -> dict[str, torch.Tensor]:
    return {
        "main_flow_pred": torch.zeros_like(probe.noisy_batch.target_flow_main),
        "mcp_depth1_flow_pred": torch.zeros_like(
            probe.noisy_batch.target_flow_depths[0]
        ),
        "mcp_depth2_flow_pred": torch.zeros_like(
            probe.noisy_batch.target_flow_depths[1]
        ),
        "mcp_depth3_flow_pred": torch.zeros_like(
            probe.noisy_batch.target_flow_depths[2]
        ),
    }


def _identity(split: str, split_index: int, sample_index: int) -> str:
    prompt_sha = PROMPT_SHA_A if split == "train" else PROMPT_SHA_B
    return (
        f"sample_index={sample_index}|sample_id={split}-{split_index:03d}|"
        f"split={split}|split_index={split_index}|prompt_sha256={prompt_sha}"
    )


def _entry(split: str, split_index: int, sample_index: int) -> dict[str, Any]:
    prompt_sha = PROMPT_SHA_A if split == "train" else PROMPT_SHA_B
    return {
        "identity": _identity(split, split_index, sample_index),
        "sample_index": sample_index,
        "sample_id": f"{split}-{split_index:03d}",
        "split": split,
        "split_index": split_index,
        "prompt_sha256": prompt_sha,
    }


def _sample_plan(work_dir: Path, *, train_count: int = 3) -> dict[str, Any]:
    train_entries = [
        _entry("train", index, index)
        for index in range(train_count)
    ]
    validation_entries = [
        _entry("validation", index, 100 + index)
        for index in range(2)
    ]
    plan = {
        "schema": m4.M4_SAMPLE_PLAN_SCHEMA,
        "manifest_path": str((work_dir / "manifest.json").resolve()),
        "manifest_sha256": MANIFEST_SHA,
        "dataset_root": None,
        "ordering_rule": m4.M4_SAMPLE_ORDERING_RULE,
        "train_subset_size": len(train_entries),
        "validation_subset_size": len(validation_entries),
        "train_sample_identities": [
            entry["identity"] for entry in train_entries
        ],
        "validation_sample_identities": [
            entry["identity"] for entry in validation_entries
        ],
        "fixed_decode_validation_identity": validation_entries[0]["identity"],
        "samples": {
            "train": train_entries,
            "validation": validation_entries,
        },
    }
    plan["sample_plan_sha256"] = m4.m4_sample_plan_sha256(plan)
    m4.validate_m4_sample_plan(plan)
    return plan


def _resolved_config(work_dir: Path, plan: Mapping[str, Any]) -> dict[str, Any]:
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
            "mode": "joint",
            "manifest": str((work_dir / "manifest.json").resolve()),
            "dataset_root": None,
            "sample_index": 100,
            "sample_id": "validation-000",
            "split": "validation",
            "split_index": 0,
            "train_seed": 55,
            "probe_seed": 44,
            "optimizer_steps": 10,
            "timing_warmup_steps": 1,
            "log_interval": 10,
            "checkpoint_interval": 10,
            "backbone_lr": 1.0e-6,
            "patch_embedding_lr": 2.0e-6,
            "mcp_lr": 3.0e-5,
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
            "dtype": "bf16",
            "device": "cuda:0",
        },
        "m4": {
            "enabled": True,
            "sample_plan_path": str((work_dir / "m4_sample_plan.json").resolve()),
            "sample_plan_sha256": str(plan["sample_plan_sha256"]),
            "train_sample_identities": list(plan["train_sample_identities"]),
            "validation_sample_identities": list(
                plan["validation_sample_identities"]
            ),
            "train_subset_size": int(plan["train_subset_size"]),
            "validation_subset_size": int(plan["validation_subset_size"]),
            "validation_seed": 66,
            "validation_steps": [0, 5, 10],
            "checkpoint_steps": [0, 10],
            "fixed_decode_validation_identity": str(
                plan["fixed_decode_validation_identity"]
            ),
            "sample_ordering_rule": str(plan["ordering_rule"]),
            "ordering_rule": str(plan["ordering_rule"]),
        },
    }


def _make_grouped_model_and_optimizer() -> tuple[nn.ModuleDict, torch.optim.AdamW]:
    model = nn.ModuleDict(
        {
            group_name: nn.Linear(1, 1)
            for group_name in m3.M3_PARAMETER_GROUP_NAMES
        }
    )
    lr_by_name = {
        "backbone": 1.0e-6,
        "patch_embedding": 2.0e-6,
    }
    optimizer_groups = []
    for group_name in m3.M3_PARAMETER_GROUP_NAMES:
        lr = lr_by_name.get(group_name, 3.0e-5)
        optimizer_groups.append(
            {
                "name": group_name,
                "params": list(model[group_name].parameters()),
                "lr": lr,
            }
        )
    optimizer = torch.optim.AdamW(
        optimizer_groups,
        betas=(0.0, 0.999),
        eps=1.0e-8,
        weight_decay=0.01,
    )
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return model, optimizer


def _checkpoint_payload(
    work_dir: Path,
    *,
    global_step: int = 4,
) -> tuple[dict[str, Any], dict[str, Any]]:
    plan = _sample_plan(work_dir)
    state = m3.select_m3_selected_state(_latent())
    probe = make_m3_probe(
        state,
        scheduler_main=_scheduler(5.0),
        scheduler_mcp=_scheduler(10.0),
        seed=44,
    )
    model, optimizer = _make_grouped_model_and_optimizer()
    payload = make_m3_checkpoint_payload(
        generator=model,
        optimizer=optimizer,
        global_step=global_step,
        train_rng=make_cpu_generator(55),
        probe=probe,
        probe_summary={
            "probe_losses": {
                "main_loss": 1.0,
                "mcp_depth1_loss": 2.0,
                "mcp_depth2_loss": 3.0,
                "mcp_depth3_loss": 4.0,
                "total_loss": 5.0,
            }
        },
        probe_outputs=_probe_outputs_for(probe),
        selected_sample_metadata={
            "sample_index": 100,
            "sample_id": "validation-000",
            "split": "validation",
            "split_index": 0,
            "prompt": "validation prompt",
            "prompt_sha256": PROMPT_SHA_B,
            "manifest_path": str((work_dir / "manifest.json").resolve()),
            "manifest_sha256": MANIFEST_SHA,
            "chunk_frames": 3,
            "target_latent": {"sha256": "target-sha"},
        },
        resolved_config=_resolved_config(work_dir, plan),
        git_sha=TEST_GIT_SHA,
        reference_checkpoint_path=work_dir / "self_forcing_dmd.pt",
        reference_checkpoint_sha256=REFERENCE_SHA,
        train_seed=55,
        probe_seed=44,
        prompt_embedding={"prompt_embeds": torch.zeros((1, 1, 1))},
    )
    return payload, plan


def _contract_and_current(
    work_dir: Path,
    *,
    global_step: int = 4,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    payload, plan = _checkpoint_payload(work_dir, global_step=global_step)
    contract = m5.build_resume_contract(
        payload,
        parent_checkpoint_path=work_dir / "checkpoint_step000004.pt",
        parent_checkpoint_sha256=PARENT_SHA,
        sample_plan=plan,
    )
    return contract, _current_run_fields(work_dir, plan), plan


def _current_run_fields(
    work_dir: Path,
    plan: Mapping[str, Any],
    *,
    git_sha: str = TEST_GIT_SHA,
    config_mutator: Callable[[dict[str, Any]], None] | None = None,
    optimizer_state_mutator: Callable[[dict[str, Any]], None] | None = None,
    optimizer_summary_mutator: Callable[[list[dict[str, Any]]], None] | None = None,
) -> dict[str, Any]:
    current_config = _resolved_config(work_dir, plan)
    if config_mutator is not None:
        config_mutator(current_config)
    _, optimizer = _make_grouped_model_and_optimizer()
    optimizer_state = optimizer.state_dict()
    optimizer_summary = m3.optimizer_group_lr_summary(optimizer)
    if optimizer_state_mutator is not None:
        optimizer_state_mutator(optimizer_state)
    if optimizer_summary_mutator is not None:
        optimizer_summary_mutator(optimizer_summary)
    return m5.build_resume_run_fields(
        resolved_config=current_config,
        reference_checkpoint={
            "path": work_dir / "self_forcing_dmd.pt",
            "sha256": REFERENCE_SHA,
        },
        git_sha=git_sha,
        optimizer_state_dict=optimizer_state,
        optimizer_group_lrs=optimizer_summary,
        sample_plan=plan,
        manifest_sha256=MANIFEST_SHA,
    )


def _validate_pass(
    contract: Mapping[str, Any],
    current: Mapping[str, Any],
    plan: Mapping[str, Any],
    *,
    target_global_step: int = 7,
) -> dict[str, Any]:
    return m5.validate_resume_contract(
        contract,
        current,
        target_global_step=target_global_step,
        sample_plan=plan,
        output_dir=Path("resume-output"),
        target_validation_steps=(0, target_global_step),
        target_checkpoint_steps=(0, target_global_step),
        allow_legacy_missing_global_rng=True,
        global_rng_independence_evidence=True,
    )


def _expect_field_rejected(
    work_dir: Path,
    mutator: Callable[[dict[str, Any]], None],
    field_path: str,
) -> None:
    contract, current, plan = _contract_and_current(work_dir)
    mutator(current)
    with pytest.raises(m5.ResumeContractError) as exc_info:
        _validate_pass(contract, current, plan)
    message = str(exc_info.value)
    assert field_path in message
    assert "expected=" in message
    assert "actual=" in message


def _payload_fingerprint(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        tensor = value.reshape(1) if value.dim() == 0 else value
        return {
            "tensor": True,
            "dtype": str(value.dtype),
            "shape": [int(dim) for dim in value.shape],
            "sha256": m3.tensor_sha256(tensor),
        }
    if isinstance(value, Mapping):
        return {
            str(key): _payload_fingerprint(item)
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return tuple(_payload_fingerprint(item) for item in value)
    if isinstance(value, list):
        return [_payload_fingerprint(item) for item in value]
    return copy.deepcopy(value)


def test_resume_contract_passes_when_locked_fields_match(work_dir: Path) -> None:
    contract, current, plan = _contract_and_current(work_dir)
    report = _validate_pass(contract, current, plan)

    assert report["schema"] == m5.M5_RESUME_SCHEMA
    assert report["status"] == "PASS"
    assert report["parent_checkpoint_sha256"] == PARENT_SHA
    assert report["source_verified"] is False
    assert report["resumed_global_step"] == 4
    assert report["target_global_step"] == 7
    assert report["first_resumed_step"] == 5
    assert report["first_resumed_sample_identity"] == plan[
        "train_sample_identities"
    ][1]
    assert report["next_sample_after_target_identity"] == plan[
        "train_sample_identities"
    ][1]
    assert report["optimizer_restore"]["status"] == "ready"
    assert report["lr_scheduler"] is None
    assert report["lr_scheduler_restore"] == "not_applicable"
    assert report["incompatibilities"] == []


def test_resume_rejects_mode_difference(work_dir: Path) -> None:
    _expect_field_rejected(
        work_dir,
        lambda current: current.__setitem__("m3.mode", "frozen"),
        "m3.mode",
    )


def test_resume_rejects_sample_plan_sha_difference(work_dir: Path) -> None:
    _expect_field_rejected(
        work_dir,
        lambda current: current.__setitem__("m4.sample_plan_sha256", "0" * 64),
        "m4.sample_plan_sha256",
    )


def test_resume_rejects_git_sha_difference(work_dir: Path) -> None:
    contract, _, plan = _contract_and_current(work_dir)
    current = _current_run_fields(work_dir, plan, git_sha="1" * 40)

    with pytest.raises(m5.ResumeContractError, match="git_sha"):
        _validate_pass(contract, current, plan)


def test_resume_rejects_train_and_validation_identity_difference(
    work_dir: Path,
) -> None:
    _expect_field_rejected(
        work_dir,
        lambda current: current["m4.train_sample_identities"].reverse(),
        "m4.train_sample_identities",
    )
    _expect_field_rejected(
        work_dir,
        lambda current: current["m4.validation_sample_identities"].reverse(),
        "m4.validation_sample_identities",
    )


@pytest.mark.parametrize(
    "field_path",
    ["m3.train_seed", "m3.probe_seed", "m4.validation_seed"],
)
def test_resume_rejects_any_seed_difference(
    work_dir: Path,
    field_path: str,
) -> None:
    _expect_field_rejected(
        work_dir,
        lambda current: current.__setitem__(field_path, current[field_path] + 1),
        field_path,
    )


@pytest.mark.parametrize(
    "field_path",
    ["m3.backbone_lr", "m3.patch_embedding_lr", "m3.mcp_lr"],
)
def test_resume_rejects_any_lr_difference(
    work_dir: Path,
    field_path: str,
) -> None:
    _expect_field_rejected(
        work_dir,
        lambda current: current.__setitem__(field_path, current[field_path] * 2.0),
        field_path,
    )


def test_resume_rejects_weight_decay_difference(work_dir: Path) -> None:
    _expect_field_rejected(
        work_dir,
        lambda current: current.__setitem__("m3.weight_decay", 0.02),
        "m3.weight_decay",
    )


def test_resume_rejects_grid_auxiliary_weight_difference(work_dir: Path) -> None:
    _expect_field_rejected(
        work_dir,
        lambda current: current.__setitem__("m3.mcp1_grid_aux_weight", 0.0),
        "m3.mcp1_grid_aux_weight",
    )


@pytest.mark.parametrize(
    ("mutator", "field_path"),
    [
        (
            lambda config: config["m3"].__setitem__(
                "mcp1_grid_aux_enabled",
                False,
            ),
            "resolved_config.locked.m3.mcp1_grid_aux_enabled",
        ),
        (
            lambda config: config["m3"].__setitem__(
                "mcp1_grid_timesteps",
                [999.0, 937.5, 833.3333129882812, 625.0],
            ),
            "resolved_config.locked.m3.mcp1_grid_timesteps[0]",
        ),
        (
            lambda config: config["m3"].__setitem__(
                "mcp1_grid_schedule",
                {"source": "changed"},
            ),
            "resolved_config.locked.m3.mcp1_grid_schedule.source",
        ),
    ],
)
def test_resume_rejects_grid_config_difference(
    work_dir: Path,
    mutator: Callable[[dict[str, Any]], None],
    field_path: str,
) -> None:
    contract, _, plan = _contract_and_current(work_dir)
    current = _current_run_fields(work_dir, plan, config_mutator=mutator)

    with pytest.raises(m5.ResumeContractError) as exc_info:
        _validate_pass(contract, current, plan)
    assert field_path in str(exc_info.value)


@pytest.mark.parametrize(
    ("field_name", "replacement"),
    [
        ("timing_warmup_steps", 2),
        ("log_interval", 20),
        ("checkpoint_interval", 20),
    ],
)
def test_resume_rejects_timing_log_checkpoint_interval_difference(
    work_dir: Path,
    field_name: str,
    replacement: int,
) -> None:
    contract, _, plan = _contract_and_current(work_dir)
    current = _current_run_fields(
        work_dir,
        plan,
        config_mutator=lambda config: config["m3"].__setitem__(
            field_name,
            replacement,
        ),
    )

    with pytest.raises(m5.ResumeContractError) as exc_info:
        _validate_pass(contract, current, plan)
    assert f"resolved_config.locked.m3.{field_name}" in str(exc_info.value)


def test_resume_rejects_dtype_difference(work_dir: Path) -> None:
    _expect_field_rejected(
        work_dir,
        lambda current: current.__setitem__("m3.dtype", "float32"),
        "m3.dtype",
    )


def test_resume_rejects_reference_checkpoint_sha_difference(
    work_dir: Path,
) -> None:
    _expect_field_rejected(
        work_dir,
        lambda current: current.__setitem__("reference_checkpoint.sha256", "1" * 64),
        "reference_checkpoint.sha256",
    )


def test_resume_rejects_model_config_difference(work_dir: Path) -> None:
    def mutate(current: dict[str, Any]) -> None:
        changed = copy.deepcopy(current["model_config"])
        changed["model_kwargs"]["timestep_shift"] = 10.0
        current["model_config"] = changed

    _expect_field_rejected(work_dir, mutate, "model_config")


@pytest.mark.parametrize(
    ("field_path", "replacement"),
    [
        ("optimizer.betas", [0.9, 0.999]),
        ("optimizer.eps", 1.0e-7),
    ],
)
def test_resume_rejects_optimizer_betas_or_eps_difference(
    work_dir: Path,
    field_path: str,
    replacement: Any,
) -> None:
    _expect_field_rejected(
        work_dir,
        lambda current: current.__setitem__(field_path, replacement),
        field_path,
    )


def test_resume_rejects_optimizer_actual_param_group_lr_mismatch(
    work_dir: Path,
) -> None:
    _, _, plan = _contract_and_current(work_dir)

    with pytest.raises(
        m5.ResumeContractError,
        match=r"optimizer\.param_groups\[0\]\.lr",
    ):
        _current_run_fields(
            work_dir,
            plan,
            optimizer_state_mutator=lambda state: state["param_groups"][
                0
            ].__setitem__("lr", 9.0e-6),
        )


@pytest.mark.parametrize(
    ("field_name", "replacement"),
    [
        ("betas", (0.9, 0.999)),
        ("eps", 1.0e-7),
        ("weight_decay", 0.02),
    ],
)
def test_resume_rejects_optimizer_actual_hyperparameter_mismatch(
    work_dir: Path,
    field_name: str,
    replacement: Any,
) -> None:
    _, _, plan = _contract_and_current(work_dir)

    with pytest.raises(
        m5.ResumeContractError,
        match=rf"optimizer\.param_groups\[0\]\.{field_name}",
    ):
        _current_run_fields(
            work_dir,
            plan,
            optimizer_state_mutator=lambda state: state["param_groups"][
                0
            ].__setitem__(field_name, replacement),
        )


def test_resume_rejects_optimizer_group_lr_summary_mismatch(
    work_dir: Path,
) -> None:
    _, _, plan = _contract_and_current(work_dir)

    with pytest.raises(
        m5.ResumeContractError,
        match=r"optimizer_group_lrs\[0\]\.lr",
    ):
        _current_run_fields(
            work_dir,
            plan,
            optimizer_summary_mutator=lambda summary: summary[0].__setitem__(
                "lr",
                9.0e-6,
            ),
        )


@pytest.mark.parametrize("target_global_step", [4, 3])
def test_resume_rejects_target_step_not_greater_than_resume_step(
    work_dir: Path,
    target_global_step: int,
) -> None:
    contract, current, plan = _contract_and_current(work_dir, global_step=4)
    with pytest.raises(m5.ResumeContractError, match="target_global_step"):
        _validate_pass(
            contract,
            current,
            plan,
            target_global_step=target_global_step,
        )


def test_optimizer_state_device_migration_preserves_structure_and_values() -> None:
    original = {
        "state": {
            0: {
                "step": torch.tensor(2),
                "exp_avg": torch.arange(3, dtype=torch.float32),
                "nested": (
                    {"buf": torch.tensor([1.5])},
                    [torch.tensor([4], dtype=torch.int64)],
                ),
            }
        },
        "param_groups": [{"params": [0], "lr": 0.1, "name": "linear"}],
    }
    moved = m5.move_optimizer_state_to_device(original, device="cpu")

    assert isinstance(moved["state"][0]["nested"], tuple)
    assert isinstance(moved["state"][0]["nested"][1], list)
    assert torch.equal(moved["state"][0]["step"], torch.tensor(2))
    assert torch.equal(
        moved["state"][0]["exp_avg"],
        torch.arange(3, dtype=torch.float32),
    )
    assert torch.equal(moved["state"][0]["nested"][0]["buf"], torch.tensor([1.5]))
    original["state"][0]["exp_avg"].add_(100.0)
    assert torch.equal(
        moved["state"][0]["exp_avg"],
        torch.arange(3, dtype=torch.float32),
    )


def test_loaded_optimizer_state_in_place_migration_audits_state() -> None:
    _, optimizer = _make_grouped_model_and_optimizer()

    report = m5.move_loaded_optimizer_state_to_device(optimizer, device="cpu")

    assert report["status"] == "moved"
    assert report["device"] == "cpu"
    assert report["state_entry_count"] == len(optimizer.state)
    assert report["state_tensor_count"] > 0
    for state in optimizer.state.values():
        for value in state.values():
            if isinstance(value, torch.Tensor):
                assert value.device.type == "cpu"


def test_train_generator_state_restore_replays_exact_sequence() -> None:
    source = torch.Generator(device="cpu")
    source.manual_seed(123)
    state = source.get_state()
    expected = torch.rand((8,), generator=source)

    restored = m5.restore_torch_generator_from_state(state, device="cpu")
    actual = torch.rand((8,), generator=restored)

    assert torch.equal(actual, expected)


def test_global_python_and_torch_cpu_rng_restore_replays_exact_sequence() -> None:
    random.seed(123)
    python_state = random.getstate()
    expected_python = [random.random() for _ in range(5)]

    torch.random.manual_seed(456)
    torch_cpu_state = torch.random.get_rng_state()
    expected_torch = torch.rand((5,))

    random.seed(999)
    torch.random.manual_seed(999)
    m5.restore_global_rng_states(
        {
            "python_random_state": python_state,
            "torch_cpu_rng_state": torch_cpu_state,
        }
    )

    assert [random.random() for _ in range(5)] == expected_python
    assert torch.equal(torch.rand((5,)), expected_torch)


def test_sample_plan_resume_step_identity_helpers(work_dir: Path) -> None:
    plan = _sample_plan(work_dir, train_count=3)

    assert m5.first_resumed_global_step(4) == 5
    assert m5.first_resumed_sample_identity(plan, 4) == plan[
        "train_sample_identities"
    ][1]
    assert m5.next_sample_after_target_identity(plan, 7) == plan[
        "train_sample_identities"
    ][1]


def test_resume_report_is_strict_json_serializable(work_dir: Path) -> None:
    contract, current, plan = _contract_and_current(work_dir)
    report = _validate_pass(contract, current, plan)

    json.dumps(report, allow_nan=False)


def test_resume_helpers_do_not_modify_checkpoint_payload(work_dir: Path) -> None:
    payload, plan = _checkpoint_payload(work_dir)
    before = _payload_fingerprint(payload)

    contract = m5.build_resume_contract(
        payload,
        parent_checkpoint_path=work_dir / "checkpoint_step000004.pt",
        parent_checkpoint_sha256=PARENT_SHA,
        sample_plan=plan,
    )
    current = _current_run_fields(work_dir, plan)
    _validate_pass(contract, current, plan)

    assert _payload_fingerprint(payload) == before


def test_resume_missing_required_field_rejects_with_field_path(
    work_dir: Path,
) -> None:
    payload, plan = _checkpoint_payload(work_dir)
    del payload["resolved_config"]["m4"]["sample_plan_sha256"]

    with pytest.raises(m5.ResumeContractError) as exc_info:
        m5.build_resume_contract(
            payload,
            parent_checkpoint_path=work_dir / "checkpoint_step000004.pt",
            parent_checkpoint_sha256=PARENT_SHA,
            sample_plan=plan,
        )

    message = str(exc_info.value)
    assert "resolved_config.m4.sample_plan_sha256" in message
    assert "expected=" in message
    assert "actual=" in message


def test_resume_rejects_missing_parent_provenance(work_dir: Path) -> None:
    payload, plan = _checkpoint_payload(work_dir)

    with pytest.raises(m5.ResumeContractError, match="parent_checkpoint_path"):
        m5.build_resume_contract(
            payload,
            parent_checkpoint_sha256=PARENT_SHA,
            sample_plan=plan,
        )

    with pytest.raises(m5.ResumeContractError, match="parent_checkpoint_sha256"):
        m5.build_resume_contract(
            payload,
            parent_checkpoint_path=work_dir / "checkpoint_step000004.pt",
            sample_plan=plan,
        )


def test_parent_checkpoint_source_sha_mismatch_rejects(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "checkpoint.pt"
    checkpoint_path.write_bytes(b"checkpoint bytes")
    actual_sha = m3.file_sha256(checkpoint_path)

    record = m5.checkpoint_source_record(checkpoint_path, sha256=actual_sha)
    assert record["source_verified"] is True
    assert record["sha256"] == actual_sha

    with pytest.raises(m5.ResumeContractError, match="parent_checkpoint_sha256"):
        m5.checkpoint_source_record(checkpoint_path, sha256=PARENT_SHA)


@pytest.mark.parametrize(
    ("mutator", "field_path"),
    [
        (
            lambda payload: payload["resolved_config"]["m3"].__setitem__(
                "log_interval",
                float("nan"),
            ),
            "resolved_config.m3.log_interval",
        ),
        (
            lambda payload: payload["resolved_config"]["m3"].__setitem__(
                "unsupported",
                object(),
            ),
            "resolved_config.m3.unsupported",
        ),
    ],
)
def test_resume_rejects_invalid_canonical_config_values(
    work_dir: Path,
    mutator: Callable[[dict[str, Any]], None],
    field_path: str,
) -> None:
    payload, plan = _checkpoint_payload(work_dir)
    mutator(payload)

    with pytest.raises(m5.ResumeContractError) as exc_info:
        m5.build_resume_contract(
            payload,
            parent_checkpoint_path=work_dir / "checkpoint_step000004.pt",
            parent_checkpoint_sha256=PARENT_SHA,
            sample_plan=plan,
        )
    assert field_path in str(exc_info.value)


def test_legacy_checkpoint_missing_global_rng_requires_explicit_evidence(
    work_dir: Path,
) -> None:
    contract, current, plan = _contract_and_current(work_dir)

    with pytest.raises(m5.ResumeContractError) as exc_info:
        m5.validate_resume_contract(
            contract,
            current,
            target_global_step=7,
            sample_plan=plan,
        )
    assert "rng_restore.missing_global_rng_fields" in str(exc_info.value)

    report = _validate_pass(contract, current, plan)
    assert report["rng_restore"]["status"] == "PASS_LEGACY_MISSING_GLOBAL_RNG"
    assert report["rng_restore"]["missing_global_rng_fields"] == [
        "python_random_state",
        "torch_cpu_rng_state",
        "torch_cuda_rng_states",
    ]


def test_m5_rng_extension_validates_global_rng_without_legacy_override(
    work_dir: Path,
) -> None:
    payload, plan = _checkpoint_payload(work_dir)
    payload[m5.M5_RNG_EXTENSION_FIELD] = m5.capture_m5_global_rng_extension(
        include_cuda=False
    )
    contract = m5.build_resume_contract(
        payload,
        parent_checkpoint_path=work_dir / "checkpoint_step000004.pt",
        parent_checkpoint_sha256=PARENT_SHA,
        sample_plan=plan,
    )
    current = _current_run_fields(work_dir, plan)
    report = m5.validate_resume_contract(
        contract,
        current,
        target_global_step=7,
        sample_plan=plan,
    )

    assert report["status"] == "PASS"
    assert report["rng_restore"]["status"] == "PASS"
    assert report["rng_restore"]["missing_global_rng_fields"] == []


def test_empty_cuda_rng_rejected_when_single_cuda_device_expected(
    work_dir: Path,
) -> None:
    payload, plan = _checkpoint_payload(work_dir)
    payload[m5.M5_RNG_EXTENSION_FIELD] = m5.capture_m5_global_rng_extension(
        include_cuda=False
    )
    contract = m5.build_resume_contract(
        payload,
        parent_checkpoint_path=work_dir / "checkpoint_step000004.pt",
        parent_checkpoint_sha256=PARENT_SHA,
        sample_plan=plan,
    )
    current = _current_run_fields(work_dir, plan)

    with pytest.raises(
        m5.ResumeContractError,
        match="rng_restore.torch_cuda_rng_states",
    ):
        m5.validate_resume_contract(
            contract,
            current,
            target_global_step=7,
            sample_plan=plan,
            expected_cuda_device_count=1,
        )
