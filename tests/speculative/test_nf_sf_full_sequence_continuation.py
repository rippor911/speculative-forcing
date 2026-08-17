from __future__ import annotations

import copy
import inspect
import json
import random
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import utils.nf_sf_full_sequence_continuation as cont
import utils.nf_sf_training as nf_sf_training
from scripts import continue_nf_sf_full_sequence_next_forcing as runner
from utils.nf_sf_m3 import file_sha256
from utils.nf_sf_training import (
    FULL_SEQUENCE_CHECKPOINT_STEPS,
    FULL_SEQUENCE_OBJECTIVE_VERSION,
    FULL_SEQUENCE_RUN_KIND,
    FULL_SEQUENCE_TARGET_GLOBAL_STEP,
    FULL_SEQUENCE_TRAINER_SCHEMA,
    OFFICIAL_SELF_FORCING_CHECKPOINT_SHA256,
    nf_sf_full_sequence_train_cursor,
)
import utils.nf_sf_first_mcp_route_equivalence as route_eq


BASE_GIT = cont.BASE_TRAINING_GIT_SHA
CONT_GIT = "c" * 40
SAMPLE_SHA = "b" * 64
MANIFEST_SHA = "d" * 64
CONDITIONAL_SHA = "e" * 64


def semantic_config() -> dict[str, Any]:
    return {
        "schema": FULL_SEQUENCE_TRAINER_SCHEMA,
        "run_kind": FULL_SEQUENCE_RUN_KIND,
        "objective_version": FULL_SEQUENCE_OBJECTIVE_VERSION,
        "objective_mode": "next_forcing_full",
        "expected_git_sha": BASE_GIT,
        "current_git_sha": BASE_GIT,
        "checkpoint_sha256": OFFICIAL_SELF_FORCING_CHECKPOINT_SHA256,
        "checkpoint_size_bytes": 11,
        "sample_plan_sha256": SAMPLE_SHA,
        "manifest_sha256": MANIFEST_SHA,
        "conditionals_artifact_sha256": CONDITIONAL_SHA,
        "train_seed": 101,
        "validation_seed": 202,
        "global_seed": 303,
        "backbone_lr": 1.0e-6,
        "patch_embedding_lr": 1.0e-6,
        "mcp_lr": 1.0e-5,
        "weight_decay": 0.01,
        "adam_betas": [0.0, 0.999],
        "adam_eps": 1.0e-8,
        "dtype": "bf16",
        "device": "cuda:0",
        "num_frame_per_block": 3,
        "gradient_checkpointing": True,
        "full_teacher_frames": 21,
        "chunk_frames": 3,
        "num_chunks": 7,
        "main_shift": 5.0,
        "mcp_shift": 10.0,
        "depth_weights": [0.5, 0.2, 0.1],
        "tap_layers": [3, 11, 19, 29],
        "mcp_blocks_per_depth": 3,
        "rng_draw_order_version": "nf_sf_full_sequence_rng_v1",
        "validation_tensor_slot": "nf_sf_full_sequence_next_forcing_v1",
        "validation_seed_derivation": "derive_m4_validation_seed",
        "validation_identity_noise_is_paired_across_steps": True,
        "production_target_global_step": 5000,
        "production_checkpoint_steps": [0, 500, 2000, 5000],
        "production_validation_steps": [0, 500, 2000, 5000],
        "main_backbone_forward_count_per_train_sample": 1,
        "anchor_micro_loop": True,
        "paper_exact_reproduction": False,
        "no_125_step_pilot": True,
    }


def provenance(*, continuation: dict[str, Any] | None = None) -> dict[str, Any]:
    record = {
        "schema": FULL_SEQUENCE_TRAINER_SCHEMA,
        "run_kind": FULL_SEQUENCE_RUN_KIND,
        "objective_version": FULL_SEQUENCE_OBJECTIVE_VERSION,
        "paper_exact_reproduction": False,
        "future_embedding_order": "depth_major",
        "rng_draw_order_version": "nf_sf_full_sequence_rng_v1",
        "objective": {
            "joint_backbone": True,
            "shared_patch_embedding": True,
            "self_rollout": False,
            "dmd": False,
            "generated_history": False,
            "noisy_history_augmentation": False,
        },
    }
    if continuation is not None:
        record["continuation"] = continuation
    return record


def optimizer_state_and_contract() -> tuple[dict[str, Any], dict[str, Any]]:
    groups = [
        ("backbone", 1.0e-6, [0, 1]),
        ("patch_embedding", 1.0e-6, [2]),
        ("mcp_fusion", 1.0e-5, [3]),
        ("mcp_depth1", 1.0e-5, [4]),
        ("mcp_depth2", 1.0e-5, [5]),
        ("mcp_depth3", 1.0e-5, [6]),
    ]
    state = {
        "state": {},
        "param_groups": [
            {
                "name": name,
                "lr": lr,
                "weight_decay": 0.01,
                "params": params,
            }
            for name, lr, params in groups
        ],
    }
    contract = {
        "class": "AdamW",
        "betas": [float(value) for value in cont.ADAMW_BETAS],
        "eps": float(cont.ADAMW_EPS),
        "weight_decay": 0.01,
        "param_groups": [
            {
                "name": group["name"],
                "lr": group["lr"],
                "weight_decay": group["weight_decay"],
                "param_count": len(group["params"]),
            }
            for group in state["param_groups"]
        ],
    }
    return state, contract


def continuation_lineage(parent_step: int, target_step: int) -> dict[str, Any]:
    return {
        "schema": cont.CONTINUATION_SCHEMA,
        "continuation_only_training_horizon_extension": True,
        "objective_changed": False,
        "architecture_changed": False,
        "optimizer_hparams_changed": False,
        "data_contract_changed": False,
        "rng_semantics_changed": False,
        "freeze_policy_changed": False,
        "parent_global_step": parent_step,
        "target_global_step": target_step,
        "runtime_git_sha": CONT_GIT,
        "base_training_git_sha": BASE_GIT,
        "parent_checkpoint_git_sha": BASE_GIT if parent_step == 5000 else CONT_GIT,
        "parent_checkpoint_sha256": "a" * 64,
    }


def payload_for_step(step: int, *, git_sha: str | None = None) -> dict[str, Any]:
    optimizer_state, optimizer_contract = optimizer_state_and_contract()
    resolved = semantic_config()
    prov = provenance()
    if step in (6500, 8000):
        parent_step = 5000 if step == 6500 else 6500
        lineage = continuation_lineage(parent_step, step)
        resolved.update(
            {
                "current_git_sha": CONT_GIT,
                "expected_git_sha": CONT_GIT,
                "continuation_schema": cont.CONTINUATION_SCHEMA,
                "continuation_parent_global_step": parent_step,
                "continuation_target_global_step": step,
                "continuation_checkpoint_steps": [step],
                "continuation_validation_steps": [step],
                "base_training_git_sha": BASE_GIT,
                "parent_checkpoint_git_sha": lineage["parent_checkpoint_git_sha"],
                "parent_checkpoint_sha256": "a" * 64,
            }
        )
        prov = provenance(continuation=lineage)
    return {
        "schema": FULL_SEQUENCE_TRAINER_SCHEMA,
        "run_kind": FULL_SEQUENCE_RUN_KIND,
        "objective_version": FULL_SEQUENCE_OBJECTIVE_VERSION,
        "objective_mode": "next_forcing_full",
        "status": "PRODUCTION",
        "global_step": step,
        "git_sha": git_sha or (CONT_GIT if step in (6500, 8000) else BASE_GIT),
        "generator": {"model.weight": torch.zeros(1), "mcp.depth1.weight": torch.ones(1)},
        "optimizer": optimizer_state,
        "train_rng_state": torch.get_rng_state(),
        "validation_seed": 202,
        "validation_base_rng_state": torch.get_rng_state(),
        "python_random_state": random.getstate(),
        "torch_cpu_global_rng_state": torch.get_rng_state(),
        "torch_cuda_global_rng_state": torch.get_rng_state(),
        "sample_cursor": nf_sf_full_sequence_train_cursor(step),
        "sample_plan_sha256": SAMPLE_SHA,
        "manifest_sha256": MANIFEST_SHA,
        "conditionals_artifact_sha256": CONDITIONAL_SHA,
        "resolved_config": resolved,
        "provenance": prov,
        "reference_checkpoint": {
            "path": "checkpoints/self_forcing_dmd.pt",
            "sha256": OFFICIAL_SELF_FORCING_CHECKPOINT_SHA256,
            "size_bytes": 11,
        },
        "optimizer_contract": optimizer_contract,
    }


def write_checkpoint(tmp_path: Path, step: int, *, payload_updates=None, validation_updates=None) -> Path:
    payload = payload_for_step(step)
    if payload_updates:
        payload.update(payload_updates)
    path = tmp_path / f"checkpoint_step{step:06d}.pt"
    torch.save(payload, path)
    sha = file_sha256(path)
    path.with_suffix("").with_suffix(".sha256.txt").write_text(
        f"{sha}  {path.name}\n",
        encoding="utf-8",
    )
    validation = {
        "status": "PASS",
        "path": str(path.resolve()),
        "sha256": sha,
        "size_bytes": int(path.stat().st_size),
        "schema": cont.CHECKPOINT_VALIDATION_SCHEMA,
        "run_kind": FULL_SEQUENCE_RUN_KIND,
        "objective_version": FULL_SEQUENCE_OBJECTIVE_VERSION,
        "objective_mode": "next_forcing_full",
        "global_step": step,
        "generator_key_count": len(payload["generator"]),
        "optimizer_state_entry_count": 0,
    }
    if validation_updates:
        validation.update(validation_updates)
    path.with_suffix("").with_suffix(".validation.json").write_text(
        json.dumps(validation),
        encoding="utf-8",
    )
    return path


def load_parent(tmp_path: Path, step: int, **kwargs):
    path = write_checkpoint(tmp_path, step, **kwargs)
    return cont.load_continuation_parent_checkpoint(
        path,
        expected_parent_checkpoint_sha256=file_sha256(path),
        expected_parent_global_step=step,
        expected_parent_checkpoint_git_sha=CONT_GIT if step == 6500 else BASE_GIT,
        sample_plan_sha256=SAMPLE_SHA,
        manifest_sha256=MANIFEST_SHA,
        conditionals_artifact_sha256=CONDITIONAL_SHA,
    )


def test_pair_contract() -> None:
    assert cont.validate_continuation_stage_pair(5000, 6500) == (5000, 6500)
    assert cont.validate_continuation_stage_pair(6500, 8000) == (6500, 8000)
    with pytest.raises(ValueError, match="5000->6500"):
        cont.validate_continuation_stage_pair(5000, 8000)
    with pytest.raises(ValueError):
        cont.validate_continuation_stage_pair(8000, 10000)


def test_canonical_baseline_constants_unchanged() -> None:
    assert FULL_SEQUENCE_TARGET_GLOBAL_STEP == 5000
    assert FULL_SEQUENCE_CHECKPOINT_STEPS == (0, 500, 2000, 5000)


def test_parent_step5000_strict_payload_loads(tmp_path: Path) -> None:
    parent = load_parent(tmp_path, 5000)
    assert parent.parent_global_step == 5000
    assert parent.parent_git_sha == BASE_GIT
    assert parent.semantic_lock_fingerprint == cont.semantic_lock_fingerprint(
        parent.payload["resolved_config"]
    )


def test_real_canonical_optimizer_group_names_are_locked() -> None:
    assert cont.CANONICAL_OPTIMIZER_GROUP_NAMES == (
        "backbone",
        "patch_embedding",
        "mcp_fusion",
        "mcp_depth1",
        "mcp_depth2",
        "mcp_depth3",
    )
    payload = payload_for_step(5000)
    group_names = tuple(
        group["name"] for group in payload["optimizer_contract"]["param_groups"]
    )
    assert group_names == cont.CANONICAL_OPTIMIZER_GROUP_NAMES
    assert "mcp" not in group_names
    source = inspect.getsource(
        nf_sf_training.configure_nf_sf_full_sequence_optimizer_plan
    )
    for name in cont.CANONICAL_OPTIMIZER_GROUP_NAMES:
        if name.startswith("mcp_depth"):
            assert "mcp_depth" in source
        else:
            assert name in source
    assert 'trainable_names.add("mcp")' not in source
    assert '"name": "mcp"' not in source


@pytest.mark.parametrize(
    ("payload_updates", "validation_updates", "match"),
    [
        ({"status": "NON_PRODUCTION_SMOKE"}, None, "status"),
        ({"schema": "bad"}, None, "schema"),
        ({"objective_mode": "main_only_full_control"}, None, "objective_mode"),
        ({"global_step": 2000}, None, "global_step"),
        ({"git_sha": CONT_GIT}, None, "git_sha"),
        ({"sample_plan_sha256": "1" * 64}, None, "sample_plan_sha256"),
        ({"manifest_sha256": "1" * 64}, None, "manifest_sha256"),
        ({"conditionals_artifact_sha256": "1" * 64}, None, "conditionals_artifact_sha256"),
        ({"reference_checkpoint": {"sha256": "1" * 64}}, None, "official SHA"),
        ({"sample_cursor": {"position": 99}}, None, "sample_cursor"),
        ({"provenance": {**provenance(), "paper_exact_reproduction": True}}, None, "paper_exact"),
        (None, {"path": "wrong.pt"}, "path"),
        (None, {"generator_key_count": 0}, "generator_key_count"),
        (None, {"optimizer_state_entry_count": 99}, "optimizer_state_entry_count"),
        (None, {"status": "FAIL"}, "PASS"),
        (None, {"schema": "bad"}, "schema"),
    ],
)
def test_parent_checkpoint_tamper_rejects(
    tmp_path: Path,
    payload_updates,
    validation_updates,
    match: str,
) -> None:
    path = write_checkpoint(
        tmp_path,
        5000,
        payload_updates=payload_updates,
        validation_updates=validation_updates,
    )
    with pytest.raises((RuntimeError, ValueError), match=match):
        cont.load_continuation_parent_checkpoint(
            path,
            expected_parent_checkpoint_sha256=file_sha256(path),
            expected_parent_global_step=5000,
            expected_parent_checkpoint_git_sha=BASE_GIT,
            sample_plan_sha256=SAMPLE_SHA,
            manifest_sha256=MANIFEST_SHA,
            conditionals_artifact_sha256=CONDITIONAL_SHA,
        )


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("backbone_lr", 2.0e-6),
        ("weight_decay", 0.02),
        ("depth_weights", [1.0, 0.2, 0.1]),
        ("tap_layers", [3, 11, 19, 28]),
        ("main_shift", 6.0),
        ("mcp_shift", 11.0),
        ("num_frame_per_block", 1),
        ("gradient_checkpointing", False),
    ],
)
def test_semantic_lock_tamper_rejects(key: str, value: Any) -> None:
    resolved = semantic_config()
    resolved[key] = value
    if key in ("backbone_lr", "weight_decay"):
        payload = payload_for_step(5000)
        payload["resolved_config"] = resolved
        with pytest.raises(RuntimeError, match="semantic lock"):
            cont._validate_parent_payload_contract(
                payload,
                expected_parent_global_step=5000,
                expected_parent_checkpoint_git_sha=BASE_GIT,
                sample_plan_sha256=SAMPLE_SHA,
                manifest_sha256=MANIFEST_SHA,
                conditionals_artifact_sha256=CONDITIONAL_SHA,
            )
    else:
        with pytest.raises(RuntimeError, match="semantic lock"):
            cont.validate_semantic_lock(resolved)


@pytest.mark.parametrize(
    ("key", "value", "match"),
    [
        ("sample_plan_sha256", "1" * 64, "resolved_config sample_plan_sha256"),
        ("manifest_sha256", "1" * 64, "resolved_config manifest_sha256"),
        (
            "conditionals_artifact_sha256",
            "1" * 64,
            "resolved_config conditionals_artifact_sha256",
        ),
    ],
)
def test_resolved_artifact_sha_tamper_rejects(key: str, value: str, match: str) -> None:
    payload = payload_for_step(5000)
    payload["resolved_config"] = copy.deepcopy(payload["resolved_config"])
    payload["resolved_config"][key] = value
    with pytest.raises(RuntimeError, match=match):
        cont._validate_parent_payload_contract(
            payload,
            expected_parent_global_step=5000,
            expected_parent_checkpoint_git_sha=BASE_GIT,
            sample_plan_sha256=SAMPLE_SHA,
            manifest_sha256=MANIFEST_SHA,
            conditionals_artifact_sha256=CONDITIONAL_SHA,
        )


@pytest.mark.parametrize(
    ("mutator", "match"),
    [
        (
            lambda p: p["provenance"].__setitem__("future_embedding_order", "anchor_major"),
            "future_embedding_order",
        ),
        (
            lambda p: p["provenance"].__setitem__("rng_draw_order_version", "bad"),
            "rng_draw_order_version",
        ),
        (
            lambda p: p["provenance"]["objective"].__setitem__("joint_backbone", False),
            "joint_backbone",
        ),
        (
            lambda p: p["provenance"]["objective"].__setitem__("generated_history", True),
            "generated_history",
        ),
    ],
)
def test_provenance_semantic_tamper_rejects(mutator, match: str) -> None:
    payload = payload_for_step(5000)
    mutator(payload)
    with pytest.raises(RuntimeError, match=match):
        cont._validate_parent_payload_contract(
            payload,
            expected_parent_global_step=5000,
            expected_parent_checkpoint_git_sha=BASE_GIT,
            sample_plan_sha256=SAMPLE_SHA,
            manifest_sha256=MANIFEST_SHA,
            conditionals_artifact_sha256=CONDITIONAL_SHA,
        )


def test_optimizer_contract_mismatch_rejects() -> None:
    payload = payload_for_step(5000)
    payload["optimizer_contract"] = copy.deepcopy(payload["optimizer_contract"])
    payload["optimizer_contract"]["param_groups"][0]["lr"] = 9.0
    with pytest.raises(RuntimeError, match="LR"):
        cont.validate_optimizer_contract_for_continuation(payload)


def test_optimizer_mcp_alias_group_rejects() -> None:
    payload = payload_for_step(5000)
    payload["optimizer"]["param_groups"][2]["name"] = "mcp"
    payload["optimizer_contract"]["param_groups"][2]["name"] = "mcp"
    with pytest.raises(RuntimeError, match="names/order"):
        cont.validate_optimizer_contract_for_continuation(payload)


class TinyGenerator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Linear(1, 1)
        self.patch_embedding = nn.Linear(1, 1, bias=False)
        self.mcp_fusion = nn.Linear(1, 1, bias=False)
        self.mcp = nn.Module()
        self.mcp.depth1 = nn.Linear(1, 1, bias=False)
        self.mcp.depth2 = nn.Linear(1, 1, bias=False)
        self.mcp.depth3 = nn.Linear(1, 1, bias=False)


def _tiny_optimizer(module: TinyGenerator) -> torch.optim.AdamW:
    return torch.optim.AdamW(
        [
            {"params": list(module.model.parameters()), "name": "backbone", "lr": 1e-6, "weight_decay": 0.01},
            {"params": list(module.patch_embedding.parameters()), "name": "patch_embedding", "lr": 1e-6, "weight_decay": 0.01},
            {"params": list(module.mcp_fusion.parameters()), "name": "mcp_fusion", "lr": 1e-5, "weight_decay": 0.01},
            {"params": list(module.mcp.depth1.parameters()), "name": "mcp_depth1", "lr": 1e-5, "weight_decay": 0.01},
            {"params": list(module.mcp.depth2.parameters()), "name": "mcp_depth2", "lr": 1e-5, "weight_decay": 0.01},
            {"params": list(module.mcp.depth3.parameters()), "name": "mcp_depth3", "lr": 1e-5, "weight_decay": 0.01},
        ],
        betas=cont.ADAMW_BETAS,
        eps=cont.ADAMW_EPS,
        weight_decay=0.01,
    )


def _optimizer_state_equal(left: dict[str, Any], right: dict[str, Any]) -> bool:
    if len(left["param_groups"]) != len(right["param_groups"]):
        return False
    for l_group, r_group in zip(left["param_groups"], right["param_groups"]):
        for key in ("name", "lr", "weight_decay", "params"):
            if l_group.get(key) != r_group.get(key):
                return False
    if set(left["state"].keys()) != set(right["state"].keys()):
        return False
    for key in left["state"]:
        l_state = left["state"][key]
        r_state = right["state"][key]
        if set(l_state.keys()) != set(r_state.keys()):
            return False
        for state_key in l_state:
            l_value = l_state[state_key]
            r_value = r_state[state_key]
            if torch.is_tensor(l_value):
                if not torch.equal(l_value.cpu(), r_value.cpu()):
                    return False
            elif l_value != r_value:
                return False
    return True


def test_exact_restore_generator_optimizer_and_rng() -> None:
    source = TinyGenerator()
    optim = _tiny_optimizer(source)
    x = torch.ones(1, 1)
    loss = (
        source.model(x).sum()
        + source.patch_embedding(x).sum()
        + source.mcp_fusion(x).sum()
        + source.mcp.depth1(x).sum()
        + source.mcp.depth2(x).sum()
        + source.mcp.depth3(x).sum()
    )
    loss.backward()
    optim.step()
    train_rng = torch.Generator(device="cpu")
    validation_rng = torch.Generator(device="cpu")
    train_rng.manual_seed(7)
    validation_rng.manual_seed(8)
    payload = {
        "generator": copy.deepcopy(source.state_dict()),
        "optimizer": copy.deepcopy(optim.state_dict()),
        "train_rng_state": train_rng.get_state(),
        "validation_base_rng_state": validation_rng.get_state(),
        "python_random_state": random.getstate(),
        "torch_cpu_global_rng_state": torch.get_rng_state(),
        "torch_cuda_global_rng_state": None,
    }
    target = TinyGenerator()
    target_optim = _tiny_optimizer(target)
    restored_train = torch.Generator(device="cpu")
    restored_validation = torch.Generator(device="cpu")
    report = cont.restore_continuation_state(
        generator=target,
        optimizer=target_optim,
        train_rng=restored_train,
        validation_base_rng=restored_validation,
        payload=payload,
        device=torch.device("cpu"),
    )
    for key, tensor in source.state_dict().items():
        assert torch.equal(tensor, target.state_dict()[key])
    assert _optimizer_state_equal(optim.state_dict(), target_optim.state_dict())
    assert report["status"] == "PASS"
    assert torch.equal(restored_train.get_state(), train_rng.get_state())
    assert torch.equal(restored_validation.get_state(), validation_rng.get_state())
    assert random.getstate() == payload["python_random_state"]
    assert torch.equal(torch.get_rng_state(), payload["torch_cpu_global_rng_state"])
    assert len(target_optim.state) == len(optim.state)


def test_exact_restore_cuda_rng_contract_with_fake_cuda(monkeypatch) -> None:
    source = TinyGenerator()
    optim = _tiny_optimizer(source)
    payload = {
        "generator": copy.deepcopy(source.state_dict()),
        "optimizer": copy.deepcopy(optim.state_dict()),
        "train_rng_state": torch.Generator(device="cpu").get_state(),
        "validation_base_rng_state": torch.Generator(device="cpu").get_state(),
        "python_random_state": random.getstate(),
        "torch_cpu_global_rng_state": torch.get_rng_state(),
        "torch_cuda_global_rng_state": torch.arange(8, dtype=torch.uint8),
    }
    seen = {}
    monkeypatch.setattr(cont, "move_optimizer_state_to_device", lambda *args, **kwargs: None)
    monkeypatch.setattr(cont.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        cont.torch.cuda,
        "set_rng_state",
        lambda state, device=None: seen.setdefault("set", (state.clone(), str(device))),
    )
    monkeypatch.setattr(
        cont.torch.cuda,
        "get_rng_state",
        lambda device=None: payload["torch_cuda_global_rng_state"].clone(),
    )
    report = cont.restore_continuation_state(
        generator=TinyGenerator(),
        optimizer=_tiny_optimizer(TinyGenerator()),
        train_rng=torch.Generator(device="cpu"),
        validation_base_rng=torch.Generator(device="cpu"),
        payload=payload,
        device=torch.device("cuda:0"),
    )
    assert torch.equal(seen["set"][0], payload["torch_cuda_global_rng_state"])
    assert seen["set"][1] == "cuda:0"
    assert report["rng_fingerprint"]["torch_cuda_global_rng_state_sha256"] is not None


def test_no_seed_reset_source_guard() -> None:
    source = Path("scripts/continue_nf_sf_full_sequence_next_forcing.py").read_text(encoding="utf-8")
    utility = Path("utils/nf_sf_full_sequence_continuation.py").read_text(encoding="utf-8")
    assert "reset_global_seed" not in source
    assert "reset_global_seed" not in utility


def test_first_resumed_step_and_schedule_contract() -> None:
    assert cont.continuation_start_step(5000, 6500) == 5001
    assert cont.continuation_start_step(6500, 8000) == 6501
    assert cont.continuation_checkpoint_steps(5000, 6500) == (6500,)
    assert cont.continuation_validation_steps(6500, 8000) == (8000,)
    contract = cont.first_continuation_step_contract(
        parent_step=5000,
        target_step=6500,
        train_identity="id-5001",
        sample_cursor=nf_sf_full_sequence_train_cursor(5001),
    )
    assert contract["first_global_step"] == 5001


def test_continuation_lineage_required_for_step6500(tmp_path: Path) -> None:
    parent = load_parent(tmp_path, 6500)
    cont.validate_continuation_lineage(
        parent.payload,
        expected_parent_step=5000,
        expected_target_step=6500,
    )
    tampered = copy.deepcopy(parent.payload)
    tampered["provenance"]["continuation"]["objective_changed"] = True
    with pytest.raises(RuntimeError, match="objective_changed"):
        cont.validate_continuation_lineage(tampered, expected_target_step=6500)


@pytest.mark.parametrize(
    ("mutator", "match"),
    [
        (
            lambda p: p["resolved_config"].__setitem__(
                "continuation_parent_global_step", 6500
            ),
            "resolved_config parent",
        ),
        (
            lambda p: p["resolved_config"].__setitem__(
                "continuation_checkpoint_steps", [5000, 6500]
            ),
            "checkpoint schedule",
        ),
        (
            lambda p: p["resolved_config"].__setitem__(
                "continuation_validation_steps", [5000, 6500]
            ),
            "validation schedule",
        ),
        (
            lambda p: p["provenance"]["continuation"].__setitem__(
                "parent_global_step", 6500
            ),
            "provenance parent",
        ),
        (
            lambda p: p["provenance"]["continuation"].__setitem__(
                "runtime_git_sha", BASE_GIT
            ),
            "runtime git",
        ),
        (
            lambda p: p["resolved_config"].__setitem__("current_git_sha", BASE_GIT),
            "current git",
        ),
        (
            lambda p: p["resolved_config"].__setitem__("expected_git_sha", BASE_GIT),
            "expected git",
        ),
        (
            lambda p: p["resolved_config"].__setitem__("base_training_git_sha", CONT_GIT),
            "base training git",
        ),
        (
            lambda p: p["resolved_config"].__setitem__(
                "parent_checkpoint_git_sha", CONT_GIT
            ),
            "parent checkpoint git",
        ),
        (
            lambda p: p["resolved_config"].__setitem__(
                "parent_checkpoint_sha256", "b" * 64
            ),
            "parent checkpoint SHA",
        ),
    ],
)
def test_continuation_lineage_strict_tamper_rejects(mutator, match: str) -> None:
    payload = payload_for_step(6500)
    mutator(payload)
    with pytest.raises(RuntimeError, match=match):
        cont.validate_continuation_lineage(
            payload,
            expected_parent_step=5000,
            expected_target_step=6500,
        )


def test_route_equivalence_loader_accepts_continuation_6500_8000(tmp_path: Path) -> None:
    for step in (6500, 8000):
        path = write_checkpoint(tmp_path, step)
        record = route_eq.load_route_equivalence_checkpoint_record(
            path,
            expected_checkpoint_step=step,
            expected_training_git_sha=CONT_GIT,
        )
        assert record.checkpoint_type == f"full_sequence_step{step}"
        assert record.global_step == step


def test_runner_cli_requires_stage_pair() -> None:
    argv = [
        "--parent_checkpoint",
        "checkpoint_step005000.pt",
        "--expected_parent_checkpoint_sha256",
        "a" * 64,
        "--expected_parent_global_step",
        "5000",
        "--expected_parent_checkpoint_git_sha",
        BASE_GIT,
        "--target_global_step",
        "6500",
        "--expected_runtime_git_sha",
        CONT_GIT,
        "--sample_plan",
        "sample_plan.json",
        "--manifest",
        "manifest.json",
        "--dataset_root",
        "dataset",
        "--conditionals_artifact",
        "conditionals",
        "--output_dir",
        "out",
    ]
    assert runner.parse_args(argv).target_global_step == 6500
    with pytest.raises(SystemExit):
        runner.parse_args(argv[:8] + ["8000"] + argv[9:])
