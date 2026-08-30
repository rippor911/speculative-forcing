from __future__ import annotations

import ast
import copy
import types
from pathlib import Path

import pytest
import torch
from torch import nn

import scripts.train_nf_sf_tiny_mcp1_paper_fidelity_ab as tiny_ab
from utils.nf_sf_training import (
    NFSFFullSequenceNoisyBatch,
    build_full_sequence_mcp_anchor_specs,
)


ROOT = Path(__file__).resolve().parents[2]


def _loss_batch(*, anchor1_target: float = 0.0) -> NFSFFullSequenceNoisyBatch:
    clean = torch.zeros((1, 21, 1, 1, 1), dtype=torch.float32)
    mcp1_target = torch.zeros((1, 6, 3, 1, 1, 1), dtype=torch.float32)
    mcp1_target[:, tiny_ab.ANCHOR_INDEX].fill_(float(anchor1_target))
    return NFSFFullSequenceNoisyBatch(
        clean_target=clean,
        noisy_main=clean.clone(),
        target_flow_main=torch.zeros_like(clean),
        epsilon_main=torch.zeros_like(clean),
        raw_timestep_main=torch.zeros((1, 7), dtype=torch.int64),
        timestep_main=torch.zeros((1, 21), dtype=torch.float32),
        noisy_mcp_depths=(
            torch.zeros((1, 6, 3, 1, 1, 1), dtype=torch.float32),
            torch.zeros((1, 5, 3, 1, 1, 1), dtype=torch.float32),
            torch.zeros((1, 4, 3, 1, 1, 1), dtype=torch.float32),
        ),
        target_flow_mcp_depths=(
            mcp1_target,
            torch.full((1, 5, 3, 1, 1, 1), 100.0, dtype=torch.float32),
            torch.full((1, 4, 3, 1, 1, 1), 200.0, dtype=torch.float32),
        ),
        epsilon_mcp_depths=(
            torch.zeros((1, 6, 3, 1, 1, 1), dtype=torch.float32),
            torch.zeros((1, 5, 3, 1, 1, 1), dtype=torch.float32),
            torch.zeros((1, 4, 3, 1, 1, 1), dtype=torch.float32),
        ),
        raw_timestep_mcp_depths=(
            torch.zeros((1, 6), dtype=torch.int64),
            torch.zeros((1, 5), dtype=torch.int64),
            torch.zeros((1, 4), dtype=torch.int64),
        ),
        timestep_mcp_depths=(
            torch.zeros((1, 6, 3), dtype=torch.float32),
            torch.zeros((1, 5, 3), dtype=torch.float32),
            torch.zeros((1, 4, 3), dtype=torch.float32),
        ),
        anchor_specs=build_full_sequence_mcp_anchor_specs(),
    )


class FakeForwardGenerator:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def forward_full_sequence_next_forcing(self, **kwargs):
        self.calls.append(dict(kwargs))
        noisy = kwargs["noisy_image_or_video"]
        b, _frames, c, h, w = noisy.shape
        mcp1 = torch.empty((b, 6, 3, c, h, w), dtype=torch.float32)
        for anchor_index in range(6):
            mcp1[:, anchor_index].fill_(float(anchor_index + 10))
        return types.SimpleNamespace(
            main_flow_pred=torch.full((b, 21, c, h, w), -1000.0),
            mcp_flow_preds_by_depth=(
                mcp1,
                torch.full((b, 5, 3, c, h, w), -2000.0),
                torch.full((b, 4, 3, c, h, w), -3000.0),
            ),
        )


class FakeBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.patch_embedding = nn.Linear(1, 1, bias=False)
        self.backbone_weight = nn.Parameter(torch.tensor(0.25))
        self.head = nn.Linear(1, 1, bias=False)


class FakeMCP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fusion = nn.Linear(1, 1, bias=False)
        self.mcp_modules = nn.ModuleList(
            [nn.Linear(1, 1, bias=False) for _ in range(3)]
        )


class FakeTrainGenerator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = FakeBackbone()
        self.mcp = FakeMCP()

    def forward_full_sequence_next_forcing(self, **kwargs):
        noisy = kwargs["noisy_image_or_video"]
        b, _frames, c, h, w = noisy.shape
        scalar = (
            self.model.backbone_weight
            + self.model.patch_embedding.weight.sum()
            + self.mcp.fusion.weight.sum()
            + self.mcp.mcp_modules[0].weight.sum()
        )
        mcp1 = scalar.reshape(1, 1, 1, 1, 1, 1).expand(b, 6, 3, c, h, w)
        return types.SimpleNamespace(
            main_flow_pred=torch.zeros((b, 21, c, h, w), device=noisy.device),
            mcp_flow_preds_by_depth=(
                mcp1,
                torch.zeros((b, 5, 3, c, h, w), device=noisy.device),
                torch.zeros((b, 4, 3, c, h, w), device=noisy.device),
            ),
        )


class FakeScheduler:
    def add_noise(
        self,
        clean: torch.Tensor,
        noise: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        return clean + noise + timestep.reshape(-1, 1, 1, 1).to(clean.dtype) / 1000.0

    def training_target(
        self,
        clean: torch.Tensor,
        noise: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        del timestep
        return noise - clean


def _synthetic_plan() -> dict:
    plan = tiny_ab._synthetic_plan_for_dry_run()
    plan["sample_plan_sha256"] = "a" * 64
    return plan


def _state_spec(
    *,
    split: str = "train",
    identity: str = "identity_0000",
    state_index: int = 0,
    split_index: int = 0,
    selected_split_position: int = 0,
    selected_identity_order: int = 0,
    raw_timestep: int = 999,
    raw_order: int = 0,
    noise_index: int = tiny_ab.NOISE_INDEX,
) -> dict:
    return {
        "state_id": (
            f"{split}_sel{selected_identity_order:02d}_"
            f"pos{selected_split_position:04d}_raw{raw_timestep:03d}_"
            f"noise{noise_index}"
        ),
        "state_index": int(state_index),
        "split": split,
        "selection_policy": "uniform_even_split_positions_identity_major_raw_order",
        "selected_identity_order": int(selected_identity_order),
        "selected_split_position": int(selected_split_position),
        "identity": identity,
        "split_index": int(split_index),
        "sample_index": int(split_index),
        "sample_id": None,
        "raw_order": int(raw_order),
        "raw_timestep": int(raw_timestep),
        "noise_index": int(noise_index),
        "anchor_index": tiny_ab.ANCHOR_INDEX,
        "current_chunk_index": tiny_ab.CURRENT_CHUNK_INDEX,
        "future_chunk_index": tiny_ab.FUTURE_CHUNK_INDEX,
        "mcp_depth": tiny_ab.MCP_DEPTH,
    }


def _eval_record(state_spec: dict, mse: float) -> dict:
    return {
        "state_identity": tiny_ab.canonical_state_identity_from_spec(state_spec),
        "state_id": str(state_spec["state_id"]),
        "state_index": int(state_spec["state_index"]),
        "split": str(state_spec["split"]),
        "split_index": int(state_spec["split_index"]),
        "identity": str(state_spec["identity"]),
        "raw_timestep": int(state_spec["raw_timestep"]),
        "noise_index": int(state_spec["noise_index"]),
        "mcp1_anchor1_mse": float(mse),
        "finite": True,
        "state_proof": {
            "future_noise_sha256": (
                f"noise-{state_spec['split']}-{state_spec['identity']}-"
                f"{state_spec['raw_timestep']}-{state_spec['noise_index']}"
            ),
            "future_target_sha256": (
                f"target-{state_spec['split']}-{state_spec['identity']}-"
                f"{state_spec['raw_timestep']}"
            ),
            "exact_fm_target_sha256": (
                f"fm-{state_spec['split']}-{state_spec['identity']}-"
                f"{state_spec['raw_timestep']}-{state_spec['noise_index']}"
            ),
        },
    }


def _eval_report(mse_by_identity_raw: dict[tuple[str, int], float]) -> dict:
    records = []
    for state_index, ((identity, raw), mse) in enumerate(mse_by_identity_raw.items()):
        split = "validation" if str(identity).startswith("val_") else "train"
        raw_order = tiny_ab.RAW_TIMESTEPS.index(int(raw))
        selected_identity_order = state_index // len(tiny_ab.RAW_TIMESTEPS)
        state = _state_spec(
            split=split,
            identity=identity,
            state_index=state_index,
            split_index=selected_identity_order,
            selected_split_position=selected_identity_order,
            selected_identity_order=selected_identity_order,
            raw_timestep=raw,
            raw_order=raw_order,
        )
        records.append(_eval_record(state, mse))
    return {
        "schema": tiny_ab.TINY_AB_EVAL_SCHEMA,
        "state_count": len(records),
        "records": records,
        "aggregate": tiny_ab.aggregate_eval_records(records),
    }


def _arm_result(*, mse: float, paper_flag: bool = False) -> dict:
    values = {
        (f"val_{index}", raw): float(mse)
        for index in range(tiny_ab.VALIDATION_IDENTITY_COUNT)
        for raw in tiny_ab.RAW_TIMESTEPS
    }
    validation = _eval_report(values)
    train = _eval_report(
        {
            (f"train_{index}", raw): float(mse)
            for index in range(tiny_ab.TRAIN_IDENTITY_COUNT)
            for raw in tiny_ab.RAW_TIMESTEPS
        }
    )
    train_order = [
        tiny_ab.fairness_key_from_state_record(record)
        for record in train["records"]
    ]
    while len(train_order) < tiny_ab.TARGET_TINY_STEP:
        train_order.extend(copy.deepcopy(train_order))
    return {
        "schema": tiny_ab.TINY_AB_ARM_SCHEMA,
        "arm": "paper_fidelity" if paper_flag else "canonical",
        "mcp_path_kind": (
            "paper_fidelity_clean_residual_mask"
            if paper_flag
            else "canonical_target_only"
        ),
        "paper_fidelity_mcp1_mask": bool(paper_flag),
        "parent_checkpoint_sha256": tiny_ab.PARENT_CHECKPOINT_SHA256,
        "parent_global_step": tiny_ab.PARENT_STEP,
        "initial_parameter_sha256": "params",
        "initial_optimizer_fingerprint_sha256": "optim",
        "initial_rng_fingerprint_sha256": "rng",
        "state_plan_fingerprint_sha256": "state-plan",
        "train_state_order": tuple(train_order[: tiny_ab.TARGET_TINY_STEP]),
        "all_loss_and_grad_finite": True,
        "validation_step0": validation,
        "validation_step200": validation,
        "train_eval_step0": train,
        "train_eval_step200": train,
        "final_state": {
            "schema": f"{tiny_ab.TINY_AB_SCHEMA}_final_state_v1",
            "parameter_updates": {"pass": True},
        },
    }


def _state_record_from_plan_state(state: dict, *, mse: float = 1.0) -> dict:
    return _eval_record(state, mse)


def _old_buggy_fairness_key_from_state_record(record: dict) -> dict:
    proof = record["state_proof"]
    return {
        "state_id": str(record["state_id"]),
        "identity": str(record["identity"]),
        "raw_timestep": int(record["raw_timestep"]),
        "noise_index": int(record["noise_index"]),
        "future_noise_sha256": str(proof["future_noise_sha256"]),
        "future_target_sha256": str(proof["future_target_sha256"]),
        "exact_fm_target_sha256": str(proof["exact_fm_target_sha256"]),
    }


def test_tiny_state_plan_has_32_train_and_32_held_out_states() -> None:
    plan = tiny_ab.build_tiny_ab_plan(_synthetic_plan())
    state_plan = plan["state_plan"]

    assert state_plan["train_state_count"] == 32
    assert state_plan["validation_state_count"] == 32
    assert state_plan["train_validation_identity_disjoint"] is True
    assert set(item["identity"] for item in state_plan["train_states"]).isdisjoint(
        set(item["identity"] for item in state_plan["validation_states"])
    )
    assert [state["raw_timestep"] for state in state_plan["train_states"][:4]] == [
        999,
        750,
        500,
        250,
    ]
    assert [state["selected_split_position"] for state in state_plan["train_states"][::4]] == [
        0,
        292,
        585,
        877,
        1170,
        1462,
        1755,
        2047,
    ]
    assert plan["sample_data_provenance"]["noise_source"] == (
        "formal_sample_source_noise_index0"
    )


def test_fairness_keys_are_unique_for_all_train_and_validation_states() -> None:
    plan = tiny_ab.build_tiny_ab_plan(_synthetic_plan())
    train_records = [
        _state_record_from_plan_state(state)
        for state in plan["state_plan"]["train_states"]
    ]
    validation_records = [
        _state_record_from_plan_state(state)
        for state in plan["state_plan"]["validation_states"]
    ]
    train_keys = [
        tiny_ab.fairness_key_from_state_record(record)
        for record in train_records
    ]
    validation_keys = [
        tiny_ab.fairness_key_from_state_record(record)
        for record in validation_records
    ]
    train_identity_hashes = {
        key["state_identity"]["state_identity_sha256"]
        for key in train_keys
    }
    validation_identity_hashes = {
        key["state_identity"]["state_identity_sha256"]
        for key in validation_keys
    }

    assert len(train_keys) == 32
    assert len(validation_keys) == 32
    assert len(train_identity_hashes) == 32
    assert len(validation_identity_hashes) == 32
    assert train_identity_hashes.isdisjoint(validation_identity_hashes)


def test_fairness_key_contract_distinguishes_state_axes_and_arm_metadata_is_ignored() -> None:
    base = _state_spec()
    base_record = _state_record_from_plan_state(base)
    canonical_key = tiny_ab.fairness_key_from_state_record(
        {**base_record, "arm": "canonical", "paper_fidelity_mcp1_mask": False}
    )
    paper_key = tiny_ab.fairness_key_from_state_record(
        {**base_record, "arm": "paper_fidelity", "paper_fidelity_mcp1_mask": True}
    )
    assert canonical_key == paper_key

    raw_changed = {**base, "raw_timestep": 750, "raw_order": 1}
    identity_changed = {
        **base,
        "identity": "identity_0001",
        "split_index": 1,
        "selected_split_position": 1,
        "selected_identity_order": 1,
    }
    noise_changed = {**base, "noise_index": 1}

    assert tiny_ab.fairness_key_from_state_record(
        _state_record_from_plan_state(raw_changed)
    ) != canonical_key
    assert tiny_ab.fairness_key_from_state_record(
        _state_record_from_plan_state(identity_changed)
    ) != canonical_key
    assert tiny_ab.fairness_key_from_state_record(
        _state_record_from_plan_state(noise_changed)
    ) != canonical_key


def test_fairness_key_fails_closed_without_state_identity_or_with_empty_fields() -> None:
    record = _state_record_from_plan_state(_state_spec())
    missing_identity = dict(record)
    del missing_identity["state_identity"]
    with pytest.raises(RuntimeError, match="missing canonical state_identity"):
        tiny_ab.fairness_key_from_state_record(missing_identity)

    bad_identity = copy.deepcopy(record)
    bad_identity["state_identity"]["identity"] = ""
    with pytest.raises(RuntimeError, match="empty field: identity"):
        tiny_ab.fairness_key_from_state_record(bad_identity)

    bad_proof = copy.deepcopy(record)
    bad_proof["state_proof"]["future_noise_sha256"] = ""
    with pytest.raises(RuntimeError, match="empty field: future_noise_sha256"):
        tiny_ab.fairness_key_from_state_record(bad_proof)


def test_update_schedule_is_fixed_200_step_cycle_without_early_stop() -> None:
    plan = tiny_ab.build_tiny_ab_plan(_synthetic_plan())
    schedule = plan["update_schedule"]

    assert len(schedule) == 200
    assert plan["early_stop_enabled"] is False
    assert schedule[0]["state_id"] == plan["state_plan"]["train_states"][0]["state_id"]
    assert schedule[31]["state_id"] == plan["state_plan"]["train_states"][31]["state_id"]
    assert schedule[32]["state_id"] == plan["state_plan"]["train_states"][0]["state_id"]
    assert schedule[199]["state_id"] == plan["state_plan"]["train_states"][7]["state_id"]


def test_arm_flags_make_mask_flag_the_only_scientific_variable() -> None:
    arms = tiny_ab.arm_specs()

    assert [arm["name"] for arm in arms] == ["canonical", "paper_fidelity"]
    assert [arm["paper_fidelity_mcp1_mask"] for arm in arms] == [False, True]
    assert [arm["mcp_path_kind"] for arm in arms] == [
        "canonical_target_only",
        "paper_fidelity_clean_residual_mask",
    ]
    plan = tiny_ab.build_tiny_ab_plan(_synthetic_plan())
    assert plan["only_variable"] == "paper_fidelity_mcp1_mask"
    assert plan["objective"] == {
        "loss": "MCP1 exact Flow Matching MSE for anchor1 only",
        "anchor_index": 1,
        "current_chunk_index": 1,
        "future_chunk_index": 2,
        "main_loss": False,
        "mcp2_loss": False,
        "mcp3_loss": False,
        "auxiliary_loss": False,
    }


def test_forward_sets_requested_flag_and_uses_only_anchor1_mcp1_loss() -> None:
    generator = FakeForwardGenerator()
    batch = _loss_batch(anchor1_target=7.0)

    result = tiny_ab.run_tiny_anchor1_forward_loss(
        generator,
        conditional_dict={"prompt_embeds": torch.zeros(1, 1, 1)},
        noisy_batch=batch,
        state_spec=_state_spec(),
        paper_fidelity_mcp1_mask=True,
    )

    call = generator.calls[0]
    assert call["paper_fidelity_mcp1_mask"] is True
    assert "direct_clean_context_kv" not in call
    assert len(call["mcp_anchor_inputs"]) == 6
    assert result.loss.item() == pytest.approx((11.0 - 7.0) ** 2)


def test_fixed_raw_batch_uses_formal_noise_index0_and_anchor1_future_chunk() -> None:
    clean = torch.arange(21, dtype=torch.float32).reshape(1, 21, 1, 1, 1)
    noise = (torch.arange(21, dtype=torch.float32) + 100).reshape(1, 21, 1, 1, 1)
    scheduler = FakeScheduler()

    batch = tiny_ab.build_fixed_raw_noisy_batch(
        clean_target=clean,
        source_noise=noise,
        raw_timestep=750,
        scheduler_main=scheduler,
        scheduler_mcp=scheduler,
    )

    expected_future = noise[:, 6:9]
    assert torch.equal(batch.epsilon_mcp_depths[0][:, tiny_ab.ANCHOR_INDEX], expected_future)
    assert torch.equal(
        batch.target_flow_mcp_depths[0][:, tiny_ab.ANCHOR_INDEX],
        expected_future - clean[:, 6:9],
    )
    assert batch.raw_timestep_main.unique().tolist() == [750]
    assert batch.raw_timestep_mcp_depths[0].unique().tolist() == [750]


def test_tiny_update_updates_backbone_patch_fusion_mcp1_only() -> None:
    generator = FakeTrainGenerator()
    before = tiny_ab.parameter_group_sha256_report(generator)
    optimizer = torch.optim.SGD(generator.parameters(), lr=0.01)

    record = tiny_ab.run_tiny_update_on_batch(
        generator=generator,
        optimizer=optimizer,
        conditional_dict={},
        noisy_batch=_loss_batch(anchor1_target=0.0),
        state_spec=_state_spec(),
        paper_fidelity_mcp1_mask=False,
    )
    after = tiny_ab.parameter_group_sha256_report(generator)
    updates = tiny_ab.updated_group_report(before, after)

    assert record["finite"] is True
    assert updates["pass"] is True
    for group in ("backbone", "patch_embedding", "mcp_fusion", "mcp_depth1"):
        assert group in updates["updated_groups"]
    for group in ("mcp_depth2", "mcp_depth3", "main_final_head"):
        assert group in updates["unchanged_groups"]


def test_real_update_record_builder_reproduces_old_state_id_bug_and_fixed_key() -> None:
    plan = tiny_ab.build_tiny_ab_plan(_synthetic_plan())
    state = plan["state_plan"]["train_states"][4]
    generator = FakeTrainGenerator()
    optimizer = torch.optim.SGD(generator.parameters(), lr=0.01)

    record = tiny_ab.run_tiny_update_on_batch(
        generator=generator,
        optimizer=optimizer,
        conditional_dict={},
        noisy_batch=_loss_batch(anchor1_target=0.0),
        state_spec=state,
        paper_fidelity_mcp1_mask=False,
    )

    assert sorted(record.keys()) == [
        "finite",
        "gradient_gate",
        "gradient_report",
        "loss",
        "state_identity",
        "state_proof",
    ]
    assert "state_id" not in record
    with pytest.raises(KeyError, match="state_id"):
        _old_buggy_fairness_key_from_state_record(record)

    fixed_key = tiny_ab.fairness_key_from_state_record(record)
    assert fixed_key["schema"] == tiny_ab.FAIRNESS_KEY_SCHEMA
    assert fixed_key["state_identity"]["state_id"] == state["state_id"]
    assert fixed_key["state_identity"]["split"] == "train"
    assert fixed_key["state_identity"]["raw_timestep"] == 999


def test_state_identity_schema_failure_happens_before_optimizer_step() -> None:
    generator = FakeTrainGenerator()
    before = tiny_ab.parameter_group_sha256_report(generator)
    optimizer = torch.optim.SGD(generator.parameters(), lr=0.01)
    bad_state = _state_spec()
    del bad_state["state_id"]

    with pytest.raises(RuntimeError, match="state identity missing fields"):
        tiny_ab.run_tiny_update_on_batch(
            generator=generator,
            optimizer=optimizer,
            conditional_dict={},
            noisy_batch=_loss_batch(anchor1_target=0.0),
            state_spec=bad_state,
            paper_fidelity_mcp1_mask=False,
        )

    after = tiny_ab.parameter_group_sha256_report(generator)
    assert before["aggregate_sha256"] == after["aggregate_sha256"]


def test_gradient_contract_rejects_mcp2_or_main_head_grad() -> None:
    generator = FakeTrainGenerator()
    for param in generator.parameters():
        param.grad = None
    generator.model.backbone_weight.grad = torch.ones_like(generator.model.backbone_weight)
    generator.model.patch_embedding.weight.grad = torch.ones_like(
        generator.model.patch_embedding.weight
    )
    generator.mcp.fusion.weight.grad = torch.ones_like(generator.mcp.fusion.weight)
    generator.mcp.mcp_modules[0].weight.grad = torch.ones_like(
        generator.mcp.mcp_modules[0].weight
    )
    generator.mcp.mcp_modules[1].weight.grad = torch.ones_like(
        generator.mcp.mcp_modules[1].weight
    )

    with pytest.raises(RuntimeError, match="mcp_depth2:expected_no_grad"):
        tiny_ab.validate_gradient_group_report(tiny_ab.gradient_group_report(generator))

    generator.mcp.mcp_modules[1].weight.grad = None
    generator.model.head.weight.grad = torch.ones_like(generator.model.head.weight)
    with pytest.raises(RuntimeError, match="main_final_head:expected_no_grad"):
        tiny_ab.validate_gradient_group_report(tiny_ab.gradient_group_report(generator))


def test_nonfinite_loss_fails_closed() -> None:
    class NonFiniteGenerator(FakeForwardGenerator):
        def forward_full_sequence_next_forcing(self, **kwargs):
            output = super().forward_full_sequence_next_forcing(**kwargs)
            output.mcp_flow_preds_by_depth[0][:, tiny_ab.ANCHOR_INDEX].fill_(float("nan"))
            return output

    with pytest.raises(RuntimeError, match="non-finite anchor1 loss"):
        tiny_ab.run_tiny_update_on_batch(
            generator=NonFiniteGenerator(),
            optimizer=torch.optim.SGD([torch.nn.Parameter(torch.ones(()))], lr=0.1),
            conditional_dict={},
            noisy_batch=_loss_batch(),
            state_spec=_state_spec(),
            paper_fidelity_mcp1_mask=True,
        )


def test_decision_gates_are_preregistered() -> None:
    identities = [f"val_{index}" for index in range(8)]
    control = _eval_report(
        {(identity, raw): 10.0 for identity in identities for raw in tiny_ab.RAW_TIMESTEPS}
    )
    support_treatment = _eval_report(
        {(identity, raw): 8.8 for identity in identities for raw in tiny_ab.RAW_TIMESTEPS}
    )
    support = tiny_ab.classify_tiny_ab_decision(
        comparison=tiny_ab.compare_eval_aggregates(
            control=control,
            treatment=support_treatment,
        ),
        all_loss_and_grad_finite=True,
    )
    assert support["decision"] == tiny_ab.SUPPORT_PAPER_FIDELITY_MCP1

    bad_treatment = _eval_report(
        {(identity, raw): 10.6 for identity in identities for raw in tiny_ab.RAW_TIMESTEPS}
    )
    no_support = tiny_ab.classify_tiny_ab_decision(
        comparison=tiny_ab.compare_eval_aggregates(control=control, treatment=bad_treatment),
        all_loss_and_grad_finite=True,
    )
    assert no_support["decision"] == tiny_ab.NO_SUPPORT

    weak_treatment = _eval_report(
        {(identity, raw): 9.4 for identity in identities for raw in tiny_ab.RAW_TIMESTEPS}
    )
    inconclusive = tiny_ab.classify_tiny_ab_decision(
        comparison=tiny_ab.compare_eval_aggregates(
            control=control,
            treatment=weak_treatment,
        ),
        all_loss_and_grad_finite=True,
    )
    assert inconclusive["decision"] == tiny_ab.INCONCLUSIVE


def test_summary_schema_and_fairness_proof_cover_parent_rng_and_paired_tensors() -> None:
    plan = tiny_ab.build_tiny_ab_plan(_synthetic_plan())
    control = _arm_result(mse=10.0, paper_flag=False)
    treatment = _arm_result(mse=8.8, paper_flag=True)
    control["state_plan_fingerprint_sha256"] = plan["state_plan"][
        "state_plan_fingerprint_sha256"
    ]
    treatment["state_plan_fingerprint_sha256"] = plan["state_plan"][
        "state_plan_fingerprint_sha256"
    ]

    summary = tiny_ab.build_summary(plan=plan, control=control, treatment=treatment)

    assert tiny_ab.validate_summary_schema(summary)["status"] == "PASS"
    assert summary["status"] == "PASS"
    assert summary["fairness"]["same_parent_checkpoint_sha"] is True
    assert summary["fairness"]["same_parent_global_step"] is True
    assert summary["fairness"]["same_initial_parameter_sha"] is True
    assert summary["fairness"]["same_optimizer_fingerprint"] is True
    assert summary["fairness"]["same_rng_fingerprint"] is True
    assert summary["fairness"]["same_sample_order_and_paired_state_tensors"] is True
    assert summary["decision"] == tiny_ab.SUPPORT_PAPER_FIDELITY_MCP1

    broken = copy.deepcopy(treatment)
    broken["validation_step200"]["records"][0]["state_proof"][
        "exact_fm_target_sha256"
    ] = "different"
    fairness = tiny_ab.compare_arm_fairness(control, broken)
    assert fairness["status"] == "FAIL"
    assert fairness["same_sample_order_and_paired_state_tensors"] is False


def test_dry_run_preregisters_plan_without_real_cuda_or_data_access() -> None:
    args = tiny_ab.parse_args(["--output_dir", str(ROOT / "_unused_tiny_ab_dry_run")])

    summary = tiny_ab.run_tiny_ab(args)

    assert summary["status"] == "DRY_RUN"
    assert summary["plan"]["update_count"] == 200
    assert summary["decision_thresholds_preregistered"]["support"][
        "validation_relative_improvement_min"
    ] == 0.10


def test_tiny_ab_source_avoids_disallowed_paths_and_checkpoint_writes() -> None:
    source_path = ROOT / "scripts" / "train_nf_sf_tiny_mcp1_paper_fidelity_ab.py"
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    imported_modules.update(
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    )

    banned_fragments = (
        "teacher_flow",
        "run_nf_sf_teacher_flow_audit",
        "oracle",
        "verifier",
        "predicted_current",
        "predicted-current",
        "opd",
    )
    assert not any(
        fragment in module.lower()
        for module in imported_modules
        for fragment in banned_fragments
    )
    for fragment in banned_fragments:
        assert fragment not in source.lower()
    assert "run_full_sequence_train_step" not in source
    assert "torch.save" not in source
    assert "atomic_torch_save" not in source
