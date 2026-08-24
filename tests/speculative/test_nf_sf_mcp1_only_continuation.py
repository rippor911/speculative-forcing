from __future__ import annotations

import json
import shutil
import types
import uuid
from pathlib import Path

import pytest
import torch
from torch import nn

from tests.speculative.test_nf_sf_full_sequence_next_forcing import (
    _loss_batch,
    _scheduler,
)
from utils.nf_sf_mcp1_only_continuation import (
    INCONCLUSIVE,
    MCP1_ONLY_FROZEN_GROUPS,
    MCP1_ONLY_TRAINABLE_GROUPS,
    NF_SF_MCP1_ONLY_CONTINUATION_PARENT_STEP,
    NF_SF_MCP1_ONLY_CONTINUATION_SCHEMA,
    NF_SF_MCP1_ONLY_CONTINUATION_TARGET_STEP,
    NF_SF_MCP1_ONLY_CONTINUATION_UPDATE_COUNT,
    NF_SF_MCP1_ONLY_PARENT_CHECKPOINT_SHA256,
    NF_SF_MCP1_ONLY_PARENT_GIT_SHA,
    NO_SUPPORT,
    SUPPORT_JOINT_TRAINING_INTERFERENCE,
    build_mcp1_only_optimizer_from_canonical_state,
    build_mcp1_only_provenance,
    build_mcp1_only_run_plan,
    classify_mcp1_only_comparison,
    compare_parameter_sha256_reports,
    configure_mcp1_only_trainable_parameters,
    forbidden_feature_contract,
    load_matching_control_artifact_bundle,
    mcp1_only_first_step_contract,
    mcp1_only_step_numbers,
    parameter_sha256_report,
    run_mcp1_only_forward_loss,
    trainable_parameter_delta_report,
    trainable_parameter_snapshot,
    validate_matching_control_provenance,
)
from utils.nf_sf_tensors import DEFAULT_S_MAIN, DEFAULT_S_MCP
from utils.nf_sf_training import (
    collect_nf_sf_parameter_groups,
    nf_sf_full_sequence_train_cursor,
)


ROOT = Path(__file__).resolve().parents[2]


class _FakeBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.patch_embedding = nn.Linear(1, 1, bias=False)
        self.backbone_weight = nn.Parameter(torch.tensor([[0.25]]))
        self.num_frame_per_block = 3


class _FakeMCP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fusion = nn.Linear(1, 1, bias=False)
        self.mcp_modules = nn.ModuleList(
            [nn.Linear(1, 1, bias=False) for _ in range(3)]
        )


class _FakeGenerator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _FakeBackbone()
        self.mcp = _FakeMCP()

    def forward_full_sequence_next_forcing(self, **kwargs):
        clean = kwargs["clean_x"]
        batch, _frames, channels, height, width = clean.shape
        seed = torch.ones((1, 1), dtype=clean.dtype, device=clean.device)
        fusion = self.mcp.fusion(seed).reshape(())
        depth_values = [
            self.mcp.mcp_modules[index](seed).reshape(())
            for index in range(3)
        ]
        main = self.model.backbone_weight.reshape(()) * torch.ones_like(clean)
        return types.SimpleNamespace(
            main_flow_pred=main,
            mcp_flow_preds_by_depth=(
                (fusion + depth_values[0])
                * torch.ones((batch, 6, 3, channels, height, width), dtype=clean.dtype),
                (10.0 * fusion + depth_values[1])
                * torch.ones((batch, 5, 3, channels, height, width), dtype=clean.dtype),
                (20.0 * fusion + depth_values[2])
                * torch.ones((batch, 4, 3, channels, height, width), dtype=clean.dtype),
            ),
            tap_shapes=((batch, 32760, 1),) * 4,
            anchor_token_slices=tuple((i * 4680, (i + 1) * 4680) for i in range(7)),
            main_backbone_forward_count=1,
            future_embedding_order="depth_major",
        )


def _canonical_optimizer_with_state(generator: _FakeGenerator) -> torch.optim.AdamW:
    groups = collect_nf_sf_parameter_groups(generator)
    optimizer = torch.optim.AdamW(
        [
            {"name": name, "params": [param for _, param in named], "lr": 1.0e-3}
            for name, named in groups.items()
        ],
        betas=(0.0, 0.999),
        eps=1.0e-8,
        weight_decay=0.01,
    )
    for param in generator.parameters():
        param.requires_grad_(True)
    optimizer.zero_grad(set_to_none=True)
    loss = sum(param.float().sum() for param in generator.parameters())
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return optimizer


def test_plan_and_first_step_contract_are_locked() -> None:
    steps = mcp1_only_step_numbers()
    assert steps[0] == 6501
    assert steps[-1] == 7000
    assert len(steps) == NF_SF_MCP1_ONLY_CONTINUATION_UPDATE_COUNT
    plan = build_mcp1_only_run_plan()
    assert plan.parent_step == NF_SF_MCP1_ONLY_CONTINUATION_PARENT_STEP
    assert plan.target_step == NF_SF_MCP1_ONLY_CONTINUATION_TARGET_STEP
    assert plan.diagnostic_only is True
    assert plan.non_canonical is True
    assert plan.canonical_training_eligible is False
    assert plan.deployment_eligible is False
    first = mcp1_only_first_step_contract(
        train_identity="train-6501",
        sample_cursor=nf_sf_full_sequence_train_cursor(6501),
    )
    assert first["first_global_step"] == 6501
    with pytest.raises(RuntimeError, match="sample_cursor"):
        mcp1_only_first_step_contract(
            train_identity="bad",
            sample_cursor=nf_sf_full_sequence_train_cursor(6502),
        )


def test_only_fusion_and_mcp1_trainable_and_optimizer_inherits_allowed_state() -> None:
    generator = _FakeGenerator()
    canonical_optimizer = _canonical_optimizer_with_state(generator)
    optimizer, selection, report = build_mcp1_only_optimizer_from_canonical_state(
        generator,
        canonical_optimizer,
        mcp_lr=1.0e-3,
        weight_decay=0.01,
    )

    groups = collect_nf_sf_parameter_groups(generator)
    assert tuple(group["name"] for group in optimizer.param_groups) == MCP1_ONLY_TRAINABLE_GROUPS
    assert set(selection.summary["trainable_groups"]) == set(MCP1_ONLY_TRAINABLE_GROUPS)
    assert report["transferred_state_entry_count"] == len(selection.trainable_named_parameters)
    for group_name, named_params in groups.items():
        expected = group_name in MCP1_ONLY_TRAINABLE_GROUPS
        assert all(param.requires_grad is expected for _, param in named_params)
    optimizer_ids = {
        id(param)
        for group in optimizer.param_groups
        for param in group["params"]
    }
    assert optimizer_ids == set(selection.allowed_param_ids)


def test_only_mcp1_loss_participates_and_mcp23_cannot_reach_fusion() -> None:
    batch = _loss_batch()
    generator = _FakeGenerator()
    with torch.no_grad():
        generator.mcp.fusion.weight.fill_(1.0)
        generator.mcp.mcp_modules[0].weight.fill_(-1.0)
        generator.mcp.mcp_modules[1].weight.fill_(3.0)
        generator.mcp.mcp_modules[2].weight.fill_(5.0)
    selection = configure_mcp1_only_trainable_parameters(
        generator,
        mcp_lr=1.0e-3,
        weight_decay=0.01,
    )
    result = run_mcp1_only_forward_loss(
        generator,
        conditional_dict={"prompt_embeds": torch.zeros((1, 1, 1))},
        noisy_batch=batch,
    )

    assert result.total_loss is result.mcp1_loss
    assert result.main_observation_loss.requires_grad is False
    assert all(loss.requires_grad is False for loss in result.mcp_depth_observation_losses)
    assert float(result.mcp1_loss.detach().item()) == pytest.approx(0.0)
    result.total_loss.backward()

    grad_by_name = {
        name: param.grad
        for name, param in selection.trainable_named_parameters
    }
    assert all(
        grad is None or float(grad.detach().abs().max().item()) == 0.0
        for grad in grad_by_name.values()
    )


class _Sample:
    def __init__(self, target: torch.Tensor) -> None:
        self.target_latent = target
        self.source_noise = torch.zeros_like(target)


class _Acquire:
    def __init__(self, value) -> None:
        self.value = value

    def __enter__(self):
        return self.value

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


class _TinyTeacherStore:
    def __init__(self, target: torch.Tensor) -> None:
        self.target = target

    def train_identity_for_step(self, step: int) -> str:
        assert int(step) == 6501
        return "train0"

    def acquire(self, identity: str):
        return _Acquire(_Sample(self.target.clone()))


class _TinyConditionalStore:
    def acquire(self, identity: str):
        return _Acquire({"prompt_embeds": torch.zeros((1, 1, 1))})


def _cpu_helpers():
    def capture_global_rng_state(device):
        assert torch.device(device).type == "cpu"
        return {"cpu": torch.get_rng_state().detach().cpu().clone()}

    def assert_global_rng_equal(before, after):
        assert before.keys() == after.keys()
        for key in before:
            assert torch.equal(before[key], after[key])

    def assert_finite_loss(loss, name):
        assert name == "mcp1_only_loss"
        assert torch.isfinite(loss.detach()).all()

    return {
        "capture_global_rng_state": capture_global_rng_state,
        "assert_global_rng_equal": assert_global_rng_equal,
        "assert_finite_loss": assert_finite_loss,
        "target_latent_from_sample": lambda sample: sample.target_latent,
    }


def test_train_step_freezes_main_patch_mcp23_and_preserves_rng_contract() -> None:
    from scripts import train_nf_sf_mcp1_only_continuation as runner

    target = torch.zeros((1, 21, 1, 1, 1), dtype=torch.float32)
    generator = _FakeGenerator()
    selection = configure_mcp1_only_trainable_parameters(
        generator,
        mcp_lr=0.1,
        weight_decay=0.0,
    )
    optimizer = torch.optim.AdamW(
        list(selection.optimizer_param_groups),
        betas=(0.0, 0.999),
        eps=1.0e-8,
        weight_decay=0.0,
    )
    frozen_before = parameter_sha256_report(generator, groups=MCP1_ONLY_FROZEN_GROUPS)
    trainable_before = trainable_parameter_snapshot(selection.trainable_named_parameters)
    train_rng = torch.Generator(device="cpu")
    train_rng.manual_seed(17)

    record = runner.run_mcp1_only_train_step(
        helpers=_cpu_helpers(),
        generator=generator,
        optimizer=optimizer,
        scheduler_main=_scheduler(DEFAULT_S_MAIN),
        scheduler_mcp=_scheduler(DEFAULT_S_MCP),
        teacher_store=_TinyTeacherStore(target),
        conditional_store=_TinyConditionalStore(),
        train_rng=train_rng,
        global_step=6501,
        device=torch.device("cpu"),
        dtype=torch.float32,
        allowed_param_ids=selection.allowed_param_ids,
        capture_draw_fingerprint=True,
    )

    frozen_after = parameter_sha256_report(generator, groups=MCP1_ONLY_FROZEN_GROUPS)
    unchanged = compare_parameter_sha256_reports(frozen_before, frozen_after)
    delta = trainable_parameter_delta_report(
        selection.trainable_named_parameters,
        trainable_before,
    )
    assert unchanged["all_sha256_exact_match"] is True
    assert delta["aggregate_l2"] > 0.0
    assert record["sample_cursor"] == nf_sf_full_sequence_train_cursor(6501)
    assert record["train_rng_before_sha256"] != record["train_rng_after_sha256"]
    assert "draw_fingerprint" in record
    assert record["optimizer_isolation"]["optimizer_ids_equal_allowed_ids"] is True
    assert record["gradient_audit"]["forbidden_gradients_absent_or_zero"] is True


def _matching_control() -> dict:
    return {
        "schema": NF_SF_MCP1_ONLY_CONTINUATION_SCHEMA,
        "status": "PASS",
        "arm": "control",
        "parent_step": 6500,
        "target_step": 7000,
        "train_record_count": 500,
        "parent_checkpoint": {
            "sha256": NF_SF_MCP1_ONLY_PARENT_CHECKPOINT_SHA256,
            "git_sha": NF_SF_MCP1_ONLY_PARENT_GIT_SHA,
        },
        "run_plan": {
            "first_step": 6501,
            "target_step": 7000,
            "update_count": 500,
        },
        "first_continuation_step": {
            "first_sample_cursor": nf_sf_full_sequence_train_cursor(6501),
        },
        "validation": {
            "global_step": 7000,
            "paired_identity_noise_across_steps": True,
        },
        "fixed_probe": {
            "raw_timestep": 999,
            "depth": 1,
        },
    }


def test_matching_control_provenance_validation() -> None:
    audit = validate_matching_control_provenance(_matching_control())
    assert audit["CONTROL_REUSABLE"] is True

    bad = _matching_control()
    bad["run_plan"] = {**bad["run_plan"], "first_step": 6502}
    audit = validate_matching_control_provenance(bad)
    assert audit["CONTROL_REUSABLE"] is False
    assert any("first_step" in failure for failure in audit["failures"])


def _hex(char: str, length: int) -> str:
    return str(char) * int(length)


def _direct_control_summary(checkpoint_sha: str = _hex("7", 64)) -> dict:
    return {
        "schema": "nf_sf_mcp_direct_clean_kv_ablation_v1",
        "status": "PASS",
        "arm": "control",
        "parent_step": 6500,
        "target_step": 7000,
        "train_record_count": 500,
        "validation": {
            "global_step": 7000,
            "paired_identity_noise_across_steps": True,
        },
        "fixed_probe": {
            "raw_timestep": 999,
            "depth": 1,
        },
        "checkpoint_sha256": checkpoint_sha,
    }


def _direct_control_metadata(runtime_git_sha: str = _hex("8", 40)) -> dict:
    rng = {
        "train_rng_state_sha256": _hex("a", 64),
        "validation_base_rng_state_sha256": _hex("b", 64),
        "python_random_state_sha256": _hex("c", 64),
        "torch_cpu_global_rng_state_sha256": _hex("d", 64),
        "torch_cuda_global_rng_state_sha256": _hex("e", 64),
    }
    return {
        "schema": "nf_sf_mcp_direct_clean_kv_ablation_v1",
        "run_plan": {
            "arm": "control",
            "first_step": 6501,
            "target_step": 7000,
            "update_count": 500,
            "direct_clean_context_kv": False,
        },
        "provenance": {
            "schema": "nf_sf_mcp_direct_clean_kv_ablation_v1",
            "arm": "control",
            "parent_step": 6500,
            "target_step": 7000,
            "update_count": 500,
            "parent_checkpoint_sha256": NF_SF_MCP1_ONLY_PARENT_CHECKPOINT_SHA256,
            "parent_git_sha": NF_SF_MCP1_ONLY_PARENT_GIT_SHA,
            "runtime_git_sha": runtime_git_sha,
            "direct_clean_context_kv": False,
            "data_changed": False,
            "rng_changed": False,
            "objective_changed": False,
            "optimizer_changed": False,
        },
        "restore_contract": {
            "status": "PASS",
            "rng_fingerprint": rng,
        },
    }


def _direct_control_metrics(
    *,
    first_step: int = 6501,
    count: int = 500,
) -> list[dict]:
    records = [
        {
            "schema": "nf_sf_mcp_direct_clean_kv_ablation_v1_validation_v1",
            "global_step": 7000,
        }
    ]
    for step in range(first_step, first_step + count):
        records.append(
            {
                "schema": "nf_sf_mcp_direct_clean_kv_ablation_v1",
                "arm": "control",
                "global_step": step,
                "sample_cursor": nf_sf_full_sequence_train_cursor(step),
                "sample_identity": f"sample-{step}",
                "train_rng_before_sha256": _hex("1", 64),
                "train_rng_after_sha256": _hex("2", 64),
            }
        )
    records.append({"schema": "other", "global_step": 7000})
    return records


def _write_direct_control_bundle(tmp_path: Path, mutator=None) -> Path:
    checkpoint_sha = _hex("7", 64)
    payloads = {
        "summary": _direct_control_summary(checkpoint_sha=checkpoint_sha),
        "run_metadata": _direct_control_metadata(),
        "metrics": _direct_control_metrics(),
        "checkpoint_validation": {
            "status": "PASS",
            "global_step": 7000,
            "sha256": checkpoint_sha,
        },
    }
    if mutator is not None:
        mutator(payloads)
    paths = {
        "summary": tmp_path / "ablation_summary.json",
        "run_metadata": tmp_path / "run_metadata.json",
        "metrics": tmp_path / "metrics.jsonl",
        "checkpoint_validation": tmp_path / "checkpoint_step007000.validation.json",
    }
    if payloads.get("summary") is not None:
        paths["summary"].write_text(
            json.dumps(payloads["summary"]),
            encoding="utf-8",
        )
    if payloads.get("run_metadata") is not None:
        paths["run_metadata"].write_text(
            json.dumps(payloads["run_metadata"]),
            encoding="utf-8",
        )
    if payloads.get("metrics") is not None:
        paths["metrics"].write_text(
            "\n".join(json.dumps(record) for record in payloads["metrics"]) + "\n",
            encoding="utf-8",
        )
    if payloads.get("checkpoint_validation") is not None:
        paths["checkpoint_validation"].write_text(
            json.dumps(payloads["checkpoint_validation"]),
            encoding="utf-8",
        )
    return paths["summary"]


@pytest.fixture
def control_bundle_dir():
    parent = ROOT / "_codex_tmp_nf_sf_mcp1_control_bundle_tests"
    path = parent / uuid.uuid4().hex
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def test_direct_clean_control_bundle_with_companion_provenance_passes(
    control_bundle_dir: Path,
) -> None:
    summary_path = _write_direct_control_bundle(control_bundle_dir)

    audit = load_matching_control_artifact_bundle(summary_path)

    assert audit["CONTROL_REUSABLE"] is True
    assert audit["control_summary_path"].endswith("ablation_summary.json")
    assert audit["run_metadata_path"].endswith("run_metadata.json")
    assert audit["metrics_path"].endswith("metrics.jsonl")
    assert audit["checkpoint_validation_path"].endswith(
        "checkpoint_step007000.validation.json"
    )
    assert len(audit["source_sha256"]) == 4
    assert all(len(value) == 64 for value in audit["source_sha256"].values())


@pytest.mark.parametrize(
    ("mutator", "match"),
    [
        (
            lambda payloads: payloads.__setitem__("run_metadata", None),
            "run_metadata",
        ),
        (
            lambda payloads: payloads["run_metadata"]["provenance"].__setitem__(
                "parent_checkpoint_sha256",
                _hex("0", 64),
            ),
            "parent_checkpoint_sha256",
        ),
        (
            lambda payloads: payloads["run_metadata"]["provenance"].__setitem__(
                "parent_git_sha",
                _hex("0", 40),
            ),
            "parent_git_sha",
        ),
        (
            lambda payloads: payloads.__setitem__(
                "metrics",
                _direct_control_metrics(first_step=6502, count=500),
            ),
            "global_step sequence",
        ),
        (
            lambda payloads: next(
                record
                for record in payloads["metrics"]
                if record.get("schema") == "nf_sf_mcp_direct_clean_kv_ablation_v1"
            ).__setitem__(
                "sample_cursor",
                nf_sf_full_sequence_train_cursor(6502),
            ),
            "first sample_cursor",
        ),
        (
            lambda payloads: payloads.__setitem__(
                "metrics",
                _direct_control_metrics(count=499),
            ),
            "record count",
        ),
        (
            lambda payloads: payloads["checkpoint_validation"].__setitem__(
                "sha256",
                _hex("9", 64),
            ),
            "checkpoint validation SHA",
        ),
    ],
)
def test_direct_clean_control_bundle_fail_closed_cases(
    control_bundle_dir: Path,
    mutator,
    match: str,
) -> None:
    summary_path = _write_direct_control_bundle(control_bundle_dir, mutator=mutator)

    audit = load_matching_control_artifact_bundle(summary_path)

    assert audit["CONTROL_REUSABLE"] is False
    assert any(match in failure for failure in audit["failures"])


def test_decision_thresholds_are_preregistered() -> None:
    support = classify_mcp1_only_comparison(
        baseline_step6500_mse=0.12,
        control_step7000_mse=0.10,
        treatment_step7000_mse=0.089,
        control_paired_mcp1_mse=0.20,
        treatment_paired_mcp1_mse=0.19,
    )
    assert support["decision"] == SUPPORT_JOINT_TRAINING_INTERFERENCE

    no_support = classify_mcp1_only_comparison(
        baseline_step6500_mse=0.12,
        control_step7000_mse=0.10,
        treatment_step7000_mse=0.096,
        control_paired_mcp1_mse=0.20,
        treatment_paired_mcp1_mse=0.199,
    )
    assert no_support["decision"] == NO_SUPPORT

    inconclusive = classify_mcp1_only_comparison(
        baseline_step6500_mse=0.12,
        control_step7000_mse=0.10,
        treatment_step7000_mse=0.093,
        control_paired_mcp1_mse=0.20,
        treatment_paired_mcp1_mse=0.19,
    )
    assert inconclusive["decision"] == INCONCLUSIVE


def test_forbidden_features_and_provenance_noncanonical_flags() -> None:
    forbidden = forbidden_feature_contract()
    assert forbidden
    assert all(value is False for value in forbidden.values())
    provenance = build_mcp1_only_provenance(
        runtime_git_sha="c3f89888bf6da31b48650f0a680dd6534943f56f",
        semantic_lock_fingerprint="abc",
    )
    assert provenance["schema"] == NF_SF_MCP1_ONLY_CONTINUATION_SCHEMA
    assert provenance["diagnostic_only"] is True
    assert provenance["non_canonical"] is True
    assert provenance["canonical_training_eligible"] is False
    assert provenance["deployment_eligible"] is False
    assert provenance["dataset_changed"] is False
    assert provenance["rng_semantics_changed"] is False
    assert provenance["timestep_distribution_changed"] is False
    assert provenance["noise_distribution_changed"] is False
    assert provenance["trainable_scope_changed"] is True
    assert provenance["loss_scope_changed"] is True


def test_dry_run_and_canonical_trainer_source_unchanged() -> None:
    from scripts import train_nf_sf_mcp1_only_continuation as runner
    from utils.nf_sf_training import validate_nf_sf_full_sequence_objective_mode

    args = runner.parse_args(
        [
            "--parent_checkpoint",
            "checkpoint_step006500.pt",
            "--expected_runtime_git_sha",
            "c3f89888bf6da31b48650f0a680dd6534943f56f",
            "--output_dir",
            str(ROOT.parent / "nf_sf_mcp1_only_outside_repo"),
        ]
    )
    summary = runner.run_mcp1_only_continuation(args)
    assert summary["dry_run"] is True
    assert summary["diagnostic_only"] is True
    assert summary["canonical_training_eligible"] is False
    assert summary["control_reuse_audit"]["CONTROL_REUSABLE"] is False

    with pytest.raises(ValueError):
        validate_nf_sf_full_sequence_objective_mode("mcp1_only_continuation")
    canonical_source = (ROOT / "scripts" / "train_nf_sf_full_sequence_next_forcing.py").read_text(
        encoding="utf-8"
    )
    assert "mcp1_only_continuation" not in canonical_source
