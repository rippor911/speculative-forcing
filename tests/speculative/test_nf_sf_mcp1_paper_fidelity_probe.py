from __future__ import annotations

import types
from pathlib import Path

import pytest
import torch
from torch import nn

import scripts.probe_nf_sf_mcp1_paper_fidelity as probe
from utils.nf_sf_training import (
    NFSFFullSequenceNoisyBatch,
    build_full_sequence_mcp_anchor_specs,
)


ROOT = Path(__file__).resolve().parents[2]


def _loss_batch() -> NFSFFullSequenceNoisyBatch:
    clean = torch.zeros((1, 21, 1, 1, 1), dtype=torch.float32)
    return NFSFFullSequenceNoisyBatch(
        clean_target=clean,
        noisy_main=clean.clone(),
        target_flow_main=torch.zeros_like(clean),
        epsilon_main=torch.zeros_like(clean),
        raw_timestep_main=torch.zeros((1, 7), dtype=torch.int64),
        timestep_main=torch.zeros((1, 21), dtype=torch.float32),
        noisy_mcp_depths=(
            torch.zeros((1, 6, 3, 1, 1, 1)),
            torch.zeros((1, 5, 3, 1, 1, 1)),
            torch.zeros((1, 4, 3, 1, 1, 1)),
        ),
        target_flow_mcp_depths=(
            torch.zeros((1, 6, 3, 1, 1, 1)),
            torch.zeros((1, 5, 3, 1, 1, 1)),
            torch.zeros((1, 4, 3, 1, 1, 1)),
        ),
        epsilon_mcp_depths=(
            torch.zeros((1, 6, 3, 1, 1, 1)),
            torch.zeros((1, 5, 3, 1, 1, 1)),
            torch.zeros((1, 4, 3, 1, 1, 1)),
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
        mcp1 = torch.ones((1, 6, 3, 1, 1, 1), dtype=torch.float32)
        mcp2 = torch.full((1, 5, 3, 1, 1, 1), 2.0, dtype=torch.float32)
        mcp3 = torch.full((1, 4, 3, 1, 1, 1), 3.0, dtype=torch.float32)
        return types.SimpleNamespace(
            main_flow_pred=torch.full((1, 21, 1, 1, 1), 7.0),
            mcp_flow_preds_by_depth=(mcp1, mcp2, mcp3),
            tap_shapes=((1, 1, 1),),
            anchor_token_slices=((0, 4680),),
            main_backbone_forward_count=1,
            future_embedding_order="depth_major",
        )


class FakeBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.patch_embedding = nn.Linear(1, 1, bias=False)
        self.backbone_weight = nn.Parameter(torch.ones(()))
        self.head = nn.Linear(1, 1)


class FakeMCP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fusion = nn.Linear(1, 1, bias=False)
        self.mcp_modules = nn.ModuleList(
            [nn.Linear(1, 1, bias=False) for _ in range(3)]
        )


class FakeGradientGenerator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = FakeBackbone()
        self.mcp = FakeMCP()


def _set_grad(param: torch.nn.Parameter, value: float) -> None:
    param.grad = torch.full_like(param, float(value))


def _passing_gradient_generator() -> FakeGradientGenerator:
    generator = FakeGradientGenerator()
    _set_grad(generator.model.backbone_weight, 1.0)
    _set_grad(generator.model.patch_embedding.weight, 1.0)
    _set_grad(generator.mcp.fusion.weight, 1.0)
    _set_grad(generator.mcp.mcp_modules[0].weight, 1.0)
    return generator


def _fusion_record(
    *,
    anchor_index: int,
    role: str,
    token_count: int,
    grad_value: float | None,
) -> dict:
    output = torch.ones((1, max(int(token_count), 1), 1), requires_grad=True)
    if grad_value is not None:
        output.grad = torch.full_like(output, float(grad_value))
    return {
        "anchor_index": int(anchor_index),
        "role": role,
        "token_count": int(token_count),
        "requires_grad": True,
        "output": output,
    }


def _passing_fusion_records() -> list[dict]:
    records: list[dict] = []
    for contract in probe.build_six_anchor_plan():
        anchor_index = int(contract["anchor_index"])
        records.append(
            _fusion_record(
                anchor_index=anchor_index,
                role="noisy_current_h_fuse",
                token_count=int(contract["target_token_count"]),
                grad_value=1.0,
            )
        )
        if int(contract["clean_token_count"]) > 0:
            records.append(
                _fusion_record(
                    anchor_index=anchor_index,
                    role="clean_h_fuse",
                    token_count=int(contract["clean_token_count"]),
                    grad_value=1.0,
                )
            )
    return records


def _valid_anchor_reports() -> list[dict]:
    return [
        {
            **dict(contract),
            "flow_shape": [1, 3, 1, 1, 1],
            "finite": True,
            "exact_fm_mse": 0.0,
        }
        for contract in probe.build_six_anchor_plan()
    ]


def test_six_anchor_plan_and_token_counts() -> None:
    plan = probe.build_six_anchor_plan()

    assert len(plan) == 6
    assert [item["clean_token_count"] for item in plan] == [
        0,
        4680,
        9360,
        14040,
        18720,
        23400,
    ]
    assert [item["total_token_count"] for item in plan] == [
        4680,
        9360,
        14040,
        18720,
        23400,
        28080,
    ]
    assert [item["future_start_frame"] for item in plan] == [3, 6, 9, 12, 15, 18]


def test_probe_forward_sets_paper_flag_and_uses_only_mcp1_exact_loss() -> None:
    generator = FakeForwardGenerator()
    batch = _loss_batch()

    result = probe.run_paper_fidelity_mcp1_forward_loss(
        generator,
        conditional_dict={"prompt_embeds": torch.zeros(1, 1, 1)},
        noisy_batch=batch,
    )

    call = generator.calls[0]
    assert call["paper_fidelity_mcp1_mask"] is True
    assert "direct_clean_context_kv" not in call
    assert len(call["mcp_anchor_inputs"]) == 6
    assert [loss.item() for loss in result.mcp1_anchor_losses] == [1.0] * 6
    assert result.probe_loss.item() == pytest.approx(1.0)


def test_anchor_report_rejects_nonfinite_anchor() -> None:
    reports = _valid_anchor_reports()
    reports[2]["finite"] = False

    with pytest.raises(probe.ProbeContractError) as exc:
        probe.validate_anchor_reports(reports)

    assert exc.value.code == probe.PROBE_FAIL_NONFINITE


def test_gradient_expectations_accept_mcp1_probe_pattern() -> None:
    report = probe.gradient_report_for_probe(_passing_gradient_generator())

    gate = probe.validate_gradient_report_for_probe(report)

    assert gate["status"] == "PASS"
    assert report["mcp_depth2"]["grad_tensors"] == 0
    assert report["mcp_depth3"]["grad_tensors"] == 0
    assert report["main_final_head"]["grad_tensors"] == 0
    assert report["backbone"]["aggregate_grad_norm"] > 0.0


def test_gradient_expectations_reject_mcp2_or_main_head_grad() -> None:
    generator = _passing_gradient_generator()
    _set_grad(generator.mcp.mcp_modules[1].weight, 1.0)
    report = probe.gradient_report_for_probe(generator)

    with pytest.raises(probe.ProbeContractError) as exc:
        probe.validate_gradient_report_for_probe(report)

    assert exc.value.code == probe.PROBE_FAIL_GRADIENT
    assert "mcp_depth2:expected_no_grad" in str(exc.value)

    generator = _passing_gradient_generator()
    _set_grad(generator.model.head.weight, 1.0)
    report = probe.gradient_report_for_probe(generator)
    with pytest.raises(probe.ProbeContractError) as exc:
        probe.validate_gradient_report_for_probe(report)

    assert exc.value.code == probe.PROBE_FAIL_GRADIENT
    assert "main_final_head:expected_no_grad" in str(exc.value)


def test_gradient_expectations_reject_nonfinite_grad() -> None:
    generator = _passing_gradient_generator()
    _set_grad(generator.model.backbone_weight, float("nan"))
    report = probe.gradient_report_for_probe(generator)

    with pytest.raises(probe.ProbeContractError) as exc:
        probe.validate_gradient_report_for_probe(report)

    assert exc.value.code == probe.PROBE_FAIL_NONFINITE


def test_clean_path_gradient_gate_requires_anchor1_to5_clean_grad() -> None:
    report = probe.build_clean_path_report(_passing_fusion_records())

    probe.validate_clean_path_report(report)
    assert report["status"] == "PASS"
    assert report["clean_path_required_anchor_indices"] == [1, 2, 3, 4, 5]
    assert report["anchor_reports"][0]["clean_h_fuse"]["present"] is False
    assert report["anchor_reports"][5]["clean_h_fuse"]["grad"]["norm"] > 0.0


def test_clean_path_gradient_gate_fails_closed_on_zero_clean_grad() -> None:
    records = _passing_fusion_records()
    for record in records:
        if record["anchor_index"] == 3 and record["role"] == "clean_h_fuse":
            record["output"].grad.zero_()

    report = probe.build_clean_path_report(records)
    with pytest.raises(probe.ProbeContractError) as exc:
        probe.validate_clean_path_report(report)

    assert exc.value.code == probe.PROBE_FAIL_CLEAN_PATH
    assert report["clean_path_failure_anchor_indices"] == [3]


def test_parameter_sha_unchanged_contract_detects_backward_vs_mutation() -> None:
    module = nn.Linear(2, 1)
    before = probe.parameter_sha256_report(module)
    module(torch.ones(1, 2)).sum().backward()
    after_backward = probe.parameter_sha256_report(module)

    unchanged = probe.parameter_sha_unchanged_report(before, after_backward)
    assert unchanged["unchanged"] is True

    with torch.no_grad():
        module.weight.add_(1.0)
    after_mutation = probe.parameter_sha256_report(module)
    changed = probe.parameter_sha_unchanged_report(before, after_mutation)
    assert changed["unchanged"] is False


def test_probe_artifact_schema_and_safety_flags() -> None:
    report = probe.build_probe_artifact(
        status="PASS",
        runtime_git_sha="a" * 40,
        checkpoint_provenance={
            "parent_checkpoint_path": str(probe.STEP6500_PARENT_CHECKPOINT),
            "parent_checkpoint_sha256": probe.STEP6500_PARENT_CHECKPOINT_SHA256,
            "parent_global_step": probe.STEP6500_PARENT_STEP,
        },
        anchors=_valid_anchor_reports(),
        gradient_report=probe.gradient_report_for_probe(_passing_gradient_generator()),
        gradient_gate={"status": "PASS"},
        clean_path_report=probe.build_clean_path_report(_passing_fusion_records()),
        memory_report={
            "snapshots": {
                "before_model": {"label": "before_model", "cuda": False},
                "after_model_load": {"label": "after_model_load", "cuda": False},
                "after_forward": {"label": "after_forward", "cuda": False},
                "after_backward": {"label": "after_backward", "cuda": False},
                "after_cleanup": {"label": "after_cleanup", "cuda": False},
            },
            "anchors": [],
            "anchor5": None,
            "overall_peak": {"cuda": False, "max_allocated": 0, "max_reserved": 0, "total": 0},
        },
        safety_report={
            "optimizer_step_executed": False,
            "checkpoint_written": False,
            "parameter_sha_unchanged": True,
            "trainable_parameter_count_before": 12,
            "trainable_parameter_count_after": 12,
            "trainable_parameter_count_unchanged": True,
        },
        sample_identity="train_000000",
        sample_cursor={"global_step": 6501, "sample_position": 452},
    )

    validation = probe.validate_probe_artifact_schema(report)
    assert validation["status"] == "PASS"
    assert report["status"] == probe.PROBE_PASS_LABEL
    assert report["paper_fidelity_mcp1_mask"] is True
    assert report["canonical_path_modified_by_probe"] is False
    assert report["optimizer_step_executed"] is False
    assert report["checkpoint_written"] is False
    assert report["parameter_sha_unchanged"] is True


def test_probe_source_contains_no_optimizer_step_or_checkpoint_write() -> None:
    source = (ROOT / "scripts" / "probe_nf_sf_mcp1_paper_fidelity.py").read_text(
        encoding="utf-8"
    )

    assert "optimizer.step(" not in source
    assert "torch.save" not in source
    assert "atomic_torch_save" not in source
    assert "run_full_sequence_train_step" not in source
