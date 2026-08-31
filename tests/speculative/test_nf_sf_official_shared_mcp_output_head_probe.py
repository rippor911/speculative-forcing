from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import scripts.probe_nf_sf_official_shared_mcp_output_head as probe


class FakeDepth(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(1, 1, bias=False)
        self.blocks = nn.ModuleList([nn.Linear(1, 1, bias=False)])
        self.head = nn.Linear(1, 1, bias=False)


class FakeMCP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fusion = nn.Linear(1, 1, bias=False)
        self.mcp_modules = nn.ModuleList([FakeDepth() for _ in range(3)])
        self.calls = []

    def forward(self, **kwargs):
        self.calls.append(dict(kwargs))
        return []


class FakeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.head = nn.Linear(1, 1, bias=False)


class FakeGenerator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = FakeModel()
        self.mcp = FakeMCP()


def _set_grad(module: nn.Module, value: float | None) -> None:
    for parameter in module.parameters():
        parameter.grad = None if value is None else torch.full_like(parameter, value)


def _passing_gradient_generator() -> FakeGenerator:
    generator = FakeGenerator()
    _set_grad(generator.model.head, 1.0)
    _set_grad(generator.mcp.fusion, 1.0)
    for module in generator.mcp.mcp_modules:
        _set_grad(module.proj, 1.0)
        _set_grad(module.blocks, 1.0)
        _set_grad(module.head, None)
    return generator


def test_weighted_mcp_only_loss_uses_all_depths_and_excludes_main() -> None:
    main = torch.tensor(100.0)
    losses = SimpleNamespace(
        main_loss=main,
        mcp_depth_losses=(torch.tensor(2.0), torch.tensor(3.0), torch.tensor(5.0)),
        total_loss=main + 0.5 * 2.0 + 0.2 * 3.0 + 0.1 * 5.0,
    )

    value = probe.weighted_mcp_only_loss(losses)

    assert value.item() == pytest.approx(2.1)
    report = probe.loss_report_for_pass(
        losses,
        backward_loss=value,
        pass_kind="mcp_only",
    )
    assert report["main_loss_in_backward"] is False
    assert report["backward_loss"] == pytest.approx(2.1)


def test_probe_gradient_contract_accepts_shared_head_and_dormant_heads() -> None:
    report = probe.gradient_report_for_probe(_passing_gradient_generator())

    gate = probe.validate_gradient_contract(report, pass_kind="mcp_only")

    assert gate["status"] == "PASS"
    assert report["main_final_head"]["aggregate_grad_norm"] > 0.0
    assert report["mcp_depth1_projection"]["aggregate_grad_norm"] > 0.0
    assert report["mcp_depth1_blocks"]["aggregate_grad_norm"] > 0.0
    assert report["dormant_mcp_depth1_independent_head"]["active_grad_tensors"] == 0


def test_probe_gradient_contract_rejects_missing_main_head_or_active_dormant_head() -> None:
    generator = _passing_gradient_generator()
    _set_grad(generator.model.head, None)
    report = probe.gradient_report_for_probe(generator)
    with pytest.raises(probe.ProbeContractError) as exc:
        probe.validate_gradient_contract(report, pass_kind="mcp_only")
    assert exc.value.code == probe.PROBE_FAIL_GRADIENT
    assert "main_final_head" in str(exc.value)

    generator = _passing_gradient_generator()
    _set_grad(generator.mcp.mcp_modules[1].head, 1.0)
    report = probe.gradient_report_for_probe(generator)
    with pytest.raises(probe.ProbeContractError) as exc:
        probe.validate_gradient_contract(report, pass_kind="full_joint")
    assert exc.value.code == probe.PROBE_FAIL_GRADIENT
    assert "dormant_mcp_depth2_independent_head" in str(exc.value)


def test_parameter_fingerprint_survives_backward_and_detects_mutation() -> None:
    module = nn.Linear(2, 1, bias=False)
    before = probe.parameter_fingerprint_report(module)
    module(torch.ones(1, 2)).sum().backward()
    after_backward = probe.parameter_fingerprint_report(module)
    assert probe.compare_parameter_fingerprints(before, after_backward)["unchanged"] is True

    with torch.no_grad():
        module.weight.add_(1.0)
    after_mutation = probe.parameter_fingerprint_report(module)
    assert probe.compare_parameter_fingerprints(before, after_mutation)["unchanged"] is False


def test_shared_head_route_recorder_requires_model_head_identity() -> None:
    generator = FakeGenerator()
    recorder = probe.SharedHeadRouteRecorder()
    recorder.install(generator)
    try:
        recorder.phase = "mcp_only"
        generator.mcp(
            features=(),
            future_embeds=[torch.zeros(1, 1, 1)],
            official_shared_mcp_output_head=True,
            main_output_head=generator.model.head,
        )
    finally:
        recorder.remove()

    gate = probe.validate_route_report(recorder.calls)
    assert gate["status"] == "PASS"
    assert recorder.calls[0]["main_output_head_is_model_head"] is True

    bad = [dict(recorder.calls[0], main_output_head_is_model_head=False)]
    with pytest.raises(probe.ProbeContractError) as exc:
        probe.validate_route_report(bad)
    assert exc.value.code == probe.PROBE_FAIL_ROUTE


def test_dry_run_artifact_records_no_step_shared_head_contract(tmp_path: Path) -> None:
    report = probe.dry_run_artifact(output_dir=tmp_path)

    assert report["status"] == "DRY_RUN"
    assert report["fresh_parent"] == "official_self_forcing_no_mcp_checkpoint"
    assert report["official_shared_mcp_output_head"] is True
    assert report["optimizer_step_executed"] is False
    assert report["checkpoint_written"] is False
    assert report["mcp_only_backward_loss_formula"] == "0.5*MCP1 + 0.2*MCP2 + 0.1*MCP3"
    assert (tmp_path / probe.PROBE_ARTIFACT_FILENAME).is_file()
    assert probe.validate_probe_artifact_schema(report)["status"] == "PASS"


def test_parse_args_dry_run_does_not_require_real_paths(tmp_path: Path) -> None:
    args = probe.parse_args(["--output_dir", str(tmp_path)])

    assert args.execute_real_probe is False
    assert args.checkpoint == Path("checkpoints/self_forcing_dmd.pt")
    assert args.expected_checkpoint_sha256 == probe.OFFICIAL_SELF_FORCING_CHECKPOINT_SHA256
