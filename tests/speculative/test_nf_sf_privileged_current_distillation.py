from __future__ import annotations

import argparse
import inspect
import sys
import types
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch import nn


def _install_wan_import_stubs() -> None:
    if "wan.modules.causal_model" in sys.modules:
        return

    wan = types.ModuleType("wan")
    wan_modules = types.ModuleType("wan.modules")
    tokenizers = types.ModuleType("wan.modules.tokenizers")
    model = types.ModuleType("wan.modules.model")
    vae = types.ModuleType("wan.modules.vae")
    t5 = types.ModuleType("wan.modules.t5")
    causal_model = types.ModuleType("wan.modules.causal_model")
    mcp = types.ModuleType("wan.modules.mcp")

    class DummyTokenizer:
        def __init__(self, *args, **kwargs) -> None:
            pass

    class DummyWanModel(nn.Module):
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

    class DummyCausalWanModel(DummyWanModel):
        pass

    class DummyRegisterTokens(nn.Module):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__()

    class DummyGanAttentionBlock(nn.Module):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__()

    class DummyMCPStack(nn.Module):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__()

        def init_from_backbone(self, blocks) -> None:
            _ = blocks

    tokenizers.HuggingfaceTokenizer = DummyTokenizer
    model.WanModel = DummyWanModel
    model.RegisterTokens = DummyRegisterTokens
    model.GanAttentionBlock = DummyGanAttentionBlock
    vae._video_vae = lambda *args, **kwargs: nn.Identity()
    t5.umt5_xxl = lambda *args, **kwargs: nn.Identity()
    causal_model.CausalWanModel = DummyCausalWanModel
    mcp.MCPStack = DummyMCPStack
    mcp.MCP_INPUT_TIMESTEP = 1000

    sys.modules["wan"] = wan
    sys.modules["wan.modules"] = wan_modules
    sys.modules["wan.modules.tokenizers"] = tokenizers
    sys.modules["wan.modules.model"] = model
    sys.modules["wan.modules.vae"] = vae
    sys.modules["wan.modules.t5"] = t5
    sys.modules["wan.modules.causal_model"] = causal_model
    sys.modules["wan.modules.mcp"] = mcp


_install_wan_import_stubs()

import scripts.train_nf_sf_privileged_current_distillation as runner
import utils.nf_sf_full_sequence_eval as ev
import utils.nf_sf_privileged_current_distillation as pcd
import utils.nf_sf_teacher_flow_audit as teacher_audit
from utils.nf_sf_tensors import (
    DEFAULT_S_MAIN,
    DEFAULT_S_MCP,
    FULL_SEQUENCE_CHUNK_FRAMES,
    FULL_SEQUENCE_DEPTHS,
    FULL_SEQUENCE_FRAME_COUNT,
    FULL_SEQUENCE_NUM_CHUNKS,
    make_cpu_generator,
)
from utils.nf_sf_training import (
    build_full_sequence_mcp_anchor_specs,
    prepare_nf_sf_full_sequence_noisy_batch,
    run_nf_sf_full_sequence_forward_loss,
)


FRAME_SEQ_LENGTH = 1


class FakeBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.patch_embedding = nn.Conv3d(1, 1, kernel_size=1, bias=False)
        self.backbone_weight = nn.Parameter(torch.tensor(0.5))
        self.head = FakeMainHead()
        self.num_frame_per_block = FULL_SEQUENCE_CHUNK_FRAMES


class FakeMainHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.modulation = nn.Parameter(torch.ones(1, 2, 1))
        self.head = nn.Linear(1, 1)


class FakeMCPModule(nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.head = nn.Module()
        self.head.head = nn.Linear(1, 1, bias=False)
        nn.init.constant_(self.head.head.weight, value)

    @property
    def weight(self) -> torch.Tensor:
        return self.head.head.weight


class FakeMCP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fusion = nn.Linear(1, 1, bias=False)
        nn.init.constant_(self.fusion.weight, 0.2)
        self.mcp_modules = nn.ModuleList(
            [FakeMCPModule(0.4), FakeMCPModule(0.6), FakeMCPModule(0.8)]
        )


class FakeStudentGenerator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = FakeBackbone()
        self.mcp = FakeMCP()
        self.calls: list[dict] = []

    def forward_full_sequence_next_forcing(
        self,
        *,
        noisy_image_or_video,
        clean_x,
        conditional_dict,
        timestep_main,
        mcp_anchor_inputs=(),
    ):
        _ = (clean_x, conditional_dict, timestep_main)
        self.calls.append({"anchor_count": len(mcp_anchor_inputs)})
        main_scale = (
            self.model.backbone_weight
            + self.model.patch_embedding.weight.reshape(()).to(noisy_image_or_video)
            + self.model.head.modulation.sum().to(noisy_image_or_video)
            + self.model.head.head.weight.reshape(()).to(noisy_image_or_video)
            + self.model.head.head.bias.reshape(()).to(noisy_image_or_video)
        )
        main = noisy_image_or_video * main_scale
        by_depth: list[torch.Tensor] = []
        shared_feature_scale = (
            self.model.backbone_weight
            + self.model.patch_embedding.weight.reshape(()).to(noisy_image_or_video)
        )
        for depth in FULL_SEQUENCE_DEPTHS:
            chunks = []
            for anchor in mcp_anchor_inputs:
                depths = tuple(anchor["depths"])
                if depth not in depths:
                    continue
                local = depths.index(depth)
                future = anchor["future_noises"][local]
                scale = (
                    self.mcp.mcp_modules[depth - 1].weight.reshape(())
                    * (1.0 + self.mcp.fusion.weight.reshape(()))
                    + 0.1 * shared_feature_scale
                ).to(future)
                chunks.append(torch.ones_like(future) * scale)
            by_depth.append(torch.stack(chunks, dim=1))
        return types.SimpleNamespace(
            main_flow_pred=main,
            mcp_flow_preds_by_depth=tuple(by_depth),
            tap_shapes=((1, 1, 1),),
            anchor_token_slices=((0, 1),),
            main_backbone_forward_count=1,
            future_embedding_order="depth_major",
        )


class FakeTeacherGenerator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.mcp = None
        self.calls: list[dict] = []

    def forward(self, **kwargs):
        chunk = kwargs["noisy_image_or_video"]
        timestep = kwargs["timestep"]
        current_start = int(kwargs["current_start"])
        kv_cache = kwargs["kv_cache"]
        self.calls.append(
            {
                "current_start": current_start,
                "chunk_sha256": ev.tensor_sha256(chunk.detach().cpu()),
                "timestep": float(timestep.flatten()[0].detach().cpu().item()),
            }
        )
        token_count = int(chunk.shape[1]) * FRAME_SEQ_LENGTH
        token_end = current_start + token_count
        for layer in kv_cache:
            layer["k"][:, current_start:token_end] = len(self.calls)
            layer["v"][:, current_start:token_end] = len(self.calls) + 1
            layer["global_end_index"].fill_(token_end)
            layer["local_end_index"].fill_(token_end)
        return chunk * 0.3, chunk


def _clean_target() -> torch.Tensor:
    return torch.linspace(
        -0.5,
        0.5,
        FULL_SEQUENCE_FRAME_COUNT,
        dtype=torch.float32,
    ).reshape(1, FULL_SEQUENCE_FRAME_COUNT, 1, 1, 1)


def _source_noise() -> torch.Tensor:
    return torch.linspace(
        0.25,
        1.25,
        FULL_SEQUENCE_FRAME_COUNT,
        dtype=torch.float32,
    ).reshape(1, FULL_SEQUENCE_FRAME_COUNT, 1, 1, 1)


def _scheduler(shift: float):
    return teacher_audit.build_flow_match_scheduler(
        shift=shift,
        device=torch.device("cpu"),
    )


def _noisy_batch():
    return prepare_nf_sf_full_sequence_noisy_batch(
        _clean_target(),
        scheduler_main=_scheduler(DEFAULT_S_MAIN),
        scheduler_mcp=_scheduler(DEFAULT_S_MCP),
        rng=make_cpu_generator(17),
    )


def _runtime(generator: FakeTeacherGenerator) -> ev.DeploymentRuntime:
    capacity = FULL_SEQUENCE_FRAME_COUNT * FRAME_SEQ_LENGTH
    kv_cache = [
        {
            "k": torch.zeros((1, capacity, 1, 1), dtype=torch.float32),
            "v": torch.zeros((1, capacity, 1, 1), dtype=torch.float32),
            "global_end_index": torch.tensor([0], dtype=torch.long),
            "local_end_index": torch.tensor([0], dtype=torch.long),
        }
    ]
    return ev.DeploymentRuntime(
        generator=generator,
        scheduler=_scheduler(DEFAULT_S_MAIN),
        kv_cache=kv_cache,
        crossattn_cache=[{"is_init": False}],
        frame_seq_length=FRAME_SEQ_LENGTH,
        num_frame_per_block=FULL_SEQUENCE_CHUNK_FRAMES,
        context_noise=0,
    )


def _teacher_factory(generator: FakeTeacherGenerator):
    return lambda: _runtime(generator)


def _conditional() -> dict[str, torch.Tensor]:
    return {"prompt_embeds": torch.zeros((1, 2, 3), dtype=torch.float32)}


def _teacher_payload() -> dict[str, int]:
    return {"rollout_seed": 1234}


def _run_forward(lambda_priv: float = pcd.PRIVILEGED_CURRENT_LAMBDA):
    student = FakeStudentGenerator()
    teacher = FakeTeacherGenerator().eval().requires_grad_(False)
    batch = _noisy_batch()
    result = pcd.run_privileged_current_forward_loss(
        student,
        teacher_runtime_factory=_teacher_factory(teacher),
        conditional_dict=_conditional(),
        noisy_batch=batch,
        source_noise=_source_noise(),
        teacher_payload=_teacher_payload(),
        mcp_scheduler=_scheduler(DEFAULT_S_MCP),
        lambda_priv=lambda_priv,
    )
    return student, teacher, batch, result


def _canonical_and_privileged_grad_reports():
    student, _, _, result = _run_forward(lambda_priv=0.0)
    result.canonical_loss.backward()
    canonical = pcd.gradient_group_report(student)

    student, _, _, result = _run_forward(lambda_priv=0.0)
    result.privileged_loss.backward()
    privileged = pcd.gradient_group_report(student)
    comparison = pcd.compare_gradient_reports(
        canonical,
        privileged,
        lambda_priv=0.25,
    )
    return canonical, privileged, comparison


def test_canonical_objective_contract_is_not_modified() -> None:
    plan = pcd.privileged_run_plan()
    provenance = pcd.provenance_contract()

    assert plan["canonical_objective_unchanged"] is True
    assert plan["inference_graph_changed"] is False
    assert provenance["exact_fm_replaced"] is False
    assert provenance["depth_weights"] == [0.5, 0.2, 0.1]


def test_lambda_fixed_for_formal_contract() -> None:
    assert pcd.validate_lambda_priv(0.25, formal=True) == 0.25
    with pytest.raises(RuntimeError, match="lambda=0.25"):
        pcd.validate_lambda_priv(0.5, formal=True)
    assert "half the direct MCP1" in pcd.privileged_run_plan()["lambda_rationale"]


def test_only_mcp1_gets_privileged_auxiliary_loss() -> None:
    _, _, batch, result = _run_forward()

    assert result.loss_record["auxiliary_depths"] == [1]
    assert tuple(result.teacher_targets.target_flows.shape) == tuple(
        batch.noisy_mcp_depths[0].shape
    )
    assert len(result.canonical.mcp_flow_preds_by_depth) == 3
    assert "mcp2_privileged_auxiliary_loss" in pcd.provenance_contract()["forbidden"]
    assert "mcp3_privileged_auxiliary_loss" in pcd.provenance_contract()["forbidden"]


def test_teacher_frozen_main_only_report_rejects_trainable_or_mcp() -> None:
    teacher = FakeTeacherGenerator().eval().requires_grad_(False)
    report = pcd.teacher_frozen_report(teacher)
    assert report["eval_mode"] is True
    assert report["requires_grad_false"] is True
    assert report["mcp_tensor_count"] == 0

    trainable = FakeTeacherGenerator().eval()
    trainable.weight.requires_grad_(True)
    with pytest.raises(RuntimeError, match="trainable"):
        pcd.teacher_frozen_report(trainable)

    with_mcp = FakeTeacherGenerator().eval().requires_grad_(False)
    with_mcp.mcp = nn.Linear(1, 1).requires_grad_(False)
    with pytest.raises(RuntimeError, match="Main-only"):
        pcd.teacher_frozen_report(with_mcp)


def test_mcp1_anchor_semantics_exact_future_sigma_and_context_boundary() -> None:
    batch = _noisy_batch()
    semantics = pcd.mcp1_anchor_semantics(
        noisy_batch=batch,
        mcp_scheduler=_scheduler(DEFAULT_S_MCP),
    )

    assert len(semantics) == FULL_SEQUENCE_NUM_CHUNKS - 1
    assert [item.anchor_index for item in semantics] == list(range(6))
    assert semantics[0].recache_chunk_indices == (0,)
    assert semantics[-1].recache_chunk_indices == (0, 1, 2, 3, 4, 5)
    assert semantics[-1].future_chunk_index == 6
    for item in semantics:
        proof = item.proof
        assert proof["same_future_tensor_as_student"] is True
        assert proof["future_clean_leakage"] is False
        assert proof["raw_timestep_directly_used_for_teacher"] is False
        assert proof["teacher_timestep"] == pytest.approx(
            item.physical_sigma * 1000.0
        )
        assert torch.equal(
            item.future_state,
            batch.noisy_mcp_depths[0][:, item.anchor_index],
        )


def test_teacher_target_route_marks_privileged_and_preserves_rng() -> None:
    teacher = FakeTeacherGenerator().eval().requires_grad_(False)
    before = pcd.parameter_sha256_report(teacher)["aggregate_sha256"]
    targets = pcd.build_privileged_mcp1_teacher_targets(
        teacher_runtime_factory=_teacher_factory(teacher),
        noisy_batch=_noisy_batch(),
        source_noise=_source_noise(),
        teacher_payload=_teacher_payload(),
        conditional_dict=_conditional(),
        mcp_scheduler=_scheduler(DEFAULT_S_MCP),
    )
    after = pcd.parameter_sha256_report(teacher)["aggregate_sha256"]

    assert targets.rng_guard["unchanged"] is True
    assert targets.target_flows.requires_grad is False
    assert before == after
    assert len(targets.anchor_records) == 6
    for record in targets.anchor_records:
        proof = record["proof"]
        assert proof["privileged_clean_current"] is True
        assert proof["same_information_as_mcp"] is False
        assert proof["history_context_latent_exact_clean"] is False
        assert proof["uses_ground_truth_future_x0_for_conversion"] is False
        assert proof["uses_wrapper_auto_x0"] is False
        assert proof["future_forward_rng"]["unchanged"] is True
        assert max(proof["recache_chunk_indices"]) == record["anchor_index"]
        assert record["future_chunk_index"] == record["anchor_index"] + 1


def test_lambda_zero_reproduces_canonical_numerical_objective() -> None:
    student, _, batch, privileged = _run_forward(lambda_priv=0.0)
    canonical = run_nf_sf_full_sequence_forward_loss(
        student,
        conditional_dict=_conditional(),
        noisy_batch=batch,
        objective_mode="next_forcing_full",
    )

    assert float(privileged.total_loss.detach()) == pytest.approx(
        float(privileged.canonical_loss.detach())
    )
    assert float(privileged.total_loss.detach()) == pytest.approx(
        float(canonical.losses.total_loss.detach())
    )


def test_gradient_report_and_comparison_cover_required_groups() -> None:
    canonical, _, comparison = _canonical_and_privileged_grad_reports()

    assert set(comparison) == {"backbone", "patch_embedding", "mcp_fusion", "mcp_depth1"}
    assert comparison["mcp_depth1"]["lambda_scaled_aux_to_canonical_norm_ratio"] is not None
    stripped = pcd.strip_gradient_flats(canonical)
    assert all("_flat" not in item for item in stripped.values())
    assert all("_full_flat" not in item for item in stripped.values())
    assert all("_by_name" not in item for item in stripped.values())


def test_one_side_missing_grad_has_full_space_cosine_and_missing_names() -> None:
    canonical, privileged, comparison = _canonical_and_privileged_grad_reports()
    backbone = comparison["backbone"]
    missing = backbone["one_side_missing_grad_parameters"]

    assert canonical["backbone"]["missing_grad_tensors"] == 0
    assert privileged["backbone"]["missing_grad_tensors"] == 3
    assert [item["name"] for item in missing] == [
        "model.head.modulation",
        "model.head.head.weight",
        "model.head.head.bias",
    ]
    assert all(item["canonical_grad_present"] is True for item in missing)
    assert all(item["privileged_grad_present"] is False for item in missing)
    assert backbone["legacy_present_gradient_cosine"] is None
    assert backbone["full_parameter_space_cosine"] is not None


def test_shared_only_cosine_matches_manual_shared_parameter_space() -> None:
    canonical, privileged, comparison = _canonical_and_privileged_grad_reports()
    shared_left = []
    shared_right = []
    for item in canonical["backbone"]["parameters"]:
        name = item["name"]
        left = canonical["backbone"]["_by_name"][name]["grad"]
        right = privileged["backbone"]["_by_name"][name]["grad"]
        if left is not None and right is not None:
            shared_left.append(left)
            shared_right.append(right)
    expected = F.cosine_similarity(
        torch.cat(shared_left),
        torch.cat(shared_right),
        dim=0,
    ).item()

    assert comparison["backbone"]["shared_nonzero_gradient_cosine"] == (
        pytest.approx(expected)
    )


def test_full_grad_groups_new_cosines_match_legacy_cosine() -> None:
    _, _, comparison = _canonical_and_privileged_grad_reports()

    for group in ("patch_embedding", "mcp_fusion", "mcp_depth1"):
        item = comparison[group]
        assert item["one_side_missing_grad_parameter_count"] == 0
        assert item["full_parameter_space_cosine"] == pytest.approx(
            item["legacy_present_gradient_cosine"]
        )
        assert item["shared_nonzero_gradient_cosine"] == pytest.approx(
            item["legacy_present_gradient_cosine"]
        )


def test_zero_fill_full_space_keeps_existing_norms() -> None:
    canonical, privileged, comparison = _canonical_and_privileged_grad_reports()

    for group, item in comparison.items():
        assert item["canonical_full_parameter_space_norm"] == pytest.approx(
            canonical[group]["norm"]
        )
        assert item["privileged_full_parameter_space_norm"] == pytest.approx(
            privileged[group]["norm"]
        )
        assert item["full_space_norm_matches_present_norm"] is True


def test_optimizer_fingerprint_changes_only_after_optimizer_step() -> None:
    model = nn.Linear(1, 1)
    optimizer = torch.optim.AdamW([{"params": list(model.parameters()), "name": "w"}])
    before = pcd.optimizer_state_fingerprint(optimizer)
    loss = model(torch.ones(1, 1)).sum()
    loss.backward()
    after_backward = pcd.optimizer_state_fingerprint(optimizer)
    optimizer.step()
    after_step = pcd.optimizer_state_fingerprint(optimizer)

    assert after_backward == before
    assert after_step != before


def test_first_step_contract_uses_canonical_cursor() -> None:
    cursor = runner.nf_sf_full_sequence_train_cursor(6501)
    report = pcd.first_step_contract(6501, cursor)
    assert report["status"] == "PASS"
    with pytest.raises(RuntimeError, match="first step"):
        pcd.first_step_contract(6502, runner.nf_sf_full_sequence_train_cursor(6502))


def test_control_reuse_fails_closed_and_can_record_strict_pass(monkeypatch) -> None:
    assert pcd.validate_control_reuse(None)["CONTROL_REUSABLE"] is False

    def fake_loader(path):
        assert str(path) == "summary.json"
        return {"CONTROL_REUSABLE": True, "failures": [], "status": "PASS"}

    monkeypatch.setattr(pcd, "load_matching_control_artifact_bundle", fake_loader)
    report = pcd.validate_control_reuse("summary.json")
    assert report["CONTROL_REUSABLE"] is True
    assert report["required_update_count"] == 500
    assert report["requires_no_direct_clean_context_kv"] is True


def test_teacher_forward_target_is_rng_invariant() -> None:
    _, _, _, result = _run_forward()

    assert result.teacher_targets.rng_guard["unchanged"] is True
    for record in result.teacher_targets.anchor_records:
        assert record["proof"]["future_forward_rng"]["unchanged"] is True


def test_no_inference_or_deployment_graph_change_static_contract() -> None:
    utility = Path("utils/nf_sf_privileged_current_distillation.py").read_text(
        encoding="utf-8"
    )
    script = Path("scripts/train_nf_sf_privileged_current_distillation.py").read_text(
        encoding="utf-8"
    )
    assert "run_nf_sf_full_sequence_forward_loss" in utility
    assert "run_nf_sf_full_sequence_forward_loss" in script
    assert "deployment_eligible\": False" in utility
    assert "inference_mcp" not in utility
    assert "decode_to_pixel" not in utility


def test_gradient_probe_source_contains_no_optimizer_step() -> None:
    source = inspect.getsource(runner.run_gradient_safety_probe)
    assert "optimizer.step(" not in source
    assert "\"optimizer_step_executed\": False" in source
    assert "optimizer_state_fingerprint" in source


def test_ab_decision_thresholds_are_preregistered() -> None:
    support = pcd.classify_privileged_current_ab(
        control_raw999_mcp1_mse=1.0,
        treatment_raw999_mcp1_mse=0.89,
        control_validation_mcp1_mse=1.0,
        treatment_validation_mcp1_mse=0.94,
        control_main_mse=1.0,
        treatment_main_mse=1.04,
    )
    assert support["decision"] == pcd.STRONG_SUPPORT

    no_support = pcd.classify_privileged_current_ab(
        control_raw999_mcp1_mse=1.0,
        treatment_raw999_mcp1_mse=0.96,
        control_validation_mcp1_mse=1.0,
        treatment_validation_mcp1_mse=0.96,
        control_main_mse=1.0,
        treatment_main_mse=1.00,
    )
    assert no_support["decision"] == pcd.NO_SUPPORT

    main_bad = pcd.classify_privileged_current_ab(
        control_raw999_mcp1_mse=1.0,
        treatment_raw999_mcp1_mse=0.50,
        control_validation_mcp1_mse=1.0,
        treatment_validation_mcp1_mse=0.50,
        control_main_mse=1.0,
        treatment_main_mse=1.05,
    )
    assert main_bad["decision"] == pcd.NO_SUPPORT

    mixed = pcd.classify_privileged_current_ab(
        control_raw999_mcp1_mse=1.0,
        treatment_raw999_mcp1_mse=0.93,
        control_validation_mcp1_mse=1.0,
        treatment_validation_mcp1_mse=0.94,
        control_main_mse=1.0,
        treatment_main_mse=1.00,
    )
    assert mixed["decision"] == pcd.INCONCLUSIVE


def test_dry_run_summary_records_provenance_and_is_non_deployable() -> None:
    args = argparse.Namespace(
        lambda_priv=0.25,
        matching_control_summary=None,
        execute_real_run=False,
        parent_checkpoint=Path("checkpoint_step006500.pt"),
        output_dir=Path("out"),
    )
    summary = runner.run_privileged_current_distillation(args)

    assert summary["status"] == "DRY_RUN"
    assert summary["diagnostic_only"] is True
    assert summary["deployment_eligible"] is False
    assert summary["run_plan"]["update_count"] == 500
    assert summary["provenance"]["teacher_trained"] is False
    assert summary["control_reuse_audit"]["CONTROL_REUSABLE"] is False


def test_real_run_static_contract_requires_canonical_config_and_teacher_sha(
    monkeypatch,
) -> None:
    args = argparse.Namespace(
        config=runner.CANONICAL_CONFIG_PATH,
        device="cuda:0",
        dtype="bf16",
        log_interval=1,
        memory_log_interval=1,
        teacher_checkpoint=Path("teacher.pt"),
        checkpoint=Path("teacher.pt"),
    )
    config = types.SimpleNamespace(num_frame_per_block=3, context_noise=0)
    monkeypatch.setattr(
        runner,
        "file_sha256",
        lambda path: pcd.PRIVILEGED_CURRENT_TEACHER_CHECKPOINT_SHA256,
    )
    runner._validate_real_run_static_contract(args, config)

    args.device = "cpu"
    with pytest.raises(RuntimeError, match="cuda:0"):
        runner._validate_real_run_static_contract(args, config)


def test_checkpoint_summary_contract_is_noncanonical_and_has_teacher_sha() -> None:
    metadata = {
        "run_plan": pcd.privileged_run_plan(),
        "provenance": pcd.provenance_contract(),
        "control_reuse_audit": {"CONTROL_REUSABLE": True},
    }
    teacher = FakeTeacherGenerator().eval().requires_grad_(False)
    teacher_report = pcd.teacher_frozen_report(teacher)
    summary = runner.build_training_summary(
        metadata=metadata,
        metrics_path=Path("metrics.jsonl"),
        train_record_count=500,
        final_record={"losses": {"privileged_mcp1_loss": 0.1}},
        checkpoint_records=({"path": "checkpoint_step007000.pt"},),
        validation_summaries=({"path": "validation.json"},),
        fixed_probe_path=Path("probe.json"),
        fixed_probe={"status": "PASS"},
        gradient_probe={"status": "PASS", "optimizer_step_executed": False},
        teacher_before=teacher_report,
        teacher_after=teacher_report,
        memory_maxima={"allocated_bytes": 123},
    )

    assert summary["status"] == "DONE"
    assert summary["canonical_training_eligible"] is False
    assert summary["deployment_eligible"] is False
    assert summary["teacher_sha_unchanged"] is True
    assert summary["train_record_count"] == 500


def test_anchor_specs_remain_canonical_full_sequence() -> None:
    specs = build_full_sequence_mcp_anchor_specs()
    assert [(spec.depth, spec.anchor_index) for spec in specs[:6]] == [
        (1, 0),
        (1, 1),
        (1, 2),
        (1, 3),
        (1, 4),
        (1, 5),
    ]
    assert pcd.privileged_run_plan()["auxiliary_depths"] == [1]
