from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import utils.nf_sf_full_sequence_eval as ev
import utils.nf_sf_teacher_flow_audit as audit
from scripts import run_nf_sf_teacher_flow_audit as runner
from utils.nf_sf_tensors import DEFAULT_S_MAIN, DEFAULT_S_MCP


RUNTIME_GIT_SHA = "f" * 40
TRAINING_GIT_SHA = "e" * 40
TEST_SHA = "a" * 64
FRAME_SEQ_LENGTH = 1


class FakeMCP(nn.Module):
    def forward(
        self,
        *,
        features,
        future_embeds,
        future_grid_sizes,
        future_start_frames,
        timesteps,
        freqs=None,
    ):
        _ = (features, future_grid_sizes, future_start_frames, timesteps, freqs)
        return [
            torch.zeros(
                (embed.shape[0], 1, 3, 1, 1),
                device=embed.device,
                dtype=embed.dtype,
            )
            for embed in future_embeds
        ]


class FakeModel:
    def __init__(self) -> None:
        self.block_mask = None


@dataclass
class FakeFullSequenceOutputs:
    main_flow_pred: torch.Tensor
    mcp_flow_preds_by_depth: tuple[torch.Tensor, ...]
    tap_shapes: tuple[tuple[int, ...], ...] = ((1, 1, 1),)
    anchor_token_slices: tuple[tuple[int, int], ...] = ((0, 1),)
    future_embedding_order: str = "depth_major"
    main_backbone_forward_count: int = 1


class FakeStudentGenerator(nn.Module):
    def __init__(self, *, flow_scale: float = 0.25) -> None:
        super().__init__()
        self.model = FakeModel()
        self.mcp = FakeMCP()
        self.flow_scale = float(flow_scale)
        self.forward_calls = []
        self.serial_forward_calls = 0

    def forward(self, **kwargs):
        _ = kwargs
        self.serial_forward_calls += 1
        raise RuntimeError("student audit must not use deployment forward")

    def forward_full_sequence_next_forcing(
        self,
        *,
        noisy_image_or_video,
        clean_x,
        conditional_dict,
        timestep_main,
        mcp_anchor_inputs=(),
        aug_t=None,
        direct_clean_context_kv=False,
    ):
        _ = (clean_x, conditional_dict, timestep_main, aug_t, direct_clean_context_kv)
        self.forward_calls.append(tuple(mcp_anchor_inputs))
        self.model.block_mask = torch.ones(1, dtype=torch.bool)
        by_depth: dict[int, list[torch.Tensor]] = {1: [], 2: [], 3: []}
        for anchor in mcp_anchor_inputs:
            current = noisy_image_or_video[
                :,
                int(anchor["anchor_index"]) * 3:(int(anchor["anchor_index"]) + 1) * 3,
            ]
            features = (current.reshape(current.shape[0], -1, 1),)
            future_embeds = [
                future.reshape(future.shape[0], -1, 1)
                for future in anchor["future_noises"]
            ]
            self.mcp(
                features=features,
                future_embeds=future_embeds,
                future_grid_sizes=[
                    torch.tensor([1, 1, 1], device=current.device)
                    for _ in future_embeds
                ],
                future_start_frames=list(anchor["future_start_frames"]),
                timesteps=list(anchor["timesteps"]),
                freqs=None,
            )
            for depth, future in zip(anchor["depths"], anchor["future_noises"]):
                by_depth[int(depth)].append(future * self.flow_scale)
        return FakeFullSequenceOutputs(
            main_flow_pred=noisy_image_or_video * 0.0,
            mcp_flow_preds_by_depth=tuple(
                torch.stack(by_depth[depth], dim=1) for depth in (1, 2, 3)
            ),
        )


class FakeTeacherGenerator(nn.Module):
    def __init__(self, *, matched_scale: float = 0.2, clean_scale: float = 0.1) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(()), requires_grad=False)
        self.mcp = None
        self.matched_scale = float(matched_scale)
        self.clean_scale = float(clean_scale)
        self.calls: list[dict[str, Any]] = []
        self.current_means: list[float] = []

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
        if current_start == 3 * FRAME_SEQ_LENGTH:
            self.current_means.append(float(chunk.detach().float().mean().item()))
        scale = (
            self.clean_scale
            if current_start == 6 * FRAME_SEQ_LENGTH and len(self.current_means) == 0
            else self.matched_scale
        )
        token_count = int(chunk.shape[1]) * FRAME_SEQ_LENGTH
        token_end = current_start + token_count
        for layer in kv_cache:
            layer["k"][:, current_start:token_end] = len(self.calls)
            layer["v"][:, current_start:token_end] = len(self.calls) + 1
            layer["global_end_index"].fill_(token_end)
            layer["local_end_index"].fill_(token_end)
        return chunk * scale, chunk


def make_runtime(generator: FakeTeacherGenerator) -> ev.DeploymentRuntime:
    capacity = ev.FULL_SEQUENCE_FRAME_COUNT * FRAME_SEQ_LENGTH
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
        scheduler=audit.build_flow_match_scheduler(
            shift=DEFAULT_S_MAIN,
            device=torch.device("cpu"),
        ),
        kv_cache=kv_cache,
        crossattn_cache=[{"is_init": False}],
        frame_seq_length=FRAME_SEQ_LENGTH,
        num_frame_per_block=ev.FULL_SEQUENCE_CHUNK_FRAMES,
        context_noise=0,
    )


def make_source_noise() -> torch.Tensor:
    return torch.linspace(
        -1.0,
        1.0,
        ev.FULL_SEQUENCE_FRAME_COUNT,
        dtype=torch.float32,
    ).reshape(1, ev.FULL_SEQUENCE_FRAME_COUNT, 1, 1, 1)


def make_teacher_target() -> torch.Tensor:
    return torch.linspace(
        0.25,
        2.25,
        ev.FULL_SEQUENCE_FRAME_COUNT,
        dtype=torch.float32,
    ).reshape(1, ev.FULL_SEQUENCE_FRAME_COUNT, 1, 1, 1)


def make_common(source_noise: torch.Tensor, teacher_target: torch.Tensor) -> tuple[dict, str]:
    common = {
        "runtime_git_sha": RUNTIME_GIT_SHA,
        "training_checkpoint_git_sha": TRAINING_GIT_SHA,
        "source_noise_sha256": ev.tensor_sha256(source_noise),
        "conditioning_sha256": ev.conditioning_json_summary(
            {"prompt_embeds": torch.zeros((1, 2, 3))}
        )["sha256"],
        "teacher_target_sha256": ev.tensor_sha256(teacher_target),
        "sample_plan_sha256": "b" * 64,
        "teacher_manifest_sha256": "c" * 64,
    }
    return common, ev.canonical_json_sha256(common)


def make_result(
    *,
    student: FakeStudentGenerator | None = None,
    teacher_factory=None,
) -> audit.TeacherFlowAuditResult:
    source = make_source_noise()
    target = make_teacher_target()
    common, fingerprint = make_common(source, target)
    main_scheduler = audit.build_flow_match_scheduler(
        shift=DEFAULT_S_MAIN,
        device=torch.device("cpu"),
    )
    mcp_scheduler = audit.build_flow_match_scheduler(
        shift=DEFAULT_S_MCP,
        device=torch.device("cpu"),
    )
    return audit.run_teacher_flow_audit(
        student_generator=student or FakeStudentGenerator(),
        teacher_runtime_factory=teacher_factory
        or (lambda: make_runtime(FakeTeacherGenerator())),
        source_noise=source,
        teacher_target=target,
        teacher_payload={"rollout_seed": 123, "prompt": "prompt"},
        conditional_dict={"prompt_embeds": torch.zeros((1, 2, 3))},
        main_scheduler=main_scheduler,
        mcp_scheduler=mcp_scheduler,
        sample_identity="validation-zero",
        checkpoint_summary={
            "checkpoint_type": "teacher_flow_student_step7000",
            "load_mode": audit.TEACHER_FLOW_AUDIT_STUDENT_LOAD_MODE,
            "sha256": TEST_SHA,
            "global_step": 7000,
        },
        teacher_summary={
            "checkpoint_type": "official_self_forcing",
            "checkpoint_sha256": "d" * 64,
            "mcp_tensor_count": 0,
            "eval_mode": True,
            "requires_grad_false": True,
        },
        common_inputs=common,
        common_inputs_fingerprint_sha256=fingerprint,
        runtime_git_sha=RUNTIME_GIT_SHA,
        training_checkpoint_git_sha=TRAINING_GIT_SHA,
    )


def test_16_state_deterministic_contract() -> None:
    source = make_source_noise()
    target = make_teacher_target()
    states = audit.build_teacher_flow_audit_states(
        source_noise=source,
        teacher_target=target,
        main_scheduler=audit.build_flow_match_scheduler(
            shift=DEFAULT_S_MAIN,
            device="cpu",
        ),
        mcp_scheduler=audit.build_flow_match_scheduler(
            shift=DEFAULT_S_MCP,
            device="cpu",
        ),
    )
    repeat = audit.build_teacher_flow_audit_states(
        source_noise=source,
        teacher_target=target,
        main_scheduler=audit.build_flow_match_scheduler(
            shift=DEFAULT_S_MAIN,
            device="cpu",
        ),
        mcp_scheduler=audit.build_flow_match_scheduler(
            shift=DEFAULT_S_MCP,
            device="cpu",
        ),
    )
    assert len(states) == 16
    assert [state.state_id for state in states] == [state.state_id for state in repeat]
    assert [state.provenance["future_state"]["sha256"] for state in states] == [
        state.provenance["future_state"]["sha256"] for state in repeat
    ]
    assert {state.raw_timestep for state in states} == {999, 750, 500, 250}
    assert all(0 <= state.noise_index <= 3 for state in states)


def test_same_future_state_exact_across_student_and_teacher_branches() -> None:
    result = make_result()
    for record in result.manifest["states"]:
        proof = record["same_state_sigma_proof"]
        assert proof["all_future_states_exact"] is True
        assert proof["student_future_state_exact"] is True
        assert proof["teacher_matched_future_state_exact"] is True
        assert proof["teacher_clean_future_state_exact"] is True
        assert record["teacher_matched_current"]["proof"][
            "history_chunk0_identity_exact"
        ] is True
        assert record["teacher_clean_current"]["proof"][
            "history_chunk0_identity_exact"
        ] is True


def test_matched_current_state_exact() -> None:
    result = make_result()
    for record in result.manifest["states"]:
        proof = record["teacher_matched_current"]["proof"]
        assert proof["matched_current_state_exact"] is True
        assert proof["current_state_sha256"] == proof["mcp_current_state_sha256"]


def test_main_sigma_and_future_sigma_are_separate() -> None:
    result = make_result()
    for record in result.manifest["states"]:
        assert record["main_sigma"] != record["future_sigma"]
        assert record["main_warped_timestep"] != record["future_warped_timestep"]


def test_teacher_timestep_uses_same_physical_sigma() -> None:
    result = make_result()
    for record in result.manifest["states"]:
        expected = record["future_sigma"] * 1000.0
        assert record["teacher_future_timestep"] == pytest.approx(expected)
        assert record["teacher_matched_current"]["proof"][
            "future_timestep_matches_physical_sigma"
        ] is True
        assert record["same_state_sigma_proof"][
            "raw_timestep_directly_used_for_teacher"
        ] is False


def test_manual_x0_conversion() -> None:
    future_state = torch.tensor([[[[[2.0]]]]])
    flow = torch.tensor([[[[[0.5]]]]])
    x0 = audit.manual_flow_to_x0(
        future_state=future_state,
        flow=flow,
        sigma=0.25,
        name="unit_x0",
    )
    assert x0.item() == pytest.approx(1.875)


def test_oracle_target_cannot_be_marked_as_teacher() -> None:
    result = make_result()
    manifest = result.manifest
    for record in manifest["states"]:
        assert record["teacher_matched_current"]["proof"][
            "uses_ground_truth_future_x0_for_conversion"
        ] is False
        assert record["teacher_clean_current"]["proof"][
            "uses_ground_truth_future_x0_for_conversion"
        ] is False
    assert any("oracle" in item for item in manifest["forbidden_comparisons"])


def test_privileged_branch_is_explicitly_marked() -> None:
    result = make_result()
    for record in result.manifest["states"]:
        clean_proof = record["teacher_clean_current"]["proof"]
        matched_proof = record["teacher_matched_current"]["proof"]
        assert clean_proof["privileged_clean_current"] is True
        assert clean_proof["same_information_as_mcp"] is False
        assert matched_proof["privileged_clean_current"] is False
        assert matched_proof["same_information_as_mcp"] is True


def test_matched_and_privileged_routes_are_not_confused() -> None:
    result = make_result()
    contracts = result.manifest["teacher_routes"]
    assert contracts[audit.TEACHER_MATCHED_CURRENT_BRANCH][
        "privileged_clean_current"
    ] is False
    assert contracts[audit.TEACHER_CLEAN_CURRENT_BRANCH][
        "privileged_clean_current"
    ] is True
    assert result.manifest["conversion_contract"]["uses_wrapper_auto_x0"] is False


def test_no_optimizer_backward_or_checkpoint_write_contract() -> None:
    result = make_result()
    manifest = result.manifest
    assert manifest["uses_optimizer"] is False
    assert manifest["runs_backward"] is False
    assert manifest["writes_checkpoint"] is False


def test_diagnostic_only_non_deployable() -> None:
    result = make_result()
    assert result.manifest["diagnostic_only"] is True
    assert result.manifest["non_deployable"] is True
    assert result.manifest["canonical_training_eligible"] is False
    assert result.manifest["canonical_deployment_eligible"] is False


def test_metric_aggregation_mean_and_max() -> None:
    result = make_result()
    states = result.manifest["states"]
    all_metrics = result.manifest["aggregates"]["all_states"]
    values = [record["metrics"]["mcp_flow_vs_exact_mse"] for record in states]
    assert all_metrics["mcp_flow_vs_exact_mse"]["mean"] == pytest.approx(
        sum(values) / len(values)
    )
    assert all_metrics["mcp_flow_vs_exact_mse"]["max"] == pytest.approx(max(values))
    for raw in ("999", "750", "500", "250"):
        assert raw in result.manifest["aggregates"]["by_raw"]


def test_diagnostic_label_uses_conservative_mixed_policy() -> None:
    records = [
        {
            "metrics": {
                "mcp_flow_vs_exact_mse": 2.0,
                "mcp_x0_vs_clean_future_mse": 2.0,
                "teacher_matched_flow_vs_exact_mse": 1.0,
                "teacher_matched_x0_vs_clean_future_mse": 1.0,
                "teacher_clean_flow_vs_exact_mse": 3.0,
                "teacher_clean_x0_vs_clean_future_mse": 3.0,
            }
        },
        {
            "metrics": {
                "mcp_flow_vs_exact_mse": 2.0,
                "mcp_x0_vs_clean_future_mse": 2.0,
                "teacher_matched_flow_vs_exact_mse": 3.0,
                "teacher_matched_x0_vs_clean_future_mse": 3.0,
                "teacher_clean_flow_vs_exact_mse": 3.0,
                "teacher_clean_x0_vs_clean_future_mse": 3.0,
            }
        },
    ]
    assert audit.diagnostic_label_from_metrics(records) == audit.INCONCLUSIVE


def test_student_uses_full_sequence_route_not_serial_rollout() -> None:
    student = FakeStudentGenerator()
    result = make_result(student=student)
    assert len(student.forward_calls) == result.manifest["state_count"]
    assert student.serial_forward_calls == 0
    for record in result.manifest["states"]:
        assert record["student"]["proof"]["route"] == "forward_full_sequence_next_forcing"
        assert record["student"]["proof"]["uses_deployment_serial_rollout"] is False


def test_teacher_noisy_current_route_fail_closed() -> None:
    class FailingTeacher(FakeTeacherGenerator):
        def forward(self, **kwargs):
            if int(kwargs["current_start"]) == 3 * FRAME_SEQ_LENGTH:
                raise RuntimeError("noisy-current rejected")
            return super().forward(**kwargs)

    with pytest.raises(RuntimeError, match="noisy-current rejected"):
        make_result(teacher_factory=lambda: make_runtime(FailingTeacher()))


def test_validate_frozen_teacher_model_rejects_mcp_tensors() -> None:
    teacher = FakeTeacherGenerator()
    teacher.eval().requires_grad_(False)
    checkpoint = ev.DeploymentCheckpointRecord(
        path="teacher.pt",
        sha256=TEST_SHA,
        checkpoint_type="official_self_forcing",
        load_mode="OFFICIAL_BACKBONE_STRICT_NO_MCP",
        generator_state_dict={"mcp.fusion.weight": torch.zeros(1)},
    )
    with pytest.raises(RuntimeError, match="MCP tensors"):
        audit.validate_frozen_teacher_model(teacher, checkpoint=checkpoint)


def test_validate_frozen_teacher_model_requires_eval_and_frozen() -> None:
    teacher = FakeTeacherGenerator()
    checkpoint = ev.DeploymentCheckpointRecord(
        path="teacher.pt",
        sha256=TEST_SHA,
        checkpoint_type="official_self_forcing",
        load_mode="OFFICIAL_BACKBONE_STRICT_NO_MCP",
        generator_state_dict={"model.weight": torch.zeros(1)},
    )
    with pytest.raises(RuntimeError, match="eval mode"):
        audit.validate_frozen_teacher_model(teacher, checkpoint=checkpoint)
    teacher.eval()
    teacher.weight.requires_grad_(True)
    with pytest.raises(RuntimeError, match="trainable"):
        audit.validate_frozen_teacher_model(teacher, checkpoint=checkpoint)


def test_select_validation_zero_identity() -> None:
    plan = {
        "validation_sample_identities": ["val0", "val1"],
        "fixed_decode_validation_identity": "val0",
    }
    assert audit.select_validation_zero_identity(plan) == "val0"
    bad = dict(plan)
    bad["fixed_decode_validation_identity"] = "val1"
    with pytest.raises(RuntimeError, match="validation identity 0"):
        audit.select_validation_zero_identity(bad)


def test_runner_parser_fixes_validation_zero_contract_fields() -> None:
    args = runner.parse_args(
        [
            "--full_sequence_checkpoint",
            "checkpoint_step007000.pt",
            "--expected_checkpoint_step",
            "7000",
            "--sample_plan",
            "sample_plan.json",
            "--teacher_manifest",
            "teacher_manifest.json",
            "--dataset_root",
            "dataset",
            "--output_dir",
            "out",
            "--expected_runtime_git_sha",
            "f" * 40,
        ]
    )
    assert args.sample_identity is None
    assert args.num_samples == 1
    assert args.student_direct_clean_context_kv is False
    assert args.expected_training_git_sha is None
    assert runner._expected_training_git_sha(args, git_sha="a" * 40) == "a" * 40


def test_artifact_identity_accepts_diagnostic_resolved_config_sha() -> None:
    plan = {
        "sample_plan_sha256": "b" * 64,
        "validation_sample_identities": ["val0"],
        "fixed_decode_validation_identity": "val0",
    }
    result = audit.validate_teacher_flow_artifact_identity(
        sample_plan=plan,
        teacher_manifest_sha256="c" * 64,
        checkpoint_payload={
            "resolved_config": {
                "sample_plan_sha256": "b" * 64,
                "manifest_sha256": "c" * 64,
            }
        },
        selected_identity="val0",
    )
    assert result["status"] == "PASS"
    assert result["checkpoint_sample_plan_sha256_source"] == (
        "resolved_config.sample_plan_sha256"
    )


def test_payload_validator_accepts_step7000_diagnostic_payload() -> None:
    payload = {
        "schema": audit.NF_SF_MCP1_ONLY_CONTINUATION_SCHEMA,
        "status": "DIAGNOSTIC_MCP1_ONLY",
        "diagnostic_only": True,
        "global_step": 7000,
        "git_sha": TRAINING_GIT_SHA,
        "generator": {
            "model.weight": torch.zeros(1),
            "mcp.fusion.weight": torch.ones(1),
        },
            "resolved_config": {"num_frame_per_block": 3},
        }
    audit._validate_student_checkpoint_payload(
        payload,
        checkpoint_sha256=TEST_SHA,
        expected_checkpoint_step=7000,
        expected_training_git_sha=TRAINING_GIT_SHA,
        expected_official_sha256=ev.OFFICIAL_SELF_FORCING_CHECKPOINT_SHA256,
    )


def test_slim_student_payload_drops_training_state() -> None:
    slim = audit._slim_student_checkpoint_payload(
        {
            "schema": audit.NF_SF_MCP1_ONLY_CONTINUATION_SCHEMA,
            "status": "DIAGNOSTIC_MCP1_ONLY",
            "global_step": 7000,
            "git_sha": TRAINING_GIT_SHA,
            "generator": {"mcp.fusion.weight": torch.ones(1)},
            "optimizer": {"state": {0: {}}},
            "train_rng_state": torch.zeros(1),
            "resolved_config": {"sample_plan_sha256": "b" * 64},
        }
    )
    assert "optimizer" not in slim
    assert "generator" not in slim
    assert "train_rng_state" not in slim
    assert slim["resolved_config"]["sample_plan_sha256"] == "b" * 64


def test_static_forbidden_operations_absent() -> None:
    repo = Path(__file__).resolve().parents[2]
    for relative in (
        "utils/nf_sf_teacher_flow_audit.py",
        "scripts/run_nf_sf_teacher_flow_audit.py",
    ):
        text = (repo / relative).read_text(encoding="utf-8")
        assert ".backward(" not in text
        assert "optimizer.step(" not in text
        assert "torch.optim" not in text
        assert "torch.save(" not in text
        assert "atomic_torch_save" not in text
