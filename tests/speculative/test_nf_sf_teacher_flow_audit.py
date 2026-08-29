from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn

import utils.nf_sf_full_sequence_eval as ev
import utils.nf_sf_teacher_flow_audit as audit
from scripts import run_nf_sf_teacher_flow_audit as runner
from utils.nf_sf_training import FULL_SEQUENCE_TRAINER_SCHEMA
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
    def __init__(
        self,
        *,
        flow_scale: float = 0.25,
        main_current_scale: float = 0.15,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(()), requires_grad=False)
        self.model = FakeModel()
        self.mcp = FakeMCP()
        self.flow_scale = float(flow_scale)
        self.main_current_scale = float(main_current_scale)
        self.forward_calls = []
        self.single_forward_calls = []
        self.serial_forward_calls = 0

    def forward(self, **kwargs):
        if kwargs.get("kv_cache") is None:
            self.serial_forward_calls += 1
            raise RuntimeError("student audit must not use serial rollout")
        if kwargs.get("clean_x") is not None:
            raise RuntimeError("predicted-current Main must not receive clean_x")
        if kwargs.get("mcp_future_noises") is not None:
            raise RuntimeError("predicted-current Main must not receive future")
        chunk = kwargs["noisy_image_or_video"]
        timestep = kwargs["timestep"]
        current_start = int(kwargs["current_start"])
        kv_cache = kwargs["kv_cache"]
        self.single_forward_calls.append(
            {
                "current_start": current_start,
                "chunk_sha256": ev.tensor_sha256(chunk.detach().cpu()),
                "timestep": float(timestep.flatten()[0].detach().cpu().item()),
                "has_clean_x": "clean_x" in kwargs and kwargs["clean_x"] is not None,
                "has_mcp_future": kwargs.get("mcp_future_noises") is not None,
            }
        )
        token_count = int(chunk.shape[1]) * FRAME_SEQ_LENGTH
        token_end = current_start + token_count
        for layer in kv_cache:
            layer["k"][:, current_start:token_end] = len(self.single_forward_calls)
            layer["v"][:, current_start:token_end] = len(self.single_forward_calls) + 1
            layer["global_end_index"].fill_(token_end)
            layer["local_end_index"].fill_(token_end)
        return chunk * self.main_current_scale, chunk

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


def make_runtime(generator: nn.Module) -> ev.DeploymentRuntime:
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


def make_bf16_oracle_failure_tensors() -> tuple[torch.Tensor, torch.Tensor]:
    source = torch.zeros((1, ev.FULL_SEQUENCE_FRAME_COUNT, 1, 1, 1), dtype=torch.bfloat16)
    target = torch.zeros((1, ev.FULL_SEQUENCE_FRAME_COUNT, 1, 1, 1), dtype=torch.bfloat16)
    source[:, 3:6] = torch.tensor(100.0, dtype=torch.bfloat16)
    return source, target


def make_oracle_state(
    *,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, tuple[audit.TeacherFlowAuditState, ...], Any, Any]:
    source = make_source_noise().to(dtype=dtype)
    target = make_teacher_target().to(dtype=dtype)
    main_scheduler = audit.build_flow_match_scheduler(
        shift=DEFAULT_S_MAIN,
        device="cpu",
    )
    mcp_scheduler = audit.build_flow_match_scheduler(
        shift=DEFAULT_S_MCP,
        device="cpu",
    )
    states = audit.build_teacher_flow_audit_states(
        source_noise=source,
        teacher_target=target,
        main_scheduler=main_scheduler,
        mcp_scheduler=mcp_scheduler,
        noise_realizations_per_raw=1,
    )
    return target, states, main_scheduler, mcp_scheduler


def oracle_failure_diagnostic(exc: BaseException) -> dict[str, Any]:
    marker = "diagnostic="
    text = str(exc)
    assert marker in text
    return json.loads(text.split(marker, 1)[1])


def make_parameter_report(
    *,
    role: str = "model",
    sha: str = TEST_SHA,
) -> dict[str, Any]:
    payload = {
        "role": role,
        "parameter_count": 1,
        "parameters": {
            "weight": {
                "sha256": sha,
                "shape": [1],
                "dtype": "torch.float32",
                "requires_grad": False,
            }
        },
    }
    return {
        **payload,
        "fingerprint_sha256": ev.canonical_json_sha256(payload),
    }


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
    student_generator = student or FakeStudentGenerator()
    main_scheduler = audit.build_flow_match_scheduler(
        shift=DEFAULT_S_MAIN,
        device=torch.device("cpu"),
    )
    mcp_scheduler = audit.build_flow_match_scheduler(
        shift=DEFAULT_S_MCP,
        device=torch.device("cpu"),
    )
    return audit.run_teacher_flow_audit(
        student_generator=student_generator,
        student_runtime_factory=lambda: make_runtime(student_generator),
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


def make_validation32_plan(count: int = 256) -> dict[str, Any]:
    identities = [f"val{index:03d}" for index in range(count)]
    return {
        "sample_plan_sha256": "b" * 64,
        "validation_sample_identities": identities,
        "fixed_decode_validation_identity": identities[0],
    }


def make_multi_record(
    *,
    identity_index: int,
    validation_position: int,
    raw_timestep: int,
    mcp_flow: float = 1.0,
    matched_flow: float = 1.1,
    predicted_flow: float = 0.8,
    privileged_flow: float = 0.5,
    mcp_x0: float = 1.0,
    matched_x0: float = 1.1,
    predicted_x0: float = 0.8,
    privileged_x0: float = 0.5,
    current_x0_hat: float = 0.2,
) -> dict[str, Any]:
    identity = f"val{validation_position:03d}"
    return {
        "state_id": f"id{identity_index:02d}_pos{validation_position:03d}"
        f"_raw{raw_timestep:03d}_noise0",
        "sample_identity": identity,
        "identity_index": int(identity_index),
        "validation_position": int(validation_position),
        "raw_timestep": int(raw_timestep),
        "noise_index": 0,
        "metrics": {
            "mcp_flow_vs_exact_mse": float(mcp_flow),
            "mcp_x0_vs_clean_future_mse": float(mcp_x0),
            "student_predicted_current_flow_vs_exact_mse": 0.25,
            "teacher_predicted_current_x0_hat_vs_clean_current_mse": float(
                current_x0_hat
            ),
            "teacher_matched_flow_vs_exact_mse": float(matched_flow),
            "teacher_matched_x0_vs_clean_future_mse": float(matched_x0),
            "mcp_flow_vs_teacher_matched_flow_mse": 0.01,
            "teacher_predicted_flow_vs_exact_mse": float(predicted_flow),
            "teacher_predicted_x0_vs_clean_future_mse": float(predicted_x0),
            "mcp_flow_vs_teacher_predicted_flow_mse": 0.015,
            "teacher_clean_flow_vs_exact_mse": float(privileged_flow),
            "teacher_clean_x0_vs_clean_future_mse": float(privileged_x0),
            "mcp_flow_vs_teacher_clean_flow_mse": 0.02,
        },
        "student_predicted_current": {
            "proof": {
                "student_frozen": True,
                "x0_hat_detached": True,
                "main_forward_uses_clean_x": False,
                "main_forward_uses_mcp_future": False,
            }
        },
        "teacher_matched_current": {
            "proof": {
                "privileged_clean_current": False,
                "same_information_as_mcp": True,
                "teacher_frozen": True,
                "teacher_rng_unchanged": True,
            }
        },
        "teacher_predicted_current": {
            "proof": {
                "privileged_clean_current": False,
                "same_information_as_mcp": False,
                "inference_information_available": True,
                "predicted_current_uses_gt_current": False,
                "predicted_current_uses_gt_future": False,
                "current_x0_hat_detached": True,
                "teacher_frozen": True,
                "student_frozen": True,
                "optimizer_step_executed": False,
                "same_future_tensor_as_other_branches": True,
            }
        },
        "teacher_privileged_current": {
            "proof": {
                "privileged_clean_current": True,
                "same_information_as_mcp": False,
                "inference_information_available": False,
                "teacher_frozen": True,
                "teacher_rng_unchanged": True,
            }
        },
        "teacher_clean_current": {
            "proof": {
                "privileged_clean_current": True,
                "same_information_as_mcp": False,
                "inference_information_available": False,
                "teacher_frozen": True,
                "teacher_rng_unchanged": True,
            }
        },
        "same_state_sigma_proof": {
            "all_future_states_exact": True,
            "raw_timestep_directly_used_for_teacher": False,
            "same_future_tensor_as_other_branches": True,
        },
    }


def make_multi_records(
    *,
    mcp_flow: float = 1.0,
    matched_flow: float = 1.1,
    predicted_flow: float = 0.8,
    privileged_flow: float = 0.5,
    mcp_x0: float = 1.0,
    matched_x0: float = 1.1,
    predicted_x0: float = 0.8,
    privileged_x0: float = 0.5,
    current_x0_hat: float = 0.2,
) -> list[dict[str, Any]]:
    records = []
    for identity_index, validation_position in enumerate(range(0, 256, 8)):
        for raw_timestep in audit.TEACHER_FLOW_AUDIT_RAW_TIMESTEPS:
            records.append(
                make_multi_record(
                    identity_index=identity_index,
                    validation_position=validation_position,
                    raw_timestep=raw_timestep,
                    mcp_flow=mcp_flow,
                    matched_flow=matched_flow,
                    predicted_flow=predicted_flow,
                    privileged_flow=privileged_flow,
                    mcp_x0=mcp_x0,
                    matched_x0=matched_x0,
                    predicted_x0=predicted_x0,
                    privileged_x0=privileged_x0,
                    current_x0_hat=current_x0_hat,
                )
            )
    return records


def make_identity_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    aggregates = audit.aggregate_teacher_flow_metrics(records)["by_identity"]
    identity_records = []
    for key, value in aggregates.items():
        identity_records.append(
            {
                "identity_index": value["identity_index"],
                "sample_identity": value["sample_identity"],
                "validation_position": value["validation_position"],
                "state_count": value["state_count"],
                "common_inputs_fingerprint_sha256": "f" * 64,
                "metrics": value["metrics"],
                "aggregate_key": key,
            }
        )
    return identity_records


def make_multi_manifest(records: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    selection = audit.select_validation32_identities(make_validation32_plan())
    state_records = records if records is not None else make_multi_records()
    return audit.build_teacher_flow_multi_identity_manifest(
        state_records=state_records,
        identity_records=make_identity_records(state_records),
        identity_selection=selection,
        student_checkpoint_contract={
            "status": "PASS",
            "required_global_step": 6500,
            "actual_global_step": 6500,
            "required_schema": FULL_SEQUENCE_TRAINER_SCHEMA,
            "actual_schema": FULL_SEQUENCE_TRAINER_SCHEMA,
            "checkpoint_sha256": TEST_SHA,
        },
        checkpoint_summary={
            "checkpoint_type": "teacher_flow_student_step6500",
            "load_mode": audit.TEACHER_FLOW_AUDIT_STUDENT_LOAD_MODE,
            "sha256": TEST_SHA,
            "global_step": 6500,
        },
        teacher_summary={
            "checkpoint_type": "official_self_forcing",
            "checkpoint_sha256": "d" * 64,
            "mcp_tensor_count": 0,
            "eval_mode": True,
            "requires_grad_false": True,
        },
        common_inputs_fingerprints_sha256={
            str(position): "f" * 64 for position in range(0, 256, 8)
        },
        runtime_git_sha=RUNTIME_GIT_SHA,
        training_checkpoint_git_sha=TRAINING_GIT_SHA,
    )


def _contains_tensor(value: Any) -> bool:
    if torch.is_tensor(value):
        return True
    if isinstance(value, dict):
        return any(_contains_tensor(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_tensor(item) for item in value)
    return False


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
        assert proof["teacher_predicted_future_state_exact"] is True
        assert proof["teacher_clean_future_state_exact"] is True
        assert proof["same_future_tensor_as_other_branches"] is True
        assert record["teacher_matched_current"]["proof"][
            "history_chunk0_identity_exact"
        ] is True
        assert record["teacher_privileged_current"]["proof"][
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
        assert record["teacher_predicted_current"]["proof"][
            "future_timestep_matches_physical_sigma"
        ] is True
        assert record["teacher_predicted_current"]["proof"][
            "student_current_prediction"
        ]["proof"]["main_sigma"] == pytest.approx(record["main_sigma"])
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


@pytest.mark.parametrize("shift", [DEFAULT_S_MAIN, DEFAULT_S_MCP])
def test_flow_match_add_noise_target_step_exact_float32(shift: float) -> None:
    scheduler = audit.build_flow_match_scheduler(shift=shift, device="cpu")
    clean = torch.tensor([0.25, 1.5, -0.75], dtype=torch.float32).reshape(
        1,
        3,
        1,
        1,
        1,
    )
    noise = torch.tensor([1.0, -0.5, 2.0], dtype=torch.float32).reshape(
        1,
        3,
        1,
        1,
        1,
    )
    timestep = audit.route_eq._timestep(750.0, clean)
    state = audit.route_eq._add_noise_chunk(
        scheduler,
        clean=clean,
        noise=noise,
        timestep=timestep,
        name="unit_state",
    )
    flow = audit.route_eq._training_target_chunk(
        scheduler,
        clean=clean,
        noise=noise,
        timestep=timestep,
        name="unit_flow",
    )
    recon = audit.reconstruct_x0_from_flow_matching(
        scheduler,
        state=state,
        flow=flow,
        timestep=timestep,
        name="unit_recon",
    )
    assert torch.allclose(recon, clean, atol=1e-6, rtol=1e-6)


def test_exact_current_flow_to_x0_conversion_oracle() -> None:
    target, states, main_scheduler, _ = make_oracle_state()
    oracle = audit.exact_current_flow_conversion_oracle(
        main_scheduler,
        state=states[0],
        teacher_target=target,
    )
    assert oracle["status"] == "PASS"
    assert oracle["derived_formula"] == "x0 = x_t - sigma * flow"
    assert oracle["formula_matches_scheduler"] is True
    assert oracle["max_abs_error"] <= audit.CURRENT_X0_ORACLE_ATOL
    diagnostic = oracle["diagnostic"]
    assert diagnostic["scheduler_actually_passed"]["shift"] == pytest.approx(
        DEFAULT_S_MAIN
    )
    assert diagnostic["main_scheduler_expected_contract"]["shift"] == pytest.approx(
        DEFAULT_S_MAIN
    )
    assert diagnostic["noisy_state_vs_regenerated"]["torch_equal"] is True
    assert diagnostic["recon_sched_vs_same_dtype_explicit"]["max_abs"] == (
        pytest.approx(0.0, abs=1e-6)
    )
    assert diagnostic["float32_reference"]["passes_existing_oracle_tolerance"] is True
    assert diagnostic["float32_reference"]["recon32_vs_clean32"]["max_abs"] == (
        pytest.approx(0.0, abs=1e-6)
    )


def test_exact_current_oracle_rejects_mcp_scheduler_for_main_state() -> None:
    target, states, _, mcp_scheduler = make_oracle_state()
    with pytest.raises(RuntimeError) as excinfo:
        audit.exact_current_flow_conversion_oracle(
            mcp_scheduler,
            state=states[0],
            teacher_target=target,
        )
    diagnostic = oracle_failure_diagnostic(excinfo.value)
    assert "scheduler_shift_not_main" in str(excinfo.value)
    assert diagnostic["scheduler_actually_passed"]["shift"] == pytest.approx(
        DEFAULT_S_MCP
    )
    assert diagnostic["main_scheduler_expected_contract"]["shift"] == pytest.approx(
        DEFAULT_S_MAIN
    )


def test_exact_current_oracle_rejects_mismatched_timestep() -> None:
    target, states, main_scheduler, _ = make_oracle_state()
    bad_state = replace(
        states[0],
        main_warped_timestep=states[0].future_warped_timestep,
    )
    with pytest.raises(RuntimeError) as excinfo:
        audit.exact_current_flow_conversion_oracle(
            main_scheduler,
            state=bad_state,
            teacher_target=target,
        )
    diagnostic = oracle_failure_diagnostic(excinfo.value)
    assert "current_timestep_not_main_shift5_raw_warp" in str(excinfo.value)
    assert diagnostic["warped_current_timestep"] == pytest.approx(
        states[0].future_warped_timestep
    )


def test_exact_current_oracle_round_trip_catches_mismatched_noise_state() -> None:
    target, states, main_scheduler, _ = make_oracle_state()
    bad_state = replace(states[0], current_noise=states[0].current_noise + 1.0)
    with pytest.raises(RuntimeError) as excinfo:
        audit.exact_current_flow_conversion_oracle(
            main_scheduler,
            state=bad_state,
            teacher_target=target,
        )
    diagnostic = oracle_failure_diagnostic(excinfo.value)
    assert "current_state_noise_timestep_round_trip_mismatch" in str(excinfo.value)
    assert diagnostic["failure_case"] == "state_noise_timestep_provenance_bug"
    assert diagnostic["noisy_state_vs_regenerated"]["torch_equal"] is False
    assert diagnostic["noisy_state_vs_regenerated"]["max_abs"] > 0.0


def test_exact_current_oracle_bf16_failure_reports_float32_reference() -> None:
    source, target = make_bf16_oracle_failure_tensors()
    main_scheduler = audit.build_flow_match_scheduler(
        shift=DEFAULT_S_MAIN,
        device="cpu",
    )
    mcp_scheduler = audit.build_flow_match_scheduler(
        shift=DEFAULT_S_MCP,
        device="cpu",
    )
    states = audit.build_teacher_flow_audit_states(
        source_noise=source,
        teacher_target=target,
        main_scheduler=main_scheduler,
        mcp_scheduler=mcp_scheduler,
        noise_realizations_per_raw=1,
    )
    with pytest.raises(RuntimeError) as excinfo:
        audit.exact_current_flow_conversion_oracle(
            main_scheduler,
            state=states[0],
            teacher_target=target,
        )
    diagnostic = oracle_failure_diagnostic(excinfo.value)
    assert diagnostic["failure_case"] == "bf16_quantized_state_contract"
    assert diagnostic["noisy_state_vs_regenerated"]["torch_equal"] is True
    assert diagnostic["recon_sched_vs_clean"]["max_abs"] > audit.CURRENT_X0_ORACLE_ATOL
    assert diagnostic["recon_same_dtype_explicit_vs_clean"]["max_abs"] > (
        audit.CURRENT_X0_ORACLE_ATOL
    )
    assert diagnostic["recon_sched_vs_same_dtype_explicit"]["max_abs"] == (
        pytest.approx(0.0, abs=1e-6)
    )
    assert diagnostic["recon_formula_actual_vs_clean"]["max_abs"] > (
        audit.CURRENT_X0_ORACLE_ATOL
    )
    assert diagnostic["float32_reference"]["passes_existing_oracle_tolerance"] is True
    assert diagnostic["float32_reference"]["recon32_vs_clean32"]["max_abs"] == (
        pytest.approx(0.0, abs=1e-6)
    )


def test_predicted_current_recheck_state_matches_formal_state_construction() -> None:
    source = make_source_noise()
    target = make_teacher_target()
    main_scheduler = audit.build_flow_match_scheduler(
        shift=DEFAULT_S_MAIN,
        device="cpu",
    )
    mcp_scheduler = audit.build_flow_match_scheduler(
        shift=DEFAULT_S_MCP,
        device="cpu",
    )
    recheck = audit.build_predicted_current_oracle_recheck_state(
        source_noise=source,
        teacher_target=target,
        main_scheduler=main_scheduler,
        mcp_scheduler=mcp_scheduler,
        state_id_prefix="id00_pos000",
        sample_identity="val000",
        validation_position=0,
        identity_index=0,
    )
    formal = audit.build_teacher_flow_audit_states(
        source_noise=source,
        teacher_target=target,
        main_scheduler=main_scheduler,
        mcp_scheduler=mcp_scheduler,
        noise_realizations_per_raw=1,
        state_id_prefix="id00_pos000",
        sample_identity="val000",
        validation_position=0,
        identity_index=0,
    )

    assert len(formal) == len(audit.TEACHER_FLOW_AUDIT_RAW_TIMESTEPS)
    assert recheck.state_id == "id00_pos000_raw999_noise0"
    assert recheck.raw_timestep == audit.PREDICTED_CURRENT_ORACLE_RECHECK_RAW_TIMESTEP
    assert recheck.noise_index == audit.PREDICTED_CURRENT_ORACLE_RECHECK_NOISE_INDEX
    assert recheck.provenance["validation_position"] == 0
    assert recheck.provenance["identity_index"] == 0
    assert torch.equal(recheck.current_state, formal[0].current_state)
    assert torch.equal(recheck.current_noise, formal[0].current_noise)
    assert recheck.main_warped_timestep == pytest.approx(
        formal[0].main_warped_timestep
    )
    assert recheck.main_sigma == pytest.approx(formal[0].main_sigma)


def test_predicted_current_recheck_all_raw_matches_formal_identity0_plan() -> None:
    source = make_source_noise()
    target = make_teacher_target()
    main_scheduler = audit.build_flow_match_scheduler(
        shift=DEFAULT_S_MAIN,
        device="cpu",
    )
    mcp_scheduler = audit.build_flow_match_scheduler(
        shift=DEFAULT_S_MCP,
        device="cpu",
    )
    all_raw = audit.build_predicted_current_oracle_recheck_validation0_all_raw_states(
        source_noise=source,
        teacher_target=target,
        main_scheduler=main_scheduler,
        mcp_scheduler=mcp_scheduler,
        state_id_prefix="id00_pos000",
        sample_identity="val000",
        validation_position=0,
        identity_index=0,
    )
    formal = audit.build_teacher_flow_audit_states(
        source_noise=source,
        teacher_target=target,
        main_scheduler=main_scheduler,
        mcp_scheduler=mcp_scheduler,
        noise_realizations_per_raw=1,
        state_id_prefix="id00_pos000",
        sample_identity="val000",
        validation_position=0,
        identity_index=0,
    )

    assert len(all_raw) == 4
    assert [state.raw_timestep for state in all_raw] == [999, 750, 500, 250]
    assert [state.noise_index for state in all_raw] == [0, 0, 0, 0]
    for recheck_state, formal_state in zip(all_raw, formal, strict=True):
        assert recheck_state.state_id == formal_state.state_id
        assert torch.equal(recheck_state.current_noise, formal_state.current_noise)
        assert torch.equal(recheck_state.future_noise, formal_state.future_noise)
        assert torch.equal(recheck_state.current_state, formal_state.current_state)
        assert torch.equal(recheck_state.future_state, formal_state.future_state)
        assert recheck_state.provenance["current_noise"]["sha256"] == (
            formal_state.provenance["current_noise"]["sha256"]
        )


def test_formal_predicted_current_consumes_identity0_states_in_builder_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = make_source_noise()
    target = make_teacher_target()
    main_scheduler = audit.build_flow_match_scheduler(
        shift=DEFAULT_S_MAIN,
        device="cpu",
    )
    mcp_scheduler = audit.build_flow_match_scheduler(
        shift=DEFAULT_S_MCP,
        device="cpu",
    )
    states = audit.build_teacher_flow_audit_states(
        source_noise=source,
        teacher_target=target,
        main_scheduler=main_scheduler,
        mcp_scheduler=mcp_scheduler,
        noise_realizations_per_raw=1,
        state_id_prefix="id00_pos000",
        sample_identity="val000",
        validation_position=0,
        identity_index=0,
    )
    seen: list[int] = []

    def fake_run_state(*, state, **kwargs):
        _ = kwargs
        seen.append(int(state.raw_timestep))
        return audit.FlowPrediction(
            state_id=state.state_id,
            branch=audit.STUDENT_PREDICTED_CURRENT_BRANCH,
            flow=torch.zeros_like(state.current_state),
            x0=torch.zeros_like(state.current_state),
            proof={},
        )

    monkeypatch.setattr(audit, "_run_student_predicted_current_state", fake_run_state)
    result = audit.run_student_predicted_current_predictions(
        runtime_factory=None,
        states=states,
        source_noise=source,
        teacher_target=target,
        teacher_payload={"rollout_seed": 456},
        conditional_dict={},
        main_scheduler=main_scheduler,
    )

    assert seen == [999, 750, 500, 250]
    assert list(result) == [state.state_id for state in states]


def test_predicted_current_recheck_exact_pass_classification() -> None:
    target, states, main_scheduler, _ = make_oracle_state()
    oracle = audit.exact_current_flow_conversion_oracle(
        main_scheduler,
        state=states[0],
        teacher_target=target,
    )
    artifact = audit.build_predicted_current_oracle_recheck_artifact(
        diagnostic=oracle["diagnostic"],
        original_bf16_oracle_pass=True,
        runtime_git_sha=RUNTIME_GIT_SHA,
        sample_identity="val000",
        identity_index=0,
        validation_position=0,
        student_parameters_before=make_parameter_report(role="student"),
        student_parameters_after=make_parameter_report(role="student"),
        teacher_parameters_before=make_parameter_report(role="teacher"),
        teacher_parameters_after=make_parameter_report(role="teacher"),
        rng_before="r",
        rng_after="r",
    )

    assert artifact["diagnostic_classification"] == audit.EXACT_PASS
    assert artifact["status"] == "PASS"
    assert artifact["original_bf16_oracle_pass"] is True
    assert artifact["backward_executed"] is False
    assert artifact["optimizer_step_executed"] is False
    assert artifact["checkpoint_written"] is False


def test_predicted_current_recheck_original_pass_ignores_float32_rounding() -> None:
    target, states, main_scheduler, _ = make_oracle_state()
    oracle = audit.exact_current_flow_conversion_oracle(
        main_scheduler,
        state=states[0],
        teacher_target=target,
    )
    diagnostic = json.loads(json.dumps(oracle["diagnostic"]))
    diagnostic["recon_sched_vs_formula_actual"]["max_abs"] = 0.0077996
    diagnostic["recon_sched_vs_float32_explicit"]["max_abs"] = 0.0077996
    diagnostic["float32_explicit_reference"]["scheduler_vs_explicit"][
        "max_abs"
    ] = 0.0077996
    diagnostic["recon_sched_vs_same_dtype_explicit"]["max_abs"] = 0.0
    diagnostic["explicit_same_dtype_reference"]["scheduler_vs_explicit"][
        "max_abs"
    ] = 0.0

    assert audit.classify_predicted_current_oracle_recheck(
        diagnostic,
        original_bf16_oracle_pass=True,
    ) == audit.EXACT_PASS


def test_predicted_current_recheck_bf16_vs_fp32_rounding_not_scheduler_mismatch() -> None:
    target, states, main_scheduler, _ = make_oracle_state()
    oracle = audit.exact_current_flow_conversion_oracle(
        main_scheduler,
        state=states[0],
        teacher_target=target,
    )
    diagnostic = json.loads(json.dumps(oracle["diagnostic"]))
    for key in ("clean", "noise", "state", "exact_flow", "reconstructed"):
        diagnostic["tensor_dtypes_devices"][key]["dtype"] = "torch.bfloat16"
    diagnostic["recon_sched_vs_formula_actual"]["max_abs"] = 0.0077996
    diagnostic["recon_sched_vs_float32_explicit"]["max_abs"] = 0.0077996
    diagnostic["recon_sched_vs_same_dtype_explicit"]["max_abs"] = 0.0

    assert audit.classify_predicted_current_oracle_recheck(
        diagnostic,
        original_bf16_oracle_pass=True,
    ) == audit.EXACT_PASS


def test_predicted_current_recheck_bf16_classification_requires_all_subgates() -> None:
    source, target = make_bf16_oracle_failure_tensors()
    main_scheduler = audit.build_flow_match_scheduler(
        shift=DEFAULT_S_MAIN,
        device="cpu",
    )
    mcp_scheduler = audit.build_flow_match_scheduler(
        shift=DEFAULT_S_MCP,
        device="cpu",
    )
    state = audit.build_predicted_current_oracle_recheck_state(
        source_noise=source,
        teacher_target=target,
        main_scheduler=main_scheduler,
        mcp_scheduler=mcp_scheduler,
        sample_identity="val000",
        validation_position=0,
        identity_index=0,
    )
    with pytest.raises(RuntimeError) as excinfo:
        audit.exact_current_flow_conversion_oracle(
            main_scheduler,
            state=state,
            teacher_target=target,
        )
    diagnostic = oracle_failure_diagnostic(excinfo.value)

    assert audit.classify_predicted_current_oracle_recheck(
        diagnostic,
        original_bf16_oracle_pass=False,
    ) == audit.BF16_QUANTIZED_STATE_CONTRACT

    state_mismatch = json.loads(json.dumps(diagnostic))
    state_mismatch["noisy_state_vs_regenerated"]["torch_equal"] = False
    assert audit.classify_predicted_current_oracle_recheck(
        state_mismatch,
        original_bf16_oracle_pass=False,
    ) == audit.STATE_PROVENANCE_MISMATCH

    scheduler_mismatch = json.loads(json.dumps(diagnostic))
    scheduler_mismatch["recon_sched_vs_same_dtype_explicit"]["max_abs"] = 0.01
    assert audit.classify_predicted_current_oracle_recheck(
        scheduler_mismatch,
        original_bf16_oracle_pass=False,
    ) == audit.SCHEDULER_MISMATCH

    semantic_mismatch = json.loads(json.dumps(diagnostic))
    semantic_mismatch["float32_reference"]["recon32_vs_clean32"]["max_abs"] = 1e-3
    assert audit.classify_predicted_current_oracle_recheck(
        semantic_mismatch,
        original_bf16_oracle_pass=False,
    ) == audit.SEMANTIC_MISMATCH


def test_predicted_current_recheck_same_dtype_disagreement_is_scheduler_mismatch() -> None:
    target, states, main_scheduler, _ = make_oracle_state()
    oracle = audit.exact_current_flow_conversion_oracle(
        main_scheduler,
        state=states[0],
        teacher_target=target,
    )
    diagnostic = json.loads(json.dumps(oracle["diagnostic"]))
    diagnostic["recon_sched_vs_same_dtype_explicit"]["max_abs"] = 0.01

    assert audit.classify_predicted_current_oracle_recheck(
        diagnostic,
        original_bf16_oracle_pass=True,
    ) == audit.SCHEDULER_MISMATCH


def test_predicted_current_recheck_scheduler_mismatch_classification() -> None:
    target, states, _, mcp_scheduler = make_oracle_state()
    with pytest.raises(RuntimeError) as excinfo:
        audit.exact_current_flow_conversion_oracle(
            mcp_scheduler,
            state=states[0],
            teacher_target=target,
        )
    diagnostic = oracle_failure_diagnostic(excinfo.value)

    assert audit.classify_predicted_current_oracle_recheck(
        diagnostic,
        original_bf16_oracle_pass=False,
    ) == audit.SCHEDULER_MISMATCH


def test_predicted_current_recheck_state_mismatch_classification() -> None:
    target, states, main_scheduler, _ = make_oracle_state()
    bad_state = replace(states[0], current_noise=states[0].current_noise + 1.0)
    with pytest.raises(RuntimeError) as excinfo:
        audit.exact_current_flow_conversion_oracle(
            main_scheduler,
            state=bad_state,
            teacher_target=target,
        )
    diagnostic = oracle_failure_diagnostic(excinfo.value)

    assert audit.classify_predicted_current_oracle_recheck(
        diagnostic,
        original_bf16_oracle_pass=False,
    ) == audit.STATE_PROVENANCE_MISMATCH


def test_predicted_current_main_does_not_read_gt_current_or_future() -> None:
    student = FakeStudentGenerator()
    result = make_result(student=student)
    current_calls = [
        call
        for call in student.single_forward_calls
        if call["current_start"] == 3 * FRAME_SEQ_LENGTH
    ]
    assert len(current_calls) == result.manifest["state_count"]
    assert all(call["has_clean_x"] is False for call in current_calls)
    assert all(call["has_mcp_future"] is False for call in current_calls)
    for record in result.manifest["states"]:
        proof = record["teacher_predicted_current"]["proof"]
        assert proof["predicted_current_uses_gt_current"] is False
        assert proof["predicted_current_uses_gt_future"] is False
        assert proof["student_current_prediction"]["proof"][
            "main_forward_uses_clean_x"
        ] is False
        assert proof["student_current_prediction"]["proof"][
            "main_forward_uses_mcp_future"
        ] is False


def test_predicted_current_x0_hat_is_detached_and_recorded() -> None:
    result = make_result()
    for state_id, tensors in result.tensors["states"].items():
        assert tensors["student_predicted_current_x0_hat"].requires_grad is False
        record = next(item for item in result.manifest["states"] if item["state_id"] == state_id)
        proof = record["teacher_predicted_current"]["proof"]
        assert proof["current_x0_hat_detached"] is True
        assert proof["current_x0_hat"]["finite"] is True
        assert len(proof["current_x0_hat"]["sha256"]) == 64
        assert "teacher_predicted_current_x0_hat_vs_clean_current_mse" in (
            record["metrics"]
        )


def test_predicted_branch_frozen_no_step_and_rng_proofs() -> None:
    result = make_result()
    for record in result.manifest["states"]:
        proof = record["teacher_predicted_current"]["proof"]
        assert proof["teacher_frozen"] is True
        assert proof["student_frozen"] is True
        assert proof["optimizer_step_executed"] is False
        assert proof["teacher_rng_unchanged"] is True
        assert proof["student_current_prediction"]["proof"][
            "student_rng_unchanged"
        ] is True
        assert record["same_state_sigma_proof"]["teacher_frozen"] is True
        assert record["same_state_sigma_proof"]["student_frozen"] is True
        assert record["same_state_sigma_proof"]["optimizer_step_executed"] is False


def test_oracle_target_cannot_be_marked_as_teacher() -> None:
    result = make_result()
    manifest = result.manifest
    for record in manifest["states"]:
        assert record["teacher_matched_current"]["proof"][
            "uses_ground_truth_future_x0_for_conversion"
        ] is False
        assert record["teacher_predicted_current"]["proof"][
            "uses_ground_truth_future_x0_for_conversion"
        ] is False
        assert record["teacher_privileged_current"]["proof"][
            "uses_ground_truth_future_x0_for_conversion"
        ] is False
    assert any("oracle" in item for item in manifest["forbidden_comparisons"])


def test_privileged_branch_is_explicitly_marked() -> None:
    result = make_result()
    for record in result.manifest["states"]:
        clean_proof = record["teacher_privileged_current"]["proof"]
        matched_proof = record["teacher_matched_current"]["proof"]
        assert clean_proof["privileged_clean_current"] is True
        assert clean_proof["same_information_as_mcp"] is False
        assert clean_proof["inference_information_available"] is False
        assert matched_proof["privileged_clean_current"] is False
        assert matched_proof["same_information_as_mcp"] is True
        assert matched_proof["inference_information_available"] is True


def test_matched_and_privileged_routes_are_not_confused() -> None:
    result = make_result()
    contracts = result.manifest["teacher_routes"]
    assert contracts[audit.TEACHER_MATCHED_CURRENT_BRANCH][
        "privileged_clean_current"
    ] is False
    assert contracts[audit.TEACHER_PRIVILEGED_CURRENT_BRANCH][
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
    assert len(student.single_forward_calls) == result.manifest["state_count"] * 2
    for record in result.manifest["states"]:
        assert record["student"]["proof"]["route"] == "forward_full_sequence_next_forcing"
        assert record["student"]["proof"]["uses_deployment_serial_rollout"] is False
        assert record["student_predicted_current"]["proof"]["route"] == (
            "student_main_single_chunk_kv_forward"
        )


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


def test_select_validation32_identities_exact_stride_and_fingerprints() -> None:
    selection = audit.select_validation32_identities(make_validation32_plan())
    assert selection["validation_identity_count"] == 256
    assert selection["selected_identity_count"] == 32
    assert selection["stride"] == 8
    assert selection["selection_rule"] == (
        "validation positions 0,8,16,...,248 from exact 256 list"
    )
    assert selection["positions"] == list(range(0, 256, 8))
    assert selection["identity_strings"][:3] == ["val000", "val008", "val016"]
    assert selection["identity_strings"][-1] == "val248"
    assert len(selection["identity_list_sha256"]) == 64
    assert len(selection["selection_fingerprint_sha256"]) == 64


def test_select_validation32_fails_closed_when_count_is_not_256() -> None:
    with pytest.raises(RuntimeError, match="exactly 256"):
        audit.select_validation32_identities(make_validation32_plan(count=255))


def test_multi_identity_state_builder_uses_one_noise_per_raw() -> None:
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
        noise_realizations_per_raw=1,
        state_id_prefix="id00_pos000",
        sample_identity="val000",
        validation_position=0,
        identity_index=0,
    )
    assert len(states) == 4
    assert [state.noise_index for state in states] == [0, 0, 0, 0]
    assert [state.raw_timestep for state in states] == [999, 750, 500, 250]
    assert states[0].state_id.startswith("id00_pos000_raw999_noise0")
    assert states[0].provenance["sample_identity"] == "val000"


def test_multi_manifest_records_exactly_128_states_and_selection() -> None:
    manifest = make_multi_manifest()
    assert manifest["mode"] == audit.TEACHER_FLOW_AUDIT_MODE_MULTI_VALIDATION32
    assert manifest["state_count"] == 128
    assert manifest["noise_realizations_per_raw"] == 1
    assert manifest["identity_selection"]["positions"] == list(range(0, 256, 8))
    assert manifest["identity_selection"]["identity_strings"][0] == "val000"
    assert manifest["identity_selection"]["identity_strings"][-1] == "val248"


def test_multi_aggregation_by_identity_and_raw() -> None:
    manifest = make_multi_manifest()
    aggregates = manifest["aggregates"]
    assert len(aggregates["by_state"]) == 128
    assert len(aggregates["by_identity"]) == 32
    assert set(aggregates["by_raw"].keys()) == {"250", "500", "750", "999"}
    identity0 = aggregates["by_identity"]["validation_position_000"]
    assert identity0["state_count"] == 4
    assert identity0["metrics"]["teacher_clean_flow_vs_exact_mse"]["mean"] == 0.5
    assert identity0["metrics"]["teacher_predicted_flow_vs_exact_mse"]["mean"] == 0.8
    assert aggregates["by_raw"]["999"]["mcp_flow_vs_exact_mse"]["mean"] == 1.0
    bridge = manifest["predicted_current_bridge_statistics"]
    assert bridge["all_states"]["matched_flow_mse"] == pytest.approx(1.1)
    assert bridge["all_states"]["predicted_flow_mse"] == pytest.approx(0.8)
    assert bridge["all_states"]["privileged_flow_mse"] == pytest.approx(0.5)
    assert bridge["all_states"]["mcp_flow_mse"] == pytest.approx(1.0)


def test_multi_paired_stats_win_rate_and_reductions() -> None:
    manifest = make_multi_manifest()
    stats = manifest["paired_statistics"]
    assert stats["privileged_identity_flow_win_count"] == 32
    assert stats["privileged_identity_flow_win_rate"] == pytest.approx(1.0)
    assert stats["matched_identity_flow_win_count"] == 0
    assert stats["privileged_identity_flow_reduction_mean"] == pytest.approx(0.5)
    assert stats["privileged_identity_flow_reduction_median"] == pytest.approx(0.5)
    assert stats["predicted_better_than_matched_win_count"] == 32
    assert stats["predicted_better_than_matched_win_rate"] == pytest.approx(1.0)
    assert stats["predicted_vs_matched_flow_reduction_mean"] == pytest.approx(
        (1.1 - 0.8) / 1.1
    )
    assert stats["gap_recovery_ratio_mean"] == pytest.approx(0.5)
    assert stats["by_raw"]["999"]["privileged_flow_win_count"] == 32
    assert stats["by_raw"]["999"]["privileged_flow_reduction_mean"] == pytest.approx(
        0.5
    )
    assert stats["by_raw"]["999"][
        "predicted_better_than_matched_win_count"
    ] == 32


def test_multi_fixture_builds_full_validation32_raw_plan() -> None:
    records = make_multi_records(matched_flow=1.0, predicted_flow=0.70)
    assert len(records) == 128
    assert len({record["sample_identity"] for record in records}) == 32
    assert {
        raw: sum(1 for record in records if int(record["raw_timestep"]) == raw)
        for raw in audit.TEACHER_FLOW_AUDIT_RAW_TIMESTEPS
    } == {999: 32, 750: 32, 500: 32, 250: 32}
    assert all(
        record["metrics"]["teacher_predicted_flow_vs_exact_mse"]
        < record["metrics"]["teacher_matched_flow_vs_exact_mse"]
        for record in records
    )


def test_predicted_current_bridge_primary_thresholds() -> None:
    policy = audit._predicted_current_bridge_policy()
    assert policy["threshold_comparison_atol"] == pytest.approx(1.0e-12)
    assert policy["strong"]["all_state_flow_reduction_min"] == pytest.approx(0.15)
    assert policy["strong"]["identity_win_rate_min"] == pytest.approx(0.75)
    assert policy["strong"]["identity_win_count_min"] == 24
    assert policy["strong"]["gap_recovery_ratio_min"] == pytest.approx(0.30)
    assert policy["strong"]["future_x0_reduction_min"] == pytest.approx(0.10)
    assert policy["no_support"]["all_state_flow_reduction_lt"] == pytest.approx(0.05)
    assert policy["no_support"]["identity_win_rate_lt"] == pytest.approx(0.60)
    assert policy["no_support"]["raw_clearly_worse_count_gte"] == 2
    assert policy["no_support"]["clearly_worse_relative_margin"] == pytest.approx(
        0.05
    )

    strong = make_multi_manifest(
        make_multi_records(
            matched_flow=1.0,
            predicted_flow=0.70,
            privileged_flow=0.0,
            matched_x0=1.0,
            predicted_x0=0.80,
        )
    )
    assert strong["primary_diagnostic_label"] == (
        audit.STRONG_PREDICTED_CURRENT_BRIDGE_SUPPORT
    )
    bridge = strong["predicted_current_bridge_statistics"]["all_states"]
    assert bridge["predicted_vs_matched_flow_reduction_pct"] == pytest.approx(30.0)
    assert bridge["gap_recovery_ratio"] == pytest.approx(0.30)

    weak_reduction = make_multi_manifest(
        make_multi_records(matched_flow=1.0, predicted_flow=0.96)
    )
    assert weak_reduction["primary_diagnostic_label"] == audit.NO_SUPPORT

    low_win_records = make_multi_records(matched_flow=1.0, predicted_flow=0.70)
    for record in low_win_records[:56]:
        record["metrics"]["teacher_predicted_flow_vs_exact_mse"] = 1.0
    low_win = make_multi_manifest(low_win_records)
    assert low_win["primary_diagnostic_label"] == audit.NO_SUPPORT

    raw_worse_records = make_multi_records(matched_flow=1.0, predicted_flow=0.70)
    for record in raw_worse_records:
        if int(record["raw_timestep"]) in (999, 750):
            record["metrics"]["teacher_predicted_flow_vs_exact_mse"] = 1.06
    raw_worse = make_multi_manifest(raw_worse_records)
    assert raw_worse["primary_diagnostic_label"] == audit.NO_SUPPORT

    mixed = make_multi_manifest(
        make_multi_records(
            matched_flow=1.0,
            predicted_flow=0.88,
            privileged_flow=0.70,
            matched_x0=1.0,
            predicted_x0=0.92,
        )
    )
    assert mixed["primary_diagnostic_label"] == audit.INCONCLUSIVE

    undefined_gap = make_multi_manifest(
        make_multi_records(matched_flow=1.0, predicted_flow=0.90, privileged_flow=1.1)
    )
    assert undefined_gap["predicted_current_bridge_statistics"]["all_states"][
        "gap_recovery_ratio"
    ] is None


def test_predicted_current_bridge_exact_numeric_thresholds_are_strong() -> None:
    manifest = make_multi_manifest(
        make_multi_records(
            matched_flow=1.0,
            predicted_flow=0.85,
            privileged_flow=0.5,
            matched_x0=1.0,
            predicted_x0=0.90,
        )
    )
    assert manifest["primary_diagnostic_label"] == (
        audit.STRONG_PREDICTED_CURRENT_BRIDGE_SUPPORT
    )
    bridge = manifest["predicted_current_bridge_statistics"]["all_states"]
    assert bridge["predicted_vs_matched_flow_reduction"] == pytest.approx(0.15)
    assert bridge["gap_recovery_ratio"] == pytest.approx(0.30)
    assert bridge["predicted_vs_matched_x0_reduction"] == pytest.approx(0.10)


def test_predicted_current_bridge_exact_identity_threshold_is_strong() -> None:
    records = make_multi_records(
        matched_flow=1.0,
        predicted_flow=0.70,
        privileged_flow=0.25,
        matched_x0=1.0,
        predicted_x0=0.80,
    )
    for record in records:
        if int(record["identity_index"]) >= 24:
            record["metrics"]["teacher_predicted_flow_vs_exact_mse"] = 1.0
    manifest = make_multi_manifest(records)
    stats = manifest["predicted_current_bridge_statistics"]
    assert stats["identity_predicted_better_than_matched_win_count"] == 24
    assert stats["identity_predicted_better_than_matched_win_rate"] == pytest.approx(
        0.75
    )
    assert stats["raw_predicted_not_worse_count"] == 4
    assert stats["all_states"]["gap_recovery_ratio"] == pytest.approx(0.30)
    assert manifest["primary_diagnostic_label"] == (
        audit.STRONG_PREDICTED_CURRENT_BRIDGE_SUPPORT
    )


def test_predicted_current_bridge_just_below_strong_threshold_is_not_strong() -> None:
    manifest = make_multi_manifest(
        make_multi_records(
            matched_flow=1.0,
            predicted_flow=0.85,
            privileged_flow=0.5,
            matched_x0=1.0,
            predicted_x0=0.900000000002,
        )
    )
    assert manifest["predicted_current_bridge_statistics"]["all_states"][
        "predicted_vs_matched_x0_reduction"
    ] < 0.10
    assert manifest["primary_diagnostic_label"] == audit.INCONCLUSIVE


def test_predicted_current_bridge_one_raw_worse_is_not_strong() -> None:
    records = make_multi_records(
        matched_flow=1.0,
        predicted_flow=0.60,
        privileged_flow=0.01,
        matched_x0=1.0,
        predicted_x0=0.80,
    )
    for record in records:
        if int(record["raw_timestep"]) == 999:
            record["metrics"]["teacher_predicted_flow_vs_exact_mse"] = 1.01
    manifest = make_multi_manifest(records)
    stats = manifest["predicted_current_bridge_statistics"]
    assert stats["raw_predicted_not_worse_count"] == 3
    assert stats["raw_predicted_clearly_worse_than_matched_count"] == 0
    assert manifest["primary_diagnostic_label"] == audit.INCONCLUSIVE


def test_privileged_current_diagnostic_thresholds_remain_unchanged() -> None:
    records = make_multi_records(privileged_flow=0.70, privileged_x0=0.70)
    aggregates = audit.aggregate_teacher_flow_metrics(records)
    paired = audit.paired_teacher_flow_statistics(records)
    assert audit.privileged_current_generalization_label(
        aggregates=aggregates,
        paired_statistics=paired,
    ) == audit.STRONG_PRIVILEGED_CURRENT_SUPPORT

    weak_reduction = make_multi_records(privileged_flow=0.95, privileged_x0=0.50)
    assert audit.privileged_current_generalization_label(
        aggregates=audit.aggregate_teacher_flow_metrics(weak_reduction),
        paired_statistics=audit.paired_teacher_flow_statistics(weak_reduction),
    ) == audit.NO_PRIVILEGED_CURRENT_SUPPORT


def test_multi_matched_timestep_dependence_is_reported_not_primary_gate() -> None:
    records = make_multi_records(matched_flow=1.1)
    for record in records:
        if int(record["raw_timestep"]) in (500, 250):
            record["metrics"]["teacher_matched_flow_vs_exact_mse"] = 0.8
    manifest = make_multi_manifest(records)
    diagnostic = manifest["matched_teacher_timestep_diagnostic"]
    assert diagnostic["label"] == audit.MATCHED_TEACHER_TIMESTEP_DEPENDENCE
    assert diagnostic["not_primary_gate"] is True
    assert diagnostic["by_raw"]["999"]["direction"] == "worse_than_mcp"
    assert diagnostic["by_raw"]["500"]["direction"] == "better_than_mcp"


def test_multi_runner_cli_contract_requires_step6500_and_regular_context() -> None:
    args = runner.parse_args(
        [
            "--multi_identity_validation32",
            "--full_sequence_checkpoint",
            "checkpoint_step006500.pt",
            "--expected_checkpoint_step",
            "6500",
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
    runner._validate_multi_identity_cli_contract(args)

    args.expected_checkpoint_step = 7000
    with pytest.raises(RuntimeError, match="step6500"):
        runner._validate_multi_identity_cli_contract(args)
    args.expected_checkpoint_step = 6500
    args.student_direct_clean_context_kv = True
    with pytest.raises(RuntimeError, match="direct_clean_context_kv=false"):
        runner._validate_multi_identity_cli_contract(args)


def test_multi_streaming_manifest_keeps_only_json_safe_records() -> None:
    manifest = make_multi_manifest()
    assert manifest["streaming_contract"]["retains_full_state_tensors"] is False
    assert manifest["streaming_contract"]["retains_full_flow_tensors"] is False
    assert "tensors" not in manifest
    assert not _contains_tensor(manifest)


def test_default_single_identity_mode_behavior_is_unchanged() -> None:
    result = make_result()
    assert result.manifest["mode"] == audit.TEACHER_FLOW_AUDIT_MODE_SINGLE
    assert result.manifest["state_count"] == 16
    assert result.manifest["noise_realizations_per_raw"] == 4
    assert len(result.tensors["states"]) == 16


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
    assert args.multi_identity_validation32 is False
    assert args.predicted_current_oracle_recheck_only is False
    assert args.predicted_current_oracle_recheck_validation0_all_raw is False
    assert args.expected_training_git_sha is None
    assert runner._expected_training_git_sha(args, git_sha="a" * 40) == "a" * 40


def test_predicted_current_oracle_recheck_flag_parsing_and_mutex() -> None:
    base = [
        "--full_sequence_checkpoint",
        "checkpoint_step006500.pt",
        "--expected_checkpoint_step",
        "6500",
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
    args = runner.parse_args(["--predicted_current_oracle_recheck_only", *base])
    assert args.predicted_current_oracle_recheck_only is True
    assert args.multi_identity_validation32 is False
    assert args.predicted_current_oracle_recheck_validation0_all_raw is False
    runner._validate_multi_identity_cli_contract(args)

    all_raw_args = runner.parse_args(
        ["--predicted_current_oracle_recheck_validation0_all_raw", *base]
    )
    assert all_raw_args.predicted_current_oracle_recheck_validation0_all_raw is True
    assert all_raw_args.predicted_current_oracle_recheck_only is False
    assert all_raw_args.multi_identity_validation32 is False
    runner._validate_multi_identity_cli_contract(all_raw_args)

    with pytest.raises(SystemExit):
        runner.parse_args(
            [
                "--multi_identity_validation32",
                "--predicted_current_oracle_recheck_only",
                *base,
            ]
        )

    with pytest.raises(SystemExit):
        runner.parse_args(
            [
                "--multi_identity_validation32",
                "--predicted_current_oracle_recheck_validation0_all_raw",
                *base,
            ]
        )

    with pytest.raises(SystemExit):
        runner.parse_args(
            [
                "--predicted_current_oracle_recheck_only",
                "--predicted_current_oracle_recheck_validation0_all_raw",
                *base,
            ]
        )

    args.expected_checkpoint_step = 7000
    with pytest.raises(RuntimeError, match="step6500"):
        runner._validate_multi_identity_cli_contract(args)
    args.expected_checkpoint_step = 6500
    args.student_direct_clean_context_kv = True
    with pytest.raises(RuntimeError, match="direct_clean_context_kv=false"):
        runner._validate_multi_identity_cli_contract(args)


def _install_fake_recheck_runner(
    monkeypatch: pytest.MonkeyPatch,
    *,
    source_noise: torch.Tensor,
    teacher_target: torch.Tensor,
    oracle=None,
) -> dict[str, Any]:
    records: dict[str, Any] = {"acquired": [], "build_modes": [], "writes": []}
    plan = make_validation32_plan()
    schedule = ev.resolve_deployment_schedule()

    payload = {
        "prompt": "prompt",
        "prompt_sha256": "d" * 64,
        "noise_seed": 123,
        "rollout_seed": 456,
        "raw_denoising_steps": list(schedule.raw_schedule),
        "warped_denoising_steps": list(schedule.main_warped_schedule),
    }
    metadata = {
        "identity": "val000",
        "sample_index": 0,
        "split": "validation",
        "split_index": 0,
        "prompt_sha256": "d" * 64,
    }

    class FakeAcquire:
        def __enter__(self):
            return SimpleNamespace(
                payload=dict(payload),
                metadata=dict(metadata),
                source_noise=source_noise,
                target_latent=teacher_target,
            )

        def __exit__(self, exc_type, exc, traceback):
            return False

    class FakeStore:
        def __init__(self, **kwargs):
            records["store_kwargs"] = kwargs

        def acquire(self, identity: str):
            records["acquired"].append(str(identity))
            return FakeAcquire()

    student_checkpoint = ev.DeploymentCheckpointRecord(
        path="checkpoint_step006500.pt",
        sha256=TEST_SHA,
        checkpoint_type="teacher_flow_student_step6500",
        load_mode=audit.TEACHER_FLOW_AUDIT_STUDENT_LOAD_MODE,
        generator_state_dict={"mcp.fusion.weight": torch.ones(1)},
        global_step=6500,
        training_git_sha=RUNTIME_GIT_SHA,
        payload={
            "schema": FULL_SEQUENCE_TRAINER_SCHEMA,
            "global_step": 6500,
            "sample_plan_sha256": "b" * 64,
            "manifest_sha256": "c" * 64,
        },
    )
    teacher_checkpoint = ev.DeploymentCheckpointRecord(
        path="self_forcing_dmd.pt",
        sha256="d" * 64,
        checkpoint_type="official_self_forcing",
        load_mode="official",
        generator_state_dict={"weight": torch.zeros(1)},
    )

    def fake_build_generator(*, mode, **kwargs):
        _ = kwargs
        records["build_modes"].append(str(mode))
        if mode == runner.MODE_TRAINED_MCP1:
            generator = FakeStudentGenerator().eval().requires_grad_(False)
            generator.weight = nn.Parameter(torch.zeros(1), requires_grad=False)
            return generator
        if mode == runner.MODE_OFFICIAL_MAIN:
            generator = FakeTeacherGenerator().eval().requires_grad_(False)
            generator.weight = nn.Parameter(torch.zeros(1), requires_grad=False)
            return generator
        raise AssertionError(f"unexpected build mode: {mode}")

    def forbidden_branch(*args, **kwargs):
        _ = (args, kwargs)
        raise AssertionError("future branch must not run in recheck-only mode")

    monkeypatch.setattr(runner, "current_git_head", lambda: RUNTIME_GIT_SHA)
    monkeypatch.setattr(runner, "validate_cli_contract", lambda args, git_sha: {
        "status": "PASS",
        "runtime_git_sha": git_sha,
        "output_dir": str(args.output_dir),
    })
    monkeypatch.setattr(
        runner,
        "runtime_device",
        lambda device_arg: (torch.device("cpu"), {"device": str(device_arg)}),
    )
    monkeypatch.setattr(runner, "merge_config", lambda path: SimpleNamespace())
    monkeypatch.setattr(runner, "validate_config", lambda config: None)
    monkeypatch.setattr(runner, "load_m4_sample_plan", lambda *args, **kwargs: plan)
    monkeypatch.setattr(runner, "file_sha256", lambda path: "c" * 64)
    monkeypatch.setattr(
        runner,
        "load_teacher_flow_student_checkpoint_record",
        lambda *args, **kwargs: student_checkpoint,
    )
    monkeypatch.setattr(
        runner,
        "load_official_checkpoint_record",
        lambda path: teacher_checkpoint,
    )
    monkeypatch.setattr(runner, "M5TeacherSampleStore", FakeStore)
    monkeypatch.setattr(
        runner,
        "atomic_json_write",
        lambda payload, path: records["writes"].append(
            {"payload": dict(payload), "path": Path(path)}
        ),
    )
    monkeypatch.setattr(
        runner,
        "build_conditioning",
        lambda *, prompt, device, dtype: {
            "prompt_embeds": torch.zeros((1, 2, 3), dtype=torch.float32)
        },
    )
    monkeypatch.setattr(runner, "build_generator", fake_build_generator)
    monkeypatch.setattr(
        runner,
        "run_student_mcp_full_sequence_predictions",
        forbidden_branch,
    )
    monkeypatch.setattr(
        runner,
        "run_student_predicted_current_predictions",
        forbidden_branch,
    )
    monkeypatch.setattr(runner, "run_teacher_branch_predictions", forbidden_branch)
    if oracle is not None:
        monkeypatch.setattr(runner, "exact_current_flow_conversion_oracle", oracle)
    return records


def _recheck_argv(output_dir: Path) -> list[str]:
    return [
        "--predicted_current_oracle_recheck_only",
        "--full_sequence_checkpoint",
        "checkpoint_step006500.pt",
        "--expected_checkpoint_step",
        "6500",
        "--sample_plan",
        "sample_plan.json",
        "--teacher_manifest",
        "teacher_manifest.json",
        "--dataset_root",
        "dataset",
        "--output_dir",
        str(output_dir),
        "--expected_runtime_git_sha",
        RUNTIME_GIT_SHA,
        "--device",
        "cpu",
    ]


def _all_raw_recheck_argv(output_dir: Path) -> list[str]:
    return [
        "--predicted_current_oracle_recheck_validation0_all_raw",
        "--full_sequence_checkpoint",
        "checkpoint_step006500.pt",
        "--expected_checkpoint_step",
        "6500",
        "--sample_plan",
        "sample_plan.json",
        "--teacher_manifest",
        "teacher_manifest.json",
        "--dataset_root",
        "dataset",
        "--output_dir",
        str(output_dir),
        "--expected_runtime_git_sha",
        RUNTIME_GIT_SHA,
        "--device",
        "cpu",
    ]


def test_predicted_current_recheck_runner_bf16_artifact_and_no_branches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, target = make_bf16_oracle_failure_tensors()
    output_dir = Path("out")
    records = _install_fake_recheck_runner(
        monkeypatch,
        source_noise=source,
        teacher_target=target,
    )

    rc = runner.main(_recheck_argv(output_dir))
    assert len(records["writes"]) == 1
    artifact = records["writes"][0]["payload"]

    assert rc == 0
    assert artifact["schema"] == audit.PREDICTED_CURRENT_ORACLE_RECHECK_SCHEMA
    assert artifact["diagnostic_classification"] == (
        audit.BF16_QUANTIZED_STATE_CONTRACT
    )
    assert artifact["original_bf16_oracle_pass"] is False
    assert artifact["identity_index"] == 0
    assert artifact["validation_position"] == 0
    assert artifact["raw_timestep"] == 999
    assert artifact["noise_index"] == 0
    assert artifact["sample_identity"] == "val000"
    assert artifact["current_state_vs_regenerated"]["torch_equal"] is True
    assert artifact["scheduler_vs_explicit"]["max_abs"] == pytest.approx(0.0)
    assert artifact["float32_reference"]["max_abs"] == pytest.approx(0.0)
    assert artifact["student_parameters_unchanged"] is True
    assert artifact["teacher_parameters_unchanged"] is True
    assert artifact["rng_unchanged"] is True
    assert artifact["backward_executed"] is False
    assert artifact["optimizer_step_executed"] is False
    assert artifact["checkpoint_written"] is False
    assert records["acquired"] == ["val000"]
    assert records["writes"][0]["path"].name == (
        "predicted_current_oracle_recheck.json"
    )
    assert records["build_modes"] == [
        runner.MODE_TRAINED_MCP1,
        runner.MODE_OFFICIAL_MAIN,
    ]


def test_predicted_current_recheck_artifact_before_nonzero_scheduler_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target, states, _, mcp_scheduler = make_oracle_state()
    with pytest.raises(RuntimeError) as excinfo:
        audit.exact_current_flow_conversion_oracle(
            mcp_scheduler,
            state=states[0],
            teacher_target=target,
        )
    diagnostic = oracle_failure_diagnostic(excinfo.value)

    def failing_oracle(*args, **kwargs):
        _ = (args, kwargs)
        raise RuntimeError(
            "exact current flow-to-x0 conversion oracle failed: "
            f"scheduler_shift_not_main; diagnostic={json.dumps(diagnostic)}"
        )

    source, target_bf16 = make_bf16_oracle_failure_tensors()
    output_dir = Path("out")
    records = _install_fake_recheck_runner(
        monkeypatch,
        source_noise=source,
        teacher_target=target_bf16,
        oracle=failing_oracle,
    )

    rc = runner.main(_recheck_argv(output_dir))
    assert len(records["writes"]) == 1
    artifact_path = records["writes"][0]["path"]
    artifact = records["writes"][0]["payload"]

    assert rc == 1
    assert artifact_path.name == "predicted_current_oracle_recheck.json"
    assert artifact["status"] == "FAIL"
    assert artifact["diagnostic_classification"] == audit.SCHEDULER_MISMATCH


def test_predicted_current_all_raw_recheck_writes_four_records_after_one_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = make_source_noise().to(dtype=torch.bfloat16)
    target = make_teacher_target().to(dtype=torch.bfloat16)
    bad_diagnostic: dict[str, Any] | None = None
    seen_raws: list[int] = []

    def oracle(main_scheduler, *, state, teacher_target):
        nonlocal bad_diagnostic
        seen_raws.append(int(state.raw_timestep))
        if int(state.raw_timestep) == 750:
            if bad_diagnostic is None:
                wrong_scheduler = audit.build_flow_match_scheduler(
                    shift=DEFAULT_S_MCP,
                    device="cpu",
                )
                with pytest.raises(RuntimeError) as excinfo:
                    audit.exact_current_flow_conversion_oracle(
                        wrong_scheduler,
                        state=state,
                        teacher_target=teacher_target,
                    )
                bad_diagnostic = oracle_failure_diagnostic(excinfo.value)
            raise RuntimeError(
                "exact current flow-to-x0 conversion oracle failed: "
                f"scheduler_shift_not_main; diagnostic={json.dumps(bad_diagnostic)}"
            )
        return audit.exact_current_flow_conversion_oracle(
            main_scheduler,
            state=state,
            teacher_target=teacher_target,
        )

    output_dir = Path("out")
    records = _install_fake_recheck_runner(
        monkeypatch,
        source_noise=source,
        teacher_target=target,
        oracle=oracle,
    )

    rc = runner.main(_all_raw_recheck_argv(output_dir))
    assert len(records["writes"]) == 1
    artifact_path = records["writes"][0]["path"]
    artifact = records["writes"][0]["payload"]

    assert rc == 1
    assert artifact_path.name == (
        "predicted_current_oracle_recheck_validation0_all_raw.json"
    )
    assert seen_raws == [999, 750, 500, 250]
    assert artifact["schema"] == audit.PREDICTED_CURRENT_ORACLE_RECHECK_ALL_RAW_SCHEMA
    assert artifact["status"] == "FAIL"
    assert artifact["state_count"] == 4
    assert [record["raw_timestep"] for record in artifact["states"]] == [
        999,
        750,
        500,
        250,
    ]
    assert [record["noise_index"] for record in artifact["states"]] == [0, 0, 0, 0]
    assert artifact["states"][1]["diagnostic_classification"] == (
        audit.SCHEDULER_MISMATCH
    )
    assert artifact["states"][0]["state_vs_regenerated"]["torch_equal"] is True
    assert artifact["backward_executed"] is False
    assert artifact["optimizer_step_executed"] is False
    assert artifact["checkpoint_written"] is False
    assert artifact["student_parameters_unchanged"] is True
    assert artifact["teacher_parameters_unchanged"] is True
    assert artifact["rng_unchanged"] is True
    assert records["acquired"] == ["val000"]
    assert records["build_modes"] == [
        runner.MODE_TRAINED_MCP1,
        runner.MODE_OFFICIAL_MAIN,
    ]


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


def test_multi_student_checkpoint_contract_requires_formal_step6500() -> None:
    checkpoint = ev.DeploymentCheckpointRecord(
        path="student.pt",
        sha256=TEST_SHA,
        checkpoint_type="teacher_flow_student_step6500",
        load_mode=audit.TEACHER_FLOW_AUDIT_STUDENT_LOAD_MODE,
        generator_state_dict={"mcp.fusion.weight": torch.ones(1)},
        global_step=6500,
        training_git_sha=TRAINING_GIT_SHA,
        payload={
            "schema": FULL_SEQUENCE_TRAINER_SCHEMA,
            "global_step": 6500,
        },
    )
    result = audit.validate_multi_identity_student_checkpoint_contract(checkpoint)
    assert result["status"] == "PASS"

    diagnostic = ev.DeploymentCheckpointRecord(
        path="student.pt",
        sha256=TEST_SHA,
        checkpoint_type="teacher_flow_student_step6500",
        load_mode=audit.TEACHER_FLOW_AUDIT_STUDENT_LOAD_MODE,
        generator_state_dict={"mcp.fusion.weight": torch.ones(1)},
        global_step=6500,
        training_git_sha=TRAINING_GIT_SHA,
        payload={
            "schema": audit.NF_SF_MCP1_ONLY_CONTINUATION_SCHEMA,
            "global_step": 6500,
        },
    )
    with pytest.raises(RuntimeError, match="formal full-sequence"):
        audit.validate_multi_identity_student_checkpoint_contract(diagnostic)

    wrong_step = ev.DeploymentCheckpointRecord(
        path="student.pt",
        sha256=TEST_SHA,
        checkpoint_type="teacher_flow_student_step7000",
        load_mode=audit.TEACHER_FLOW_AUDIT_STUDENT_LOAD_MODE,
        generator_state_dict={"mcp.fusion.weight": torch.ones(1)},
        global_step=7000,
        training_git_sha=TRAINING_GIT_SHA,
        payload={
            "schema": FULL_SEQUENCE_TRAINER_SCHEMA,
            "global_step": 7000,
        },
    )
    with pytest.raises(RuntimeError, match="step6500"):
        audit.validate_multi_identity_student_checkpoint_contract(wrong_step)


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
