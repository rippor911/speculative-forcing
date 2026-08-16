from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import utils.nf_sf_first_mcp_route_equivalence as audit
import utils.nf_sf_full_sequence_eval as ev
from utils.nf_sf_tensors import DEFAULT_NUM_TRAIN_TIMESTEPS, DEFAULT_S_MAIN, DEFAULT_S_MCP
from utils.scheduler import FlowMatchScheduler


RUNTIME_GIT_SHA = "e" * 40
TRAINING_GIT_SHA = ev.TRAINING_CHECKPOINT_GIT_SHA
TEST_SHA = "a" * 64
FRAME_SEQ_LENGTH = 2


class CountingFlowMatchScheduler(FlowMatchScheduler):
    def __init__(self, *, shift: float) -> None:
        super().__init__(shift=shift, sigma_min=0.0, extra_one_step=True)
        self.set_timesteps(DEFAULT_NUM_TRAIN_TIMESTEPS, training=True)
        self.training_target_calls = 0
        self.step_calls = 0

    def training_target(self, sample, noise, timestep):
        self.training_target_calls += 1
        return super().training_target(sample, noise, timestep)

    def step(self, model_output, timestep, sample, to_final=False):
        if to_final:
            self.step_calls += 1
        return super().step(model_output, timestep, sample, to_final=to_final)


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
        outputs = []
        for embed in future_embeds:
            outputs.append(
                torch.zeros(
                    (embed.shape[0], 1, 3, 1, 1),
                    device=embed.device,
                    dtype=embed.dtype,
                )
            )
        return outputs


class FakeModel:
    def __init__(self) -> None:
        self.block_mask = None


@dataclass
class FakeFullSequenceOutputs:
    main_flow_pred: torch.Tensor
    mcp_flow_preds_by_depth: tuple[torch.Tensor, ...]
    tap_shapes: tuple[tuple[int, ...], ...]
    anchor_token_slices: tuple[tuple[int, int], ...]
    future_embedding_order: str = "depth_major"
    main_backbone_forward_count: int = 1


class FakeGenerator(nn.Module):
    def __init__(
        self,
        *,
        consume_rng_in_joint: bool = False,
        mcp_nonfinite: str | None = None,
        training_feature_offset: float = 0.0,
        training_mcp_bias: float = 0.0,
        raise_in_training_forward: bool = False,
    ) -> None:
        super().__init__()
        self.model = FakeModel()
        self.mcp = FakeMCP()
        self.calls: list[dict[str, Any]] = []
        self.deployment_joint_block_mask_before: list[bool] = []
        self.forward_full_sequence_calls = 0
        self.training_anchor_inputs: tuple[dict, ...] = ()
        self.consume_rng_in_joint = bool(consume_rng_in_joint)
        self.mcp_nonfinite = mcp_nonfinite
        self.training_feature_offset = float(training_feature_offset)
        self.training_mcp_bias = float(training_mcp_bias)
        self.raise_in_training_forward = bool(raise_in_training_forward)

    def forward(self, **kwargs):
        current = kwargs["noisy_image_or_video"]
        timestep = kwargs["timestep"]
        current_start = int(kwargs["current_start"])
        kv_cache = kwargs["kv_cache"]
        mcp_requested = kwargs.get("mcp_future_noises") is not None
        if mcp_requested:
            self.deployment_joint_block_mask_before.append(self.model.block_mask is None)
        if mcp_requested and self.consume_rng_in_joint:
            torch.randn((), device=current.device)
        token_count = int(current.shape[1]) * FRAME_SEQ_LENGTH
        token_end = current_start + token_count
        self.calls.append(
            {
                "mcp_requested": mcp_requested,
                "current_start": current_start,
                "token_end": token_end,
                "is_context": bool((timestep.detach().float() == 0).all().item()),
            }
        )
        for layer in kv_cache:
            layer["k"][:, current_start:token_end] = 1.0 + len(self.calls)
            layer["v"][:, current_start:token_end] = 2.0 + len(self.calls)
            layer["global_end_index"].fill_(token_end)
            layer["local_end_index"].fill_(token_end)
        main_flow = current * 0.125
        main_x0 = current
        if not mcp_requested:
            return main_flow, main_x0
        future = kwargs["mcp_future_noises"][0]
        features = self._features_from_chunk(current, route="deployment")
        future_embed = self._future_embed(future)
        self.mcp(
            features=features,
            future_embeds=[future_embed],
            future_grid_sizes=[torch.tensor([1, 1, 1], device=current.device)],
            future_start_frames=list(kwargs["mcp_future_start_frames"]),
            timesteps=list(kwargs["mcp_timesteps"]),
            freqs=None,
        )
        if self.mcp_nonfinite == "nan":
            mcp_flow = torch.full_like(future, float("nan"))
        elif self.mcp_nonfinite == "inf":
            mcp_flow = torch.full_like(future, float("inf"))
        else:
            mcp_flow = future * 0.25
        return main_flow, main_x0, [mcp_flow]

    def forward_full_sequence_next_forcing(
        self,
        *,
        noisy_image_or_video,
        clean_x,
        conditional_dict,
        timestep_main,
        mcp_anchor_inputs=(),
        aug_t=None,
    ):
        _ = (clean_x, conditional_dict, aug_t)
        self.forward_full_sequence_calls += 1
        self.training_anchor_inputs = tuple(mcp_anchor_inputs)
        if self.model.block_mask is not None:
            raise RuntimeError("training fake expected block_mask None before TF forward")
        self.model.block_mask = torch.ones(1, dtype=torch.bool)
        if self.raise_in_training_forward:
            raise RuntimeError("training boom")
        if self.consume_rng_in_joint:
            torch.randn((), device=noisy_image_or_video.device)
        by_depth: dict[int, list[torch.Tensor]] = {1: [], 2: [], 3: []}
        for anchor in mcp_anchor_inputs:
            anchor_index = int(anchor["anchor_index"])
            depths = tuple(int(depth) for depth in anchor["depths"])
            current = noisy_image_or_video[:, anchor_index * 3:(anchor_index + 1) * 3]
            features = self._features_from_chunk(current, route="training")
            future_embeds = [self._future_embed(tensor) for tensor in anchor["future_noises"]]
            self.mcp(
                features=features,
                future_embeds=future_embeds,
                future_grid_sizes=[
                    torch.tensor([1, 1, 1], device=noisy_image_or_video.device)
                    for _ in future_embeds
                ],
                future_start_frames=list(anchor["future_start_frames"]),
                timesteps=list(anchor["timesteps"]),
                freqs=None,
            )
            for depth, future in zip(depths, anchor["future_noises"]):
                by_depth[depth].append(future * 0.25 + self.training_mcp_bias)
        main_flow = noisy_image_or_video * 0.125
        outputs = []
        for depth in (1, 2, 3):
            outputs.append(torch.stack(by_depth[depth], dim=1))
        return FakeFullSequenceOutputs(
            main_flow_pred=main_flow,
            mcp_flow_preds_by_depth=tuple(outputs),
            tap_shapes=tuple((1, 2, 1) for _ in range(4)),
            anchor_token_slices=tuple((index * 6, (index + 1) * 6) for index in range(7)),
        )

    def _features_from_chunk(self, chunk: torch.Tensor, *, route: str):
        base = chunk.reshape(chunk.shape[0], -1, 1)
        if route == "training":
            base = base + self.training_feature_offset
        return tuple(base + float(index) for index in range(4))

    @staticmethod
    def _future_embed(future: torch.Tensor) -> torch.Tensor:
        return future.reshape(future.shape[0], -1, 1)


def make_runtime(generator: FakeGenerator) -> ev.DeploymentRuntime:
    capacity = ev.FULL_SEQUENCE_FRAME_COUNT * FRAME_SEQ_LENGTH
    kv_cache = [
        {
            "k": torch.zeros((1, capacity, 1, 1), dtype=torch.float32),
            "v": torch.zeros((1, capacity, 1, 1), dtype=torch.float32),
            "global_end_index": torch.tensor([0], dtype=torch.long),
            "local_end_index": torch.tensor([0], dtype=torch.long),
        }
        for _ in range(2)
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


def make_audit_result(
    *,
    generator: FakeGenerator | None = None,
    main_scheduler: CountingFlowMatchScheduler | None = None,
    mcp_scheduler: CountingFlowMatchScheduler | None = None,
) -> audit.FirstMCPRouteEquivalenceResult:
    generator = generator or FakeGenerator()
    source = make_source_noise()
    teacher = make_teacher_target()
    common, fingerprint = make_common(source, teacher)
    return audit.run_first_mcp_route_equivalence_audit(
        runtime_factory=lambda: make_runtime(generator),
        generator=generator,
        main_scheduler=main_scheduler
        or CountingFlowMatchScheduler(shift=DEFAULT_S_MAIN),
        mcp_scheduler=mcp_scheduler
        or CountingFlowMatchScheduler(shift=DEFAULT_S_MCP),
        source_noise=source,
        teacher_target=teacher,
        teacher_payload={"rollout_seed": 123, "prompt": "prompt"},
        conditional_dict={"prompt_embeds": torch.zeros((1, 2, 3))},
        checkpoint_summary={
            "type": "full_sequence_step5000",
            "checkpoint_type": "full_sequence_step5000",
            "load_mode": "FULL_SEQUENCE_GENERATOR_STRICT_WITH_MCP",
            "sha256": TEST_SHA,
            "global_step": 5000,
            "training_git_sha": TRAINING_GIT_SHA,
            "expected_checkpoint_step": 5000,
            "loaded_checkpoint_global_step": 5000,
            "checkpoint_loader_mode": audit.CHECKPOINT_LOADER_MODE_FINAL,
            "diagnostic_intermediate_checkpoint": False,
        },
        common_inputs=common,
        common_inputs_fingerprint_sha256=fingerprint,
        runtime_git_sha=RUNTIME_GIT_SHA,
        training_checkpoint_git_sha=TRAINING_GIT_SHA,
    )


def make_intermediate_payload(step: int) -> dict[str, Any]:
    return {
        "schema": audit.FULL_SEQUENCE_TRAINER_SCHEMA,
        "run_kind": audit.FULL_SEQUENCE_RUN_KIND,
        "objective_version": audit.FULL_SEQUENCE_OBJECTIVE_VERSION,
        "objective_mode": ev.FULL_SEQUENCE_OBJECTIVE_MODE,
        "status": "PRODUCTION",
        "global_step": int(step),
        "git_sha": TRAINING_GIT_SHA,
        "generator": {
            "model.weight": torch.zeros(1),
            "mcp.depth1.weight": torch.ones(1),
        },
        "optimizer": {"state": {}},
        "train_rng_state": torch.get_rng_state(),
        "validation_seed": 456,
        "validation_base_rng_state": torch.get_rng_state(),
        "python_random_state": (3, (), None),
        "torch_cpu_global_rng_state": torch.get_rng_state(),
        "torch_cuda_global_rng_state": None,
        "sample_cursor": audit.nf_sf_full_sequence_train_cursor(int(step)),
        "sample_plan_sha256": "b" * 64,
        "manifest_sha256": "c" * 64,
        "conditionals_artifact_sha256": "d" * 64,
        "resolved_config": {
            "num_frame_per_block": 3,
            "gradient_checkpointing": True,
        },
        "provenance": {
            "schema": audit.FULL_SEQUENCE_TRAINER_SCHEMA,
            "run_kind": audit.FULL_SEQUENCE_RUN_KIND,
            "objective_version": audit.FULL_SEQUENCE_OBJECTIVE_VERSION,
            "paper_exact_reproduction": False,
        },
        "reference_checkpoint": {
            "path": "checkpoints/self_forcing_dmd.pt",
            "sha256": ev.OFFICIAL_SELF_FORCING_CHECKPOINT_SHA256,
            "size_bytes": 11,
        },
        "optimizer_contract": {"class": "AdamW"},
    }


def write_intermediate_checkpoint(
    tmp_path: Path,
    *,
    step: int,
    payload_updates: dict[str, Any] | None = None,
    validation_updates: dict[str, Any] | None = None,
    filename: str | None = None,
    sha_sidecar_text: str | None = None,
) -> Path:
    payload = make_intermediate_payload(step)
    if payload_updates:
        payload.update(payload_updates)
    path = tmp_path / (filename or f"checkpoint_step{int(step):06d}.pt")
    torch.save(payload, path)
    actual_sha = ev.file_sha256(path)
    stem = path.with_suffix("")
    sha_text = sha_sidecar_text or f"{actual_sha}  {path.name}\n"
    stem.with_suffix(".sha256.txt").write_text(sha_text, encoding="utf-8")
    validation = {
        "status": "PASS",
        "path": str(path.resolve()),
        "sha256": actual_sha,
        "size_bytes": int(path.stat().st_size),
        "schema": ev.CHECKPOINT_VALIDATION_SCHEMA,
        "run_kind": audit.FULL_SEQUENCE_RUN_KIND,
        "objective_version": audit.FULL_SEQUENCE_OBJECTIVE_VERSION,
        "objective_mode": ev.FULL_SEQUENCE_OBJECTIVE_MODE,
        "global_step": int(step),
        "generator_key_count": len(payload["generator"])
        if isinstance(payload.get("generator"), dict)
        else 0,
        "optimizer_state_entry_count": 0,
    }
    if validation_updates:
        validation.update(validation_updates)
    stem.with_suffix(".validation.json").write_text(
        json.dumps(validation),
        encoding="utf-8",
    )
    return path


def test_raw_timestep_support_contract() -> None:
    assert audit.raw_timestep_in_training_support(999) is True
    assert audit.raw_timestep_in_training_support(1000) is False
    result = make_audit_result()
    points = result.manifest["points"]
    assert points[audit.POINT_TRAINING_EDGE]["in_training_raw_support"] is True
    assert points[audit.POINT_DEPLOYMENT_ENDPOINT]["in_training_raw_support"] is False
    support = result.manifest["training_raw_support_contract"]
    assert support["training_raw_min"] == 0
    assert support["training_raw_max_inclusive"] == 999


@pytest.mark.parametrize("step", [0, 500, 2000])
def test_intermediate_checkpoint_loader_accepts_strict_payload(
    tmp_path: Path,
    step: int,
) -> None:
    path = write_intermediate_checkpoint(tmp_path, step=step)
    record = audit.load_route_equivalence_checkpoint_record(
        path,
        expected_checkpoint_step=step,
        expected_training_git_sha=TRAINING_GIT_SHA,
    )
    assert record.global_step == step
    assert record.checkpoint_type == f"full_sequence_step{step}"
    assert record.load_mode == audit.CHECKPOINT_LOADER_MODE_INTERMEDIATE
    assert record.training_git_sha == TRAINING_GIT_SHA
    assert record.validation_sidecar["global_step"] == step
    assert record.payload["generator"]["mcp.depth1.weight"].item() == 1.0


def test_intermediate_checkpoint_loader_rejects_wrong_expected_step(
    tmp_path: Path,
) -> None:
    path = write_intermediate_checkpoint(tmp_path, step=500)
    with pytest.raises(RuntimeError, match="filename"):
        audit.load_route_equivalence_checkpoint_record(
            path,
            expected_checkpoint_step=0,
            expected_training_git_sha=TRAINING_GIT_SHA,
        )


def test_intermediate_checkpoint_loader_rejects_unsupported_step() -> None:
    with pytest.raises(ValueError, match="0, 500, 2000, 5000"):
        audit.load_route_equivalence_checkpoint_record(
            Path("checkpoint_step000125.pt"),
            expected_checkpoint_step=125,
            expected_training_git_sha=TRAINING_GIT_SHA,
        )


def test_final_checkpoint_loader_delegates_to_canonical(monkeypatch) -> None:
    calls: list[dict[str, Any]] = []

    def fake_canonical_loader(path, *, expected_training_git_sha, expected_official_sha256):
        calls.append(
            {
                "path": Path(path),
                "expected_training_git_sha": expected_training_git_sha,
                "expected_official_sha256": expected_official_sha256,
            }
        )
        return ev.DeploymentCheckpointRecord(
            path=str(Path(path)),
            sha256=TEST_SHA,
            checkpoint_type="full_sequence_step5000",
            load_mode="FULL_SEQUENCE_GENERATOR_STRICT_WITH_MCP",
            generator_state_dict={"mcp.depth1.weight": torch.ones(1)},
            global_step=5000,
            training_git_sha=expected_training_git_sha,
            payload={"global_step": 5000, "git_sha": expected_training_git_sha},
            validation_sidecar={"global_step": 5000},
        )

    monkeypatch.setattr(
        audit.deployment,
        "load_full_sequence_checkpoint_record",
        fake_canonical_loader,
    )
    record = audit.load_route_equivalence_checkpoint_record(
        Path("checkpoint_step005000.pt"),
        expected_checkpoint_step=5000,
        expected_training_git_sha=TRAINING_GIT_SHA,
    )
    assert len(calls) == 1
    assert record.checkpoint_type == "full_sequence_step5000"
    assert audit.route_equivalence_checkpoint_loader_mode(5000) == (
        audit.CHECKPOINT_LOADER_MODE_FINAL
    )


def test_route_equivalence_cli_requires_expected_checkpoint_step() -> None:
    from scripts import diagnose_nf_sf_first_mcp_route_equivalence as route_cli

    argv = [
        "--full_sequence_checkpoint",
        "checkpoint_step000500.pt",
        "--sample_plan",
        "sample_plan.json",
        "--teacher_manifest",
        "manifest.json",
        "--dataset_root",
        "dataset",
        "--output_dir",
        "out",
        "--expected_runtime_git_sha",
        RUNTIME_GIT_SHA,
    ]
    with pytest.raises(SystemExit):
        route_cli.parse_args(argv)
    parsed = route_cli.parse_args(
        argv[:2] + ["--expected_checkpoint_step", "500"] + argv[2:]
    )
    assert parsed.expected_checkpoint_step == 500
    with pytest.raises(SystemExit):
        route_cli.parse_args(
            argv[:2] + ["--expected_checkpoint_step", "125"] + argv[2:]
        )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"filename": "checkpoint_step000500.pt"}, "filename"),
        ({"sha_sidecar_text": "0" * 64 + "  checkpoint_step000000.pt\n"}, "SHA256 sidecar"),
        ({"validation_updates": {"global_step": 500}}, "global_step"),
        ({"validation_updates": {"sha256": "1" * 64}}, "validation SHA"),
        ({"validation_updates": {"path": "wrong.pt"}}, "validation path"),
        ({"validation_updates": {"status": "FAIL"}}, "status"),
        ({"validation_updates": {"schema": "bad"}}, "schema"),
        ({"payload_updates": {"schema": "bad"}}, "schema mismatch"),
        ({"payload_updates": {"global_step": 500}}, "global_step"),
        ({"validation_updates": {"objective_version": "bad"}}, "objective_version"),
        ({"payload_updates": {"git_sha": "f" * 40}}, "training git_sha"),
        (
            {
                "payload_updates": {
                    "reference_checkpoint": {
                        "path": "checkpoints/self_forcing_dmd.pt",
                        "sha256": "e" * 64,
                        "size_bytes": 11,
                    }
                }
            },
            "official parent",
        ),
        (
            {
                "payload_updates": {
                    "provenance": {
                        "schema": audit.FULL_SEQUENCE_TRAINER_SCHEMA,
                        "run_kind": audit.FULL_SEQUENCE_RUN_KIND,
                        "objective_version": audit.FULL_SEQUENCE_OBJECTIVE_VERSION,
                        "paper_exact_reproduction": True,
                    }
                }
            },
            "paper_exact_reproduction",
        ),
    ],
)
def test_intermediate_checkpoint_loader_rejects_tamper(
    tmp_path: Path,
    kwargs: dict[str, Any],
    match: str,
) -> None:
    path = write_intermediate_checkpoint(tmp_path, step=0, **kwargs)
    with pytest.raises(RuntimeError, match=match):
        audit.load_route_equivalence_checkpoint_record(
            path,
            expected_checkpoint_step=0,
            expected_training_git_sha=TRAINING_GIT_SHA,
        )


def test_intermediate_checkpoint_loader_rejects_missing_mcp(tmp_path: Path) -> None:
    path = write_intermediate_checkpoint(
        tmp_path,
        step=0,
        payload_updates={"generator": {"model.weight": torch.zeros(1)}},
    )
    with pytest.raises(RuntimeError, match="MCP tensors"):
        audit.load_route_equivalence_checkpoint_record(
            path,
            expected_checkpoint_step=0,
            expected_training_git_sha=TRAINING_GIT_SHA,
        )


def test_route_manifest_records_checkpoint_progression_metadata() -> None:
    result = make_audit_result()
    assert result.manifest["expected_checkpoint_step"] == 5000
    assert result.manifest["loaded_checkpoint_global_step"] == 5000
    assert result.manifest["checkpoint_loader_mode"] == audit.CHECKPOINT_LOADER_MODE_FINAL
    assert result.manifest["diagnostic_intermediate_checkpoint"] is False
    assert result.manifest["checkpoint_sha256"] == TEST_SHA


def test_warped_timestep_uses_repo_function(monkeypatch) -> None:
    calls = []

    def fake_shift(timestep, *, shift, num_train_timesteps):
        calls.append((float(timestep.item()), float(shift), int(num_train_timesteps)))
        return timestep.float() + float(shift)

    monkeypatch.setattr(audit, "flow_match_shift_timesteps", fake_shift)
    point = audit.build_route_equivalence_point(123)
    assert point["main_warped_timestep"] == 128.0
    assert point["mcp_warped_timestep"] == 133.0
    assert calls == [(123.0, 5.0, 1000), (123.0, 10.0, 1000)]


def test_deployment_route_contract_and_kv_rollback() -> None:
    generator = FakeGenerator()
    result = make_audit_result(generator=generator)
    point = result.manifest["points"][audit.POINT_DEPLOYMENT_ENDPOINT]
    deployment = point["deployment"]
    assert deployment["teacher_history_chunks"] == [0]
    assert deployment["current_start_frame"] == 3
    assert deployment["future_start_frame"] == 6
    assert deployment["depths_used"] == [1]
    assert deployment["mcp_call_count"] == 1
    assert deployment["forward_rng"]["unchanged"] is True
    assert deployment["kv_rollback_exact"] is True
    assert deployment["deployment_block_mask_before_is_none"] is True
    assert deployment["deployment_block_mask_after_is_none"] is True
    assert deployment["selected_mcp_pre_hook"]["future_start_frames"] == [6]
    assert any(call["current_start"] == 3 * FRAME_SEQ_LENGTH for call in generator.calls)


def test_training_route_calls_full_sequence_and_all_anchor_contract() -> None:
    generator = FakeGenerator()
    result = make_audit_result(generator=generator)
    point = result.manifest["points"][audit.POINT_TRAINING_EDGE]
    training = point["training_route"]
    assert generator.forward_full_sequence_calls == 2
    assert training["forward_full_sequence_next_forcing_called"] is True
    assert training["training_block_mask_before_is_none"] is True
    assert training["training_block_mask_created"] is True
    assert training["training_block_mask_restored_is_none"] is True
    assert training["anchor_count"] == 6
    assert training["flat_anchor_future_count"] == 15
    assert training["selected_output"] == {
        "depth": 1,
        "anchor_index": 1,
        "target_chunk_index": 2,
    }
    assert [anchor["anchor_index"] for anchor in generator.training_anchor_inputs] == [0, 1, 2, 3, 4, 5]
    assert training["selected_mcp_pre_hook"]["future_start_frames"] == [6, 9, 12]
    assert generator.model.block_mask is None


def test_training_route_restores_block_mask_before_next_deployment() -> None:
    generator = FakeGenerator()
    result = make_audit_result(generator=generator)
    endpoint = result.manifest["points"][audit.POINT_DEPLOYMENT_ENDPOINT]
    edge = result.manifest["points"][audit.POINT_TRAINING_EDGE]
    assert endpoint["training_route"]["training_block_mask_created"] is True
    assert endpoint["training_route"]["training_block_mask_restored_is_none"] is True
    assert edge["deployment"]["deployment_block_mask_before_is_none"] is True
    assert generator.deployment_joint_block_mask_before == [True, True]
    assert generator.model.block_mask is None


def test_training_route_exception_restores_hook_and_block_mask() -> None:
    generator = FakeGenerator(raise_in_training_forward=True)
    with pytest.raises(RuntimeError, match="training boom"):
        make_audit_result(generator=generator)
    assert generator.model.block_mask is None
    assert len(generator.mcp._forward_pre_hooks) == 0


def test_raw1000_states_equal_source_noise_and_raw999_uses_add_noise() -> None:
    result = make_audit_result()
    endpoint = result.manifest["points"][audit.POINT_DEPLOYMENT_ENDPOINT]
    edge = result.manifest["points"][audit.POINT_TRAINING_EDGE]
    assert endpoint["deployment"]["raw1000_current_state_equals_source"] is True
    assert endpoint["deployment"]["raw1000_future_state_equals_source"] is True
    assert endpoint["training_route"]["raw1000_current_state_equals_source"] is True
    assert endpoint["training_route"]["raw1000_future_state_equals_source"] is True
    assert edge["deployment"]["raw1000_current_state_equals_source"] is False
    assert edge["training_route"]["raw1000_future_state_equals_source"] is False


def test_exact_targets_and_x0_use_scheduler_methods() -> None:
    main_scheduler = CountingFlowMatchScheduler(shift=DEFAULT_S_MAIN)
    mcp_scheduler = CountingFlowMatchScheduler(shift=DEFAULT_S_MCP)
    _ = make_audit_result(main_scheduler=main_scheduler, mcp_scheduler=mcp_scheduler)
    assert main_scheduler.training_target_calls >= 4
    assert mcp_scheduler.training_target_calls >= 18
    assert main_scheduler.step_calls >= 4
    assert mcp_scheduler.step_calls >= 4


def test_pre_hook_selection_rules_for_training_and_deployment() -> None:
    result = make_audit_result()
    for point in result.manifest["points"].values():
        assert point["deployment"]["selected_mcp_pre_hook"]["future_start_frames"] == [6]
        assert point["training_route"]["selected_mcp_pre_hook"]["future_start_frames"] == [6, 9, 12]
        assert point["route_comparison"]["future_start_exact"] is True
        assert point["route_comparison"]["future_grid_exact"] is True
        assert point["route_comparison"]["mcp_timestep_exact"] is True
        assert point["route_comparison"]["current_state_route_sha_exact"] is True
        assert point["route_comparison"]["future_state_route_sha_exact"] is True
        assert point["route_comparison"]["main_timestep_route_exact"] is True
        assert point["route_comparison"]["mcp_timestep_route_exact"] is True


def test_route_comparison_detects_exact_and_mismatch() -> None:
    exact = make_audit_result()
    exact_cmp = exact.manifest["points"][audit.POINT_DEPLOYMENT_ENDPOINT]["route_comparison"]
    assert exact_cmp["future_embed_sha_exact"] is True
    assert exact_cmp["mcp_flow_sha_exact"] is True

    mismatch = make_audit_result(
        generator=FakeGenerator(training_feature_offset=0.5, training_mcp_bias=0.125)
    )
    mismatch_cmp = mismatch.manifest["points"][audit.POINT_TRAINING_EDGE][
        "route_comparison"
    ]
    assert mismatch_cmp["tap_feature_sha_exact_by_tap"] == [False, False, False, False]
    assert mismatch_cmp["mcp_flow_sha_exact"] is False
    assert mismatch_cmp["mcp_flow_route_mse"] > 0.0


def test_nonfinite_fail_closed_and_hook_removed() -> None:
    generator = FakeGenerator(mcp_nonfinite="nan")
    with pytest.raises(RuntimeError, match="nonfinite"):
        make_audit_result(generator=generator)
    assert len(generator.mcp._forward_pre_hooks) == 0


def test_unexpected_rng_consumption_rejects_and_hook_removed() -> None:
    generator = FakeGenerator(consume_rng_in_joint=True)
    with pytest.raises(RuntimeError, match="changed active global RNG state"):
        make_audit_result(generator=generator)
    assert len(generator.mcp._forward_pre_hooks) == 0


def test_manifest_hypotheses_all_null() -> None:
    result = make_audit_result()
    contract = result.manifest["interpretation_contract"]
    assert contract["training_deployment_route_mismatch_supported"] is None
    assert contract["deployment_endpoint_extrapolation_supported"] is None
    assert contract["mcp_model_objective_failure_supported"] is None
    assert contract["backbone_feature_route_mismatch_supported"] is None
    assert contract["mcp_input_embedding_route_mismatch_supported"] is None


def test_manifest_validator_rejects_bad_support_and_route_gate() -> None:
    result = make_audit_result()
    manifest = dict(result.manifest)
    manifest["points"] = {
        key: dict(value) for key, value in result.manifest["points"].items()
    }
    manifest["points"][audit.POINT_DEPLOYMENT_ENDPOINT][
        "in_training_raw_support"
    ] = True
    with pytest.raises(RuntimeError, match="raw1000"):
        audit.validate_first_mcp_route_equivalence_manifest(manifest)

    manifest = dict(result.manifest)
    manifest["points"] = {
        key: dict(value) for key, value in result.manifest["points"].items()
    }
    manifest["points"][audit.POINT_TRAINING_EDGE]["route_comparison"] = dict(
        manifest["points"][audit.POINT_TRAINING_EDGE]["route_comparison"]
    )
    manifest["points"][audit.POINT_TRAINING_EDGE]["route_comparison"][
        "mcp_timestep_exact"
    ] = False
    with pytest.raises(RuntimeError, match="mcp_timestep_exact"):
        audit.validate_first_mcp_route_equivalence_manifest(manifest)


@pytest.mark.parametrize(
    ("field", "match"),
    [
        ("current_state_route_sha_exact", "current_state_route_sha_exact"),
        ("future_state_route_sha_exact", "future_state_route_sha_exact"),
    ],
)
def test_manifest_validator_rejects_state_route_sha_tamper(field: str, match: str) -> None:
    result = make_audit_result()
    manifest = copy.deepcopy(result.manifest)
    manifest["points"][audit.POINT_TRAINING_EDGE]["route_comparison"][field] = False
    with pytest.raises(RuntimeError, match=match):
        audit.validate_first_mcp_route_equivalence_manifest(manifest)


@pytest.mark.parametrize(
    ("route", "field", "match"),
    [
        ("deployment", "raw1000_current_state_equals_source", "current state"),
        ("training_route", "raw1000_future_state_equals_source", "future state"),
    ],
)
def test_manifest_validator_rejects_raw1000_source_equality_tamper(
    route: str,
    field: str,
    match: str,
) -> None:
    result = make_audit_result()
    manifest = copy.deepcopy(result.manifest)
    manifest["points"][audit.POINT_DEPLOYMENT_ENDPOINT][route][field] = False
    with pytest.raises(RuntimeError, match=match):
        audit.validate_first_mcp_route_equivalence_manifest(manifest)


@pytest.mark.parametrize(
    ("route", "field", "match"),
    [
        ("deployment", "deployment_block_mask_before_is_none", "block_mask before"),
        ("deployment", "deployment_block_mask_after_is_none", "block_mask after"),
        ("training_route", "training_block_mask_before_is_none", "block_mask before"),
        ("training_route", "training_block_mask_created", "create"),
        ("training_route", "training_block_mask_restored_is_none", "restore"),
    ],
)
def test_manifest_validator_rejects_block_mask_contract_tamper(
    route: str,
    field: str,
    match: str,
) -> None:
    result = make_audit_result()
    manifest = copy.deepcopy(result.manifest)
    manifest["points"][audit.POINT_TRAINING_EDGE][route][field] = False
    with pytest.raises(RuntimeError, match=match):
        audit.validate_first_mcp_route_equivalence_manifest(manifest)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("anchor_count", 5, "6 depth1 anchors"),
        ("flat_anchor_future_count", 14, "15 MCP futures"),
        ("mcp_call_count", 5, "once per valid depth1 anchor"),
        (
            "selected_output",
            {"depth": 1, "anchor_index": 0, "target_chunk_index": 1},
            "selected output",
        ),
    ],
)
def test_manifest_validator_rejects_training_anchor_contract_tamper(
    field: str,
    value: Any,
    match: str,
) -> None:
    result = make_audit_result()
    manifest = copy.deepcopy(result.manifest)
    manifest["points"][audit.POINT_TRAINING_EDGE]["training_route"][field] = value
    with pytest.raises(RuntimeError, match=match):
        audit.validate_first_mcp_route_equivalence_manifest(manifest)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("current_chunk", 0, "current chunk"),
        ("future_chunk", 3, "future chunk"),
        ("current_start_frame", 0, "current start"),
        ("future_start_frame", 9, "future start"),
        ("deployment_depths_used", [1, 2], "depth1"),
    ],
)
def test_manifest_validator_rejects_route_contract_tamper(
    field: str,
    value: Any,
    match: str,
) -> None:
    result = make_audit_result()
    manifest = copy.deepcopy(result.manifest)
    manifest["route_contract"][field] = value
    with pytest.raises(RuntimeError, match=match):
        audit.validate_first_mcp_route_equivalence_manifest(manifest)


def test_tensor_archive_contains_required_tensors() -> None:
    result = make_audit_result()
    tensors = result.tensors
    assert tensors["teacher_chunk0"].shape == (1, 3, 1, 1, 1)
    assert tensors["teacher_chunk1"].shape == (1, 3, 1, 1, 1)
    assert tensors["teacher_chunk2"].shape == (1, 3, 1, 1, 1)
    for point in (audit.POINT_DEPLOYMENT_ENDPOINT, audit.POINT_TRAINING_EDGE):
        record = tensors["points"][point]
        assert record["deployment_main_flow"].shape == (1, 3, 1, 1, 1)
        assert record["training_mcp_flow"].shape == (1, 3, 1, 1, 1)
        assert record["deployment_future_embed"].ndim == 3
        assert record["training_future_embed"].ndim == 3


def test_source_guard_no_forbidden_old_oracle_or_method_imports() -> None:
    repo = Path(__file__).resolve().parents[2]
    sources = [
        repo / "utils" / "nf_sf_first_mcp_route_equivalence.py",
        repo / "scripts" / "diagnose_nf_sf_first_mcp_route_equivalence.py",
    ]
    forbidden = (
        "import inference_next_forcing",
        "from inference_next_forcing",
        "import inference_mcp",
        "from inference_mcp",
        "nf_sf_m6",
        "import refinement",
        "from refinement",
        "import verifier",
        "from verifier",
        "import routing",
        "from routing",
    )
    for path in sources:
        text = path.read_text(encoding="utf-8")
        for pattern in forbidden:
            assert pattern not in text
