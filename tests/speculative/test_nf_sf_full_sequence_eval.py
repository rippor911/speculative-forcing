from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch import nn

import utils.nf_sf_full_sequence_eval as ev


TEST_SHA = "a" * 64
SAMPLE_PLAN_SHA = "b" * 64
MANIFEST_SHA = "c" * 64
RUNTIME_GIT_SHA = "d" * 40
TRAINING_GIT_SHA = ev.TRAINING_CHECKPOINT_GIT_SHA
FRAME_SEQ_LENGTH = 2


class FakeScheduler:
    def add_noise(
        self,
        clean: torch.Tensor,
        noise: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        sigma = timestep.detach().float().reshape(-1, 1, 1, 1) / 1000.0
        return ((1.0 - sigma) * clean.float() + sigma * noise.float()).to(clean.dtype)

    def step(
        self,
        model_output: torch.Tensor,
        timestep: torch.Tensor,
        sample: torch.Tensor,
        *,
        to_final: bool = False,
    ) -> torch.Tensor:
        if not to_final:
            raise RuntimeError("FakeScheduler.step supports only to_final=True")
        sigma = timestep.detach().float().reshape(-1, 1, 1, 1) / 1000.0
        return (sample.float() - sigma * model_output.float()).to(sample.dtype)


class FakeGenerator(nn.Module):
    def __init__(self, *, mcp_output_count: int = 1, consume_rng: bool = False) -> None:
        super().__init__()
        self.mcp_output_count = int(mcp_output_count)
        self.consume_rng = bool(consume_rng)
        self.calls: list[dict] = []
        self.mcp_call_count = 0

    def forward(self, **kwargs):
        current = kwargs["noisy_image_or_video"]
        timestep = kwargs["timestep"]
        kv_cache = kwargs["kv_cache"]
        current_start = int(kwargs["current_start"])
        mcp_requested = kwargs.get("mcp_future_noises") is not None
        if mcp_requested:
            self.mcp_call_count += 1
        if self.consume_rng:
            _ = torch.randn((), device=current.device)

        pre_local = int(kv_cache[0]["local_end_index"].item())
        token_count = int(current.shape[1]) * FRAME_SEQ_LENGTH
        token_end = current_start + token_count
        self.calls.append(
            {
                "mcp_requested": mcp_requested,
                "mcp_future_count": (
                    None
                    if kwargs.get("mcp_future_noises") is None
                    else len(kwargs["mcp_future_noises"])
                ),
                "mcp_future_start_frames": kwargs.get("mcp_future_start_frames"),
                "mcp_timesteps": kwargs.get("mcp_timesteps"),
                "pre_local": pre_local,
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

        flow = torch.zeros_like(current)
        clean = current + 1.0
        if mcp_requested:
            future = kwargs["mcp_future_noises"][0]
            return flow, clean, [
                torch.zeros_like(future)
                for _ in range(self.mcp_output_count)
            ]
        return flow, clean


def make_runtime(generator: FakeGenerator | None = None) -> ev.DeploymentRuntime:
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
        generator=generator or FakeGenerator(),
        scheduler=FakeScheduler(),
        kv_cache=kv_cache,
        crossattn_cache=[{"is_init": False}],
        frame_seq_length=FRAME_SEQ_LENGTH,
        num_frame_per_block=ev.FULL_SEQUENCE_CHUNK_FRAMES,
        context_noise=0,
    )


def make_source_noise() -> torch.Tensor:
    return torch.arange(
        ev.FULL_SEQUENCE_FRAME_COUNT,
        dtype=torch.float32,
    ).reshape(1, ev.FULL_SEQUENCE_FRAME_COUNT, 1, 1, 1)


def make_payload(source_noise: torch.Tensor | None = None) -> dict:
    source_noise = make_source_noise() if source_noise is None else source_noise
    schedule = ev.resolve_deployment_schedule()
    return {
        "source_noise": source_noise,
        "target_latent": torch.zeros_like(source_noise),
        "prompt": "prompt",
        "prompt_sha256": TEST_SHA,
        "noise_seed": 123,
        "rollout_seed": 456,
        "raw_denoising_steps": list(schedule.raw_schedule),
        "warped_denoising_steps": list(schedule.main_warped_schedule),
    }


def make_metadata() -> dict:
    return {
        "identity": "validation-0",
        "sample_index": 0,
        "sample_id": "sample-0",
        "split": "validation",
        "split_index": 0,
        "latent_file_sha256": TEST_SHA,
        "prompt_sha256": TEST_SHA,
    }


def make_sample_plan() -> dict:
    return {
        "sample_plan_sha256": SAMPLE_PLAN_SHA,
        "fixed_decode_validation_identity": "validation-0",
        "samples": {
            "train": [{"identity": "train-0"}],
            "validation": [{"identity": "validation-0"}],
        },
    }


def make_common(source_noise: torch.Tensor | None = None) -> tuple[dict, str]:
    source_noise = make_source_noise() if source_noise is None else source_noise
    return ev.build_common_inputs_record(
        sample_identity="validation-0",
        teacher_metadata=make_metadata(),
        teacher_payload=make_payload(source_noise),
        source_noise=source_noise,
        conditioning={"prompt_embeds": torch.zeros((1, 2, 3), dtype=torch.float32)},
        runtime_git_sha=RUNTIME_GIT_SHA,
        training_checkpoint_git_sha=TRAINING_GIT_SHA,
        fps=16,
        sample_plan_sha256=SAMPLE_PLAN_SHA,
        teacher_manifest_sha256=MANIFEST_SHA,
        selected_validation_position=0,
    )


def make_checkpoint_record(kind: str) -> ev.DeploymentCheckpointRecord:
    state = {"model.weight": torch.ones(1)}
    if kind != "official":
        state["mcp.0.weight"] = torch.ones(1)
    return ev.DeploymentCheckpointRecord(
        path="/tmp/checkpoint.pt",
        sha256=TEST_SHA,
        checkpoint_type=kind,
        load_mode="TEST",
        generator_state_dict=state,
        global_step=None if kind == "official" else ev.FULL_SEQUENCE_GLOBAL_STEP,
    )


def full_payload(**overrides) -> dict:
    payload = {
        "schema": ev.FULL_SEQUENCE_TRAINER_SCHEMA,
        "run_kind": ev.FULL_SEQUENCE_RUN_KIND,
        "objective_version": ev.FULL_SEQUENCE_OBJECTIVE_VERSION,
        "objective_mode": ev.FULL_SEQUENCE_OBJECTIVE_MODE,
        "status": "PRODUCTION",
        "global_step": ev.FULL_SEQUENCE_GLOBAL_STEP,
        "git_sha": TRAINING_GIT_SHA,
        "generator": {"model.weight": torch.ones(1), "mcp.0.weight": torch.ones(1)},
        "reference_checkpoint": {
            "sha256": ev.OFFICIAL_SELF_FORCING_CHECKPOINT_SHA256,
        },
        "sample_plan_sha256": SAMPLE_PLAN_SHA,
        "manifest_sha256": MANIFEST_SHA,
        "resolved_config": {
            "num_frame_per_block": ev.FULL_SEQUENCE_CHUNK_FRAMES,
            "gradient_checkpointing": True,
        },
        "provenance": {"paper_exact_reproduction": False},
    }
    payload.update(overrides)
    return payload


def write_full_checkpoint(tmp_path: Path, payload: dict | None = None) -> Path:
    path = tmp_path / "checkpoint_step005000.pt"
    torch.save(full_payload() if payload is None else payload, path)
    sha = ev.file_sha256(path)
    sidecars = ev.checkpoint_sidecar_paths(path)
    sidecars["sha256"].write_text(f"{sha}  {path.name}\n", encoding="utf-8")
    sidecars["validation"].write_text(
        json.dumps(
            {
                "status": "PASS",
                "path": str(path.resolve()),
                "sha256": sha,
                "size_bytes": path.stat().st_size,
                "schema": ev.CHECKPOINT_VALIDATION_SCHEMA,
                "run_kind": ev.FULL_SEQUENCE_RUN_KIND,
                "objective_version": ev.FULL_SEQUENCE_OBJECTIVE_VERSION,
                "objective_mode": ev.FULL_SEQUENCE_OBJECTIVE_MODE,
                "global_step": ev.FULL_SEQUENCE_GLOBAL_STEP,
                "generator_key_count": 2,
                "optimizer_state_entry_count": 1,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_exact_deployment_schedules() -> None:
    schedule = ev.resolve_deployment_schedule()
    assert schedule.raw_schedule == ev.RAW_DEPLOYMENT_SCHEDULE
    assert schedule.main_warped_schedule == ev.MAIN_DEPLOYMENT_SCHEDULE
    assert schedule.mcp_warped_schedule == ev.MCP_DEPLOYMENT_SCHEDULE


def test_role_map_and_execution_plan_are_locked() -> None:
    assert ev.full_sequence_role_map() == {
        "bootstrap": [0],
        "main_current": [1, 3, 5],
        "mcp_next": [2, 4, 6],
    }
    plan = ev.build_mcp1_execution_plan()
    assert [item["phase"] for item in plan] == [
        "bootstrap",
        "paired_round",
        "paired_round",
        "paired_round",
    ]
    assert [item["chunk_indices"] for item in plan] == [[0], [1, 2], [3, 4], [5, 6]]
    assert [item["cursor_after"] for item in plan] == [1, 3, 5, 7]


@pytest.mark.parametrize(
    ("updates", "match"),
    [
        ({"status": "NON_PRODUCTION_SMOKE"}, "status"),
        ({"global_step": 500}, "global_step"),
        ({"global_step": 2000}, "global_step"),
        ({"schema": "wrong"}, "schema"),
        ({"objective_mode": "main_only_full_control"}, "objective_mode"),
        ({"git_sha": "0" * 40}, "git_sha"),
        ({"generator": {"model.weight": torch.ones(1)}}, "MCP"),
    ],
)
def test_full_checkpoint_payload_rejects_invalid_contract(updates, match) -> None:
    with pytest.raises((RuntimeError, TypeError), match=match):
        ev.validate_full_sequence_checkpoint_payload(
            full_payload(**updates),
            checkpoint_sha256=TEST_SHA,
            expected_training_git_sha=TRAINING_GIT_SHA,
            expected_official_sha256=ev.OFFICIAL_SELF_FORCING_CHECKPOINT_SHA256,
        )


def test_full_checkpoint_sidecar_rejects_bad_sidecar(tmp_path: Path) -> None:
    path = write_full_checkpoint(tmp_path)
    sidecars = ev.checkpoint_sidecar_paths(path)
    sidecars["sha256"].write_text("0" * 64 + "  checkpoint.pt\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="sidecar mismatch"):
        ev.load_full_sequence_checkpoint_record(
            path,
            expected_training_git_sha=TRAINING_GIT_SHA,
        )


def test_full_checkpoint_record_accepts_valid_payload(tmp_path: Path) -> None:
    path = write_full_checkpoint(tmp_path)
    record = ev.load_full_sequence_checkpoint_record(
        path,
        expected_training_git_sha=TRAINING_GIT_SHA,
    )
    assert record.global_step == ev.FULL_SEQUENCE_GLOBAL_STEP
    assert record.checkpoint_type == "full_sequence_step5000"
    assert ev.count_mcp_tensors(record.generator_state_dict) == 1
    assert record.training_git_sha == TRAINING_GIT_SHA


def test_runtime_git_and_training_checkpoint_git_can_differ() -> None:
    ev.validate_full_sequence_checkpoint_payload(
        full_payload(),
        checkpoint_sha256=TEST_SHA,
        expected_training_git_sha=TRAINING_GIT_SHA,
        expected_official_sha256=ev.OFFICIAL_SELF_FORCING_CHECKPOINT_SHA256,
    )
    common, _ = make_common()
    assert common["runtime_git_sha"] == RUNTIME_GIT_SHA
    assert common["training_checkpoint_git_sha"] == TRAINING_GIT_SHA
    assert common["runtime_git_sha"] != common["training_checkpoint_git_sha"]


def test_training_checkpoint_git_mismatch_rejects() -> None:
    with pytest.raises(RuntimeError, match="training git_sha"):
        ev.validate_full_sequence_checkpoint_payload(
            full_payload(),
            checkpoint_sha256=TEST_SHA,
            expected_training_git_sha="0" * 40,
            expected_official_sha256=ev.OFFICIAL_SELF_FORCING_CHECKPOINT_SHA256,
        )


def preflight_facts(**overrides) -> dict:
    facts = {
        "repo_root": "D:/repo",
        "git_top_level": "D:/repo",
        "current_runtime_git_sha": RUNTIME_GIT_SHA,
        "expected_runtime_git_sha": RUNTIME_GIT_SHA,
        "tracked_worktree_dirty_paths": [],
        "staged_index_dirty_paths": [],
        "output_dir": "D:/eval-output",
        "output_dir_inside_repo": False,
    }
    facts.update(overrides)
    return facts


def test_repo_preflight_facts_computes_output_inside_repo(monkeypatch) -> None:
    repo_root = Path("D:/repo")
    monkeypatch.setattr(ev, "git_top_level", lambda: repo_root)
    monkeypatch.setattr(ev, "current_git_head", lambda: RUNTIME_GIT_SHA)
    monkeypatch.setattr(ev, "git_changed_paths", lambda *, cached: ())

    outside = ev.repo_preflight_facts(
        expected_runtime_git_sha=RUNTIME_GIT_SHA,
        output_dir=Path("D:/eval-output"),
        repo_root=repo_root,
    )
    inside = ev.repo_preflight_facts(
        expected_runtime_git_sha=RUNTIME_GIT_SHA,
        output_dir=repo_root / "eval-output",
        repo_root=repo_root,
    )

    assert outside["output_dir_inside_repo"] is False
    assert inside["output_dir_inside_repo"] is True


def test_repo_preflight_runtime_git_mismatch_rejects() -> None:
    with pytest.raises(RuntimeError, match="runtime git SHA"):
        ev.validate_repo_preflight_facts(
            preflight_facts(current_runtime_git_sha="0" * 40)
        )


def test_repo_preflight_tracked_dirty_rejects() -> None:
    with pytest.raises(RuntimeError, match="tracked worktree"):
        ev.validate_repo_preflight_facts(
            preflight_facts(tracked_worktree_dirty_paths=["utils/file.py"])
        )


def test_repo_preflight_staged_dirty_rejects() -> None:
    with pytest.raises(RuntimeError, match="staged index"):
        ev.validate_repo_preflight_facts(
            preflight_facts(staged_index_dirty_paths=["utils/file.py"])
        )


def test_repo_preflight_output_inside_repo_rejects() -> None:
    with pytest.raises(RuntimeError, match="output_dir"):
        ev.validate_repo_preflight_facts(
            preflight_facts(output_dir="D:/repo/eval", output_dir_inside_repo=True)
        )


def test_artifact_identity_rejects_sample_plan_sha_mismatch() -> None:
    with pytest.raises(RuntimeError, match="sample_plan SHA"):
        ev.validate_eval_artifact_identity(
            sample_plan={**make_sample_plan(), "sample_plan_sha256": "0" * 64},
            teacher_manifest_sha256=MANIFEST_SHA,
            checkpoint_payload=full_payload(),
            selected_identity="validation-0",
        )


def test_artifact_identity_rejects_teacher_manifest_sha_mismatch() -> None:
    with pytest.raises(RuntimeError, match="teacher manifest SHA"):
        ev.validate_eval_artifact_identity(
            sample_plan=make_sample_plan(),
            teacher_manifest_sha256="0" * 64,
            checkpoint_payload=full_payload(),
            selected_identity="validation-0",
        )


def test_artifact_identity_rejects_train_identity() -> None:
    with pytest.raises(RuntimeError, match="validation split"):
        ev.validate_eval_artifact_identity(
            sample_plan=make_sample_plan(),
            teacher_manifest_sha256=MANIFEST_SHA,
            checkpoint_payload=full_payload(),
            selected_identity="train-0",
        )


def test_official_checkpoint_exact_sha_and_mcp_rejection(tmp_path: Path) -> None:
    path = tmp_path / "official.pt"
    torch.save({"model.weight": torch.ones(1)}, path)
    sha = ev.file_sha256(path)
    record = ev.load_official_checkpoint_record(path, expected_sha256=sha)
    assert record.checkpoint_type == "official_self_forcing"
    with pytest.raises(RuntimeError, match="SHA256"):
        ev.load_official_checkpoint_record(path, expected_sha256="0" * 64)

    mcp_path = tmp_path / "official_mcp.pt"
    torch.save({"mcp.0.weight": torch.ones(1)}, mcp_path)
    with pytest.raises(RuntimeError, match="MCP"):
        ev.load_official_checkpoint_record(
            mcp_path,
            expected_sha256=ev.file_sha256(mcp_path),
        )


def test_common_fingerprint_is_stable_and_source_noise_shared() -> None:
    source = make_source_noise()
    first, first_fp = make_common(source)
    second, second_fp = make_common(source.clone())
    assert first == second
    assert first_fp == second_fp
    assert first["source_noise_sha256"] == ev.tensor_sha256(source)


def test_rng_plan_is_stable_and_global_rng_is_restored() -> None:
    source = make_source_noise()
    torch.manual_seed(999)
    before = torch.get_rng_state().clone()
    first = ev.build_absolute_chunk_rng_plan(source_noise=source, rollout_seed=77)
    after = torch.get_rng_state().clone()
    second = ev.build_absolute_chunk_rng_plan(source_noise=source, rollout_seed=77)
    assert torch.equal(before, after)
    assert first["trace"]["draw_count"] == ev.FULL_SEQUENCE_NUM_CHUNKS * 4
    assert first["trace"]["draws"][0]["noise"]["sha256"] == second["trace"]["draws"][0]["noise"]["sha256"]
    assert first["trace"]["rng_plan_fingerprint_sha256"] == second["trace"]["rng_plan_fingerprint_sha256"]


def test_rng_plan_fingerprint_changes_when_draw_semantics_change() -> None:
    source = make_source_noise()
    plan = ev.build_absolute_chunk_rng_plan(source_noise=source, rollout_seed=77)
    changed = dict(plan["trace"])
    changed["draws"] = [dict(item) for item in plan["trace"]["draws"]]
    changed["draws"][0] = dict(changed["draws"][0])
    changed["draws"][0]["purpose"] = "changed_purpose"
    assert ev.rng_plan_fingerprint(changed) != plan["trace"]["rng_plan_fingerprint_sha256"]


def test_rng_plan_fingerprint_mismatch_rejects() -> None:
    with pytest.raises(RuntimeError, match="RNG plan fingerprint differs"):
        ev.assert_rng_plan_fingerprints(
            {
                ev.MODE_OFFICIAL_MAIN: {"rng_plan_fingerprint_sha256": "a"},
                ev.MODE_TRAINED_MAIN: {"rng_plan_fingerprint_sha256": "b"},
            }
        )


def test_trained_main_rollout_has_zero_mcp_calls() -> None:
    source = make_source_noise()
    common, fingerprint = make_common(source)
    generator = FakeGenerator()
    result = ev.run_main_only_deployment(
        mode=ev.MODE_TRAINED_MAIN,
        runtime=make_runtime(generator),
        source_noise=source,
        teacher_payload=make_payload(source),
        teacher_metadata=make_metadata(),
        conditional_dict={"prompt_embeds": torch.zeros((1, 2, 3))},
        checkpoint=make_checkpoint_record("full_sequence_step5000"),
        git_sha=RUNTIME_GIT_SHA,
        common_inputs=common,
        common_inputs_fingerprint_sha256=fingerprint,
    )
    assert result.summary["mcp_call_count"] == 0
    assert result.summary["static_runtime_counts"]["main_solver_forward_count"] == 28
    assert len(result.trace["chunks"]) == 7
    assert result.summary["generation_elapsed_ms"] >= 0.0
    assert result.summary["runtime_measurement_status"] == ev.RUNTIME_MEASUREMENT_STATUS
    ev.validate_trained_main_trace(result.trace)


def test_source_noise_sha_mismatch_rejects() -> None:
    common_source = make_source_noise()
    actual_source = common_source + 1.0
    common, fingerprint = make_common(common_source)
    with pytest.raises(RuntimeError, match="source_noise SHA"):
        ev.run_main_only_deployment(
            mode=ev.MODE_TRAINED_MAIN,
            runtime=make_runtime(FakeGenerator()),
            source_noise=actual_source,
            teacher_payload=make_payload(actual_source),
            teacher_metadata=make_metadata(),
            conditional_dict={"prompt_embeds": torch.zeros((1, 2, 3))},
            checkpoint=make_checkpoint_record("full_sequence_step5000"),
            git_sha=RUNTIME_GIT_SHA,
            common_inputs=common,
            common_inputs_fingerprint_sha256=fingerprint,
        )


def test_conditioning_sha_mismatch_rejects() -> None:
    source = make_source_noise()
    common, fingerprint = make_common(source)
    with pytest.raises(RuntimeError, match="conditioning SHA"):
        ev.run_main_only_deployment(
            mode=ev.MODE_TRAINED_MAIN,
            runtime=make_runtime(FakeGenerator()),
            source_noise=source,
            teacher_payload=make_payload(source),
            teacher_metadata=make_metadata(),
            conditional_dict={"prompt_embeds": torch.ones((1, 2, 3))},
            checkpoint=make_checkpoint_record("full_sequence_step5000"),
            git_sha=RUNTIME_GIT_SHA,
            common_inputs=common,
            common_inputs_fingerprint_sha256=fingerprint,
        )


def test_trained_mcp1_depth1_only_three_paired_rounds_and_recache_order() -> None:
    source = make_source_noise()
    common, fingerprint = make_common(source)
    generator = FakeGenerator()
    result = ev.run_mcp1_deployment(
        runtime=make_runtime(generator),
        mcp_scheduler=FakeScheduler(),
        source_noise=source,
        teacher_payload=make_payload(source),
        teacher_metadata=make_metadata(),
        conditional_dict={"prompt_embeds": torch.zeros((1, 2, 3))},
        checkpoint=make_checkpoint_record("full_sequence_step5000"),
        git_sha=RUNTIME_GIT_SHA,
        common_inputs=common,
        common_inputs_fingerprint_sha256=fingerprint,
    )
    summary = result.summary
    assert summary["role_map"] == ev.full_sequence_role_map()
    assert summary["mcp_call_count"] == 12
    assert summary["per_depth_call_counts"] == {"1": 12, "2": 0, "3": 0}
    assert summary["static_runtime_counts"]["main_solver_forward_count"] == 16
    assert summary["static_runtime_counts"]["clean_recache_forward_count"] == 7
    assert summary["generation_elapsed_ms"] >= 0.0
    assert [item["clean_recache_order"] for item in result.trace["parallel_rounds"]] == [
        [1, 2],
        [3, 4],
        [5, 6],
    ]
    for round_record in result.trace["parallel_rounds"]:
        assert len(round_record["joint_solver_steps"]) == 4
    ev.validate_trained_mcp1_trace(result.trace)


def test_mcp1_rejects_depth2_or_depth3_outputs() -> None:
    source = make_source_noise()
    common, fingerprint = make_common(source)
    with pytest.raises(RuntimeError, match="depth1 only"):
        ev.run_mcp1_deployment(
            runtime=make_runtime(FakeGenerator(mcp_output_count=2)),
            mcp_scheduler=FakeScheduler(),
            source_noise=source,
            teacher_payload=make_payload(source),
            teacher_metadata=make_metadata(),
            conditional_dict={"prompt_embeds": torch.zeros((1, 2, 3))},
            checkpoint=make_checkpoint_record("full_sequence_step5000"),
            git_sha=RUNTIME_GIT_SHA,
            common_inputs=common,
            common_inputs_fingerprint_sha256=fingerprint,
        )


def test_common_input_fingerprint_mismatch_rejects() -> None:
    with pytest.raises(RuntimeError, match="differs"):
        ev.assert_common_input_fingerprints(
            {
                ev.MODE_OFFICIAL_MAIN: {"common_inputs_fingerprint_sha256": "a"},
                ev.MODE_TRAINED_MAIN: {"common_inputs_fingerprint_sha256": "b"},
            }
        )


def test_comparison_reports_include_role_aware_metrics() -> None:
    left = make_source_noise()
    right = left + 1.0
    report = ev.build_comparison_report(
        name="trained_main_vs_trained_mcp1",
        left_mode=ev.MODE_TRAINED_MAIN,
        right_mode=ev.MODE_TRAINED_MCP1,
        latent_left=left,
        latent_right=right,
        pixel_left=torch.zeros((81, 2, 2, 3), dtype=torch.uint8),
        pixel_right=torch.ones((81, 2, 2, 3), dtype=torch.uint8),
        role_map=ev.full_sequence_role_map(),
    )
    assert report["visual_review_status"] == "PENDING"
    assert report["visual_quality_pass"] is None
    assert set(report["role_aware_latent"]) == {"bootstrap", "main_current", "mcp_next"}
    assert len(report["latent"]["per_chunk_mse"]) == 7
    assert report["pixel"]["pixel_chunk_mapping_status"] == "UNAVAILABLE"
    assert report["pixel"]["per_latent_chunk_pixel_mse"] is None
    assert len(report["pixel"]["per_frame_mse"]) == 81


def test_output_manifest_schema_and_engineering_acceptance() -> None:
    common, fingerprint = make_common()
    rng_fp = "e" * 64
    conditioning_sha = str(common["conditioning_sha256"])
    mode_summaries = {
        ev.MODE_OFFICIAL_MAIN: {
            "status": "PASS",
            "common_inputs_fingerprint_sha256": fingerprint,
            "latent_sha256": TEST_SHA,
            "source_noise_sha256": common["source_noise_sha256"],
            "conditioning_sha256": conditioning_sha,
            "rng_plan_fingerprint_sha256": rng_fp,
            "generation_elapsed_ms": 1.0,
            "runtime_measurement_status": ev.RUNTIME_MEASUREMENT_STATUS,
            "mcp_call_count": 0,
            "video": {"sha256": TEST_SHA, "size_bytes": 1},
        },
        ev.MODE_TRAINED_MAIN: {
            "status": "PASS",
            "common_inputs_fingerprint_sha256": fingerprint,
            "latent_sha256": TEST_SHA,
            "source_noise_sha256": common["source_noise_sha256"],
            "conditioning_sha256": conditioning_sha,
            "rng_plan_fingerprint_sha256": rng_fp,
            "generation_elapsed_ms": 2.0,
            "runtime_measurement_status": ev.RUNTIME_MEASUREMENT_STATUS,
            "mcp_call_count": 0,
            "video": {"sha256": TEST_SHA, "size_bytes": 1},
        },
        ev.MODE_TRAINED_MCP1: {
            "status": "PASS",
            "common_inputs_fingerprint_sha256": fingerprint,
            "latent_sha256": TEST_SHA,
            "source_noise_sha256": common["source_noise_sha256"],
            "conditioning_sha256": conditioning_sha,
            "rng_plan_fingerprint_sha256": rng_fp,
            "generation_elapsed_ms": 3.0,
            "runtime_measurement_status": ev.RUNTIME_MEASUREMENT_STATUS,
            "mcp_call_count": 12,
            "role_map": ev.full_sequence_role_map(),
            "video": {"sha256": TEST_SHA, "size_bytes": 1},
        },
    }
    manifest = ev.build_eval_manifest(
        common_inputs=common,
        common_inputs_fingerprint_sha256=fingerprint,
        mode_summaries=mode_summaries,
        comparisons={
            "official_vs_trained_main": {"visual_review_status": "PENDING", "visual_quality_pass": None},
            "trained_main_vs_trained_mcp1": {"visual_review_status": "PENDING", "visual_quality_pass": None},
        },
        output_dir=Path("."),
        git_sha=RUNTIME_GIT_SHA,
    )
    assert manifest["schema"] == ev.EVAL_SCHEMA
    assert manifest["status"] == "PASS"
    assert manifest["runtime_git_sha"] == RUNTIME_GIT_SHA
    assert manifest["training_checkpoint_git_sha"] == TRAINING_GIT_SHA
    assert manifest["rng_plan_fingerprint_sha256"] == rng_fp
    assert manifest["engineering_acceptance"]["trained_main_zero_mcp_calls"] is True
    assert manifest["engineering_acceptance"]["rng_plan_fingerprints_exact"] is True


def test_source_guard_no_old_oracle_or_inference_imports() -> None:
    repo = Path(__file__).resolve().parents[2]
    sources = [
        repo / "utils" / "nf_sf_full_sequence_eval.py",
        repo / "scripts" / "eval_nf_sf_full_sequence_deployment.py",
    ]
    forbidden = (
        "import inference_next_forcing",
        "from inference_next_forcing",
        "import inference_mcp",
        "from inference_mcp",
        "nf_sf_m6",
    )
    for path in sources:
        text = path.read_text(encoding="utf-8")
        for pattern in forbidden:
            assert pattern not in text
