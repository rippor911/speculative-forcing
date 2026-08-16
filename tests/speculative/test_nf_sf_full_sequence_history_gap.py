from __future__ import annotations

import copy
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
    def __init__(self, *, mcp_output_count: int = 1) -> None:
        super().__init__()
        self.mcp_output_count = int(mcp_output_count)
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


def make_checkpoint_record() -> ev.DeploymentCheckpointRecord:
    return ev.DeploymentCheckpointRecord(
        path="/tmp/checkpoint.pt",
        sha256=TEST_SHA,
        checkpoint_type="full_sequence_step5000",
        load_mode="TEST",
        generator_state_dict={"model.weight": torch.ones(1), "mcp.0.weight": torch.ones(1)},
        global_step=ev.FULL_SEQUENCE_GLOBAL_STEP,
        training_git_sha=TRAINING_GIT_SHA,
    )


def run_canonical_mcp(source: torch.Tensor | None = None) -> ev.DeploymentResult:
    source = make_source_noise() if source is None else source
    common, fingerprint = make_common(source)
    return ev.run_mcp1_deployment(
        runtime=make_runtime(FakeGenerator()),
        mcp_scheduler=FakeScheduler(),
        source_noise=source,
        teacher_payload=make_payload(source),
        teacher_metadata=make_metadata(),
        conditional_dict={"prompt_embeds": torch.zeros((1, 2, 3))},
        checkpoint=make_checkpoint_record(),
        git_sha=RUNTIME_GIT_SHA,
        common_inputs=common,
        common_inputs_fingerprint_sha256=fingerprint,
    )


def run_intervention(
    mode: str,
    *,
    source: torch.Tensor | None = None,
    history: torch.Tensor | None = None,
    history_source: str,
    common: dict | None = None,
    fingerprint: str | None = None,
) -> ev.DeploymentResult:
    source = make_source_noise() if source is None else source
    if common is None or fingerprint is None:
        common, fingerprint = make_common(source)
    return ev.run_mcp1_history_intervention_deployment(
        mode=mode,  # type: ignore[arg-type]
        runtime=make_runtime(FakeGenerator()),
        mcp_scheduler=FakeScheduler(),
        source_noise=source,
        teacher_payload=make_payload(source),
        teacher_metadata=make_metadata(),
        conditional_dict={"prompt_embeds": torch.zeros((1, 2, 3))},
        checkpoint=make_checkpoint_record(),
        git_sha=RUNTIME_GIT_SHA,
        common_inputs=common,
        common_inputs_fingerprint_sha256=fingerprint,
        history_recache_tensor=history,
        history_source=history_source,
    )


def make_diagnostic_results() -> tuple[dict, str, dict[str, ev.DeploymentResult]]:
    source = make_source_noise()
    teacher = source + 200.0
    common, _ = make_common(source)
    common["teacher_target_sha256"] = ev.tensor_sha256(teacher)
    fingerprint = ev.canonical_json_sha256(common)
    trained_main_reference = ev.run_main_only_deployment(
        mode=ev.DIAG_MODE_TRAINED_MAIN_REFERENCE,
        runtime=make_runtime(FakeGenerator()),
        source_noise=source,
        teacher_payload=make_payload(source),
        teacher_metadata=make_metadata(),
        conditional_dict={"prompt_embeds": torch.zeros((1, 2, 3))},
        checkpoint=make_checkpoint_record(),
        git_sha=RUNTIME_GIT_SHA,
        common_inputs=common,
        common_inputs_fingerprint_sha256=fingerprint,
    )
    reference = trained_main_reference.latent
    return common, fingerprint, {
        ev.DIAG_MODE_TRAINED_MAIN_REFERENCE: trained_main_reference,
        ev.DIAG_MODE_MCP1_LIVE_HISTORY: run_intervention(
            ev.DIAG_MODE_MCP1_LIVE_HISTORY,
            source=source,
            history=None,
            history_source="generated_output",
            common=common,
            fingerprint=fingerprint,
        ),
        ev.DIAG_MODE_MCP1_MAIN_HISTORY_REPAIR: run_intervention(
            ev.DIAG_MODE_MCP1_MAIN_HISTORY_REPAIR,
            source=source,
            history=reference,
            history_source="trained_main_reference",
            common=common,
            fingerprint=fingerprint,
        ),
        ev.DIAG_MODE_MCP1_TEACHER_HISTORY_ORACLE: run_intervention(
            ev.DIAG_MODE_MCP1_TEACHER_HISTORY_ORACLE,
            source=source,
            history=teacher,
            history_source="teacher_target",
            common=common,
            fingerprint=fingerprint,
        ),
    }


def mode_summaries_with_video(
    results: dict[str, ev.DeploymentResult],
) -> dict[str, dict]:
    summaries: dict[str, dict] = {}
    for mode, result in results.items():
        summary = dict(result.summary)
        summary["video"] = {"sha256": TEST_SHA, "size_bytes": 1}
        summaries[mode] = summary
    return summaries


def mode_traces(results: dict[str, ev.DeploymentResult]) -> dict[str, dict]:
    return {mode: copy.deepcopy(result.trace) for mode, result in results.items()}


def build_manifest_from_results(
    *,
    common: dict,
    fingerprint: str,
    results: dict[str, ev.DeploymentResult],
    summaries: dict[str, dict] | None = None,
    traces: dict[str, dict] | None = None,
) -> dict:
    return ev.build_history_diagnostic_manifest(
        common_inputs=common,
        common_inputs_fingerprint_sha256=fingerprint,
        mode_summaries=mode_summaries_with_video(results) if summaries is None else summaries,
        mode_traces=mode_traces(results) if traces is None else traces,
        comparisons={},
        output_dir=Path("."),
        git_sha=RUNTIME_GIT_SHA,
    )


def chunk_record(result: ev.DeploymentResult, chunk_index: int) -> dict:
    for chunk in result.trace["chunks"]:
        if int(chunk["chunk_index"]) == int(chunk_index):
            return chunk
    raise AssertionError(f"missing chunk {chunk_index}")


def chunk_sha(tensor: torch.Tensor, chunk_index: int) -> str:
    start = int(chunk_index) * ev.FULL_SEQUENCE_CHUNK_FRAMES
    return ev.tensor_sha256(
        tensor[:, start:start + ev.FULL_SEQUENCE_CHUNK_FRAMES].detach().cpu()
    )


def test_canonical_mcp1_deployment_behavior_unchanged() -> None:
    source = make_source_noise()
    canonical = run_canonical_mcp(source)
    live = run_intervention(
        ev.DIAG_MODE_MCP1_LIVE_HISTORY,
        source=source,
        history=None,
        history_source="generated_output",
    )
    assert canonical.trace["mode"] == ev.MODE_TRAINED_MCP1
    assert "diagnostic_only" not in canonical.trace
    assert canonical.summary["mcp_call_count"] == 12
    assert canonical.summary["per_depth_call_counts"] == {"1": 12, "2": 0, "3": 0}
    assert torch.equal(canonical.latent, live.latent)


def test_live_intervention_recaches_generated_output() -> None:
    result = run_intervention(
        ev.DIAG_MODE_MCP1_LIVE_HISTORY,
        history=None,
        history_source="generated_output",
    )
    assert result.summary["history_recache_full_tensor_sha256"] is None
    assert result.summary["history_recache_chunk_sha256_by_chunk"] is None
    for chunk in result.trace["chunks"]:
        recache = chunk["clean_recache"]
        assert recache["history_recache_source"] == "generated_output"
        assert recache["history_recache_matches_generated_output"] is True


def test_live_intervention_rejects_external_history_tensor() -> None:
    source = make_source_noise()
    with pytest.raises(RuntimeError, match="must not receive history tensor"):
        run_intervention(
            ev.DIAG_MODE_MCP1_LIVE_HISTORY,
            source=source,
            history=source + 1.0,
            history_source="generated_output",
        )


def test_main_history_repair_keeps_generated_output_but_recaches_reference() -> None:
    source = make_source_noise()
    common, fingerprint = make_common(source)
    reference = ev.run_main_only_deployment(
        mode=ev.DIAG_MODE_TRAINED_MAIN_REFERENCE,
        runtime=make_runtime(FakeGenerator()),
        source_noise=source,
        teacher_payload=make_payload(source),
        teacher_metadata=make_metadata(),
        conditional_dict={"prompt_embeds": torch.zeros((1, 2, 3))},
        checkpoint=make_checkpoint_record(),
        git_sha=RUNTIME_GIT_SHA,
        common_inputs=common,
        common_inputs_fingerprint_sha256=fingerprint,
    ).latent
    result = run_intervention(
        ev.DIAG_MODE_MCP1_MAIN_HISTORY_REPAIR,
        source=source,
        history=reference,
        history_source="trained_main_reference",
        common=common,
        fingerprint=fingerprint,
    )
    recache = chunk_record(result, 2)["clean_recache"]
    assert chunk_record(result, 2)["output_produced_by"] == "MCP1"
    assert result.summary["trained_main_reference_latent_sha256"] == ev.tensor_sha256(
        reference
    )
    assert recache["history_recache_source"] == "trained_main_reference"
    assert recache["history_recache_tensor_sha256"] == chunk_sha(reference, 2)
    assert recache["generated_output_tensor_sha256"] != recache["history_recache_tensor_sha256"]
    assert recache["history_recache_matches_generated_output"] is False


def test_teacher_history_oracle_keeps_generated_output_but_recaches_teacher() -> None:
    source = make_source_noise()
    teacher = source + 200.0
    result = run_intervention(
        ev.DIAG_MODE_MCP1_TEACHER_HISTORY_ORACLE,
        source=source,
        history=teacher,
        history_source="teacher_target",
    )
    recache = chunk_record(result, 2)["clean_recache"]
    assert chunk_record(result, 2)["output_produced_by"] == "MCP1"
    assert result.summary["teacher_target_sha256"] == ev.tensor_sha256(teacher)
    assert recache["history_recache_source"] == "teacher_target"
    assert recache["history_recache_tensor_sha256"] == chunk_sha(teacher, 2)
    assert recache["generated_output_tensor_sha256"] != recache["history_recache_tensor_sha256"]


def test_exact_recache_order_and_kv_advancement() -> None:
    source = make_source_noise()
    common, fingerprint = make_common(source)
    reference = ev.run_main_only_deployment(
        mode=ev.DIAG_MODE_TRAINED_MAIN_REFERENCE,
        runtime=make_runtime(FakeGenerator()),
        source_noise=source,
        teacher_payload=make_payload(source),
        teacher_metadata=make_metadata(),
        conditional_dict={"prompt_embeds": torch.zeros((1, 2, 3))},
        checkpoint=make_checkpoint_record(),
        git_sha=RUNTIME_GIT_SHA,
        common_inputs=common,
        common_inputs_fingerprint_sha256=fingerprint,
    ).latent
    result = run_intervention(
        ev.DIAG_MODE_MCP1_MAIN_HISTORY_REPAIR,
        source=source,
        history=reference,
        history_source="trained_main_reference",
        common=common,
        fingerprint=fingerprint,
    )
    assert [item["clean_recache_order"] for item in result.trace["parallel_rounds"]] == [
        [1, 2],
        [3, 4],
        [5, 6],
    ]
    recache = chunk_record(result, 2)["clean_recache"]
    assert recache["before"]["local_end_index"] == 6 * FRAME_SEQ_LENGTH
    assert recache["after"]["local_end_index"] == 9 * FRAME_SEQ_LENGTH


def test_same_rng_and_common_fingerprints_across_diagnostic_modes() -> None:
    source = make_source_noise()
    teacher = source + 200.0
    common, _ = make_common(source)
    common["teacher_target_sha256"] = ev.tensor_sha256(teacher)
    fingerprint = ev.canonical_json_sha256(common)
    reference = ev.run_main_only_deployment(
        mode=ev.DIAG_MODE_TRAINED_MAIN_REFERENCE,
        runtime=make_runtime(FakeGenerator()),
        source_noise=source,
        teacher_payload=make_payload(source),
        teacher_metadata=make_metadata(),
        conditional_dict={"prompt_embeds": torch.zeros((1, 2, 3))},
        checkpoint=make_checkpoint_record(),
        git_sha=RUNTIME_GIT_SHA,
        common_inputs=common,
        common_inputs_fingerprint_sha256=fingerprint,
    ).latent
    results = [
        ev.run_main_only_deployment(
            mode=ev.DIAG_MODE_TRAINED_MAIN_REFERENCE,
            runtime=make_runtime(FakeGenerator()),
            source_noise=source,
            teacher_payload=make_payload(source),
            teacher_metadata=make_metadata(),
            conditional_dict={"prompt_embeds": torch.zeros((1, 2, 3))},
            checkpoint=make_checkpoint_record(),
            git_sha=RUNTIME_GIT_SHA,
            common_inputs=common,
            common_inputs_fingerprint_sha256=fingerprint,
        ),
        run_intervention(
            ev.DIAG_MODE_MCP1_LIVE_HISTORY,
            source=source,
            history=None,
            history_source="generated_output",
            common=common,
            fingerprint=fingerprint,
        ),
        run_intervention(
            ev.DIAG_MODE_MCP1_MAIN_HISTORY_REPAIR,
            source=source,
            history=reference,
            history_source="trained_main_reference",
            common=common,
            fingerprint=fingerprint,
        ),
        run_intervention(
            ev.DIAG_MODE_MCP1_TEACHER_HISTORY_ORACLE,
            source=source,
            history=teacher,
            history_source="teacher_target",
            common=common,
            fingerprint=fingerprint,
        ),
    ]
    rng = {result.summary["rng_plan_fingerprint_sha256"] for result in results}
    common = {result.summary["common_inputs_fingerprint_sha256"] for result in results}
    source_shas = {result.summary["source_noise_sha256"] for result in results}
    conditioning = {result.summary["conditioning_sha256"] for result in results}
    assert len(rng) == 1
    assert len(common) == 1
    assert len(source_shas) == 1
    assert len(conditioning) == 1


def test_depth1_only_three_pairs_and_four_steps() -> None:
    result = run_intervention(
        ev.DIAG_MODE_MCP1_LIVE_HISTORY,
        history=None,
        history_source="generated_output",
    )
    assert result.summary["per_depth_call_counts"] == {"1": 12, "2": 0, "3": 0}
    assert len(result.trace["parallel_rounds"]) == 3
    for round_record in result.trace["parallel_rounds"]:
        assert len(round_record["joint_solver_steps"]) == 4
    for chunk in result.trace["chunks"]:
        assert len(chunk["solver_steps"]) == 4


def test_history_tensor_shape_required() -> None:
    source = make_source_noise()
    bad = source[:, :-3]
    with pytest.raises(RuntimeError, match="match source_noise shape"):
        run_intervention(
            ev.DIAG_MODE_MCP1_TEACHER_HISTORY_ORACLE,
            source=source,
            history=bad,
            history_source="teacher_target",
        )
    with pytest.raises(RuntimeError, match="match source_noise shape"):
        run_intervention(
            ev.DIAG_MODE_MCP1_MAIN_HISTORY_REPAIR,
            source=source,
            history=bad,
            history_source="trained_main_reference",
        )


def test_diagnostic_manifest_hypotheses_start_null() -> None:
    common, fingerprint, results = make_diagnostic_results()
    manifest = build_manifest_from_results(
        common=common,
        fingerprint=fingerprint,
        results=results,
    )
    contract = manifest["interpretation_contract"]
    assert manifest["schema"] == ev.HISTORY_DIAGNOSTIC_SCHEMA
    assert manifest["status"] == "PASS"
    assert manifest["engineering_acceptance"]["history_source_provenance_exact"] is True
    assert manifest["engineering_acceptance"][
        "diagnostic_interventions_declared_non_deployable"
    ] is True
    assert contract["history_contamination_supported"] is None
    assert contract["teacher_history_rescues_mcp"] is None
    assert contract["mcp_intrinsic_generation_failure_supported"] is None


def test_manifest_rejects_tampered_role_map() -> None:
    common, fingerprint, results = make_diagnostic_results()
    traces = mode_traces(results)
    traces[ev.DIAG_MODE_MCP1_LIVE_HISTORY]["role_map"] = {"bad": [0]}
    with pytest.raises(RuntimeError, match="role map"):
        build_manifest_from_results(
            common=common,
            fingerprint=fingerprint,
            results=results,
            traces=traces,
        )


def test_manifest_rejects_tampered_pair_count() -> None:
    common, fingerprint, results = make_diagnostic_results()
    traces = mode_traces(results)
    traces[ev.DIAG_MODE_MCP1_LIVE_HISTORY]["parallel_rounds"] = traces[
        ev.DIAG_MODE_MCP1_LIVE_HISTORY
    ]["parallel_rounds"][:2]
    with pytest.raises(RuntimeError, match="three paired rounds"):
        build_manifest_from_results(
            common=common,
            fingerprint=fingerprint,
            results=results,
            traces=traces,
        )


def test_manifest_rejects_tampered_solver_count() -> None:
    common, fingerprint, results = make_diagnostic_results()
    traces = mode_traces(results)
    traces[ev.DIAG_MODE_MCP1_LIVE_HISTORY]["parallel_rounds"][0][
        "joint_solver_steps"
    ] = traces[ev.DIAG_MODE_MCP1_LIVE_HISTORY]["parallel_rounds"][0][
        "joint_solver_steps"
    ][:3]
    with pytest.raises(RuntimeError, match="four raw solver steps"):
        build_manifest_from_results(
            common=common,
            fingerprint=fingerprint,
            results=results,
            traces=traces,
        )


def test_manifest_rejects_tampered_recache_order() -> None:
    common, fingerprint, results = make_diagnostic_results()
    traces = mode_traces(results)
    traces[ev.DIAG_MODE_MCP1_LIVE_HISTORY]["parallel_rounds"][0][
        "clean_recache_order"
    ] = [2, 1]
    with pytest.raises(RuntimeError, match="recache order"):
        build_manifest_from_results(
            common=common,
            fingerprint=fingerprint,
            results=results,
            traces=traces,
        )


def test_manifest_rejects_tampered_history_source_sha() -> None:
    common, fingerprint, results = make_diagnostic_results()
    traces = mode_traces(results)
    traces[ev.DIAG_MODE_MCP1_MAIN_HISTORY_REPAIR][
        "history_recache_chunk_sha256_by_chunk"
    ]["2"] = "0" * 64
    with pytest.raises(RuntimeError, match="recache chunk SHA"):
        build_manifest_from_results(
            common=common,
            fingerprint=fingerprint,
            results=results,
            traces=traces,
        )


def test_manifest_rejects_summary_trace_history_sha_mismatch() -> None:
    common, fingerprint, results = make_diagnostic_results()
    summaries = mode_summaries_with_video(results)
    summaries[ev.DIAG_MODE_MCP1_TEACHER_HISTORY_ORACLE][
        "history_recache_full_tensor_sha256"
    ] = "0" * 64
    with pytest.raises(RuntimeError, match="summary/trace provenance"):
        build_manifest_from_results(
            common=common,
            fingerprint=fingerprint,
            results=results,
            summaries=summaries,
        )


def test_manifest_rejects_repair_source_not_actual_trained_main_reference() -> None:
    common, fingerprint, results = make_diagnostic_results()
    summaries = mode_summaries_with_video(results)
    traces = mode_traces(results)
    bad_sha = "1" * 64
    for record in (
        summaries[ev.DIAG_MODE_MCP1_MAIN_HISTORY_REPAIR],
        traces[ev.DIAG_MODE_MCP1_MAIN_HISTORY_REPAIR],
    ):
        record["history_recache_full_tensor_sha256"] = bad_sha
        record["trained_main_reference_latent_sha256"] = bad_sha
    with pytest.raises(RuntimeError, match="repair history source"):
        build_manifest_from_results(
            common=common,
            fingerprint=fingerprint,
            results=results,
            summaries=summaries,
            traces=traces,
        )


def test_manifest_rejects_oracle_source_not_common_teacher_target() -> None:
    common, fingerprint, results = make_diagnostic_results()
    summaries = mode_summaries_with_video(results)
    traces = mode_traces(results)
    bad_sha = "2" * 64
    for record in (
        summaries[ev.DIAG_MODE_MCP1_TEACHER_HISTORY_ORACLE],
        traces[ev.DIAG_MODE_MCP1_TEACHER_HISTORY_ORACLE],
    ):
        record["history_recache_full_tensor_sha256"] = bad_sha
        record["teacher_target_sha256"] = bad_sha
    with pytest.raises(RuntimeError, match="oracle history source"):
        build_manifest_from_results(
            common=common,
            fingerprint=fingerprint,
            results=results,
            summaries=summaries,
            traces=traces,
        )


def test_manifest_rejects_missing_common_teacher_target_sha() -> None:
    common, fingerprint, results = make_diagnostic_results()
    common_missing = dict(common)
    common_missing.pop("teacher_target_sha256")
    with pytest.raises(RuntimeError, match="teacher_target_sha256 missing"):
        build_manifest_from_results(
            common=common_missing,
            fingerprint=fingerprint,
            results=results,
        )


def test_manifest_rejects_invalid_common_teacher_target_sha() -> None:
    common, fingerprint, results = make_diagnostic_results()
    common_invalid = dict(common)
    common_invalid["teacher_target_sha256"] = "not-a-sha"
    with pytest.raises(RuntimeError, match="teacher_target_sha256 invalid"):
        build_manifest_from_results(
            common=common_invalid,
            fingerprint=fingerprint,
            results=results,
        )


def test_manifest_rejects_tampered_non_deployable_binding() -> None:
    common, fingerprint, results = make_diagnostic_results()
    summaries = mode_summaries_with_video(results)
    traces = mode_traces(results)
    summaries[ev.DIAG_MODE_MCP1_MAIN_HISTORY_REPAIR]["non_deployable"] = False
    traces[ev.DIAG_MODE_MCP1_MAIN_HISTORY_REPAIR]["non_deployable"] = False
    with pytest.raises(RuntimeError, match="non_deployable"):
        build_manifest_from_results(
            common=common,
            fingerprint=fingerprint,
            results=results,
            summaries=summaries,
            traces=traces,
        )


def test_pixel_mapping_remains_unavailable_in_diagnostic_comparison() -> None:
    left = make_source_noise()
    right = left + 1.0
    report = ev.build_history_diagnostic_comparison(
        name="diagnostic",
        left_mode=ev.DIAG_MODE_MCP1_LIVE_HISTORY,
        right_mode=ev.DIAG_MODE_TRAINED_MAIN_REFERENCE,
        latent_left=left,
        latent_right=right,
        pixel_left=torch.zeros((81, 2, 2, 3), dtype=torch.uint8),
        pixel_right=torch.ones((81, 2, 2, 3), dtype=torch.uint8),
    )
    assert report["pixel"]["pixel_chunk_mapping_status"] == "UNAVAILABLE"
    assert report["pixel"]["per_latent_chunk_pixel_mse"] is None
    assert report["diagnostic_metrics"]["later_main_recovery_chunks"] == [3, 5]
    assert report["diagnostic_metrics"]["mcp_direct_quality_chunks"] == [2, 4, 6]


def test_source_guard_no_forbidden_old_oracle_imports() -> None:
    repo = Path(__file__).resolve().parents[2]
    sources = [
        repo / "utils" / "nf_sf_full_sequence_eval.py",
        repo / "scripts" / "diagnose_nf_sf_full_sequence_history_gap.py",
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
