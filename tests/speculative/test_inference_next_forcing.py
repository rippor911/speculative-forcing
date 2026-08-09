from __future__ import annotations

import json
from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import inference_next_forcing as inf
import utils.nf_sf_m6 as m6

FRAME_SEQ_LENGTH = 2
TEST_GIT_SHA = "a" * 40
TEST_SHA256 = "b" * 64


class FakeScheduler:
    def add_noise(
        self,
        clean: torch.Tensor,
        noise: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        sigma = timestep.detach().float().reshape(-1, 1, 1, 1) / 1000.0
        return ((1.0 - sigma) * clean.float() + sigma * noise.float()).to(clean.dtype)


class FakeGenerator(nn.Module):
    def __init__(
        self,
        *,
        nonfinite: bool = False,
        overwrite_visible: bool = False,
        constant_clean: float | None = None,
        recache_token_offset: int = 0,
        inconsistent_boundaries: bool = False,
    ) -> None:
        super().__init__()
        self.nonfinite = bool(nonfinite)
        self.overwrite_visible = bool(overwrite_visible)
        self.constant_clean = constant_clean
        self.recache_token_offset = int(recache_token_offset)
        self.inconsistent_boundaries = bool(inconsistent_boundaries)
        self.calls: list[dict] = []
        self.mcp_call_count = 0

    def forward(self, **kwargs):
        current = kwargs["noisy_image_or_video"]
        timestep = kwargs["timestep"]
        kv_cache = kwargs["kv_cache"]
        current_start = int(kwargs["current_start"])
        if kwargs.get("mcp_future_noises") is not None:
            self.mcp_call_count += 1
        is_context = bool((timestep.detach().float() == 0).all().item())
        pre_local = int(kv_cache[0]["local_end_index"].item())
        token_count = int(current.shape[1]) * FRAME_SEQ_LENGTH
        token_end = current_start + token_count
        if is_context:
            token_end += self.recache_token_offset
        call_index = len(self.calls)
        self.calls.append(
            {
                "is_context": is_context,
                "timestep": timestep.detach().clone(),
                "current_start": current_start,
                "pre_local": pre_local,
                "token_end": token_end,
                "mcp_requested": kwargs.get("mcp_future_noises") is not None,
            }
        )

        for layer_index, layer in enumerate(kv_cache):
            if self.overwrite_visible and not is_context and pre_local > 0:
                layer["k"][:, :pre_local] = -1000.0 - call_index - layer_index
                layer["v"][:, :pre_local] = -2000.0 - call_index - layer_index
            layer["k"][:, current_start:token_end] = 10.0 + call_index + layer_index
            layer["v"][:, current_start:token_end] = 20.0 + call_index + layer_index
            layer_end = token_end + (1 if self.inconsistent_boundaries and layer_index == 1 else 0)
            layer["global_end_index"].fill_(layer_end)
            layer["local_end_index"].fill_(layer_end)

        if self.nonfinite and not is_context:
            clean = torch.full_like(current, float("inf"))
        elif self.constant_clean is not None and not is_context:
            clean = torch.full_like(current, float(self.constant_clean))
        else:
            clean = current + 1.0
        flow = torch.zeros_like(current)
        return flow, clean


def make_config(schedule=None):
    return SimpleNamespace(
        denoising_step_list=[1000, 750, 500, 250] if schedule is None else schedule,
        model_kwargs={"timestep_shift": 5.0},
        num_frame_per_block=3,
        context_noise=0,
    )


def make_cache(*, frames: int = 6, layers: int = 2):
    capacity = frames * FRAME_SEQ_LENGTH
    return [
        {
            "k": torch.zeros((1, capacity, 1, 1), dtype=torch.float32),
            "v": torch.zeros((1, capacity, 1, 1), dtype=torch.float32),
            "global_end_index": torch.tensor([0], dtype=torch.long),
            "local_end_index": torch.tensor([0], dtype=torch.long),
        }
        for _ in range(layers)
    ]


def make_runtime(generator: FakeGenerator | None = None) -> m6.M6OracleRuntime:
    return m6.M6OracleRuntime(
        generator=generator or FakeGenerator(),
        scheduler=FakeScheduler(),
        kv_cache=make_cache(),
        crossattn_cache=[{"is_init": False}, {"is_init": False}],
        frame_seq_length=FRAME_SEQ_LENGTH,
        num_frame_per_block=3,
        context_noise=0,
    )


def make_payload(source_noise: torch.Tensor | None = None):
    if source_noise is None:
        source_noise = torch.arange(6, dtype=torch.float32).reshape(1, 6, 1, 1, 1)
    schedule = m6.resolve_m6_schedule(make_config())
    return {
        "source_noise": source_noise,
        "target_latent": torch.zeros_like(source_noise),
        "noise_seed": 123,
        "rollout_seed": 456,
        "prompt": "prompt",
        "prompt_sha256": TEST_SHA256,
        "raw_denoising_steps": list(schedule.raw_schedule),
        "warped_denoising_steps": list(schedule.main_warped_schedule),
    }


def make_metadata():
    return {
        "manifest_path": "/tmp/manifest.json",
        "manifest_sha256": TEST_SHA256,
        "dataset_root": "/tmp/dataset",
        "sample_index": 0,
        "sample_id": "sample-train-0",
        "split": "train",
        "split_index": 0,
        "source_line_index": 7,
        "shard_id": 0,
        "plan_index": 0,
        "latent_path": "/tmp/payload.pt",
        "latent_file_sha256": TEST_SHA256,
        "prompt_sha256": TEST_SHA256,
    }


def make_checkpoint(kind: str = "A", *, sha256: str = TEST_SHA256) -> m6.M6CheckpointRecord:
    checkpoint_type = (
        m6.M6_CHECKPOINT_OFFICIAL if kind == "A" else m6.M6_CHECKPOINT_FORMAL_STEP0
    )
    return m6.M6CheckpointRecord(
        path="/tmp/checkpoint.pt",
        sha256=sha256,
        checkpoint_type=checkpoint_type,
        load_mode="test",
        generator_state_dict={},
        global_step=None if kind == "A" else 0,
        mcp_tensor_count=0 if kind == "A" else 1,
    )


def make_formal_step0_payload(
    *,
    metadata: dict | None = None,
    top_level: dict | None = None,
) -> dict:
    stage_contract = {
        "target_global_step": 500,
        "parent_global_step": None,
        "validation_steps": [0, 500],
        "checkpoint_steps": [0, 500],
        "is_resume_stage": False,
    }
    formal_metadata = {
        "schema": m6.M5_FORMAL_TRAINER_SCHEMA,
        "status": "PASS",
        "formal_enabled": True,
        "stage": "stage_a",
        "stage_contract": dict(stage_contract),
        "sample_plan_sha256": "1" * 64,
        "teacher_manifest_sha256": "2" * 64,
        "conditional_artifact_sha256": "3" * 64,
        "validation_implementation_schema": m6.M5_STREAMING_VALIDATION_SCHEMA,
    }
    if metadata is not None:
        formal_metadata.update(metadata)
    payload = {
        "format": m6.M3_CHECKPOINT_FORMAT,
        "git_sha": TEST_GIT_SHA,
        "global_step": 0,
        "generator": {"mcp.weight": torch.zeros(1)},
        "m5_formal_trainer": formal_metadata,
        "resolved_config": {
            "m5_formal": {
                "schema": m6.M5_FORMAL_TRAINER_SCHEMA,
                "enabled": True,
                "sample_plan_sha256": formal_metadata["sample_plan_sha256"],
                "teacher_manifest_sha256": formal_metadata["teacher_manifest_sha256"],
                "conditional_artifact_sha256": formal_metadata["conditional_artifact_sha256"],
                "validation_implementation_schema": m6.M5_STREAMING_VALIDATION_SCHEMA,
            }
        },
    }
    if top_level is not None:
        payload.update(top_level)
    return payload


def run_fake_oracle(
    *,
    oracle_kind: str = "A",
    generator: FakeGenerator | None = None,
    source_noise: torch.Tensor | None = None,
    checkpoint_sha256: str = TEST_SHA256,
    tolerance: float | None = None,
):
    payload = make_payload(source_noise=source_noise)
    runtime = make_runtime(generator)
    return m6.run_main_only_oracle(
        oracle_kind=oracle_kind,
        runtime=runtime,
        source_noise=payload["source_noise"],
        teacher_payload=payload,
        teacher_metadata=make_metadata(),
        conditional_dict={"prompt_embeds": torch.zeros((1, 4, 2), dtype=torch.float32)},
        schedule=m6.resolve_m6_schedule(make_config()),
        checkpoint=make_checkpoint(oracle_kind, sha256=checkpoint_sha256),
        git_sha=TEST_GIT_SHA,
        resolved_config_canonical_sha256=TEST_SHA256,
        device_runtime_contract={"WORLD_SIZE": "1", "device": "cpu", "runtime": "fake_cpu"},
        tolerance=tolerance,
    ), runtime


def install_test_writers(monkeypatch) -> dict[str, int]:
    calls = {"json": 0, "torch": 0}

    def write_json(payload, path):
        calls["json"] += 1
        m6.validate_json_payload(payload)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    def save_torch(payload, path):
        calls["torch"] += 1
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(payload), path)

    monkeypatch.setattr(m6, "atomic_json_write", write_json)
    monkeypatch.setattr(m6, "atomic_torch_save", save_torch)
    return calls


def test_raw_to_warped_schedule_resolver_contract() -> None:
    raw = [1000, 750, 500, 250]
    config = make_config(schedule=raw)
    schedule = m6.resolve_m6_schedule(config)

    assert raw == [1000, 750, 500, 250]
    assert schedule.raw_schedule == (1000.0, 750.0, 500.0, 250.0)
    assert schedule.main_warped_schedule == pytest.approx(
        (1000.0, 937.5, 833.3333333, 625.0)
    )
    assert schedule.mcp_enabled is False
    assert schedule.mcp_warped_schedule is None
    assert schedule.main_warped_schedule != (1000.0,)


@pytest.mark.parametrize(
    "bad_schedule",
    [
        [],
        [1000],
        [1000, 500, 750, 250],
        [1000, 750, 500, 250, 0],
        [1000, float("nan"), 500, 250],
    ],
)
def test_schedule_resolver_rejects_illegal_or_non_locked_schedule(bad_schedule) -> None:
    with pytest.raises((ValueError, RuntimeError)):
        m6.resolve_m6_schedule(make_config(schedule=bad_schedule))


def test_cli_parse_args_accepts_argv() -> None:
    args = inf.parse_args(
        [
            "--oracle",
            "A",
            "--config",
            "config.yaml",
            "--checkpoint",
            "checkpoint.pt",
            "--teacher_manifest",
            "manifest.json",
            "--dataset_root",
            "dataset",
            "--output_dir",
            "out",
        ]
    )

    assert args.oracle == "A"
    assert args.output_dir == Path("out")


def test_oracle_b_cli_contract_rejects_before_output_dir_creation(tmp_path: Path) -> None:
    args = SimpleNamespace(
        oracle="B",
        oracle_a_dir=None,
        tolerance=0.0,
        output_dir=tmp_path / "should_not_exist",
    )

    with pytest.raises(ValueError, match="oracle_a_dir"):
        inf.validate_oracle_b_cli_contract(args)
    assert not args.output_dir.exists()


def test_oracle_b_cli_contract_requires_tolerance_and_nonempty_a_artifacts(
    tmp_path: Path,
) -> None:
    oracle_a_dir = tmp_path / "a"
    oracle_a_dir.mkdir()
    args = SimpleNamespace(
        oracle="B",
        oracle_a_dir=oracle_a_dir,
        tolerance=None,
        output_dir=tmp_path / "out",
    )
    with pytest.raises(ValueError, match="tolerance"):
        inf.validate_oracle_b_cli_contract(args)

    args.tolerance = -1.0
    with pytest.raises(ValueError, match="non-negative"):
        inf.validate_oracle_b_cli_contract(args)

    args.tolerance = 0.0
    (oracle_a_dir / "oracle_trace.json").write_text("{}", encoding="utf-8")
    (oracle_a_dir / "oracle_summary.json").write_text("{}", encoding="utf-8")
    (oracle_a_dir / "output_latent.pt").write_bytes(b"")
    with pytest.raises(FileNotFoundError, match="non-empty"):
        inf.validate_oracle_b_cli_contract(args)
    assert not args.output_dir.exists()


def test_runtime_device_requires_world_size_cuda0_and_single_gpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WORLD_SIZE", "2")
    with pytest.raises(RuntimeError, match="WORLD_SIZE"):
        inf.runtime_device("cuda:0")

    monkeypatch.setenv("WORLD_SIZE", "1")
    with pytest.raises(RuntimeError, match="cuda:0"):
        inf.runtime_device("cuda")

    monkeypatch.setattr(inf.torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA"):
        inf.runtime_device("cuda:0")

    monkeypatch.setattr(inf.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(inf.torch.cuda, "device_count", lambda: 2)
    with pytest.raises(RuntimeError, match="device_count"):
        inf.runtime_device("cuda:0")

    calls = []
    monkeypatch.setattr(inf.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(inf.torch.cuda, "set_device", lambda index: calls.append(index))
    device, contract = inf.runtime_device("cuda:0")

    assert str(device) == "cuda:0"
    assert calls == [0]
    assert contract["WORLD_SIZE"] == "1"


def test_schedule_config_and_trace_are_consistent() -> None:
    result, _ = run_fake_oracle()

    assert result.trace["schedule"]["raw_schedule"] == [1000, 750, 500, 250]
    assert result.trace["schedule"]["main_warped_schedule"] == pytest.approx(
        [1000.0, 937.5, 833.3333333, 625.0]
    )
    assert result.trace["mcp_enabled"] is False
    assert result.trace["mcp_warped_schedule"] is None


def test_exact_stored_source_noise_is_used_and_different_noise_is_rejected() -> None:
    payload = make_payload()
    source = payload["source_noise"]
    result, _ = run_fake_oracle(source_noise=source)
    first_step = result.trace["chunks"][0]["solver_steps"][0]

    assert result.trace["source_noise"]["sha256"] == m6.tensor_sha256(source)
    assert first_step["input_tensor"]["sha256"] == m6.tensor_sha256(source[:, 0:3])

    with pytest.raises(RuntimeError, match="source_noise"):
        m6.run_main_only_oracle(
            oracle_kind="A",
            runtime=make_runtime(),
            source_noise=source + 1.0,
            teacher_payload=payload,
            teacher_metadata=make_metadata(),
            conditional_dict={"prompt_embeds": torch.zeros((1, 4, 2))},
            schedule=m6.resolve_m6_schedule(make_config()),
            checkpoint=make_checkpoint("A"),
            git_sha=TEST_GIT_SHA,
            resolved_config_canonical_sha256=TEST_SHA256,
            device_runtime_contract={"WORLD_SIZE": "1", "device": "cpu", "runtime": "fake_cpu"},
        )


def test_rollout_rng_draw_order_transition_hashes_and_context_consumption_reproduce() -> None:
    first, _ = run_fake_oracle()
    second, _ = run_fake_oracle()

    first_draws = first.trace["rng"]["draws"]
    second_draws = second.trace["rng"]["draws"]
    assert len(first_draws) == 9
    assert [draw["draw_order"] for draw in first_draws] == list(range(9))
    assert [draw["purpose"] for draw in first_draws] == [
        "teacher_exit_flag_randint_compatibility",
        "transition_re_noise",
        "transition_re_noise",
        "transition_re_noise",
        "context_clean_recache_noise",
        "transition_re_noise",
        "transition_re_noise",
        "transition_re_noise",
        "context_clean_recache_noise",
    ]
    compatibility = first_draws[0]
    assert compatibility["operation"] == "torch.randint"
    assert compatibility["low"] == 0
    assert compatibility["high"] == 4
    assert compatibility["size"] == [2]
    assert compatibility["dtype"] == "torch.int64"
    assert compatibility["device"] == "cpu"
    assert compatibility["values_discarded"] is True
    assert compatibility["reason"] == (
        "match formal Teacher generate_and_sync_list(last_step_only=True) RNG consumption"
    )
    assert first.trace["rng"]["post_reset_global_rng_state_hash"] == compatibility[
        "state_before_hash"
    ]
    assert first.trace["rng"]["pre_solver_global_rng_state_hash"] == compatibility[
        "state_after_hash"
    ]
    for before, after in pairwise(first_draws):
        assert before["state_after_hash"] == after["state_before_hash"]
    assert compatibility["state_after_hash"] == first_draws[1]["state_before_hash"]
    assert compatibility["values"] == second_draws[0]["values"]
    assert [draw["state_before_hash"] for draw in first_draws] == [
        draw["state_before_hash"] for draw in second_draws
    ]
    assert [draw["state_after_hash"] for draw in first_draws] == [
        draw["state_after_hash"] for draw in second_draws
    ]
    assert [draw["noise"]["sha256"] for draw in first_draws[1:]] == [
        draw["noise"]["sha256"] for draw in second_draws[1:]
    ]


def test_teacher_compatibility_draw_is_called_once(monkeypatch: pytest.MonkeyPatch) -> None:
    original = m6.consume_teacher_exit_flag_rng_compatibility_draw
    calls = []

    def spy(**kwargs):
        calls.append(dict(kwargs))
        return original(**kwargs)

    monkeypatch.setattr(m6, "consume_teacher_exit_flag_rng_compatibility_draw", spy)

    run_fake_oracle()

    assert calls == [
        {
            "num_chunks": 2,
            "num_denoising_steps": 4,
            "device": torch.device("cpu"),
            "draw_order": 0,
        }
    ]


def test_teacher_compatibility_draw_aligns_next_rng_draw_with_manual_reference() -> None:
    seed = 789
    reference_tensor = torch.zeros((2, 3), dtype=torch.float32)

    m6.reset_rollout_rng(seed, "cpu")
    unburned_next = torch.randn_like(reference_tensor)

    m6.reset_rollout_rng(seed, "cpu")
    reference_values = torch.randint(
        low=0,
        high=4,
        size=(2,),
        device="cpu",
        dtype=torch.long,
    )
    reference_next = torch.randn_like(reference_tensor)

    m6.reset_rollout_rng(seed, "cpu")
    state_before = m6.global_rng_state_hash("cpu")
    record = m6.consume_teacher_exit_flag_rng_compatibility_draw(
        num_chunks=2,
        num_denoising_steps=4,
        device="cpu",
        draw_order=0,
    )
    actual_next = torch.randn_like(reference_tensor)

    assert not torch.equal(unburned_next, reference_next)
    assert torch.equal(actual_next, reference_next)
    assert record["state_before_hash"] == state_before
    assert record["values"] == [int(value) for value in reference_values.tolist()]
    assert record["values_discarded"] is True


def test_common_inputs_bind_teacher_rng_compatibility_contract_not_checkpoint() -> None:
    result_a, _ = run_fake_oracle(oracle_kind="A")
    result_b, _ = run_fake_oracle(
        oracle_kind="B",
        checkpoint_sha256="c" * 64,
    )

    common_inputs = result_a.summary["common_inputs"]
    assert common_inputs["rng_draw_contract_version"] == (
        "m6_ab_teacher_compatible_rng_draw_contract_v2"
    )
    assert common_inputs["rng_compatibility_contract"] == {
        "operation": "torch.randint",
        "purpose": "teacher_exit_flag_randint_compatibility",
        "low": 0,
        "high": 4,
        "size": [2],
        "dtype": "torch.int64",
        "values_discarded": True,
    }
    assert "checkpoint" not in common_inputs
    assert "checkpoint_sha256" not in common_inputs
    assert result_b.summary["common_inputs"] == common_inputs
    assert (
        result_b.summary["common_inputs_fingerprint_sha256"]
        == result_a.summary["common_inputs_fingerprint_sha256"]
    )


def test_each_chunk_has_four_forwards_each_rolled_back_and_one_clean_recache() -> None:
    generator = FakeGenerator()
    result, runtime = run_fake_oracle(generator=generator)
    denoise_calls = [call for call in generator.calls if not call["is_context"]]
    context_calls = [call for call in generator.calls if call["is_context"]]

    assert len(denoise_calls) == 8
    assert len(context_calls) == 2
    assert runtime.kv_cache[0]["local_end_index"].item() == 12
    for chunk in result.trace["chunks"]:
        assert len(chunk["solver_steps"]) == 4
        for step in chunk["solver_steps"]:
            before = step["kv"]["before"]["local_end_index"]
            rollback = step["kv"]["rollback_after_forward"]["local_end_index"]
            temp = step["kv"]["temporary_after_forward"]["local_end_index"]
            assert rollback == before
            assert temp > before
            assert step["kv"]["visible_data_restored"] is True
        assert chunk["clean_recache"]["after"]["local_end_index"] > chunk["clean_recache"]["before"]["local_end_index"]


def test_final_solver_forward_is_rolled_back_before_clean_recache() -> None:
    result, _ = run_fake_oracle()

    first_chunk = result.trace["chunks"][0]
    final_step = first_chunk["solver_steps"][-1]
    assert final_step["transition"] is None
    assert final_step["kv"]["rollback_after_forward"]["local_end_index"] == 0
    assert first_chunk["clean_recache"]["before"]["local_end_index"] == 0
    assert first_chunk["clean_recache"]["after"]["local_end_index"] == 6


def test_visible_kv_data_does_not_leak_when_forward_overwrites_visible_history() -> None:
    generator = FakeGenerator(overwrite_visible=True)
    result, runtime = run_fake_oracle(generator=generator)

    second_chunk_steps = result.trace["chunks"][1]["solver_steps"]
    assert all(step["kv"]["visible_data_restored"] is True for step in second_chunk_steps)
    assert torch.all(runtime.kv_cache[0]["k"][:, :6] >= 0)
    assert torch.all(runtime.kv_cache[0]["v"][:, :6] >= 0)


def test_kv_restore_false_fails_immediately(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(m6.M6KVSnapshot, "restore", lambda self, cache: False)

    with pytest.raises(RuntimeError, match="KV visible data restore failed"):
        run_fake_oracle()


def test_visible_kv_corruption_after_restore_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    original_restore = m6.M6KVSnapshot.restore

    def corrupt_after_restore(self, cache):
        restored = original_restore(self, cache)
        if int(cache[0]["local_end_index"].item()) > 0:
            cache[0]["k"][:, 0:1] = -999.0
        return restored

    monkeypatch.setattr(m6.M6KVSnapshot, "restore", corrupt_after_restore)

    with pytest.raises(RuntimeError, match="KV visible data restore failed"):
        run_fake_oracle()


def test_layer_boundary_inconsistency_fails_closed() -> None:
    with pytest.raises(RuntimeError, match="boundaries are inconsistent"):
        run_fake_oracle(generator=FakeGenerator(inconsistent_boundaries=True))


def test_kv_rollback_boundary_mismatch_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    original_restore = m6.M6KVSnapshot.restore

    def bad_restore(self, cache):
        restored = original_restore(self, cache)
        for layer in cache:
            layer["local_end_index"].add_(1)
            layer["global_end_index"].add_(1)
        return restored

    monkeypatch.setattr(m6.M6KVSnapshot, "restore", bad_restore)

    with pytest.raises(RuntimeError, match="rollback boundary mismatch"):
        run_fake_oracle()


@pytest.mark.parametrize("offset", [-1, 1])
def test_clean_recache_wrong_advancement_fails(offset: int) -> None:
    with pytest.raises(RuntimeError, match="clean recache after boundary mismatch"):
        run_fake_oracle(generator=FakeGenerator(recache_token_offset=offset))


def test_cursor_contract_rejects_wrong_cursor() -> None:
    with pytest.raises(RuntimeError, match="cursor_before"):
        m6._require_commit_contract(
            cursor_before=1,
            cursor_after=3,
            start_frame=0,
            chunk_frames=3,
        )


def test_nonzero_mcp_call_count_fails_closed() -> None:
    generator = FakeGenerator()
    generator.mcp_call_count = 1

    with pytest.raises(RuntimeError, match="mcp_call_count=0"):
        run_fake_oracle(generator=generator)


def test_oracle_a_and_b_do_not_call_mcp() -> None:
    for oracle_kind in ("A", "B"):
        generator = FakeGenerator()
        result, _ = run_fake_oracle(oracle_kind=oracle_kind, generator=generator)
        assert generator.mcp_call_count == 0
        assert all(not call["mcp_requested"] for call in generator.calls)
        assert result.trace["mcp_call_count"] == 0


def test_checkpoint_type_rejection_rules(monkeypatch) -> None:
    monkeypatch.setattr(m6, "file_sha256", lambda path: TEST_SHA256)
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: {"format": m6.M3_CHECKPOINT_FORMAT})

    with pytest.raises(RuntimeError, match="official"):
        m6.load_oracle_checkpoint(path=Path("formal.pt"), oracle_kind="A")

    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: {"generator": {"w": torch.zeros(1)}})
    with pytest.raises(RuntimeError, match="formal"):
        m6.load_oracle_checkpoint(path=Path("official.pt"), oracle_kind="B")

    generic_step0_payload = {
        "format": m6.M3_CHECKPOINT_FORMAT,
        "git_sha": TEST_GIT_SHA,
        "global_step": 0,
        "generator": {"mcp.weight": torch.zeros(1)},
    }
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: generic_step0_payload)
    monkeypatch.setattr(m6, "load_m3_checkpoint", lambda path: generic_step0_payload)
    with pytest.raises(TypeError, match="m5_formal_trainer"):
        m6.load_oracle_checkpoint(path=Path("generic_step0.pt"), oracle_kind="B")

    smoke_shaped_payload = {
        "format": m6.M3_CHECKPOINT_FORMAT,
        "git_sha": TEST_GIT_SHA,
        "global_step": 0,
        "generator": {"mcp.weight": torch.zeros(1)},
        "m5_formal_smoke": {"schema": "nf_sf_m5_formal_smoke_v1"},
    }
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: smoke_shaped_payload)
    monkeypatch.setattr(
        m6,
        "load_m3_checkpoint",
        lambda path: smoke_shaped_payload,
    )
    with pytest.raises(RuntimeError, match="smoke"):
        m6.load_oracle_checkpoint(path=Path("smoke_step0.pt"), oracle_kind="B")

    hybrid_payload = make_formal_step0_payload(top_level={"m5_formal_smoke": {"status": "PASS"}})
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: hybrid_payload)
    monkeypatch.setattr(m6, "load_m3_checkpoint", lambda path: hybrid_payload)
    with pytest.raises(RuntimeError, match="smoke"):
        m6.load_oracle_checkpoint(path=Path("hybrid_step0.pt"), oracle_kind="B")

    formal_payload = make_formal_step0_payload()
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: formal_payload)
    monkeypatch.setattr(
        m6,
        "load_m3_checkpoint",
        lambda path: formal_payload,
    )
    record = m6.load_oracle_checkpoint(path=Path("step0.pt"), oracle_kind="B")
    assert record.checkpoint_type == m6.M6_CHECKPOINT_FORMAL_STEP0
    assert record.global_step == 0
    assert record.mcp_tensor_count == 1
    assert record.formal_metadata["stage"] == "stage_a"


@pytest.mark.parametrize(
    ("payload_kwargs", "path", "match"),
    [
        ({"metadata": {"status": "FAIL"}}, "bad_step0.pt", "status"),
        ({"metadata": {"formal_enabled": False}}, "bad_step0.pt", "marker"),
        ({"metadata": {"smoke_enabled": True}}, "bad_step0.pt", "smoke"),
        ({"metadata": {"run_kind": "short_smoke"}}, "bad_step0.pt", "short_smoke"),
        ({"top_level": {"git_sha": "not-a-git-sha"}}, "bad_step0.pt", "git"),
        ({"metadata": {"sample_plan_sha256": "A" * 64}}, "bad_step0.pt", "lowercase"),
        ({}, "bad_step0.pt.tmp", ".tmp"),
    ],
)
def test_formal_step0_checkpoint_rejects_bad_marker(
    monkeypatch: pytest.MonkeyPatch,
    payload_kwargs: dict,
    path: str,
    match: str,
) -> None:
    monkeypatch.setattr(m6, "file_sha256", lambda path: TEST_SHA256)
    bad_payload = make_formal_step0_payload(**payload_kwargs)
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: bad_payload)
    monkeypatch.setattr(m6, "load_m3_checkpoint", lambda path: bad_payload)

    with pytest.raises((RuntimeError, ValueError), match=match):
        m6.load_oracle_checkpoint(path=Path(path), oracle_kind="B")


def test_oracle_b_vs_a_comparison_reports_metrics_and_not_generation_only_pass() -> None:
    actual = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float32).reshape(1, 3, 1, 1, 1)
    expected = torch.tensor([0.0, 2.0, 4.0], dtype=torch.float32).reshape(1, 3, 1, 1, 1)

    comparison = m6.compare_latents(actual, expected, tolerance=0.1)

    assert comparison["shape_match"] is True
    assert comparison["dtype_match"] is True
    assert comparison["exact_equality"] is False
    assert comparison["max_abs_diff"] == pytest.approx(2.0)
    assert comparison["mean_abs_diff"] == pytest.approx(1.0)
    assert comparison["mse"] == pytest.approx(5.0 / 3.0)
    assert comparison["reproduction_pass"] is False
    assert comparison["per_chunk"][0]["mse"] == pytest.approx(5.0 / 3.0)


def test_gate_fields_fail_closed_without_explicit_tolerance() -> None:
    result, _ = run_fake_oracle()

    assert result.summary["execution_status"] == "COMPLETED"
    assert result.summary["protocol_pass"] is True
    assert result.summary["target_reproduction_pass"] is None
    assert result.summary["oracle_a_reproduction_pass"] is None
    assert result.summary["oracle_gate_pass"] is None
    assert result.summary["status"] == "REPORT_ONLY"
    assert "TARGET_REPRODUCTION_TOLERANCE_NOT_PROVIDED" in result.summary["gate_reasons"]


def test_oracle_a_gate_pass_requires_target_reproduction_with_tolerance() -> None:
    result, _ = run_fake_oracle(
        generator=FakeGenerator(constant_clean=0.0),
        tolerance=0.0,
    )

    assert result.summary["target_reproduction_pass"] is True
    assert result.summary["oracle_gate_pass"] is True
    assert result.summary["status"] == "PASS"
    assert result.summary["gate_reasons"] == []


def test_oracle_b_gate_requires_oracle_a_comparison() -> None:
    result, _ = run_fake_oracle(
        oracle_kind="B",
        generator=FakeGenerator(constant_clean=0.0),
        tolerance=0.0,
    )

    assert result.summary["protocol_pass"] is True
    assert result.summary["target_reproduction_pass"] is True
    assert result.summary["oracle_a_reproduction_pass"] is False
    assert result.summary["oracle_gate_pass"] is False
    assert result.summary["status"] == "FAIL"
    assert "ORACLE_A_COMPARISON_MISSING" in result.summary["gate_reasons"]

    comparison = m6.compare_latents(result.latent, result.latent, tolerance=0.0)
    finalized = m6.finalize_oracle_gate(result, oracle_a_comparison=comparison)

    assert finalized.summary["oracle_a_reproduction_pass"] is True
    assert finalized.summary["oracle_gate_pass"] is True
    assert finalized.summary["status"] == "PASS"


def test_oracle_b_target_comparison_is_diagnostic_not_gate() -> None:
    result, _ = run_fake_oracle(oracle_kind="B", tolerance=0.0)
    comparison = m6.compare_latents(result.latent, result.latent, tolerance=0.0)
    finalized = m6.finalize_oracle_gate(result, oracle_a_comparison=comparison)

    assert finalized.summary["target_reproduction_pass"] is False
    assert finalized.summary["oracle_a_reproduction_pass"] is True
    assert finalized.summary["oracle_gate_pass"] is True
    assert finalized.summary["status"] == "PASS"
    assert finalized.summary["gate_reasons"] == []


def test_oracle_stdout_payload_uses_final_gate_status() -> None:
    result, _ = run_fake_oracle()
    payload = m6.oracle_stdout_payload(
        result=result,
        output_dir=Path("."),
        artifact_hashes={"oracle_summary_json_sha256": TEST_SHA256},
    )

    assert payload["status"] == "REPORT_ONLY"
    assert payload["oracle_gate_pass"] is None
    assert payload["gate_reasons"] == result.summary["gate_reasons"]


def test_nonfinite_latent_is_rejected() -> None:
    with pytest.raises(RuntimeError, match="non-finite"):
        run_fake_oracle(generator=FakeGenerator(nonfinite=True))


def test_json_payload_rejects_tensors_and_nonfinite_numbers() -> None:
    with pytest.raises(TypeError, match="tensors"):
        m6.validate_json_payload({"tensor": torch.zeros(1)})
    with pytest.raises(ValueError, match="non-finite"):
        m6.validate_json_payload({"value": float("inf")})


def test_non_empty_output_dir_is_rejected(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    (output_dir / "existing.txt").write_text("occupied", encoding="utf-8")

    with pytest.raises(FileExistsError, match="output_dir"):
        m6.prepare_output_dir(output_dir)


def test_trace_schema_required_fields_and_artifact_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    writer_calls = install_test_writers(monkeypatch)
    result, _ = run_fake_oracle(tolerance=0.0)
    output_dir = tmp_path / "oracle"
    hashes = m6.write_oracle_artifacts(
        output_dir=output_dir,
        resolved_config={"m6": {"schedule": result.trace["schedule"]}},
        result=result,
    )
    loaded = m6.load_output_latent_artifact(output_dir / "output_latent.pt")

    assert result.trace["schema"] == m6.M6_ORACLE_SCHEMA
    for key in (
        "status",
        "execution_status",
        "protocol_pass",
        "target_reproduction_pass",
        "oracle_a_reproduction_pass",
        "oracle_gate_pass",
        "gate_reasons",
        "oracle_kind",
        "git_sha",
        "checkpoint",
        "teacher_identity",
        "source_noise",
        "prompt_conditioning",
        "schedule",
        "rng",
        "chunks",
        "finite_checks",
        "mcp_call_count",
        "common_inputs",
        "common_inputs_fingerprint_sha256",
    ):
        assert key in result.trace
    assert torch.equal(loaded, result.latent)
    latent_payload = m6.load_output_latent_payload(output_dir / "output_latent.pt")
    assert latent_payload["common_inputs"] == result.trace["common_inputs"]
    assert (
        latent_payload["common_inputs_fingerprint_sha256"]
        == result.trace["common_inputs_fingerprint_sha256"]
        == result.summary["common_inputs_fingerprint_sha256"]
    )
    assert writer_calls == {"json": 3, "torch": 1}
    assert set(hashes) == {
        "resolved_config_json_sha256",
        "oracle_trace_json_sha256",
        "oracle_summary_json_sha256",
        "output_latent_pt_sha256",
    }


def test_write_oracle_b_comparison_from_oracle_a_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_test_writers(monkeypatch)
    result_a, _ = run_fake_oracle(
        oracle_kind="A",
        generator=FakeGenerator(constant_clean=0.0),
        tolerance=0.0,
    )
    result_b, _ = run_fake_oracle(
        oracle_kind="B",
        generator=FakeGenerator(constant_clean=0.0),
        tolerance=0.0,
    )
    root = tmp_path
    oracle_a_dir = root / "a"
    oracle_b_dir = root / "b"
    m6.write_oracle_artifacts(
        output_dir=oracle_a_dir,
        resolved_config={"oracle": "A"},
        result=result_a,
    )
    m6.prepare_output_dir(oracle_b_dir)

    comparison = m6.write_oracle_comparison(
        output_dir=oracle_b_dir,
        oracle_b_latent=result_b.latent,
        oracle_a_latent_path=oracle_a_dir / "output_latent.pt",
        tolerance=0.0,
    )

    assert comparison["exact_equality"] is True
    assert comparison["reproduction_pass"] is True
    assert (oracle_b_dir / "oracle_comparison.json").is_file()


def test_b_artifacts_are_written_after_final_gate_comparison(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    writer_calls = install_test_writers(monkeypatch)
    result_b, _ = run_fake_oracle(
        oracle_kind="B",
        generator=FakeGenerator(constant_clean=0.0),
        tolerance=0.0,
    )
    comparison = m6.compare_latents(result_b.latent, result_b.latent, tolerance=0.0)
    finalized = m6.finalize_oracle_gate(result_b, oracle_a_comparison=comparison)
    output_dir = tmp_path / "oracle"

    hashes = m6.write_oracle_artifacts(
        output_dir=output_dir,
        resolved_config={"oracle": "B"},
        result=finalized,
        oracle_comparison=comparison,
    )
    summary = json.loads((output_dir / "oracle_summary.json").read_text(encoding="utf-8"))

    assert summary["status"] == "PASS"
    assert summary["oracle_a_reproduction_pass"] is True
    assert (output_dir / "oracle_comparison.json").is_file()
    assert "oracle_comparison_json_sha256" in hashes
    assert writer_calls == {"json": 4, "torch": 1}


def write_pass_oracle_a_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    install_test_writers(monkeypatch)
    result, _ = run_fake_oracle(
        oracle_kind="A",
        generator=FakeGenerator(constant_clean=0.0),
        tolerance=0.0,
    )
    output_dir = tmp_path / "oracle_a"
    m6.write_oracle_artifacts(
        output_dir=output_dir,
        resolved_config={"oracle": "A"},
        result=result,
    )
    return output_dir, result


def test_validate_oracle_a_artifact_dir_accepts_strict_pass(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir, result = write_pass_oracle_a_artifacts(tmp_path, monkeypatch)

    record = m6.validate_oracle_a_artifact_dir(
        output_dir,
        expected_common_inputs_fingerprint_sha256=result.summary[
            "common_inputs_fingerprint_sha256"
        ],
    )

    assert record.latent_sha256 == m6.tensor_sha256(record.latent)
    assert record.common_inputs == result.summary["common_inputs"]


def test_validate_oracle_a_artifact_dir_rejects_b_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_test_writers(monkeypatch)
    result, _ = run_fake_oracle(
        oracle_kind="B",
        generator=FakeGenerator(constant_clean=0.0),
        tolerance=0.0,
    )
    comparison = m6.compare_latents(result.latent, result.latent, tolerance=0.0)
    result = m6.finalize_oracle_gate(result, oracle_a_comparison=comparison)
    output_dir = tmp_path / "oracle_b"
    m6.write_oracle_artifacts(
        output_dir=output_dir,
        resolved_config={"oracle": "B"},
        result=result,
    )

    with pytest.raises(RuntimeError, match="Oracle A"):
        m6.validate_oracle_a_artifact_dir(
            output_dir,
            expected_common_inputs_fingerprint_sha256=result.summary[
                "common_inputs_fingerprint_sha256"
            ],
        )


@pytest.mark.parametrize("tolerance", [None, 0.0])
def test_validate_oracle_a_artifact_dir_rejects_report_only_or_fail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tolerance: float | None,
) -> None:
    install_test_writers(monkeypatch)
    result, _ = run_fake_oracle(oracle_kind="A", tolerance=tolerance)
    output_dir = tmp_path / f"oracle_a_{tolerance}"
    m6.write_oracle_artifacts(
        output_dir=output_dir,
        resolved_config={"oracle": "A"},
        result=result,
    )

    with pytest.raises(RuntimeError, match="status"):
        m6.validate_oracle_a_artifact_dir(
            output_dir,
            expected_common_inputs_fingerprint_sha256=result.summary[
                "common_inputs_fingerprint_sha256"
            ],
        )


def test_validate_oracle_a_artifact_dir_rejects_fingerprint_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir, _ = write_pass_oracle_a_artifacts(tmp_path, monkeypatch)

    with pytest.raises(RuntimeError, match="fingerprint"):
        m6.validate_oracle_a_artifact_dir(
            output_dir,
            expected_common_inputs_fingerprint_sha256="0" * 64,
        )


def test_validate_oracle_a_artifact_dir_rejects_cross_file_fingerprint_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir, result = write_pass_oracle_a_artifacts(tmp_path, monkeypatch)
    summary_path = output_dir / "oracle_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["common_inputs_fingerprint_sha256"] = "0" * 64
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(RuntimeError, match="fingerprint"):
        m6.validate_oracle_a_artifact_dir(
            output_dir,
            expected_common_inputs_fingerprint_sha256=result.summary[
                "common_inputs_fingerprint_sha256"
            ],
        )


def test_validate_oracle_a_artifact_dir_rejects_tampered_latent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir, result = write_pass_oracle_a_artifacts(tmp_path, monkeypatch)
    latent_path = output_dir / "output_latent.pt"
    payload = torch.load(latent_path, map_location="cpu", weights_only=False)
    payload["latent"] = payload["latent"] + 1.0
    torch.save(payload, latent_path)

    with pytest.raises(RuntimeError, match="SHA256"):
        m6.validate_oracle_a_artifact_dir(
            output_dir,
            expected_common_inputs_fingerprint_sha256=result.summary[
                "common_inputs_fingerprint_sha256"
            ],
        )


def test_new_entry_does_not_import_legacy_mcp_entrypoint() -> None:
    source = Path("inference_next_forcing.py").read_text(encoding="utf-8")
    assert "import inference_mcp" not in source
    assert "from inference_mcp" not in source


def test_original_inference_entrypoint_is_not_rewired_to_m6() -> None:
    source = Path("inference.py").read_text(encoding="utf-8")
    assert "nf_sf_m6" not in source
    assert "inference_next_forcing" not in source
