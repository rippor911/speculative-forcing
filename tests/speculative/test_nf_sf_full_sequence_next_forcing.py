from __future__ import annotations

import inspect
import importlib.util
import json
import random
import shutil
import sys
import types
import uuid
from contextlib import contextmanager
from pathlib import Path

import pytest
import torch
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
            pass

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

_TRAINER_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "train_nf_sf_full_sequence_next_forcing.py"
)
_TRAINER_SPEC = importlib.util.spec_from_file_location(
    "train_nf_sf_full_sequence_next_forcing",
    _TRAINER_PATH,
)
assert _TRAINER_SPEC is not None
assert _TRAINER_SPEC.loader is not None
trainer = importlib.util.module_from_spec(_TRAINER_SPEC)
sys.modules[_TRAINER_SPEC.name] = trainer
_TRAINER_SPEC.loader.exec_module(trainer)
from utils.nf_sf_tensors import (
    DEFAULT_S_MAIN,
    DEFAULT_S_MCP,
    FULL_SEQUENCE_CHUNK_FRAMES,
    FULL_SEQUENCE_DEPTHS,
    FULL_SEQUENCE_FRAME_COUNT,
    FULL_SEQUENCE_NUM_CHUNKS,
    FULL_SEQUENCE_RNG_DRAW_ORDER_VERSION,
    expand_raw_chunk_timesteps,
    flow_match_shift_timesteps,
    make_cpu_generator,
    sample_nf_sf_full_sequence_noise_and_timesteps,
)
from utils.nf_sf_training import (
    FULL_SEQUENCE_CHECKPOINT_STEPS,
    FULL_SEQUENCE_CHUNK_TOKENS,
    FULL_SEQUENCE_DEPTH_WEIGHTS,
    FULL_SEQUENCE_TARGET_GLOBAL_STEP,
    NFSFFullSequenceNoisyBatch,
    NFSFSelectedState,
    build_full_sequence_mcp_anchor_inputs,
    build_full_sequence_mcp_anchor_specs,
    build_nf_sf_full_sequence_provenance,
    compute_nf_sf_full_sequence_losses,
    configure_nf_sf_full_sequence_optimizer_plan,
    full_sequence_anchor_token_slice,
    nf_sf_full_sequence_checkpoint_steps,
    nf_sf_full_sequence_train_cursor,
    prepare_nf_sf_full_sequence_noisy_batch,
    prepare_nf_sf_noisy_batch,
    require_nf_sf_full_sequence_runtime,
    validate_nf_sf_full_sequence_provenance,
)
from utils.scheduler import FlowMatchScheduler
from utils.wan_wrapper import (
    FULL_SEQUENCE_FUTURE_EMBEDDING_ORDER,
    WanDiffusionWrapper,
)


def _latent(batch: int = 2) -> torch.Tensor:
    total = batch * FULL_SEQUENCE_FRAME_COUNT
    return torch.arange(total, dtype=torch.float32).reshape(
        batch,
        FULL_SEQUENCE_FRAME_COUNT,
        1,
        1,
        1,
    )


def _scheduler(shift: float) -> FlowMatchScheduler:
    scheduler = FlowMatchScheduler(shift=shift, sigma_min=0.0, extra_one_step=True)
    scheduler.set_timesteps(1000, training=True)
    return scheduler


GIT_SHA = "bdd981c8c9b708f4c6b2fd1e3e75697c33b0d1ee"
SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64


@pytest.fixture
def tmp_path():
    parent = trainer.ROOT / "_codex_tmp_nf_sf_full_tests"
    path = parent / uuid.uuid4().hex
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _assert_samples_equal(left, right) -> None:
    assert torch.equal(left.epsilon_main, right.epsilon_main)
    assert all(
        torch.equal(a, b)
        for a, b in zip(left.epsilon_mcp_depths, right.epsilon_mcp_depths)
    )
    assert torch.equal(left.raw_timestep_main, right.raw_timestep_main)
    assert all(
        torch.equal(a, b)
        for a, b in zip(left.raw_timestep_mcp_depths, right.raw_timestep_mcp_depths)
    )
    assert torch.equal(left.timestep_main, right.timestep_main)
    assert all(
        torch.equal(a, b)
        for a, b in zip(left.timestep_mcp_depths, right.timestep_mcp_depths)
    )


def test_full_tensor_sampler_shapes_draw_order_and_restore_determinism() -> None:
    clean = _latent(batch=2)
    seed = 123
    actual = sample_nf_sf_full_sequence_noise_and_timesteps(
        clean,
        generator=make_cpu_generator(seed),
    )

    manual_rng = make_cpu_generator(seed)
    expected_epsilon_main = torch.randn(
        clean.shape,
        dtype=clean.dtype,
        device=clean.device,
        generator=manual_rng,
    )
    expected_epsilons = tuple(
        torch.randn(
            (2, FULL_SEQUENCE_NUM_CHUNKS - depth, 3, 1, 1, 1),
            dtype=clean.dtype,
            device=clean.device,
            generator=manual_rng,
        )
        for depth in FULL_SEQUENCE_DEPTHS
    )
    expected_raw_main = torch.randint(
        0,
        1000,
        (2, 7),
        dtype=torch.int64,
        generator=manual_rng,
    )
    expected_raw_mcp = tuple(
        torch.randint(
            0,
            1000,
            (2, FULL_SEQUENCE_NUM_CHUNKS - depth),
            dtype=torch.int64,
            generator=manual_rng,
        )
        for depth in FULL_SEQUENCE_DEPTHS
    )

    assert actual.rng_draw_order_version == FULL_SEQUENCE_RNG_DRAW_ORDER_VERSION
    assert actual.epsilon_main.shape == (2, 21, 1, 1, 1)
    assert [tuple(t.shape) for t in actual.epsilon_mcp_depths] == [
        (2, 6, 3, 1, 1, 1),
        (2, 5, 3, 1, 1, 1),
        (2, 4, 3, 1, 1, 1),
    ]
    assert torch.equal(actual.epsilon_main, expected_epsilon_main)
    assert all(
        torch.equal(actual_tensor, expected_tensor)
        for actual_tensor, expected_tensor in zip(
            actual.epsilon_mcp_depths,
            expected_epsilons,
        )
    )
    assert torch.equal(actual.raw_timestep_main, expected_raw_main)
    assert all(
        torch.equal(actual_tensor, expected_tensor)
        for actual_tensor, expected_tensor in zip(
            actual.raw_timestep_mcp_depths,
            expected_raw_mcp,
        )
    )

    rng = make_cpu_generator(77)
    _ = sample_nf_sf_full_sequence_noise_and_timesteps(clean, generator=rng)
    saved_state = rng.get_state()
    next_a = sample_nf_sf_full_sequence_noise_and_timesteps(clean, generator=rng)
    restored = make_cpu_generator(0)
    restored.set_state(saved_state)
    next_b = sample_nf_sf_full_sequence_noise_and_timesteps(
        clean,
        generator=restored,
    )
    _assert_samples_equal(next_a, next_b)


def test_raw_timestep_domain_and_shifted_timestep_formulas() -> None:
    samples = sample_nf_sf_full_sequence_noise_and_timesteps(
        _latent(batch=1),
        generator=make_cpu_generator(9),
    )

    assert samples.raw_timestep_main.dtype == torch.int64
    assert int(samples.raw_timestep_main.min()) >= 0
    assert int(samples.raw_timestep_main.max()) < 1000
    for raw in samples.raw_timestep_mcp_depths:
        assert int(raw.min()) >= 0
        assert int(raw.max()) < 1000

    expected_main = flow_match_shift_timesteps(
        samples.raw_timestep_main,
        shift=DEFAULT_S_MAIN,
    ).unsqueeze(-1).expand(-1, -1, 3).reshape(1, 21)
    torch.testing.assert_close(samples.timestep_main, expected_main)
    for raw, shifted in zip(
        samples.raw_timestep_mcp_depths,
        samples.timestep_mcp_depths,
    ):
        expected_mcp = expand_raw_chunk_timesteps(
            raw,
            chunk_frames=3,
            shift=DEFAULT_S_MCP,
        )
        torch.testing.assert_close(shifted, expected_mcp)


def test_prepare_noisy_batch_shapes_and_valid_anchor_targets() -> None:
    clean = _latent(batch=1)
    batch = prepare_nf_sf_full_sequence_noisy_batch(
        clean,
        scheduler_main=_scheduler(DEFAULT_S_MAIN),
        scheduler_mcp=_scheduler(DEFAULT_S_MCP),
        rng=make_cpu_generator(5),
    )

    assert batch.noisy_main.shape == (1, 21, 1, 1, 1)
    assert batch.target_flow_main.shape == (1, 21, 1, 1, 1)
    assert batch.raw_timestep_main.shape == (1, 7)
    assert [tuple(t.shape) for t in batch.noisy_mcp_depths] == [
        (1, 6, 3, 1, 1, 1),
        (1, 5, 3, 1, 1, 1),
        (1, 4, 3, 1, 1, 1),
    ]
    assert [tuple(t.shape) for t in batch.timestep_mcp_depths] == [
        (1, 6, 3),
        (1, 5, 3),
        (1, 4, 3),
    ]
    assert torch.equal(batch.clean_target[:, 3:6], clean[:, 3:6])
    assert torch.equal(batch.target_flow_mcp_depths[0][:, 0], batch.epsilon_mcp_depths[0][:, 0] - clean[:, 3:6])
    assert torch.equal(batch.target_flow_mcp_depths[2][:, 3], batch.epsilon_mcp_depths[2][:, 3] - clean[:, 18:21])


def test_main_anchor_token_slices_cover_noisy_half_without_overlap() -> None:
    slices = [full_sequence_anchor_token_slice(i) for i in range(7)]

    assert [(s.start, s.stop) for s in slices] == [
        (0, 4680),
        (4680, 9360),
        (9360, 14040),
        (14040, 18720),
        (18720, 23400),
        (23400, 28080),
        (28080, 32760),
    ]
    assert all(s.stop - s.start == FULL_SEQUENCE_CHUNK_TOKENS for s in slices)
    assert slices[0].start == 0
    assert slices[-1].stop == 32760
    assert all(left.stop == right.start for left, right in zip(slices, slices[1:]))


def test_mcp_valid_anchor_table_and_future_start_frames() -> None:
    specs = build_full_sequence_mcp_anchor_specs()
    by_depth = {
        depth: [spec for spec in specs if spec.depth == depth]
        for depth in FULL_SEQUENCE_DEPTHS
    }

    assert [len(by_depth[depth]) for depth in FULL_SEQUENCE_DEPTHS] == [6, 5, 4]
    assert [(spec.anchor_index, spec.target_chunk_index) for spec in by_depth[1]] == [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5),
        (5, 6),
    ]
    assert [(spec.anchor_index, spec.target_chunk_index) for spec in by_depth[2]] == [
        (0, 2),
        (1, 3),
        (2, 4),
        (3, 5),
        (4, 6),
    ]
    assert [(spec.anchor_index, spec.target_chunk_index) for spec in by_depth[3]] == [
        (0, 3),
        (1, 4),
        (2, 5),
        (3, 6),
    ]
    assert [spec.future_start_frame for spec in specs] == [
        3,
        6,
        9,
        12,
        15,
        18,
        6,
        9,
        12,
        15,
        18,
        9,
        12,
        15,
        18,
    ]
    assert [spec.flat_index for spec in specs] == list(range(15))


def test_anchor_inputs_are_anchor_major_with_stable_depth_major_flat_indices() -> None:
    batch = prepare_nf_sf_full_sequence_noisy_batch(
        _latent(batch=1),
        scheduler_main=_scheduler(DEFAULT_S_MAIN),
        scheduler_mcp=_scheduler(DEFAULT_S_MCP),
        rng=make_cpu_generator(15),
    )
    anchors = build_full_sequence_mcp_anchor_inputs(batch)

    assert [anchor["anchor_index"] for anchor in anchors] == [0, 1, 2, 3, 4, 5]
    assert anchors[0]["depths"] == (1, 2, 3)
    assert anchors[0]["future_start_frames"] == (3, 6, 9)
    assert anchors[0]["flat_indices"] == (0, 6, 11)
    assert anchors[4]["depths"] == (1, 2)
    assert anchors[4]["future_start_frames"] == (15, 18)
    assert anchors[4]["flat_indices"] == (4, 10)
    assert anchors[5]["depths"] == (1,)
    assert anchors[5]["future_start_frames"] == (18,)
    assert anchors[5]["flat_indices"] == (5,)


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


def test_full_loss_formula_counts_and_tail_exclusion() -> None:
    batch = _loss_batch()
    main_pred = torch.zeros_like(batch.target_flow_main)
    for chunk_index in range(7):
        main_pred[:, chunk_index * 3 : (chunk_index + 1) * 3] = float(chunk_index + 1)
    mcp_preds = (
        torch.ones((1, 6, 3, 1, 1, 1)) * 2.0,
        torch.ones((1, 5, 3, 1, 1, 1)) * 3.0,
        torch.ones((1, 4, 3, 1, 1, 1)) * 4.0,
    )

    losses = compute_nf_sf_full_sequence_losses(
        main_flow_pred=main_pred,
        mcp_flow_preds_by_depth=mcp_preds,
        noisy_batch=batch,
    )

    expected_main = sum(float((index + 1) ** 2) for index in range(7)) / 7.0
    expected_total = expected_main + 0.5 * 4.0 + 0.2 * 9.0 + 0.1 * 16.0
    assert len(losses.main_chunk_losses) == 7
    assert [len(values) for values in losses.mcp_anchor_losses] == [6, 5, 4]
    assert [float(loss.item()) for loss in losses.mcp_depth_losses] == [4.0, 9.0, 16.0]
    assert losses.main_loss.item() == pytest.approx(expected_main)
    assert losses.total_loss.item() == pytest.approx(expected_total)

    invalid_tail = (torch.ones((1, 7, 3, 1, 1, 1)), mcp_preds[1], mcp_preds[2])
    with pytest.raises(ValueError, match="MCP depth 1 shape mismatch"):
        compute_nf_sf_full_sequence_losses(
            main_flow_pred=main_pred,
            mcp_flow_preds_by_depth=invalid_tail,
            noisy_batch=batch,
        )


def test_main_only_full_control_loss_contract() -> None:
    batch = _loss_batch()
    main_pred = torch.ones_like(batch.target_flow_main)

    losses = compute_nf_sf_full_sequence_losses(
        main_flow_pred=main_pred,
        mcp_flow_preds_by_depth=(),
        noisy_batch=batch,
        objective_mode="main_only_full_control",
    )

    assert losses.total_loss.item() == pytest.approx(losses.main_loss.item())
    assert losses.mcp_depth_losses == ()
    assert losses.mcp_anchor_losses == ()
    with pytest.raises(ValueError, match="must not receive MCP"):
        compute_nf_sf_full_sequence_losses(
            main_flow_pred=main_pred,
            mcp_flow_preds_by_depth=(torch.zeros((1, 6, 3, 1, 1, 1)),),
            noisy_batch=batch,
            objective_mode="main_only_full_control",
        )


class FakeBackboneForRoute(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.patch_embedding = nn.Conv3d(1, 1, kernel_size=1, bias=False)
        self.backbone_weight = nn.Parameter(torch.tensor(1.0))
        self.num_frame_per_block = 3
        self.gradient_checkpointing = True
        self.freqs = torch.empty(0)
        self.calls = []

    def forward(self, x, **kwargs):
        self.calls.append({"x": x, **kwargs})
        batch = x.shape[0]
        features = tuple(torch.ones((batch, 32760, 2)) for _ in range(4))
        patch_inputs = kwargs.get("mcp_patch_inputs") or []
        embeds = tuple(torch.ones((chunk.shape[0], 4680, 2)) for chunk in patch_inputs)
        grids = tuple(torch.tensor([[3, 60, 104]]) for _ in patch_inputs)
        flow = x * self.backbone_weight
        if kwargs.get("return_features") is None:
            return flow
        return flow, {
            "features": features,
            "mcp_embeds": embeds,
            "mcp_grid_sizes": grids,
        }


class FakeMCPForRoute(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fusion = nn.Linear(2, 2, bias=False)
        self.mcp_modules = nn.ModuleList(
            [nn.Linear(2, 2, bias=False) for _ in range(3)]
        )
        self.calls = []

    def forward(
        self,
        *,
        features,
        future_embeds,
        future_grid_sizes,
        future_start_frames,
        timesteps,
        freqs,
    ):
        self.calls.append(
            {
                "feature_shapes": [tuple(feature.shape) for feature in features],
                "depth_count": len(future_embeds),
                "future_start_frames": list(future_start_frames),
                "timesteps": [t.detach().clone() for t in timesteps],
            }
        )
        preds = []
        for index, embed in enumerate(future_embeds):
            batch = embed.shape[0]
            value = torch.full((batch, 1, 3, 1, 1), float(index + 1))
            preds.append(value)
        return preds


class FakeFullRouteWrapper(WanDiffusionWrapper):
    def __init__(self) -> None:
        nn.Module.__init__(self)
        self.model = FakeBackboneForRoute()
        self.mcp = FakeMCPForRoute()
        self.mcp_tap_layers = (3, 11, 19, 29)
        self.uniform_timestep = False
        self.seq_len = 32760


def test_wrapper_full_route_runs_main_backbone_once_and_anchor_mcp_micro_loop() -> None:
    clean = torch.ones((1, 21, 1, 1, 1))
    batch = prepare_nf_sf_full_sequence_noisy_batch(
        clean,
        scheduler_main=_scheduler(DEFAULT_S_MAIN),
        scheduler_mcp=_scheduler(DEFAULT_S_MCP),
        rng=make_cpu_generator(4),
    )
    anchors = build_full_sequence_mcp_anchor_inputs(batch)
    wrapper = FakeFullRouteWrapper()

    outputs = wrapper.forward_full_sequence_next_forcing(
        noisy_image_or_video=batch.noisy_main,
        clean_x=batch.clean_target,
        conditional_dict={"prompt_embeds": torch.zeros((1, 1, 1))},
        timestep_main=batch.timestep_main,
        mcp_anchor_inputs=anchors,
    )

    assert len(wrapper.model.calls) == 1
    assert outputs.main_backbone_forward_count == 1
    assert outputs.main_flow_pred.shape == (1, 21, 1, 1, 1)
    assert [tuple(t.shape) for t in outputs.mcp_flow_preds_by_depth] == [
        (1, 6, 3, 1, 1, 1),
        (1, 5, 3, 1, 1, 1),
        (1, 4, 3, 1, 1, 1),
    ]
    assert outputs.future_embedding_order == FULL_SEQUENCE_FUTURE_EMBEDDING_ORDER
    assert outputs.tap_shapes == ((1, 32760, 2),) * 4
    assert outputs.anchor_token_slices == (
        (0, 4680),
        (4680, 9360),
        (9360, 14040),
        (14040, 18720),
        (18720, 23400),
        (23400, 28080),
        (28080, 32760),
    )
    assert len(wrapper.mcp.calls) == 6
    assert [call["depth_count"] for call in wrapper.mcp.calls] == [3, 3, 3, 3, 2, 1]
    assert [call["future_start_frames"] for call in wrapper.mcp.calls] == [
        [3, 6, 9],
        [6, 9, 12],
        [9, 12, 15],
        [12, 15, 18],
        [15, 18],
        [18],
    ]
    assert all(
        shape == (1, 4680, 2)
        for call in wrapper.mcp.calls
        for shape in call["feature_shapes"]
    )


def test_flattened_future_embedding_mapping_is_depth_major() -> None:
    batch = prepare_nf_sf_full_sequence_noisy_batch(
        torch.ones((1, 21, 1, 1, 1)),
        scheduler_main=_scheduler(DEFAULT_S_MAIN),
        scheduler_mcp=_scheduler(DEFAULT_S_MCP),
        rng=make_cpu_generator(8),
    )
    anchors = build_full_sequence_mcp_anchor_inputs(batch)
    wrapper = FakeFullRouteWrapper()
    entries = wrapper._flatten_full_sequence_mcp_anchor_inputs(anchors)

    assert [(entry["anchor_index"], entry["depth"]) for entry in entries] == [
        (0, 1),
        (1, 1),
        (2, 1),
        (3, 1),
        (4, 1),
        (5, 1),
        (0, 2),
        (1, 2),
        (2, 2),
        (3, 2),
        (4, 2),
        (0, 3),
        (1, 3),
        (2, 3),
        (3, 3),
    ]
    assert [entry["flat_index"] for entry in entries] == list(range(15))


class FakeBackboneForOptimizer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.patch_embedding = nn.Conv3d(1, 1, kernel_size=1, bias=False)
        self.backbone_weight = nn.Parameter(torch.tensor(0.25))
        self.num_frame_per_block = 3
        self.gradient_checkpointing = True


class FakeMCPForOptimizer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fusion = nn.Linear(1, 1, bias=False)
        self.mcp_modules = nn.ModuleList(
            [nn.Linear(1, 1, bias=False) for _ in range(3)]
        )


class FakeGeneratorForOptimizer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = FakeBackboneForOptimizer()
        self.mcp = FakeMCPForOptimizer()


def test_shared_patch_embedding_reused_and_optimizer_control_contract() -> None:
    generator = FakeGeneratorForOptimizer()

    full_plan = configure_nf_sf_full_sequence_optimizer_plan(
        generator,
        objective_mode="next_forcing_full",
        group_lrs={
            "backbone": 1.0,
            "patch_embedding": 2.0,
            "mcp": 3.0,
            "mcp_fusion": 3.0,
        },
    )
    full_groups = [group["name"] for group in full_plan.optimizer_param_groups]
    assert full_groups == [
        "backbone",
        "patch_embedding",
        "mcp_fusion",
        "mcp_depth1",
        "mcp_depth2",
        "mcp_depth3",
    ]

    state_keys = tuple(generator.state_dict().keys())
    assert sum(key.startswith("model.patch_embedding.") for key in state_keys) == 1
    assert not any(key.startswith("mcp.patch_embedding.") for key in state_keys)

    control_plan = configure_nf_sf_full_sequence_optimizer_plan(
        generator,
        objective_mode="main_only_full_control",
        group_lrs={
            "backbone": 1.0,
            "patch_embedding": 2.0,
            "mcp": 3.0,
            "mcp_fusion": 3.0,
        },
    )
    assert [group["name"] for group in control_plan.optimizer_param_groups] == [
        "backbone",
        "patch_embedding",
    ]
    assert all(
        not audit.requires_grad
        for audit in control_plan.audits
        if audit.name.startswith("mcp")
    )


def test_old_wrapper_forward_signature_and_fixed_window_api_remain_available() -> None:
    signature = inspect.signature(WanDiffusionWrapper.forward)
    assert "clean_x" in signature.parameters
    assert "mcp_future_noises" in signature.parameters
    assert "mcp_timesteps" in signature.parameters

    current = torch.zeros((1, 3, 1, 1, 1))
    state = NFSFSelectedState(
        clean_history=torch.zeros((1, 3, 1, 1, 1)),
        current_target=current,
        future_targets=(current + 1.0, current + 2.0, current + 3.0),
        current_start_frame=3,
    )
    old_batch = prepare_nf_sf_noisy_batch(
        state,
        scheduler_main=_scheduler(DEFAULT_S_MAIN),
        scheduler_mcp=_scheduler(DEFAULT_S_MCP),
        rng=make_cpu_generator(22),
    )
    assert old_batch.noisy_current.shape == current.shape
    assert [future.shape for future in old_batch.noisy_futures] == [
        current.shape,
        current.shape,
        current.shape,
    ]


def test_provenance_schema_records_self_forcing_adaptation_and_rejects_paper_exact() -> None:
    provenance = build_nf_sf_full_sequence_provenance(
        objective_mode="next_forcing_full",
        git_sha="bdd981c8c9b708f4c6b2fd1e3e75697c33b0d1ee",
    )

    assert provenance["paper_exact_reproduction"] is False
    assert provenance["adaptation_differences"]["paper_exact_mcp_attention"] is False
    assert provenance["adaptation_differences"]["mcp_history_via_fused_main_features"] is True
    assert provenance["adaptation_differences"]["mcp_future_attention_single_chunk_only"] is True
    assert "m5_formal" not in jsonish_keys(provenance)

    config = types.SimpleNamespace(num_frame_per_block=3, gradient_checkpointing=True)
    args = _checkpoint_args(trainer.ROOT, objective_mode="next_forcing_full")
    scripted_metadata = trainer.build_step0_metadata(
        args=args,
        config=config,
        sample_plan={"sample_plan_sha256": SHA_A},
        git_sha=GIT_SHA,
        preflight_report=_preflight_report(),
    )
    assert args.objective_mode == "next_forcing_full"
    scripted_provenance = scripted_metadata["provenance"]
    assert scripted_provenance["objective"]["mcp_zero_head_bootstrap"] is True
    assert (
        scripted_provenance["objective"]["first_step_fusion_zero_grad_expected"]
        is True
    )

    invalid = dict(provenance)
    invalid["paper_exact_reproduction"] = True
    with pytest.raises(ValueError, match="paper-exact"):
        validate_nf_sf_full_sequence_provenance(invalid)


def jsonish_keys(value) -> str:
    if isinstance(value, dict):
        return " ".join(str(key) + " " + jsonish_keys(child) for key, child in value.items())
    if isinstance(value, (list, tuple)):
        return " ".join(jsonish_keys(child) for child in value)
    return str(value)


def test_full_schedule_target_and_resume_cursor_mapping() -> None:
    assert nf_sf_full_sequence_checkpoint_steps() == FULL_SEQUENCE_CHECKPOINT_STEPS
    assert FULL_SEQUENCE_CHECKPOINT_STEPS == (0, 500, 2000, 5000)
    assert FULL_SEQUENCE_TARGET_GLOBAL_STEP == 5000
    with pytest.raises(ValueError, match="5000"):
        nf_sf_full_sequence_checkpoint_steps(target_global_step=2000)

    assert nf_sf_full_sequence_train_cursor(0) == {
        "global_step": 0,
        "sample_position": None,
        "cycle_index": 0,
        "next_sample_position": 0,
    }
    assert nf_sf_full_sequence_train_cursor(1)["sample_position"] == 0
    assert nf_sf_full_sequence_train_cursor(2048)["sample_position"] == 2047
    assert nf_sf_full_sequence_train_cursor(2049)["sample_position"] == 0
    assert nf_sf_full_sequence_train_cursor(5000) == {
        "global_step": 5000,
        "sample_position": 903,
        "cycle_index": 2,
        "next_sample_position": 904,
    }


def test_runtime_guard_requires_nfpb3_and_gradient_checkpointing() -> None:
    config = types.SimpleNamespace(num_frame_per_block=3)
    generator = types.SimpleNamespace(
        model=types.SimpleNamespace(
            num_frame_per_block=3,
            gradient_checkpointing=True,
        ),
        mcp=object(),
    )

    report = require_nf_sf_full_sequence_runtime(
        config=config,
        generator=generator,
        objective_mode="next_forcing_full",
    )
    assert report["model_num_frame_per_block"] == 3
    assert report["gradient_checkpointing"] is True

    generator.model.gradient_checkpointing = False
    with pytest.raises(ValueError, match="gradient_checkpointing"):
        require_nf_sf_full_sequence_runtime(
            config=config,
            generator=generator,
            objective_mode="next_forcing_full",
        )


def test_resolved_config_has_full_schema_and_no_formal_stage_metadata() -> None:
    config = types.SimpleNamespace(
        num_frame_per_block=3,
        gradient_checkpointing=True,
    )
    args = types.SimpleNamespace(
        config=trainer.CANONICAL_CONFIG_PATH,
        checkpoint=Path("checkpoints/self_forcing_dmd.pt"),
        sample_plan=Path("sample_plan.json"),
        manifest=Path("manifest.json"),
        dataset_root=Path("dataset"),
        conditionals_artifact=Path("conditionals"),
        objective_mode="next_forcing_full",
        train_seed=11,
        validation_seed=22,
        global_seed=33,
        backbone_lr=1.0e-5,
        patch_embedding_lr=2.0e-5,
        mcp_lr=3.0e-5,
        weight_decay=0.01,
        dtype="bf16",
        device="cuda:0",
    )

    resolved = trainer.full_sequence_resolved_config(
        config,
        args,
        preflight_report=_preflight_report(),
    )

    assert resolved["schema"] == "nf_sf_full_sequence_next_forcing_trainer_v1"
    assert resolved["run_kind"] == "nf_sf_full_sequence_next_forcing_v1"
    assert resolved["objective_mode"] == "next_forcing_full"
    assert resolved["expected_git_sha"] == GIT_SHA
    assert resolved["sample_plan_sha256"] == SHA_A
    assert resolved["manifest_sha256"] == SHA_B
    assert resolved["conditionals_artifact_sha256"] == SHA_C
    assert resolved["checkpoint_size_bytes"] == 123
    assert resolved["train_seed"] == 11
    assert resolved["validation_seed"] == 22
    assert resolved["global_seed"] == 33
    assert resolved["backbone_lr"] == pytest.approx(1.0e-5)
    assert resolved["patch_embedding_lr"] == pytest.approx(2.0e-5)
    assert resolved["mcp_lr"] == pytest.approx(3.0e-5)
    assert resolved["weight_decay"] == pytest.approx(0.01)
    assert resolved["adam_betas"] == [0.0, 0.999]
    assert resolved["adam_eps"] == pytest.approx(1.0e-8)
    assert resolved["device"] == "cuda:0"
    assert resolved["dtype"] == "bf16"
    assert resolved["production_target_global_step"] == 5000
    assert "m5_formal" not in jsonish_keys(resolved)


def test_sample_plan_contract_requires_2048_train_and_256_validation() -> None:
    plan = {
        "train_sample_identities": [f"train-{index}" for index in range(2048)],
        "validation_sample_identities": [
            f"validation-{index}" for index in range(256)
        ],
    }
    trainer.validate_sample_plan_contract(plan)

    bad = dict(plan)
    bad["train_sample_identities"] = plan["train_sample_identities"][:-1]
    with pytest.raises(RuntimeError, match="2048"):
        trainer.validate_sample_plan_contract(bad)


class FakeTrainBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.patch_embedding = nn.Conv3d(1, 1, kernel_size=1, bias=False)
        self.backbone_weight = nn.Parameter(torch.tensor(0.5))
        self.num_frame_per_block = 3
        self.gradient_checkpointing = True


class FakeTrainMCP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fusion = nn.Linear(1, 1, bias=False)
        nn.init.ones_(self.fusion.weight)
        self.mcp_modules = nn.ModuleList(
            [FakeTrainMCPModule() for _ in range(3)]
        )


class FakeTrainMCPModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.head = nn.Module()
        self.head.head = nn.Linear(1, 1, bias=False)
        nn.init.zeros_(self.head.head.weight)

    @property
    def weight(self) -> torch.Tensor:
        return self.head.head.weight


class FakeTrainGenerator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = FakeTrainBackbone()
        self.mcp = FakeTrainMCP()

    def forward_full_sequence_next_forcing(
        self,
        *,
        noisy_image_or_video,
        clean_x,
        conditional_dict,
        timestep_main,
        mcp_anchor_inputs=(),
    ):
        main_scale = (
            self.model.backbone_weight
            + self.model.patch_embedding.weight.reshape(()).to(noisy_image_or_video)
        )
        main = noisy_image_or_video * main_scale
        by_depth = []
        for depth in FULL_SEQUENCE_DEPTHS:
            chunks = []
            for anchor in mcp_anchor_inputs:
                depths = tuple(anchor["depths"])
                if depth not in depths:
                    continue
                local = depths.index(depth)
                noise = anchor["future_noises"][local]
                scale = (
                    self.mcp.mcp_modules[depth - 1].weight.reshape(())
                    * (1.0 + self.mcp.fusion.weight.reshape(()))
                ).to(noise)
                chunks.append(torch.ones_like(noise) * scale)
            if chunks:
                by_depth.append(torch.stack(chunks, dim=1))
        return types.SimpleNamespace(
            main_flow_pred=main,
            mcp_flow_preds_by_depth=tuple(by_depth),
            tap_shapes=((1, 32760, 2),) * 4 if mcp_anchor_inputs else (),
            anchor_token_slices=trainer.expected_anchor_token_slices(),
            main_backbone_forward_count=1,
            future_embedding_order="depth_major",
        )


class FakeSample:
    def __init__(self, target_latent: torch.Tensor) -> None:
        self.target_latent = target_latent
        self.metadata = {"identity": "sample-0"}


class FakeStore:
    def __init__(self, identities) -> None:
        self.train_identities = tuple(identities)
        self.validation_identities = tuple(identities)
        self.sample = FakeSample(torch.ones((1, 21, 1, 1, 1)))

    def train_identity_for_step(self, step: int) -> str:
        assert step == 1
        return self.train_identities[0]

    @contextmanager
    def acquire(self, identity: str):
        assert identity in set(self.train_identities) | set(self.validation_identities)
        yield self.sample


class FakeConditionalStore:
    @contextmanager
    def acquire(self, identity: str):
        yield {"prompt_embeds": torch.zeros((1, 1, 1))}


def test_smoke_contract_executes_one_optimizer_step_with_adam_state() -> None:
    generator = FakeTrainGenerator()
    optimizer, _ = trainer.build_optimizer(
        generator,
        objective_mode="next_forcing_full",
        backbone_lr=1e-4,
        patch_embedding_lr=1e-4,
        mcp_lr=1e-4,
        weight_decay=0.0,
    )

    record = trainer.run_full_sequence_train_step(
        generator=generator,
        optimizer=optimizer,
        scheduler_main=_scheduler(DEFAULT_S_MAIN),
        scheduler_mcp=_scheduler(DEFAULT_S_MCP),
        teacher_store=FakeStore(["sample-0"]),
        conditional_store=FakeConditionalStore(),
        train_rng=make_cpu_generator(31),
        global_step=1,
        objective_mode="next_forcing_full",
        device=torch.device("cpu"),
        dtype=torch.float32,
        smoke=True,
    )

    assert record["global_step"] == 1
    assert record["target_latent_shape"] == [1, 21, 1, 1, 1]
    assert record["main_pred_shape"] == [1, 21, 1, 1, 1]
    assert record["mcp_pred_shapes"] == [
        [1, 6, 3, 1, 1, 1],
        [1, 5, 3, 1, 1, 1],
        [1, 4, 3, 1, 1, 1],
    ]
    assert record["optimizer_state_entries"] > 0
    assert record["smoke"] is True
    assert record["structural_report"]["status"] == "PASS"
    assert record["structural_report"]["main_backbone_forward_count"] == 1
    assert record["structural_report"]["anchor_token_slices"] == [
        [0, 4680],
        [4680, 9360],
        [9360, 14040],
        [14040, 18720],
        [18720, 23400],
        [23400, 28080],
        [28080, 32760],
    ]
    fusion_report = record["gradient_report"]["mcp_fusion"]
    assert fusion_report["aggregate_grad_norm"] == pytest.approx(0.0)
    assert fusion_report["bootstrap_zero_grad_allowed"] is True
    assert fusion_report["bootstrap_reason"] == "zero_initialized_mcp_output_heads"
    for depth in ("mcp_depth1", "mcp_depth2", "mcp_depth3"):
        assert record["gradient_report"][depth]["aggregate_grad_norm"] > 0.0
    head_report = record["mcp_output_head_bootstrap_report"]
    assert head_report["mcp_output_heads_left_zero_init_after_step1"] is True
    assert all(item["norm_after_step"] > 0.0 for item in head_report["per_depth"])
    assert "after_cleanup" in record["memory"]
    assert record["smoke_memory_gate"]["status"] == "PASS"


def test_validation_rng_isolation() -> None:
    generator = FakeTrainGenerator()
    train_rng = make_cpu_generator(101)
    train_state_before = train_rng.get_state().clone()

    report = trainer.run_full_sequence_validation(
        generator=generator,
        scheduler_main=_scheduler(DEFAULT_S_MAIN),
        scheduler_mcp=_scheduler(DEFAULT_S_MCP),
        teacher_store=FakeStore(["sample-0", "sample-1"]),
        conditional_store=FakeConditionalStore(),
        validation_seed=202,
        train_rng=train_rng,
        objective_mode="next_forcing_full",
        device=torch.device("cpu"),
        dtype=torch.float32,
        global_step=0,
    )
    report_again = trainer.run_full_sequence_validation(
        generator=generator,
        scheduler_main=_scheduler(DEFAULT_S_MAIN),
        scheduler_mcp=_scheduler(DEFAULT_S_MCP),
        teacher_store=FakeStore(["sample-0", "sample-1"]),
        conditional_store=FakeConditionalStore(),
        validation_seed=202,
        train_rng=train_rng,
        objective_mode="next_forcing_full",
        device=torch.device("cpu"),
        dtype=torch.float32,
        global_step=500,
    )

    assert torch.equal(train_state_before, train_rng.get_state())
    assert report["identity_count"] == 2
    assert report["seed_derivation"] == "derive_m4_validation_seed"
    assert report["tensor_slot"] == trainer.VALIDATION_TENSOR_SLOT
    assert [item["derived_seed"] for item in report["per_sample"]] == [
        item["derived_seed"] for item in report_again["per_sample"]
    ]
    assert report["weighted_total"] == pytest.approx(report_again["weighted_total"])
    assert len(report["main_per_chunk"]) == 7
    assert [len(values) for values in report["mcp_per_anchor"]] == [6, 5, 4]


def test_validation_identity_seeds_are_distinct() -> None:
    seed_a = trainer.derive_m4_validation_seed(
        base_seed=202,
        sample_identity="sample-0",
        tensor_slot=trainer.VALIDATION_TENSOR_SLOT,
    )
    seed_b = trainer.derive_m4_validation_seed(
        base_seed=202,
        sample_identity="sample-1",
        tensor_slot=trainer.VALIDATION_TENSOR_SLOT,
    )

    assert seed_a != seed_b


def _grad_item(
    *,
    norm: float,
    expected_trainable: bool = True,
    trainable_tensors: int = 1,
    missing_grad_tensors: int = 0,
    all_finite: bool = True,
) -> dict:
    return {
        "expected_trainable": expected_trainable,
        "trainable_tensors": trainable_tensors,
        "grad_tensors": trainable_tensors - missing_grad_tensors,
        "missing_grad_tensors": missing_grad_tensors,
        "all_finite": all_finite,
        "aggregate_grad_norm": float(norm),
        "pass": (
            expected_trainable
            and trainable_tensors > 0
            and missing_grad_tensors == 0
            and all_finite
            and float(norm) > 0.0
        ),
    }


def _full_grad_report(*, fusion_norm: float = 1.0, depth_norms=(1.0, 1.0, 1.0)):
    return {
        "backbone": _grad_item(norm=1.0),
        "patch_embedding": _grad_item(norm=1.0),
        "mcp_fusion": _grad_item(norm=fusion_norm),
        "mcp_depth1": _grad_item(norm=depth_norms[0]),
        "mcp_depth2": _grad_item(norm=depth_norms[1]),
        "mcp_depth3": _grad_item(norm=depth_norms[2]),
    }


def test_step1_next_forcing_allows_only_mcp_fusion_zero_bootstrap() -> None:
    report = trainer.validate_full_sequence_gradient_audit(
        _full_grad_report(fusion_norm=0.0),
        objective_mode="next_forcing_full",
        global_step=1,
    )

    assert report["mcp_fusion"]["pass"] is True
    assert report["mcp_fusion"]["bootstrap_zero_grad_allowed"] is True
    assert (
        report["mcp_fusion"]["bootstrap_reason"]
        == "zero_initialized_mcp_output_heads"
    )


def test_step1_mcp_fusion_missing_grad_fails() -> None:
    report = _full_grad_report(fusion_norm=0.0)
    report["mcp_fusion"] = _grad_item(norm=0.0, missing_grad_tensors=1)

    with pytest.raises(RuntimeError, match="mcp_fusion"):
        trainer.validate_full_sequence_gradient_audit(
            report,
            objective_mode="next_forcing_full",
            global_step=1,
        )


def test_step1_mcp_depth_zero_norm_still_fails() -> None:
    with pytest.raises(RuntimeError, match="mcp_depth2"):
        trainer.validate_full_sequence_gradient_audit(
            _full_grad_report(fusion_norm=0.0, depth_norms=(1.0, 0.0, 1.0)),
            objective_mode="next_forcing_full",
            global_step=1,
        )


def test_step2_mcp_fusion_zero_norm_fails_strict_contract() -> None:
    with pytest.raises(RuntimeError, match="mcp_fusion"):
        trainer.validate_full_sequence_gradient_audit(
            _full_grad_report(fusion_norm=0.0),
            objective_mode="next_forcing_full",
            global_step=2,
        )


def test_main_only_control_does_not_accept_mcp_fusion_bootstrap_exception() -> None:
    report = {
        "backbone": _grad_item(norm=1.0),
        "patch_embedding": _grad_item(norm=1.0),
        "mcp_fusion": _grad_item(norm=0.0),
    }

    with pytest.raises(RuntimeError, match="mcp_fusion"):
        trainer.validate_full_sequence_gradient_audit(
            report,
            objective_mode="main_only_full_control",
            global_step=1,
        )


def test_step1_output_heads_still_zero_after_step_fails() -> None:
    generator = FakeTrainGenerator()
    before = trainer.capture_mcp_output_head_bootstrap_before_step(generator)

    with pytest.raises(RuntimeError, match="did not leave zero"):
        trainer.validate_mcp_output_heads_left_zero_init_after_step1(
            generator,
            before_report=before,
        )


def test_step1_output_heads_nonzero_after_step_passes() -> None:
    generator = FakeTrainGenerator()
    before = trainer.capture_mcp_output_head_bootstrap_before_step(generator)
    with torch.no_grad():
        for weight in trainer.mcp_output_head_weight_tensors(generator):
            weight.fill_(0.25)

    report = trainer.validate_mcp_output_heads_left_zero_init_after_step1(
        generator,
        before_report=before,
    )

    assert report["mcp_output_heads_left_zero_init_after_step1"] is True
    assert all(item["norm_after_step"] > 0.0 for item in report["per_depth"])


def _preflight_report(**overrides) -> dict:
    report = {
        "status": "PASS",
        "current_git_sha": GIT_SHA,
        "expected_git_sha": GIT_SHA,
        "sample_plan_sha256": SHA_A,
        "manifest_sha256": SHA_B,
        "conditionals_artifact_sha256": SHA_C,
        "checkpoint_sha256": trainer.OFFICIAL_SELF_FORCING_CHECKPOINT_SHA256,
        "checkpoint_size_bytes": 123,
    }
    report.update(overrides)
    return report


def _preflight_facts(tmp_path: Path, **overrides) -> dict:
    facts = {
        "git_top_level": str(trainer.ROOT.resolve()),
        "root": str(trainer.ROOT.resolve()),
        "current_git_sha": GIT_SHA,
        "expected_git_sha": GIT_SHA,
        "tracked_dirty": False,
        "staged_dirty": False,
        "output_dir": str(
            (trainer.ROOT.parent / "_nf_sf_full_sequence_outside" / tmp_path.name)
            .resolve()
        ),
        "config_path": str(trainer.CANONICAL_CONFIG_PATH),
        "device": "cuda:0",
        "dtype": "bf16",
        "cuda_available": True,
        "checkpoint_sha256": trainer.OFFICIAL_SELF_FORCING_CHECKPOINT_SHA256,
        "checkpoint_size_bytes": 123,
        "sample_plan_sha256": SHA_A,
        "expected_sample_plan_sha256": SHA_A,
        "manifest_sha256": SHA_B,
        "expected_manifest_sha256": SHA_B,
        "conditionals_artifact_sha256": SHA_C,
        "expected_conditionals_artifact_sha256": SHA_C,
    }
    facts.update(overrides)
    return facts


@pytest.mark.parametrize(
    ("override", "match"),
    [
        ({"tracked_dirty": True}, "tracked worktree"),
        ({"staged_dirty": True}, "staged index"),
        ({"expected_git_sha": "0" * 40}, "git HEAD"),
        ({"output_dir": str(trainer.ROOT / "runs" / "inside")}, "output_dir"),
        ({"config_path": str(Path("other.yaml").resolve())}, "config path"),
        ({"expected_sample_plan_sha256": "0" * 64}, "sample_plan SHA256"),
        ({"expected_manifest_sha256": "0" * 64}, "manifest SHA256"),
        (
            {"expected_conditionals_artifact_sha256": "0" * 64},
            "conditionals_artifact SHA256",
        ),
    ],
)
def test_repo_preflight_synthetic_rejects_invalid_identity(
    tmp_path: Path,
    override: dict,
    match: str,
) -> None:
    facts = _preflight_facts(tmp_path, **override)

    with pytest.raises(RuntimeError, match=match):
        trainer.validate_repo_preflight_facts(facts)


def test_repo_preflight_synthetic_pass_records_shas(tmp_path: Path) -> None:
    report = trainer.validate_repo_preflight_facts(_preflight_facts(tmp_path))

    assert report["status"] == "PASS"
    assert report["current_git_sha"] == GIT_SHA
    assert report["sample_plan_sha256"] == SHA_A
    assert report["manifest_sha256"] == SHA_B
    assert report["conditionals_artifact_sha256"] == SHA_C


def _checkpoint_args(tmp_path: Path, objective_mode: str = "next_forcing_full"):
    return types.SimpleNamespace(
        config=trainer.CANONICAL_CONFIG_PATH,
        checkpoint=tmp_path / "official_self_forcing.pt",
        sample_plan=tmp_path / "sample_plan.json",
        manifest=tmp_path / "manifest.json",
        dataset_root=tmp_path / "dataset",
        conditionals_artifact=tmp_path / "conditionals",
        objective_mode=objective_mode,
        train_seed=101,
        validation_seed=202,
        global_seed=303,
        backbone_lr=1.0e-4,
        patch_embedding_lr=1.0e-4,
        mcp_lr=1.0e-4,
        weight_decay=0.0,
        dtype="bf16",
        device="cpu",
    )


def _checkpoint_metadata(tmp_path: Path, objective_mode: str = "next_forcing_full"):
    config = types.SimpleNamespace(num_frame_per_block=3, gradient_checkpointing=True)
    args = _checkpoint_args(tmp_path, objective_mode=objective_mode)
    resolved = trainer.full_sequence_resolved_config(
        config,
        args,
        preflight_report=_preflight_report(),
    )
    provenance = build_nf_sf_full_sequence_provenance(
        objective_mode=objective_mode,
        git_sha=GIT_SHA,
    )
    return args, resolved, provenance


def _new_train_fixture(objective_mode: str = "next_forcing_full"):
    generator = FakeTrainGenerator()
    optimizer, _ = trainer.build_optimizer(
        generator,
        objective_mode=objective_mode,
        backbone_lr=1.0e-4,
        patch_embedding_lr=1.0e-4,
        mcp_lr=1.0e-4,
        weight_decay=0.0,
    )
    return generator, optimizer


def _save_checkpoint_fixture(
    tmp_path: Path,
    *,
    smoke: bool = False,
    global_step: int = 0,
):
    args, resolved, provenance = _checkpoint_metadata(tmp_path)
    generator, optimizer = _new_train_fixture()
    train_rng = make_cpu_generator(123)
    validation_base_rng = make_cpu_generator(456)
    path = trainer.save_full_sequence_checkpoint(
        output_dir=tmp_path / "run",
        generator=generator,
        optimizer=optimizer,
        global_step=global_step,
        train_rng=train_rng,
        validation_base_rng=validation_base_rng,
        validation_seed=202,
        sample_plan={"sample_plan_sha256": SHA_A},
        resolved_config=resolved,
        provenance=provenance,
        git_sha=GIT_SHA,
        reference_checkpoint_path=args.checkpoint,
        objective_mode="next_forcing_full",
        smoke=smoke,
    )
    return {
        "path": path,
        "resolved": resolved,
        "provenance": provenance,
        "optimizer_contract": trainer.optimizer_contract(optimizer),
        "train_rng_state": train_rng.get_state().clone(),
        "validation_rng_state": validation_base_rng.get_state().clone(),
    }


def _load_checkpoint_fixture(fixture: dict, **overrides):
    generator, optimizer = _new_train_fixture()
    train_rng = make_cpu_generator(0)
    validation_base_rng = make_cpu_generator(0)
    path = Path(overrides.get("path", fixture["path"]))
    payload = trainer.load_full_sequence_checkpoint(
        path,
        generator=generator,
        optimizer=optimizer,
        train_rng=train_rng,
        validation_base_rng=validation_base_rng,
        objective_mode=overrides.get("objective_mode", "next_forcing_full"),
        expected_resume_checkpoint_sha256=overrides.get(
            "expected_resume_checkpoint_sha256",
            trainer.file_sha256(path),
        ),
        expected_git_sha=overrides.get("expected_git_sha", GIT_SHA),
        expected_resolved_config=overrides.get(
            "expected_resolved_config",
            fixture["resolved"],
        ),
        expected_provenance=overrides.get(
            "expected_provenance",
            fixture["provenance"],
        ),
        expected_optimizer_contract=overrides.get(
            "expected_optimizer_contract",
            trainer.optimizer_contract(optimizer),
        ),
    )
    return payload, train_rng, validation_base_rng


def test_checkpoint_atomic_save_writes_sha_and_validation_sidecars(tmp_path: Path) -> None:
    fixture = _save_checkpoint_fixture(tmp_path)

    sidecars = trainer.checkpoint_sidecar_paths(fixture["path"])
    assert fixture["path"].name == "checkpoint_step000000.pt"
    assert sidecars["sha256"].is_file()
    assert sidecars["validation"].is_file()
    validation = trainer.validate_checkpoint_sidecars(path=fixture["path"])
    assert validation["status"] == "PASS"
    assert validation["schema"] == trainer.CHECKPOINT_VALIDATION_SCHEMA
    assert validation["global_step"] == 0
    assert validation["generator_key_count"] > 0

    args, resolved, provenance = _checkpoint_metadata(tmp_path)
    generator, optimizer = _new_train_fixture()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        trainer.save_full_sequence_checkpoint(
            output_dir=fixture["path"].parent,
            generator=generator,
            optimizer=optimizer,
            global_step=0,
            train_rng=make_cpu_generator(1),
            validation_base_rng=make_cpu_generator(2),
            validation_seed=202,
            sample_plan={"sample_plan_sha256": SHA_A},
            resolved_config=resolved,
            provenance=provenance,
            git_sha=GIT_SHA,
            reference_checkpoint_path=args.checkpoint,
            objective_mode="next_forcing_full",
            smoke=False,
        )


def test_resume_accepts_strict_production_checkpoint_and_restores_rng(
    tmp_path: Path,
) -> None:
    fixture = _save_checkpoint_fixture(tmp_path)

    payload, train_rng, validation_base_rng = _load_checkpoint_fixture(fixture)

    assert payload["restore_contract"]["status"] == "PASS"
    assert torch.equal(train_rng.get_state(), fixture["train_rng_state"])
    assert torch.equal(
        validation_base_rng.get_state(),
        fixture["validation_rng_state"],
    )


def test_resume_rejects_wrong_checkpoint_sha(tmp_path: Path) -> None:
    fixture = _save_checkpoint_fixture(tmp_path)

    with pytest.raises(RuntimeError, match="resume checkpoint SHA256"):
        _load_checkpoint_fixture(
            fixture,
            expected_resume_checkpoint_sha256="0" * 64,
        )


def test_resume_rejects_wrong_git(tmp_path: Path) -> None:
    fixture = _save_checkpoint_fixture(tmp_path)

    with pytest.raises(RuntimeError, match="git_sha"):
        _load_checkpoint_fixture(fixture, expected_git_sha="0" * 40)


@pytest.mark.parametrize(
    ("field", "match"),
    [
        ("sample_plan_sha256", "sample plan SHA"),
        ("manifest_sha256", "manifest SHA"),
        ("conditionals_artifact_sha256", "conditional artifact SHA"),
    ],
)
def test_resume_rejects_artifact_sha_drift(
    tmp_path: Path,
    field: str,
    match: str,
) -> None:
    fixture = _save_checkpoint_fixture(tmp_path)
    resolved = dict(fixture["resolved"])
    resolved[field] = "0" * 64

    with pytest.raises(RuntimeError, match=match):
        _load_checkpoint_fixture(fixture, expected_resolved_config=resolved)


def test_resume_rejects_semantic_config_drift(tmp_path: Path) -> None:
    fixture = _save_checkpoint_fixture(tmp_path)
    resolved = dict(fixture["resolved"])
    resolved["backbone_lr"] = 9.0e-4

    with pytest.raises(RuntimeError, match="resolved_config semantic"):
        _load_checkpoint_fixture(fixture, expected_resolved_config=resolved)


def test_resume_rejects_bad_cursor(tmp_path: Path) -> None:
    fixture = _save_checkpoint_fixture(tmp_path)
    payload = torch.load(fixture["path"], map_location="cpu", weights_only=False)
    payload["sample_cursor"] = {"global_step": 0, "sample_position": 99}
    bad_path = tmp_path / "bad" / "checkpoint_step000000.pt"
    bad_path.parent.mkdir(parents=True)
    trainer.atomic_torch_save(payload, bad_path)
    generator, optimizer = _new_train_fixture()
    trainer.write_checkpoint_sidecars(path=bad_path, payload=payload, optimizer=optimizer)

    with pytest.raises(RuntimeError, match="sample_cursor"):
        _load_checkpoint_fixture(fixture, path=bad_path)


def test_resume_rejects_non_production_smoke(tmp_path: Path) -> None:
    fixture = _save_checkpoint_fixture(tmp_path, smoke=True, global_step=1)

    with pytest.raises(RuntimeError, match="PRODUCTION"):
        _load_checkpoint_fixture(fixture)


def test_global_and_python_rng_save_restore_synthetic(tmp_path: Path) -> None:
    args, resolved, provenance = _checkpoint_metadata(tmp_path)
    generator, optimizer = _new_train_fixture()
    train_rng = make_cpu_generator(555)
    validation_base_rng = make_cpu_generator(666)
    random.seed(777)
    torch.manual_seed(888)
    expected_python_state = random.getstate()
    expected_cpu_state = torch.get_rng_state().clone()
    path = trainer.save_full_sequence_checkpoint(
        output_dir=tmp_path / "rng-run",
        generator=generator,
        optimizer=optimizer,
        global_step=0,
        train_rng=train_rng,
        validation_base_rng=validation_base_rng,
        validation_seed=202,
        sample_plan={"sample_plan_sha256": SHA_A},
        resolved_config=resolved,
        provenance=provenance,
        git_sha=GIT_SHA,
        reference_checkpoint_path=args.checkpoint,
        objective_mode="next_forcing_full",
        smoke=False,
    )
    random.seed(999)
    torch.manual_seed(999)
    fixture = {
        "path": path,
        "resolved": resolved,
        "provenance": provenance,
    }

    _load_checkpoint_fixture(fixture)

    assert random.getstate() == expected_python_state
    assert torch.equal(torch.get_rng_state(), expected_cpu_state)


def test_step0_validation_persistence_contract() -> None:
    assert trainer.should_run_step0_validation(
        resume_checkpoint=None,
        validation_steps=(0, 500, 2000, 5000),
    )
    assert not trainer.should_run_step0_validation(
        resume_checkpoint=Path("checkpoint_step000500.pt"),
        validation_steps=(0, 500, 2000, 5000),
    )


def test_smoke_after_cleanup_memory_schema() -> None:
    memory = {
        label: trainer.memory_snapshot(label, torch.device("cpu"))
        for label in (
            "before_sample",
            "after_forward",
            "after_backward",
            "after_optimizer_step",
            "after_cleanup",
        )
    }

    gate = trainer.validate_smoke_memory_gate(memory)

    assert gate["status"] == "PASS"
    assert gate["cuda"] is False


def test_full_gradient_audit_and_gc_use_interval() -> None:
    common = {
        "checkpoint_steps": (500, 2000, 5000),
        "validation_steps": (500, 2000, 5000),
        "memory_log_interval": 100,
    }

    assert trainer.should_run_full_gradient_audit(1, smoke=False, **common)
    assert trainer.should_run_full_gradient_audit(2, smoke=False, **common)
    assert not trainer.should_run_full_gradient_audit(3, smoke=False, **common)
    assert trainer.should_run_full_gradient_audit(100, smoke=False, **common)
    assert trainer.should_run_full_gradient_audit(500, smoke=False, **common)
    assert trainer.should_run_cleanup_gc(500, smoke=False, **common)
    assert not trainer.should_run_cleanup_gc(501, smoke=False, **common)
    assert trainer.should_capture_memory(2, smoke=True, **common)


def test_metrics_jsonl_append_and_summary_exclude_large_train_records(
    tmp_path: Path,
) -> None:
    metrics_path = tmp_path / "metrics.jsonl"
    record = {
        "global_step": 1,
        "sample_identity": "sample-0",
        "sample_cursor": nf_sf_full_sequence_train_cursor(1),
        "total_loss": 1.0,
        "main_loss": 1.0,
        "elapsed_ms": 12.0,
    }
    trainer.append_jsonl(metrics_path, record)

    assert json.loads(metrics_path.read_text(encoding="utf-8").strip()) == record
    summary = trainer.build_training_summary(
        objective_mode="next_forcing_full",
        status="DONE",
        target_global_step=5000,
        completed_global_step=1,
        checkpoint_steps=(0, 500, 2000, 5000),
        validation_steps=(0, 500, 2000, 5000),
        metrics_path=metrics_path,
        train_record_count=1,
        final_train_record=record,
        checkpoint_records=(),
        validation_summaries=(),
        memory_maxima={},
        resume_payload=None,
        smoke=False,
        reference_checkpoint_immutability={"status": "PASS"},
    )

    assert summary["metrics_jsonl"].endswith("metrics.jsonl")
    assert summary["train_record_count"] == 1
    assert summary["reference_checkpoint_immutability"]["status"] == "PASS"
    assert "train_records" not in summary


def test_teacher_store_source_guard_avoids_per_sample_official_rehash() -> None:
    source = inspect.getsource(trainer.run_training)
    store_block = source[
        source.index("teacher_store = M5TeacherSampleStore") :
        source.index("validate_store_identity_order")
    ].replace(" ", "")

    assert "reference_checkpoint_path=None" in store_block
    assert "reference_checkpoint_path=args.checkpoint" not in store_block


def test_source_guard_no_inference_mcp_or_runtime_decode_or_self_rollout() -> None:
    text = Path("scripts/train_nf_sf_full_sequence_next_forcing.py").read_text(
        encoding="utf-8"
    )

    assert "inference_mcp" not in text
    assert "decode_to_pixel" not in text
    assert "self_rollout" not in text
    assert "run_nf_sf_mcp1_grid_point_loss" not in text
