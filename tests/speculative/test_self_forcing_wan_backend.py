from __future__ import annotations

import unittest
from types import SimpleNamespace
from typing import Any, Optional

import torch

from speculative.adapters.runtime_state import RuntimeStateTransactionManager
from speculative.adapters.self_forcing_runtime import (
    RuntimeWindowDescriptor,
    SelfForcingMCPRolloutPlan,
    SelfForcingMCPRuntimeConfig,
    SelfForcingMCPRuntimeContext,
)
from speculative.adapters.self_forcing_wan_backend import SelfForcingWanMCPBackend
from speculative.adapters.wan_state_planner import (
    SUPPORTED_ATTENTION_MODE,
    TensorRangeDescriptor,
    WanCacheLayout,
    WanStateMutationPlan,
    WanTouchedRangePlanner,
)
from speculative.types import BlockRef, CommitRequest, ControlRequest, DraftCandidate


FRAME_SEQ_LENGTH = 4
BLOCK_FRAMES = 2
TOTAL_FRAMES = 6
LAYERS = 2
CROSS_CAPACITY = 4


def make_config(
    *,
    total_frames: int = TOTAL_FRAMES,
    block_frames: int = BLOCK_FRAMES,
    mcp_depth: int = 2,
) -> SelfForcingMCPRuntimeConfig:
    return SelfForcingMCPRuntimeConfig(
        anchor_denoising_steps=(1000,),
        frame_seq_length=FRAME_SEQ_LENGTH,
        num_frame_per_block=block_frames,
        all_num_frames=total_frames,
        mcp_depth=mcp_depth,
        attention_mode=SUPPORTED_ATTENTION_MODE,
        validated_attention_mode=SUPPORTED_ATTENTION_MODE,
    )


def make_rollout_plan(config: SelfForcingMCPRuntimeConfig) -> SelfForcingMCPRolloutPlan:
    blocks: list[BlockRef] = []
    starts: list[int] = []
    start = 0
    remaining = config.all_num_frames
    while remaining > 0:
        frames = min(config.num_frame_per_block, remaining)
        blocks.append(BlockRef(index=len(blocks), start_frame=start, num_frames=frames))
        starts.append(start)
        start += frames
        remaining -= frames
    period = config.mcp_depth + 1
    return SelfForcingMCPRolloutPlan(
        blocks=tuple(blocks),
        block_starts=tuple(starts),
        anchor_block_indices=tuple(range(0, len(blocks), period)),
        period=period,
    )


def make_planner(
    *,
    total_frames: int = TOTAL_FRAMES,
    block_frames: int = BLOCK_FRAMES,
    mcp_depth: int = 2,
    layers: int = LAYERS,
    local_attn_size: int = -1,
    sink_size: int = 0,
    runtime_device_type: str = "cpu",
) -> WanTouchedRangePlanner:
    layout = WanCacheLayout(
        num_layers=layers,
        cache_capacity=total_frames * FRAME_SEQ_LENGTH,
        cross_attention_capacity=CROSS_CAPACITY,
    )
    return WanTouchedRangePlanner(
        frame_seq_length=FRAME_SEQ_LENGTH,
        num_frame_per_block=block_frames,
        mcp_depth=mcp_depth,
        cache_layout=layout,
        local_attn_size=local_attn_size,
        sink_size=sink_size,
        runtime_device_type=runtime_device_type,  # type: ignore[arg-type]
    )


def make_kv_cache(
    *,
    layers: int = LAYERS,
    capacity: int = TOTAL_FRAMES * FRAME_SEQ_LENGTH,
) -> list[dict[str, Any]]:
    cache = []
    for _ in range(layers):
        cache.append(
            {
                "k": torch.zeros((1, capacity, 1, 1), dtype=torch.float32),
                "v": torch.zeros((1, capacity, 1, 1), dtype=torch.float32),
                "global_end_index": torch.tensor([0], dtype=torch.long),
                "local_end_index": torch.tensor([0], dtype=torch.long),
            }
        )
    return cache


def make_cross_cache(
    *,
    layers: int = LAYERS,
    capacity: int = CROSS_CAPACITY,
    fill: float = 0.0,
) -> list[dict[str, Any]]:
    cache = []
    for _ in range(layers):
        cache.append(
            {
                "k": torch.full((1, capacity, 1, 1), fill, dtype=torch.float32),
                "v": torch.full((1, capacity, 1, 1), fill, dtype=torch.float32),
                "is_init": False,
            }
        )
    return cache


class FakeModel:
    def __init__(
        self,
        *,
        model_type: str = "t2v",
        local_attn_size: int = -1,
        sink_size: int = 0,
        patch_device: str = "cpu",
        freqs_device: str = "cpu",
    ) -> None:
        self.model_type = model_type
        self.local_attn_size = local_attn_size
        self.patch_embedding = SimpleNamespace(
            weight=torch.empty((1,), device=patch_device)
        )
        self.freqs = torch.ones((2,), device=freqs_device)
        self.blocks = [
            SimpleNamespace(self_attn=SimpleNamespace(sink_size=sink_size))
        ]


class FakeGenerator:
    def __init__(self, model: FakeModel, *, uniform_timestep: object = False) -> None:
        self.model = model
        self.uniform_timestep = uniform_timestep
        self.calls: list[dict[str, Any]] = []
        self.raise_on_call: Optional[Exception] = None
        self.invalid_staging_layer: Optional[int] = None
        self.result_override: Optional[Any] = None

    def __call__(self, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        self.calls.append(kwargs)
        if self.raise_on_call is not None:
            raise self.raise_on_call

        noisy = kwargs["noisy_image_or_video"]
        current_start = int(kwargs["current_start"])
        cache_start = int(kwargs.get("cache_start", current_start))
        kv_cache = kwargs["kv_cache"]
        crossattn_cache = kwargs["crossattn_cache"]
        token_count = noisy.shape[1] * FRAME_SEQ_LENGTH
        cache_end = cache_start + token_count

        for layer_index, layer in enumerate(kv_cache):
            layer["k"][:, cache_start:cache_end].fill_(10.0 + layer_index)
            layer["v"][:, cache_start:cache_end].fill_(20.0 + layer_index)
            layer["global_end_index"].fill_(cache_end)
            layer["local_end_index"].fill_(cache_end)

        for layer_index, layer in enumerate(crossattn_cache):
            if not layer["is_init"]:
                layer["k"] = torch.full_like(layer["k"], 100.0 + layer_index)
                layer["v"] = torch.full_like(layer["v"], 200.0 + layer_index)
                layer["is_init"] = True

        if self.invalid_staging_layer is not None:
            layer = crossattn_cache[self.invalid_staging_layer]
            layer["v"] = torch.zeros((1, CROSS_CAPACITY + 1, 1, 1))

        if self.result_override is not None:
            return self.result_override(**kwargs)

        flow = torch.full_like(noisy, 0.25)
        denoised = noisy + 1.0
        if "mcp_future_noises" not in kwargs:
            return flow, denoised

        mcp_flows = [
            torch.full_like(noise, float(depth))
            for depth, noise in enumerate(kwargs["mcp_future_noises"], start=1)
        ]
        return flow, denoised, mcp_flows


class FakePipeline:
    def __init__(
        self,
        *,
        model: Optional[FakeModel] = None,
        layers: int = LAYERS,
        total_frames: int = TOTAL_FRAMES,
        block_frames: int = BLOCK_FRAMES,
        mcp_modules: int = 2,
        uniform_timestep: object = False,
    ) -> None:
        self.denoising_step_list = [1000]
        self.i2v = False
        self.independent_first_frame = False
        self.mcp_num_modules = mcp_modules
        self.mcp_accel_depths = mcp_modules
        self.frame_seq_length = FRAME_SEQ_LENGTH
        self.num_frame_per_block = block_frames
        self.context_noise = 0
        self.num_transformer_blocks = layers
        self.kv_cache_size = total_frames * FRAME_SEQ_LENGTH
        self.generator = FakeGenerator(
            model or FakeModel(),
            uniform_timestep=uniform_timestep,
        )
        self.kv_cache1 = make_kv_cache(layers=layers, capacity=total_frames * FRAME_SEQ_LENGTH)
        self.crossattn_cache = make_cross_cache(layers=layers)
        self.init_calls: list[tuple[int, torch.dtype, torch.device]] = []
        self.future_calls: list[dict[str, Any]] = []
        self.future_override: Optional[tuple[list[Any], list[Any]]] = None
        self.last_future_noises: Optional[list[Any]] = None
        self.last_future_starts: Optional[list[Any]] = None
        self.commit_calls: list[dict[str, Any]] = []
        self.bad_commit_end = False

    def _initialize_crossattn_cache(
        self,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        self.init_calls.append((batch_size, dtype, device))
        self.crossattn_cache = []
        for _ in range(self.num_transformer_blocks):
            self.crossattn_cache.append(
                {
                    "k": torch.zeros((batch_size, CROSS_CAPACITY, 1, 1), dtype=dtype, device=device),
                    "v": torch.zeros((batch_size, CROSS_CAPACITY, 1, 1), dtype=dtype, device=device),
                    "is_init": False,
                }
            )

    def _mcp_future_chunks(
        self,
        block_index: int,
        block_starts: tuple[int, ...],
        all_num_frames: tuple[int, ...],
        noise: torch.Tensor,
        num_input_frames: int,
    ) -> tuple[list[Any], list[Any]]:
        self.future_calls.append(
            {
                "block_index": block_index,
                "block_starts": block_starts,
                "all_num_frames": all_num_frames,
                "noise": noise,
                "num_input_frames": num_input_frames,
            }
        )
        if self.future_override is not None:
            noises, starts = self.future_override
            self.last_future_noises = noises
            self.last_future_starts = starts
            return noises, starts

        current_frames = all_num_frames[block_index]
        noises: list[Any] = []
        starts: list[Any] = []
        for depth in range(self.mcp_num_modules):
            future_index = block_index + depth + 1
            if future_index >= len(all_num_frames):
                noises.append(None)
                starts.append(None)
                continue
            future_frames = all_num_frames[future_index]
            lo = block_starts[future_index] - num_input_frames
            hi = lo + future_frames
            if future_frames != current_frames or lo < 0 or hi > noise.shape[1]:
                noises.append(None)
                starts.append(None)
                continue
            noises.append(noise[:, lo:hi])
            starts.append(block_starts[future_index])
        self.last_future_noises = noises
        self.last_future_starts = starts
        return noises, starts

    def _cache_start(self) -> int:
        return int(self.kv_cache1[0]["global_end_index"].item())

    def _commit_context_block(
        self,
        latent: torch.Tensor,
        start_frame: int,
        conditional_dict: dict[str, Any],
    ) -> None:
        self.commit_calls.append(
            {
                "latent": latent,
                "start_frame": start_frame,
                "conditional_dict": conditional_dict,
            }
        )
        torch.randn_like(latent.flatten(0, 1))
        timestep = torch.full((latent.shape[0], latent.shape[1]), self.context_noise, dtype=torch.int64)
        self.generator(
            noisy_image_or_video=latent,
            conditional_dict=conditional_dict,
            timestep=timestep,
            kv_cache=self.kv_cache1,
            crossattn_cache=self.crossattn_cache,
            current_start=start_frame * FRAME_SEQ_LENGTH,
            cache_start=self._cache_start(),
        )
        if self.bad_commit_end:
            self.kv_cache1[-1]["local_end_index"].add_(1)


class Harness:
    def __init__(
        self,
        *,
        config: Optional[SelfForcingMCPRuntimeConfig] = None,
        planner: Optional[WanTouchedRangePlanner] = None,
        pipeline: Optional[FakePipeline] = None,
        total_frames: int = TOTAL_FRAMES,
        block_frames: int = BLOCK_FRAMES,
        mcp_depth: int = 2,
    ) -> None:
        self.config = config or make_config(
            total_frames=total_frames,
            block_frames=block_frames,
            mcp_depth=mcp_depth,
        )
        self.plan = make_rollout_plan(self.config)
        self.planner = planner or make_planner(
            total_frames=total_frames,
            block_frames=block_frames,
            mcp_depth=mcp_depth,
        )
        self.pipeline = pipeline or FakePipeline(
            total_frames=total_frames,
            block_frames=block_frames,
            mcp_modules=mcp_depth,
        )
        self.source_noise = torch.arange(
            total_frames * 4,
            dtype=torch.float32,
        ).reshape(1, total_frames, 1, 2, 2)
        self.output = torch.zeros_like(self.source_noise)
        self.pipeline.kv_cache1 = make_kv_cache(
            capacity=total_frames * FRAME_SEQ_LENGTH,
        )
        self.pipeline.crossattn_cache = make_cross_cache()
        self.context = SelfForcingMCPRuntimeContext(
            config=self.config,
            rollout_plan=self.plan,
            source_noise=self.source_noise,
            output=self.output,
            kv_cache=self.pipeline.kv_cache1,
            cross_attention_cache=self.pipeline.crossattn_cache,
        )
        self.backend = SelfForcingWanMCPBackend(
            pipeline=self.pipeline,
            conditional_dict={"prompt_embeds": torch.zeros((1, 3, 1))},
            planner=self.planner,
        )


class SelfForcingWanBackendConfigTests(unittest.TestCase):
    def test_unsupported_schedule_fails_fast(self) -> None:
        pipeline = FakePipeline()
        pipeline.denoising_step_list = [999]

        with self.assertRaisesRegex(ValueError, "denoising_step_list"):
            SelfForcingWanMCPBackend(
                pipeline=pipeline,
                conditional_dict={"prompt_embeds": torch.zeros((1, 1, 1))},
                planner=make_planner(),
            )

    def test_t2v_only_fails_fast(self) -> None:
        pipeline = FakePipeline()
        pipeline.i2v = True
        with self.assertRaisesRegex(ValueError, "T2V"):
            SelfForcingWanMCPBackend(
                pipeline=pipeline,
                conditional_dict={"prompt_embeds": torch.zeros((1, 1, 1))},
                planner=make_planner(),
            )

        pipeline = FakePipeline(model=FakeModel(model_type="i2v"))
        with self.assertRaisesRegex(ValueError, "T2V"):
            SelfForcingWanMCPBackend(
                pipeline=pipeline,
                conditional_dict={"prompt_embeds": torch.zeros((1, 1, 1))},
                planner=make_planner(),
            )

    def test_local_attention_and_sink_size_fail_fast(self) -> None:
        with self.assertRaisesRegex(ValueError, "local_attn_size"):
            SelfForcingWanMCPBackend(
                pipeline=FakePipeline(),
                conditional_dict={"prompt_embeds": torch.zeros((1, 1, 1))},
                planner=make_planner(local_attn_size=1),
            )

        with self.assertRaisesRegex(ValueError, "sink_size"):
            SelfForcingWanMCPBackend(
                pipeline=FakePipeline(model=FakeModel(sink_size=1)),
                conditional_dict={"prompt_embeds": torch.zeros((1, 1, 1))},
                planner=make_planner(),
            )

    def test_causal_wrapper_false_constructs_backend(self) -> None:
        pipeline = FakePipeline(uniform_timestep=False)

        backend = SelfForcingWanMCPBackend(
            pipeline=pipeline,
            conditional_dict={"prompt_embeds": torch.zeros((1, 1, 1))},
            planner=make_planner(),
        )

        self.assertIs(backend.pipeline, pipeline)

    def test_non_causal_wrapper_values_are_rejected(self) -> None:
        for value in (True, None):
            with self.subTest(uniform_timestep=value):
                pipeline = FakePipeline(uniform_timestep=value)
                with self.assertRaisesRegex(ValueError, "uniform_timestep"):
                    SelfForcingWanMCPBackend(
                        pipeline=pipeline,
                        conditional_dict={"prompt_embeds": torch.zeros((1, 1, 1))},
                        planner=make_planner(),
                    )

        pipeline = FakePipeline()
        delattr(pipeline.generator, "uniform_timestep")
        with self.assertRaisesRegex(ValueError, "uniform_timestep"):
            SelfForcingWanMCPBackend(
                pipeline=pipeline,
                conditional_dict={"prompt_embeds": torch.zeros((1, 1, 1))},
                planner=make_planner(),
            )

    def test_pipeline_planner_alignment_fields_fail_fast(self) -> None:
        mismatches = {
            "frame_seq_length": FRAME_SEQ_LENGTH + 1,
            "num_frame_per_block": BLOCK_FRAMES + 1,
            "num_transformer_blocks": LAYERS + 1,
            "kv_cache_size": TOTAL_FRAMES * FRAME_SEQ_LENGTH + 1,
            "mcp_num_modules": 3,
            "mcp_accel_depths": 3,
            "context_noise": 1,
        }
        for field_name, bad_value in mismatches.items():
            with self.subTest(field_name=field_name, kind="mismatch"):
                pipeline = FakePipeline()
                setattr(pipeline, field_name, bad_value)
                with self.assertRaisesRegex(ValueError, f"pipeline.{field_name}"):
                    SelfForcingWanMCPBackend(
                        pipeline=pipeline,
                        conditional_dict={"prompt_embeds": torch.zeros((1, 1, 1))},
                        planner=make_planner(),
                    )

        for field_name in mismatches:
            with self.subTest(field_name=field_name, kind="missing"):
                pipeline = FakePipeline()
                delattr(pipeline, field_name)
                with self.assertRaisesRegex(ValueError, f"pipeline.{field_name}"):
                    SelfForcingWanMCPBackend(
                        pipeline=pipeline,
                        conditional_dict={"prompt_embeds": torch.zeros((1, 1, 1))},
                        planner=make_planner(),
                    )

    def test_planner_runtime_device_must_match_source_noise_device(self) -> None:
        harness = Harness(planner=make_planner(runtime_device_type="cuda"))
        request = ControlRequest(anchor_block=harness.plan.blocks[0], max_depth=1)

        with self.assertRaisesRegex(ValueError, "runtime_device_type"):
            harness.backend.propose_anchor_and_drafts(request, harness.context)

    def test_constructor_migrates_freqs_to_patch_embedding_device(self) -> None:
        model = FakeModel(patch_device="meta", freqs_device="cpu")
        pipeline = FakePipeline(model=model)

        SelfForcingWanMCPBackend(
            pipeline=pipeline,
            conditional_dict={"prompt_embeds": torch.zeros((1, 1, 1))},
            planner=make_planner(),
        )

        self.assertEqual(model.freqs.device.type, "meta")

    def test_operation_fails_when_freqs_rebound_to_wrong_device(self) -> None:
        harness = Harness()
        harness.pipeline.generator.model.freqs = torch.empty((2,), device="meta")
        request = ControlRequest(anchor_block=harness.plan.blocks[0], max_depth=1)

        with self.assertRaisesRegex(RuntimeError, "freqs device"):
            harness.backend.propose_anchor_and_drafts(request, harness.context)


class SelfForcingWanBackendBinderTests(unittest.TestCase):
    def test_kv_range_binds_to_tensor_region_spec(self) -> None:
        harness = Harness()
        specs = harness.backend.temporary_state_specs("prepare", None, harness.context)

        region = next(item for item in specs.tensor_regions if item.name == "kv_cache.layer_000.k")
        self.assertIs(region.tensor, harness.context.kv_cache[0]["k"])
        self.assertEqual((region.dim, region.start, region.end), (1, 0, 8))

    def test_index_values_bind_to_actual_scalar_tensors(self) -> None:
        harness = Harness()
        specs = harness.backend.temporary_state_specs("prepare", None, harness.context)

        self.assertTrue(
            any(tensor is harness.context.kv_cache[0]["global_end_index"] for tensor in specs.tensor_values)
        )
        self.assertTrue(
            any(tensor is harness.context.kv_cache[0]["local_end_index"] for tensor in specs.tensor_values)
        )

    def test_crossattn_is_init_binds_to_object_state_spec(self) -> None:
        harness = Harness()
        specs = harness.backend.prepare_persistent_state_specs(harness.context)

        names = [item.name for item in specs.object_states]
        self.assertIn("crossattn_cache.layer_000.is_init", names)

    def test_object_state_restore_only_restores_is_init(self) -> None:
        harness = Harness()
        specs = harness.backend.prepare_persistent_state_specs(harness.context)
        layer = harness.context.cross_attention_cache[0]
        manager = RuntimeStateTransactionManager(object_states=(specs.object_states[0],))
        original_k = layer["k"].clone()

        with manager.transaction():
            layer["is_init"] = True
            layer["k"].fill_(9.0)

        self.assertFalse(layer["is_init"])
        self.assertFalse(torch.equal(layer["k"], original_k))

    def test_output_ranges_and_runtime_bookkeeping_are_not_bound(self) -> None:
        harness = Harness()
        plan = WanStateMutationPlan.from_parts(
            output_ranges=(TensorRangeDescriptor("output", 1, 0, 2),)
        )

        specs = harness.backend._state_specs_from_plan(plan, harness.context)

        self.assertEqual(specs.tensor_regions, ())
        self.assertEqual(specs.tensor_values, ())
        self.assertEqual(specs.object_states, ())
        with self.assertRaises(ValueError):
            harness.backend._bind_python_value(
                "self_forcing_runtime_committed_blocks",
                harness.context,
            )

    def test_unknown_state_name_is_rejected(self) -> None:
        harness = Harness()
        plan = WanStateMutationPlan.from_parts(
            backend_tensor_ranges=(TensorRangeDescriptor("unknown.layer_000.k", 1, 0, 1),)
        )

        with self.assertRaises(ValueError):
            harness.backend._state_specs_from_plan(plan, harness.context)

    def test_missing_layer_or_key_is_rejected(self) -> None:
        harness = Harness()
        with self.assertRaises(IndexError):
            harness.backend._bind_tensor_value(
                "kv_cache.layer_999.global_end_index",
                harness.context,
            )

        del harness.context.kv_cache[0]["k"]
        plan = WanStateMutationPlan.from_parts(
            backend_tensor_ranges=(TensorRangeDescriptor("kv_cache.layer_000.k", 1, 0, 1),)
        )
        with self.assertRaises(KeyError):
            harness.backend._state_specs_from_plan(plan, harness.context)

    def test_wrong_cache_and_scalar_types_are_rejected(self) -> None:
        harness = Harness()
        bad_context = SelfForcingMCPRuntimeContext(
            config=harness.config,
            rollout_plan=harness.plan,
            source_noise=harness.source_noise,
            output=harness.output,
            kv_cache=tuple(harness.context.kv_cache),
            cross_attention_cache=harness.context.cross_attention_cache,
        )
        with self.assertRaises(TypeError):
            harness.backend._bind_tensor_value(
                "kv_cache.layer_000.global_end_index",
                bad_context,
            )

        harness.context.kv_cache[0]["global_end_index"] = torch.zeros((2,), dtype=torch.long)
        with self.assertRaises(ValueError):
            harness.backend._bind_tensor_value(
                "kv_cache.layer_000.global_end_index",
                harness.context,
            )

    def test_wrong_is_init_type_is_rejected(self) -> None:
        harness = Harness()
        harness.context.cross_attention_cache[0]["is_init"] = torch.tensor([False])

        with self.assertRaises(TypeError):
            harness.backend._bind_python_value(
                "crossattn_cache.layer_000.is_init",
                harness.context,
            )


class SelfForcingWanBackendPrepareTests(unittest.TestCase):
    def test_prepare_uses_staging_and_publishes_after_validation(self) -> None:
        harness = Harness()
        live_cache = harness.context.cross_attention_cache
        live_dict_ids = [id(layer) for layer in live_cache]
        live_k_ids = [id(layer["k"]) for layer in live_cache]
        live_v_ids = [id(layer["v"]) for layer in live_cache]

        harness.backend.prepare_cross_attention(harness.context)

        self.assertEqual(len(harness.pipeline.init_calls), 1)
        call = harness.pipeline.generator.calls[-1]
        self.assertIs(call["kv_cache"], harness.context.kv_cache)
        self.assertIsNot(call["crossattn_cache"], live_cache)
        self.assertEqual(
            call["noisy_image_or_video"].untyped_storage().data_ptr(),
            harness.source_noise.untyped_storage().data_ptr(),
        )
        self.assertEqual(call["current_start"], 0)
        self.assertNotIn("mcp_future_noises", call)
        self.assertIs(harness.pipeline.crossattn_cache, live_cache)
        self.assertEqual([id(layer) for layer in live_cache], live_dict_ids)
        self.assertEqual([id(layer["k"]) for layer in live_cache], live_k_ids)
        self.assertEqual([id(layer["v"]) for layer in live_cache], live_v_ids)
        self.assertTrue(all(layer["is_init"] for layer in live_cache))
        self.assertTrue(torch.equal(live_cache[0]["k"], torch.full_like(live_cache[0]["k"], 100.0)))
        self.assertTrue(torch.equal(live_cache[1]["v"], torch.full_like(live_cache[1]["v"], 201.0)))

    def test_invalid_later_staging_layer_publishes_nothing(self) -> None:
        harness = Harness()
        live_cache = harness.context.cross_attention_cache
        expected = [(layer["k"].clone(), layer["v"].clone(), layer["is_init"]) for layer in live_cache]
        harness.pipeline.generator.invalid_staging_layer = 1

        with self.assertRaises(RuntimeError):
            harness.backend.prepare_cross_attention(harness.context)

        for layer, (expected_k, expected_v, expected_is_init) in zip(live_cache, expected):
            self.assertTrue(torch.equal(layer["k"], expected_k))
            self.assertTrue(torch.equal(layer["v"], expected_v))
            self.assertEqual(layer["is_init"], expected_is_init)

    def test_prepare_restores_pipeline_crossattn_on_exception(self) -> None:
        harness = Harness()
        live_cache = harness.context.cross_attention_cache
        harness.pipeline.generator.raise_on_call = RuntimeError("prepare failed")

        with self.assertRaisesRegex(RuntimeError, "prepare failed"):
            harness.backend.prepare_cross_attention(harness.context)

        self.assertIs(harness.pipeline.crossattn_cache, live_cache)


class SelfForcingWanBackendProposalTests(unittest.TestCase):
    def test_proposal_calls_future_helper_and_returns_anchor_and_drafts(self) -> None:
        harness = Harness()
        request = ControlRequest(anchor_block=harness.plan.blocks[0], max_depth=2)

        batch = harness.backend.propose_anchor_and_drafts(request, harness.context)

        self.assertEqual(harness.pipeline.future_calls[0]["block_index"], 0)
        self.assertIs(harness.pipeline.future_calls[0]["noise"], harness.source_noise)
        call = harness.pipeline.generator.calls[-1]
        self.assertEqual(call["current_start"], 0)
        self.assertEqual(call["timestep"].dtype, torch.int64)
        self.assertTrue(torch.equal(call["timestep"], torch.full((1, 2), 1000, dtype=torch.int64)))
        self.assertEqual(tuple(call["mcp_future_start_frames"]), (2, 4))
        self.assertIs(call["mcp_future_noises"][0], harness.pipeline.last_future_noises[0])
        self.assertIs(call["mcp_future_noises"][1], harness.pipeline.last_future_noises[1])
        self.assertEqual(batch.anchor.source, "anchor")
        self.assertTrue(torch.equal(batch.anchor.latent, harness.source_noise[:, 0:2] + 1.0))
        self.assertEqual([draft.depth for draft in batch.drafts], [1, 2])
        self.assertIs(batch.drafts[0].source_noise, harness.pipeline.last_future_noises[0])
        self.assertTrue(
            torch.equal(
                batch.drafts[0].latent,
                batch.drafts[0].source_noise - torch.ones_like(batch.drafts[0].source_noise),
            )
        )
        self.assertTrue(torch.equal(harness.output, torch.zeros_like(harness.output)))
        self.assertEqual(harness.pipeline.commit_calls, [])
        self.assertEqual(len(harness.pipeline.generator.calls), 1)

    def test_request_max_depth_truncates_mcp_arguments(self) -> None:
        harness = Harness(mcp_depth=3, pipeline=FakePipeline(mcp_modules=3))
        request = ControlRequest(anchor_block=harness.plan.blocks[0], max_depth=1)

        batch = harness.backend.propose_anchor_and_drafts(request, harness.context)

        call = harness.pipeline.generator.calls[-1]
        self.assertEqual(len(call["mcp_future_noises"]), 1)
        self.assertEqual(len(batch.drafts), 1)

    def test_first_none_stops_draft_chain(self) -> None:
        harness = Harness(mcp_depth=3, pipeline=FakePipeline(mcp_modules=3))
        noise1 = harness.source_noise[:, 2:4]
        noise3 = harness.source_noise[:, 4:6]
        harness.pipeline.future_override = ([noise1, None, noise3], [2, None, 4])
        request = ControlRequest(anchor_block=harness.plan.blocks[0], max_depth=3)

        batch = harness.backend.propose_anchor_and_drafts(request, harness.context)

        call = harness.pipeline.generator.calls[-1]
        self.assertEqual(len(call["mcp_future_noises"]), 1)
        self.assertEqual([draft.depth for draft in batch.drafts], [1])

    def test_wrong_future_start_is_rejected(self) -> None:
        harness = Harness()
        noise1 = harness.source_noise[:, 2:4]
        harness.pipeline.future_override = ([noise1, None], [4, None])
        request = ControlRequest(anchor_block=harness.plan.blocks[0], max_depth=2)

        with self.assertRaisesRegex(RuntimeError, "future start frame"):
            harness.backend.propose_anchor_and_drafts(request, harness.context)

    def test_bool_or_float_future_start_is_rejected(self) -> None:
        for start in (True, 2.0):
            with self.subTest(start=start):
                harness = Harness()
                noise1 = harness.source_noise[:, 2:4]
                harness.pipeline.future_override = ([noise1, None], [start, None])
                request = ControlRequest(anchor_block=harness.plan.blocks[0], max_depth=2)

                with self.assertRaisesRegex(TypeError, "strict int"):
                    harness.backend.propose_anchor_and_drafts(request, harness.context)

    def test_none_noise_with_non_none_start_is_rejected(self) -> None:
        harness = Harness()
        harness.pipeline.future_override = ([None, None], [2, None])
        request = ControlRequest(anchor_block=harness.plan.blocks[0], max_depth=2)

        with self.assertRaisesRegex(RuntimeError, "None noise"):
            harness.backend.propose_anchor_and_drafts(request, harness.context)

    def test_no_future_chunk_omits_mcp_arguments(self) -> None:
        harness = Harness(total_frames=5, block_frames=2, mcp_depth=2)
        request = ControlRequest(anchor_block=harness.plan.blocks[1], max_depth=2)

        batch = harness.backend.propose_anchor_and_drafts(request, harness.context)

        call = harness.pipeline.generator.calls[-1]
        self.assertNotIn("mcp_future_noises", call)
        self.assertEqual(batch.drafts, ())

    def test_final_short_mismatched_block_does_not_create_invalid_draft(self) -> None:
        harness = Harness(total_frames=5, block_frames=2, mcp_depth=2)
        request = ControlRequest(anchor_block=harness.plan.blocks[0], max_depth=2)

        batch = harness.backend.propose_anchor_and_drafts(request, harness.context)

        self.assertEqual([draft.block.index for draft in batch.drafts], [1])

    def test_proposal_rejects_mcp_flow_count_or_shape_mismatch(self) -> None:
        harness = Harness()

        def bad_call(**kwargs: Any) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
            noisy = kwargs["noisy_image_or_video"]
            return torch.zeros_like(noisy), noisy + 1, [torch.zeros_like(noisy)]

        harness.pipeline.generator.result_override = bad_call
        request = ControlRequest(anchor_block=harness.plan.blocks[0], max_depth=2)

        with self.assertRaises(RuntimeError):
            harness.backend.propose_anchor_and_drafts(request, harness.context)


class SelfForcingWanBackendFallbackTests(unittest.TestCase):
    def test_fallback_uses_candidate_source_noise_identity(self) -> None:
        harness = Harness()
        block = harness.plan.blocks[1]
        source_noise = harness.source_noise[:, 2:4]
        candidate = DraftCandidate(
            block=block,
            depth=1,
            latent=torch.zeros_like(source_noise),
            source_noise=source_noise,
        )

        result = harness.backend.generate_target_fallback(candidate, harness.context)

        call = harness.pipeline.generator.calls[-1]
        self.assertIs(call["noisy_image_or_video"], source_noise)
        self.assertEqual(call["current_start"], 8)
        self.assertNotIn("mcp_future_noises", call)
        self.assertIs(result.source_noise, source_noise)
        self.assertEqual(result.block, block)
        self.assertTrue(torch.equal(result.latent, source_noise + 1.0))
        self.assertTrue(torch.equal(harness.output, torch.zeros_like(harness.output)))
        self.assertEqual(harness.pipeline.commit_calls, [])


class SelfForcingWanBackendCommitTests(unittest.TestCase):
    def test_commit_delegates_once_and_validates_end_indices(self) -> None:
        harness = Harness()
        block = harness.plan.blocks[0]
        latent = torch.full_like(harness.source_noise[:, 0:2], 7.0)
        request = CommitRequest(block=block, latent=latent, source="anchor")

        harness.backend.commit_context_block(request, harness.context)

        self.assertEqual(len(harness.pipeline.commit_calls), 1)
        self.assertIs(harness.pipeline.commit_calls[0]["latent"], latent)
        self.assertEqual(harness.pipeline.commit_calls[0]["start_frame"], 0)
        for layer in harness.context.kv_cache:
            self.assertEqual(int(layer["global_end_index"].item()), 8)
            self.assertEqual(int(layer["local_end_index"].item()), 8)
        self.assertTrue(torch.equal(harness.output, torch.zeros_like(harness.output)))

    def test_commit_checks_cache_start_before_helper(self) -> None:
        harness = Harness()
        block = harness.plan.blocks[1]
        latent = torch.full_like(harness.source_noise[:, 2:4], 7.0)
        request = CommitRequest(block=block, latent=latent, source="anchor")

        with self.assertRaisesRegex(RuntimeError, "cache start"):
            harness.backend.commit_context_block(request, harness.context)

        self.assertEqual(harness.pipeline.commit_calls, [])

    def test_commit_postcondition_failure_raises(self) -> None:
        harness = Harness()
        harness.pipeline.bad_commit_end = True
        block = harness.plan.blocks[0]
        request = CommitRequest(
            block=block,
            latent=torch.full_like(harness.source_noise[:, 0:2], 7.0),
            source="anchor",
        )

        with self.assertRaisesRegex(RuntimeError, "local_end_index"):
            harness.backend.commit_context_block(request, harness.context)


class SelfForcingWanBackendPlannerMappingTests(unittest.TestCase):
    def test_spec_methods_use_expected_planner_operations(self) -> None:
        harness = Harness()
        prepare_persistent = harness.backend.prepare_persistent_state_specs(harness.context)
        prepare_temp = harness.backend.temporary_state_specs("prepare", None, harness.context)
        proposal = harness.backend.temporary_state_specs(
            "proposal",
            ControlRequest(anchor_block=harness.plan.blocks[0], max_depth=2),
            harness.context,
        )
        fallback = harness.backend.temporary_state_specs(
            "fallback",
            DraftCandidate(
                block=harness.plan.blocks[1],
                depth=1,
                latent=torch.zeros_like(harness.source_noise[:, 2:4]),
                source_noise=harness.source_noise[:, 2:4],
            ),
            harness.context,
        )

        self.assertTrue(any(region.name.startswith("crossattn_cache") for region in prepare_persistent.tensor_regions))
        self.assertTrue(any(region.name.startswith("kv_cache") for region in prepare_temp.tensor_regions))
        self.assertTrue(any(region.name.startswith("kv_cache") for region in proposal.tensor_regions))
        self.assertTrue(any(region.name.startswith("kv_cache") for region in fallback.tensor_regions))
        self.assertFalse(any(region.tensor is harness.output for region in proposal.tensor_regions))
        self.assertFalse(any(region.tensor is harness.output for region in fallback.tensor_regions))

    def test_window_specs_use_actual_allowed_blocks_and_preserve_rng_flags(self) -> None:
        harness = Harness()
        window = RuntimeWindowDescriptor(
            anchor_block=harness.plan.blocks[0],
            allowed_blocks=(harness.plan.blocks[0], harness.plan.blocks[1]),
        )

        specs = harness.backend.window_state_specs(window, harness.context)

        region = next(item for item in specs.tensor_regions if item.name == "kv_cache.layer_000.k")
        self.assertEqual((region.start, region.end), (0, 16))
        self.assertTrue(specs.capture_rng)
        self.assertFalse(specs.capture_cuda_rng)
        self.assertFalse(any(region.tensor is harness.output for region in specs.tensor_regions))

    def test_unknown_operation_fails_fast(self) -> None:
        harness = Harness()

        with self.assertRaises(ValueError):
            harness.backend.temporary_state_specs("commit", None, harness.context)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
