from __future__ import annotations

import unittest
from dataclasses import FrozenInstanceError, fields

from speculative.adapters.self_forcing_runtime import RuntimeWindowDescriptor
from speculative.adapters.wan_state_planner import (
    SUPPORTED_CROSSATTN_CONTAINER,
    SUPPORTED_KV_CONTAINER,
    TensorRangeDescriptor,
    WanCacheLayout,
    WanStateMutationPlan,
    WanTouchedRangePlanner,
)
from speculative.types import BlockRef, CommitRequest, ControlRequest, DraftCandidate


FRAME_SEQ_LENGTH = 10
BLOCK_FRAMES = 3


def block(index: int, *, frames: int = BLOCK_FRAMES) -> BlockRef:
    return BlockRef(
        index=index,
        start_frame=index * BLOCK_FRAMES,
        num_frames=frames,
    )


def planner(
    *,
    capacity: int = 240,
    layers: int = 2,
    mcp_depth: int = 3,
    local_attn_size: int = -1,
    sink_size: int = 0,
    runtime_device_type: str = "cuda",
    batch_size: int = 1,
    cfg_batch_multiplier: int = 1,
    kv_container: str = SUPPORTED_KV_CONTAINER,
    crossattn_container: str = SUPPORTED_CROSSATTN_CONTAINER,
    latent_layout: str = "B,F,C,H,W",
) -> WanTouchedRangePlanner:
    layout = WanCacheLayout(
        num_layers=layers,
        cache_capacity=capacity,
        kv_container=kv_container,
        crossattn_container=crossattn_container,
        batch_size=batch_size,
        cfg_batch_multiplier=cfg_batch_multiplier,
        latent_layout=latent_layout,
    )
    return WanTouchedRangePlanner(
        frame_seq_length=FRAME_SEQ_LENGTH,
        num_frame_per_block=BLOCK_FRAMES,
        mcp_depth=mcp_depth,
        cache_layout=layout,
        local_attn_size=local_attn_size,
        sink_size=sink_size,
        runtime_device_type=runtime_device_type,  # type: ignore[arg-type]
    )


def range_for(plan: WanStateMutationPlan, state_name: str) -> TensorRangeDescriptor:
    matches = [item for item in plan.backend_tensor_ranges if item.state_name == state_name]
    if len(matches) != 1:
        raise AssertionError(f"expected one range for {state_name}, got {matches}")
    return matches[0]


def output_range_for(plan: WanStateMutationPlan) -> TensorRangeDescriptor:
    matches = [item for item in plan.output_ranges if item.state_name == "output"]
    if len(matches) != 1:
        raise AssertionError(f"expected one output range, got {matches}")
    return matches[0]


class WanStatePlannerTests(unittest.TestCase):
    def test_current_start_is_derived_from_block_ref(self) -> None:
        self.assertEqual(planner().current_start_for(block(2)), 60)

    def test_planner_has_no_mutable_frame_cursor(self) -> None:
        uut = planner()
        self.assertFalse(hasattr(uut, "current_start"))
        self.assertFalse(hasattr(uut, "current_start_frame"))
        self.assertFalse(hasattr(uut, "__dict__"))

    def test_single_complete_block_range(self) -> None:
        plan = planner(layers=1).plan_commit(block(0))
        self.assertEqual(range_for(plan, "kv_cache.layer_000.k").start, 0)
        self.assertEqual(range_for(plan, "kv_cache.layer_000.k").end, 30)
        self.assertEqual(range_for(plan, "kv_cache.layer_000.v").end, 30)
        self.assertEqual(plan.operation_ranges[0].global_end_index, 30)
        self.assertEqual(plan.operation_ranges[0].local_end_index, 30)
        self.assertEqual(output_range_for(plan), TensorRangeDescriptor("output", 1, 0, 3))

    def test_final_short_block_range(self) -> None:
        short = BlockRef(index=2, start_frame=6, num_frames=1)
        plan = planner(layers=1).plan_commit(short)
        self.assertEqual(range_for(plan, "kv_cache.layer_000.k"), TensorRangeDescriptor("kv_cache.layer_000.k", 1, 60, 70))
        self.assertEqual(output_range_for(plan), TensorRangeDescriptor("output", 1, 6, 7))

    def test_mcp_depth_zero_allows_anchor_only(self) -> None:
        uut = planner(mcp_depth=0)
        blocks = tuple(block(i) for i in range(5))
        request = ControlRequest(anchor_block=blocks[0], max_depth=3)
        self.assertEqual(uut.allowed_blocks_for_request(request, blocks), (blocks[0],))

    def test_mcp_depth_one_allows_one_draft(self) -> None:
        uut = planner(mcp_depth=1)
        blocks = tuple(block(i) for i in range(5))
        request = ControlRequest(anchor_block=blocks[0], max_depth=3)
        self.assertEqual(uut.allowed_blocks_for_request(request, blocks), blocks[:2])

    def test_mcp_depth_three_allows_three_drafts(self) -> None:
        uut = planner(mcp_depth=3)
        blocks = tuple(block(i) for i in range(5))
        request = ControlRequest(anchor_block=blocks[0], max_depth=3)
        self.assertEqual(uut.allowed_blocks_for_request(request, blocks), blocks[:4])

    def test_actual_proposal_blocks_can_be_less_than_max_depth(self) -> None:
        uut = planner(mcp_depth=3)
        blocks = tuple(block(i) for i in range(2))
        request = ControlRequest(anchor_block=blocks[0], max_depth=3)
        self.assertEqual(uut.allowed_blocks_for_request(request, blocks), blocks)

    def test_always_accept_second_window_uses_next_uncommitted_block(self) -> None:
        uut = planner(capacity=300)
        second_anchor = block(4)
        plan = uut.plan_proposal(ControlRequest(anchor_block=second_anchor, max_depth=3))
        self.assertEqual(plan.operation_ranges[0].current_start, 120)
        self.assertEqual(plan.operation_ranges[0].global_start, 120)

    def test_always_reject_continues_from_next_uncommitted_block(self) -> None:
        uut = planner()
        blocks = tuple(block(i) for i in range(6))
        request = ControlRequest(anchor_block=blocks[2], max_depth=3)
        self.assertEqual(uut.allowed_blocks_for_request(request, blocks), blocks[2:6])

    def test_fallback_only_covers_rejected_block(self) -> None:
        rejected = DraftCandidate(
            block=block(2),
            depth=2,
            latent=object(),
            source_noise=object(),
        )
        plan = planner(layers=1).plan_fallback(rejected)
        self.assertEqual(output_range_for(plan), TensorRangeDescriptor("output", 1, 6, 9))
        self.assertEqual(len(plan.operation_ranges), 1)
        self.assertEqual(plan.operation_ranges[0].block_index, 2)

    def test_commit_only_covers_submitted_block(self) -> None:
        request = CommitRequest(block=block(1), latent=object(), source="anchor")
        plan = planner(layers=1).plan_commit(request)
        self.assertEqual(output_range_for(plan), TensorRangeDescriptor("output", 1, 3, 6))
        self.assertEqual(range_for(plan, "kv_cache.layer_000.k"), TensorRangeDescriptor("kv_cache.layer_000.k", 1, 30, 60))

    def test_window_plan_uses_actual_allowed_blocks(self) -> None:
        blocks = tuple(block(i) for i in range(2))
        window = RuntimeWindowDescriptor(anchor_block=blocks[0], allowed_blocks=blocks)
        plan = planner(layers=1).plan_window(window)
        self.assertEqual(output_range_for(plan), TensorRangeDescriptor("output", 1, 0, 6))
        self.assertEqual(range_for(plan, "kv_cache.layer_000.k"), TensorRangeDescriptor("kv_cache.layer_000.k", 1, 0, 60))

    def test_global_cache_append_range_is_reported(self) -> None:
        plan = planner(capacity=300).plan_commit(block(2))
        operation_range = plan.operation_ranges[0]
        self.assertEqual((operation_range.global_start, operation_range.global_end), (60, 90))

    def test_local_cache_normal_append_range(self) -> None:
        plan = planner(capacity=100, local_attn_size=2).plan_commit(block(1))
        self.assertFalse(plan.operation_ranges[0].rolls_local_cache)
        self.assertEqual(plan.operation_ranges[0].local_ranges[0].start, 30)
        self.assertEqual(plan.operation_ranges[0].local_ranges[0].end, 60)

    def test_local_cache_rollover_boundary_does_not_roll(self) -> None:
        plan = planner(capacity=60, local_attn_size=2).plan_commit(block(1))
        self.assertFalse(plan.operation_ranges[0].rolls_local_cache)
        self.assertEqual(plan.operation_ranges[0].local_ranges[0].end, 60)

    def test_local_cache_cross_rollover_reports_segments(self) -> None:
        plan = planner(capacity=50, local_attn_size=2, layers=1).plan_commit(block(1))
        self.assertTrue(plan.operation_ranges[0].rolls_local_cache)
        self.assertEqual(
            plan.operation_ranges[0].local_ranges,
            (
                type(plan.operation_ranges[0].local_ranges[0])(0, 20),
                type(plan.operation_ranges[0].local_ranges[0])(20, 50),
            ),
        )
        self.assertEqual(range_for(plan, "kv_cache.layer_000.k"), TensorRangeDescriptor("kv_cache.layer_000.k", 1, 0, 50))

    def test_global_and_local_index_mutation_descriptors(self) -> None:
        plan = planner(layers=2).plan_commit(block(0))
        self.assertIn("kv_cache.layer_000.global_end_index", plan.backend_tensor_value_names)
        self.assertIn("kv_cache.layer_000.local_end_index", plan.backend_tensor_value_names)
        self.assertIn("kv_cache.layer_001.global_end_index", plan.backend_tensor_value_names)
        self.assertIn("kv_cache.layer_001.local_end_index", plan.backend_tensor_value_names)

    def test_prepare_cross_attention_descriptor(self) -> None:
        plan = planner(layers=1).plan_prepare_cross_attention()
        self.assertEqual(
            range_for(plan, "crossattn_cache.layer_000.k"),
            TensorRangeDescriptor("crossattn_cache.layer_000.k", 1, 0, 512),
        )
        self.assertEqual(
            range_for(plan, "crossattn_cache.layer_000.v"),
            TensorRangeDescriptor("crossattn_cache.layer_000.v", 1, 0, 512),
        )
        self.assertIn("crossattn_cache.layer_000.is_init", plan.backend_python_value_names)
        self.assertEqual(plan.backend_tensor_value_names, ())
        self.assertEqual(plan.output_ranges, ())

    def test_prepare_plan_has_no_duplicate_state_semantics(self) -> None:
        plan = planner(layers=1).plan_prepare_cross_attention()
        tensor_names = {item.state_name for item in plan.backend_tensor_ranges}
        value_names = set(plan.backend_tensor_value_names) | set(
            plan.backend_python_value_names
        )
        self.assertFalse(tensor_names & value_names)
        with self.assertRaisesRegex(ValueError, "both tensor range and value"):
            WanStateMutationPlan.from_parts(
                backend_tensor_ranges=(TensorRangeDescriptor("x", 0, 0, 1),),
                backend_python_value_names=("x",),
            )

    def test_prepare_kv_are_backend_tensor_ranges_only(self) -> None:
        plan = planner(layers=1).plan_prepare_cross_attention()
        tensor_names = {item.state_name for item in plan.backend_tensor_ranges}
        value_names = set(plan.backend_tensor_value_names) | set(
            plan.backend_python_value_names
        )
        self.assertIn("crossattn_cache.layer_000.k", tensor_names)
        self.assertIn("crossattn_cache.layer_000.v", tensor_names)
        self.assertNotIn("crossattn_cache.layer_000.k", value_names)
        self.assertNotIn("crossattn_cache.layer_000.v", value_names)

    def test_prepare_is_init_is_python_value_only(self) -> None:
        plan = planner(layers=1).plan_prepare_cross_attention()
        tensor_names = {item.state_name for item in plan.backend_tensor_ranges}
        self.assertIn("crossattn_cache.layer_000.is_init", plan.backend_python_value_names)
        self.assertNotIn("crossattn_cache.layer_000.is_init", tensor_names)
        self.assertNotIn(
            "crossattn_cache.layer_000.is_init",
            plan.backend_tensor_value_names,
        )

    def test_prepare_scratch_covers_block_zero_kv_and_indices(self) -> None:
        plan = planner(layers=1).plan_prepare_scratch(block(0))
        self.assertEqual(plan.operation_ranges[0].operation, "prepare_scratch")
        self.assertEqual(
            range_for(plan, "kv_cache.layer_000.k"),
            TensorRangeDescriptor("kv_cache.layer_000.k", 1, 0, 30),
        )
        self.assertEqual(
            range_for(plan, "kv_cache.layer_000.v"),
            TensorRangeDescriptor("kv_cache.layer_000.v", 1, 0, 30),
        )
        self.assertIn("kv_cache.layer_000.global_end_index", plan.backend_tensor_value_names)
        self.assertIn("kv_cache.layer_000.local_end_index", plan.backend_tensor_value_names)
        self.assertFalse(plan.capture_rng)
        self.assertFalse(plan.capture_cuda_rng)

    def test_prepare_scratch_has_no_output_range(self) -> None:
        plan = planner(layers=1).plan_prepare_scratch(block(0))
        self.assertEqual(plan.output_ranges, ())

    def test_prepare_scratch_rejects_nonzero_first_block(self) -> None:
        with self.assertRaisesRegex(ValueError, "index 0"):
            planner().plan_prepare_scratch(block(1))
        with self.assertRaisesRegex(ValueError, "start_frame == 0"):
            planner().plan_prepare_scratch(
                BlockRef(index=0, start_frame=3, num_frames=BLOCK_FRAMES)
            )

    def test_proposal_backend_ranges_exclude_output(self) -> None:
        blocks = (block(0), block(1))
        request = ControlRequest(anchor_block=blocks[0], max_depth=1)
        plan = planner(layers=1).plan_proposal(request, allowed_blocks=blocks)
        self.assertNotIn(
            "output",
            {item.state_name for item in plan.backend_tensor_ranges},
        )
        self.assertEqual(output_range_for(plan), TensorRangeDescriptor("output", 1, 0, 6))

    def test_fallback_backend_ranges_exclude_output(self) -> None:
        plan = planner(layers=1).plan_fallback(block(1))
        self.assertNotIn(
            "output",
            {item.state_name for item in plan.backend_tensor_ranges},
        )
        self.assertEqual(output_range_for(plan), TensorRangeDescriptor("output", 1, 3, 6))

    def test_commit_backend_ranges_exclude_output(self) -> None:
        plan = planner(layers=1).plan_commit(block(1))
        self.assertNotIn(
            "output",
            {item.state_name for item in plan.backend_tensor_ranges},
        )
        self.assertEqual(output_range_for(plan), TensorRangeDescriptor("output", 1, 3, 6))

    def test_window_backend_ranges_exclude_output(self) -> None:
        blocks = (block(0), block(1))
        window = RuntimeWindowDescriptor(anchor_block=blocks[0], allowed_blocks=blocks)
        plan = planner(layers=1).plan_window(window)
        self.assertNotIn(
            "output",
            {item.state_name for item in plan.backend_tensor_ranges},
        )
        self.assertEqual(output_range_for(plan), TensorRangeDescriptor("output", 1, 0, 6))

    def test_output_ranges_remain_separately_observable(self) -> None:
        plan = planner(layers=1).plan_commit(block(0))
        self.assertEqual(plan.output_ranges, (TensorRangeDescriptor("output", 1, 0, 3),))
        self.assertNotEqual(plan.backend_tensor_ranges, plan.output_ranges)

    def test_window_plan_excludes_runtime_committed_bookkeeping(self) -> None:
        blocks = (block(0), block(1))
        window = RuntimeWindowDescriptor(anchor_block=blocks[0], allowed_blocks=blocks)
        plan = planner(layers=1).plan_window(window)
        forbidden = "self_forcing_runtime_committed_blocks"
        self.assertNotIn(forbidden, {item.state_name for item in plan.backend_tensor_ranges})
        self.assertNotIn(forbidden, plan.backend_tensor_value_names)
        self.assertNotIn(forbidden, plan.backend_python_value_names)

    def test_cpu_rng_capture_flag_for_commit(self) -> None:
        plan = planner(runtime_device_type="cpu").plan_commit(block(0))
        self.assertTrue(plan.capture_rng)
        self.assertFalse(plan.capture_cuda_rng)

    def test_cuda_rng_capture_flag_for_commit(self) -> None:
        plan = planner(runtime_device_type="cuda").plan_commit(block(0))
        self.assertFalse(plan.capture_rng)
        self.assertTrue(plan.capture_cuda_rng)

    def test_unsupported_attention_mode_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported attention mode"):
            WanCacheLayout(
                num_layers=1,
                cache_capacity=100,
                attention_mode="bidirectional",
            )

    def test_unsupported_cache_layout_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported KV cache container"):
            planner(kv_container="tuple[k,v]")

    def test_cache_capacity_overflow_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "exceeds capacity"):
            planner(capacity=20).plan_commit(block(0))

    def test_local_attn_size_zero_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "local_attn_size=0"):
            planner(local_attn_size=0)

    def test_global_attention_with_nonzero_sink_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "sink_size > 0"):
            planner(local_attn_size=-1, sink_size=1)

    def test_oversized_block_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be <="):
            planner().plan_commit(
                BlockRef(index=0, start_frame=0, num_frames=BLOCK_FRAMES + 1)
            )

    def test_explicit_proposal_blocks_exceeding_request_depth_rejected(self) -> None:
        blocks = (block(0), block(1), block(2))
        request = ControlRequest(anchor_block=blocks[0], max_depth=1)
        with self.assertRaisesRegex(ValueError, "request.max_depth"):
            planner(mcp_depth=3).plan_proposal(request, allowed_blocks=blocks)

    def test_explicit_proposal_blocks_exceeding_mcp_depth_rejected(self) -> None:
        blocks = (block(0), block(1), block(2))
        request = ControlRequest(anchor_block=blocks[0], max_depth=3)
        with self.assertRaisesRegex(ValueError, "planner mcp_depth"):
            planner(mcp_depth=1).plan_proposal(request, allowed_blocks=blocks)

    def test_negative_tensor_dimension_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "dimension must be >= 0"):
            TensorRangeDescriptor("x", -1, 0, 1)

    def test_bool_float_and_string_integer_inputs_are_rejected(self) -> None:
        bad_cases = (
            lambda: WanCacheLayout(num_layers=True, cache_capacity=100),  # type: ignore[arg-type]
            lambda: WanTouchedRangePlanner(
                frame_seq_length=10.0,  # type: ignore[arg-type]
                num_frame_per_block=3,
                mcp_depth=1,
                cache_layout=WanCacheLayout(num_layers=1, cache_capacity=100),
            ),
            lambda: TensorRangeDescriptor("x", "1", 0, 1),  # type: ignore[arg-type]
        )
        for case in bad_cases:
            with self.subTest(case=case):
                with self.assertRaises(ValueError):
                    case()

    def test_descriptors_do_not_store_mutable_cache_model_or_runtime(self) -> None:
        plan = planner(layers=1).plan_commit(block(0))
        field_names = {item.name for item in fields(plan)}
        self.assertEqual(
            field_names,
            {
                "backend_tensor_ranges",
                "backend_tensor_value_names",
                "backend_python_value_names",
                "output_ranges",
                "capture_rng",
                "capture_cuda_rng",
                "operation_ranges",
            },
        )
        self.assertTrue(all(not isinstance(value, dict) for value in plan.backend_tensor_ranges))

    def test_outputs_are_immutable_tuple_dataclasses(self) -> None:
        plan = planner(layers=1).plan_commit(block(0))
        self.assertIsInstance(plan.backend_tensor_ranges, tuple)
        self.assertIsInstance(plan.backend_tensor_value_names, tuple)
        self.assertIsInstance(plan.backend_python_value_names, tuple)
        self.assertIsInstance(plan.output_ranges, tuple)
        with self.assertRaises(FrozenInstanceError):
            plan.capture_rng = True  # type: ignore[misc]
        with self.assertRaises(FrozenInstanceError):
            plan.backend_tensor_ranges[0].start = 99  # type: ignore[misc]

    def test_range_sorting_is_stable(self) -> None:
        plan = WanStateMutationPlan.from_parts(
            backend_tensor_ranges=(
                TensorRangeDescriptor("z", 1, 4, 5),
                TensorRangeDescriptor("a", 1, 2, 3),
                TensorRangeDescriptor("a", 1, 0, 1),
            )
        )
        self.assertEqual(
            plan.backend_tensor_ranges,
            (
                TensorRangeDescriptor("a", 1, 0, 1),
                TensorRangeDescriptor("a", 1, 2, 3),
                TensorRangeDescriptor("z", 1, 4, 5),
            ),
        )

    def test_same_tensor_adjacent_ranges_are_merged(self) -> None:
        blocks = (block(0), block(1))
        window = RuntimeWindowDescriptor(anchor_block=blocks[0], allowed_blocks=blocks)
        plan = planner(layers=1).plan_window(window)
        self.assertEqual(range_for(plan, "kv_cache.layer_000.k"), TensorRangeDescriptor("kv_cache.layer_000.k", 1, 0, 60))

    def test_different_tensor_ranges_are_not_merged(self) -> None:
        plan = planner(layers=1).plan_commit(block(0))
        names = {descriptor.state_name for descriptor in plan.backend_tensor_ranges}
        self.assertIn("kv_cache.layer_000.k", names)
        self.assertIn("kv_cache.layer_000.v", names)
        self.assertEqual(len([name for name in names if name.startswith("kv_cache")]), 2)

    def test_multi_layer_kv_ranges_are_independent(self) -> None:
        plan = planner(layers=2).plan_commit(block(0))
        names = {descriptor.state_name for descriptor in plan.backend_tensor_ranges}
        self.assertIn("kv_cache.layer_000.k", names)
        self.assertIn("kv_cache.layer_001.k", names)
        self.assertIn("kv_cache.layer_000.v", names)
        self.assertIn("kv_cache.layer_001.v", names)

    def test_batch_or_cfg_layout_outside_baseline_fails_fast(self) -> None:
        bad_cases = (
            lambda: planner(batch_size=2),
            lambda: planner(cfg_batch_multiplier=2),
            lambda: planner(latent_layout="B,2,F,C,H,W"),
        )
        for case in bad_cases:
            with self.subTest(case=case):
                with self.assertRaises(ValueError):
                    case()


if __name__ == "__main__":
    unittest.main()
