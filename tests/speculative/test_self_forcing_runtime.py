from __future__ import annotations

import unittest
from typing import Optional

import torch

from speculative.adapters.runtime_state import ObjectStateSpec
from speculative.adapters.self_forcing_mcp import (
    SelfForcingMCPCommitter,
    SelfForcingMCPFallbackGenerator,
    SelfForcingMCPProposalSource,
)
from speculative.adapters.self_forcing_runtime import (
    RuntimeStateSpecBundle,
    RuntimeWindowDescriptor,
    SelfForcingMCPRuntime,
    SelfForcingMCPRuntimeConfig,
    SelfForcingMCPRuntimeContext,
)
from speculative.controller import SpeculativeController
from speculative.factory import create_policy
from speculative.types import (
    BlockRef,
    CommitRequest,
    ControlRequest,
    DraftCandidate,
    Evaluation,
    FallbackResult,
    ProposalBatch,
)


VALIDATED_MODE = "validated-global-cache"


def make_config(
    *,
    total_frames: int = 7,
    block_frames: int = 3,
    frame_seq_length: int = 10,
    mcp_depth: int = 2,
    steps: tuple[int, ...] = (1000,),
    attention_mode: str = VALIDATED_MODE,
) -> SelfForcingMCPRuntimeConfig:
    return SelfForcingMCPRuntimeConfig(
        anchor_denoising_steps=steps,
        frame_seq_length=frame_seq_length,
        num_frame_per_block=block_frames,
        all_num_frames=total_frames,
        mcp_depth=mcp_depth,
        attention_mode=attention_mode,
        validated_attention_mode=VALIDATED_MODE,
    )


def block_latent(block: BlockRef, value: float) -> torch.Tensor:
    assert block.num_frames is not None
    return torch.full((1, block.num_frames, 1), value, dtype=torch.float32)


def assert_tensor_equal(test: unittest.TestCase, actual: torch.Tensor, expected: torch.Tensor) -> None:
    test.assertTrue(torch.equal(actual, expected), f"\nactual={actual}\nexpected={expected}")


def unsafe_proposal_batch(
    anchor: CommitRequest,
    drafts: tuple[DraftCandidate, ...],
) -> ProposalBatch:
    batch = object.__new__(ProposalBatch)
    object.__setattr__(batch, "anchor", anchor)
    object.__setattr__(batch, "drafts", drafts)
    return batch


class IdentityEvaluator:
    def __init__(self) -> None:
        self.calls: list[DraftCandidate] = []

    def evaluate(self, candidate: DraftCandidate) -> Evaluation:
        self.calls.append(candidate)
        return Evaluation(candidate=candidate, value={"depth": candidate.depth})


class FakeRuntimeBackend:
    def __init__(self) -> None:
        self.prepare_calls: list[SelfForcingMCPRuntimeContext] = []
        self.proposal_calls: list[tuple[ControlRequest, SelfForcingMCPRuntimeContext]] = []
        self.fallback_calls: list[tuple[DraftCandidate, SelfForcingMCPRuntimeContext]] = []
        self.commit_calls: list[tuple[CommitRequest, SelfForcingMCPRuntimeContext]] = []
        self.state_spec_calls: list[tuple[str, object]] = []
        self.temporary_spec_bundles: list[RuntimeStateSpecBundle] = []
        self.prepare_persistent_spec_bundles: list[RuntimeStateSpecBundle] = []
        self.window_state_spec_calls: list[RuntimeWindowDescriptor] = []
        self.window_spec_bundles: list[RuntimeStateSpecBundle] = []
        self.call_order: list[str] = []
        self.scratch_state: list[str] = []
        self.prepare_error: Optional[Exception] = None
        self.proposal_error: Optional[Exception] = None
        self.fallback_error: Optional[Exception] = None
        self.commit_error: Optional[Exception] = None
        self.fail_window_capture = False
        self.proposal_result: Optional[ProposalBatch] = None
        self.fallback_result: Optional[FallbackResult] = None
        self.current_starts: list[int] = []

    def prepare_cross_attention(self, runtime_context: SelfForcingMCPRuntimeContext) -> None:
        self.call_order.append("prepare")
        self.prepare_calls.append(runtime_context)
        runtime_context.kv_cache.add_(1.0)
        runtime_context.cross_attention_cache["prepared"] = "partial"
        runtime_context.cross_attention_cache["partial"] = True
        self.scratch_state.append("prepare-temp")
        torch.rand(1)
        if self.prepare_error is not None:
            raise self.prepare_error
        runtime_context.cross_attention_cache["prepared"] = True
        runtime_context.cross_attention_cache.pop("partial", None)

    def propose_anchor_and_drafts(
        self,
        request: ControlRequest,
        runtime_context: SelfForcingMCPRuntimeContext,
    ) -> ProposalBatch:
        self.call_order.append("proposal")
        self.proposal_calls.append((request, runtime_context))
        self.current_starts.append(runtime_context.current_start_for(request.anchor_block))
        runtime_context.kv_cache.add_(2.0)
        self._write_output_block(runtime_context, request.anchor_block, 20.0)
        self.scratch_state.append("proposal-temp")
        torch.rand(1)
        if self.proposal_error is not None:
            raise self.proposal_error
        if self.proposal_result is not None:
            return self.proposal_result
        return self._make_batch(request, runtime_context)

    def generate_target_fallback(
        self,
        candidate: DraftCandidate,
        runtime_context: SelfForcingMCPRuntimeContext,
    ) -> FallbackResult:
        self.call_order.append("fallback")
        self.fallback_calls.append((candidate, runtime_context))
        self.current_starts.append(runtime_context.current_start_for(candidate.block))
        runtime_context.kv_cache.add_(3.0)
        self._write_output_block(runtime_context, candidate.block, 30.0)
        self.scratch_state.append("fallback-temp")
        torch.rand(1)
        if self.fallback_error is not None:
            raise self.fallback_error
        if self.fallback_result is not None:
            return self.fallback_result
        return FallbackResult(
            block=candidate.block,
            latent=block_latent(candidate.block, 300.0 + candidate.depth),
            source_noise=candidate.source_noise,
        )

    def commit_context_block(
        self,
        request: CommitRequest,
        runtime_context: SelfForcingMCPRuntimeContext,
    ) -> None:
        self.call_order.append("commit")
        self.commit_calls.append((request, runtime_context))
        self.current_starts.append(runtime_context.current_start_for(request.block))
        runtime_context.kv_cache.add_(10.0 + request.block.index)
        self.scratch_state.append(f"commit-{request.block.index}")
        torch.rand(1)
        if self.commit_error is not None:
            raise self.commit_error

    def temporary_state_specs(
        self,
        operation: str,
        target: object,
        runtime_context: SelfForcingMCPRuntimeContext,
    ) -> RuntimeStateSpecBundle:
        self.call_order.append(f"{operation}-specs")
        self.state_spec_calls.append((operation, target))
        bundle = RuntimeStateSpecBundle(
            tensor_values=(runtime_context.kv_cache,),
            object_states=(self._scratch_state_spec(),),
            capture_rng=True,
        )
        self.temporary_spec_bundles.append(bundle)
        return bundle

    def prepare_persistent_state_specs(
        self,
        runtime_context: SelfForcingMCPRuntimeContext,
    ) -> RuntimeStateSpecBundle:
        self.call_order.append("prepare-persistent-specs")
        bundle = RuntimeStateSpecBundle(
            object_states=(self._cross_attention_state_spec(runtime_context),),
        )
        self.prepare_persistent_spec_bundles.append(bundle)
        return bundle

    def window_state_specs(
        self,
        window: RuntimeWindowDescriptor,
        runtime_context: SelfForcingMCPRuntimeContext,
    ) -> RuntimeStateSpecBundle:
        self.call_order.append("window-specs")
        self.window_state_spec_calls.append(window)
        object_states: tuple[ObjectStateSpec[object], ...]
        object_states = (self._scratch_state_spec(),)
        if self.fail_window_capture:
            object_states = object_states + (self._failing_capture_spec(),)
        bundle = RuntimeStateSpecBundle(
            tensor_values=(runtime_context.kv_cache,),
            object_states=object_states,
            capture_rng=True,
        )
        self.window_spec_bundles.append(bundle)
        return bundle

    def _write_output_block(
        self,
        runtime_context: SelfForcingMCPRuntimeContext,
        block: BlockRef,
        value: float,
    ) -> None:
        assert block.start_frame is not None and block.num_frames is not None
        start = block.start_frame
        end = start + block.num_frames
        runtime_context.output[:, start:end] = torch.full_like(
            runtime_context.output[:, start:end],
            value,
        )

    def _scratch_state_spec(self) -> ObjectStateSpec[tuple[str, ...]]:
        return ObjectStateSpec(
            getter=lambda: tuple(self.scratch_state),
            setter=self._restore_scratch_state,
            copy_fn=tuple,
            name="fake_backend_scratch_state",
        )

    def _failing_capture_spec(self) -> ObjectStateSpec[object]:
        def fail() -> object:
            raise RuntimeError("window capture failed")

        return ObjectStateSpec(
            getter=fail,
            setter=lambda value: None,
            name="fake_backend_capture_failure",
        )

    def _restore_scratch_state(self, value: object) -> None:
        self.scratch_state = list(value)  # type: ignore[arg-type]

    def _cross_attention_state_spec(
        self,
        runtime_context: SelfForcingMCPRuntimeContext,
    ) -> ObjectStateSpec[dict[str, object]]:
        return ObjectStateSpec(
            getter=lambda: dict(runtime_context.cross_attention_cache),
            setter=lambda value: self._restore_cross_attention_state(runtime_context, value),
            copy_fn=dict,
            name="fake_cross_attention_cache",
        )

    def _restore_cross_attention_state(
        self,
        runtime_context: SelfForcingMCPRuntimeContext,
        value: object,
    ) -> None:
        runtime_context.cross_attention_cache.clear()
        runtime_context.cross_attention_cache.update(value)  # type: ignore[arg-type]

    def _make_batch(
        self,
        request: ControlRequest,
        runtime_context: SelfForcingMCPRuntimeContext,
    ) -> ProposalBatch:
        plan = runtime_context.rollout_plan
        anchor = CommitRequest(
            block=request.anchor_block,
            latent=block_latent(request.anchor_block, 100.0 + request.anchor_block.index),
            source="anchor",
        )
        remaining = len(plan.blocks) - request.anchor_block.index - 1
        depth_count = min(request.max_depth, runtime_context.config.mcp_depth, remaining)
        drafts: list[DraftCandidate] = []
        for depth in range(1, depth_count + 1):
            block = plan.blocks[request.anchor_block.index + depth]
            drafts.append(
                DraftCandidate(
                    block=block,
                    depth=depth,
                    latent=block_latent(block, 200.0 + depth),
                    source_noise=runtime_context.source_noise[block.index],
                )
            )
        return ProposalBatch(anchor=anchor, drafts=tuple(drafts))


class RuntimeHarness:
    def __init__(
        self,
        *,
        config: Optional[SelfForcingMCPRuntimeConfig] = None,
        backend: Optional[FakeRuntimeBackend] = None,
    ) -> None:
        self.config = make_config() if config is None else config
        self.backend = FakeRuntimeBackend() if backend is None else backend
        self.source_noise = [object() for _ in range(self.config.all_num_frames)]
        self.output = torch.zeros((1, self.config.all_num_frames, 1), dtype=torch.float32)
        self.kv_cache = torch.zeros(4, dtype=torch.float32)
        self.cross_attention_cache: dict[str, object] = {"prepared": False}
        self.runtime = SelfForcingMCPRuntime(
            config=self.config,
            backend=self.backend,
            source_noise=self.source_noise,
            output=self.output,
            kv_cache=self.kv_cache,
            cross_attention_cache=self.cross_attention_cache,
        )

    def prepare(self) -> None:
        self.runtime.prepare()


class SelfForcingRuntimeTest(unittest.TestCase):
    def test_config_strict_integer_validation_rejects_bool_float_and_str(self) -> None:
        cases = [
            {"frame_seq_length": True},
            {"frame_seq_length": 1.5},
            {"frame_seq_length": "10"},
            {"num_frame_per_block": False},
            {"num_frame_per_block": 3.0},
            {"all_num_frames": "7"},
            {"mcp_depth": True},
            {"mcp_depth": 1.0},
        ]
        for kwargs in cases:
            with self.subTest(kwargs=kwargs):
                values = {
                    "anchor_denoising_steps": (1000,),
                    "frame_seq_length": 10,
                    "num_frame_per_block": 3,
                    "all_num_frames": 7,
                    "mcp_depth": 2,
                    "attention_mode": VALIDATED_MODE,
                    "validated_attention_mode": VALIDATED_MODE,
                }
                values.update(kwargs)
                with self.assertRaises(ValueError):
                    SelfForcingMCPRuntimeConfig(**values)

    def test_non_1000_schedule_fails_fast(self) -> None:
        for steps in [(999,), (1000, 0), (1000, 999)]:
            with self.subTest(steps=steps):
                with self.assertRaises(ValueError):
                    make_config(steps=steps)

    def test_unsupported_attention_mode_fails_fast(self) -> None:
        with self.assertRaises(ValueError):
            make_config(attention_mode="unsupported-local-cache")

    def test_rollout_plan_full_blocks_and_final_short_block(self) -> None:
        full = RuntimeHarness(config=make_config(total_frames=6, block_frames=3)).runtime
        short = RuntimeHarness(config=make_config(total_frames=8, block_frames=3)).runtime

        self.assertEqual([(b.start_frame, b.num_frames) for b in full.rollout_plan.blocks], [(0, 3), (3, 3)])
        self.assertEqual(
            [(b.start_frame, b.num_frames) for b in short.rollout_plan.blocks],
            [(0, 3), (3, 3), (6, 2)],
        )

    def test_anchor_indices_follow_period(self) -> None:
        runtime = RuntimeHarness(
            config=make_config(total_frames=13, block_frames=3, mcp_depth=2)
        ).runtime

        self.assertEqual(runtime.rollout_plan.period, 3)
        self.assertEqual(runtime.rollout_plan.anchor_block_indices, (0, 3))

    def test_current_start_is_derived_from_blockref_without_frame_cursor(self) -> None:
        runtime = RuntimeHarness(config=make_config(total_frames=8, frame_seq_length=11)).runtime
        block = runtime.rollout_plan.blocks[2]

        self.assertEqual(runtime.current_start_for(block), 66)
        self.assertFalse(hasattr(runtime, "current_start"))
        self.assertFalse(hasattr(runtime, "_current_start"))

    def test_prepare_success_required_before_runtime_methods(self) -> None:
        harness = RuntimeHarness()
        runtime = harness.runtime
        request = ControlRequest(anchor_block=runtime.rollout_plan.blocks[0], max_depth=1)
        candidate = DraftCandidate(
            block=runtime.rollout_plan.blocks[1],
            depth=1,
            latent=block_latent(runtime.rollout_plan.blocks[1], 1.0),
            source_noise=harness.source_noise[1],
        )
        commit = CommitRequest(block=runtime.rollout_plan.blocks[0], latent=block_latent(runtime.rollout_plan.blocks[0], 1.0), source="anchor")

        with self.assertRaises(RuntimeError):
            runtime.propose_window(request)
        with self.assertRaises(RuntimeError):
            runtime.generate_target_fallback(candidate)
        with self.assertRaises(RuntimeError):
            runtime.begin_window()
        with self.assertRaises(RuntimeError):
            runtime.commit_block(commit)

        runtime.prepare()
        result = runtime.propose_window(request)
        self.assertIsInstance(result, ProposalBatch)

    def test_prepare_can_succeed_only_once(self) -> None:
        runtime = RuntimeHarness().runtime

        runtime.prepare()
        with self.assertRaises(RuntimeError):
            runtime.prepare()

    def test_prepare_failure_can_retry_and_restores_state(self) -> None:
        harness = RuntimeHarness()
        error = RuntimeError("prepare failed")
        harness.backend.prepare_error = error
        torch.manual_seed(123)
        expected_rng = torch.random.get_rng_state().clone()
        expected_kv = harness.kv_cache.clone()
        expected_cross_attention = dict(harness.cross_attention_cache)

        with self.assertRaises(RuntimeError) as context:
            harness.runtime.prepare()

        self.assertIs(context.exception, error)
        self.assertFalse(harness.runtime.is_prepared)
        assert_tensor_equal(self, harness.kv_cache, expected_kv)
        assert_tensor_equal(self, torch.random.get_rng_state(), expected_rng)
        self.assertEqual(harness.cross_attention_cache, expected_cross_attention)
        self.assertEqual(harness.backend.scratch_state, [])

        harness.backend.prepare_error = None
        harness.runtime.prepare()
        self.assertTrue(harness.runtime.is_prepared)
        self.assertTrue(harness.cross_attention_cache["prepared"])

    def test_partial_cross_attention_prepare_failure_restores_state(self) -> None:
        harness = RuntimeHarness()
        harness.backend.prepare_error = RuntimeError("prepare failed")

        with self.assertRaises(RuntimeError):
            harness.runtime.prepare()

        self.assertEqual(harness.cross_attention_cache, {"prepared": False})
        self.assertFalse(harness.runtime.is_prepared)

    def test_prepare_temporary_state_and_rng_roll_back(self) -> None:
        harness = RuntimeHarness()
        torch.manual_seed(456)
        expected_rng = torch.random.get_rng_state().clone()
        expected_kv = harness.kv_cache.clone()
        expected_output = harness.output.clone()

        harness.runtime.prepare()

        assert_tensor_equal(self, harness.kv_cache, expected_kv)
        assert_tensor_equal(self, harness.output, expected_output)
        assert_tensor_equal(self, torch.random.get_rng_state(), expected_rng)
        self.assertEqual(harness.backend.scratch_state, [])
        self.assertTrue(harness.cross_attention_cache["prepared"])

    def test_proposal_request_identity_is_preserved(self) -> None:
        harness = RuntimeHarness()
        harness.prepare()
        request = ControlRequest(anchor_block=harness.runtime.rollout_plan.blocks[0], max_depth=2)

        harness.runtime.propose_window(request)

        self.assertIs(harness.backend.proposal_calls[0][0], request)
        self.assertIs(harness.backend.state_spec_calls[-1][1], request)

    def test_proposal_result_identity_is_preserved(self) -> None:
        harness = RuntimeHarness()
        harness.prepare()
        request = ControlRequest(anchor_block=harness.runtime.rollout_plan.blocks[0], max_depth=1)
        expected = ProposalBatch(
            anchor=CommitRequest(block=request.anchor_block, latent=block_latent(request.anchor_block, 1.0), source="anchor"),
            drafts=(),
        )
        harness.backend.proposal_result = expected

        result = harness.runtime.propose_window(request)

        self.assertIs(result, expected)

    def test_proposal_restores_kv_output_bookkeeping_and_rng(self) -> None:
        harness = RuntimeHarness()
        harness.prepare()
        torch.manual_seed(789)
        expected_rng = torch.random.get_rng_state().clone()
        expected_kv = harness.kv_cache.clone()
        expected_output = harness.output.clone()

        harness.runtime.propose_window(ControlRequest(anchor_block=harness.runtime.rollout_plan.blocks[0], max_depth=2))

        assert_tensor_equal(self, harness.kv_cache, expected_kv)
        assert_tensor_equal(self, harness.output, expected_output)
        assert_tensor_equal(self, torch.random.get_rng_state(), expected_rng)
        self.assertEqual(harness.runtime.committed_blocks, ())
        self.assertEqual(harness.backend.scratch_state, [])

    def test_proposal_exception_restores_state(self) -> None:
        harness = RuntimeHarness()
        harness.prepare()
        error = RuntimeError("proposal failed")
        harness.backend.proposal_error = error
        torch.manual_seed(321)
        expected_rng = torch.random.get_rng_state().clone()
        expected_kv = harness.kv_cache.clone()
        expected_output = harness.output.clone()

        with self.assertRaises(RuntimeError) as context:
            harness.runtime.propose_window(ControlRequest(anchor_block=harness.runtime.rollout_plan.blocks[0], max_depth=2))

        self.assertIs(context.exception, error)
        assert_tensor_equal(self, harness.kv_cache, expected_kv)
        assert_tensor_equal(self, harness.output, expected_output)
        assert_tensor_equal(self, torch.random.get_rng_state(), expected_rng)
        self.assertEqual(harness.backend.scratch_state, [])
        with self.assertRaises(RuntimeError):
            harness.runtime.begin_window()

    def test_malformed_proposal_does_not_leave_pending(self) -> None:
        harness = RuntimeHarness()
        harness.prepare()
        request = ControlRequest(anchor_block=harness.runtime.rollout_plan.blocks[0], max_depth=2)
        bad_draft = DraftCandidate(
            block=harness.runtime.rollout_plan.blocks[1],
            depth=2,
            latent=block_latent(harness.runtime.rollout_plan.blocks[1], 2.0),
            source_noise=harness.source_noise[1],
        )
        harness.backend.proposal_result = unsafe_proposal_batch(
            CommitRequest(block=request.anchor_block, latent=block_latent(request.anchor_block, 1.0), source="anchor"),
            (bad_draft,),
        )

        with self.assertRaises(RuntimeError):
            harness.runtime.propose_window(request)

        self.assertFalse(harness.runtime.has_active_window)
        with self.assertRaises(RuntimeError):
            harness.runtime.begin_window()

    def test_malformed_proposal_can_retry(self) -> None:
        harness = RuntimeHarness()
        harness.prepare()
        request = ControlRequest(anchor_block=harness.runtime.rollout_plan.blocks[0], max_depth=2)
        bad_draft = DraftCandidate(
            block=harness.runtime.rollout_plan.blocks[1],
            depth=2,
            latent=block_latent(harness.runtime.rollout_plan.blocks[1], 2.0),
            source_noise=harness.source_noise[1],
        )
        harness.backend.proposal_result = unsafe_proposal_batch(
            CommitRequest(block=request.anchor_block, latent=block_latent(request.anchor_block, 1.0), source="anchor"),
            (bad_draft,),
        )
        with self.assertRaises(RuntimeError):
            harness.runtime.propose_window(request)

        harness.backend.proposal_result = None
        batch = harness.runtime.propose_window(request)
        harness.runtime.begin_window()

        self.assertIsInstance(batch, ProposalBatch)
        self.assertTrue(harness.runtime.has_active_window)

    def test_descriptor_uses_actual_proposal_blocks(self) -> None:
        harness = RuntimeHarness()
        harness.prepare()
        request = ControlRequest(anchor_block=harness.runtime.rollout_plan.blocks[0], max_depth=2)
        harness.backend.proposal_result = self._batch_for(harness, anchor_index=0, depths=(1,))

        harness.runtime.propose_window(request)
        harness.runtime.begin_window()

        descriptor = harness.backend.window_state_spec_calls[0]
        self.assertEqual([block.index for block in descriptor.allowed_blocks], [0, 1])
        self.assertEqual(descriptor.anchor_block, request.anchor_block)
        self.assertFalse(hasattr(descriptor, "request"))

    def test_descriptor_does_not_retain_request_object(self) -> None:
        harness = RuntimeHarness()
        harness.prepare()
        request = ControlRequest(
            anchor_block=harness.runtime.rollout_plan.blocks[0],
            max_depth=1,
            metadata={"mutable": []},
        )

        harness.runtime.propose_window(request)
        harness.runtime.begin_window()

        descriptor = harness.backend.window_state_spec_calls[0]
        self.assertFalse(hasattr(descriptor, "request"))
        self.assertTrue(all(value is not request for value in descriptor.__dict__.values()))

    def test_mutating_request_metadata_after_proposal_cannot_change_window_specs(self) -> None:
        harness = RuntimeHarness()
        harness.prepare()
        metadata: dict[str, object] = {"mutable": []}
        request = ControlRequest(
            anchor_block=harness.runtime.rollout_plan.blocks[0],
            max_depth=2,
            metadata=metadata,
        )
        harness.backend.proposal_result = self._batch_for(harness, anchor_index=0, depths=(1,))

        harness.runtime.propose_window(request)
        metadata["mutable"] = [2]
        metadata["allowed_blocks"] = [0, 1, 2]
        harness.runtime.begin_window()

        descriptor = harness.backend.window_state_spec_calls[0]
        self.assertFalse(hasattr(descriptor, "metadata"))
        self.assertEqual([block.index for block in descriptor.allowed_blocks], [0, 1])

    def test_initial_future_anchor_is_rejected_before_backend_call(self) -> None:
        harness = RuntimeHarness()
        harness.prepare()
        request = ControlRequest(anchor_block=harness.runtime.rollout_plan.blocks[1], max_depth=1)

        with self.assertRaises(RuntimeError):
            harness.runtime.propose_window(request)

        self.assertEqual(harness.backend.proposal_calls, [])
        self.assertFalse(any(name == "proposal" for name, _ in harness.backend.state_spec_calls))

    def test_proposal_latent_alias_is_rejected(self) -> None:
        harness = RuntimeHarness()
        harness.prepare()
        request = ControlRequest(anchor_block=harness.runtime.rollout_plan.blocks[0], max_depth=1)
        harness.backend.proposal_result = ProposalBatch(
            anchor=CommitRequest(
                block=request.anchor_block,
                latent=harness.output[:, 0:3],
                source="anchor",
            ),
            drafts=(),
        )
        expected_output = harness.output.clone()

        with self.assertRaises(RuntimeError):
            harness.runtime.propose_window(request)

        assert_tensor_equal(self, harness.output, expected_output)
        self.assertFalse(harness.runtime.has_active_window)
        with self.assertRaises(RuntimeError):
            harness.runtime.begin_window()

    def test_proposal_non_tensor_latent_is_rejected(self) -> None:
        harness = RuntimeHarness()
        harness.prepare()
        request = ControlRequest(anchor_block=harness.runtime.rollout_plan.blocks[0], max_depth=1)
        harness.backend.proposal_result = ProposalBatch(
            anchor=CommitRequest(block=request.anchor_block, latent="not-a-tensor", source="anchor"),
            drafts=(),
        )
        expected_output = harness.output.clone()

        with self.assertRaises(RuntimeError):
            harness.runtime.propose_window(request)

        assert_tensor_equal(self, harness.output, expected_output)
        with self.assertRaises(RuntimeError):
            harness.runtime.begin_window()

    def test_proposal_broadcasting_shape_is_rejected(self) -> None:
        harness = RuntimeHarness()
        harness.prepare()
        request = ControlRequest(anchor_block=harness.runtime.rollout_plan.blocks[0], max_depth=1)
        harness.backend.proposal_result = ProposalBatch(
            anchor=CommitRequest(
                block=request.anchor_block,
                latent=torch.zeros((1, 1, 1), dtype=harness.output.dtype),
                source="anchor",
            ),
            drafts=(),
        )

        with self.assertRaises(RuntimeError):
            harness.runtime.propose_window(request)

        with self.assertRaises(RuntimeError):
            harness.runtime.begin_window()

    def test_proposal_dtype_mismatch_is_rejected(self) -> None:
        harness = RuntimeHarness()
        harness.prepare()
        request = ControlRequest(anchor_block=harness.runtime.rollout_plan.blocks[0], max_depth=1)
        harness.backend.proposal_result = ProposalBatch(
            anchor=CommitRequest(
                block=request.anchor_block,
                latent=torch.zeros((1, 3, 1), dtype=torch.float64),
                source="anchor",
            ),
            drafts=(),
        )

        with self.assertRaises(RuntimeError):
            harness.runtime.propose_window(request)

        with self.assertRaises(RuntimeError):
            harness.runtime.begin_window()

    def test_fallback_candidate_result_and_source_noise_identity_are_preserved(self) -> None:
        harness = RuntimeHarness()
        harness.prepare()
        block = harness.runtime.rollout_plan.blocks[1]
        candidate = DraftCandidate(
            block=block,
            depth=1,
            latent=block_latent(block, 2.0),
            source_noise=harness.source_noise[1],
        )
        expected = FallbackResult(
            block=block,
            latent=block_latent(block, 3.0),
            source_noise=candidate.source_noise,
        )
        harness.backend.fallback_result = expected

        result = harness.runtime.generate_target_fallback(candidate)

        self.assertIs(harness.backend.fallback_calls[0][0], candidate)
        self.assertIs(result, expected)
        self.assertIs(result.source_noise, candidate.source_noise)

    def test_fallback_restores_state(self) -> None:
        harness = RuntimeHarness()
        harness.prepare()
        candidate = self._candidate_for(harness, block_index=1, depth=1)
        torch.manual_seed(654)
        expected_rng = torch.random.get_rng_state().clone()
        expected_kv = harness.kv_cache.clone()
        expected_output = harness.output.clone()

        harness.runtime.generate_target_fallback(candidate)

        assert_tensor_equal(self, harness.kv_cache, expected_kv)
        assert_tensor_equal(self, harness.output, expected_output)
        assert_tensor_equal(self, torch.random.get_rng_state(), expected_rng)
        self.assertEqual(harness.runtime.committed_blocks, ())
        self.assertEqual(harness.backend.scratch_state, [])

    def test_fallback_exception_restores_state(self) -> None:
        harness = RuntimeHarness()
        harness.prepare()
        candidate = self._candidate_for(harness, block_index=1, depth=1)
        error = RuntimeError("fallback failed")
        harness.backend.fallback_error = error
        torch.manual_seed(987)
        expected_rng = torch.random.get_rng_state().clone()
        expected_kv = harness.kv_cache.clone()
        expected_output = harness.output.clone()

        with self.assertRaises(RuntimeError) as context:
            harness.runtime.generate_target_fallback(candidate)

        self.assertIs(context.exception, error)
        assert_tensor_equal(self, harness.kv_cache, expected_kv)
        assert_tensor_equal(self, harness.output, expected_output)
        assert_tensor_equal(self, torch.random.get_rng_state(), expected_rng)
        self.assertEqual(harness.backend.scratch_state, [])

    def test_fallback_latent_alias_is_rejected(self) -> None:
        harness = RuntimeHarness()
        harness.prepare()
        candidate = self._candidate_for(harness, block_index=1, depth=1)
        harness.backend.fallback_result = FallbackResult(
            block=candidate.block,
            latent=harness.output[:, 3:6],
            source_noise=candidate.source_noise,
        )
        expected_output = harness.output.clone()

        with self.assertRaises(RuntimeError):
            harness.runtime.generate_target_fallback(candidate)

        assert_tensor_equal(self, harness.output, expected_output)
        self.assertFalse(harness.runtime.has_active_window)

    def test_fallback_incompatible_latent_is_rejected(self) -> None:
        harness = RuntimeHarness()
        harness.prepare()
        candidate = self._candidate_for(harness, block_index=1, depth=1)
        harness.backend.fallback_result = FallbackResult(
            block=candidate.block,
            latent=torch.zeros((1, 1, 1), dtype=harness.output.dtype),
            source_noise=candidate.source_noise,
        )
        expected_output = harness.output.clone()

        with self.assertRaises(RuntimeError):
            harness.runtime.generate_target_fallback(candidate)

        assert_tensor_equal(self, harness.output, expected_output)
        self.assertEqual(harness.backend.commit_calls, [])

    def test_fallback_block_mismatch_is_rejected_inside_runtime(self) -> None:
        harness = RuntimeHarness()
        harness.prepare()
        candidate = self._candidate_for(harness, block_index=1, depth=1)
        wrong_block = harness.runtime.rollout_plan.blocks[2]
        harness.backend.fallback_result = FallbackResult(
            block=wrong_block,
            latent=block_latent(wrong_block, 3.0),
            source_noise=candidate.source_noise,
        )
        expected_output = harness.output.clone()

        with self.assertRaises(RuntimeError):
            harness.runtime.generate_target_fallback(candidate)

        assert_tensor_equal(self, harness.output, expected_output)
        self.assertEqual(harness.backend.commit_calls, [])

    def test_fallback_source_noise_identity_mismatch_is_rejected_inside_runtime(self) -> None:
        harness = RuntimeHarness()
        harness.prepare()
        candidate = self._candidate_for(harness, block_index=1, depth=1)
        harness.backend.fallback_result = FallbackResult(
            block=candidate.block,
            latent=block_latent(candidate.block, 3.0),
            source_noise=object(),
        )
        expected_output = harness.output.clone()

        with self.assertRaises(RuntimeError):
            harness.runtime.generate_target_fallback(candidate)

        assert_tensor_equal(self, harness.output, expected_output)
        self.assertEqual(harness.backend.commit_calls, [])

    def test_alias_rejection_leaves_no_pending_or_active_state(self) -> None:
        harness = RuntimeHarness()
        harness.prepare()
        request = ControlRequest(anchor_block=harness.runtime.rollout_plan.blocks[0], max_depth=1)
        harness.backend.proposal_result = ProposalBatch(
            anchor=CommitRequest(
                block=request.anchor_block,
                latent=harness.output[:, 0:3],
                source="anchor",
            ),
            drafts=(),
        )

        with self.assertRaises(RuntimeError):
            harness.runtime.propose_window(request)

        self.assertFalse(harness.runtime.has_active_window)
        with self.assertRaises(RuntimeError):
            harness.runtime.begin_window()

    def test_begin_without_pending_proposal_is_rejected(self) -> None:
        harness = RuntimeHarness()
        harness.prepare()

        with self.assertRaises(RuntimeError):
            harness.runtime.begin_window()

    def test_begin_cannot_be_nested(self) -> None:
        harness = RuntimeHarness()
        self._begin_first_window(harness)

        with self.assertRaises(RuntimeError):
            harness.runtime.begin_window()

    def test_begin_capture_failure_does_not_leave_active_window(self) -> None:
        harness = RuntimeHarness()
        harness.prepare()
        harness.runtime.propose_window(ControlRequest(anchor_block=harness.runtime.rollout_plan.blocks[0], max_depth=1))
        harness.backend.fail_window_capture = True

        with self.assertRaises(RuntimeError):
            harness.runtime.begin_window()

        self.assertFalse(harness.runtime.has_active_window)
        harness.backend.fail_window_capture = False
        harness.runtime.begin_window()
        self.assertTrue(harness.runtime.has_active_window)

    def test_commit_requires_active_window(self) -> None:
        harness = RuntimeHarness()
        harness.prepare()
        commit = CommitRequest(
            block=harness.runtime.rollout_plan.blocks[0],
            latent=block_latent(harness.runtime.rollout_plan.blocks[0], 1.0),
            source="anchor",
        )

        with self.assertRaises(RuntimeError):
            harness.runtime.commit_block(commit)

    def test_commit_order_is_validated_against_plan(self) -> None:
        harness = RuntimeHarness()
        self._begin_first_window(harness)
        block = harness.runtime.rollout_plan.blocks[1]

        with self.assertRaises(RuntimeError):
            harness.runtime.commit_block(
                CommitRequest(
                    block=block,
                    latent=block_latent(block, 2.0),
                    source="draft",
                    depth=1,
                    source_noise=harness.source_noise[1],
                )
            )

    def test_commit_revalidates_latent_after_proposal(self) -> None:
        harness = RuntimeHarness()
        batch = self._begin_first_window(harness)
        invalid_commit = CommitRequest(
            block=batch.anchor.block,
            latent=torch.zeros((1, 1, 1), dtype=harness.output.dtype),
            source="anchor",
        )
        expected_output = harness.output.clone()

        with self.assertRaises(RuntimeError):
            harness.runtime.commit_block(invalid_commit)

        self.assertEqual(harness.backend.commit_calls, [])
        self.assertEqual(harness.runtime.committed_blocks, ())
        assert_tensor_equal(self, harness.output, expected_output)

    def test_invalid_commit_does_not_call_backend(self) -> None:
        harness = RuntimeHarness()
        batch = self._begin_first_window(harness)
        invalid_commit = CommitRequest(
            block=batch.anchor.block,
            latent=torch.zeros_like(batch.anchor.latent, dtype=torch.float64),
            source="anchor",
        )

        with self.assertRaises(RuntimeError):
            harness.runtime.commit_block(invalid_commit)

        self.assertEqual(harness.backend.commit_calls, [])
        self.assertEqual(harness.runtime.committed_blocks, ())

    def test_commit_rejects_latent_rebound_to_output_storage(self) -> None:
        harness = RuntimeHarness()
        harness.prepare()
        request = ControlRequest(anchor_block=harness.runtime.rollout_plan.blocks[0], max_depth=1)
        batch = harness.runtime.propose_window(request)
        harness.runtime.begin_window()
        expected_output = harness.output.clone()

        batch.anchor.latent.set_(harness.output[:, 0:3])

        with self.assertRaises(RuntimeError):
            harness.runtime.commit_block(batch.anchor)

        self.assertEqual(harness.backend.commit_calls, [])
        self.assertEqual(harness.runtime.committed_blocks, ())
        assert_tensor_equal(self, harness.output, expected_output)
        self.assertTrue(harness.runtime.has_active_window)

        harness.runtime.rollback_window()
        assert_tensor_equal(self, harness.output, expected_output)
        self.assertFalse(harness.runtime.has_active_window)

    def test_commit_rejects_latent_rebound_to_backend_window_tensor_storage(self) -> None:
        harness = RuntimeHarness()
        harness.prepare()
        request = ControlRequest(anchor_block=harness.runtime.rollout_plan.blocks[0], max_depth=1)
        batch = harness.runtime.propose_window(request)
        harness.runtime.begin_window()
        expected_output = harness.output.clone()
        expected_kv = harness.kv_cache.clone()
        kv_view = torch.as_strided(harness.kv_cache, size=(1, 3, 1), stride=(3, 1, 1))

        batch.anchor.latent.set_(kv_view)

        with self.assertRaises(RuntimeError):
            harness.runtime.commit_block(batch.anchor)

        self.assertEqual(harness.backend.commit_calls, [])
        self.assertEqual(harness.runtime.committed_blocks, ())
        assert_tensor_equal(self, harness.output, expected_output)
        self.assertTrue(harness.runtime.has_active_window)

        harness.runtime.rollback_window()
        assert_tensor_equal(self, harness.output, expected_output)
        assert_tensor_equal(self, harness.kv_cache, expected_kv)
        self.assertFalse(harness.runtime.has_active_window)

    def test_commit_backend_exception_does_not_update_bookkeeping(self) -> None:
        harness = RuntimeHarness()
        batch = self._begin_first_window(harness)
        harness.backend.commit_error = RuntimeError("commit failed")
        expected_output = harness.output.clone()

        with self.assertRaises(RuntimeError):
            harness.runtime.commit_block(batch.anchor)

        self.assertEqual(harness.runtime.committed_blocks, ())
        assert_tensor_equal(self, harness.output, expected_output)
        harness.runtime.rollback_window()
        self.assertEqual(harness.backend.scratch_state, [])

    def test_commit_success_writes_output_and_updates_bookkeeping(self) -> None:
        harness = RuntimeHarness()
        batch = self._begin_first_window(harness)

        harness.runtime.commit_block(batch.anchor)

        block = batch.anchor.block
        assert block.start_frame is not None and block.num_frames is not None
        assert_tensor_equal(
            self,
            harness.output[:, block.start_frame:block.start_frame + block.num_frames],
            batch.anchor.latent,
        )
        self.assertEqual(harness.runtime.committed_blocks, (block,))

    def test_complete_preserves_kv_output_bookkeeping_and_rng_changes(self) -> None:
        harness = RuntimeHarness()
        batch = self._begin_first_window(harness)
        torch.manual_seed(111)
        initial_rng = torch.random.get_rng_state().clone()

        harness.runtime.commit_block(batch.anchor)
        changed_kv = harness.kv_cache.clone()
        changed_output = harness.output.clone()
        harness.runtime.complete_window()

        assert_tensor_equal(self, harness.kv_cache, changed_kv)
        assert_tensor_equal(self, harness.output, changed_output)
        self.assertEqual(harness.runtime.committed_blocks, (batch.anchor.block,))
        self.assertFalse(torch.equal(torch.random.get_rng_state(), initial_rng))
        self.assertFalse(harness.runtime.has_active_window)

    def test_rollback_restores_kv_output_bookkeeping_and_rng(self) -> None:
        harness = RuntimeHarness()
        torch.manual_seed(222)
        expected_rng = torch.random.get_rng_state().clone()
        batch = self._begin_first_window(harness)
        expected_kv = harness.kv_cache.clone()
        expected_output = harness.output.clone()

        harness.runtime.commit_block(batch.anchor)
        self.assertNotEqual(harness.runtime.committed_blocks, ())
        harness.runtime.rollback_window()

        assert_tensor_equal(self, harness.kv_cache, expected_kv)
        assert_tensor_equal(self, harness.output, expected_output)
        assert_tensor_equal(self, torch.random.get_rng_state(), expected_rng)
        self.assertEqual(harness.runtime.committed_blocks, ())
        self.assertEqual(harness.backend.scratch_state, [])
        self.assertFalse(harness.runtime.has_active_window)

    def test_backend_window_specs_omit_output_but_rollback_restores_it(self) -> None:
        harness = RuntimeHarness()
        batch = self._begin_first_window(harness)
        harness.runtime.commit_block(batch.anchor)
        window_specs = harness.backend.window_spec_bundles[-1]
        self.assertTrue(all(tensor is not harness.output for tensor in window_specs.tensor_values))
        self.assertFalse(any(region.tensor is harness.output for region in window_specs.tensor_regions))

        harness.runtime.rollback_window()

        assert_tensor_equal(self, harness.output, torch.zeros_like(harness.output))

    def test_backend_temporary_specs_omit_output_but_proposal_and_fallback_restore_it(self) -> None:
        harness = RuntimeHarness()
        harness.prepare()
        request = ControlRequest(anchor_block=harness.runtime.rollout_plan.blocks[0], max_depth=1)
        expected_output = harness.output.clone()

        harness.runtime.propose_window(request)
        assert_tensor_equal(self, harness.output, expected_output)
        proposal_specs = harness.backend.temporary_spec_bundles[-1]
        self.assertTrue(all(tensor is not harness.output for tensor in proposal_specs.tensor_values))
        self.assertFalse(any(region.tensor is harness.output for region in proposal_specs.tensor_regions))

        candidate = self._candidate_for(harness, block_index=1, depth=1)
        harness.runtime.generate_target_fallback(candidate)
        fallback_specs = harness.backend.temporary_spec_bundles[-1]
        self.assertTrue(all(tensor is not harness.output for tensor in fallback_specs.tensor_values))
        self.assertFalse(any(region.tensor is harness.output for region in fallback_specs.tensor_regions))
        assert_tensor_equal(self, harness.output, expected_output)

    def test_rollback_failure_clears_active_window(self) -> None:
        harness = RuntimeHarness()
        batch = self._begin_first_window(harness)
        harness.runtime.commit_block(batch.anchor)
        harness.kv_cache.resize_(5)

        with self.assertRaises(RuntimeError):
            harness.runtime.rollback_window()

        self.assertFalse(harness.runtime.has_active_window)

    def test_complete_and_rollback_repeated_calls_are_rejected(self) -> None:
        complete_harness = RuntimeHarness()
        self._begin_first_window(complete_harness)
        complete_harness.runtime.complete_window()
        with self.assertRaises(RuntimeError):
            complete_harness.runtime.complete_window()

        rollback_harness = RuntimeHarness()
        self._begin_first_window(rollback_harness)
        rollback_harness.runtime.rollback_window()
        with self.assertRaises(RuntimeError):
            rollback_harness.runtime.rollback_window()

    def test_new_proposal_is_rejected_while_window_is_active(self) -> None:
        harness = RuntimeHarness()
        self._begin_first_window(harness)

        with self.assertRaises(RuntimeError):
            harness.runtime.propose_window(ControlRequest(anchor_block=harness.runtime.rollout_plan.blocks[0], max_depth=1))

    def test_pending_proposal_cannot_be_silently_overwritten(self) -> None:
        harness = RuntimeHarness()
        harness.prepare()
        request = ControlRequest(anchor_block=harness.runtime.rollout_plan.blocks[0], max_depth=1)
        harness.runtime.propose_window(request)

        with self.assertRaises(RuntimeError):
            harness.runtime.propose_window(request)

        harness.runtime.begin_window()
        descriptor = harness.backend.window_state_spec_calls[0]
        self.assertFalse(hasattr(descriptor, "request"))
        self.assertEqual(descriptor.anchor_block, request.anchor_block)

    def test_controller_wrappers_runtime_always_accept(self) -> None:
        harness = RuntimeHarness(config=make_config(total_frames=9, block_frames=3, mcp_depth=2))
        harness.prepare()
        controller, evaluator = self._controller_for(harness, "always_accept")

        result = controller.run(ControlRequest(anchor_block=harness.runtime.rollout_plan.blocks[0], max_depth=2))

        self.assertEqual([block.index for block in harness.runtime.committed_blocks], [0, 1, 2])
        self.assertEqual([call[0].source for call in harness.backend.commit_calls], ["anchor", "draft", "draft"])
        self.assertEqual([candidate.depth for candidate in evaluator.calls], [1, 2])
        self.assertEqual(result.accepted_depth, 2)

    def test_always_accept_second_window_uses_reference_next_anchor(self) -> None:
        harness = RuntimeHarness(config=make_config(total_frames=12, block_frames=3, mcp_depth=2))
        harness.prepare()
        controller, _ = self._controller_for(harness, "always_accept")

        controller.run(ControlRequest(anchor_block=harness.runtime.rollout_plan.blocks[0], max_depth=2))
        controller.run(ControlRequest(anchor_block=harness.runtime.rollout_plan.blocks[3], max_depth=2))

        self.assertEqual(harness.runtime.rollout_plan.anchor_block_indices, (0, 3))
        self.assertEqual([call[0].anchor_block.index for call in harness.backend.proposal_calls], [0, 3])
        self.assertEqual([block.index for block in harness.runtime.committed_blocks], [0, 1, 2, 3])

    def test_controller_wrappers_runtime_always_reject(self) -> None:
        harness = RuntimeHarness(config=make_config(total_frames=9, block_frames=3, mcp_depth=2))
        harness.prepare()
        controller, evaluator = self._controller_for(harness, "always_reject")

        result = controller.run(ControlRequest(anchor_block=harness.runtime.rollout_plan.blocks[0], max_depth=2))

        self.assertEqual([block.index for block in harness.runtime.committed_blocks], [0, 1])
        self.assertEqual([call[0].source for call in harness.backend.commit_calls], ["anchor", "fallback"])
        self.assertEqual([candidate.depth for candidate in evaluator.calls], [1])
        self.assertEqual(result.rejected_depth, 1)
        self.assertEqual([candidate.block.index for candidate in result.invalidated], [2])

    def test_always_reject_multi_window_continues_from_next_uncommitted_block(self) -> None:
        harness = RuntimeHarness(config=make_config(total_frames=9, block_frames=3, mcp_depth=2))
        harness.prepare()
        controller, _ = self._controller_for(harness, "always_reject")

        controller.run(ControlRequest(anchor_block=harness.runtime.rollout_plan.blocks[0], max_depth=2))
        controller.run(ControlRequest(anchor_block=harness.runtime.rollout_plan.blocks[2], max_depth=2))

        self.assertEqual([call[0].anchor_block.index for call in harness.backend.proposal_calls], [0, 2])
        self.assertEqual([block.index for block in harness.runtime.committed_blocks], [0, 1, 2])

    def test_max_depth_limited_multi_window_continues_correctly(self) -> None:
        harness = RuntimeHarness(config=make_config(total_frames=9, block_frames=3, mcp_depth=2))
        harness.prepare()
        controller, _ = self._controller_for(harness, "always_accept")

        controller.run(ControlRequest(anchor_block=harness.runtime.rollout_plan.blocks[0], max_depth=1))
        controller.run(ControlRequest(anchor_block=harness.runtime.rollout_plan.blocks[2], max_depth=2))

        self.assertEqual([call[0].anchor_block.index for call in harness.backend.proposal_calls], [0, 2])
        self.assertEqual([block.index for block in harness.runtime.committed_blocks], [0, 1, 2])

    def test_final_short_block_can_be_next_anchor_when_not_returned_as_draft(self) -> None:
        harness = RuntimeHarness(config=make_config(total_frames=8, block_frames=3, mcp_depth=2))
        harness.prepare()
        controller, _ = self._controller_for(harness, "always_accept")

        controller.run(ControlRequest(anchor_block=harness.runtime.rollout_plan.blocks[0], max_depth=1))
        controller.run(ControlRequest(anchor_block=harness.runtime.rollout_plan.blocks[2], max_depth=2))

        self.assertEqual(harness.runtime.rollout_plan.blocks[2].num_frames, 2)
        self.assertEqual([call[0].anchor_block.index for call in harness.backend.proposal_calls], [0, 2])
        self.assertEqual([block.index for block in harness.runtime.committed_blocks], [0, 1, 2])

    def test_final_short_block_requires_exact_short_latent_shape(self) -> None:
        harness = RuntimeHarness(config=make_config(total_frames=8, block_frames=3, mcp_depth=2))
        harness.prepare()
        first_request = ControlRequest(anchor_block=harness.runtime.rollout_plan.blocks[0], max_depth=1)
        first_batch = harness.runtime.propose_window(first_request)
        harness.runtime.begin_window()
        harness.runtime.commit_block(first_batch.anchor)
        first_draft = first_batch.drafts[0]
        harness.runtime.commit_block(
            CommitRequest(
                block=first_draft.block,
                latent=first_draft.latent,
                source="draft",
                depth=first_draft.depth,
                source_noise=first_draft.source_noise,
            )
        )
        harness.runtime.complete_window()
        short_block = harness.runtime.rollout_plan.blocks[2]
        harness.backend.proposal_result = ProposalBatch(
            anchor=CommitRequest(
                block=short_block,
                latent=torch.zeros((1, 3, 1), dtype=harness.output.dtype),
                source="anchor",
            ),
            drafts=(),
        )

        with self.assertRaises(RuntimeError):
            harness.runtime.propose_window(ControlRequest(anchor_block=short_block, max_depth=2))

        with self.assertRaises(RuntimeError):
            harness.runtime.begin_window()

    def test_exact_compatible_latent_still_preserves_identity(self) -> None:
        harness = RuntimeHarness()
        harness.prepare()
        request = ControlRequest(anchor_block=harness.runtime.rollout_plan.blocks[0], max_depth=1)
        anchor_latent = block_latent(request.anchor_block, 9.0)
        draft_block = harness.runtime.rollout_plan.blocks[1]
        draft_latent = block_latent(draft_block, 10.0)
        harness.backend.proposal_result = ProposalBatch(
            anchor=CommitRequest(block=request.anchor_block, latent=anchor_latent, source="anchor"),
            drafts=(
                DraftCandidate(
                    block=draft_block,
                    depth=1,
                    latent=draft_latent,
                    source_noise=harness.source_noise[1],
                ),
            ),
        )

        result = harness.runtime.propose_window(request)
        harness.runtime.begin_window()
        harness.runtime.commit_block(result.anchor)

        self.assertIs(result.anchor.latent, anchor_latent)
        self.assertIs(result.drafts[0].latent, draft_latent)
        self.assertIs(harness.backend.commit_calls[0][0].latent, anchor_latent)

    def test_controller_wrappers_runtime_reject_at_depth(self) -> None:
        harness = RuntimeHarness(config=make_config(total_frames=9, block_frames=3, mcp_depth=2))
        harness.prepare()
        controller, evaluator = self._controller_for(harness, "reject_at_depth", reject_depth=2)

        result = controller.run(ControlRequest(anchor_block=harness.runtime.rollout_plan.blocks[0], max_depth=2))

        self.assertEqual([block.index for block in harness.runtime.committed_blocks], [0, 1, 2])
        self.assertEqual([call[0].source for call in harness.backend.commit_calls], ["anchor", "draft", "fallback"])
        self.assertEqual([candidate.depth for candidate in evaluator.calls], [1, 2])
        self.assertEqual(result.accepted_depth, 1)
        self.assertEqual(result.rejected_depth, 2)

    def test_controller_commit_exception_triggers_runtime_rollback(self) -> None:
        harness = RuntimeHarness(config=make_config(total_frames=9, block_frames=3, mcp_depth=2))
        harness.prepare()
        harness.backend.commit_error = RuntimeError("commit failed")
        controller, _ = self._controller_for(harness, "always_accept")
        expected_kv = harness.kv_cache.clone()
        expected_output = harness.output.clone()

        with self.assertRaises(RuntimeError):
            controller.run(ControlRequest(anchor_block=harness.runtime.rollout_plan.blocks[0], max_depth=2))

        self.assertFalse(harness.runtime.has_active_window)
        self.assertEqual(harness.runtime.committed_blocks, ())
        assert_tensor_equal(self, harness.kv_cache, expected_kv)
        assert_tensor_equal(self, harness.output, expected_output)

    def _begin_first_window(self, harness: RuntimeHarness) -> ProposalBatch:
        harness.prepare()
        request = ControlRequest(anchor_block=harness.runtime.rollout_plan.blocks[0], max_depth=2)
        batch = harness.runtime.propose_window(request)
        harness.runtime.begin_window()
        return batch

    def _candidate_for(self, harness: RuntimeHarness, *, block_index: int, depth: int) -> DraftCandidate:
        block = harness.runtime.rollout_plan.blocks[block_index]
        return DraftCandidate(
            block=block,
            depth=depth,
            latent=block_latent(block, float(depth)),
            source_noise=harness.source_noise[block_index],
        )

    def _batch_for(
        self,
        harness: RuntimeHarness,
        *,
        anchor_index: int,
        depths: tuple[int, ...],
    ) -> ProposalBatch:
        anchor_block = harness.runtime.rollout_plan.blocks[anchor_index]
        anchor = CommitRequest(
            block=anchor_block,
            latent=block_latent(anchor_block, 100.0 + anchor_index),
            source="anchor",
        )
        drafts = tuple(
            DraftCandidate(
                block=harness.runtime.rollout_plan.blocks[anchor_index + depth],
                depth=depth,
                latent=block_latent(
                    harness.runtime.rollout_plan.blocks[anchor_index + depth],
                    200.0 + depth,
                ),
                source_noise=harness.source_noise[anchor_index + depth],
            )
            for depth in depths
        )
        return ProposalBatch(anchor=anchor, drafts=drafts)

    def _controller_for(
        self,
        harness: RuntimeHarness,
        policy_name: str,
        *,
        reject_depth: Optional[int] = None,
    ) -> tuple[SpeculativeController, IdentityEvaluator]:
        kwargs = {} if reject_depth is None else {"reject_depth": reject_depth}
        evaluator = IdentityEvaluator()
        controller = SpeculativeController(
            proposer=SelfForcingMCPProposalSource(harness.runtime),
            evaluator=evaluator,
            policy=create_policy(policy_name, **kwargs),
            fallback=SelfForcingMCPFallbackGenerator(harness.runtime),
            committer=SelfForcingMCPCommitter(harness.runtime),
        )
        return controller, evaluator


if __name__ == "__main__":
    unittest.main()
