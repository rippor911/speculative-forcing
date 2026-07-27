from __future__ import annotations

import gc
import unittest
import weakref
from typing import Any, Sequence

import torch

import speculative.adapters.wan_vae_transaction as transaction_module
from speculative.adapters.wan_vae_transaction import (
    WanVAECacheRestoreError,
    fingerprint_wan_vae_cache,
    fingerprints_identity_equal,
    fingerprints_numerically_equal,
    fingerprints_structurally_equal,
)
from speculative.adapters.wan_vae_window import (
    WanVAECandidateDecoder,
    WanVAEPreviewRollbackError,
    WanVAEWindowCoordinator,
    WanVAEWindowError,
    WanVAEWindowRollbackError,
)
from speculative.evaluation import CompositeCandidateEvaluator, DecodedCandidate
from speculative.scoring import RawScoreResult
from speculative.types import BlockRef, DraftCandidate, Evaluation


class FakeModel:
    def __init__(
        self,
        *,
        conv_idx: list[object] | None = None,
        feat_map: list[object] | None = None,
    ) -> None:
        self._conv_idx = [0] if conv_idx is None else conv_idx
        self._feat_map = [None, None, None] if feat_map is None else feat_map
        self._conv_num = len(self._feat_map)
        self._enc_conv_num = 0
        self._enc_conv_idx = [0]
        self._enc_feat_map: list[object] = []


class NonWeakrefFakeModel:
    __slots__ = (
        "_conv_idx",
        "_feat_map",
        "_conv_num",
        "_enc_conv_num",
        "_enc_conv_idx",
        "_enc_feat_map",
    )

    def __init__(self) -> None:
        self._conv_idx = [0]
        self._feat_map = [None, None, None]
        self._conv_num = len(self._feat_map)
        self._enc_conv_num = 0
        self._enc_conv_idx = [0]
        self._enc_feat_map: list[object] = []


class FailOnceSliceList(list):
    def __init__(self, values: Sequence[object]) -> None:
        super().__init__(values)
        self.failures_remaining = 1

    def __setitem__(self, key: object, value: object) -> None:
        if isinstance(key, slice) and self.failures_remaining:
            self.failures_remaining -= 1
            raise RuntimeError("slice restore failed once")
        super().__setitem__(key, value)


class FakeWanVAEWrapper:
    def __init__(
        self,
        model: object | None = None,
        *,
        fail_after_cache_write: bool = False,
        fail_message: str = "decode failed after cache write",
        reentrant_callback: Any = None,
    ) -> None:
        self.model = FakeModel() if model is None else model
        self.fail_after_cache_write = fail_after_cache_write
        self.fail_message = fail_message
        self.reentrant_callback = reentrant_callback
        self.decode_calls: list[tuple[Any, bool]] = []
        self.returned_pixels: list[torch.Tensor] = []

    def decode_to_pixel(self, latent: Any, use_cache: bool = True) -> torch.Tensor:
        self.decode_calls.append((latent, use_cache))
        if use_cache is not True:
            raise ValueError("Fake cached decode requires use_cache=True.")
        if self.reentrant_callback is not None:
            self.reentrant_callback()

        model = self.model
        position = int(model._conv_idx[0])
        value = latent_scalar(latent)
        model._conv_idx = [position + 1]
        if position >= len(model._feat_map):
            model._feat_map.extend([None] * (position - len(model._feat_map) + 1))
        model._feat_map[position] = torch.tensor([value], dtype=torch.float32)

        if self.fail_after_cache_write:
            raise RuntimeError(self.fail_message)

        pixels = torch.tensor([value, float(position)], dtype=torch.float32)
        self.returned_pixels.append(pixels)
        return pixels


class GuardedWanVAEWrapper(FakeWanVAEWrapper):
    @property
    def policy(self) -> object:
        raise AssertionError("coordinator must not read policy")

    @property
    def threshold(self) -> object:
        raise AssertionError("coordinator must not read threshold")


class FakeScorer:
    def __init__(self) -> None:
        self.payloads: list[torch.Tensor] = []

    def score(self, decoded: DecodedCandidate) -> RawScoreResult:
        self.payloads.append(decoded.payload)
        return RawScoreResult(
            per_frame_scores=(0.4, 0.9),
            scorer_name="fake_scorer",
            metadata={"depth": decoded.candidate.depth},
        )


class FakeAggregator:
    name = "fake_aggregator"

    def __init__(self) -> None:
        self.calls: list[tuple[float, ...]] = []

    def aggregate(self, scores: Sequence[float]) -> float:
        normalized = tuple(float(score) for score in scores)
        self.calls.append(normalized)
        return normalized[-1]


def latent_scalar(latent: Any) -> float:
    if isinstance(latent, torch.Tensor):
        return float(latent.detach().reshape(-1)[0].item())
    return float(latent)


def fingerprint(wrapper: FakeWanVAEWrapper) -> dict[str, object]:
    return fingerprint_wan_vae_cache(wrapper, include_digest=True)


def clear_owner_record(model: FakeModel) -> None:
    with transaction_module._ACTIVE_OWNER_LOCK:
        transaction_module._ACTIVE_OWNERS.pop(id(model), None)


def candidate_at_depth(depth: int, latent: Any | None = None) -> DraftCandidate:
    return DraftCandidate(
        block=BlockRef(index=depth),
        depth=depth,
        latent=float(depth) if latent is None else latent,
        source_noise=object(),
    )


class WanVAEWindowCoordinatorTest(unittest.TestCase):
    def assertSameCache(
        self,
        left: dict[str, object],
        right: dict[str, object],
        *,
        identity: bool = True,
    ) -> None:
        self.assertTrue(fingerprints_structurally_equal(left, right))
        self.assertTrue(fingerprints_numerically_equal(left, right))
        if identity:
            self.assertTrue(fingerprints_identity_equal(left, right))

    def assertAllOperationsRejected(self, coordinator: WanVAEWindowCoordinator) -> None:
        operations = (
            ("begin", coordinator.begin_window),
            ("preview", lambda: coordinator.preview_latent(1.0)),
            ("commit", lambda: coordinator.commit_latent(1.0)),
            ("complete", coordinator.complete_window),
            ("rollback", coordinator.rollback_window),
        )
        for operation_name, operation in operations:
            with self.subTest(operation=operation_name):
                with self.assertRaises(WanVAEWindowError):
                    operation()

    def test_begin_enters_active_state(self) -> None:
        coordinator = WanVAEWindowCoordinator(FakeWanVAEWrapper()).begin_window()
        try:
            self.assertEqual(coordinator.state, WanVAEWindowCoordinator.ACTIVE)
            self.assertTrue(coordinator.is_active)
            self.assertFalse(coordinator.preview_active)
        finally:
            coordinator.rollback_window()

    def test_operations_before_begin_are_rejected(self) -> None:
        coordinator = WanVAEWindowCoordinator(FakeWanVAEWrapper())

        with self.assertRaisesRegex(WanVAEWindowError, "cannot commit"):
            coordinator.commit_latent(1.0)
        with self.assertRaisesRegex(WanVAEWindowError, "cannot preview"):
            coordinator.preview_latent(1.0)
        with self.assertRaisesRegex(WanVAEWindowError, "cannot complete"):
            coordinator.complete_window()
        with self.assertRaisesRegex(WanVAEWindowError, "cannot rollback"):
            coordinator.rollback_window()

    def test_double_begin_is_rejected(self) -> None:
        coordinator = WanVAEWindowCoordinator(FakeWanVAEWrapper()).begin_window()
        try:
            with self.assertRaisesRegex(WanVAEWindowError, "cannot begin"):
                coordinator.begin_window()
        finally:
            coordinator.rollback_window()

    def test_begin_failure_is_exception_atomic(self) -> None:
        wrapper = FakeWanVAEWrapper(NonWeakrefFakeModel())
        coordinator = WanVAEWindowCoordinator(wrapper)

        with self.assertRaisesRegex(TypeError, "weak references"):
            coordinator.begin_window()

        self.assertEqual(coordinator.state, WanVAEWindowCoordinator.NEW)
        self.assertIsNone(coordinator._transaction)
        self.assertFalse(coordinator.preview_active)

        wrapper.model = FakeModel()
        coordinator.begin_window()
        coordinator.rollback_window()

    def test_preview_restores_cache_fingerprint(self) -> None:
        wrapper = FakeWanVAEWrapper()
        coordinator = WanVAEWindowCoordinator(wrapper).begin_window()
        before = fingerprint(wrapper)

        try:
            coordinator.preview_latent(1.0)
            after = fingerprint(wrapper)
            self.assertSameCache(before, after)
        finally:
            coordinator.rollback_window()

    def test_preview_returns_original_pixels_object(self) -> None:
        wrapper = FakeWanVAEWrapper()
        coordinator = WanVAEWindowCoordinator(wrapper).begin_window()

        try:
            pixels = coordinator.preview_latent(3.0)
            self.assertIs(pixels, wrapper.returned_pixels[-1])
        finally:
            coordinator.rollback_window()

    def test_commit_keeps_cache_mutation(self) -> None:
        wrapper = FakeWanVAEWrapper()
        coordinator = WanVAEWindowCoordinator(wrapper).begin_window()
        before = fingerprint(wrapper)

        try:
            pixels = coordinator.commit_latent(5.0)
            after = fingerprint(wrapper)
            self.assertIs(pixels, wrapper.returned_pixels[-1])
            self.assertFalse(fingerprints_numerically_equal(before, after))
            self.assertEqual(wrapper.model._conv_idx, [1])
            self.assertTrue(torch.equal(wrapper.model._feat_map[0], torch.tensor([5.0])))
        finally:
            coordinator.rollback_window()

    def test_anchor_commit_then_preview_restores_anchor_prefix(self) -> None:
        wrapper = FakeWanVAEWrapper()
        coordinator = WanVAEWindowCoordinator(wrapper).begin_window()

        try:
            coordinator.commit_latent(10.0)
            anchor_prefix = fingerprint(wrapper)
            coordinator.preview_latent(20.0)
            after_preview = fingerprint(wrapper)
            self.assertSameCache(anchor_prefix, after_preview)
        finally:
            coordinator.rollback_window()

    def test_anchor_preview_commit_same_draft_matches_direct_decode(self) -> None:
        wrapper = FakeWanVAEWrapper()
        coordinator = WanVAEWindowCoordinator(wrapper).begin_window()
        try:
            coordinator.commit_latent(1.0)
            coordinator.preview_latent(2.0)
            coordinator.commit_latent(2.0)
            actual = fingerprint(wrapper)
            coordinator.complete_window()
        except Exception:
            if coordinator.state in (
                WanVAEWindowCoordinator.ACTIVE,
                WanVAEWindowCoordinator.FAILED,
            ):
                coordinator.rollback_window()
            raise

        direct = FakeWanVAEWrapper()
        direct.decode_to_pixel(1.0, use_cache=True)
        direct.decode_to_pixel(2.0, use_cache=True)
        expected = fingerprint(direct)
        self.assertSameCache(expected, actual, identity=False)

    def test_anchor_preview_commit_fallback_matches_direct_decode(self) -> None:
        wrapper = FakeWanVAEWrapper()
        coordinator = WanVAEWindowCoordinator(wrapper).begin_window()
        try:
            coordinator.commit_latent(1.0)
            coordinator.preview_latent(2.0)
            coordinator.commit_latent(9.0)
            actual = fingerprint(wrapper)
            coordinator.complete_window()
        except Exception:
            if coordinator.state in (
                WanVAEWindowCoordinator.ACTIVE,
                WanVAEWindowCoordinator.FAILED,
            ):
                coordinator.rollback_window()
            raise

        direct = FakeWanVAEWrapper()
        direct.decode_to_pixel(1.0, use_cache=True)
        direct.decode_to_pixel(9.0, use_cache=True)
        expected = fingerprint(direct)
        self.assertSameCache(expected, actual, identity=False)

    def test_preview_decode_exception_restores_and_stays_active(self) -> None:
        wrapper = FakeWanVAEWrapper(fail_after_cache_write=True, fail_message="preview boom")
        coordinator = WanVAEWindowCoordinator(wrapper).begin_window()
        before = fingerprint(wrapper)

        try:
            with self.assertRaisesRegex(RuntimeError, "preview boom"):
                coordinator.preview_latent(4.0)
            after = fingerprint(wrapper)
            self.assertSameCache(before, after)
            self.assertFalse(coordinator.preview_active)
            self.assertEqual(coordinator.state, WanVAEWindowCoordinator.ACTIVE)
        finally:
            coordinator.rollback_window()

    def test_preview_success_but_local_restore_failure_allows_outer_rollback(self) -> None:
        wrapper = FakeWanVAEWrapper(
            FakeModel(conv_idx=FailOnceSliceList([0])),
        )
        coordinator = WanVAEWindowCoordinator(wrapper).begin_window()
        before = fingerprint(wrapper)

        with self.assertRaises(WanVAECacheRestoreError):
            coordinator.preview_latent(4.0)

        self.assertEqual(len(wrapper.returned_pixels), 1)
        self.assertEqual(coordinator.state, WanVAEWindowCoordinator.FAILED)
        self.assertFalse(coordinator.preview_active)
        for operation in (
            lambda: coordinator.preview_latent(1.0),
            lambda: coordinator.commit_latent(1.0),
            coordinator.complete_window,
        ):
            with self.assertRaises(WanVAEWindowError):
                operation()

        coordinator.rollback_window()
        self.assertEqual(coordinator.state, WanVAEWindowCoordinator.ROLLED_BACK)
        self.assertSameCache(before, fingerprint(wrapper))

    def test_preview_decode_and_restore_failure_fails_and_only_allows_rollback(self) -> None:
        wrapper = FakeWanVAEWrapper(
            FakeModel(conv_idx=FailOnceSliceList([0])),
            fail_after_cache_write=True,
            fail_message="preview boom",
        )
        coordinator = WanVAEWindowCoordinator(wrapper).begin_window()

        with self.assertRaises(WanVAEPreviewRollbackError) as raised:
            coordinator.preview_latent(4.0)

        self.assertIsInstance(raised.exception.original_exception, RuntimeError)
        self.assertIsInstance(raised.exception.restore_exception, WanVAECacheRestoreError)
        self.assertEqual(coordinator.state, WanVAEWindowCoordinator.FAILED)
        self.assertFalse(coordinator.preview_active)
        for operation in (
            lambda: coordinator.begin_window(),
            lambda: coordinator.preview_latent(1.0),
            lambda: coordinator.commit_latent(1.0),
            coordinator.complete_window,
        ):
            with self.assertRaises(WanVAEWindowError):
                operation()

        coordinator.rollback_window()
        self.assertEqual(coordinator.state, WanVAEWindowCoordinator.ROLLED_BACK)

    def test_commit_exception_marks_failed_and_outer_rollback_restores_begin(self) -> None:
        wrapper = FakeWanVAEWrapper(fail_after_cache_write=True, fail_message="commit boom")
        coordinator = WanVAEWindowCoordinator(wrapper).begin_window()
        before = fingerprint(wrapper)

        with self.assertRaisesRegex(RuntimeError, "commit boom"):
            coordinator.commit_latent(7.0)

        self.assertEqual(coordinator.state, WanVAEWindowCoordinator.FAILED)
        self.assertFalse(fingerprints_numerically_equal(before, fingerprint(wrapper)))
        coordinator.rollback_window()
        self.assertSameCache(before, fingerprint(wrapper))

    def test_complete_keeps_final_cache(self) -> None:
        wrapper = FakeWanVAEWrapper()
        coordinator = WanVAEWindowCoordinator(wrapper).begin_window()
        coordinator.commit_latent(7.0)
        committed = fingerprint(wrapper)

        coordinator.complete_window()

        self.assertEqual(coordinator.state, WanVAEWindowCoordinator.COMPLETED)
        self.assertSameCache(committed, fingerprint(wrapper))

    def test_rollback_restores_begin_fingerprint(self) -> None:
        wrapper = FakeWanVAEWrapper()
        coordinator = WanVAEWindowCoordinator(wrapper).begin_window()
        before = fingerprint(wrapper)

        coordinator.commit_latent(1.0)
        coordinator.commit_latent(2.0)
        coordinator.rollback_window()

        self.assertEqual(coordinator.state, WanVAEWindowCoordinator.ROLLED_BACK)
        self.assertSameCache(before, fingerprint(wrapper))

    def test_completed_state_rejects_all_operations(self) -> None:
        coordinator = WanVAEWindowCoordinator(FakeWanVAEWrapper()).begin_window()
        coordinator.complete_window()

        self.assertAllOperationsRejected(coordinator)

    def test_rolled_back_state_rejects_all_operations(self) -> None:
        coordinator = WanVAEWindowCoordinator(FakeWanVAEWrapper()).begin_window()
        coordinator.rollback_window()

        self.assertAllOperationsRejected(coordinator)

    def test_nested_preview_is_rejected(self) -> None:
        wrapper = FakeWanVAEWrapper()
        coordinator = WanVAEWindowCoordinator(wrapper).begin_window()
        errors: list[Exception] = []

        def callback() -> None:
            try:
                coordinator.preview_latent(99.0)
            except Exception as error:
                errors.append(error)

        wrapper.reentrant_callback = callback
        try:
            coordinator.preview_latent(1.0)
            self.assertIsInstance(errors[0], WanVAEWindowError)
            self.assertIn("preview", str(errors[0]))
            self.assertIn("preview active", str(errors[0]))
        finally:
            coordinator.rollback_window()

    def test_preview_rejects_reentrant_commit(self) -> None:
        self.assert_reentrant_operation_rejected(
            lambda coordinator: coordinator.commit_latent(99.0),
            "commit",
        )

    def test_preview_rejects_reentrant_complete(self) -> None:
        self.assert_reentrant_operation_rejected(
            lambda coordinator: coordinator.complete_window(),
            "complete",
        )

    def test_preview_rejects_reentrant_rollback(self) -> None:
        self.assert_reentrant_operation_rejected(
            lambda coordinator: coordinator.rollback_window(),
            "rollback",
        )

    def assert_reentrant_operation_rejected(
        self,
        operation: Any,
        operation_name: str,
    ) -> None:
        wrapper = FakeWanVAEWrapper()
        coordinator = WanVAEWindowCoordinator(wrapper).begin_window()
        errors: list[Exception] = []

        def callback() -> None:
            try:
                operation(coordinator)
            except Exception as error:
                errors.append(error)

        wrapper.reentrant_callback = callback
        try:
            coordinator.preview_latent(1.0)
            self.assertIsInstance(errors[0], WanVAEWindowError)
            self.assertIn(operation_name, str(errors[0]))
            self.assertIn("preview active", str(errors[0]))
        finally:
            coordinator.rollback_window()

    def test_same_model_two_active_coordinators_rejected(self) -> None:
        wrapper = FakeWanVAEWrapper()
        first = WanVAEWindowCoordinator(wrapper).begin_window()
        try:
            with self.assertRaisesRegex(RuntimeError, "already active"):
                WanVAEWindowCoordinator(wrapper).begin_window()
        finally:
            first.rollback_window()

    def test_different_models_can_be_active(self) -> None:
        left = WanVAEWindowCoordinator(FakeWanVAEWrapper()).begin_window()
        right = WanVAEWindowCoordinator(FakeWanVAEWrapper()).begin_window()

        self.assertTrue(left.is_active)
        self.assertTrue(right.is_active)

        right.rollback_window()
        left.rollback_window()

    def test_gc_abandoned_active_coordinator_poisons_live_model_owner_gate(self) -> None:
        wrapper = FakeWanVAEWrapper()
        coordinator = WanVAEWindowCoordinator(wrapper).begin_window()
        model_id = id(wrapper.model)
        coordinator_ref = weakref.ref(coordinator)

        try:
            del coordinator
            gc.collect()

            self.assertIsNone(coordinator_ref())
            with self.assertRaisesRegex(RuntimeError, "abandoned/poisoned"):
                WanVAEWindowCoordinator(wrapper).begin_window()
        finally:
            with transaction_module._ACTIVE_OWNER_LOCK:
                transaction_module._ACTIVE_OWNERS.pop(model_id, None)

    def test_context_normal_exit_rolls_back_without_complete(self) -> None:
        wrapper = FakeWanVAEWrapper()
        before = fingerprint(wrapper)
        coordinator = WanVAEWindowCoordinator(wrapper)

        with coordinator as active:
            active.commit_latent(1.0)

        self.assertEqual(coordinator.state, WanVAEWindowCoordinator.ROLLED_BACK)
        self.assertSameCache(before, fingerprint(wrapper))

    def test_context_explicit_complete_keeps_state(self) -> None:
        wrapper = FakeWanVAEWrapper()
        coordinator = WanVAEWindowCoordinator(wrapper)

        with coordinator as active:
            active.commit_latent(1.0)
            committed = fingerprint(wrapper)
            active.complete_window()

        self.assertEqual(coordinator.state, WanVAEWindowCoordinator.COMPLETED)
        self.assertSameCache(committed, fingerprint(wrapper))

    def test_context_body_exception_rolls_back_and_reraises_original(self) -> None:
        wrapper = FakeWanVAEWrapper()
        before = fingerprint(wrapper)
        coordinator = WanVAEWindowCoordinator(wrapper)

        with self.assertRaisesRegex(ValueError, "body failed"):
            with coordinator as active:
                active.commit_latent(1.0)
                raise ValueError("body failed")

        self.assertEqual(coordinator.state, WanVAEWindowCoordinator.ROLLED_BACK)
        self.assertSameCache(before, fingerprint(wrapper))

    def test_context_body_exception_and_rollback_failure_preserves_both(self) -> None:
        wrapper = FakeWanVAEWrapper()
        original_model = wrapper.model
        coordinator = WanVAEWindowCoordinator(wrapper)

        try:
            with self.assertRaises(WanVAEWindowRollbackError) as raised:
                with coordinator:
                    wrapper.model = FakeModel()
                    raise ValueError("body failed")

            self.assertIsInstance(raised.exception.original_exception, ValueError)
            self.assertIsInstance(raised.exception.rollback_exception, WanVAECacheRestoreError)
            self.assertEqual(coordinator.state, WanVAEWindowCoordinator.POISONED)
            with self.assertRaisesRegex(RuntimeError, "abandoned/poisoned"):
                WanVAEWindowCoordinator(FakeWanVAEWrapper(original_model)).begin_window()
        finally:
            clear_owner_record(original_model)

    def test_candidate_decoder_returns_decoded_candidate(self) -> None:
        wrapper = FakeWanVAEWrapper()
        coordinator = WanVAEWindowCoordinator(wrapper).begin_window()
        decoder = WanVAECandidateDecoder(
            coordinator,
            metadata={"scope": "cpu"},
        )
        candidate = candidate_at_depth(2, latent=8.0)

        try:
            decoded = decoder.decode(candidate)
            self.assertIsInstance(decoded, DecodedCandidate)
            self.assertIs(decoded.payload, wrapper.returned_pixels[-1])
            self.assertEqual(decoded.metadata["decoder_name"], "wan_vae_cached_preview")
            self.assertEqual(decoded.metadata["block_index"], 2)
            self.assertEqual(decoded.metadata["depth"], 2)
            self.assertEqual(decoded.metadata["decoder_metadata"]["scope"], "cpu")
        finally:
            coordinator.rollback_window()

    def test_candidate_decoder_preserves_candidate_identity(self) -> None:
        coordinator = WanVAEWindowCoordinator(FakeWanVAEWrapper()).begin_window()
        candidate = candidate_at_depth(1, latent=3.0)

        try:
            decoded = WanVAECandidateDecoder(coordinator).decode(candidate)
            self.assertIs(decoded.candidate, candidate)
        finally:
            coordinator.rollback_window()

    def test_candidate_decoder_restores_cache_before_return(self) -> None:
        wrapper = FakeWanVAEWrapper()
        coordinator = WanVAEWindowCoordinator(wrapper).begin_window()
        coordinator.commit_latent(1.0)
        before = fingerprint(wrapper)

        try:
            WanVAECandidateDecoder(coordinator).decode(candidate_at_depth(1, latent=2.0))
            after = fingerprint(wrapper)
            self.assertSameCache(before, after)
        finally:
            coordinator.rollback_window()

    def test_composite_evaluator_leaves_cache_without_draft_preview(self) -> None:
        wrapper = FakeWanVAEWrapper()
        coordinator = WanVAEWindowCoordinator(wrapper).begin_window()
        coordinator.commit_latent(1.0)
        before = fingerprint(wrapper)
        scorer = FakeScorer()
        aggregator = FakeAggregator()
        evaluator = CompositeCandidateEvaluator(
            decoder=WanVAECandidateDecoder(coordinator),
            scorer=scorer,
            aggregator=aggregator,
        )

        try:
            evaluation = evaluator.evaluate(candidate_at_depth(1, latent=2.0))
            after = fingerprint(wrapper)
            self.assertIsInstance(evaluation, Evaluation)
            self.assertEqual(evaluation.metadata["aggregator_name"], "fake_aggregator")
            self.assertEqual(aggregator.calls, [(0.4, 0.9)])
            self.assertEqual(len(scorer.payloads), 1)
            self.assertSameCache(before, after)
        finally:
            coordinator.rollback_window()

    def test_coordinator_does_not_read_policy_or_threshold(self) -> None:
        wrapper = GuardedWanVAEWrapper()
        coordinator = WanVAEWindowCoordinator(wrapper).begin_window()

        try:
            coordinator.preview_latent(1.0)
            coordinator.commit_latent(2.0)
        finally:
            coordinator.rollback_window()

    def test_model_binding_changed_makes_rollback_poisoned(self) -> None:
        wrapper = FakeWanVAEWrapper()
        coordinator = WanVAEWindowCoordinator(wrapper).begin_window()
        original_model = wrapper.model
        wrapper.model = FakeModel()

        try:
            with self.assertRaises(WanVAECacheRestoreError):
                coordinator.rollback_window()

            self.assertEqual(coordinator.state, WanVAEWindowCoordinator.POISONED)
            with self.assertRaisesRegex(RuntimeError, "abandoned/poisoned"):
                WanVAEWindowCoordinator(FakeWanVAEWrapper(original_model)).begin_window()
        finally:
            clear_owner_record(original_model)

    def test_poisoned_live_model_cannot_be_reused_after_coordinator_gc(self) -> None:
        wrapper = FakeWanVAEWrapper()
        coordinator = WanVAEWindowCoordinator(wrapper).begin_window()
        original_model = wrapper.model
        wrapper.model = FakeModel()

        try:
            with self.assertRaises(WanVAECacheRestoreError):
                coordinator.rollback_window()
            coordinator_ref = weakref.ref(coordinator)
            del coordinator
            gc.collect()

            self.assertIsNone(coordinator_ref())
            with self.assertRaisesRegex(RuntimeError, "abandoned/poisoned"):
                WanVAEWindowCoordinator(FakeWanVAEWrapper(original_model)).begin_window()
        finally:
            clear_owner_record(original_model)

    def test_new_coordinator_cannot_bypass_poisoned_live_model(self) -> None:
        wrapper = FakeWanVAEWrapper()
        coordinator = WanVAEWindowCoordinator(wrapper).begin_window()
        original_model = wrapper.model
        wrapper.model = FakeModel()

        try:
            with self.assertRaises(WanVAECacheRestoreError):
                coordinator.rollback_window()
            for _ in range(2):
                with self.assertRaisesRegex(RuntimeError, "abandoned/poisoned"):
                    WanVAEWindowCoordinator(FakeWanVAEWrapper(original_model)).begin_window()
        finally:
            clear_owner_record(original_model)

    def test_complete_releases_snapshot_old_tensor_refs(self) -> None:
        old_tensor = torch.tensor([1.0], dtype=torch.float32)
        old_ref = weakref.ref(old_tensor)
        wrapper = FakeWanVAEWrapper(FakeModel(feat_map=[old_tensor, None]))
        coordinator = WanVAEWindowCoordinator(wrapper).begin_window()
        coordinator.commit_latent(7.0)

        del old_tensor
        gc.collect()
        self.assertIsNotNone(old_ref())

        coordinator.complete_window()
        gc.collect()
        self.assertIsNone(old_ref())

    def test_rollback_releases_temporary_cache_tensor_refs(self) -> None:
        old_tensor = torch.tensor([1.0], dtype=torch.float32)
        wrapper = FakeWanVAEWrapper(FakeModel(feat_map=[old_tensor, None]))
        coordinator = WanVAEWindowCoordinator(wrapper).begin_window()
        coordinator.commit_latent(7.0)
        temporary_tensor = wrapper.model._feat_map[0]
        temporary_ref = weakref.ref(temporary_tensor)

        coordinator.rollback_window()
        del temporary_tensor
        gc.collect()

        self.assertIsNone(temporary_ref())
        self.assertIs(wrapper.model._feat_map[0], old_tensor)


if __name__ == "__main__":
    unittest.main()
