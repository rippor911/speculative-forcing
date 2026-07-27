from __future__ import annotations

import gc
import json
import unittest
import weakref

import torch

import speculative.adapters.wan_vae_transaction as transaction_module
from speculative.adapters.wan_vae_transaction import (
    WanVAECacheRestoreError,
    WanVAECacheRollbackError,
    WanVAECacheSnapshot,
    WanVAECacheTransaction,
    fingerprint_wan_vae_cache,
    fingerprints_identity_equal,
    fingerprints_numerically_equal,
    fingerprints_structurally_equal,
)


class FakeModel:
    def __init__(
        self,
        *,
        conv_idx: list[object] | None = None,
        feat_map: list[object] | None = None,
    ) -> None:
        self._conv_idx = [0] if conv_idx is None else conv_idx
        self._feat_map = [None] if feat_map is None else feat_map
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
        self._feat_map = [None]
        self._conv_num = len(self._feat_map)
        self._enc_conv_num = 0
        self._enc_conv_idx = [0]
        self._enc_feat_map: list[object] = []


class FakeWrapper:
    def __init__(self, model: object) -> None:
        self.model = model


class FailingSliceList(list):
    def __setitem__(self, key, value):  # type: ignore[no-untyped-def]
        if isinstance(key, slice):
            raise RuntimeError("slice restore failed")
        return super().__setitem__(key, value)


def make_wrapper(*, feat_map: list[object] | None = None) -> FakeWrapper:
    return FakeWrapper(FakeModel(feat_map=feat_map))


def clear_owner_record(model: FakeModel) -> None:
    with transaction_module._ACTIVE_OWNER_LOCK:
        transaction_module._ACTIVE_OWNERS.pop(id(model), None)


def mutate_like_cached_decode(model: FakeModel) -> None:
    model._conv_idx = [7]
    model._feat_map[0] = torch.tensor([99.0], dtype=torch.float32)
    if len(model._feat_map) > 1:
        model._feat_map[1] = "Rep"


def tensor_digest(fingerprint: dict[str, object], attr: str = "_feat_map", index: int = 0) -> str:
    attrs = fingerprint["model_attributes"]  # type: ignore[index]
    entries = attrs[attr]["entries"]  # type: ignore[index]
    return entries[index]["digest"]  # type: ignore[index]


def tensor_object_id(fingerprint: dict[str, object], attr: str = "_feat_map", index: int = 0) -> int:
    attrs = fingerprint["model_attributes"]  # type: ignore[index]
    entries = attrs[attr]["entries"]  # type: ignore[index]
    return entries[index]["object_id"]  # type: ignore[index]


def tensor_data_ptr(fingerprint: dict[str, object], attr: str = "_feat_map", index: int = 0) -> int:
    attrs = fingerprint["model_attributes"]  # type: ignore[index]
    entries = attrs[attr]["entries"]  # type: ignore[index]
    return entries[index]["data_ptr"]  # type: ignore[index]


class WanVAECacheTransactionTest(unittest.TestCase):
    def test_capture_does_not_modify_original_state(self) -> None:
        tensor = torch.tensor([1.0])
        wrapper = make_wrapper(feat_map=[None, tensor, "Rep"])
        model = wrapper.model
        conv_idx_id = id(model._conv_idx)
        feat_map_id = id(model._feat_map)
        entries = tuple(model._feat_map)

        snapshot = WanVAECacheSnapshot.capture(wrapper)

        self.assertIs(snapshot.model, model)
        self.assertEqual(id(model._conv_idx), conv_idx_id)
        self.assertEqual(model._conv_idx, [0])
        self.assertEqual(id(model._feat_map), feat_map_id)
        self.assertEqual(tuple(model._feat_map), entries)

    def test_rollback_restores_conv_idx_original_list_identity(self) -> None:
        wrapper = make_wrapper()
        original = wrapper.model._conv_idx
        tx = WanVAECacheTransaction(wrapper).begin()
        wrapper.model._conv_idx = [5]

        tx.rollback()

        self.assertIs(wrapper.model._conv_idx, original)

    def test_rollback_restores_conv_idx_original_content(self) -> None:
        wrapper = make_wrapper()
        wrapper.model._conv_idx[:] = [3, 4]
        tx = WanVAECacheTransaction(wrapper).begin()
        wrapper.model._conv_idx = [8]

        tx.rollback()

        self.assertEqual(wrapper.model._conv_idx, [3, 4])

    def test_rollback_restores_feat_map_original_list_identity(self) -> None:
        wrapper = make_wrapper(feat_map=[None, torch.tensor([1.0])])
        original = wrapper.model._feat_map
        tx = WanVAECacheTransaction(wrapper).begin()
        wrapper.model._feat_map = [torch.tensor([9.0])]

        tx.rollback()

        self.assertIs(wrapper.model._feat_map, original)

    def test_rollback_restores_all_original_entry_references(self) -> None:
        tensor = torch.tensor([1.0])
        sentinel = "Rep"
        wrapper = make_wrapper(feat_map=[None, tensor, sentinel])
        original_entries = tuple(wrapper.model._feat_map)
        tx = WanVAECacheTransaction(wrapper).begin()
        mutate_like_cached_decode(wrapper.model)

        tx.rollback()

        for restored, original in zip(wrapper.model._feat_map, original_entries):
            self.assertIs(restored, original)

    def test_original_tensor_entry_is_restored_by_identity_not_clone(self) -> None:
        tensor = torch.tensor([1.0])
        wrapper = make_wrapper(feat_map=[tensor])
        tx = WanVAECacheTransaction(wrapper).begin()
        wrapper.model._feat_map[0] = tensor.clone()

        tx.rollback()

        self.assertIs(wrapper.model._feat_map[0], tensor)
        self.assertEqual(wrapper.model._feat_map[0].data_ptr(), tensor.data_ptr())

    def test_transaction_complete_keeps_modifications(self) -> None:
        wrapper = make_wrapper(feat_map=[None])
        tx = WanVAECacheTransaction(wrapper).begin()
        replacement = torch.tensor([5.0])
        wrapper.model._conv_idx = [9]
        wrapper.model._feat_map[0] = replacement

        tx.complete()

        self.assertEqual(tx.state, WanVAECacheTransaction.COMPLETED)
        self.assertEqual(wrapper.model._conv_idx, [9])
        self.assertIs(wrapper.model._feat_map[0], replacement)

    def test_complete_clears_snapshot_reference(self) -> None:
        tx = WanVAECacheTransaction(make_wrapper()).begin()

        self.assertIsNotNone(tx._snapshot)
        tx.complete()

        self.assertIsNone(tx._snapshot)
        self.assertIsNone(tx._model_id)

    def test_complete_releases_old_cache_tensor_for_gc_without_other_refs(self) -> None:
        tensor = torch.tensor([1.0])
        tensor_ref = weakref.ref(tensor)
        wrapper = make_wrapper(feat_map=[tensor])
        tx = WanVAECacheTransaction(wrapper).begin()
        wrapper.model._feat_map[0] = torch.tensor([2.0])

        del tensor
        gc.collect()
        self.assertIsNotNone(tensor_ref())

        tx.complete()
        gc.collect()

        self.assertIsNone(tensor_ref())

    def test_context_normal_exit_without_complete_rolls_back(self) -> None:
        tensor = torch.tensor([1.0])
        wrapper = make_wrapper(feat_map=[tensor])
        original_conv_idx = wrapper.model._conv_idx
        original_feat_map = wrapper.model._feat_map

        with WanVAECacheTransaction(wrapper):
            mutate_like_cached_decode(wrapper.model)

        self.assertIs(wrapper.model._conv_idx, original_conv_idx)
        self.assertIs(wrapper.model._feat_map, original_feat_map)
        self.assertIs(wrapper.model._feat_map[0], tensor)

    def test_context_body_exception_rolls_back_and_reraises_original(self) -> None:
        tensor = torch.tensor([1.0])
        wrapper = make_wrapper(feat_map=[tensor])

        with self.assertRaisesRegex(ValueError, "body failed"):
            with WanVAECacheTransaction(wrapper):
                mutate_like_cached_decode(wrapper.model)
                raise ValueError("body failed")

        self.assertIs(wrapper.model._feat_map[0], tensor)

    def test_body_exception_plus_restore_exception_preserves_both_and_poisons_model(self) -> None:
        wrapper = FakeWrapper(FakeModel(conv_idx=FailingSliceList([0]), feat_map=[None]))
        original_model = wrapper.model

        try:
            with self.assertRaises(WanVAECacheRollbackError) as raised:
                with WanVAECacheTransaction(wrapper):
                    wrapper.model._conv_idx = [9]
                    raise ValueError("body failed")

            self.assertIsInstance(raised.exception.original_exception, ValueError)
            self.assertIsInstance(raised.exception.restore_exception, WanVAECacheRestoreError)
            self.assertIn("body failed", str(raised.exception))
            with self.assertRaisesRegex(RuntimeError, "abandoned/poisoned"):
                WanVAECacheTransaction(FakeWrapper(original_model)).begin()
        finally:
            clear_owner_record(original_model)

    def test_with_explicit_complete_does_not_rollback_on_normal_exit(self) -> None:
        tensor = torch.tensor([1.0])
        replacement = torch.tensor([5.0])
        wrapper = make_wrapper(feat_map=[tensor])

        with WanVAECacheTransaction(wrapper) as tx:
            wrapper.model._feat_map[0] = replacement
            tx.complete()

        self.assertEqual(tx.state, WanVAECacheTransaction.COMPLETED)
        self.assertIs(wrapper.model._feat_map[0], replacement)
        self.assertIsNone(tx._snapshot)

    def test_double_begin_is_rejected(self) -> None:
        wrapper = make_wrapper()
        tx = WanVAECacheTransaction(wrapper).begin()

        with self.assertRaisesRegex(RuntimeError, "cannot begin"):
            tx.begin()

        tx.rollback()

    def test_complete_before_active_is_rejected(self) -> None:
        tx = WanVAECacheTransaction(make_wrapper())

        with self.assertRaisesRegex(RuntimeError, "cannot complete"):
            tx.complete()

    def test_rollback_before_active_is_rejected(self) -> None:
        tx = WanVAECacheTransaction(make_wrapper())

        with self.assertRaisesRegex(RuntimeError, "cannot rollback"):
            tx.rollback()

    def test_double_complete_is_rejected(self) -> None:
        tx = WanVAECacheTransaction(make_wrapper()).begin()
        tx.complete()

        with self.assertRaisesRegex(RuntimeError, "cannot complete"):
            tx.complete()

    def test_double_rollback_is_rejected(self) -> None:
        tx = WanVAECacheTransaction(make_wrapper()).begin()
        tx.rollback()

        with self.assertRaisesRegex(RuntimeError, "cannot rollback"):
            tx.rollback()

    def test_same_model_nested_transaction_is_rejected(self) -> None:
        wrapper = make_wrapper()
        tx = WanVAECacheTransaction(wrapper).begin()

        with self.assertRaisesRegex(RuntimeError, "already active"):
            WanVAECacheTransaction(wrapper).begin()

        tx.rollback()

    def test_different_model_transactions_can_be_active_together(self) -> None:
        left = make_wrapper()
        right = make_wrapper()
        left_tx = WanVAECacheTransaction(left).begin()
        right_tx = WanVAECacheTransaction(right).begin()

        self.assertTrue(left_tx.is_active)
        self.assertTrue(right_tx.is_active)

        right_tx.rollback()
        left_tx.rollback()

    def test_capture_failure_does_not_leave_active_owner(self) -> None:
        wrapper = make_wrapper(feat_map=[object()])

        with self.assertRaisesRegex(TypeError, "_feat_map\\[0\\]"):
            WanVAECacheTransaction(wrapper).begin()

        wrapper.model._feat_map = [None]
        tx = WanVAECacheTransaction(wrapper).begin()
        tx.rollback()

    def test_begin_weakref_failure_is_exception_atomic(self) -> None:
        model = NonWeakrefFakeModel()
        wrapper = FakeWrapper(model)
        tx = WanVAECacheTransaction(wrapper)

        with self.assertRaisesRegex(TypeError, "weak references"):
            tx.begin()

        self.assertEqual(tx.state, WanVAECacheTransaction.NEW)
        self.assertIsNone(tx._snapshot)
        self.assertIsNone(tx._model_id)
        with transaction_module._ACTIVE_OWNER_LOCK:
            self.assertNotIn(id(model), transaction_module._ACTIVE_OWNERS)

        wrapper.model = FakeModel()
        tx.begin()
        tx.rollback()

    def test_gc_abandoned_active_transaction_poisons_live_model(self) -> None:
        wrapper = make_wrapper()
        tx = WanVAECacheTransaction(wrapper).begin()
        model_id = id(wrapper.model)
        mutate_like_cached_decode(wrapper.model)
        tx_ref = weakref.ref(tx)

        try:
            del tx
            gc.collect()

            self.assertIsNone(tx_ref())
            with self.assertRaisesRegex(RuntimeError, "abandoned/poisoned"):
                WanVAECacheTransaction(wrapper).begin()
        finally:
            with transaction_module._ACTIVE_OWNER_LOCK:
                transaction_module._ACTIVE_OWNERS.pop(model_id, None)

    def test_rollback_failure_poisons_live_model(self) -> None:
        wrapper = make_wrapper()
        original_model = wrapper.model
        tx = WanVAECacheTransaction(wrapper).begin()
        wrapper.model = FakeModel()

        try:
            with self.assertRaises(WanVAECacheRestoreError):
                tx.rollback()

            self.assertEqual(tx.state, WanVAECacheTransaction.FAILED)
            self.assertIsNone(tx._snapshot)
            with self.assertRaisesRegex(RuntimeError, "abandoned/poisoned"):
                WanVAECacheTransaction(FakeWrapper(original_model)).begin()
        finally:
            clear_owner_record(original_model)

    def test_rollback_failure_then_transaction_gc_keeps_live_model_poisoned(self) -> None:
        wrapper = make_wrapper()
        original_model = wrapper.model
        tx = WanVAECacheTransaction(wrapper).begin()
        wrapper.model = FakeModel()

        try:
            with self.assertRaises(WanVAECacheRestoreError):
                tx.rollback()
            tx_ref = weakref.ref(tx)
            del tx
            gc.collect()

            self.assertIsNone(tx_ref())
            with self.assertRaisesRegex(RuntimeError, "abandoned/poisoned"):
                WanVAECacheTransaction(FakeWrapper(original_model)).begin()
        finally:
            clear_owner_record(original_model)

    def test_gc_poisoned_model_stale_record_can_be_cleared_and_other_model_is_unaffected(self) -> None:
        wrapper = make_wrapper()
        original_model = wrapper.model
        model_id = id(original_model)
        model_ref = weakref.ref(original_model)
        tx = WanVAECacheTransaction(wrapper).begin()
        wrapper.model = FakeModel()

        with self.assertRaises(WanVAECacheRestoreError):
            tx.rollback()

        del original_model
        gc.collect()

        try:
            self.assertIsNone(model_ref())
            with transaction_module._ACTIVE_OWNER_LOCK:
                transaction_module._check_owner_available_locked(model_id)
                self.assertNotIn(model_id, transaction_module._ACTIVE_OWNERS)

            other = make_wrapper()
            other_tx = WanVAECacheTransaction(other).begin()
            other_tx.rollback()
        finally:
            with transaction_module._ACTIVE_OWNER_LOCK:
                transaction_module._ACTIVE_OWNERS.pop(model_id, None)

    def test_rollback_clears_snapshot_reference(self) -> None:
        wrapper = make_wrapper()
        tx = WanVAECacheTransaction(wrapper).begin()

        tx.rollback()

        self.assertIsNone(tx._snapshot)
        self.assertIsNone(tx._model_id)

    def test_rollback_failure_clears_snapshot_reference(self) -> None:
        wrapper = make_wrapper()
        original_model = wrapper.model
        tx = WanVAECacheTransaction(wrapper).begin()
        wrapper.model = FakeModel()

        try:
            with self.assertRaises(WanVAECacheRestoreError):
                tx.rollback()

            self.assertIsNone(tx._snapshot)
            self.assertIsNone(tx._model_id)
        finally:
            clear_owner_record(original_model)

    def test_complete_allows_same_model_immediate_new_transaction(self) -> None:
        wrapper = make_wrapper()
        tx = WanVAECacheTransaction(wrapper).begin()
        tx.complete()

        next_tx = WanVAECacheTransaction(wrapper).begin()
        next_tx.rollback()

    def test_rollback_allows_same_model_immediate_new_transaction(self) -> None:
        wrapper = make_wrapper()
        tx = WanVAECacheTransaction(wrapper).begin()
        tx.rollback()

        next_tx = WanVAECacheTransaction(wrapper).begin()
        next_tx.rollback()

    def test_old_transaction_delayed_release_does_not_delete_new_owner(self) -> None:
        wrapper = make_wrapper()
        old_tx = WanVAECacheTransaction(wrapper).begin()
        model_id = old_tx._model_id
        self.assertIsNotNone(model_id)

        with transaction_module._ACTIVE_OWNER_LOCK:
            transaction_module._ACTIVE_OWNERS.pop(model_id, None)

        new_tx = WanVAECacheTransaction(wrapper).begin()
        try:
            old_tx._release_owner()

            with transaction_module._ACTIVE_OWNER_LOCK:
                record = transaction_module._ACTIVE_OWNERS.get(model_id)
                self.assertIsNotNone(record)
                self.assertEqual(record.model_id, model_id)
                self.assertIs(record.model_ref(), wrapper.model)
                self.assertIs(record.transaction_ref(), new_tx)
            with self.assertRaisesRegex(RuntimeError, "already active"):
                WanVAECacheTransaction(wrapper).begin()
        finally:
            old_tx._snapshot = None
            old_tx._state = WanVAECacheTransaction.FAILED
            if new_tx.is_active:
                new_tx.rollback()

    def test_restore_errors_aggregate_multiple_steps(self) -> None:
        wrapper = FakeWrapper(
            FakeModel(
                conv_idx=FailingSliceList([0]),
                feat_map=FailingSliceList([None]),
            )
        )
        original_model = wrapper.model
        tx = WanVAECacheTransaction(wrapper).begin()
        wrapper.model._conv_idx = [7]
        wrapper.model._feat_map = [torch.tensor([1.0])]

        try:
            with self.assertRaises(WanVAECacheRestoreError) as raised:
                tx.rollback()

            self.assertEqual(len(raised.exception.errors), 2)
            with self.assertRaisesRegex(RuntimeError, "abandoned/poisoned"):
                WanVAECacheTransaction(FakeWrapper(original_model)).begin()
        finally:
            clear_owner_record(original_model)

    def test_rollback_failure_releases_snapshot_old_tensor_reference(self) -> None:
        old_tensor = torch.tensor([1.0], dtype=torch.float32)
        old_ref = weakref.ref(old_tensor)
        wrapper = make_wrapper(feat_map=[old_tensor])
        original_model = wrapper.model
        tx = WanVAECacheTransaction(wrapper).begin()
        original_model._feat_map[0] = torch.tensor([2.0], dtype=torch.float32)
        wrapper.model = FakeModel()

        del old_tensor
        gc.collect()
        self.assertIsNotNone(old_ref())

        try:
            with self.assertRaises(WanVAECacheRestoreError):
                tx.rollback()
            gc.collect()
            self.assertIsNone(old_ref())
        finally:
            clear_owner_record(original_model)

    def test_model_binding_replaced_restore_is_rejected(self) -> None:
        wrapper = make_wrapper()
        snapshot = WanVAECacheSnapshot.capture(wrapper)
        wrapper.model = FakeModel()

        with self.assertRaisesRegex(WanVAECacheRestoreError, "Wan VAE cache restore failed"):
            snapshot.restore()

    def test_non_list_conv_idx_is_rejected(self) -> None:
        wrapper = make_wrapper()
        wrapper.model._conv_idx = (0,)  # type: ignore[assignment]

        with self.assertRaisesRegex(TypeError, "_conv_idx must be list"):
            WanVAECacheSnapshot.capture(wrapper)

    def test_non_list_feat_map_is_rejected(self) -> None:
        wrapper = make_wrapper()
        wrapper.model._feat_map = (None,)  # type: ignore[assignment]

        with self.assertRaisesRegex(TypeError, "_feat_map must be list"):
            WanVAECacheSnapshot.capture(wrapper)

    def test_illegal_cache_entry_type_is_rejected(self) -> None:
        wrapper = make_wrapper(feat_map=[{"bad": "entry"}])

        with self.assertRaisesRegex(TypeError, "_feat_map\\[0\\]"):
            WanVAECacheSnapshot.capture(wrapper)

    def test_fingerprint_does_not_modify_original_state(self) -> None:
        tensor = torch.tensor([1.0])
        wrapper = make_wrapper(feat_map=[None, "Rep", tensor])
        conv_idx = wrapper.model._conv_idx
        feat_map = wrapper.model._feat_map
        entries = tuple(feat_map)

        fingerprint_wan_vae_cache(wrapper, include_digest=True)

        self.assertIs(wrapper.model._conv_idx, conv_idx)
        self.assertIs(wrapper.model._feat_map, feat_map)
        self.assertEqual(tuple(wrapper.model._feat_map), entries)

    def test_fingerprint_is_json_safe(self) -> None:
        wrapper = make_wrapper(feat_map=[None, "Rep", torch.tensor([1.0])])
        fingerprint = fingerprint_wan_vae_cache(wrapper, include_digest=True)

        json.dumps(fingerprint, allow_nan=False)

    def test_fingerprint_records_none_sentinel_and_tensor(self) -> None:
        tensor = torch.tensor([1.0])
        wrapper = make_wrapper(feat_map=[None, "Rep", tensor])

        entries = fingerprint_wan_vae_cache(wrapper)["model_attributes"]["_feat_map"]["entries"]  # type: ignore[index]

        self.assertEqual(entries[0]["kind"], "none")
        self.assertEqual(entries[1]["kind"], "sentinel")
        self.assertEqual(entries[1]["value"], "Rep")
        self.assertEqual(entries[2]["kind"], "tensor")
        self.assertEqual(entries[2]["shape"], [1])

    def test_digest_for_same_tensor_is_stable(self) -> None:
        tensor = torch.tensor([1.0, 2.0], dtype=torch.float32)
        wrapper = make_wrapper(feat_map=[tensor])

        first = fingerprint_wan_vae_cache(wrapper, include_digest=True)
        second = fingerprint_wan_vae_cache(wrapper, include_digest=True)

        self.assertEqual(tensor_digest(first), tensor_digest(second))

    def test_float16_digest_is_stable(self) -> None:
        tensor = torch.tensor([1.0, 2.0], dtype=torch.float16)
        wrapper = make_wrapper(feat_map=[tensor])

        first = fingerprint_wan_vae_cache(wrapper, include_digest=True)
        second = fingerprint_wan_vae_cache(wrapper, include_digest=True)

        self.assertEqual(tensor_digest(first), tensor_digest(second))

    def test_bfloat16_digest_is_stable(self) -> None:
        tensor = torch.tensor([1.0, 2.0], dtype=torch.bfloat16)
        wrapper = make_wrapper(feat_map=[tensor])

        first = fingerprint_wan_vae_cache(wrapper, include_digest=True)
        second = fingerprint_wan_vae_cache(wrapper, include_digest=True)

        self.assertEqual(tensor_digest(first), tensor_digest(second))

    def test_digest_changes_after_tensor_value_change(self) -> None:
        tensor = torch.tensor([1.0, 2.0], dtype=torch.float32)
        wrapper = make_wrapper(feat_map=[tensor])
        before = fingerprint_wan_vae_cache(wrapper, include_digest=True)

        tensor.add_(1.0)
        after = fingerprint_wan_vae_cache(wrapper, include_digest=True)

        self.assertNotEqual(tensor_digest(before), tensor_digest(after))

    def test_same_values_new_tensor_preserve_numerical_not_identity_equality(self) -> None:
        tensor = torch.tensor([1.0, 2.0], dtype=torch.float32)
        wrapper = make_wrapper(feat_map=[tensor])
        before = fingerprint_wan_vae_cache(wrapper, include_digest=True)

        wrapper.model._feat_map[0] = tensor.clone()
        after = fingerprint_wan_vae_cache(wrapper, include_digest=True)

        self.assertEqual(tensor_digest(before), tensor_digest(after))
        self.assertNotEqual(tensor_object_id(before), tensor_object_id(after))
        self.assertNotEqual(tensor_data_ptr(before), tensor_data_ptr(after))
        self.assertTrue(fingerprints_structurally_equal(before, after))
        self.assertTrue(fingerprints_numerically_equal(before, after))
        self.assertFalse(fingerprints_identity_equal(before, after))

    def test_rollback_restores_structural_numerical_and_identity_fingerprint(self) -> None:
        tensor = torch.tensor([1.0, 2.0], dtype=torch.float32)
        wrapper = make_wrapper(feat_map=[tensor, None])
        before = fingerprint_wan_vae_cache(wrapper, include_digest=True)
        tx = WanVAECacheTransaction(wrapper).begin()
        mutate_like_cached_decode(wrapper.model)

        tx.rollback()
        after = fingerprint_wan_vae_cache(wrapper, include_digest=True)

        self.assertTrue(fingerprints_structurally_equal(before, after))
        self.assertTrue(fingerprints_numerically_equal(before, after))
        self.assertTrue(fingerprints_identity_equal(before, after))

    def test_shallow_rollback_is_enough_when_cached_decode_only_replaces_entries(self) -> None:
        tensor = torch.tensor([1.0], dtype=torch.float32)
        wrapper = make_wrapper(feat_map=[tensor])
        tx = WanVAECacheTransaction(wrapper).begin()
        wrapper.model._feat_map[0] = torch.tensor([9.0], dtype=torch.float32)

        tx.rollback()

        self.assertIs(wrapper.model._feat_map[0], tensor)
        self.assertTrue(torch.equal(wrapper.model._feat_map[0], torch.tensor([1.0])))

    def test_in_place_old_tensor_mutation_remains_f4b2_risk_detected_by_digest(self) -> None:
        tensor = torch.tensor([1.0], dtype=torch.float32)
        wrapper = make_wrapper(feat_map=[tensor])
        before = fingerprint_wan_vae_cache(wrapper, include_digest=True)
        tx = WanVAECacheTransaction(wrapper).begin()

        tensor.add_(10.0)
        tx.rollback()
        after = fingerprint_wan_vae_cache(wrapper, include_digest=True)

        self.assertIs(wrapper.model._feat_map[0], tensor)
        self.assertTrue(torch.equal(wrapper.model._feat_map[0], torch.tensor([11.0])))
        self.assertTrue(fingerprints_structurally_equal(before, after))
        self.assertFalse(fingerprints_numerically_equal(before, after))
        self.assertTrue(fingerprints_identity_equal(before, after))


if __name__ == "__main__":
    unittest.main()
