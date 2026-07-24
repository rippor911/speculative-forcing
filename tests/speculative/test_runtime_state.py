from __future__ import annotations

import unittest

import torch

from speculative.adapters.runtime_state import (
    ObjectStateSnapshot,
    ObjectStateSpec,
    RuntimeStateRestoreError,
    RuntimeStateRollbackError,
    RuntimeStateSnapshot,
    RuntimeStateTransactionManager,
    TensorRegionSnapshot,
    TensorRegionSpec,
    TensorValueSnapshot,
)


class RuntimeStateTest(unittest.TestCase):
    def test_direct_append_region_rolls_back(self) -> None:
        cache = torch.arange(16, dtype=torch.float32).reshape(2, 8)
        original = cache.clone()
        manager = RuntimeStateTransactionManager(
            tensor_regions=[TensorRegionSpec(cache, dim=1, start=4, end=6)]
        )

        tx = manager.begin()
        cache[:, 4:6] = 100
        tx.rollback()

        self.assertTrue(torch.equal(cache, original))

    def test_direct_overwrite_region_rolls_back(self) -> None:
        cache = torch.arange(24, dtype=torch.float32).reshape(3, 8)
        original = cache.clone()
        manager = RuntimeStateTransactionManager(
            tensor_regions=[TensorRegionSpec(cache, dim=1, start=2, end=5)]
        )

        tx = manager.begin()
        cache[:, 2:5] = torch.tensor([[-1.0, -2.0, -3.0]]).expand(3, 3)
        tx.rollback()

        self.assertTrue(torch.equal(cache, original))

    def test_simulated_local_roll_region_rolls_back(self) -> None:
        cache = torch.arange(10, dtype=torch.float32).reshape(1, 10)
        original = cache.clone()
        manager = RuntimeStateTransactionManager(
            tensor_regions=[TensorRegionSpec(cache, dim=1, start=1, end=8)]
        )

        tx = manager.begin()
        cache[:, 1:6] = cache[:, 3:8].clone()
        cache[:, 6:8] = torch.tensor([[101.0, 102.0]])
        tx.rollback()

        self.assertTrue(torch.equal(cache, original))

    def test_global_local_index_tensor_restore_preserves_objects(self) -> None:
        global_end_index = torch.tensor([6], dtype=torch.long)
        local_end_index = torch.tensor([6], dtype=torch.long)
        global_ref = global_end_index
        local_ref = local_end_index
        manager = RuntimeStateTransactionManager(
            tensor_values=[global_end_index, local_end_index]
        )

        tx = manager.begin()
        global_end_index.fill_(12)
        local_end_index.fill_(10)
        tx.rollback()

        self.assertIs(global_end_index, global_ref)
        self.assertIs(local_end_index, local_ref)
        self.assertEqual(global_end_index.item(), 6)
        self.assertEqual(local_end_index.item(), 6)

    def test_output_slice_restore(self) -> None:
        output = torch.arange(30, dtype=torch.float32).reshape(1, 5, 6)
        original = output.clone()
        manager = RuntimeStateTransactionManager(
            tensor_regions=[TensorRegionSpec(output, dim=1, start=2, end=4)]
        )

        tx = manager.begin()
        output[:, 2:4] = -5
        tx.rollback()

        self.assertTrue(torch.equal(output, original))

    def test_object_ordering_state_restore_uses_independent_copy(self) -> None:
        box = {
            "state": {
                "committed": [0, 1],
                "seen": {0, 1},
                "meta": {"next_block": 2},
            }
        }
        original = {
            "committed": [0, 1],
            "seen": {0, 1},
            "meta": {"next_block": 2},
        }
        manager = RuntimeStateTransactionManager(
            object_states=[
                ObjectStateSpec(
                    getter=lambda: box["state"],
                    setter=lambda value: box.__setitem__("state", value),
                )
            ]
        )

        tx = manager.begin()
        box["state"]["committed"].append(2)
        box["state"]["seen"].add(2)
        box["state"]["meta"]["next_block"] = 3
        tx.rollback()

        self.assertEqual(box["state"], original)
        self.assertIsNot(box["state"], original)
        box["state"]["committed"].append(9)
        self.assertEqual(original["committed"], [0, 1])

    def test_object_state_spec_rejects_tensor_backed_state(self) -> None:
        tensor = torch.zeros(1)
        cycle: list[object] = []
        cycle.append(cycle)
        cycle.append({"tensor": tensor})
        cases = [
            tensor,
            [tensor],
            ({"nested": tensor},),
            {"key": tensor},
            {("value", tensor)},
            cycle,
        ]

        for value in cases:
            with self.subTest(value_type=type(value).__name__):
                manager = RuntimeStateTransactionManager(
                    object_states=[
                        ObjectStateSpec(
                            getter=lambda value=value: value,
                            setter=lambda restored: None,
                        )
                    ]
                )
                with self.assertRaisesRegex(TypeError, "tensor-backed"):
                    manager.begin()

    def test_identity_copy_fn_does_not_alias_source_state(self) -> None:
        box = {"state": {"order": [1, 2]}}
        manager = RuntimeStateTransactionManager(
            object_states=[
                ObjectStateSpec(
                    getter=lambda: box["state"],
                    setter=lambda value: box.__setitem__("state", value),
                    copy_fn=lambda value: value,
                )
            ]
        )

        tx = manager.begin()
        box["state"]["order"].append(3)
        tx.rollback()

        self.assertEqual(box["state"], {"order": [1, 2]})

    def test_identity_copy_fn_cannot_expose_internal_backup(self) -> None:
        box = {"state": {"order": [1, 2]}}
        snapshot = ObjectStateSnapshot.capture(
            ObjectStateSpec(
                getter=lambda: box["state"],
                setter=lambda value: box.__setitem__("state", value),
                copy_fn=lambda value: value,
            )
        )

        exposed = snapshot.value
        exposed["order"].append(99)
        box["state"]["order"].append(3)
        snapshot.restore()

        self.assertEqual(box["state"], {"order": [1, 2]})

    def test_cpu_rng_restore(self) -> None:
        torch.manual_seed(123)
        expected = torch.rand(5)
        torch.manual_seed(123)
        manager = RuntimeStateTransactionManager(capture_rng=True)

        tx = manager.begin()
        torch.rand(17)
        tx.rollback()
        actual = torch.rand(5)

        self.assertTrue(torch.equal(actual, expected))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_cuda_rng_restore(self) -> None:
        torch.cuda.manual_seed_all(321)
        expected = torch.rand(5, device="cuda")
        torch.cuda.manual_seed_all(321)
        manager = RuntimeStateTransactionManager(capture_rng=True, capture_cuda_rng=True)

        tx = manager.begin()
        torch.rand(17, device="cuda")
        tx.rollback()
        actual = torch.rand(5, device="cuda")

        self.assertTrue(torch.equal(actual.cpu(), expected.cpu()))

    def test_complete_keeps_modifications(self) -> None:
        cache = torch.zeros(1, 4)
        manager = RuntimeStateTransactionManager(
            tensor_regions=[TensorRegionSpec(cache, dim=1, start=1, end=3)]
        )

        tx = manager.begin()
        cache[:, 1:3] = 7
        tx.complete()

        self.assertFalse(manager.is_active)
        self.assertTrue(torch.equal(cache, torch.tensor([[0.0, 7.0, 7.0, 0.0]])))

    def test_context_manager_exception_exit_rolls_back(self) -> None:
        cache = torch.arange(5, dtype=torch.float32)
        original = cache.clone()
        manager = RuntimeStateTransactionManager(
            tensor_regions=[TensorRegionSpec(cache, dim=0, start=1, end=4)]
        )

        with self.assertRaisesRegex(RuntimeError, "temporary failure"):
            with manager.transaction():
                cache[1:4] = -1
                raise RuntimeError("temporary failure")

        self.assertFalse(manager.is_active)
        self.assertTrue(torch.equal(cache, original))

    def test_context_manager_normal_exit_without_complete_rolls_back(self) -> None:
        cache = torch.arange(5, dtype=torch.float32)
        original = cache.clone()
        manager = RuntimeStateTransactionManager(
            tensor_regions=[TensorRegionSpec(cache, dim=0, start=1, end=4)]
        )

        with manager.transaction():
            cache[1:4] = 99

        self.assertFalse(manager.is_active)
        self.assertTrue(torch.equal(cache, original))

    def test_same_transaction_context_reentry_is_rejected(self) -> None:
        cache = torch.zeros(4)
        manager = RuntimeStateTransactionManager(
            tensor_regions=[TensorRegionSpec(cache, dim=0, start=0, end=2)]
        )
        tx = manager.begin()

        self.assertIs(tx.__enter__(), tx)
        with self.assertRaisesRegex(RuntimeError, "already been entered"):
            tx.__enter__()
        tx.rollback()
        with self.assertRaisesRegex(RuntimeError, "already closed"):
            tx.__enter__()

        tx2 = manager.begin()
        tx2.complete()
        self.assertFalse(manager.is_active)

    def test_nested_transaction_is_rejected(self) -> None:
        cache = torch.zeros(4)
        manager = RuntimeStateTransactionManager(
            tensor_regions=[TensorRegionSpec(cache, dim=0, start=0, end=2)]
        )

        with manager.transaction():
            with self.assertRaisesRegex(RuntimeError, "already active"):
                manager.begin()

        self.assertFalse(manager.is_active)

    def test_double_complete_and_rollback_are_rejected(self) -> None:
        cache = torch.zeros(4)
        manager = RuntimeStateTransactionManager(
            tensor_regions=[TensorRegionSpec(cache, dim=0, start=0, end=2)]
        )

        tx = manager.begin()
        tx.complete()
        with self.assertRaisesRegex(RuntimeError, "already closed"):
            tx.complete()
        with self.assertRaisesRegex(RuntimeError, "already closed"):
            tx.rollback()

        tx2 = manager.begin()
        tx2.rollback()
        with self.assertRaisesRegex(RuntimeError, "already closed"):
            tx2.rollback()
        with self.assertRaisesRegex(RuntimeError, "already closed"):
            tx2.complete()

    def test_capture_data_does_not_alias_source_tensor(self) -> None:
        tensor = torch.arange(6, dtype=torch.float32)
        region_spec = TensorRegionSpec(tensor, dim=0, start=1, end=4)
        region_snapshot = TensorRegionSnapshot.capture(region_spec)
        value_snapshot = TensorValueSnapshot.capture(tensor)

        self.assertNotEqual(region_snapshot.value.data_ptr(), tensor[1:4].data_ptr())
        self.assertNotEqual(value_snapshot.value.data_ptr(), tensor.data_ptr())
        tensor[1:4] = -10

        self.assertTrue(torch.equal(region_snapshot.value, torch.tensor([1.0, 2.0, 3.0])))
        self.assertTrue(torch.equal(value_snapshot.value, torch.arange(6, dtype=torch.float32)))

    def test_dtype_change_is_rejected_for_full_tensor_snapshot(self) -> None:
        tensor = torch.arange(4, dtype=torch.float32)
        manager = RuntimeStateTransactionManager(tensor_values=[tensor])

        tx = manager.begin()
        tensor.data = torch.arange(4, dtype=torch.int64)
        with self.assertRaises(RuntimeStateRestoreError) as raised:
            tx.rollback()

        self.assertIn("dtype changed", str(raised.exception.errors[0]))
        self.assertFalse(manager.is_active)

    def test_dtype_change_is_rejected_for_region_snapshot(self) -> None:
        tensor = torch.arange(4, dtype=torch.float32)
        manager = RuntimeStateTransactionManager(
            tensor_regions=[TensorRegionSpec(tensor, dim=0, start=1, end=3)]
        )

        tx = manager.begin()
        tensor.data = torch.arange(4, dtype=torch.int64)
        with self.assertRaises(RuntimeStateRestoreError) as raised:
            tx.rollback()

        self.assertIn("dtype changed", str(raised.exception.errors[0]))
        self.assertFalse(manager.is_active)

    def test_source_tensor_shape_change_is_rejected(self) -> None:
        tensor = torch.arange(6, dtype=torch.float32)
        manager = RuntimeStateTransactionManager(
            tensor_regions=[TensorRegionSpec(tensor, dim=0, start=2, end=5)]
        )

        tx = manager.begin()
        tensor.resize_(2, 3)
        with self.assertRaises(RuntimeStateRestoreError) as raised:
            tx.rollback()

        self.assertIn("shape changed", str(raised.exception.errors[0]))
        self.assertFalse(manager.is_active)

    def test_exposed_backup_mutation_cannot_corrupt_restore(self) -> None:
        tensor = torch.arange(5, dtype=torch.float32)
        tensor_snapshot = TensorValueSnapshot.capture(tensor)
        exposed_tensor_backup = tensor_snapshot.value
        exposed_tensor_backup.fill_(-100)
        tensor.fill_(7)
        tensor_snapshot.restore()
        self.assertTrue(torch.equal(tensor, torch.arange(5, dtype=torch.float32)))

        box = {"state": {"order": [1, 2]}}
        object_snapshot = ObjectStateSnapshot.capture(
            ObjectStateSpec(
                getter=lambda: box["state"],
                setter=lambda value: box.__setitem__("state", value),
            )
        )
        exposed_object_backup = object_snapshot.value
        exposed_object_backup["order"].append(99)
        box["state"]["order"].append(3)
        object_snapshot.restore()
        self.assertEqual(box["state"], {"order": [1, 2]})

    def test_invalid_regions_and_integer_types_are_rejected(self) -> None:
        tensor = torch.zeros(4, 4)
        invalid_values = [True, 1.0, "1"]

        for value in invalid_values:
            with self.subTest(field="dim", value=value):
                with self.assertRaisesRegex(ValueError, "integer"):
                    TensorRegionSpec(tensor, dim=value, start=0, end=1)  # type: ignore[arg-type]
            with self.subTest(field="start", value=value):
                with self.assertRaisesRegex(ValueError, "integer"):
                    TensorRegionSpec(tensor, dim=0, start=value, end=1)  # type: ignore[arg-type]
            with self.subTest(field="end", value=value):
                with self.assertRaisesRegex(ValueError, "integer"):
                    TensorRegionSpec(tensor, dim=0, start=0, end=value)  # type: ignore[arg-type]

        invalid_specs = [
            {"dim": 2, "start": 0, "end": 1},
            {"dim": 0, "start": -1, "end": 1},
            {"dim": 0, "start": 2, "end": 2},
            {"dim": 0, "start": 2, "end": 1},
            {"dim": 0, "start": 0, "end": 5},
        ]
        for kwargs in invalid_specs:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    TensorRegionSpec(tensor, **kwargs)

        with self.assertRaises(TypeError):
            TensorRegionSpec("not a tensor", dim=0, start=0, end=1)  # type: ignore[arg-type]

    def test_region_capture_revalidates_bounds_at_begin(self) -> None:
        tensor = torch.arange(6, dtype=torch.float32)
        spec = TensorRegionSpec(tensor, dim=0, start=2, end=5)
        manager = RuntimeStateTransactionManager(tensor_regions=[spec])
        tensor.resize_(3)

        with self.assertRaisesRegex(RuntimeError, "invalid"):
            manager.begin()

        self.assertFalse(manager.is_active)

    def test_full_and_region_snapshot_conflict_is_rejected(self) -> None:
        tensor = torch.zeros(6)

        with self.assertRaisesRegex(ValueError, "both full and region"):
            RuntimeStateTransactionManager(
                tensor_regions=[TensorRegionSpec(tensor, dim=0, start=1, end=3)],
                tensor_values=[tensor],
            )

    def test_direct_snapshot_capture_rejects_conflicting_specs(self) -> None:
        tensor = torch.zeros(6)

        with self.assertRaisesRegex(ValueError, "both full and region"):
            RuntimeStateSnapshot.capture(
                tensor_regions=[TensorRegionSpec(tensor, dim=0, start=1, end=3)],
                tensor_values=[tensor],
            )

    def test_duplicate_full_tensor_snapshot_is_rejected(self) -> None:
        tensor = torch.zeros(6)

        with self.assertRaisesRegex(ValueError, "more than once"):
            RuntimeStateTransactionManager(tensor_values=[tensor, tensor])

    def test_overlapping_region_snapshots_are_rejected(self) -> None:
        tensor = torch.zeros(8)

        with self.assertRaisesRegex(ValueError, "overlapping"):
            RuntimeStateTransactionManager(
                tensor_regions=[
                    TensorRegionSpec(tensor, dim=0, start=1, end=4),
                    TensorRegionSpec(tensor, dim=0, start=3, end=5),
                ]
            )

    def test_same_tensor_different_dim_regions_are_rejected(self) -> None:
        tensor = torch.zeros(4, 4)

        with self.assertRaisesRegex(ValueError, "different dims"):
            RuntimeStateTransactionManager(
                tensor_regions=[
                    TensorRegionSpec(tensor, dim=0, start=1, end=3),
                    TensorRegionSpec(tensor, dim=1, start=1, end=3),
                ]
            )

    def test_disjoint_same_dim_region_snapshots_are_allowed(self) -> None:
        tensor = torch.arange(8, dtype=torch.float32)
        original = tensor.clone()
        manager = RuntimeStateTransactionManager(
            tensor_regions=[
                TensorRegionSpec(tensor, dim=0, start=1, end=3),
                TensorRegionSpec(tensor, dim=0, start=5, end=7),
            ]
        )

        tx = manager.begin()
        tensor[1:3] = -1
        tensor[5:7] = -2
        tx.rollback()

        self.assertTrue(torch.equal(tensor, original))

    def test_outside_region_modifications_are_not_overwritten(self) -> None:
        tensor = torch.arange(6, dtype=torch.float32)
        manager = RuntimeStateTransactionManager(
            tensor_regions=[TensorRegionSpec(tensor, dim=0, start=2, end=4)]
        )

        tx = manager.begin()
        tensor[0:2] = -1
        tensor[2:4] = 50
        tx.rollback()

        self.assertTrue(torch.equal(tensor, torch.tensor([-1.0, -1.0, 2.0, 3.0, 4.0, 5.0])))

    def test_restore_continues_after_one_failure(self) -> None:
        failing = torch.arange(4, dtype=torch.float32)
        restored = torch.arange(6, dtype=torch.float32)
        restored_original = restored.clone()
        manager = RuntimeStateTransactionManager(
            tensor_regions=[
                TensorRegionSpec(failing, dim=0, start=0, end=2),
                TensorRegionSpec(restored, dim=0, start=2, end=5),
            ]
        )

        tx = manager.begin()
        failing.data = torch.arange(4, dtype=torch.int64)
        restored[2:5] = -10
        with self.assertRaises(RuntimeStateRestoreError) as raised:
            tx.rollback()

        self.assertEqual(len(raised.exception.errors), 1)
        self.assertTrue(torch.equal(restored, restored_original))
        self.assertFalse(manager.is_active)

    def test_rng_restore_is_attempted_after_earlier_failure(self) -> None:
        torch.manual_seed(987)
        expected = torch.rand(4)
        torch.manual_seed(987)
        tensor = torch.arange(4, dtype=torch.float32)
        manager = RuntimeStateTransactionManager(tensor_values=[tensor], capture_rng=True)

        tx = manager.begin()
        torch.rand(11)
        tensor.data = torch.arange(4, dtype=torch.int64)
        with self.assertRaises(RuntimeStateRestoreError):
            tx.rollback()
        actual = torch.rand(4)

        self.assertTrue(torch.equal(actual, expected))
        self.assertFalse(manager.is_active)

    def test_body_exception_and_rollback_exception_are_preserved(self) -> None:
        tensor = torch.arange(4, dtype=torch.float32)
        manager = RuntimeStateTransactionManager(tensor_values=[tensor])

        with self.assertRaises(RuntimeStateRollbackError) as raised:
            with manager.transaction():
                tensor.data = torch.arange(4, dtype=torch.int64)
                raise ValueError("body failed")

        self.assertIsInstance(raised.exception.original_exception, ValueError)
        self.assertIsInstance(raised.exception.restore_exception, RuntimeStateRestoreError)
        self.assertIn("body failed", str(raised.exception))
        self.assertFalse(manager.is_active)

    def test_manager_closes_after_restore_failure(self) -> None:
        tensor = torch.arange(4, dtype=torch.float32)
        manager = RuntimeStateTransactionManager(tensor_values=[tensor])

        tx = manager.begin()
        tensor.data = torch.arange(4, dtype=torch.int64)
        with self.assertRaises(RuntimeStateRestoreError):
            tx.rollback()

        self.assertFalse(manager.is_active)
        with self.assertRaisesRegex(RuntimeError, "already closed"):
            tx.rollback()
        tx2 = manager.begin()
        tx2.rollback()

    def test_capture_failure_does_not_leave_active_transaction(self) -> None:
        tensor = torch.zeros(4)

        def fail_getter() -> dict[str, int]:
            raise RuntimeError("capture failed")

        manager = RuntimeStateTransactionManager(
            tensor_regions=[TensorRegionSpec(tensor, dim=0, start=0, end=2)],
            object_states=[
                ObjectStateSpec(
                    getter=fail_getter,
                    setter=lambda value: None,
                )
            ],
        )

        with self.assertRaisesRegex(RuntimeError, "capture failed"):
            manager.begin()

        self.assertFalse(manager.is_active)
        valid_manager = RuntimeStateTransactionManager(
            tensor_regions=[TensorRegionSpec(tensor, dim=0, start=0, end=2)]
        )
        tx = valid_manager.begin()
        self.assertTrue(valid_manager.is_active)
        tx.rollback()

    def test_runtime_snapshot_restore_order_is_tensor_region_value_object_rng(self) -> None:
        order: list[str] = []

        class RecordingRegion:
            def restore(self) -> None:
                order.append("region")

        class RecordingValue:
            def restore(self) -> None:
                order.append("value")

        class RecordingObject:
            def restore(self) -> None:
                order.append("object")

        class RecordingRNG:
            def restore(self) -> None:
                order.append("rng")

        snapshot = RuntimeStateSnapshot(
            tensor_regions=(RecordingRegion(),),  # type: ignore[arg-type]
            tensor_values=(RecordingValue(),),  # type: ignore[arg-type]
            object_states=(RecordingObject(),),  # type: ignore[arg-type]
            rng_state=RecordingRNG(),  # type: ignore[arg-type]
        )

        snapshot.restore()

        self.assertEqual(order, ["region", "value", "object", "rng"])


if __name__ == "__main__":
    unittest.main()
