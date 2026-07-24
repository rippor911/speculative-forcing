from __future__ import annotations

import json
import math
import unittest

from speculative.trace import TRACE_SCHEMA_VERSION, TraceEvent, TraceRecorder
from speculative.types import BlockRef, Decision


class TraceSchemaTest(unittest.TestCase):
    def test_recorder_assigns_sequence_numbers(self) -> None:
        recorder = TraceRecorder()

        first = recorder.emit("proposal_requested", block=BlockRef(index=0))
        second = recorder.emit("transaction_begin", block=BlockRef(index=0))

        self.assertEqual(first.sequence, 0)
        self.assertEqual(second.sequence, 1)
        self.assertEqual([event.sequence for event in recorder.events()], [0, 1])

    def test_trace_to_dict_schema(self) -> None:
        recorder = TraceRecorder()
        recorder.emit(
            "decision",
            block=BlockRef(index=1),
            depth=1,
            decision="reject",
            reason="scripted",
            metadata={"score": 0.0},
        )

        payload = recorder.to_dict()

        self.assertEqual(payload["schema_version"], TRACE_SCHEMA_VERSION)
        self.assertEqual(
            payload["events"],
            [
                {
                    "sequence": 0,
                    "name": "decision",
                    "block_index": 1,
                    "depth": 1,
                    "source": None,
                    "decision": "reject",
                    "reason": "scripted",
                    "metadata": {"score": 0.0},
                }
            ],
        )

    def test_unknown_trace_event_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown trace event"):
            TraceEvent(sequence=0, name="surprise")

    def test_non_json_trace_metadata_is_rejected(self) -> None:
        recorder = TraceRecorder()

        with self.assertRaisesRegex(ValueError, "JSON-safe"):
            recorder.emit("decision", metadata={"tensor_like": object()})
        with self.assertRaisesRegex(ValueError, "keys must be strings"):
            recorder.emit("decision", metadata={1: "bad"})
        with self.assertRaisesRegex(ValueError, "finite float"):
            recorder.emit("decision", metadata={"score": math.nan})

    def test_to_dict_is_json_serializable_with_allow_nan_false(self) -> None:
        recorder = TraceRecorder()
        recorder.emit(
            "evaluated",
            block=BlockRef(index=1),
            depth=1,
            metadata={
                "score": 1.25,
                "passed": True,
                "labels": ("draft", "safe"),
                "nested": {"count": 2, "none": None},
            },
        )

        payload = recorder.to_dict()

        json.dumps(payload, allow_nan=False)
        self.assertEqual(payload["events"][0]["metadata"]["labels"], ["draft", "safe"])

    def test_trace_metadata_is_copied_on_emit(self) -> None:
        recorder = TraceRecorder()
        metadata = {"labels": ["before"], "nested": {"score": 1.0}}

        recorder.emit("evaluated", metadata=metadata)
        metadata["labels"].append("after")
        metadata["nested"]["score"] = 2.0

        payload = recorder.to_dict()
        self.assertEqual(
            payload["events"][0]["metadata"],
            {"labels": ["before"], "nested": {"score": 1.0}},
        )

    def test_event_metadata_cannot_be_modified(self) -> None:
        recorder = TraceRecorder()
        event = recorder.emit(
            "evaluated",
            metadata={"labels": ["before"], "nested": {"score": 1.0}},
        )

        with self.assertRaises(TypeError):
            event.metadata["labels"] = ["after"]  # type: ignore[index]
        with self.assertRaises(TypeError):
            event.metadata["nested"]["score"] = 2.0  # type: ignore[index]
        with self.assertRaises(AttributeError):
            event.metadata["labels"].append("after")  # type: ignore[attr-defined]

    def test_modifying_to_dict_result_does_not_affect_recorder(self) -> None:
        recorder = TraceRecorder()
        recorder.emit("evaluated", metadata={"labels": ["before"], "nested": {"score": 1.0}})

        payload = recorder.to_dict()
        payload["events"][0]["metadata"]["labels"].append("after")
        payload["events"][0]["metadata"]["nested"]["score"] = 2.0

        self.assertEqual(
            recorder.to_dict()["events"][0]["metadata"],
            {"labels": ["before"], "nested": {"score": 1.0}},
        )

    def test_multiple_to_dict_results_are_consistent(self) -> None:
        recorder = TraceRecorder()
        recorder.emit("evaluated", metadata={"labels": ["before"]})

        self.assertEqual(recorder.to_dict(), recorder.to_dict())

    def test_trace_event_integer_fields_reject_bool_float_and_str(self) -> None:
        invalid_values = [True, 1.0, "1"]

        for value in invalid_values:
            with self.subTest(field="sequence", value=value):
                with self.assertRaisesRegex(ValueError, "integer"):
                    TraceEvent(sequence=value, name="decision")  # type: ignore[arg-type]
            with self.subTest(field="block_index", value=value):
                with self.assertRaisesRegex(ValueError, "integer"):
                    TraceEvent(sequence=0, name="decision", block_index=value)  # type: ignore[arg-type]
            with self.subTest(field="depth", value=value):
                with self.assertRaisesRegex(ValueError, "integer"):
                    TraceEvent(sequence=0, name="decision", depth=value)  # type: ignore[arg-type]

    def test_trace_event_string_fields_reject_non_strings(self) -> None:
        with self.assertRaisesRegex(ValueError, "source must be a string"):
            TraceEvent(sequence=0, name="commit", source=1)  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "decision must be a string"):
            TraceEvent(sequence=0, name="decision", decision=1)  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "reason must be a string"):
            TraceEvent(sequence=0, name="decision", reason=1)  # type: ignore[arg-type]

    def test_decision_reason_must_be_string(self) -> None:
        with self.assertRaisesRegex(ValueError, "reason must be a string"):
            Decision.accept(reason=1)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
