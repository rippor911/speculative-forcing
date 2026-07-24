from __future__ import annotations

import json
import math
from collections.abc import Mapping as MappingABC
from collections.abc import Sequence as SequenceABC
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Optional

from speculative.types import BlockRef


TRACE_SCHEMA_VERSION = 1

TRACE_EVENT_NAMES = frozenset(
    {
        "proposal_requested",
        "proposal_ready",
        "transaction_begin",
        "evaluate",
        "evaluated",
        "decision",
        "invalidated",
        "fallback_requested",
        "fallback_ready",
        "commit",
        "transaction_complete",
        "error",
        "transaction_rollback",
        "transaction_rollback_error",
    }
)


def _is_strict_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _require_optional_int(name: str, value: Optional[int]) -> None:
    if value is not None and not _is_strict_int(value):
        raise ValueError(f"{name} must be an integer or None, got {type(value).__name__}.")


def _require_optional_str(name: str, value: Optional[str]) -> None:
    if value is not None and not isinstance(value, str):
        raise ValueError(f"{name} must be a string or None, got {type(value).__name__}.")


def _freeze_json_value(value: Any, path: str = "metadata") -> Any:
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} must be a finite float, got {value!r}.")
        return value
    if isinstance(value, MappingABC):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{path} keys must be strings, got {type(key).__name__}.")
            normalized[key] = _freeze_json_value(item, f"{path}.{key}")
        return MappingProxyType(normalized)
    if isinstance(value, SequenceABC) and not isinstance(value, (bytes, bytearray)):
        if isinstance(value, str):
            return value
        return tuple(
            _freeze_json_value(item, f"{path}[{index}]")
            for index, item in enumerate(value)
        )
    raise ValueError(
        f"{path} is not JSON-safe: {type(value).__name__}. "
        "Allowed values are None, bool, str, int, finite float, "
        "string-key mappings, and sequences."
    )


def freeze_metadata(metadata: Optional[Mapping[str, Any]]) -> Mapping[str, Any]:
    if metadata is None:
        return MappingProxyType({})
    if not isinstance(metadata, MappingABC):
        raise ValueError("metadata must be a mapping with string keys.")
    return _freeze_json_value(metadata)


def _to_json_container(value: Any) -> Any:
    if isinstance(value, MappingABC):
        return {key: _to_json_container(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_to_json_container(item) for item in value]
    if isinstance(value, list):
        return [_to_json_container(item) for item in value]
    return value


@dataclass(frozen=True)
class TraceEvent:
    """Serializable event emitted by the speculative controller."""

    sequence: int
    name: str
    block_index: Optional[int] = None
    depth: Optional[int] = None
    source: Optional[str] = None
    decision: Optional[str] = None
    reason: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not _is_strict_int(self.sequence):
            raise ValueError(f"sequence must be an integer, got {type(self.sequence).__name__}.")
        if self.sequence < 0:
            raise ValueError(f"sequence must be >= 0, got {self.sequence}.")
        _require_optional_int("block_index", self.block_index)
        _require_optional_int("depth", self.depth)
        _require_optional_str("source", self.source)
        _require_optional_str("decision", self.decision)
        _require_optional_str("reason", self.reason)
        if self.name not in TRACE_EVENT_NAMES:
            raise ValueError(f"Unknown trace event name: {self.name!r}.")
        object.__setattr__(self, "metadata", freeze_metadata(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "sequence": self.sequence,
            "name": self.name,
            "block_index": self.block_index,
            "depth": self.depth,
            "source": self.source,
            "decision": self.decision,
            "reason": self.reason,
            "metadata": _to_json_container(self.metadata),
        }
        json.dumps(payload, allow_nan=False)
        return payload


class TraceRecorder:
    """Collects deterministic trace events for one control run."""

    def __init__(self) -> None:
        self._events: list[TraceEvent] = []

    def emit(
        self,
        name: str,
        *,
        block: Optional[BlockRef] = None,
        depth: Optional[int] = None,
        source: Optional[str] = None,
        decision: Optional[str] = None,
        reason: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> TraceEvent:
        frozen_metadata = freeze_metadata(metadata)
        event = TraceEvent(
            sequence=len(self._events),
            name=name,
            block_index=None if block is None else block.index,
            depth=depth,
            source=source,
            decision=decision,
            reason=reason,
            metadata=frozen_metadata,
        )
        self._events.append(event)
        return event

    def events(self) -> tuple[TraceEvent, ...]:
        return tuple(self._events)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema_version": TRACE_SCHEMA_VERSION,
            "events": [event.to_dict() for event in self._events],
        }
        json.dumps(payload, allow_nan=False)
        return payload
