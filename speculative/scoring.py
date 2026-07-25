from __future__ import annotations

import math
from collections.abc import Mapping as MappingABC
from collections.abc import Sequence as SequenceABC
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Sequence


def _is_finite_score(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def require_finite_score(name: str, value: object) -> float:
    if not _is_finite_score(value):
        raise ValueError(f"{name} must be a finite int or float, got {value!r}.")
    return float(value)


def _freeze_metadata_value(value: Any, path: str) -> Any:
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} must be finite, got {value!r}.")
        return value
    if isinstance(value, MappingABC):
        frozen: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{path} keys must be strings.")
            frozen[key] = _freeze_metadata_value(item, f"{path}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, SequenceABC) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(
            _freeze_metadata_value(item, f"{path}[{index}]")
            for index, item in enumerate(value)
        )
    raise ValueError(f"{path} must be metadata-safe, got {type(value).__name__}.")


def freeze_metadata(metadata: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if metadata is None:
        return MappingProxyType({})
    if not isinstance(metadata, MappingABC):
        raise ValueError("metadata must be a mapping.")
    return _freeze_metadata_value(metadata, "metadata")


def normalize_score_sequence(scores: Sequence[float], *, name: str = "scores") -> tuple[float, ...]:
    if isinstance(scores, (str, bytes, bytearray)) or not isinstance(scores, SequenceABC):
        raise ValueError(f"{name} must be a non-empty sequence of finite scores.")
    normalized = tuple(
        require_finite_score(f"{name}[{index}]", score)
        for index, score in enumerate(scores)
    )
    if not normalized:
        raise ValueError(f"{name} must not be empty.")
    return normalized


def _require_name(field_name: str, value: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a non-empty string.")


@dataclass(frozen=True)
class RawScoreResult:
    """Raw per-frame scores produced by a scorer before aggregation."""

    per_frame_scores: Sequence[float]
    scorer_name: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_name("scorer_name", self.scorer_name)
        object.__setattr__(
            self,
            "per_frame_scores",
            normalize_score_sequence(self.per_frame_scores, name="per_frame_scores"),
        )
        object.__setattr__(self, "metadata", freeze_metadata(self.metadata))


@dataclass(frozen=True)
class ScoreResult:
    """Aggregated block score carried inside the existing Evaluation object."""

    per_frame_scores: Sequence[float]
    block_score: float
    scorer_name: str
    aggregator_name: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_name("scorer_name", self.scorer_name)
        _require_name("aggregator_name", self.aggregator_name)
        object.__setattr__(
            self,
            "per_frame_scores",
            normalize_score_sequence(self.per_frame_scores, name="per_frame_scores"),
        )
        object.__setattr__(
            self,
            "block_score",
            require_finite_score("block_score", self.block_score),
        )
        object.__setattr__(self, "metadata", freeze_metadata(self.metadata))


@dataclass(frozen=True)
class MinScoreAggregator:
    """Aggregates frame scores by taking the worst frame score."""

    name: str = "min_frame"

    def aggregate(self, scores: Sequence[float]) -> float:
        return min(normalize_score_sequence(scores))


@dataclass(frozen=True)
class MeanScoreAggregator:
    """Aggregates frame scores by arithmetic mean."""

    name: str = "mean_frame"

    def aggregate(self, scores: Sequence[float]) -> float:
        normalized = normalize_score_sequence(scores)
        return require_finite_score("mean score", sum(normalized) / len(normalized))


@dataclass(frozen=True)
class ScriptedCandidateScorer:
    """Checkpoint-free scorer that returns explicit per-depth score sequences."""

    scores_by_depth: Mapping[int, Sequence[float]]
    scorer_name: str = "scripted"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_name("scorer_name", self.scorer_name)
        if not isinstance(self.scores_by_depth, MappingABC):
            raise ValueError("scores_by_depth must be a mapping from depth to scores.")
        normalized: dict[int, tuple[float, ...]] = {}
        for depth, scores in self.scores_by_depth.items():
            if not isinstance(depth, int) or isinstance(depth, bool) or depth <= 0:
                raise ValueError("scores_by_depth keys must be positive integer depths.")
            normalized[depth] = normalize_score_sequence(
                scores,
                name=f"scores_by_depth[{depth}]",
            )
        object.__setattr__(self, "scores_by_depth", MappingProxyType(normalized))
        object.__setattr__(self, "metadata", freeze_metadata(self.metadata))

    def score(self, decoded: Any) -> RawScoreResult:
        candidate = getattr(decoded, "candidate", None)
        depth = getattr(candidate, "depth", None)
        try:
            scores = self.scores_by_depth[depth]
        except KeyError as exc:
            raise ValueError(f"No scripted scores configured for depth {depth!r}.") from exc
        return RawScoreResult(
            per_frame_scores=scores,
            scorer_name=self.scorer_name,
            metadata={
                "depth": depth,
                "scripted": True,
                "scorer_metadata": self.metadata,
            },
        )


__all__ = [
    "MeanScoreAggregator",
    "MinScoreAggregator",
    "RawScoreResult",
    "ScoreResult",
    "ScriptedCandidateScorer",
    "freeze_metadata",
    "normalize_score_sequence",
    "require_finite_score",
]
