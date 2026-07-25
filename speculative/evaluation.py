from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from speculative.scoring import RawScoreResult, ScoreResult, freeze_metadata
from speculative.types import DraftCandidate, Evaluation


@dataclass(frozen=True)
class DecodedCandidate:
    """Decoder output for scorer input.

    It preserves the original `DraftCandidate` identity and carries only the
    representation needed by a scorer. It does not score, decide, fallback, or
    mutate generation state.
    """

    candidate: DraftCandidate
    payload: Any
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.candidate, DraftCandidate):
            raise TypeError("DecodedCandidate.candidate must be a DraftCandidate.")
        object.__setattr__(self, "metadata", freeze_metadata(self.metadata))


@dataclass(frozen=True)
class IdentityCandidateDecoder:
    """Checkpoint-free decoder that exposes the candidate latent unchanged."""

    decoder_name: str = "identity"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.decoder_name, str) or not self.decoder_name:
            raise ValueError("decoder_name must be a non-empty string.")
        object.__setattr__(self, "metadata", freeze_metadata(self.metadata))

    def decode(self, candidate: DraftCandidate) -> DecodedCandidate:
        if not isinstance(candidate, DraftCandidate):
            raise TypeError("candidate must be a DraftCandidate.")
        return DecodedCandidate(
            candidate=candidate,
            payload=candidate.latent,
            metadata={
                "decoder_name": self.decoder_name,
                "block_index": candidate.block.index,
                "depth": candidate.depth,
                "decoder_metadata": self.metadata,
            },
        )


@dataclass(frozen=True)
class CompositeCandidateEvaluator:
    """Runs decoder -> scorer -> aggregator and returns existing Evaluation."""

    decoder: Any
    scorer: Any
    aggregator: Any

    def evaluate(self, candidate: DraftCandidate) -> Evaluation:
        decoded = self.decoder.decode(candidate)
        if not isinstance(decoded, DecodedCandidate):
            raise TypeError("CandidateDecoder.decode() must return DecodedCandidate.")
        if decoded.candidate is not candidate:
            raise ValueError("DecodedCandidate.candidate must be the current candidate.")

        raw_scores = self.scorer.score(decoded)
        if not isinstance(raw_scores, RawScoreResult):
            raise TypeError("CandidateScorer.score() must return RawScoreResult.")

        block_score = self.aggregator.aggregate(raw_scores.per_frame_scores)
        aggregator_name = getattr(self.aggregator, "name", type(self.aggregator).__name__)
        score_result = ScoreResult(
            per_frame_scores=raw_scores.per_frame_scores,
            block_score=block_score,
            scorer_name=raw_scores.scorer_name,
            aggregator_name=aggregator_name,
            metadata={
                "decoded": decoded.metadata,
                "raw_score": raw_scores.metadata,
            },
        )
        return Evaluation(
            candidate=candidate,
            value=score_result,
            metadata={
                "block_score": score_result.block_score,
                "per_frame_scores": score_result.per_frame_scores,
                "scorer_name": score_result.scorer_name,
                "aggregator_name": score_result.aggregator_name,
            },
        )


__all__ = [
    "CompositeCandidateEvaluator",
    "DecodedCandidate",
    "IdentityCandidateDecoder",
]
