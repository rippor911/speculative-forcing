from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from speculative.scoring import ScoreResult, freeze_metadata, require_finite_score
from speculative.types import Decision, Evaluation


@dataclass(frozen=True)
class FixedThresholdPolicy:
    """Pure policy accepting candidates with block_score >= threshold."""

    threshold: float
    policy_name: str = "fixed_threshold"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.policy_name, str) or not self.policy_name:
            raise ValueError("policy_name must be a non-empty string.")
        object.__setattr__(
            self,
            "threshold",
            require_finite_score("threshold", self.threshold),
        )
        object.__setattr__(self, "metadata", freeze_metadata(self.metadata))

    def decide(self, evaluation: Evaluation) -> Decision:
        if not isinstance(evaluation, Evaluation):
            raise TypeError("FixedThresholdPolicy.decide() requires an Evaluation.")
        if not isinstance(evaluation.value, ScoreResult):
            raise TypeError("FixedThresholdPolicy requires Evaluation.value to be ScoreResult.")

        block_score = require_finite_score("block_score", evaluation.value.block_score)
        accepted = block_score >= self.threshold
        metadata = freeze_metadata(
            {
                "policy_name": self.policy_name,
                "threshold": self.threshold,
                "block_score": block_score,
                "scorer_name": evaluation.value.scorer_name,
                "aggregator_name": evaluation.value.aggregator_name,
                "accepted": accepted,
                "policy_metadata": self.metadata,
            }
        )
        reason = (
            f"{self.policy_name}: block_score {block_score} >= threshold {self.threshold}"
            if accepted
            else f"{self.policy_name}: block_score {block_score} < threshold {self.threshold}"
        )
        if accepted:
            return Decision.accept(reason=reason, metadata=metadata)
        return Decision.reject(reason=reason, metadata=metadata)


__all__ = ["FixedThresholdPolicy"]
