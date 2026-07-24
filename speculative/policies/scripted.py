from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from speculative.types import Decision, Evaluation


ScriptedPolicyMode = Literal["always_accept", "always_reject", "reject_at_depth"]


@dataclass(frozen=True)
class ScriptedPolicy:
    """Deterministic policy for controller tests and early integration."""

    mode: ScriptedPolicyMode
    reject_depth: Optional[int] = None

    def __post_init__(self) -> None:
        if self.mode not in ("always_accept", "always_reject", "reject_at_depth"):
            raise ValueError(f"Unsupported scripted policy mode: {self.mode!r}.")
        if self.mode == "reject_at_depth":
            if not isinstance(self.reject_depth, int) or isinstance(self.reject_depth, bool):
                raise ValueError("reject_at_depth requires reject_depth to be an int.")
            if self.reject_depth <= 0:
                raise ValueError("reject_at_depth requires reject_depth > 0.")
        elif self.reject_depth is not None:
            raise ValueError(f"{self.mode} does not accept reject_depth.")

    @classmethod
    def always_accept(cls) -> "ScriptedPolicy":
        return cls("always_accept")

    @classmethod
    def always_reject(cls) -> "ScriptedPolicy":
        return cls("always_reject")

    @classmethod
    def reject_at_depth(cls, depth: int) -> "ScriptedPolicy":
        return cls("reject_at_depth", reject_depth=depth)

    def decide(self, evaluation: Evaluation) -> Decision:
        depth = evaluation.candidate.depth
        if self.mode == "always_accept":
            return Decision.accept("scripted_always_accept")
        if self.mode == "always_reject":
            return Decision.reject("scripted_always_reject")
        if self.reject_depth is not None and depth >= self.reject_depth:
            return Decision.reject("scripted_reject_at_depth")
        return Decision.accept("scripted_before_reject_depth")
