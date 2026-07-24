"""Model-free speculative decoding control core."""

from speculative.controller import SpeculativeController
from speculative.factory import create_policy
from speculative.types import (
    BlockRef,
    CommitRequest,
    ControlRequest,
    ControlResult,
    Decision,
    DraftCandidate,
    Evaluation,
    FallbackResult,
    ProposalBatch,
    SpeculativeControlError,
    SpeculativeRollbackError,
)

__all__ = [
    "BlockRef",
    "CommitRequest",
    "ControlRequest",
    "ControlResult",
    "Decision",
    "DraftCandidate",
    "Evaluation",
    "FallbackResult",
    "ProposalBatch",
    "SpeculativeControlError",
    "SpeculativeRollbackError",
    "SpeculativeController",
    "create_policy",
]
