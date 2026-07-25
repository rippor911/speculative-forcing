from __future__ import annotations

from typing import Protocol, Sequence, TYPE_CHECKING

from speculative.types import (
    CommitRequest,
    ControlRequest,
    Decision,
    DraftCandidate,
    Evaluation,
    FallbackResult,
    ProposalBatch,
)

if TYPE_CHECKING:
    from speculative.evaluation import DecodedCandidate
    from speculative.scoring import RawScoreResult


class CandidateDecoder(Protocol):
    """Converts a draft candidate into scorer input.

    It must not score, decide, call fallback, call generators, or mutate
    persistent generation state.
    """

    def decode(self, candidate: DraftCandidate) -> "DecodedCandidate":
        ...


class CandidateScorer(Protocol):
    """Produces raw scores from decoder output without aggregation or policy."""

    def score(self, decoded: "DecodedCandidate") -> "RawScoreResult":
        ...


class ScoreAggregator(Protocol):
    """Aggregates raw finite scores into one block score.

    It must not read thresholds, decide acceptance, or mutate generation state.
    """

    @property
    def name(self) -> str:
        ...

    def aggregate(self, scores: Sequence[float]) -> float:
        ...


class ProposalSource(Protocol):
    """Produces one anchor and speculative drafts for a control window.

    The adapter may run temporary model/cache forwards, but it must snapshot and
    restore any such local state before returning. It must not permanently
    advance KV, output buffers, or generation cursors.
    """

    def propose(self, request: ControlRequest) -> ProposalBatch:
        ...


class Evaluator(Protocol):
    """Scores or otherwise evaluates a draft candidate.

    Evaluation state is adapter-local. Temporary model/cache mutation must be
    restored before returning the `Evaluation`.
    """

    def evaluate(self, candidate: DraftCandidate) -> Evaluation:
        ...


class Policy(Protocol):
    """Maps an evaluation to a pure accept/reject decision."""

    def decide(self, evaluation: Evaluation) -> Decision:
        ...


class FallbackGenerator(Protocol):
    """Generates the non-speculative replacement for a rejected draft.

    Fallback generation may use adapter-local transactions, but it must return
    without permanent KV/output/cursor mutation. Permanent state changes happen
    only through `Committer.commit`.
    """

    def generate(self, rejected: DraftCandidate) -> FallbackResult:
        ...


class Committer(Protocol):
    """Owns the speculative window transaction and permanent generation state.

    This is the only protocol allowed to permanently modify Transformer KV,
    output storage, and the generation cursor. `begin`/`complete`/`rollback`
    describe the controller window transaction, distinct from any adapter-local
    snapshot/restore performed by proposal, evaluation, or fallback adapters.

    `begin` must be exception-atomic: if it raises, it must leave no permanent
    state change behind. The controller treats a failed `begin` as "transaction
    not started" and will not call `rollback`.
    """

    def begin(self) -> None:
        ...

    def commit(self, request: CommitRequest) -> None:
        ...

    def complete(self) -> None:
        ...

    def rollback(self) -> None:
        ...
