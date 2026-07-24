from __future__ import annotations

from typing import Optional

from speculative.interfaces import (
    Committer,
    Evaluator,
    FallbackGenerator,
    Policy,
    ProposalSource,
)
from speculative.trace import TraceRecorder
from speculative.types import (
    CommitRequest,
    ControlRequest,
    ControlResult,
    Decision,
    DraftCandidate,
    Evaluation,
    FallbackResult,
    ProposalBatch,
    BlockRef,
    SpeculativeControlError,
    SpeculativeRollbackError,
    validate_contiguous_block_range,
)


class SpeculativeController:
    """Model-agnostic speculative decoding controller.

    The controller owns only ordering, longest-prefix acceptance, fallback, and
    trace. Model calls and cache mutation stay behind protocol boundaries.
    """

    def __init__(
        self,
        *,
        proposer: ProposalSource,
        evaluator: Evaluator,
        policy: Policy,
        fallback: FallbackGenerator,
        committer: Committer,
    ) -> None:
        self.proposer = proposer
        self.evaluator = evaluator
        self.policy = policy
        self.fallback = fallback
        self.committer = committer
        self._committed_blocks: set[int] = set()
        self._last_committed_block: Optional[BlockRef] = None

    def run(self, request: ControlRequest) -> ControlResult:
        trace = TraceRecorder()
        committed: list[CommitRequest] = []
        invalidated: list[DraftCandidate] = []
        accepted_depth = 0
        rejected_depth: Optional[int] = None
        transaction_started = False

        trace.emit(
            "proposal_requested",
            block=request.anchor_block,
            metadata={"max_depth": request.max_depth},
        )
        batch = self.proposer.propose(request)
        self._validate_batch(request, batch)
        trace.emit(
            "proposal_ready",
            block=batch.anchor.block,
            metadata={"draft_depths": [draft.depth for draft in batch.drafts]},
        )

        previous_committed = set(self._committed_blocks)
        previous_last = self._last_committed_block

        try:
            trace.emit("transaction_begin", block=batch.anchor.block)
            self.committer.begin()
            transaction_started = True

            self._commit_once(batch.anchor, trace, committed)
            for offset, candidate in enumerate(batch.drafts):
                trace.emit("evaluate", block=candidate.block, depth=candidate.depth)
                evaluation = self.evaluator.evaluate(candidate)
                if not isinstance(evaluation, Evaluation):
                    raise SpeculativeControlError("Evaluator.evaluate() must return Evaluation.")
                if evaluation.candidate is not candidate:
                    raise SpeculativeControlError(
                        "Evaluation.candidate must be the current candidate."
                    )
                trace.emit(
                    "evaluated",
                    block=candidate.block,
                    depth=candidate.depth,
                    metadata=evaluation.metadata,
                )

                decision = self.policy.decide(evaluation)
                if not isinstance(decision, Decision):
                    raise SpeculativeControlError("Policy.decide() must return Decision.")
                trace.emit(
                    "decision",
                    block=candidate.block,
                    depth=candidate.depth,
                    decision=decision.action,
                    reason=decision.reason,
                    metadata=decision.metadata,
                )

                if decision.accepted:
                    accepted_depth = candidate.depth
                    self._commit_once(
                        CommitRequest(
                            block=candidate.block,
                            latent=candidate.latent,
                            source="draft",
                            depth=candidate.depth,
                            source_noise=candidate.source_noise,
                            metadata=candidate.metadata,
                        ),
                        trace,
                        committed,
                    )
                    continue

                rejected_depth = candidate.depth
                for deeper in batch.drafts[offset + 1:]:
                    invalidated.append(deeper)
                    trace.emit(
                        "invalidated",
                        block=deeper.block,
                        depth=deeper.depth,
                        reason="first_reject",
                    )

                trace.emit("fallback_requested", block=candidate.block, depth=candidate.depth)
                fallback_result = self.fallback.generate(candidate)
                self._validate_fallback(candidate, fallback_result)
                trace.emit(
                    "fallback_ready",
                    block=fallback_result.block,
                    depth=candidate.depth,
                    metadata=fallback_result.metadata,
                )
                self._commit_once(
                    CommitRequest(
                        block=fallback_result.block,
                        latent=fallback_result.latent,
                        source="fallback",
                        depth=candidate.depth,
                        source_noise=fallback_result.source_noise,
                        metadata=fallback_result.metadata,
                    ),
                    trace,
                    committed,
                )
                break

            self.committer.complete()
            trace.emit("transaction_complete", block=committed[-1].block if committed else None)
        except Exception as exc:
            self._committed_blocks = previous_committed
            self._last_committed_block = previous_last
            trace.emit(
                "error",
                metadata={"type": type(exc).__name__, "message": str(exc)},
            )
            if transaction_started:
                try:
                    self.committer.rollback()
                    trace.emit("transaction_rollback")
                except Exception as rollback_error:
                    trace.emit(
                        "transaction_rollback_error",
                        metadata={
                            "type": type(rollback_error).__name__,
                            "message": str(rollback_error),
                        },
                    )
                    raise SpeculativeRollbackError(exc, rollback_error) from exc
            raise

        return ControlResult(
            accepted_depth=accepted_depth,
            rejected_depth=rejected_depth,
            committed=committed,
            invalidated=invalidated,
            trace=trace.events(),
        )

    def _validate_batch(self, request: ControlRequest, batch: ProposalBatch) -> None:
        if not isinstance(batch, ProposalBatch):
            raise SpeculativeControlError("ProposalSource.propose() must return ProposalBatch.")
        if batch.anchor.block != request.anchor_block:
            raise SpeculativeControlError(
                "Proposal anchor block does not match the control request."
            )
        if len(batch.drafts) > request.max_depth:
            raise SpeculativeControlError(
                f"Proposal returned {len(batch.drafts)} drafts for max_depth={request.max_depth}."
            )
        try:
            validate_contiguous_block_range(batch.anchor.block, batch.drafts)
        except ValueError as exc:
            raise SpeculativeControlError(str(exc)) from exc

        for expected_depth, candidate in enumerate(batch.drafts, start=1):
            if candidate.depth != expected_depth:
                raise SpeculativeControlError(
                    f"Draft depths must be contiguous from 1; expected "
                    f"{expected_depth}, got {candidate.depth}."
                )
            expected_block = batch.anchor.block.index + expected_depth
            if candidate.block.index != expected_block:
                raise SpeculativeControlError(
                    f"Draft depth {candidate.depth} must target block "
                    f"{expected_block}, got {candidate.block.index}."
                )

    def _validate_fallback(
        self,
        rejected: DraftCandidate,
        fallback_result: FallbackResult,
    ) -> None:
        if not isinstance(fallback_result, FallbackResult):
            raise SpeculativeControlError(
                "FallbackGenerator.generate() must return FallbackResult."
            )
        if fallback_result.block != rejected.block:
            raise SpeculativeControlError("Fallback must target the rejected draft block.")
        if fallback_result.source_noise is not rejected.source_noise:
            raise SpeculativeControlError("Fallback must use the rejected draft source_noise.")

    def _commit_once(
        self,
        request: CommitRequest,
        trace: TraceRecorder,
        committed: list[CommitRequest],
    ) -> None:
        block_index = request.block.index
        if block_index in self._committed_blocks:
            raise SpeculativeControlError(f"Block {block_index} has already been committed.")
        if self._last_committed_block is not None:
            expected = self._last_committed_block.index + 1
            if block_index != expected:
                raise SpeculativeControlError(
                    f"Out-of-order commit: expected block {expected}, got {block_index}."
                )
            previous = self._last_committed_block
            current = request.block
            previous_has_range = previous.start_frame is not None or previous.num_frames is not None
            current_has_range = current.start_frame is not None or current.num_frames is not None
            if previous_has_range or current_has_range:
                if not (previous_has_range and current_has_range):
                    raise SpeculativeControlError(
                        "Contiguous frame validation requires both adjacent blocks "
                        "to provide start_frame and num_frames."
                    )
                expected_start = previous.start_frame + previous.num_frames
                if current.start_frame != expected_start:
                    raise SpeculativeControlError(
                        f"Out-of-order frame range: expected block {current.index} "
                        f"to start at {expected_start}, got {current.start_frame}."
                    )

        self.committer.commit(request)
        self._committed_blocks.add(block_index)
        self._last_committed_block = request.block
        committed.append(request)
        trace.emit(
            "commit",
            block=request.block,
            depth=request.depth,
            source=request.source,
        )
