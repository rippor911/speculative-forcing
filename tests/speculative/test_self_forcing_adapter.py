from __future__ import annotations

import unittest
from typing import Optional

from speculative.adapters.self_forcing_mcp import (
    SelfForcingMCPCommitter,
    SelfForcingMCPFallbackGenerator,
    SelfForcingMCPProposalSource,
)
from speculative.controller import SpeculativeController
from speculative.factory import create_policy
from speculative.types import (
    BlockRef,
    CommitRequest,
    ControlRequest,
    DraftCandidate,
    Evaluation,
    FallbackResult,
    ProposalBatch,
)


def block(index: int) -> BlockRef:
    return BlockRef(index=index, start_frame=index * 4, num_frames=4)


def make_batch(anchor_index: int, depth_count: int) -> tuple[ProposalBatch, list[object]]:
    anchor = CommitRequest(
        block=block(anchor_index),
        latent=f"anchor-{anchor_index}",
        source="anchor",
    )
    noises = [object() for _ in range(depth_count)]
    drafts = tuple(
        DraftCandidate(
            block=block(anchor_index + depth),
            depth=depth,
            latent=f"draft-{depth}",
            source_noise=noise,
        )
        for depth, noise in enumerate(noises, start=1)
    )
    return ProposalBatch(anchor=anchor, drafts=drafts), noises


class FakeSharedRuntime:
    def __init__(
        self,
        *,
        proposal_batch: Optional[ProposalBatch] = None,
        fallback_result: Optional[FallbackResult] = None,
    ) -> None:
        self.proposal_batch = proposal_batch
        self.fallback_result = fallback_result
        self.propose_requests: list[ControlRequest] = []
        self.fallback_candidates: list[DraftCandidate] = []
        self.begin_count = 0
        self.commit_requests: list[CommitRequest] = []
        self.complete_count = 0
        self.rollback_count = 0
        self.call_order: list[str] = []
        self.proposal_error: Optional[Exception] = None
        self.fallback_error: Optional[Exception] = None
        self.commit_error: Optional[Exception] = None
        self.rollback_error: Optional[Exception] = None

    def propose_window(self, request: ControlRequest) -> ProposalBatch:
        self.call_order.append("propose")
        self.propose_requests.append(request)
        if self.proposal_error is not None:
            raise self.proposal_error
        if self.proposal_batch is None:
            raise AssertionError("proposal_batch is required")
        return self.proposal_batch

    def generate_target_fallback(self, candidate: DraftCandidate) -> FallbackResult:
        self.call_order.append("fallback")
        self.fallback_candidates.append(candidate)
        if self.fallback_error is not None:
            raise self.fallback_error
        if self.fallback_result is not None:
            return self.fallback_result
        return FallbackResult(
            block=candidate.block,
            latent=f"fallback-{candidate.depth}",
            source_noise=candidate.source_noise,
        )

    def begin_window(self) -> None:
        self.call_order.append("begin")
        self.begin_count += 1

    def commit_block(self, request: CommitRequest) -> None:
        self.call_order.append("commit")
        self.commit_requests.append(request)
        if self.commit_error is not None:
            raise self.commit_error

    def complete_window(self) -> None:
        self.call_order.append("complete")
        self.complete_count += 1

    def rollback_window(self) -> None:
        self.call_order.append("rollback")
        self.rollback_count += 1
        if self.rollback_error is not None:
            raise self.rollback_error


class IdentityEvaluator:
    def __init__(self) -> None:
        self.calls: list[DraftCandidate] = []

    def evaluate(self, candidate: DraftCandidate) -> Evaluation:
        self.calls.append(candidate)
        return Evaluation(candidate=candidate, value={"depth": candidate.depth})


class SelfForcingAdapterTest(unittest.TestCase):
    def make_wrappers(
        self,
        runtime: FakeSharedRuntime,
    ) -> tuple[
        SelfForcingMCPProposalSource,
        SelfForcingMCPFallbackGenerator,
        SelfForcingMCPCommitter,
    ]:
        return (
            SelfForcingMCPProposalSource(runtime),
            SelfForcingMCPFallbackGenerator(runtime),
            SelfForcingMCPCommitter(runtime),
        )

    def make_controller(
        self,
        runtime: FakeSharedRuntime,
        *,
        policy_name: str,
        reject_depth: Optional[int] = None,
    ) -> tuple[SpeculativeController, IdentityEvaluator]:
        proposer, fallback, committer = self.make_wrappers(runtime)
        kwargs = {} if reject_depth is None else {"reject_depth": reject_depth}
        evaluator = IdentityEvaluator()
        return (
            SpeculativeController(
                proposer=proposer,
                evaluator=evaluator,
                policy=create_policy(policy_name, **kwargs),
                fallback=fallback,
                committer=committer,
            ),
            evaluator,
        )

    def test_wrappers_share_same_runtime_object(self) -> None:
        runtime = FakeSharedRuntime()
        proposer, fallback, committer = self.make_wrappers(runtime)

        self.assertIs(proposer.runtime, runtime)
        self.assertIs(fallback.runtime, runtime)
        self.assertIs(committer.runtime, runtime)

    def test_proposal_delegates_with_control_request_identity(self) -> None:
        batch, _ = make_batch(0, 1)
        runtime = FakeSharedRuntime(proposal_batch=batch)
        proposer = SelfForcingMCPProposalSource(runtime)
        request = ControlRequest(anchor_block=block(0), max_depth=1)

        proposer.propose(request)

        self.assertEqual(runtime.propose_requests, [request])
        self.assertIs(runtime.propose_requests[0], request)

    def test_proposal_returns_runtime_batch_identity(self) -> None:
        batch, _ = make_batch(0, 1)
        runtime = FakeSharedRuntime(proposal_batch=batch)
        proposer = SelfForcingMCPProposalSource(runtime)

        result = proposer.propose(ControlRequest(anchor_block=block(0), max_depth=1))

        self.assertIs(result, batch)

    def test_fallback_delegates_with_candidate_identity(self) -> None:
        batch, _ = make_batch(0, 1)
        candidate = batch.drafts[0]
        runtime = FakeSharedRuntime()
        fallback = SelfForcingMCPFallbackGenerator(runtime)

        fallback.generate(candidate)

        self.assertEqual(runtime.fallback_candidates, [candidate])
        self.assertIs(runtime.fallback_candidates[0], candidate)

    def test_fallback_returns_result_and_source_noise_identity(self) -> None:
        batch, _ = make_batch(0, 1)
        candidate = batch.drafts[0]
        result = FallbackResult(
            block=candidate.block,
            latent=object(),
            source_noise=candidate.source_noise,
        )
        runtime = FakeSharedRuntime(fallback_result=result)
        fallback = SelfForcingMCPFallbackGenerator(runtime)

        returned = fallback.generate(candidate)

        self.assertIs(returned, result)
        self.assertIs(returned.source_noise, candidate.source_noise)

    def test_commit_delegates_with_commit_request_identity(self) -> None:
        runtime = FakeSharedRuntime()
        committer = SelfForcingMCPCommitter(runtime)
        request = CommitRequest(block=block(0), latent=object(), source="anchor")

        committer.commit(request)

        self.assertEqual(runtime.commit_requests, [request])
        self.assertIs(runtime.commit_requests[0], request)

    def test_committer_delegates_lifecycle_in_order(self) -> None:
        runtime = FakeSharedRuntime()
        committer = SelfForcingMCPCommitter(runtime)
        request = CommitRequest(block=block(0), latent=object(), source="anchor")

        committer.begin()
        committer.commit(request)
        committer.complete()
        committer.rollback()

        self.assertEqual(runtime.call_order, ["begin", "commit", "complete", "rollback"])
        self.assertEqual(runtime.begin_count, 1)
        self.assertEqual(runtime.complete_count, 1)
        self.assertEqual(runtime.rollback_count, 1)

    def test_runtime_proposal_exception_propagates(self) -> None:
        error = RuntimeError("proposal failed")
        runtime = FakeSharedRuntime()
        runtime.proposal_error = error
        proposer = SelfForcingMCPProposalSource(runtime)

        with self.assertRaises(RuntimeError) as context:
            proposer.propose(ControlRequest(anchor_block=block(0), max_depth=0))

        self.assertIs(context.exception, error)

    def test_runtime_fallback_exception_propagates(self) -> None:
        batch, _ = make_batch(0, 1)
        error = RuntimeError("fallback failed")
        runtime = FakeSharedRuntime()
        runtime.fallback_error = error
        fallback = SelfForcingMCPFallbackGenerator(runtime)

        with self.assertRaises(RuntimeError) as context:
            fallback.generate(batch.drafts[0])

        self.assertIs(context.exception, error)

    def test_runtime_commit_exception_propagates(self) -> None:
        error = RuntimeError("commit failed")
        runtime = FakeSharedRuntime()
        runtime.commit_error = error
        committer = SelfForcingMCPCommitter(runtime)

        with self.assertRaises(RuntimeError) as context:
            committer.commit(CommitRequest(block=block(0), latent=object(), source="anchor"))

        self.assertIs(context.exception, error)

    def test_runtime_rollback_exception_propagates(self) -> None:
        error = RuntimeError("rollback failed")
        runtime = FakeSharedRuntime()
        runtime.rollback_error = error
        committer = SelfForcingMCPCommitter(runtime)

        with self.assertRaises(RuntimeError) as context:
            committer.rollback()

        self.assertIs(context.exception, error)

    def test_controller_always_accept_commits_anchor_then_all_drafts(self) -> None:
        batch, _ = make_batch(0, 3)
        runtime = FakeSharedRuntime(proposal_batch=batch)
        controller, evaluator = self.make_controller(runtime, policy_name="always_accept")

        result = controller.run(ControlRequest(anchor_block=block(0), max_depth=3))

        self.assertEqual([request.block.index for request in runtime.commit_requests], [0, 1, 2, 3])
        self.assertEqual([request.source for request in runtime.commit_requests], ["anchor", "draft", "draft", "draft"])
        self.assertEqual([candidate.depth for candidate in evaluator.calls], [1, 2, 3])
        self.assertEqual(runtime.fallback_candidates, [])
        self.assertEqual(result.committed, tuple(runtime.commit_requests))
        self.assertEqual(runtime.call_order, ["propose", "begin", "commit", "commit", "commit", "commit", "complete"])

    def test_controller_always_reject_commits_anchor_and_fallback_only(self) -> None:
        batch, noises = make_batch(0, 3)
        runtime = FakeSharedRuntime(proposal_batch=batch)
        controller, evaluator = self.make_controller(runtime, policy_name="always_reject")

        result = controller.run(ControlRequest(anchor_block=block(0), max_depth=3))

        self.assertEqual([request.block.index for request in runtime.commit_requests], [0, 1])
        self.assertEqual([request.source for request in runtime.commit_requests], ["anchor", "fallback"])
        self.assertEqual([candidate.depth for candidate in evaluator.calls], [1])
        self.assertEqual(runtime.fallback_candidates, [batch.drafts[0]])
        self.assertEqual(result.accepted_depth, 0)
        self.assertEqual(result.rejected_depth, 1)
        self.assertEqual(result.invalidated, tuple(batch.drafts[1:]))
        self.assertIs(runtime.commit_requests[1].source_noise, noises[0])

    def test_controller_reject_at_depth_preserves_prefix_fallback_and_invalidation(self) -> None:
        batch, noises = make_batch(0, 3)
        runtime = FakeSharedRuntime(proposal_batch=batch)
        controller, evaluator = self.make_controller(
            runtime,
            policy_name="reject_at_depth",
            reject_depth=2,
        )

        result = controller.run(ControlRequest(anchor_block=block(0), max_depth=3))

        self.assertEqual([candidate.depth for candidate in evaluator.calls], [1, 2])
        self.assertEqual([request.block.index for request in runtime.commit_requests], [0, 1, 2])
        self.assertEqual([request.source for request in runtime.commit_requests], ["anchor", "draft", "fallback"])
        self.assertEqual(result.accepted_depth, 1)
        self.assertEqual(result.rejected_depth, 2)
        self.assertEqual(result.invalidated, (batch.drafts[2],))
        self.assertIs(runtime.fallback_candidates[0], batch.drafts[1])
        self.assertIs(runtime.commit_requests[1].source_noise, noises[0])
        self.assertIs(runtime.commit_requests[2].source_noise, noises[1])

    def test_controller_exception_calls_runtime_rollback(self) -> None:
        batch, _ = make_batch(0, 1)
        error = RuntimeError("fallback failed")
        runtime = FakeSharedRuntime(proposal_batch=batch)
        runtime.fallback_error = error
        controller, _ = self.make_controller(runtime, policy_name="always_reject")

        with self.assertRaises(RuntimeError) as context:
            controller.run(ControlRequest(anchor_block=block(0), max_depth=1))

        self.assertIs(context.exception, error)
        self.assertEqual(runtime.rollback_count, 1)
        self.assertEqual(runtime.complete_count, 0)
        self.assertEqual(runtime.call_order, ["propose", "begin", "commit", "fallback", "rollback"])

    def test_wrappers_do_not_copy_latent_noise_or_blockref(self) -> None:
        anchor_block = block(0)
        anchor_latent = object()
        draft_block = block(1)
        draft_latent = object()
        source_noise = object()
        anchor = CommitRequest(block=anchor_block, latent=anchor_latent, source="anchor")
        draft = DraftCandidate(
            block=draft_block,
            depth=1,
            latent=draft_latent,
            source_noise=source_noise,
        )
        batch = ProposalBatch(anchor=anchor, drafts=(draft,))
        fallback_result = FallbackResult(
            block=draft_block,
            latent=object(),
            source_noise=source_noise,
        )
        runtime = FakeSharedRuntime(proposal_batch=batch, fallback_result=fallback_result)
        proposer, fallback, committer = self.make_wrappers(runtime)

        proposed = proposer.propose(ControlRequest(anchor_block=anchor_block, max_depth=1))
        returned_fallback = fallback.generate(draft)
        committer.commit(anchor)

        self.assertIs(proposed.anchor.block, anchor_block)
        self.assertIs(proposed.anchor.latent, anchor_latent)
        self.assertIs(proposed.drafts[0].block, draft_block)
        self.assertIs(proposed.drafts[0].latent, draft_latent)
        self.assertIs(proposed.drafts[0].source_noise, source_noise)
        self.assertIs(returned_fallback.block, draft_block)
        self.assertIs(returned_fallback.source_noise, source_noise)
        self.assertIs(runtime.commit_requests[0], anchor)

    def test_wrappers_expose_only_shared_runtime_state(self) -> None:
        runtime = FakeSharedRuntime()
        wrappers = self.make_wrappers(runtime)

        for wrapper in wrappers:
            self.assertFalse(hasattr(wrapper, "__dict__"))
            self.assertEqual(getattr(type(wrapper), "__slots__"), ("_runtime",))
            with self.assertRaises(AttributeError):
                setattr(wrapper, "output", object())


if __name__ == "__main__":
    unittest.main()
