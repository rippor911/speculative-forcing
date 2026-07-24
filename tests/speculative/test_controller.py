from __future__ import annotations

import json
import unittest

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


def block(index: int) -> BlockRef:
    return BlockRef(index=index, start_frame=index * 3, num_frames=3)


def make_batch(anchor_index: int, depths: tuple[int, ...]) -> tuple[ProposalBatch, list[object]]:
    anchor = CommitRequest(block=block(anchor_index), latent=f"anchor-{anchor_index}", source="anchor")
    noises = [object() for _ in depths]
    drafts = [
        DraftCandidate(
            block=block(anchor_index + depth),
            depth=depth,
            latent=f"draft-{depth}",
            source_noise=noise,
        )
        for depth, noise in zip(depths, noises)
    ]
    return ProposalBatch(anchor=anchor, drafts=tuple(drafts)), noises


def unsafe_batch(anchor: CommitRequest, drafts: tuple[DraftCandidate, ...]) -> ProposalBatch:
    batch = object.__new__(ProposalBatch)
    object.__setattr__(batch, "anchor", anchor)
    object.__setattr__(batch, "drafts", drafts)
    return batch


class FakeProposal:
    def __init__(self, batch: ProposalBatch) -> None:
        self.batch = batch
        self.requests: list[ControlRequest] = []

    def propose(self, request: ControlRequest) -> ProposalBatch:
        self.requests.append(request)
        return self.batch


class FakeEvaluator:
    def __init__(
        self,
        *,
        return_value: object | None = None,
        mismatch_candidate: bool = False,
        fail_at_depth: int | None = None,
    ) -> None:
        self.return_value = return_value
        self.mismatch_candidate = mismatch_candidate
        self.fail_at_depth = fail_at_depth
        self.calls: list[DraftCandidate] = []

    def evaluate(self, candidate: DraftCandidate) -> object:
        self.calls.append(candidate)
        if candidate.depth == self.fail_at_depth:
            raise RuntimeError(f"evaluator failed at depth {candidate.depth}")
        if self.return_value is not None:
            return self.return_value
        if self.mismatch_candidate:
            other = DraftCandidate(
                block=candidate.block,
                depth=candidate.depth,
                latent=candidate.latent,
                source_noise=candidate.source_noise,
            )
            return Evaluation(candidate=other, value={"depth": candidate.depth})
        return Evaluation(candidate=candidate, value={"depth": candidate.depth})


class FakeFallback:
    def __init__(self, *, wrong_noise: bool = False, fail: bool = False) -> None:
        self.wrong_noise = wrong_noise
        self.fail = fail
        self.calls: list[DraftCandidate] = []

    def generate(self, rejected: DraftCandidate) -> FallbackResult:
        self.calls.append(rejected)
        if self.fail:
            raise RuntimeError(f"fallback failed at depth {rejected.depth}")
        noise = object() if self.wrong_noise else rejected.source_noise
        return FallbackResult(
            block=rejected.block,
            latent=f"fallback-{rejected.depth}",
            source_noise=noise,
        )


class FakeCommitter:
    def __init__(
        self,
        *,
        fail_on_block: int | None = None,
        fail_rollback: bool = False,
        fail_begin: bool = False,
    ) -> None:
        self.fail_on_block = fail_on_block
        self.fail_rollback = fail_rollback
        self.fail_begin = fail_begin
        self.commits: list[CommitRequest] = []
        self._open: list[CommitRequest] = []
        self.persistent_cursor = 0
        self.begins = 0
        self.completes = 0
        self.rollbacks = 0

    def begin(self) -> None:
        if self.fail_begin:
            raise RuntimeError("begin failed")
        self.begins += 1
        self._open = []

    def commit(self, request: CommitRequest) -> None:
        if request.block.index == self.fail_on_block:
            raise RuntimeError(f"commit failed for block {request.block.index}")
        self.commits.append(request)
        self._open.append(request)
        self.persistent_cursor = request.block.index + 1

    def complete(self) -> None:
        self.completes += 1
        self._open = []

    def rollback(self) -> None:
        self.rollbacks += 1
        if self.fail_rollback:
            raise RuntimeError("rollback failed")
        for request in reversed(self._open):
            if self.commits and self.commits[-1] is request:
                self.commits.pop()
        self.persistent_cursor = self.commits[-1].block.index + 1 if self.commits else 0
        self._open = []


class TrackingPolicy:
    def __init__(self, *, reject_depth: int | None = None, fail: bool = False) -> None:
        self.reject_depth = reject_depth
        self.fail = fail
        self.calls: list[int] = []

    def decide(self, evaluation: Evaluation) -> Decision:
        self.calls.append(evaluation.candidate.depth)
        if self.fail:
            raise RuntimeError("policy failed")
        if self.reject_depth is not None and evaluation.candidate.depth >= self.reject_depth:
            return Decision.reject("tracking_reject")
        return Decision.accept("tracking_accept")


class ControllerTest(unittest.TestCase):
    def build_controller(
        self,
        *,
        batch: ProposalBatch,
        policy_name: str,
        reject_depth: int | None = None,
        committer: FakeCommitter | None = None,
        fallback: FakeFallback | None = None,
        evaluator: FakeEvaluator | None = None,
        policy: object | None = None,
    ) -> tuple[SpeculativeController, FakeEvaluator, FakeFallback, FakeCommitter]:
        evaluator = FakeEvaluator() if evaluator is None else evaluator
        fallback = FakeFallback() if fallback is None else fallback
        committer = FakeCommitter() if committer is None else committer
        kwargs = {} if reject_depth is None else {"reject_depth": reject_depth}
        policy = create_policy(policy_name, **kwargs) if policy is None else policy
        controller = SpeculativeController(
            proposer=FakeProposal(batch),
            evaluator=evaluator,
            policy=policy,
            fallback=fallback,
            committer=committer,
        )
        return controller, evaluator, fallback, committer

    def test_always_accept_commits_anchor_then_all_drafts(self) -> None:
        batch, _ = make_batch(0, (1, 2, 3))
        controller, evaluator, fallback, committer = self.build_controller(
            batch=batch,
            policy_name="always_accept",
        )

        result = controller.run(ControlRequest(anchor_block=block(0), max_depth=3))

        self.assertEqual(result.accepted_depth, 3)
        self.assertIsNone(result.rejected_depth)
        self.assertEqual([c.block.index for c in committer.commits], [0, 1, 2, 3])
        self.assertEqual([c.source for c in committer.commits], ["anchor", "draft", "draft", "draft"])
        self.assertEqual([c.depth for c in evaluator.calls], [1, 2, 3])
        self.assertEqual(fallback.calls, [])
        self.assertEqual(committer.completes, 1)

    def test_always_reject_commits_anchor_then_fallback(self) -> None:
        batch, _ = make_batch(0, (1, 2, 3))
        controller, evaluator, fallback, committer = self.build_controller(
            batch=batch,
            policy_name="always_reject",
        )

        result = controller.run(ControlRequest(anchor_block=block(0), max_depth=3))

        self.assertEqual(result.accepted_depth, 0)
        self.assertEqual(result.rejected_depth, 1)
        self.assertEqual([c.block.index for c in committer.commits], [0, 1])
        self.assertEqual([c.source for c in committer.commits], ["anchor", "fallback"])
        self.assertEqual([c.depth for c in evaluator.calls], [1])
        self.assertEqual([c.depth for c in fallback.calls], [1])
        self.assertEqual([d.depth for d in result.invalidated], [2, 3])

    def test_reject_at_depth_commits_prefix_then_fallback(self) -> None:
        batch, _ = make_batch(4, (1, 2, 3))
        controller, evaluator, fallback, committer = self.build_controller(
            batch=batch,
            policy_name="reject_at_depth",
            reject_depth=2,
        )

        result = controller.run(ControlRequest(anchor_block=block(4), max_depth=3))

        self.assertEqual(result.accepted_depth, 1)
        self.assertEqual(result.rejected_depth, 2)
        self.assertEqual([c.block.index for c in committer.commits], [4, 5, 6])
        self.assertEqual([c.source for c in committer.commits], ["anchor", "draft", "fallback"])
        self.assertEqual([c.depth for c in evaluator.calls], [1, 2])
        self.assertEqual([d.depth for d in result.invalidated], [3])

    def test_duplicate_committed_block_is_rejected(self) -> None:
        first_batch, _ = make_batch(0, ())
        second_batch, _ = make_batch(0, ())
        evaluator = FakeEvaluator()
        fallback = FakeFallback()
        committer = FakeCommitter()
        controller = SpeculativeController(
            proposer=FakeProposal(first_batch),
            evaluator=evaluator,
            policy=create_policy("always_accept"),
            fallback=fallback,
            committer=committer,
        )
        controller.run(ControlRequest(anchor_block=block(0), max_depth=0))
        controller.proposer = FakeProposal(second_batch)

        with self.assertRaisesRegex(SpeculativeControlError, "already been committed"):
            controller.run(ControlRequest(anchor_block=block(0), max_depth=0))

        self.assertEqual([c.block.index for c in committer.commits], [0])
        self.assertEqual(committer.rollbacks, 1)

    def test_out_of_order_draft_proposal_does_not_commit(self) -> None:
        anchor = CommitRequest(block=block(0), latent="anchor-0", source="anchor")
        out_of_order = unsafe_batch(
            anchor,
            (
                DraftCandidate(block=block(2), depth=2, latent="draft-2", source_noise=object()),
                DraftCandidate(block=block(1), depth=1, latent="draft-1", source_noise=object()),
            ),
        )
        controller, _, _, committer = self.build_controller(
            batch=out_of_order,
            policy_name="always_accept",
        )

        with self.assertRaisesRegex(SpeculativeControlError, "contiguous"):
            controller.run(ControlRequest(anchor_block=block(0), max_depth=2))

        self.assertEqual(committer.begins, 0)
        self.assertEqual(committer.commits, [])

    def test_fallback_commits_with_rejected_source_noise(self) -> None:
        batch, noises = make_batch(0, (1,))
        controller, _, _, committer = self.build_controller(
            batch=batch,
            policy_name="always_reject",
        )

        controller.run(ControlRequest(anchor_block=block(0), max_depth=1))

        fallback_commit = committer.commits[-1]
        self.assertEqual(fallback_commit.source, "fallback")
        self.assertIs(fallback_commit.source_noise, noises[0])

    def test_wrong_noise_fallback_rolls_back(self) -> None:
        batch, _ = make_batch(0, (1,))
        fallback = FakeFallback(wrong_noise=True)
        controller, _, _, committer = self.build_controller(
            batch=batch,
            policy_name="always_reject",
            fallback=fallback,
        )

        with self.assertRaisesRegex(SpeculativeControlError, "source_noise"):
            controller.run(ControlRequest(anchor_block=block(0), max_depth=1))

        self.assertEqual(committer.commits, [])
        self.assertEqual(committer.rollbacks, 1)

    def test_commit_exception_cleans_transaction_and_controller_state(self) -> None:
        batch, _ = make_batch(0, (1,))
        committer = FakeCommitter(fail_on_block=1)
        controller, _, _, _ = self.build_controller(
            batch=batch,
            policy_name="always_accept",
            committer=committer,
        )

        with self.assertRaisesRegex(RuntimeError, "commit failed"):
            controller.run(ControlRequest(anchor_block=block(0), max_depth=1))

        self.assertEqual(committer.commits, [])
        self.assertEqual(committer.rollbacks, 1)
        committer.fail_on_block = None
        controller.run(ControlRequest(anchor_block=block(0), max_depth=1))
        self.assertEqual([c.block.index for c in committer.commits], [0, 1])

    def test_wrong_evaluator_return_type_rolls_back(self) -> None:
        batch, _ = make_batch(0, (1,))
        evaluator = FakeEvaluator(return_value="not an evaluation")
        controller, _, _, committer = self.build_controller(
            batch=batch,
            policy_name="always_accept",
            evaluator=evaluator,
        )

        with self.assertRaisesRegex(SpeculativeControlError, "Evaluator.evaluate"):
            controller.run(ControlRequest(anchor_block=block(0), max_depth=1))

        self.assertEqual(committer.commits, [])
        self.assertEqual(committer.rollbacks, 1)

    def test_evaluation_candidate_identity_mismatch_rolls_back(self) -> None:
        batch, _ = make_batch(0, (1,))
        evaluator = FakeEvaluator(mismatch_candidate=True)
        controller, _, _, committer = self.build_controller(
            batch=batch,
            policy_name="always_accept",
            evaluator=evaluator,
        )

        with self.assertRaisesRegex(SpeculativeControlError, "current candidate"):
            controller.run(ControlRequest(anchor_block=block(0), max_depth=1))

        self.assertEqual(committer.commits, [])
        self.assertEqual(committer.rollbacks, 1)

    def test_evaluator_exception_rolls_back(self) -> None:
        batch, _ = make_batch(0, (1,))
        evaluator = FakeEvaluator(fail_at_depth=1)
        controller, _, _, committer = self.build_controller(
            batch=batch,
            policy_name="always_accept",
            evaluator=evaluator,
        )

        with self.assertRaisesRegex(RuntimeError, "evaluator failed"):
            controller.run(ControlRequest(anchor_block=block(0), max_depth=1))

        self.assertEqual(committer.commits, [])
        self.assertEqual(committer.rollbacks, 1)

    def test_policy_exception_rolls_back(self) -> None:
        batch, _ = make_batch(0, (1,))
        policy = TrackingPolicy(fail=True)
        controller, _, _, committer = self.build_controller(
            batch=batch,
            policy_name="always_accept",
            policy=policy,
        )

        with self.assertRaisesRegex(RuntimeError, "policy failed"):
            controller.run(ControlRequest(anchor_block=block(0), max_depth=1))

        self.assertEqual(policy.calls, [1])
        self.assertEqual(committer.commits, [])
        self.assertEqual(committer.rollbacks, 1)

    def test_fallback_exception_rolls_back(self) -> None:
        batch, _ = make_batch(0, (1,))
        fallback = FakeFallback(fail=True)
        controller, _, _, committer = self.build_controller(
            batch=batch,
            policy_name="always_reject",
            fallback=fallback,
        )

        with self.assertRaisesRegex(RuntimeError, "fallback failed"):
            controller.run(ControlRequest(anchor_block=block(0), max_depth=1))

        self.assertEqual(committer.commits, [])
        self.assertEqual(committer.rollbacks, 1)

    def test_rollback_failure_reports_original_and_rollback_errors(self) -> None:
        batch, _ = make_batch(0, (1,))
        committer = FakeCommitter(fail_on_block=1, fail_rollback=True)
        controller, _, _, _ = self.build_controller(
            batch=batch,
            policy_name="always_accept",
            committer=committer,
        )

        with self.assertRaises(SpeculativeRollbackError) as raised:
            controller.run(ControlRequest(anchor_block=block(0), max_depth=1))

        error = raised.exception
        self.assertIsInstance(error.original_exception, RuntimeError)
        self.assertIsInstance(error.rollback_exception, RuntimeError)
        self.assertIn("commit failed for block 1", str(error))
        self.assertIn("rollback failed", str(error))
        self.assertEqual(committer.rollbacks, 1)

    def test_reject_skips_deeper_evaluator_and_policy_calls(self) -> None:
        batch, _ = make_batch(0, (1, 2, 3))
        policy = TrackingPolicy(reject_depth=1)
        controller, evaluator, fallback, _ = self.build_controller(
            batch=batch,
            policy_name="always_accept",
            policy=policy,
        )

        controller.run(ControlRequest(anchor_block=block(0), max_depth=3))

        self.assertEqual([candidate.depth for candidate in evaluator.calls], [1])
        self.assertEqual(policy.calls, [1])
        self.assertEqual([candidate.depth for candidate in fallback.calls], [1])

    def test_invalid_commit_request_source_depth_combinations(self) -> None:
        with self.assertRaisesRegex(ValueError, "depth=None"):
            CommitRequest(block=block(0), latent="anchor", source="anchor", depth=1)
        with self.assertRaisesRegex(ValueError, "source_noise=None"):
            CommitRequest(block=block(0), latent="anchor", source="anchor", source_noise=object())
        with self.assertRaisesRegex(ValueError, "positive integer depth"):
            CommitRequest(block=block(1), latent="draft", source="draft", source_noise=object())
        with self.assertRaisesRegex(ValueError, "positive integer depth"):
            CommitRequest(block=block(1), latent="draft", source="draft", depth=0, source_noise=object())
        with self.assertRaisesRegex(ValueError, "preserve source_noise"):
            CommitRequest(block=block(1), latent="fallback", source="fallback", depth=1)
        with self.assertRaisesRegex(ValueError, "Unsupported commit source"):
            CommitRequest(block=block(1), latent="bad", source="bad")  # type: ignore[arg-type]

    def test_discontinuous_start_frame_is_rejected(self) -> None:
        anchor = CommitRequest(
            block=BlockRef(index=0, start_frame=0, num_frames=3),
            latent="anchor",
            source="anchor",
        )
        draft = DraftCandidate(
            block=BlockRef(index=1, start_frame=9, num_frames=3),
            depth=1,
            latent="draft",
            source_noise=object(),
        )

        with self.assertRaisesRegex(ValueError, "continuous"):
            ProposalBatch(anchor=anchor, drafts=(draft,))

    def test_trace_order_for_reject(self) -> None:
        batch, _ = make_batch(0, (1, 2, 3))
        controller, _, _, _ = self.build_controller(
            batch=batch,
            policy_name="always_reject",
        )

        result = controller.run(ControlRequest(anchor_block=block(0), max_depth=3))

        for event in result.trace:
            json.dumps(event.to_dict(), allow_nan=False)
        self.assertEqual(
            [event.name for event in result.trace],
            [
                "proposal_requested",
                "proposal_ready",
                "transaction_begin",
                "commit",
                "evaluate",
                "evaluated",
                "decision",
                "invalidated",
                "invalidated",
                "fallback_requested",
                "fallback_ready",
                "commit",
                "transaction_complete",
            ],
        )

    def test_cross_window_discontinuous_start_frame_is_rejected(self) -> None:
        first_batch, _ = make_batch(0, ())
        bad_anchor = CommitRequest(
            block=BlockRef(index=1, start_frame=9, num_frames=3),
            latent="anchor-1",
            source="anchor",
        )
        second_batch = ProposalBatch(anchor=bad_anchor)
        controller, _, _, committer = self.build_controller(
            batch=first_batch,
            policy_name="always_accept",
        )
        controller.run(ControlRequest(anchor_block=block(0), max_depth=0))
        controller.proposer = FakeProposal(second_batch)

        with self.assertRaisesRegex(SpeculativeControlError, "frame range"):
            controller.run(ControlRequest(anchor_block=bad_anchor.block, max_depth=0))

        self.assertEqual([commit.block.index for commit in committer.commits], [0])
        self.assertEqual(committer.rollbacks, 1)

    def test_cross_window_contiguous_start_frame_is_accepted(self) -> None:
        first_batch, _ = make_batch(0, ())
        second_anchor = CommitRequest(block=block(1), latent="anchor-1", source="anchor")
        second_batch = ProposalBatch(anchor=second_anchor)
        controller, _, _, committer = self.build_controller(
            batch=first_batch,
            policy_name="always_accept",
        )
        controller.run(ControlRequest(anchor_block=block(0), max_depth=0))
        controller.proposer = FakeProposal(second_batch)

        controller.run(ControlRequest(anchor_block=block(1), max_depth=0))

        self.assertEqual([commit.block.index for commit in committer.commits], [0, 1])
        self.assertEqual(committer.rollbacks, 0)

    def test_rollback_restores_last_block_ref(self) -> None:
        first_batch, _ = make_batch(0, ())
        failing_batch, _ = make_batch(1, (1,))
        replacement_batch, _ = make_batch(1, ())
        committer = FakeCommitter(fail_on_block=2)
        controller, _, _, _ = self.build_controller(
            batch=first_batch,
            policy_name="always_accept",
            committer=committer,
        )
        controller.run(ControlRequest(anchor_block=block(0), max_depth=0))
        controller.proposer = FakeProposal(failing_batch)

        with self.assertRaisesRegex(RuntimeError, "commit failed"):
            controller.run(ControlRequest(anchor_block=block(1), max_depth=1))

        self.assertEqual([commit.block.index for commit in committer.commits], [0])
        self.assertEqual(committer.persistent_cursor, 1)
        committer.fail_on_block = None
        controller.proposer = FakeProposal(replacement_batch)
        controller.run(ControlRequest(anchor_block=block(1), max_depth=0))
        self.assertEqual([commit.block.index for commit in committer.commits], [0, 1])

    def test_begin_exception_is_atomic_and_does_not_rollback(self) -> None:
        first_batch, _ = make_batch(0, ())
        second_batch, _ = make_batch(1, ())
        committer = FakeCommitter()
        controller, _, _, _ = self.build_controller(
            batch=first_batch,
            policy_name="always_accept",
            committer=committer,
        )
        controller.run(ControlRequest(anchor_block=block(0), max_depth=0))
        controller.proposer = FakeProposal(second_batch)
        committer.fail_begin = True

        with self.assertRaisesRegex(RuntimeError, "begin failed"):
            controller.run(ControlRequest(anchor_block=block(1), max_depth=0))

        self.assertEqual([commit.block.index for commit in committer.commits], [0])
        self.assertEqual(committer.persistent_cursor, 1)
        self.assertEqual(committer.rollbacks, 0)
        committer.fail_begin = False
        controller.run(ControlRequest(anchor_block=block(1), max_depth=0))
        self.assertEqual([commit.block.index for commit in committer.commits], [0, 1])

    def test_public_integer_fields_reject_bool_float_and_str(self) -> None:
        cases = [
            lambda value: BlockRef(index=value),
            lambda value: BlockRef(index=0, start_frame=value, num_frames=3),
            lambda value: BlockRef(index=0, start_frame=0, num_frames=value),
            lambda value: ControlRequest(anchor_block=block(0), max_depth=value),
            lambda value: DraftCandidate(
                block=block(1),
                depth=value,
                latent="draft",
                source_noise=object(),
            ),
            lambda value: CommitRequest(
                block=block(1),
                latent="draft",
                source="draft",
                depth=value,
                source_noise=object(),
            ),
        ]
        invalid_values = [True, 1.0, "1"]

        for make_value in cases:
            for value in invalid_values:
                with self.subTest(factory=make_value, value=value):
                    with self.assertRaisesRegex(ValueError, "integer"):
                        make_value(value)

        for value in invalid_values:
            with self.subTest(field="accepted_depth", value=value):
                with self.assertRaisesRegex(ValueError, "integer"):
                    ControlResult(
                        accepted_depth=value,
                        rejected_depth=None,
                        committed=(),
                        invalidated=(),
                        trace=(),
                    )
            with self.subTest(field="rejected_depth", value=value):
                with self.assertRaisesRegex(ValueError, "integer"):
                    ControlResult(
                        accepted_depth=0,
                        rejected_depth=value,
                        committed=(),
                        invalidated=(),
                        trace=(),
                    )


if __name__ == "__main__":
    unittest.main()
