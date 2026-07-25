from __future__ import annotations

import unittest

from speculative.controller import SpeculativeController
from speculative.evaluation import CompositeCandidateEvaluator, DecodedCandidate, IdentityCandidateDecoder
from speculative.factory import create_verifier_from_config
from speculative.policies.fixed_threshold import FixedThresholdPolicy
from speculative.scoring import MeanScoreAggregator, MinScoreAggregator, RawScoreResult
from speculative.types import (
    BlockRef,
    CommitRequest,
    ControlRequest,
    DraftCandidate,
    FallbackResult,
    ProposalBatch,
)


def block(index: int) -> BlockRef:
    return BlockRef(index=index, start_frame=index * 3, num_frames=3)


def make_batch(depths: tuple[int, ...] = (1, 2, 3)) -> tuple[ProposalBatch, list[object]]:
    anchor = CommitRequest(block=block(0), latent="anchor-0", source="anchor")
    noises = [object() for _ in depths]
    drafts = tuple(
        DraftCandidate(
            block=block(depth),
            depth=depth,
            latent=f"draft-{depth}",
            source_noise=noise,
        )
        for depth, noise in zip(depths, noises)
    )
    return ProposalBatch(anchor=anchor, drafts=drafts), noises


class FakeProposal:
    def __init__(self, batch: ProposalBatch) -> None:
        self.batch = batch
        self.calls: list[ControlRequest] = []

    def propose(self, request: ControlRequest) -> ProposalBatch:
        self.calls.append(request)
        return self.batch


class FakeFallback:
    def __init__(self) -> None:
        self.calls: list[DraftCandidate] = []

    def generate(self, rejected: DraftCandidate) -> FallbackResult:
        self.calls.append(rejected)
        return FallbackResult(
            block=rejected.block,
            latent=f"fallback-{rejected.depth}",
            source_noise=rejected.source_noise,
        )


class FakeCommitter:
    def __init__(self) -> None:
        self.commits: list[CommitRequest] = []
        self._open: list[CommitRequest] = []
        self.fake_kv = {"cursor": 0}
        self.output: list[tuple[int, str]] = []
        self.begins = 0
        self.completes = 0
        self.rollbacks = 0
        self._snapshot: tuple[int, dict[str, int], list[tuple[int, str]]] | None = None

    def begin(self) -> None:
        self.begins += 1
        self._open = []
        self._snapshot = (len(self.commits), dict(self.fake_kv), list(self.output))

    def commit(self, request: CommitRequest) -> None:
        self.commits.append(request)
        self._open.append(request)
        self.fake_kv["cursor"] = request.block.index + 1
        self.output.append((request.block.index, request.source))

    def complete(self) -> None:
        self.completes += 1
        self._open = []
        self._snapshot = None

    def rollback(self) -> None:
        self.rollbacks += 1
        if self._snapshot is None:
            return
        commit_count, fake_kv, output = self._snapshot
        del self.commits[commit_count:]
        self.fake_kv = fake_kv
        self.output = output
        self._open = []
        self._snapshot = None


class RecordingDecoder:
    def __init__(self, *, fail_depth: int | None = None) -> None:
        self.fail_depth = fail_depth
        self.calls: list[int] = []
        self.inner = IdentityCandidateDecoder()

    def decode(self, candidate: DraftCandidate) -> DecodedCandidate:
        self.calls.append(candidate.depth)
        if candidate.depth == self.fail_depth:
            raise RuntimeError(f"decoder failed at depth {candidate.depth}")
        return self.inner.decode(candidate)


class RecordingScorer:
    def __init__(
        self,
        scores_by_depth: dict[int, tuple[float, ...]],
        *,
        fail_depth: int | None = None,
    ) -> None:
        self.scores_by_depth = scores_by_depth
        self.fail_depth = fail_depth
        self.calls: list[int] = []

    def score(self, decoded: DecodedCandidate) -> RawScoreResult:
        depth = decoded.candidate.depth
        self.calls.append(depth)
        if depth == self.fail_depth:
            raise RuntimeError(f"scorer failed at depth {depth}")
        return RawScoreResult(
            per_frame_scores=self.scores_by_depth[depth],
            scorer_name="recording",
        )


class RecordingAggregator:
    def __init__(self, inner: object, *, fail: bool = False) -> None:
        self.inner = inner
        self.fail = fail
        self.name = inner.name
        self.calls: list[tuple[float, ...]] = []

    def aggregate(self, scores: tuple[float, ...]) -> float:
        self.calls.append(tuple(scores))
        if self.fail:
            raise RuntimeError("aggregator failed")
        return self.inner.aggregate(scores)


class RecordingPolicy:
    def __init__(self, threshold: float) -> None:
        self.inner = FixedThresholdPolicy(threshold=threshold)
        self.calls: list[int] = []

    def decide(self, evaluation):
        self.calls.append(evaluation.candidate.depth)
        return self.inner.decide(evaluation)


def build_controller(
    *,
    scores_by_depth: dict[int, tuple[float, ...]],
    depths: tuple[int, ...] = (1, 2, 3),
    aggregator: object | None = None,
    threshold: float = 0.5,
    decoder: RecordingDecoder | None = None,
    scorer: RecordingScorer | None = None,
    policy: RecordingPolicy | None = None,
) -> tuple[
    SpeculativeController,
    RecordingDecoder,
    RecordingScorer,
    object,
    RecordingPolicy,
    FakeFallback,
    FakeCommitter,
]:
    batch, _ = make_batch(depths)
    decoder = RecordingDecoder() if decoder is None else decoder
    scorer = RecordingScorer(scores_by_depth) if scorer is None else scorer
    aggregator = RecordingAggregator(MinScoreAggregator()) if aggregator is None else aggregator
    policy = RecordingPolicy(threshold) if policy is None else policy
    fallback = FakeFallback()
    committer = FakeCommitter()
    controller = SpeculativeController(
        proposer=FakeProposal(batch),
        evaluator=CompositeCandidateEvaluator(
            decoder=decoder,
            scorer=scorer,
            aggregator=aggregator,
        ),
        policy=policy,
        fallback=fallback,
        committer=committer,
    )
    return controller, decoder, scorer, aggregator, policy, fallback, committer


class VerifierControllerLoopTest(unittest.TestCase):
    def test_all_depths_above_threshold_accepts_all_without_fallback(self) -> None:
        controller, decoder, scorer, _, policy, fallback, committer = build_controller(
            scores_by_depth={1: (0.8,), 2: (0.7,), 3: (0.6,)},
        )

        result = controller.run(ControlRequest(anchor_block=block(0), max_depth=3))

        self.assertEqual(result.accepted_depth, 3)
        self.assertIsNone(result.rejected_depth)
        self.assertEqual([commit.source for commit in committer.commits], ["anchor", "draft", "draft", "draft"])
        self.assertEqual(decoder.calls, [1, 2, 3])
        self.assertEqual(scorer.calls, [1, 2, 3])
        self.assertEqual(policy.calls, [1, 2, 3])
        self.assertEqual(fallback.calls, [])

    def test_depth_one_rejects_uses_same_noise_fallback_and_skips_deeper(self) -> None:
        controller, decoder, scorer, _, policy, fallback, committer = build_controller(
            scores_by_depth={1: (0.1,), 2: (0.9,), 3: (0.9,)},
        )

        result = controller.run(ControlRequest(anchor_block=block(0), max_depth=3))

        self.assertEqual(result.accepted_depth, 0)
        self.assertEqual(result.rejected_depth, 1)
        self.assertEqual([commit.source for commit in committer.commits], ["anchor", "fallback"])
        self.assertEqual(decoder.calls, [1])
        self.assertEqual(scorer.calls, [1])
        self.assertEqual(policy.calls, [1])
        self.assertEqual([candidate.depth for candidate in fallback.calls], [1])
        self.assertIs(committer.commits[-1].source_noise, fallback.calls[0].source_noise)
        self.assertEqual([candidate.depth for candidate in result.invalidated], [2, 3])

    def test_depth_two_reject_commits_prefix_then_fallback_and_skips_depth_three(self) -> None:
        controller, decoder, scorer, _, policy, fallback, committer = build_controller(
            scores_by_depth={1: (0.8,), 2: (0.1,), 3: (0.9,)},
        )

        result = controller.run(ControlRequest(anchor_block=block(0), max_depth=3))

        self.assertEqual(result.accepted_depth, 1)
        self.assertEqual(result.rejected_depth, 2)
        self.assertEqual([commit.source for commit in committer.commits], ["anchor", "draft", "fallback"])
        self.assertEqual([commit.block.index for commit in committer.commits], [0, 1, 2])
        self.assertEqual(decoder.calls, [1, 2])
        self.assertEqual(scorer.calls, [1, 2])
        self.assertEqual(policy.calls, [1, 2])
        self.assertEqual([candidate.depth for candidate in result.invalidated], [3])
        self.assertEqual([candidate.depth for candidate in fallback.calls], [2])

    def test_score_equal_to_threshold_accepts(self) -> None:
        controller, _, _, _, _, fallback, committer = build_controller(
            scores_by_depth={1: (0.5,), 2: (0.8,), 3: (0.9,)},
            depths=(1,),
        )

        result = controller.run(ControlRequest(anchor_block=block(0), max_depth=1))

        self.assertEqual(result.accepted_depth, 1)
        self.assertEqual(fallback.calls, [])
        self.assertEqual([commit.source for commit in committer.commits], ["anchor", "draft"])

    def test_same_scores_min_rejects_mean_accepts_without_controller_changes(self) -> None:
        min_controller, _, _, _, _, min_fallback, _ = build_controller(
            scores_by_depth={1: (0.1, 0.9)},
            depths=(1,),
            aggregator=RecordingAggregator(MinScoreAggregator()),
            threshold=0.5,
        )
        mean_controller, _, _, _, _, mean_fallback, _ = build_controller(
            scores_by_depth={1: (0.1, 0.9)},
            depths=(1,),
            aggregator=RecordingAggregator(MeanScoreAggregator()),
            threshold=0.5,
        )

        min_result = min_controller.run(ControlRequest(anchor_block=block(0), max_depth=1))
        mean_result = mean_controller.run(ControlRequest(anchor_block=block(0), max_depth=1))

        self.assertEqual(min_result.rejected_depth, 1)
        self.assertEqual([candidate.depth for candidate in min_fallback.calls], [1])
        self.assertEqual(mean_result.accepted_depth, 1)
        self.assertEqual(mean_fallback.calls, [])

    def test_decoder_exception_uses_existing_rollback_semantics(self) -> None:
        decoder = RecordingDecoder(fail_depth=1)
        controller, _, _, _, _, _, committer = build_controller(
            scores_by_depth={1: (0.8,), 2: (0.8,), 3: (0.8,)},
            decoder=decoder,
        )

        with self.assertRaisesRegex(RuntimeError, "decoder failed"):
            controller.run(ControlRequest(anchor_block=block(0), max_depth=3))

        self.assertEqual(committer.commits, [])
        self.assertEqual(committer.rollbacks, 1)

    def test_scorer_exception_uses_existing_rollback_semantics(self) -> None:
        scorer = RecordingScorer(
            {1: (0.8,), 2: (0.8,), 3: (0.8,)},
            fail_depth=1,
        )
        controller, _, _, _, _, _, committer = build_controller(
            scores_by_depth={1: (0.8,), 2: (0.8,), 3: (0.8,)},
            scorer=scorer,
        )

        with self.assertRaisesRegex(RuntimeError, "scorer failed"):
            controller.run(ControlRequest(anchor_block=block(0), max_depth=3))

        self.assertEqual(committer.commits, [])
        self.assertEqual(committer.rollbacks, 1)

    def test_aggregator_exception_uses_existing_rollback_semantics(self) -> None:
        aggregator = RecordingAggregator(MinScoreAggregator(), fail=True)
        controller, _, _, _, _, _, committer = build_controller(
            scores_by_depth={1: (0.8,), 2: (0.8,), 3: (0.8,)},
            aggregator=aggregator,
        )

        with self.assertRaisesRegex(RuntimeError, "aggregator failed"):
            controller.run(ControlRequest(anchor_block=block(0), max_depth=3))

        self.assertEqual(committer.commits, [])
        self.assertEqual(committer.rollbacks, 1)

    def test_factory_config_constructs_controller_loop_components(self) -> None:
        evaluator, policy = create_verifier_from_config(
            {
                "speculative": {
                    "evaluator": {
                        "decoder": {"type": "identity"},
                        "scorer": {
                            "type": "scripted",
                            "scores_by_depth": {1: (0.8,), 2: (0.1,)},
                        },
                        "aggregator": {"type": "min_frame"},
                    },
                    "acceptance": {"type": "fixed_threshold", "threshold": 0.5},
                }
            }
        )
        batch, _ = make_batch((1, 2))
        fallback = FakeFallback()
        committer = FakeCommitter()
        controller = SpeculativeController(
            proposer=FakeProposal(batch),
            evaluator=evaluator,
            policy=policy,
            fallback=fallback,
            committer=committer,
        )

        result = controller.run(ControlRequest(anchor_block=block(0), max_depth=2))

        self.assertEqual(result.accepted_depth, 1)
        self.assertEqual(result.rejected_depth, 2)
        self.assertEqual([commit.source for commit in committer.commits], ["anchor", "draft", "fallback"])


if __name__ == "__main__":
    unittest.main()
