from __future__ import annotations

import math
import unittest

from speculative.policies.fixed_threshold import FixedThresholdPolicy
from speculative.scoring import ScoreResult
from speculative.types import BlockRef, DraftCandidate, Evaluation


def evaluation_with_score(block_score: float) -> Evaluation:
    candidate = DraftCandidate(
        block=BlockRef(index=1),
        depth=1,
        latent={"latent": 1},
        source_noise=object(),
    )
    return Evaluation(
        candidate=candidate,
        value=ScoreResult(
            per_frame_scores=(block_score,),
            block_score=block_score,
            scorer_name="scripted",
            aggregator_name="min_frame",
        ),
    )


class FixedThresholdPolicyTest(unittest.TestCase):
    def test_accepts_when_block_score_is_above_threshold(self) -> None:
        decision = FixedThresholdPolicy(threshold=0.5).decide(evaluation_with_score(0.7))

        self.assertTrue(decision.accepted)
        self.assertEqual(decision.action, "accept")
        self.assertEqual(decision.metadata["policy_name"], "fixed_threshold")
        self.assertEqual(decision.metadata["threshold"], 0.5)
        self.assertEqual(decision.metadata["block_score"], 0.7)

    def test_accepts_when_block_score_equals_threshold(self) -> None:
        decision = FixedThresholdPolicy(threshold=0.5).decide(evaluation_with_score(0.5))

        self.assertTrue(decision.accepted)

    def test_rejects_when_block_score_is_below_threshold(self) -> None:
        decision = FixedThresholdPolicy(threshold=0.5).decide(evaluation_with_score(0.49))

        self.assertFalse(decision.accepted)
        self.assertEqual(decision.action, "reject")
        self.assertIn("< threshold", decision.reason)

    def test_threshold_must_be_explicit_finite_number(self) -> None:
        for value in (None, True, math.nan, math.inf, -math.inf, "0.5"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    FixedThresholdPolicy(threshold=value)  # type: ignore[arg-type]

    def test_requires_score_result_evaluation_value(self) -> None:
        candidate = DraftCandidate(
            block=BlockRef(index=1),
            depth=1,
            latent=object(),
            source_noise=object(),
        )
        with self.assertRaisesRegex(TypeError, "ScoreResult"):
            FixedThresholdPolicy(threshold=0.5).decide(Evaluation(candidate=candidate, value=0.5))

    def test_decision_metadata_is_frozen(self) -> None:
        decision = FixedThresholdPolicy(threshold=0.5).decide(evaluation_with_score(0.7))

        with self.assertRaises(TypeError):
            decision.metadata["threshold"] = 0.1  # type: ignore[index]

    def test_policy_does_not_mutate_candidate_or_generation_state(self) -> None:
        latent = {"value": "draft"}
        source_noise = object()
        candidate = DraftCandidate(
            block=BlockRef(index=1),
            depth=1,
            latent=latent,
            source_noise=source_noise,
        )
        evaluation = Evaluation(
            candidate=candidate,
            value=ScoreResult(
                per_frame_scores=(0.7,),
                block_score=0.7,
                scorer_name="scripted",
                aggregator_name="min_frame",
            ),
        )
        fake_kv = {"cursor": 0}
        fake_output = ["anchor"]
        fake_commits: list[str] = []

        before = (dict(fake_kv), list(fake_output), list(fake_commits))
        decision = FixedThresholdPolicy(threshold=0.5).decide(evaluation)

        self.assertTrue(decision.accepted)
        self.assertIs(candidate.latent, latent)
        self.assertIs(candidate.source_noise, source_noise)
        self.assertEqual((fake_kv, fake_output, fake_commits), before)


if __name__ == "__main__":
    unittest.main()
