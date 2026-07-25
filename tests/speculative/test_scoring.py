from __future__ import annotations

import math
import unittest

from speculative.evaluation import IdentityCandidateDecoder
from speculative.scoring import (
    MeanScoreAggregator,
    MinScoreAggregator,
    RawScoreResult,
    ScoreResult,
    ScriptedCandidateScorer,
)
from speculative.types import BlockRef, DraftCandidate


def candidate_at_depth(depth: int) -> DraftCandidate:
    return DraftCandidate(
        block=BlockRef(index=depth),
        depth=depth,
        latent=f"latent-{depth}",
        source_noise=object(),
    )


class ScoringTest(unittest.TestCase):
    def test_min_score_aggregator_returns_worst_frame(self) -> None:
        self.assertEqual(MinScoreAggregator().aggregate((0.8, 0.1, 0.7)), 0.1)

    def test_mean_score_aggregator_returns_arithmetic_mean(self) -> None:
        self.assertAlmostEqual(MeanScoreAggregator().aggregate((0.8, 0.1, 0.7)), 1.6 / 3)

    def test_mean_overflow_result_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "mean score"):
            MeanScoreAggregator().aggregate((1e308, 1e308))

    def test_aggregators_reject_empty_or_non_finite_scores(self) -> None:
        invalid_scores = ((), (math.nan,), (math.inf,), (-math.inf,), (True,))
        for scores in invalid_scores:
            with self.subTest(scores=scores):
                with self.assertRaises(ValueError):
                    MinScoreAggregator().aggregate(scores)  # type: ignore[arg-type]
                with self.assertRaises(ValueError):
                    MeanScoreAggregator().aggregate(scores)  # type: ignore[arg-type]

    def test_raw_score_result_rejects_invalid_scores(self) -> None:
        with self.assertRaisesRegex(ValueError, "per_frame_scores"):
            RawScoreResult(per_frame_scores=(), scorer_name="scripted")
        with self.assertRaisesRegex(ValueError, "finite"):
            RawScoreResult(per_frame_scores=(False,), scorer_name="scripted")  # type: ignore[arg-type]

    def test_score_result_rejects_invalid_block_score(self) -> None:
        with self.assertRaisesRegex(ValueError, "block_score"):
            ScoreResult(
                per_frame_scores=(0.1,),
                block_score=math.nan,
                scorer_name="scripted",
                aggregator_name="min_frame",
            )

    def test_score_metadata_is_frozen_and_copied(self) -> None:
        metadata = {"labels": ["before"], "nested": {"score": 1.0}}
        result = RawScoreResult(
            per_frame_scores=(0.1,),
            scorer_name="scripted",
            metadata=metadata,
        )
        metadata["labels"].append("after")
        metadata["nested"]["score"] = 2.0

        self.assertEqual(result.metadata["labels"], ("before",))
        self.assertEqual(result.metadata["nested"]["score"], 1.0)
        with self.assertRaises(TypeError):
            result.metadata["labels"] = ("after",)  # type: ignore[index]

    def test_scripted_scorer_uses_explicit_depth_scores(self) -> None:
        scorer = ScriptedCandidateScorer(scores_by_depth={1: (0.8, 0.7)})
        decoded = IdentityCandidateDecoder().decode(candidate_at_depth(1))

        result = scorer.score(decoded)

        self.assertEqual(result.per_frame_scores, (0.8, 0.7))
        self.assertEqual(result.scorer_name, "scripted")
        self.assertEqual(result.metadata["depth"], 1)

    def test_scripted_scorer_rejects_missing_depth(self) -> None:
        scorer = ScriptedCandidateScorer(scores_by_depth={1: (0.8,)})
        decoded = IdentityCandidateDecoder().decode(candidate_at_depth(2))

        with self.assertRaisesRegex(ValueError, "No scripted scores"):
            scorer.score(decoded)

    def test_scripted_scorer_rejects_invalid_table(self) -> None:
        bad_tables = (
            {True: (0.1,)},
            {1: ()},
            {1: (math.inf,)},
        )
        for table in bad_tables:
            with self.subTest(table=table):
                with self.assertRaises(ValueError):
                    ScriptedCandidateScorer(scores_by_depth=table)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
