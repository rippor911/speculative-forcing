from __future__ import annotations

import unittest

from speculative.evaluation import (
    CompositeCandidateEvaluator,
    DecodedCandidate,
    IdentityCandidateDecoder,
)
from speculative.scoring import MinScoreAggregator, RawScoreResult, ScoreResult
from speculative.types import BlockRef, DraftCandidate, Evaluation


def candidate_at_depth(depth: int) -> DraftCandidate:
    return DraftCandidate(
        block=BlockRef(index=depth),
        depth=depth,
        latent={"latent": depth},
        source_noise=object(),
    )


class ConstantScorer:
    def score(self, decoded: DecodedCandidate) -> RawScoreResult:
        return RawScoreResult(
            per_frame_scores=(0.8, 0.7),
            scorer_name="constant",
            metadata={"depth": decoded.candidate.depth},
        )


class BadDecoder:
    def decode(self, candidate: DraftCandidate) -> DecodedCandidate:
        other = DraftCandidate(
            block=candidate.block,
            depth=candidate.depth,
            latent=candidate.latent,
            source_noise=candidate.source_noise,
        )
        return DecodedCandidate(candidate=other, payload=candidate.latent)


class BadScorer:
    def score(self, decoded: DecodedCandidate) -> object:
        return "not raw scores"


class EvaluationTest(unittest.TestCase):
    def test_identity_decoder_preserves_candidate_and_latent_identity(self) -> None:
        candidate = candidate_at_depth(1)
        decoded = IdentityCandidateDecoder().decode(candidate)

        self.assertIs(decoded.candidate, candidate)
        self.assertIs(decoded.payload, candidate.latent)
        self.assertEqual(decoded.metadata["depth"], 1)

    def test_decoded_candidate_metadata_is_frozen(self) -> None:
        metadata = {"labels": ["before"]}
        decoded = DecodedCandidate(
            candidate=candidate_at_depth(1),
            payload=object(),
            metadata=metadata,
        )
        metadata["labels"].append("after")

        self.assertEqual(decoded.metadata["labels"], ("before",))
        with self.assertRaises(TypeError):
            decoded.metadata["labels"] = ("after",)  # type: ignore[index]

    def test_composite_evaluator_returns_existing_evaluation_with_score_result(self) -> None:
        candidate = candidate_at_depth(1)
        evaluator = CompositeCandidateEvaluator(
            decoder=IdentityCandidateDecoder(),
            scorer=ConstantScorer(),
            aggregator=MinScoreAggregator(),
        )

        evaluation = evaluator.evaluate(candidate)

        self.assertIsInstance(evaluation, Evaluation)
        self.assertIs(evaluation.candidate, candidate)
        self.assertIsInstance(evaluation.value, ScoreResult)
        self.assertEqual(evaluation.value.per_frame_scores, (0.8, 0.7))
        self.assertEqual(evaluation.value.block_score, 0.7)
        self.assertEqual(evaluation.value.scorer_name, "constant")
        self.assertEqual(evaluation.value.aggregator_name, "min_frame")
        self.assertEqual(evaluation.metadata["block_score"], 0.7)

    def test_composite_evaluator_rejects_decoder_candidate_mismatch(self) -> None:
        evaluator = CompositeCandidateEvaluator(
            decoder=BadDecoder(),
            scorer=ConstantScorer(),
            aggregator=MinScoreAggregator(),
        )

        with self.assertRaisesRegex(ValueError, "current candidate"):
            evaluator.evaluate(candidate_at_depth(1))

    def test_composite_evaluator_rejects_wrong_scorer_return_type(self) -> None:
        evaluator = CompositeCandidateEvaluator(
            decoder=IdentityCandidateDecoder(),
            scorer=BadScorer(),
            aggregator=MinScoreAggregator(),
        )

        with self.assertRaisesRegex(TypeError, "RawScoreResult"):
            evaluator.evaluate(candidate_at_depth(1))


if __name__ == "__main__":
    unittest.main()
