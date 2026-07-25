from __future__ import annotations

import unittest

from speculative.evaluation import CompositeCandidateEvaluator, IdentityCandidateDecoder
from speculative.factory import (
    AGGREGATOR_FACTORIES,
    DECODER_FACTORIES,
    POLICY_FACTORIES,
    SCORER_FACTORIES,
    create_acceptance_policy_from_config,
    create_aggregator_from_config,
    create_decoder_from_config,
    create_evaluator_from_config,
    create_policy,
    create_scorer_from_config,
    create_verifier_from_config,
)
from speculative.policies.fixed_threshold import FixedThresholdPolicy
from speculative.policies.scripted import ScriptedPolicy
from speculative.scoring import MeanScoreAggregator, MinScoreAggregator, ScriptedCandidateScorer
from speculative.types import BlockRef, DraftCandidate


class FactoryTest(unittest.TestCase):
    def test_factory_maps_are_explicit_whitelists(self) -> None:
        self.assertIn("identity", DECODER_FACTORIES)
        self.assertIn("scripted", SCORER_FACTORIES)
        self.assertIn("min_frame", AGGREGATOR_FACTORIES)
        self.assertIn("mean_frame", AGGREGATOR_FACTORIES)
        self.assertIn("fixed_threshold", POLICY_FACTORIES)

    def test_create_policy_keeps_existing_scripted_policy_api(self) -> None:
        self.assertIsInstance(create_policy("always_accept"), ScriptedPolicy)
        self.assertIsInstance(create_policy("always_reject"), ScriptedPolicy)
        self.assertIsInstance(create_policy("reject_at_depth", reject_depth=1), ScriptedPolicy)
        self.assertIsInstance(create_policy("fixed_threshold", threshold=0.5), FixedThresholdPolicy)

    def test_component_config_factories_construct_known_components(self) -> None:
        self.assertIsInstance(create_decoder_from_config({"type": "identity"}), IdentityCandidateDecoder)
        self.assertIsInstance(
            create_scorer_from_config(
                {"type": "scripted", "scores_by_depth": {1: (0.8,)}}
            ),
            ScriptedCandidateScorer,
        )
        self.assertIsInstance(create_aggregator_from_config({"type": "min_frame"}), MinScoreAggregator)
        self.assertIsInstance(create_aggregator_from_config({"type": "mean_frame"}), MeanScoreAggregator)
        self.assertIsInstance(
            create_acceptance_policy_from_config(
                {"type": "fixed_threshold", "threshold": 0.5}
            ),
            FixedThresholdPolicy,
        )

    def test_create_verifier_from_config_constructs_evaluator_and_policy(self) -> None:
        evaluator, policy = create_verifier_from_config(
            {
                "speculative": {
                    "evaluator": {
                        "decoder": {"type": "identity"},
                        "scorer": {
                            "type": "scripted",
                            "scores_by_depth": {1: (0.8, 0.7)},
                        },
                        "aggregator": {"type": "min_frame"},
                    },
                    "acceptance": {
                        "type": "fixed_threshold",
                        "threshold": 0.5,
                    },
                }
            }
        )

        candidate = DraftCandidate(
            block=BlockRef(index=1),
            depth=1,
            latent=object(),
            source_noise=object(),
        )
        evaluation = evaluator.evaluate(candidate)
        decision = policy.decide(evaluation)

        self.assertIsInstance(evaluator, CompositeCandidateEvaluator)
        self.assertTrue(decision.accepted)
        self.assertEqual(evaluation.value.block_score, 0.7)

    def test_unknown_type_errors_include_field_path(self) -> None:
        with self.assertRaisesRegex(ValueError, "speculative.evaluator.decoder.type"):
            create_decoder_from_config({"type": "dynamic.import.Path"})
        with self.assertRaisesRegex(ValueError, "speculative.acceptance.type"):
            create_acceptance_policy_from_config({"type": "dynamic.import.Path"})

    def test_unknown_config_root_field_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "config has unknown field"):
            create_verifier_from_config(
                {
                    "speculative": {
                        "evaluator": {
                            "decoder": {"type": "identity"},
                            "scorer": {
                                "type": "scripted",
                                "scores_by_depth": {1: (0.8,)},
                            },
                            "aggregator": {"type": "min_frame"},
                        },
                        "acceptance": {
                            "type": "fixed_threshold",
                            "threshold": 0.5,
                        },
                    },
                    "other": {},
                }
            )

    def test_unknown_speculative_field_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "speculative has unknown field"):
            create_verifier_from_config(
                {
                    "speculative": {
                        "evaluator": {
                            "decoder": {"type": "identity"},
                            "scorer": {
                                "type": "scripted",
                                "scores_by_depth": {1: (0.8,)},
                            },
                            "aggregator": {"type": "min_frame"},
                        },
                        "acceptance": {
                            "type": "fixed_threshold",
                            "threshold": 0.5,
                        },
                        "verifer": {"type": "typo"},
                    },
                }
            )

    def test_canonical_sibling_config_is_accepted_but_not_constructed(self) -> None:
        evaluator, policy = create_verifier_from_config(
            {
                "speculative": {
                    "controller": {"type": "longest_prefix"},
                    "proposal": {"type": "self_forcing_mcp", "depth": 3},
                    "evaluator": {
                        "decoder": {"type": "identity"},
                        "scorer": {
                            "type": "scripted",
                            "scores_by_depth": {1: (0.8,)},
                        },
                        "aggregator": {"type": "min_frame"},
                    },
                    "acceptance": {
                        "type": "fixed_threshold",
                        "threshold": 0.5,
                    },
                    "fallback": {
                        "type": "target_regenerate",
                        "reuse_candidate_noise": True,
                    },
                    "verification": {"mode": "sequential"},
                    "trace": {"enabled": True, "schema_version": 1},
                }
            }
        )

        candidate = DraftCandidate(
            block=BlockRef(index=1),
            depth=1,
            latent=object(),
            source_noise=object(),
        )
        evaluation = evaluator.evaluate(candidate)
        decision = policy.decide(evaluation)

        self.assertIsInstance(evaluator, CompositeCandidateEvaluator)
        self.assertTrue(decision.accepted)
        self.assertEqual(evaluation.value.block_score, 0.8)

    def test_unknown_evaluator_field_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "speculative.evaluator has unknown field"):
            create_evaluator_from_config(
                {
                    "decoder": {"type": "identity"},
                    "scorer": {
                        "type": "scripted",
                        "scores_by_depth": {1: (0.8,)},
                    },
                    "aggregator": {"type": "min_frame"},
                    "threshold": 0.5,
                }
            )

    def test_missing_type_errors_include_field_path(self) -> None:
        with self.assertRaisesRegex(ValueError, "speculative.evaluator.scorer.type must be set"):
            create_scorer_from_config({"scores_by_depth": {1: (0.8,)}})

    def test_missing_threshold_errors_include_field_path(self) -> None:
        with self.assertRaisesRegex(ValueError, "speculative.acceptance.threshold must be set"):
            create_acceptance_policy_from_config({"type": "fixed_threshold"})

    def test_missing_scores_errors_include_field_path(self) -> None:
        with self.assertRaisesRegex(ValueError, "speculative.evaluator.scorer.scores_by_depth"):
            create_scorer_from_config({"type": "scripted"})

    def test_depth_thresholds_are_not_silently_enabled(self) -> None:
        with self.assertRaisesRegex(ValueError, "depth_thresholds"):
            create_acceptance_policy_from_config(
                {
                    "type": "fixed_threshold",
                    "threshold": 0.5,
                    "depth_thresholds": {1: 0.5},
                }
            )


if __name__ == "__main__":
    unittest.main()
