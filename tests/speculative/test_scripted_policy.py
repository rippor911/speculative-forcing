from __future__ import annotations

import unittest

from speculative.factory import create_policy
from speculative.policies.scripted import ScriptedPolicy
from speculative.types import BlockRef, DraftCandidate, Evaluation


def evaluation_at_depth(depth: int) -> Evaluation:
    candidate = DraftCandidate(
        block=BlockRef(index=depth),
        depth=depth,
        latent=f"draft-{depth}",
        source_noise=object(),
    )
    return Evaluation(candidate=candidate)


class ScriptedPolicyTest(unittest.TestCase):
    def test_always_accept(self) -> None:
        policy = ScriptedPolicy.always_accept()

        decision = policy.decide(evaluation_at_depth(3))

        self.assertTrue(decision.accepted)
        self.assertEqual(decision.action, "accept")

    def test_always_reject(self) -> None:
        policy = ScriptedPolicy.always_reject()

        decision = policy.decide(evaluation_at_depth(1))

        self.assertFalse(decision.accepted)
        self.assertEqual(decision.action, "reject")

    def test_reject_at_depth(self) -> None:
        policy = ScriptedPolicy.reject_at_depth(2)

        self.assertEqual(policy.decide(evaluation_at_depth(1)).action, "accept")
        self.assertEqual(policy.decide(evaluation_at_depth(2)).action, "reject")
        self.assertEqual(policy.decide(evaluation_at_depth(3)).action, "reject")

    def test_factory_map_creates_known_policies(self) -> None:
        self.assertEqual(create_policy("always_accept").decide(evaluation_at_depth(1)).action, "accept")
        self.assertEqual(create_policy("always_reject").decide(evaluation_at_depth(1)).action, "reject")
        self.assertEqual(
            create_policy("reject_at_depth", reject_depth=1).decide(evaluation_at_depth(1)).action,
            "reject",
        )

    def test_factory_rejects_unknown_policy(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown speculative policy"):
            create_policy("dynamic_import_me")

    def test_factory_rejects_unknown_kwargs(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown argument"):
            create_policy("always_accept", reject_depth=1)
        with self.assertRaisesRegex(ValueError, "Unknown argument"):
            create_policy("always_reject", typo=True)
        with self.assertRaisesRegex(ValueError, "Unknown argument"):
            create_policy("reject_at_depth", depth=1)
        with self.assertRaisesRegex(ValueError, "Unknown argument"):
            create_policy("reject_at_depth", reject_depth=1, typo=True)

    def test_reject_at_depth_requires_positive_depth(self) -> None:
        with self.assertRaisesRegex(ValueError, "reject_depth > 0"):
            ScriptedPolicy.reject_at_depth(0)
        with self.assertRaisesRegex(ValueError, "requires reject_depth"):
            create_policy("reject_at_depth")
        with self.assertRaisesRegex(ValueError, "must be an integer"):
            create_policy("reject_at_depth", reject_depth="1")

    def test_direct_reject_depth_requires_positive_strict_integer(self) -> None:
        for value in (True, 1.0, "1", 0, -1):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    ScriptedPolicy.reject_at_depth(value)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
