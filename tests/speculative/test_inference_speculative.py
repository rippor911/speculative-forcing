from __future__ import annotations

import argparse
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

from inference_speculative import (
    ReferenceRNGPlan,
    ScriptedCandidateEvaluator,
    build_controller,
    build_policy,
    build_trace_payload,
    consume_reference_rollout_setup_rng,
    next_control_request,
    run_speculative_rollout,
    validate_policy_args,
    validate_speculative_config,
)
from speculative.adapters.self_forcing_runtime import SelfForcingMCPRolloutPlan
from speculative.types import BlockRef, CommitRequest, ControlRequest, DraftCandidate, FallbackResult, ProposalBatch


def block(index: int, *, frames: int = 3) -> BlockRef:
    return BlockRef(index=index, start_frame=index * frames, num_frames=frames)


def args_for(
    *,
    policy: str = "always_accept",
    reject_depth: Optional[int] = None,
    mcp_depth: int = 3,
    checkpoint: str = "checkpoint.pt",
    num_frames: int = 12,
) -> argparse.Namespace:
    return argparse.Namespace(
        config="config.yaml",
        checkpoint=checkpoint,
        prompt="prompt",
        output="out.mp4",
        seed=0,
        num_frames=num_frames,
        mcp_depth=mcp_depth,
        policy=policy,
        reject_depth=reject_depth,
        save_trace=None,
        fps=16,
        device="cuda",
    )


class ScalarIndex:
    def __init__(self, value: int = 0) -> None:
        self.value = int(value)

    def item(self) -> int:
        return self.value

    def set(self, value: int) -> None:
        self.value = int(value)


class FakeRuntime:
    def __init__(self, *, block_count: int, mcp_depth: int, frames: int = 3) -> None:
        blocks = tuple(block(index, frames=frames) for index in range(block_count))
        self.rollout_plan = SelfForcingMCPRolloutPlan(
            blocks=blocks,
            block_starts=tuple(item.start_frame for item in blocks),
            anchor_block_indices=tuple(range(0, block_count, mcp_depth + 1)),
            period=mcp_depth + 1,
        )
        self._committed_blocks: list[BlockRef] = []
        self.prepare_calls = 0
        self.requests: list[ControlRequest] = []
        self.fallbacks: list[DraftCandidate] = []
        self.commits: list[CommitRequest] = []
        self.noises = [object() for _ in range(block_count)]
        self.kv_cache = [
            {
                "global_end_index": ScalarIndex(0),
                "local_end_index": ScalarIndex(0),
            }
        ]
        self._open_start = 0

    @property
    def committed_blocks(self) -> tuple[BlockRef, ...]:
        return tuple(self._committed_blocks)

    def prepare(self) -> None:
        self.prepare_calls += 1

    def propose_window(self, request: ControlRequest) -> ProposalBatch:
        self.requests.append(request)
        anchor = CommitRequest(
            block=request.anchor_block,
            latent=f"anchor-{request.anchor_block.index}",
            source="anchor",
        )
        remaining = len(self.rollout_plan.blocks) - request.anchor_block.index - 1
        depth_count = min(request.max_depth, remaining)
        drafts = []
        for depth in range(1, depth_count + 1):
            target = self.rollout_plan.blocks[request.anchor_block.index + depth]
            drafts.append(
                DraftCandidate(
                    block=target,
                    depth=depth,
                    latent=f"draft-{target.index}",
                    source_noise=self.noises[target.index],
                )
            )
        return ProposalBatch(anchor=anchor, drafts=tuple(drafts))

    def generate_target_fallback(self, candidate: DraftCandidate) -> FallbackResult:
        self.fallbacks.append(candidate)
        return FallbackResult(
            block=candidate.block,
            latent=f"fallback-{candidate.block.index}",
            source_noise=candidate.source_noise,
        )

    def begin_window(self) -> None:
        self._open_start = len(self._committed_blocks)

    def commit_block(self, request: CommitRequest) -> None:
        self.commits.append(request)
        self._committed_blocks.append(request.block)
        end_index = request.block.start_frame + request.block.num_frames
        for layer in self.kv_cache:
            layer["global_end_index"].set(end_index)
            layer["local_end_index"].set(end_index)

    def complete_window(self) -> None:
        self._open_start = len(self._committed_blocks)

    def rollback_window(self) -> None:
        del self._committed_blocks[self._open_start:]
        del self.commits[self._open_start:]


class FakeRolloutPipeline:
    def __init__(self) -> None:
        self.denoising_step_list = [1000]
        self.calls: list[tuple[int, int, object]] = []

    def generate_and_sync_list(
        self,
        num_blocks: int,
        num_denoising_steps: int,
        device: object,
    ) -> list[int]:
        self.calls.append((num_blocks, num_denoising_steps, device))
        return [0] * num_blocks


class InferenceSpeculativeValidationTest(unittest.TestCase):
    def test_reject_at_depth_requires_positive_depth_within_mcp_depth(self) -> None:
        for bad_depth in (None, 0, -1):
            with self.subTest(bad_depth=bad_depth):
                with self.assertRaises(ValueError):
                    validate_policy_args(args_for(policy="reject_at_depth", reject_depth=bad_depth))
        with self.assertRaisesRegex(ValueError, "<= --mcp_depth"):
            validate_policy_args(args_for(policy="reject_at_depth", reject_depth=3, mcp_depth=2))

    def test_other_policies_forbid_reject_depth(self) -> None:
        for policy in ("always_accept", "always_reject"):
            with self.subTest(policy=policy):
                with self.assertRaisesRegex(ValueError, "does not accept"):
                    validate_policy_args(args_for(policy=policy, reject_depth=1))

    def test_config_validation_reuses_mcp_constraints(self) -> None:
        checkpoint = Path("inference_mcp.py").resolve()
        config = SimpleNamespace(
            i2v=False,
            num_frame_per_block=3,
            mcp_num_modules=3,
        )
        validate_speculative_config(
            config,
            args_for(checkpoint=str(checkpoint), num_frames=12, mcp_depth=3),
        )
        with self.assertRaisesRegex(ValueError, "divisible"):
            validate_speculative_config(
                config,
                args_for(checkpoint=str(checkpoint), num_frames=10, mcp_depth=3),
            )


class InferenceSpeculativePolicyTest(unittest.TestCase):
    def test_policy_construction_uses_factory(self) -> None:
        candidate = DraftCandidate(
            block=block(1),
            depth=1,
            latent="draft",
            source_noise=object(),
        )
        evaluation = ScriptedCandidateEvaluator().evaluate(candidate)

        self.assertEqual(build_policy(args_for(policy="always_accept")).decide(evaluation).action, "accept")
        self.assertEqual(build_policy(args_for(policy="always_reject")).decide(evaluation).action, "reject")
        self.assertEqual(
            build_policy(args_for(policy="reject_at_depth", reject_depth=1)).decide(evaluation).action,
            "reject",
        )
        self.assertIs(evaluation.candidate, candidate)


class InferenceSpeculativeLoopTest(unittest.TestCase):
    def run_fake(self, *, policy: str, block_count: int, mcp_depth: int, reject_depth: Optional[int] = None) -> tuple[FakeRuntime, list[dict[str, Any]]]:
        runtime = FakeRuntime(block_count=block_count, mcp_depth=mcp_depth)
        controller = build_controller(
            runtime,
            build_policy(args_for(policy=policy, reject_depth=reject_depth, mcp_depth=mcp_depth)),
        )
        windows = run_speculative_rollout(
            runtime=runtime,
            controller=controller,
            mcp_depth=mcp_depth,
            kv_cache=runtime.kv_cache,
        )
        return runtime, windows

    def test_next_request_uses_runtime_committed_prefix(self) -> None:
        runtime = FakeRuntime(block_count=4, mcp_depth=2)
        request = next_control_request(runtime, 2)
        self.assertEqual(request.anchor_block.index, 0)
        self.assertEqual(request.max_depth, 2)
        runtime._committed_blocks.extend(runtime.rollout_plan.blocks[:2])
        request = next_control_request(runtime, 2)
        self.assertEqual(request.anchor_block.index, 2)
        self.assertEqual(request.max_depth, 1)

    def test_multi_window_always_accept(self) -> None:
        runtime, windows = self.run_fake(policy="always_accept", block_count=5, mcp_depth=2)

        self.assertEqual(runtime.prepare_calls, 1)
        self.assertEqual([request.anchor_block.index for request in runtime.requests], [0, 3])
        self.assertEqual([request.max_depth for request in runtime.requests], [2, 1])
        self.assertEqual([block.index for block in runtime.committed_blocks], [0, 1, 2, 3, 4])
        self.assertEqual([window["next_anchor"]["index"] if window["next_anchor"] else None for window in windows], [3, None])

    def test_multi_window_always_reject(self) -> None:
        runtime, windows = self.run_fake(policy="always_reject", block_count=5, mcp_depth=2)

        self.assertEqual([request.anchor_block.index for request in runtime.requests], [0, 2, 4])
        self.assertEqual([request.max_depth for request in runtime.requests], [2, 2, 0])
        self.assertEqual([commit.source for commit in runtime.commits], ["anchor", "fallback", "anchor", "fallback", "anchor"])
        self.assertEqual([block.index for block in runtime.committed_blocks], [0, 1, 2, 3, 4])
        self.assertEqual([item["depth"] for item in windows[0]["invalidated"]], [2])

    def test_reject_at_depth_two_commits_prefix_fallback_and_invalidates(self) -> None:
        runtime, windows = self.run_fake(
            policy="reject_at_depth",
            reject_depth=2,
            block_count=4,
            mcp_depth=3,
        )

        self.assertEqual([commit.source for commit in runtime.commits[:3]], ["anchor", "draft", "fallback"])
        self.assertEqual([commit.block.index for commit in runtime.commits[:3]], [0, 1, 2])
        self.assertEqual(windows[0]["accepted_depth"], 1)
        self.assertEqual(windows[0]["rejected_depth"], 2)
        self.assertEqual([item["depth"] for item in windows[0]["invalidated"]], [3])
        self.assertEqual(runtime.fallbacks[0].depth, 2)
        self.assertEqual(windows[0]["next_anchor"]["index"], 3)

    def test_last_window_max_depth_can_be_less_than_mcp_depth(self) -> None:
        runtime, _ = self.run_fake(policy="always_accept", block_count=6, mcp_depth=3)

        self.assertEqual([request.anchor_block.index for request in runtime.requests], [0, 4])
        self.assertEqual([request.max_depth for request in runtime.requests], [3, 1])

    def test_trace_payload_is_json_serializable(self) -> None:
        runtime, windows = self.run_fake(policy="always_reject", block_count=3, mcp_depth=2)
        trace = build_trace_payload(
            args=args_for(policy="always_reject", mcp_depth=2, num_frames=9),
            config=SimpleNamespace(num_frame_per_block=3),
            load_mode="MCP_COMPLETE_STRICT_RESTORE",
            mcp_tensor_count=172,
            runtime=runtime,
            windows=windows,
            kv_cache=runtime.kv_cache,
            reference_rng=ReferenceRNGPlan(
                mode="vanilla_target_only",
                draw_count=3,
                exit_flags=(0, 0, 0),
            ),
        )

        json.dumps(trace, allow_nan=False)
        self.assertEqual(trace["windows"][0]["anchor_block"]["index"], 0)
        self.assertEqual(trace["final_kv_index"], {"global": 9, "local": 9})
        self.assertEqual(trace["reference_rng_mode"], "vanilla_target_only")
        self.assertEqual(trace["reference_rng_draw_count"], 3)


class InferenceSpeculativeReferenceRNGTest(unittest.TestCase):
    def consume(
        self,
        *,
        policy: str,
        block_count: int,
        mcp_depth: int,
        reject_depth: Optional[int] = None,
    ) -> tuple[ReferenceRNGPlan, FakeRolloutPipeline]:
        runtime = FakeRuntime(block_count=block_count, mcp_depth=mcp_depth)
        pipeline = FakeRolloutPipeline()
        plan = consume_reference_rollout_setup_rng(
            rollout_pipeline=pipeline,
            runtime=runtime,
            device="cpu",
            policy_name=policy,
            reject_depth=reject_depth,
        )
        return plan, pipeline

    def test_always_accept_rng_draw_count_matches_frozen_mcp_anchors(self) -> None:
        plan, pipeline = self.consume(
            policy="always_accept",
            block_count=10,
            mcp_depth=3,
        )

        self.assertEqual(plan.mode, "frozen_mcp_always_accept")
        self.assertEqual(plan.draw_count, 3)
        self.assertEqual(pipeline.calls[0][0], 3)
        self.assertEqual(plan.exit_flags, (0, 0, 0))

    def test_always_reject_rng_draw_count_matches_vanilla_block_count(self) -> None:
        plan, pipeline = self.consume(
            policy="always_reject",
            block_count=5,
            mcp_depth=3,
        )

        self.assertEqual(plan.mode, "vanilla_target_only")
        self.assertEqual(plan.draw_count, 5)
        self.assertEqual(pipeline.calls[0][0], 5)

    def test_reject_at_depth_two_rng_draw_count_matches_scripted_window_count(self) -> None:
        plan, pipeline = self.consume(
            policy="reject_at_depth",
            reject_depth=2,
            block_count=8,
            mcp_depth=3,
        )

        self.assertEqual(plan.mode, "scripted_reject_at_depth")
        self.assertEqual(plan.draw_count, 3)
        self.assertEqual(pipeline.calls[0][0], 3)

    def test_reject_at_depth_final_short_window_accepts_remaining_draft(self) -> None:
        plan, pipeline = self.consume(
            policy="reject_at_depth",
            reject_depth=2,
            block_count=5,
            mcp_depth=3,
        )

        self.assertEqual(plan.draw_count, 2)
        self.assertEqual(pipeline.calls[0][0], 2)


if __name__ == "__main__":
    unittest.main()
