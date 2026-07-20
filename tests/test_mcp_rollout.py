import unittest
from types import SimpleNamespace

import torch

from pipeline.self_forcing_training import SelfForcingTrainingPipeline


def make_pipeline(**overrides):
    kwargs = {
        "denoising_step_list": torch.tensor([1000, 750, 500, 250]),
        "scheduler": SimpleNamespace(),
        "generator": SimpleNamespace(),
        "num_frame_per_block": 3,
        "num_max_frames": 21,
        "mcp_num_modules": 3,
        "mcp_accel_depths": 1,
    }
    kwargs.update(overrides)
    return SelfForcingTrainingPipeline(**kwargs)


class MCPRolloutLayoutTest(unittest.TestCase):
    def test_future_chunks_follow_absolute_block_offsets(self):
        pipeline = make_pipeline()
        noise = torch.arange(21.0).reshape(1, 21, 1, 1, 1)
        block_starts = pipeline._block_start_frames(0, [3] * 7)

        chunks, starts = pipeline._mcp_future_chunks(
            block_index=0,
            block_starts=block_starts,
            all_num_frames=[3] * 7,
            noise=noise,
            num_input_frames=0,
        )

        self.assertEqual(starts, [3, 6, 9])
        torch.testing.assert_close(chunks[0].flatten(), torch.tensor([3.0, 4.0, 5.0]))
        torch.testing.assert_close(chunks[1].flatten(), torch.tensor([6.0, 7.0, 8.0]))
        torch.testing.assert_close(chunks[2].flatten(), torch.tensor([9.0, 10.0, 11.0]))

    def test_future_chunks_stop_at_video_boundary(self):
        pipeline = make_pipeline()
        noise = torch.zeros(1, 21, 1, 1, 1)
        block_starts = pipeline._block_start_frames(0, [3] * 7)

        chunks, starts = pipeline._mcp_future_chunks(
            block_index=4,
            block_starts=block_starts,
            all_num_frames=[3] * 7,
            noise=noise,
            num_input_frames=0,
        )

        self.assertIsNotNone(chunks[0])
        self.assertIsNotNone(chunks[1])
        self.assertIsNone(chunks[2])
        self.assertEqual(starts, [15, 18, None])

    def test_acceleration_depth_must_not_exceed_trained_heads(self):
        with self.assertRaisesRegex(ValueError, "mcp_accel_depths"):
            make_pipeline(mcp_num_modules=2, mcp_accel_depths=3)

    def test_memory_gap_and_mcp_rollout_are_rejected_together(self):
        with self.assertRaisesRegex(ValueError, "not composable"):
            make_pipeline(memory_gap_blocks=1)


if __name__ == "__main__":
    unittest.main()
