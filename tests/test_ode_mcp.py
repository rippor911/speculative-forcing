import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from model.ode_regression import ODERegression


def make_uninitialized_ode_model():
    """Construct only the lightweight state needed by MCP helper methods."""
    model = ODERegression.__new__(ODERegression)
    torch.nn.Module.__init__(model)
    model.num_frame_per_block = 2
    model.mcp_num_modules = 2
    model.mcp_depth_weights = [0.5, 0.2]
    model.denoising_step_list = torch.tensor([1000, 500, 0])
    model.scheduler = SimpleNamespace(
        timesteps=torch.tensor([1000.0, 500.0, 0.0]),
        sigmas=torch.tensor([1.0, 0.5, 0.0]),
    )
    return model


class ODEMCPDataTest(unittest.TestCase):
    def setUp(self):
        self.model = make_uninitialized_ode_model()

    def test_shift_chunks_repeats_last_chunk(self):
        latent = torch.arange(6).reshape(1, 6, 1, 1, 1)
        shifted_one = self.model._shift_chunks(latent, 1)
        shifted_two = self.model._shift_chunks(latent, 2)

        torch.testing.assert_close(
            shifted_one.flatten(), torch.tensor([2, 3, 4, 5, 4, 5])
        )
        torch.testing.assert_close(
            shifted_two.flatten(), torch.tensor([4, 5, 4, 5, 4, 5])
        )

    def test_shift_chunks_rejects_partial_chunk(self):
        latent = torch.zeros(1, 5, 1, 1, 1)
        with self.assertRaisesRegex(ValueError, "divisible"):
            self.model._shift_chunks(latent, 1)

    def test_valid_mask_excludes_padded_tail(self):
        mask_one = self.model._mcp_valid_frame_mask(6, 1, "cpu")
        mask_two = self.model._mcp_valid_frame_mask(6, 2, "cpu")
        torch.testing.assert_close(
            mask_one, torch.tensor([True, True, True, True, False, False])
        )
        torch.testing.assert_close(
            mask_two, torch.tensor([True, True, False, False, False, False])
        )

    def test_prepare_inputs_aligns_shift_mask_timestep_sigma_and_rope(self):
        # [B=2, levels=3, frames=6, C=H=W=1]. Every level/frame pair has a
        # distinct value, making an accidental shift or level mix visible.
        ode_latent = torch.empty(2, 3, 6, 1, 1, 1)
        for batch in range(2):
            for level in range(3):
                ode_latent[batch, level, :, 0, 0, 0] = (
                    batch * 1000 + level * 100 + torch.arange(6)
                )

        sampled_levels = [torch.tensor([0, 2]), torch.tensor([1, 1])]
        with patch("torch.randint", side_effect=sampled_levels):
            prepared = self.model._prepare_mcp_inputs(ode_latent)

        expected_depth1 = torch.stack(
            [ode_latent[0, 0], ode_latent[1, 2]], dim=0
        )
        expected_depth2 = torch.stack(
            [ode_latent[0, 1], ode_latent[1, 1]], dim=0
        )
        torch.testing.assert_close(
            prepared["noisy"][0], self.model._shift_chunks(expected_depth1, 1)
        )
        torch.testing.assert_close(
            prepared["noisy"][1], self.model._shift_chunks(expected_depth2, 2)
        )
        torch.testing.assert_close(
            prepared["clean"][0], self.model._shift_chunks(ode_latent[:, -1], 1)
        )
        torch.testing.assert_close(
            prepared["clean"][1], self.model._shift_chunks(ode_latent[:, -1], 2)
        )

        torch.testing.assert_close(
            prepared["timesteps"][0],
            torch.tensor([[1000.0] * 6, [0.0] * 6]),
        )
        torch.testing.assert_close(
            prepared["timesteps"][1], torch.tensor([[500.0] * 6] * 2)
        )
        torch.testing.assert_close(
            prepared["sigmas"][0].flatten(), torch.tensor([1.0, 0.0])
        )
        torch.testing.assert_close(
            prepared["sigmas"][1].flatten(), torch.tensor([0.5, 0.5])
        )
        self.assertEqual(prepared["start_frames"], [2, 4])
        self.assertEqual(
            [mask.sum().item() for mask in prepared["masks"]], [4, 2]
        )

    def test_depth_weighted_x0_loss(self):
        inputs = {
            "noisy": [torch.zeros(1, 6, 1, 1, 1)] * 2,
            "clean": [
                torch.ones(1, 6, 1, 1, 1),
                torch.full((1, 6, 1, 1, 1), 2.0),
            ],
            "sigmas": [torch.ones(1, 1, 1, 1, 1)] * 2,
            "masks": [
                torch.tensor([True, True, True, True, False, False]),
                torch.tensor([True, True, False, False, False, False]),
            ],
        }
        flow_predictions = [
            torch.zeros(1, 6, 1, 1, 1),
            torch.zeros(1, 6, 1, 1, 1),
        ]

        loss, log = self.model._compute_mcp_loss(inputs, flow_predictions)

        self.assertAlmostEqual(loss.item(), 0.5 * 1.0 + 0.2 * 4.0)
        self.assertAlmostEqual(log["mcp_loss_depth1"].item(), 1.0)
        self.assertAlmostEqual(log["mcp_loss_depth2"].item(), 4.0)


if __name__ == "__main__":
    unittest.main()
