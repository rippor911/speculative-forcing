import unittest
from unittest.mock import patch

import torch

from wan.modules.mcp import MCPStack, mcp_unpatchify


class MCPModuleTest(unittest.TestCase):
    def test_unpatchify_restores_bcfhw_layout(self):
        patches = torch.arange(16.0).reshape(1, 2, 4, 2)
        grid_sizes = torch.tensor([[2, 2, 2]])

        output = mcp_unpatchify(
            patches, grid_sizes, patch_size=(1, 1, 1), out_dim=2
        )
        expected = patches.reshape(1, 2, 2, 2, 2).permute(0, 4, 1, 2, 3)

        self.assertEqual(output.shape, (1, 2, 2, 2, 2))
        torch.testing.assert_close(output, expected)

    def test_stack_shapes_offsets_and_gradient_flow(self):
        stack = MCPStack(
            dim=4,
            ffn_dim=8,
            num_heads=2,
            out_dim=2,
            patch_size=(1, 1, 1),
            freq_dim=4,
            num_modules=2,
            num_layers=1,
            tap_layers=(0, 1),
        )
        # MCP heads are deliberately zero-initialized. Give them non-zero weights
        # here so this structural test can verify gradient flow through the entire
        # chain rather than only the output projection's first optimization step.
        for module in stack.mcp_modules:
            torch.nn.init.normal_(module.head.head.weight, std=0.1)

        features = [
            torch.randn(1, 8, 4, requires_grad=True),
            torch.randn(1, 8, 4, requires_grad=True),
        ]
        future_embeds = [
            torch.randn(1, 8, 4, requires_grad=True),
            torch.randn(1, 8, 4, requires_grad=True),
        ]
        grid_sizes = [torch.tensor([[2, 2, 2]])] * 2
        timesteps = [torch.full((1, 2), 1000.0)] * 2
        rope_starts = []

        def fake_rope(x, _grid_sizes, _freqs, start_frame):
            rope_starts.append(start_frame)
            return x

        def cpu_attention(q, k, v):
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            weights = torch.softmax(q @ k.transpose(-1, -2) / q.shape[-1] ** 0.5, dim=-1)
            return (weights @ v).transpose(1, 2).contiguous()

        with patch("wan.modules.mcp.causal_rope_apply", side_effect=fake_rope), patch(
            "wan.modules.mcp.attention", side_effect=cpu_attention
        ):
            outputs = stack(
                features=features,
                future_embeds=future_embeds,
                future_grid_sizes=grid_sizes,
                future_start_frames=[2, 4],
                timesteps=timesteps,
                freqs=None,
            )

        self.assertEqual(len(outputs), 2)
        self.assertEqual(outputs[0].shape, (1, 2, 2, 2, 2))
        self.assertEqual(outputs[1].shape, (1, 2, 2, 2, 2))
        # q and k are roped independently for each module.
        self.assertEqual(rope_starts, [2, 2, 4, 4])

        sum(output.square().mean() for output in outputs).backward()
        for tensor in features + future_embeds:
            self.assertIsNotNone(tensor.grad)
            self.assertTrue(torch.isfinite(tensor.grad).all())
            self.assertGreater(tensor.grad.abs().sum().item(), 0.0)

        for name, parameter in stack.named_parameters():
            self.assertIsNotNone(parameter.grad, msg=f"missing gradient for {name}")
            self.assertTrue(
                torch.isfinite(parameter.grad).all(), msg=f"non-finite gradient for {name}"
            )

    def test_none_future_chunk_stops_the_chain(self):
        stack = MCPStack(
            dim=4,
            ffn_dim=8,
            num_heads=2,
            out_dim=2,
            patch_size=(1, 1, 1),
            freq_dim=4,
            num_modules=2,
            num_layers=1,
            tap_layers=(0,),
        )
        with patch("wan.modules.mcp.causal_rope_apply", side_effect=lambda x, *args, **kwargs: x), patch(
            "wan.modules.mcp.attention", side_effect=lambda _q, _k, v: v
        ):
            outputs = stack(
                features=[torch.randn(1, 8, 4)],
                future_embeds=[torch.randn(1, 8, 4), None],
                future_grid_sizes=[torch.tensor([[2, 2, 2]]), None],
                future_start_frames=[2, 4],
                timesteps=[torch.full((1, 2), 1000.0)] * 2,
                freqs=None,
            )
        self.assertEqual(len(outputs), 1)


if __name__ == "__main__":
    unittest.main()
