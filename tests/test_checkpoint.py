import unittest

import torch
from torch import nn

from utils.checkpoint import (
    extract_generator_state_dict,
    is_mcp_state_key,
    load_state_dict_allowing_mcp_mismatch,
    normalize_state_key,
)


class TinyGenerator(nn.Module):
    def __init__(self, with_mcp: bool):
        super().__init__()
        self.backbone = nn.Linear(2, 2, bias=False)
        if with_mcp:
            self.mcp = nn.Linear(2, 2, bias=False)


class CheckpointCompatibilityTest(unittest.TestCase):
    def test_extracts_supported_checkpoint_layouts(self):
        state = {"backbone.weight": torch.ones(2, 2)}
        self.assertIs(extract_generator_state_dict({"generator": state}), state)
        self.assertIs(extract_generator_state_dict({"model": state}), state)
        self.assertIs(extract_generator_state_dict(state), state)

    def test_normalizes_wrapped_mcp_keys(self):
        key = "_fsdp_wrapped_module._checkpoint_wrapped_module.mcp.head.weight"
        self.assertEqual(normalize_state_key(key), "mcp.head.weight")
        self.assertTrue(is_mcp_state_key(key))
        self.assertFalse(is_mcp_state_key("_orig_mod.backbone.weight"))

    def test_backbone_only_checkpoint_can_initialize_mcp_model(self):
        source = TinyGenerator(with_mcp=False)
        target = TinyGenerator(with_mcp=True)
        expected = torch.full_like(source.backbone.weight, 3.0)
        source.backbone.weight.data.copy_(expected)

        missing, unexpected = load_state_dict_allowing_mcp_mismatch(
            target, source.state_dict()
        )

        self.assertEqual(missing, ["mcp.weight"])
        self.assertEqual(unexpected, [])
        torch.testing.assert_close(target.backbone.weight, expected)

    def test_mcp_checkpoint_can_initialize_backbone_only_model(self):
        source = TinyGenerator(with_mcp=True)
        target = TinyGenerator(with_mcp=False)

        missing, unexpected = load_state_dict_allowing_mcp_mismatch(
            target, source.state_dict()
        )

        self.assertEqual(missing, [])
        self.assertEqual(unexpected, ["mcp.weight"])

    def test_non_mcp_mismatch_is_rejected(self):
        target = TinyGenerator(with_mcp=True)
        with self.assertRaisesRegex(RuntimeError, "missing .*non-MCP"):
            load_state_dict_allowing_mcp_mismatch(
                target, {"mcp.weight": target.mcp.weight.detach().clone()}
            )

    def test_shape_mismatch_is_rejected(self):
        target = TinyGenerator(with_mcp=True)
        bad_state = target.state_dict()
        bad_state["backbone.weight"] = torch.ones(3, 3)
        with self.assertRaises(RuntimeError):
            load_state_dict_allowing_mcp_mismatch(target, bad_state)


if __name__ == "__main__":
    unittest.main()
