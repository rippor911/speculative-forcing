# MCP validation tests

These tests are intentionally independent of Wan checkpoints and video data. Run
them in the project's normal Python environment:

```bash
python -m unittest discover -s tests -v
```

They cover:

- checkpoint loading with and without `mcp.*` parameters;
- ODE future-chunk shifting, masks, timestep/sigma lookup, RoPE offsets, and
  depth-weighted loss;
- MCP module output shapes and end-to-end gradient flow on tiny tensors;
- accelerated-rollout future-chunk layout and configuration guards.

The module test replaces FlashAttention and RoPE with CPU test doubles, so it
checks MCP wiring without requiring a GPU. It does not replace a later A100 smoke
test of the real Wan attention path.
