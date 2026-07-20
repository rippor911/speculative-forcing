"""Checkpoint helpers shared by the ODE and distillation trainers."""

from collections.abc import Mapping


_STATE_DICT_WRAPPER_PREFIXES = (
    "_fsdp_wrapped_module.",
    "_checkpoint_wrapped_module.",
    "_orig_mod.",
)


def normalize_state_key(name: str) -> str:
    """Remove transparent wrapper prefixes from a state-dict key."""
    for prefix in _STATE_DICT_WRAPPER_PREFIXES:
        name = name.replace(prefix, "")
    return name


def is_mcp_state_key(name: str) -> bool:
    """Return whether a (possibly wrapped) state-dict key belongs to MCP."""
    return normalize_state_key(name).startswith("mcp.")


def extract_generator_state_dict(checkpoint: Mapping) -> Mapping:
    """Accept the checkpoint layouts used across this repository."""
    if not isinstance(checkpoint, Mapping):
        raise TypeError(
            f"generator checkpoint must be a mapping, got {type(checkpoint).__name__}"
        )
    if "generator" in checkpoint:
        return checkpoint["generator"]
    if "model" in checkpoint:
        return checkpoint["model"]
    return checkpoint


def load_state_dict_allowing_mcp_mismatch(module, state_dict: Mapping):
    """Load a generator while allowing only MCP keys to differ.

    This supports both directions needed by the training pipeline:

    * a backbone-only checkpoint initializes a generator with newly attached MCP
      modules (missing ``mcp.*`` keys);
    * a checkpoint with more MCP depths initializes a configuration that keeps
      fewer or no heads (unexpected ``mcp.*`` keys).

    Shape mismatches still raise in ``load_state_dict``. Any missing or unexpected
    non-MCP key is also rejected, so ``strict=False`` cannot hide a broken
    backbone checkpoint.
    """
    missing, unexpected = module.load_state_dict(state_dict, strict=False)
    non_mcp_missing = [key for key in missing if not is_mcp_state_key(key)]
    non_mcp_unexpected = [key for key in unexpected if not is_mcp_state_key(key)]
    if non_mcp_missing or non_mcp_unexpected:
        raise RuntimeError(
            "Unexpected generator checkpoint mismatch.\n"
            f"  missing (non-MCP): {non_mcp_missing}\n"
            f"  unexpected (non-MCP): {non_mcp_unexpected}"
        )
    return missing, unexpected
