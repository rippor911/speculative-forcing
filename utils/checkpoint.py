"""Checkpoint helpers shared by the ODE and distillation trainers."""

from collections.abc import Mapping


_STATE_DICT_WRAPPER_PREFIXES = (
    "_fsdp_wrapped_module.",
    "_checkpoint_wrapped_module.",
    "_orig_mod.",
)

BACKBONE_ONLY_INITIALIZE_MCP = "BACKBONE_ONLY_INITIALIZE_MCP"
MCP_COMPLETE_STRICT_RESTORE = "MCP_COMPLETE_STRICT_RESTORE"


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
    if "generator_ema" in checkpoint:
        return checkpoint["generator_ema"]
    if "model" in checkpoint:
        return checkpoint["model"]
    return checkpoint


def load_state_dict_allowing_mcp_mismatch(module, state_dict: Mapping):
    """Load a generator while only relaxing MCP keys for backbone checkpoints."""
    checkpoint_has_mcp = any(is_mcp_state_key(key) for key in state_dict.keys())
    missing, unexpected = module.load_state_dict(state_dict, strict=False)

    if checkpoint_has_mcp:
        load_state_dict_allowing_mcp_mismatch.last_load_mode = (
            MCP_COMPLETE_STRICT_RESTORE
        )
        if missing or unexpected:
            raise RuntimeError(
                "MCP generator checkpoint must match exactly.\n"
                f"  missing: {missing}\n"
                f"  unexpected: {unexpected}"
            )
        return missing, unexpected

    load_state_dict_allowing_mcp_mismatch.last_load_mode = (
        BACKBONE_ONLY_INITIALIZE_MCP
    )
    non_mcp_missing = [key for key in missing if not is_mcp_state_key(key)]
    if non_mcp_missing or unexpected:
        raise RuntimeError(
            "Backbone-only generator checkpoint mismatch.\n"
            f"  missing (non-MCP): {non_mcp_missing}\n"
            f"  unexpected: {unexpected}"
        )
    return missing, unexpected


load_state_dict_allowing_mcp_mismatch.last_load_mode = None
