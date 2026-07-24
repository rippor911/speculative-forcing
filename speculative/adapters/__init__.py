"""Adapter-side utilities for speculative decoding.

Milestone 2A exposes generic snapshot/restore primitives. Milestone 2B1 adds
thin Self-Forcing MCP protocol wrappers while the real runtime remains future
work.
"""

from speculative.adapters.runtime_state import (
    ObjectStateSnapshot,
    ObjectStateSpec,
    RuntimeStateRestoreError,
    RuntimeStateRollbackError,
    RuntimeStateSnapshot,
    RuntimeStateTransaction,
    RuntimeStateTransactionManager,
    TensorRegionSnapshot,
    TensorRegionSpec,
    TensorValueSnapshot,
    TorchRNGSnapshot,
)
from speculative.adapters.self_forcing_mcp import (
    SelfForcingMCPCommitter,
    SelfForcingMCPFallbackGenerator,
    SelfForcingMCPProposalSource,
    SelfForcingMCPRuntimeProtocol,
)

__all__ = [
    "ObjectStateSnapshot",
    "ObjectStateSpec",
    "RuntimeStateRestoreError",
    "RuntimeStateRollbackError",
    "RuntimeStateSnapshot",
    "RuntimeStateTransaction",
    "RuntimeStateTransactionManager",
    "TensorRegionSnapshot",
    "TensorRegionSpec",
    "TensorValueSnapshot",
    "TorchRNGSnapshot",
    "SelfForcingMCPCommitter",
    "SelfForcingMCPFallbackGenerator",
    "SelfForcingMCPProposalSource",
    "SelfForcingMCPRuntimeProtocol",
]
