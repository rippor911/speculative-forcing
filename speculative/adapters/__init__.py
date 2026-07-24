"""Adapter-side utilities for speculative decoding.

Milestone 2A exposes generic snapshot/restore primitives. Milestone 2B1 adds
thin Self-Forcing MCP protocol wrappers. Milestone 2B2A adds runtime
orchestration with an injectable backend; the real Wan backend remains future
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
from speculative.adapters.self_forcing_runtime import (
    RuntimeBackendProtocol,
    RuntimeStateSpecBundle,
    RuntimeWindowDescriptor,
    SelfForcingMCPRolloutPlan,
    SelfForcingMCPRuntime,
    SelfForcingMCPRuntimeConfig,
    SelfForcingMCPRuntimeContext,
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
    "RuntimeBackendProtocol",
    "RuntimeStateSpecBundle",
    "RuntimeWindowDescriptor",
    "SelfForcingMCPRolloutPlan",
    "SelfForcingMCPRuntime",
    "SelfForcingMCPRuntimeConfig",
    "SelfForcingMCPRuntimeContext",
    "TensorRegionSnapshot",
    "TensorRegionSpec",
    "TensorValueSnapshot",
    "TorchRNGSnapshot",
    "SelfForcingMCPCommitter",
    "SelfForcingMCPFallbackGenerator",
    "SelfForcingMCPProposalSource",
    "SelfForcingMCPRuntimeProtocol",
]
