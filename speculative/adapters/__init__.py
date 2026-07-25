"""Adapter-side utilities for speculative decoding.

Milestone 2A exposes generic snapshot/restore primitives. Milestone 2B1 adds
thin Self-Forcing MCP protocol wrappers. Milestone 2B2A adds runtime
orchestration with an injectable backend. Milestone F2A adds the minimal Wan
MCP backend binding.
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
from speculative.adapters.self_forcing_wan_backend import SelfForcingWanMCPBackend

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
    "SelfForcingWanMCPBackend",
]
