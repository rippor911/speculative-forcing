"""Adapter-side runtime state utilities for speculative decoding.

Milestone 2A exposes only generic snapshot/restore primitives. Real
Self-Forcing, Wan, MCP, and evaluator adapters are intentionally out of scope.
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
]
