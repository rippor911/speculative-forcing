from __future__ import annotations

from typing import Protocol

from speculative.types import (
    CommitRequest,
    ControlRequest,
    DraftCandidate,
    FallbackResult,
    ProposalBatch,
)


class SelfForcingMCPRuntimeProtocol(Protocol):
    """Milestone 2B1 runtime contract for thin MCP adapter wrappers.

    This protocol describes the shared state owner that future runtime work will
    implement. It is not a runtime implementation and does not own, snapshot, or
    mutate generation state by itself.
    """

    def propose_window(self, request: ControlRequest) -> ProposalBatch:
        ...

    def generate_target_fallback(self, candidate: DraftCandidate) -> FallbackResult:
        ...

    def begin_window(self) -> None:
        ...

    def commit_block(self, request: CommitRequest) -> None:
        ...

    def complete_window(self) -> None:
        ...

    def rollback_window(self) -> None:
        ...


class SelfForcingMCPProposalSource:
    """ProposalSource wrapper that delegates to a shared MCP runtime."""

    __slots__ = ("_runtime",)

    def __init__(self, runtime: SelfForcingMCPRuntimeProtocol) -> None:
        self._runtime = runtime

    @property
    def runtime(self) -> SelfForcingMCPRuntimeProtocol:
        return self._runtime

    def propose(self, request: ControlRequest) -> ProposalBatch:
        return self._runtime.propose_window(request)


class SelfForcingMCPFallbackGenerator:
    """FallbackGenerator wrapper that delegates to a shared MCP runtime."""

    __slots__ = ("_runtime",)

    def __init__(self, runtime: SelfForcingMCPRuntimeProtocol) -> None:
        self._runtime = runtime

    @property
    def runtime(self) -> SelfForcingMCPRuntimeProtocol:
        return self._runtime

    def generate(self, rejected: DraftCandidate) -> FallbackResult:
        return self._runtime.generate_target_fallback(rejected)


class SelfForcingMCPCommitter:
    """Committer wrapper that delegates controller transactions to a shared MCP runtime."""

    __slots__ = ("_runtime",)

    def __init__(self, runtime: SelfForcingMCPRuntimeProtocol) -> None:
        self._runtime = runtime

    @property
    def runtime(self) -> SelfForcingMCPRuntimeProtocol:
        return self._runtime

    def begin(self) -> None:
        self._runtime.begin_window()

    def commit(self, request: CommitRequest) -> None:
        self._runtime.commit_block(request)

    def complete(self) -> None:
        self._runtime.complete_window()

    def rollback(self) -> None:
        self._runtime.rollback_window()


__all__ = [
    "SelfForcingMCPCommitter",
    "SelfForcingMCPFallbackGenerator",
    "SelfForcingMCPProposalSource",
    "SelfForcingMCPRuntimeProtocol",
]
