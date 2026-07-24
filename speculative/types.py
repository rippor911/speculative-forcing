from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Mapping, Optional, Sequence


DecisionAction = Literal["accept", "reject"]
CommitSource = Literal["anchor", "draft", "fallback"]

if TYPE_CHECKING:
    from speculative.trace import TraceEvent


class SpeculativeControlError(RuntimeError):
    """Raised when speculative control invariants are violated."""


class SpeculativeRollbackError(SpeculativeControlError):
    """Raised when the committer rollback fails after an earlier error."""

    def __init__(self, original_exception: Exception, rollback_exception: Exception) -> None:
        self.original_exception = original_exception
        self.rollback_exception = rollback_exception
        super().__init__(
            "Speculative rollback failed after "
            f"{type(original_exception).__name__}: {original_exception}; "
            "rollback raised "
            f"{type(rollback_exception).__name__}: {rollback_exception}"
        )


def _is_strict_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _require_optional_strict_int(name: str, value: Optional[int]) -> None:
    if value is not None and not _is_strict_int(value):
        raise ValueError(f"{name} must be an integer or None, got {type(value).__name__}.")


def _require_strict_int(name: str, value: int) -> None:
    if not _is_strict_int(value):
        raise ValueError(f"{name} must be an integer, got {type(value).__name__}.")


def _require_non_negative(name: str, value: Optional[int]) -> None:
    _require_optional_strict_int(name, value)
    if value is not None and value < 0:
        raise ValueError(f"{name} must be >= 0, got {value}.")


def _require_positive(name: str, value: Optional[int]) -> None:
    _require_optional_strict_int(name, value)
    if value is not None and value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}.")


def _require_required_non_negative(name: str, value: int) -> None:
    _require_strict_int(name, value)
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value}.")


def _require_required_positive(name: str, value: int) -> None:
    _require_strict_int(name, value)
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}.")


@dataclass(frozen=True)
class BlockRef:
    """Stable identity for one decoded latent block."""

    index: int
    start_frame: Optional[int] = None
    num_frames: Optional[int] = None

    def __post_init__(self) -> None:
        _require_required_non_negative("index", self.index)
        _require_non_negative("start_frame", self.start_frame)
        _require_positive("num_frames", self.num_frames)
        if (self.start_frame is None) != (self.num_frames is None):
            raise ValueError("start_frame and num_frames must be provided together.")


@dataclass(frozen=True)
class ControlRequest:
    """One speculative control window request."""

    anchor_block: BlockRef
    max_depth: int
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_required_non_negative("max_depth", self.max_depth)


@dataclass(frozen=True)
class DraftCandidate:
    """Candidate block predicted speculatively from preserved source noise."""

    block: BlockRef
    depth: int
    latent: Any
    source_noise: Any
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_required_positive("depth", self.depth)
        if self.source_noise is None:
            raise ValueError("DraftCandidate.source_noise is required.")


@dataclass(frozen=True)
class Evaluation:
    """Evaluator output consumed by a policy."""

    candidate: DraftCandidate
    value: Any = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Decision:
    """A policy's complete output."""

    action: DecisionAction
    reason: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.action not in ("accept", "reject"):
            raise ValueError(f"Unsupported decision action: {self.action!r}.")
        if not isinstance(self.reason, str):
            raise ValueError("Decision.reason must be a string.")

    @property
    def accepted(self) -> bool:
        return self.action == "accept"

    @classmethod
    def accept(cls, reason: str = "", metadata: Optional[Mapping[str, Any]] = None) -> "Decision":
        return cls("accept", reason=reason, metadata={} if metadata is None else metadata)

    @classmethod
    def reject(cls, reason: str = "", metadata: Optional[Mapping[str, Any]] = None) -> "Decision":
        return cls("reject", reason=reason, metadata={} if metadata is None else metadata)


@dataclass(frozen=True)
class FallbackResult:
    """Non-speculative replacement for a rejected draft block."""

    block: BlockRef
    latent: Any
    source_noise: Any
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.source_noise is None:
            raise ValueError("FallbackResult.source_noise is required.")


@dataclass(frozen=True)
class CommitRequest:
    """A finalized block ready to mutate the external commit target."""

    block: BlockRef
    latent: Any
    source: CommitSource
    depth: Optional[int] = None
    source_noise: Any = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.source not in ("anchor", "draft", "fallback"):
            raise ValueError(f"Unsupported commit source: {self.source!r}.")
        if self.source == "anchor":
            if self.depth is not None:
                raise ValueError("anchor commits must have depth=None.")
            if self.source_noise is not None:
                raise ValueError("anchor commits must have source_noise=None.")
            return
        if not _is_strict_int(self.depth) or self.depth <= 0:
            raise ValueError(f"{self.source} commits must have a positive integer depth.")
        if self.source_noise is None:
            raise ValueError(f"{self.source} commits must preserve source_noise.")


@dataclass(frozen=True)
class ProposalBatch:
    """Anchor plus ordered speculative drafts for one window."""

    anchor: CommitRequest
    drafts: Sequence[DraftCandidate] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.anchor.source != "anchor":
            raise ValueError("ProposalBatch.anchor must have source='anchor'.")
        object.__setattr__(self, "drafts", tuple(self.drafts))
        validate_contiguous_block_range(self.anchor.block, self.drafts)


@dataclass(frozen=True)
class ControlResult:
    """Result of one speculative control window."""

    accepted_depth: int
    rejected_depth: Optional[int]
    committed: Sequence[CommitRequest]
    invalidated: Sequence[DraftCandidate]
    trace: Sequence["TraceEvent"]

    def __post_init__(self) -> None:
        _require_required_non_negative("accepted_depth", self.accepted_depth)
        _require_positive("rejected_depth", self.rejected_depth)
        object.__setattr__(self, "committed", tuple(self.committed))
        object.__setattr__(self, "invalidated", tuple(self.invalidated))
        object.__setattr__(self, "trace", tuple(self.trace))


def validate_contiguous_block_range(
    anchor: BlockRef,
    drafts: Sequence[DraftCandidate],
) -> None:
    """Validate draft depth/index and optional frame ranges."""

    for expected_depth, candidate in enumerate(drafts, start=1):
        if candidate.depth != expected_depth:
            raise ValueError(
                f"Draft depths must be contiguous from 1; expected "
                f"{expected_depth}, got {candidate.depth}."
            )
        expected_block = anchor.index + expected_depth
        if candidate.block.index != expected_block:
            raise ValueError(
                f"Draft depth {candidate.depth} must target block "
                f"{expected_block}, got {candidate.block.index}."
            )

    blocks = (anchor,) + tuple(candidate.block for candidate in drafts)
    has_frame_range = any(
        block.start_frame is not None or block.num_frames is not None
        for block in blocks
    )
    if not has_frame_range:
        return

    for block in blocks:
        if block.start_frame is None or block.num_frames is None:
            raise ValueError(
                "start_frame and num_frames must be provided for every block "
                "when any block has a frame range."
            )

    for previous, current in zip(blocks, blocks[1:]):
        expected_start = previous.start_frame + previous.num_frames
        if current.start_frame != expected_start:
            raise ValueError(
                f"Block frame ranges must be continuous; expected block "
                f"{current.index} to start at {expected_start}, got "
                f"{current.start_frame}."
            )
