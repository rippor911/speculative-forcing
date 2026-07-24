from __future__ import annotations

from contextlib import contextmanager
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Literal, Optional, Protocol, Sequence

import torch

from speculative.adapters.runtime_state import (
    ObjectStateSpec,
    RuntimeStateTransaction,
    RuntimeStateTransactionManager,
    TensorRegionSpec,
)
from speculative.types import (
    BlockRef,
    CommitRequest,
    ControlRequest,
    DraftCandidate,
    FallbackResult,
    ProposalBatch,
    validate_contiguous_block_range,
)


RuntimeOperation = Literal["prepare", "proposal", "fallback"]


def _is_strict_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _require_strict_int(name: str, value: int) -> None:
    if not _is_strict_int(value):
        raise ValueError(f"{name} must be an integer, got {type(value).__name__}.")


def _require_positive_int(name: str, value: int) -> None:
    _require_strict_int(name, value)
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}.")


def _require_non_negative_int(name: str, value: int) -> None:
    _require_strict_int(name, value)
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value}.")


@dataclass(frozen=True)
class RuntimeStateSpecBundle:
    """Explicit 2A state specs for one runtime operation."""

    tensor_regions: Sequence[TensorRegionSpec] = field(default_factory=tuple)
    tensor_values: Sequence[torch.Tensor] = field(default_factory=tuple)
    object_states: Sequence[ObjectStateSpec[Any]] = field(default_factory=tuple)
    capture_rng: bool = False
    capture_cuda_rng: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.capture_rng, bool):
            raise ValueError("capture_rng must be a bool.")
        if not isinstance(self.capture_cuda_rng, bool):
            raise ValueError("capture_cuda_rng must be a bool.")
        object.__setattr__(self, "tensor_regions", tuple(self.tensor_regions))
        object.__setattr__(self, "tensor_values", tuple(self.tensor_values))
        object.__setattr__(self, "object_states", tuple(self.object_states))


@dataclass(frozen=True)
class SelfForcingMCPRuntimeConfig:
    """Narrow, validated configuration for 2B2A runtime orchestration."""

    anchor_denoising_steps: Sequence[int]
    frame_seq_length: int
    num_frame_per_block: int
    all_num_frames: int
    mcp_depth: int
    attention_mode: str
    validated_attention_mode: str

    def __post_init__(self) -> None:
        steps = tuple(self.anchor_denoising_steps)
        for index, step in enumerate(steps):
            _require_strict_int(f"anchor_denoising_steps[{index}]", step)
        if steps != (1000,):
            raise ValueError("anchor_denoising_steps must be exactly (1000,).")
        object.__setattr__(self, "anchor_denoising_steps", steps)

        _require_positive_int("frame_seq_length", self.frame_seq_length)
        _require_positive_int("num_frame_per_block", self.num_frame_per_block)
        _require_positive_int("all_num_frames", self.all_num_frames)
        _require_non_negative_int("mcp_depth", self.mcp_depth)
        if not isinstance(self.attention_mode, str) or not self.attention_mode:
            raise ValueError("attention_mode must be a non-empty string.")
        if not isinstance(self.validated_attention_mode, str) or not self.validated_attention_mode:
            raise ValueError("validated_attention_mode must be a non-empty string.")
        if self.attention_mode != self.validated_attention_mode:
            raise ValueError(
                f"Unsupported attention/cache mode {self.attention_mode!r}; "
                f"expected validated mode {self.validated_attention_mode!r}."
            )


@dataclass(frozen=True)
class SelfForcingMCPRolloutPlan:
    """Immutable block layout for one source-noise rollout."""

    blocks: tuple[BlockRef, ...]
    block_starts: tuple[int, ...]
    anchor_block_indices: tuple[int, ...]
    period: int

    def block_for_index(self, index: int) -> BlockRef:
        if index < 0 or index >= len(self.blocks):
            raise ValueError(f"Block index {index} is outside the rollout plan.")
        return self.blocks[index]

    def contains_block(self, block: BlockRef) -> bool:
        return 0 <= block.index < len(self.blocks) and self.blocks[block.index] == block

    def is_anchor(self, block: BlockRef) -> bool:
        return self.contains_block(block) and block.index in self.anchor_block_indices


@dataclass(frozen=True)
class RuntimeWindowDescriptor:
    """Read-only descriptor consumed by `begin_window()` after proposal."""

    anchor_block: BlockRef
    allowed_blocks: tuple[BlockRef, ...]


@dataclass(frozen=True)
class SelfForcingMCPRuntimeContext:
    """Borrowed runtime state passed to the backend without transferring ownership."""

    config: SelfForcingMCPRuntimeConfig
    rollout_plan: SelfForcingMCPRolloutPlan
    source_noise: Any
    output: Any
    kv_cache: Any
    cross_attention_cache: Any

    def current_start_for(self, block: BlockRef) -> int:
        if block.start_frame is None:
            raise ValueError("BlockRef.start_frame is required to derive current_start.")
        return block.start_frame * self.config.frame_seq_length


class RuntimeBackendProtocol(Protocol):
    """Backend interface for model/cache operations under runtime-owned transactions."""

    def prepare_cross_attention(self, runtime_context: SelfForcingMCPRuntimeContext) -> None:
        ...

    def propose_anchor_and_drafts(
        self,
        request: ControlRequest,
        runtime_context: SelfForcingMCPRuntimeContext,
    ) -> ProposalBatch:
        ...

    def generate_target_fallback(
        self,
        candidate: DraftCandidate,
        runtime_context: SelfForcingMCPRuntimeContext,
    ) -> FallbackResult:
        ...

    def commit_context_block(
        self,
        request: CommitRequest,
        runtime_context: SelfForcingMCPRuntimeContext,
    ) -> None:
        ...

    def temporary_state_specs(
        self,
        operation: RuntimeOperation,
        target: object,
        runtime_context: SelfForcingMCPRuntimeContext,
    ) -> RuntimeStateSpecBundle:
        ...

    def prepare_persistent_state_specs(
        self,
        runtime_context: SelfForcingMCPRuntimeContext,
    ) -> RuntimeStateSpecBundle:
        ...

    def window_state_specs(
        self,
        window: RuntimeWindowDescriptor,
        runtime_context: SelfForcingMCPRuntimeContext,
    ) -> RuntimeStateSpecBundle:
        ...


@dataclass
class _ActiveWindow:
    descriptor: RuntimeWindowDescriptor
    transaction: RuntimeStateTransaction
    manager: RuntimeStateTransactionManager
    transaction_specs: RuntimeStateSpecBundle
    committed_offsets: list[int] = field(default_factory=list)


class SelfForcingMCPRuntime:
    """State owner and transaction orchestrator for Self-Forcing MCP integration."""

    def __init__(
        self,
        *,
        config: SelfForcingMCPRuntimeConfig,
        backend: RuntimeBackendProtocol,
        source_noise: Any,
        output: Any,
        kv_cache: Any,
        cross_attention_cache: Any,
    ) -> None:
        _validate_output_tensor(output, config)
        self._config = config
        self._backend = backend
        self._source_noise = source_noise
        self._output = output
        self._kv_cache = kv_cache
        self._cross_attention_cache = cross_attention_cache
        self._rollout_plan = _build_rollout_plan(config)
        self._context = SelfForcingMCPRuntimeContext(
            config=config,
            rollout_plan=self._rollout_plan,
            source_noise=source_noise,
            output=output,
            kv_cache=kv_cache,
            cross_attention_cache=cross_attention_cache,
        )
        self._is_prepared = False
        self._pending_window: Optional[RuntimeWindowDescriptor] = None
        self._active_window: Optional[_ActiveWindow] = None
        self._committed_blocks: list[BlockRef] = []

    @property
    def is_prepared(self) -> bool:
        return self._is_prepared

    @property
    def has_active_window(self) -> bool:
        return self._active_window is not None

    @property
    def rollout_plan(self) -> SelfForcingMCPRolloutPlan:
        return self._rollout_plan

    @property
    def committed_blocks(self) -> tuple[BlockRef, ...]:
        return tuple(self._committed_blocks)

    @property
    def output(self) -> Any:
        """Borrowed output buffer; callers must not mutate it directly."""

        return self._output

    def current_start_for(self, block: BlockRef) -> int:
        return self._context.current_start_for(block)

    def prepare(self) -> None:
        if self._is_prepared:
            raise RuntimeError("SelfForcingMCPRuntime.prepare() has already succeeded.")
        persistent_specs = self._backend.prepare_persistent_state_specs(self._context)
        persistent_manager = self._manager_from_specs(
            self._with_runtime_output_specs(persistent_specs, ()),
            include_runtime_commit_state=False,
        )
        persistent_transaction = persistent_manager.begin()
        try:
            temporary_specs = self._backend.temporary_state_specs("prepare", None, self._context)
            with self._temporary_transaction(
                self._with_runtime_output_specs(temporary_specs, ())
            ):
                self._backend.prepare_cross_attention(self._context)
            persistent_transaction.complete()
        except Exception:
            try:
                persistent_transaction.rollback()
            finally:
                self._is_prepared = False
            raise
        self._is_prepared = True

    def propose_window(self, request: ControlRequest) -> ProposalBatch:
        self._require_prepared()
        if self._active_window is not None:
            raise RuntimeError("Cannot propose while a window transaction is active.")
        if self._pending_window is not None:
            raise RuntimeError("Cannot replace a pending proposal before begin_window().")
        self._validate_proposal_request(request)
        specs = self._backend.temporary_state_specs("proposal", request, self._context)
        output_blocks = self._proposal_output_blocks(request)
        transaction_specs = self._with_runtime_output_specs(specs, output_blocks)
        with self._temporary_transaction(transaction_specs):
            batch = self._backend.propose_anchor_and_drafts(request, self._context)
            self._validate_proposal_batch(request, batch)
            self._reject_latent_aliases(
                (batch.anchor.latent,) + tuple(draft.latent for draft in batch.drafts),
                transaction_specs,
            )
            descriptor = self._build_window_descriptor_from_batch(batch)
        self._pending_window = descriptor
        return batch

    def generate_target_fallback(self, candidate: DraftCandidate) -> FallbackResult:
        self._require_prepared()
        self._require_plan_block(candidate.block)
        specs = self._backend.temporary_state_specs("fallback", candidate, self._context)
        transaction_specs = self._with_runtime_output_specs(specs, (candidate.block,))
        with self._temporary_transaction(transaction_specs):
            result = self._backend.generate_target_fallback(candidate, self._context)
            self._validate_fallback_result(candidate, result)
            self._reject_latent_aliases((result.latent,), transaction_specs)
            return result

    def begin_window(self) -> None:
        self._require_prepared()
        if self._active_window is not None:
            raise RuntimeError("Runtime window transaction is already active.")
        if self._pending_window is None:
            raise RuntimeError("begin_window() requires a pending proposal.")

        descriptor = self._pending_window
        specs = self._backend.window_state_specs(descriptor, self._context)
        transaction_specs = self._with_runtime_output_specs(specs, descriptor.allowed_blocks)
        manager = self._manager_from_specs(transaction_specs, include_runtime_commit_state=True)
        transaction = manager.begin()
        self._active_window = _ActiveWindow(
            descriptor=descriptor,
            transaction=transaction,
            manager=manager,
            transaction_specs=transaction_specs,
        )
        self._pending_window = None

    def commit_block(self, request: CommitRequest) -> None:
        self._require_prepared()
        active = self._require_active_window()
        expected_offset = len(active.committed_offsets)
        if expected_offset >= len(active.descriptor.allowed_blocks):
            raise RuntimeError("Active window has no remaining blocks to commit.")
        expected_block = active.descriptor.allowed_blocks[expected_offset]
        if request.block != expected_block:
            raise RuntimeError(
                f"Out-of-order runtime commit: expected block {expected_block.index}, "
                f"got {request.block.index}."
            )
        global_expected_index = len(self._committed_blocks)
        if request.block.index != global_expected_index:
            raise RuntimeError(
                f"Out-of-order runtime commit: expected global block "
                f"{global_expected_index}, got {request.block.index}."
            )

        self._validate_latent_for_block(request.block, request.latent, "commit")
        self._reject_latent_aliases((request.latent,), active.transaction_specs)
        self._backend.commit_context_block(request, self._context)
        self._write_output(request)
        self._committed_blocks.append(request.block)
        active.committed_offsets.append(expected_offset)

    def complete_window(self) -> None:
        active = self._require_active_window()
        try:
            active.transaction.complete()
        finally:
            self._active_window = None
            self._pending_window = None

    def rollback_window(self) -> None:
        active = self._require_active_window()
        try:
            active.transaction.rollback()
        finally:
            self._active_window = None
            self._pending_window = None

    def _require_prepared(self) -> None:
        if not self._is_prepared:
            raise RuntimeError("SelfForcingMCPRuntime.prepare() must succeed before use.")

    def _validate_proposal_request(self, request: ControlRequest) -> None:
        self._require_plan_block(request.anchor_block)
        next_index = len(self._committed_blocks)
        if next_index >= len(self._rollout_plan.blocks):
            raise RuntimeError("Rollout plan has no uncommitted block for a new proposal.")
        expected_anchor = self._rollout_plan.blocks[next_index]
        if request.anchor_block != expected_anchor:
            raise RuntimeError(
                f"Proposal anchor must be next uncommitted block "
                f"{expected_anchor.index}, got {request.anchor_block.index}."
            )
        if request.max_depth > self._config.mcp_depth:
            raise RuntimeError(
                f"request.max_depth={request.max_depth} exceeds runtime mcp_depth="
                f"{self._config.mcp_depth}."
            )

    def _build_window_descriptor_from_batch(
        self,
        batch: ProposalBatch,
    ) -> RuntimeWindowDescriptor:
        allowed = (batch.anchor.block,) + tuple(draft.block for draft in batch.drafts)
        return RuntimeWindowDescriptor(
            anchor_block=batch.anchor.block,
            allowed_blocks=allowed,
        )

    def _proposal_output_blocks(self, request: ControlRequest) -> tuple[BlockRef, ...]:
        remaining = len(self._rollout_plan.blocks) - request.anchor_block.index - 1
        depth_count = min(request.max_depth, self._config.mcp_depth, remaining)
        start = request.anchor_block.index
        return self._rollout_plan.blocks[start:start + depth_count + 1]

    def _validate_proposal_batch(
        self,
        request: ControlRequest,
        batch: ProposalBatch,
    ) -> None:
        if not isinstance(batch, ProposalBatch):
            raise RuntimeError("Backend proposal must return ProposalBatch.")
        if batch.anchor.block != request.anchor_block:
            raise RuntimeError("Proposal anchor block does not match the request anchor.")
        if len(batch.drafts) > request.max_depth:
            raise RuntimeError(
                f"Proposal returned {len(batch.drafts)} drafts for "
                f"request.max_depth={request.max_depth}."
            )
        if len(batch.drafts) > self._config.mcp_depth:
            raise RuntimeError(
                f"Proposal returned {len(batch.drafts)} drafts for runtime "
                f"mcp_depth={self._config.mcp_depth}."
            )
        try:
            validate_contiguous_block_range(batch.anchor.block, batch.drafts)
        except ValueError as exc:
            raise RuntimeError(str(exc)) from exc
        self._require_plan_block(batch.anchor.block)
        self._validate_latent_for_block(batch.anchor.block, batch.anchor.latent, "proposal anchor")
        for draft in batch.drafts:
            self._require_plan_block(draft.block)
            self._validate_latent_for_block(draft.block, draft.latent, "proposal draft")

    def _validate_fallback_result(
        self,
        candidate: DraftCandidate,
        result: object,
    ) -> None:
        if not isinstance(result, FallbackResult):
            raise RuntimeError("Backend fallback must return FallbackResult.")
        if result.block != candidate.block:
            raise RuntimeError("Fallback result block does not match candidate block.")
        if result.source_noise is not candidate.source_noise:
            raise RuntimeError("Fallback result source_noise must preserve candidate identity.")
        self._validate_latent_for_block(result.block, result.latent, "fallback")

    def _validate_latent_for_block(
        self,
        block: BlockRef,
        latent: object,
        role: str,
    ) -> None:
        if not isinstance(latent, torch.Tensor):
            raise RuntimeError(f"{role} latent must be a torch.Tensor.")
        self._require_plan_block(block)
        if block.start_frame is None or block.num_frames is None:
            raise RuntimeError(f"{role} BlockRef must include start_frame and num_frames.")
        start = block.start_frame
        end = start + block.num_frames
        target = self._output[:, start:end]
        if latent.shape != target.shape:
            raise RuntimeError(
                f"{role} latent shape must match output slice shape exactly: "
                f"got {tuple(latent.shape)}, expected {tuple(target.shape)}."
            )
        if latent.dtype != target.dtype:
            raise RuntimeError(
                f"{role} latent dtype must match output dtype: "
                f"got {latent.dtype}, expected {target.dtype}."
            )
        if latent.device != target.device:
            raise RuntimeError(
                f"{role} latent device must match output device: "
                f"got {latent.device}, expected {target.device}."
            )
        if latent.layout != target.layout:
            raise RuntimeError(
                f"{role} latent layout must match output layout: "
                f"got {latent.layout}, expected {target.layout}."
            )

    def _require_plan_block(self, block: BlockRef) -> None:
        if not self._rollout_plan.contains_block(block):
            raise RuntimeError(f"BlockRef {block!r} is outside the runtime rollout plan.")

    def _require_active_window(self) -> _ActiveWindow:
        if self._active_window is None:
            raise RuntimeError("No active runtime window transaction.")
        return self._active_window

    @contextmanager
    def _temporary_transaction(self, specs: RuntimeStateSpecBundle) -> Iterator[None]:
        manager = self._manager_from_specs(specs, include_runtime_commit_state=False)
        with manager.transaction():
            yield
        if manager.is_active:
            raise RuntimeError("Temporary runtime transaction is still active after exit.")

    def _with_runtime_output_specs(
        self,
        specs: RuntimeStateSpecBundle,
        blocks: Sequence[BlockRef],
    ) -> RuntimeStateSpecBundle:
        self._reject_backend_output_specs(specs)
        output_region = self._output_region_for_blocks(blocks)
        tensor_regions = tuple(specs.tensor_regions)
        if output_region is not None:
            tensor_regions = tensor_regions + (output_region,)
        return RuntimeStateSpecBundle(
            tensor_regions=tensor_regions,
            tensor_values=specs.tensor_values,
            object_states=specs.object_states,
            capture_rng=specs.capture_rng,
            capture_cuda_rng=specs.capture_cuda_rng,
        )

    def _reject_backend_output_specs(self, specs: RuntimeStateSpecBundle) -> None:
        for tensor in specs.tensor_values:
            if isinstance(tensor, torch.Tensor) and _shares_storage(tensor, self._output):
                raise RuntimeError("Backend state specs must not declare the runtime output tensor.")
        for region in specs.tensor_regions:
            if _shares_storage(region.tensor, self._output):
                raise RuntimeError("Backend state specs must not declare runtime output regions.")

    def _output_region_for_blocks(
        self,
        blocks: Sequence[BlockRef],
    ) -> Optional[TensorRegionSpec]:
        if not blocks:
            return None
        starts: list[int] = []
        ends: list[int] = []
        for block in blocks:
            self._require_plan_block(block)
            if block.start_frame is None or block.num_frames is None:
                raise RuntimeError("Output snapshots require block frame ranges.")
            starts.append(block.start_frame)
            ends.append(block.start_frame + block.num_frames)
        return TensorRegionSpec(
            tensor=self._output,
            dim=1,
            start=min(starts),
            end=max(ends),
            name="self_forcing_runtime_output",
        )

    def _reject_latent_aliases(
        self,
        latents: Sequence[object],
        specs: RuntimeStateSpecBundle,
    ) -> None:
        spec_tensors = tuple(specs.tensor_values) + tuple(
            region.tensor for region in specs.tensor_regions
        )
        for latent in latents:
            if not isinstance(latent, torch.Tensor):
                continue
            for tensor in spec_tensors:
                if isinstance(tensor, torch.Tensor) and _shares_storage(latent, tensor):
                    raise RuntimeError("Returned latent must not alias transaction state tensors.")

    def _manager_from_specs(
        self,
        specs: RuntimeStateSpecBundle,
        *,
        include_runtime_commit_state: bool,
    ) -> RuntimeStateTransactionManager:
        object_states = tuple(specs.object_states)
        if include_runtime_commit_state:
            object_states = object_states + (self._committed_blocks_state_spec(),)
        return RuntimeStateTransactionManager(
            tensor_regions=specs.tensor_regions,
            tensor_values=specs.tensor_values,
            object_states=object_states,
            capture_rng=specs.capture_rng,
            capture_cuda_rng=specs.capture_cuda_rng,
        )

    def _committed_blocks_state_spec(self) -> ObjectStateSpec[tuple[BlockRef, ...]]:
        return ObjectStateSpec(
            getter=lambda: tuple(self._committed_blocks),
            setter=self._restore_committed_blocks,
            copy_fn=tuple,
            name="self_forcing_runtime_committed_blocks",
        )

    def _restore_committed_blocks(self, committed_blocks: object) -> None:
        self._committed_blocks = list(committed_blocks)  # type: ignore[arg-type]

    def _write_output(self, request: CommitRequest) -> None:
        block = request.block
        if block.start_frame is None or block.num_frames is None:
            raise RuntimeError("Committed BlockRef must include start_frame and num_frames.")
        start = block.start_frame
        end = start + block.num_frames
        self._output[:, start:end] = request.latent


def _build_rollout_plan(config: SelfForcingMCPRuntimeConfig) -> SelfForcingMCPRolloutPlan:
    blocks: list[BlockRef] = []
    starts: list[int] = []
    start_frame = 0
    remaining = config.all_num_frames
    while remaining > 0:
        block_frames = min(config.num_frame_per_block, remaining)
        block = BlockRef(
            index=len(blocks),
            start_frame=start_frame,
            num_frames=block_frames,
        )
        blocks.append(block)
        starts.append(start_frame)
        start_frame += block_frames
        remaining -= block_frames

    period = config.mcp_depth + 1
    anchor_indices = tuple(range(0, len(blocks), period))
    return SelfForcingMCPRolloutPlan(
        blocks=tuple(blocks),
        block_starts=tuple(starts),
        anchor_block_indices=anchor_indices,
        period=period,
    )


def _validate_output_tensor(output: object, config: SelfForcingMCPRuntimeConfig) -> None:
    if not isinstance(output, torch.Tensor):
        raise TypeError(f"output must be a torch.Tensor, got {type(output).__name__}.")
    if output.ndim < 2:
        raise ValueError("output must have at least two dimensions with frames on dim=1.")
    if output.shape[1] < config.all_num_frames:
        raise ValueError(
            f"output frame dimension must be at least {config.all_num_frames}, "
            f"got {output.shape[1]}."
        )


def _shares_storage(left: torch.Tensor, right: torch.Tensor) -> bool:
    return left.untyped_storage().data_ptr() == right.untyped_storage().data_ptr()


__all__ = [
    "RuntimeBackendProtocol",
    "RuntimeStateSpecBundle",
    "RuntimeWindowDescriptor",
    "SelfForcingMCPRolloutPlan",
    "SelfForcingMCPRuntime",
    "SelfForcingMCPRuntimeConfig",
    "SelfForcingMCPRuntimeContext",
]
