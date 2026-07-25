from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional, Sequence

from speculative.adapters.self_forcing_runtime import RuntimeWindowDescriptor
from speculative.types import BlockRef, CommitRequest, ControlRequest, DraftCandidate


SUPPORTED_ATTENTION_MODE = "causal_self_attention"
SUPPORTED_KV_CONTAINER = "list[dict[k,v,global_end_index,local_end_index]]"
SUPPORTED_CROSSATTN_CONTAINER = "list[dict[k,v,is_init]]"
SUPPORTED_LATENT_LAYOUT = "B,F,C,H,W"
SUPPORTED_MAX_MCP_DEPTH = 3

WanOperation = Literal[
    "prepare_cross_attention",
    "prepare_scratch",
    "proposal",
    "fallback",
    "commit",
    "window",
]


def _is_strict_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _require_strict_int(name: str, value: int) -> None:
    if not _is_strict_int(value):
        raise ValueError(f"{name} must be an integer, got {type(value).__name__}.")


def _require_bool(name: str, value: bool) -> None:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a bool, got {type(value).__name__}.")


def _require_positive_int(name: str, value: int) -> None:
    _require_strict_int(name, value)
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}.")


def _require_non_negative_int(name: str, value: int) -> None:
    _require_strict_int(name, value)
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value}.")


def _require_string(name: str, value: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string.")


@dataclass(frozen=True, slots=True, order=True)
class TensorRangeDescriptor:
    """Pure descriptor for a contiguous tensor slice.

    The descriptor intentionally does not hold a tensor. A future Wan backend
    binds `state_name` to a real borrowed tensor and converts the range to 2A
    `TensorRegionSpec`.
    """

    state_name: str
    dimension: int
    start: int
    end: int

    def __post_init__(self) -> None:
        _require_string("state_name", self.state_name)
        _require_strict_int("dimension", self.dimension)
        if self.dimension < 0:
            raise ValueError(f"dimension must be >= 0, got {self.dimension}.")
        _require_non_negative_int("start", self.start)
        _require_non_negative_int("end", self.end)
        if self.end < self.start:
            raise ValueError(
                f"end must be >= start, got start={self.start}, end={self.end}."
            )


@dataclass(frozen=True, slots=True, order=True)
class TokenRangeDescriptor:
    """Pure token range used to expose the pre-merge operation formula."""

    start: int
    end: int

    def __post_init__(self) -> None:
        _require_non_negative_int("start", self.start)
        _require_non_negative_int("end", self.end)
        if self.end < self.start:
            raise ValueError(
                f"end must be >= start, got start={self.start}, end={self.end}."
            )


@dataclass(frozen=True, slots=True)
class WanCacheLayout:
    """Validated description of the current Wan inference cache containers."""

    num_layers: int
    cache_capacity: int
    cross_attention_capacity: int = 512
    self_attention_dim: int = 1
    cross_attention_dim: int = 1
    attention_mode: str = SUPPORTED_ATTENTION_MODE
    kv_container: str = SUPPORTED_KV_CONTAINER
    crossattn_container: str = SUPPORTED_CROSSATTN_CONTAINER
    has_kv_cache: bool = True
    has_crossattn_cache: bool = True
    has_global_index: bool = True
    has_local_index: bool = True
    batch_size: int = 1
    cfg_batch_multiplier: int = 1
    latent_layout: str = SUPPORTED_LATENT_LAYOUT

    def __post_init__(self) -> None:
        _require_positive_int("num_layers", self.num_layers)
        _require_positive_int("cache_capacity", self.cache_capacity)
        _require_positive_int("cross_attention_capacity", self.cross_attention_capacity)
        _require_strict_int("self_attention_dim", self.self_attention_dim)
        _require_strict_int("cross_attention_dim", self.cross_attention_dim)
        _require_positive_int("batch_size", self.batch_size)
        _require_positive_int("cfg_batch_multiplier", self.cfg_batch_multiplier)
        for name in (
            "has_kv_cache",
            "has_crossattn_cache",
            "has_global_index",
            "has_local_index",
        ):
            _require_bool(name, getattr(self, name))
        _require_string("attention_mode", self.attention_mode)
        _require_string("kv_container", self.kv_container)
        _require_string("crossattn_container", self.crossattn_container)
        _require_string("latent_layout", self.latent_layout)

        if self.attention_mode != SUPPORTED_ATTENTION_MODE:
            raise ValueError(
                f"Unsupported attention mode {self.attention_mode!r}; "
                f"expected {SUPPORTED_ATTENTION_MODE!r}."
            )
        if self.kv_container != SUPPORTED_KV_CONTAINER:
            raise ValueError(
                f"Unsupported KV cache container {self.kv_container!r}; "
                f"expected {SUPPORTED_KV_CONTAINER!r}."
            )
        if self.crossattn_container != SUPPORTED_CROSSATTN_CONTAINER:
            raise ValueError(
                f"Unsupported cross-attention cache container "
                f"{self.crossattn_container!r}; expected "
                f"{SUPPORTED_CROSSATTN_CONTAINER!r}."
            )
        if self.self_attention_dim != 1:
            raise ValueError("Wan KV cache token dimension must be 1.")
        if self.cross_attention_dim != 1:
            raise ValueError("Wan cross-attention cache token dimension must be 1.")
        if not self.has_kv_cache:
            raise ValueError("Wan planner requires the current KV cache container.")
        if not self.has_crossattn_cache:
            raise ValueError("Wan planner requires the current cross-attention cache.")
        if not self.has_global_index or not self.has_local_index:
            raise ValueError("Wan planner requires both global and local cache indices.")
        if self.batch_size != 1:
            raise ValueError("Wan planner currently supports batch_size == 1 only.")
        if self.cfg_batch_multiplier != 1:
            raise ValueError("Wan planner does not support an extra CFG batch dimension.")
        if self.latent_layout != SUPPORTED_LATENT_LAYOUT:
            raise ValueError(
                f"Unsupported latent layout {self.latent_layout!r}; "
                f"expected {SUPPORTED_LATENT_LAYOUT!r}."
            )


@dataclass(frozen=True, slots=True)
class WanOperationRange:
    """Concrete token-index formula for one Wan generator/cache operation."""

    operation: WanOperation
    block_index: int
    start_frame: int
    num_frames: int
    current_start: int
    cache_start: int
    global_start: int
    global_end: int
    local_ranges: tuple[TokenRangeDescriptor, ...]
    global_end_index: int
    local_end_index: int
    rolls_local_cache: bool = False

    def __post_init__(self) -> None:
        _require_string("operation", self.operation)
        _require_non_negative_int("block_index", self.block_index)
        _require_non_negative_int("start_frame", self.start_frame)
        _require_positive_int("num_frames", self.num_frames)
        _require_non_negative_int("current_start", self.current_start)
        _require_non_negative_int("cache_start", self.cache_start)
        _require_non_negative_int("global_start", self.global_start)
        _require_non_negative_int("global_end", self.global_end)
        _require_non_negative_int("global_end_index", self.global_end_index)
        _require_non_negative_int("local_end_index", self.local_end_index)
        _require_bool("rolls_local_cache", self.rolls_local_cache)
        if self.global_end < self.global_start:
            raise ValueError(
                f"global_end must be >= global_start, got "
                f"{self.global_start}, {self.global_end}."
            )
        object.__setattr__(self, "local_ranges", tuple(self.local_ranges))
        for local_range in self.local_ranges:
            if not isinstance(local_range, TokenRangeDescriptor):
                raise TypeError("local_ranges must contain TokenRangeDescriptor values.")


@dataclass(frozen=True, slots=True)
class WanStateMutationPlan:
    """Immutable planner output for later binding to runtime state specs."""

    backend_tensor_ranges: tuple[TensorRangeDescriptor, ...] = field(default_factory=tuple)
    backend_tensor_value_names: tuple[str, ...] = field(default_factory=tuple)
    backend_python_value_names: tuple[str, ...] = field(default_factory=tuple)
    output_ranges: tuple[TensorRangeDescriptor, ...] = field(default_factory=tuple)
    capture_rng: bool = False
    capture_cuda_rng: bool = False
    operation_ranges: tuple[WanOperationRange, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        _require_bool("capture_rng", self.capture_rng)
        _require_bool("capture_cuda_rng", self.capture_cuda_rng)
        object.__setattr__(self, "backend_tensor_ranges", tuple(self.backend_tensor_ranges))
        object.__setattr__(self, "backend_tensor_value_names", tuple(self.backend_tensor_value_names))
        object.__setattr__(self, "backend_python_value_names", tuple(self.backend_python_value_names))
        object.__setattr__(self, "output_ranges", tuple(self.output_ranges))
        object.__setattr__(self, "operation_ranges", tuple(self.operation_ranges))
        for descriptor in self.backend_tensor_ranges:
            if not isinstance(descriptor, TensorRangeDescriptor):
                raise TypeError(
                    "backend_tensor_ranges must contain TensorRangeDescriptor values."
                )
            if descriptor.state_name == "output":
                raise ValueError("output must not appear in backend_tensor_ranges.")
            if descriptor.state_name == "self_forcing_runtime_committed_blocks":
                raise ValueError(
                    "runtime committed-block bookkeeping must not be a backend descriptor."
                )
        for descriptor in self.output_ranges:
            if not isinstance(descriptor, TensorRangeDescriptor):
                raise TypeError("output_ranges must contain TensorRangeDescriptor values.")
            if descriptor.state_name != "output":
                raise ValueError("output_ranges must use state_name='output'.")
        for name in self.backend_tensor_value_names:
            _require_string("backend tensor value name", name)
            if name in ("output", "self_forcing_runtime_committed_blocks"):
                raise ValueError(f"{name} must not be a backend tensor value descriptor.")
        for name in self.backend_python_value_names:
            _require_string("backend python value name", name)
            if name in ("output", "self_forcing_runtime_committed_blocks"):
                raise ValueError(f"{name} must not be a backend python value descriptor.")
        tensor_state_names = {descriptor.state_name for descriptor in self.backend_tensor_ranges}
        value_state_names = set(self.backend_tensor_value_names) | set(self.backend_python_value_names)
        duplicate_semantics = sorted(tensor_state_names & value_state_names)
        if duplicate_semantics:
            raise ValueError(
                "same backend state cannot be both tensor range and value "
                f"descriptor: {duplicate_semantics}."
            )
        for descriptor in self.operation_ranges:
            if not isinstance(descriptor, WanOperationRange):
                raise TypeError("operation_ranges must contain WanOperationRange values.")

    @classmethod
    def from_parts(
        cls,
        *,
        backend_tensor_ranges: Sequence[TensorRangeDescriptor] = (),
        backend_tensor_value_names: Sequence[str] = (),
        backend_python_value_names: Sequence[str] = (),
        output_ranges: Sequence[TensorRangeDescriptor] = (),
        capture_rng: bool = False,
        capture_cuda_rng: bool = False,
        operation_ranges: Sequence[WanOperationRange] = (),
    ) -> "WanStateMutationPlan":
        return cls(
            backend_tensor_ranges=_merge_tensor_ranges(backend_tensor_ranges),
            backend_tensor_value_names=tuple(dict.fromkeys(backend_tensor_value_names)),
            backend_python_value_names=tuple(dict.fromkeys(backend_python_value_names)),
            output_ranges=_merge_tensor_ranges(output_ranges),
            capture_rng=capture_rng,
            capture_cuda_rng=capture_cuda_rng,
            operation_ranges=tuple(operation_ranges),
        )


@dataclass(frozen=True, slots=True)
class WanTouchedRangePlanner:
    """Pure Wan touched-range planner.

    This object holds only immutable configuration. It never stores a pipeline,
    model, runtime, cache, generator, tensor, mutable frame cursor, or RNG object.
    """

    frame_seq_length: int
    num_frame_per_block: int
    mcp_depth: int
    cache_layout: WanCacheLayout
    local_attn_size: int = -1
    sink_size: int = 0
    runtime_device_type: Literal["cpu", "cuda"] = "cuda"

    def __post_init__(self) -> None:
        _require_positive_int("frame_seq_length", self.frame_seq_length)
        _require_positive_int("num_frame_per_block", self.num_frame_per_block)
        _require_non_negative_int("mcp_depth", self.mcp_depth)
        _require_strict_int("local_attn_size", self.local_attn_size)
        _require_non_negative_int("sink_size", self.sink_size)
        _require_string("runtime_device_type", self.runtime_device_type)
        if self.local_attn_size < -1:
            raise ValueError(
                f"local_attn_size must be -1 or a positive integer, got "
                f"{self.local_attn_size}."
            )
        if self.local_attn_size == 0:
            raise ValueError("local_attn_size=0 is unsupported; use -1 or a positive integer.")
        if self.local_attn_size == -1 and self.sink_size > 0:
            raise ValueError("sink_size > 0 requires positive local_attn_size.")
        if self.mcp_depth > SUPPORTED_MAX_MCP_DEPTH:
            raise ValueError(
                f"mcp_depth must be <= {SUPPORTED_MAX_MCP_DEPTH}, got "
                f"{self.mcp_depth}."
            )
        if self.runtime_device_type not in ("cpu", "cuda"):
            raise ValueError(
                f"runtime_device_type must be 'cpu' or 'cuda', got "
                f"{self.runtime_device_type!r}."
            )

    def current_start_for(self, block: BlockRef) -> int:
        start_frame, _ = self._block_frame_range(block)
        return start_frame * self.frame_seq_length

    def allowed_blocks_for_request(
        self,
        request: ControlRequest,
        rollout_blocks: Sequence[BlockRef],
    ) -> tuple[BlockRef, ...]:
        if not isinstance(request, ControlRequest):
            raise TypeError("request must be a ControlRequest.")
        blocks = tuple(rollout_blocks)
        self._validate_rollout_blocks(blocks)
        anchor = request.anchor_block
        if anchor.index >= len(blocks) or blocks[anchor.index] != anchor:
            raise ValueError("request.anchor_block must identify a rollout block.")
        remaining = len(blocks) - anchor.index - 1
        depth_count = min(request.max_depth, self.mcp_depth, remaining)
        return blocks[anchor.index:anchor.index + depth_count + 1]

    def plan_prepare_cross_attention(self) -> WanStateMutationPlan:
        ranges: list[TensorRangeDescriptor] = []
        python_value_names: list[str] = []
        for layer in range(self.cache_layout.num_layers):
            for field_name in ("k", "v"):
                name = self._crossattn_state_name(layer, field_name)
                ranges.append(
                    TensorRangeDescriptor(
                        state_name=name,
                        dimension=self.cache_layout.cross_attention_dim,
                        start=0,
                        end=self.cache_layout.cross_attention_capacity,
                    )
                )
            python_value_names.append(self._crossattn_state_name(layer, "is_init"))
        return WanStateMutationPlan.from_parts(
            backend_tensor_ranges=ranges,
            backend_python_value_names=python_value_names,
        )

    def plan_prepare_scratch(self, block: BlockRef) -> WanStateMutationPlan:
        if not isinstance(block, BlockRef):
            raise TypeError("block must be a BlockRef.")
        start_frame, _ = self._block_frame_range(block)
        if block.index != 0:
            raise ValueError("prepare scratch block must be rollout block index 0.")
        if start_frame != 0:
            raise ValueError("prepare scratch baseline requires start_frame == 0.")
        operation_range = self._operation_range_for_block(
            block,
            operation="prepare_scratch",
        )
        return self._plan_from_operation_ranges(
            operation_ranges=(operation_range,),
            output_blocks=(),
            capture_commit_rng=False,
        )

    def plan_proposal(
        self,
        request: ControlRequest,
        allowed_blocks: Optional[Sequence[BlockRef]] = None,
    ) -> WanStateMutationPlan:
        if not isinstance(request, ControlRequest):
            raise TypeError("request must be a ControlRequest.")
        blocks = (request.anchor_block,) if allowed_blocks is None else tuple(allowed_blocks)
        self._validate_allowed_blocks(request.anchor_block, blocks)
        draft_count = len(blocks) - 1
        if draft_count > request.max_depth:
            raise ValueError(
                f"allowed_blocks contains {draft_count} draft blocks, exceeding "
                f"request.max_depth={request.max_depth}."
            )
        if draft_count > self.mcp_depth:
            raise ValueError(
                f"allowed_blocks contains {draft_count} draft blocks, exceeding "
                f"planner mcp_depth={self.mcp_depth}."
            )
        operation_range = self._operation_range_for_block(
            request.anchor_block,
            operation="proposal",
        )
        return self._plan_from_operation_ranges(
            operation_ranges=(operation_range,),
            output_blocks=blocks,
            capture_commit_rng=False,
        )

    def plan_fallback(self, candidate: DraftCandidate | BlockRef) -> WanStateMutationPlan:
        block = candidate.block if isinstance(candidate, DraftCandidate) else candidate
        operation_range = self._operation_range_for_block(block, operation="fallback")
        return self._plan_from_operation_ranges(
            operation_ranges=(operation_range,),
            output_blocks=(block,),
            capture_commit_rng=False,
        )

    def plan_commit(self, request: CommitRequest | BlockRef) -> WanStateMutationPlan:
        block = request.block if isinstance(request, CommitRequest) else request
        operation_range = self._operation_range_for_block(block, operation="commit")
        return self._plan_from_operation_ranges(
            operation_ranges=(operation_range,),
            output_blocks=(block,),
            capture_commit_rng=True,
        )

    def plan_window(self, window: RuntimeWindowDescriptor) -> WanStateMutationPlan:
        if not isinstance(window, RuntimeWindowDescriptor):
            raise TypeError("window must be a RuntimeWindowDescriptor.")
        self._validate_allowed_blocks(window.anchor_block, window.allowed_blocks)
        operation_ranges = tuple(
            self._operation_range_for_block(block, operation="commit")
            for block in window.allowed_blocks
        )
        plan = self._plan_from_operation_ranges(
            operation_ranges=operation_ranges,
            output_blocks=window.allowed_blocks,
            capture_commit_rng=True,
        )
        return WanStateMutationPlan.from_parts(
            backend_tensor_ranges=plan.backend_tensor_ranges,
            backend_tensor_value_names=plan.backend_tensor_value_names,
            backend_python_value_names=plan.backend_python_value_names,
            output_ranges=plan.output_ranges,
            capture_rng=plan.capture_rng,
            capture_cuda_rng=plan.capture_cuda_rng,
            operation_ranges=plan.operation_ranges,
        )

    def plan_operation(
        self,
        operation: WanOperation,
        target: object = None,
    ) -> WanStateMutationPlan:
        if operation == "prepare_cross_attention":
            return self.plan_prepare_cross_attention()
        if operation == "prepare_scratch":
            if not isinstance(target, BlockRef):
                raise TypeError("prepare_scratch target must be a BlockRef.")
            return self.plan_prepare_scratch(target)
        if operation == "proposal":
            if not isinstance(target, ControlRequest):
                raise TypeError("proposal target must be a ControlRequest.")
            return self.plan_proposal(target)
        if operation == "fallback":
            if not isinstance(target, (DraftCandidate, BlockRef)):
                raise TypeError("fallback target must be a DraftCandidate or BlockRef.")
            return self.plan_fallback(target)
        if operation == "commit":
            if not isinstance(target, (CommitRequest, BlockRef)):
                raise TypeError("commit target must be a CommitRequest or BlockRef.")
            return self.plan_commit(target)
        if operation == "window":
            if not isinstance(target, RuntimeWindowDescriptor):
                raise TypeError("window target must be a RuntimeWindowDescriptor.")
            return self.plan_window(target)
        raise ValueError(f"Unsupported Wan operation {operation!r}.")

    def _plan_from_operation_ranges(
        self,
        *,
        operation_ranges: Sequence[WanOperationRange],
        output_blocks: Sequence[BlockRef],
        capture_commit_rng: bool,
    ) -> WanStateMutationPlan:
        backend_tensor_ranges: list[TensorRangeDescriptor] = []
        backend_tensor_value_names: list[str] = []
        output_ranges: list[TensorRangeDescriptor] = []
        for operation_range in operation_ranges:
            for layer in range(self.cache_layout.num_layers):
                for field_name in ("k", "v"):
                    state_name = self._kv_state_name(layer, field_name)
                    for local_range in operation_range.local_ranges:
                        if local_range.start == local_range.end:
                            continue
                        backend_tensor_ranges.append(
                            TensorRangeDescriptor(
                                state_name=state_name,
                                dimension=self.cache_layout.self_attention_dim,
                                start=local_range.start,
                                end=local_range.end,
                            )
                        )
                backend_tensor_value_names.append(self._kv_state_name(layer, "global_end_index"))
                backend_tensor_value_names.append(self._kv_state_name(layer, "local_end_index"))

        for block in output_blocks:
            start_frame, num_frames = self._block_frame_range(block)
            output_ranges.append(
                TensorRangeDescriptor(
                    state_name="output",
                    dimension=1,
                    start=start_frame,
                    end=start_frame + num_frames,
                )
            )

        capture_rng = capture_commit_rng and self.runtime_device_type == "cpu"
        capture_cuda_rng = capture_commit_rng and self.runtime_device_type == "cuda"
        return WanStateMutationPlan.from_parts(
            backend_tensor_ranges=backend_tensor_ranges,
            backend_tensor_value_names=backend_tensor_value_names,
            output_ranges=output_ranges,
            capture_rng=capture_rng,
            capture_cuda_rng=capture_cuda_rng,
            operation_ranges=operation_ranges,
        )

    def _operation_range_for_block(
        self,
        block: BlockRef,
        *,
        operation: WanOperation,
    ) -> WanOperationRange:
        start_frame, num_frames = self._block_frame_range(block)
        current_start = start_frame * self.frame_seq_length
        num_new_tokens = num_frames * self.frame_seq_length
        cache_start = current_start
        pre_global_end_index = cache_start
        pre_local_end_index = self._local_index_before(cache_start)
        cache_end = cache_start + num_new_tokens
        sink_tokens = self.sink_size * self.frame_seq_length

        if self.local_attn_size == -1 and cache_end > self.cache_layout.cache_capacity:
            raise ValueError(
                f"KV cache range [{cache_start}, {cache_end}) exceeds capacity "
                f"{self.cache_layout.cache_capacity}."
            )
        if self.local_attn_size != -1 and num_new_tokens + sink_tokens > self.cache_layout.cache_capacity:
            raise ValueError(
                "A single Wan block plus sink tokens must fit inside the KV cache."
            )

        rolls_local_cache = False
        local_ranges: tuple[TokenRangeDescriptor, ...]
        if self.local_attn_size != -1 and (
            cache_end > pre_global_end_index
        ) and (num_new_tokens + pre_local_end_index > self.cache_layout.cache_capacity):
            rolls_local_cache = True
            num_evicted_tokens = (
                num_new_tokens
                + pre_local_end_index
                - self.cache_layout.cache_capacity
            )
            num_rolled_tokens = (
                pre_local_end_index - num_evicted_tokens - sink_tokens
            )
            if num_rolled_tokens < 0:
                raise ValueError(
                    "Local Wan cache rollover would evict beyond the sink range."
                )
            local_end_index = (
                pre_local_end_index
                + cache_end
                - pre_global_end_index
                - num_evicted_tokens
            )
            local_start_index = local_end_index - num_new_tokens
            segments = []
            if num_rolled_tokens > 0:
                segments.append(
                    TokenRangeDescriptor(
                        sink_tokens,
                        sink_tokens + num_rolled_tokens,
                    )
                )
            segments.append(TokenRangeDescriptor(local_start_index, local_end_index))
            local_ranges = tuple(segments)
        else:
            local_end_index = (
                pre_local_end_index + cache_end - pre_global_end_index
            )
            local_start_index = local_end_index - num_new_tokens
            local_ranges = (TokenRangeDescriptor(local_start_index, local_end_index),)

        for local_range in local_ranges:
            self._validate_cache_range(local_range.start, local_range.end)
        self._validate_cache_range(local_start_index, local_end_index)

        return WanOperationRange(
            operation=operation,
            block_index=block.index,
            start_frame=start_frame,
            num_frames=num_frames,
            current_start=current_start,
            cache_start=cache_start,
            global_start=cache_start,
            global_end=cache_end,
            local_ranges=local_ranges,
            global_end_index=cache_end,
            local_end_index=local_end_index,
            rolls_local_cache=rolls_local_cache,
        )

    def _local_index_before(self, current_start: int) -> int:
        if self.local_attn_size == -1:
            return current_start
        return min(current_start, self.cache_layout.cache_capacity)

    def _validate_cache_range(self, start: int, end: int) -> None:
        if start < 0 or end < start or end > self.cache_layout.cache_capacity:
            raise ValueError(
                f"KV cache range [{start}, {end}) is outside capacity "
                f"{self.cache_layout.cache_capacity}."
            )

    def _block_frame_range(self, block: BlockRef) -> tuple[int, int]:
        if not isinstance(block, BlockRef):
            raise TypeError("block must be a BlockRef.")
        if block.start_frame is None or block.num_frames is None:
            raise ValueError("BlockRef.start_frame and num_frames are required.")
        if block.num_frames > self.num_frame_per_block:
            raise ValueError(
                f"BlockRef.num_frames ({block.num_frames}) must be <= "
                f"num_frame_per_block ({self.num_frame_per_block})."
            )
        return block.start_frame, block.num_frames

    def _validate_allowed_blocks(
        self,
        anchor_block: BlockRef,
        allowed_blocks: Sequence[BlockRef],
    ) -> None:
        blocks = tuple(allowed_blocks)
        if not blocks:
            raise ValueError("allowed_blocks must not be empty.")
        if blocks[0] != anchor_block:
            raise ValueError("allowed_blocks must start with the anchor block.")
        self._validate_rollout_blocks(blocks)

    def _validate_rollout_blocks(self, blocks: Sequence[BlockRef]) -> None:
        if not blocks:
            raise ValueError("rollout_blocks must not be empty.")
        for block in blocks:
            self._block_frame_range(block)
        for previous, current in zip(blocks, blocks[1:]):
            if current.index != previous.index + 1:
                raise ValueError("rollout blocks must have contiguous indices.")
            expected_start = previous.start_frame + previous.num_frames
            if current.start_frame != expected_start:
                raise ValueError("rollout blocks must have contiguous frame ranges.")

    @staticmethod
    def _kv_state_name(layer: int, field_name: str) -> str:
        return f"kv_cache.layer_{layer:03d}.{field_name}"

    @staticmethod
    def _crossattn_state_name(layer: int, field_name: str) -> str:
        return f"crossattn_cache.layer_{layer:03d}.{field_name}"


def _merge_tensor_ranges(
    tensor_ranges: Sequence[TensorRangeDescriptor],
) -> tuple[TensorRangeDescriptor, ...]:
    ordered = sorted(tuple(tensor_ranges))
    if not ordered:
        return ()

    merged: list[TensorRangeDescriptor] = []
    for descriptor in ordered:
        if descriptor.start == descriptor.end:
            continue
        if not merged:
            merged.append(descriptor)
            continue
        previous = merged[-1]
        if (
            previous.state_name == descriptor.state_name
            and previous.dimension == descriptor.dimension
            and descriptor.start <= previous.end
        ):
            merged[-1] = TensorRangeDescriptor(
                state_name=previous.state_name,
                dimension=previous.dimension,
                start=previous.start,
                end=max(previous.end, descriptor.end),
            )
        else:
            merged.append(descriptor)
    return tuple(merged)


__all__ = [
    "SUPPORTED_ATTENTION_MODE",
    "SUPPORTED_CROSSATTN_CONTAINER",
    "SUPPORTED_KV_CONTAINER",
    "SUPPORTED_LATENT_LAYOUT",
    "SUPPORTED_MAX_MCP_DEPTH",
    "TensorRangeDescriptor",
    "TokenRangeDescriptor",
    "WanCacheLayout",
    "WanOperationRange",
    "WanStateMutationPlan",
    "WanTouchedRangePlanner",
]
