from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Callable, Generic, Mapping, Optional, Sequence, TypeVar

import torch


T = TypeVar("T")


class RuntimeStateRestoreError(RuntimeError):
    """Raised after best-effort restore attempts one or more state restores.

    The `errors` tuple preserves every restore exception in the order it was
    observed. Restore always attempts tensor regions, tensor values, object
    states, then RNG before raising this aggregate error.
    """

    def __init__(self, errors: Sequence[Exception]) -> None:
        self.errors = tuple(errors)
        summary = ", ".join(type(error).__name__ for error in self.errors)
        super().__init__(f"runtime state restore failed with {len(self.errors)} error(s): {summary}")


class RuntimeStateRollbackError(RuntimeError):
    """Raised when context-body failure is followed by rollback failure."""

    def __init__(self, original_exception: BaseException, restore_exception: Exception) -> None:
        self.original_exception = original_exception
        self.restore_exception = restore_exception
        super().__init__(
            "runtime state rollback failed after "
            f"{type(original_exception).__name__}: {original_exception}; "
            "restore raised "
            f"{type(restore_exception).__name__}: {restore_exception}"
        )


def _is_strict_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _require_strict_int(name: str, value: int) -> None:
    if not _is_strict_int(value):
        raise ValueError(f"{name} must be an integer, got {type(value).__name__}.")


def _require_tensor(name: str, value: torch.Tensor) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(value).__name__}.")


def _default_object_copy(value: T) -> T:
    return copy.deepcopy(value)


def _tensor_stride(tensor: torch.Tensor) -> tuple[int, ...]:
    return tuple(tensor.stride())


def _validate_tensor_metadata(
    *,
    tensor: torch.Tensor,
    source_shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
    layout: torch.layout,
    stride: tuple[int, ...],
    label: str,
) -> None:
    if tuple(tensor.shape) != source_shape:
        raise RuntimeError(
            f"{label} shape changed from {source_shape} to {tuple(tensor.shape)}."
        )
    if tensor.dtype != dtype:
        raise RuntimeError(f"{label} dtype changed from {dtype} to {tensor.dtype}.")
    if tensor.device != device:
        raise RuntimeError(f"{label} device changed from {device} to {tensor.device}.")
    if tensor.layout != layout:
        raise RuntimeError(f"{label} layout changed from {layout} to {tensor.layout}.")
    current_stride = _tensor_stride(tensor)
    if current_stride != stride:
        raise RuntimeError(f"{label} stride changed from {stride} to {current_stride}.")


def _validate_region_bounds(spec: "TensorRegionSpec") -> None:
    tensor = spec.tensor
    if tensor.ndim == 0:
        raise RuntimeError("tensor region restore requires a tensor with at least one dimension.")
    if spec.dim < 0 or spec.dim >= tensor.ndim:
        raise RuntimeError(
            f"tensor region dim {spec.dim} is invalid for tensor ndim {tensor.ndim}."
        )
    if spec.start < 0 or spec.end <= spec.start or spec.end > tensor.shape[spec.dim]:
        raise RuntimeError(
            f"tensor region [{spec.start}, {spec.end}) is invalid for "
            f"tensor.shape[{spec.dim}]={tensor.shape[spec.dim]}."
        )


def _contains_tensor(value: object, seen: Optional[set[int]] = None) -> bool:
    if isinstance(value, torch.Tensor):
        return True
    if seen is None:
        seen = set()
    object_id = id(value)
    if object_id in seen:
        return False

    if isinstance(value, Mapping):
        seen.add(object_id)
        return any(_contains_tensor(k, seen) or _contains_tensor(v, seen) for k, v in value.items())
    if isinstance(value, (list, tuple, set, frozenset)):
        seen.add(object_id)
        return any(_contains_tensor(item, seen) for item in value)
    return False


def _reject_tensor_backed_object(value: object, name: str) -> None:
    if _contains_tensor(value):
        label = f"ObjectStateSpec {name!r}" if name else "ObjectStateSpec"
        raise TypeError(f"{label} cannot snapshot tensor-backed object state.")


@dataclass(frozen=True)
class TensorRegionSpec:
    """Borrowed tensor region to snapshot.

    The spec describes one contiguous slice along a single dimension. It does
    not infer model-specific ranges, cache semantics, or Wan attention behavior;
    callers must provide the exact touched region they want protected.
    """

    tensor: torch.Tensor
    dim: int
    start: int
    end: int
    name: str = ""

    def __post_init__(self) -> None:
        _require_tensor("tensor", self.tensor)
        _require_strict_int("dim", self.dim)
        _require_strict_int("start", self.start)
        _require_strict_int("end", self.end)
        if self.tensor.ndim == 0:
            raise ValueError("tensor region snapshots require a tensor with at least one dimension.")

        dim = self.dim
        if dim < 0:
            dim += self.tensor.ndim
        if dim < 0 or dim >= self.tensor.ndim:
            raise ValueError(
                f"dim must be in [{-self.tensor.ndim}, {self.tensor.ndim - 1}], got {self.dim}."
            )
        if self.start < 0:
            raise ValueError(f"start must be >= 0, got {self.start}.")
        if self.end <= self.start:
            raise ValueError(f"end must be > start, got start={self.start}, end={self.end}.")
        if self.end > self.tensor.shape[dim]:
            raise ValueError(
                f"end must be <= tensor.shape[{dim}] ({self.tensor.shape[dim]}), got {self.end}."
            )
        object.__setattr__(self, "dim", dim)

    def index(self) -> tuple[slice, ...]:
        """Return the tuple index for this region."""

        slices = [slice(None)] * self.tensor.ndim
        slices[self.dim] = slice(self.start, self.end)
        return tuple(slices)


@dataclass(frozen=True)
class TensorRegionSnapshot:
    """Cloned value of a borrowed tensor region.

    Restore uses in-place `copy_` into the original tensor slice. The tensor
    object itself is never replaced.
    """

    spec: TensorRegionSpec
    _value: torch.Tensor = field(repr=False)
    _source_shape: tuple[int, ...]
    _dtype: torch.dtype
    _device: torch.device
    _layout: torch.layout
    _stride: tuple[int, ...]

    @classmethod
    def capture(cls, spec: TensorRegionSpec) -> "TensorRegionSnapshot":
        _validate_region_bounds(spec)
        region = spec.tensor[spec.index()]
        return cls(
            spec=spec,
            _value=region.detach().clone(),
            _source_shape=tuple(spec.tensor.shape),
            _dtype=spec.tensor.dtype,
            _device=spec.tensor.device,
            _layout=spec.tensor.layout,
            _stride=_tensor_stride(spec.tensor),
        )

    @property
    def value(self) -> torch.Tensor:
        """Return a clone of the captured region value."""

        return self._value.detach().clone()

    def restore(self) -> None:
        _validate_tensor_metadata(
            tensor=self.spec.tensor,
            source_shape=self._source_shape,
            dtype=self._dtype,
            device=self._device,
            layout=self._layout,
            stride=self._stride,
            label="tensor region source",
        )
        _validate_region_bounds(self.spec)
        target = self.spec.tensor[self.spec.index()]
        if tuple(target.shape) != tuple(self._value.shape):
            raise RuntimeError(
                f"tensor region shape changed from {tuple(self._value.shape)} "
                f"to {tuple(target.shape)} during restore."
            )
        with torch.no_grad():
            target.copy_(self._value)


@dataclass(frozen=True)
class TensorValueSnapshot:
    """Cloned value of a full tensor state.

    This is for scalar or small tensor state such as cache indices. Restore is
    in-place and preserves all caller-held references to the original tensor.
    """

    tensor: torch.Tensor
    _value: torch.Tensor = field(repr=False)
    _source_shape: tuple[int, ...]
    _dtype: torch.dtype
    _device: torch.device
    _layout: torch.layout
    _stride: tuple[int, ...]
    name: str = ""

    @classmethod
    def capture(cls, tensor: torch.Tensor, name: str = "") -> "TensorValueSnapshot":
        _require_tensor("tensor", tensor)
        return cls(
            tensor=tensor,
            _value=tensor.detach().clone(),
            _source_shape=tuple(tensor.shape),
            _dtype=tensor.dtype,
            _device=tensor.device,
            _layout=tensor.layout,
            _stride=_tensor_stride(tensor),
            name=name,
        )

    @property
    def value(self) -> torch.Tensor:
        """Return a clone of the captured tensor value."""

        return self._value.detach().clone()

    def restore(self) -> None:
        _validate_tensor_metadata(
            tensor=self.tensor,
            source_shape=self._source_shape,
            dtype=self._dtype,
            device=self._device,
            layout=self._layout,
            stride=self._stride,
            label="tensor value source",
        )
        with torch.no_grad():
            self.tensor.copy_(self._value)


@dataclass(frozen=True)
class ObjectStateSpec(Generic[T]):
    """Getter/setter pair for Python object state.

    This covers commit-order bookkeeping, cursor-like metadata, and
    mapping/list/set state. `copy_fn` is a capture-time transformation from
    caller state into small restorable Python state. The default transformation
    is `deepcopy`, intended only for small Python bookkeeping.

    Do not use this for tensors, KV caches, models, schedulers, generators,
    CUDA streams, or CUDA events. Tensor state must be protected through
    explicit tensor snapshots. `getter` and `copy_fn` must be side-effect-free
    and must not consume RNG. The transformed value must be tensor-free and
    deepcopy-able. `setter` must restore only the declared transformed object
    state and must not mutate tensor-managed state.
    """

    getter: Callable[[], T]
    setter: Callable[[Any], None]
    copy_fn: Callable[[T], Any] = _default_object_copy
    name: str = ""

    def __post_init__(self) -> None:
        if not callable(self.getter):
            raise TypeError("getter must be callable.")
        if not callable(self.setter):
            raise TypeError("setter must be callable.")
        if not callable(self.copy_fn):
            raise TypeError("copy_fn must be callable.")


@dataclass(frozen=True)
class ObjectStateSnapshot:
    """Independent copy of transformed Python object state.

    `ObjectStateSpec.copy_fn` is used only during capture. The transformed
    value is deep-copied into a private backup. `value` and `restore()` always
    deep-copy that backup rather than calling a user-supplied copy function.
    """

    setter: Callable[[Any], None]
    _value: Any = field(repr=False)
    name: str = ""

    @classmethod
    def capture(cls, spec: ObjectStateSpec[Any]) -> "ObjectStateSnapshot":
        source = spec.getter()
        _reject_tensor_backed_object(source, spec.name)
        transformed = spec.copy_fn(source)
        _reject_tensor_backed_object(transformed, spec.name)
        backup = copy.deepcopy(transformed)
        return cls(
            setter=spec.setter,
            _value=backup,
            name=spec.name,
        )

    @property
    def value(self) -> Any:
        """Return an independent copy of the captured Python object state."""

        return copy.deepcopy(self._value)

    def restore(self) -> None:
        self.setter(copy.deepcopy(self._value))


@dataclass(frozen=True)
class TorchRNGSnapshot:
    """Torch RNG state snapshot.

    CPU RNG is always captured. CUDA RNG is captured only when requested and
    CUDA is available. This class does not require CUDA for CPU-only tests.
    """

    _cpu_state: torch.Tensor = field(repr=False)
    _cuda_states: Optional[tuple[torch.Tensor, ...]] = field(default=None, repr=False)

    @classmethod
    def capture(cls, *, include_cuda: bool = False) -> "TorchRNGSnapshot":
        cuda_states: Optional[tuple[torch.Tensor, ...]] = None
        if include_cuda:
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA RNG capture was requested, but CUDA is not available.")
            cuda_states = tuple(state.detach().clone() for state in torch.cuda.get_rng_state_all())
        return cls(
            _cpu_state=torch.random.get_rng_state().detach().clone(),
            _cuda_states=cuda_states,
        )

    @property
    def cpu_state(self) -> torch.Tensor:
        """Return a clone of the captured CPU RNG state."""

        return self._cpu_state.detach().clone()

    @property
    def cuda_states(self) -> Optional[tuple[torch.Tensor, ...]]:
        """Return clones of captured CUDA RNG states, when present."""

        if self._cuda_states is None:
            return None
        return tuple(state.detach().clone() for state in self._cuda_states)

    def restore(self) -> None:
        torch.random.set_rng_state(self._cpu_state.detach().clone())
        if self._cuda_states is not None:
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA RNG restore was requested, but CUDA is not available.")
            torch.cuda.set_rng_state_all([state.detach().clone() for state in self._cuda_states])


@dataclass(frozen=True)
class RuntimeStateSnapshot:
    """Aggregate snapshot for adapter-local and committer transactions.

    Restore order is fixed:
    1. tensor regions;
    2. full tensor values;
    3. Python object states;
    4. torch RNG state.

    Tensor data is restored before metadata, and RNG is restored last so restore
    bookkeeping cannot perturb the final random sequence.
    """

    tensor_regions: tuple[TensorRegionSnapshot, ...] = field(default_factory=tuple)
    tensor_values: tuple[TensorValueSnapshot, ...] = field(default_factory=tuple)
    object_states: tuple[ObjectStateSnapshot, ...] = field(default_factory=tuple)
    rng_state: Optional[TorchRNGSnapshot] = None

    @classmethod
    def capture(
        cls,
        *,
        tensor_regions: Sequence[TensorRegionSpec] = (),
        tensor_values: Sequence[torch.Tensor] = (),
        object_states: Sequence[ObjectStateSpec[Any]] = (),
        capture_rng: bool = False,
        capture_cuda_rng: bool = False,
    ) -> "RuntimeStateSnapshot":
        """Capture the caller-specified state.

        This method only snapshots explicit inputs. It does not walk Python
        object graphs or infer which model cache slices were touched.
        """

        _validate_snapshot_specs(tensor_regions, tensor_values)
        region_snapshots = tuple(TensorRegionSnapshot.capture(spec) for spec in tensor_regions)
        tensor_snapshots = tuple(TensorValueSnapshot.capture(tensor) for tensor in tensor_values)
        object_snapshots = tuple(ObjectStateSnapshot.capture(spec) for spec in object_states)
        rng_snapshot = (
            TorchRNGSnapshot.capture(include_cuda=capture_cuda_rng)
            if capture_rng or capture_cuda_rng
            else None
        )
        return cls(
            tensor_regions=region_snapshots,
            tensor_values=tensor_snapshots,
            object_states=object_snapshots,
            rng_state=rng_snapshot,
        )

    def restore(self) -> None:
        errors: list[Exception] = []
        for snapshot in self.tensor_regions:
            try:
                snapshot.restore()
            except Exception as error:
                errors.append(error)
        for snapshot in self.tensor_values:
            try:
                snapshot.restore()
            except Exception as error:
                errors.append(error)
        for snapshot in self.object_states:
            try:
                snapshot.restore()
            except Exception as error:
                errors.append(error)
        if self.rng_state is not None:
            try:
                self.rng_state.restore()
            except Exception as error:
                errors.append(error)
        if errors:
            raise RuntimeStateRestoreError(errors)


def _validate_snapshot_specs(
    tensor_regions: Sequence[TensorRegionSpec],
    tensor_values: Sequence[torch.Tensor],
) -> None:
    full_tensor_ids: set[int] = set()
    for tensor in tensor_values:
        _require_tensor("tensor value", tensor)
        tensor_id = id(tensor)
        if tensor_id in full_tensor_ids:
            raise ValueError("same tensor cannot be full-snapshotted more than once.")
        full_tensor_ids.add(tensor_id)

    regions_by_tensor: dict[int, list[TensorRegionSpec]] = {}
    for spec in tensor_regions:
        if not isinstance(spec, TensorRegionSpec):
            raise TypeError(
                f"tensor_regions must contain TensorRegionSpec, got {type(spec).__name__}."
            )
        tensor_id = id(spec.tensor)
        if tensor_id in full_tensor_ids:
            raise ValueError("same tensor cannot have both full and region snapshots.")
        regions_by_tensor.setdefault(tensor_id, []).append(spec)

    for tensor_id, regions in regions_by_tensor.items():
        dims = {spec.dim for spec in regions}
        if len(dims) > 1:
            raise ValueError("same tensor cannot have region snapshots across different dims.")
        ordered = sorted(regions, key=lambda spec: spec.start)
        for left, right in zip(ordered, ordered[1:]):
            if left.end > right.start:
                raise ValueError("same tensor cannot have overlapping region snapshots.")


class RuntimeStateTransactionManager:
    """Factory and single-active-owner for runtime state transactions.

    A manager is configured with explicit state specs. It rejects nested
    transactions and marks a transaction active only after every capture step
    succeeds, so `begin()` is exception-atomic with respect to manager state.
    """

    def __init__(
        self,
        *,
        tensor_regions: Sequence[TensorRegionSpec] = (),
        tensor_values: Sequence[torch.Tensor] = (),
        object_states: Sequence[ObjectStateSpec[Any]] = (),
        capture_rng: bool = False,
        capture_cuda_rng: bool = False,
    ) -> None:
        _validate_snapshot_specs(tensor_regions, tensor_values)
        self._tensor_regions = tuple(tensor_regions)
        self._tensor_values = tuple(tensor_values)
        self._object_states = tuple(object_states)
        self._capture_rng = capture_rng
        self._capture_cuda_rng = capture_cuda_rng
        self._active_transaction: Optional[RuntimeStateTransaction] = None

    @property
    def is_active(self) -> bool:
        return self._active_transaction is not None

    def begin(self) -> "RuntimeStateTransaction":
        """Capture state and open one transaction.

        The method captures before marking the manager active. If any capture
        component fails, no transaction remains active.
        """

        if self._active_transaction is not None:
            raise RuntimeError("runtime state transaction is already active.")
        snapshot = RuntimeStateSnapshot.capture(
            tensor_regions=self._tensor_regions,
            tensor_values=self._tensor_values,
            object_states=self._object_states,
            capture_rng=self._capture_rng,
            capture_cuda_rng=self._capture_cuda_rng,
        )
        transaction = RuntimeStateTransaction(manager=self, snapshot=snapshot)
        self._active_transaction = transaction
        return transaction

    def transaction(self) -> "RuntimeStateTransaction":
        """Open a context-manager transaction."""

        return self.begin()

    def _close(self, transaction: "RuntimeStateTransaction") -> None:
        if self._active_transaction is not transaction:
            raise RuntimeError("transaction does not belong to this manager or is not active.")
        self._active_transaction = None

    def _force_close(self, transaction: "RuntimeStateTransaction") -> None:
        if self._active_transaction is transaction:
            self._active_transaction = None


class RuntimeStateTransaction:
    """One runtime state transaction.

    `complete()` closes the transaction and keeps modifications. `rollback()`
    restores the captured snapshot and closes the transaction. Context-manager
    exit rolls back unless `complete()` has already been called.
    """

    def __init__(self, *, manager: RuntimeStateTransactionManager, snapshot: RuntimeStateSnapshot) -> None:
        self._manager = manager
        self._snapshot = snapshot
        self._closed = False
        self._entered = False

    @property
    def is_closed(self) -> bool:
        return self._closed

    def complete(self) -> None:
        """Keep modifications and close this transaction."""

        self._require_open()
        self._closed = True
        self._manager._close(self)

    def rollback(self) -> None:
        """Restore the captured snapshot and close this transaction."""

        self._require_open()
        try:
            self._snapshot.restore()
        finally:
            self._closed = True
            self._manager._force_close(self)

    def __enter__(self) -> "RuntimeStateTransaction":
        self._require_open()
        if self._entered:
            raise RuntimeError("runtime state transaction context has already been entered.")
        self._entered = True
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        if not self._closed:
            try:
                self.rollback()
            except RuntimeStateRestoreError as restore_error:
                if exc is not None:
                    raise RuntimeStateRollbackError(exc, restore_error) from exc
                raise
        return False

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("runtime state transaction is already closed.")
