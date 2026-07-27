from __future__ import annotations

import hashlib
import threading
import weakref
from dataclasses import dataclass
from typing import Mapping, Sequence

import torch


_ACTIVE_OWNER_LOCK = threading.RLock()
_MISSING = object()
_IDENTITY_KEYS = {"wrapper_object_id", "model_object_id", "object_id", "list_object_id", "data_ptr"}
_NUMERIC_KEYS = {"digest", "finite"}


class WanVAECacheRestoreError(RuntimeError):
    """Raised after best-effort Wan VAE cache restore observes failures."""

    def __init__(self, errors: Sequence[Exception]) -> None:
        self.errors = tuple(errors)
        summary = ", ".join(type(error).__name__ for error in self.errors)
        super().__init__(f"Wan VAE cache restore failed with {len(self.errors)} error(s): {summary}")


class WanVAECacheRollbackError(RuntimeError):
    """Raised when context-body failure is followed by rollback failure."""

    def __init__(self, original_exception: BaseException, restore_exception: Exception) -> None:
        self.original_exception = original_exception
        self.restore_exception = restore_exception
        super().__init__(
            "Wan VAE cache rollback failed after "
            f"{type(original_exception).__name__}: {original_exception}; "
            "restore raised "
            f"{type(restore_exception).__name__}: {restore_exception}"
        )


def _is_strict_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _get_model(wrapper: object) -> object:
    if not hasattr(wrapper, "model"):
        raise TypeError("wrapper.model is required for Wan VAE cache transaction.")
    model = getattr(wrapper, "model")
    if model is None:
        raise TypeError("wrapper.model must not be None.")
    return model


def _require_list_attr(model: object, attr_name: str) -> list[object]:
    if not hasattr(model, attr_name):
        raise TypeError(f"model.{attr_name} is required for Wan VAE cache transaction.")
    value = getattr(model, attr_name)
    if not isinstance(value, list):
        raise TypeError(f"model.{attr_name} must be list, got {type(value).__name__}.")
    return value


def _validate_conv_idx(value: list[object], attr_name: str = "_conv_idx") -> None:
    if not value:
        raise ValueError(f"model.{attr_name} must contain at least one integer scratch index.")
    for index, entry in enumerate(value):
        if not _is_strict_int(entry):
            raise TypeError(
                f"model.{attr_name}[{index}] must be int scratch index, "
                f"got {type(entry).__name__}."
            )


def _validate_cache_entries(value: list[object], attr_name: str = "_feat_map") -> None:
    for index, entry in enumerate(value):
        if entry is None or isinstance(entry, str) or isinstance(entry, torch.Tensor):
            continue
        raise TypeError(
            f"model.{attr_name}[{index}] has unsupported cache entry type "
            f"{type(entry).__name__}; expected None, Tensor, or str sentinel."
        )


def _validate_wrapper_for_snapshot(wrapper: object) -> tuple[object, list[object], list[object]]:
    model = _get_model(wrapper)
    conv_idx = _require_list_attr(model, "_conv_idx")
    feat_map = _require_list_attr(model, "_feat_map")
    _validate_conv_idx(conv_idx)
    _validate_cache_entries(feat_map)
    return model, conv_idx, feat_map


def _replace_list_contents(target: list[object], values: Sequence[object]) -> None:
    target[:] = list(values)


@dataclass(frozen=True)
class _ActiveOwnerRecord:
    model_id: int
    model_ref: weakref.ReferenceType[object]
    transaction_ref: weakref.ReferenceType["WanVAECacheTransaction"]


_ACTIVE_OWNERS: dict[int, _ActiveOwnerRecord] = {}


def _model_weakref(model: object) -> weakref.ReferenceType[object]:
    try:
        return weakref.ref(model)
    except TypeError as error:
        raise TypeError("wrapper.model must support weak references for Wan VAE cache ownership.") from error


def _check_owner_available_locked(model_id: int) -> None:
    record = _ACTIVE_OWNERS.get(model_id)
    if record is None:
        return

    model = record.model_ref()
    if model is None:
        _ACTIVE_OWNERS.pop(model_id, None)
        return

    owner = record.transaction_ref()
    if owner is not None:
        raise RuntimeError("Wan VAE cache transaction is already active for this model.")

    raise RuntimeError(
        "Wan VAE cache transaction was abandoned/poisoned for this live model; "
        "explicit complete() or rollback() is required before reuse."
    )


@dataclass(frozen=True)
class WanVAECacheSnapshot:
    """Shallow snapshot of the Wan VAE cached-decode Python cache state."""

    wrapper: object
    model: object
    conv_idx_list: list[object]
    conv_idx_contents: tuple[int, ...]
    feat_map_list: list[object]
    feat_map_entries: tuple[object, ...]

    @classmethod
    def capture(cls, wrapper: object) -> "WanVAECacheSnapshot":
        model, conv_idx, feat_map = _validate_wrapper_for_snapshot(wrapper)
        return cls(
            wrapper=wrapper,
            model=model,
            conv_idx_list=conv_idx,
            conv_idx_contents=tuple(int(entry) for entry in conv_idx),
            feat_map_list=feat_map,
            feat_map_entries=tuple(feat_map),
        )

    def restore(self) -> None:
        current_model = getattr(self.wrapper, "model", _MISSING)
        if current_model is not self.model:
            raise WanVAECacheRestoreError([
                RuntimeError("wrapper.model binding changed; refusing to restore Wan VAE cache snapshot.")
            ])

        errors: list[Exception] = []
        try:
            _replace_list_contents(self.conv_idx_list, self.conv_idx_contents)
        except Exception as error:
            errors.append(error)
        try:
            setattr(self.model, "_conv_idx", self.conv_idx_list)
        except Exception as error:
            errors.append(error)
        try:
            _replace_list_contents(self.feat_map_list, self.feat_map_entries)
        except Exception as error:
            errors.append(error)
        try:
            setattr(self.model, "_feat_map", self.feat_map_list)
        except Exception as error:
            errors.append(error)

        if errors:
            raise WanVAECacheRestoreError(errors)


def capture_wan_vae_cache_snapshot(wrapper: object) -> WanVAECacheSnapshot:
    return WanVAECacheSnapshot.capture(wrapper)


class WanVAECacheTransaction:
    """Single-model Wan VAE cache transaction for temporary cached decode work."""

    NEW = "new"
    ACTIVE = "active"
    COMPLETED = "completed"
    ROLLED_BACK = "rolled_back"
    FAILED = "failed"

    def __init__(self, wrapper: object) -> None:
        self.wrapper = wrapper
        self._state = self.NEW
        self._snapshot: WanVAECacheSnapshot | None = None
        self._model_id: int | None = None

    @property
    def state(self) -> str:
        return self._state

    @property
    def is_active(self) -> bool:
        return self._state == self.ACTIVE

    def begin(self) -> "WanVAECacheTransaction":
        if self._state != self.NEW:
            raise RuntimeError(f"Wan VAE cache transaction cannot begin from state {self._state!r}.")

        with _ACTIVE_OWNER_LOCK:
            model = _get_model(self.wrapper)
            model_id = id(model)
            _check_owner_available_locked(model_id)

            snapshot = WanVAECacheSnapshot.capture(self.wrapper)
            snapshot_model_id = id(snapshot.model)
            if snapshot_model_id != model_id:
                _check_owner_available_locked(snapshot_model_id)

            self._snapshot = snapshot
            self._model_id = snapshot_model_id
            _ACTIVE_OWNERS[snapshot_model_id] = _ActiveOwnerRecord(
                model_id=snapshot_model_id,
                model_ref=_model_weakref(snapshot.model),
                transaction_ref=weakref.ref(self),
            )
            self._state = self.ACTIVE
        return self

    def complete(self) -> None:
        self._require_active("complete")
        self._state = self.COMPLETED
        self._snapshot = None
        self._release_owner()

    def rollback(self) -> None:
        self._require_active("rollback")
        snapshot = self._snapshot
        assert snapshot is not None
        try:
            snapshot.restore()
        except WanVAECacheRestoreError:
            self._state = self.FAILED
            raise
        except Exception as error:
            self._state = self.FAILED
            raise WanVAECacheRestoreError([error]) from error
        else:
            self._state = self.ROLLED_BACK
        finally:
            self._snapshot = None
            self._release_owner()

    def __enter__(self) -> "WanVAECacheTransaction":
        return self.begin()

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        if self._state == self.ACTIVE:
            try:
                self.rollback()
            except WanVAECacheRestoreError as restore_error:
                if exc is not None:
                    raise WanVAECacheRollbackError(exc, restore_error) from exc
                raise
        return False

    def _require_active(self, operation: str) -> None:
        if self._state != self.ACTIVE:
            raise RuntimeError(
                f"Wan VAE cache transaction cannot {operation} from state {self._state!r}."
            )

    def _release_owner(self) -> None:
        model_id = self._model_id
        if model_id is None:
            return
        with _ACTIVE_OWNER_LOCK:
            record = _ACTIVE_OWNERS.get(model_id)
            owner = record.transaction_ref() if record is not None else None
            if owner is self:
                _ACTIVE_OWNERS.pop(model_id, None)
        self._model_id = None


def _tensor_finite(tensor: torch.Tensor, attr_name: str, index: int) -> bool:
    try:
        return bool(torch.isfinite(tensor.detach()).all().item())
    except Exception as error:
        raise RuntimeError(
            f"failed to compute finite status for model.{attr_name}[{index}]: {error}"
        ) from error


def _tensor_digest(tensor: torch.Tensor, attr_name: str, index: int) -> str:
    try:
        value = tensor.detach().contiguous().cpu()
        digest_value = value.to(torch.float32) if value.dtype is torch.bfloat16 else value
        data = digest_value.numpy().tobytes()
    except Exception as error:
        raise RuntimeError(
            f"failed to digest tensor cache entry model.{attr_name}[{index}]: {error}"
        ) from error

    digest = hashlib.sha256()
    header = (
        f"shape={tuple(tensor.shape)};"
        f"dtype={tensor.dtype};"
        f"digest_dtype={digest_value.dtype};"
    )
    digest.update(header.encode("utf-8"))
    digest.update(data)
    return digest.hexdigest()


def _fingerprint_tensor_entry(
    tensor: torch.Tensor,
    *,
    attr_name: str,
    index: int,
    include_digest: bool,
) -> dict[str, object]:
    record: dict[str, object] = {
        "index": int(index),
        "kind": "tensor",
        "entry_type": type(tensor).__name__,
        "object_id": int(id(tensor)),
        "data_ptr": int(tensor.data_ptr()),
        "shape": [int(dim) for dim in tensor.shape],
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "stride": [int(dim) for dim in tensor.stride()],
        "storage_offset": int(tensor.storage_offset()),
        "requires_grad": bool(tensor.requires_grad),
        "finite": _tensor_finite(tensor, attr_name, index),
    }
    if include_digest:
        record["digest"] = _tensor_digest(tensor, attr_name, index)
    return record


def _fingerprint_list_entry(
    entry: object,
    *,
    attr_name: str,
    index: int,
    include_digest: bool,
) -> dict[str, object]:
    if entry is None:
        return {
            "index": int(index),
            "kind": "none",
            "entry_type": "NoneType",
            "object_id": int(id(entry)),
        }
    if isinstance(entry, str):
        return {
            "index": int(index),
            "kind": "sentinel",
            "entry_type": "str",
            "object_id": int(id(entry)),
            "value": entry,
        }
    if isinstance(entry, torch.Tensor):
        return _fingerprint_tensor_entry(
            entry,
            attr_name=attr_name,
            index=index,
            include_digest=include_digest,
        )
    if _is_strict_int(entry):
        return {
            "index": int(index),
            "kind": "int",
            "entry_type": "int",
            "object_id": int(id(entry)),
            "value": int(entry),
        }
    raise TypeError(
        f"model.{attr_name}[{index}] has unsupported fingerprint entry type "
        f"{type(entry).__name__}."
    )


def _fingerprint_list_attr(
    model: object,
    attr_name: str,
    *,
    include_digest: bool,
    required: bool,
    conv_idx: bool = False,
) -> dict[str, object]:
    if not hasattr(model, attr_name):
        if required:
            raise TypeError(f"model.{attr_name} is required for Wan VAE cache fingerprint.")
        return {"attribute": attr_name, "exists": False}

    value = getattr(model, attr_name)
    if not isinstance(value, list):
        raise TypeError(f"model.{attr_name} must be list, got {type(value).__name__}.")
    if conv_idx:
        _validate_conv_idx(value, attr_name)
    else:
        _validate_cache_entries(value, attr_name)

    return {
        "attribute": attr_name,
        "exists": True,
        "type": "list",
        "list_object_id": int(id(value)),
        "length": int(len(value)),
        "entries": [
            _fingerprint_list_entry(
                entry,
                attr_name=attr_name,
                index=index,
                include_digest=include_digest,
            )
            for index, entry in enumerate(value)
        ],
    }


def _fingerprint_scalar_attr(model: object, attr_name: str) -> dict[str, object]:
    if not hasattr(model, attr_name):
        return {"attribute": attr_name, "exists": False}
    value = getattr(model, attr_name)
    record: dict[str, object] = {
        "attribute": attr_name,
        "exists": True,
        "type": type(value).__name__,
        "object_id": int(id(value)),
    }
    if value is None or isinstance(value, (str, int, bool)):
        record["value"] = value
    else:
        record["value_repr"] = repr(value)
    return record


def fingerprint_wan_vae_cache(wrapper: object, *, include_digest: bool = False) -> dict[str, object]:
    """Return a JSON-safe, read-only fingerprint of Wan VAE cache attributes."""

    model = _get_model(wrapper)
    attributes = {
        "_conv_num": _fingerprint_scalar_attr(model, "_conv_num"),
        "_conv_idx": _fingerprint_list_attr(
            model, "_conv_idx", include_digest=include_digest, required=True, conv_idx=True
        ),
        "_feat_map": _fingerprint_list_attr(
            model, "_feat_map", include_digest=include_digest, required=True
        ),
        "_enc_conv_num": _fingerprint_scalar_attr(model, "_enc_conv_num"),
        "_enc_conv_idx": _fingerprint_list_attr(
            model, "_enc_conv_idx", include_digest=include_digest, required=False, conv_idx=True
        ),
        "_enc_feat_map": _fingerprint_list_attr(
            model, "_enc_feat_map", include_digest=include_digest, required=False
        ),
    }
    return {
        "wrapper_object_id": int(id(wrapper)),
        "model_object_id": int(id(model)),
        "model_attributes": attributes,
    }


def _strip_keys(value: object, keys: set[str]) -> object:
    if isinstance(value, Mapping):
        return {
            str(key): _strip_keys(item, keys)
            for key, item in value.items()
            if key not in keys
        }
    if isinstance(value, list):
        return [_strip_keys(item, keys) for item in value]
    return value


def _all_tensor_entries_have_digest(value: object) -> bool:
    if isinstance(value, Mapping):
        if value.get("kind") == "tensor" and "digest" not in value:
            return False
        return all(_all_tensor_entries_have_digest(item) for item in value.values())
    if isinstance(value, list):
        return all(_all_tensor_entries_have_digest(item) for item in value)
    return True


def fingerprints_structurally_equal(left: Mapping[str, object], right: Mapping[str, object]) -> bool:
    """Compare cache structure while ignoring identity and numerical digest fields."""

    ignored = _IDENTITY_KEYS | _NUMERIC_KEYS
    return _strip_keys(left, ignored) == _strip_keys(right, ignored)


def fingerprints_numerically_equal(left: Mapping[str, object], right: Mapping[str, object]) -> bool:
    """Compare cache numerics using tensor digests while ignoring identity fields."""

    if not _all_tensor_entries_have_digest(left) or not _all_tensor_entries_have_digest(right):
        return False
    return _strip_keys(left, _IDENTITY_KEYS) == _strip_keys(right, _IDENTITY_KEYS)


def fingerprints_identity_equal(left: Mapping[str, object], right: Mapping[str, object]) -> bool:
    """Compare cache identity while ignoring tensor digest and finite value state."""

    return _strip_keys(left, _NUMERIC_KEYS) == _strip_keys(right, _NUMERIC_KEYS)
