from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from speculative.adapters.wan_vae_transaction import (
    WanVAECacheSnapshot,
    WanVAECacheTransaction,
)
from speculative.evaluation import DecodedCandidate
from speculative.scoring import freeze_metadata
from speculative.types import DraftCandidate


class WanVAEWindowError(RuntimeError):
    """Raised when a Wan VAE window lifecycle invariant is violated."""


class WanVAEPreviewRollbackError(WanVAEWindowError):
    """Raised when preview decode fails and preview cache restore also fails."""

    def __init__(
        self,
        original_exception: BaseException,
        restore_exception: Exception,
    ) -> None:
        self.original_exception = original_exception
        self.restore_exception = restore_exception
        super().__init__(
            "Wan VAE preview rollback failed after "
            f"{type(original_exception).__name__}: {original_exception}; "
            "restore raised "
            f"{type(restore_exception).__name__}: {restore_exception}"
        )


class WanVAEWindowRollbackError(WanVAEWindowError):
    """Raised when context-body failure is followed by window rollback failure."""

    def __init__(
        self,
        original_exception: BaseException,
        rollback_exception: Exception,
    ) -> None:
        self.original_exception = original_exception
        self.rollback_exception = rollback_exception
        super().__init__(
            "Wan VAE window rollback failed after "
            f"{type(original_exception).__name__}: {original_exception}; "
            "rollback raised "
            f"{type(rollback_exception).__name__}: {rollback_exception}"
        )


class WanVAEWindowCoordinator:
    """Coordinates one cached Wan VAE decode window around an outer transaction."""

    NEW = "NEW"
    ACTIVE = "ACTIVE"
    FAILED = "FAILED"
    COMPLETED = "COMPLETED"
    ROLLED_BACK = "ROLLED_BACK"
    POISONED = "POISONED"

    def __init__(self, wrapper: object) -> None:
        self.wrapper = wrapper
        self._state = self.NEW
        self._transaction: WanVAECacheTransaction | None = None
        self._preview_active = False
        self._validate_wrapper()

    @property
    def state(self) -> str:
        return self._state

    @property
    def is_active(self) -> bool:
        return self._state == self.ACTIVE

    @property
    def preview_active(self) -> bool:
        return self._preview_active

    def begin_window(self) -> "WanVAEWindowCoordinator":
        if self._state != self.NEW:
            raise WanVAEWindowError(
                f"Wan VAE window cannot begin from state {self._state!r}."
            )

        transaction = WanVAECacheTransaction(self.wrapper)
        transaction.begin()
        self._transaction = transaction
        self._state = self.ACTIVE
        return self

    def preview_latent(self, latent: Any) -> Any:
        self._require_active_without_preview("preview")

        snapshot = WanVAECacheSnapshot.capture(self.wrapper)
        self._preview_active = True
        try:
            try:
                pixels = self.wrapper.decode_to_pixel(latent, use_cache=True)  # type: ignore[attr-defined]
            except BaseException as decode_error:
                try:
                    snapshot.restore()
                except Exception as restore_error:
                    self._state = self.FAILED
                    raise WanVAEPreviewRollbackError(
                        decode_error,
                        restore_error,
                    ) from decode_error
                raise

            try:
                snapshot.restore()
            except Exception:
                self._state = self.FAILED
                raise
            return pixels
        finally:
            self._preview_active = False

    def commit_latent(self, latent: Any) -> Any:
        self._require_active_without_preview("commit")

        try:
            return self.wrapper.decode_to_pixel(latent, use_cache=True)  # type: ignore[attr-defined]
        except BaseException:
            self._state = self.FAILED
            raise

    def complete_window(self) -> None:
        self._require_active_without_preview("complete")
        transaction = self._require_transaction("complete")

        transaction.complete()
        self._state = self.COMPLETED
        self._transaction = None

    def rollback_window(self) -> None:
        if self._preview_active:
            raise WanVAEWindowError("Wan VAE window cannot rollback while preview active.")
        if self._state not in (self.ACTIVE, self.FAILED):
            raise WanVAEWindowError(
                f"Wan VAE window cannot rollback from state {self._state!r}."
            )

        transaction = self._transaction
        try:
            if transaction is not None and transaction.is_active:
                transaction.rollback()
        except Exception:
            self._state = self.POISONED
            self._transaction = None
            raise

        self._state = self.ROLLED_BACK
        self._transaction = None

    def __enter__(self) -> "WanVAEWindowCoordinator":
        return self.begin_window()

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        if self._state in (self.ACTIVE, self.FAILED):
            try:
                self.rollback_window()
            except Exception as rollback_error:
                if exc is not None:
                    raise WanVAEWindowRollbackError(exc, rollback_error) from exc
                raise
        return False

    def _validate_wrapper(self) -> None:
        if not hasattr(self.wrapper, "model"):
            raise TypeError("wrapper.model is required for Wan VAE window coordination.")
        if getattr(self.wrapper, "model") is None:
            raise TypeError("wrapper.model must not be None.")
        decode = getattr(self.wrapper, "decode_to_pixel", None)
        if not callable(decode):
            raise TypeError("wrapper.decode_to_pixel(latent, use_cache=True) is required.")

    def _require_active_without_preview(self, operation: str) -> None:
        if self._state != self.ACTIVE:
            raise WanVAEWindowError(
                f"Wan VAE window cannot {operation} from state {self._state!r}."
            )
        if self._preview_active:
            raise WanVAEWindowError(
                f"Wan VAE window cannot {operation} while preview active."
            )

    def _require_transaction(self, operation: str) -> WanVAECacheTransaction:
        if self._transaction is None:
            raise WanVAEWindowError(
                f"Wan VAE window cannot {operation} without an active transaction."
            )
        return self._transaction


@dataclass(frozen=True)
class WanVAECandidateDecoder:
    """CandidateDecoder adapter backed by temporary Wan VAE cached preview decode."""

    coordinator: WanVAEWindowCoordinator
    decoder_name: str = "wan_vae_cached_preview"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.decoder_name, str) or not self.decoder_name:
            raise ValueError("decoder_name must be a non-empty string.")
        object.__setattr__(self, "metadata", freeze_metadata(self.metadata))

    def decode(self, candidate: DraftCandidate) -> DecodedCandidate:
        if not isinstance(candidate, DraftCandidate):
            raise TypeError("candidate must be a DraftCandidate.")

        pixels = self.coordinator.preview_latent(candidate.latent)
        return DecodedCandidate(
            candidate=candidate,
            payload=pixels,
            metadata={
                "decoder_name": self.decoder_name,
                "block_index": candidate.block.index,
                "depth": candidate.depth,
                "decoder_metadata": self.metadata,
            },
        )


__all__ = [
    "WanVAECandidateDecoder",
    "WanVAEPreviewRollbackError",
    "WanVAEWindowCoordinator",
    "WanVAEWindowError",
    "WanVAEWindowRollbackError",
]
