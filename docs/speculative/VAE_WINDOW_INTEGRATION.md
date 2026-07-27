# Wan VAE window integration

Milestone: F4B3A

F4B3A is engineering infrastructure only. It introduces an independent Wan VAE
window coordinator and a `CandidateDecoder` adapter for cached preview decode.
It does not define or validate a verifier algorithm, scorer, aggregator,
accept/reject policy, quality-equivalence claim, or speed claim.

## Window Lifecycle

The new lifecycle is:

```text
begin window
-> commit anchor latent and permanently advance the VAE cache
-> preview draft latent with temporary cached decode
-> restore the preview cache before returning pixels to the evaluator
-> policy decides outside the coordinator
-> commit the accepted draft or fallback latent
-> complete window
```

If an exception requires abandoning the window, `rollback_window()` restores the
VAE cache to the state captured at `begin_window()`.

Preview decode is temporary. `preview_latent()` captures a local
`WanVAECacheSnapshot`, calls `wrapper.decode_to_pixel(latent, use_cache=True)`,
then restores the snapshot before returning the pixel object. The returned
pixels are the scorer input, but the VAE cache no longer contains the preview
draft state when evaluation returns.

Commit decode is permanent inside the active VAE window.
`commit_latent()` calls the same cached decode path and keeps the resulting VAE
cache mutation. The coordinator does not distinguish anchor, draft, and
fallback latents. The caller decides which latent to pass.

`complete_window()` closes the outer `WanVAECacheTransaction` and keeps the
final committed VAE cache state. `rollback_window()` rolls that transaction
back and restores the cache captured at `begin_window()`.

## Controller Ordering

The existing controller order remains unchanged:

```text
evaluator.evaluate(candidate)
policy.decide(evaluation)
committer.commit(...)
```

Because the policy runs after evaluation, `WanVAECandidateDecoder` must restore
the temporary preview state before `evaluate()` returns. If a draft is accepted,
the current correctness-first plan is to decode that latent again in a later
VAE-aware commit path, which will permanently advance the VAE cache.

That duplicate decode is intentional for the F4B3A baseline. It prioritizes
clear correctness and rollback semantics over performance. It is not the final
performance design.

F4B3A does not implement the production composite committer that will perform
accepted-draft or fallback VAE commits.

## Boundaries

The coordinator owns only the VAE cache window. It does not own the Transformer
transaction, Transformer KV cache, output buffers, RNG state, controller cursor,
or runtime window.

F4B3A intentionally does not modify:

- `SpeculativeController`;
- committer protocols;
- runtime orchestration;
- production inference entrypoints;
- Wan VAE implementation code.

F4B3B is the milestone that should compose the Transformer and VAE lifecycles.
That composition still needs an explicit design for cross-resource error
handling and commit ordering.

The verifier, scorer, aggregator, and policy are still independent components.
They are not selected by this coordinator and are not read by it. In
particular, the coordinator has no threshold input and makes no accept/reject
decision.

## Failure Semantics

If preview decode fails, the coordinator still attempts to restore the local
preview snapshot. If restore succeeds, the original decode exception is
re-raised and the window remains active.

If both preview decode and local preview restore fail,
`WanVAEPreviewRollbackError` preserves both exception objects and the
coordinator enters `FAILED`. After that, only `rollback_window()` is allowed.

If commit decode fails after partially mutating the VAE cache, the coordinator
enters `FAILED`. The caller must roll back the window to restore the
`begin_window()` cache state.

If outer rollback fails, the coordinator enters `POISONED` and drops its strong
reference to the transaction. A poisoned rollback means the model cache state is
unknown. The current process must not continue treating that VAE instance as a
clean model state.

`POISONED` is not only local coordinator state. The underlying
`WanVAECacheTransaction` owner registry also keeps the original live model
blocked as abandoned/poisoned. Creating another wrapper or a new
`WanVAEWindowCoordinator` around that same live model must still fail. The safe
recovery path is to rebuild the VAE model; once the poisoned model object dies,
the stale weakref owner record may be cleaned.

There is no `__del__`, weakref callback, finalizer, or production bypass API for
automatic rollback. Active abandoned windows are rejected by the underlying
`WanVAECacheTransaction` owner gate.

## Non-Goals

F4B3A does not:

- implement a real verifier;
- implement ImageReward;
- implement a scoring strategy;
- implement an aggregation strategy;
- implement an accept/reject threshold;
- modify controller, committer, or runtime ordering;
- claim quality equivalence;
- claim speedup;
- load checkpoints;
- require CUDA.
