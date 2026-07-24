# Speculative Control Core Architecture

## Scope

This package contains only the control layer for speculative decoding. It does
not load Wan, ImageReward, checkpoints, videos, schedulers, or training code.

The current implementation is intentionally model-agnostic:

- proposal is supplied through a `ProposalSource` protocol;
- draft quality is supplied through an `Evaluator` protocol;
- accept/reject behavior is supplied through a `Policy` protocol;
- rejected-block recovery is supplied through a `FallbackGenerator` protocol;
- KV/cache mutation is hidden behind a `Committer` protocol.

## Real MCP Boundary

The validated MCP reference path is still `inference_mcp.py` at
`dev@d1dde97`. That file is not modified by this package.

The existing always-accept path currently lives inside
`SelfForcingTrainingPipeline._inference_with_trajectory_mcp_accelerated`:

1. the backbone denoises an anchor block;
2. MCP heads draft future blocks from their source noise;
3. anchor and accepted drafts are committed by re-running the generator with
   `kv_cache`;
4. the final latent is decoded by VAE outside the speculative loop.

This package extracts the decision and commit ordering rules without binding to
those concrete calls. Policy is pure decision logic; the controller is
model-agnostic orchestration.

## Modules

- `speculative.types`: dataclasses for blocks, drafts, decisions, fallback
  outputs, commit requests, and controller results.
- `speculative.interfaces`: protocols for proposal, evaluation, policy,
  fallback, and commit boundaries.
- `speculative.controller`: stateful controller enforcing ordered commit,
  longest accepted prefix, first-reject invalidation, fallback, and trace.
- `speculative.trace`: stable trace event schema and recorder.
- `speculative.policies.scripted`: deterministic scripted policies used for
  tests and early integration.
- `speculative.factory`: explicit policy factory map.

## Transaction Model

The controller calls `committer.begin()` before the first commit in a window and
`committer.complete()` after all commits succeed. If any evaluation, policy,
fallback, or commit step raises, the controller restores its own bookkeeping and
calls `committer.rollback()`. If rollback itself fails, the controller raises a
rollback-specific exception that includes both the original error and the
rollback error.

`committer.begin()` must be exception-atomic. If it raises, it must not leave
permanent KV, output, or cursor changes behind. The controller does not call
`rollback()` when `begin()` has not completed successfully.

The real generator/KV integration should implement `Committer` with a KV
snapshot and restore operation. The model-agnostic core deliberately does not
know how the snapshot is stored.

## State Ownership

`ProposalSource`, `Evaluator`, `Policy`, and `FallbackGenerator` must return
without retaining permanent generation-state mutations. If an adapter needs a
temporary model/cache forward, it owns an adapter-local snapshot/restore before
returning to the controller.

`Committer` is the only component allowed to permanently mutate Transformer KV,
output storage, and the generation cursor. Its transaction is the controller
window transaction. This is separate from adapter-local transactions used by
proposal, evaluation, or fallback implementations.

The controller keeps the full last committed `BlockRef` across windows. Every
commit must advance by exactly one block index. If either adjacent block uses a
frame range, both must provide `start_frame` and `num_frames`, and the new
`start_frame` must equal `previous.start_frame + previous.num_frames`.

Latent tensors, source noise, scores, and other payloads passed through core
dataclasses are read-only borrowed references from the controller's perspective.
Adapters and policies must not modify those payloads in place.

Trace metadata is copied on emit and normalized to JSON-safe values. Arbitrary
objects, tensors, non-string mapping keys, and non-finite floats are rejected.
Internally, trace metadata is stored as recursive read-only mappings and tuples;
`to_dict()` returns fresh mutable dict/list containers for serialization.

## VAE Boundary

VAE decode is outside this control core. The current validated path decodes the
complete latent with `use_cache=False`. Wan VAE has a `cached_decode()` path, but
its cache has mutable temporal state and no controller-level rollback semantics.
Do not couple speculative block decisions to VAE cache until that cache has a
tested transaction story.
