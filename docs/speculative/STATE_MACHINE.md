# Speculative Controller State Machine

## Window Inputs

A speculative window starts from a `ControlRequest`. The `ProposalSource`
returns one anchor `CommitRequest` and zero or more ordered `DraftCandidate`
objects. Draft depth is one-based: depth 1 is the first block after the anchor.

Each `DraftCandidate` must keep its `source_noise`. If a draft is rejected, the
fallback generator receives that same candidate and must return a fallback result
with the same `source_noise` object.

## States

1. `proposal_requested`
2. `proposal_ready`
3. `transaction_begin`
4. `commit(anchor)`
5. For each draft in increasing depth:
   - `evaluate`
   - `evaluated`
   - `decision`
   - if accepted: `commit(draft)`
   - if rejected:
     - `invalidated` for every deeper draft in this window
     - `fallback_requested`
     - `fallback_ready`
     - `commit(fallback)`
     - stop evaluating this window
6. `transaction_complete`

If any step after transaction begin raises, the state machine emits `error`,
calls rollback, and emits `transaction_rollback`. If rollback raises, the
controller raises a rollback failure exception containing both the original
exception and the rollback exception.

If `transaction_begin` is emitted but `committer.begin()` raises, the committer
must have left no permanent state behind. The controller treats that as a
non-started transaction and does not call rollback.

## Invariants

- A policy returns only a `Decision`; it never mutates proposal, KV, fallback, or
  controller state.
- Proposal, evaluation, and fallback adapters must snapshot/restore any
  temporary model or cache forwards before returning.
- `Committer` is the only permanent owner of Transformer KV, output buffers, and
  generation cursor updates.
- The controller commits only the longest accepted prefix.
- The first reject invalidates all deeper drafts in the current window.
- Deeper drafts are not evaluated after the first reject.
- A block index can be committed at most once per controller lifetime.
- Commit order is strictly increasing by block index.
- Commit order is validated across speculative windows using the full previous
  `BlockRef`, including frame continuity when frame ranges are present.
- A fallback commit uses the rejected draft block and the rejected draft
  `source_noise`.

## Out-of-Order Proposals

Drafts must arrive as a contiguous depth sequence beginning at 1, and each draft
block index must equal `anchor_block.index + depth`. A malformed proposal fails
before any commit.

When `start_frame` and `num_frames` are provided, they must be provided for the
anchor and every draft, and the frame ranges must be continuous. A proposal with
correct block indexes but a skipped `start_frame` is invalid.

Payloads such as latent tensors and `source_noise` are read-only borrowed
references. The controller may carry them between adapters, but adapters must
not mutate them in place.
