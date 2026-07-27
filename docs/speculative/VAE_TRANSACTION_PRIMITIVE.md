# Wan VAE transaction primitive

Milestone: F4B1

This document describes the standalone Wan VAE cache snapshot/restore primitive
added for CPU synthetic validation. It is infrastructure only. It is not a
candidate decoder, verifier, scorer, controller integration, production runtime
transaction, or GPU validation.

## Scope

F4B1 proves only this local state operation:

```text
clean committed VAE cache
-> capture snapshot
-> temporary cached-decode-like cache mutation
-> rollback
-> restore the previous cache binding, contents, and entry identity
```

The implementation lives in:

```text
speculative/adapters/wan_vae_transaction.py
```

It accepts a WanVAEWrapper-compatible object by duck typing. It does not import
`WanVAEWrapper`, load a checkpoint, call `decode_to_pixel`, call
`cached_decode`, call `clear_cache`, move devices, change dtype, or change
train/eval state.

## Static State Set

F4A found that pure `WanVAE_.cached_decode()` statically mutates only the decode
cache frontier:

- `_conv_idx` is rebound per latent frame and used as the scratch index list.
- `_feat_map` stays as the cache list object, but entries are replaced in place.

Therefore F4B1 snapshots:

- the `wrapper.model` object reference;
- the original `_conv_idx` list object and a tuple copy of its integer contents;
- the original `_feat_map` list object and a tuple of its entry references.

Rollback restores, in order:

1. original `_conv_idx` list contents;
2. `model._conv_idx` binding to the original list object;
3. original `_feat_map` list entries;
4. `model._feat_map` binding to the original list object.

The optional/fingerprint-only attributes are `_conv_num`, `_enc_conv_num`,
`_enc_conv_idx`, `_enc_feat_map`, and wrapper-level `mean`, `std`, `model`
identity. They are not part of the static minimal pure-`cached_decode` rollback
set unless future code mixes in `decode()`, `encode()`, or `clear_cache()`.

## Shallow Entries

F4B1 intentionally shallow-copies `_feat_map` entries. Existing Tensor entries
are restored with `is` identity, not cloned.

The reason is code evidence from F4A:

- old cache Tensor entries are read;
- new Tensor objects are assigned into `_feat_map` slots;
- no old cache Tensor in-place numerical mutation was found in source.

Cloning old Tensor entries would break identity equality and `data_ptr`
equality. That would make the primitive unable to test whether wrapper-level
rollback preserves the exact pre-call cache identity.

The remaining blocking question is GPU-only:

```text
Does the real CUDA cached-decode path ever mutate an old cache Tensor in place?
```

F4B2 must answer that with A100 Path A / Path B fingerprints. If an old Tensor's
object id and `data_ptr` stay the same but its digest changes during rejected
draft decode, then shallow rollback is insufficient and clone-based numerical
backup must be reconsidered.

F4B2 must also measure CUDA `allocated` and `reserved` memory across repeated
reject/rollback cycles. F4B1 CPU garbage-collection tests only prove that closed
transactions no longer hold Python snapshot references; they do not prove CUDA
allocator behavior for the real VAE path.

## Fingerprint Equality

`fingerprint_wan_vae_cache(wrapper, include_digest=False)` returns only
JSON-safe Python containers and scalars. It records:

- wrapper object id;
- model object id;
- `_conv_num`, if present;
- `_conv_idx`;
- `_feat_map`;
- `_enc_conv_num`, if present;
- `_enc_conv_idx`, if present;
- `_enc_feat_map`, if present.

For list attributes, it records list object id, length, and every entry index,
kind, type, and object id.

Entry kinds are:

- `none`;
- `sentinel` for string cache sentinels such as `'Rep'`;
- `tensor`;
- `int` for `_conv_idx` scratch indices.

Tensor entries record shape, dtype, device, stride, storage offset,
`requires_grad`, finite status, object id, and `data_ptr`. With
`include_digest=True`, the fingerprint also records a deterministic digest from
a detached contiguous CPU representation. The digest path supports float32,
float16, and bfloat16.

The comparison helpers distinguish three questions:

- structural equality ignores object ids, list ids, Tensor ids, `data_ptr`,
  finite state, and digests;
- numerical equality requires Tensor digests to match but ignores object ids and
  `data_ptr`;
- identity equality requires wrapper/model/list/entry identity and Tensor
  `data_ptr` to match, but ignores Tensor digest.

This distinction prevents a new Tensor with identical values from being reported
as a numerical difference, and prevents an in-place Tensor value mutation from
being hidden by unchanged identity.

## Transaction Lifecycle

`WanVAECacheTransaction` has these states:

```text
new -> active -> completed
new -> active -> rolled_back
new -> active -> failed
```

Rules:

- `begin()` captures the snapshot and opens the transaction.
- `complete()` closes the transaction and keeps current cache modifications.
- `rollback()` restores the snapshot and closes the transaction.
- `begin()` may be called only once.
- `complete()` and `rollback()` require `active`.
- closed transactions cannot be completed or rolled back again.
- one model may have only one active transaction at a time.
- different model objects may have simultaneous active transactions.
- active ownership is recorded only after capture succeeds.
- active ownership is released even if rollback fails.

Active ownership is tracked as:

```text
model object id -> active owner record
```

Each active owner record stores the model object id, a weak reference to the
model, and a weak reference to the owning transaction. The registry is protected
by a lock. `begin()` handles existing records while holding the lock:

- live transaction: reject as an already-active transaction for that model;
- dead transaction with live model: reject as abandoned/poisoned state;
- dead model: delete the expired record;
- no valid record: capture the snapshot and register the current transaction.

An abandoned active transaction is a programming error. The primitive does not
infer that the cache is clean merely because the transaction object was garbage
collected; the model may still contain temporary cache mutations. It therefore
fails closed instead of continuing on an unknown cache state.

Only explicit `complete()` or `rollback()` normally deletes the current owner.
Release removes a registry entry only when the current record's transaction
weak reference still resolves to the releasing transaction, so delayed cleanup
from an old transaction cannot delete a newer owner. There is no `__del__`,
weakref callback, or garbage-collection finalizer that attempts automatic
rollback.

Closed transactions immediately release snapshot references. `complete()` drops
`self._snapshot` after closing and keeping current cache state. `rollback()`
keeps the snapshot until restore has been attempted, then drops
`self._snapshot` whether restore succeeds or fails. Both paths clear the stored
model id after releasing active ownership. This matters because the snapshot
contains shallow references to the original `_feat_map` entries; keeping it
alive after close could keep old cache Tensor objects, and on GPU their CUDA
storage, alive longer than intended.

Context-manager semantics:

```python
with WanVAECacheTransaction(wrapper) as tx:
    ...
```

If the body exits normally without `tx.complete()`, rollback is automatic. If
the body raises, rollback is attempted and the original exception is re-raised.
If the body raises and rollback also fails, `WanVAECacheRollbackError` preserves
both the original exception and the restore exception.

Restore is best-effort: each restore step is attempted and all errors are
collected into `WanVAECacheRestoreError`.

## Non-Goals

F4B1 does not:

- implement `WanVAECandidateDecoder`;
- call real Wan VAE decode;
- call generator or MCP heads;
- modify Transformer KV;
- modify `SpeculativeController`, `Evaluator`, `Committer`, or runtime
  transactions;
- decide accept/reject;
- score candidates;
- add ImageReward;
- run GPU;
- load checkpoints;
- claim production-safe VAE transactions;
- claim quality equivalence;
- claim speedup.

The unresolved integration questions remain exactly the F4A gate items:

- evaluator return-time handling of candidate VAE state;
- accept/fallback permanent VAE cache owner;
- whether controller window rollback covers VAE;
- Transformer commit and VAE commit atomicity;
- whether accepted candidates may be decoded twice.
