# MCP Adapter Runtime State

This document freezes Milestone 2A: the generic runtime state transaction layer
used by future Self-Forcing MCP adapters.

## Scope

`speculative/adapters/runtime_state.py` solves one problem: protect caller-owned
mutable state while proposal, fallback, or commit code performs temporary work.
It can snapshot and restore:

- explicit tensor regions;
- full small tensor values, such as `global_end_index` and `local_end_index`;
- Python object state through getter/setter pairs;
- torch CPU RNG, and CUDA RNG only when requested on a CUDA-capable runtime.

It does not call Wan, MCP, ImageReward, checkpoints, VAE, or any real generator.

## Non-Goals

The runtime state layer does not compute Wan touched ranges. It does not know
which K/V slices a model forward will write, whether attention will append,
overwrite, or roll local cache content, or which output block is being committed.
Those decisions belong to the future `SelfForcingMCPRuntime` in Milestone 2B.

The layer also does not clone a whole real KV cache. Callers must provide exact
regions that need protection.

It also does not reason about different tensor views sharing storage. Conflict
checks use tensor object identity. If two distinct tensor objects alias the same
storage, the future runtime must avoid conflicting specs itself.

## Tensor Safety

Tensor snapshots treat source tensors as borrowed objects. Restore never
replaces caller-held tensors; it uses in-place `copy_`.

Every tensor backup is cloned at capture. Backup storage is private; exposed
properties return clones, so mutating an exposed value cannot corrupt rollback.

Tensor region and full tensor snapshots record this source metadata at capture
and validate it before restore:

- full source shape;
- dtype;
- device;
- layout;
- stride.

Metadata changes fail fast. This prevents `copy_` from silently converting dtype
or device. Region snapshots also re-check that their declared region is still
valid in the current full tensor shape. Region capture also validates bounds
immediately before slicing, so a tensor resized after spec creation cannot be
silently truncated by Python slicing.

## Object State Contract

`ObjectStateSpec` is only for small Python bookkeeping such as commit order,
cursor-like metadata, mappings, lists, and sets.

Do not use it for:

- tensors or containers containing tensors;
- Transformer KV cache objects;
- models, schedulers, generators, or pipelines;
- CUDA streams or events.

`copy_fn` is a capture-time transformation. Capture reads `source = getter()`,
runs `transformed = copy_fn(source)`, rejects tensors in both values, then stores
an independent `deepcopy(transformed)` as the internal backup. The default
`copy_fn` is `deepcopy`, intended only for small Python state.

The transformed value must be small, tensor-free, and deepcopy-able. `getter`
and `copy_fn` must be side-effect-free and must not consume RNG. `setter` must
restore only the declared transformed object state and must not modify state
managed by tensor snapshots.

Tensor-backed object state is rejected at capture. Common containers
(`dict`, `list`, `tuple`, `set`, `frozenset`) are checked recursively with cycle
protection.

`ObjectStateSnapshot.value` returns `deepcopy(internal_backup)`. Restore passes
`deepcopy(internal_backup)` to the setter. A custom `copy_fn` is never called
while reading the backup or restoring it.

## Spec Conflict Rules

`RuntimeStateTransactionManager` and `RuntimeStateSnapshot.capture()` reject
ambiguous tensor specs through the same validation rules before capture:

- the same tensor object cannot have both a full snapshot and region snapshot;
- the same tensor object cannot be full-snapshotted twice;
- region snapshots for the same tensor object cannot overlap;
- non-empty regions for the same tensor object across different dimensions are
  considered conflicting;
- disjoint regions on the same dimension are allowed.

## Transaction Use

The same primitives support two transaction types:

- adapter-local transactions used by proposal and fallback, which must return
  without permanent model/cache/output/RNG changes;
- committer window transactions used by `Committer.begin()` and
  `Committer.rollback()`, which protect permanent state while the controller
  commits a window.

The transaction layer does not decide which type it is running. The caller
chooses the tensor regions, tensor values, object states, and RNG state to
capture.

## Lifecycle

`RuntimeStateTransactionManager.begin()` captures the configured state and
returns a `RuntimeStateTransaction`.

- `complete()` closes the transaction and keeps modifications.
- `rollback()` restores the snapshot and closes the transaction.
- Context-manager exception exit rolls back.
- Context-manager normal exit also rolls back unless `complete()` was called.
- A closed transaction rejects any later `complete()` or `rollback()`.
- The same transaction context can be entered only once.
- A transaction opened with `begin()` can still be completed or rolled back
  directly without entering it as a context manager.
- A manager allows only one active transaction at a time.
- `begin()` is exception-atomic with respect to manager state: capture failure
  leaves no active transaction behind.

## Capture And Restore Order

`RuntimeStateSnapshot.restore()` runs in this order:

1. tensor regions;
2. full tensor values;
3. Python object states;
4. torch RNG state.

Tensor data is restored before metadata. RNG is restored last so restore
bookkeeping cannot perturb the final random sequence.

Restore is best-effort. If one restore step fails, later steps are still
attempted in the same order, including RNG last. After all attempts finish,
`RuntimeStateRestoreError` is raised with every captured restore exception.

If a context body raises and automatic rollback also fails, the transaction
raises `RuntimeStateRollbackError`. It preserves both the original body
exception and the restore exception. The transaction and manager are closed even
after restore failure.

## Future Runtime Example

Milestone 2B should wrap this layer inside `SelfForcingMCPRuntime`; protocol
wrappers should not call these low-level specs directly unless the runtime owns
the state selection.

Example shape:

```python
manager = RuntimeStateTransactionManager(
    tensor_regions=[
        TensorRegionSpec(k_cache, dim=1, start=local_start, end=local_end),
        TensorRegionSpec(v_cache, dim=1, start=local_start, end=local_end),
        TensorRegionSpec(output, dim=1, start=start_frame, end=end_frame),
    ],
    tensor_values=[global_end_index, local_end_index],
    object_states=[
        ObjectStateSpec(
            getter=lambda: runtime.commit_order_state,
            setter=runtime.set_commit_order_state,
        ),
    ],
    capture_rng=True,
    capture_cuda_rng=runtime.capture_cuda_rng,
)

with manager.transaction() as tx:
    runtime.run_temporary_forward(...)
    tx.complete()  # only when the temporary mutation should be kept
```

Proposal and fallback would omit `complete()` so temporary changes roll back.
Committer transactions would call `complete()` only after the controller window
finishes successfully.

## Memory Principle

Snapshot only touched regions. For Wan KV, the future runtime should snapshot
the visible or written K/V slices that the next operation can mutate. Full
allocated K/V cloning is reserved for diagnostics, not normal adapter runtime.
