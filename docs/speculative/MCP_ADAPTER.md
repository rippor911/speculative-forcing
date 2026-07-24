# MCP Adapter

This document records Milestone 2A runtime state primitives, Milestone 2B1 thin
wrapper boundaries, and Milestone 2B2A runtime orchestration for future
Self-Forcing MCP adapters.

## Scope

`speculative/adapters/runtime_state.py` solves one problem: protect caller-owned
mutable state while proposal, fallback, or commit code performs temporary work.
It can snapshot and restore:

- explicit tensor regions;
- full small tensor values, such as `global_end_index` and `local_end_index`;
- Python object state through getter/setter pairs;
- torch CPU RNG, and CUDA RNG only when requested on a CUDA-capable runtime.

It does not call Wan, MCP, ImageReward, checkpoints, VAE, or any real generator.

Milestone 2B1 adds only protocol wrappers in
`speculative/adapters/self_forcing_mcp.py`:

- `SelfForcingMCPProposalSource`;
- `SelfForcingMCPFallbackGenerator`;
- `SelfForcingMCPCommitter`.

These wrappers are a stateless delegation layer. They hold one shared runtime
object and expose the existing controller protocols without owning generation
state.

## Non-Goals

The runtime state layer does not compute Wan touched ranges. It does not know
which K/V slices a model forward will write, whether attention will append,
overwrite, or roll local cache content, or which output block is being committed.
Those decisions belong to the future real Wan backend in Milestone 2B2B.

The layer also does not clone a whole real KV cache. Callers must provide exact
regions that need protection.

It also does not reason about different tensor views sharing storage. Conflict
checks use tensor object identity. If two distinct tensor objects alias the same
storage, the future runtime must avoid conflicting specs itself.

Milestone 2B2A implements `SelfForcingMCPRuntime` as an orchestration layer, but
does not add `inference_speculative.py` and does not connect Wan, checkpoints,
ImageReward, VAE, or real fallback scoring. The real Wan backend and K/V
touched-range planner remain Milestone 2B2B work.

## Milestone 2B1 Thin Wrappers

The 2B1 adapter shape is:

```text
SpeculativeController
  -> thin wrapper
  -> shared SelfForcingMCPRuntime
```

All three wrappers must be constructed with the same runtime object. The
wrappers do not own KV cache, output buffers, source noise, rollout plans,
commit bookkeeping, transaction snapshots, or RNG state. Those states belong to
the future shared `SelfForcingMCPRuntime`.

The 2B1 runtime protocol is intentionally narrow and names runtime operations
by controller-window role:

- `propose_window(request) -> ProposalBatch`;
- `generate_target_fallback(candidate) -> FallbackResult`;
- `begin_window()`;
- `commit_block(request)`;
- `complete_window()`;
- `rollback_window()`.

The public wrapper APIs remain the existing controller protocols:

- `SelfForcingMCPProposalSource.propose(request)`;
- `SelfForcingMCPFallbackGenerator.generate(rejected)`;
- `SelfForcingMCPCommitter.begin()`;
- `SelfForcingMCPCommitter.commit(request)`;
- `SelfForcingMCPCommitter.complete()`;
- `SelfForcingMCPCommitter.rollback()`.

The wrapper methods pass through the same request or candidate objects they
receive and return the same objects produced by the runtime. They do not copy or
replace latents, source noise, `BlockRef`, or metadata. They also do not create
transactions, maintain a second active flag, retry failed calls, or alter the
controller rollback behavior.

The wrappers must not call pipeline private methods directly. Private pipeline
helpers remain behind the future runtime boundary, where state ownership and
rollback behavior can be reviewed in one place.

## Milestone 2B2A Runtime Orchestration

Milestone 2B2A adds `SelfForcingMCPRuntime` as the only owner of mutable
generation state. The wrappers remain a pure delegation layer:

```text
SpeculativeController
  -> thin wrapper
  -> shared SelfForcingMCPRuntime
  -> RuntimeBackendProtocol
```

The runtime owns the immutable source noise reference, output latent buffer,
immutable rollout plan, KV cache references, cross-attention cache references,
commit-order bookkeeping, runtime-state transaction managers, and backend
reference. It exposes only read-only observability needed by integration tests:

- `is_prepared`;
- `has_active_window`;
- `rollout_plan`;
- `committed_blocks`;
- `output`, as a borrowed buffer the caller must not mutate directly.

The runtime does not expose replaceable KV/cache references or transaction
snapshots.

The backend is a controlled model/cache operation interface. It provides state
specs and executes prepare, proposal, fallback, and commit operations, but it
does not own transactions and must not maintain a second commit-order state.
The 2B2A backend is injectable so tests can use a fake backend; this milestone
still does not call the real Wan generator.

Backend state specs must not declare the runtime output tensor or regions of
that tensor. The runtime adds its own `TensorRegionSpec` for output protection
so rollback still restores output if a backend forgets about it.

`SelfForcingMCPRuntimeConfig` is a narrow fail-fast config surface. It requires
`anchor_denoising_steps == (1000,)`, strict integer frame and depth fields, a
legal positive frame plan with a possible final short block, and an explicitly
validated attention/cache mode. Unsupported modes fail during config/runtime
setup rather than silently falling back.

The rollout plan is immutable. It records block starts, `BlockRef` values with
`start_frame` and `num_frames`, planned anchor indices, and
`period = mcp_depth + 1`. The runtime never stores a mutable frame cursor:
`current_start` is always derived as
`BlockRef.start_frame * frame_seq_length`.

Legal proposal anchors are dynamic. `anchor_block_indices` records the
always-accept reference schedule only; it is not a legality gate. A new
`ControlRequest.anchor_block` must equal
`rollout_plan.blocks[len(committed_blocks)]`, so reject paths, smaller
`max_depth` windows, and short returned proposal batches continue from the next
uncommitted block rather than from the next reference anchor.

Because the controller's `begin()` API carries no request, `propose_window()`
creates a read-only pending window descriptor after a successful proposal. The
descriptor contains only runtime-constructed immutable plan information:
`anchor_block` and `allowed_blocks`. It does not retain the caller's
`ControlRequest`, request metadata, `ProposalBatch`, or any caller-owned mutable
container. The runtime rejects stale or future anchors before calling the
backend. After the backend returns, while the temporary transaction is still
open, `propose_window()` validates the returned `ProposalBatch`, including
`validate_contiguous_block_range`, plan membership, anchor match, returned draft
count, and latent compatibility. Only then does it build the pending descriptor
from the actual returned anchor and draft blocks. Malformed proposals roll back
and leave no pending descriptor, so a later valid proposal can retry.

`begin_window()` consumes the latest pending descriptor and turns it into
active-window metadata only after transaction capture succeeds.
`complete_window()` and `rollback_window()` clear the active descriptor; a
second proposal cannot silently replace a pending descriptor, and no proposal is
accepted while a window is active.

There are three runtime-owned lifecycles:

- `prepare()` lifecycle: runs once successfully. An outer persistent transaction
  from `backend.prepare_persistent_state_specs(...)` protects cross-attention
  initialization state. Inside it, a temporary transaction from
  `backend.temporary_state_specs("prepare", ...)` protects scratch
  self-attention/RNG state. Success rolls back the inner transaction and
  completes the outer transaction. Failure rolls back both, leaves the runtime
  unprepared, and remains retryable without half-initialized cross-attention
  state.
- proposal/fallback temporary lifecycle: `propose_window()` and
  `generate_target_fallback()` open 2A transactions from backend
  `temporary_state_specs(...)`, call the backend, and roll back on both success
  and failure so KV/output/bookkeeping/RNG temporary mutations do not persist.
  Proposal validates returned anchor and draft latents before rollback.
  Fallback validates that the backend returned `FallbackResult`, that
  `result.block == candidate.block`, and that
  `result.source_noise is candidate.source_noise`, then validates the fallback
  latent before rollback. Proposal/fallback also reject returned tensor latents
  that share storage with any transaction tensor value or tensor region source
  before rollback. The runtime does not copy returned latents.
- committer window lifecycle: `begin_window()` opens a long-lived 2A
  transaction from backend `window_state_specs(...)` plus runtime-owned commit
  bookkeeping state. `complete_window()` keeps committed state. `rollback_window()`
  restores KV, output, bookkeeping, and RNG, and clears the active runtime
  reference even when restore raises.

Commit order is explicit: `commit_block()` requires an active window, validates
that the committed block is the next plan block allowed by the active
descriptor, revalidates latent compatibility, rejects latent storage aliasing
against the active window transaction specs, calls
`backend.commit_context_block(...)`, writes the original latent into the output
slice, then updates runtime commit bookkeeping. The alias check covers backend
window tensor regions, backend window tensor values, and the runtime-owned
output `TensorRegionSpec`. If validation or the backend raises, output and
bookkeeping are not updated; the controller's rollback path restores transaction
state.

Latent compatibility is exact. For a block, the target is
`output[:, block.start_frame:block.start_frame + block.num_frames]`. Proposal
anchor latents, proposal draft latents, fallback latents, and commit latents
must be `torch.Tensor` objects with exactly the same shape, dtype, device, and
layout as that target slice. Broadcasting is never accepted. The runtime checks
these rules without copying the latent object.

Milestone 2B2A still does not implement the real Wan backend, real K/V
touched-range planning, checkpoint loading, ImageReward, VAE, or scoring.
Milestone 2B2B should implement the Wan backend and touched-range planner behind
`RuntimeBackendProtocol`. Milestone 2B2C should add `inference_speculative.py`
and server GPU parity against frozen `inference_mcp.py`.

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

Milestone 2B2A wraps this layer inside `SelfForcingMCPRuntime`; protocol
wrappers do not call these low-level specs directly because the runtime owns
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
