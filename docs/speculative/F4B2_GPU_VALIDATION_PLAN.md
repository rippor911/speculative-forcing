# F4B2 GPU Validation Plan

Milestone F4B2 validates the Wan VAE cache transaction primitive against the
real CUDA cached-decode path on an A100. It adds only a standalone server-side
validation script; it does not modify production inference.

## Latent Sources

The script obtains four real latents from one real `ODERegression` /
`SelfForcingMCPRuntime` flow with the fixed experiment shape:

- `anchor_latent`: block 0 backbone anchor from `runtime.propose_window()`.
- `draft_latent`: block 1 MCP depth-1 draft from the same proposal batch.
- `target_latent`: block 1 target fallback generated from that draft, requiring
  `fallback.source_noise is draft.source_noise`.
- `following_latent`: block 2 backbone anchor after block 0 anchor and block 1
  fallback are committed and the window is completed.

Each latent is immediately detached, cloned, and moved to CPU. The VAE
comparison then reuses those frozen tensors for both paths.

## Why Freeze Latents First

The goal is to isolate Wan VAE cached-decode cache behavior. Re-running the
generator inside Path A or Path B would mix generator CUDA nondeterminism,
runtime cache state, and VAE cache state into one comparison. Freezing the four
latents makes Path A and Path B consume identical process-local inputs.

The script does not save or commit latent tensors.

## Path A

Path A starts from `model.vae.model.clear_cache()` and decodes:

1. `anchor_latent` with `use_cache=True`, expecting 9 pixel frames.
2. `target_latent` with `use_cache=True`, expecting 12 pixel frames.
3. `following_latent` with `use_cache=True`, expecting 12 pixel frames.

It fingerprints the VAE cache after the target decode and after the following
decode with `include_digest=True`.

## Path B

Path B starts from a fresh `clear_cache()` and decodes:

1. `anchor_latent`, expecting 9 pixel frames.
2. Captures the pre-draft cache fingerprint and the old `_feat_map` entry
   Python references.
3. Opens `WanVAECacheTransaction(model.vae).begin()`.
4. Decodes `draft_latent`, expecting 12 pixel frames.
5. Re-digests the original old `_feat_map` Tensor objects before rollback.
6. Calls `rollback()`.
7. Decodes `target_latent`, expecting 12 pixel frames.
8. Decodes `following_latent`, expecting 12 pixel frames.

Path B fingerprints the rollback state, target-post state, and following-post
state with `include_digest=True`.

## Equality Checks

Rollback is checked three ways:

- structural equality: same cache shape and metadata, ignoring identity and
  numeric digest fields;
- numerical equality: same Tensor digests, ignoring Python object identity;
- identity equality: same wrapper/model/list/Tensor identity and data pointers,
  ignoring digest fields.

Path A and Path B target/following post-cache states are compared only with
structural and numerical equality. Cross-path identity is not required because
each path begins from `clear_cache()` and creates independent Python list and
Tensor objects.

The old Tensor digest check answers whether real CUDA cached decode mutated any
previous cache Tensor object in place. A digest change on an old `_feat_map`
Tensor is blocking because the F4B1 transaction restores shallow entry
references.

## CUDA Memory

The memory loop starts from a committed VAE context:

```text
clear_cache()
decode anchor_latent
```

It records a baseline cache fingerprint and CUDA memory, then performs two
warmup reject/rollback loops. After warmup it synchronizes and records:

```text
post_warmup_allocated_baseline = torch.cuda.memory_allocated(device)
```

It then resets peak memory stats so measured `max_memory_allocated` excludes
warmup peaks. After warmup it runs `repeat_loops` measured loops; the CLI
requires at least 5 loops and defaults to 20:

```text
begin transaction
decode draft_latent
delete draft pixels
rollback
gc.collect()
torch.cuda.synchronize()
```

Each measured loop records `memory_allocated`, `memory_reserved`,
`max_memory_allocated`, and rollback structural/numerical/identity equality.
The loop does not call `torch.cuda.empty_cache()`.

`memory_allocated` is memory currently held by active PyTorch Tensor objects.
After each measured loop deletes temporary draft pixels, rolls back the VAE
cache, runs Python GC, and synchronizes CUDA, it should return to the stable
post-warmup baseline. The default blocking criterion is:

```text
allocated_all_return_to_post_warmup_baseline = all(delta == 0)
```

where each delta is measured against `post_warmup_allocated_baseline`.

`memory_reserved` is memory held by the CUDA caching allocator. It may remain
above the initial baseline after warmup, so reserved memory is diagnostic only.
The report still records reserved final delta, tail range, and monotonic/trend
information, but reserved memory returning to the initial value is not a
standalone failure condition.

## Device Gate

The script records `gpu_name`, `compute_capability`, and `running_on_a100`.
It requires `"A100"` in the device name and compute capability `(8, 0)`. A
non-A100 run cannot produce `overall_pass=true`.

## Passing Criteria

The validation passes only if all blocking checks pass:

- block 0 cached decode outputs 9 pixel frames;
- each later cached 3-latent block outputs 12 pixel frames;
- all decoded pixel tensors are rank-5, batch 1, channel count 3, and finite;
- rollback structural, numerical, and identity equality all hold;
- old `_feat_map` Tensor digests do not change during rejected draft decode;
- Path A and Path B target pixels are equal within the CLI max-abs tolerance;
- Path A and Path B following pixels are equal within the CLI max-abs tolerance;
- target/following Path A/B post-cache structural and numerical equality hold;
- memory-loop rollback structural, numerical, and identity equality hold;
- every measured rollback returns `memory_allocated` to the post-warmup
  baseline.

Default pixel max-abs tolerance is `0.0`, so the default requires exact target
and following pixel equality.

## Non-Goals

This experiment does not decide transaction ownership across evaluator,
committer, or controller. It does not implement candidate decode integration,
accepted-candidate production handoff, ImageReward, acceptance thresholds,
quality experiments, speed experiments, or any claim of quality equivalence or
acceleration.
