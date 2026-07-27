# Wan Backend State Mutation Audit

Milestone: 2B2B1

Branch: `feat/speculative-mcp-adapter`

Baseline:

- `inference_mcp.py` is the frozen reference oracle.
- `inference_mcp.py` defines `ANCHOR_DENOISING_STEPS = [1000]`.
- This audit does not load checkpoints, call real Wan forward, use GPU, VAE, or
  ImageReward.
- This audit supports only the current verified Self-Forcing MCP shape. It does
  not claim parity or acceleration.

## Real Checkpoint Validation

Milestone F2B added a server-only real checkpoint smoke for the completed
`SelfForcingWanMCPBackend` and `SelfForcingMCPRuntime` integration. This
section records that validation without changing the earlier checkpoint-free
mutation audit evidence.

Command:

```bash
cd /home/dataset-assist-0/luojy/efficiency/rippor/speculative-forcing

PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES=0 \
/home/dataset-assist-0/luojy/efficiency/rippor/envs/speculative-forcing/bin/python \
  scripts/smoke_speculative_wan_backend_real.py \
  --config configs/ode_init.yaml \
  --checkpoint logs/ode_cont3_301/checkpoint_model_000300/model.pt \
  --prompt "A small boat moves slowly across a calm lake." \
  --seed 0 \
  --num_frames 6 \
  --mcp_depth 1 \
  --device cuda:0 \
  2>&1 | tee /tmp/speculative_f2b_real_smoke.txt
```

Configuration and checkpoint:

- Validation branch: `feat/speculative-wan-backend`.
- F2A backend commit:
  `11574b26dff7a16df0b7217720c4de9b726c3654`.
- F2B validation commit:
  `ffc6589013ee391daa904c01ce7d8b1594bb91d1`.
- GPU: NVIDIA A100-SXM4-80GB.
- Checkpoint:
  `logs/ode_cont3_301/checkpoint_model_000300/model.pt`.
- Checkpoint restore mode: `MCP_COMPLETE_STRICT_RESTORE`.
- MCP tensor count: 172.
- `num_frames`: 6.
- `mcp_depth`: 1.

Observed smoke results:

- `prepare=PASS`.
- `proposal_rollback=PASS`.
- `fallback_rollback=PASS`.
- `window_rollback=PASS`.
- `window_complete=PASS`.
- Cross-attention identity preserved: `True`.
- Same-noise fallback: `True`.
- CUDA RNG rollback: `True`.
- Final committed blocks: `[0, 1]`.
- Final KV index: `global_end_index == local_end_index == 9360`.
- Peak allocated CUDA memory: 18.437 GiB.

The peak memory value is diagnostic only. It includes other components loaded
during `ODERegression` initialization, including VAE, and must not be treated
as a backend memory result, speed result, or performance claim.

Validated support scope:

- T2V.
- Batch size 1.
- Denoising schedule `[1000]`.
- MCP depth 1/2/3 is supported by code.
- This real checkpoint smoke validated depth 1.
- Global attention with `local_attn_size == -1`.
- No local rolling.

Not validated by this smoke:

- Real smoke for `mcp_depth` 2 or 3.
- Multiple prompts.
- Full controller inference entrypoint.
- Output video parity.
- Quality metrics.
- Speed metrics.
- ImageReward.
- VAE verifier.

Warnings observed and considered non-blocking for F2B:

- `torch.load` `FutureWarning` is not a blocker for this milestone.
- `sink_size` config attribute deprecation warning is not a blocker for this
  milestone.
- The smoke script does not call VAE decode, but `ODERegression` construction
  loads VAE as part of the existing model object.

## Source Evidence

Read sources:

- `inference_mcp.py`
- `pipeline/self_forcing_training.py`
- `speculative/adapters/self_forcing_runtime.py`
- `speculative/adapters/runtime_state.py`
- `speculative/adapters/self_forcing_mcp.py`
- `docs/speculative/MCP_ADAPTER.md`
- `docs/speculative/MCP_ADAPTER_AUDIT.md`
- `model/base.py`
- `model/dmd.py`
- `model/ode_regression.py`
- `utils/wan_wrapper.py`
- `wan/modules/causal_model.py`
- `wan/modules/mcp.py`

Additional implementation fragment read for cross-attention mutation evidence:

- `wan/modules/model.py::WanT2VCrossAttention.forward`

## Operation Table

| Operation | Actual helper | Input | Return value | Mutated state | Mutation range | Persistent/temporary |
|---|---|---|---|---|---|---|
| prepare cross-attention | `SelfForcingTrainingPipeline._initialize_crossattn_cache`; `WanT2VCrossAttention.forward` on a staging cache; future backend `copy_` from staging into live cache | `batch_size`, `dtype`, `device`; prompt `context` padded to 512 tokens; block-0 scratch latent/timestep for the prepare forward | `_initialize_crossattn_cache` returns `None`; prepare forward return is ignored for persistent state | staging `crossattn_cache[layer]["k"]`, `["v"]`, `["is_init"]`; after full success live preallocated `k/v` tensor contents and live `is_init` | staging forward may replace staging dict values; live copy mutates `k/v` prompt-token range `[0, 512)` on dim 1; live `is_init` changes as a Python value | live cross-attn is persistent after prepare; staging cache and scratch self-attn are temporary |
| block-0 scratch prepare | same one-step `WanDiffusionWrapper.forward`/`CausalWanModel._forward_inference` path used only to force cross-attn initialization | rollout first block only; `current_start = 0`; `timestep = 1000`; no MCP futures | forward output ignored | scratch self-attn `kv_cache[layer]["k"]`, `["v"]`, `global_end_index`, `local_end_index` | block-0 self-attn range `[0, block.num_frames * frame_seq_length)` on dim 1 | Temporary; not part of persistent backend state |
| model freqs device migration | `CausalWanModel._forward_inference` | any first forward after `model.to(device)` when `self.freqs.device != patch_embedding.weight.device` | no direct return change | `self.freqs` Python attribute is rebound to `self.freqs.to(device)` | no tensor slice; whole ordinary Tensor attribute binding changes | Persistent setup mutation; must be completed before runtime construction, not inside proposal/fallback/commit |
| proposal anchor generation | `SelfForcingTrainingPipeline._inference_with_trajectory_mcp_accelerated` calling `WanDiffusionWrapper.forward` with `kv_cache` | `noise[:, start:start + m]`; `timestep = ones([B, m]) * 1000`; `current_start = start * frame_seq_length`; optional MCP futures | `(flow_pred, denoised_pred)` or `(flow_pred, denoised_pred, mcp_flow_preds)` | Temporary `kv_cache1[layer]["k"]`, `["v"]`, `global_end_index`, `local_end_index`; cross-attn only if prepare did not run | self-attn physical token ranges from formulas below; for current global append, `[current_start, current_start + block.num_frames * frame_seq_length)` on dim 1 | Temporary in adapter proposal; must roll back before returning |
| MCP future-chunk generation | `_mcp_future_chunks`; `WanDiffusionWrapper._run_mcp`; `MCPStack.forward` | future noise views `noise[:, lo:hi]`; future start frames; default MCP timestep 1000 | `_mcp_future_chunks` returns `(noises, starts)`; `_run_mcp` returns list of flow predictions | `_mcp_future_chunks` mutates nothing; MCP heads mutate no KV/cross-attn cache | No KV range beyond the anchor backbone forward; draft output frame ranges are future block frame intervals | Temporary proposal payload |
| target fallback generation | same backbone one-step call as proposal, without MCP futures | rejected `DraftCandidate.source_noise`; `timestep = 1000`; `current_start = rejected.block.start_frame * frame_seq_length` | `FallbackResult(block, latent, same source_noise)` in adapter contract | Temporary self-attn KV and indices; no output write; no commit bookkeeping | fallback block self-attn range from formulas below | Temporary; must roll back before returning |
| context commit | `SelfForcingTrainingPipeline._commit_context_block` | accepted latent; `context_timestep = context_noise`; `torch.randn_like(latent.flatten(0, 1))`; `current_start = start_frame * frame_seq_length`; `cache_start = self._cache_start()` | `None`; generator output ignored | Permanent self-attn `k/v`; `global_end_index`; `local_end_index`; runtime writes output slice separately; consumes RNG | commit block self-attn range from formulas below; output frame range `[start_frame, start_frame + num_frames)` | Persistent if controller window completes; protected by window transaction until then |
| window completion | `SelfForcingMCPRuntime.complete_window` | active transaction | `None` | transaction state is closed; active window metadata cleared | No tensor range; it keeps previously committed KV/output/RNG effects | Persistent |
| rollback required state | `RuntimeStateTransaction.rollback` via committer rollback path | window transaction specs from backend planner plus runtime output and committed-block specs | `None` or restore error | self-attn `k/v` touched by allowed commit blocks; global/local index tensors; output frame range; runtime committed-block bookkeeping; RNG state if commit consumed random context noise | union of allowed block commit ranges; same tensor/dim adjacent ranges may be restored as one larger range | Temporary protection for in-flight window |

## Call Chains

### Reference entrypoint

`inference_mcp.main` builds the model and rollout pipeline, prepares prompt
embeddings, makes source noise with a separate CPU `torch.Generator`, resets
runtime seeds, and calls `rollout_pipeline.inference_with_trajectory(...)`.
Evidence:

- `inference_mcp.py:9-11`: MCP tensor count, `[1000]`, always-accept policy.
- `inference_mcp.py:80-112`: T2V-only, positive block size, `mcp_depth <= config.mcp_num_modules`.
- `inference_mcp.py:147-163`: pipeline construction uses `[1000]`,
  `same_step_across_blocks=False`, `last_step_only=True`, `context_noise=0`,
  and MCP module/depth count from CLI depth.
- `inference_mcp.py:166-176`: source noise is `[1, num_frames, 16, 60, 104]`,
  sampled on CPU with a separate generator, cast to bf16, then moved to device.
- `inference_mcp.py:482-488`: seeds are reset immediately before rollout.

### Model freqs device migration

`CausalWanModel.__init__` assigns `self.freqs = torch.cat([...], dim=1)` as a
plain Python Tensor attribute. It is not registered with `register_buffer`, so
`model.to(device)` does not move it with parameters and buffers.

`CausalWanModel._forward_inference` then begins by reading the model parameter
device and rebinding the attribute when needed:

```text
device = self.patch_embedding.weight.device
if self.freqs.device != device:
    self.freqs = self.freqs.to(device)
```

This is a persistent Python attribute mutation, not a cache token-range
mutation. The 2B2B2 backend must complete this migration idempotently during
backend setup, before constructing the runtime and before any controller
transaction can start. Proposal, fallback, commit, and prepare scratch must
fail fast if `model.freqs.device` still differs from
`model.patch_embedding.weight.device`; none of those operations may hide the
first migration inside a transactional forward.

### Proposal and MCP futures

The accelerated rollout path is

```text
SelfForcingTrainingPipeline.inference_with_trajectory
  -> _inference_with_trajectory_mcp_accelerated
  -> _mcp_future_chunks(...)
  -> WanDiffusionWrapper.forward(..., kv_cache, crossattn_cache,
                                current_start, mcp_future_noises,
                                mcp_future_start_frames)
  -> CausalWanModel._forward_inference(..., return_features, mcp_patch_inputs)
  -> CausalWanAttentionBlock.forward
  -> CausalWanSelfAttention.forward
  -> WanDiffusionWrapper._run_mcp
  -> MCPStack.forward
```

Evidence:

- `pipeline/self_forcing_training.py:185-192`: `m`, period, block starts,
  one-step schedule and exit flags.
- `pipeline/self_forcing_training.py:223-282`: anchor loop calls the generator
  once at the exit step; with valid future chunks it passes `mcp_future_noises`
  and `mcp_future_start_frames`.
- `pipeline/self_forcing_training.py:112-139`: `_mcp_future_chunks` returns
  borrowed future noise views and start frames, or `None` entries when invalid.
- `utils/wan_wrapper.py:336-345`: MCP inputs require attached MCP modules and
  aligned future-noise/start lists.
- `utils/wan_wrapper.py:361-382`: future chunks are converted to patch inputs
  and passed into the causal model in the same wrapper forward.
- `utils/wan_wrapper.py:465-515`: `_run_mcp` builds per-depth timesteps,
  defaults them to `MCP_INPUT_TIMESTEP`, and invokes `self.mcp`.
- `wan/modules/mcp.py:34-42`: `MCP_INPUT_TIMESTEP = 1000`.
- `wan/modules/mcp.py:425-438`: MCP depth chain stops at first missing future
  embed and returns one flow prediction per module that ran.

### Target fallback

There is no fallback in `inference_mcp.py` because it is always-accept. The
audited target fallback operation is the same one-step target backbone call as
the anchor path, but without `mcp_future_noises`. The adapter contract requires
that `FallbackResult.source_noise is rejected.source_noise`, matching
`SelfForcingMCPRuntime._validate_fallback_result`.

Evidence:

- `speculative/adapters/self_forcing_runtime.py:314-328`: fallback runs inside a
  temporary transaction, validates the result, then rolls back temporary state.
- `speculative/adapters/self_forcing_runtime.py:455-463`: fallback must return
  `FallbackResult`, the same block, and the identical `source_noise` object.
- `utils/wan_wrapper.py:373-394`: causal KV forward without MCP futures returns
  two values.

### Context commit

The accelerated reference has a local `commit_to_cache` closure and the pipeline
also has `_commit_context_block`. Both add context noise and call the generator
at `context_noise`; `_commit_context_block` additionally passes
`cache_start=self._cache_start()`.

Evidence:

- `pipeline/self_forcing_training.py:195-215`: local accelerated closure.
- `pipeline/self_forcing_training.py:300-304`: anchor and drafts are committed in
  deployment order.
- `pipeline/self_forcing_training.py:352-374`: `_commit_context_block` uses
  `torch.randn_like`, `scheduler.add_noise`, `current_start`, and
  `cache_start=self._cache_start()`.
- `pipeline/self_forcing_training.py:349-350`: `_cache_start()` reads
  `kv_cache1[0]["global_end_index"]`.
- `speculative/adapters/self_forcing_runtime.py:345-383`: runtime commit order,
  backend commit call, output write, and bookkeeping update.

## Frozen Prepare Strategy

The actual cross-attention module lazily initializes prompt K/V inside
`WanT2VCrossAttention.forward`. Source evidence in
`wan/modules/model.py:174-183` shows that the first forward sets
`crossattn_cache["is_init"] = True` and assigns new `k` and `v` tensors into
the dict. It does not write into the tensors allocated by
`SelfForcingTrainingPipeline._initialize_crossattn_cache`.

The supported 2B2B2 backend must therefore prepare with a separate staging
cross-attention cache:

1. Allocate the live cross-attention cache with
   `_initialize_crossattn_cache`; its preallocated live `k/v` Tensor objects
   are the stable objects the runtime can snapshot.
2. Run the prepare forward with an independent staging cross-attention cache
   and block-0 scratch self-attention KV cache. The staging forward may replace
   staging dict `k/v` bindings.
3. Only after the full staging forward succeeds, copy staging prompt K/V into
   the already allocated live tensors with in-place `copy_`.
4. Set each live layer's `is_init=True` after the live K/V copies complete.
5. Never replace the live dict's `k` or `v` object bindings.

Rollback typing follows from that contract:

- live cross-attention `k/v` use backend tensor ranges on prompt-token dim 1,
  range `[0, 512)`;
- live `is_init` uses a backend Python value/object descriptor;
- staging cross-attention dicts and block-0 scratch self-attention KV/index
  state are temporary and are not persistent backend descriptors;
- output and runtime committed-block bookkeeping remain runtime-owned state.

`plan_prepare_scratch(block)` freezes the scratch side of this path. The block
must be rollout block 0 with `start_frame == 0`; `current_start` is still
`block.start_frame * frame_seq_length`; the schedule is `[1000]`; no MCP future
chunks are passed; self-attention K/V and index writes are temporary; there is
no output range, no persistent cross-attention descriptor, and no commit RNG
capture.

## Self-Attention KV Mutation Formula

Definitions:

- `T = frame_seq_length`.
- `S = BlockRef.start_frame * T`.
- `N = BlockRef.num_frames * T`.
- `current_start = S`.
- proposal and fallback omit `cache_start`, so
  `CausalWanSelfAttention.forward` sets `cache_start = current_start`.
- commit helper passes `cache_start = self._cache_start()`; the supported
  in-order adapter contract requires this equals `S` for the committed block.
- `cache_end = cache_start + N`.
- All intervals are half-open `[start, end)`.

Actual self-attention code:

- `wan/modules/causal_model.py:193-201`: computes `frame_seqlen`,
  `current_start_frame`, RoPE offsets, and `cache_end`.
- `wan/modules/causal_model.py:204-205`: reads physical KV capacity and
  `num_new_tokens`.
- `wan/modules/causal_model.py:206-222`: local rolling branch moves existing
  K/V then writes new K/V.
- `wan/modules/causal_model.py:223-228`: direct write branch.
- `wan/modules/causal_model.py:229-235`: attention reads the visible local
  window, then fills both index tensors.

### Global/default attention (`local_attn_size == -1`)

The current verified `WanDiffusionWrapper` default is `local_attn_size=-1`.
`CausalWanSelfAttention.max_attention_size` becomes `32760`, and no rolling
branch can run.

For one block:

```text
global_start = S
global_end = S + N
local_start = S
local_end = S + N
mutated k/v physical range = [S, S + N) on dim 1
global_end_index = S + N
local_end_index = S + N
```

This requires `S + N <= cache_capacity`.

### Local rolling branch (`local_attn_size != -1`)

The formula below is from source, but local attention is not the current
validated inference configuration. A backend may use it only after explicit
runtime validation.

For a block derived from `BlockRef`:

```text
pre_global = S
pre_local = min(S, cache_capacity)
cache_end = S + N
sink_tokens = sink_size * T
```

No roll if `N + pre_local <= cache_capacity`:

```text
local_start = pre_local
local_end = pre_local + N
mutated k/v physical range = [local_start, local_end)
global_end_index = S + N
local_end_index = local_end
```

Roll if `N + pre_local > cache_capacity`:

```text
num_evicted_tokens = N + pre_local - cache_capacity
num_rolled_tokens = pre_local - num_evicted_tokens - sink_tokens
local_end = cache_capacity
local_start = cache_capacity - N
rolled destination range = [sink_tokens, sink_tokens + num_rolled_tokens)
new write range = [local_start, local_end)
global_end_index = S + N
local_end_index = cache_capacity
```

The physical restore range is the union of the rolled destination range and the
new write range. If these are adjacent or overlap for the same layer/K-or-V
tensor and dim, one combined snapshot is rollback-equivalent.

Fail-fast requirements:

- `0 <= local_start <= local_end <= cache_capacity`.
- `0 <= sink_tokens <= cache_capacity`.
- `N + sink_tokens <= cache_capacity`.
- `num_rolled_tokens >= 0`.

### Final short block

The code uses `roped_query.shape[1]` as `num_new_tokens`; therefore the final
short block does not need a separate formula. It uses the same equations with
`N = short_block.num_frames * T`.

## Operation-Specific Ranges

### 0. Prepare cross-attention and block-0 scratch

Persistent live cross-attention descriptors:

```text
crossattn k/v range = [0, 512) on dim 1, per layer
crossattn is_init = Python value descriptor, per layer
```

Scratch self-attention prepare block `P0`:

```text
P0.index == 0
P0.start_frame == 0
current_start = P0.start_frame * T = 0
timestep = 1000
self-attn touched range = formula(P0)
output range = none
commit RNG capture = false
```

### 1. Single proposal anchor

Input block `B`:

```text
current_start = B.start_frame * T
timestep = 1000
self-attn touched range = formula(B)
output candidate range = [B.start_frame, B.start_frame + B.num_frames)
```

No `torch.randn_like` occurs on the one-step `[1000]` proposal path. The
proposal generator call can initialize cross-attention only if `prepare()` did
not already initialize it; the supported adapter path runs prepare first.

### 2. K MCP drafts

The same anchor backbone forward supplies tapped features for all returned MCP
drafts. MCP heads do not touch `kv_cache` or `crossattn_cache`. For returned
draft block `D_k`:

```text
source noise view = noise[:, lo:hi]
draft frame output range = [D_k.start_frame, D_k.start_frame + D_k.num_frames)
```

The KV touched range remains the anchor range only.

### 3. Target fallback single block

For rejected block `R`:

```text
current_start = R.start_frame * T
timestep = 1000
self-attn touched range = formula(R)
output candidate range = [R.start_frame, R.start_frame + R.num_frames)
```

Fallback does not consume RNG in the `[1000]` one-step target call and must not
call `_commit_context_block`.

### 4. Commit single block

For committed block `C`:

```text
current_start = C.start_frame * T
cache_start = _cache_start()
supported in-order invariant: cache_start == current_start
self-attn touched range = formula(C)
output persistent range = [C.start_frame, C.start_frame + C.num_frames)
```

Commit consumes RNG through `torch.randn_like(latent.flatten(0, 1))`. In the
reference GPU path this is CUDA RNG; if a CPU-only future test uses CPU latents,
it is CPU RNG.

### 5. ProposalBatch window union

For `RuntimeWindowDescriptor.allowed_blocks = (B0, B1, ..., Bj)`, the window
rollback plan is the union of `formula(Bi)` for every allowed block, plus output
frame range:

```text
output range = [min(Bi.start_frame), max(Bi.start_frame + Bi.num_frames))
```

The runtime constructs `allowed_blocks` from the actual returned
`ProposalBatch`, so if the backend returns fewer drafts than `request.max_depth`,
the window plan must use the shorter actual list.

### 6. Global append

For the current verified global attention layout, logical and physical cache
positions are the same:

```text
global append range = [S, S + N)
physical k/v range = [S, S + N)
```

### 7. Local rolling

For local mode, logical global append remains `[S, S + N)`, while physical
mutation follows the local formulas above. If rolling crosses the capacity
boundary, rollback must restore both the rolled destination and the new write
range for every layer and for K and V independently.

### 8. Index tensors

Every generator forward with `kv_cache` fills:

```text
kv_cache[layer]["global_end_index"] = S + N
kv_cache[layer]["local_end_index"] = local_end
```

These are scalar tensor values and must be snapshotted as value descriptors, not
as K/V token ranges.

## Supported Configuration

This milestone supports only the current verified Self-Forcing MCP inference
shape:

- T2V only: `inference_mcp.py` rejects `config.i2v=True`.
- Batch size is exactly 1 from `make_noise`.
- No CFG-expanded generation batch is present in the reference rollout; only
  `conditional_dict["prompt_embeds"]` is passed to the generator.
- Latent layout is `[B, F, C, H, W]`; frame/time dimension is dim 1.
- Source noise shape in the reference is `[1, num_frames, 16, 60, 104]`.
- Denoising schedule is exactly `[1000]`.
- `num_frame_per_block` comes from config and must be positive; the frozen
  reference also requires `num_frames % num_frame_per_block == 0`.
- `frame_seq_length = 1560`.
- Pipeline KV layer count is `num_transformer_blocks = 30`.
- KV cache container is a list of per-layer dicts with keys `k`, `v`,
  `global_end_index`, and `local_end_index`.
- KV tensor allocation in the pipeline is `[batch_size, kv_cache_size, 12, 128]`;
  token dimension is dim 1.
- `kv_cache_size = num_max_frames * frame_seq_length`.
- Global and local indices coexist in the same per-layer KV dict; there is no
  separate global-cache tensor and local-cache tensor.
- Cross-attention cache container is a list of per-layer dicts with `k`, `v`,
  and `is_init`.
- Cross-attention allocation is `[batch_size, 512, 12, 128]`; prompt-token
  dimension is dim 1.
- Prepare uses a staging cross-attention cache and copies into the live
  preallocated `k/v` tensors. Live `k/v` object bindings must remain stable;
  only live tensor contents and live `is_init` may change.
- `model.freqs.device` must already match
  `model.patch_embedding.weight.device` before runtime construction and before
  any prepare/proposal/fallback/commit forward.
- Current verified attention uses causal self-attention with
  `local_attn_size=-1` and `sink_size=0`.
- `local_attn_size` must be `-1` or a positive integer; `0` is rejected.
- `sink_size > 0` is supported only with positive `local_attn_size`.
- MCP effective depth is 0 for disabled/anchor-only planning, otherwise 1, 2,
  or 3. The reference CLI accepts 1, 2, or 3 when MCP is enabled.
- Explicit proposal `allowed_blocks` may not contain more draft blocks than
  either `ControlRequest.max_depth` or planner `mcp_depth`.
- Rolling-cache formulas are documented and tested at descriptor level, but a
  real backend must fail fast until local attention is separately validated.

Unsupported layouts must fail fast:

- batch size other than 1;
- CFG-added batch dimension;
- latent time dimension not dim 1;
- missing global or local index tensors;
- cache containers other than the current list-of-dicts layout;
- attention mode other than the audited causal KV path;
- MCP depth greater than 3;
- `BlockRef.num_frames > num_frame_per_block`;
- K/V token ranges outside `[0, cache_capacity)`.

## RNG Audit

- `inference_mcp.make_noise` uses a dedicated CPU generator for source noise and
  returns the tensor to the target device. This is outside per-operation planner
  ranges.
- `reset_runtime_seed` resets Python, NumPy, CPU torch, and CUDA torch RNG
  before rollout.
- `generate_and_sync_list` calls `torch.randint` on `noise.device` at rollout
  setup. With the reference schedule length of 1 and `last_step_only=True`, the
  values are deterministic zeros, but the call still belongs to the rollout RNG
  sequence.
- Proposal and fallback with schedule `[1000]` do not call `torch.randn_like`.
- Commit calls `torch.randn_like` inside context recache and therefore requires
  RNG capture for rollback. In the reference runtime this is CUDA RNG.
- MCP `_run_mcp` uses `torch.full` for missing explicit MCP timesteps; it does
  not consume RNG.

## Planner Mapping

`speculative/adapters/wan_state_planner.py` implements only pure descriptors:

- `WanCacheLayout`
- `TensorRangeDescriptor`
- `TokenRangeDescriptor`
- `WanOperationRange`
- `WanStateMutationPlan`
- `WanTouchedRangePlanner`

The planner does not create `TensorRegionSpec`; it does not hold tensors,
models, pipelines, runtimes, generators, mutable caches, RNG objects, or a frame
cursor. Its `WanStateMutationPlan` deliberately separates state ownership:

- `backend_tensor_ranges`: backend-owned tensor slices such as Wan self-attn
  K/V and live cross-attn K/V;
- `backend_tensor_value_names`: backend-owned scalar tensor values such as
  `global_end_index` and `local_end_index`;
- `backend_python_value_names`: backend-owned Python values such as live
  cross-attn `is_init`;
- `output_ranges`: runtime-owned output frame intervals, exposed only for
  audit and for the runtime to bind;
- `operation_ranges`: immutable formula evidence for each Wan operation;
- `capture_rng` and `capture_cuda_rng`: rollback requirements for commit
  context-noise RNG.

A future backend binds only the `backend_*` fields to actual borrowed tensors,
scalar tensor values, and Python value specs. It must not bind `output_ranges`
or `self_forcing_runtime_committed_blocks`; those remain owned by
`SelfForcingMCPRuntime`.

## Resolved Prepare Path

The previous prompt-prepare ambiguity is resolved for this milestone by the
staging-cache plus block-0 scratch prepare contract above. 2B2B2 must implement
that exact shape without modifying `inference_mcp.py`: complete static
`model.freqs` migration first, run prepare through staging cross-attention and
scratch self-attention state, copy staging `k/v` into stable live tensors, then
set live `is_init=True`. No remaining 2B2B1 touched-range formula is marked
`UNRESOLVED`.
