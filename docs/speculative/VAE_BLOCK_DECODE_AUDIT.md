# Wan causal VAE block decode audit

Milestone: F4A

Scope: audit only. This document does not implement a candidate decoder,
transaction class, scorer, controller change, runtime change, or test.

Preflight observed:

- Branch: `feat/speculative-vae-transaction`
- HEAD: `0526d399031bbed0a886a013df6c0ddfeac2be2f`
- Initial `git status --short`: clean

## Files and searches audited

Required files read completely:

- `SPEC.md`
- `inference_mcp.py`
- `inference_speculative.py`
- `model/base.py`
- `utils/wan_wrapper.py`
- `wan/modules/vae.py`
- `docs/speculative/ARCHITECTURE.md`
- `docs/speculative/F3_GPU_VALIDATION.md`

Additional code inspected because repository searches referenced it:

- `model/ode_regression.py`
- `pipeline/causal_inference.py`
- `inference.py`
- `demo_utils/constant.py`
- `demo_utils/vae.py`
- `demo_utils/vae_block3.py`
- `demo_utils/vae_torch2trt.py`

Repository searches executed:

- `rg -n decode_to_pixel .`
- `rg -n cached_decode .`
- `rg -n clear_cache .`
- `rg -n _feat_map .`
- `rg -n _conv_idx .`
- `rg -n use_cache .`
- `rg -n vae .`
- `rg -n VAE .`
- targeted follow-up searches for `register_buffer`, `ZERO_VAE_CACHE`,
  `feat_cache`, `feat_idx`, `CACHE_T`, `upsample3d`, and VAE class/function
  definitions.

Important search note: one parallel `rg` run without explicit `.` returned no
matches, contradicting already-read source. It was discarded and rerun with
explicit repository path `.`; the explicit-path search results are the audit
evidence.

## 1. Current VAE call chain

### Construction path

Current F3/F4 inference entrypoints instantiate `ODERegression`, which creates
`self.vae = WanVAEWrapper()`:

- `inference_mcp.py:447` constructs `ODERegression`; final decode occurs at
  `inference_mcp.py:534-538`.
- `inference_speculative.py:560` constructs `ODERegression`; final decode occurs
  at `inference_speculative.py:670-674`.
- `model/ode_regression.py:94-112` overrides `_initialize_models()` and creates
  `WanVAEWrapper` at `model/ode_regression.py:103-107`.
- `model/base.py:26-67` also documents the base initialization path and creates
  `WanVAEWrapper` at `model/base.py:60-64`.
- `utils/wan_wrapper.py:74-93` defines `WanVAEWrapper.__init__`, creates
  `mean`, `std`, and `self.model = _video_vae(...)`.
- `wan/modules/vae.py:612-636` defines `_video_vae`; it constructs `WanVAE_`
  under `torch.device('meta')`, loads the VAE checkpoint with `assign=True`, and
  returns the module.
- `wan/modules/vae.py:483-509` defines `WanVAE_.__init__`; it creates encoder,
  decoder, `conv1`, `conv2`, then calls `self.clear_cache()`.

The current scripted speculative runtime decodes only the final complete latent
with `use_cache=False`. `inference_speculative.py:52-67` defines a scripted noop
evaluator that explicitly does not decode candidates, and
`inference_speculative.py:673-674` performs only the final decode.

### `WanVAEWrapper.decode_to_pixel(..., use_cache=False)`

Code path:

1. `utils/wan_wrapper.py:110-113`: `decode_to_pixel()` receives latent as
   `[B, T, C, H, W]` and permutes it to `[B, C, T, H, W]`.
2. `utils/wan_wrapper.py:117-119`: creates per-call `scale` tensors from
   `WanVAEWrapper.mean` and `WanVAEWrapper.std`; this does not rebind or mutate
   the wrapper attributes.
3. `utils/wan_wrapper.py:121-124`: with `use_cache=False`, selects
   `self.model.decode`.
4. `utils/wan_wrapper.py:126-129`: loops over batch items and calls
   `decode_function(u.unsqueeze(0), scale)`, then applies `.float().clamp_(-1, 1)`
   to the returned output tensor.
5. `wan/modules/vae.py:545-569`: `WanVAE_.decode()` is the final VAE method.
6. `wan/modules/vae.py:546`: `decode()` calls `self.clear_cache()` before any
   latent-frame decode.
7. `wan/modules/vae.py:548-554`: unscales latent and applies `self.conv2(z)`.
8. `wan/modules/vae.py:555-567`: loops one latent frame at a time, rebinding
   `self._conv_idx = [0]` for each frame and calling `self.decoder(...,
   feat_cache=self._feat_map, feat_idx=self._conv_idx)`.
9. `wan/modules/vae.py:568`: `decode()` calls `self.clear_cache()` after the
   decode loop succeeds.
10. `utils/wan_wrapper.py:129-133`: wrapper stacks batch outputs and permutes
    back to `[B, T, C, H, W]`.

Final call: `WanVAE_.decode`, not `cached_decode`.

Cache behavior:

- This path calls `clear_cache()` at entry and on successful exit.
- `clear_cache()` is not a reset of the old list contents; it rebinds new lists
  at `wan/modules/vae.py:602-609`.
- A successful call leaves a clean cache state, but not the same object identity
  as the state before the call.
- If an exception occurs inside `WanVAE_.decode()` after line 546 and before
  line 568, the initial clean cache may be partially populated and left behind.
  There is no `try/finally`.
- If an exception occurs in the wrapper after `WanVAE_.decode()` returns, the
  model cache has already been cleared by line 568.

### `WanVAEWrapper.decode_to_pixel(..., use_cache=True)`

Code path:

1. `utils/wan_wrapper.py:110-113`: same input permute as above.
2. `utils/wan_wrapper.py:114-115`: asserts `latent.shape[0] == 1`; cached decode
   supports batch size 1 only at wrapper level.
3. `utils/wan_wrapper.py:117-119`: creates per-call scale tensors.
4. `utils/wan_wrapper.py:121-123`: selects `self.model.cached_decode`.
5. `utils/wan_wrapper.py:126-129`: loops over batch items. Because of the batch
   assertion, this loop has one element in valid cached use.
6. `wan/modules/vae.py:571-593`: `WanVAE_.cached_decode()` is the final VAE
   method.
7. `wan/modules/vae.py:573-579`: unscales latent and applies `self.conv2(z)`.
8. `wan/modules/vae.py:580-592`: loops one latent frame at a time, rebinding
   `self._conv_idx = [0]` for each frame and calling `self.decoder(...,
   feat_cache=self._feat_map, feat_idx=self._conv_idx)`.
9. `wan/modules/vae.py:593`: returns without calling `clear_cache()`.
10. `utils/wan_wrapper.py:129-133`: wrapper stacks and permutes the returned
    pixels.

Final call: `WanVAE_.cached_decode`.

Cache behavior:

- This path does not call `clear_cache()` before or after decode.
- It preserves and advances `self.model._feat_map`.
- It leaves `self.model._conv_idx` bound to the last per-frame scratch list used
  by the decode loop.
- If an exception occurs inside `cached_decode()`, any cache entries already
  replaced during that call remain. There is no rollback or `try/finally`.
- If an exception occurs in the wrapper after `cached_decode()` returns, the VAE
  cache remains in the post-decode state even if output post-processing fails.

## 2. Mutable state inventory

The cache touched by current `WanVAEWrapper.encode_to_latent`,
`decode_to_pixel(..., use_cache=False)`, and `decode_to_pixel(..., use_cache=True)`
is owned by `WanVAEWrapper` and its `WanVAE_` submodule. No `register_buffer`
usage was found in `utils/wan_wrapper.py`, `wan/modules/vae.py`, or the inspected
demo decoder files.

`demo_utils/constant.py:5-41` defines `ZERO_VAE_CACHE`, and
`demo_utils/vae.py` / `demo_utils/vae_block3.py` define independent decoder
wrappers with explicit cache inputs/outputs. They are not imported by
`WanVAEWrapper`, `inference_mcp.py`, or `inference_speculative.py`; therefore
they are not part of the current WanVAEWrapper runtime call chain.

| State name | Owner | Type | Initialization | Modification | decode uses? | cached_decode uses? | Success retention | Exception residue | Transaction handling | Rollback target |
|---|---|---|---|---|---|---|---|---|---|---|
| `mean` | `WanVAEWrapper` | plain `torch.Tensor` attr | `utils/wan_wrapper.py:77-86` | `.to(...)` creates per-call tensors at `utils/wan_wrapper.py:97-99`, `117-119`; no attr rebind | yes, scale | yes, scale | retained unchanged | unchanged | fingerprint binding/value; no clone needed for VAE cache rollback | original attr binding and tensor identity |
| `std` | `WanVAEWrapper` | plain `torch.Tensor` attr | `utils/wan_wrapper.py:81-86` | same as `mean` | yes | yes | retained unchanged | unchanged | fingerprint binding/value; no clone needed | original attr binding and tensor identity |
| `model` | `WanVAEWrapper` | `WanVAE_` submodule | `utils/wan_wrapper.py:88-92` | `.to(...)` on wrapper may move submodule parameters; encode/decode/cached_decode mutate internal cache attrs | yes | yes | retained | internal dirty cache possible | snapshot internal cache attrs; do not clone model parameters | original submodule binding |
| `_conv_num` | `WanVAE_` | `int` | `clear_cache()` in `wan/modules/vae.py:602-603`, called by `WanVAE_.__init__` at line 509 | rebound to `count_conv3d(self.decoder)` by every `clear_cache()` | yes indirectly for list length | yes indirectly if cache was initialized | retained after cached decode; reset after normal decode | may reflect latest `clear_cache()` | snapshot attr value | original int value |
| `_conv_idx` | `WanVAE_` | Python list, usually one int | `wan/modules/vae.py:604` | rebound in `clear_cache()`; rebound per latent frame at `wan/modules/vae.py:556` and `581`; incremented in decoder layers at `423-469`, residual blocks at `202-217`, resample at `101-159` | yes, scratch index | yes, scratch index | normal decode leaves new `[0]`; cached decode leaves last scratch list with final index | partial final index possible | snapshot binding and contents if identity equality matters | original list object and value, or equivalent `[0]` if only structural state is required |
| `_feat_map` | `WanVAE_` | Python list of `None`, `Tensor`, or `'Rep'` | `wan/modules/vae.py:605` | list entries replaced by decoder/residual/resample cache writes | yes | yes | normal decode clears to new all-`None` list; cached decode preserves populated list | partial entries may remain | snapshot list binding, entry types, Tensor entries | original list identity plus original entries; clone Tensor entries for conservative numerical rollback |
| `_enc_conv_num` | `WanVAE_` | `int` | `wan/modules/vae.py:607` | rebound by `clear_cache()` | encode only | no direct use | reset by normal decode because `clear_cache()` resets both encode and decode state | can be rebound by failed normal decode after initial clear | include because `decode()` and `clear_cache()` touch it | original int value |
| `_enc_conv_idx` | `WanVAE_` | Python list, usually one int | `wan/modules/vae.py:608` | rebound in `clear_cache()` and per encode chunk at `wan/modules/vae.py:524`; incremented by encoder/residual/resample paths | encode only | no direct use | reset by normal decode; encode success clears | partial encode state possible | include in whole-VAE transaction if wrapper may call encode or normal decode | original list object and value |
| `_enc_feat_map` | `WanVAE_` | Python list of `None` or `Tensor` | `wan/modules/vae.py:609` | encoder path replaces entries at `wan/modules/vae.py:329-363`, `Resample.forward` downsample branch `143-159`, residual block `202-217` | encode only | no direct use | reset by normal decode; encode success clears | partial encode state possible | include because `clear_cache()` rebinds it | original list identity plus original entries; clone Tensors conservatively |
| `CausalConv3d._padding` | each causal conv | tuple | `wan/modules/vae.py:24-26` | no runtime attr mutation; forward uses local `padding = list(self._padding)` at `28-34` | yes | yes | unchanged | unchanged | fingerprint optional; not part of VAE transaction | original value |
| `RMS_norm.gamma`, `RMS_norm.bias` | RMS norm modules | `Parameter` or float `0.` | `wan/modules/vae.py:48-49` | no encode/decode mutation | yes | yes | unchanged | unchanged | no cache snapshot; model weights only | no rollback needed |
| Conv/attention/module weights | encoder/decoder/conv modules | parameters | constructors and checkpoint load | no encode/decode mutation in eval | yes | yes | unchanged | unchanged | no cache snapshot | no rollback needed |
| `feat_idx` default arg | `ResidualBlock.forward`, `Resample.forward`, etc. | default list object `[0]` at function definition | function signatures `wan/modules/vae.py:101`, `202`, `318`, `423` | only mutated when caller does not pass `feat_idx`; WanVAE_ always passes `self._conv_idx` / `_enc_conv_idx` when caching | not via default in current path | not via default in current path | default object is a latent hazard for direct calls | possible if direct external call omits `feat_idx` while passing cache | wrapper transaction does not need it for current path; direct decoder use should avoid defaults | not applicable to current wrapper path |
| local `cache_x` | decoder/encoder layer forwards | Tensor local | slices and `.clone()` in layer forward methods | new Tensor, then assigned into cache list | yes | yes | may become cache entry | if assigned before exception, entry remains | clone in snapshot if restoring numerical values | previous Tensor entry |
| output tensor after decode | wrapper local | Tensor | return from `decode`/`cached_decode` | `.float().clamp_(-1, 1)` mutates this output tensor at `utils/wan_wrapper.py:128` | yes | yes | returned to caller | output may exist with dirty cache | not part of VAE cache transaction except discard on reject | discard rejected output |

Static count note: for the default `_video_vae` config (`dim=96`,
`z_dim=16`, `dim_mult=[1,2,4,4]`, temporal downsample `[False, True, True]`),
`_feat_map` is expected to have 33 decoder cache slots. The code defines this
dynamically with `count_conv3d(self.decoder)`, so a fingerprint should record the
actual runtime length rather than hard-code the count. `demo_utils/constant.py`
also has 33 `ZERO_VAE_CACHE` entries, but that is a separate demo/TRT path.

## 3. Exact cache modification behavior

### `clear_cache()`

`wan/modules/vae.py:602-609` rebinds all six cache attributes:

- `_conv_num = count_conv3d(self.decoder)`
- `_conv_idx = [0]`
- `_feat_map = [None] * self._conv_num`
- `_enc_conv_num = count_conv3d(self.encoder)`
- `_enc_conv_idx = [0]`
- `_enc_feat_map = [None] * self._enc_conv_num`

It does not mutate the previous `_feat_map` or `_enc_feat_map` list in place.
Any external reference to the old list would continue pointing at the old list.

### `cached_decode()`

`cached_decode()` mutates cache through `Decoder3d.forward()`:

- `cached_decode()` rebinds `_conv_idx = [0]` for each latent frame at
  `wan/modules/vae.py:581`.
- `Decoder3d.forward()` writes `feat_cache[idx] = cache_x` at
  `wan/modules/vae.py:436` and `468`.
- `ResidualBlock.forward()` writes `feat_cache[idx] = cache_x` at
  `wan/modules/vae.py:216`.
- `Resample.forward(mode='upsample3d')` writes sentinel `'Rep'` for a first
  temporal-upsample encounter at `wan/modules/vae.py:106-108`; later calls replace
  that entry with a Tensor at `wan/modules/vae.py:111-132`.

Cache entry types confirmed in current code:

- `None`: clean slots from `clear_cache()`.
- `Tensor`: cloned feature cache entries from causal conv/residual/resample
  layers.
- sentinel string `'Rep'`: only for first cached encounter in `upsample3d`.
- No other entry type was found in `wan/modules/vae.py`.

Tensor content behavior:

- Current code saves new Tensor objects into cache entries via `.clone()` or
  `torch.cat(...)`; it does not perform in-place writes into existing cache
  Tensor contents.
- Previous cache Tensor entries are read, sometimes moved with `.to(...)`, and
  concatenated. They are not mutated in place by the audited code.
- Cache Tensor identity still matters for an identity-level rollback/fingerprint,
  and for detecting memory growth.

List identity behavior:

- `cached_decode()` uses the current `_feat_map` list object and modifies its
  entries in place.
- `clear_cache()` rebinds `_feat_map` to a new list object.
- The audited WanVAE code does not compare list identity; it reads
  `self._feat_map` each call. However, external references would observe
  in-place changes during `cached_decode()` and would not follow a later
  `clear_cache()` rebind.

Successful `use_cache=False` semantics:

- Clean-state equivalence: yes, after success the VAE has freshly-created clean
  lists containing `None` and index lists containing `[0]`.
- Value equivalence to a clean cache: yes for cache values, assuming the decoder
  and encoder module structure did not change.
- Object-identity equivalence to the pre-call state: no. `decode()` calls
  `clear_cache()` at entry and exit, and `clear_cache()` rebinds new list objects.
- Value equivalence to the pre-call state: only if the pre-call state was already
  clean. If the pre-call cache contained committed incremental state,
  `use_cache=False` destroys it and leaves a new clean state.

## 4. Latent frame to pixel frame mapping

The mapping below is derived from code, not from a guessed formula.

Relevant code:

- `CACHE_T = 2` at `wan/modules/vae.py:14`.
- `_video_vae()` config uses `temperal_downsample=[False, True, True]` at
  `wan/modules/vae.py:617-624`.
- `WanVAE_.__init__` reverses it into `self.temperal_upsample` at
  `wan/modules/vae.py:500` and passes it to `Decoder3d` at `507-508`.
- `Decoder3d.__init__` appends `Resample(..., mode='upsample3d')` when
  `temperal_upsample[i]` is true at `wan/modules/vae.py:399-416`. For the default
  config, the decoder has two temporal upsample stages followed by one spatial
  only upsample stage.
- `Resample.forward(mode='upsample3d')` has causal first-frame behavior:
  `feat_cache[idx] is None` stores `'Rep'` and skips temporal upsample at
  `wan/modules/vae.py:103-108`. On later calls, it runs `time_conv`, reshapes, and
  interleaves two temporal outputs at `wan/modules/vae.py:127-137`.
- `decode()` and `cached_decode()` both call the decoder one latent frame at a
  time, with a cache list, at `wan/modules/vae.py:555-567` and `580-592`.

Consequences:

- The first latent frame from a clean VAE cache produces 1 pixel frame, because
  both temporal upsample stages take the first-frame sentinel path.
- Each later latent frame with valid preceding cache produces 4 pixel frames,
  because two temporal upsample stages each double the time dimension.
- Cumulative pixel frames for a contiguous prefix of `n >= 1` latent frames are
  `1 + 4 * (n - 1)`.
- A zero-length latent decode is not supported by `decode()` / `cached_decode()`
  as written because `out` would never be assigned. As a prefix count before any
  call, cumulative pixel frames are 0.

| latent frames | cumulative pixel frames |
|---:|---:|
| 0 | 0 as an empty prefix; direct decode with `T=0` is unsupported |
| 1 | 1 |
| 2 | 5 |
| 3 | 9 |
| 4 | 13 |
| 6 | 21 |
| 9 | 33 |
| 12 | 45 |

For current speculative runtime, `num_frame_per_block = 3`:

- Block 0's 3 latent frames produce 9 pixel frames from a clean cache.
- Block 1 adds 12 pixel frames if decoded with block 0 cache retained.
- Block 2 adds 12 pixel frames if decoded with preceding cache retained.
- Later 3-latent blocks also add 12 pixel frames each.
- `cached_decode()` returns the pixels produced by the current call, not the
  cumulative prefix. Therefore the first cached call with 3 latent frames returns
  9 frames, while later cached calls with 3 latent frames return 12 frames.

Need real GPU verification:

- The frame-count derivation is code-level. It still needs real A100 validation
  against actual tensors to confirm shapes, boundary numerical equality, and no
  repeated or omitted visible boundary frames after concatenation.

## 5. Incremental block decode semantics

1. First `cached_decode` input should be the first contiguous latent prefix from
   the beginning of the video, normally block 0. For the current block size, that
   is 3 latent frames if the runtime wants to initialize VAE cache by block.
2. Subsequent calls should input only the next contiguous latent block(s), in
   order, with the committed VAE cache preserved.
3. First and subsequent calls behave differently. The first call from clean cache
   takes the `upsample3d` sentinel path and produces fewer pixel frames.
4. Each call returns the current call's decoded pixels, not the cumulative prefix.
5. Decoding a later block alone without prior cache uses first-frame/sentinel
   behavior for that block and will not match boundary-conditioned decode.
6. A candidate must be decoded with the already-committed context represented in
   VAE cache. That can be done by replaying committed context blocks into a clean
   VAE cache or by snapshotting a maintained committed VAE state.
7. If an accepted draft's decoded pixels become part of the committed visible
   prefix, its VAE cache should become permanent after the corresponding latent
   block is committed.
8. If a draft is rejected, rollback must restore at least `_feat_map`,
   `_conv_idx`, `_conv_num`, and any encode cache attrs touched or rebound by the
   chosen wrapper path. Rejected pixels must be discarded.
9. After rollback, decoding the target fallback with the same source noise should
   advance the VAE cache to the new permanent target state once that fallback is
   committed.
10. If deeper drafts are invalidated, any VAE cache state produced by decoding
    those deeper drafts must be fully discarded. The valid state is the committed
    prefix plus any accepted longest-prefix drafts only.
11. Duplicate pixel frames are possible if an implementation appends cumulative
    prefix outputs or replays a block with fresh cache and appends it as if it were
    boundary-conditioned incremental output.
12. Boundary pixel frames can be omitted if the implementation assumes every
    3-latent block maps to exactly 3 or 9 pixel frames. After block 0, a
    3-latent block maps to 12 newly returned pixel frames when cache is valid.

## 6. Transaction ownership and lifecycle

Current controller execution order is:

1. `committer.begin()` starts the controller window transaction
   (`speculative/controller.py:77-80`).
2. Anchor is committed through `committer.commit(...)`
   (`speculative/controller.py:82`, `_commit_once()` at
   `speculative/controller.py:235-274`).
3. For each draft candidate, the controller calls
   `evaluator.evaluate(candidate)` first (`speculative/controller.py:83-97`).
4. Then it calls `policy.decide(evaluation)`
   (`speculative/controller.py:99-109`).
5. Only after an accepted decision does it call `committer.commit(...)` for the
   draft (`speculative/controller.py:111-124`).
6. On rejection, it calls `fallback.generate(candidate)` and then
   `committer.commit(...)` for the fallback block
   (`speculative/controller.py:127-158`).
7. If all commits succeed, it calls `committer.complete()`
   (`speculative/controller.py:160-161`).
8. On any evaluation, policy, fallback, or commit exception after `begin()`, it
   restores controller bookkeeping and calls `committer.rollback()`
   (`speculative/controller.py:162-182`).

This creates a real ownership mismatch for VAE decode transactions:

- `SPEC.md:222-236` requires future causal VAE cache transaction to span draft
  decode, scoring, policy decision, accept complete, or reject rollback followed
  by target fallback decode and commit.
- The current `Evaluator` protocol says evaluation state is adapter-local and
  temporary model/cache mutation must be restored before returning
  `Evaluation` (`speculative/interfaces.py:64-72`).
- The current `CandidateDecoder` protocol also forbids persistent generation
  state mutation (`speculative/interfaces.py:20-28`).
- The current `Committer` protocol is the only boundary allowed to permanently
  modify Transformer KV, output storage, and generation cursor
  (`speculative/interfaces.py:94-117`). It does not currently mention VAE cache.

Therefore, a candidate VAE transaction cannot be considered fully owned yet. A
static VAE snapshot/restore primitive can be designed, but the integration owner
must be chosen before F4B implements production behavior.

### Integration option A: evaluator keeps a pending transaction

The evaluator decodes and scores the candidate, then returns `Evaluation` while
keeping a pending VAE transaction open. A later component completes or rolls back
that transaction based on `policy.decide(evaluation)`.

- Controller modification: likely required. The current controller has no hook
  between `policy.decide()` and `_commit_once()` except the existing committer
  path. A hidden evaluator-owned pending transaction would also need explicit
  cleanup when policy or commit raises.
- Evaluator contract: violates the current contract, because temporary VAE cache
  mutation would not be restored before returning `Evaluation`.
- Exception/window rollback: fragile unless controller rollback also knows how
  to find and roll back evaluator-owned VAE state. If `policy.decide()` raises,
  there is no current evaluator cleanup callback.
- Duplicate VAE decode: avoids duplicate decode for accepted drafts if the
  pending decoded candidate cache is completed.
- Permanent-state rule: weak. Permanent state would be staged by evaluator and
  completed by another owner, so permanent state would not be modified only
  through the existing commit path.

### Integration option B: evaluator always rolls back, committer decodes again

The evaluator decodes and scores inside an adapter-local VAE transaction, then
always rolls back before returning `Evaluation`. If a draft is accepted, or if a
fallback block is committed, the committer decodes that latent again and advances
the permanent VAE cache.

- Controller modification: not required if the committer owns permanent VAE
  cache advancement internally.
- Evaluator contract: satisfies the current contract because all temporary VAE
  mutation is restored before `Evaluation` returns.
- Exception/window rollback: aligns with the current controller window
  transaction if committer rollback restores both Transformer state and VAE
  permanent cache. If evaluation or policy raises, no evaluator VAE state should
  remain.
- Duplicate VAE decode: yes. Accepted candidates are decoded once for scoring
  and again during commit. Rejected target fallback is decoded during commit
  after fallback generation.
- Permanent-state rule: strong. Permanent VAE cache advances only in the commit
  path.

This is the simplest fit to the current contracts, but it may add enough VAE
cost to matter. That performance tradeoff must be measured later; F4A does not
claim speedup.

### Integration option C: explicit VAE transaction coordinator

Add an explicit VAE transaction coordinator called by the controller or by a
composite committer. It owns candidate decode staging, policy-time lifetime, and
final completion or rollback.

- Controller modification: required if the controller directly calls coordinator
  hooks around evaluation/policy/commit. Not required only if a composite
  evaluator/committer pair hides the coordinator behind existing protocols, but
  that coupling would still need a documented ownership contract.
- Evaluator contract: can be made compliant only if the coordinator, not the
  evaluator, owns the pending state. If evaluator returns with coordinator-held
  uncommitted VAE state, the current protocol text still needs revision.
- Exception/window rollback: can be clean if coordinator rollback is included in
  the same window transaction as Transformer KV/output rollback. It must define
  ordering for failures in VAE rollback versus Transformer rollback.
- Duplicate VAE decode: can avoid duplicate accepted-candidate decode if the
  staged candidate cache is completed, but only with explicit lifetime tracking.
- Permanent-state rule: can preserve the rule if coordinator completion is called
  only from the commit path. If policy/controller completes VAE directly, the
  rule becomes broader and must be updated.

F4A conclusion for integration: option B is the least invasive contract fit, but
F4A does not select an implementation. Options A and C require ownership changes
that are outside this audit.

## 7. F4 real A100 differential experiment design

This is a server-side design only. Thresholds below are initial suggestions; the
final pass/fail threshold must be based on real measurements.

Use the same checkpoint, config, prompt, seed, source noise, dtype, device, and
latent blocks as the F3 A100 setup where possible. Run with `torch.no_grad()` and
single GPU. Do not use MP4 encoding as the comparison artifact; compare tensors
before video serialization.

### Shared setup

1. Build the current `ODERegression` / VAE wrapper in eval mode.
2. Generate and then freeze exact latent tensors before either VAE path is run:
   - block 0: real backbone anchor latent;
   - block 1 draft: real MCP depth-1 latent;
   - block 1 target: real target fallback latent generated from the same
     `source_noise` object as the rejected draft;
   - block 2 following: real next-block latent generated after target block 1 is
     formally committed.
3. Detach and clone these four latent tensors into fixed comparison inputs. Path
   A and Path B must consume these copied tensors, not rerun the generator. This
   isolates VAE behavior from generator nondeterminism.
4. Use `model.vae.decode_to_pixel(..., use_cache=True)` for block decode.
5. Use a read-only VAE cache fingerprint before and after every step.
6. Record `torch.cuda.memory_allocated()` and `torch.cuda.memory_reserved()` after
   synchronization at each step.

### Path A

1. Clear VAE cache.
2. Decode block 0 context from the frozen real backbone anchor latent.
3. Decode block 1 target from the frozen real target fallback latent.
4. Save target pixels.
5. Save VAE post-state fingerprint.
6. Decode block 2 following from the frozen real following latent.
7. Save following-block pixels and final fingerprint.

### Path B

1. Clear VAE cache.
2. Decode block 0 context from the same frozen anchor latent.
3. Snapshot VAE state.
4. Decode rejected block 1 draft from the frozen real MCP depth-1 latent.
5. Roll back VAE state.
6. Decode the exact same frozen block 1 target latent as Path A.
7. Save target pixels.
8. Save VAE post-state fingerprint.
9. Decode the exact same frozen block 2 following latent as Path A.
10. Save following-block pixels and final fingerprint.

### Required comparisons

- target pixel shape;
- target pixel finite check;
- target max absolute difference;
- target mean absolute difference;
- target PSNR or equivalent numeric metric;
- following-block pixel shape and finite check;
- following-block max absolute difference;
- following-block mean absolute difference;
- following-block PSNR or equivalent;
- cache list structure;
- cache entry type sequence;
- each Tensor entry shape;
- each Tensor entry dtype;
- each Tensor entry device;
- each Tensor numerical summary and/or digest;
- list identity before snapshot, after rejected decode, after rollback, after target;
- cache Tensor identity before snapshot, after rejected decode, after rollback,
  after target;
- allocated CUDA memory;
- reserved CUDA memory;
- repeated reject/rollback memory trend across many iterations;
- digest of old cache Tensor entries before rejected draft decode and after
  rejected draft decode, before rollback, to detect any in-place mutation.

Provisional tolerances:

- Shapes, finiteness, structure, dtype, and device should match exactly.
- If deterministic kernels and identical dtype/device paths are used, target and
  following tensors should ideally have max absolute difference 0.
- Because CUDA kernels and mixed precision may introduce environment-specific
  differences, start by recording exact max/mean/PSNR without failing on an
  arbitrary threshold.
- Thresholds are only initial suggestions; final tolerances must be set from real
  A100 measurements.

## 8. Cache fingerprint design

The fingerprint must be read-only. It must not call `clear_cache()`, must not
write list entries, and must not mutate tensors in place. It may allocate
temporary detached CPU copies to compute digests.

For each audited attribute:

- attribute name;
- owner type;
- attribute exists;
- attribute type;
- `id(attribute_value)`;
- for scalar-like values: value;
- for list values:
  - list object id;
  - list length;
  - per-entry index;
  - per-entry type;
  - per-entry object id;
  - sentinel string value when entry is a string;
  - `None` marker when entry is `None`;
  - Tensor metadata when entry is a Tensor.

For each Tensor entry:

- shape;
- dtype;
- device;
- requires_grad;
- finite boolean;
- object id;
- `data_ptr`;
- stride and storage offset, useful for alias/debugging;
- numeric checksum or digest, for example SHA256 over a detached contiguous CPU
  representation;
- optional summary: min, max, mean, sum, L2 norm, number of finite elements.

Comparison modes:

- Structural equality depends on attribute existence, attribute type, list
  length, entry type sequence, sentinel values, Tensor shapes, dtypes, devices,
  strides, and requires_grad.
- Numerical equality depends on Tensor finite flags, digests/checksums, and
  optional pairwise max/mean difference metrics.
- Identity equality depends on attribute object ids, list object ids, entry
  object ids, Tensor object ids, and Tensor `data_ptr` values.

The fingerprint should record `_conv_num`, `_conv_idx`, `_feat_map`,
`_enc_conv_num`, `_enc_conv_idx`, `_enc_feat_map`, and wrapper-level `mean`,
`std`, and `model` bindings. It can optionally include model parameter/buffer
identity summaries to prove that transaction logic did not replace the VAE model,
but it should not clone or serialize full weights.

## 9. Transaction feasibility conclusion

### VAE snapshot/restore primitive

Selected conclusion: B.

Incremental cached decode looks feasible, but real GPU differential experiments
are still required before claiming the transaction is correct.

Code evidence:

- The current wrapper already exposes `use_cache=True` and dispatches to
  `WanVAE_.cached_decode()` at `utils/wan_wrapper.py:110-129`.
- `cached_decode()` preserves `_feat_map` after returning and does not call
  `clear_cache()` at `wan/modules/vae.py:571-593`.
- Cache state appears localized to `WanVAE_` attributes and Python list entries
  touched by `Decoder3d`, `ResidualBlock`, and `Resample`; no `register_buffer`
  cache was found.
- The first-frame sentinel and two temporal upsample stages provide a plausible
  incremental temporal mapping: block 0 gives 9 pixel frames, later 3-latent
  blocks give 12 newly returned pixel frames.

Unresolved risks:

- No A100 Path A/Path B tensor comparison has been run for rejected draft rollback
  followed by target fallback decode.
- Exception paths have no built-in `try/finally`; wrapper transaction must handle
  dirty partial cache state.
- `use_cache=False` destroys any existing incremental cache and returns only a
  new clean state, so it cannot be mixed into an incremental transaction unless
  that destruction is intentional and snapshotted.
- The exact pixel boundary semantics need GPU confirmation with real latent
  tensors.
- Memory behavior after repeated reject/rollback loops is unknown.

Minimum snapshot set for `cached_decode` rollback:

Must restore:

- `WanVAE_._conv_idx` binding and contents. `cached_decode()` rebinds it per
  latent frame at `wan/modules/vae.py:581`, and decoder layers mutate the passed
  one-element list as a scratch index.
- `WanVAE_._feat_map` binding and all call-before entry references. The current
  code does not rebind `_feat_map` inside `cached_decode()`, but it replaces list
  entries through decoder/residual/resample cache writes.

Fingerprint only for `cached_decode` static minimal rollback:

- `WanVAE_._conv_num`: read only for initialized cache shape; no
  `cached_decode()` write found.
- `WanVAE_._enc_conv_num`: no `cached_decode()` write found.
- `WanVAE_._enc_conv_idx`: no `cached_decode()` write found.
- `WanVAE_._enc_feat_map`: no `cached_decode()` write found.
- `WanVAEWrapper.mean`: used to create per-call scale; no attr write found.
- `WanVAEWrapper.std`: used to create per-call scale; no attr write found.
- `WanVAEWrapper.model`: used as the owner of `cached_decode`; no wrapper attr
  write found.

If the implementation uses `decode()` or `clear_cache()` inside a transaction
instead of pure `cached_decode()`, this minimum set is no longer sufficient
because `clear_cache()` rebinds all decode and encode cache attributes.

Shallow vs clone:

- Current code does not show old cache Tensor objects being modified in place.
  Existing entries are read and new Tensor objects are assigned into list slots.
- Therefore, shallow copying the old `_feat_map` entry references is the static
  minimum rollback scheme.
- Cloning old cache Tensor entries would break identity equality, because
  rollback would restore different Tensor objects and likely different
  `data_ptr`s.
- The A100 experiment must compare old cache Tensor digests before rejected draft
  decode and after rejected draft decode, before rollback. If those digests
  change while object identity is unchanged, then the code or kernel path has an
  in-place mutation and clones become required for numerical rollback.
- Only a measured or code-proven in-place mutation should require Tensor clones.

List and binding restoration:

- To restore identity equality, restore original list object identity for
  `_feat_map` and `_conv_idx`, then restore their entries/values.
- To restore only clean-state/value equivalence, rebinding equivalent new lists is
  enough, but this is weaker and will fail identity fingerprints.
- Encode-cache bindings are fingerprint-only for pure `cached_decode`; they become
  rollback targets only if a future wrapper mixes in `decode()`, `encode()`, or
  `clear_cache()`.

Wrapper-only feasibility:

- Static evidence suggests wrapper-only transaction is possible because the
  mutable state is reachable through wrapper-owned `model` attributes.
- It should not require modifying `wan/modules/vae.py` if GPU Path A/Path B passes
  with wrapper snapshot/restore.
- It does touch the current frozen boundary semantically by adding a new VAE
  transaction wrapper around an existing frozen VAE implementation, but it does
  not require editing frozen VAE source.
- This is not confirmed enough for conclusion A.

### Controller/evaluator integration

Selected conclusion: D.

Existing evidence is insufficient to choose a safe integration owner. The VAE
snapshot/restore primitive can be tested independently, but the current
controller/evaluator/committer lifecycle does not yet define who owns a candidate
VAE state after `evaluator.evaluate(candidate)` returns and before
`policy.decide(evaluation)` is known.

The unresolved integration questions are:

- whether evaluator is allowed to return with pending VAE state;
- whether accepted candidates may be decoded twice;
- whether permanent VAE cache advancement belongs to the existing committer;
- whether controller window rollback must include VAE state;
- how Transformer commit and VAE commit become atomic.

## F4B Implementation Gate

| Gate item | Status | Evidence / next action |
|---|---|---|
| Calling chain confirmed | PASS | Wrapper dispatch and inference call sites traced above. |
| Cache state inventory complete | PASS | Current wrapper/VAE attrs, decoder layers, `register_buffer`, and demo cache paths audited. |
| First-block frame mapping confirmed | PASS | Code-derived: 3 latent frames from clean cache produce 9 pixel frames; needs real GPU validation. |
| Later-block frame mapping confirmed | PASS | Code-derived: later 3-latent blocks produce 12 new pixel frames; needs real GPU validation. |
| `cached_decode` input semantics confirmed | PASS | Must receive contiguous sequence from current cache frontier. |
| `cached_decode` output semantics confirmed | PASS | Returns current-call pixels, not cumulative prefix. |
| Reject rollback state set confirmed | PASS | Static minimum for pure `cached_decode`: restore `_conv_idx` binding/content and `_feat_map` binding/entry references. |
| Target fallback decode post-state semantics confirmed | UNKNOWN | Static semantics are clear, but permanent owner is not selected. |
| Fingerprint scheme clear | PASS | Section 8 defines structural, numerical, and identity fields. |
| Path A / Path B experiment clear | PASS | Section 7 fixes latent sources and VAE-only comparison path. |
| Wrapper-only VAE snapshot primitive feasibility confirmed | UNKNOWN | Static evidence supports it, but A100 differential test is required. |
| Evaluator returns before candidate VAE state handling | UNKNOWN | Current `Evaluator` contract requires temporary state restored before return; pending candidate state would violate it unless contract changes. |
| Accept after permanent VAE cache advancement owner | UNKNOWN | Could be evaluator pending transaction, committer re-decode, or coordinator; not selected. |
| Fallback after permanent VAE cache advancement owner | UNKNOWN | Most consistent owner is committer, but not specified yet. |
| Controller window rollback covers VAE | UNKNOWN | Current rollback calls only `committer.rollback()`; VAE is covered only if committer owns it. |
| Transformer commit and VAE commit atomicity | UNKNOWN | No current contract defines cross-state atomic ordering or rollback failure behavior. |
| Accepted candidate duplicate decode allowed | UNKNOWN | Option B requires duplicate decode; option A/C can avoid it but need ownership changes. |
| Controller/evaluator integration feasibility confirmed | UNKNOWN | Section 6 selects D for integration ownership. |
| GPU-only unknowns have experiment plan | PASS | Section 7 covers pixel, cache, identity, old Tensor digest, and memory comparisons. |

F4B may proceed only after the UNKNOWN ownership items above are deliberately
resolved or scoped out. A narrow F4B can implement and test the VAE
snapshot/fingerprint primitive, but production candidate decode integration must
not proceed until evaluator/committer/controller ownership and atomic rollback
semantics are explicit.
