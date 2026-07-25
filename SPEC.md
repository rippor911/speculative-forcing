# SPEC.md — Self-Forcing MCP Next-Forcing 视频生成加速项目规范

> 状态：Canonical / 项目唯一主线规范  
> 适用仓库：`rippor911/speculative-forcing`  
> 集成基线：`feat/speculative-mcp-adapter @ a5c8270`  
> 当前实现阶段：Milestone F1 complete / F2 next  
> 研究目标：以 Self-Forcing + MCP/Next-Forcing 为基础，实现可插拔 verifier 的视频 speculative decoding 框架，并验证质量—速度折中  
> 论文目标：面向 CVPR 2027 的研究原型与实验体系  
> 最高优先级约束：**任何开发任务都必须直接服务于主线，不得为了通用性、抽象完整性或测试数量而脱离研究目标。**

---

## 0. 本规范的地位

本文件是项目后续设计、实现、审查、实验和 Git 操作的统一基准。

任何新任务开始前，都必须先回答：

> 这项工作是否直接服务于  
> `MCP draft → verifier → acceptance policy → longest-prefix controller → fallback/commit → 质量与加速实验`？

如果不能直接映射到这条链路，且不是让该链路闭环运行所必需的工作，则默认不做。

本规范优先级高于临时讨论、一次性建议和局部实现偏好。若后续需要修改主线、里程碑或架构边界，必须先更新本文件，再继续编码。

---

# 1. 项目目标

## 1.1 核心研究目标

以 Self-Forcing 作为视频自回归生成 backbone，以 MCP heads 预测未来多个 chunk，形成 Next-Forcing 风格的 draft，再通过可插拔 verifier 和 acceptance policy 决定：

- 接受 MCP draft，并将其正式提交到生成状态；
- 或拒绝 MCP draft，由同一 backbone 使用相同 source noise 重新生成 target block；
- 第一次拒绝后，本轮更深 MCP draft 全部失效；
- 在保证质量可控的前提下减少 target backbone 的实际生成次数，实现视频生成加速。

最终研究问题：

1. MCP heads 预测的未来 chunk 是否足以作为低成本 draft？
2. draft 质量如何随 MCP depth 增大而退化？
3. 哪类 verifier 能有效区分“可接受 draft”和“应 fallback 的 draft”？
4. verifier、VAE 解码和 fallback 的额外开销是否小于节省的 target 计算？
5. 能否得到稳定、可解释、可复现的质量—速度折中曲线？
6. 相比 ImageReward 等图像级信号，新的 block-level / temporal verifier 是否能更好识别时间一致性与运动质量问题？

## 1.2 工程目标

实现一个固定控制语义、策略可插拔的 speculative framework：

```text
Self-Forcing Backbone
        │
        ├── 当前 anchor chunk
        └── MCP heads → future draft chunks
                           │
                           ▼
                 CandidateEvaluator
                 ├── CandidateDecoder
                 ├── Frame/Block Scorer
                 └── ScoreAggregator
                           │
                           ▼
                  AcceptancePolicy
                           │
                           ▼
                SpeculativeController
                 ├── accept → commit draft
                 └── reject → target fallback → commit target
```

框架必须支持：

- scripted policy 正确性验证；
- MCP proposer；
- longest-prefix acceptance；
- same-noise target fallback；
- trace；
- 可插拔 decoder / scorer / aggregator / policy；
- 配置驱动消融；
- 后续 ImageReward、latent verifier、learned verifier 接入；
- 真实 Wan backend 与 frozen MCP baseline 的差分验证。

## 1.3 非目标

当前阶段不追求：

- 通用视频生成框架；
- 支持任意 attention mode；
- 支持多 batch、CFG batch expansion、I2V 等未验证配置；
- 支持任意模型、任意 KV layout；
- 重写 Self-Forcing 训练代码；
- 修复所有 MCP 训练问题后才开发推理框架；
- 一开始就实现 learned verifier；
- 一开始就实现 verify/generate 并行；
- 一开始就声明加速收益；
- 追求单元测试数量本身；
- 为未来假设场景提前构建大型抽象。

---

# 2. 项目来源与差异

## 2.1 基础来源

### Self-Forcing

项目 backbone 和训练/推理基础来自 Self-Forcing：

- 自回归 chunk-by-chunk 视频生成；
- diffusion / flow-based block generation；
- causal KV cache；
- few-step / one-step 推理；
- target backbone 负责 anchor 与 fallback。

### MCP / Next-Forcing

当前项目使用 MCP heads 预测未来多个 chunk：

- depth 1：预测 next¹ chunk；
- depth 2：预测 next² chunk；
- depth 3：预测 next³ chunk；
- MCP heads 基于 anchor backbone 中间特征和未来 source noise 形成 draft；
- 当前形态接近 Next-Forcing 的多层未来 chunk 预测思想，但代码和权重并非直接来自公开 Next-Forcing 实现。

### SDVG

SDVG 提供路由基线思想：

- VAE 解码 draft；
- ImageReward 逐帧评分；
- min-frame 聚合；
- 固定阈值；
- 拒绝后 target 重生成；
- same-noise fallback。

## 2.2 与 SDVG 的关键差异

| 维度 | SDVG | 当前项目 |
|---|---|---|
| Drafter | 独立小模型 | 同一 backbone 上的 MCP heads |
| Target | 独立大模型 | Self-Forcing backbone |
| 第 0 块 | 可视为 drafter/target 控制的一部分 | 天然由 backbone 生成，是 anchor |
| KV | drafter 与 target 各自维护 | 同一 runtime 中共享 backbone/cache |
| Draft 深度 | 逐块 drafter | MCP next¹/next²/next³ |
| 接受语义 | 逐块路由 | longest accepted prefix |
| 研究空间 | 图像奖励路由 | block/temporal verifier、depth-aware policy、MCP-specific routing |

不得直接复制 SDVG 的双模型代码结构。可借鉴其 verifier 基线和实验方法，但必须保持当前 MCP + shared-backbone 架构。

---

# 3. 冻结的核心语义

## 3.1 Longest-prefix acceptance

对一个 anchor 产生的 MCP draft：

```text
anchor
→ depth 1
→ depth 2
→ depth 3
```

控制规则：

- depth 1 接受：commit depth 1，继续验证 depth 2；
- depth 2 接受：commit depth 2，继续验证 depth 3；
- 任一 depth 拒绝：不 commit 该 draft；
- 使用该 candidate 的原始 source noise 生成 target fallback；
- commit fallback；
- 本轮更深 draft 全部 invalidated；
- 下一未提交 block 成为新的 anchor。

禁止出现：

```text
accept, reject, accept
```

因为 reject 后上下文已经变化，更深 draft 条件失效。

## 3.2 Anchor 语义

- anchor 始终由 backbone 生成；
- block 0 天然是 target/backbone anchor；
- 不额外实现“首块强制拒绝”；
- anchor 必须先正式 commit，然后才能处理其 MCP drafts。

## 3.3 Commit 语义

只有 `SpeculativeController → Committer` 路径可以永久修改：

- Transformer KV；
- output latent buffer；
- committed block bookkeeping；
- generation ordering state。

Evaluator、Scorer、Aggregator、Policy 均不得永久修改生成状态。

## 3.4 Fallback 语义

Fallback 必须：

- 使用被拒绝 candidate 的同一个 `source_noise` 对象；
- 使用当前已 commit 上下文；
- 运行 target backbone one-step；
- 不复用 draft latent；
- 返回与目标 output slice shape/dtype/device/layout 完全一致的 latent；
- 由 controller/committer 正式提交。

## 3.5 Transformer KV 语义

- 未接受的 draft 不得被正式 commit 到 persistent Transformer KV；
- verifier 本身不得写 Transformer KV；
- 真实 proposal/fallback helper 若临时触碰共享 KV，则由 adapter-local transaction 恢复；
- commit 后的 KV 由 window transaction 保护，失败时 rollback，成功时 complete；
- 不为 ImageReward verifier 额外设计 Transformer KV rollback 路径。

## 3.6 VAE cache 语义

未来 causal VAE block decode 若推进 cache：

```text
begin VAE transaction
→ decode draft
→ score
→ accept: commit VAE transaction
→ reject: rollback
         → decode target fallback
         → commit fallback decode
```

VAE transaction 只在完成真实 VAE cache 审计后实现，不能凭假设编码。

---

# 4. 核心模块边界

## 4.1 DraftProvider / ProposalSource

责任：

- 使用 Self-Forcing backbone 生成 anchor；
- 使用 MCP heads 生成实际可用的 future drafts；
- 保留 source noise；
- 返回按 depth 排序的 `ProposalBatch`。

不负责：

- 解码；
- 评分；
- 接受决策；
- fallback；
- 永久 commit；
- 阈值处理。

## 4.2 CandidateDecoder

责任：

- 将 candidate latent 解码为 scorer 所需表示；
- 第一版真实实现为 Wan causal VAE block decode；
- 管理自身 decoder transaction 接口。

不负责：

- 打分；
- 聚合；
- 决策；
- target fallback；
- Transformer KV。

## 4.3 BlockScorer / FrameScorer

责任：

- 对 decoder 输出产生逐帧或逐块分数；
- 返回原始评分及必要 metadata。

可替换实现：

- fake/scripted scorer；
- ImageReward；
- CLIP/VisionReward；
- latent error；
- learned block verifier；
- temporal quality model。

不负责：

- 决策；
- commit；
- rollback；
- 调 generator。

## 4.4 ScoreAggregator

责任：

- 将逐帧或多维评分聚合为 block score。

第一版：

- `min-frame`；
- `mean-frame` 作为消融。

未来：

- percentile；
- depth-aware aggregation；
- temporal consistency aggregation；
- learned aggregation。

不负责：

- 阈值；
- generator；
- cache；
- fallback。

## 4.5 CandidateEvaluator

推荐为组合对象：

```text
CandidateDecoder
→ BlockScorer
→ ScoreAggregator
→ ScoreResult
```

它只负责评估，不负责接受决策。

## 4.6 AcceptancePolicy

责任：

- 根据 `ScoreResult`、depth、预算等上下文返回 `Decision`。

第一版：

- `always_accept`；
- `always_reject`；
- `reject_at_depth`；
- `fixed_threshold`。

后续：

- depth-specific threshold；
- adaptive threshold；
- budget-aware policy；
- learned policy。

Policy 禁止：

- 调 generator；
- 调 VAE；
- 修改 KV；
- commit/rollback；
- 丢弃未来 draft；
- 修改 output。

## 4.7 FallbackGenerator

责任：

- 用 candidate 的 source noise 生成 target replacement；
- 返回 `FallbackResult`。

不负责：

- 决定何时 fallback；
- 修改 output；
- 修改 controller cursor；
- 处理更深 draft invalidation。

## 4.8 SpeculativeController

固定核心，只负责：

- anchor/draft 顺序；
- longest-prefix 语义；
- accept/reject 状态机；
- fallback 调用；
- commit；
- invalidation；
- trace；
- 异常时 transaction rollback。

Controller 中禁止出现：

- ImageReward 类名；
- 具体阈值；
- 具体 VAE 类；
- scorer 实现细节；
- 模型路径；
- checkpoint 路径。

## 4.9 Trace

每个状态转换必须记录：

- anchor；
- candidate；
- depth；
- score；
- threshold；
- decision；
- fallback；
- invalidated drafts；
- commit；
- timing；
- cache/index 状态；
- VAE transaction 结果。

Trace schema 必须带版本号。

---

# 5. 数据接口规范

所有核心数据结构优先采用不可变 dataclass。

## 5.1 DraftCandidate

必须至少包含：

```python
@dataclass(frozen=True)
class DraftCandidate:
    block: BlockRef
    depth: int
    latent: torch.Tensor
    source_noise: torch.Tensor
```

约束：

- `depth` 为严格正整数；
- `latent` 与目标 output slice exact-compatible；
- `source_noise` 保留对象 identity；
- 不允许广播 shape；
- 不允许静默 dtype/device 转换。

## 5.2 ScoreResult

建议字段：

```python
@dataclass(frozen=True)
class ScoreResult:
    per_frame_scores: tuple[float, ...]
    block_score: float
    scorer_name: str
    aggregator_name: str
    metadata: Mapping[str, Any]
```

## 5.3 Decision

建议字段：

```python
@dataclass(frozen=True)
class Decision:
    accepted: bool
    policy_name: str
    reason: str
    threshold: float | None
    metadata: Mapping[str, Any]
```

## 5.4 FallbackResult

必须至少包含：

```python
@dataclass(frozen=True)
class FallbackResult:
    block: BlockRef
    latent: torch.Tensor
    source_noise: torch.Tensor
```

要求：

```python
result.block == candidate.block
result.source_noise is candidate.source_noise
```

---

# 6. 当前已完成状态

## 6.1 MCP baseline

已完成：

- Self-Forcing 基础推理；
- 标准 MCP inference 入口；
- MCP depth 1/3 运行；
- checkpoint strict restore；
- vanilla / MCP 输出长度验证；
- 真实 ODE 数据链路；
- MCP 模型结构和 rollout 基础验证。

冻结参考：

- `inference_mcp.py` 是标准 MCP always-accept reference oracle；
- 原则上不得修改；
- 后续差分测试必须以其为基准。

## 6.2 Milestone 1：控制核心

已完成：

- core types；
- interfaces；
- longest-prefix controller；
- scripted policies；
- trace schema；
- fake component tests；
- 43-test milestone；
- 合入 `dev` 基线。

## 6.3 Milestone 2A：runtime state transaction

已完成：

- tensor region snapshot；
- tensor value snapshot；
- object state snapshot；
- CPU/CUDA RNG；
- rollback/complete；
- restore ordering；
- alias/metadata safety；
- server validation。

## 6.4 Milestone 2B1：thin wrappers

已完成：

- proposal wrapper；
- fallback wrapper；
- committer wrapper；
- shared runtime identity；
- wrapper purity；
- server validation。

## 6.5 Milestone 2B2A：runtime orchestration

已完成：

- `SelfForcingMCPRuntime`；
- prepare/proposal/fallback/window transaction 生命周期；
- dynamic next-uncommitted anchor；
- exact latent compatibility；
- storage alias revalidation；
- commit ordering；
- output/bookkeeping ownership；
- 159 tests；
- A100 CUDA RNG 测试通过。

## 6.6 Milestone 2B2B1：Wan mutation audit + pure planner

已完成并冻结：

- Wan helper 调用链审计；
- self-attention KV touched range；
- cross-attention staging prepare 规范；
- `model.freqs` 静态迁移规范；
- backend-owned / runtime-owned descriptor 分离；
- block-0 prepare scratch plan；
- global baseline fail-fast；
- local rolling 仅 descriptor 级记录，不启用真实 backend；
- 209 tests；
- A100 全部通过。

---

# 7. 后续里程碑

后续里程碑以“先补齐可插拔 framework，再接真实模型，再做 verifier”为主线。

## Milestone F1：可插拔 verifier framework

### 目标

在不接真实 Wan/VAE/ImageReward 的情况下，补齐：

- `CandidateDecoder`；
- `BlockScorer`；
- `ScoreAggregator`；
- `CompositeCandidateEvaluator`；
- `FixedThresholdPolicy`；
- 显式 factory/config assembly；
- fake/scripted 组件；
- controller 端到端组合测试。

### 必须回答

- decoder、scorer、aggregator、policy 是否能独立替换？
- controller 是否完全不知道具体 scorer 和 threshold？
- 固定阈值是否能通过配置切换？
- fake evaluator 是否能触发 accept/reject/fallback？
- mean/min 消融是否只需换配置？
- factory 构造错误是否给出明确字段路径？

### 不做

- Wan backend；
- VAE；
- ImageReward；
- checkpoint；
- GPU；
- 新 cache 抽象；
- learned verifier。

### 完成标准

```text
fake decoder
→ fake scorer
→ min/mean aggregator
→ fixed-threshold policy
→ controller
→ accept/reject/fallback/trace
```

完整闭环可通过配置切换，不修改 controller。

## Milestone F2：真实 Self-Forcing MCP adapter/backend

### 目标

实现当前 global baseline 的最小真实 backend：

- `SelfForcingWanMCPBackend`；
- planner descriptor 到 2A state specs 的绑定；
- `model.freqs` 静态迁移；
- staging cross-attention prepare；
- proposal；
- fallback；
- commit；
- 当前 KV layout；
- no local rolling。

### 支持范围

```text
T2V
batch_size = 1
无 CFG batch expansion
schedule = [1000]
local_attn_size = -1
sink_size = 0
MCP depth = 1/2/3
当前 list-of-dicts KV/cross-attention cache
```

### 完成标准

先 fake backend contract tests，再服务器真实 smoke：

- prepare；
- proposal temporary rollback；
- fallback temporary rollback；
- commit + rollback；
- commit + complete；
- live cross-attention `k/v` identity 不变；
- CUDA RNG rollback；
- output/bookkeeping ownership正确。

## Milestone F3：`inference_speculative.py` + scripted GPU parity

### 目标

新增独立推理入口，冻结基线不动。

第一版只支持：

- `always_accept`；
- `always_reject`；
- `reject_at_depth`。

### 验收

#### Always accept

```text
inference_speculative.py --policy always_accept
vs
inference_mcp.py
```

比较：

- 同 checkpoint；
- 同 prompt；
- 同 seed；
- 同 source noise；
- anchor/draft/commit trace；
- latent；
- visible KV；
- indices；
- next anchor；
- 输出尺寸；
- MP4。

#### Always reject

与 vanilla backbone rollout 比较：

- 每个 draft 均 fallback；
- 更深 draft invalidated；
- final KV index 正确；
- latent 差异可解释。

#### Reject at depth

至少验证：

```text
anchor 0
accept depth 1
reject depth 2
invalidate depth 3
next anchor = next uncommitted block
```

## Milestone F4：VAE block decode audit + transaction

### 目标

先审计，再实现：

- causal VAE 是否支持增量 block decode；
- cache 位置；
- snapshot/restore 对象；
- 首块/后续块像素帧数；
- latent block 到 pixel frames 的映射；
- reject 后无重复帧；
- cache fingerprint 一致；
- 无显存持续增长。

### 核心差分

```text
Path A:
clean VAE cache
→ decode target block

Path B:
clean VAE cache
→ decode rejected draft
→ rollback
→ decode same target block
```

要求：

- Path A/B pixel 输出一致；
- 后续 block 输出一致；
- cache fingerprint 一致。

## Milestone F5：SDVG 风格 verifier baseline

### 默认组合

```text
MCP proposer
+ Wan causal VAE block decoder
+ ImageReward per-frame scorer
+ min-frame aggregator
+ fixed-threshold policy
+ same-noise target fallback
+ longest-prefix controller
```

### 阈值

SDVG 的 `-0.7` 只能作为 reference config，不能作为当前默认值。

正式配置：

```yaml
threshold: null
```

或要求 CLI 显式传入，直到完成验证集扫描。

### Smoke

- 1 个 prompt；
- 6 latent frames；
- finite scores；
- min 聚合正确；
- 极低阈值触发 accept；
- 极高阈值触发 reject；
- fallback 正常；
- VAE cache 正常。

## Milestone F6：阈值标定与基础消融

必须至少覆盖：

| 维度 | 对照 |
|---|---|
| 路由信号 | ImageReward / random / always accept |
| 聚合 | min / mean |
| MCP depth | 1 / 2 / 3 |
| 阈值 | 多个固定阈值 |
| depth 策略 | shared / depth-specific |
| fallback | immediate target |
| verification | sequential |

产出：

- acceptance rate；
- prefix length；
- fallback calls；
- quality；
- latency；
- peak memory；
- 质量—速度曲线。

## Milestone F7：新 verifier / 研究创新

优先研究：

- latent-space verifier；
- backbone feature verifier；
- block-level learned verifier；
- temporal consistency verifier；
- depth-aware verifier；
- learned aggregation；
- adaptive threshold；
- verifier cost-aware policy。

创新评估必须相对：

- always accept；
- always reject；
- random；
- ImageReward + min + fixed threshold；
- target-only；
- draft-only。

---

# 8. 当前下一步

当前唯一下一阶段：

> **Milestone F2：真实 Self-Forcing MCP adapter/backend。**

本阶段不得直接跳到：

- VAE；
- ImageReward；
- learned verifier；
- 大规模实验；
- local rolling；
- `inference_speculative.py` scripted parity；
- 阈值标定；
- 继续横向扩展 verifier framework。

完成 F2 后，进入 F3 scripted parity，将框架接到真实 MCP Next-Forcing 推理闭环。

---

# 9. 配置规范

推荐配置结构：

```yaml
speculative:
  controller:
    type: longest_prefix

  proposal:
    type: self_forcing_mcp
    depth: 3

  evaluator:
    decoder:
      type: fake
    scorer:
      type: scripted
    aggregator:
      type: min_frame

  acceptance:
    type: fixed_threshold
    threshold: null
    depth_thresholds: null

  fallback:
    type: target_regenerate
    reuse_candidate_noise: true

  verification:
    mode: sequential

  trace:
    enabled: true
    schema_version: 1
```

约束：

- 所有消融因素来自配置；
- 不使用任意 Python import-path registry；
- 采用显式 factory map；
- 未知 type fail fast；
- 缺失 threshold 给出字段路径；
- 不静默使用论文魔数；
- controller type 当前只允许 `longest_prefix`。

示例工厂：

```python
POLICY_FACTORIES = {
    "always_accept": build_always_accept,
    "always_reject": build_always_reject,
    "reject_at_depth": build_reject_at_depth,
    "fixed_threshold": build_fixed_threshold,
}
```

---

# 10. Trace 规范

单次 candidate 事件至少记录：

```json
{
  "schema_version": 1,
  "mode": "image_reward",
  "block_start": 6,
  "block_index": 2,
  "depth": 2,
  "per_frame_scores": [-0.42, -0.91, -0.35],
  "block_score": -0.91,
  "threshold": -0.7,
  "accepted": false,
  "fallback_used": true,
  "source_noise_reused": true,
  "invalidated_future_starts": [9],
  "transformer_cache_before": 9360,
  "transformer_cache_after": 14040,
  "vae_transaction": "rolled_back",
  "timing_ms": {
    "draft": 0.0,
    "decode": 0.0,
    "score": 0.0,
    "fallback": 0.0,
    "commit": 0.0
  }
}
```

汇总指标至少包含：

```text
draft_candidates
accepted
rejected
invalidated
acceptance_rate
prefix_length_per_anchor
fallback_calls
draft_time
vae_decode_time
score_time
fallback_time
commit_time
total_generation_time
peak_cuda_memory
```

---

# 11. 实验规范

## 11.1 基线

必须包含：

- target-only / vanilla Self-Forcing；
- frozen MCP always-accept；
- speculative always-reject；
- random routing；
- ImageReward baseline；
- 新 verifier。

## 11.2 计时

正式速度实验必须：

- 排除模型加载；
- 排除首次 warmup；
- 相同 prompt、seed、分辨率、帧数；
- 至少重复 3 次；
- 报告均值和标准差；
- 分项记录 generator、MCP、VAE、scorer、fallback、commit；
- 报告 peak CUDA memory；
- 不使用历史零散运行时间作为论文加速比。

## 11.3 质量

至少记录：

- 视觉质量；
- 文本对齐；
- 时间一致性；
- 运动自然性；
- draft-target latent 差异；
- acceptance depth 分布；
- reject block 类型；
- 后续 block 传播误差。

## 11.4 诊断实验

在正式 verifier 前，保存同一 candidate 的：

```text
MCP draft latent
target fallback latent
decoded draft frames
decoded target frames
depth
prompt
context block
```

用于回答：

- depth 1/2/3 的误差分布；
- 哪些内容更容易被接受；
- verifier 应使用 latent、feature 还是 pixel；
- reject 是否集中在运动/边界/文本复杂 prompt。

## 11.5 已知限制

当前训练存在跨 chunk 依赖问题。它不阻塞 framework，但所有质量结论必须区分：

```text
MCP 本身没学好
vs
verifier 路由失败
```

不得将二者混为同一个问题。

---

# 12. Git 规约

## 12.1 分支

- `dev`：功能集成基线；
- `main`：最终同步；
- 每个 milestone 使用独立 feature branch；
- 未审查前不合并 `dev`；
- 不直接在 `main` 开发。

## 12.2 安全操作

禁止：

```text
git add .
git add -N
git reset
```

评审 tracked 文件：

```bash
git diff --check -- <files>
git diff --stat -- <files>
git diff -- <files>
```

评审 untracked 文件：

- 直接上传完整文件；
- 或复制完整文件到 review directory；
- 或在 Git Bash 使用：

```bash
git diff --no-index /dev/null <file> || true
```

## 12.3 暂存

只允许显式路径：

```bash
git add -- \
  path/to/file1 \
  path/to/file2
```

提交前检查：

```bash
git diff --cached --check
git diff --cached --stat
git diff --cached --name-status
git status --short
```

## 12.4 Commit

- 每个 commit 只完成一个可审查目标；
- 文档、测试、功能可以拆分；
- 不一次提交 controller、ImageReward、CLI、实验脚本；
- Codex 不得自动 commit；
- Codex 不得自动 push；
- 未明确授权不得 merge。

## 12.5 禁止入库

- checkpoints；
- MP4；
- profiler logs；
- unittest logs；
- 大型中间 tensor；
- model weights；
- 临时 review artifacts。

---

# 13. 代码规约

1. Controller 不出现具体 verifier 名称。
2. Policy 不出现 generator、KV、VAE、视频保存。
3. Scorer 不做接受决策。
4. Aggregator 不读 threshold。
5. Fallback 不决定何时调用。
6. 所有 persistent state 修改必须经过 runtime/committer。
7. 所有 public interface 写明“负责什么”和“不负责什么”。
8. 不传递超大可变 dict。
9. 复杂状态用不可变 dataclass。
10. 不复制 pipeline private helper 实现体；通过 adapter 包装。
11. `inference_mcp.py` 冻结。
12. 不顺手重构训练代码。
13. 不为未验证配置增加兼容路径。
14. 当前 global baseline fail fast，禁止静默 fallback。
15. 每个新增策略必须有实现、配置示例、测试和文档说明。

---

# 14. 测试规约

## 14.1 测试层次

### 纯单元测试

- types；
- controller；
- policies；
- aggregators；
- factory；
- transaction；
- planner。

### fake integration

- fake decoder；
- fake scorer；
- fake fallback；
- fake backend；
- controller complete loop。

### server smoke

- 真实 cache；
- 真实 checkpoint；
- 真实 MCP proposal；
- 真实 fallback；
- 真实 CUDA RNG；
- 真实 VAE；
- 真实 ImageReward。

### parity

- always-accept vs `inference_mcp.py`；
- always-reject vs vanilla；
- reject-at-depth trace。

## 14.2 通过标准

单元测试通过不等于 milestone 完成。

Milestone 完成必须同时满足：

- 代码审查；
- 测试审查；
- 本地测试；
- server test；
- 真实 smoke 或 parity（若该 milestone 涉及真实模型）；
- Git 范围正确；
- 文档更新；
- 无未解释的状态差异。

## 14.3 不追求测试数量

测试只验证研究闭环和状态正确性。不得为了增加数字扩展不支持配置或构造无实际风险的边界。

---

# 15. 主线偏离检查表

每个新任务开始前必须逐项检查：

- [ ] 是否直接服务于 MCP draft？
- [ ] 是否直接服务于 verifier 或 acceptance policy？
- [ ] 是否直接服务于 fallback/commit 闭环？
- [ ] 是否是 scripted parity、VAE transaction 或真实推理入口所必需？
- [ ] 是否会帮助后续质量—速度实验？
- [ ] 是否会帮助 verifier 消融或论文贡献？
- [ ] 是否在当前 milestone 范围内？
- [ ] 是否避免扩展未验证配置？
- [ ] 是否避免为了通用性增加复杂度？
- [ ] 是否有明确完成标准？

出现以下情况即视为偏离：

- 为 local rolling 持续开发，而真实 baseline 不使用；
- 支持多 batch/CFG，但没有当前实验需要；
- 持续完善 planner，而真实 framework 仍未闭环；
- 提前实现大型 learned verifier；
- 未完成 scripted parity 就开始跑正式速度；
- 未审计 VAE 就接 ImageReward；
- 以“测试更多”为目标，而不是验证假设；
- 重构训练代码但与 speculative inference 无直接关系；
- 把 SDVG 双模型结构照搬到 shared-backbone MCP 项目。

---

# 16. 决策优先级

当多个任务竞争时，按以下顺序：

1. 让完整 MCP speculative framework 闭环；
2. 保证 longest-prefix / fallback / commit 正确；
3. 真实 MCP parity；
4. VAE transaction；
5. ImageReward baseline；
6. 阈值标定；
7. 新 verifier；
8. 性能优化；
9. 通用化和代码美化。

---

# 17. 当前项目状态摘要

```text
Self-Forcing baseline                         完成
MCP training/inference prototype             完成
Standard MCP inference oracle                完成
Speculative control core                     完成
Scripted policies                            完成
Trace schema                                 完成
Runtime state transactions                   完成
MCP thin wrappers                            完成
SelfForcingMCPRuntime orchestration           完成
Wan mutation audit + pure planner            完成并冻结
Pluggable evaluator stack                    完成
Real Wan backend                             下一步
inference_speculative.py                     未完成
Scripted GPU parity                          未完成
VAE cache audit/transaction                  未完成
ImageReward baseline                         未完成
Threshold calibration                        未完成
New verifier                                 未完成
Formal quality-speed experiments             未完成
```

---

# 18. 下一步任务边界

下一任务只能做 Milestone F2。

允许：

- `SelfForcingWanMCPBackend`；
- planner descriptor 到 2A state specs 的绑定；
- `model.freqs` 静态迁移；
- staging cross-attention prepare；
- proposal；
- fallback；
- commit；
- 当前 KV layout；
- no local rolling。

禁止：

- 修改 `inference_mcp.py`；
- 修改 frozen controller semantics；
- 修改 runtime transaction；
- 修改 Wan planner；
- 接 VAE；
- 接 ImageReward；
- 加 checkpoint；
- 用 GPU；
- 实现 learned verifier；
- 扩展 local rolling；
- 继续扩展 F1 verifier framework；
- commit/push/merge，除非审查后明确授权。

完成后必须立即进入 scripted parity，不再横向扩展 framework。

---

# 19. 变更管理

本规范发生以下变化时，必须显式记录：

- 项目研究假设变化；
- 默认 verifier 变化；
- controller invariant 变化；
- fallback 语义变化；
- baseline 配置变化；
- 里程碑重排；
- 支持范围扩大；
- 实验指标变化。

更新格式：

```text
Date:
Change:
Reason:
Evidence:
Affected milestones:
Backward compatibility:
```

未经更新 SPEC，不得以“临时方便”为理由改变主线。

---

# 20. 最终成功标准

项目达到可投稿研究原型，至少满足：

1. MCP Next-Forcing draft 可稳定运行；
2. speculative controller 完整闭环；
3. scripted parity 通过；
4. causal VAE transaction 正确；
5. ImageReward baseline 可复现；
6. threshold 扫描得到质量—速度曲线；
7. 至少一种新 verifier 或 policy 明显优于基础路由；
8. 报告 acceptance、fallback、latency、quality、memory；
9. 结果可复现；
10. 代码模块边界清晰；
11. 所有借鉴与差异有 provenance；
12. 不把 MCP 训练缺陷与 verifier 缺陷混为一谈；
13. 不声称未经验证的加速；
14. 论文贡献直接围绕 MCP speculative video generation，而不是工程框架本身。

---

## 一句话主线

> **用 Self-Forcing backbone + MCP heads 实现 Next-Forcing 风格的多 chunk draft，通过可插拔 verifier 和 longest-prefix speculative controller 决定接受或 same-noise target fallback，最终研究新的视频 block verifier 与质量—速度折中。**
