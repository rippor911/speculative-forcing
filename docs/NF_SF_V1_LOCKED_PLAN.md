# NF-SF v1 锁定计划

## 0. 计划状态

本计划是当前项目后续开发的唯一主线。

基础状态：

- 分支：`next-forcing`
- 基准提交：`5b97a46cdf342eebe33aba28721c69fa9a0f15f2`
- 本地负责开发、提交和推送；
- 服务器只负责 `pull --ff-only`、测试、训练和生成实验产物；
- 继续使用单一线性分支，不增加无必要的长期分支。

审计确认当前仓库已经具备 MCP（多视频块预测模块）的主体骨架和经过验证的 speculative runtime（推测式运行框架），无需回到早期提交重写。

------

# 1. 项目目标

在官方 Self-Forcing Wan-1.3B 模型上实现一套 **Next-Forcing-style（类 Next Forcing）多视频块预测架构**：

1. 主干模型生成当前 chunk（视频块）；
2. MCP-1、MCP-2、MCP-3 分别学习预测后续第 1、2、3 个 chunk；
3. 主干、特征融合层和三个 MCP 通过 Flow Matching（流匹配）联合训练；
4. 推理时主干生成当前 chunk，MCP-1 同步生成下一 chunk；
5. 第一阶段采用 all-accept（所有 MCP 输出直接接受），直接检验 MCP 的真实质量；
6. 根据实际结果，再决定是否加入 Self-Forcing/DMD 后训练或 verifier（质量验证器）。

最终需要回答：

> 联合训练后的 MCP 能否生成清晰、连续的未来 chunk，并通过并行去噪获得真实推理加速。

------

# 2. 已锁定的技术决策

| 项目                          | 锁定方案                                                |
| ----------------------------- | ------------------------------------------------------- |
| 初始化权重                    | 官方 `self_forcing_dmd.pt`                              |
| Reference Teacher（参考教师） | 永久冻结，只生成训练 latent 和基线结果                  |
| NF Main（新主干）             | 允许训练                                                |
| chunk 大小                    | 第一阶段固定 3 个 latent frames                         |
| chunk 参数化                  | 代码从配置读取，不允许散落硬编码 `3`                    |
| MCP 深度                      | next1、next2、next3                                     |
| 每个 MCP 层数                 | 3 个 Transformer blocks                                 |
| 主干特征层                    | 代码索引 `[3, 11, 19, 29]`，对应论文第 4、12、20、30 层 |
| MCP 条件                      | multi-layer backbone feature fusion（多层主干特征融合） |
| 主模型 timestep shift         | `s_main = 5`                                            |
| MCP timestep shift            | `s_mcp = 10`                                            |
| MCP loss 权重                 | `[0.5, 0.2, 0.1]`                                       |
| 第一阶段训练                  | Flow Matching 联合训练                                  |
| 第一阶段推理                  | depth-1 并行多步去噪                                    |
| 接受策略                      | all-accept                                              |
| verifier                      | 暂不开发                                                |
| refinement（额外修复）        | 暂不开发                                                |
| DMD（分布匹配蒸馏）           | 暂不加入第一阶段                                        |
| direct history attention      | 暂不加入 v1                                             |

上述 MCP 层数、特征层、timestep shift 和 loss 权重来自 Next Forcing 的默认设计；论文使用主模型与三个 MCP 的 Flow Matching 损失联合训练，推理时只保留 depth-1 MCP 并行生成下一 chunk。

固定 3 个 latent frames 与当前 Self-Forcing chunk-wise 模型设置一致；代码仅保留未来扩展空间，不在第一阶段验证其他大小。

------

# 3. 整体架构

## 3.1 模型角色

### Reference Teacher（参考教师）

```text
官方 self_forcing_dmd.pt
```

用途：

- 离线生成 clean latent 视频；
- 提供 Target-only 基线；
- 不参与梯度更新；
- 不作为最终部署模型。

### NF Main（联合训练后的主干）

用途：

- 预测当前 chunk 的 flow（去噪方向）；
- 提供第 4、12、20、30 层特征；
- 接收主任务和 MCP 的反向梯度；
- 推理时生成当前 chunk；
- 后续若加入 fallback（回退），也使用该主干。

### MCP-1/2/3

用途：

- MCP-1 预测 next1；
- MCP-2 预测 next2；
- MCP-3 预测 next3；
- 三者形成隐藏特征的因果链；
- 默认推理只使用 MCP-1。

------

## 3.2 训练数据流

以历史 `C0、C1`，当前块 `C2` 为例：

```text
历史：C0、C1
当前目标：C2
next1：C3
next2：C4
next3：C5
```

分别独立采样：

```text
noise：ε0、ε1、ε2、ε3
timestep：t0、t1、t2、t3
```

构造：

```text
C2_t0 → NF Main
C3_t1 → MCP-1
C4_t2 → MCP-2
C5_t3 → MCP-3
```

要求：

- 四套 noise 相互独立；
- 四个 timestep 独立采样；
- 主干使用 `s_main=5`；
- MCP 使用 `s_mcp=10`；
- noise 和 timestep 在线生成，不固定使用旧 teacher source noise。

Next Forcing 明确使用 temporally shifted targets（时间平移目标）以及每个未来深度独立的噪声和时间步。

------

## 3.3 多层主干特征条件

主干输入：

```text
clean history + noisy current chunk
```

主干当前块 token 在计算时已经结合：

- 历史 KV cache（历史键值缓存）；
- prompt（文本提示）；
- 当前噪声；
- 当前 timestep。

从主干四个层级提取特征：

```text
h4、h12、h20、h30
```

经过 fusion MLP（特征融合层）：

```text
[h4; h12; h20; h30]
        ↓
      h_fuse
```

MCP-1 使用：

```text
noisy next1 + h_fuse
```

MCP-2 使用：

```text
noisy next2 + MCP-1 hidden
```

MCP-3 使用：

```text
noisy next3 + MCP-2 hidden
```

当前 MCP 不直接读取历史 KV；历史信息通过已经看过历史的主干特征间接传入。该行为已由仓库审计确认。

------

## 3.4 损失

主模型：

\[
L_{\text{main}}
=
\left\|
v_0-(\epsilon_0-C_2)
\right\|_2^2
\]

三个 MCP：

\[
L_k
=
\left\|
v_k-(\epsilon_k-C_{2+k})
\right\|_2^2
\]

总损失：

\[
L = L_{\text{main}} + 0.5L_1 + 0.2L_2 + 0.1L_3
\]

必须分别记录：

```text
main_loss
mcp_depth1_loss
mcp_depth2_loss
mcp_depth3_loss
total_loss
```

不得只记录总 loss。

------

## 3.5 参数更新范围

### Joint（主实验）

训练：

- backbone（主干）；
- shared patch embedding（共享 latent 嵌入层）；
- fusion MLP；
- MCP-1；
- MCP-2；
- MCP-3。

冻结：

- Reference Teacher；
- VAE；
- text encoder（文本编码器）。

### Frozen（短对照）

训练：

- fusion MLP；
- MCP-1/2/3。

冻结：

- backbone；
- shared patch embedding。

Frozen 只作为控制实验，不投入与 Joint 相同的完整训练预算。

------

# 4. 推理流程

新增独立推理路径，不修改旧 oracle（参考入口）。

每轮维护两个状态：

```text
current_state：当前 chunk
next_state：下一 chunk
```

每个去噪 timestep：

```text
NF Main(current_state, history, t)
    → current flow
    → 主干中间特征

MCP-1(next_state, fused features, t)
    → next flow

scheduler：
    current_state → 下一 timestep
    next_state    → 下一 timestep
```

完成全部去噪步骤后：

```text
commit current chunk
commit MCP next chunk
下一轮从 i+2 开始
```

禁止使用：

```text
MCP 一步得到最终 x0
→ 重新加噪
→ Target refinement
```

这属于旧 E0211 路径，不属于本主线。

------

# 5. 里程碑与验收

## M0：规范锁定

修改：

- 更新 `SPEC.md` 或对应主线文档；
- 记录本计划中的模型角色、训练目标和冻结范围；
- 明确 E0209/E0210/E0211 为历史对照，不再作为主方法。

验收：

- 论文设计、仓库现状和项目适配三者明确分开；
- 不声称完整复现原论文；
- 不声称使用真实视频数据训练。

------

## M1：纯 CPU tensor（张量）数据构造

实现：

- `chunk_frames` 参数化；
- next1/2/3 shift；
- independent noise（独立噪声）；
- independent timestep（独立时间步）；
- 有效位置 mask；
- `s_main` 与 `s_mcp` 的采样接口。

验收：

- 使用小 CPU tensor 精确验证每个 future target；
- 三个 MCP noise 不相同；
- 越界位置不进入 loss；
- 相同 seed 可复现；
- `chunk_frames=1/2/3/4` 的纯 tensor 测试通过；
- 真实训练仍只允许 `chunk_frames=3`。

失败条件：

- shift、mask 或 timestep 语义不能唯一确定时停止，不进入真实模型测试。

------

## M2：真实模型 forward 与梯度路径

服务器执行最小真实模型测试。

检查：

- timestep 不限于 1000；
- main 和三个 MCP 输出 shape 正确；
- 无 NaN/Inf；
- bf16 可运行；
- 显存有记录；
- Frozen 和 Joint 的梯度范围正确。

由于 MCP 输出头采用零初始化，至少执行 2～3 个 optimizer steps 后再检查深层梯度。

验收：

| 参数组          | Frozen | Joint  |
| --------------- | ------ | ------ |
| MCP-1/2/3       | 有梯度 | 有梯度 |
| fusion          | 有梯度 | 有梯度 |
| backbone        | 无梯度 | 有梯度 |
| patch embedding | 无梯度 | 有梯度 |

失败条件：

- optimizer 漏参数；
- Joint 模式 backbone 始终无梯度；
- Frozen 模式 backbone 参数发生变化；
- 出现非有限梯度。

------

## M3：单样本过拟合

数据：

- 1 个 teacher 视频；
- 至少包含一个完整的 current + next1/2/3 状态。

输出：

- main 和三个 MCP loss 曲线；
- checkpoint；
- fresh-process restore（新进程恢复）结果；
- 单块解码结果；
- 固定 teacher history 下的 MCP-1 standalone multi-step denoising reconstruction。

验收：

- 四项 loss 均持续下降；
- checkpoint 在新进程恢复后输出一致；
- 固定 teacher history 下，MCP-1 standalone reconstruction 不再只是低频轮廓；
- 结果可解码且无明显数值异常。

停止条件：

- 单样本无法过拟合；
- restore 后结果不一致；
- 未来块 target、RoPE 或 chunk 对齐存在疑问。

单样本失败时禁止扩大数据和训练步数。

M3 不要求正式 depth-1 parallel runtime，不要求 current/next 双状态并行，不要求 KV commit，也不要求 `i+2` 推进；这些边界属于 M6。

------

## M4：短程 Frozen/Joint 对照

使用相同数据、batch、step 数、seed 和验证集。

目的：

> 判断训练 backbone 是否确实优于只训练 MCP。

先进行：

- 100-step 性能和稳定性测试；
- Frozen 短训练；
- Joint 等预算短训练；
- 固定 prompt 的视觉对照。

验收：

- 有可比较的 loss 曲线；
- 有相同 checkpoint 步数的解码视频；
- 记录单步耗时、峰值显存和训练稳定性；
- Joint 的未来块质量或验证 loss 至少有一个明确优于 Frozen。

停止条件：

- Joint 没有任何改善且主干质量下降；
- 训练指标与视频质量完全矛盾且无法解释；
- 出现不稳定或无法恢复的 checkpoint。

------

## M5：现有数据正式训练

第一轮只使用现有：

```text
2048 train
256 validation
```

不立即扩展到 16k。

训练预算通过 M4 的真实 step 耗时确定；初始检查点设置为：

```text
early checkpoint
middle checkpoint
final checkpoint
```

建议对应约：

```text
500 / 2000 / 5000 optimizer steps
```

这些是项目门控点，不是论文规定值。

每个检查点固定生成同一组 prompt。

验收：

- main 与三个 MCP loss 有完整曲线；
- validation 不只记录 latent MSE；
- 固定视频可直接横向比较；
- Joint 主干的标准单块生成能力未明显退化；
- depth-1 并行结果随训练出现可解释改善。

停止条件：

- 到中期 checkpoint 仍与 E0209 一样系统性模糊；
- final checkpoint 没有视觉改善趋势；
- 主干质量明显退化；
- checkpoint resume 不可靠。

正式长训练前必须修复审计发现的 resume 问题：当前 trainer 不恢复 optimizer、scheduler 和 global step，不能承担可靠的中断续训。

------

## M6：并行 all-accept 推理

新增独立入口，例如：

```text
inference_next_forcing.py
```

对比：

- Target-only；
- E0209 one-shot；
- NF-Frozen parallel；
- NF-Joint parallel；
- NF-Joint zero-overhead（丢弃 MCP，仅用训练后主干）。

验收：

- current 与 next 使用完整多步去噪；
- commit 顺序正确；
- 下一轮从 `i+2` 开始；
- 完整视频来自同一条实际生成轨迹；
- 不再把新块拼入旧 teacher 视频；
- 记录主干时间、MCP 时间、KV commit 时间和端到端时间。

------

## M7：第一次方向结论

固定一组未参与训练的 prompts，至少检查：

- 单块清晰度；
- 人物、背景和构图连续性；
- chunk 边界跳变；
- 长视频是否逐块漂移；
- Target-only 与 NF-Joint 的耗时；
- 峰值显存；
- 失败案例。

结果分流：

### A. 单块清晰，完整视频稳定

继续扩大训练或数据规模，并正式评估加速。

### B. 单块清晰，但历史连续性差

考虑 direct clean-history attention（MCP 直接读取历史）或 Self-Forcing 后训练。

### C. 短视频正常，长视频逐渐漂移

加入 self-rollout（模型按真实推理流程生成历史）和 DMD 后训练。Self-Forcing 的目的正是让训练历史来自模型自身，减小训练—推理分布差。

### D. 少数 chunk 失败，多数质量良好

再接入现有 verifier 和 same-noise fallback。

### E. 单块始终模糊

不做 verifier、不扩数据，优先检查：

- future target 对齐；
- timestep；
- noise；
- Flow Matching target；
- gradient；
- MCP 容量；
- 特征条件。

------

# 6. 冻结范围

以下内容第一阶段禁止修改：

- `inference_mcp.py`；
- speculative controller 的状态机；
- longest-prefix acceptance；
- first-rejection 语义；
- same-noise fallback；
- transaction / rollback；
- commit ordering；
- KV index validation；
- trace 基本结构。

当前优先问题是 Draft 质量；在没有新证据前，不重写已经通过测试和真实 GPU smoke 的 speculative 控制框架。

------

# 7. 第一阶段禁止事项

第一阶段不做：

- verifier；
- ImageReward；
- accept/reject；
- refinement；
- DMD；
- self-rollout；
- direct clean-history attention；
- 随机 chunk size 真实训练；
- 16k 数据生成；
- 大规模消融；
- 重写 MCPStack；
- 重写 speculative runtime；
- 修改旧 `inference_mcp.py`；
- 根据 latent MSE 单独下结论。

------

# 8. 开发与实验约束

## Git

- 只使用 `next-forcing`；
- 小步 commit；
- 不使用 `git add .`；
- 不自动 merge；
- 不改写已推送历史；
- 不在服务器直接修改 tracked 源码。

## 本地

负责：

- 代码修改；
- 单元测试；
- 静态检查；
- commit；
- push。

## 服务器

只允许：

```text
git pull --ff-only
运行测试
运行训练
生成 checkpoint、日志和视频
```

服务器发现问题后的处理：

```text
记录日志
→ 本地修复
→ push
→ 服务器 pull --ff-only
→ 重测
```

## 实验记录

每次真实训练必须保存：

- Git SHA；
- config；
- checkpoint 来源及 SHA256；
- prompt 划分；
- seed；
- GPU 数量；
- global batch；
- optimizer steps；
- 单步耗时；
- 峰值显存；
- loss 曲线；
- 固定 prompt 视频。

没有这些信息的实验不得作为后续决策依据。

------

# 9. 第一批施工范围

M0 文档锁定后，下一次 Codex 施工只包含 M1：

1. `chunk_frames` 参数化；
2. next1/2/3 shift；
3. independent noise；
4. independent timestep；
5. valid mask；
6. `s_main` / `s_mcp` 采样接口；
7. 纯 CPU tensor tests。

M1 不包含：

- main + depth1/2/3 Flow Matching loss；
- Frozen/Joint 参数开关；
- 真实模型 forward；
- 梯度检查；
- M3 训练入口；
- 正式并行推理；
- 大规模训练。

M1 验收后再进入 M2；不得在 M1 中预做 M2/M3 施工。
