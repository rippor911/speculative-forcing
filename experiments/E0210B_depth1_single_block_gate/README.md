# E0210B：Depth-1 单块替换诊断

本实验严格停留在 depth-1 主线，不训练新模型，不启用 depth-2，也不引入 verifier。

目的：区分两类问题：

1. 单个 depth-1 draft block 本身已经模糊、丢主体；
2. 单块尚可，但四个分别基于教师历史得到的 draft 拼在一起互不兼容。

对 legacy sample 004、005，各输出：

- `target`：完整教师视频；
- `e0209_all4`：四个预测块一起替换，复现上一阶段失败现象；
- `e0209_anchor0` 到 `e0209_anchor3`：每次只替换一个未来块，其余块全部保持教师结果。

共 2 个样本、8 个预测状态、12 个视频。

运行：

```bash
GPU=0 bash experiments/E0210B_depth1_single_block_gate/run.sh
```

查看：

```bash
bash experiments/E0210B_depth1_single_block_gate/inspect.sh
```

打开：

```text
experiments/E0210B_depth1_single_block_gate/results/review.html
```

打包：

```bash
bash experiments/E0210B_depth1_single_block_gate/package_results.sh
```

决策：

- 单块也失败：保持 depth-1，检查损失目标、latent 目标语义或模型表达能力；
- 单块通过而 all4 失败：下一步做 depth-1 顺序/on-policy 历史诊断；
- 两者都通过：再做 depth-1 closed-loop gate，仍不直接进入 depth-2。
