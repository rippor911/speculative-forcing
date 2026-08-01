# E0211 v2（shared-noise control） — MCP Draft + Target refinement curve

该实验严格对应交接计划书中的唯一下一步：

```text
Draft + 0 Target refinement
Draft + 1 Target refinement
Draft + 2 Target refinement
Draft + 3 Target refinement
完整 Target 4-step
```

目的不是继续训练，也不是提前接 verifier，而是判断当前模糊的 depth-1 Draft
能否作为 Target 的粗初始化。

## 设计

对每个样本和 anchor 固定：

- prompt
- seed
- source noise
- Target 历史
- anchor
- VAE 解码设置

Refinement 使用：

```text
MCP draft x0
→ 在剩余 Target 轨迹的起始 timestep 重新加噪
→ 运行 Target schedule suffix
```

例如：

```text
1-step:  re-noise 到 625，运行 [625]
2-step:  re-noise 到 833.33，运行 [833.33, 625]
3-step:  re-noise 到 937.5，运行 [937.5, 833.33, 625]
```

## 安装到服务器

在仓库根目录执行：

```bash
mkdir -p experiments/E0211_target_refinement_curve
```

把以下两个文件放入该目录：

```text
run_refinement_curve.py
run.sh
```

然后：

```bash
chmod +x experiments/E0211_target_refinement_curve/run.sh
```

## 运行

建议只开一个 tmux：

```bash
cd /home/dataset-assist-0/luojy/efficiency/rippor/speculative-forcing
tmux new -s e0211
bash experiments/E0211_target_refinement_curve/run.sh
```

退出但保留：

```text
Ctrl-b
d
```

查看：

```bash
tmux capture-pane -pt e0211:0 -S -100
tail -f experiments/E0211_target_refinement_curve/run.log
```

## 必须产物

```text
report.json
metrics.csv
review.html
videos/
timing.json
checkpoint_contract.json
latents.pt
manual_review_template.csv
```

打开 `review.html` 后，人工填写：

```text
manual_review_template.csv
```

重点检查：

- 最差帧是否清晰；
- 主体是否保留；
- 替换块边界是否闪烁。

## 决策

```text
1 次 refinement 通过
→ 进入 MCP + Target 1-step hybrid

2 次 refinement 通过
→ 先核算完整 wall-clock

3 次或完整 Target 才通过
→ 当前 MCP 加速价值有限，再讨论训练目标、loss 或架构
```

## 说明

`measured_step_speedup` 只是局部计算比值：

```text
Target 4-step 时间
/
(MCP proposal forward + refinement 时间)
```

它不是完整视频端到端加速结论。最终加速必须在 hybrid runtime 中重新测量。


## v2 修正

v1 的每个 refinement 分支使用不同随机噪声，并将结果与另一条旧教师轨迹比较，
因此无法形成有效曲线。v2 改为：

```text
Target4 controlled: source_noise --z0--> --z1--> --z2-->
Draft+3:              Draft --z0--> --z1--> --z2-->
Draft+2:                      Draft --z1--> --z2-->
Draft+1:                              Draft --z2-->
```

所有曲线指标以 `target4_controlled` 为局部参考；旧 `target_teacher` 和
`target_teacher_rng_replay` 只作为独立 parity 诊断。

建议在空闲 GPU 上运行：

```bash
CUDA_VISIBLE_DEVICES=1 bash experiments/E0211_target_refinement_curve_v2/run.sh
```
