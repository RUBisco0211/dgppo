# GRAFO 六组完整序列实验记录

2026 年 10 月 13—14 日，我们在 TS、CS、GFS 三种任务序列上分别运行了 FROG+EWC(M) 和直接微调（FT），共六组完整实验。所有 run 均正常结束，阶段评估、GIF、检查点和本地 CSV 均已生成，并同步至 W&B 项目 `cgmarl`。

## 共同配置

| 项目 | 配置 |
| --- | --- |
| 实现 | 本仓库的 JAX InforMARL、GRAFO 与持续学习训练器 |
| 动作与到达判定 | 二维连续动作；到达阈值 0.35 |
| 训练量 | 每任务请求 500 万环境帧；实际每任务 5,001,216 帧，即 1221 次 PPO update |
| 采集与优化 | 32 个并行环境；rollout 长度 128；batch size 512；4 个 PPO epoch |
| 评估 | 每 100 次 update 及任务结束时，在当前与全部已学任务上各评估 64 个 episode |
| 随机种子 | seed 0；每组只有一个种子 |
| 持续学习方式 | 两种方法都按序训练，并为每个任务保留独立的 actor 输出头；FT 不使用 FROG 或 EWC |
| 正则系数 | FROG+EWC(M) 显式设为 1；FT 的实际正则系数为 0 |

FT 的 run 名称也含有 `lambda1`，这是启动脚本的命名问题，**并不表示 FT 使用了 EWC**。

## 汇总指标

AP 是各任务刚学完时成功率的平均值；AFP 是整条序列结束后所有任务成功率的平均值；AF 是旧任务“最终成功率减去刚学完时成功率”的平均值。三项均越高越好，AF 为负表示遗忘。下表将 AP 和 AFP 写成百分比，AF 写成百分点。

| Setting | 方法与 W&B | AP ↑ | AF ↑ | AFP ↑ |
| --- | --- | ---: | ---: | ---: |
| TS | [FROG+EWC(M)](https://wandb.ai/1205456072-southeast-university/cgmarl/runs/gedlk4xk) | 88.02% | −5.31 | 83.59% |
| TS | [FT](https://wandb.ai/1205456072-southeast-university/cgmarl/runs/0mtf9c6w) | 98.70% | −86.25 | 26.82% |
| CS | [FROG+EWC(M)](https://wandb.ai/1205456072-southeast-university/cgmarl/runs/0icid46i) | 94.38% | −14.84 | 82.50% |
| CS | [FT](https://wandb.ai/1205456072-southeast-university/cgmarl/runs/fddzgv1b) | 96.88% | −92.97 | 22.50% |
| GFS | [FROG+EWC(M)](https://wandb.ai/1205456072-southeast-university/cgmarl/runs/1rc0mfaa) | 98.44% | −1.04 | 97.66% |
| GFS | [FT](https://wandb.ai/1205456072-southeast-university/cgmarl/runs/05di40zr) | 99.22% | −98.44 | 25.39% |

## 序列结束后的逐任务成功率

下列数值按任务**训练顺序**排列，均来自最后一个阶段结束时的 64-episode 评估；它们的平均值就是对应 run 的 AFP。

| Setting | 任务顺序 |
| --- | --- |
| TS | GRAFO-5 → GRAFO-6(1) → GRAFO-7 → GRAFO-4 → GRAFO-3 → GRAFO-6(3) |
| CS | GRAFO-6(1) → GRAFO-6(2) → GRAFO-6(3) → GRAFO-6(4) → GRAFO-6(5) |
| GFS | GRAFO-4(1) → GRAFO-4(2) → GRAFO-4(3) → GRAFO-4(4) |

| Setting | 方法 | 各任务最终成功率，按上表顺序 |
| --- | --- | --- |
| TS | FROG+EWC(M) | 100%、85.94%、89.06%、87.50%、100%、39.06% |
| TS | FT | 0%、0%、0%、0%、60.94%、100% |
| CS | FROG+EWC(M) | 71.88%、89.06%、71.88%、82.81%、96.88% |
| CS | FT | 0%、9.38%、0%、3.12%、100% |
| GFS | FROG+EWC(M) | 100%、93.75%、96.88%、100% |
| GFS | FT | 1.56%、0%、0%、100% |

## 简要解读

- **FT 的可塑性高，但遗忘严重。**三种 setting 的 AP 都在 96% 以上，说明大多数任务刚学完时可以完成；整条序列结束后，旧任务成功率大幅下降，AFP 仅为 22.50%—26.82%。
- **FROG+EWC(M) 的旧任务保留明显更好。**在三种 setting 中，它的 AFP 均高于 FT，AF 也更接近 0；GFS 的最终四任务成功率均超过 93%。
- **TS 仍有新任务学习问题。**FROG+EWC(M) 在最后的 GRAFO-6(3) 上只达到 39.06%，拉低了 AP。不能只看 TS 的较高 AFP 就认为所有任务都学会了。

这些是当前仓库内的**单种子、修改系数后的方法对照**，只支持定性判断。[论文主实验](https://arxiv.org/html/2610.10302v1)统计十个种子，FROG+EWC(M) 的系数为 `10^4`，PPO batch size 为 128；作者 GRAFO 的准确到达阈值尚未核实。因此不能将本表作为论文数值结果的直接复现。

## 原始记录

每个 run 的本地目录为 `logs/cgmarl/full_cl/<run_name>/`，其中 `evaluation.csv` 记录每次阶段评估，`summary.json` 保存最终成功率矩阵和 AP/AF/AFP，`metrics.jsonl` 保存训练指标，`media/` 保存评估 GIF。六个 run 名称如下：

```text
full_ts_frog_ewcm_lambda1_b512_continuous_r035_seed0
full_ts_ft_lambda1_b512_continuous_r035_seed0
full_cs_frog_ewcm_lambda1_b512_continuous_r035_seed0
full_cs_ft_lambda1_b512_continuous_r035_seed0
full_gfs_frog_ewcm_lambda1_b512_continuous_r035_seed0
full_gfs_ft_lambda1_b512_continuous_r035_seed0
```

`logs/` 不提交到 Git；上表中的 W&B 链接可查看云端曲线和评估媒体。
