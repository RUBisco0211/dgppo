# GPUDrive BC actor 预训练与 PPO 续训

状态：GPU 已恢复；轨迹跟踪数据和两种 BC actor 已完成。IPPO/attention PPO 均已在 GPU2 的 tmux 会话中启动；step-0 eval 与训练进度以各 run 目录为准，不能提前宣称训练完成。

## 样本及评估

- BC 训练：mini 训练集 352 个 simple/medium 场景，约 1178 辆受控车；BC 验证：mini 验证集 48 个不同场景，约 151 辆受控车。全部每场景 2–5 受控车。
- PPO：保留上一轮的 10 个训练场景和 5 个评估场景。5 个最终评估场景从 BC 验证清单排除；BC 未使用 testing 集。
- GPUDrive 对 `classic` 动力学不生成逆动作，`get_expert_actions()` 在此模式下全为零，不能直接用于 BC。现在对同一 7×13=91 个离散 classic 动作逐一做一步动力学预测，选出最接近记录轨迹下一状态的动作。这是**轨迹跟踪教师**，不宣称恢复了原始驾驶员控制。PPO 转向范围仍为 ±1。
- 数据生成在离散跟踪回放中记录真实 GPUDrive observation、离散动作和存活有效 mask；审计到达、碰撞、越界、动作分布与一步跟踪误差。动作几乎全零或回放明显失败则禁止启动 BC。
- 预计最多约 1178×91≈10.7 万 BC 训练状态-动作对，实际有效样本数以 `audit.json` 为准。BC 15 epoch 上限、验证加权交叉熵早停；PPO 两种网络各 1000 万有效受控车 step，32 worlds，seed0。
- BC 只更新 actor 与动作输出层；PPO 导入 actor 权重，重新创建 PPO optimizer，critic 保持原初始化。两边使用完全相同的数据集与 BC 超参数。

## 2026-10-11 数据与 BC 基线

- 训练集 352 场景、1169 受控车、79559 条有效状态—动作；验证集 48 场景、151 车、10600 条。轨迹跟踪教师回放到达率 99.0%/99.3%，平均一步误差 0.178/0.161。此为带未来轨迹的教师成绩，**不是** BC 策略成绩。
- 验证集精确动作分类率：IPPO 22.5%，attention 21.7%，最频动作基线 22.0%。BC 模型不能靠分类率宣称已学好。
- 真正闭环 deterministic 策略，在相同 48 个验证场景：零动作 41.1% 到达、11.3% 碰撞、20.5% 越界；IPPO-BC 48.3% 到达、17.2% 碰撞、18.5% 越界；attention-BC 51.0% 到达、14.6% 碰撞、21.2% 越界。BC 有到达率增益但安全性尚未改善。
- 另外 5 个最终保留场景共15辆车：零动作6辆到达，IPPO-BC与attention-BC各7辆到达。样本小，PPO 微调与后续多场景评估仍必要。

## 目录和日志

- 场景清单 `gpudrive/data/manifests/bc_seed0/`
- 数据 `gpudrive/data/derived/bc_seed0_tracking/{training,validation}/batch_*.pt`，每 split 的 `audit.json`。旧的 `bc_seed0` 全零伪标签仅留作错误诊断，不能训练。
- BC run `logs/gpudrive/bc_pretrain/{ippo,attention_ippo}/bc_<algo>_seed000_<stamp>/`：`metrics.jsonl`、`best_actor.pt`、`metadata.json`、W&B、控制台
- PPO run `logs/gpudrive/bc_warmstart/{ippo,attention_ippo}/<algo>_bc_seed000_<stamp>/`：原 PPO 指标、step0 的 `bc_initial_eval.json`、checkpoint、每100 iter评估、五场景GIF和W&B
- W&B project `gpudrive`。BC group `bc_actor_seed0`，横轴 `bc/epoch`；PPO group `bc_warmstart_seed0`，横轴有效 agent-step `global_step`。

## 执行顺序

在根目录执行：

```bash
python scripts/start_gpudrive_bc_pair.py data --data-device cpu
# 等两个 data tmux 完成，检查两个 audit.json 的回放到达率
python scripts/start_gpudrive_bc_pair.py bc --stamp MMDDhhmmss
# 等 BC 两个 run 完成并检查验证损失与模型
python scripts/start_gpudrive_bc_pair.py ppo --stamp 同一个BC时间戳 --algorithm ippo
python scripts/start_gpudrive_bc_pair.py ppo --stamp 同一个BC时间戳 --algorithm attention_ippo
```

数据生成使用 CPU。当前 GPU1/3 有其他任务占用，BC 和 PPO 改在物理 GPU2（A100，`sm_80`）上运行。PPO 两个算法并行时 GPU2 显存峰值（一个评估子进程）约 30 GB / 40 GB；两边同时评估可能接近 38 GB，需注意余量。入口固定 `CUDA_DEVICE_ORDER=PCI_BUS_ID`。Madrona 缓存使用 `.cache/gpudrive/gpu2_pci_ippo_aligned_16w/madrona_kernel_cache.bin`，已核对内容为 `sm_80`。**文件名中的 gpu2 不足以证明架构**：`gpu2_ippo_aligned_16w` 实际是 `sm_89`，在 A100 上会报 `CUDA_ERROR_NO_BINARY_FOR_GPU`。启动脚本只起 tmux，不启动周期性监控。

源码入口：[数据构建](/home/linminlei/code/dgppo/gpudrive/baselines/ppo/build_bc_dataset.py)、[BC训练](/home/linminlei/code/dgppo/gpudrive/baselines/ppo/train_bc_actor.py)、[PPO接入](/home/linminlei/code/dgppo/gpudrive/baselines/ppo/ppo_sb3.py)。
