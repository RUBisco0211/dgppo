# 小规模邻车注意力对照

目的：只验证 actor 邻车聚合是否有帮助，不再同时更换道路编码、critic 或奖励。

## 固定设置

- Torch/SB3 IPPO vs `NeighborAttentionNet`：后者继承原 LateFusion，仅将 actor 的邻车 max pooling 换成 ego-query 的 masked dot-product attention。
- 原 ego、road、neighbor MLP 和完整 critic 不变。同 seed 下原有参数（包含 action/value head）初始化一致。没有邻车时聚合为零；不加距离平滑门控、BC、安全层。
- 原生 weighted_combination：到达目标 +1、碰撞 -0.5、越界 -0.5；**没有距离奖励**（该模式不使用 log_distance_weight）。classic 动力学、转向 [-1,1]，其余 PPO 设置沿用。
- seed 0，32 worlds，rollout 91，各约 3,000,000 **有效存活受控车辆 step**。不指定 total_iterations，最后完整 rollout 可能略超预算。不是 300 万 world-step，也不是承诺收敛。
- 10 个训练场景、5 个固定验证场景。原始 JSON 不修改，每场景 2–5 受控车辆，其他车辆维持原生行为。筛选背景车辆总数不超过20，受控起终距离5–100m，记录轨迹末端距目标不超过2m；这些是筛选标准，不是道路可行性的形式证明。
- 两算法同一 manifest、种子、训练预算、超参数。有效采样数与场景重采样触发时刻依赖各自策略，因此轨迹不可能完全一样。

## 日志与评估

正式目录 `logs/gpudrive/neighbor_attention_ablation/{ippo,attention_ippo}/<run_name>/`。
每个 run 含 config.json、console.log、metrics.jsonl、tensorboard、wandb、checkpoints、eval、videos。
W&B project `gpudrive`，group `neighbor_attention_ablation_seed0`，横轴 `global_step` 为有效 agent-step。
新实验 opt-in `log_after_update=true`：每次更新后统一记录 loss、KL、entropy、clip fraction、value explained variance、reward、到达/碰撞/越界率和吞吐；checkpoint 与 eval 对应该次更新后的模型。不修改优化顺序。
不再把 chunk reward / episode starts 冒称 episode return。

每50次更新保存与评估，末次另行评估。评估使用独立进程、固定 seed10000、5个验证场景；禁止自动重置混入第二轮，输出 aggregate 和 per_scene 指标、5份GIF。媒体/评估失败不应中止训练；`eval/error` 暴露失败。
短测（W&B disabled）仅放 `artifacts/gpudrive_attention_smoke`；场景初始图放 `artifacts/gpudrive_attention_scene_audit`，不与正式实验混放。
历史 runs 原位置保留，不删除、不覆盖、不冒充新实验。

## 启动与缓存

环境 `/home/linminlei/.conda/envs/gpudrive`，启动脚本 `scripts/run_gpudrive_attention_ablation.sh`。
GPU1 跑 IPPO，GPU3 跑 attention-IPPO。显式 `CUDA_DEVICE_ORDER=PCI_BUS_ID`，`CUDA_VISIBLE_DEVICES` 分别1、3。
缓存 `.cache/gpudrive/attention_ablation_gpu{1,3}/{train.bin,eval.bin}`，不与旧实验混用。首次编译后须测试加载缓存的第二次启动。
这是 PyTorch 实验，不使用 JAX 编译缓存。环境变量 CUDA_HOME 指向上述 conda 环境，OMP_NUM_THREADS=4，MPLBACKEND=Agg，PYTHONUNBUFFERED=1。
checkpoint 为 SB3 ZIP，保留 optimizer；未保存模拟器与全部随机状态，不能宣称精确续训。

## 验证

`scripts/check_gpudrive_attention_ablation.py` 验证原参数一致、critic输出一致、空邻居前反向有限和邻居排列不变性。
`scripts/audit_gpudrive_attention_scenes.py --split training|validation` 验证 simulator 受控车数与清单一致，并渲染初始布局。每个 split 单独进程，避免 Madrona 同进程重复创建的 CUDA heap 问题。
训练仅作为一 seed 的低成本筛查；不要由本次结果直接宣称统计显著或泛化提升。
