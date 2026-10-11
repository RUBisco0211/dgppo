# GPUDrive 上的 InforMARL / DGPPO 实现规范

## 当前基线

- 单一同质直道路场景复制为 8 个并行 world。
- 每个 world 物理场景严格保留 15 辆机动车：5 辆受控车和 10 辆按 GPUDrive 录制轨迹回放的自由车，其余原始车辆移除。
- 训练时可在这 15 辆内部重新选择 5 辆受控车；评估使用固定的 5+10 布局。
- 受控车初始位置必须在车道内，彼此相对集中、间距至少 8m，目标沿相近方向且距离 20–85m。
- 使用 GPUDrive `classic` 连续动力学，动作是 acceleration 与 steering；默认 steering 上限为 ±0.3rad。
- 碰撞采用 `collision_behavior="ignore"`：碰撞产生奖励惩罚、cost 和评估事件，但不终止；单车到达目标后只 mask 该车。
- rollout 和 eval horizon 均为 91。InforMARL 首轮实验使用 8 worlds、4 worlds/minibatch。

## 奖励、团队回报与终止

采用 GPUDrive 官方 weighted-combination 奖励，并只增加一个目标距离改善项：

```text
r_i = clip(0.1 * (d_goal,t - d_goal,t+1), -0.1, 0.1)
      + 1.0 * goal_achieved
      - 0.75 * vehicle_collision
      - 0.75 * offroad
```

不加入绝对目标距离、车道中心偏离或人为存活奖励。除目标距离改善量外，其他奖励权重保持官方设置不变。InforMARL 的集中式任务回报为 5 辆受控车单步奖励之和，使用一个团队 `Vl` 和共享 team advantage。

- 单车到达目标：该 agent 后续 actor/PPO loss 与奖励被 mask。
- `Vl` 只有在 5 辆受控车全部完成或场景达到 91 步上限时停止 bootstrap。
- 碰撞不设置 done。
- 行为克隆保留命令行配置接口，默认关闭；当前正式基线从随机策略开始训练。

## Actor 的 ego-centric 图

每个 world 的 actor 图包含 5 个互不相连的 ego 星形子图：

- 5 个 ego root，每个 root 输出一辆受控车的动作。
- 每个 ego 从 GPUDrive partner observation 中按纯欧氏距离选择 50m 内最近 10 辆车。
- 邻车不区分是否受控参与排序，但节点特征含 `is_controlled` 标记。
- 每个 ego 与自己的邻车双向连边；不存在邻车—邻车边，也不存在不同 ego 子图之间的边。
- 邻车进出范围使用与 InforMARL 相同的硬 top-k/mask，不额外设计平滑机制。

道路不是 GNN 节点。每个 ego 从官方 `agent_roadmap_tensor` 选择最近 32 条、50m 范围内的结构化道路段，以共享 MLP 编码后做 masked mean + max pooling，再拼接到 ego 表征。道路信息不产生 road-to-vehicle 图边。

默认轻量模型为 Actor 1 层 GNN、`Vl` 1 层 GNN并关闭 RNN；保留 `dgppo_default` 深度配置供消融。

## 集中式任务 critic `Vl`

`Vl` 使用可扩展的固定容量稀疏图：

- 最多 64 个物理车辆节点，padding 由有效 mask 排除。
- 场景内 5+10 共 15 辆物理机动车全部进入图。
- 每个车辆节点只接收 50m 内最近 10 辆车辆的消息。
- 前 5 个节点固定为受控车，只有这些节点带道路 pooled context；其余车辆仍通过稀疏消息影响受控节点。
- 5 个受控节点的最终表示经 mean pooling 后输出单个 team value。

## 连续安全 cost

DGPPO 的 `Vh` 使用两个连续通道，InforMARL 只记录这些指标而不训练 `Vh`：

```text
c_vehicle,i = max_j (0.5m - signed_vehicle_clearance_ij)
c_road,i    = 0.5m - min_k road_edge_clearance_ik
```

- 支持 `oriented_box` 和 `ellipse` 两种车辆几何模式。
- vehicle cost 对这 15 辆中的所有其他车辆计算，而不是只对 actor 的最近 10 辆。
- road cost 可以继续使用 GPUDrive LiDAR RoadEdge 命中做几何 clearance；这与 actor 的结构化道路 observation 相互独立。
- cost 裁剪到 `[-1, 1]`；`cost > 0` 表示违反 0.5m 安全裕量。

## Torch/JAX 边界与静态形状

GPUDrive/Madrona 环境保留在 Torch/CUDA 侧，rollout runner 在 Python 外层批量 step；张量通过 DLPack 传入 JAX。actor 和 critic 图均用固定容量、padding node 与 mask，网络前向和 PPO update 才进入 JAX JIT。

当前 8-world 实际固定形状：

```text
actor nodes:  [8, 56, 274]   # 5 ego + 5*10 neighbor + padding
actor edges:  [8, 100, 7]
critic nodes: [8, 65, 274]   # 64 vehicles + padding
critic edges: [8, 640, 7]    # 64*10 sparse incoming edges
cost:         [8, 5, 2]
```

改变 world 数、horizon、车辆/道路容量或 GNN 结构都会产生新的 JAX executable；保持这些静态配置不变才可复用持久编译缓存。

## 入口、缓存与评估

训练使用独立入口 `train_gpudrive.py`，不修改原来的 `train.py`。W&B 项目固定为 `gpudrive`，run 名称遵循 `<algorithm>_seedNNN_<timestamp>_<id>`。

推荐启动环境变量：

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_COMPILATION_CACHE_DIR=/home/linminlei/.cache/jax/gpudrive-informarl-5c-cu124
# 仅在排查慢编译时启用：
export XLA_FLAGS=--xla_dump_to=/tmp/gpudrive_xla_dump
```

Madrona 缓存由 GPUDrive 构建目录管理；C++、CUDA、编译 flags 或 GPU 架构不变时可复用。JAX 缓存要求 jax/jaxlib、CUDA、设备架构、静态 shape、dtype 与被编译函数保持一致。

固定评估布局可用：

```bash
conda run -n gpudrive python scripts/render_gpudrive_eval_layout.py \
  --num-worlds 8 --output artifacts/gpudrive_eval_layout_5c10n.png
```

短 smoke test：

```bash
conda run -n gpudrive python train_gpudrive.py \
  --algorithm informarl --num-worlds 8 --eval-worlds 8 \
  --iterations 1 --horizon 2 --debug --run-name smoke_informarl_5c10n
```

完整实验开始前先检查：图 shape、所有 loss finite、padding rate 为 0、固定 eval 布局、GIF 多帧、`Vl/grad_norm`、reward 三个组成项、成功率和碰撞率。
