# DGPPO Navigation / NavigationObs 迁移与排查记录

本文记录从第一版 JAX navigation/navigation_obs 环境完成后，用户提出的反馈、分析结论、实际修改和仍待处理的问题。后续修改前应先阅读本文，避免重复引入已经发现的问题。

## 当前目标与约束

- 使用原始 JAX DGPPO 算法。
- 环境 API 遵循 `dgppo` 包内的 `MultiAgentEnv`、`GraphsTuple`、reward/cost 范式。
- 底层物理使用 `dgppo/dgppo/env/vmas/physax` 复刻 VMAS 行为，不直接依赖 PyTorch VMAS。
- 场景包括 `navigation` 和自定义 `navigation_obs`。
- 当前决定：不使用 lidar，使用 DGPPO graph observation 表达 agent-agent 和 agent-obstacle 信息。
- 当前决定：物理边界保持关闭，即 `enforce_bounds=False`。矩形仅表示出生区域和渲染参考框，允许 agent 移动到矩形外。
- `train_navigation.py` 负责 GemsMARL 风格参数、W&B 日志、视频和 checkpoint。

## 第一版环境与训练脚本

### 用户需求

- 在 DGPPO 仓库中实现 JAX 版本的 `navigation` 和 `navigation_obs`。
- 能用原始 DGPPO 训练脚本跑通 rollout。
- 后续使用 GemsMARL 风格参数、日志字段、W&B 视频和 checkpoint。

### 第一版实现

- 新增 `VMASNavigation` 和 `VMASNavigationObs`。
- 第一版 navigation 使用手写动力学：

  ```python
  a_vel = damping * old_vel + action * dt
  a_pos = a_pos + a_vel * dt
  ```

- 第一版 navigation_obs 只把最近障碍物的相对向量和距离写入 agent node feature。
- cost 使用 agent-agent、agent-obstacle 的几何 signed distance。
- 新增 `dgppo/train_navigation.py`，支持 navigation/navigation_obs、GemsMARL 风格日志、W&B、视频和 checkpoint。

### 早期验证

- 使用 `mamba run -n dgppo` 跑通 reset、step、rollout 和最小训练。
- 本机 `.venv` 存在 CUDA/NCCL 兼容问题，因此原始 JAX DGPPO 统一使用 mamba 的 `dgppo` 环境。

## 日志、W&B 与 checkpoint 反馈

### 用户反馈：W&B 横轴周期不一致

用户发现 DGPPO 的 `collection/reward/episode_reward_mean` 横轴达到约 100k，而 GemsMARL 其他算法横轴按 `max_n_iters` 记录。

### 修复

- W&B `step` 改为 iteration，而不是 total frames。
- 同一 iteration 内使用 `commit=False`，iteration 末尾统一 commit。
- collection、train、eval、timer、counter 日志时机按 GemsMARL 风格整理。

### 用户反馈：指标字段需要和 GemsMARL 语义对应

### 修复

- 增加 GemsMARL 风格字段，包括：
  - `collection/reward/episode_reward_*`
  - `collection/agents/reward/episode_reward_*`
  - `eval/reward/episode_reward_*`
  - `eval/agents/reward/episode_reward_*`
  - `Safe/unsafe_rate`
  - `Safe/cost_mean`
  - `Safe/cost_max`
  - `train/agents/*`
  - `counters/*`
  - `timers/*`

### 用户反馈：需要每个 iteration 保存 checkpoint，最终只保留最新版本

### 修复

- 每个 iteration 结束保存：

  ```text
  models/latest/actor.pkl
  models/latest/Vl.pkl
  models/latest/Vh.pkl
  ```

- 保存前删除旧的 `models/latest`，只保留最新 checkpoint。

### 用户反馈：eval 视频未上传 W&B

### 问题

- 使用 raw numpy frames 调用 `wandb.Video` 时，缺少 `moviepy` 导致训练直接终止。

### 修复

- 保持与 GemsMARL 相同的 raw-frame `wandb.Video` 路径。
- 启动训练前检查 `moviepy`。
- 缺失时明确提示安装：

  ```bash
  mamba run -n dgppo pip install 'wandb[media]'
  ```

- 不自行增加 GIF fallback。

## 速度与视频反馈

### 用户反馈：视频中 agent 速度很慢

### 分析

- 第一版手写动力学使用较小的 dt/动作缩放，与 VMAS 默认动力学不一致。
- VMAS 默认主要参数为：
  - `dt=0.1`
  - `substeps=2`
  - `drag=0.25`
  - `u_range=1.0`
  - `u_multiplier=1.0`
  - 默认无 `max_speed` / `v_range`

### 早期修正

- 手写动力学阶段曾把 navigation/navigation_obs 调整为较大的 `dt` 和 `u_multiplier`。
- 后续改用 Physax 后，手写动力学被替换。

### 用户反馈：navigation_obs 视频没有显示障碍物

### 修复

- `navigation_obs.render_video()` 绘制 obstacle。
- `train_navigation.py` 的 W&B eval frame 渲染检测 `o_pos` 并绘制 obstacle。
- 障碍物使用深灰填充、黑色边框，提高可见性。

## 避障效果差与环境语义分析

### 用户反馈：navigation_obs eval 中 agent 根本没有避障

### 第一轮分析

- 第一版 navigation_obs 只提供最近障碍物信息，多个障碍物信息不完整。
- obstacle 没有作为 graph node。
- cost 只有真正碰撞后才明显变正，碰撞前安全信号不足。
- cost 曾基于旧 state 计算，动作与安全反馈存在一拍延迟。
- Physax 接触后会把 agent 推开，如果只看 post-step state，可能漏掉本 step 已经发生的碰撞。

### 建议

- obstacle 应参与真实底层物理碰撞。
- obstacle 信息应作为 graph node 或完整相对信息提供给 GNN。
- cost 使用真实 signed distance。
- transition cost 使用 old/new state 的最大风险。
- 可增加碰撞前安全 buffer。

## VMAS 复刻方案调整

### 用户要求

参考原始 DGPPO 的：

- `vmas_reverse_transport.py`
- `vmas_wheel.py`

使用 `physax` API 复刻 VMAS navigation/navigation_obs，环境 API 仍按 DGPPO 范式提供。

### 分析结论

- 原始 DGPPO 的 VMAS 环境不是调用 VMAS 库，而是使用 Physax 自己复刻底层物理。
- `wheel/reverse_transport` 使用 `World`、`Agent`、`Entity`、`Sphere/Box/Line`。
- observation 会针对 DGPPO graph 重新组织，不要求逐字复刻 VMAS flat observation。

## Navigation Physax 复刻

### 已实施修改

- `VMASNavigation.step()` 从手写积分改为 Physax：

  ```python
  World.step([Agent, ...])
  ```

- 对齐 VMAS 默认物理参数：
  - `dt=0.1`
  - `substeps=2`
  - `drag=0.25`
  - `u_multiplier=1.0`
  - action range `[-1, 1]`
  - 不设置 `max_speed` / `v_range`
- agent 使用 Physax `Sphere`，agent-agent 碰撞产生真实接触力。
- reward 改为 VMAS shared progress 语义，即所有 agent progress 求和。
- goal relative feature 改为 `agent_pos - goal_pos`。
- transition cost 使用 old/new state 最大 signed collision risk。

### Physax 修复

发现 `physax.world._sphere_sphere_collision()` 调用了不存在的 `self.update_env_forces()`，任何真实 sphere-sphere 碰撞都会报错。

已改为使用现有 `update_forcetorque()` 写回双方碰撞力。

影响检查：

- navigation 会使用 sphere-sphere collision。
- wheel 只允许 agent 与 line 碰撞，不走该路径。
- reverse_transport 只允许 agent 与 box 碰撞，不走该路径。
- wheel/reverse_transport 最小 step 均验证通过。

## Lidar 决策

### 用户反馈与讨论

- VMAS 原始 navigation 在 `collisions=True` 时包含 lidar observation，默认检测其他 agent。
- 用户询问 DGPPO graph 是否可以不依赖 lidar。

### 分析结论

- DGPPO graph edge 已提供精确的 agent-agent 相对位置和速度，信息量不弱于 lidar。
- obstacle 也可通过 graph node 和 agent-obstacle edge 提供。
- 如果目标是严格复刻 VMAS flat observation，应保留 lidar。
- 如果目标是让原始 DGPPO 在该任务上训练并避障，可以不使用 lidar。

### 当前决定

- navigation 和 navigation_obs 均不使用 lidar。
- navigation：
  - agent node feature：position、velocity、relative goal。
  - agent-agent edge：relative position、relative velocity。
- navigation_obs：
  - obstacle 作为 graph node。
  - agent-obstacle edge 提供相对状态。
- 渲染不绘制 lidar ray。

## NavigationObs Physax 复刻

### 已实施修改

- obstacle 使用 Physax 静态 `Entity`：
  - `movable=False`
  - `rotatable=False`
  - `collide=True`
  - `shape=Sphere(obstacle_radius)`
- agents 和 obstacles 一起传入 `World.step()`。
- obstacle 作为 graph `OBS` node。
- policy 仍只对 `node_type=AGENT` 输出动作。
- graph 包含：
  - agent-agent edges
  - obstacle -> agent edges
- cost 包含：
  - agent collisions
  - obstacle collisions
- obstacle cost 使用 transition old/new 最大风险。
- reset 同时采样 agents 和 obstacles，避免初始重叠。
- obstacle 在环境视频和 W&B eval 视频中均可见。

### 验证

- reset/step/JIT 通过。
- obstacle collision case 中 obstacle cost 能正确变正。
- `train_navigation.py --scenario navigation_obs` 一轮最小训练冒烟测试通过。

## Navigation 完整训练后的最新反馈

### 用户反馈 1：agent 会从矩形框边界外部运动到目标点

### 排查结论

- 原因是 `enforce_bounds=False`。
- 矩形仅是出生区域和渲染参考，不是物理墙。
- `state_lim()` 不会自动约束 Physax 状态，因为 Physax step 不调用 `clip_state()`。
- VMAS navigation 默认同样是 `enforce_bounds=False`。

### 当前决定

- 用户明确要求：物理边界可以不开启。
- 因此保留 `enforce_bounds=False`。
- 后续不应把“agent 越出矩形”当成 Physax bug。
- 如保留当前渲染矩形，需理解它仅表示出生区域；必要时可修改渲染样式，避免误认为物理边界。

### 用户反馈 2：相比 MPE，agent-agent 避障仍很容易碰撞

### 最新完整训练结果

运行目录：

```text
dgppo/outputs/navigation/dgppo/dgppo_navigation_seed0_20260606_013830
```

主要结果：

```text
最终 eval reward ≈ 5.39
最终 eval unsafe_rate = 0.60
训练期间 eval unsafe_rate 在 0.20 到 0.90 之间波动
最终 collection unsafe_rate ≈ 0.53
```

结论：策略已经学会高效到达目标，但安全约束没有收敛。

### 已确认的高风险原因

#### 1. 安全 cost 只在接触后变正

当前 cost：

```python
2 * agent_radius - min_dist
```

仅当 agent 中心距离小于 `0.2` 时变正，此时两个 Physax sphere 已经接触/重叠。

`cost_margin=0.5` 只是对正负标签做平移，不会扩大安全距离：

```python
cost <= 0 -> cost - 0.5
cost > 0  -> cost + 0.5
```

因此策略缺少碰撞前的正 cost 预警。

建议但尚未实施：

```python
safe_distance = 2 * agent_radius + safety_buffer
cost = safe_distance - min_dist
```

建议先测试 `safety_buffer=0.05~0.1`。

#### 2. `dt=0.1` 与 DGPPO `alpha=10` 可能不匹配

DGPPO 使用：

```python
(V_next - V) / dt + alpha * V
```

当前：

```text
dt=0.1
alpha=10
alpha * dt = 1
```

相比原始 DGPPO 常用 `dt=0.03, alpha=10`，当前离散 CBF 条件更激进，可能导致安全优势不稳定。

建议但尚未实施：

- 在保持 `dt=0.1` 时先测试 `alpha≈3`。

#### 3. 没有速度上限

- Physax navigation 当前没有 `max_speed` / `v_range`。
- 正常控制下 drag 会限制稳定速度，但碰撞接触力可能产生较大瞬时速度。
- 较高速度会缩短避让时间，且 old/new 两点 cost 可能无法完整表达中间风险。

建议但尚未实施：

- 统计训练后策略的速度峰值。
- 根据 VMAS 实验需要考虑设置合理 `max_speed`，或增加 substeps。

#### 4. 当前 graph 为全连接

当前：

```python
comm_radius = 10.0
```

场景出生区域宽度仅约 `2.0`，因此 agent graph 始终全连接。MPE 默认 `comm_radius=0.5`。

全连接可能让远距离无关 agent 的消息与近距离危险 agent 竞争 attention。

建议但尚未实施：

- 测试 `comm_radius=0.5` 或适合 agent 半径/速度的局部感知范围。

#### 5. 训练参数与原始 DGPPO 默认差异较大

当前完整训练：

```text
PPO epochs = 45
minibatch = 400
lr = 3e-5
```

原始 DGPPO 默认 `epoch_ppo=1`。重复使用同一批 rollout 更新 45 次可能导致 actor 和安全 critic 振荡。

最新日志：

```text
最终 actor grad norm ≈ 5.65
clip_grad_val = 1.0
```

说明 actor 更新长期处于强梯度裁剪状态。

建议但尚未实施：

- PPO epochs 先从 45 降为 1 或 5。
- 观察 unsafe rate、safe ratio、actor grad norm 是否稳定。

#### 6. Reward 不再惩罚碰撞

当前 `agent_collision_penalty=0.0`。

这是为了避免 reward penalty 与 DGPPO safety cost 重复，但结果是：

- reward 只鼓励快速到达目标；
- 如果安全项没有学稳，reward 最优策略会直接穿行或碰撞。

这解释了最新训练中 reward 已收敛但 unsafe rate 很高。

是否恢复 reward collision penalty 尚未决定。优先应先修正安全 cost 预警范围和 DGPPO 超参数，再决定是否加入辅助 penalty。

## 下一轮建议实验顺序

保持物理边界关闭，按以下顺序做消融，每次只改一到两个变量：

1. 增加 agent-agent `safety_buffer`，让 cost 在物理接触前变正。
2. 将 `alpha` 从 `10` 调整为约 `3`。
3. 将 PPO epochs 从 `45` 降至 `1` 或 `5`。
4. 将 `comm_radius` 从 `10.0` 改为局部范围，例如 `0.5`。
5. 记录最大速度、最小 agent-agent 距离、越界比例等诊断指标。
6. 若高速仍导致避碰失败，再评估 `max_speed` 或更多 Physax substeps。
7. 若 safety cost 仍无法约束策略，再考虑恢复较小的 collision reward penalty 作为辅助信号。

## 注意事项

- `dgppo/` 当前在父仓库 git 状态中是未跟踪目录，`git diff` 无法提供该目录的历史基线。
- `make_env()` 当前直接取得类的 `PARAMS` 字典并可能原地修改；后续调整参数时应注意跨环境实例污染风险。
- navigation 和 navigation_obs 共享 navigation 的部分参数和方法，修改 navigation 的物理参数时必须回归验证 navigation_obs。
- 修改 Physax `world.py` 前必须检查 wheel/reverse_transport 是否受影响。
- 所有 JAX DGPPO 验证统一使用：

  ```bash
  mamba run -n dgppo ...
  ```

## 2026-06-06：最小修改方向复核

### 用户反馈与约束

- 后续每次用户反馈和修复建议都必须追加到本文档。
- 用户不希望对当前已经较稳定的 reward 训练做大改动。
- 用户对前一轮建议的反馈：
  1. 询问其他 DGPPO 环境的 safety cost 是否同样只在物理接触后变正。
  2. `dt` 和 `alpha` 应优先参考已有 JAX VMAS 环境实现。
  3. 速度上限可能是原因。
  4. 询问全连接 graph 是否真的有问题，以及其他环境如何设置。
  5. 训练参数尽量保持当前配置，因为 reward 曲线已比较稳定。
  6. 询问为什么之前把 reward collision penalty 改为 0。

### 对照检查结论

#### 1. 其他 DGPPO 环境的 cost 也通常在接触后变正

MPE、LidarEnv、VMASWheel、VMASReverseTransport 的 agent-agent cost 都使用：

```python
2 * radius - min_dist
```

因此只有 agent 几何体接触/重叠后，原始 signed cost 才为正。随后统一增加 `eps=0.5` 的正负 margin。

结论：

- 当前 navigation 的几何 contact cost 与原始 DGPPO 环境范式一致。
- 直接增加 `safety_buffer` 虽可能改善避碰，但会改变 unsafe 定义，不应作为第一优先修复。

#### 2. `dt=0.1, alpha=10` 与已有 JAX VMAS 环境一致

- `VMASWheel` 使用 `dt=0.1`。
- `VMASReverseTransport` 使用 `dt=0.1`。
- 原始 DGPPO 训练默认 `alpha=10`。

结论：

- 不应仅因为 MPE 使用 `dt=0.03` 就把 navigation 的 alpha 改成 3。
- `dt/alpha` 暂时保持不变，除非后续消融实验明确证明该组合导致问题。

#### 3. 速度上限可能有帮助，但不是 VMAS 默认行为

- 当前 Physax navigation 与 VMAS navigation 默认一致：没有 `max_speed` / `v_range`。
- MPE 使用 `clip_state()`，速度会被限制到状态范围。
- Physax 接触力可能产生较大的瞬时速度，进而增加后续再次碰撞的概率。

结论：

- 速度上限属于可考虑的小范围稳定性修改。
- 修改前应先增加最大速度诊断指标，确认训练后策略或碰撞响应是否真的产生异常高速。
- 如果速度峰值正常，不应为了改善避碰而偏离 VMAS 默认物理。

#### 4. 全连接 graph 不是明确错误

- MPE/LidarEnv 默认使用局部连接，`comm_radius=0.5`。
- VMASWheel 和 VMASReverseTransport 虽声明 `comm_radius=0.4`，但实际 agent-agent edge mask 是除自身外全连接。
- 当前 navigation 使用 `comm_radius=10.0`，因此也是全连接。

结论：

- 全连接 graph 在原始 JAX VMAS 复刻环境中已有先例，不是首要问题。
- navigation 仅有 4 个 agent，全连接带来的 attention 干扰有限。
- 暂时保持全连接；局部连接可作为后续消融，不作为第一轮修复。

#### 5. 当前训练参数暂时保持

- 用户反馈 reward 曲线已较稳定。
- 虽然 `epoch_ppo=45` 与原始 DGPPO 默认差异较大，并且 actor grad norm 较高，但此时直接修改训练参数会同时改变 reward 学习行为。

结论：

- 第一轮修复不修改 PPO epochs、minibatch、lr 等训练参数。
- 优先修复环境与 DGPPO safety critic 的语义不一致。

#### 6. Reward collision penalty 改为 0 的原因

- VMAS 原始 navigation reward 包含 agent collision penalty。
- 原始 DGPPO 的 MPE/Lidar 等环境通常把碰撞作为 safety cost，而不在任务 reward 中重复惩罚。
- 之前将 navigation 的 `agent_collision_penalty` 改为 0，是为了：
  - 避免 reward objective 与 safety constraint 重复表达碰撞；
  - 保持 reward 曲线主要反映任务完成情况；
  - 让安全性由 DGPPO 的 Vh/CBF 机制负责。

结论：

- 当前保持 `agent_collision_penalty=0.0`。
- 若 safety 机制修复后仍无法避碰，再考虑加入较小 reward penalty 作为辅助，但这会降低与其他 DGPPO 环境的安全约束可比性。

### 新发现：transition cost 与 DGPPO Vh 输入语义不一致

当前 navigation/navigation_obs 的 `step()` 返回：

```python
cost = get_transition_cost(old_state, next_state)
```

该修改最初是为了避免 Physax 接触力把 agent 推开后，只看 next state 会漏掉本 step 的碰撞。

但是 DGPPO 的安全价值网络 `Vh` 输入是当前 `graph/state`，并不输入 action。原始 DGPPO 环境也统一在 step 中使用：

```python
cost = get_cost(graph)
```

因此 transition cost 会造成：

- 同一个当前 graph，在采取不同 action 后产生不同 cost 标签；
- `Vh(graph)` 无法仅根据当前 graph 准确预测该 action-dependent 标签；
- safety critic 和 CBF advantage 训练目标不一致；
- 可能解释 reward 已收敛但 unsafe rate 长期波动的问题。

### 当前最攸关、最小范围的修复建议

第一优先级：

1. 将 navigation 的 rollout cost 恢复为 state-based：

   ```python
   cost = self.get_cost(graph)
   ```

2. navigation_obs 后续同步恢复为 state-based cost，保持两个环境与原始 DGPPO cost 范式一致。

3. 保留现有 Physax 碰撞、reward、dt、alpha、全连接 graph 和训练参数不变。

第二优先级，仅做诊断，不立即改变行为：

1. 日志增加最大速度、平均速度和最小 agent-agent 距离。
2. 根据诊断结果决定是否增加 `max_speed`。

暂不优先修改：

- 不增加 safety buffer。
- 不修改 alpha。
- 不修改全连接 graph。
- 不修改 PPO epochs/lr/minibatch。
- 不恢复 reward collision penalty。

## 2026-06-06：实施最小 safety cost 修复并增加诊断日志

### 实际修改

1. `VMASNavigation.step()` 和 `VMASNavigationObs.step()` 的 rollout cost 从：

   ```python
   cost = self.get_transition_cost(env_state, env_state_new)
   ```

   恢复为原始 DGPPO 范式：

   ```python
   cost = self.get_cost(graph)
   ```

2. `train_navigation.py` 为 collection 和 eval 增加以下诊断指标：

   - `diagnostics/agent_speed_mean`
   - `diagnostics/agent_speed_max`
   - `diagnostics/min_agent_distance_mean`
   - `diagnostics/min_agent_distance_min`

3. 未修改 Physax 物理实现、边界、reward、碰撞 penalty、`dt`、`alpha`、图连接方式和训练参数。

### 验证结果

- navigation 和 navigation_obs 的普通 `step`、`jax.jit(step)` 返回 cost 均与当前 graph 的 `get_cost(graph)` 一致。
- 人工令两个 agent 重叠后：
  - navigation 当前 cost 与 rollout 返回 cost 均为 `0.7`；
  - navigation_obs 的 agent collision cost 当前值与 rollout 返回值也均为 `0.7`。
- 使用 `mamba` 的 `dgppo` 环境完成 navigation 最小训练冒烟测试：
  - 4 个并行环境；
  - 每环境 10 步；
  - 1 个训练 iter；
  - 1 次 PPO epoch。
- collection/eval 的 8 个新增诊断字段均已成功写入 `metrics.csv`。

### 后续判断依据

- 需要重新进行完整训练，观察 state-based cost 是否降低 eval unsafe rate；本次冒烟测试不能证明避碰效果已经改善。
- 根据完整训练中的 `agent_speed_max` 和 `min_agent_distance_min` 再决定是否增加 `max_speed` 或 safety buffer，避免同时引入多项行为变化。

## 2026-06-06：同步检查并修复 NavigationObs

### 用户反馈

- 在 navigation 完成最小 safety cost 修复后，检查 navigation_obs 是否存在需要同步适配的部分，并一起修复。

### 对比检查结论

- navigation_obs 已通过继承或自身实现同步以下 navigation 行为：
  - `dt`、`substeps`、collision force、contact margin；
  - agent 的 `u_multiplier`、drag、radius 和 Physax 构造；
  - `enforce_bounds=False`；
  - agent-agent 图边；
  - state-based rollout cost。
- agent-obstacle edge 方向正确：
  - receiver 是 agent；
  - sender 是 obstacle；
  - 与原始 DGPPO 的 MPE/Lidar obstacle edge 范式一致。
- 不需要重新加入 lidar。

### 发现的问题

1. navigation 和 navigation_obs 分别重复构造 `World`，后续修改物理参数时容易再次出现不同步。
2. navigation_obs 原 reset 分别采样实体和目标：
   - agents 与 obstacles 之间不会初始重叠；
   - 但 goals 未避开 agents 和 obstacles；
   - 目标可能生成在障碍物附近，造成任务不合理或诱导 agent 穿过障碍物。
3. 通用诊断日志只有 agent-agent 最小距离，navigation_obs 缺少 agent-obstacle 最小距离。
4. `make_env()` 原地修改类级 `PARAMS`，例如一次使用自定义 `n_obs` 后可能污染后续环境实例。

### 实际修改

1. 在 `VMASNavigation` 中增加公共 `_make_world()`，navigation 和 navigation_obs 均通过该方法构造 Physax World。
2. navigation_obs reset 改为一次性联合采样：
   - agents；
   - obstacles；
   - goals。
3. 联合采样最小间距按 agent/obstacle 最大直径组合计算，并额外增加 `0.05`。
4. navigation_obs 的 collection/eval 日志新增：
   - `diagnostics/min_obstacle_distance_mean`
   - `diagnostics/min_obstacle_distance_min`
5. `make_env()` 改为复制类级 `PARAMS` 后再写入实例配置，避免跨实例污染。

### 保持不变

- Physax 碰撞计算；
- reward 和 collision penalty；
- safety cost 阈值与 margin；
- state-based cost；
- graph edge 方向和全连接范围；
- 训练参数；
- 无 lidar 的 graph observation 方案。

### 验证结果

- Python 静态编译检查通过。
- 对 64 个 navigation_obs reset 批量检查：
  - agents、obstacles、goals 的全局最小间距为 `0.2514`；
  - 满足设定最小间距 `0.25`。
- navigation 与 navigation_obs 的 JIT step 均通过。
- navigation_obs 普通/JIT step 返回 cost 均与当前 graph 的 `get_cost(graph)` 一致。
- `make_env(..., num_obs=7)` 后：
  - 实例使用 `n_obs=7`；
  - 类级默认值仍保持 `n_obs=3`。
- navigation_obs 一轮最小训练冒烟测试通过。
- collection/eval 的 obstacle 最小距离字段均成功写入 `metrics.csv`。
- obstacle 场景首次 XLA 编译耗时较长，本次单 iter 冒烟测试总耗时约 7 分 23 秒；未发现运行时错误。
