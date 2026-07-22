# HMM 约束流形方法接入 InforMARL 的可行性分析

本文分析 `.papers/HMM.pdf` 中的 Hierarchical Manifold Multi-Agent PPO 思路，是否可以直接用本仓库已有的单步 InforMARL actor-critic 接上约束流形投影方法。

## 结论

不能把当前单步 InforMARL **无改动地** 直接当成 HMM 的高层策略来接约束流形方法。

可以做一个较小改动的安全过滤版本：保留 InforMARL 每个环境 step 输出低层 action，然后把该 action 当作 `u_ref` 输入约束流形投影器，得到安全动作 `u_s` 再交给环境。这种接法工程上可行，但它不是 HMM 论文里的方法，更接近“InforMARL nominal action + manifold safety filter”。

如果要复现 HMM 的核心方法，则需要把 InforMARL 改造成“高层 subgoal policy”：每 `tau` 个环境 step 输出一次相对 subgoal，低层固定控制器在这 `tau` 步内追踪 subgoal，并在每个环境 step 做约束流形投影。同时 PPO 的 rollout、reward 累积、折扣和 advantage 都要按 semi-MDP 决策 epoch 重写。

## HMM 论文方法的关键结构

HMM 是分层方法：

- 高层 RL 每隔 `tau` 个环境 step 为每个 agent 输出一个 subgoal。
- subgoal 是相对当前位置的短期 waypoint，而不是直接环境 action。
- 低层 controller 把 subgoal 转换成 nominal reference action `u_ref`。
- 约束流形 controller 对 `u_ref` 做 tangent-space projection，得到最终安全动作 `u_s`。
- 安全保证来自低层固定 controller，而不是来自 RL policy 本身。

论文超参数中给出：

- `Subgoal interval tau = 8`
- `Relative subgoal max Delta = 0.2`
- actor/critic 学习率、PPO clip、GAE 等仍保持 MAPPO/InforMARL 风格。

论文也说明 baseline 中的 InforMARL 是把约束违反作为 penalty 加入 reward；HMM 则把安全约束下放到固定低层 manifold controller。

## 当前仓库 InforMARL 的接口

当前实现中，InforMARL 是普通单步 actor-critic：

- `dgppo/algo/informarl.py`
  - `step(graph, rnn_state, key)` 每个环境 step 采样一个 action。
  - action shape 为 `(n_agents, action_dim)`。
  - rollout 中每一步立刻调用 `env.step(graph, action)`。

- `dgppo/trainer/utils.py`
  - `rollout()` 使用 `jax.lax.scan` 跑 `env.max_episode_steps` 步。
  - 每一步都记录 `(graph, action, rnn_state, reward, cost, done, log_pi, next_graph)`。
  - 当前 `Rollout.actions` 存的是每个环境 step 的 action，不是每个高层 epoch 的 subgoal。

- `dgppo/algo/module/policy.py`
  - `PPOPolicy` 输出 `TanhNormal` 分布，动作经过 tanh 映射。
  - 语义是直接 action，不是 waypoint/subgoal。
  - 虽然 action_dim 在 2D Lidar 环境中也是 2，但这只是维度相同，语义不同。

- `dgppo/env/lidar_env/base.py`
  - 环境 action 被裁剪后直接用于动力学：
    `x_dot = concat([velocity, action * 10.])`
  - 因此当前 action 语义更接近加速度/控制输入，而不是相对目标点。

## 为什么不能“直接”接成 HMM

### 1. 时间尺度不一致

HMM 的 RL 决策周期是 `tau` 步，当前 InforMARL 是每步决策。HMM 的 Bellman backup 使用有效折扣 `gamma ** tau`，reward 是 `tau` 步累计 reward。当前代码使用每环境 step 的 `gamma` 和每步 reward 计算 GAE。

如果只把当前 InforMARL action 后接 projection，训练过程仍是普通 MDP，不是 HMM 的 CSMDP。

### 2. action 语义不一致

HMM 高层输出 `z_i^k`，表示相对 subgoal/waypoint。低层 nominal controller 再根据 `(state, subgoal)` 生成 `u_ref`。

当前 InforMARL 输出的是环境 action，直接进入 `env.step()` 影响加速度。把这个 action 当 subgoal 用会改变语义；把它当 `u_ref` 用则可行，但不再是论文 HMM。

### 3. rollout 数据结构不一致

HMM 需要在高层 epoch 记录：

- epoch 起点 graph
- subgoal
- subgoal log probability
- `tau` 步累计 reward
- epoch 末 graph
- 可能还要记录低层执行中的 safety/cost 指标

当前 `Rollout` 是每个环境 step 一条 transition。若要训练高层 subgoal policy，应该新增或改造 rollout 函数，而不是直接复用 `trainer/utils.py::rollout()`。

### 4. 低层 controller 必须固定

论文证明高层学习过程 stationary 的关键条件是：低层 policy 没有可学习参数。当前 InforMARL actor 本身就是每步 learnable low-level action policy；如果它继续每步输出动作，那么不满足 HMM 对固定低层 dynamics 的建模。

### 5. 安全投影需要额外状态和约束计算

约束流形 controller 需要：

- pairwise constraint `h_ij`
- slack variable `mu`
- constraint residual `c = h + mu`
- `J_u`
- drift term `psi`
- pseudo-inverse `J_u^dagger`
- null-space basis `B_u`
- subgoal tracking产生的 `u_ref`

当前仓库已有 cost 计算，但没有维护 slack variable，也没有 manifold projection controller。

## 可以直接复用的部分

可以复用：

- InforMARL 的 GNN actor/critic backbone。
- `GraphsTuple` 图输入结构。
- Lidar 环境中的 agent/goal/obstacle graph 表达。
- PPO policy/value 模块的大部分网络代码。
- baseline 的 reward penalty / lagrangian 对比逻辑。

尤其是高层 subgoal policy 可以直接沿用 `PPOPolicy` 的 GNN + MLP + optional RNN 结构，只需要把输出语义改成相对 subgoal，并限制范围到 `[-subgoal_max, subgoal_max]`。

## 两种可行实现路线

### 路线 A：最小工程接入，但不是 HMM

目标：快速验证 manifold projection 是否能提升 InforMARL safety。

做法：

1. 保持当前 InforMARL 每步输出 action。
2. 在 `env.step()` 前增加 `manifold_project(graph, action)`。
3. 把 actor action 当作 `u_ref`。
4. 投影后的 `u_s` 进入环境动力学。
5. PPO 仍按每步 MDP 训练。

优点：

- 改动小。
- 可以复用当前 `Rollout`、`InforMARL.update()`、`PPOPolicy`。
- 适合做 ablation：`InforMARL + manifold filter`。

缺点：

- 不是论文 HMM。
- 没有高层 subgoal，也没有 `tau` 步 semi-MDP。
- 理论上的 stationary high-level MDP 结论不能照搬。
- 如果 projection 和环境离散动力学不一致，安全保证也需要重新审视。

### 路线 B：实现 HMM 风格，推荐用于复现论文方法

目标：复现 HMM：高层 InforMARL-style GNN subgoal policy + 固定低层 manifold controller。

需要新增：

1. `HMM` 算法类，例如 `dgppo/algo/hmm.py`。
2. 高层 rollout，例如 `hierarchical_rollout()`：
   - 外层 scan 长度为 `max_episode_steps // tau`。
   - 每个 epoch 采样一次 subgoal。
   - 内层 scan 执行 `tau` 个低层 step。
   - 累计 reward，记录 epoch 末 graph。
3. Subgoal policy：
   - 可以复用 `PPOPolicy`。
   - `action_dim = spatial_dim`。
   - 输出乘以 `subgoal_max = 0.2`。
4. Low-level nominal controller：
   - `u_ref = K * (subgoal_position - current_position) - D * velocity`
   - 或按论文设置 viability/tracking gains。
5. Manifold projection controller：
   - 计算 top-k neighbor constraints。
   - 维护每个 agent 的 slack `mu`。
   - 计算 `psi, J_u, c, J_u^dagger, B_u`。
   - 输出 `u_s`。
6. PPO update 改成 high-level semi-MDP：
   - reward 用 `tau` 步累计。
   - bootstrap 使用 epoch 末 state。
   - discount 使用 `gamma ** tau`。
   - GAE 使用 `(gamma ** tau) * lambda`。
   - policy log_prob 对应 subgoal，而不是低层每步 action。

优点：

- 与 HMM 论文结构一致。
- 可以保留“低层固定，因此高层 MDP stationary”的论证。
- 更适合复现实验表述。

缺点：

- 改动较大。
- 需要仔细处理 JAX scan、slack state、top-k neighbor switching 和离散时间误差。

## 对用户问题的直接回答

“能不能直接用单步的 InforMARL 接上这个约束流形投影方法？”

可以，但只能作为 **单步 action safety filter**：

```text
graph -> InforMARL actor -> u_ref -> constraint manifold projection -> u_s -> env.step
```

这条路工程上可行，且最小改动。

但如果目标是论文 HMM 的方法，则不能直接这么接。HMM 需要：

```text
graph at epoch k -> high-level InforMARL-style GNN policy -> relative subgoal z
for tau low-level steps:
    subgoal tracking controller -> u_ref
    constraint manifold projection -> u_s
    env.step(u_s)
high-level PPO update on aggregated tau-step transition
```

因此，建议命名上区分：

- `informarl_manifold_filter`：单步 InforMARL + manifold projection，最小可跑。
- `hmm`：高层 subgoal + 低层 manifold controller，论文复现路线。

## 推荐下一步

先实现路线 A 作为快速 sanity check，验证：

- projection controller 是否能在当前 LidarSpread 动力学中稳定运行；
- JAX `jit/scan/vmap` 下的 pseudo-inverse 和 top-k 约束是否数值稳定；
- safety rate 是否明显改善；
- task reward 是否因过强 projection 降低。

路线 A 跑通后，再推进路线 B。这样可以把问题拆开：先验证 manifold controller，再改 high-level semi-MDP 训练。
