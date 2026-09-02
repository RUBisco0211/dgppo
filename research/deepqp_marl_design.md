# Deep-QP in MARL 当前方案与训练指南

本文是当前实现的唯一方案入口，统一记录理论设计、代码边界、训练命令、日志和已知问题。`research/` 中其余文档仅作为背景调研，不作为当前实现或运行流程的规格。

## 1. 结论与定位

这个方案可以作为一个可验证的研究基线实现，并继续遵循 CTDE：安全 critic 离线集中训练，RL 阶段使用集中可见的联合动作计算安全更新信号，执行阶段仍然只有共享的局部图 actor。

它不是运行时安全过滤器，也不提供逐步硬安全保证。更准确的名称是：使用 Deep-QP HJ 损失预训练分布式 Graph-HJ critic，再照搬 DGPPO 的逐样本 task/safety advantage 混合机制训练策略。

本文统一将第二阶段称为“Graph-HJ 混合更新”或“HJ-DGPPO 式更新”。

本实现采用以下边界：

- 支持 LidarEnv，以及 `VMASNavigation` 和 `VMASNavigationObs`。
- 不调用环境显式动力学函数。
- 不修改 LidarEnv/VMAS 的 reward、cost 或 step 动力学；连续安全约束只读取图观测。
- HJ critic 在 PPO 前单独用 off-policy 探索数据训练；PPO 阶段冻结。
- 不在 actor 执行路径中求解 QP。

## 2. 为什么旧的逐智能体 QP 路径需要移除

旧原型让每个局部值函数只输出 ego 动作的方向导数系数，然后独立投影 ego 动作。若局部值函数聚合了邻居状态，它通常也依赖邻居动作。真实的局部导数应写成：

$$
\dot V_i(z_i, u_{\mathcal N_i})
=
\sum_{j \in \mathcal N_i} a_{ij}(z_i)^\top u_j + b_i(z_i)
$$

只保留其中的 ego 项会把其他智能体动作造成的变化错误归入漂移项，而且多个 ego QP 不能保证所有耦合约束同时成立。因此以下结构被判定为多余并移除：

- `SafeRollout` 中 nominal action 与 executed action 的双重语义；
- 逐智能体 box-halfspace QP；
- 训练和评估时的 runtime filter 分支；
- filter intervention、infeasible、action delta 指标；
- PPO 期间在线更新 safety critic 的 replay 路径。

保留的结构包括直接读取环境 cost 的安全目标、独立 replay、Deep-QP 双值头、目标网络、HJ 损失和安全 checkpoint。

## 3. 分布式 Graph-HJ critic

### 3.1 输入和局部性

每个智能体拥有一个共享参数的局部值函数：

$$
V_i = V_\psi(z_i)
$$

其中局部图观测为：

$$
z_i = \mathcal G_i(x)
$$

critic 复用项目中的 `GraphTransformerGNN`。任务 goal 特征和 centralized `env_states` 会从 safety critic 输入中移除。actor 仍然使用原始任务图。

### 3.2 联合动作方向导数头

GNN 先产生每个 agent 的上下文嵌入。共享 pair head 对每个有效图边输出一个方向导数系数：

$$
a_{ij} = f_{\mathrm{pair}}(h_i, h_j)
$$

系数只在 agent 到 agent 的局部图边以及自环上有效：

$$
a_{ij} = 0, \qquad j \notin \mathcal N_i
$$

因此 critic 的动作相关输出形状为：

$$
A \in \mathbb R^{N \times N \times d_u}
$$

这个 shared pair head 对 agent 排列是等变的，不使用固定 agent ID 输出槽。

### 3.3 Deep-QP 导数分解

代码保持 Deep-QP 的 support-function 参数化。动作盒集合记为：

$$
\mathcal U = [u_{\min}, u_{\max}]^N
$$

联合动作系数的 support function 为：

$$
\sigma_{\mathcal U}(a_i)
=
\max_{u \in \mathcal U}
\sum_j a_{ij}^\top u_j
$$

critic 对实际联合动作给出的导数为：

$$
\widehat{\dot V}_i
=
\sum_j a_{ij}^\top u_j
-
\sigma_{\mathcal U}(a_i)
+
d_i
$$

其中标量项为：

$$
d_i
=
\delta_i
-
\lambda(c_i - V_i)
$$

HJ 训练继续使用双值头、target network、Bellman 值回归以及方向导数分解损失。关键变化只是把原来的自身动作内积改为完整局部联合动作内积。

### 3.4 Value head 与导数 head 必须联合训练

当前实现中的 Value 网络和导数网络不是两个可以独立开关的 critic。它们共享
同一个 `LocalSafetyGNN` 编码器，只在输出端分成两个 Value head、
`PairCoefficientHead` 和 `ScalarHead`。第一阶段的总损失为：

$$
L_{\mathrm{safety}}
=
w_V L_V
+
w_D\left(L_{\mathrm{coefficient}}+L_{\mathrm{scalar}}\right).
$$

导数部分通过两条路径影响 HJ Value：

1. derivative loss 会通过共享 GNN 的梯度改变 Value head 使用的图嵌入；
2. Value 的 Bellman target 本身使用 target coefficient、target scalar 和动作
   support function 构造，因此导数输出不是单纯的辅助监督。

所以，简单把 `derivative_loss_weight` 设为 0 并不能得到有效的 value-only
训练：未训练的导数 head 仍会进入 Value target，其随机或失真的输出会污染
Value 学习。当前第二阶段还直接使用 coefficient/scalar head 构造
$\widehat{\dot V}$ 和 HJ/CBF residual，因此禁用导数训练也会使策略约束失效。

若未来把第二阶段改成 rollout 有限差分 residual，才可以另外设计真正的
value-only 版本；届时必须同时重写 Value Bellman target、网络输出结构和
第二阶段 residual，不能只关闭 derivative loss。当前训练流程必须保留导数
head 的联合训练。若需要降低训练耗时，应优先调整预训练更新次数、batch size
或网络宽度。

## 4. 独立 off-policy 预训练

预训练器用 OU、均匀和 bang-bang 动作的混合分布收集 transition：

$$
(\mathcal G_t, u_t, c_t, \mathcal G_{t+1}, c_{t+1}, d_t)
$$

采样动作覆盖完整联合动作空间。replay 对安全边界附近及不安全样本做优先采样。预训练过程不读取 task reward，也不执行当前 critic 导出的 QP 或策略，避免 collector 与 critic 共同形成自举过滤闭环。

完整两阶段训练使用第 8 节的 `--algo deepqp` 入口。若只需单独预训练 safety critic，可使用底层入口：

```bash
python train_safety_filter.py \
  --env LidarSpread -n 3 --obs 3 \
  --output-dir ./logs/deep_qp_safety/lidar_spread
```

预训练完成后，可以在保持 Graph-HJ 参数树完全不变的前提下，把冻结
critic 加载到不同 agent 数量的第二阶段：

```bash
python train.py \
  --env LidarSpread --algo informarl_deep_qp -n 8 --obs 3 \
  --deep-qp-checkpoint \
    ./logs/deep_qp_safety/lidar_spread/deep_qp_safety.pkl \
  --deep-qp-allow-agent-count-transfer
```

实现会按照目标数量重新创建环境、Graph-HJ 输出张量和 actor RNN state，
然后加载共享 GNN/value/scalar/pair-head 参数。该选项只跳过 checkpoint 中
`n_agents` 的不一致；网络结构、动力学、通信半径、动作范围、环境 cost 来源和
`top_k_rays` 仍按原规则严格检查。默认不启用该选项，因此原有同数量训练和
恢复训练保持原行为。数量迁移仅提供工程上的 eval/冻结使用能力，不把未见
局部邻居密度下的安全泛化当作理论保证。

## 5. RL 阶段的 CBF 残差

冻结 critic 后，对 rollout 中的联合动作计算：

$$
r_i
=
\widehat{\dot V}_i
+
\alpha(V_i - m)
$$

违反量定义为：

$$
\ell_i
=
\left[\epsilon-r_i\right]_+
$$

其中安全符号约定为值越大越安全，因此期望残差非负。

每个动作拥有者需要对所有显式依赖其动作的局部约束负责。其安全违反量为：

$$
\ell_j^{\mathrm{owner}}
=
\max_{i:\,j\in\mathcal N_i}\ell_i
$$

这个归因步骤只在 centralized training 使用。执行时 actor 不需要其他智能体动作。

## 6. DGPPO 式逐样本 advantage 混合

任务 advantage 继续使用 InforMARL 的 reward critic 和 GAE，记为：

$$
A_{j,t}^{\mathrm{task}}
$$

对每个动作拥有者，先判断它参与的局部 HJ 约束是否全部安全：

$$
s_{j,t}
=
\mathbf 1
\left[
\ell_{j,t}^{\mathrm{owner}}=0
\right]
$$

DGPPO 的混合规则被直接迁移为：安全样本保留 task advantage；不安全样本把 task advantage 置零，并沿降低 HJ violation 的方向更新。

$$
A_{j,t}
=
s_{j,t} A_{j,t}^{\mathrm{task}}
-
w_t \ell_{j,t}^{\mathrm{owner}}
$$

CBF 权重使用与 DGPPO 相同的分段调度：

$$
w_t
=
\begin{cases}
w_0,
& t < 0.5T
\\
2w_0,
& 0.5T \leq t < 0.75T
\\
4w_0,
& t \geq 0.75T
\end{cases}
$$

混合后的 advantage 进入原有 clipped PPO objective。task value critic 始终更新；HJ critic 在 RL 阶段始终冻结。同一个 batch 内可以同时包含 task 样本和 safety-repair 样本。

## 7. CTDE 数据流

预训练阶段：

```text
局部图 + 随机联合动作
          |
          v
off-policy replay -> Deep-QP HJ loss -> frozen Graph-HJ checkpoint
```

RL 阶段：

```text
共享局部 actor -> 普通联合 rollout -> reward GAE
                               |
                               +-> frozen Graph-HJ residual
                                           |
                                           v
                              DGPPO per-sample advantage mixing
                                           |
                                           v
                                      clipped PPO
```

执行阶段：

```text
局部图 -> 共享 actor -> 本智能体动作
```

## 8. 训练入口、参数与日志

### 8.1 一条命令完成两阶段训练

LidarEnv 上的推荐命令为：

```bash
python train.py --env LidarSpread --algo deepqp -n 3 --obs 3
```

`deepqp` 是组合训练入口，会顺序执行：

1. 创建统一实验目录、W&B run ID 和本地 metrics 文件；
2. 使用 off-policy replay 预训练 Graph-HJ critic；
3. 保存并冻结 `deep_qp_safety.pkl`；
4. 使用冻结 critic 启动第二阶段 Graph-HJ 混合 PPO 训练。

`--steps` 表示第二阶段 PPO 训练步数，第一阶段使用独立的 `--deep-qp-pretrain-steps`。无需 shell 包装器，也无需手工传递两个阶段之间的 checkpoint。

### 8.2 参数组织

以下参数会同时传给两个阶段，必须保持一致：

- `--deep-qp-gnn-layers`
- `--deep-qp-gnn-out-dim`
- `--deep-qp-hidden-dim`
- `--deep-qp-hidden-layers`
- `--deep-qp-lambda-init`
- `--deep-qp-lambda-final`
- `--deep-qp-lambda-decay-steps`
- `--deep-qp-constraint-scale`

Graph-HJ 不定义额外的几何 margin。第一阶段直接读取环境的
`env.get_cost(graph)`；由于环境 cost 是 unsafe-positive 且含多个通道，
存入 Graph-HJ replay 的标量为 `-max(cost_channels)`。

第一阶段的采样与优化参数使用 `--deep-qp-pretrain-*` 参数组，包括 environment 数、rollout 长度、warmup、batch size、replay size、更新次数以及保存/评估间隔。

第二阶段继续使用 `--steps`、`--n-env-train`、`--batch-size`、`--hj-cbf-alpha`、`--hj-cbf-margin`、`--hj-cbf-eps` 和 `--cbf-weight` 等原有 RL 参数。组合入口不会覆盖其他算法的优化超参数。

### 8.3 输出目录与统一日志

```text
logs/<env>/deepqp/seed<seed>_<timestamp>_<id>/
├── training_metrics.jsonl
├── wandb_run_id
├── deep-qp/
│   ├── deep_qp_safety.pkl
│   ├── deep_qp_replay.pkl
│   └── deep_qp_training_state.pkl
└── rl/<env>/<stage2-run>/
    ├── config.yaml
    ├── models/latest/
    └── videos/latest_eval.mp4
```

两个阶段共同追加根目录的 `training_metrics.jsonl`，并复用同一个 W&B project、run ID 和 run name。第一阶段指标全部位于 `deep-qp/*` 命名空间，主要包括：

- value、derivative、coefficient 和 scalar 分项 loss；
- derivative residual、value bound violation、coefficient norm 和 gradient norm；
- replay size、constraint mean/min、unsafe sample rate；
- 固定 validation transitions 上的 `deep-qp/eval/safety/*` 指标。

第二阶段记录 reward/cost、unsafe rate、actor/value loss、entropy、clip fraction、HJ residual、violation、CBF weight、吞吐与耗时，其中策略训练产生的 HJ 指标统一位于 `deep-qp/policy/*` 命名空间。训练过程周期性生成 deterministic eval 视频，主 W&B 横轴为 `counters/total_frames`。没有 FFmpeg 时视频降级为 GIF；渲染失败不会阻止 scalar 日志或 checkpoint 保存。

第一阶段没有策略视频，因为该阶段只训练值函数。固定验证集只能检查经验误差和数值退化，不能替代连续状态空间上的前向不变性证明。

#### 8.3.1 第一阶段指标使用的量

记 `constraint_scale` 为约束归一化尺度，环境给出的局部安全约束为 `constraint`，网络实际使用的归一化约束为：

$$
\widetilde c_i=\frac{c_i}{s_c}.
$$

当前符号约定是 `constraint >= 0` 表示安全，`constraint < 0` 表示已经违反环境安全约束。网络输出两个 value head，并使用二者的较小值作为安全 value：

$$
V_i=\min\left(V_i^{(1)},V_i^{(2)}\right).
$$

对局部证书所有者 `i`，联合动作方向导数头输出系数 `a_ij` 和标量项。动作盒约束的 support function 为：

$$
\sigma_{\mathcal U}(a_i)
=
\max_{\mathbf u\in\mathcal U}
\sum_j a_{ij}^{\mathsf T}u_j.
$$

代码中的动作相关导数项为：

$$
q_i(\mathbf u)
=
\sum_j a_{ij}^{\mathsf T}u_j
-
\sigma_{\mathcal U}(a_i).
$$

以下第一阶段指标均在最新 replay minibatch 上计算，并写入 `deep-qp/safety/*`：

| 指标 | 含义 | 期望趋势 |
|---|---|---|
| `deep-qp/safety/loss` | 实际反向传播的总损失，即 value loss 与 derivative loss 的加权和。 | 下降并稳定；不能单独作为证书有效性的结论。 |
| `deep-qp/safety/value_loss` | 两个 value head 对固定点/Bellman target 的均方误差之和，并按环境时间步长缩放。 | 下降。 |
| `deep-qp/safety/derivative_loss` | 结构化方向导数拟合误差，等于 coefficient loss 与 scalar loss 之和。 | 下降。 |
| `deep-qp/safety/coefficient_loss` | 检查联合动作系数产生的动作项与 target 导数分解是否一致。 | 下降。 |
| `deep-qp/safety/scalar_loss` | 检查与动作无关的标量导数项是否补足 target 导数。 | 下降。 |
| `deep-qp/safety/value_bound_violation` | minibatch 中满足 `V > normalized_constraint` 的样本比例。HJ safety value 应不大于瞬时约束，因此该事件表示 value 上界被破坏。 | 越接近 0 越好。 |
| `deep-qp/safety/value_mean` | minibatch 上安全 value 的均值。它反映当前数据整体离危险边界的相对位置。 | 仅诊断；不能简单认为越大或越小越好。 |
| `deep-qp/safety/target_online_gap` | online value 与 target value 的平均绝对差。 | 通常应逐渐缩小并保持有限；过大表示 target 跟踪或训练不稳定。 |
| `deep-qp/safety/derivative_residual` | 预测总方向导数与经验 target 导数之间的平均绝对残差。 | 越小越好，是判断导数头是否学到有效信号的核心指标之一。 |
| `deep-qp/safety/coefficient_norm` | 联合动作系数张量的平均范数。 | 仅诊断；突然爆炸通常表示导数头不稳定，接近 0 则可能表示动作影响被忽略。 |
| `deep-qp/safety/lambda` | 当前 contraction/HJ 折扣系数，按配置从初值五次多项式衰减到终值。 | 由 schedule 决定，不是训练质量指标。 |
| `deep-qp/safety/grad_norm` | safety critic 本次更新的原始全局梯度范数。优化器随后再执行梯度裁剪。 | 应保持有限；持续尖峰需要检查 loss、数据尺度或学习率。 |
| `deep-qp/safety/has_nan` | 本次 loss 或梯度是否出现非有限值；`1` 表示异常。 | 必须为 0。当前实现检测到 `1` 会停止训练。 |
| `deep-qp/safety/update_applied` | 本次 safety critic 参数更新是否实际执行；`1` 表示成功。 | 正常训练应为 1。 |

其中总损失关系为：

$$
L_{\mathrm{safety}}
=
w_V L_V+w_D
\left(L_{\mathrm{coefficient}}+L_{\mathrm{scalar}}\right).
$$

第一阶段还记录采样、进度和性能指标：

| 指标 | 含义 | 备注 |
|---|---|---|
| `deep-qp/counters/update` | 已成功完成的 safety critic 梯度更新次数。 | 第一阶段 W&B 指标的横轴。 |
| `deep-qp/counters/replay_size` | 当前 replay buffer 中的 transition 数量。 | 达到 warmup 和 batch size 要求后才开始更新。 |
| `deep-qp/data/constraint_mean` | 最新采集 batch 中原始环境约束的均值。 | 未除以 `constraint_scale`。 |
| `deep-qp/data/constraint_min` | 最新采集 batch 中最小的原始环境约束。 | 小于 0 表示 batch 内至少存在违反样本。 |
| `deep-qp/data/unsafe_rate` | 最新采集 batch 中 `constraint < 0` 的元素比例。 | 衡量采样数据中的环境约束违反率。 |
| `deep-qp/time/elapsed_sec` | 第一阶段从本次启动开始累计的墙钟时间。 | 续训后重新计时。 |
| `deep-qp/performance/updates_per_sec` | 平均每秒 safety critic 更新次数。 | 用于吞吐诊断。 |

`deep-qp/eval/safety/*` 与 `deep-qp/eval/data/*` 使用相同定义，但始终在启动时固定采集的 validation transitions 上计算。它们比训练 minibatch 指标更适合观察泛化和过拟合。eval loss 不执行反向传播，因此没有 eval 版本的 `grad_norm`、`has_nan` 和 `update_applied`。

#### 8.3.2 第二阶段策略指标

第二阶段冻结第一阶段的 target HJ 网络。对 rollout 中每个局部证书所有者 `i`，代码计算：

$$
r_i
=
\sum_j a_{ij}^{\mathsf T}u_j
-
\sigma_{\mathcal U}(a_i)
+
d_i
+
\alpha
\left(
V_i-\frac{m}{s_c}
\right),
$$

其中 `d_i` 是 critic 输出并经过 contraction 修正的最大导数标量项，`m` 对应 `hj_cbf_margin`。当前 safe-positive 符号约定要求：

$$
r_i\geq\varepsilon.
$$

因此局部违反量为：

$$
\ell_i
=
\max\left(\varepsilon-r_i,0\right).
$$

动作所有者 `j` 接收所有显式依赖 `u_j` 的局部证书中最坏的违反量：

$$
\ell_j^{\mathrm{owner}}
=
\max_{i:M_{ij}=1}\ell_i.
$$

第二阶段新增的 W&B 和本地日志指标如下：

| 指标 | 含义 | 期望趋势 |
|---|---|---|
| `deep-qp/policy/constraint_estimate` | 先在每个 rollout 时刻对所有局部证书的违反量取最大值，再对 batch 和时间求均值，即“平均最坏局部 HJ 违反量”。它不是环境 cost。 | 越接近 0 越好。 |
| `deep-qp/policy/safe_data` | 所有 batch、时间和动作所有者位置中满足 `owner_violation == 0` 的比例。 | 越接近 1 越好。 |
| `deep-qp/policy/cbf_weight` | 当前混合策略更新中 HJ 违反项的权重。启用 schedule 时会在训练进度达到一半和四分之三时分别增大。 | 由 schedule 决定，不是性能指标。 |
| `deep-qp/policy/violation_mean` | 所有局部证书违反量的平均值。 | 越接近 0 越好。 |
| `deep-qp/policy/violation_max` | 当前 PPO rollout 中出现的最大局部证书违反量。 | 越接近 0 越好，用于发现均值掩盖的极端失败。 |
| `deep-qp/policy/residual_min` | 当前 PPO rollout 中最小的 HJ/CBF residual。 | 越大越安全；低于 `hj_cbf_eps` 表示至少存在违反。 |
| `deep-qp/policy/value_min` | 当前 PPO rollout 上所有局部 HJ value 的最小值。 | 仅诊断危险边界；更负通常表示存在更危险状态，但必须结合 value 标定和环境 cost 判断。 |
| `deep-qp/policy/neighborhood_density` | `action_mask` 中 `True` 的比例，即局部证书与联合动作变量之间的平均耦合密度。 | 仅诊断图规模和 credit assignment 复杂度，不是安全率。 |

`train/agents/safe_ratio` 是 `deep-qp/policy/safe_data` 的通用训练面板别名，两者数值相同。策略指标来自用于 PPO 更新的训练 rollout；当前 deterministic eval 仍以环境 reward、cost 和 unsafe rate 为主，并不会重新命名成 `deep-qp/policy/*`。

#### 8.3.3 指标解释边界

- `deep-qp/data/unsafe_rate` 使用环境约束 `constraint < 0` 定义；`deep-qp/policy/safe_data` 使用学习到的 HJ residual 定义。二者既不是同一个标签，也不是严格互补量。
- loss 下降只能说明网络更好地拟合当前 replay/validation target，不能单独证明学习到的 value 满足连续状态空间上的 CBF 条件。
- `safe_data` 接近 1 只说明当前训练 rollout 上网络判断 residual 满足阈值；仍需结合环境 `eval/unsafe_frac`、各项 eval cost 和分布外测试判断真实安全性。
- `value_mean`、`value_min` 和 `coefficient_norm` 主要用于检测漂移、塌缩或爆炸，不应作为独立的模型选择目标。
- 模型选择时应联合观察 validation `derivative_residual`、`value_bound_violation`、第二阶段 `violation_mean/max`、环境 eval unsafe rate 与任务回报，避免只优化单一指标。

### 8.4 恢复边界

`--algo deepqp` 当前只负责 fresh two-stage training，不接受 `--resume-dir`。第一阶段可通过 `train_safety_filter.py --resume <checkpoint>` 单独恢复；统一入口暂未提供第二阶段恢复选项。

### 8.5 代码组织

- `dgppo/algo/module/deep_qp_safety.py`：Graph-HJ 网络、联合方向导数、Deep-QP loss、checkpoint。
- `dgppo/trainer/safety_buffer.py`：离线 HJ replay。
- `train_safety_filter.py`：独立 off-policy critic 预训练。
- `dgppo/algo/informarl_deep_qp.py`：冻结 critic 的 DGPPO 式逐样本混合 PPO 更新。
- `tests/`：约束、联合系数、HJ update、advantage 混合、动作责任归因以及 VMAS/LidarEnv rollout/update 测试。

### 8.6 最小端到端检查

下面的命令只用于验证阶段切换、checkpoint 传递和日志，不代表正式训练配置：

```bash
python train.py --env LidarSpread --algo deepqp -n 2 --obs 1 \
  --deep-qp-pretrain-steps 1 --deep-qp-pretrain-n-env 1 \
  --deep-qp-pretrain-rollout-steps 2 \
  --deep-qp-pretrain-updates-per-collect 1 \
  --deep-qp-pretrain-warmup 1 --deep-qp-pretrain-batch-size 1 \
  --deep-qp-pretrain-replay-size 4 --deep-qp-pretrain-save-interval 1 \
  --deep-qp-pretrain-log-interval 1 --deep-qp-pretrain-eval-interval 1 \
  --deep-qp-pretrain-eval-n-env 1 --deep-qp-gnn-out-dim 8 \
  --deep-qp-hidden-dim 16 --deep-qp-hidden-layers 1 --steps 0 \
  --n-env-train 1 --n-env-test 1 --batch-size 128 \
  --no-rnn --no-video --wandb-mode disabled --log-dir /tmp/deepqp-smoke
```

## 9. 与原有算法的兼容性边界

这里把“核心算法流程”和“训练外围设施”分开定义。核心算法流程包括 rollout 采样语义、advantage 计算、loss、梯度计算和参数更新；日志、视频、checkpoint 文件组织和 W&B 连接方式不属于核心算法流程。

从 commit `42a1b59aee784385b331a025280a2cd19721d812` 到当前实现，兼容性结论如下：

- 原有算法实现文件没有修改，其 rollout、CBF/cost advantage、loss 和更新公式保持不变。
- `informarl` 的 PPO rollout、reward GAE、clipped objective 和参数更新没有修改；仅 `save/load` 增加了优化器、训练步数和 PRNG 状态保存。
- `informarl_manifold` 继承 `informarl`，因此核心更新不变，但同样继承新的 checkpoint 行为。
- 只有 `deepqp` 组合入口会预训练并实例化冻结的 Graph-HJ critic，执行第 5、6 节的安全 residual 和混合 advantage；其他 `--algo` 值仍直接进入原有 RL 训练分支。
- 新增的 `SafetyBatch`、safety replay 和 Graph-HJ 网络不会进入其他算法的 update 路径。Graph-HJ 直接读取原有环境的 `env.get_cost()`；环境的 reward、cost 和 dynamics 文件没有被修改。

共享 `Trainer` 存在外围行为变化，适用于所有算法：增加本地 JSONL 指标、W&B mode/run ID、计时指标、视频失败容错、训练状态 sidecar 和循环结束后的最终 checkpoint。联网检测也从固定返回在线改为真实探测。这些变化不参与 loss 或梯度计算，因此不会改变 fresh training 的数值更新规则，但会改变日志、checkpoint 内容和部分续训语义。当前 `counters/stage=2` 是 Trainer 的统一日志标签，普通算法出现该字段不表示它执行了 Deep-QP 第二阶段。

`test.py --stochastic` 的 actor 调用签名也被修正为当前 `Algorithm.step(graph, rnn_state, key)` 接口；deterministic evaluation 路径不变。

当前已有测试覆盖 Deep-QP/Graph-HJ 混合更新的关键路径；本次兼容性审计也验证了原有四个算法仍可由 factory 正常构造。尚未建立从上述 commit 出发、对原算法完整训练轨迹做逐参数 bitwise 对比的回归测试。因此这里能确认的是“核心公式和代码路径未修改”，而不是跨硬件、跨 JAX 版本的逐位一致性保证。

## 10. 当前仍然存在的问题

### 10.1 没有运行时硬安全保证

混合 PPO 更新改善的是采样分布上的期望安全，不会像可行 CBF-QP 那样逐步拒绝危险动作。测试时仍可能违反约束。

### 10.2 学习值函数不自动获得严格 CBF 性质

有限 replay、函数逼近误差和优化误差意味着 HJ residual 只在训练分布上近似成立。当前没有对整个连续状态空间验证：

$$
\sup_{x\in\mathcal X}
\left[
-\max_{u\in\mathcal U}
\left(
\dot V(x,u)+\alpha V(x)
\right)
\right]_+
=0
$$

因此不能声称已证明前向不变性。

### 10.3 联合可行性没有被直接检查

多个局部 HJ 约束可能冲突。DGPPO 风格的 soft policy update 绕开在线联合 QP，但没有证明存在一个联合动作同时满足全部约束。后续应增加仅用于离线诊断的 centralized joint-QP feasibility oracle。

### 10.4 HJ 分解的可辨识性

标量项和动作系数项只通过 transition 导数监督，可能存在互相补偿。需要监控 coefficient norm、derivative residual，并用充分激励的联合动作采样和 held-out action counterfactual 检验方向导数。

### 10.5 动态拓扑和局部 Markov gap

通信边出现或消失会让输入图和 action mask 离散变化。局部图未必是闭合的 Markov 状态，邻居刚进入感知范围时尤其明显。当前实现保证的是结构一致性，不是 HJ 收敛定理。

### 10.6 混合更新的尺度敏感性

HJ violation 没有像 task advantage 一样标准化，因此更新强度直接依赖 critic residual 的标度和 `cbf_weight`。应联合监控 safe-data ratio、当前 CBF weight、violation max 和 constraint estimate，并对 CBF 权重做消融。

### 10.7 checkpoint 不向后兼容旧方向导数头

旧原型的系数形状为每个 agent 一个自身动作向量，新实现是共享 pair head。旧 checkpoint 无法安全迁移，必须重新预训练。

## 11. 最小验证顺序

1. 单步测试联合系数 shape、邻接 mask、goal 特征不变性。
2. 小 replay 上确认 HJ loss、梯度、target update 有限。
3. held-out transition 检查预测导数与有限差分导数误差。
4. 固定 actor rollout，对比 HJ violation 与 VMAS 原生 cost 的 precision、recall。
5. 短程训练确认安全样本保留 task advantage，不安全样本只产生 HJ 修复项。
6. 与 InforMARL、DGPPO 比较 return、unsafe rate、最大违反和训练稳定性。
7. 最后再增加 centralized joint-QP oracle，量化局部约束冲突率。
