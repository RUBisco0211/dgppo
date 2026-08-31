# InforMARL + Graph-HJ + DGPPO 混合更新设计

## 1. 结论与定位

这个方案可以作为一个可验证的研究基线实现，并继续遵循 CTDE：安全 critic 离线集中训练，RL 阶段使用集中可见的联合动作计算安全更新信号，执行阶段仍然只有共享的局部图 actor。

它不是运行时安全过滤器，也不提供逐步硬安全保证。更准确的名称是：使用 Deep-QP HJ 损失预训练分布式 Graph-HJ critic，再照搬 DGPPO 的逐样本 task/safety advantage 混合机制训练策略。

本实现采用以下边界：

- 只支持 `VMASNavigation` 和 `VMASNavigationObs`。
- 不调用环境显式动力学函数。
- 不修改 VMAS reward、cost 或 step 动力学。
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

保留的结构包括连续安全约束适配器、独立 replay、Deep-QP 双值头、目标网络、HJ 损失和安全 checkpoint。

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

## 4. 独立 off-policy 预训练

预训练器用 OU、均匀和 bang-bang 动作的混合分布收集 transition：

$$
(\mathcal G_t, u_t, c_t, \mathcal G_{t+1}, c_{t+1}, d_t)
$$

采样动作覆盖完整联合动作空间。replay 对安全边界附近及不安全样本做优先采样。预训练过程不读取 task reward，也不执行当前 critic 导出的 QP 或策略，避免 collector 与 critic 共同形成自举过滤闭环。

训练命令示例：

```bash
python train_safety_filter.py \
  --env VMASNavigationObs -n 3 --obs 3 \
  --output-dir ./logs/deep_qp_safety/vmas_navigation_obs
```

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

## 6. DGPPO 风格的混合策略更新

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

混合后的 advantage 进入原有 clipped PPO objective。task value critic 始终更新；HJ critic 在 RL 阶段始终冻结。与上一版 batch-level CRPO switch 不同，这里没有全局阈值，也不会让整个 batch 在 task update 和 safety update 之间二选一。

训练命令示例：

```bash
python train.py \
  --env VMASNavigationObs --algo informarl_hj_crpo \
  -n 3 --obs 3 --no-rnn \
  --deep-qp-checkpoint ./logs/deep_qp_safety/vmas_navigation_obs/deep_qp_safety.pkl \
  --hj-cbf-alpha 1.0 --cbf-weight 1.0
```

`informarl_deep_qp` 暂时保留为同一实现的兼容别名，新实验应使用 `informarl_hj_crpo`。

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

## 8. 代码组织

- `dgppo/env/safety_constraint.py`：只读图状态的连续安全 margin。
- `dgppo/algo/module/deep_qp_safety.py`：Graph-HJ 网络、联合方向导数、Deep-QP loss、checkpoint。
- `dgppo/trainer/safety_buffer.py`：离线 HJ replay。
- `train_safety_filter.py`：独立 off-policy critic 预训练。
- `dgppo/algo/informarl_deep_qp.py`：冻结 critic 的 DGPPO 风格混合 PPO 更新。
- `tests/test_deep_qp_safety.py`：约束、联合系数、HJ update、标准 rollout 语义测试。

## 9. 当前仍然存在的问题

### 9.1 没有运行时硬安全保证

混合 PPO 更新改善的是采样分布上的期望安全，不会像可行 CBF-QP 那样逐步拒绝危险动作。测试时仍可能违反约束。

### 9.2 学习值函数不自动获得严格 CBF 性质

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

### 9.3 联合可行性没有被直接检查

多个局部 HJ 约束可能冲突。DGPPO 风格的 soft policy update 绕开在线联合 QP，但没有证明存在一个联合动作同时满足全部约束。后续应增加仅用于离线诊断的 centralized joint-QP feasibility oracle。

### 9.4 HJ 分解的可辨识性

标量项和动作系数项只通过 transition 导数监督，可能存在互相补偿。需要监控 coefficient norm、derivative residual，并用充分激励的联合动作采样和 held-out action counterfactual 检验方向导数。

### 9.5 动态拓扑和局部 Markov gap

通信边出现或消失会让输入图和 action mask 离散变化。局部图未必是闭合的 Markov 状态，邻居刚进入感知范围时尤其明显。当前实现保证的是结构一致性，不是 HJ 收敛定理。

### 9.6 混合更新的尺度敏感性

HJ violation 没有像 task advantage 一样标准化，因此更新强度直接依赖 critic residual 的标度和 `cbf_weight`。应联合监控 `hj_crpo/safe_data`、`hj_crpo/cbf_weight`、`violation_max` 和 `constraint_estimate`，并对 CBF 权重做消融。

### 9.7 checkpoint 不向后兼容旧方向导数头

旧原型的系数形状为每个 agent 一个自身动作向量，新实现是共享 pair head。旧 checkpoint 无法安全迁移，必须重新预训练。

## 10. 最小验证顺序

1. 单步测试联合系数 shape、邻接 mask、goal 特征不变性。
2. 小 replay 上确认 HJ loss、梯度、target update 有限。
3. held-out transition 检查预测导数与有限差分导数误差。
4. 固定 actor rollout，对比 HJ violation 与 VMAS 原生 cost 的 precision、recall。
5. 短程训练确认安全样本保留 task advantage，不安全样本只产生 HJ 修复项。
6. 与 InforMARL、DGPPO 比较 return、unsafe rate、最大违反和训练稳定性。
7. 最后再增加 centralized joint-QP oracle，量化局部约束冲突率。
