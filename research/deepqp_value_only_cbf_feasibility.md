# Deep-QP 的 value-only、on-policy CBF 简化：可行性审计

> 结论基于本地 Deep-QP 原论文、DGPPO 原论文和当前仓库实现。本文只保留决定可行性的公式与证明条件；它是理论设计审计，不是对现有神经网络 checkpoint 的安全认证。

## 1. 结论

**可以去掉导数网络，并改成 on-policy 更新；但必须改变问题定义和 Bellman loss，不能只把 derivative loss 关掉。** 最干净的方案是学习最终确定性分布式策略的 HJ **policy value**

$$
V_i^\mu(x)=\inf_{k\ge 0}c_i(x_k^\mu),
$$

并把它作为离散时间 Graph CBF。若该 value 精确、同一最终策略 $\mu$ 被部署，且局部图是充分状态，则其零上水平集天然前向不变，不需要 $\nabla V$、导数网络或 CBF-QP。

但结论有三条硬边界：

1. 这保证的是**固定策略 $\mu$ 的不变集**，一般小于 Deep-QP 最优 HJ value 的最大 viability kernel；不能继续声称“最大可控不变集”。
2. SGD、有限 rollout 和有限 on-policy 样本本身不构成严格证明。严格保证还需要对最终网络和最终策略做全域 one-step 条件验证，或给出有效的 uniform error/Lipschitz 界。
3. 用每步有限差分得到的是**离散时刻**前向不变性。若要声称连续时间内不碰撞，还必须观测区间内的最小 constraint，或加入 inter-sample margin。

因此，本项目最合理的简化定位是：

> **Value-only On-policy Graph Reachability Barrier**：Deep-QP 的 HJ min-return 语义 + DGPPO 式确定性 policy evaluation + 无导数的离散 barrier policy update。

它理论上可行，工程复杂度会显著下降；但“训练后严格可证”必须由独立 verification 步骤闭环。

## 2. 为什么导数网络可以删，但现有实现不能直接删

Deep-QP 的 undiscounted value 是

$$
V^*(x)=\sup_\pi\inf_{t\ge0}c(x_t^\pi),
$$

其零上水平集是最大 control-invariant safe set；原文随后引入 $Q_s(x,u)$，是因为最优安全策略未知，式 (1) 不能直接用任意行为数据学习（Deep-QP §3.2，原文第 68--92 行）。导数网络承担了两个额外角色：

- 用任意动作 transition 学习最优 HJ value 时，估计 action-dependent advantage $q^\lambda(x,u)$；
- 将方向导数写成 action-affine 形式，供连续时间 QP 约束使用（Deep-QP §3.3--3.4，第 130--170、190--252 行）。

一旦不再做 QP，并限定为 on-policy policy evaluation，就可以直接观察
$(x_t,c_t,x_{t+1})$，不必另外拟合 $\dot V(x,u)$。复杂度不会凭空消失，而是把“任意 off-policy 动作的校正”替换为“当前策略的数据分布和策略改进”。

当前代码不能只设 `derivative_loss_weight=0`，因为：

- 网络同时输出双 value、pair coefficient 和 scalar head（`dgppo/algo/module/deep_qp_safety.py:143-213`）；
- value target 的 `q_part_for_value` 仍由 coefficient/scalar 构造（同文件 `:541-559`）；
- PPO 安全信号直接调用导数头形成 residual（同文件 `:463-480`；`dgppo/algo/informarl_deep_qp.py:252-280`）；
- safety critic 在 PPO 阶段只以 frozen target params 使用，没有 on-policy 更新（`dgppo/algo/informarl_deep_qp.py:162-193`）；
- 独立预训练器使用随机动作混合、replay 和多次更新，是 off-policy 流程（`train_safety_filter.py:92-168,299-310`）。

所以简化必须同时替换网络输出、critic target、policy residual 和采样流程。

## 3. 方案 A：固定策略的 Policy-HJ CBF（推荐先做）

### 定义和 Bellman 关系

对最终部署的确定性联合策略 $\mu$，定义每个 agent 的正值安全 constraint：

$$
V_i^\mu(x)=\inf_{k\ge0}c_i(x_k^\mu).
$$

在确定性离散动力学 $x^+=F(x,\mu(x))$ 下有精确递推：

$$
V_i^\mu(x)=\min\{c_i(x),V_i^\mu(x^+)\}. \tag{A1}
$$

这正是原 Deep-QP HJ min-return 在固定策略、采样时刻下的版本；DGPPO 论文也明确使用确定性策略的 infinite-horizon constraint value 构造 DCBF（DGPPO §4.1，第 112--126 行）。

### 零上水平集不变性的证明骨架

令

$$
\mathcal C_\mu=\bigcap_i\{x:V_i^\mu(x)\ge0\}.
$$

由 (A1)，$V_i^\mu(x)\ge0$ 意味着 $c_i(x)\ge0$ 且
$V_i^\mu(x^+)\ge V_i^\mu(x)\ge0$。对所有 $i$ 同时成立，故
$x\in\mathcal C_\mu\Rightarrow x^+\in\mathcal C_\mu$；归纳即得离散时间前向不变，并且 $\mathcal C_\mu$ 包含在物理安全集内。

更一般地，只要最终网络 $h_i$ 和最终策略满足

$$
h_i(x)\ge0\ \forall i
\Longrightarrow
h_i(F(x,\mu(x)))\ge(1-\kappa)h_i(x),\quad 0\le\kappa\le1, \tag{A2}
$$

且 $h_i(x)\le c_i(x)$，同一归纳证明仍成立。这是 value-only 所需的全部 barrier 条件，不需要方向导数。

### On-policy 训练

每轮固定当前 actor 的确定性 mode $\mu_k$：

1. 用 $\mu_k$ 收集 deterministic rollouts；
2. 用反向 min-return/bootstrapped target 更新单一 Graph value head：

   $$
   y_{i,t}=\min\{c_{i,t},c_{i,t+1},\ldots,c_{i,t+H-1},
   \bar V_i(x_{t+H})\};
   $$

3. 固定 value target，在实际 stochastic PPO rollout 上计算一步残差

   $$
   r_{i,t}=h_i(x_{t+1})-(1-\kappa)h_i(x_t),
   \qquad \ell_{i,t}=[\varepsilon-r_{i,t}]_+;
   $$

4. 用 score-function/PPO surrogate 降低 $\ell$，安全样本再优化 task advantage；
5. actor 更新后重新收集数据、重新 evaluation。训练结束后同时冻结 actor/value，再做证书验证。

这与 DGPPO 的“确定性 rollout 学 constraint value、stochastic rollout 更新 actor”原则一致（DGPPO §4.5，第 247--265 行），但 value target 应保留 HJ 的 `min` 语义并使用正值安全符号。

**证明等级：** 在精确 policy evaluation 和 (A2) 全域成立时严格；普通神经网络训练只提供经验近似。该方案不保留最大 viability kernel，只认证最终策略能够维持的子集。

## 4. 方案 B：最优 HJ value-only generalized policy iteration

若必须保留 Deep-QP 的“最优可达集”语义，应先把多智能体 constraint 合成

$$
C(x)=\min_i c_i(x),
$$

并在**同一个可实现的分布式策略类** $\Pi_{\rm dec}$ 上定义

$$
V^*(x)=\sup_{\mu\in\Pi_{\rm dec}}\inf_{k\ge0}C(x_k^\mu). \tag{B1}
$$

不能分别对每个 $i$ 求 $\sup_\pi V_i$：各 agent 的最优值可能由互不兼容的联合策略实现，“每个局部约束存在控制”不推出“存在一个联合控制同时满足全部约束”。

对应的离散 Bellman 最优性关系是

$$
V^*(x)=\min\left\{C(x),\sup_{u\in\mathcal U_{\rm dec}(x)}V^*(F(x,u))\right\}. \tag{B2}
$$

用一个 value head 和一个分布式 safety actor 交替做：

- policy evaluation：on-policy min-return 回归；
- policy improvement：把观测到的 $V(x_{t+1})$ 或 barrier residual 当 pseudo-advantage，用 PPO/REINFORCE 提高其最坏局部值；
- 逐渐降低探索，最终部署 deterministic mode。

若 $V^*$ 精确且 actor 精确实现 (B2) 的共同 maximizer，则
$V^*(x)\ge0\Rightarrow C(x)\ge0$ 且 $V^*(x^+)\ge0$，所以零上水平集是 viability kernel 并前向不变。这保留了原 Deep-QP 的核心理论，却不再需要导数头。

**证明等级：** 只在全局 Bellman fixed point、精确共同 selector 和充分观测下严格。纯 on-policy PPO 一般只能得到局部最优策略，也没有覆盖所有 action，因此不能据训练收敛曲线声称学到了 (B1) 的最大 kernel。相比方案 A，它的理论目标更强，但实际认证明显更难。

原论文的 $\lambda>0$ discounted operator 可作为训练 continuation，但原文只证明 $\lambda\downarrow0$ 时回到 undiscounted value（Deep-QP §3.3，第 94--128 行）。若最终要声称原始 viability kernel，应在 $\lambda=0$ 的目标或直接验证的 (A2) 上落脚；有限 $\lambda$ 的网络不能仅凭 contraction 就获得前向不变性。

## 5. 神经网络何时才称得上“可证明”

训练 loss 小不等于全域条件成立。对最终候选 $h_i$，至少需要同时证明：

1. **集合包含：** $h_i(x)\le c_i(x)$。可用无 offset 的结构
   $h_i=c_i-\operatorname{softplus}(r_\psi)$ 强制；当前实现训练时允许 unclipped value，推理 clip 也不是独立的全域验证（`deep_qp_safety.py:204-210,501-512`）。
2. **一步闭包：** 对所有 $x\in\cap_i\{h_i\ge0\}$，最终 joint policy 均满足 (A2)，而不只是 rollout 平均成立。
3. **局部充分性：** 同一个局部图不能对应安全上互相冲突的隐藏状态/下一图；否则 $h_i(G_i)$ 不是 Markov value。当前 safety graph 会删除 goal、全局 states 和 `env_states`（`deep_qp_safety.py:407-428`），该假设必须单独论证。
4. **共同可行性：** 所有局部 inequality 必须由同一个联合分布式策略同时满足。
5. **最终策略一致：** critic 认证的是冻结后的部署策略；每次 PPO 更新都会使旧 $V^{\mu_k}$ 失效。

有限 on-policy 样本无法单独证明连续状态空间上的第 2 项：总能构造一个与所有已观测 transition 相同、但在未访问状态跳出安全集的黑箱动力学。可接受的严格闭环只有以下几类：

- 已知动力学/网络的 interval、SMT 或 reachability verification；
- 紧状态域上的有限覆盖，加已知且有效的 dynamics、policy、value Lipschitz 界，把网格 margin 推广到全域；
- 有限状态系统的穷举；
- 运行时 backup/shield。最后一种可以保证执行安全，但保证来自 backup，不是 learned value 本身。

随机策略还要更谨慎：期望 residual 非负不保证单条轨迹安全。若部署随机策略，(A2) 必须对其 support 中几乎所有动作成立；对常见的 full-support 连续分布通常过强。推荐训练时 stochastic、认证和部署时 deterministic mode。

## 6. 连续时间与多智能体边界

原 Deep-QP 假设连续时间 control-affine dynamics，并用导数 QP 强制
$\dot V+\alpha(V)\ge0$（原文第 52--60、242--252 行）。value-only finite difference 改写后，证明对象变成采样系统 $x_{k+1}=F_{\Delta t}(x_k,u_k)$。

若只读取端点 cost，最多证明 $c(x_k)\ge0$。要升级到
$c(x(t))\ge0$ 对所有 $t$，需要二选一：

- transition 同时返回 $\min_{\tau\in[0,\Delta t]}c(x(\tau))$，对应 Deep-QP 式 (1)/(2) 中的区间最小值；
- 已知 $|\dot c|\le L_c$ 等界，并把安全集收紧至少 $L_c\Delta t$。

当前 collector 只在 `env.step` 前后读取 `env.get_cost`（`train_safety_filter.py:128-139`），所以现实现不能声称 inter-sample 连续安全。

多智能体下还需明确：由局部 $h_i(G_i)$ 定义全局集合
$\mathcal C=\cap_i\{h_i\ge0\}$；只有所有 local next-step 条件同步成立，才可归纳得到全局不变。agent 数量迁移、hard sensing 邻域变化和 GNN 泛化都不是自动的证明。

## 7. 建议的取舍

建议先实现方案 A 作为最小可行消融：单 value head、deterministic on-policy min-return、finite-difference safety advantage，完全删除 pair/scalar derivative heads、support function、导数 loss、独立 replay 和 λ scheduler。它最容易检验“导数网络是否真的有贡献”，也能复用现有 PPO 数据流。

若研究目标必须包含“HJ 最大可达集”，再做方案 B；但应把结论写成**在分布式策略类内的条件性 viability 保证**，并增加 uniform verifier。否则最稳妥的论文表述是：

> exact value/policy pair 满足前向不变性定理；learned pair 通过经验 residual 验证提高安全性，但没有有限样本硬保证。

还需注意，方案 A 与 DGPPO 的 policy constraint-value/DGCBF 理论高度接近；单独把 Deep-QP 导数头删除并不构成强创新。真正可区分的部分应来自方案 B 的 reachability policy iteration、对局部可实现策略类的定义，以及 approximation-to-invariance 的可验证误差闭环。

## 本地证据索引

- Deep-QP 原文：`.papers/deep-qp/Kim和Kr - Deep QP Safety Filter Model-free Learning for Reachability-based Safety Filter/auto/Kim和Kr - Deep QP Safety Filter Model-free Learning for Reachability-based Safety Filter.md`，重点为 §3.1--3.4（第 45--275 行）。
- DGPPO 原文：`.papers/dgppo/Zhang 等 - 2025 - DISCRETE GCBF PROXIMAL POLICY OPTIMIZATION FOR MULTI-AGENT SAFE OPTIMAL CONTROL/auto/Zhang 等 - 2025 - DISCRETE GCBF PROXIMAL POLICY OPTIMIZATION FOR MULTI-AGENT SAFE OPTIMAL CONTROL.md`，重点为 §3.2、§4.1--4.5（第 72--265 行）及全局不变性证明（第 740--798 行）。
- 当前 Graph-HJ critic：`dgppo/algo/module/deep_qp_safety.py`。
- 当前 Deep-QP/PPO 接口：`dgppo/algo/informarl_deep_qp.py`。
- 当前独立预训练器：`train_safety_filter.py`。
