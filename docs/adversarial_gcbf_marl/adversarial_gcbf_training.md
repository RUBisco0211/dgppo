# 对抗性图 CBF 的同步训练方法

## 1. 学习对象

环境给出安全约束

$$
c_i(x)=\min_m c_{im}(x),
$$

其中 $c_i(x)<0$ 表示碰撞或其他约束被违反。

对每个 ego agent 学习共享参数的动作条件安全 Critic：

$$
Q_\phi(o_i,u_i,d_i),
\qquad d_i=u_{\mathcal N_i}.
$$

它同时输入 ego 动作 $u_i$ 和当前邻居动作 $d_i$。相应的安全值为

$$
V^*(o_i)=\max_{u_i}\min_{d_i}Q^*(o_i,u_i,d_i).
$$

这里的 $V^*$ 是理论中的精确 minimax constraint-value。神经网络只能得到近似值 $\widehat V$，除非额外验证，否则称为 **candidate adversarial DGCBF**。

任务回报仍使用单独的 reward critic。安全 Critic 不是折扣累计回报 Critic。

## 2. GNN 结构与动态图

输入只使用当前可感知邻域 $\mathcal N_i^k$，沿用 DGPPO/GCBF+ 的图设定，不为感知半径外的未观测 agent 增加隐藏状态或额外对手。消息可写成

$$
m_{ij}=g(r_{ij})\,\phi_e(x_i,x_j,u_i,u_j),
\qquad r_{ij}=\|p_i-p_j\|.
$$

$V$ 的消息不输入动作；$Q$ 的消息输入 $u_i,u_j$。两者必须使用同一个距离 gate。

设感知半径为 $R$，agent 单步最多移动 $\bar d$，安全距离为 $R_{\rm safe}$。采用

$$
R-2\bar d>R_{\rm safe},
$$

并令

$$
g(r)=0,\qquad r\ge R-2\bar d.
$$

这表示刚进入或即将离开邻域的 agent 暂时不影响 $V$ 和 $Q$。对于真正的进入/离开转移，在理论文档所列反事实转移假设下，安全值可约化到前后两步的共同邻域。离散时间继承论证要求 gate 严格为零，但不要求连续时间 GCBF 中的 $g'(R)=0$。

普通 softmax attention 只能学到“接近零”的权重。若要使用动态图定理，应在 attention 外显式乘上述 gate。[DGPPO 2025](<../../.papers/Zhang 等 - 2025 - DISCRETE GCBF PROXIMAL POLICY OPTIMIZATION FOR MULTI-AGENT SAFE OPTIMAL CONTROL.pdf>)，[GCBF+ 2025](<../../.papers/Zhang 等 - 2025 - GCBF+ A Neural Graph Control Barrier Function Framework for Distributed Safe Multiagent Control.pdf>)。

训练数据必须给每个一步转移附加 transition type：

1. **stable：**邻域集合和 gate 活动状态均不变；
2. **enter：**有节点进入 $\mathcal N_i$；
3. **leave：**有节点离开 $\mathcal N_i$；
4. **activate：**共同邻居从 $g=0$ 变为 $g\ne0$；
5. **deactivate：**共同邻居从 $g\ne0$ 变为 $g=0$。

动态图继承引理只把 enter/leave 约化到固定邻域转移；activate/deactivate 本身属于固定邻域前提，必须直接进入 Bellman/DCBF loss 和 held-out 验证。训练实现不能把五类样本合并后只报告一个平均 residual。

## 3. 安全 Critic 的训练

### 3.1 Minimax Bellman 更新

用 ego maximizer $\mu_\eta$ 和邻居 adversary $\beta_\psi$ 近似内外层优化：

$$
u_i^*=\mu_\eta(o_i),
\qquad
d_i^*=\beta_\psi(o_i,u_i^*).
$$

目标网络给出下一状态值

$$
\bar V(o_i')=
Q_{\bar\phi}\!\left(o_i',
\mu_{\bar\eta}(o_i'),
\beta_{\bar\psi}(o_i',\mu_{\bar\eta}(o_i'))\right).
$$

与理论一致的无折扣参照 target 为

$$
y_i^{(0)}=\min\{c_i(x),\bar V(o_i')\}.
$$

它用于 $\gamma_s=1$ 分支和无折扣 Bellman residual 诊断。默认实现训练所用的 $y_i^{\gamma_s}$ 与 $L_Q$ 在第 3.2 节定义。

随后交替最小化两个损失：

$$
L_{\mu}=-Q_\phi(o_i,\mu_\eta(o_i),\beta_\psi(o_i,\mu_\eta(o_i))),
$$

$$
L_{\beta}=Q_\phi(o_i,\mu_\eta(o_i),\beta_\psi(o_i,\mu_\eta(o_i))).
$$

动作都投影到环境允许的集合。连续动作可用 adversary 网络，也可用多起点投影梯度下降。Liu 的对抗 actor--critic 可作为实现参考；基本理论仍保留无折扣 $\min$ Bellman backup，实现折扣按第 3.2 节单独处理。[Liu 2026](<../../.papers/Liu - 2026 - Safe multi-agent reinforcement learning based on adversarial strategy and control barrier function f.pdf>)。

### 3.2 理论无折扣、实现引入安全折扣

需要把理论定义、任务折扣与安全训练折扣分开：

- reward critic/PPO 可以继续使用 DGPPO 的 $\gamma_r=0.99$ 和 GAE $\lambda=0.95$；
- 理论文档只定义无折扣 minimax constraint-value，不引入 $\gamma_s$；
- 实现中的 safety target 加入独立的 $\gamma_s$ 以改善数值传播。

DGPPO 的理论 constraint-value 是无限时域最坏约束值，本身不带奖励式折扣；但其实际 on-policy 代码在 GAE-like 安全 target 中使用 $\gamma=0.99$。按本文的安全为正符号，对应的 discounted surrogate 可写成

$$
y_i^{\gamma_s}
=\min\!\left\{
c_i(x),\,
(1-\gamma_s)c_i(x)+\gamma_s\bar V(o_i')
\right\},
\qquad 0<\gamma_s<1.
\tag{20}
$$

默认 critic loss 为

$$
L_Q=
\left(
Q_\phi(o_i,u_i,d_i)-y_i^{\gamma_s}
\right)^2.
$$

该算子比 $\gamma_s=1$ 更容易形成 contraction-like 的数值传播，但它不再与式 (6) 的原始轨迹最小值完全相同，零超水平集也不能未经证明就等同于 robust viability kernel。因此 $\gamma_s$ 只出现在实现与实验部分，不写入基本理论、定理或证明。

实现阶段的决定如下：

1. 默认从 $\gamma_s=0.99$ 开始，与 DGPPO 的实践设置对齐；$\gamma_s$ 与 reward discount 分别配置，禁止共用一个含义不清的参数；
2. 同时保留 $\gamma_s=1$ 的无折扣 target 作为理论对照，并在 held-out rollout 上始终计算真实的 undiscounted trajectory minimum；
3. 采用 DGPPO 式新鲜 deterministic on-policy rollout、target network、Polyak averaging、约束归一化和有限时域 curriculum 稳定训练；
4. 可额外使用多步 min-return：
   $$
   G_{i,t}^{(n)}
   =\min\{c_{i,t},c_{i,t+1},\ldots,c_{i,t+n-1},\bar V(o_{i,t+n})\};
   $$
5. 做 $\gamma_s\in\{0.95,0.99,1.0\}$ 消融，比较无折扣 Bellman residual、候选安全集大小和真实碰撞率。实现输出应称为 discounted safety surrogate/candidate DGCBF，而不是宣称折扣网络精确等于 $V^*$。

### 3.3 DGPPO 式 $V_h$ 监督

像 DGPPO 一样，安全值的主体数据使用当前网络产生的新鲜 deterministic on-policy rollout。ego 使用 $\mu_\eta$，邻居使用 $\beta_\psi$；随机动作、多起点动作搜索和边界缓冲样本只作为覆盖增强。对长度 $H$ 的轨迹构造

$$
G_{i,t}^{H}=\min_{0\le k\le H}c_i(x_{t+k}).
$$

用独立的 state-value head $V_\omega(o_i)$ 拟合该标签：

$$
L_H=(V_\omega(o_{i,t})-G_{i,t}^{H})^2.
$$

再令它与动作 Critic 的 saddle value 对齐：

$$
L_{\rm align}=
\left[V_\omega(o_i)-Q_\phi(o_i,u_i^*,d_i^*)\right]^2.
$$

$G^H$ 只是当前对抗策略下的无折扣有限时域标签。它能稳定训练，但单独使用不能证明无限时域安全；在 $\mu_\eta,\beta_\psi$ 尚未接近 saddle point 时，它与最优 $Q^*$ 存在策略偏差，与 $\gamma_s<1$ 的 critic 也存在目标差异。这正是当前接受的 DGPPO 式理论—实现 gap。

第一版先保留单个 $V_\omega$，令 $\lambda_H,\lambda_a$ 从较小值 warm up，Bellman 更新为主体，并监控 $L_H$ 与 $L_{\rm align}$ 的梯度夹角。若两者持续冲突，再拆成 discounted saddle head $V_{\omega,\gamma}$ 和 undiscounted rollout head $V_{\omega,0}$；不在尚无实验信号时提前增加双 head 复杂度。

### 3.4 DGCBF 结构损失

精确值满足 $V^*(o_i)\le c_i(x)$，因此加入

$$
L_{\rm dom}=[V_\omega(o_i)-c_i(x)]_+^2.
$$

在候选安全集 $V_\omega(o_i)\ge0$ 上，对 adversary、随机动作和多起点搜索得到的候选动作集合 $\widehat D_i(o_i)$ 取最大违约：

$$
L_{\rm cbf}=
\sum_{z\in\mathcal T}w_z\,
\mathbb E_{(o_i,z)\sim\mathcal B_z}
\left[
\mathbf 1_{\{V_\omega(o_i)\ge0\}}
\max_{d_i\in\widehat D_i(o_i)}
\left[(1-\kappa)V_\omega(o_i)-V_\omega(o_i'(d_i))\right]_+^2
\right],
\qquad 0<\kappa\le1.
$$

其中 $\mathcal T=\{\text{stable, enter, leave, activate, deactivate}\}$，$\mathcal B_z$ 是对应类型的分层 batch，$w_z$ 防止稀少的边界样本被 stable 样本淹没。总安全损失可取

$$
L_{\rm safe}=L_Q+\lambda_HL_H+\lambda_aL_{\rm align}
+\lambda_dL_{\rm dom}+\lambda_bL_{\rm cbf}.
$$

不要把所有 $c_i(x)\ge0$ 的状态都标成 $V_\omega\ge0$：有些状态当前未碰撞，但下一步已经无法避免碰撞。

“训练时保证动态图前提”在当前 candidate 语义下具体指：

- 每个 batch 都含五类转移；缺少某一类时主动从边界场景生成；
- 对 activate/deactivate 直接优化上述 DCBF loss；
- 对 enter/leave 同时检查 gate 输出严格为零、删除变化节点前后的 $V/Q/\mu$ 数值不变性；
- checkpoint 只有在五类 held-out worst residual 都低于预设阈值时才被接受。

这保证训练流程不会遗漏引理前提，但仍是经验验证，不把有限样本升级为全域数学证明。

## 4. 与 MARL 同步训练

首个实现分支选择 **方案 A：DGPPO 式 residual policy update** 作为主线，先验证对抗性 DGCBF 是否能稳定影响 PPO actor。这是当前算法选择，不表示已经证明有限样本下的任务策略继承精确 $V^*$ 的前向不变性。每轮中的闭环是：

1. 用任务策略 $\pi_\theta$ 采集 stochastic rollout，计算 reward critic/PPO 所需的 reward advantage，并构造用于 $Q_\phi$ 的 task-action counterfactual 转移；
2. round-robin 指定 ego，用当前 $\mu_\eta$ 与 $\beta_\psi$ 采集 adversarial deterministic on-policy rollout；
3. 从当前 rollout 生成五类动态图样本，并用少量边界缓存补足稀有类别；
4. 更新 $Q_\phi,V_\omega$，再用多步内层更新 $\beta_\psi$ 和 $\mu_\eta$；
5. 更新 target networks并运行分类型 held-out 检验。

任务策略与安全 critic 之间保留三个接口，其中 A 已是首个实现的主分支，B/C 作为后续消融：

### 4.1 方案 A：DGPPO 式 residual policy update

根据当前任务动作造成的 DCBF residual 决定 PPO 更新方向：没有违反时优化任务回报，违反时优先降低 residual。该方案与 DGPPO 最接近，但需要把原来的固定策略 residual 改成最坏邻居响应下的 residual，并处理 PPO stochastic action 与 deterministic safety rollout 的对应关系。

### 4.2 方案 B：safety-Q regularization

在 PPO loss 中加入

$$
L_\pi=L_{\rm PPO}
-\lambda_\pi\,\mathbb E\!\left[
Q_\phi(o_i,u_i,\beta_\psi(o_i,u_i))
\right].
$$

该方案实现最简单，但 $\lambda_\pi$ 会引入任务—安全权衡，而且 critic 误差会直接影响策略。实现时必须对 $Q_\phi$ stop-gradient，并优先只在 $V_\omega(o_i)$ 接近边界时启用。

### 4.3 方案 C：独立安全选择器或执行期 override

保留独立 $\mu_\eta$ 近似 Bellman 式中的 $\max$；$\pi_\theta$ 只学习任务。当 $V_\omega$ 低于阈值时切换到 $\mu_\eta$，或者在候选动作中选择最坏邻居响应下安全值最大的动作。该方案的理论职责最清楚，但可能造成动作切换和性能损失。

当前先用 A 跑通固定图基础验证。待安全 critic 达到可用的 checkpoint 后，再在同一 checkpoint 上比较 A/B/C 的任务回报、对抗安全率、DCBF residual、干预比例和训练稳定性。完成该消融前，不声称最终任务策略继承精确 $V^*$ 的前向不变性。

## 5. 整体算法流程

**输入：**多智能体环境、约束函数 $c_i(x)$、任务策略 $\pi_\theta$、安全策略 $\mu_\eta$、邻居对手 $\beta_\psi$、安全 Critic $Q_\phi,V_\omega$。

**每轮训练：**

1. **采集任务数据。** 所有 agent 使用 $\pi_\theta$ 采集 stochastic rollout，供 reward critic 与 PPO 使用。
2. **采集安全数据。** round-robin 选择 ego；ego 使用 $\mu_\eta$，当前可观测邻居使用 $\beta_\psi$，采集新鲜 deterministic on-policy rollout。
3. **分类动态图转移。** 把安全数据分为 stable、enter、leave、activate、deactivate，并从边界缓存补足稀有类型。
4. **构造安全标签。** 训练 target 使用实现折扣 $\gamma_s=0.99$；同时保存无折扣一步 target 和 trajectory minimum 作为理论一致性诊断。
5. **更新安全 Critic。** 最小化 $L_{\rm safe}$，使 $Q_\phi$ 近似动作条件 discounted safety surrogate，使 $V_\omega$ 对齐其 saddle value 和有限时域最坏裕度。
6. **更新安全博弈策略。** 每次 critic 更新后，先对 $\beta_\psi$ 做若干最小化内层步，再对 $\mu_\eta$ 做最大化步；动作始终投影到物理范围。
7. **更新任务策略。** 对 stochastic PPO rollout 中的每个 ego 动作，保留该 ego 动作，令当前可观测邻居使用 $\beta_\psi$ 最坏响应，由 $Q_\phi$ 计算 robust DCBF residual。满足 residual 的样本使用 reward advantage；违反样本改用降低 residual 的 advantage，再进入标准 clipped PPO loss。计算该 advantage 时对 $Q_\phi,\mu_\eta,\beta_\psi$ stop-gradient。
8. **更新与验收。** Polyak 更新 target networks；只有五类 held-out residual、gate invariance 和多起点 adversary 检验均通过阈值时才接受 checkpoint。

**当前阶段输出：**任务策略 $\pi_\theta$、安全选择器候选 $\mu_\eta$、对手 $\beta_\psi$、discounted candidate adversarial DGCBF $V_\omega$ 和动作 Critic $Q_\phi$，以及对应的无折扣验证指标。

训练阶段的关系可概括为：

$$
\text{任务采样与确定性对抗采样}
\longrightarrow
\text{更新 }Q_\phi,V_\omega
\longrightarrow
\text{更新 }\beta_\psi,\mu_\eta
\longrightarrow
\text{robust residual advantage + clipped PPO}.
$$

当前训练后可直接评估 $\pi_\theta$，但执行期是否再用 $\mu_\eta$ override 仍是后续消融项。在接口消融完成前，不能把“部署只运行 $\pi_\theta$”写成已继承精确安全保证。

### 5.1 当前代码实现范围

- 算法名为 `adversarial_dgppo`（别名 `adv_dgppo`）；
- $Q_\phi,V_\omega,\mu_\eta,\beta_\psi$ 都是 agent 间共享参数的前馈 GNN，ego 身份和当前邻居 mask 作为条件；
- PPO/$V_l$ 使用原有 stochastic task rollout，安全值主体使用独立 deterministic adversarial rollout，并用 task-action counterfactual 转移增强 $Q_\phi$ 的动作覆盖；
- 安全折扣、target Polyak 和内层步数分别由 `--safety-gamma`、`--adv-target-tau` 和 `--adv-inner-steps` 配置；
- 所有本方法特有日志统一使用 `adv_dgcbf/` 前缀，不复用 DGPPO 的 `Vh/` 分类；
- 当前是固定图/核心 actor-update 验证版。compact-support gate、五类动态转移分层采样与 checkpoint 门限属于阶段 II，尚未实现。

## 6. 训练数据与检验

仅用普通协作 rollout 不能识别未尝试过的邻居动作。训练时至少需要：

- 当前 $\mu_\eta,\beta_\psi$ 的 deterministic on-policy rollout；
- 随机邻居动作、多起点投影搜索及必要的 adversary ensemble；
- $c_i\approx0$、$V_\omega\approx0$ 和五类动态图转移；
- 在计算预算内尽可能覆盖环境允许的邻居动作集合 $D_i$；
- round-robin ego 角色覆盖。

训练后至少报告：

1. $\gamma_s=0.99$ 训练 target residual 与 $\gamma_s=1$ 无折扣 Bellman residual；
2. stable、enter、leave、activate、deactivate 五类各自的最坏 DCBF residual 和违反率；
3. 多起点 adversary 搜索后的最坏值，以及单 adversary 与多起点搜索之间的 gap；
4. enter/leave 节点删除前后的 $V/Q/\mu$ invariance error；
5. $V_\omega\le c_i$ 的违反率；
6. 不同 agent 数量、密度、速度和可观测邻居数下的对抗安全率；
7. 对 $\{o_i:V_\omega(o_i)\ge0\}$ 的反例搜索结果；
8. $\gamma_s\in\{0.95,0.99,1.0\}$ 对候选安全集大小、真实碰撞率和训练稳定性的影响。

## 7. 必须预先确定的事项

当前已经确定：

1. 用固定物理控制上界定义邻居动作集合 $D_i$；理论保证只覆盖这个集合。
2. 从环境时间步长和速度上界计算 $\bar d$，检查 $R-2\bar d>R_{\rm safe}$；另记录 $R-4\bar d>R_{\rm safe}$ 是否成立，但不把后者当作 viability 定理。
3. 沿用 DGPPO/GCBF+ 的有限感知问题，不建模范围外 agent；同时确认它们不会直接改变 ego 与保留邻居的一步动力学。
4. 理论中不引入安全折扣；实现独立使用 $\gamma_s=0.99$，并强制保留 $\gamma_s=1$ 对照。
5. 只声称 candidate adversarial DGCBF，接受 DGPPO/GCBF+ 同类工作的理论—实现 gap。
6. 随机动力学和概率保证暂缓。

仍待实验确定：

1. 已实现的 DGPPO residual 分支是否作为最终主方案，safety-Q regularization 与执行期 override 作为对照；
2. 独立 $\mu_\eta$ 是否仅用于训练 saddle value，还是也用于执行期 override；
3. adversary 内层步数、restart 数量、ensemble 数量及 checkpoint 阈值。

## 8. 分阶段实现路线

### 阶段 I：固定图安全博弈

- 2–3 agents、确定性动力学、固定邻域；
- 实现 action-conditioned $Q_\phi$、$\mu_\eta$、$\beta_\psi$；
- 对比 $\gamma_s=0.99$ 与 $1.0$，首先确认 max/min 方向和安全值符号正确。

通过标准：多起点 adversary 找到的值与 $\beta_\psi$ 给出的值接近，且无折扣 trajectory minimum 与 critic 排序一致。

### 阶段 II：动态邻域

- 实现 compact-support gate 和五类 transition classifier；
- 使用定向场景生成 enter/leave/activate/deactivate；
- 分类型训练和验收，不使用总体平均掩盖边界失败。

通过标准：五类 held-out DCBF residual 分别达标，enter/leave invariance error 接近数值精度。

### 阶段 III：策略接口

- 冻结相同的安全 critic checkpoint；
- 分别实现 residual update、safety-Q regularization、override；
- 比较任务回报、安全率、干预比例、动作平滑度和训练敏感性。

通过标准：选出一个主方案，并据其实际闭环重新撰写“策略如何继承安全性质”的命题；其余方案保留为消融。

### 阶段 IV：规模与泛化

- round-robin ego；
- 扫描 agent 数、密度、速度与邻居数；
- 检查共享 GNN 的角色一致性和 adversary 强度。

### 阶段 V：随机性

只在前四阶段基本成立后，再考虑过程噪声、观测误差、执行误差以及概率安全指标。

## 参考原文

- Liu et al., 2026：minimax CBF action-value 与交替训练。[Liu 2026](<../../.papers/Liu - 2026 - Safe multi-agent reinforcement learning based on adversarial strategy and control barrier function f.pdf>)。
- Zhang et al., ICLR 2025：deterministic rollout 学习 constraint-value、PPO 安全梯度及离散换邻条件。[DGPPO 2025](<../../.papers/Zhang 等 - 2025 - DISCRETE GCBF PROXIMAL POLICY OPTIMIZATION FOR MULTI-AGENT SAFE OPTIMAL CONTROL.pdf>)。
- Zhang et al., TRO 2025：GNN/GCBF 与感知边界零影响结构。[GCBF+ 2025](<../../.papers/Zhang 等 - 2025 - GCBF+ A Neural Graph Control Barrier Function Framework for Distributed Safe Multiagent Control.pdf>)。
