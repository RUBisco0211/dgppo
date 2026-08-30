# GCBF+ 理论能否迁移到 Multi-Agent Deep-QP

日期：2026-08-31

## 结论

可以迁移，而且 GCBF+ 最有价值的部分正好对应当前 Multi-Agent Deep-QP 的两个主要缺口：

1. 它给出了“共享的局部图标量函数如何组合成全局多智能体安全集合”的定义和前向不变性定理；
2. 它明确指出局部图函数的时间导数必须包含 ego 与所有邻居智能体的控制输入，而不能只包含 ego action。

因此，推荐把两篇工作的职责拆开：

- Deep-QP 负责从黑箱 transition 中学习局部 HJ safety value 及其动作仿射方向导数；
- GCBF+ 负责规定局部图函数、动态图边界条件、局部集合到全局集合的组合方式，以及在联合约束成立时的前向不变性结论。

但是，GCBF+ 不能直接证明当前实现安全。它的定理假设局部图函数确实是连续可微 GCBF、联合动作确实满足所有局部导数约束，并且证书集合包含于物理安全集合。神经网络训练误差、Deep-QP Bellman 逼近误差、局部观测的 Markov 性和独立逐智能体 QP 的动作冲突都不在其证明范围内。

## 1. GCBF+ 的关键理论

GCBF+ 考虑每个智能体具有控制仿射动力学：

$$
\dot{x}_i=f(x_i)+g(x_i)u_i.
$$

对智能体 $i$ 的局部邻域 $\mathcal N_i$，共享的局部图障碍函数写为：

$$
h_i=h(\bar x_{\mathcal N_i}).
$$

它的完整时间导数是：

$$
\dot h_i=
\sum_{j\in\mathcal N_i}
\frac{\partial h(\bar x_{\mathcal N_i})}{\partial x_j}
\left(f(x_j)+g(x_j)u_j\right).
$$

因此只要 $h_i$ 依赖邻居状态，其导数一般就必然依赖邻居动作。GCBF+ 在 Definition 1 和式 (7) 中明确写出了这一点；训练部分也明确说明，局部导数 loss 会向 ego controller 和所有邻居 controller 反向传播。[GCBF+ Definition 1 与局部联合导数](https://arxiv.org/html/2401.14554#S3.E7) [GCBF+ derivative loss 的邻居动作说明](https://arxiv.org/html/2401.14554#S4.SS2)

### 1.1 动态邻域的边界条件

GCBF+ 要求邻居到达感知边界 $R$ 时对局部证书没有影响：

$$
\frac{\partial h}{\partial x_j}=0,
\qquad
j\in\mathcal N_i\setminus\mathcal N_i^{<R},
$$

以及：

$$
h(\bar x_{\mathcal N_i})
=h(\bar x_{\mathcal N_i^{<R}}).
$$

这两个条件使邻居进入或离开局部图时，$t\mapsto h(\bar x_{\mathcal N_i(t)}(t))$ 仍然连续可微，从而可以在动态图上使用连续时间 CBF 证明。[GCBF+ 边界条件](https://arxiv.org/html/2401.14554#S3.E8) [邻域变化 Lemma](https://arxiv.org/html/2401.14554#ThmLemma1)

这部分可以直接用于修复当前硬半径建图、top-k 切换带来的 critic 跳变问题。但不应只依赖普通 attention “自己学会在边界变成零”；工程实现应显式使用满足以下条件的紧支撑平滑 gate：

$$
w(R)=0,
\qquad
w'(R)=0.
$$

### 1.2 从局部证书到全局安全

对每个智能体定义局部证书集合：

$$
\mathcal C_{N,i}
=\left\{\bar x: h(\bar x_{\mathcal N_i})\ge 0\right\},
$$

全局证书集合是局部集合的交：

$$
\mathcal C_N=\bigcap_{i=1}^{N}\mathcal C_{N,i}.
$$

如果：

$$
\mathcal C_N\subseteq\mathcal S_N,
$$

并且所有局部 GCBF 条件同时成立：

$$
\dot h_i+\alpha(h_i)\ge 0,
\qquad \forall i,
$$

那么 GCBF+ Theorem 1 证明 $\mathcal C_N$ 前向不变，并且这个组合结构可用于任意规模 $N$。[GCBF+ 全局集合与 Theorem 1](https://arxiv.org/html/2401.14554#ThmTheorem1)

这说明 Multi-Agent Deep-QP 不必训练一个集中式全局标量。可以训练一个参数共享的逐智能体局部图 value，再通过所有局部 superlevel set 的交定义全局证书。

## 2. 与当前 Deep-QP 的直接拼接

### 2.1 局部 HJ safety value

令 $z_i$ 表示智能体 $i$ 的局部安全图，只包含与安全有关的 agent、obstacle、相对位置和相对速度。定义正数表示安全的连续局部物理约束：

$$
c_i(z_i)=\min\left\{
\min_{j\ne i}\left(\lVert p_i-p_j\rVert-2r\right),
\min_o\left(d(p_i,o)-r\right)
\right\}.
$$

然后学习局部折扣 HJ value：

$$
V_i^\lambda(z_i).
$$

Deep-QP 有：

$$
V_i^\lambda(z_i)\le c_i(z_i).
$$

因此：

$$
V_i^\lambda(z_i)\ge 0
\Longrightarrow
c_i(z_i)\ge 0.
$$

这为 GCBF+ 所要求的 $\mathcal C_{N,i}\subseteq\mathcal S_{N,i}$ 提供了自然连接。再取所有局部集合的交，就能得到全局集合包含关系。

### 2.2 必须改成局部 joint-action derivative

当前 own-action-only 参数化：

$$
\widehat{\partial V_i}(z_i,u_i)=a_i(z_i)u_i+b_i(z_i)
$$

遗漏了邻居状态随 $u_j$ 变化对 $V_i$ 的贡献。应改成 edge-wise、置换等变的局部联合动作参数化：

$$
\widehat{\partial V_i}(z_i,\bar u_{\mathcal N_i})
=\sum_{j\in\mathcal N_i}a_{ij}(z_i)u_j
-\max_{\bar v_{\mathcal N_i}\in\mathcal U^{|\mathcal N_i|}}
\sum_{j\in\mathcal N_i}a_{ij}(z_i)v_j
+b_i(z_i).
$$

当动作集合是逐智能体 Cartesian product 时：

$$
\max_{\bar v_{\mathcal N_i}}
\sum_j a_{ij}v_j
=\sum_j\max_{v_j\in\mathcal U_j}a_{ij}v_j,
$$

所以 Deep-QP 的解析 action maximization 仍可扩展到邻域 joint action，不需要枚举联合动作。Replay transition 必须保存实际执行的 $\bar u_{\mathcal N_i}$，HJ derivative loss 也必须以完整邻域动作作为条件。

这条路线比当前用一个 GNN embedding 后只输出 ego coefficient 更接近 Deep-QP 原文的“完整状态、完整控制输入”假设。[Deep-QP joint control 应视作完整控制向量的依据](https://arxiv.org/html/2601.21297#S3.SS4)

## 3. QP 会发生什么变化

把学到的 $V_i$ 当作局部图 CBF 后，理论上最直接的过滤器是：

$$
\min_{\bar u}\sum_i\lVert u_i-u_i^{\mathrm{nom}}\rVert^2
$$

subject to：

$$
\widehat{\partial V_i}(z_i,\bar u_{\mathcal N_i})
+\alpha(V_i(z_i))
\ge \varepsilon_i,
\qquad \forall i.
$$

$\varepsilon_i$ 是覆盖 derivative/value 估计误差的鲁棒 margin。由于每条约束只涉及局部邻域动作，这个 QP 是稀疏的，但不同智能体控制仍被约束耦合。

GCBF+ 自己也给出了同构的联合 CBF-QP，并明确指出：该 QP 不是独立的逐智能体分布式 QP，因为每个局部约束包含所有邻居动作。[GCBF+ 联合 QP 及其非分布式性质](https://arxiv.org/html/2401.14554#S3.E17)

所以 GCBF+ 理论能够解释并修复当前导数模型，却不能让耦合约束自动变成每个 agent 独立求解的一条 QP。

### 3.1 保持 CTDE 的可选方案

1. **集中式训练 QP teacher，分布式策略蒸馏。** 训练时求稀疏联合 QP，并把结果蒸馏到共享局部 filter policy。执行时只用局部图。GCBF+ 采用的就是相近思路；缺点是执行时不再是显式 QP，神经蒸馏误差需要 margin 或 verification。
2. **分布式迭代 QP。** 保留显式 QP，通过邻居通信、ADMM 或其他分布式稀疏 QP solver 达成动作一致。理论最干净，但不再是一次前向的独立 local QP。
3. **责任分配。** 把每条联合约束的安全责任分配给相关 agent，使各 agent 解局部 QP；需要证明责任预算之和确实推出原联合约束。GCBF+ 没有提供这部分理论。
4. **最坏邻居动作。** 将邻居动作视为 disturbance，构造 $\max_{u_i}\min_{u_{-i}}$ HJ value，再做 ego-only robust QP。它可实现独立执行，但会更保守，并且需要重新证明 max-min Deep-QP Bellman operator，而不是直接引用 GCBF+。

近期最稳妥的验证顺序是先实现第 1 种的联合 QP teacher/oracle，用它验证 joint-action critic 是否正确；在证书与 QP 数值行为成立后，再选择蒸馏还是分布式求解。

## 4. 不能从 GCBF+ 直接继承的内容

### 4.1 它没有证明神经网络训练一定得到 GCBF

GCBF+ Theorem 1 是条件定理：假设 $h$ 已经满足 Definition 1。论文最后也承认神经网络控制器难以形式验证。因此不能因为用了它的 GNN/CBF loss，就声称学到的函数自动具有前向不变性。[GCBF+ limitations](https://arxiv.org/html/2401.14554#S8)

### 4.2 它没有 Deep-QP 的 HJ/Bellman 收敛理论

GCBF+ 采用有限差分 CBF loss 和有限 rollout 的 control-invariant 标签。Deep-QP 的价值在于通过折扣 HJ Bellman operator 学 viability value，并对固定项下的算子给出 contraction。两者不能简单替换。

建议保留 Deep-QP HJ loss，只借用 GCBF+ 的图证书定义、联合导数结构和全局组合定理。GCBF+ 的有限 rollout 标签可以作为 replay curriculum 或评估指标，但不应替代 HJ target。

### 4.3 局部图未必是闭合的 Markov state

GCBF+ 的邻域切换 Lemma 解决的是 $h$ 在图节点加入/移除时的连续可微性，不是局部 HJ transition 的 Markov 性。局部图之外的智能体未来可能进入邻域，同一个 $z_i$ 仍可能产生不同的 $z_i'$。

建议使用两个半径：

$$
R_{\mathrm{active}}<R_{\mathrm{observe}},
$$

在较大半径内观察和存储潜在邻居，只在较小半径内让其对证书产生非零影响，并在 $R_{\mathrm{active}}$ 使用紧支撑平滑 gate。这样新进入 active set 的邻居已经提前出现在 critic 输入中，但这仍是近似闭合，需要用 action/state branching 实验检查条件方差。

### 4.4 GCBF+ 假设已知动力学，Deep-QP 才是黑箱导数学习

GCBF+ 的解析式使用 $f$、$g$ 计算联合导数。当前项目不能调用 VMAS 显式动力学，因此应由 Deep-QP 的 edge coefficient 和 scalar head 学习联合导数。只要估计误差可控，GCBF+ 的 invariance argument 不要求在线求导时必须知道模型；但有限误差保证需要额外建立。

可对真实导数 $d_i$ 与学习导数 $\hat d_i$ 假设或校准：

$$
d_i\ge \hat d_i-\varepsilon_i,
$$

并在 QP 中强制：

$$
\hat d_i+\alpha(V_i)\ge\varepsilon_i.
$$

这比把固定 heuristic margin 加在物理 cost 上更有理论针对性。

## 5. 对当前实现的具体判断

当前 `DeepQPSafetyNet` 的共享 GNN trunk 可以保留，但下列部分必须改：

1. own-action coefficient head 改为按 receiver-sender edge 输出 $a_{ij}$；
2. safety replay 保存 ego 局部图中所有 agent node 的 executed action 和有效 mask；
3. Deep-QP value/derivative target 中的 action maximum 改为邻域联合最大值；
4. graph builder 增加显式 $C^1$ compact-support gate，并区分 observe radius 与 active radius；
5. 首先实现集中式稀疏 joint QP 作为 teacher/oracle，不能继续把独立 ego QP 称为 GCBF+ 理论支持的安全过滤器；
6. 分别测量 local Markov residual、joint derivative residual、拓扑切换连续性、QP feasibility 和全局最小 certificate。

最终可以形成如下理论链条：

$$
\text{Deep-QP joint-action HJ learning}
\Longrightarrow
\text{local graph safety value and derivative}
\Longrightarrow
\text{all local CBF inequalities}
\Longrightarrow
\text{GCBF+ intersection theorem}
\Longrightarrow
\text{global forward invariance}.
$$

目前真正缺失的环节是前两个蕴含在神经逼近、局部观测和有限数据下的误差界，以及如何在分布式执行时一致地满足联合动作约束。

## 一手来源

- [GCBF+ 论文](https://arxiv.org/html/2401.14554)
- [GCBF+ Definition 1、动态图条件与全局安全定理](https://arxiv.org/html/2401.14554#S3.SS2)
- [GCBF+ 联合 CBF-QP](https://arxiv.org/html/2401.14554#S3.E17)
- [GCBF+ 神经 GCBF 与训练 loss](https://arxiv.org/html/2401.14554#S4)
- [GCBF+ 官方代码](https://github.com/MIT-REALM/gcbfplus)
- [Deep-QP Safety Filter 论文](https://arxiv.org/html/2601.21297)
- [Deep-QP model-free Bellman operators、重参数化与 QP](https://arxiv.org/html/2601.21297#S3.SS4)

