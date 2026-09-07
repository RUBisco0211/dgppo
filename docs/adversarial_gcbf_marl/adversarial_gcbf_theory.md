# 对抗性图安全值函数：基本理论

## 结论

- 在确定性离散系统中，对 ego 动作取 **max**、对图邻域动作取 **min**，再沿时间取最小安全裕度，可得到鲁棒 viability value。其零超水平集在适当假设下是鲁棒前向不变集，且该值函数是一个离散时间 CBF（DCBF）。
- 它可以作为 MARL 中的 minimax safety critic，但不是折扣累计回报 critic。GNN 只是该 critic 的局部、置换不变参数化；有限样本训练得到的只能称为 *candidate adversarial GCBF*。
- 相对 DGPPO，核心变化不是再次引入“最坏时刻”，而是把固定联合策略下的 **temporal worst case** 扩展为邻居可偏离协作策略的 **strategic worst case**。
- 本文沿用 DGPPO/GCBF+ 的有限感知设定：安全博弈只对当前图中可观测的邻居建模，不额外对感知半径外的未观测 agent 建立 hidden-state adversary。因此，理论保证也只在这一感知模型内成立。

本文尽量沿用 DGPPO 的离散时间符号，但采用用户约定：约束值在安全时非负、碰撞/不安全时为负。若 DGPPO 的 avoid 函数记为 $h_i$（不安全为正），则本文取

$$
c_i=-h_i,\qquad V_i^c=-V_i^h.
$$

因此本文的安全集是零**超**水平集，而 DGPPO 的安全集是 avoid value 的零**次**水平集。[Liu 2026](<../../.papers/Liu - 2026 - Safe multi-agent reinforcement learning based on adversarial strategy and control barrier function f.pdf>)，[DGPPO 2025](<../../.papers/Zhang 等 - 2025 - DISCRETE GCBF PROXIMAL POLICY OPTIMIZATION FOR MULTI-AGENT SAFE OPTIMAL CONTROL.pdf>)，[GCBF+ 2025](<../../.papers/Zhang 等 - 2025 - GCBF+ A Neural Graph Control Barrier Function Framework for Distributed Safe Multiagent Control.pdf>)。

## 1. DGPPO 符号下的局部安全博弈

考虑 joint state、joint action 和确定性离散动力学

$$
x^k=(x_1^k,\ldots,x_N^k),\qquad
u^k=(u_1^k,\ldots,u_N^k),\qquad
x^{k+1}=f(x^k,u^k).
\tag{1}
$$

agent $i$ 的 ego-rooted 图观测为

$$
o_i^k=O_i(x^k)
=\bigl(x_i^k,\{e_{ij}^k,x_j^k:j\in\mathcal N_i^k\},\text{obstacles}\bigr).
\tag{2}
$$

其确定性策略写作 $u_i^k=\mu_i(o_i^k)$。把图邻域动作记为

$$
d_i^k:=u_{\mathcal N_i^k}^k\in
D_i(o_i^k):=\prod_{j\in\mathcal N_i^k}U_j(o_j^k).
\tag{3}
$$

环境给出的碰撞约束为 $c_{im}(x^k)\ge0$；负值表示违反约束。对 agent $i$ 合并多项约束：

$$
c_i(x):=\min_m c_{im}(x),\qquad
S_i:=\{x:c_i(x)\ge0\}.
\tag{4}
$$

若约束和一步演化可由局部图完整表示，可写成 $c_i(o_i)$ 及

$$
o_i^{k+1}=F_i(o_i^k,u_i^k,d_i^k)
=O_i\!\left(f(x^k,u^k)\right).
\tag{5}
$$

为使用局部 Bellman 方程，需要明确两点：

1. agent 动力学彼此解耦，碰撞约束只依赖当前可感知邻居；
2. 采用 $\exists u_i^k\,\forall d_i^k$：ego 选择动作后，对 $D_i$ 内所有允许邻居动作都要求安全。

### 1.1 本文采用的邻域建模范围

本文与 DGPPO/GCBF+ 一样，把当前感知图 $o_i^k$ 作为 agent $i$ 的局部安全状态，不显式建模当前位于感知半径 $R$ 之外的 agent。这是本文的问题定义，而不是对任意隐藏状态的额外鲁棒保证。

为使局部 Bellman 方程在这一建模范围内可用，预先采用以下假设：

1. agent 动力学解耦；一步内的局部约束只取决于 ego、当前已观测邻居以及当前已观测障碍物；
2. 感知半径与单步运动上界足以使新邻居在进入图时尚未违反物理安全约束；
3. 训练与验证直接覆盖邻域不变、进入、离开以及 gate 激活/去激活的一步转移。

若未观测 agent 能在进入感知范围前直接影响 ego 或保留邻居的一步动力学，上述问题定义不再适用。这种情形需要扩大感知范围或另建部分可观测模型，不在当前第一阶段理论中处理。

### 1.2 动态邻域继承的准确前提

采用 GCBF+ 的“感知边界零影响”思想及 DGPPO 的离散时间缓冲条件。设感知半径为 $R$，每个 agent 单步位移不超过 $\bar d$，碰撞距离为 $R_{\rm safe}$，要求

$$
R-2\bar d>R_{\rm safe}.
\tag{5d}
$$

在外层缓冲带 $[R-2\bar d,R]$ 内，强制节点对 GNN 输出没有影响：

$$
q_i=\sum_{j\in\mathcal N_i}g(\|p_i-p_j\|)\,\phi(e_{ij}),
\qquad g(r)=0\quad\forall r\ge R-2\bar d.
\tag{5e}
$$

除普通的邻域不变转移外，必须明确区分三类边界转移：

- **进入/离开：**对象进入或离开 $\mathcal N_i$；
- **激活：**$j\in\mathcal N_i^k\cap\mathcal N_i^{k+1}$，但 $g(r_{ij}^k)=0$、$g(r_{ij}^{k+1})\ne0$；
- **去激活：**$j\in\mathcal N_i^k\cap\mathcal N_i^{k+1}$，但 $g(r_{ij}^k)\ne0$、$g(r_{ij}^{k+1})=0$。

激活/去激活不改变邻域集合，因而它们应由“所有固定邻域转移已满足鲁棒 DCBF 条件”这一前提直接覆盖，而不是由进入/离开的零影响论证自动推出。

对真正的进入/离开转移，还需要 DGPPO 式反事实假设：可以删除变化节点，同时保持 ego、共同邻居及已观测非 agent 对象的一步运动不变。为此需要：

- 剩余 agent 的动力学不受被删除远端节点直接影响；
- gate 在缓冲带中严格为零，且使用 gate 后的 $V$、$Q$ 和安全选择器对这些节点不变；
- 所有邻域集合不变的转移，包括激活/去激活，已经满足鲁棒 DCBF 条件。

**动态图继承引理。** 若上述前提成立，且固定邻域转移满足

$$
\min_{d_i}\left[V_i(o_i^{k+1})-V_i(o_i^k)+\alpha(V_i(o_i^k))\right]\ge0,
$$

则在进入/离开时，变化节点在相应边界时刻的消息贡献为零，该转移可约化为共同邻域上的反事实固定邻域转移，因而同一不等式仍成立。

这是一个**条件性引理**。零 gate 不会自动证明激活转移安全；训练阶段必须对这类转移显式施加 $L_{\rm cbf}$，验证阶段必须单独报告其最坏 residual。$R-4\bar d>R_{\rm safe}$ 可作为额外的保守几何检查，但它只能排除“首次激活时已经碰撞”，不能单独证明后续存在可行避碰动作。

普通 softmax attention 只让权重接近零，不能支持严格继承论证。实现中应在 attention 外显式乘 compact-support gate。离散时间证明不要求 GCBF+ 连续时间设定中的时间导数连续性。

## 2. Minimax constraint-value 与 Bellman 方程

对初始局部观测 $o_i^0=o$，定义有限时域值

$$
V_{i,H}(o)
:=\sup_{\mu_i}\inf_{\beta_i}
\min_{0\le k\le H}c_i(o_i^k),
\tag{6}
$$

其中 $\mu_i$ 为 ego 的确定性反馈策略，$\beta_i$ 为可利用历史及当前 ego 动作的非预见邻居策略。这里有两层“最坏情况”：

- $\min_k$：轨迹上的最坏时刻，即 DGPPO 的 temporal worst case 取负后的形式；
- $\inf_{\beta_i}$：邻居控制的最坏响应，即 strategic worst case。

有限时域 Bellman 递推为

$$
V_{i,0}(o)=c_i(o),
$$

$$
V_{i,H+1}(o)=
\min\!\left\{
c_i(o),
\max_{u_i\in U_i(o)}\min_{d_i\in D_i(o)}
V_{i,H}\!\left(F_i(o,u_i,d_i)\right)
\right\}.
\tag{7}
$$

$V_{i,H}$ 随 $H$ 单调不增。定义无限时域值

$$
V_i(o):=\lim_{H\to\infty}V_{i,H}(o).
\tag{8}
$$

在动态规划及最优选择器等假设成立时，

$$
V_i(o)=
\min\!\left\{
c_i(o),
\max_{u_i}\min_{d_i}
V_i\!\left(F_i(o,u_i,d_i)\right)
\right\}.
\tag{9}
$$

定义 action-conditioned safety value

$$
Q_i^V(o,u_i,d_i)
:=\min\!\left\{c_i(o),V_i(F_i(o,u_i,d_i))\right\},
\tag{10}
$$

则

$$
V_i(o)=\max_{u_i}\min_{d_i}Q_i^V(o,u_i,d_i).
\tag{11}
$$

式 (10)–(11) 是适合 GNN critic 训练的形式，因为邻居动作依赖没有被错误地消去。

在第 1 节规定的有限感知模型及动态图前提下，式 (6)–(11) 用于当前可观测图。它不对尚未进入感知范围的 agent 作额外预测或安全承诺；新邻居进入后的转移由第 1.2 节的进入、激活训练与验证条件处理。

## 3. 鲁棒前向不变性与 DCBF 性质

令

$$
K_i:=\{o:V_i(o)\ge0\},
\tag{12}
$$

并令 $\mu_i^*(o)$ 取到式 (9) 中的最大值。

**定理 1（鲁棒 viability/DCBF）。** 在第 1 节假设下：

1. $K_i\subseteq\{o:c_i(o)\ge0\}$；
2. 对任意 $o\in K_i$ 和任意 $d_i\in D_i(o)$，
   $$
   V_i(F_i(o,\mu_i^*(o),d_i))\ge V_i(o)\ge0;
   \tag{13}
   $$
   因而 $K_i$ 对邻居允许动作鲁棒前向不变；
3. $K_i$ 是局部安全集内最大的鲁棒控制不变集，即相应的 robust viability kernel；
4. 对任意合适的 class-$\mathcal K$ 函数 $\alpha$，在 $K_i$ 上
   $$
   \min_{d_i}\bigl[
   V_i(F_i(o,\mu_i^*(o),d_i))-V_i(o)+\alpha(V_i(o))
   \bigr]\ge0,
   \tag{14}
   $$
   所以 $V_i$ 是 $K_i$ 上的鲁棒 DCBF。

**证明。** 由式 (9)，$V_i(o)\le c_i(o)$，故第 1 条成立。记

$$
G_i(o):=\max_{u_i}\min_{d_i}V_i(F_i(o,u_i,d_i)).
$$

式 (9) 给出 $V_i(o)\le G_i(o)$。取最大化动作后，对所有 $d_i$ 都有

$$
V_i(F_i(o,\mu_i^*(o),d_i))\ge G_i(o)\ge V_i(o),
$$

得到式 (13)。任何安全集内的鲁棒控制不变集合都存在 ego 策略，使任意邻居策略下整条轨迹满足 $c_i(o_i^k)\ge0$；由式 (6)–(8)，其状态必属于 $K_i$，故 $K_i$ 最大。式 (14) 则直接由式 (13) 及 $\alpha(V_i(o))\ge0$ 得到。证毕。

若定义 joint safe set

$$
K:=\bigcap_i O_i^{-1}(K_i),
\tag{15}
$$

且每个 $\mu_i^*$ 均对邻域内**任意**联合动作满足式 (13)，则所有策略同步执行时，实际邻居动作属于各自的全称量化范围，故 $K$ 前向不变。该结论不依赖在线分布式 CBF-QP，但通常较保守。

## 4. 与 DGPPO 最坏情况安全值相比，创新在哪里

DGPPO 对固定确定性联合策略 $\mu=(\mu_1,\ldots,\mu_N)$ 定义 avoid-positive constraint-value；按本文符号取负后，本质为

$$
V_i^{\mu}(o_i^0)=\min_{k\ge0}c_i(o_i^k),
\qquad u_j^k=\mu_j(o_j^k).
\tag{16}
$$

它回答的是：**所有 agent 都按当前联合策略执行时，轨迹最危险的一刻是否安全？** 本文式 (6) 回答的是：**ego 能否选择策略，使得即使图邻域 agent 在允许动作内作最坏偏离，所有时刻仍安全？**

| 比较项 | DGPPO constraint-value | 本文对抗性 constraint-value |
|---|---|---|
| 最坏情况 | 固定联合策略下对时间 $k$ 取最坏 | 对时间取最坏，并对邻居策略/动作取最坏 |
| 安全集 | policy-dependent invariant set | 局部鲁棒不变集；在局部 Markov 假设下为 robust viability kernel |
| Bellman critic | 以当前确定性联合策略 rollout 的 state-value 为主 | $Q_i^V(o_i,u_i,d_i)$ 与 $\max_{u_i}\min_{d_i}$ backup |
| 对其他 agent 的假设 | 固定、通常同质协作策略 | 允许邻居偏离、失配或产生对抗响应 |
| 证书含义 | 对给定 $\mu$ 的闭环安全性 | ego 对允许邻居动作集合的鲁棒可控安全性 |

因此，潜在创新应明确落在以下组合上：

1. 把 DGPPO 的局部动态图 constraint-value 从“固定协作联合策略”扩展为“每个 ego 对当前可观测邻居允许动作的 strategic worst case”；
2. 使用对变长邻域置换等变的 action-conditioned GNN critic 表示 $Q_i^V(o_i,u_i,d_i)$，并显式处理进入、离开、激活和去激活样本；
3. 使用与原始安全裕度一致的无折扣 minimax backup，避免把 Liu 的折扣 surrogate 误当成同一个 viability value；
4. 给出与 MARL 训练相接的候选证书验证协议。首个实现分支使用 DGPPO 式 robust residual PPO，但这一工程选择不在当前理论贡献中被预先声称为继承安全定理。

必须诚实限定：**max-min、action-conditioned CBF Q 和 viability value 都不是单独的新原理。** Liu 2026 已有全状态 target-team/free-team 的 action-conditioned max-min，DGPPO 已有动态局部图和最坏时刻 constraint-value。这里真正需要证明和验证的是二者结合后新增的 per-ego 局部鲁棒语义、动态图训练前提以及无折扣安全 backup；若缺少这些内容，就只是在拼接已有模块。

## 5. 与 Liu 一步 CBF 余量的关系

Liu 使用一步余量

$$
r_{i,\varepsilon}(o,u_i,d_i)
:=c_i(F_i(o,u_i,d_i))-(1-\varepsilon)c_i(o),
\qquad 0<\varepsilon<1.
\tag{17}
$$

若沿用该设计，正确的对抗 Bellman 方程应保留当前联合动作：

$$
W_i(o)=\max_{u_i}\min_{d_i}
\min\!\left\{
r_{i,\varepsilon}(o,u_i,d_i),
W_i(F_i(o,u_i,d_i))
\right\}.
\tag{18}
$$

令 $H_i(o)=\min\{c_i(o),W_i(o)\}$。在最优动作可取到等条件下，$H_i(o)\ge0$ 蕴含

$$
H_i(o_i^{k+1})\ge(1-\varepsilon)H_i(o_i^k),
\tag{19}
$$

故 $H_i$ 也是指数型 DCBF。但 $r_{i,\varepsilon}$ 依赖 $(u_i,d_i)$，通常不能移到 max-min 外。环境已提供 $c_i$ 时，式 (6)–(14) 更直接；式 (17)–(19) 可作为强调一步收缩率的备选。

## 6. 当前阶段的理论边界

$V_i$ 或 $Q_i^V$ 可以作为 minimax safety critic，但当前阶段采用以下边界：

- 只考虑确定性动力学和当前可观测邻域；随机动力学、观测噪声与概率保证留到基本算法验证后。
- 精确理论使用轨迹最小安全裕度，不使用奖励式折扣。若只用当前 $(\mu,\beta)$ 的 deterministic rollout，得到的是当前博弈策略下的有限时域标签，而不是自动得到式 (8) 的无限时域最优值。
- 沿用 DGPPO/GCBF+ 的理论—实现边界：神经网络与有限样本只称为 candidate certificate，不额外承诺全域形式化验证。
- adversary 覆盖到的动作决定经验鲁棒范围；理论中的“对所有 $d_i\in D_i$”仍是条件性结论。
- $V_i^*$ 只证明安全动作存在。首个实现用 robust residual 切换 PPO advantage，但近似 critic 和有限样本下仍不声称最终任务策略继承式 (13)。

| 必须项 | 当前处理 |
|---|---|
| 未观测邻居 | 沿用 DGPPO/GCBF+ 的有限感知问题定义，不纳入当前博弈；保证范围明确限制在当前感知模型内。 |
| 动态图前提 | 显式 hard/compact-support gate；训练与验证必须分别覆盖邻域不变、进入、离开、激活和去激活转移。 |
| 邻居动作范围 | 优先使用与局部观测无关的固定物理控制上界定义 $D_i$；若动作约束依赖状态，必须保证可由 $o_i$ 构造。 |
| max-min 顺序 | 当前采用 $\exists u_i\,\forall d_i$：ego 先选，邻居作最坏响应。保证只覆盖这一较保守的信息结构。 |
| 策略使用方式 | 首个实现选择 DGPPO 式 robust residual policy update；后续与 safety-Q regularization、执行期安全选择器比较。 |
| 网络误差 | 与 DGPPO/GCBF+ 一样只声称 candidate DGCBF，并报告 held-out residual、最坏动作搜索和对抗安全率。 |
| 随机性 | 暂缓，待确定性版本在小规模环境中闭合后再扩展。 |

## 7. “适当假设”具体是什么意思

式 (9) 到定理 1 的推导很短，但其中隐含了若干数学条件。它们不是额外算法模块，而是为了说明什么时候可以把上确界/下确界写成最大值/最小值，以及什么时候真的存在可执行的安全策略。

### 7.1 动作集合紧致与极值可达

若 $U_i$、$D_i$ 是闭且有界的集合，并且 $Q_i^V$ 对动作连续，则最大值和最小值能够由某个具体动作取到。否则只能写 $\sup/\inf$；可能存在一串越来越好的动作，却没有任何一个动作真正达到最优值，此时 $\mu_i^*(o)$ 未必存在。

实现上最简单的选择是使用环境给定的 box action bounds：

$$
U_i=[u_i^{\min},u_i^{\max}],\qquad
D_i=\prod_{j\in\mathcal N_i}[u_j^{\min},u_j^{\max}].
\tag{20}
$$

### 7.2 有限时域策略类

式 (7) 的有限时域动态规划允许策略随剩余时域变化，即第 $k$ 步可使用 $\mu_{i,k}$、$\beta_{i,k}$。如果一开始就把 $\mu_i$ 限死为单个 stationary network，有限时域 Bellman 递推未必等于对该受限策略类直接优化的结果。

当前理论可采用标准处理：先在允许时变 Markov 策略的有限时域博弈上定义 $V_{i,H}$，再在无限时域固定点存在时选择 stationary maximizer $\mu_i^*$。训练中的共享网络只是对该 selector 的近似。

### 7.3 无限时域极限与安全策略存在

$V_{i,H}$ 随 $H$ 单调不增；还需约束值有下界，才能保证极限不是 $-\infty$。要从“每个有限 $H$ 都有安全动作”推出“一条策略对所有时间都安全”，通常还需要状态/动作紧致性、转移连续性以及安全集闭合等条件。

本文不展开一般拓扑空间证明，而在实验环境中预先检查：

- 状态和动作均有界；
- $c_i$ 连续且有界；
- 离散动力学连续；
- 每个候选安全状态至少存在一个允许动作；
- 训练时发现无安全动作的状态不标入候选安全集。

### 7.4 对手的信息与动作矩形性

$\max_{u_i}\min_{d_i}$ 表示邻居在知道 ego 当前动作后选择最坏联合动作。这比同时行动更保守。还默认

$$
D_i=\prod_{j\in\mathcal N_i}U_j
$$

是 rectangular 的，即每个邻居动作可以独立组合。若邻居之间存在联合动力学、共享资源或耦合动作约束，就必须把这些约束直接写入 $D_i$，不能继续使用简单笛卡尔积。

### 7.5 局部定理与共享网络

定理 1 对每个 ego 给出一个可能不同的 selector $\mu_i^*$。实现采用共享参数 $\mu_\eta$ 时，还隐含假设同一个置换等变网络有足够容量同时近似所有 ego 的 selector。参数共享有利于泛化，但不是由定理自动保证的；需要按 agent 角色轮换训练并分别报告 residual。

### 7.6 精确对象与 learned candidate

定理中的 $V_i$、内层最小化和 DCBF 不等式都是精确对象。实际网络、有限 adversary steps 和有限样本都只是近似。当前工作沿用 DGPPO/GCBF+ 的处理：理论证明精确对象，算法学习 candidate，并用实验检验接近程度；不把经验 residual 写成形式化全域保证。

## 8. 下一步如何细化

按以下顺序推进，可以让每一步都对应一个可验证的问题，而不是同时改动所有模块。

1. **固定图最小闭环。** 先在 2–3 agent、固定邻域、确定性动力学中，以实现默认的折扣 target 训练，同时计算无折扣 minimax Bellman target 与真实 trajectory minimum 作为理论诊断，确认 $Q(o,u,d)$ 的动作排序、$\mu$ 最大化和 $\beta$ 最小化方向全部正确。
2. **动态图前提闭环。** 加入 compact-support gate，并按邻域不变、进入、离开、激活、去激活五类统计 DCBF residual；只有五类都通过阈值后才进入下一阶段。
3. **对手强度闭环。** 比较单 adversary、随机动作、多起点投影梯度和 adversary ensemble，确认弱对手不会系统性高估安全值。
4. **策略接口消融。** 以已实现的 DGPPO 式 robust residual/CRPO 更新为首个主分支，在同一 critic checkpoint 上与 safety-Q 正则、$V$ 阈值触发的 $\mu$ override 比较。先用安全率与任务回报选最终方案，再补相应理论陈述。
5. **规模与角色泛化。** round-robin 轮换 ego，改变 agent 数量、密度、速度和可观测邻居数量，检查共享 GNN 是否仍满足各 ego 的 residual。
6. **最后再扩展随机性。** 确定性版本基本成立后，再把过程噪声、观测误差和执行误差加入扰动集合或概率模型，避免当前阶段混淆“邻居动作鲁棒”和“随机动力学鲁棒”。

现阶段可采用的理论贡献表述是：

> 我们将 DGPPO 在固定协作联合策略下的局部图 constraint-value 扩展为确定性局部零和安全博弈，将当前可观测邻居的有界动作视为最坏响应。对于精确 minimax constraint-value，在所列动作、策略选择与动态图前提下，其零超水平集是鲁棒前向不变的，并构成鲁棒 DGCBF。

算法贡献暂写为：

> 我们使用置换等变的 action-conditioned GNN critic 与确定性对抗 rollout 近似该值，并要求动态图版本显式训练和验证五类转移。该网络在未完成全域验证时称为 candidate adversarial DGCBF；首个实现以 robust residual 影响 PPO，最终接口仍由后续消融决定。
