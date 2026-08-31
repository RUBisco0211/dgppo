# Multi-Agent Graph-HJ Gap：HJB-GNN、GCBF+ 理论与修复方案

更新时间：2026-08-31

本文合并 GNN 聚合下的 Deep-QP gap、HJB-GNN 的结构启发和 GCBF+ 的局部到全局证书理论。它是理论背景与备选修复路线，不是当前 `informarl_hj_crpo` 实现规格；当前方案以 [统一设计文档](informarl_hj_crpo_design.md) 为准。

## 1. 结论

可以继续使用 GNN 对每个智能体的整个局部邻域进行信息聚合，并输出 node-wise HJ Safety Value。问题从来不在“使用 GNN”，而在于值函数的状态依赖、方向导数的动作依赖和执行时可用信息必须一致。

旧 ego-only 原型遗漏邻居动作，造成 derivative target 混叠。当前实现已经保留整邻域标量，并用 local-joint-action pair head、完整 replay joint action 和独立 off-policy 训练修复了这一层结构 gap；RL 阶段冻结 critic，通过 soft HJ-CRPO 更新 actor，不执行 runtime QP。

仍需处理的主要问题是：

1. 用平滑感知边界处理邻居进出图的不连续；
2. 用历史或 recurrent GNN 缓解局部观测非 Markov；
3. 校准函数逼近与 derivative residual，避免把经验拟合当作形式证书；
4. 如果未来恢复独立 local QP，必须使用与训练算子一致的 robust neighbor elimination，不能把 joint constraint 直接拆成 ego-only constraint。

这不要求修改 VMAS/LidarEnv 的 reward，也不要求写出解析动力学方程。需要的仍然只是已有的图转移、实际执行的全体动作、安全约束值和下一图。

但必须明确：该修改修复的是当前最直接的“状态依赖与动作依赖不一致”。它不会自动带来完整的前向不变性证明；局部观测充分性、动作仿射误差、函数逼近误差、图拓扑切换和 QP 可行性仍需分别验证。

## 2. 旧 ego-only 原型的 gap 与当前修复状态

旧 `DeepQPSafetyNet` 原型先用 `GraphTransformerGNN` 聚合邻居信息，然后由同一个 ego embedding 输出两个 Safety Value、一个 ego 动作系数和一个标量项。可以把旧结构概括为：

$$
V_i = V_i(G_i),
$$

$$
q_i(G_i,u_i)
=a_{ii}(G_i)^\top u_i
-\sigma_{\mathcal U_i}\!\left(a_{ii}(G_i)\right)
+d_i(G_i),
$$

其中支持函数为：

$$
\sigma_{\mathcal U}(a)=\max_{u\in\mathcal U}a^\top u.
$$

在旧代码中，`coefficient` 的形状只覆盖每个智能体自己的动作维度；loss 中也只计算同索引的 `coefficient * batch.actions`。然而，GNN 输出的 `V_i` 和 `a_ii` 已经依赖邻居状态。实际局部图转移应当写成：

$$
G_{i,t+1}=F_i\!\left(G_{i,t},u_{i,t},u_{\mathcal N_i^a,t},w_t\right),
$$

其中 `w_t` 表示未观测环境运动、接触和传感噪声等因素。因此一般有：

$$
\frac{\mathrm d}{\mathrm dt}V_i(G_i)
=b_i(G_i)
+a_{ii}(G_i)^\top u_i
+\sum_{j\in\mathcal N_i^a}a_{ij}(G_i)^\top u_j.
$$

若训练输入只给出 `G_i` 和 `u_i`，则同一个声明的输入会对应多个方向导数目标：

$$
\operatorname{Var}\!\left[
\dot V_i\mid G_i,u_i
\right] > 0.
$$

此时标量项 `d_i(G_i)` 会混合邻居策略的条件均值、动作噪声和真正的 drift。它可能在固定训练策略下得到较小的经验误差，但不能再解释为只关于 `G_i` 和 `u_i` 的确定性控制仿射方向导数；更换 RL policy 或进入密集交互区后，这个条件分布还会漂移。

当前 `GraphHJSafetyCritic` 已完成第一层结构修复：使用共享 pair head 输出置换等变的局部 joint-action coefficient，并在 HJ loss 中使用完整 replay joint action。当前 RL 阶段没有恢复独立 local QP，而是冻结该 critic，通过 DGPPO 风格的 soft HJ-CRPO advantage 更新策略。因此，本节描述的是修复动机；仍未解决的是局部 Markov 性、动态图边界、函数逼近误差、联合可行性和形式证书。

## 3. HJB-GNN 能提供什么证据

HJB-GNN 的 graph CBF 是整个局部增强邻域上的一个标量，而不是每条边各自一个 HJ Value。它把 graph CBF 的时间导数明确写成所有可控邻居贡献之和：

$$
\dot h\!\left(\bar x_{\hat{\mathcal N}_i}\right)
=\sum_{\ell\in\hat{\mathcal N}_i^a}
\nabla_{x_\ell}h\!\left(\bar x_{\hat{\mathcal N}_i}\right)
\left(f(x_\ell)+g(x_\ell)u_\ell\right).
$$

这个式子直接支持本文对 gap 的判断：如果图标量依赖多个可控智能体的状态，它的导数就应包含这些智能体的动作。HJB-GNN 随后的拉格朗日乘子中也显式含有邻居控制贡献；论文自己指出这种 neighbor-control coupling 会阻碍直接的分布式执行。

HJB-GNN 还给出了一个对动态图很有价值的条件：对感知范围边缘的补充节点，graph CBF 的值应保持不变，对这些节点状态的梯度应当为零。这说明仅仅 padding 或 hard radius cut-off 不足以保证连续性，邻居贡献应在离开有效邻域前衰减到零。

不过，HJB-GNN 不能直接移植到当前工程：

- 它假设已知连续时间控制仿射动力学。
- 它使用 goal error 上的任务 Value、二次任务代价和受约束 HJB/KKT 结构，联合训练任务控制器、graph CBF 和任务 Value。
- 它通过解析动力学和自动微分计算 CBF 导数，再让分布式 GNN policy 模仿耦合的解析控制器。
- 它的定理假设存在精确有效的 graph CBF 和最优任务 Value；神经网络 loss 是促使这些条件近似成立，而不是对 SGD 得到全局有效证书的证明。

因此，值得借鉴的是“整邻域标量 + 全部邻居动作贡献”“图边界零贡献”和“用局部 GNN 蒸馏耦合 teacher”这三种组织方式，而不是它依赖已知动力学、任务 cost 和 KKT 的具体 loss。

## 4. GCBF+ 可迁移的组合理论

GCBF+ 的价值不是提供另一个 Deep-QP loss，而是规定局部图证书如何形成全局安全集合。对共享的局部图函数：

$$
h_i=h(\bar x_{\mathcal N_i}),
$$

完整时间导数为：

$$
\dot h_i=
\sum_{j\in\mathcal N_i}
\frac{\partial h(\bar x_{\mathcal N_i})}{\partial x_j}
\left(f(x_j)+g(x_j)u_j\right).
$$

这再次说明：只要局部标量依赖邻居状态，其导数一般就必须包含邻居动作。Deep-QP 可以负责从黑箱 transition 学习这个 joint-action derivative；GCBF+ 则提供以下三项结构条件。

### 4.1 动态邻域边界条件

邻居到达感知边界时应对证书值和梯度没有贡献：

$$
\frac{\partial h}{\partial x_j}=0,
\qquad
h(\bar x_{\mathcal N_i})=h(\bar x_{\mathcal N_i^{<R}}).
$$

工程上不能假设普通 attention 会自动满足该条件。紧支撑 gate 至少应满足：

$$
w(R)=0,
\qquad
w'(R)=0.
$$

这与第 7 节的 smooth graph boundary 设计一致，并给出了其理论来源。

### 4.2 从局部证书到全局安全

定义每个智能体的局部证书集合和它们的交：

$$
\mathcal C_{N,i}
=
\left\{\bar x:h(\bar x_{\mathcal N_i})\ge 0\right\},
$$

$$
\mathcal C_N=\bigcap_{i=1}^{N}\mathcal C_{N,i}.
$$

如果：

$$
\mathcal C_N\subseteq\mathcal S_N,
$$

且所有局部约束同时成立：

$$
\dot h_i+\alpha(h_i)\ge 0,
\qquad \forall i,
$$

则 GCBF+ 的条件定理给出全局集合的前向不变性。Multi-Agent Deep-QP 因而不必学习一个集中式全局标量；参数共享的 node-wise Graph-HJ value 可以通过局部 superlevel set 的交形成全局证书候选。

对正数表示安全的局部物理 margin $c_i$，若 Deep-QP 学到：

$$
V_i^\lambda(z_i)\le c_i(z_i),
$$

则：

$$
V_i^\lambda(z_i)\ge 0
\Longrightarrow
c_i(z_i)\ge 0,
$$

这为局部证书集合包含于局部物理安全集合提供了自然连接。

### 4.3 不能直接继承的保证

GCBF+ 不会证明神经网络训练一定得到有效 GCBF，也没有 Deep-QP 的 Bellman/HJ contraction 结论。它还假设已知 control-affine dynamics，而本项目只允许从 transition 学导数。可以校准保守误差界：

$$
d_i\ge \hat d_i-\varepsilon_i,
$$

并要求：

$$
\hat d_i+\alpha(V_i)\ge\varepsilon_i,
$$

但 $\varepsilon_i$ 的全局有效性、局部图 Markov 性、动态图连续性、函数逼近误差和所有联合约束的同时可行性仍需额外证明或验证。

## 5. 修复后的 Safety Critic

### 5.1 整邻域标量保持不变

每个智能体仍输出一个依赖完整局部图的 Safety Value：

$$
V_i=V_\theta(G_i).
$$

GNN embedding 可以表示 ego 同时受到多个邻居夹逼、多个障碍共同封闭通道等高阶关系。这正是相比纯 pair-wise Value 应当保留的能力。

改变的只是方向导数头。对每个 ego 智能体，网络输出：

$$
\left\{
a_{ii},\{a_{ij}\}_{j\in\mathcal N_i^a},d_i
\right\}=D_\phi(G_i).
$$

建议用整图上下文生成每条可控边的系数：

$$
a_{ij}=\psi_\phi(z_i,z_j,e_{ji},z_{G_i}).
$$

这里的 `a_ij` 虽然按边存储，但它依赖整邻域上下文；它不是 pair-wise Safety Value，也不要求把全局安全集合写成两两安全集合的交集。因此，它不会重新引入 pair-wise minimum 固有的信息损失。

障碍节点没有可控动作，不输出动作系数；静态或动态障碍对值变化的影响留在标量项和时序表征中。

### 5.2 合作式联合动作参数化

若训练或 teacher QP 可以同时决定局部所有智能体动作，可以采用：

$$
q_i^{\mathrm{coop}}(G_i,u_{\mathcal A_i})
=d_i(G_i)
+\sum_{\ell\in\mathcal A_i}
\left[
a_{i\ell}(G_i)^\top u_\ell
-\sigma_{\mathcal U_\ell}\!\left(a_{i\ell}(G_i)\right)
\right],
$$

其中：

$$
\mathcal A_i=\{i\}\cup\mathcal N_i^a.
$$

该重参数化满足：

$$
\max_{u_{\mathcal A_i}}q_i^{\mathrm{coop}}(G_i,u_{\mathcal A_i})=d_i(G_i).
$$

它与原始 Deep-QP 的 action-affine reparameterization 最接近，适合作为诊断模型或集中式 teacher。但运行时若直接解局部联合 QP，需要邻域动作通信、迭代一致性或集中求解，不能称为完全独立的 decentralized QP。

### 5.3 面向独立局部 QP 的 robust 参数化

若执行时每个智能体只能决定自己的动作，就必须规定如何处理邻居动作。最干净、与策略分布变化无关的办法是把邻居视作有界扰动，学习 ego-max、neighbor-min 的 robust safety game：

$$
q_i^{\mathrm{rob}}(G_i,u_{\mathcal A_i})
=d_i(G_i)
+a_{ii}^\top u_i
-\sigma_{\mathcal U_i}(a_{ii})
+\sum_{j\in\mathcal N_i^a}
\left[
a_{ij}^\top u_j
+\sigma_{\mathcal U_j}(-a_{ij})
\right].
$$

于是：

$$
\max_{u_i\in\mathcal U_i}
\min_{u_{\mathcal N_i^a}\in\mathcal U_{\mathcal N_i^a}}
q_i^{\mathrm{rob}}(G_i,u_{\mathcal A_i})
=d_i(G_i).
$$

对 box action space，支持函数和最坏情况邻居动作都能解析计算，不需要动力学方程，也不需要在运行时枚举邻居动作。

经过最坏情况消元，运行时的安全约束仍只关于 ego 动作。沿用当前实现中 `d_i` 表示最大安全方向导数、`m` 表示 margin 的记号，可写为：

$$
a_{ii}^\top u_i
\geq
\sigma_{\mathcal U_i}(a_{ii})
-d_i
-\alpha\!\left(V_i-m\right).
$$

因此若恢复 runtime filter，只需引入一个 ego box + half-space projector。与当前 soft HJ-CRPO 实现不同，robust 路线中的 `d_i` 和 `a_ii` 必须在包含邻居动作系数的 max-min HJ 参数化下学习，不能把邻居动作影响重新当成不可辨识噪声。

需要强调：不能只在推理时给当前 QP 加一个 worst-case margin，却继续用 cooperative 或只含 ego action 的 Bellman/HJ target。这样训练的 Value 仍对应另一个控制问题。robust 消元必须与训练时的 max-min 算子和方向导数参数化一起修改。

## 6. 不依赖解析动力学的训练方法

### 6.1 训练数据

继续单独、off-policy 地训练 Safety Critic。每条样本应当包含：

$$
\left(
G_t,
u_{1:N,t},
c_t,
G_{t+1},
c_{t+1},
\mathrm{done}_t
\right).
$$

当前 `SafetyBatch` 和 replay 已经存储全体智能体动作与完整图，因此数据主体不需要重做。loss 在构造智能体 `i` 的局部目标时，根据图中的可控邻居 mask 取出对应动作即可。

原 Deep-QP 的 Value loss 和 derivative loss 可以保留整体结构，但应做如下替换：

$$
x\longrightarrow G_i,
$$

$$
u\longrightarrow u_{\mathcal A_i},
$$

$$
q^\lambda(x,u)\longrightarrow
q_i^{\mathrm{coop}}(G_i,u_{\mathcal A_i})
\quad\text{或}\quad
q_i^{\mathrm{rob}}(G_i,u_{\mathcal A_i}),
$$

并把下一状态上的动作优化分别替换为 joint-max 或 ego-max/neighbor-min。

Deep-QP 原文的收缩结论针对完整状态、完整控制输入和其定义的算子。max 和 min 在上确界范数下都是非扩张映射，因此 robust 算子的收缩证明有合理的延伸路径；但在完成局部观测、动态图和函数逼近条件下的正式推导前，不能直接引用原定理宣称当前 multi-agent robust loss 已经收敛。

### 6.2 为什么仍建议 off-policy

只使用 PPO 当前策略产生的 on-policy 动作，局部联合动作通常高度相关，难以辨识各个 `a_ij`。安全过滤器介入后，实际动作覆盖会进一步收窄。建议 replay 混合：

- 当前与历史 policy 的实际 executed actions。
- action box 内的独立与相关随机扰动。
- 安全边界附近的高优先级样本。
- 如果 VMAS 的 step 是纯函数，可从同一个图状态分支执行若干组联合动作，形成 counterfactual action branches。

最后一种方法特别适合本项目：它不要求知道 `f` 和 `g`，也不修改 reward，只利用模拟器本身检查并学习动作对下一安全值的影响。

### 6.3 避免 policy distribution leakage

训练、验证和测试应至少使用多个行为策略或不同训练阶段的 checkpoint。否则 `d_i` 可能仍通过图特征猜测某个固定邻居 policy 的动作，而没有真正学到 joint-action effect。建议在 minibatch 中随机打散策略来源，并报告跨 policy 的方向导数误差。

## 7. GNN 本身还需要的两个通用修复

### 7.1 平滑邻居进出图

hard sensing radius 会让 `G_i` 在邻居刚好越过阈值时离散变化，不符合连续 HJ/CBF 推导。参考 HJB-GNN 的“外层邻居零梯度”条件，可以设置内外两个半径，并对消息和 `a_ij` 使用连续门控：

$$
w(r)=
\begin{cases}
1, & r\leq R_{\mathrm{in}},\\
\frac{1}{2}\left[
1+\cos\!\left(
\pi\frac{r-R_{\mathrm{in}}}{R_{\mathrm{out}}-R_{\mathrm{in}}}
\right)
\right],
& R_{\mathrm{in}}<r<R_{\mathrm{out}},\\
0, & r\geq R_{\mathrm{out}}.
\end{cases}
$$

让边消息、邻居动作系数和可选的 attention logit 同时乘以该门控。在训练中增加同一物理状态、不同 padding/外层节点表示之间的 value consistency loss，使外层节点的加入不会产生突跳。

### 7.2 局部观测的 Markov 性

即使加入全部可控邻居动作，如果单帧 LiDAR 图没有速度、遮挡后物体运动或接触状态，下一图仍不一定由当前图和联合动作决定。无需建立显式动力学，可以按成本从低到高采用：

1. 在安全图中加入相邻帧差分得到的相对速度和上一时刻动作。
2. 使用短帧堆叠。
3. 把 Safety Critic encoder 改成 recurrent GNN，并在 safety replay 中保存短序列。

这些方法修复的是 observation aliasing，不应与邻居动作缺失混为同一个问题。

## 8. 执行方式的取舍

| 执行方式 | 是否保持独立局部 QP | 安全语义 | 主要代价 |
|---|---:|---|---|
| 只含 ego action 的旧原型 | 是 | 对固定邻居策略的隐式平均 | derivative target 有混叠，换策略会漂移 |
| cooperative local-joint QP | 否 | 邻域内合作控制 | 需通信、迭代或集中 solver |
| robust neighbor elimination | 是 | 对 action box 内任意邻居动作最坏情况 | 保守、可能 deadlock 或不可行 |
| 邻居 nominal action exchange | 需要一次通信 | 对交换动作条件化 | 同步动作与过滤后动作不一致 |
| 将 joint/robust QP 蒸馏成 GNN policy | 是，且无在线 QP | 经验继承 teacher | 蒸馏误差破坏 hard certificate |

如果研究目标是恢复显式且独立的 local QP，优先考虑 robust neighbor elimination，因为它同时满足：不改 reward、不依赖解析动力学、保留 GNN 高阶邻域建模、RL 训练继续 CTDE、执行时每个智能体只解自己的局部 QP。当前工程选择的是表格之外的 soft HJ-CRPO 路线：它保持 CTDE 和分布式 actor 执行，但不求解 runtime QP，也不提供逐步硬安全保证。

HJB-GNN 的 policy imitation 可以作为后续加速方案：用本文的数据驱动 joint/robust QP 生成 teacher action，再训练本地 GNN safety correction policy。但它应当是可选部署后端，不应替代第一阶段对 QP certificate 的验证。

### 8.1 GCBF+ 不会把联合约束自动变成独立 QP

如果直接把所有局部 Graph-HJ value 作为 CBF，理论上得到的是稀疏联合 QP：

$$
\min_{\bar u}\sum_i\lVert u_i-u_i^{\mathrm{nom}}\rVert^2
$$

subject to：

$$
\widehat{\partial V_i}(z_i,\bar u_{\mathcal N_i})
+\alpha(V_i(z_i))
\ge\varepsilon_i,
\qquad \forall i.
$$

不同局部约束共享邻居动作，所以不能直接拆成互相独立的 ego QP。保持 CTDE 的选择包括：

1. 集中训练 joint-QP teacher，再蒸馏到共享局部 policy；
2. 使用邻居通信和 ADMM 等分布式稀疏 QP solver；
3. 证明安全责任预算后再做责任分配；
4. 把邻居视为有界 disturbance，重新训练 ego-max/neighbor-min robust HJ value；
5. 像当前方案一样放弃 runtime QP，只在 centralized training 中用 joint-action HJ residual 构造 soft constrained policy update。

前四种路线分别引入蒸馏误差、通信迭代、额外责任分配定理或 robust 保守性；第五种保持分布式执行，但不提供逐步硬安全保证。

## 9. 若继续 robust local-QP 路线的代码改动

### 9.1 `dgppo/algo/module/deep_qp_safety.py`

- 保留 `GraphTransformerGNN` 和 twin Value heads。
- 复用当前 shared pair coefficient head；若需要单独提取 ego 项和邻居项，应从现有 joint coefficient tensor 与 action mask 中分解，不再恢复 ego-only head。
- edge coefficient 继续由边两端 embedding 和局部图上下文生成，并使用 padding/节点类型 mask。
- `SafetyNetworkOutput` 和 `SafetyCertificate` 增加邻居系数及其 edge-to-ego、edge-to-agent 索引。
- `_support` 扩展为同时计算 ego support 和 robust neighbor lower support。
- loss 使用 `batch.actions` 中所有可控邻居的动作，以 segment sum 聚合每个 ego 的联合方向导数项。
- robust 模式下，下一状态 derivative optimum 使用 ego-max/neighbor-min 的解析标量 `d_i`。
- checkpoint metadata 增加参数化版本；旧 checkpoint 不能静默加载到新 head。

### 9.2 `dgppo/algo/informarl_deep_qp.py`

- RL actor、PPO update、nominal/executed action 语义保持不变。
- `project_action` 在 robust 模式下仍调用当前单 half-space projector。
- 增加 joint-action derivative residual、neighbor coefficient norm、robust margin、边界连续性和 adversarial violation 指标。
- Safety Critic 仍建议先离线预训练并冻结；在线更新仅作为可选实验，不改变 PPO 的 on-policy buffer。

### 9.3 图与 replay

- 复用现有完整 joint action replay。
- 增加从 agent-agent edge 到 sender action 和 receiver ego 的稳定索引/mask。
- 若加入 recurrent critic，再扩展 replay 为固定长度序列；不要把 RNN hidden state 当作跨 checkpoint 可复用的物理状态。

## 10. 必须先做的诊断实验

在大规模重训前，建议先在同一批边界状态上做 action branching。对固定 `G_i` 和 `u_i`，只改变邻居动作，比较有限差分目标：

$$
y_i=
\frac{V_{\mathrm{target}}(G_{i,t+1})-V_{\mathrm{target}}(G_{i,t})}{\Delta t}.
$$

需要报告：

$$
\operatorname{Var}[y_i\mid G_i,u_i],
$$

$$
\operatorname{Var}[y_i\mid G_i,u_i,u_{\mathcal N_i^a}],
$$

以及 joint affine head 的 held-out residual。若第二个条件方差没有明显下降，说明主要问题不是邻居动作缺失，而是局部观测非 Markov、动作非仿射、接触离散性或图边界跳变，应先修复第 6 节问题。

建议最小消融组：

1. 当前 ego-only Deep-QP。
2. joint-action head，但用 cooperative target。
3. joint-action head + robust target + 独立局部 QP。
4. 第 3 组 + smooth graph boundary。
5. 第 4 组 + history/recurrent encoder。

除 reward、return 和原有安全率外，还应报告跨邻居策略的 derivative residual、QP infeasible rate、worst-case sampled violation、intervention rate、最小安全距离和 deadlock rate。

## 11. 仍未解决和不能过度声明的部分

1. **局部信息充分性。** 有限 LiDAR/GNN 观测可能不是安全动力学的 Markov state；joint action head 只能修复已观测邻居动作缺失。
2. **动作仿射假设。** VMAS 在 action clipping、接触、碰撞响应和离散积分处可能不满足全局 affine-in-action。应通过同状态 action branching 测量误差，而不是先反推动力学公式。
3. **HJ Value 的非光滑性。** 整邻域标量能表示多邻居高阶几何，但精确 reachability Value 在 switching surface 或 corner 仍可能不可微；单个 affine head 只给出一种局部近似。必要时再研究多方向 head 或 nonsmooth certificate，不能把 twin Value heads误当成两个独立 CBF 约束。
4. **robust 保守性。** 把所有邻居动作视为 adversarial 会缩小可行集；独立局部 QP 与低保守性不能同时免费获得。可在后续用可信动作集合替代整个 action box，但这会把保证改成集合条件保证。
5. **QP 不可行和 slack。** 增加 slack 可以保证总有输出，却会放松安全约束；必须把 slack 使用率和幅值单独报告。
6. **函数逼近误差。** Bellman/HJ residual 小不等价于全状态域 CBF 有效。可以用 held-out worst-case residual 分位数设置额外 margin，但它首先是经验校准，不是全局形式化验证。
7. **收敛性证明。** Deep-QP 的固定算子收缩结论不能原封不动地覆盖局部图、max-min、多网络同步更新和 SGD。新方案应先称为“结构一致的扩展”，直到补出 robust graph operator 的正式证明。
8. **CTDE 的含义。** Safety Critic 可以在集中训练阶段读取全体 replay actions，但部署只能使用本地图。若运行时读取全局 joint action 或调用全局 QP，就不再是本文要求的 decentralized execution。

## 12. 推荐实施顺序

### P0：先证实 gap

实现同状态 joint-action branching 和条件方差指标，验证 neighbor action omission 确实是当前主要误差源。

### P1：结构修复

当前实现已完成 context-conditioned pair coefficient head、joint-action HJ loss 和相应 mask。下一步应先用 action branching 检查其可辨识性、拟合误差和跨策略泛化。

### P2：恢复独立执行

切换为 robust max-min target，并复用现有 box-half-space QP。比较 safety、deadlock、return 和 infeasible rate。

### P3：处理动态图与部分可观测

加入 smooth radius gate；若 branching residual 仍大，再加入历史特征或 recurrent GNN。

### P4：可选降保守与加速

研究可信邻居动作集合、一次动作通信或 HJB-GNN 风格的 teacher-policy distillation。它们属于性能优化，不应先于 P0 至 P3。

## 13. 一手资料

- Wang, Shu, He, Zhao, *Safe Multi-Agent Navigation via Constrained HJB-Informed Learning*, arXiv:2506.22117v2, 2026：[论文 HTML](https://arxiv.org/html/2506.22117)。关键位置包括式 (1)、式 (4)、式 (10) 至式 (12)、Theorem 1 和 Section IV。
- HJB-GNN 作者官方实现：[hublan24/HJB-GNN](https://github.com/hublan24/HJB-GNN)。当前 README 显示公开环境主要为 CrazyFlie，并包含 graph CBF、controller、value 与 replay 相关训练参数。
- HJB-GNN 作者项目页：[NUS CORE HJB-GNN](https://nus-core.github.io/assets/standalone/HJB-GNN/index.html)。
- Hsu et al., *Deep QP Safety Filter: Model-free Learning for Reachability-based Safety Filter*, arXiv:2601.21297, 2026：[论文 HTML](https://arxiv.org/html/2601.21297)。关键位置为方向导数重参数化、off-policy target networks、式 (10) 至式 (11) 和固定算子收缩结论。
- GCBF+：[论文](https://arxiv.org/html/2401.14554) · [Definition 1、动态图条件与全局安全定理](https://arxiv.org/html/2401.14554#S3.SS2) · [联合 CBF-QP](https://arxiv.org/html/2401.14554#S3.E17) · [官方代码](https://github.com/MIT-REALM/gcbfplus)
