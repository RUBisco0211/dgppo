# Multi-Agent Deep-QP 的邻居动作缺口与 Pairwise HJ / Layered Safety 修正路线

## 1. 结论

当前初版 Multi-Agent Deep-QP 的主要理论缺口不是“Safety Critic 使用了 GNN”，而是：局部 safety value 已通过 GNN 依赖邻居状态，但方向导数只显式参数化本智能体动作。对于一般多智能体动力学，同一个局部图和本智能体动作在不同邻居动作下会产生不同的下一局部图，因此当前 derivative target 不是局部状态与本智能体动作的单值函数。

《Resolving Conflicting Constraints in Multi-Agent Reinforcement Learning with Layered Safety》提供了直接相关的修正骨架：

1. 在每条 agent-agent 边上建立相对状态；
2. 定义 pairwise HJ value；
3. 在相对动力学中显式保留双方动作；
4. cooperative 情况使用双方动作的二智能体联合 CBVF-QP；
5. non-cooperative 情况把邻居动作作为最坏情况扰动，使用 min-max HJ value；
6. 多邻居时只激活最危险 pair，并让 MARL 策略提前避免进入多重冲突区域。

因此，这篇论文能够解决当前方案中“邻居动作被错误吸收到 state-only scalar head”这一建模缺口，并为 edge-wise neural HJ critic 提供正确的状态、动作和 QP 接口。但是，它不能单独解决任意多智能体同时交互下的联合前向不变性：论文自身明确把形式化保证限制在 pairwise interaction，多重 engagement 和 leaky corner 仍通过优先级、保守冲突区域与 MARL 策略进行缓解，而不是得到完整证明。[论文 §IV–V](https://arxiv.org/html/2505.02293#S4)、[论文局限性](https://arxiv.org/html/2505.02293#S7)、[RSS 2025 官方页面](https://www.roboticsproceedings.org/rss21/p094.html)

## 2. 当前初版的具体缺口

### 2.1 当前网络与数据依赖

当前 `LocalDeepQPSafetyFilter` 使用共享局部 GNN，对每个智能体输出 twin value、own-action coefficient 和 scalar deviation：

$$
\operatorname{LocalGNN}(G)_i
\longrightarrow
\left(v_{i,1},v_{i,2},a_{ii},b_i\right).
$$

智能体的 embedding 聚合通信范围内的邻居状态，因此 value 实际是局部邻域的函数：

$$
v_i=v_i(o_i),
\qquad
o_i=\psi_i(x_i,\{x_j\}_{j\in\mathcal N_i}).
$$

但当前 derivative head 只使用本智能体动作：

$$
\widehat{\partial v_i}(o_i,u_i)
=
a_{ii}(o_i)^\top u_i
-
\sigma_{\mathcal U_i}(a_{ii}(o_i))
+
b_i(o_i).
$$

当前实现可见 [Deep-QP Safety Critic](/Users/rubisco/Desktop/code/rl/dgppo/dgppo/algo/module/deep_qp_safety.py)；其中 GNN 产生逐智能体 embedding，loss 只把每个智能体自己的 action 与自己的 coefficient 相乘。

### 2.2 正确的联合动作导数

若联合系统是控制仿射的：

$$
\dot x
=
f(x)
+
\sum_{j=1}^{N}g_j(x)u_j,
$$

则任何依赖邻居状态的局部 value 一般满足：

$$
\dot v_i
=
L_fv_i
+
L_{g_i}v_i\,u_i
+
\sum_{j\in\mathcal N_i}L_{g_j}v_i\,u_j.
$$

相应的学习参数化应至少包含：

$$
\widehat{\dot v_i}
=
b_i(o_i)
+
a_{ii}(o_i)^\top u_i
+
\sum_{j\in\mathcal N_i}a_{ij}(o_i)^\top u_j.
$$

只有在邻居动作是固定且能由局部状态唯一确定的情况下，才能把邻居项吸收到 scalar：

$$
b_i^{\mu_{-i}}(o_i)
=
b_i(o_i)
+
\sum_{j\in\mathcal N_i}a_{ij}(o_i)^\top\mu_j(o_j).
$$

此时 critic 将依赖邻居策略；一旦 RL actor 改变，预训练且冻结的 scalar 就会失配，也不再是 Deep-QP 所要求的任务无关 HJ safety critic。

### 2.3 对 off-policy derivative learning 的影响

当前 replay 中真实的有限差分 target 来自联合动作：

$$
y_i
=
\frac{v_i(o_i')-v_i(o_i)}{\Delta t}
\approx
b_i(o_i)
+
a_{ii}(o_i)^\top u_i
+
\sum_j a_{ij}(o_i)^\top u_j.
$$

当网络只条件化于局部状态和本智能体动作时，相同输入可能对应多个 target：

$$
(o_i,u_i)
\longrightarrow
\left\{y_i^{(1)},y_i^{(2)},\ldots\right\},
$$

差别来自 replay 中不同的邻居动作。平方损失更可能得到 behavior distribution 下的条件均值：

$$
\widehat{\dot v_i}(o_i,u_i)
\approx
\mathbb E
\left[
\dot v_i
\mid o_i,u_i
\right].
$$

CBF 需要逐状态、逐动作或最坏情况意义下的约束，平均导数不能排除 false-safe。Deep-QP 原论文能从 arbitrary behavior transitions 进行 off-policy 学习，是因为 transition 对完整状态和完整控制输入条件化；当前局部版本遗漏邻居动作后不能直接继承这一论证。[Deep-QP §3.4](https://arxiv.org/html/2601.21297#S3.SS4)

### 2.4 对收敛结论的影响

Deep-QP 证明的是条件 Bellman operator contraction：固定 derivative/safety-advantage 项时，value operator 收缩；固定 value 时，derivative operator 收缩。它本身不是有限 replay、神经函数逼近与联合 SGD 的全局收敛证明。[Deep-QP Theorem 3.5–3.6](https://arxiv.org/html/2601.21297#S3.SS4)

在当前多智能体初版中：

- 折扣 value backup 可能仍在一个固定经验 transition kernel 下具有收缩外形；
- 但其 fixed point 可能只是特定邻居 behavior mixture 下的 observation-level value；
- derivative target 对当前输入不是单值函数，原 derivative operator 的定义域与固定点已发生改变；
- 因此原文的 derivative contraction 不能直接迁移；
- 即使训练 loss 稳定下降，也不能据此断言学到了正确的多智能体 HJ derivative 或前向不变安全集。

## 3. Layered Safety 论文实际做了什么

### 3.1 它不是对整张多智能体图求联合 HJ

论文对每一对智能体建立相对状态：

$$
s^{(ij)}
=
\operatorname{rel}(s^{(i)},s^{(j)}),
$$

并显式写出双方动作驱动的相对动力学：

$$
\dot s^{(ij)}
=
f^{(ij)}
\left(
s^{(ij)},a^{(i)},a^{(j)}
\right).
$$

论文要求智能体的局部观测足以重建可见邻居的相对状态。这与图中的 edge feature 很契合：一条边可以直接承载相对位置、相对速度、相对朝向及双方自身速度，而无需先把所有邻居池化为一个不可分解的 node embedding。[论文问题建模 §III](https://arxiv.org/html/2505.02293#S3)

### 3.2 Pairwise cooperative HJ value

论文首先定义 pairwise 最小未来距离：

$$
J
\left(
s_0^{(ij)},\boldsymbol a^{(i)},\boldsymbol a^{(j)}
\right)
=
\min_{t\geq0}
\operatorname{dist}
\left(s^{(ij)}(t)\right).
$$

双方为安全协作时，pairwise HJ value 为：

$$
V(s_0^{(ij)})
=
\max_{\boldsymbol a^{(i)},\boldsymbol a^{(j)}}
J
\left(
s_0^{(ij)},\boldsymbol a^{(i)},\boldsymbol a^{(j)}
\right).
$$

对应的 pairwise safe set 为：

$$
\mathcal S^{(ij)}
=
\left\{
s^{(ij)}
\mid
V(s^{(ij)})\geq r_{\mathrm{safety}}
\right\}.
$$

论文令：

$$
B(s^{(ij)})
=
V(s^{(ij)})-r_{\mathrm{safety}},
$$

并在 value 几乎处处可微时把它作为 CBVF。[论文 §IV-A1–A2](https://arxiv.org/html/2505.02293#S4.SS1)

### 3.3 Cooperative pairwise filter显式优化双方动作

论文 cooperative safety filter 不是只求本智能体动作，而是求一个 pair 的两个动作：

$$
\begin{aligned}
(a_{\mathrm{safe}}^{(i)},a_{\mathrm{safe}}^{(j)})
=
\arg\min_{a^{(i)},a^{(j)}}\quad
&
\|a^{(i)}-a_{\mathrm{marl}}^{(i)}\|^2
+
\|a^{(j)}-a_{\mathrm{marl}}^{(j)}\|^2
\\
\mathrm{s.t.}\quad
&
\nabla B(s^{(ij)})^\top
f^{(ij)}
\left(s^{(ij)},a^{(i)},a^{(j)}\right)
+
\gamma B(s^{(ij)})
\geq0.
\end{aligned}
$$

论文称其可分散执行，是因为 pair 中两个智能体可以基于一致信息分别求解相同的二智能体优化，再各自执行属于自己的动作分量。它仍然需要双方 nominal action、一致的相对状态和一致的 pair 选择；这不是当前 own-action-only 的完全独立 QP。[论文 cooperative CBVF filter](https://arxiv.org/html/2505.02293#S4.SS1.SSS2)

### 3.4 Non-cooperative 版本显式采用最坏邻居动作

若不能假设邻居为安全协作，论文定义 differential-game value：

$$
V_{\mathrm{worst}}(s_0^{(ij)})
=
\min_{\boldsymbol a^{(j)}}
\max_{\boldsymbol a^{(i)}}
J
\left(
s_0^{(ij)},\boldsymbol a^{(i)},\boldsymbol a^{(j)}
\right),
$$

并要求本智能体 QP 对邻居动作的最坏情况仍满足 barrier 条件：

$$
\min_{a^{(j)}\in\mathcal A_j}
\nabla B(s^{(ij)})^\top
f^{(ij)}
\left(s^{(ij)},a^{(i)},a^{(j)}\right)
+
\gamma B(s^{(ij)})
\geq0.
$$

这与“edge-wise coefficient + 对邻居动作取支持函数下界”的分布式鲁棒 QP 路线直接对应。[论文 non-cooperative filter](https://arxiv.org/html/2505.02293#S4.SS1.SSS2)

### 3.5 论文使用的是网格化 HJ PDE 求解，不是 grid search

论文使用 `hj_reachability` 离线数值求解低维相对状态上的 HJ PDE。运行示例的 pairwise relative state 是五维，论文报告该 value 的离线计算在 GPU 上一小时内完成。准确术语是 grid-based HJ reachability 或网格化动态规划，不是通常用于调超参数的 grid search。[论文 §IV-A1](https://arxiv.org/html/2505.02293#S4.SS1.SSS1)

这种方法的优点是：在已知低维动力学、足够细网格和正确数值设置下，可以直接得到 pairwise value 与梯度。缺点是维数灾难，无法直接扩展到整张多智能体联合状态。论文因此复用同一个低维 pairwise value 模板，而不是为所有邻居联合求解 HJ。[作者官方代码仓库](https://github.com/DINaMo-MIT/Layered-Safe-MARL)

## 4. 它能解决什么，不能解决什么

| 当前问题 | Layered Safety 是否提供解法 | 判断 |
|---|---|---|
| value 依赖邻居状态，但导数遗漏邻居动作 | 是 | pairwise relative dynamics 显式依赖双方动作 |
| off-policy derivative target 对本地输入非单值 | 部分可以 | 将训练输入改为 pair state 与 pair joint action 后恢复单值性 |
| 如何保持纯本地 QP | 是，但需选择语义 | cooperative 需一致地解 pair joint QP；non-cooperative 可对邻居动作取 worst case |
| GNN 聚合后难以识别是哪条边造成危险 | 是 | pairwise CBVF 为每条边独立打分并选最危险边 |
| 多条 pairwise CBF 约束互相冲突 | 不能彻底解决 | 论文优先一个 pair，并用策略避免进入冲突区 |
| 任意多智能体联合前向不变性 | 否 | 论文明确仅保证 pairwise interaction |
| 未知黑箱动力学下的 pairwise HJ | 未直接解决 | 论文用已知动力学做网格 HJ；需要与 Deep-QP 学习方法融合 |
| 神经 HJ critic 的收敛证明 | 未直接解决 | 网格数值解的保证不能直接转移到 GNN/MLP 与 SGD |

因此不能把论文结论表述为“Layered Safety 已解决 Multi-Agent Deep-QP 的全部理论问题”。更准确的表述是：

> Layered Safety 给出了修复 pairwise 状态—动作建模的正确分解，并给出了 cooperative 与 robust 两类执行语义；它把多约束冲突作为独立的 higher-order 问题，通过优先级和策略层规避，而不是用 pairwise HJ 证明联合安全。

论文也明确指出，pairwise safe set 的交集不一定是真正的多智能体 safe set；三个及以上智能体可进入 leaky corner，使所有 pair 分别安全但联合约束已经不可同时满足。[论文 §IV-B](https://arxiv.org/html/2505.02293#S4.SS2)

## 5. 建议的融合方案：Neural Pairwise Deep-QP

### 5.1 从 node-wise scalar value 改成 edge-wise pair value

对每条可见 agent-agent 边构造相对安全状态：

$$
z_{ij}
=
\operatorname{PairState}
\left(x_i,x_j,e_{ij}\right).
$$

建议至少包含：

- 相对位置；
- 相对速度；
- 双方速度或其他决定动力学的自身状态；
- 相对朝向；
- 实体类型与动作上下界；
- 必要时与感知边界相关的 mask。

共享 pairwise critic 对所有同类型边复用参数：

$$
F_{\phi}^{\mathrm{pair}}(z_{ij})
\longrightarrow
\left(
v_{ij,1},
v_{ij,2},
a_{ij}^{(i)},
a_{ij}^{(j)},
b_{ij}
\right).
$$

agent-obstacle 边则只需要可控一侧 coefficient：

$$
F_{\phi}^{\mathrm{obs}}(z_{ik})
\longrightarrow
\left(
v_{ik,1},
v_{ik,2},
a_{ik}^{(i)},
b_{ik}
\right).
$$

这比先把整个邻域聚合成一个 node embedding 再输出单个 value 更容易保持动作归因和约束可诊断性。GNN 仍可用于构造 edge context，但 pairwise value 与 pairwise derivative heads 不应在输出前丢失边身份。

### 5.2 Cooperative Deep-QP 重参数化

把双方动作视为一个 joint control：

$$
\bar u_{ij}
=
\begin{bmatrix}
u_i\\
u_j
\end{bmatrix},
\qquad
\bar a_{ij}
=
\begin{bmatrix}
a_{ij}^{(i)}\\
a_{ij}^{(j)}
\end{bmatrix}.
$$

相应的 Deep-QP derivative 参数化为：

$$
\widehat{\partial v}_{ij}
=
(a_{ij}^{(i)})^\top u_i
+
(a_{ij}^{(j)})^\top u_j
-
\sigma_{\mathcal U_i}(a_{ij}^{(i)})
-
\sigma_{\mathcal U_j}(a_{ij}^{(j)})
+
b_{ij}.
$$

于是：

$$
\max_{u_i,u_j}
\widehat{\partial v}_{ij}
=
b_{ij}.
$$

这恢复了 Deep-QP 重参数化所需的 action maximization identity。HJ replay loss 必须使用双方实际执行动作：

$$
(z_{ij},u_i,u_j,c_{ij},z_{ij}',c_{ij}').
$$

### 5.3 Cooperative pair QP

每个激活 pair 求解：

$$
\begin{aligned}
(u_i^{\mathrm{safe}},u_j^{\mathrm{safe}})
=
\arg\min_{u_i,u_j}\quad
&
\|u_i-u_i^{\mathrm{nom}}\|^2
+
\|u_j-u_j^{\mathrm{nom}}\|^2
\\
\mathrm{s.t.}\quad
&
\widehat{\partial v}_{ij}(z_{ij},u_i,u_j)
+
\alpha(v_{ij}-m)
\geq0,\\
&
u_i\in\mathcal U_i,
\qquad
u_j\in\mathcal U_j.
\end{aligned}
$$

这种方案最贴近论文，但需要 pair 内交换 nominal action 或通过确定性协议获得完全一致的 joint QP 输入。若一个 agent 同时属于多个激活 pair，还需要统一 neighborhood QP、责任分配或只激活一个 pair，否则不同 pair 可能给同一 agent 产生不同安全动作。

### 5.4 Robust decentralized QP

若不希望交换或预测邻居即时动作，必须学习 differential-game / worst-case pair value，而不是直接把 cooperative value 塞进鲁棒约束。

当 learned derivative 为：

$$
\widehat{\dot v}_{ij}
=
b_{ij}
+
(a_{ij}^{(i)})^\top u_i
+
(a_{ij}^{(j)})^\top u_j,
$$

本智能体约束可写成：

$$
(a_{ij}^{(i)})^\top u_i
+
b_{ij}
+
\min_{u_j\in\mathcal U_j}
(a_{ij}^{(j)})^\top u_j
+
\alpha(v_{ij}-m)
\geq0.
$$

对于对称 box：

$$
\min_{u_j\in[-1,1]^{d_u}}
(a_{ij}^{(j)})^\top u_j
=
-\|a_{ij}^{(j)}\|_1.
$$

因此每个 agent 仍可只优化自己的动作：

$$
(a_{ij}^{(i)})^\top u_i
\geq
-b_{ij}
+
\|a_{ij}^{(j)}\|_1
-
\alpha(v_{ij}-m).
$$

但对应的 neural HJ loss 也必须从 cooperative max HJ 改为 max-min discounted HJ。Deep-QP 原论文只给出单控制最大化版本；max-min Bellman operator、重参数化、target 构造与 contraction 需要重新推导，不能只修改推理期 QP。

### 5.5 Layered conflict handling

在 edge-wise critic 之外，可复用论文的三层策略：

1. 用 pairwise value 评估所有邻边；
2. 选择最小 pairwise barrier 的邻居作为优先 pair；
3. 只对 mutual-priority pair 激活 cooperative filter；
4. 根据 worst-case pairwise value 推导 potential conflict range；
5. 当两个以上邻居进入 potential conflict range 时，对 MARL reward 加间接 conflict penalty；
6. 通过 curriculum 逐步增大 safety radius 和 conflict radius。

这能降低 pair constraints 同时冲突的频率，却不是对联合安全的证明。实验中必须单独统计：

- mutual pair 选择一致率；
- 一个 agent 同时被多个 pair 选择的频率；
- pair QP 不一致率；
- multi-engagement / leaky-corner rate；
- QP infeasible rate；
- pairwise false-safe rate；
- 未被优先 pair 的碰撞比例。

## 6. 对收敛证明可以恢复到什么程度

### 6.1 Cooperative pairwise 情况

如果满足：

1. 相对状态对 pair 动力学是 Markov 且充分的；
2. pair transition 只由双方状态和双方动作决定；
3. replay 保存完整 pair joint action；
4. value、derivative 和 HJ action optimization 都使用同一个 cooperative 语义；
5. 固定图拓扑区间内 transition 采样满足 Deep-QP 的时间离散近似；

则可以把 Deep-QP 原来的单一控制向量替换为拼接 joint action，并在 pairwise state space 上复用 operator-level contraction 证明：

$$
u
\leftarrow
\begin{bmatrix}u_i\\u_j\end{bmatrix}.
$$

这能够修复当前 target 非单值和 Hamiltonian 不一致的问题。但它仍不自动证明：

- 共享神经 trunk 的 SGD 全局收敛；
- 有限 replay 覆盖整个 pair state-action space；
- learned value 的误差不会造成 false-safe；
- 多条 pairwise safe set 的交集前向不变；
- prioritized filter 不会忽略真正危险的非优先边。

### 6.2 Non-cooperative 情况

若采用 worst-case neighbor action，则 value 定义、Hamiltonian 与 Bellman operator都发生变化：

$$
\max_{u_i}\min_{u_j}
\widehat{\dot v}_{ij}.
$$

需要单独证明 max-min operator 在固定交叉项时仍具有相应的折扣 contraction，并明确 Isaacs 条件、动作优化顺序及采样数据如何覆盖 disturbance action。Layered Safety 的网格 min-max HJ 说明该控制语义是正确方向，但没有给出 neural off-policy Deep-QP 的训练证明。

## 7. 推荐决策

不建议继续把当前 node-wise value 与 own-action derivative 作为主方案。建议按以下顺序推进：

1. 将 agent-agent safety value 改成共享参数的 edge-wise pair critic；
2. derivative head 同时输出 ego 与 neighbor action coefficient；
3. 先实现 cooperative joint-action HJ loss，并在双智能体 toy system 上对照网格 HJ value；
4. 先实现 mutual-priority pair joint QP，验证和 Layered Safety 网格基线的一致性；
5. 再加入多邻居 prioritization、potential-conflict penalty 与 curriculum；
6. 如果部署要求不交换 nominal action，再推导并实现 max-min neural HJ 与 robust local QP；
7. 在任何完整多智能体安全声明之前，保留“pairwise guarantee / learned approximation / no general multi-engagement guarantee”的限定。

最适合当前项目的第一条可验证路线是：

$$
\boxed{
\text{pairwise relative state}
+
\text{joint-action Deep-QP HJ loss}
+
\text{mutual-priority pair QP}
+
\text{InforMARL conflict avoidance layer}
}
$$

它比当前 own-action-only 方案更符合 Deep-QP 的重参数化和 off-policy 条件，也比直接为整个多智能体联合状态学习 HJ value 更可扩展；代价是安全保证仍停留在 pairwise 层面，多重交互只能先缓解、监测和保守处理。

## 8. 一手来源

- [Choi et al., Resolving Conflicting Constraints in Multi-Agent Reinforcement Learning with Layered Safety, arXiv / RSS 2025](https://arxiv.org/html/2505.02293)
- [Robotics: Science and Systems XXI 官方论文页](https://www.roboticsproceedings.org/rss21/p094.html)
- [作者官方 Layered-Safe-MARL 代码仓库](https://github.com/DINaMo-MIT/Layered-Safe-MARL)
- [作者项目页](https://dinamo-mit.github.io/Layered-Safe-MARL/)
- [Kim and Kim, Deep QP Safety Filter, PMLR 2026](https://proceedings.mlr.press/v331/kim26c.html)
- [Deep-QP Safety Filter §3.4：off-policy Bellman operators 与 target networks](https://arxiv.org/html/2601.21297#S3.SS4)
