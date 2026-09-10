# 从 Deep-QP Graph-HJ 回到 GCBF+：可辩护的创新路线

> 调研日期：2026-09-08。本文只使用原论文、作者主页、出版社或会议官方页面。它是定向文献核查，不是穷尽式查新；“新颖性风险”表示与已发现工作的重合风险，而非正式的 novelty 判定。

## 1. 结论先行

不建议继续把当前 `Graph-HJ + advantage mixing` 作为主理论线。它既没有原 Deep-QP 的在线 QP safety filter，也没有证明 learned HJ value 相比环境几何约束带来足够大的可恢复集差异；在多智能体论文中，完整说明 joint-action HJ、局部可观测性、分布式执行和函数逼近误差的成本很高。

更实用的主线是：**以 GCBF+ 为骨架，学习一个短时域、速度敏感、保守且带校准误差裕量的 graph safety residual**。它专门回答一个可验证问题：

> 在当前几何约束仍为正，但由于相对速度、朝向、制动距离或邻居运动，未来已高风险的状态中，能否比原始 GCBF+ 更早预警，同时保留可变规模的分布式执行？

建议把创新组合限定为四点：

1. `h = c - positive_residual`，结构上保证 learned safe set 不超出原始几何安全集；
2. residual 监督来自短时域最小 future clearance，而不是完整 Deep-QP/HJ fixed point；
3. 用 ensemble/quantile/conformal calibration 把预测误差变成 barrier tightening；
4. 用边界反例采样专门覆盖 `c > 0` 但未来会碰撞的高速/夹逼状态。

这些元素分别已有先例，因此贡献不能写成“首次使用预测、uncertainty 或 velocity”。可辩护的贡献应是：**黑箱离散时间、图局部、任意 agent 数量、无需在线 joint QP 的统一方法，以及预测误差到 tightened graph-barrier 条件的明确界。**

## 2. 候选方向与撞车风险

| 方向 | 文献状态 | 单独作为创新的风险 | 对本仓库的建议 |
|---|---|---:|---|
| 集中 joint CBF-QP 教师蒸馏到局部 GNN | CBF-QP imitation、集中专家到分布式 GNN、集中安全网络到局部安全网络均已有 | 中高 | 可作为训练工具；只有加入误差界、反例 DAgger、residual gate/OOD 回退才适合作主贡献 |
| ADMM/consensus/message-passing 分布式 CBF-QP | 从自然责任分摊到精确分布式 QP、ADMM、截断迭代已有完整路线 | 很高 | 不建议只做“QP 分布式化”；若做，应聚焦有限通信轮 residual 到安全裕量的定理 |
| 一步 learned safety residual/action correction | 2018 已有 learned one-step linear safety model；2021 已扩展到 MARL | 很高 | 只适合作 baseline；必须升级为短时域、图局部、保守校准版本 |
| uncertainty-aware / robust barrier | GP robust multi-agent CBF、conformal decentralized CBF 已有 | 高 | uncertainty 必须服务于明确的 approximation bound，而不能只加 ensemble 方差 |
| velocity/reachability-aware barrier | velocity obstacle、predictive multi-robot CBF、neural HJ/reachability barrier 已有 | 高 | “加入速度输入”不新；可以做结构保守的 finite-horizon residual，并证明/验证提前量 |
| GCBF+ + finite-horizon calibrated residual | 已有组件较多，但尚有清晰系统 gap | 中 | **最推荐**：工程改动小、问题可证伪，也能直接与 GCBF+ 和当前 Graph-HJ 对比 |
| 学生策略 + certified residual gate + 必要时继续求解 | optimizer amortization 已有，但 time-varying graph CBF、有限通信和 fallback 的组合仍有空间 | 中 | 第二推荐，偏系统/控制；实现和理论成本明显高于上一项 |

## 3. 为什么五个直觉方向不能直接声称新颖

### 3.1 一步安全预测已经很成熟

[Dalal et al., *Safe Exploration in Continuous Action Spaces*](https://arxiv.org/abs/1801.08757) 从任意历史动作数据学习 safety signal 的一步、action-linear 变化模型，并在策略后解析修正动作。[Sheebaelhamd et al., *Safe Deep Reinforcement Learning for Multi-Agent Systems with Continuous Action Spaces*](https://arxiv.org/abs/2108.03952) 已把这种单步线性化 safety layer 扩展到多智能体连续动作，并用 soft constraints 处理不可行性。

因此，“预测下一步 constraint，再惩罚或修正动作”本身是高撞车路线。真正可区分的部分必须来自多步提前量、graph-local scalability、保守误差界或新的理论关系。

### 3.2 uncertainty-aware barrier 也不是空白

[Cheng et al., *Safe Multi-Agent Interaction through Robust Control Barrier Functions with Learned Uncertainties*](https://arxiv.org/abs/2004.05273) 已用 Matrix-Variate GP 学其他 agent 动力学的不确定界，并将 min-max robust CBF 化为实时 QP。[Huriot and Sibai, *Safe Decentralized Multi-Agent Control using Black-Box Predictors, Conformal Decision Policies, and Control Barrier Functions*](https://arxiv.org/abs/2409.18862) 又将 black-box trajectory predictor 的误差通过 conformal decision theory 转成自适应 CBF 约束。

因此，“给 GCBF+ 加 ensemble/GP/conformal”不够。需要明确说明误差对象究竟是 future-clearance、barrier residual、模型误差还是 student action error，并把它推导为训练/执行时的 tightening。

### 3.3 velocity-sensitive、predictive 和 reachability barrier 均已有先例

[Li et al., *A Predictive Cooperative Collision Avoidance for Multi-Robot Systems Using Control Barrier Function*](https://arxiv.org/abs/2501.10447) 已针对多机器人 CBF 缺少 look-ahead 的问题引入预测安全项。[Sánchez Roncero et al., *Multi-Agent Obstacle Avoidance using Velocity Obstacles and Control Barrier Functions*](https://arxiv.org/abs/2409.10117) 已组合 velocity obstacle 与 CBF。[So et al., *Reachability Barrier Networks*](https://arxiv.org/abs/2505.11755) 则直接学习 HJ 解形成平滑 CBF，并在 9D multi-vehicle collision avoidance 上报告相对 neural CBF 更安全且更不保守。

此外，[Wang et al., *Learning Distributed Safe Multi-Agent Navigation via Infinite-Horizon Optimal Graph Control*](https://arxiv.org/abs/2506.22117) 已把 graph CBF、分布式 GNN policy 和无限时域 HJB optimal control 放在同一框架中。因此，“Graph + HJ/long horizon”这个标题级组合也有很高重合风险。

### 3.4 centralized teacher 到 decentralized student 已有直接邻近工作

[Yaghoubi et al., *Training Neural Network Controllers Using Control Barrier Functions in the Presence of Disturbances*](https://arxiv.org/abs/2001.08088) 明确用 imitation learning 将在线 CBF-QP 蒸馏为 NN controller。[Cosner et al., *End-to-End Imitation Learning with Safety Guarantees*](https://arxiv.org/abs/2212.11365) 进一步将模仿误差作为扰动，给出由 robust CBF expert 导出的 input-to-state safety 保证。

多智能体侧，[Lee et al., *Graph Neural Networks for Decentralized Multi-Agent Perimeter Defense*](https://arxiv.org/abs/2301.09689) 使用 centralized expert imitation 训练可扩展 decentralized GNN；[Jiang and Guo, *Multi-Robot Guided Policy Search for Learning Decentralized Swarm Control*](https://doi.org/10.1109/LCSYS.2020.3005441) 使用分布式轨迹优化/Guided Policy Search 学习仅依赖局部观测的 swarm policy。更近的 MTT-SN 也采用“centralized safety filter 修正 joint risky actions，再由各 UAV 的 local safety network 模仿”的两阶段结构：[出版社页面](https://www.sciencedirect.com/science/article/pii/S0952197626016544)。

所以简单地用 joint QP 产生标签、再做 action MSE，不宜作为主创新。相对可辩护的是：variable-size graph local realizability、干预区的 DAgger/反例覆盖、action imitation error 到 Graph-CBF tightening 的界，以及 student 无法通过 residual check 时的回退机制。

### 3.5 多智能体并非“不能做分布式 CBF-QP”

正确表述应是：**多个互不协调的 ego-only QP 一般不能复现带耦合约束的 joint QP；但有责任分摊或通信迭代时，分布式实现是可以做的。**

- [Wang, Ames and Egerstedt, *Safety Barrier Certificates for Collisions-Free Multirobot Systems*](https://liwanggt.github.io/files/A2_Safe_Swarm.pdf) 从集中 QP 推到只处理邻居约束的自然去中心化，并讨论可行性、急刹和 deadlock。
- [Tan and Dimarogonas, *Distributed Implementation of Control Barrier Functions for Multi-agent Systems*](https://people.kth.se/~dimos/pdfs/DistributedCBF_LCSS_2022.pdf) 不预先分摊 coupling constraint，通过邻居辅助变量通信，在有限时间达到集中 QP 最优解，并在迭代过程中维持 CBF 条件。
- [Pereira et al., *Decentralized Safe Multi-agent Stochastic Optimal Control using Deep FBSDEs and ADMM*](https://arxiv.org/abs/2202.10658) 使用邻居控制副本、consensus ADMM 和 CADMM-OSQP 隐式层，分解随机多机器人安全 QP。
- [Wang et al., *Distributed Safe Control Design and Probabilistic Safety Verification for Multi-Agent Systems*](https://arxiv.org/abs/2303.12610) 给出合作处理 infeasibility 的分布式迭代 CBF-QP、截断实现与概率安全验证。
- [Yun et al., *Safe Multi-Agent Reinforcement Learning through Decentralized Multiple Control Barrier Functions*](https://arxiv.org/abs/2103.12553) 已将局部 CBF shields 用于 safe MARL。

所以“ADMM + Graph CBF”本身新颖性很低，而且会牺牲当前 actor 一次前向传播的执行优势。

## 4. 最推荐的具体方法：Calibrated Predictive-Residual GCBF+

### 4.1 参数化

保留环境原始 signed clearance `c_i(G)`，只学习非负的保守收缩量：

$$
h_i(G)=c_i(G)-\operatorname{softplus}(r_\theta(G)).
$$

于是天然有 `h_i <= c_i`，从而：

$$
h_i(G)\ge 0 \Longrightarrow c_i(G)\ge 0.
$$

这避免 full HJ value 完全自由漂移，也直接解释为何网络与几何约束相似：网络只负责学习“由于动力学和交互需要额外退让多少”。`r_theta` 使用 GCBF+ 同类的共享局部 GNN，并显式输入相对位置、相对速度、heading/action limit 等动力学相关量。

### 4.2 监督信号

对 replay 中长度为 `H` 的短 rollout 定义：

$$
y_{i,t}^{(H)}=\min_{0\le k\le H} c_i(G_{t+k}).
$$

训练 residual 预测 `c_i(G_t)-y_i^(H)`，重点重采样：

- `c_t > 0` 但 `y_t^(H) < 0` 的提前碰撞样本；
- 高 closing speed、有限制动能力、Dubins 不利朝向；
- 两个以上邻居夹逼、窄通道、动态图邻居进入/离开；
- density、mass、action bound、sensor noise 的 OOD 组合。

为了避免标签只代表某一固定行为策略，应混合 nominal/current policy、扰动动作和局部 adversarial candidate actions，并明确论文保证针对哪一种 action distribution 或 bounded disturbance set。

### 4.3 校准裕量

用 ensemble、lower quantile 或 split conformal 在独立 calibration set 上估计预测误差 `epsilon(G)`，训练/执行时约束离散 barrier residual：

$$
h_i(G_{t+1})-(1-\alpha)h_i(G_t) \ge \epsilon_i(G_t).
$$

论文的理论核心不应是“ensemble 更安全”，而应是：在什么误差事件、局部 Markov 假设和邻域覆盖条件下，上式推出 finite-horizon/per-step 的安全结论；对 conformal 方法必须区分 marginal coverage、trajectory-level safety 和 hard forward invariance，不能混称。

### 4.4 策略训练和部署

最简版本继续采用 GCBF+ 的 distributed actor，不做 online QP。训练时用 tightened residual loss 更新 actor；部署时每个 agent 只读取局部图并一次前向传播。

增强版本可加一个轻量 residual gate：

- `LCB residual >= 0`：直接执行 student action；
- 不通过：触发若干轮邻居通信/局部 correction；
- 仍不可行：执行显式 emergency brake/backup policy。

若引入 gate，应分别报告纯 actor 和 actor+fallback，避免把 fallback 的硬安全效果归功于 learned policy。

## 5. 第二选择：有限轮 Graph-QP amortization

如果研究目标仍要求运行时硬过滤，可把 centralized joint Graph-CBF QP 当 teacher，让每个 message-passing layer 对应一轮 primal-dual/ADMM 更新，预测 primal action、dual multiplier 和 warm start。固定 `K` 轮后检查真实 primal feasibility；只有 residual 通过才执行，否则继续迭代或回退完整 QP。

可形成的贡献是：

1. time-varying graph CBF-QP 的 permutation-equivariant amortization；
2. `K` 轮优化 residual、student approximation error 与 barrier tightening 的关系；
3. 动态邻域、通信丢包/延迟下的安全 gate；
4. agent-count/density OOD 下，与 centralized QP、ADMM、GCBF+ 的 wall-clock/safety 对比。

但风险高于预测残差路线：distributed CBF-QP 和 GNN 加速分布式优化都已存在，且系统会重新引入通信和 solver/fallback，削弱 GCBF+ 简洁的 decentralized execution 叙事。

## 6. 必须做的实验，才能证明“复杂度有价值”

### 6.1 核心对照

1. 原始 GCBF+；
2. `h=c`，直接用环境约束；
3. 一步 safety predictor（Dalal-style baseline）；
4. 当前 Deep-QP Graph-HJ；
5. `H`-step predictive residual；
6. predictive residual + calibrated tightening；
7. 可选 centralized joint QP oracle。

### 6.2 最关键的诊断指标

- 在 `c_t>0` 的状态中，预测未来 `H` 步碰撞的 AUROC/AUPRC 和 false-negative rate；
- 首次告警相对首次碰撞的提前时间；
- `h` 与 `c` 的 zero-level-set 差异，按 closing speed/heading 分层；
- agent safety rate 之外的 system failure rate、time-to-first-failure、violation 次数与最大穿透深度；
- success、travel time、deadlock、control effort；
- density、agent count、动力学参数、观测噪声、通信边变化下的 OOD 曲线；
- 每步延迟、训练环境交互数和 accelerator hours。

### 6.3 环境优先级

优先在 DoubleIntegrator 和 DubinsCar 上做，因为相对速度、制动距离和朝向会让 viability boundary 明显偏离几何 collision boundary。SingleIntegrator 上二者接近可能是问题结构导致的，不应作为“learned reachability 无效”的主要证据。

## 7. 最小可发表闭环

最稳妥的题目叙事不是“多智能体 Deep-QP”，而是类似：

> **Calibrated Predictive Graph Barrier Functions for Model-Free Distributed Multi-Agent Safety**

最低限度需要同时具备：

1. 一个结构保守的 predictive graph-barrier 参数化；
2. 一个从 rollout prediction error 到 tightened discrete-time barrier condition 的定理；
3. 一个 boundary/counterexample-focused 数据采集算法；
4. 对 GCBF+、一步 predictor 和当前 Graph-HJ 的严格消融；
5. 在 `c>0` 但动力学上危险的状态中展示稳定的提前预警收益；
6. agent-count、density 和 dynamics shift 至少两类 OOD 验证。

如果第 5 点不能成立，即 learned residual 没有比 `c` 提供显著提前量，那么最诚实也最有工程价值的结论就是：该环境族无需 HJ 复杂度，应直接采用 GCBF+。
