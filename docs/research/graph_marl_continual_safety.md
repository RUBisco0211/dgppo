# 图 MARL 的持续学习与安全：论文调研和 DGPPO 迁移判断

调研日期：2026-10-12。本文只依据论文原文、会议页面、作者项目页和官方代码仓库。

## 结论先行

截至调研日期，没有找到一篇已经把以下四项完整结合并做系统实验的工作：

1. 用 GNN 表示多智能体局部交互；
2. 按未知任务序列持续训练，而不只是训练后零样本泛化；
3. 显式约束碰撞、cost 或控制屏障函数；
4. 同时评估旧任务的性能遗忘和安全遗忘。

用户给出的 [*Continual Graph Multi-Agent Reinforcement Learning*（CGMARL）](https://arxiv.org/abs/2610.10302) 是目前最直接的“图 MARL + 持续学习”工作，但它**不研究安全**。另一方面，[DGPPO](https://mit-realm.github.io/dgppo/)、[GCBF+](https://mit-realm.github.io/gcbfplus/) 和 [SS-MARL](https://arxiv.org/abs/2501.13727) 研究“图 MARL + 安全/可扩展性”，但都没有持续任务序列下的灾难性遗忘问题。因此，合理的研究路线是把 CGMARL 的防遗忘框架与 DGPPO 的安全 critic 分层组合，而不是声称已经存在一个可直接复现的完整算法。

最值得先做的低复杂度版本是：**策略图编码器采用 FROG 式冻结或 EWC；DGPPO 的安全网络 `Vh` 不直接照搬 FROG，而是使用旧任务安全边界样本重放和保守蒸馏；PPO 仍只使用当前任务的 on-policy rollout。**

## 1. 对 arXiv:2610.10302 的核实

### 1.1 它到底做了什么

论文题名是 [*Continual Graph Multi-Agent Reinforcement Learning*](https://arxiv.org/abs/2610.10302)。它正式定义了 CGMARL：任务序列中的每个任务由一个属性图表示，图决定智能体数量以及任务的转移和/或奖励。论文把任务变化分成三类：

- Topology Switch（TS）：节点数和边可跨任务变化；
- Connectivity Switch（CS）：节点固定、连接结构变化；
- Graph Feature Switch（GFS）：拓扑固定、节点或边特征变化。

它提出 GRAFO（Graph-based Formation）基准：智能体根据任务图到达相应编队位置；不同任务图对应不同目标结构。这里“图”不是仅用于通信的临时邻接图，而是会决定任务奖励或动力学的结构。[论文原文与附录](https://arxiv.org/pdf/2610.10302)

该论文确实属于持续学习，而不是普通的 zero-shot scalability：任务依次到达，训练新任务后还要重新评估旧任务；指标包括 Average Final Performance（AFP）、Average Forgetting（AF）和 Average Plasticity（AP）。不过它采用的是相对容易的 **task-incremental** 设置：评估时任务身份已知，并为不同任务保留独立输出 head。[论文第 6 节与附录](https://arxiv.org/pdf/2610.10302)

### 1.2 InforMARL 在其中是什么角色

[InforMARL](https://proceedings.mlr.press/v202/nayak23a.html) 本身是一个基于 GNN 信息聚合的 MAPPO/actor–critic 架构，目标是局部观测下的多智能体导航和变规模泛化；其[官方代码](https://github.com/nsidn98/InforMARL)不包含持续学习机制。

在 CGMARL 论文中，InforMARL 是**单任务图策略骨架和所有持续学习方法共同依赖的基础网络**，而不是一个已经具备持续学习能力的 baseline。作者在它上面比较：

- Fine-tuning（不做防遗忘）；
- L2 参数约束；
- EWC；
- MAS；
- 上述方法分别加或不加 FROG。

论文另外用 InforMARL 做了单任务、zero-shot transfer 和 sequential continual 三种对照，说明：它能单独学会各个 GRAFO 任务，但直接迁移到另一结构时很差；顺序微调后又会忘记旧任务。[论文第 5 节、表 1 和附录 E.1](https://arxiv.org/pdf/2610.10302)

### 1.3 FROG 的实际机制和局限

FROG（Frozen Graph Encoder）的机制非常简单：先在第一个任务上训练图编码器，然后冻结它；后续任务只更新动作预测模块，并在该模块上叠加 L2、EWC 或 MAS。这样可以消除“图表征模块参数继续变化”这一项造成的遗忘，但不能消除动作模块的遗忘。[论文第 5.2 节](https://arxiv.org/pdf/2610.10302)

需要注意三点：

1. 它的效果依赖第一个任务。论文的任务顺序实验表明，从协调要求最低的小图开始可能损害后续可塑性。
2. 它不是完全 task-free：主实验使用已知任务身份和任务专属 head。
3. 冻结 GNN 只是在参数层面阻止表征漂移，不等于证明表征足以覆盖未来图、未来动力学或未来安全边界。

### 1.4 它是否兼顾安全

没有。论文的 Dec-POMDP 目标是最大化奖励，没有 CMDP cost 预算、约束优化、CBF/GCBF 条件或安全 critic。GRAFO 的成功定义是限时到达编队；论文附录的实验超参数还明确关闭了 collision。论文没有报告碰撞率、安全率、危险漏报率或旧任务安全保持率。[论文问题定义、GRAFO 与实验附录](https://arxiv.org/pdf/2610.10302)

因此它回答的是：

> 图结构决定任务时，怎样减少策略对旧图任务的遗忘？

它没有回答：

> 持续适应新图任务时，怎样保证旧任务和新任务的安全集合都不被遗忘？

论文引用了 [JaxInforMARL](https://github.com/jselvaraaj/JaxInforMARL) 作为环境/实现基础，并说明持续学习部分适配了 [MEAL 官方仓库](https://github.com/TTomilin/MEAL)。截至调研日期，论文页面没有给出作者自己的 GRAFO/FROG 发布仓库，所以具体实现仍不能完全通过官方代码核验。

## 2. 真正的交集与相邻方向

### 2.1 图 MARL + 持续学习（有持续学习，无安全）

| 工作 | 核心内容 | 缺少什么 | 对本仓库价值 |
| --- | --- | --- | --- |
| [CGMARL / FROG](https://arxiv.org/abs/2610.10302) | 将任务序列表示成图序列；冻结首任务训练后的 GNN，配合 EWC/MAS | 没有安全约束；任务身份已知；安全网络是否可冻结未经验证 | 最直接，可用于 actor 的持续学习基线 |
| [MACPro](https://arxiv.org/abs/2305.13937) | 共享特征提取器、逐步增加任务 head；每个 agent 根据局部信息选择 head | 不是图方法，没有安全模块；动态扩 head 和任务识别更复杂 | 可用于未知任务身份，但复杂度明显高于 FROG |
| [MEAL benchmark](https://arxiv.org/abs/2506.14990) / [代码](https://github.com/TTomilin/MEAL) | 首个专门的 continual MARL benchmark；JAX/GPU 支持最多 100-task 序列，揭示短序列看不到的失败模式 | 本身不是图安全算法 | 长序列、稳定性/可塑性指标和 EWC/MAS 实现的参考来源 |
| [RPG](https://arxiv.org/abs/2507.06004) | 从动态实体观测中用 attention 提取跨任务 relation pattern；条件 hypernetwork 生成任务特定决策器，并同时正则 relation capturer 与 hypernetwork | 实验基于 QMIX、SMAC/LBF；没有显式图输入或安全约束 | 比 FROG 更有可塑性，但迁移到连续动作 PPO/DGPPO 的工作量大 |
| [Continuous Coordination / Lifelong Hanabi](https://proceedings.mlr.press/v139/nekoei21a.html) | 用不断变化的搭档评估 zero/few-shot lifelong coordination | Hanabi，不是 GNN 物理控制，也不研究碰撞安全 | 适合借鉴“对未见搭档保持协调”的评估思想 |

### 2.2 图 MARL + 开放规模或开放任务（有适应性，不一定是严格持续学习）

| 工作 | 核心内容 | 与 continual learning 的区别 | 对本仓库价值 |
| --- | --- | --- | --- |
| [InforMARL](https://proceedings.mlr.press/v202/nayak23a.html) / [代码](https://github.com/nsidn98/InforMARL) | GNN 聚合局部邻居信息，测试时泛化到不同数量的 agent/obstacle | 训练后泛化，不要求顺序学习后保留旧任务 | DGPPO 已与其结构高度接近，是基线而不是防遗忘方法 |
| [MRA: Meta Representations for Agents](https://proceedings.mlr.press/v232/zhang23a.html) | 学习跨不同 population Markov games 的共享和任务特定策略模式 | 重点是跨规模/游戏的元表征，不是安全，也不是标准 CL 训练流 | 可作为变 agent 数策略多模态化的高复杂度方案 |
| [MOHITO](https://proceedings.mlr.press/v286/anil25a.html) | 用动态三元超图表示 agent、任务和动作，处理任务随时出现/消失的开放系统 | 论文明确将 task-open 与传统顺序 lifelong setting 区分 | 如果将来任务节点动态出现有价值；对当前 MPE/Lidar 过重 |

“变智能体数量能 zero-shot 工作”不能自动等价为持续学习。只有当模型在任务 1、2、3 上依次继续更新，并在每个阶段回测所有历史任务，才能测出灾难性遗忘。

### 2.3 图 MARL + 安全（有安全，无持续学习）

| 工作 | 安全机制 | 是否持续学习 | 对 DGPPO 的关系 |
| --- | --- | --- | --- |
| [DGPPO](https://mit-realm.github.io/dgppo/) / [论文](https://arxiv.org/abs/2502.03640) / [代码](https://github.com/MIT-REALM/dgppo) | 学习离散 GCBF constraint-value，并据其 violation 选择 PPO 更新方向 | 否 | 当前仓库的直接基础 |
| [GCBF+](https://mit-realm.github.io/gcbfplus/) | 图神经 GCBF、安全策略和 QP 安全控制，用局部图支持规模泛化 | 否 | 提供图安全证书结构，但没有安全遗忘处理 |
| [SS-MARL](https://arxiv.org/abs/2501.13727) | GNN actor/critic/cost critic + constrained joint policy optimization；从小规模零样本迁移到大规模 | 否 | 说明图安全策略可跨规模，但不是顺序适应，且优化比 PPO 重 |
| [Graph-Based Safe RL with Time-Varying Topology](https://arxiv.org/abs/2609.08802) | CBLF 动作筛选层 + attention actor + GAT centralized critic | 否 | 动作筛选适合作为执行期兜底，但这是较新的预印本，不能替代持续安全训练 |

这些工作说明图安全方法可以适应每一步变化的邻接关系或测试时不同规模，但都没有回答“在任务 B 上继续训练后，任务 A 的安全函数是否失效”。

### 2.4 持续学习 + 安全（有二者，但不是深度图 MARL）

[*Safe Policy Search for Lifelong Reinforcement Learning with Sublinear Regret*](https://proceedings.mlr.press/v37/ammar15.html) 是较早的严格交集：它按顺序学习多个单智能体控制任务，通过共享线性 latent basis 转移知识，并将每个任务的已知线性策略参数约束投影进可行集合。它给出 sublinear regret，但安全约束是预先给定的参数约束，并不学习类似 `Vh` 的神经安全函数。[论文原文](https://proceedings.mlr.press/v37/ammar15.pdf)

这篇工作证明“lifelong + safety”并非空白概念，但其假设和算法不能直接迁移到 DGPPO：DGPPO 的约束集合由动态图上的近似神经安全值决定，且策略是多智能体非凸 PPO，而不是已知线性约束下的投影策略搜索。

更直接的新工作是 [*Safe Continual Reinforcement Learning in Non-stationary Environments*](https://arxiv.org/abs/2604.19737)，它明确指出当前 safe RL 和 continual RL 方法通常不能同时做到低安全代价与低遗忘，并提出两个 PPO+EWC 变体：[论文原文](https://arxiv.org/pdf/2604.19737)，[官方代码](https://github.com/MACS-Research-Lab/safe-crl)。

- **Safe EWC**：使用 `r_safe = r - beta * cost` 做 reward shaping，再对 actor 加普通 EWC。它确实显著降低实验 cost，但代价是奖励或任务成功率下降；论文在 Safe Continual World 中报告的结果尤其清楚地展示了这一 safety–performance trade-off。它保留 cost critic 只做监控，策略本身仍由 shaped reward 优化，并不等于显式满足 CMDP 约束。
- **CF-EWC（Cost-Fisher EWC）**：在 Fisher importance 中引入 cost，希望允许改变旧任务中导致高 cost 的参数、保护低 cost 行为相关参数。实验结果显示它的行为更接近 PPO/PPO+EWC：奖励和遗忘较好，但**不能稳定防止安全违反**。作者据此明确指出，“保护看起来安全的权重”本身不足以保证安全。

这项结果对 DGPPO 很重要：不能把 actor 上的 cost-weighted EWC 当成 `Vh` 或 GCBF 的替代品。Safe EWC 可以成为一个简单 baseline，但它通过 reward shaping 换安全，正好会混淆 DGPPO 原本想分开的任务最优性和安全约束；CF-EWC 则可以作为负面基线，不能作为主要安全机制。该论文也只处理单智能体、已知任务边界和 soft/期望约束，没有图 MARL 或 hard certificate 保证。

## 3. 对 DGPPO / InforMARL 的可迁移性

### 3.1 先区分三种目标

1. **规模泛化**：用 3 agents 训练，直接测试 8 agents；参数不再更新。这是 InforMARL、GCBF+、SS-MARL 已涉及的目标。
2. **在线适应**：部署到新环境后继续训练，只关心新环境。这不要求保留旧任务。
3. **持续学习**：在新环境继续训练，同时旧环境的任务性能和安全性不能明显退化。这才需要 FROG/EWC/replay 和历史任务评估矩阵。

如果用户真正需要的只是不同 agent 数、障碍数下的零样本安全泛化，就没有必要引入完整持续学习系统。只有场景/动力学/目标会依次变化且旧场景必须继续可用时，CGMARL 才是正确问题定义。

### 3.2 FROG 直接加到 InforMARL：可行性高，复杂度低到中

本仓库已经有 `GrafoContinualInforMARL`、GRAFO 任务序列、任务 head、L2/EWC/MAS 和 FROG encoder mask。因此在 InforMARL 上验证论文问题是最低风险路线。需要保持论文限定：任务身份已知、不同任务可有独立输出 head；如果取消任务身份，就变成 task-free continual MARL，需要额外的任务变化检测或 MACPro 式 head 选择器。

FROG 适合保护 actor 的结构表征，但它不能自动保护 reward critic；旧任务 actor 的 AFP/AF/AP 仍需通过完整回测验证。

### 3.3 FROG 直接加到 DGPPO：工程容易，理论上不能一刀切

DGPPO 有三类需要分别考虑的网络：actor、奖励 value `Vl` 和安全 value `Vh`。对它们使用同一个“冻结全部 GNN”的规则并不合理：

- **actor encoder**：可以先试 FROG。它主要承担结构化行为表征，冻结能减少旧策略遗忘。
- **`Vl` encoder**：奖励随任务变化最大。更适合任务 head 或 EWC，不适合假设首任务表示永远足够。
- **`Vh` encoder**：最敏感。若新任务改变动力学、感知半径、障碍类型或安全距离，冻结首任务安全 GNN 可能直接阻止它学习新危险；若完全继续微调，又可能忘记旧安全边界。

因此，FROG 对 DGPPO 的正确迁移不是“冻结整个 DGPPO”，而是**模块化稳定—可塑性分配**。

### 3.4 推荐的最小安全持续学习方案

#### 方案 A：先做可诊断基线（低复杂度）

- 构造 3–5 个顺序任务，优先保持动力学和碰撞定义相同，只改变编队目标、agent 数、障碍布局或连边结构。
- 每完成一个任务，冻结 checkpoint，并回测所有历史任务。
- 同时记录 reward retention 和 safety retention，不修改原 DGPPO 更新。

这一步先判断仓库中是否真的存在“性能遗忘”和“安全遗忘”；二者可能不同步。

#### 方案 B：actor 用 FROG/EWC，`Vh` 用安全记忆（中等复杂度，最推荐）

- actor：冻结图 encoder，或只对 encoder 做 EWC；任务输出 head 可独立保存。
- `Vl`：当前任务正常训练，必要时任务专属 head。
- `Vh`：保留一个小型分层 safety memory，每个旧任务保存：危险状态、接近 `Vh=0` 的边界状态、危险前若干步、少量正常安全状态。
- 新任务训练 `Vh` 时，把当前 deterministic rollout 与旧安全 memory 混合；对旧样本做 target/distillation 或原始安全监督。
- PPO actor 更新仍然只用当前任务新鲜的 on-policy rollout。旧任务 transition 只训练 `Vh`，不能直接混入 PPO ratio，否则破坏 on-policy 语义。

这一方案比给每个任务保存整套 DGPPO 网络更轻，也比完整动态多 head/task detector 简单。它正面处理安全函数遗忘，而不是假设策略不遗忘就等于安全不遗忘。

#### 方案 C：安全 head 或 progressive modules（高复杂度）

当不同任务连安全语义或动力学都不同，可为 `Vh` 使用共享 encoder + task-specific head，或 MACPro 式动态增加安全模块。代价是：

- 需要可靠的任务身份或在线变化检测；
- 每个新任务增加参数和存储；
- head 选错本身会产生安全风险；
- 很难声称任务未知时的统一安全保证。

只有在方案 B 明确失败且失败来自相互冲突的安全边界时，才值得采用。

#### RPG 与 DGPPO 的组合边界

[RPG](https://arxiv.org/pdf/2507.06004) 的 relation capturer 与 DGPPO 的局部图 attention 在功能上部分重叠：两者都从 ego 与周围实体的关系中提取置换友好的交互表征。RPG 真正额外提供的是两件事：

1. 用 return-aware parameter importance 保护跨任务共享的 relation pattern；
2. 用 task embedding 条件化 hypernetwork，生成适应不同任务/动作空间的决策模块，并蒸馏旧任务生成结果。

因此不建议把 RPG 整套堆到 DGPPO。最小可比较迁移应是：

- 将 RPG 的 relation regularization 只加到 actor GNN，和 FROG、EWC 做三方比较；
- 连续动作 actor 仍由 DGPPO 原来的分布头输出，不先引入生成完整 actor 参数的 hypernetwork；
- `Vh` 保持独立安全网络，用边界 replay/蒸馏，不共享 RPG 的 reward-based importance；
- 只有当任务间动作语义或 agent 类型确实变化时，再加入 task embedding + hypernetwork head。

完整 RPG 迁移复杂度高：原论文是离散动作 QMIX/TD loss，DGPPO 是连续动作 on-policy PPO；需要重新定义参数重要性、任务条件动作分布以及旧任务 head 的保持方式，而且 RPG 原文没有 cost 或安全结论。它更适合作为后续高容量 continual actor，而不是第一版安全持续 DGPPO。

#### Safe EWC / CF-EWC 与 DGPPO 的组合边界

- `DGPPO + actor EWC`：合理的普通 continual baseline，但不能宣称安全持续学习。
- `DGPPO + Safe EWC`：可作为 reward-shaping baseline；它很可能降低碰撞，同时牺牲 reach/reward，并会改变 DGPPO 的任务/安全梯度分工。
- `DGPPO + CF-EWC`：应作为消融或负面对照；原论文已经发现其安全违反仍然很高。
- `DGPPO + actor FROG/EWC + Vh boundary replay`：最推荐。actor 的性能遗忘与 `Vh` 的安全边界遗忘分别处理，且不把旧 off-policy 数据混入 PPO。
- `DGPPO + RPG actor + Vh boundary replay`：第二阶段方案；仅当 FROG 的冻结造成明显 plasticity loss 时再做。

这里的核心判断来自 Safe EWC/CF-EWC 的实验：参数重要性正则最多帮助“记住某些旧行为”，不能单独执行当前状态下的安全约束。DGPPO 的 safety critic、DCBF residual 和真实碰撞验证仍必须保留。

### 3.5 为什么不能只冻结 `Vh`

若所有任务共享完全相同的局部动力学、安全距离和 observation semantics，冻结经过充分预训练的 `Vh` 是一个合理 baseline；它相当于把安全模块视为跨任务基础模型。但 CGMARL 的 FROG 论文只证明冻结能移除由 encoder 参数变化引起的一项遗忘上界，并没有证明固定表征在新图安全边界上正确。[FROG 理论分析](https://arxiv.org/pdf/2610.10302)

在下面任一变化发生时，盲目冻结都很危险：

- 单步最大位移或动力学改变；
- 安全距离或障碍几何改变；
- 新节点/边特征训练时未出现；
- 感知范围改变导致可观测邻域语义变化；
- 新任务要求以前从未覆盖的交互密度。

此时更稳妥的策略是冻结底层通用几何层、允许上层安全层适应，并使用旧边界样本防止回退。

## 4. 建议实验设计

### 4.1 第一阶段：安全定义不变的图任务序列

目的：隔离“结构/目标变化引起的遗忘”，避免一开始同时改变安全语义。

可选任务维度：

- agent 数：3 → 5 → 7；
- 图拓扑/编队目标：line → formation → spread；
- 障碍密度或布局；
- 通信/感知邻接结构。

所有任务保持相同碰撞距离、动作边界和基础动力学。比较：

1. sequential fine-tuning；
2. FROG actor；
3. EWC actor；
4. Safe EWC 与 CF-EWC actor baseline；
5. FROG/EWC actor + `Vh` safety memory；
6. RPG-style actor regularization + `Vh` safety memory（第二阶段）；
7. 每任务独立训练（upper bound，不是 continual 方法）。

### 4.2 第二阶段：安全边界变化

只在第一阶段稳定后，再改变安全距离、动力学或感知半径。该阶段专门检验冻结 `Vh` 的局限，并比较：

- frozen `Vh`；
- full fine-tune `Vh`；
- partial freeze + replay；
- shared encoder + safety task heads。

### 4.3 必须同时报告两套 retention matrix

普通 CL 只记录任务成功率矩阵 `R[i,j]`：学完任务 `i` 后，在任务 `j` 上的成功率。安全持续学习至少还要有安全矩阵 `S[i,j]`。建议每个历史任务记录：

- episode safety rate / collision rate；
- 平均与峰值 cost；
- `Vh` 危险漏报率（真实危险但预测安全）；
- 固定边界验证集上的 `Vh` 误差或 calibration；
- DCBF residual violation rate；
- 任务成功率和 return。

由 `R` 计算 AFP/AF/AP，由 `S` 单独计算 safety-AFP 和 safety forgetting。不能用任务 success 的 AF 替代安全遗忘，因为策略回报保持不变时，`Vh` 仍可能过度估计安全区域；反过来，任务性能下降也不一定意味着更危险。

### 4.4 关键消融

- 只冻结 actor encoder vs. 只冻结 `Vh` encoder vs. 两者都冻结；
- `Vh` 无 replay vs. 均匀 replay vs. 危险/边界分层 replay；
- 已知 task ID + task head vs. 单一共享 head；
- 任务从简单到复杂 vs. 从复杂到简单。FROG 原文已经表明首任务会影响后续可塑性；安全场景更应优先用覆盖丰富的首任务预训练 encoder。

## 5. 总体判断

### 研究新颖性

“continual graph MARL with retained safety certificates”仍然有明显空位。CGMARL/FROG 解决策略图表征遗忘，DGPPO/GCBF+ 解决单任务动态图安全，但二者之间缺少对安全函数遗忘、旧安全集合保持以及 safety–plasticity trade-off 的系统研究。

### 可实现性

- InforMARL + FROG/EWC：高可行性，仓库已有主要组件。
- DGPPO actor + FROG：高可行性，但只能算策略防遗忘。
- DGPPO `Vh` + safety memory：中高可行性，是最值得新增的部分。
- 任务自动识别 + progressive safety heads：可行但复杂。
- 对近似 GNN `Vh` 给出任意未知任务序列上的形式化持续安全保证：目前不现实，应限定为 candidate certificate 并用 held-out safety tests 验证。

### 推荐论文主线

最干净的主线不是堆叠多个复杂模块，而是：

> 证明并实证区分 graph policy forgetting 与 graph safety-value forgetting；提出 policy encoder stabilization + boundary-aware safety replay，使 DGPPO 在顺序图任务中同时保持旧任务性能和旧安全边界。

这比直接把 MACPro、FROG、DGPPO、QP 和多个任务 head 全部拼在一起更容易形成可解释消融，也更贴近当前仓库已经观察到的安全函数估计误差问题。

## 参考一手来源

- [CGMARL / FROG 论文](https://arxiv.org/abs/2610.10302)；[PDF](https://arxiv.org/pdf/2610.10302)
- [InforMARL 论文（PMLR）](https://proceedings.mlr.press/v202/nayak23a.html)；[官方代码](https://github.com/nsidn98/InforMARL)
- [JaxInforMARL（CGMARL 论文引用的实现）](https://github.com/jselvaraaj/JaxInforMARL)
- [MEAL benchmark](https://openreview.net/forum?id=I3W8PynQU0)；[官方代码](https://github.com/TTomilin/MEAL)
- [MEAL arXiv](https://arxiv.org/abs/2506.14990)
- [RPG](https://arxiv.org/abs/2507.06004)
- [MACPro](https://arxiv.org/abs/2305.13937)；[IEEE DOI](https://doi.org/10.1109/TNNLS.2024.3394513)
- [Continuous Coordination / Lifelong Hanabi](https://proceedings.mlr.press/v139/nekoei21a.html)；[官方代码](https://github.com/chandar-lab/Lifelong-Hanabi)
- [MRA](https://proceedings.mlr.press/v232/zhang23a.html)
- [MOHITO](https://proceedings.mlr.press/v286/anil25a.html)
- [DGPPO 项目页](https://mit-realm.github.io/dgppo/)；[论文](https://arxiv.org/abs/2502.03640)；[官方代码](https://github.com/MIT-REALM/dgppo)
- [GCBF+ 项目页](https://mit-realm.github.io/gcbfplus/)
- [SS-MARL](https://arxiv.org/abs/2501.13727)
- [Graph-Based Safe RL with Time-Varying Topology](https://arxiv.org/abs/2609.08802)
- [Safe Policy Search for Lifelong RL（PMLR）](https://proceedings.mlr.press/v37/ammar15.html)
- [Safe Continual RL / Safe EWC / CF-EWC](https://arxiv.org/abs/2604.19737)；[官方代码](https://github.com/MACS-Research-Lab/safe-crl)
