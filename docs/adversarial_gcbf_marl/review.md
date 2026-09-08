# 评审记录与处理状态：对抗性图安全值函数

> 评审对象：[theory](adversarial_gcbf_theory.md)、[training](adversarial_gcbf_training.md)。
> 初次评审：2026-09-07。本文记录讨论后的处理决定，避免继续引用已经修正的中间判断。

## 当前结论

1. 确定性系统中的精确 minimax constraint-value、Bellman 递推及鲁棒不变集论证保留；神经实现仍只称为 candidate adversarial DGCBF。
2. 局部图沿用 DGPPO/GCBF+ 的有限感知设定，不额外建模感知半径外的 agent；理论保证相应限制在该问题定义内。
3. 动态图继承是条件性结论。训练与验证必须覆盖 stable、enter、leave、activate、deactivate 五类转移。
4. 理论不引入安全折扣；实现使用独立 $\gamma_s=0.99$ 的 discounted safety target，并保留 $\gamma_s=1$ 的无折扣诊断与消融。
5. 首个实现分支已选择 DGPPO 式 robust residual update；safety-Q regularization 和执行期 override 保留为后续消融。该实现选择暂不升格为任务策略继承精确安全定理的证明。
6. 随机动力学、观测噪声与概率安全保证推迟到确定性方案通过基本验证之后。

## 对初次反馈的修正

初次反馈把“共同邻居从零门控带进入活动区”描述成动态图继承引理本身的反例。更准确的说法是：

- activate/deactivate 不改变邻域集合，属于“固定邻域转移已经满足 DCBF 条件”这一前提；
- 因此它不是 DGPPO 式继承证明遗漏的 topology-change case；
- 真正风险是训练数据可能没有覆盖这一前提，且 hard gate 会增加该类样本的学习难度；
- 所以修改重点是显式分类、定向采样、直接施加 DCBF loss 和分类型验收，而不是仅在证明中增加一句 case；
- $R-4\bar d>R_{\rm safe}$ 只作为“首次激活未碰撞”的保守诊断，不作为已经证明的 viability 条件。

## 已落实到文档的事项

### 理论文档

- 明确不考虑未观测邻居及其适用边界；
- 补全动态图反事实转移、gate 和安全选择器不变性前提；
- 详细解释动作集紧致、极值可达、有限/无限时域策略、对手信息顺序、rectangular 动作集和共享网络假设；
- 将创新点收敛为 per-ego 局部鲁棒语义、动态图训练前提及无折扣安全 backup 的组合；
- 增加固定图、动态图、对手强度、策略接口、规模泛化和随机性六步细化路线。

### 训练文档

- 使用 DGPPO 式 deterministic on-policy adversarial rollout 作为安全数据主体；
- 为五类动态图转移建立分层 batch、定向生成和独立 checkpoint 门；
- 实现中使用 $\gamma_s=0.99$；理论本身不定义折扣，无折扣参照在实现中对应 $\gamma_s=1$ 分支；
- 任务 reward discount 与 safety discount 分开配置；
- 以 DGPPO 式 robust residual PPO 作为首个可运行主分支，同一共享 critic checkpoint 上的三路消融用于决定最终方案。
- 当前代码使用环境的有限感知动态图完成核心 actor-update 验证；compact-support gate 和五类动态转移训练仍待阶段 II 实现。

## 仍需实验回答的问题

1. $\gamma_s=0.99$ 是否比 $1.0$ 更稳定，以及是否造成安全值高估或候选安全集虚增；
2. 单 adversary 与多起点动作搜索之间的最坏值 gap；
3. activate/deactivate 样本上的最坏 DCBF residual；
4. 三种策略接口的安全率、回报、干预比例和动作平滑性；
5. 共享 GNN 在不同 ego、agent 数量、密度和速度下的 residual 泛化。

## 当前最低限度表述

> 本方法学习确定性有限感知图上的 candidate adversarial DGCBF。精确无折扣 minimax value 的安全结论依赖理论文档所列动作、策略选择和动态图条件；实现中的折扣 target、有限样本和近似 adversary 不被表述为形式化全域保证。首个实现使用 DGPPO 式 robust residual PPO，其是否作为最终接口由后续消融决定。
