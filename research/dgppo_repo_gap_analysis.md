# DGPPO 文献、仓库 Gap 与研究路线

> 审阅日期：2026-08-25。本文合并文献侧理论边界与本地仓库实现审阅，把 ICLR 2025 原论文、直接后续工作、代码现状和可执行研究路线放在同一条证据链中。本文是背景调研，不是当前 Deep-QP 实现规格。

## 1. 结论先行

DGPPO 最有价值的 gap 不是再加一个任务，而是下面四层之间仍有明显断层：

1. **精确理论与近似实现**：定理使用精确、无限时域的 deterministic-policy constraint value；代码使用有限 rollout、GAE、神经网络回归和单样本 surrogate，没有把逼近误差转成安全裕量。
2. **经验安全与部署安全**：策略更新鼓励 DGCBF 条件，但部署时没有逐步强制可行的 runtime shield；执行噪声、随机动力学、通信延迟和 OOD 状态不在原保证内。
3. **任意规模与真正的分布偏移**：论文的大规模实验保持训练密度；定理还要求局部密度上界和 agent-wise decoupled dynamics。高密度、异构、共同负载和接触耦合仍未被通用地解决。
4. **论文协议与当前仓库默认协议**：论文用 3 个训练 seed × 32 个测试初态；当前训练默认只评 1 条轨迹，最终测试默认 5 条，而且 `n_env_test` 实际未进入评估 rollout 数量。

我的建议是两条线并行规划、串行落地：

- **近期、最贴合现有代码**：把当前 manifold 分支改造成数学上正确的 `latent-action PPO + decentralized discrete-time shield`，再接到 DGPPO，而不是只接 InforMARL。
- **中长期、学术辨识度最高**：做 `approximation-aware / uncertainty-calibrated DGPPO`，给 critic 误差、Bellman residual 与 DGCBF margin 建立定量关系，并用反例搜索和 OOD stress test 验证。

前者容易较快形成可靠实验，后者更可能成为论文的核心理论贡献。

## 2. 当前实现到底做了什么

DGPPO 的主要数据流是：

```text
stochastic rollout ──> task value V^l / PPO advantage ──┐
                                                        ├─> actor update
deterministic rollout ──> constraint value V^h ──> CBF ─┘
```

代码证据：

- 每次 `update` 额外生成 deterministic rollout，随后同时清洗 stochastic/deterministic graph：[dgppo.py](../dgppo/algo/dgppo.py#L125)。
- `V^h` 是只用局部信息的 per-agent GNN value network：[dgppo.py](../dgppo/algo/dgppo.py#L74)。
- 安全 surrogate 是 `(Vh_next - Vh) / dt + alpha * Vh`，违反时取正部并与任务 advantage 合并：[dgppo.py](../dgppo/algo/dgppo.py#L240)。
- `V^h` target 来自 deterministic rollout 的 max-style OCP GAE：[dgppo.py](../dgppo/algo/dgppo.py#L263)、[utils.py](../dgppo/algo/utils.py#L11)。
- 执行时 `act` 直接输出 policy mode，没有在线求解或 backup controller：[informarl.py](../dgppo/algo/informarl.py#L230)。

所以它更准确的描述是：**学习一个长时域 safety critic，并用它塑造策略更新**；它不是部署时每一步都投影动作的 safety filter。

## 3. 文献侧理论与实验边界

### 3.1 精确 DGCBF 与神经网络近似之间仍有证书缺口

论文的安全结论要求精确的无限时域 deterministic-policy constraint value、DGCBF 不等式在相应状态成立，且策略真正满足该条件；实际算法使用有限 rollout、GAE、神经网络回归和单样本 surrogate。因此 learned network 本身没有有限样本、有限逼近误差下的前向不变性保证。

最值得建立的桥梁是：从 Bellman residual、critic error 和 policy update error 推导 tightened safe set 或 adaptive margin，再用 ensemble、quantile/conformal calibration、反例搜索或可验证网络给出数据依赖的误差上界。

### 3.2 “几乎处处安全”的理论前提强于训练可检查条件

DGPPO 的理论从非负 violation 的期望为零推导几乎处处满足约束，但 SGD 只能把有限 batch 上的经验均值降到小量，不能检查状态分布之外的区域。有限实验中的高 safety rate 不能直接升级为全状态域或无限时域保证。

### 3.3 近似梯度投影依赖较强的参数几何假设

安全/性能梯度混合被解释为近似投影，需要任务梯度与约束梯度在参数空间具有特定正交或对齐关系。一般共享 GNN actor 不会自动满足该结构，因此应直接报告梯度夹角、冲突率和投影误差，并与真实 constrained optimizer 或 epigraph/HJB 路线比较。

### 3.4 任意规模推广有局部密度和动力学解耦条件

任意规模定理不等于任意拥挤度、异构性或耦合系统都能迁移。它依赖局部 density 上界、共享局部结构以及 agent-wise decoupled dynamics。高密度、不同半径/质量/action limit、共同负载、接触耦合和 non-cooperative agent 都超出直接结论。

### 3.5 动态邻域条件没有被普通 GNN 架构硬性保证

邻域变化定理要求新进入或离开感知边界的节点对证书值和导数没有突变。hard radius、top-k 或普通 attention 不会自动保证边界零贡献，需要显式平滑 gate、边界 consistency loss 和拓扑切换测试。

### 3.6 原实验仍留下部署与评估 gap

- 离散决策时刻安全不自动覆盖步间碰撞、tracking error 和控制周期抖动。
- communication delay、dropout、stale edge 和异步动作没有通用保证。
- 原版缺少硬件证据；后续 sim-to-real 仍依赖任务特定 abstraction 和已知 tracking bound。
- 固定长度 rollout 没有系统覆盖稀疏奖励、长时域绕行和显式任务规划。
- 有限轨迹 agent safety rate 会隐藏系统级失败、violation 次数/幅度、time-to-first-failure 和尾部风险。

## 4. gap 与可做方向排序

| 优先级 | 方向 | 真正的研究问题 | 新颖性 | 预计难度 | 与本仓库匹配度 |
|---|---|---|---|---|---|
| A | Approximation-aware DGCBF | 有限 rollout/GAE/NN 误差下，安全集需要收缩多少？如何给 violation upper bound？ | 高 | 高 | 高 |
| A | Robust / stochastic DGCBF | 动力学、动作和观测有噪声时，如何得到 chance/CVaR/高概率前向不变性？ | 高 | 高 | 高 |
| A- | Latent-action PPO + shield | 非可逆动作投影下，怎样保持 PPO ratio 正确并真正约束执行动作？ | 中高 | 中 | 很高 |
| B+ | Sample-efficient DGPPO | 怎样去掉第二套完整 deterministic rollout，同时仍正确评估 mode-policy `V^h`？ | 高 | 高 | 高 |
| B+ | Density/coupling generalization | 怎样突破同密度、解耦动力学假设，覆盖拥挤和共同负载系统？ | 中高 | 高 | 中 |
| B | Async/communication robustness | delay、dropout、stale edge、异步动作下怎样修改 DGCBF 条件？ | 中高 | 中高 | 中高 |
| B- | Long-horizon hierarchy | 高层 waypoint/skill 与低层 DGCBF 如何组合出 safety + liveness？ | 中 | 中高 | 中 |

### 4.1 最推荐：Approximation-aware / uncertainty-calibrated DGPPO

核心不是把 critic 换成 ensemble 就结束，而是完成一条可检验的理论—算法链：

1. 假设 learned `V^h` 对真实 constraint value 的误差或 Bellman residual 有界。
2. 推导 approximate DGCBF 条件及需要的 tightened safe set / adaptive margin。
3. 用 ensemble、quantile critic、conformal calibration 或可验证网络得到数据依赖的误差上界。
4. 在 barrier boundary、稀有失败和 OOD 区域做主动反例搜索；把找到的反例回灌训练。
5. 报告带置信度的系统级 failure upper bound，而不只报告有限轨迹 mean safety rate。

最小发表闭环：一个近似 DGCBF theorem、一种 uncertainty-aware margin、一个独立 verifier/counterexample loop、以及 noise/density/dynamics-shift 下的显著结果。

### 4.2 Robust / stochastic DGPPO

原论文明确不覆盖 stochastic dynamics。一个有辨识度的版本应同时处理：

- action/actuation noise；
- observation noise、遮挡和 state-estimation error；
- dynamics parameter shift；
- graph edge dropout / delay。

可以把 `V^h` 改成 constraint-return distribution 或 upper-confidence critic，并约束 quantile/CVaR barrier residual。理论上要明确区分 finite-horizon chance safety、per-step chance constraint 和无限时域几乎必然安全；不能把有限测试中的 99.9% 直接称为 hard guarantee。

### 4.3 Sample-efficient DGPPO

论文消融已经说明直接用 stochastic rollout 学 `V^h` 会退化，因此“删掉 deterministic rollout”本身不是方案。更扎实的选项有：

- 学一个局部 dynamics/successor model，在 stochastic state 上 counterfactually rollout 当前 mode policy；
- replay deterministic-mode transitions，并做 policy-version conditioning / importance correction；
- 共享表示与环境状态，但保留两个 target head；
- 用 Def-MARL 的 epigraph objective 做主优化，DGCBF 只作 boundary-focused backup。

评价必须固定 environment interactions、wall-clock 和 accelerator hours；DGPPO 每次更新的双 rollout 不能只按 update 数与 baseline 比。

### 4.4 Density-shift、heterogeneous 与 coupled dynamics

只随机训练 agent 数量已经不够新：2026 的 DGPPO 多无人机工作已经做了 team-size/physical-parameter domain randomization 和 sim-to-real。更强的切口是：

- 固定 `N` 扫局部 density，区分“数量泛化”和“拥挤程度泛化”；
- heterogeneous radius、mass、action limit、sensor range 和 dynamics；
- coupled system graph：共同 payload、cable/contact/formation constraints；
- failure agent、non-cooperative agent 和动态障碍。

理论贡献可瞄准由 interaction/coupling graph 定义的 compositional DGCBF，而不是继续依赖 agent-wise decoupled dynamics。

## 5. 当前 manifold 分支：方向是对的，但现状不能宣称 hard safety

当前分支是 `InforMARLManifold`，不是 DGPPO：[algo/__init__.py](../dgppo/algo/__init__.py#L9)。它用已知 control-affine dynamics，把 nominal action 投影到一个带 slack 的连续时间切空间条件上：[manifold_filter.py](../dgppo/algo/module/manifold_filter.py#L146)。这正好可以发展成 runtime shield，但有六个需要先解决的问题。

### 5.1 PPO likelihood 用错了随机变量

rollout 先采 `action_ref`，投影成 `action_safe`，随后却计算并保存 `log pi(action_safe | state)`：[informarl_manifold.py](../dgppo/algo/informarl_manifold.py#L141)。投影通常是多对一、不可逆的，因此这不是执行动作分布的正确 log-density，PPO importance ratio 会有偏。

最简单且正确的改法是把 nominal action 当作 latent action：

```text
u_ref ~ pi_theta(. | o)
u_exec = Shield(o, u_ref)
environment.step(u_exec)
PPO ratio = pi_new(u_ref | o) / pi_old(u_ref | o)
```

rollout 必须同时保存 `actions_nominal` 与 `actions_executed`；策略 loss 用前者，动力学、reward、cost 与 intervention 指标用后者。这样即使 shield 非可逆，score-function policy gradient 仍对 latent policy 有定义。

### 5.2 投影后再 clip 会破坏等式/不等式

当前先解无 box constraint 的最小二乘投影，最后再 `clip_action`：[manifold_filter.py](../dgppo/algo/module/manifold_filter.py#L198)、[manifold_filter.py](../dgppo/algo/module/manifold_filter.py#L447)。clip 后一般不再位于所求流形上，因此不能给 hard safety。

动作上下界应作为同一个 QP/NLP 的约束；若不可行，要有显式 slack、infeasibility 状态和 emergency action，且单独报告 slack/intervention。

### 5.3 连续时间切空间条件与离散环境不一致

环境是 Euler 离散步进，原 DGPPO 也是 discrete barrier；当前 filter 使用 `h_dot` 风格的一阶切空间投影。大 `dt`、高相对速度或非线性 bicycle dynamics 下，一阶条件并不自动保证下一离散状态安全。

建议直接约束 `h(f_hat(x,u)) <= tightened_threshold`，或给 sampled-data/inter-sample margin；这也能与 DGPPO 的 discrete-time 论文定位一致。

### 5.4 当前 agent-agent 约束并非真正局部

filter 为每个 agent 枚举所有其他 agent，并全部标 active：[manifold_filter.py](../dgppo/algo/module/manifold_filter.py#L217)、[manifold_filter.py](../dgppo/algo/module/manifold_filter.py#L239)。这读取了 graph 中保存的全体真实状态，与 DGPPO 的 limited sensing / decentralized execution 假设不一致。

应只从本 agent 的可见 incoming edges / local observation 构造约束，并对刚进入 sensing radius、丢包和未知邻居设计 worst-case margin。

### 5.5 train/eval/test 的 filter state 不一致

训练与 Trainer 内评估维护跨步 slack state：[informarl_manifold.py](../dgppo/algo/informarl_manifold.py#L136)、[informarl_manifold.py](../dgppo/algo/informarl_manifold.py#L166)；普通 `act` 每步重新初始化 slack：[informarl_manifold.py](../dgppo/algo/informarl_manifold.py#L192)。`test.py` 只调用 `algo.act`，因此最终测试不是训练时的 stateful filter。

filter state 应进入统一的 actor state API，训练、在线评估、`test.py` 和真机部署共用一条 rollout 路径。

### 5.6 shield 与长时域策略的关系还没成为 DGPPO 贡献

当前是 InforMARL + handcrafted local shield。更有研究价值的版本是：

- DGPPO 提供长时域、避免 deadlock 的 nominal policy / learned DGCBF；
- local discrete-time shield 只在 critic uncertainty 高或 barrier margin 逼近边界时介入；
- adaptive intervention strength 来自 calibrated uncertainty；
- 比较 DGPPO、InforMARL+shield、DGPPO+shield、GCBF+/HJB-GNN/Def-MARL。

这样能回答一个清晰问题：**learned long-horizon certificate 与 model-based local correction 能否互补，而不是互相重复或让策略依赖 shield？**

## 6. 先修的评估与工程问题

这些不一定构成论文贡献，但不修会削弱所有后续结论。

### 6.1 训练期评估样本数和配置名不一致

CLI 有 `--n-env-test=32`，Trainer 也保存了它，但实际构造 `test_keys` 使用的是 `eval_epi`；默认 `eval_epi=1`：[train.py](../train.py#L258)、[trainer.py](../dgppo/trainer/trainer.py#L220)。这与 README 对 `n-env-test` 的描述和论文 32 个测试初态都不一致。

### 6.2 当前 checkpoint 与 `test.py` 默认查找不一致

Trainer 每次保存只保留 `models/latest`：[trainer.py](../dgppo/trainer/trainer.py#L172)；`test.py --step` 未提供时只查找数字目录：[test.py](../test.py#L49)。按 README 的默认命令可能找不到 checkpoint。

### 6.3 `test.py` 没有完整恢复环境/算法配置

- 创建环境时没有从 config 恢复 `n_rays`，`full_observation` 也由测试 CLI 默认 `False` 覆盖训练配置：[test.py](../test.py#L39)。
- manifold 算法没有走 stateful `eval_rollout_with_filter`。

stochastic test wrapper 的旧签名问题已经修正为 `Algorithm.step(graph, rnn_state, key)`，不再列为当前 gap。

### 6.4 末端 next-state safety 没有纳入指标

环境在当前 `graph` 上计算 cost，rollout 记录 `T` 个当前状态，但最后一次动作得到的 final `next_graph` 没进入 safety aggregation：[mpe/base.py](../dgppo/env/mpe/base.py#L137)、[trainer/utils.py](../dgppo/trainer/utils.py#L45)。最后一步导致的碰撞可能漏报。评估应对 `T+1` 个状态统一检查 constraint。

### 6.5 缺少自动化测试

仓库目前已有 Deep-QP/HJ-CRPO 单元测试，但仍缺少原始 DGPPO 和 manifold 分支的系统 regression test 与 CI。至少应补：

- DCBF/GAE target 的小型解析测试；
- graph neighborhood enter/leave 与 permutation/variable-N 测试；
- shield feasibility、action bound、one-step invariance 测试；
- save/load/resume 与 deterministic reproducibility；
- train/eval/test rollout 一致性。

现有 safety tests 和语法检查都无法覆盖上述 DGPPO 协议与数学语义问题。

## 7. 建议的最小实验闭环

### Phase 0：可信 baseline（先做）

- 复现论文核心环境，至少 5 个训练 seeds；每个 checkpoint 用 200–1000 个独立初态测试。
- 固定 environment steps，并另外报告 wall-clock；DGPPO 的 stochastic + deterministic samples 分开计数。
- 指标：system episode failure、agent safety rate、violations / million agent-steps、max/integrated penetration、near-miss、time-to-first-failure、task success、deadlock、intervention/slack。
- 压力轴：agent density、obstacle density、sensor/action noise、mass/friction、delay/dropout、`dt`、unseen team size，以及组合 OOD。
- 报告 bootstrap/Wilson confidence interval；near-100% 区域增加 rare-event / adversarial initial-state search。

### Phase 1：修正 manifold baseline

1. rollout 拆分 nominal/executed actions，修正 PPO ratio。
2. 统一 stateful filter API。
3. 在求解内加入 action bounds 与 infeasibility handling。
4. 改成 local-observation、discrete-time constraint。
5. 先比较 InforMARL、InforMARL+shield、DGPPO、DGPPO+shield。

### Phase 2：加入论文贡献

在以下两项中选一项做主贡献，避免一次把问题铺得过宽：

- **理论线**：critic/residual error bound -> tightened DGCBF -> calibrated uncertainty -> certificate/stress test。
- **系统线**：robust latent-action DGPPO shield -> noise/delay/density/coupled dynamics -> training-to-deployment safety。

## 8. 不建议单独立项的方向

- 只把 PPO 换成 SAC/TD3，没有新的 state-wise safety 机制。
- 只增加固定 penalty、Lagrange schedule 或普通 curriculum；已有 DGPPO、Def-MARL、HJB-GNN 和 HMARL-CBF 覆盖相邻问题。
- 只在更多 agent 数上测试但保持同密度；原论文已到 512-agent deployment。
- 只做 generic domain randomization；2026 的直接 DGPPO sim-to-real follow-up 已经做了 team-size/physical-parameter randomization。
- 只添加 one-step shield 并报告更高 safety rate，却不修正 PPO likelihood、box constraints、局部观测和离散时间一致性。

## 9. 推荐选题表述

如果偏理论：

> **Approximation-Aware Discrete Graph Barrier Policy Optimization for Multi-Agent Systems**：在有限样本 constraint-value approximation 下，以校准不确定度和收缩安全集给出高概率安全界。

如果偏算法/机器人：

> **Shielded DGPPO with Latent-Action Policy Optimization**：用正确的 latent-action PPO 训练分布式长时域策略，并以带 action constraints 的 discrete-time local shield 提供训练和部署期安全修正。

如果偏系统与鲁棒性：

> **Robust DGPPO under Density, Dynamics, and Communication Shift**：把执行、观测、图通信不确定度转成自适应 DGCBF margin，并在拥挤、异构和耦合多机器人系统上验证。

## 10. 一手资料与实现索引

- DGPPO：[ICLR 2025 paper](https://arxiv.org/abs/2502.03640) · [project](https://mit-realm.github.io/dgppo/) · [official code](https://github.com/MIT-REALM/dgppo)
- GCBF+：[T-RO paper](https://arxiv.org/abs/2401.14554) · [project](https://mit-realm.github.io/gcbfplus/) · [official code](https://github.com/MIT-REALM/gcbfplus)
- Def-MARL：[RSS 2025 paper](https://arxiv.org/abs/2504.15425) · [project](https://mit-realm.github.io/def-marl/) · [official code](https://github.com/MIT-REALM/def-marl)
- HJB-GNN：[paper](https://arxiv.org/abs/2506.22117) · [project](https://nus-core.github.io/assets/standalone/HJB-GNN/index.html) · [official code](https://github.com/hublan24/HJB-GNN)
- DGPPO sim-to-real 直接后续：[RA-L 2026 / arXiv paper](https://arxiv.org/abs/2607.20665)
- InforMARL：[ICML 2023 paper](https://proceedings.mlr.press/v202/nayak23a.html) · [official code](https://github.com/nsidn98/InforMARL)
- MACPO / MAPPO-Lagrangian：[paper](https://arxiv.org/abs/2110.02793) · [official code](https://github.com/chauncygu/Multi-Agent-Constrained-Policy-Optimisation)
- Scal-MAPPO-L：[NeurIPS 2024 paper](https://proceedings.neurips.cc/paper_files/paper/2024/hash/fa76985f05e0a25c66528308dda33de0-Abstract-Conference.html)
- WCMASAC：[AAAI 2026 paper](https://ojs.aaai.org/index.php/AAAI/article/download/40198/44159) · [official code](https://github.com/YeY-YYe/WCMASAC)
