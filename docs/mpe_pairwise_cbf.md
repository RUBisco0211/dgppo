# MPE pair-wise CBF 与 multi-cbf 责任分配迁移

本实现独立定义 MPE 的解析二阶 HOCBF，将 Sah 和 Keshavan (2026,
arXiv:2609.14935) 的可行性证书、责任分配 LP 和局部安全 QP 迁移到二维动作。
不依赖 `gcbfplus`、manifold 或学习得到的安全函数。求解器是 CPU 参考实现，
不能放入 JAX `jit/vmap/lax.scan`；新增 `informarl_cbf` 使用 host rollout 接入训练。
原 HCBFCRPO/DGPPO 和普通 InforMARL 的目标不变。

## 文件与依赖

- `dgppo/algo/module/mpe_cbf.py`：MPE 几何、HOCBF 系数、环境接口。
- `dgppo/algo/module/cbf_allocation.py`：通用向量动作/盒约束的证书 LP、分配 LP、CVXPY/Clarabel QP。
- `scripts/eval_mpe_cbf.py`：配对评估；PD 或已有 PPO actor 作为 nominal controller。
- `tests/test_mpe_cbf.py`：导数、独立优化器、原标量实现、障碍冲突、真实环境接口检查。
- `docs/licenses/multi-cbf.txt`：上游 MIT 许可及归属。

在已有 DGPPO 环境安装可选依赖：

```bash
python -m pip install -r requirements-multi-cbf.txt
```

迁移开发使用 `/tmp/dgppo-pairwise-cbf-venv`，继承本机 `dgppo` 环境，并仅在
临时虚拟环境内安装可选求解依赖；没有向原训练环境安装这些包。
验证版本：NumPy 1.26.4、SciPy 1.15.3、CVXPY 1.6.7、Clarabel 0.11.1、JAX 0.6.2。

## CBF 定义

MPE 未裁剪的连续动力学为 `p_dot=v, v_dot=10*u`，`u in [-1,1]^2`。
每一对 `i<j` 保留一条独立约束，不对邻居取 min/max：

\[
h_{ij}=\|p_i-p_j\|^2-(2r+\varepsilon)^2,\quad
\psi_{ij}=2(p_i-p_j)^T(v_i-v_j)+\alpha_1 h_{ij}.
\]

约束 `dot(psi)+alpha2*psi >= 0` 写为：

\[
G_{ij,i}u_i+G_{ij,j}u_j+b_{ij}\ge0,
\quad G_{ij,i}=20(p_i-p_j)^T,\quad G_{ij,j}=-G_{ij,i},
\]
\[
b_{ij}=2\|v_i-v_j\|^2+
2(\alpha_1+\alpha_2)(p_i-p_j)^T(v_i-v_j)+\alpha_1\alpha_2h_{ij}.
\]

静态圆障碍的公式取 `p_j=obstacle_center, v_j=0`，安全距离改成
`agent_radius+obs_radius+epsilon`，只有当前机器人一个控制输入。
安全符号为 `h>=0`，与环境的正值碰撞 cost 相反。环境原 cost/奖励保持原样用于评估。

默认 `alpha1=alpha2=3`，取自论文作为起点，并未针对 MPE 调参。
默认安全间距 `epsilon=0`，避免额外间距把原本合法的 reset 变成初始不安全。
可用 `--safety-margin` 增大间距，但应检查 `initial_hocbf_admissible`。

## 责任分配与求解

对于 pair row，LP 同时选择 `theta in [0,1]` 和所有机器人的 witness 动作，使
以下两条约束的公共余量 `t` 最大：

\[
G_{ij,i}u_i+\theta_{ij}b_{ij}\ge t,\qquad
G_{ij,j}u_j+(1-\theta_{ij})b_{ij}\ge t.
\]

障碍行的责任固定为 1，并参与同一个 max-min LP；否则分配器看不到机器人已被障碍限制。
随后局部 QP 最小化 `||u-u_nom||² + 1e6*||slack||²`。
每个机器人的 QP 有二维物理动作，另加一维/约束的非负 slack。
CVXPY 参数化问题按形状缓存，不在每步重复构造表达式。

支持模式：

| 模式 | 行为 |
| --- | --- |
| `none` | 原 nominal controller，无滤波 |
| `uniform` | 每个机器人对各承担 0.5；障碍责任为 1 |
| `capability` | 按每个机器人在约束方向上的盒输入半宽支持值分配 |
| `certificate` | 论文的最优责任分配 LP + 局部 QP |
| `centralized` | 所有动作一起求解 QP，作为参考 |

同构 MPE 中每个机器人动作界相同，pair 的控制系数互为相反数，因此
`capability` 和 `uniform` 的份额相同，这是预期行为。
向量接口也支持每个机器人不同的上下界以及非对称盒输入。

证书 `reserve=M` 是**联合**可行性余量；`allocation_margin=t` 是**局部责任分配**余量。
`M>=0` 不保证存在 `t>=0` 的分配。负余量不等于求解器故障；LP 故障直接报错，
QP 故障返回有界 nominal 动作并设置 `solver_failed`。没有将求解失败伪装成安全。

`--witness-fallback` 仅用于 certificate 模式：当 `t < -1e-4`，执行 LP 的 witness
动作，而非局部软 QP；这仍是可能违反安全约束的降级动作。
默认关闭，与上游默认 rollout 一致。

## 每个 agent 处理多少邻居

首版默认 **所有 N-1 个其他机器人 + 全部 O 个障碍**。每个 agent 的局部约束数是
`N-1+O`，而全局唯一约束数是 `N*(N-1)/2 + N*O`。

| 机器人/障碍数量 | 单 agent 局部约束数 |
| --- | --- |
| 3 / 3 | 5 |
| 4 / 3 | 6 |
| 8 / 3 | 10 |
| 16 / 3 | 18 |

先以完整约束为基线；论文中的证书和分配只对传入的约束集合有效。
例如“最近 3 个对象”可能在拥挤场景丢掉决定可行性的第 4 个对象。
大规模实验可再对比 4/8/12 个机器人邻居预算，但不把它们当作安全上限。
若将来实现独立 top-k 选择，共享 pair 应使用双方邻居集合的并集并去重；
因此一个机器人的实际约束数可能超过其主动选择的 k。

当前仅提供 `--sensing-radius` 距离筛选作为可选实验，默认不筛选。
该参数独立于 actor 的通信半径，采用对称 pair 选择。
默认滤波器使用完整状态，不能把结果描述为仅基于局部观测或完全去中心化执行。

## 使用

PD nominal controller 的配对对照（所有模式使用相同 reset seed）：

```bash
python scripts/eval_mpe_cbf.py \
  --env MPETarget --num-agents 4 --obs 3 --episodes 5 --steps 128 \
  --output artifacts/mpe_cbf/pd_4agents_3obs.json
```

接已有 DGPPO / HCBFCRPO / InforMARL 系列的 PPO actor：

```bash
python scripts/eval_mpe_cbf.py \
  --path logs/MPETarget/dgppo/<run> --step latest \
  --episodes 5 --steps 128 \
  --output artifacts/mpe_cbf/dgppo_actor.json
```

入口只加载 actor 权重与 RNN carry，不需要加载/训练安全 critic。
未显式传入的环境、agent 数量、障碍数量和 actor 完整观测设置从训练配置读取。
默认 CPU JAX；可由环境变量显式改变平台，但 SciPy/CVXPY 仍在 CPU 上运行。

直接使用环境接口：

```python
from dgppo.algo.module.mpe_cbf import MPEPairwiseCBF

safety_filter = MPEPairwiseCBF(env, mode="certificate")
# Python loop; actor 和 env.step 可以分别 JIT，滤波器不放入 JIT。
action_safe, info = safety_filter(graph, action_nominal)
next_graph, reward, cost, done, extra = env.step(graph, action_safe)
```

只支持标准 MPE 双积分器。MPEConnectSpread 还有连通性约束，当前明确拒绝；
LidarEnv 不在这次迁移范围内。修复了 MPETarget 的一个零障碍分支错误，以支持无障碍对照。

## 指标和解释

JSON 含每种模式的逐 seed 记录与均值。重点指标：

- `safe_agent_rate`：整个初始状态和回合中，从未被环境 cost 判为碰撞的 agent 比例。
- `collision_run` / `collision_step_rate`：回合是否有碰撞 / 执行后的碰撞步骤比例。
- `reach_ever_rate` / `reach_final_rate`：曾到达 / 终点到达比例，默认使用环境距离阈值。
- `infeasible_step_rate`：局部余量低于 `-1e-4` 的比例；集中式模式使用联合余量。
- `max_constraint_violation`：实际有界动作代入约束后的最大违反量，不只看求解器 slack。
- `state_clipped_step_rate`：真实下一状态与未裁剪 Euler 预测不一致的步骤比例。
- `filter_ms_*`：含行构建、证书、分配和 QP 的 CPU 时间，初始预热不计。
- `initial_hocbf_admissible`：初始选中约束同时满足 `h>=0` 和 `psi1>=0`。

`none` 模式没有可行性 LP，相关指标是 null；无约束时无穷余量在 JSON 中也表示为 null。
如果启用感知半径，CBF 诊断只覆盖选中约束，但碰撞仍使用全环境 cost 评估。
每种模式轨迹会分叉，因此“最优分配余量不差于均分”的逐状态性质不意味着其整个回合
一定更少碰撞或更快到达，也不能直接比较不同轨迹上的余量作为逐状态证明。

这里的 soft QP、离散采样、Euler 积分和状态裁剪都不支持直接宣称连续时间严格安全。
`M>=0` 只说明当前构建的仿射约束存在解；任务到达/死锁也不由该证书保证。

## 检查

```bash
JAX_PLATFORM_NAME=cpu python -m pytest -q tests/test_mpe_cbf.py
```

包括与独立 CVXPY primal max-margin 优化的对照、与原 `multi-cbf` 标量实现的一致性、
二维 HOCBF 方向导数校验、singleton 障碍责任、空约束、求解故障和 witness fallback。
原仓库对照在没有 `multi-cbf` checkout 时跳过，其余实现不依赖该 checkout。

## 首轮迁移结果

21 项测试通过。以下每组使用相同的 5 个 reset seeds（0–4），每回合 128 步，
完整约束、增益 3/3、零额外间距、默认软 QP，不启用 witness fallback。
这些是小样本接入验证，不是论文规模的性能结论。

| nominal / 配置 | 模式 | 全程无碰撞 agent 比例 | 回合内曾到达比例 |
| --- | --- | --- | --- |
| PD，4 agents / 3 obstacles | none | 40% | 50% |
| 同上 | uniform / capability | 100% | 30% |
| 同上 | certificate | 100% | 15% |
| 同上 | centralized | 100% | 15% |
| 已有 DGPPO，3 agents / 3 obstacles | none | 100% | 100% |
| 同上 | uniform / capability | 100% | 93.3% |
| 同上 | certificate | 100% | 33.3% |
| 同上 | centralized | 100% | 93.3% |

已有 actor 来自 `logs/MPETarget/dgppo/seed0_912004304_MSTG/models/latest/actor.pkl`。
全部滤波回合没有 LP/QP 故障，负可行性余量的步骤比例也都是 0。
因此这些轨迹没有展示“消除均分不可行”的收益，而最优分配显著降低了任务到达率。
论文的 max-min 目标优先优化可行性，并不保证最小任务扰动。

另一个显著现象是：certificate 模式在 PD / DGPPO 两组中分别约 54.5% / 78.0%
的步骤触发了至少一个状态分量的环境裁剪。此时连续 CBF 模型与实际步进不完全相同，
需要在后续研究中单独处理速度/边界约束，不能将零碰撞观察升级为严格安全保证。

参考实现观测到的 certificate 总滤波耗时约为每步 11–13 ms，含诊断 LP，
不是 GPU 批量训练吞吐或严格的实时性 benchmark。

输出：

- `artifacts/mpe_cbf/pd_4agents_3obs.json`
- `artifacts/mpe_cbf/dgppo_3agents_3obs.json`

额外逐状态探针：用 NumPy `default_rng(20260929)` 独立生成 200 个状态，4 个 agent
位置与 3 个障碍中心均匀采样于 `[0,1.5]^2`，速度均匀采样于 `[-1,1]^2`，半径均为
0.05，使用默认 CBF 配置。均分有 4 个局部不可行状态，最优分配有 2 个，联合约束均可行，
没有最优余量比均分更差的状态。结果在 `artifacts/mpe_cbf/synthetic_snapshots.json`。
这些不是环境合法 reset 或证明可达的轨迹状态，只用于检验分配机制。

## InforMARL + QP 动作监督训练

训练算法：`--algo informarl_cbf`；实现位于 `dgppo/algo/informarl_cbf.py`。
独立 host collector 在每步采样后调用责任分配和 QP，策略与环境计算分别 JIT。
actor/团队 critic 的架构、GAE 和 PPO clipped objective 沿用 InforMARL。

动作数据分开保存：

- `actions` / `log_pis`：策略原始采样动作及其概率，用于 PPO ratio。
- `executed_actions`：QP 修正后的实际动作，产生环境奖励、cost 和 next state。
- `intervention_mask`：该 agent 的动作修正 L2 范数大于 `1e-4`。

actor loss 在原 clipped PPO 和熵项之外加入：

\[
L_{\mathrm{actor}}=L_{\mathrm{PPO}}-\beta H(\pi_\theta)
+\lambda_{\mathrm{CBF}}\frac{\sum_{b,t,i}m_{bti}
\operatorname{mean}_{d}(\mu_\theta(o_{bti})-
\operatorname{stopgrad}(u^{\mathrm{QP}}_{bti}))^2}
{\max(\sum_{b,t,i}m_{bti},1)}.
\]

`mu` 是 tanh 后的确定性 actor 输出，RNN 状态与本次 PPO forward 使用同一套
当前参数历史；QP target 是本次采样收集的冻结目标，不随 PPO epoch 重新求解。
仅实际被修正的 agent 参与监督，无修正时此项为 0，默认
`--cbf-action-loss-coef 1.0`。这是额外的监督优化项，不是把 QP 当成可微层。
由于目标来自随机采样后的 QP，目标也可能含探索噪声；权重需由训练表现验证。

**责任分配变量只出现在 agent–agent 行。** 3 agents 对应 3 个 pair 变量；
3 个静态障碍对应 9 条单-owner 约束，固定责任 1，同时参与 LP/QP。
每个 agent 共 2+3=5 条局部约束。

启动首轮试验：

```bash
tmux new-session -s mpe_informarl_cbf \
  'env PYTHON_BIN=/tmp/dgppo-pairwise-cbf-venv/bin/python CUDA_VISIBLE_DEVICES=1 bash scripts/run_mpe_informarl_cbf.sh'
```

脚本默认 MPETarget、3 agents、3 障碍、seed 0、8 个采样环境、128 步/回合、
1024 帧/minibatch、CBF 增益 3/3、监督权重 1。原 Trainer 的循环包含端点，
因此 `--steps 999` 从头执行 1000 次更新，总共 1,024,000 环境帧。
每 25 个 iteration 评估 4 个固定 seed 回合、保存 latest，并渲染一个评估回合。
W&B 默认 auto（本次显式 online），始终记录 `training_metrics.jsonl`；LP/QP 在 CPU，网络更新在指定 GPU。

新增指标：`policy/cbf_action_loss`、`policy/cbf_action_loss_weighted`、
`policy/cbf_supervised_fraction`；`cbf/collection/*` 与 `cbf/eval/*` 包含动作修正、
不可行率、求解失败率、状态裁剪与滤波耗时；`eval/task/*` 包含任务到达指标。

权重和优化器/RNG 使用原 `actor.pkl`、`Vl.pkl`、`algo_training_state.pkl`、
`trainer_state.pkl` 格式。恢复时使用相同环境和 CBF/监督配置，例如：

```bash
PYTHON_BIN=/tmp/dgppo-pairwise-cbf-venv/bin/python \
bash scripts/run_mpe_informarl_cbf.sh --resume-dir logs/MPETarget/informarl_cbf/<run>
```

恢复会创建新的日志 run，继续 iteration 和训练 RNG；数值求解缓存会重建，
因此不声称整个过滤轨迹逐 bit 一致。标准 `test.py` 支持该算法的 host eval，
`scripts/eval_mpe_cbf.py --path ...` 则可提取训练后的 nominal actor 比较不同分配模式。

训练接入检查包含 `tests/test_informarl_cbf.py`：原始动作概率匹配、实际执行动作
与 next state 对应、QP target 梯度为 0、监督项对 actor 梯度非零、实际 PPO 更新、
保存/加载、评估状态隔离。另增加了 QP 冷启动重试检查，总计 27 项相关测试通过。
一次完整入口短跑保存了 iteration 1，再恢复至 iteration 2；评估 GIF 为 128 帧。
在首次短跑中发现一次求解失败，现已加入保持同一目标/约束的冷启动重试，
`cbf/*/solver_retry_step_rate` 和 `solver_failed_step_rate` 分别记录重试和最终失败。

2026-09-30 首轮训练已迁入 tmux，使用物理 GPU 1，W&B 在线：

- tmux：`tmux attach -t mpe_informarl_cbf`
- run：`logs/MPETarget/informarl_cbf/seed0_930030048_ONYR`
- W&B：https://wandb.ai/1205456072-southeast-university/dgppo/runs/cgmb5cms
- 启动脚本：`artifacts/mpe_cbf/informarl_training/tmux_gpu1/launch.sh`
- 控制台：`artifacts/mpe_cbf/informarl_training/tmux_gpu1/console.log`

原非 tmux 进程已停止；从原 run `seed0_930025524_UWKC` 的 latest iteration 0
恢复权重、优化器和训练 RNG，尚未保存的少量更新重新执行。网络计算使用 GPU，
SciPy LP 和 CVXPY/Clarabel QP 使用 CPU。这是 pilot，效果需根据后续评估确认。

### 纯 pairwise CBF-QP 对照（2026-09-30）

新模式 `--pairwise-cbf-mode qp` 对全部 agent-agent 和 agent-obstacle 行建立一个
联合 soft QP，直接最小化动作偏移及 slack；不调用论文的 feasibility reserve LP 或
责任分配 LP，也没有 pair 的责任比例变量。actor 继续接受实际执行 QP 动作的监督。
启动入口为 `scripts/run_mpe_informarl_cbf_qp.sh`，实际在 tmux 会话
`mpe_informarl_cbf_qp` 内运行，物理 GPU 1 处理 JAX 网络。

日志和 DGPPO 使用同一个 Trainer 的 `counters/total_frames` 横轴、reward、cost、
unsafe、eval/video、checkpoint 和 `train/agents/*` 字段。此模式增加
`policy/ppo_loss`（不含 CBF 监督项），`policy/loss` 是实际用于更新 actor 的总损失，
`policy/cbf_action_loss_weighted` 是监督项；`train/agents/loss_objective` 对齐普通
DGPPO 的 PPO loss。比较学习曲线时需按 `counters/total_frames`，不能按 iteration：
旧 DGPPO run 每次采 16384 帧，新 pilot 每次采 1024 帧。

### 仅在 eval 使用 CBF-QP 的 InforMARL 对照

`dgppo/algo/informarl_eval_cbf.py` 继承原生 InforMARL 的 collect/update，
没有 actor 监督 loss；固定 seed 的 deterministic eval 才调用 CBF 过滤器。
`cbf` 使用 `qp`，全部约束进入一个联合 soft QP，不执行论文 LP；物理 GPU 1。
`mfcbf` 使用 `certificate`，可行性/责任分配 LP 加每 agent 局部 QP；
物理 GPU 3。

公共 W&B 字段沿用原 DGPPO Trainer，按 `counters/total_frames` 对齐；
`cbf/eval/*` 和 `eval/task/*` 仅为额外诊断。InforMARL 没有 DGPPO 的 Vh，
因此不存在 Vh 专用指标，也不制造同名占位值。测试以相同 seed 对照原生
InforMARL，确认 collect 的动作与奖励相同，PPO actor/Vl 更新指标相同，
并且 eval 不消耗训练 PRNG。

### 完整 eval-only 实验

2026-09-30 从 seed 0 启动两条完整 run：
`--steps 200000`（含端点共 200001 更新）、每更新 16384 帧、
`--eval-interval 1000 --eval-epi 1 --save-interval 1000 --n-env-test 32`
和 `video_dpi=100`。这些计数和指定的 DGPPO run
`logs/MPETarget/dgppo/seed0_912004304_MSTG` 一致，预计总采样 3,276,800,000 帧。

- `cbf`：联合 QP，W&B https://wandb.ai/1205456072-southeast-university/dgppo/runs/q92rh1yo
- `mfcbf`：责任分配，W&B https://wandb.ai/1205456072-southeast-university/dgppo/runs/141f1aig

脚本 `scripts/run_mpe_informarl_eval_cbf.sh` 现在默认完整配置；
先前的两条短 pilot run 已从 W&B 和本地删除；完整 run 从头训练。

### MPELine 完整 eval-only 对照

与已有完整 DGPPO run `logs/MPELine/dgppo/seed0_914105727_MPPJ` 对齐：
3 agents/3 障碍物，seed 0，`--steps 200000`（200001 更新），
每轮 128 环境 ×128 步，评估/保存每 1000 轮，评估 1 回合。
训练仍是原生 InforMARL；只在 eval 使用过滤器。

- 联合 QP：tmux `cbf_line`，物理 GPU 0，W&B https://wandb.ai/1205456072-southeast-university/dgppo/runs/00e1q2ql
- 论文责任分配：tmux `mfcbf_line`，物理 GPU 2，W&B https://wandb.ai/1205456072-southeast-university/dgppo/runs/ry6421e2
