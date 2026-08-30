# DGPPO 的 constraint jump 与 clipping：论文、补充材料和官方代码核对

核对对象：DGPPO 原论文 *Discrete GCBF Proximal Policy Optimization for Multi-agent Safe Optimal Control*（[arXiv:2502.03640v3](https://arxiv.org/html/2502.03640)、[ICLR 2025 / OpenReview 论文 PDF](https://openreview.net/pdf?id=1X1R7P6yzt)）、[官方项目页](https://mit-realm.github.io/dgppo/)和 [MIT-REALM/dgppo 官方仓库](https://github.com/MIT-REALM/dgppo)。核对日期：2026-08-30。

## 结论先行

1. **论文正文和随论文发布的附录没有写出 `safe side -0.5 / unsafe side +0.5`，也没有写 constraint 被裁剪到 `[-1,1]`。** 论文给出的环境约束是未经该变换的有符号几何距离。论文里出现的 `[-1,1]` 是**动作范围**；表 1 中的 `gradient clip norm` 和 PPO `clip epsilon` 也不是 constraint clipping。
2. **±0.5 分段平移是官方代码中的实现事实。** 五处环境实现都带有 `# add margin`、`eps = 0.5` 和 `where(cost <= 0, cost - eps, cost + eps)`。
3. **“官方代码总是把 constraint clip 到 `[-1,1]`”并不成立。** LiDAR base、MPE ConnectSpread、VMAS ReverseTransport 使用双边裁剪；MPE base 和 VMAS Wheel 只做下界裁剪，没有上界裁剪。
4. **该处理从首次公开代码提交就存在，不是后来补丁。** `git blame` 将这些行全部追溯到 2025-01-22 的首次代码提交 [`ba94749` (`add code`)](https://github.com/MIT-REALM/dgppo/commit/ba9474981e137653c3bb827be5f7a0de26559069)。紧随其后的 [`9011f49`](https://github.com/MIT-REALM/dgppo/commit/9011f494f4e5af215935c8877ea77ab8d9abeb83) 主要把包名从 `cmarl` 改为 `dgppo`，没有引入这一变换。
5. **官方没有说明动机。** 唯一直接注释是 `# add margin`。把它理解成“拉开安全/不安全监督信号并限制数值尺度”是合理推断，但不是论文或代码明确给出的设计理由。

## 1. 先区分论文中的两种“cost”

论文第 3.1 节把任务目标记为联合 cost，并明确脚注说明它相当于 CMDP 的负 reward，不是 CMDP safety cost。安全性则由 avoid function / constraint function 定义：[第 3.1 节、式 (3c)](https://arxiv.org/html/2502.03640#S3.SS1)。

$$
h_i^{(m)}(o_i^k)\le 0
$$

避免集使用严格正号：

$$
\mathcal A_i=\left\{o_i\mid h_i^{(m)}(o_i)>0\right\}.
$$

官方 Python 环境接口却把安全约束数组命名为 `cost` / `costs`。因此本文所说的 “cost jump/clipping” 实际是 **constraint 的后处理**，不是论文任务 cost 的后处理。

## 2. 论文明确写了什么

### 2.1 理论定义没有 ±0.5 或裁剪

论文第 4.1 节把任意约束函数的 policy constraint-value 定义为轨迹上的最大值：[式 (7)](https://arxiv.org/html/2502.03640#S4.SS1)。

$$
V^{\zeta,\boldsymbol\mu}(\mathbf x)
=\max_{k\ge 0}\zeta(\mathbf x^k).
$$

该处没有给约束函数增加分段常数，也没有范围裁剪。Theorem 2 的表述允许任意约束函数，因此从纯定理形式看，后处理后的函数也可以作为新的约束函数；但论文没有在定理处说明实验代码采用了这种后处理。

### 2.2 附录 C.2 给出的是原始有符号几何约束

LiDAR 环境的 agent-agent 和 agent-obstacle 约束是 [附录 C.2.1，式 (103)–(104)](https://arxiv.org/html/2502.03640#A3.SS2.SSS1)：

$$
h^{(1)}(o_i)=2r-\min_{j\in\mathcal N_i}\lVert p_i-p_j\rVert,
$$

$$
h^{(2)}(o_i)=r-\min_{j\in\mathcal N_i}\lVert p_i-p_j\rVert.
$$

MuJoCo Transport 的式 (108) 仍是同一类原始距离约束：[附录 C.2.2](https://arxiv.org/html/2502.03640#A3.SS2.SSS2)。VMAS Transport2 的式 (110)–(111) 和 Wheel 的式 (113)–(114) 也都是未经 ±0.5 平移和裁剪的距离/角度约束：[Transport2](https://arxiv.org/html/2502.03640#A3.SS2.SSS3)、[Wheel](https://arxiv.org/html/2502.03640#A3.SS2.SSS3)。

例如 Transport2 写为：

$$
h^{(1)}(o_i)=2r-\min_{j\in\mathcal N_i}\lVert p_i-p_j\rVert,
$$

$$
h^{(2)}(o_i)=r_{\mathrm{obs}}-\min_{q\in\{1,2,3\}}\lVert p_{\mathrm{package}}-p_q\rVert.
$$

### 2.3 论文中的 “clip” 和 `[-1,1]` 指向别的对象

- 附录 C.2 在 LiDAR、MuJoCo 和 VMAS 环境中多次说明**控制输入**限制为 `[-1,1]`，例如 [LiDAR](https://arxiv.org/html/2502.03640#A3.SS2.SSS1) 和 [VMAS](https://arxiv.org/html/2502.03640#A3.SS2.SSS3)。这不是 constraint 范围。
- [附录 C.3 表 1](https://arxiv.org/html/2502.03640#A3.SS3) 的 `gradient clip norm = 2` 是梯度范数裁剪，`clip epsilon = 0.25` 是 PPO ratio clipping，也不是环境 constraint clipping。
- 对 v3 正文及附录全文核对 `0.5`、`eps`、`margin`、`clip`、`clipping` 和 `[-1,1]` 的相关命中后，没有找到安全侧减 0.5、不安全侧加 0.5或将约束裁剪到 `[-1,1]` 的文字、公式、表格或消融实验。

### 2.4 supplementary 的范围

arXiv v3 / ICLR 论文 PDF 本身包含附录 A–D；其中环境定义和实现超参数位于附录 C。论文 [附录 C.9](https://arxiv.org/html/2502.03640#A3.SS9) 另称 supplementary materials 中提供 `dgppo.zip`，并同时指向 GitHub 官方仓库。公开论文和附录没有解释 jump/clipping；可公开核对的官方仓库代码则明确包含它们。故不能把代码行为反向表述成“论文明确提出的公式”。

## 3. 官方代码实际做了什么

令附录公式对应的原始 signed constraint 为：

$$
c=h(o).
$$

代码先做分段平移：

$$
\widetilde c=
\begin{cases}
c-0.5,&c\le 0,\\
c+0.5,&c>0.
\end{cases}
$$

随后按环境做裁剪。当前 `main` 的逐处证据如下。

| 官方实现 | ±0.5 分段平移 | 后续裁剪 |
|---|---:|---:|
| [LiDAR base](https://github.com/MIT-REALM/dgppo/blob/main/dgppo/env/lidar_env/base.py#L180-L207) | 是 | 双边 `[-1,1]` |
| [MPE base](https://github.com/MIT-REALM/dgppo/blob/main/dgppo/env/mpe/base.py#L164-L191) | 是 | 仅下界 `-1` |
| [MPE ConnectSpread](https://github.com/MIT-REALM/dgppo/blob/main/dgppo/env/mpe/mpe_connect_spread.py#L105-L138) | 是 | 双边 `[-1,1]` |
| [VMAS ReverseTransport](https://github.com/MIT-REALM/dgppo/blob/main/dgppo/env/vmas/vmas_reverse_transport.py#L223-L249) | 是 | 双边 `[-1,1]` |
| [VMAS Wheel](https://github.com/MIT-REALM/dgppo/blob/main/dgppo/env/vmas/vmas_wheel.py#L235-L260) | 是 | 仅下界 `-1` |

ReverseTransport 还在该处理前分别给两种约束乘以 4 和 2；这个缩放同样没有出现在论文式 (110)–(111) 中。

这并非只用于日志或 safety-rate 统计。DGPPO 更新代码把 `rollout.costs` 和 `det_rollout.costs` 直接传给 constraint-value target / advantage 计算，[官方 `dgppo.py` 第 218–270 行](https://github.com/MIT-REALM/dgppo/blob/main/dgppo/algo/dgppo.py#L218-L270)。因此网络训练看到的是环境返回的**平移、并视环境而定被裁剪后的 constraint**。

## 4. Git 历史

- 官方仓库首次加入完整代码的提交是 [`ba9474981e137653c3bb827be5f7a0de26559069`](https://github.com/MIT-REALM/dgppo/commit/ba9474981e137653c3bb827be5f7a0de26559069)，时间为 2025-01-22。该提交中的旧包名路径已经包含 [`# add margin`、`eps = 0.5`、分段平移和双边裁剪](https://github.com/MIT-REALM/dgppo/blob/ba9474981e137653c3bb827be5f7a0de26559069/cmarl/env/lidar_env/base.py#L202-L205)。
- 下一提交 [`9011f494f4e5af215935c8877ea77ab8d9abeb83`](https://github.com/MIT-REALM/dgppo/commit/9011f494f4e5af215935c8877ea77ab8d9abeb83) 将 `cmarl` 重命名为 `dgppo`；相关行不是在这次重命名中新增。
- 当前代码的 `git blame` 仍将五处处理追溯到 `ba94749`。后续与环境相关的提交没有为该处理补充理论说明或更详细注释。

因此，现有公开历史最多支持：“作者从首次代码发布起就有意加入名为 margin 的实现处理”；它不支持：“论文推导要求必须这样处理”或“该数值来自某项论文消融”。

## 5. 明确事实与合理推断

### 5.1 由源码直接推出的数学性质

不考虑裁剪，零边界处存在大小为 1 的跳变：

$$
\widetilde c(0)=-0.5,
\qquad
\lim_{c\downarrow 0,\,c>0}\widetilde c(c)=+0.5.
$$

该变换保持安全/不安全的符号分类，且与论文用严格正号定义 avoid set 的边界约定一致：

$$
c\le 0\Longrightarrow \widetilde c<0,
\qquad
c>0\Longrightarrow \widetilde c>0.
$$

双边或仅下界 clipping 也不会翻转符号，但会在饱和区丢失原始几何距离的幅值信息，并产生平台区。这些是代码的直接数学后果，不是论文陈述。

### 5.2 只能作为推断的动机与影响

以下判断合理，但一手来源没有确认：

- ±0.5 可能意在给 safe / unsafe target 提供显式分类间隔，使边界附近的小数值不至于都集中在零附近。
- 双边裁剪可能意在统一 value target 的尺度、抑制离群几何距离；但 MPE base 和 Wheel 没有上界裁剪，使“统一到 `[-1,1]`”这一解释不能覆盖全部实现。
- jump 可能让安全符号更易分类，同时使边界附近的连续值回归更难；clipping 则可能让安全 critic 对远离边界的距离不敏感。论文没有报告针对这两点的消融。

## 6. 对理论阅读和复现的含义

1. **安全集合层面：** 后处理保留了约束的符号，因此没有改变除数值表示外的 safe/unsafe 划分；精确零边界在论文和代码中都属于非 avoid 侧。
2. **Theorem 2 层面：** 论文对任意约束函数定义 constraint-value，因此把后处理结果视为新的约束函数，并不表面上违反定理。但定理针对精确 policy constraint-value；神经网络近似误差和 jump/clipping 带来的优化效应不由此自动消失。
3. **实验复现层面：** 如果只照附录 C.2 的公式实现而不加代码中的后处理，训练 target 的尺度与官方实验实现不同，不能算严格复现官方代码路径。
4. **文档表述层面：** 应写成“官方实现额外加入 ±0.5 margin，并在部分环境双边裁剪”，而不是“论文定义了 ±0.5 jump 并统一 clip 到 `[-1,1]`”。
5. **若在当前 Multi-Agent Deep-QP 中借用：** 应把 raw physical constraint 与 critic training transform 分成两个显式接口，并分别记录 raw/shifted/clipped 指标；不要让一个名为 `cost` 的张量同时承担物理安全边界、学习 target 和报告指标而不注明语义。
