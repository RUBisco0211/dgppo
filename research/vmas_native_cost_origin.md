# VMAS 原生 `cost` 接口与本仓库 Navigation cost 的来源

日期：2026-08-30  
范围：VMAS 官方仓库 `proroklab/VectorizedMultiAgentSimulator`、本仓库 git 历史。本文只做来源核查，不评价当前 cost 变换是否适合具体算法。

## 结论

1. **VMAS 官方原生 scenario 接口没有 `cost()` / `get_cost()`。** 官方 `BaseScenario` 强制实现的是 `make_world`、`reset_world_at`、`observation` 和 `reward`，可选的是 `done`、`info` 等；官方环境 `step` 只收集 observation、reward、done/terminated/truncated 和 info，没有读取 `scenario.cost`。[官方 `BaseScenario.reward`](https://github.com/proroklab/VectorizedMultiAgentSimulator/blob/main/vmas/simulator/scenario.py#L449-L487) [官方环境收集 scenario 输出](https://github.com/proroklab/VectorizedMultiAgentSimulator/blob/main/vmas/simulator/environment/environment.py#L588-L637)
2. **官方 `navigation.py` 没有安全 cost。** 碰撞只表现为 reward 中的 `agent_collision_rew`，并通过 `info["agent_collisions"]` 暴露；它不是连续 signed-distance cost。[官方 Navigation reward](https://github.com/proroklab/VectorizedMultiAgentSimulator/blob/main/vmas/scenarios/navigation.py#L203-L252) [官方 Navigation info](https://github.com/proroklab/VectorizedMultiAgentSimulator/blob/main/vmas/scenarios/navigation.py#L294-L299)
3. 本仓库的 `VMASNavigation` 和 `VMASNavigationObs` 是 **VMAS 风格的 JAX/Physax 重实现**，不是 VMAS 官方 scenario 的直接镜像。它们遵循本仓库 `MultiAgentEnv` 的 `n_cost` / `get_cost` 接口；这个接口属于 DGPPO 仓库，而非原生 VMAS。[本仓库 `MultiAgentEnv`](../dgppo/env/base.py#L30-L114)
4. 两个 Navigation 环境及其连续 signed-distance cost 最初由本仓库提交 [`dca1bb9`](https://github.com/RUBisco0211/dgppo/commit/dca1bb9a28f5683e21339d8c58edfb2019c6acb3) 引入。**`[-1,1]` clipping 从首次引入时就存在。** [`VMASNavigation` 初版 cost](https://github.com/RUBisco0211/dgppo/blob/dca1bb9a28f5683e21339d8c58edfb2019c6acb3/dgppo/env/vmas/vmas_navigation.py#L143-L154) [`VMASNavigationObs` 初版 cost](https://github.com/RUBisco0211/dgppo/blob/dca1bb9a28f5683e21339d8c58edfb2019c6acb3/dgppo/env/vmas/vmas_navigation_obs.py#L75-L86)
5. **`cost <= 0` 时减 `0.5`、否则加 `0.5` 的平移不是 VMAS 官方逻辑。** 它由本仓库后续 Physax 提交 [`915a30b`](https://github.com/RUBisco0211/dgppo/commit/915a30ba2dce41522331e1489486653f67c9f615) 加入，同时新增 `cost_margin = 0.5` 和 transition cost。[该提交后的 `VMASNavigation` 实现](https://github.com/RUBisco0211/dgppo/blob/915a30ba2dce41522331e1489486653f67c9f615/dgppo/env/vmas/vmas_navigation.py#L187-L218) [该提交后的 `VMASNavigationObs` 实现](https://github.com/RUBisco0211/dgppo/blob/915a30ba2dce41522331e1489486653f67c9f615/dgppo/env/vmas/vmas_navigation_obs.py#L161-L188)

因此，不能把当前 cost 的平移或 clipping 解释为“VMAS 原生定义”；它们都是本仓库安全学习接口中的本地设计。

## 1. VMAS 官方 scenario 到底返回什么

官方 `BaseScenario` 没有 cost 抽象方法。其核心方法是：

- `observation(agent)`；
- `reward(agent)`；
- 默认 `done()`；
- 默认 `info(agent)`。

对应源码中，`reward` 是抽象方法，而 `done` 和 `info` 提供默认实现；没有 `cost` 方法。[官方 scenario 基类](https://github.com/proroklab/VectorizedMultiAgentSimulator/blob/main/vmas/simulator/scenario.py#L403-L554)

官方环境收集输出时逐 agent 调用：

```python
self.scenario.reward(agent)
self.scenario.observation(agent)
self.scenario.info(agent)
```

之后组合 done/terminated/truncated；源码中没有 `scenario.cost` 调用。[官方 `Environment.get_from_scenario`](https://github.com/proroklab/VectorizedMultiAgentSimulator/blob/main/vmas/simulator/environment/environment.py#L558-L637)

这也与官方 README 的扩展说明一致：创建 scenario 至少实现 `make_world`、`reset_world_at`、`observation`、`reward`，可选实现 `done`、`info`、`process_action`、`extra_render`，其中不包含 cost。[VMAS 官方 README：Creating a new scenario](https://github.com/proroklab/VectorizedMultiAgentSimulator#creating-a-new-scenario)

### 官方 Navigation 的碰撞信号

官方 Navigation 计算 pairwise 物理碰撞，并把固定的 `agent_collision_penalty` 加入 reward。`info()` 返回的 `agent_collisions` 也是这个 reward 分量：[官方碰撞 reward](https://github.com/proroklab/VectorizedMultiAgentSimulator/blob/main/vmas/scenarios/navigation.py#L203-L252) [官方 info](https://github.com/proroklab/VectorizedMultiAgentSimulator/blob/main/vmas/scenarios/navigation.py#L294-L299)。

它与本仓库连续 cost 的语义不同：

| 项目 | VMAS 官方 Navigation | 本仓库 Navigation |
|---|---|---|
| 对外接口 | reward + info | reward + cost + info |
| 碰撞量 | 碰撞阈值触发的 penalty | 最近实体 signed clearance |
| 正负约定 | penalty 通常为负 | unsafe 为正，safe 为负 |
| `0.5` 平移 | 无 | 有 |
| cost clipping | 无 cost | 有，裁剪至 `[-1,1]` |

## 2. 本仓库的引入历史

### 2.1 首次引入：`dca1bb9`

提交 [`dca1bb9a28f5683e21339d8c58edfb2019c6acb3`](https://github.com/RUBisco0211/dgppo/commit/dca1bb9a28f5683e21339d8c58edfb2019c6acb3) 的提交说明是 `add vmas navigation scenario, add gemsmarl style training script and logger`。它一次性新增：

- `dgppo/env/vmas/vmas_navigation.py`；
- `dgppo/env/vmas/vmas_navigation_obs.py`；
- 相应环境注册和训练入口。

初版 `VMASNavigation` 已定义：

```python
agent_cost = 2 * agent_radius - min_agent_distance
cost = agent_cost[:, None]
return jnp.clip(cost, a_min=-1.0, a_max=1.0)
```

源码证据见[初版 `VMASNavigation.get_cost`](https://github.com/RUBisco0211/dgppo/blob/dca1bb9a28f5683e21339d8c58edfb2019c6acb3/dgppo/env/vmas/vmas_navigation.py#L143-L154)。因此：

- signed-distance cost 是本仓库在适配 DGPPO `get_cost` 接口时添加的；
- clipping 也从本仓库首次引入该环境时就存在；
- 此时还没有 `0.5` 平移。

初版 `VMASNavigationObs` 再增加 agent-obstacle cost，并同样裁剪到 `[-1,1]`。[初版 `VMASNavigationObs.get_cost`](https://github.com/RUBisco0211/dgppo/blob/dca1bb9a28f5683e21339d8c58edfb2019c6acb3/dgppo/env/vmas/vmas_navigation_obs.py#L75-L86)

### 2.2 平移的引入：`915a30b`

提交 [`915a30ba2dce41522331e1489486653f67c9f615`](https://github.com/RUBisco0211/dgppo/commit/915a30ba2dce41522331e1489486653f67c9f615) 的提交说明是 `add physax implementation for navigation scenarios`。这个提交做了三件与 cost 直接相关的事：

1. 增加 `cost_margin = 0.5`；
2. 对状态 cost 和 transition cost 都加入分段平移；
3. 保留 `[-1,1]` clipping。

```python
margin = self.params["cost_margin"]
cost = jnp.where(cost <= 0.0, cost - margin, cost + margin)
cost = jnp.clip(cost, a_min=-1.0, a_max=1.0)
```

直接证据见 [`VMASNavigation`](https://github.com/RUBisco0211/dgppo/blob/915a30ba2dce41522331e1489486653f67c9f615/dgppo/env/vmas/vmas_navigation.py#L187-L218) 和 [`VMASNavigationObs`](https://github.com/RUBisco0211/dgppo/blob/915a30ba2dce41522331e1489486653f67c9f615/dgppo/env/vmas/vmas_navigation_obs.py#L161-L188)。

后续提交 [`7c06a37`](https://github.com/RUBisco0211/dgppo/commit/7c06a37f24aef8662ae6c04fd06a29081c555e52) 把环境 `step()` 返回的 cost 从 transition cost 改回当前状态的 `get_cost(graph)`，但没有创造或移除上述平移；当前文件仍保留它。[当前 `VMASNavigation`](../dgppo/env/vmas/vmas_navigation.py#L189-L206) [当前 `VMASNavigationObs`](../dgppo/env/vmas/vmas_navigation_obs.py#L150-L177)

## 3. 能否对应到某个 VMAS 官方 scenario / commit

只能做**概念级对应**，不能做逐行或 cost 级对应。

### 可以对应的部分

本仓库 `VMASNavigation` 明显沿用了官方 Navigation 的任务概念和若干命名：随机生成 agent/goal、位置与速度状态、goal progress shaping、`world_spawning_x/y`、`agent_radius`、`pos_shaping_factor`、`final_reward` 等。官方文件历史显示 Navigation scenario 至少从 2022 年存在，2024 年加入了可配置 spawning range 等功能。[官方 Navigation 文件历史](https://github.com/proroklab/VectorizedMultiAgentSimulator/commits/main/vmas/scenarios/navigation.py) [官方参数定义](https://github.com/proroklab/VectorizedMultiAgentSimulator/blob/main/vmas/scenarios/navigation.py#L29-L73)

### 不能对应的部分

- 本仓库首次引入时已经把官方 PyTorch/VMAS physics 改写成 JAX 状态转移；后续又换成自有 Physax world，因此不是官方源码某一提交的原样 port。
- 官方仓库只有 `vmas/scenarios/navigation.py`，没有 `navigation_obs.py`；官方 Navigation 的 Lidar 默认感知其他 agents，本仓库 `VMASNavigationObs` 则显式生成静态圆形 obstacles。这是本地扩展。[官方 scenario 文件](https://github.com/proroklab/VectorizedMultiAgentSimulator/tree/main/vmas/scenarios) [官方 Navigation Lidar entity filter](https://github.com/proroklab/VectorizedMultiAgentSimulator/blob/main/vmas/scenarios/navigation.py#L74-L126)
- 检查官方 `navigation.py` 当前源码及其完整文件历史，没有发现 `def cost`、`cost = where(...)` 或 cost clipping。官方文件历史中的功能提交也都围绕 Navigation 场景、Lidar、spawning range、bounds、格式化等，而不是 safety cost。[官方 Navigation 完整历史](https://github.com/proroklab/VectorizedMultiAgentSimulator/commits/main/vmas/scenarios/navigation.py)

所以最严谨的来源表述是：

> `VMASNavigation` 是参照 VMAS Navigation 任务语义编写的 DGPPO/JAX 环境；`get_cost`、signed-distance 定义、`0.5` margin 平移和 clipping 都是本仓库自己的安全学习适配，不能归因于 VMAS 官方 scenario。

## 4. 对当前代码解释的直接影响

当前变换把原始 signed clearance 的零点两侧拉开：原始安全值不大于零时再减 margin，原始不安全值大于零时再加 margin，随后饱和到 `[-1,1]`。因此它不是纯粹的单位缩放，而是人为加入了宽度为 `1.0` 的输出间隙，并截断远离边界的距离信息。

若后续要决定 HJ safety critic 应训练原始几何 margin 还是该变换后的 DGPPO cost，需要作为**本仓库算法设计选择**单独论证；不能以“保持 VMAS 原生语义”为理由保留该变换，因为 VMAS 原生并不存在这个 cost。

## 5. 一手来源清单

- [VMAS 官方仓库](https://github.com/proroklab/VectorizedMultiAgentSimulator)
- [VMAS 官方 `BaseScenario`](https://github.com/proroklab/VectorizedMultiAgentSimulator/blob/main/vmas/simulator/scenario.py)
- [VMAS 官方 Environment 输出收集](https://github.com/proroklab/VectorizedMultiAgentSimulator/blob/main/vmas/simulator/environment/environment.py#L558-L637)
- [VMAS 官方 Navigation scenario](https://github.com/proroklab/VectorizedMultiAgentSimulator/blob/main/vmas/scenarios/navigation.py)
- [VMAS 官方 Navigation git 历史](https://github.com/proroklab/VectorizedMultiAgentSimulator/commits/main/vmas/scenarios/navigation.py)
- [本仓库 Navigation 首次引入提交](https://github.com/RUBisco0211/dgppo/commit/dca1bb9a28f5683e21339d8c58edfb2019c6acb3)
- [本仓库 Physax 与 cost margin 引入提交](https://github.com/RUBisco0211/dgppo/commit/915a30ba2dce41522331e1489486653f67c9f615)

