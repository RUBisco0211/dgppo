# GRAFO 复现状态

训练器支持 `--action-encoding discrete --reach-threshold 0.35`，用于对照论文引用的 JaxInforMARL Formation 默认设置。此前的连续动作实验及其检查点仍单独保留。由于作者未公开 GRAFO 环境源码，不能确认其使用了相同的动作编码和阈值；这一配置应标为“参考引用源码的变体”，不能直接称为论文环境的精确复现。当前 GRAFO actor 使用两层隐藏层，并将最终的动作分布层作为任务专属输出头；更早的运行还在动作路径上额外使用了 `ScaleHid` 层。

## 环境与网络

`train_grafo_continual.py` 使用本仓库的 JAX InforMARL actor、critic 和 PPO 实现。GRAFO 任务及序列依据 [arXiv:2610.10302](https://arxiv.org/abs/2610.10302) 的图 9 和附录录入。智能体图包含任务专属的智能体间边和一个可见的参考地标。策略无法直接看到目标偏移量；偏移量只用于奖励、成功判定和评估渲染。结构等价的智能体通过最小总距离分配目标。环境关闭碰撞。

策略节点的位置以可见地标为原点表示，物理状态仍保留世界坐标。actor 将局部观测与四轮 UniMP 消息传递后的嵌入拼接；GNN 的输入特征部分采用论文超参数表中的两层。物理积分先使用旧速度更新位置。智能体到达分配的目标后会被吸附到目标并保持静止，这遵循所引用的 JaxInforMARL 目标环境，但 GRAFO 论文没有明确写出该规则。

## 训练与持续学习流程

每次 PPO update 从 32 个并行环境各采集 128 个 transition；默认划分为 32 个环境 minibatch。每个 episode 最多 50 步。采集器在 episode 边界重置环境和 actor 状态并记录 `done`；PPO 的 GAE 对 `done` 做掩码，循环 critic 状态也在边界重置。每阶段请求 500 万帧，向上取整到完整的 `32 × 128` 步 update，因此实际为 5,001,216 帧。评估使用独立的 50 步 episode。

前馈 actor 对每段 16 步训练数据并行计算。测试将其动作对数概率、熵和返回状态与原先的顺序扫描逐项比较。tanh 策略估计熵时，现使用调用方传入的 JAX key，而不是在 JIT tracing 阶段调用 NumPy 随机数。这改变了相对于修复前运行的熵正则随机流；训练器记录 `ppo_implementation_version`，并拒绝不兼容检查点的恢复。已启动的 Python 进程仍使用启动时载入的旧代码。

使用 32 个环境时，`--batch-size 128` 对应每个 PPO epoch 32 个 minibatch；`--batch-size 512` 对应 8 个，因此也改变了优化器更新次数，不能当作论文原配置。吞吐量实验应使用单独的 run。

持续学习实验在每个智能体节点上提供任务身份的 one-hot 编码，并为不同任务保留独立的 actor 输出头。切换任务时，共享 actor 和 critic 参数继续训练。EWC 与 MAS 默认从 64 条独立、各最多 50 步的轨迹估计参数重要性。多任务版本为每个已完成任务保存参数锚点和重要性矩阵；在线版本将旧重要性乘以 0.9 再加上新矩阵，并使用最新锚点。L2 使用权重为 1 的惩罚，约束 actor 参数相对上一任务结束时的值。这三种方法都只正则化共享 actor 参数，不正则化 critic 或任务专属输出头。FROG 从第二个任务起冻结已经学到的 GNN。若未显式传入 `--regularization-coefficient`，训练器按方法选择论文中的系数；EWC/MAS 重要性分数的精确归一化仍见源码审计文档。

## 与论文环境的差距

当前环境仍是**论文 Formation 环境的近似实现**。论文称 GRAFO 的状态转移与观测、动作空间沿用 Agarwal 等人（2020）的 Formation。我们对齐了[原始 InforMARL Formation 场景](https://github.com/nsidn98/InforMARL)中已知的 MPE 常量：时间步长 0.1、阻尼 0.25、动作力尺度 5、智能体初始位置范围 `[-1, 1]`、地标初始位置范围 `[-0.5, 0.5]`，以及到达阈值 0.05。

论文引用的 [JaxInforMARL 仓库](https://github.com/jselvaraaj/JaxInforMARL)包含 Target MPE 环境，但不是作者的 GRAFO 源码；其中连续动作的处理尚未实现。GRAFO 论文没有报告奖励 bonus 的大小，也无法证明其 Formation 变体采用了这些精确的初始化数值。当前实现保留二维连续动作、带可见地标的任务图，并假定一次性到达奖励为 1。论文未给出动作编码、到达半径或奖励 bonus 的具体数值。报告与论文的数值一致性之前，必须先解决这些差异或做敏感性实验。

## 运行与评估

单任务验证与持续学习共用训练入口：

```bash
bash scripts/run_grafo_single.sh GRAFO-3 \
  --seed 0 --frames-per-task 5000000 \
  --output logs/cgmarl/single_grafo3_seed0
```

此前约定的单任务验收范围是 TS、CS、GFS 中所有不同任务，每项使用种子 0、1、2。局部冒烟测试可传入 `--frames-per-task 128 --n-env 1 --wandb-mode disabled --no-gif`。论文单任务表只覆盖 TS 中的六个任务，使用五个种子，报告成功率接近 1.0；这里的更广任务范围和三种子门槛应在报告中另行标注。当前六组完整序列对照只使用 seed 0，不能替代这一验收。

```bash
bash scripts/validate_grafo_single_tasks.sh logs/cgmarl/single_task_validation
```

持续学习 TS 示例：

```bash
conda run --no-capture-output -n dgppo python train_grafo_continual.py \
  --sequence TS --frog --regularization ewc \
  --regularization-coefficient 10000 \
  --frames-per-task 5000000 --n-env 32 --batch-size 128 \
  --wandb-mode online --output logs/cgmarl/ts_frog_ewc_seed0
```

默认 W&B 项目名为 `cgmarl`。在线运行前需在 `dgppo` conda 环境中登录 W&B。训练器还会在本地写入 JSONL 指标和检查点，显示 tqdm 进度，记录评估成功率与回报，并可生成评估 GIF。`eval/<task>/success_rate` 统计 50 步内曾到达完整编队的 episode 比例，与论文的时限内成功定义一致；`eval/<task>/final_success_rate` 只检查最后一步。修正该指标之前创建的 run 将“最后一步成功率”写在 `success_rate` 字段下。可使用 `--resume <output>/checkpoints/latest` 恢复训练。
