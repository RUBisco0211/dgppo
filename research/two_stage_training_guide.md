# Graph-HJ + InforMARL 两阶段服务器训练指南

## 1. 训练流程

当前流程是正确的：

1. 使用 off-policy replay 单独训练 Graph-HJ value、局部联合动作方向导数和标量导数头。
2. 加载并冻结 Graph-HJ checkpoint，训练 InforMARL actor 和 reward critic。
3. 第二阶段使用 DGPPO 风格的逐样本混合 advantage：

$$
A_{j,t}
=
\mathbf 1_{\ell^{owner}_{j,t}=0}A^{task}_{j,t}
-
w_t\ell^{owner}_{j,t}
$$

第二阶段不会更新 HJ critic，也不会执行 runtime QP。

## 2. 一条命令运行两个阶段

`deepqp` 是 `train.py` 中的组合训练入口。LidarEnv 上直接运行：

```bash
python train.py --env LidarSpread --algo deepqp -n 3 --obs 3
```

该命令会自动完成：

1. 创建统一实验目录和 W&B run ID；
2. 运行 Stage 1 Graph-HJ off-policy 预训练；
3. 加载并冻结刚生成的 `deep_qp_safety.pkl`；
4. 运行 Stage 2 `informarl_hj_crpo`。

无需额外 shell 入口，也无需手工传递 checkpoint。`--steps` 仍表示第二阶段的 PPO 训练步数；第一阶段使用独立的 `--deep-qp-pretrain-steps`。

## 3. 需要时手工分阶段运行

正常训练不需要使用本节。只训练 HJ critic 时仍可直接调用底层入口：

```bash
python train_safety_filter.py --env LidarSpread -n 3 --obs 3 \
  --output-dir ./logs/deep_qp_safety/lidar_spread
```

已有 HJ checkpoint 时只训练 RL：

```bash
python train.py --env LidarSpread --algo informarl_hj_crpo -n 3 --obs 3 \
  --deep-qp-checkpoint ./logs/deep_qp_safety/lidar_spread/deep_qp_safety.pkl
```

## 4. 参数组织原则

以下 `train.py` 参数由组合入口同时传给两个阶段，保证 checkpoint 的网络结构和安全约束定义一致：

- `--deep-qp-gnn-layers`
- `--deep-qp-gnn-out-dim`
- `--deep-qp-hidden-dim`
- `--deep-qp-hidden-layers`
- `--deep-qp-lambda-init`
- `--deep-qp-lambda-final`
- `--deep-qp-lambda-decay-steps`
- `--deep-qp-constraint-scale`
- `--deep-qp-agent-margin`
- `--deep-qp-obstacle-margin`

第一阶段常用参数：

- `--deep-qp-pretrain-steps`
- `--deep-qp-pretrain-n-env`
- `--deep-qp-pretrain-rollout-steps`
- `--deep-qp-pretrain-updates-per-collect`
- `--deep-qp-pretrain-warmup`
- `--deep-qp-pretrain-batch-size`
- `--deep-qp-pretrain-replay-size`

第二阶段继续使用原有参数，例如 `--steps`、`--n-env-train`、`--batch-size`、`--hj-cbf-alpha`、`--hj-cbf-margin`、`--hj-cbf-eps` 和 `--cbf-weight`。组合入口没有修改其他算法的默认值；它只为 Stage 1 增加了 `--deep-qp-pretrain-*` 参数组。

## 5. 恢复说明

`--algo deepqp` 当前只负责 fresh two-stage training，不接受 `--resume-dir`。只恢复第一阶段时使用 `train_safety_filter.py --resume <checkpoint>`；只恢复第二阶段时使用 `--algo informarl_hj_crpo --resume-dir <run>`。

## 6. 输出目录

默认输出根目录为：

```text
logs/<env>/deepqp/seed<seed>_<timestamp>_<id>/
├── training_metrics.jsonl
├── wandb_run_id
├── deep-qp/
│   ├── deep_qp_safety.pkl
│   ├── deep_qp_replay.pkl
│   └── deep_qp_training_state.pkl
└── rl/<env>/informarl_hj_crpo/<run>/
    ├── config.yaml
    ├── models/latest/
    └── videos/latest_eval.mp4
```

根目录下的 `training_metrics.jsonl` 由两个阶段共同追加。组合入口生成并保存稳定的 `wandb_run_id`，两个阶段使用同一个 project、run ID 和 run name，因此页面上是一条连续的两阶段 run。

没有 FFmpeg 时视频自动降级为 GIF；渲染失败只给出警告，不会阻止 scalar 日志或 checkpoint 保存。

## 7. 日志与可视化审查结果

### 第一阶段 Graph-HJ

终端 tqdm 显示 update、总 loss、value loss、derivative loss 和 replay size。本地 JSONL 与 W&B 中的 HJ 指标全部位于 `deep-qp/*` 命名空间，例如：

- value、derivative、coefficient、scalar 分项 loss；
- derivative residual、value bound violation 和 coefficient norm；
- gradient norm、NaN 检查和 target lambda；
- replay size、constraint mean/min、unsafe sample rate；
- elapsed time 和 updates per second。

第一阶段没有策略评估视频是有意设计：它训练的是高维局部图值函数，不存在可直接渲染的执行策略。训练启动时会用独立固定 seed 采集一批不进入 replay 的 validation transitions，并周期性记录 `eval/safety/*` 的 value、derivative loss 和 residual。该验证能发现过拟合或数值退化，但它仍是经验检查，不能替代对连续状态空间前向不变性的形式化证明。

### 第二阶段 InforMARL

终端 tqdm 显示 collection episode reward、unsafe rate 和 policy loss。JSONL 与 W&B 记录：

- collection/eval episode reward 的 min、mean、max；
- 环境 cost mean/max、各 cost component 和 unsafe rate；
- actor/value loss、entropy、clip fraction 和 gradient norm；
- `hj_crpo/safe_data`、CBF weight、HJ violation、residual 和 value；
- collection、training、evaluation、checkpoint 和 iteration 耗时；
- total frames、current frames、iteration 和 update counter；
- 周期性 deterministic eval 视频。

主 W&B 横轴为 `counters/total_frames`。eval 使用独立固定 seed，不消耗训练 PRNG 流。

## 8. 开跑前检查

```bash
python train_safety_filter.py --help
python train.py --help
```

可用下面的一行命令做端到端 smoke test，而不修改任何默认参数：

```bash
python train.py --env LidarSpread --algo deepqp -n 2 --obs 1 \
  --deep-qp-pretrain-steps 1 --deep-qp-pretrain-n-env 1 \
  --deep-qp-pretrain-rollout-steps 2 \
  --deep-qp-pretrain-updates-per-collect 1 \
  --deep-qp-pretrain-warmup 1 --deep-qp-pretrain-batch-size 1 \
  --deep-qp-pretrain-replay-size 4 --deep-qp-pretrain-save-interval 1 \
  --deep-qp-pretrain-log-interval 1 --deep-qp-pretrain-eval-interval 1 \
  --deep-qp-pretrain-eval-n-env 1 --deep-qp-gnn-out-dim 8 \
  --deep-qp-hidden-dim 16 --deep-qp-hidden-layers 1 --steps 0 \
  --n-env-train 1 --n-env-test 1 --batch-size 128 \
  --no-rnn --no-video --wandb-mode disabled --log-dir /tmp/deepqp-smoke
```
