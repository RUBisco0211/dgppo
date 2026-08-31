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

训练入口与原仓库的 `train.sh` 放在同一级：

```text
train_hj_informarl.sh
```

默认配置为 LidarEnv 系列的 `LidarTarget` 场景、3 个 agent、3 个 obstacle、seed 0，并使用 W&B offline 模式：

```bash
./train_hj_informarl.sh
```

本方法的实验参数全部显式写在该文件顶部，分为以下几组：

- shared environment/logging；
- Stage 1 Graph-HJ replay、优化器和网络参数；
- 两阶段共用的 HJ 结构与连续约束参数；
- Stage 2 InforMARL/PPO 与 HJ-CBF 参数。

服务器开跑前直接编辑文件，例如把：

```bash
python_bin="/path/to/conda/env/bin/python"
wandb_mode="online"
env_id="LidarTarget"
seed=0
```

这样不会为了某一次实验而修改 `train.py` 或 `train_safety_filter.py` 原有 CLI 默认参数。

## 3. 分阶段运行

只训练 HJ critic：

```bash
./train_hj_informarl.sh hj
```

HJ checkpoint 已存在时只训练 RL：

```bash
./train_hj_informarl.sh rl
```

两个阶段分别提交时不要修改 `run_root`，第二阶段会读取 `${run_root}/deep-qp/deep_qp_safety.pkl`，并复用根目录保存的 W&B run ID。

## 4. 参数组织原则

以下变量由入口文件同时传给两个阶段，不能在中间单独修改：

- `hj_gnn_layers`
- `hj_gnn_out_dim`
- `hj_hidden_dim`
- `hj_hidden_layers`
- `hj_lambda_init`
- `hj_lambda_final`
- `hj_lambda_decay_steps`
- `hj_constraint_scale`
- `hj_agent_margin`
- `hj_obstacle_margin`

第二阶段常用参数：

- `hj_cbf_alpha`
- `hj_cbf_margin`
- `hj_cbf_eps`
- `cbf_weight`
- `rl_steps`
- `rl_n_env_train`
- `rl_batch_size`

入口显式传递这些参数，但 Python CLI 中既有参数的默认值保持不变；直接运行原来的 `train.py`、`train_safety_filter.py` 或 `train.sh` 时不会自动套用本实验配置。

## 5. 恢复说明

当前入口默认执行 fresh training；检测到同目录 HJ checkpoint 时会停止，避免误覆盖。需要恢复时直接使用 Python 入口的 `--resume` 或 `--resume-dir`，而不是改变默认参数或复用 fresh-training 脚本。

## 6. 输出目录

默认输出根目录为：

```text
logs/two_stage/<env>_n<agents>_o<obstacles>_seed<seed>/
├── console.log
├── training_metrics.jsonl
├── wandb_run_id
├── deep-qp/
└── rl/
```

第一阶段产生：

```text
deep-qp/
├── console.log
├── deep_qp_safety.pkl
├── deep_qp_replay.pkl
└── deep_qp_training_state.pkl
```

第二阶段产生：

```text
rl/
├── console.log
└── <env>/informarl_hj_crpo/<run>/
    ├── config.yaml
    ├── models/latest/
    │   ├── algo_training_state.pkl
    │   └── trainer_state.pkl
    └── videos/latest_eval.mp4
```

根目录下的 `training_metrics.jsonl` 和 `console.log` 由两个阶段共同追加。在线 W&B 模式下，脚本还会生成并保存一个稳定的 `wandb_run_id`，两个进程使用同一个 project、run ID 和 run name，因此页面上是一条连续的两阶段 run。

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
bash -n train_hj_informarl.sh
python train_safety_filter.py --help
python train.py --help
```

建议第一次运行前先在 `train_hj_informarl.sh` 顶部暂时改成小规模配置：

```bash
wandb_mode="disabled"
run_root="/tmp/hj_informarl_smoke"
hj_steps=2
hj_n_env=1
hj_rollout_steps=2
hj_updates_per_collect=1
hj_warmup=1
hj_batch_size=1
hj_replay_size=4
rl_steps=1
rl_n_env_train=1
rl_n_env_test=1
rl_batch_size=128
```

然后运行 `./train_hj_informarl.sh`。确认流程后再恢复正式参数并使用新的 `run_root`。
