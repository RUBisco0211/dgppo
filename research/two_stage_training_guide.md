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

脚本位置：

```text
scripts/train_hj_informarl_two_stage.sh
```

默认配置为 LidarEnv 系列的 `LidarTarget` 场景、3 个 agent、3 个 obstacle、seed 0，并使用 W&B offline 模式。也可以通过 `ENV_ID` 切换到 `LidarSpread`、`LidarLine`、`LidarBicycleTarget` 或 VMAS navigation 场景：

```bash
./scripts/train_hj_informarl_two_stage.sh
```

服务器已配置 W&B API key 时：

```bash
WANDB_MODE=online ./scripts/train_hj_informarl_two_stage.sh
```

指定 Python 或 Conda 环境：

```bash
PYTHON_BIN=/path/to/conda/env/bin/python \
WANDB_MODE=online \
./scripts/train_hj_informarl_two_stage.sh
```

## 3. 分阶段运行

只训练 HJ critic：

```bash
STAGE=hj ./scripts/train_hj_informarl_two_stage.sh
```

HJ checkpoint 已存在时只训练 RL：

```bash
STAGE=rl \
HJ_CHECKPOINT=/path/to/deep_qp_safety.pkl \
./scripts/train_hj_informarl_two_stage.sh
```

若两个阶段分两次提交到服务器，需保持同一个 `RUN_ROOT`；如果 HJ checkpoint 来自其他目录，则同时传入第一次生成的 `WANDB_RUN_ID`，才能继续写入同一条 W&B run。

## 4. 常用服务器配置

```bash
ENV_ID=LidarTarget \
NUM_AGENTS=3 \
NUM_OBS=3 \
N_RAYS=32 \
SEED=0 \
RUN_ROOT=/data/experiments/hj_informarl_seed0 \
HJ_STEPS=1000000 \
HJ_N_ENV=64 \
HJ_BATCH_SIZE=512 \
RL_STEPS=200000 \
RL_N_ENV_TRAIN=128 \
RL_BATCH_SIZE=16384 \
RL_EVAL_INTERVAL=1000 \
RL_SAVE_INTERVAL=1000 \
WANDB_MODE=online \
WANDB_PROJECT=dgppo \
./scripts/train_hj_informarl_two_stage.sh
```

以下结构参数会由脚本同时传给两个阶段，不能在中间单独修改：

- `HJ_GNN_LAYERS`
- `HJ_GNN_OUT_DIM`
- `HJ_HIDDEN_DIM`
- `HJ_HIDDEN_LAYERS`
- `HJ_CONSTRAINT_SCALE`
- `HJ_AGENT_MARGIN`
- `HJ_OBSTACLE_MARGIN`
- `HJ_BRAKING_ACCEL`

第二阶段常用参数：

- `HJ_CBF_ALPHA`
- `HJ_CBF_MARGIN`
- `HJ_CBF_EPS`
- `CBF_WEIGHT`
- `NO_CBF_SCHEDULE=1`
- `RL_USE_RNN=1`
- `RL_NO_VIDEO=1`

## 5. 恢复训练

第一阶段保存 online/target 参数、optimizer、replay、replay RNG 和采样 JAX PRNG key。使用同一个输出目录恢复：

```bash
STAGE=hj \
HJ_RESUME=1 \
RUN_ROOT=/data/experiments/hj_informarl_seed0 \
./scripts/train_hj_informarl_two_stage.sh
```

为避免误覆盖，HJ checkpoint 已存在时脚本会停止；应显式设置 `HJ_RESUME=1`，或使用新的 `RUN_ROOT` 开始另一组训练。

第二阶段恢复（`RL_STEPS` 是恢复后的目标总 iteration，不是追加量）：

```bash
STAGE=rl \
RL_RESUME_DIR=/path/to/previous/run/models/latest \
RL_STEPS=300000 \
./scripts/train_hj_informarl_two_stage.sh
```

新格式的第二阶段 checkpoint 会同时保存 actor/reward critic optimizer、算法与 rollout PRNG、NumPy shuffle 状态、更新计数器和冻结的 HJ critic，因此可以做完整状态续训。旧 checkpoint 缺少这些 sidecar 时会明确提示并降级为参数 warm-start。即使状态完整，不同进程、XLA 版本或硬件上的浮点归约仍可能造成细微数值差异，不承诺 bitwise identical。

训练正常结束后的 `latest` 标记指向最后一次更新的下一 iteration；用相同的 `RL_STEPS` 恢复会被识别为已经完成。若要继续，应把 `RL_STEPS` 设为更大的目标值。若启用了依赖总步数的 CBF schedule，延长目标总步数会重新定义 schedule 边界，因此不等价于从一开始就使用更长的目标训练。

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

根目录下的 `training_metrics.jsonl` 和 `console.log` 由两个阶段共同追加。在线 W&B 模式下，脚本还会生成并保存一个稳定的 `wandb_run_id`，两个进程用同一个 project、run ID 和 run name；因此页面上是一条连续的两阶段 run，而不是两个互不关联的实验。可通过 `WANDB_PROJECT`、`WANDB_NAME`、`WANDB_RUN_ID` 覆盖默认值。

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
bash -n scripts/train_hj_informarl_two_stage.sh
python train_safety_filter.py --help
python train.py --help
```

建议先用很小的步数做 smoke test：

```bash
RUN_ROOT=/tmp/hj_informarl_smoke \
HJ_STEPS=2 HJ_N_ENV=1 HJ_ROLLOUT_STEPS=2 \
HJ_UPDATES_PER_COLLECT=1 HJ_WARMUP=1 HJ_BATCH_SIZE=1 HJ_REPLAY_SIZE=4 \
HJ_SAVE_INTERVAL=1 HJ_LOG_INTERVAL=1 HJ_EVAL_INTERVAL=1 HJ_EVAL_N_ENV=1 \
RL_STEPS=1 RL_N_ENV_TRAIN=1 RL_N_ENV_TEST=1 RL_BATCH_SIZE=128 \
RL_EVAL_INTERVAL=1 RL_SAVE_INTERVAL=1 RL_NO_VIDEO=1 \
WANDB_MODE=disabled \
./scripts/train_hj_informarl_two_stage.sh
```
