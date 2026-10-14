# GRAFO reproduction status

The trainer now supports `--action-encoding discrete --reach-threshold 0.35`
as a controlled match to the cited JaxInforMARL Formation defaults. The
earlier continuous-action runs and their checkpoints remain separate. The
authors' GRAFO source does not specify these exact choices, so the new branch
must be labeled as a source-informed variant until single-task learning and
cross-task results have been validated. The GRAFO actor now uses two hidden
action layers and final distribution layers as task-specific heads; previous
runs used an additional `ScaleHid` layer in the action path.

`train_grafo_continual.py` uses this repository's JAX InforMARL actor, critic,
and PPO implementation. GRAFO tasks and sequences are transcribed from Figure
9 and the appendix of [arXiv:2610.10302](https://arxiv.org/abs/2610.10302).
The agent graph contains task-specific agent edges and one visible reference
landmark. Target offsets are hidden from the policy and are used for reward,
success checks, and evaluation rendering. Equivalent agents are assigned to
targets by minimum total distance. Collisions are disabled.
Policy node positions are expressed relative to the visible landmark; the
physics state retains world coordinates. The actor concatenates its local
observation with the four-round UniMP embedding, and the GNN has two input
feature layers as listed in the paper. The physics integrator uses the old
velocity for the position update. Agents reaching an assigned target are
snapped there and held stationary, following the cited JaxInforMARL target
environment; this particular rule is not stated in the GRAFO paper.

Training collects 128 transitions per environment and PPO update, with 32
parallel environments and 32 environment minibatches by default. Each episode
lasts 50 steps. The collector resets the environment and actor state at each
episode boundary, records `done`, and PPO uses done-masked GAE and resets the
recurrent critic state. A stage requests 5 million frames, rounded up to a
whole 32 × 128-step update. This produces 5,001,216 actual frames per stage.
Evaluation still runs independent 50-step episodes.

The feedforward actor evaluates each 16-step training chunk in parallel; a
test compares its log probabilities, entropy, and returned state with the
former sequential scan. The tanh policy's entropy sample now uses the supplied
JAX key instead of NumPy randomness at JIT trace time. This changes the
entropy regularization stream relative to runs started before the fix, so the
runner records `ppo_implementation_version` and rejects incompatible resume
checkpoints. An already running Python process retains its loaded old code.
With 32 environments, `--batch-size 128` makes 32 minibatches per PPO epoch;
`--batch-size 512` makes 8, altering the number of optimizer steps and the
paper configuration. Use a separate run for that throughput experiment.

For continual experiments, each graph gives every agent a one-hot task
identity, and the actor keeps separate output heads for previous tasks. The
task sequence transfers the shared actor and critic weights at each switch.
EWC and MAS estimate parameter importance from 64 independent 50-step
trajectories by default. Their multi variants retain one anchor and
importance matrix per completed task; online variants decay the previous
importance by 0.9 before adding the new matrix and use the latest anchor.
L2 uses a unit-weight penalty around the latest task's actor weights. All
three regularize shared actor parameters but leave the critic and task-specific
output head unregularized. FROG freezes the learned GNN from task two onward.
The runner selects the method-specific coefficients from the paper unless
`--regularization-coefficient` is explicitly provided. See the source audit
for remaining uncertainty about exact EWC/MAS score normalization.

The environment is still an **approximation of the paper's Formation
environment**. The paper says its GRAFO transitions and observation/action
spaces match Formation from Agarwal et al. (2020). We aligned the known MPE
Formation constants from the [original InforMARL Formation scenario](https://github.com/nsidn98/InforMARL)
(dt 0.1, damping 0.25, action force scale 5, initial agents in [-1, 1],
initial landmark in [-0.5, 0.5], reach threshold 0.05). The cited
[JaxInforMARL repository](https://github.com/jselvaraaj/JaxInforMARL)
contains a Target MPE environment, not the authors' GRAFO source; its
continuous-action handler is incomplete. The GRAFO paper does not report the
reward bonus magnitude or prove that its exact Formation variant used these
initialization values. We keep continuous two-axis actions, a task-edge
graph with a visible landmark, and an assumed one-time bonus of 1. The paper
does not publish an action encoding, reach radius, or bonus magnitude. These
differences must be resolved
or tested before reporting numerical agreement with the paper.

Single-task validation uses the same runner as continual learning:

```bash
bash scripts/run_grafo_single.sh GRAFO-3 \
  --seed 0 --frames-per-task 5000000 \
  --output logs/cgmarl/single_grafo3_seed0
```

Run each distinct task in TS, CS, and GFS with seeds 0, 1, and 2 before
starting a full continual comparison. For a local smoke test, pass
`--frames-per-task 128 --n-env 1 --wandb-mode disabled --no-gif`.
The paper's single-task table covers the six TS tasks with five seeds and
reports success rates near 1.0; the broader 13-task, three-seed gate here is
the agreed validation protocol and should be labeled separately in reports.

```bash
bash scripts/validate_grafo_single_tasks.sh logs/cgmarl/single_task_validation
```

Continual TS example:

```bash
conda run --no-capture-output -n dgppo python train_grafo_continual.py \
  --sequence TS --frog --regularization ewc \
  --regularization-coefficient 10000 \
  --frames-per-task 5000000 --n-env 32 --batch-size 128 \
  --wandb-mode online --output logs/cgmarl/ts_frog_ewc_seed0
```

The default W&B project is `cgmarl`. Authenticate W&B in the `dgppo` conda
environment before an online run. The runner also writes local JSONL metrics,
checkpoints, tqdm progress, eval success and returns, and optional evaluation
GIFs. `eval/<task>/success_rate` counts episodes reaching the full formation
at any step within 50 steps, matching the paper's time-limit definition;
`eval/<task>/final_success_rate` checks only the last step. Runs created
before this metric correction logged last-step success under `success_rate`.
Use `--resume <output>/checkpoints/latest` to continue a run.
