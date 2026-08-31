#!/usr/bin/env bash
set -euo pipefail

# Graph-HJ + InforMARL two-stage training.
# Edit the parameters in this file for a new experiment, then run:
#   ./train_hj_informarl.sh          # both stages
#   ./train_hj_informarl.sh hj       # only pretrain the HJ critic
#   ./train_hj_informarl.sh rl       # only train InforMARL from an existing HJ checkpoint

python_bin="python"
stage="${1:-all}"

# Shared experiment settings.
env_id="LidarTarget"
num_agents=3
num_obs=3
n_rays=32
seed=0
run_root="./logs/two_stage/${env_id}_n${num_agents}_o${num_obs}_seed${seed}"

# Shared logging settings. Use "online" on a server with a configured W&B key.
wandb_mode="offline"
wandb_project="dgppo"
wandb_name="${env_id}_hj_informarl_n${num_agents}_o${num_obs}_seed${seed}"

# Stage 1: off-policy Graph-HJ critic pretraining.
hj_steps=1000000
hj_n_env=32
hj_rollout_steps=32
hj_updates_per_collect=32
hj_warmup=20000
hj_batch_size=256
hj_replay_size=1000000
hj_learning_rate=3e-4
hj_learning_rate_final=3e-6
hj_max_grad_norm=2.0
hj_target_tau=0.005
hj_lambda_init=0.1
hj_lambda_final=0.0001
hj_lambda_decay_steps=1000000
hj_save_interval=10000
hj_log_interval=100
hj_eval_interval=1000
hj_eval_n_env=8

# Graph-HJ architecture and continuous safety-constraint settings.
# These values are passed unchanged to both stages.
hj_gnn_layers=1
hj_gnn_out_dim=64
hj_hidden_dim=256
hj_hidden_layers=2
hj_constraint_scale=0.5
hj_agent_margin=0.02
hj_obstacle_margin=0.02

# Stage 2: InforMARL + DGPPO-style HJ constrained policy updates.
rl_steps=200000
rl_n_env_train=128
rl_n_env_test=32
rl_batch_size=16384
rl_eval_interval=1000
rl_eval_episodes=4
rl_save_interval=1000
rl_actor_gnn_layers=2
rl_value_gnn_layers=2
rl_learning_rate_actor=3e-4
rl_learning_rate_value=1e-3
rl_clip_eps=0.25
rl_entropy_coef=1e-2
hj_cbf_alpha=1.0
hj_cbf_margin=0.0
hj_cbf_eps=0.0
cbf_weight=1.0

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${script_dir}"
mkdir -p "${run_root}"
run_root="$(cd "${run_root}" && pwd)"
hj_dir="${run_root}/deep-qp"
rl_log_dir="${run_root}/rl"
hj_checkpoint="${hj_dir}/deep_qp_safety.pkl"
metrics_log_file="${run_root}/training_metrics.jsonl"
console_log_file="${run_root}/console.log"
mkdir -p "${hj_dir}" "${rl_log_dir}"

# Persist one W&B ID so separately submitted stages still belong to one run.
wandb_id_file="${run_root}/wandb_run_id"
if [[ -f "${wandb_id_file}" ]]; then
  wandb_run_id="$(<"${wandb_id_file}")"
else
  wandb_run_id="two_stage_${seed}_$(date +%Y%m%d%H%M%S)_$$"
  printf '%s\n' "${wandb_run_id}" > "${wandb_id_file}"
fi

train_hj() {
  if [[ -f "${hj_checkpoint}" ]]; then
    echo "Graph-HJ checkpoint already exists: ${hj_checkpoint}" >&2
    echo "Choose a new run_root for fresh training." >&2
    exit 1
  fi

  "${python_bin}" train_safety_filter.py \
    --env "${env_id}" \
    --num-agents "${num_agents}" \
    --obs "${num_obs}" \
    --n-rays "${n_rays}" \
    --seed "${seed}" \
    --steps "${hj_steps}" \
    --n-env "${hj_n_env}" \
    --rollout-steps "${hj_rollout_steps}" \
    --updates-per-collect "${hj_updates_per_collect}" \
    --warmup "${hj_warmup}" \
    --batch-size "${hj_batch_size}" \
    --replay-size "${hj_replay_size}" \
    --gnn-layers "${hj_gnn_layers}" \
    --gnn-out-dim "${hj_gnn_out_dim}" \
    --hidden-dim "${hj_hidden_dim}" \
    --hidden-layers "${hj_hidden_layers}" \
    --lr "${hj_learning_rate}" \
    --lr-final "${hj_learning_rate_final}" \
    --max-grad-norm "${hj_max_grad_norm}" \
    --tau "${hj_target_tau}" \
    --lambda-init "${hj_lambda_init}" \
    --lambda-final "${hj_lambda_final}" \
    --lambda-decay-steps "${hj_lambda_decay_steps}" \
    --constraint-scale "${hj_constraint_scale}" \
    --agent-margin "${hj_agent_margin}" \
    --obstacle-margin "${hj_obstacle_margin}" \
    --save-interval "${hj_save_interval}" \
    --log-interval "${hj_log_interval}" \
    --eval-interval "${hj_eval_interval}" \
    --eval-n-env "${hj_eval_n_env}" \
    --output-dir "${hj_dir}" \
    --log-file "${metrics_log_file}" \
    --wandb-mode "${wandb_mode}" \
    --wandb-project "${wandb_project}" \
    --wandb-name "${wandb_name}" \
    --wandb-run-id "${wandb_run_id}" \
    2>&1 | tee -a "${console_log_file}" "${hj_dir}/console.log"
}

train_rl() {
  if [[ ! -f "${hj_checkpoint}" ]]; then
    echo "Graph-HJ checkpoint not found: ${hj_checkpoint}" >&2
    exit 1
  fi

  "${python_bin}" train.py \
    --env "${env_id}" \
    --algo informarl_hj_crpo \
    --num-agents "${num_agents}" \
    --obs "${num_obs}" \
    --n-rays "${n_rays}" \
    --seed "${seed}" \
    --steps "${rl_steps}" \
    --n-env-train "${rl_n_env_train}" \
    --n-env-test "${rl_n_env_test}" \
    --batch-size "${rl_batch_size}" \
    --eval-interval "${rl_eval_interval}" \
    --eval-epi "${rl_eval_episodes}" \
    --save-interval "${rl_save_interval}" \
    --actor-gnn-layers "${rl_actor_gnn_layers}" \
    --Vl-gnn-layers "${rl_value_gnn_layers}" \
    --lr-actor "${rl_learning_rate_actor}" \
    --lr-Vl "${rl_learning_rate_value}" \
    --clip-eps "${rl_clip_eps}" \
    --coef-ent "${rl_entropy_coef}" \
    --no-rnn \
    --log-dir "${rl_log_dir}" \
    --metrics-log-file "${metrics_log_file}" \
    --deep-qp-checkpoint "${hj_checkpoint}" \
    --deep-qp-gnn-layers "${hj_gnn_layers}" \
    --deep-qp-gnn-out-dim "${hj_gnn_out_dim}" \
    --deep-qp-hidden-dim "${hj_hidden_dim}" \
    --deep-qp-hidden-layers "${hj_hidden_layers}" \
    --deep-qp-lr "${hj_learning_rate}" \
    --deep-qp-lr-final "${hj_learning_rate_final}" \
    --deep-qp-tau "${hj_target_tau}" \
    --deep-qp-lambda-init "${hj_lambda_init}" \
    --deep-qp-lambda-final "${hj_lambda_final}" \
    --deep-qp-lambda-decay-steps "${hj_lambda_decay_steps}" \
    --deep-qp-constraint-scale "${hj_constraint_scale}" \
    --deep-qp-agent-margin "${hj_agent_margin}" \
    --deep-qp-obstacle-margin "${hj_obstacle_margin}" \
    --hj-cbf-alpha "${hj_cbf_alpha}" \
    --hj-cbf-margin "${hj_cbf_margin}" \
    --hj-cbf-eps "${hj_cbf_eps}" \
    --cbf-weight "${cbf_weight}" \
    --wandb-mode "${wandb_mode}" \
    --wandb-project "${wandb_project}" \
    --wandb-name "${wandb_name}" \
    --wandb-run-id "${wandb_run_id}" \
    2>&1 | tee -a "${console_log_file}" "${rl_log_dir}/console.log"
}

case "${stage}" in
  hj)
    train_hj
    ;;
  rl)
    train_rl
    ;;
  all)
    train_hj
    train_rl
    ;;
  *)
    echo "Usage: $0 [hj|rl|all]" >&2
    exit 2
    ;;
esac

echo "> Two-stage training completed: ${run_root}"
