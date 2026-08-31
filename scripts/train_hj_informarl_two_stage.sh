#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
cd "${repo_root}"

python_bin="${PYTHON_BIN:-python}"
stage="${STAGE:-all}"
env_id="${ENV_ID:-LidarTarget}"
num_agents="${NUM_AGENTS:-3}"
num_obs="${NUM_OBS:-3}"
n_rays="${N_RAYS:-32}"
seed="${SEED:-0}"
wandb_mode="${WANDB_MODE:-offline}"
wandb_project="${WANDB_PROJECT:-dgppo}"

run_root="${RUN_ROOT:-./logs/two_stage/${env_id}_n${num_agents}_o${num_obs}_seed${seed}}"
mkdir -p "${run_root}"
run_root="$(cd "${run_root}" && pwd)"
hj_dir="${HJ_DIR:-${run_root}/deep-qp}"
rl_log_dir="${RL_LOG_DIR:-${run_root}/rl}"
mkdir -p "${hj_dir}" "${rl_log_dir}"
hj_dir="$(cd "${hj_dir}" && pwd)"
rl_log_dir="$(cd "${rl_log_dir}" && pwd)"
hj_checkpoint="${HJ_CHECKPOINT:-${hj_dir}/deep_qp_safety.pkl}"
metrics_log_file="${METRICS_LOG_FILE:-${run_root}/training_metrics.jsonl}"
if [[ "${metrics_log_file}" != /* ]]; then
  metrics_log_file="${repo_root}/${metrics_log_file#./}"
fi
console_log_file="${run_root}/console.log"

wandb_id_file="${run_root}/wandb_run_id"
if [[ -n "${WANDB_RUN_ID:-}" ]]; then
  wandb_run_id="${WANDB_RUN_ID}"
elif [[ -f "${wandb_id_file}" ]]; then
  wandb_run_id="$(<"${wandb_id_file}")"
else
  wandb_run_id="two_stage_${seed}_$(date +%Y%m%d%H%M%S)_$$"
  printf '%s\n' "${wandb_run_id}" > "${wandb_id_file}"
fi
wandb_name="${WANDB_NAME:-${env_id}_hj_informarl_n${num_agents}_o${num_obs}_seed${seed}}"

gnn_layers="${HJ_GNN_LAYERS:-1}"
gnn_out_dim="${HJ_GNN_OUT_DIM:-64}"
hidden_dim="${HJ_HIDDEN_DIM:-256}"
hidden_layers="${HJ_HIDDEN_LAYERS:-2}"
constraint_scale="${HJ_CONSTRAINT_SCALE:-0.5}"
agent_margin="${HJ_AGENT_MARGIN:-0.02}"
obstacle_margin="${HJ_OBSTACLE_MARGIN:-0.02}"

run_hj() {
  if [[ "${HJ_RESUME:-0}" == "1" && ! -f "${hj_checkpoint}" ]]; then
    echo "HJ_RESUME=1 but checkpoint was not found: ${hj_checkpoint}" >&2
    exit 1
  fi
  if [[ -f "${hj_checkpoint}" && "${HJ_RESUME:-0}" != "1" ]]; then
    echo "Graph-HJ checkpoint already exists: ${hj_checkpoint}" >&2
    echo "Set HJ_RESUME=1 to continue or choose a new RUN_ROOT to start over." >&2
    exit 1
  fi

  local -a command=(
    "${python_bin}" train_safety_filter.py
    --env "${env_id}"
    -n "${num_agents}"
    --obs "${num_obs}"
    --n-rays "${n_rays}"
    --seed "${seed}"
    --steps "${HJ_STEPS:-1000000}"
    --n-env "${HJ_N_ENV:-32}"
    --rollout-steps "${HJ_ROLLOUT_STEPS:-32}"
    --updates-per-collect "${HJ_UPDATES_PER_COLLECT:-32}"
    --warmup "${HJ_WARMUP:-20000}"
    --batch-size "${HJ_BATCH_SIZE:-256}"
    --replay-size "${HJ_REPLAY_SIZE:-1000000}"
    --gnn-layers "${gnn_layers}"
    --gnn-out-dim "${gnn_out_dim}"
    --hidden-dim "${hidden_dim}"
    --hidden-layers "${hidden_layers}"
    --constraint-scale "${constraint_scale}"
    --agent-margin "${agent_margin}"
    --obstacle-margin "${obstacle_margin}"
    --save-interval "${HJ_SAVE_INTERVAL:-10000}"
    --log-interval "${HJ_LOG_INTERVAL:-100}"
    --eval-interval "${HJ_EVAL_INTERVAL:-1000}"
    --eval-n-env "${HJ_EVAL_N_ENV:-8}"
    --output-dir "${hj_dir}"
    --log-file "${metrics_log_file}"
    --wandb-mode "${wandb_mode}"
    --wandb-project "${wandb_project}"
    --wandb-name "${wandb_name}"
    --wandb-run-id "${wandb_run_id}"
  )

  if [[ "${FULL_OBSERVATION:-0}" == "1" ]]; then
    command+=(--full-observation)
  fi
  if [[ -n "${HJ_BRAKING_ACCEL:-}" ]]; then
    command+=(--braking-accel "${HJ_BRAKING_ACCEL}")
  fi
  if [[ "${HJ_RESUME:-0}" == "1" && -f "${hj_checkpoint}" ]]; then
    command+=(--resume "${hj_checkpoint}")
  fi

  echo "> Stage 1/2: training Graph-HJ critic"
  "${command[@]}" 2>&1 | tee -a "${console_log_file}" "${hj_dir}/console.log"
}

run_rl() {
  if [[ ! -f "${hj_checkpoint}" ]]; then
    echo "Graph-HJ checkpoint not found: ${hj_checkpoint}" >&2
    exit 1
  fi

  local -a command=(
    "${python_bin}" train.py
    --env "${env_id}"
    --algo informarl_hj_crpo
    -n "${num_agents}"
    --obs "${num_obs}"
    --n-rays "${n_rays}"
    --seed "${seed}"
    --steps "${RL_STEPS:-200000}"
    --n-env-train "${RL_N_ENV_TRAIN:-128}"
    --n-env-test "${RL_N_ENV_TEST:-32}"
    --batch-size "${RL_BATCH_SIZE:-16384}"
    --eval-interval "${RL_EVAL_INTERVAL:-1000}"
    --eval-epi "${RL_EVAL_EPISODES:-4}"
    --save-interval "${RL_SAVE_INTERVAL:-1000}"
    --log-dir "${rl_log_dir}"
    --name "${RUN_NAME:-graph_hj_dgppo_mix}"
    --metrics-log-file "${metrics_log_file}"
    --deep-qp-checkpoint "${hj_checkpoint}"
    --deep-qp-gnn-layers "${gnn_layers}"
    --deep-qp-gnn-out-dim "${gnn_out_dim}"
    --deep-qp-hidden-dim "${hidden_dim}"
    --deep-qp-hidden-layers "${hidden_layers}"
    --deep-qp-constraint-scale "${constraint_scale}"
    --deep-qp-agent-margin "${agent_margin}"
    --deep-qp-obstacle-margin "${obstacle_margin}"
    --hj-cbf-alpha "${HJ_CBF_ALPHA:-1.0}"
    --hj-cbf-margin "${HJ_CBF_MARGIN:-0.0}"
    --hj-cbf-eps "${HJ_CBF_EPS:-0.0}"
    --cbf-weight "${CBF_WEIGHT:-1.0}"
    --wandb-mode "${wandb_mode}"
    --wandb-project "${wandb_project}"
    --wandb-name "${wandb_name}"
    --wandb-run-id "${wandb_run_id}"
  )

  if [[ "${RL_USE_RNN:-0}" != "1" ]]; then
    command+=(--no-rnn)
  fi
  if [[ "${FULL_OBSERVATION:-0}" == "1" ]]; then
    command+=(--full-observation)
  fi
  if [[ -n "${HJ_BRAKING_ACCEL:-}" ]]; then
    command+=(--deep-qp-braking-accel "${HJ_BRAKING_ACCEL}")
  fi
  if [[ "${RL_NO_VIDEO:-0}" == "1" ]]; then
    command+=(--no-video)
  fi
  if [[ "${NO_CBF_SCHEDULE:-0}" == "1" ]]; then
    command+=(--no-cbf-schedule)
  fi
  if [[ -n "${RL_RESUME_DIR:-}" ]]; then
    command+=(--resume-dir "${RL_RESUME_DIR}" --resume-step "${RL_RESUME_STEP:-latest}")
  fi

  echo "> Stage 2/2: training InforMARL with DGPPO-style HJ constraints"
  "${command[@]}" 2>&1 | tee -a "${console_log_file}" "${rl_log_dir}/console.log"
}

case "${stage}" in
  hj)
    run_hj
    ;;
  rl)
    run_rl
    ;;
  all)
    run_hj
    run_rl
    ;;
  *)
    echo "STAGE must be one of: hj, rl, all" >&2
    exit 2
    ;;
esac

echo "> Two-stage command completed. Artifacts: ${run_root}"
