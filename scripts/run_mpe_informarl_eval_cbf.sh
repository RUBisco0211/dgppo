#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mode="${1:?usage: run_mpe_informarl_eval_cbf.sh qp|certificate [extra train args]}"
shift
case "$mode" in qp|certificate) ;; *) echo "mode must be qp or certificate" >&2; exit 2 ;; esac
mpe_env="${MPE_ENV:-MPETarget}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
exec "${PYTHON_BIN:-python}" -u -c 'from train import main; main()' \
  --env "$mpe_env" --num-agents 3 --obs 3 --algo informarl_eval_cbf \
  --seed 0 --steps 200000 --n-env-train 128 --batch-size 16384 \
  --n-env-test 32 --eval-epi 1 --eval-interval 1000 --save-interval 1000 \
  --pairwise-cbf-mode "$mode" --pairwise-cbf-alpha1 3 --pairwise-cbf-alpha2 3 \
  --pairwise-cbf-safety-margin 0 --pairwise-cbf-slack-penalty 1000000 \
  --cbf-action-loss-coef 0 \
  --actor-gnn-layers 2 --Vl-gnn-layers 2 --rnn-step 16 \
  --lr-actor 0.0003 --lr-Vl 0.001 --coef-ent 0.01 \
  --video-dpi 100 --wandb-mode auto \
  --name "eval_${mode}_full" "$@"
