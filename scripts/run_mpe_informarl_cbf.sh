#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg

# The existing Trainer includes both endpoints: --steps 999 runs 1000
# updates from scratch, each collecting 8*128=1024 environment frames.
# Call main directly so an unattended exception exits instead of entering ipdb.
exec "${PYTHON_BIN:-python}" -u -c 'from train import main; main()' \
  --env MPETarget --num-agents 3 --obs 3 --algo informarl_cbf \
  --seed 0 --steps 999 --n-env-train 8 --batch-size 1024 \
  --n-env-test 4 --eval-epi 4 --eval-interval 25 --save-interval 25 \
  --pairwise-cbf-mode certificate \
  --pairwise-cbf-alpha1 3 --pairwise-cbf-alpha2 3 \
  --pairwise-cbf-safety-margin 0 --pairwise-cbf-slack-penalty 1000000 \
  --cbf-action-loss-coef 1.0 \
  --actor-gnn-layers 2 --Vl-gnn-layers 2 --rnn-step 16 \
  --lr-actor 0.0003 --lr-Vl 0.001 --coef-ent 0.01 \
  --video-dpi 70 --wandb-mode auto \
  --name allocation_distill_pilot "$@"
