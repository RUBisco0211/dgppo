#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Plain joint pairwise CBF-QP baseline. It retains PPO-to-QP action
# supervision and the DGPPO Trainer's W&B/eval/checkpoint metrics.
exec bash scripts/run_mpe_informarl_cbf.sh \
  --pairwise-cbf-mode qp \
  --name plain_pairwise_qp_pilot "$@"
