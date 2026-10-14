#!/usr/bin/env bash
set -euo pipefail

# Single-task validation through the same 128-step collector used by the
# sequential experiment. Additional flags are forwarded to the runner.
task_name="${1:-GRAFO-3}"
if [[ $# -gt 0 ]]; then shift; fi
exec conda run --no-capture-output -n dgppo python train_grafo_continual.py \
  --tasks "$task_name" --action-encoding discrete --reach-threshold 0.35 \
  --n-env 32 --batch-size 128 \
  --eval-episodes 64 --wandb-mode online \
  "$@"
