#!/usr/bin/env bash
set -euo pipefail

# Run each distinct task in the TS, CS, and GFS sequences with three seeds.
# Pass extra train_grafo_continual.py flags after the output directory.
output_root="${1:-logs/cgmarl/single_task_validation}"
if [[ $# -gt 0 ]]; then shift; fi

tasks=(
  GRAFO-3 GRAFO-4 'GRAFO-4(1)' 'GRAFO-4(2)' 'GRAFO-4(3)'
  'GRAFO-4(4)' GRAFO-5 'GRAFO-6(1)' 'GRAFO-6(2)'
  'GRAFO-6(3)' 'GRAFO-6(4)' 'GRAFO-6(5)' GRAFO-7
)

for task in "${tasks[@]}"; do
  safe_task="${task//[^a-zA-Z0-9]/_}"
  for seed in 0 1 2; do
    run_dir="${output_root}/${safe_task}_seed${seed}"
    echo "Starting ${task}, seed ${seed}: ${run_dir}"
    conda run --no-capture-output -n dgppo python train_grafo_continual.py \
      --tasks "${task}" --action-encoding discrete --reach-threshold 0.35 \
      --seed "${seed}" --frames-per-task 5000000 \
      --n-env 32 --batch-size 128 --wandb-mode online \
      --name "single_${safe_task}_seed${seed}" --output "${run_dir}" "$@"
  done
done
