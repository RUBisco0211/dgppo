#!/usr/bin/env bash
set -euo pipefail

sequence="${1:?usage: run_grafo_cl_compare.sh TS|CS|GFS frog_ewcm|ft}"
method="${2:?usage: run_grafo_cl_compare.sh TS|CS|GFS frog_ewcm|ft}"
case "$sequence" in TS|CS|GFS) ;; *) exit 2 ;; esac
case "$method" in frog_ewcm|ft) ;; *) exit 2 ;; esac

# Two jobs may share the otherwise idle GPU 3; other sessions wait in tmux.
mkdir -p logs/cgmarl/full_cl
while :; do
  for slot in 1 2; do
    exec {lock_fd}>"logs/cgmarl/full_cl/.gpu3_slot${slot}.lock"
    if flock -n "$lock_fd"; then
      echo "GPU 3 slot ${slot} acquired"
      break 2
    fi
    exec {lock_fd}>&-
  done
  echo "Waiting for GPU 3 training slot..."
  sleep 30
done

run_name="full_${sequence,,}_${method}_lambda1_b512_continuous_r035_seed0"
output="logs/cgmarl/full_cl/${run_name}"
mkdir -p "$output"
flags=()
if [[ "$method" == frog_ewcm ]]; then
  flags=(--frog --regularization ewc --importance-mode multi --regularization-coefficient 1)
else
  flags=(--regularization none)
fi

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=3
export XLA_PYTHON_CLIENT_PREALLOCATE=false
echo "Run: ${run_name}; GPU: ${CUDA_VISIBLE_DEVICES}; output: ${output}"
set +e
conda run --no-capture-output -n dgppo python -u train_grafo_continual.py \
  --sequence "$sequence" "${flags[@]}" --seed 0 \
  --action-encoding continuous --reach-threshold 0.35 \
  --frames-per-task 5000000 --n-env 32 --batch-size 512 \
  --eval-episodes 64 --eval-interval 100 --save-interval 100 \
  --wandb-mode online --name "$run_name" --output "$output" \
  2>&1 | tee "$output/terminal.log"
status="${PIPESTATUS[0]}"
echo "EXIT_STATUS:${status}"
exit "$status"
