#!/usr/bin/env bash
set -euo pipefail

sequence="${1:?usage: run_grafo_cl_full.sh TS|CS|GFS [frog_ewcm|frog|ft] [continuous|discrete] [reach_threshold]}"
method="${2:-frog_ewcm}"
action_encoding="${3:-discrete}"
case "$sequence" in
  TS|CS|GFS) ;;
  *) echo "unsupported sequence: $sequence" >&2; exit 2 ;;
esac
case "$action_encoding" in
  continuous) reach_threshold=0.05 ;;
  discrete) reach_threshold=0.35 ;;
  *) echo "unsupported action encoding: $action_encoding" >&2; exit 2 ;;
esac
reach_threshold="${4:-$reach_threshold}"
case "$method" in
  frog_ewcm) method_flags=(--frog --regularization ewc --importance-mode multi) ;;
  frog) method_flags=(--frog --regularization none) ;;
  ft) method_flags=(--regularization none) ;;
  *) echo "unsupported method: $method" >&2; exit 2 ;;
esac

sequence_lower="${sequence,,}"
run_name="full_${sequence_lower}_${method}_seed0_b512_${action_encoding}_r${reach_threshold//./}"
output="logs/cgmarl/full_cl/${run_name}"
mkdir -p "$output"

exec conda run --no-capture-output -n dgppo python -u train_grafo_continual.py \
  --sequence "$sequence" "${method_flags[@]}" --seed 0 \
  --action-encoding "$action_encoding" --reach-threshold "$reach_threshold" \
  --frames-per-task 5000000 --n-env 32 --batch-size 512 \
  --eval-episodes 64 --eval-interval 100 --save-interval 100 \
  --wandb-mode online --name "$run_name" --output "$output"
