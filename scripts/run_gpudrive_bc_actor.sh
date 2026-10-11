#!/usr/bin/env bash
set -euo pipefail
algorithm="${1:?ippo, attention_ippo, or gnn_ippo}"
output="${2:?run directory}"
dataset="${3:?dataset directory}"
case "$algorithm" in
  ippo) gpu=2 ;;
  attention_ippo) gpu=2 ;;
  gnn_ippo) gpu=2 ;;
  *) exit 2 ;;
esac
cd /home/linminlei/code/dgppo/gpudrive
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$gpu"
export CUDA_HOME=/home/linminlei/.conda/envs/gpudrive
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
config="baselines/ppo/config/ppo_${algorithm}_ablation.yaml"
epochs=15
patience=4
if [[ "$algorithm" == "gnn_ippo" ]]; then
  config="baselines/ppo/config/ppo_gnn_bc_scenes.yaml"
  epochs=100
  patience=20
fi
exec "$CUDA_HOME/bin/python" baselines/ppo/train_bc_actor.py \
  --algorithm "$algorithm" --dataset-dir "$dataset" --output-dir "$output" \
  --config "$config" \
  --epochs "$epochs" --patience "$patience" \
  --batch-size 1024 --lr 0.0003 --wandb-mode online
