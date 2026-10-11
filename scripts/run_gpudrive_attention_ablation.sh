#!/usr/bin/env bash
set -euo pipefail
algorithm="${1:?ippo or attention_ippo}"
case "$algorithm" in
  ippo) gpu=1 ;;
  attention_ippo) gpu=3 ;;
  *) exit 2 ;;
esac
cd /home/linminlei/code/dgppo/gpudrive
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$gpu"
export CUDA_HOME=/home/linminlei/.conda/envs/gpudrive
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export OMP_NUM_THREADS=4
export MADRONA_MWGPU_KERNEL_CACHE="/home/linminlei/code/dgppo/.cache/gpudrive/attention_ablation_gpu${gpu}/train.bin"
mkdir -p "$(dirname "$MADRONA_MWGPU_KERNEL_CACHE")"
exec "$CUDA_HOME/bin/python" baselines/ppo/ppo_sb3.py \
  --config "baselines/ppo/config/ppo_${algorithm}_ablation.yaml" "${@:2}"
