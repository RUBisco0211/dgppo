#!/usr/bin/env bash
set -euo pipefail

cd /home/linminlei/code/dgppo/gpudrive
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
export CUDA_HOME=/home/linminlei/.conda/envs/gpudrive
export MADRONA_MWGPU_KERNEL_CACHE="${MADRONA_MWGPU_KERNEL_CACHE:-/home/linminlei/code/dgppo/.cache/gpudrive/gpu3_ippo_mini_16w/madrona_kernel_cache.bin}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/gpudrive_ippo_mini_gpu3}"
export WANDB_PROJECT=gpudrive
export PYTHONUNBUFFERED=1

mkdir -p "$(dirname "$MADRONA_MWGPU_KERNEL_CACHE")" "$MPLCONFIGDIR" \
  /home/linminlei/code/dgppo/logs/gpudrive/ippo

run_stamp="$(date +%m%d%H%M%S)"
exec /home/linminlei/.conda/envs/gpudrive/bin/python \
  baselines/ppo/ppo_sb3.py \
  --config baselines/ppo/config/ppo_ippo_mini_seed0.yaml \
  --run-name "ippo_mini_seed000_${run_stamp}"
