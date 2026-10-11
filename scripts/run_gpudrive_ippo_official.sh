#!/usr/bin/env bash
set -euo pipefail

cd /home/linminlei/code/dgppo/gpudrive
export CUDA_VISIBLE_DEVICES=0
export CUDA_HOME=/home/linminlei/.conda/envs/gpudrive
export MADRONA_MWGPU_KERNEL_CACHE=/home/linminlei/code/dgppo/.cache/gpudrive/gpu0_ippo_official/madrona_kernel_cache.bin
export MPLCONFIGDIR=/tmp/gpudrive_ippo_mpl_gpu0
export WANDB_PROJECT=gpudrive
export PYTHONUNBUFFERED=1

mkdir -p "$(dirname "$MADRONA_MWGPU_KERNEL_CACHE")" "$MPLCONFIGDIR"

exec /home/linminlei/.conda/envs/gpudrive/bin/python \
  baselines/ppo/ppo_sb3.py
