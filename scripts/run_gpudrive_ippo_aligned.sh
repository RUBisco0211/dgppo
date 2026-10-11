#!/usr/bin/env bash
set -euo pipefail

cd /home/linminlei/code/dgppo/gpudrive
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1
export CUDA_HOME=/home/linminlei/.conda/envs/gpudrive
export MADRONA_MWGPU_KERNEL_CACHE=/home/linminlei/code/dgppo/.cache/gpudrive/gpu1_pci_ippo_native_16w/madrona_kernel_cache.bin
export MPLCONFIGDIR=/tmp/gpudrive_ippo_native_gpu1
export WANDB_PROJECT=gpudrive
export PYTHONUNBUFFERED=1

mkdir -p "$(dirname "$MADRONA_MWGPU_KERNEL_CACHE")" "$MPLCONFIGDIR" \
  /home/linminlei/code/dgppo/logs/gpudrive/ippo

exec /home/linminlei/.conda/envs/gpudrive/bin/python \
  baselines/ppo/ppo_sb3.py \
  --config baselines/ppo/config/ppo_aligned_gnn_comparison.yaml
