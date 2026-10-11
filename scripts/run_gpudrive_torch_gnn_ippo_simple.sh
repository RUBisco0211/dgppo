#!/usr/bin/env bash
set -euo pipefail

cd /home/linminlei/code/dgppo/gpudrive
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export CUDA_HOME=/home/linminlei/.conda/envs/gpudrive
export MADRONA_MWGPU_KERNEL_CACHE="${MADRONA_MWGPU_KERNEL_CACHE:-/home/linminlei/code/dgppo/.cache/gpudrive/gpu2_pci_gnn_simple_128w_smoke/madrona_kernel_cache.bin}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/gpudrive_gnn_simple_gpu2}"
export WANDB_PROJECT=gpudrive
export PYTHONUNBUFFERED=1

mkdir -p "$(dirname "$MADRONA_MWGPU_KERNEL_CACHE")" "$MPLCONFIGDIR" \
  /home/linminlei/code/dgppo/logs/gpudrive/torch_gnn_ippo
exec /home/linminlei/.conda/envs/gpudrive/bin/python \
  baselines/ppo/ppo_sb3.py \
  --config baselines/ppo/config/ppo_gnn_simple_seed0.yaml \
  --run-name "torch_gnn_ippo_simple_seed000_0929185003"
