#!/usr/bin/env bash
set -euo pipefail

cd /home/linminlei/code/dgppo/gpudrive
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=3
export CUDA_HOME=/home/linminlei/.conda/envs/gpudrive
export MADRONA_MWGPU_KERNEL_CACHE=/home/linminlei/code/dgppo/.cache/gpudrive/gpu3_pci_torch_gnn_ippo_native_16w/madrona_kernel_cache.bin
export MPLCONFIGDIR=/tmp/gpudrive_torch_gnn_ippo_smoke_gpu3
export PYTHONUNBUFFERED=1

mkdir -p "$(dirname "$MADRONA_MWGPU_KERNEL_CACHE")" "$MPLCONFIGDIR" \
  /home/linminlei/code/dgppo/logs/gpudrive/torch_gnn_ippo_smoke

run_stamp="$(date +%m%d%H%M%S)"
exec /home/linminlei/.conda/envs/gpudrive/bin/python \
  baselines/ppo/ppo_sb3.py \
  --config baselines/ppo/config/ppo_gnn_aligned_jax.yaml \
  --total-iterations 20 \
  --eval-policy-freq 20 \
  --save-policy-freq 20 \
  --output-root /home/linminlei/code/dgppo/logs/gpudrive/torch_gnn_ippo_smoke \
  --run-name "torch_gnn_ippo_smoke_seed000_${run_stamp}" \
  --wandb-mode disabled
