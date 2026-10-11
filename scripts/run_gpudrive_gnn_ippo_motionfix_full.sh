#!/usr/bin/env bash
set -euo pipefail

cd /home/linminlei/code/dgppo
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=2
export CUDA_HOME=/home/linminlei/.conda/envs/gpudrive
export MADRONA_MWGPU_KERNEL_CACHE=/home/linminlei/code/dgppo/.cache/gpudrive/gpu2_pci_gnn_ippo_native_16w/madrona_kernel_cache.bin
export JAX_COMPILATION_CACHE_DIR=/home/linminlei/code/dgppo/.cache/jax/gpu2_pci_gnn_ippo_native_16w
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=1
export JAX_LOG_COMPILES=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR=/tmp/gpudrive_gnn_ippo_native_gpu2
unset XLA_FLAGS

mkdir -p "$(dirname "$MADRONA_MWGPU_KERNEL_CACHE")" \
  "$JAX_COMPILATION_CACHE_DIR" "$MPLCONFIGDIR"

run_stamp="$(date +%m%d%H%M%S)"
exec /home/linminlei/.conda/envs/gpudrive/bin/python \
  train_gpudrive_gnn_ippo.py \
  --action-mode discrete \
  --num-worlds 16 \
  --eval-worlds 16 \
  --iterations 1000 \
  --horizon 91 \
  --eval-horizon 91 \
  --minibatch-worlds 4 \
  --epoch-ppo 4 \
  --coef-ent 1e-2 \
  --eval-interval 50 \
  --save-interval 50 \
  --video-interval 50 \
  --log-dir logs/gpudrive/gnn_ippo \
  --run-name "gnn_ippo_native_seed000_${run_stamp}" \
  --wandb-project gpudrive \
  --wandb-group ippo_gnn_native_comparison_seed0 \
  --wandb-mode online
