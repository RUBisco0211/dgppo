#!/usr/bin/env bash
set -euo pipefail

cd /home/linminlei/code/dgppo
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_HOME=/home/linminlei/.conda/envs/gpudrive
export MADRONA_MWGPU_KERNEL_CACHE=/home/linminlei/code/dgppo/.cache/gpudrive/gpu0_gnn_ippo_smoke/madrona_kernel_cache.bin
export JAX_COMPILATION_CACHE_DIR=/home/linminlei/code/dgppo/.cache/jax/gpu0_gnn_ippo_smoke
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=1
export JAX_LOG_COMPILES=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR=/tmp/gpudrive_gnn_ippo_smoke_mpl
unset XLA_FLAGS

mkdir -p "$(dirname "$MADRONA_MWGPU_KERNEL_CACHE")" \
  "$JAX_COMPILATION_CACHE_DIR" "$MPLCONFIGDIR"

exec /home/linminlei/.conda/envs/gpudrive/bin/python \
  train_gpudrive_gnn_ippo.py \
  --action-mode discrete \
  --num-worlds 1 \
  --eval-worlds 1 \
  --iterations 1 \
  --horizon 2 \
  --eval-horizon 2 \
  --eval-interval 1 \
  --save-interval 1 \
  --video-interval 1 \
  --no-video \
  --minibatch-worlds 1 \
  --epoch-ppo 1 \
  --wandb-mode disabled \
  --run-name gnn_ippo_discrete_smoke_1w_h2
