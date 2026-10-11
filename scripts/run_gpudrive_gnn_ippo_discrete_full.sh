#!/usr/bin/env bash
set -euo pipefail

cd /home/linminlei/code/dgppo
export CUDA_VISIBLE_DEVICES=2
export CUDA_HOME=/home/linminlei/.conda/envs/gpudrive
export MADRONA_MWGPU_KERNEL_CACHE=/home/linminlei/code/dgppo/.cache/gpudrive/gpu2_gnn_ippo_discrete_16w/madrona_kernel_cache.bin
export JAX_COMPILATION_CACHE_DIR=/home/linminlei/code/dgppo/.cache/jax/gpu2_gnn_ippo_discrete_16w
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=1
export JAX_LOG_COMPILES=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# Unlike the default BFC allocator, release transient rollout/update buffers
# between iterations. The full 64-vehicle road graphs otherwise fragment VRAM
# after the first PPO update even though the card has sufficient total memory.
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR=/tmp/gpudrive_gnn_ippo_discrete_gpu2
unset XLA_FLAGS

mkdir -p "$(dirname "$MADRONA_MWGPU_KERNEL_CACHE")" \
  "$JAX_COMPILATION_CACHE_DIR" "$MPLCONFIGDIR"

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
  --eval-interval 50 \
  --save-interval 50 \
  --video-interval 50 \
  --wandb-project gpudrive \
  --wandb-mode online
