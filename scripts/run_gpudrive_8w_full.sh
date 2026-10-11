#!/usr/bin/env bash
set -euo pipefail

cd /home/linminlei/code/dgppo
export CUDA_VISIBLE_DEVICES=2
export CUDA_HOME=/home/linminlei/.conda/envs/gpudrive
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MADRONA_MWGPU_KERNEL_CACHE=/home/linminlei/code/dgppo/.cache/gpudrive/gpu2_8w_straight3p5/madrona_kernel_cache.bin
export JAX_COMPILATION_CACHE_DIR=/home/linminlei/code/dgppo/.cache/gpudrive/gpu2_8w_straight3p5/jax
export MPLCONFIGDIR=/tmp/gpudrive_mpl_gpu2_straight3p5
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=-1
export JAX_LOG_COMPILES=1
# tmux servers retain their creation-time environment. Do not inherit an old
# diagnostic XLA dump path into the full run.
unset XLA_FLAGS
export HTTP_PROXY=http://127.0.0.1:7890
export HTTPS_PROXY=http://127.0.0.1:7890
export ALL_PROXY=socks5h://127.0.0.1:7890
export http_proxy="$HTTP_PROXY"
export https_proxy="$HTTPS_PROXY"
export all_proxy="$ALL_PROXY"

mkdir -p "$(dirname "$MADRONA_MWGPU_KERNEL_CACHE")" "$JAX_COMPILATION_CACHE_DIR" "$MPLCONFIGDIR"

exec /home/linminlei/.conda/envs/gpudrive/bin/python train_gpudrive.py \
  --device cuda \
  --num-worlds 8 \
  --eval-worlds 8 \
  --dataset-size 1 \
  --scene-file tfrecord-00000-of-01000_402.json \
  --iterations 1000 \
  --horizon 64 \
  --max-steer-rad 0.5 \
  --eval-horizon 91 \
  --eval-interval 50 \
  --save-interval 50 \
  --video-interval 50 \
  --minibatch-worlds 4 \
  --epoch-ppo 1 \
  --wandb-mode online \
  --wandb-project gpudrive
