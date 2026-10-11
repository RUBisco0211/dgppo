#!/usr/bin/env bash
set -euo pipefail

cd /home/linminlei/code/dgppo
export CUDA_VISIBLE_DEVICES=1
export CUDA_HOME=/home/linminlei/.conda/envs/gpudrive
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MADRONA_MWGPU_KERNEL_CACHE=/home/linminlei/code/dgppo/.cache/gpudrive/gpu1_infor_allcars_32w/madrona.bin
export JAX_COMPILATION_CACHE_DIR=/home/linminlei/code/dgppo/.cache/gpudrive/gpu1_infor_allcars_32w/jax
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=-1
export MPLCONFIGDIR=/tmp/gpudrive_infor_allcars_32w
export PYTHONUNBUFFERED=1
export HTTP_PROXY=http://127.0.0.1:7890
export HTTPS_PROXY=http://127.0.0.1:7890
export ALL_PROXY=socks5h://127.0.0.1:7890

mkdir -p "$(dirname "$MADRONA_MWGPU_KERNEL_CACHE")" \
  "$JAX_COMPILATION_CACHE_DIR" "$MPLCONFIGDIR"

exec /home/linminlei/.conda/envs/gpudrive/bin/python \
  train_gpudrive_informarl.py \
  --device cuda \
  --control-all-vehicles \
  --keep-all-vehicles \
  --num-worlds 32 \
  --eval-worlds 32 \
  --dataset-size 4 \
  --scene-file tfrecord \
  --iterations 5000 \
  --total-agent-steps 100000000 \
  --horizon 91 \
  --eval-horizon 91 \
  --eval-interval 50 \
  --save-interval 100 \
  --video-interval 100 \
  --minibatch-worlds 4 \
  --epoch-ppo 1 \
  --coef-ent 0.0001 \
  --wandb-mode online \
  --wandb-project gpudrive
