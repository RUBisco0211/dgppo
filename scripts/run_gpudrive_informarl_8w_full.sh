#!/usr/bin/env bash
set -euo pipefail
cd /home/linminlei/code/dgppo
export CUDA_VISIBLE_DEVICES=1
export CUDA_HOME=/home/linminlei/.conda/envs/gpudrive
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MADRONA_MWGPU_KERNEL_CACHE=/home/linminlei/code/dgppo/.cache/gpudrive/gpu1_8w_informarl_5c10n_progress01_goalpark_v1/madrona_kernel_cache.bin
export JAX_COMPILATION_CACHE_DIR=/home/linminlei/code/dgppo/.cache/gpudrive/gpu1_8w_informarl_5c10n_progress01/jax
export MPLCONFIGDIR=/tmp/gpudrive_mpl_gpu1_informarl
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=-1
export JAX_LOG_COMPILES=1
unset XLA_FLAGS
export HTTP_PROXY=http://127.0.0.1:7890 HTTPS_PROXY=http://127.0.0.1:7890
export ALL_PROXY=socks5h://127.0.0.1:7890
export http_proxy="$HTTP_PROXY" https_proxy="$HTTPS_PROXY" all_proxy="$ALL_PROXY"
mkdir -p "$(dirname "$MADRONA_MWGPU_KERNEL_CACHE")" "$JAX_COMPILATION_CACHE_DIR" "$MPLCONFIGDIR"
exec /home/linminlei/.conda/envs/gpudrive/bin/python train_gpudrive_informarl.py \
  --device cuda --num-worlds 8 --eval-worlds 8 --dataset-size 1 \
  --scene-file tfrecord-00000-of-01000_402.json --iterations 1000 --horizon 91 \
  --max-steer-rad 0.3 --eval-horizon 91 --eval-interval 50 \
  --save-interval 50 --video-interval 50 --minibatch-worlds 4 --epoch-ppo 1 \
  --wandb-mode online --wandb-project gpudrive
