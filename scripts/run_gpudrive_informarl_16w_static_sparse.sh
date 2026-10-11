#!/usr/bin/env bash
set -euo pipefail
cd /home/linminlei/code/dgppo
export CUDA_VISIBLE_DEVICES=1
export CUDA_HOME=/home/linminlei/.conda/envs/gpudrive
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MADRONA_MWGPU_KERNEL_CACHE=/home/linminlei/code/dgppo/.cache/gpudrive/gpu1_informarl_goalpark_staticruntime_v2/madrona_kernel_cache.bin
export JAX_COMPILATION_CACHE_DIR=/home/linminlei/code/dgppo/.cache/gpudrive/gpu1_16w_informarl_5c10static_sparse/jax
export MPLCONFIGDIR=/tmp/gpudrive_mpl_gpu1_informarl_16w_static_sparse
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=-1
export JAX_LOG_COMPILES=1
unset XLA_FLAGS
export HTTP_PROXY=http://127.0.0.1:7890 HTTPS_PROXY=http://127.0.0.1:7890
export ALL_PROXY=socks5h://127.0.0.1:7890
export http_proxy="$HTTP_PROXY" https_proxy="$HTTPS_PROXY" all_proxy="$ALL_PROXY"
mkdir -p "$(dirname "$MADRONA_MWGPU_KERNEL_CACHE")" "$JAX_COMPILATION_CACHE_DIR" "$MPLCONFIGDIR"
exec /home/linminlei/.conda/envs/gpudrive/bin/python train_gpudrive_informarl.py \
  --device cuda --num-worlds 16 --eval-worlds 16 --dataset-size 1 \
  --scene-file tfrecord-00000-of-01000_402.json --iterations 1000 --horizon 91 \
  --background-behavior static --goal-progress-weight 0 \
  --max-steer-rad 0.3 --eval-horizon 91 --eval-interval 50 \
  --save-interval 50 --video-interval 50 --minibatch-worlds 4 --epoch-ppo 1 \
  --wandb-mode online --wandb-project gpudrive
