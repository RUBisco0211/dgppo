#!/usr/bin/env bash
set -euo pipefail

cd /home/linminlei/code/dgppo
export CUDA_VISIBLE_DEVICES=3
export CUDA_HOME=/home/linminlei/.conda/envs/gpudrive
export MADRONA_MWGPU_KERNEL_CACHE=/home/linminlei/code/dgppo/.cache/gpudrive/gpu3_gnn_ippo_eval_1w/madrona_kernel_cache.bin
export JAX_COMPILATION_CACHE_DIR=/home/linminlei/code/dgppo/.cache/jax/gpu3_gnn_ippo_eval_1w
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR=/tmp/gpudrive_gnn_ippo_eval_950_gpu3

run_dir=/home/linminlei/code/dgppo/logs/gpudrive/gnn_ippo/gnn_ippo_discrete_motionfix_ent1e2_seed000_0927154801
checkpoint="$run_dir/checkpoints/950/training_state.pkl"
output_dir="$run_dir/eval_checkpoint_950_other_maps"
mkdir -p "$(dirname "$MADRONA_MWGPU_KERNEL_CACHE")" "$JAX_COMPILATION_CACHE_DIR" "$MPLCONFIGDIR" "$output_dir"

for scene in \
  tfrecord-00000-of-01000_325.json \
  tfrecord-00000-of-01000_4.json \
  tfrecord-00002-of-01000_407.json
do
  echo "> Evaluating $scene"
  /home/linminlei/.conda/envs/gpudrive/bin/python eval_gpudrive_gnn_ippo.py \
    --checkpoint "$checkpoint" \
    --scene-file "$scene" \
    --output-dir "$output_dir" \
    --worlds 1 \
    --horizon 91 \
    --eval-seed 10000
done
