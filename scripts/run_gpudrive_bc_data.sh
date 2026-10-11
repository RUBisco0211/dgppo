#!/usr/bin/env bash
set -euo pipefail
split="${1:?training or validation}"
case "$split" in
  training) gpu=1; batch_worlds=32 ;;
  validation) gpu=3; batch_worlds=16 ;;
  *) exit 2 ;;
esac
data_device="${GPUDRIVE_BC_DATA_DEVICE:-cuda}"
if [ "$data_device" = cpu ]; then
  batch_worlds=16
elif [ "$data_device" != cuda ]; then
  exit 2
fi
cd /home/linminlei/code/dgppo/gpudrive
export CUDA_DEVICE_ORDER=PCI_BUS_ID
if [ "$data_device" = cpu ]; then
  export CUDA_VISIBLE_DEVICES=-1
else
  export CUDA_VISIBLE_DEVICES="$gpu"
fi
export CUDA_HOME=/home/linminlei/.conda/envs/gpudrive
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MADRONA_MWGPU_KERNEL_CACHE="/home/linminlei/code/dgppo/.cache/gpudrive/bc_gpu${gpu}_${batch_worlds}w.bin"
mkdir -p "$(dirname "$MADRONA_MWGPU_KERNEL_CACHE")"
exec "$CUDA_HOME/bin/python" baselines/ppo/build_bc_dataset.py \
  --split "$split" \
  --manifest "/home/linminlei/code/dgppo/gpudrive/data/manifests/bc_seed0/${split}.manifest" \
  --output-dir /home/linminlei/code/dgppo/gpudrive/data/derived/bc_seed0_tracking \
  --batch-worlds "$batch_worlds" --device "$data_device"
