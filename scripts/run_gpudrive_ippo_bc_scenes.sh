#!/usr/bin/env bash
set -euo pipefail
run_name="${1:?run name required}"
cd /home/linminlei/code/dgppo/gpudrive
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=2
export CUDA_HOME=/home/linminlei/.conda/envs/gpudrive
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MPLBACKEND=Agg
export MADRONA_MWGPU_KERNEL_CACHE=/home/linminlei/code/dgppo/.cache/gpudrive/gpu2_pci_ippo_aligned_16w/madrona_kernel_cache.bin
if ! rg -a -q '\.target sm_80' "$MADRONA_MWGPU_KERNEL_CACHE"; then
  echo "GPU2 is A100/sm_80 but Madrona cache does not match" >&2
  exit 1
fi
exec "$CUDA_HOME/bin/python" baselines/ppo/ppo_sb3.py \
  --config baselines/ppo/config/ppo_ippo_bc_scenes_10m.yaml \
  --run-name "$run_name"
