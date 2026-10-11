#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 5 ]]; then
  echo "usage: $0 CHECKPOINT CONFIG OUTPUT_DIR GPU CACHE_PATH" >&2
  exit 2
fi

checkpoint="$1"
config="$2"
output_dir="$3"
gpu="$4"
cache_path="$5"
manifest_root=/home/linminlei/code/dgppo/gpudrive/data/manifests/mini_seed0
chunk_dir="$output_dir/manifests"
mkdir -p "$chunk_dir" "$output_dir"
split -d -a 2 -l 16 "$manifest_root/validation.manifest" "$chunk_dir/chunk_"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$gpu"
export CUDA_HOME=/home/linminlei/.conda/envs/gpudrive
export MADRONA_MWGPU_KERNEL_CACHE="$cache_path"
export MPLCONFIGDIR="/tmp/gpudrive_validation150_gpu${gpu}"
export PYTHONUNBUFFERED=1
mkdir -p "$(dirname "$cache_path")" "$MPLCONFIGDIR"

cd /home/linminlei/code/dgppo/gpudrive
for manifest in "$chunk_dir"/chunk_*; do
  chunk="$(basename "$manifest")"
  worlds="$(wc -l < "$manifest")"
  /home/linminlei/.conda/envs/gpudrive/bin/python baselines/ppo/eval_sb3.py \
    --checkpoint "$checkpoint" \
    --config "$config" \
    --output-dir "$output_dir/$chunk" \
    --num-worlds "$worlds" \
    --scene-manifest "$manifest" \
    --seed 10000 \
    --device cuda
done

/home/linminlei/.conda/envs/gpudrive/bin/python - "$output_dir" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = [json.loads(path.read_text()) for path in sorted(root.glob("chunk_*/metrics.json"))]
agents = sum(row["controlled_agents"] for row in rows)
summary = {
    "chunks": len(rows),
    "worlds": sum(row["num_worlds"] for row in rows),
    "controlled_agents": agents,
    "goal_count": sum(row["goal_count"] for row in rows),
    "collision_count": sum(row["collision_count"] for row in rows),
    "offroad_count": sum(row["offroad_count"] for row in rows),
    "episode_reward_sum": sum(row["episode_reward_sum"] for row in rows),
}
for name in ("goal", "collision", "offroad"):
    summary[f"{name}_rate"] = summary[f"{name}_count"] / max(agents, 1)
(root / "summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
PY
