"""Verify native controlled counts and save initial layouts for selected scenes."""
import dataclasses
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from gpudrive.env.config import EnvConfig, RenderConfig
from gpudrive.env.dataset import SceneDataLoader
from gpudrive.env.env_torch import GPUDriveTorchEnv

ROOT = Path(__file__).resolve().parents[1]
manifest_dir = ROOT / "gpudrive/data/manifests/attention_ablation_seed0"
selection = json.loads((manifest_dir / "selection.json").read_text())
output = ROOT / "artifacts/gpudrive_attention_scene_audit"
output.mkdir(parents=True, exist_ok=True)
audit = {}
parser = argparse.ArgumentParser()
parser.add_argument("--split", choices=("training", "validation"), required=True)
args = parser.parse_args()
for split in (args.split,):
    records = selection[split]
    loader = SceneDataLoader(root=str(ROOT / "gpudrive/data/processed" / split),
                             batch_size=len(records), dataset_size=len(records),
                             manifest_path=str(manifest_dir / f"{split}.manifest"))
    env = GPUDriveTorchEnv(config=dataclasses.replace(EnvConfig(), remove_non_vehicles=True,
                           collision_behavior="ignore"), data_loader=loader,
                           max_cont_agents=5, device="cuda", render_config=RenderConfig())
    try:
        counts = env.cont_agent_mask.sum(-1).tolist()
        expected = [r["controllable_count"] for r in records]
        assert counts == expected, (split, counts, expected)
        figures = env.vis.plot_simulator_state(list(range(len(records))))
        for i, figure in enumerate(figures):
            figure.axes[0].set_title(f"{split} {i}: {counts[i]} controlled | {Path(records[i]['path']).name}")
            figure.savefig(output / f"{split}_{i}.png", dpi=100)
            plt.close(figure)
        audit[split] = counts
    finally:
        env.close()
(output / f"{args.split}_native_counts.json").write_text(json.dumps(audit, indent=2))
print(audit)
