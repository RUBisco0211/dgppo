"""Small CPU check of BC loss, checkpoint format and actor-only weights."""
import dataclasses
from pathlib import Path
import subprocess
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "gpudrive"))
from gpudrive.env.config import EnvConfig
from gpudrive.env import constants

torch.set_num_threads(4)
cfg = EnvConfig()
width = (constants.EGO_FEAT_DIM + (cfg.max_num_agents_in_scene - 1) * constants.PARTNER_FEAT_DIM +
         cfg.roadgraph_top_k * constants.ROAD_GRAPH_FEAT_DIM)
out = ROOT / "artifacts/gpudrive_bc_offline_check_nogpu"
for split in ("training", "validation"):
    directory = out / "dataset" / split
    directory.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0 if split == "training" else 1)
    obs = torch.randn(64, width)
    obs[:, constants.EGO_FEAT_DIM:] = 0
    actions = torch.randint(0, 91, (64,))
    torch.save({"observations": obs, "actions": actions}, directory / "batch_000.pt")
for algorithm in ("ippo", "attention_ippo"):
    output = out / algorithm
    subprocess.run([
        sys.executable, str(ROOT / "gpudrive/baselines/ppo/train_bc_actor.py"),
        "--algorithm", algorithm, "--dataset-dir", str(out / "dataset"),
        "--output-dir", str(output), "--config", str(ROOT / f"gpudrive/baselines/ppo/config/ppo_{algorithm}_ablation.yaml"),
        "--epochs", "2", "--batch-size", "32", "--device", "cpu", "--wandb-mode", "disabled",
    ], cwd=ROOT / "gpudrive", check=True)
    checkpoint = torch.load(output / "best_actor.pt", map_location="cpu", weights_only=True)
    assert checkpoint["metadata"]["action_count"] == 91
    assert not any(key.startswith("mlp_extractor.val_") for key in checkpoint["state_dict"])
print("PASS: both BC actors train and checkpoint only actor weights")
