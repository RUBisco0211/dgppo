"""Check whether a BC checkpoint predicts meaningful actions on held-out scenes."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "gpudrive"))

import numpy as np
import torch
import yaml
from box import Box
from gymnasium import spaces

from gpudrive.env.config import EnvConfig
from gpudrive.networks.neighbor_attention import NeighborAttentionNet
from gpudrive.networks.perm_eq_late_fusion import LateFusionNet, LateFusionPolicy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-dir", default=str(ROOT / "gpudrive/data/derived/bc_seed0_tracking"))
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    metadata = checkpoint["metadata"]
    algorithm = metadata["algorithm"]
    config = Box(yaml.safe_load((ROOT / f"gpudrive/baselines/ppo/config/ppo_{algorithm}_ablation.yaml").read_text()))
    chunks = [torch.load(path, map_location="cpu", weights_only=True)
              for path in sorted((Path(args.dataset_dir) / "validation").glob("batch_*.pt"))]
    if not chunks:
        raise FileNotFoundError("No validation chunks")
    obs = torch.cat([chunk["observations"] for chunk in chunks])
    target = torch.cat([chunk["actions"] for chunk in chunks])
    env_config = EnvConfig()
    cls = LateFusionNet if algorithm == "ippo" else NeighborAttentionNet
    policy = LateFusionPolicy(spaces.Box(-np.inf, np.inf, shape=(obs.shape[1],), dtype=np.float32),
                              env_config, config, spaces.Discrete(91), lambda _: 3e-4,
                              mlp_class=cls)
    policy.load_state_dict(checkpoint["state_dict"], strict=False)
    policy.set_training_mode(False)
    predictions = []
    with torch.no_grad():
        for chunk in obs.split(1024):
            predictions.append(policy.get_distribution(chunk).distribution.logits.argmax(-1))
    predicted = torch.cat(predictions)
    zero_action = 3 * 13 + 6
    target_nonzero = target != zero_action
    predicted_nonzero = predicted != zero_action
    result = {
        "algorithm": algorithm,
        "samples": len(target),
        "majority_class": int(torch.bincount(target, minlength=91).argmax()),
        "majority_baseline_accuracy": float(torch.bincount(target, minlength=91).max() / len(target)),
        "exact_accuracy": float((predicted == target).float().mean()),
        "target_nonzero_fraction": float(target_nonzero.float().mean()),
        "predicted_nonzero_fraction": float(predicted_nonzero.float().mean()),
        "nonzero_action_recall": float(((predicted == target) & target_nonzero).sum() /
                                       target_nonzero.sum().clamp_min(1)),
        "predicted_unique_actions": int(predicted.unique().numel()),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
