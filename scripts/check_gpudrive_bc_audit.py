"""Print BC expert replay quality and block uninformative demonstrations."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
directory = ROOT / "gpudrive/data/derived/bc_seed0_tracking"
for split in ("training", "validation"):
    path = directory / split / "audit.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing completed {split} dataset: {path}")
    audit = json.loads(path.read_text())
    cars = audit["controlled_agents"]
    discrete = audit["expert_discrete_goals"] / cars
    summary = {"split": split, "scenes": audit["scenes"], "cars": cars,
               "valid_samples": audit["samples"],
               "discrete_goal_rate": discrete,
               "discrete_collision_rate": audit["expert_discrete_collisions"] / cars,
               "discrete_offroad_rate": audit["expert_discrete_offroads"] / cars,
               "nonzero_accel_fraction": audit["nonzero_accel_samples"] / audit["samples"],
               "nonzero_steer_fraction": audit["nonzero_steer_samples"] / audit["samples"],
               "tracking_error_mean": audit["tracking_error_mean"]}
    print(json.dumps(summary, indent=2))
    if audit["samples"] < cars * 10:
        raise RuntimeError(f"Too few useful expert states for {split}")
    if summary["nonzero_accel_fraction"] < .05 or summary["nonzero_steer_fraction"] < .01:
        raise RuntimeError(f"{split} tracking teacher collapsed to near-zero actions")
    if discrete < .2 or summary["tracking_error_mean"] > 1.0:
        raise RuntimeError(
            f"{split} tracking replay is not reliable enough for BC warm start.")
