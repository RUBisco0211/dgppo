"""Reproducible small native-scene ablation; never modify source scenes."""
import json
import math
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "gpudrive/data/manifests/attention_ablation_seed0"


def eligible(record):
    if record["complexity_tier"] != "simple" or not 2 <= record["controllable_count"] <= 5:
        return None
    if record["vehicle_count"] > 20:
        return None
    scene = json.loads(Path(record["path"]).read_text())
    distances = []
    for obj in scene["objects"]:
        if obj["type"] != "vehicle" or not obj["valid"][0] or obj.get("mark_as_expert", False):
            continue
        start, goal = obj["position"][0], obj["goalPosition"]
        distance = math.hypot(goal["x"] - start["x"], goal["y"] - start["y"])
        if distance < .2:
            continue
        # Recorded trajectory provides a reachability witness, not a learned-policy guarantee.
        points = [p for p, valid in zip(obj["position"], obj["valid"]) if valid]
        end = points[-1]
        if not 5 <= distance <= 100 or math.hypot(end["x"]-goal["x"], end["y"]-goal["y"]) > 2:
            return None
        distances.append(distance)
    if len(distances) != record["controllable_count"]:
        return None
    return {**record, "controlled_goal_distances_m": distances}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    records = [json.loads(line) for line in (ROOT / "gpudrive/data/manifests/mini_seed0/scene_index.jsonl").read_text().splitlines()]
    audit = {}
    for split, count in (("training", 10), ("validation", 5)):
        candidates = [item for r in records if r["split"] == split and (item := eligible(r))]
        candidates.sort(key=lambda r: (r["complexity_score"], r["path"]))
        selected = []
        # Round-robin counts rather than selecting exclusively two-agent scenes.
        while len(selected) < count:
            before = len(selected)
            for n in range(2, 6):
                matching = [r for r in candidates if r["controllable_count"] == n and r not in selected]
                if matching and len(selected) < count:
                    selected.append(matching[0])
            if len(selected) == before:
                raise RuntimeError(f"Not enough qualified scenes in {split}: {len(selected)}")
        (OUT / f"{split}.manifest").write_text("".join(r["path"] + "\n" for r in selected))
        audit[split] = selected
    (OUT / "selection.json").write_text(json.dumps(audit, indent=2))
    base = yaml.safe_load((ROOT / "gpudrive/baselines/ppo/config/ppo_ippo_simple_seed0.yaml").read_text())
    for key in ("resume_checkpoint", "resume_iteration", "run_name", "total_iterations"):
        base.pop(key, None)
    base.update(scene_manifest=str(OUT / "training.manifest"), eval_scene_manifest=str(OUT / "validation.manifest"),
                num_worlds=32, k_unique_scenes=10, total_timesteps=3_000_000,
                eval_num_worlds=5, eval_policy_freq=50, save_policy_freq=50,
                group_name="neighbor_attention_ablation_seed0", sync_tensorboard=False,
                log_after_update=True, eval_render_3d=False)
    for algorithm, model, gpu in (("ippo", "late_fusion", 1), ("attention_ippo", "neighbor_attention", 3)):
        config = {**base, "mlp_class": model, "eval_cuda_visible_devices": gpu,
                  "tags": [algorithm, "actor_neighbor_ablation", "10train_5val", "3m_agent_steps"],
                  "output_root": str(ROOT / "logs/gpudrive/neighbor_attention_ablation" / algorithm),
                  "eval_madrona_cache": str(ROOT / f".cache/gpudrive/attention_ablation_gpu{gpu}/eval.bin")}
        path = ROOT / f"gpudrive/baselines/ppo/config/ppo_{algorithm}_ablation.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False))
    print(json.dumps({k: [(Path(r["path"]).name, r["controllable_count"]) for r in v] for k,v in audit.items()}, indent=2))


if __name__ == "__main__":
    main()
