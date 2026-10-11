"""Deterministically select BC scenes with 2–5 controlled cars."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "gpudrive/data/manifests/mini_seed0/scene_index.jsonl"
OUT = ROOT / "gpudrive/data/manifests/bc_seed0"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in SOURCE.read_text().splitlines()]
    heldout_eval = set((ROOT / "gpudrive/data/manifests/attention_ablation_seed0/validation.manifest")
                       .read_text().splitlines())
    counts = {}
    for split, count in (("training", 352), ("validation", 48)):
        eligible = [r for r in rows if r["split"] == split and
                    r["complexity_tier"] in ("simple", "medium") and
                    2 <= r["controllable_count"] <= 5 and
                    r["path"] not in heldout_eval]
        eligible.sort(key=lambda r: (r["complexity_tier"] != "simple",
                                     r["complexity_score"], r["path"]))
        assert len(eligible) >= count, (split, len(eligible))
        chosen = eligible[:count]
        (OUT / f"{split}.manifest").write_text(
            "".join(r["path"] + "\n" for r in chosen), encoding="utf-8")
        counts[split] = {"scenes": count, "controlled_vehicle_estimate": sum(
            r["controllable_count"] for r in chosen), "simple": sum(
            r["complexity_tier"] == "simple" for r in chosen)}
    (OUT / "selection.json").write_text(json.dumps(counts, indent=2))
    print(counts)


if __name__ == "__main__":
    main()
