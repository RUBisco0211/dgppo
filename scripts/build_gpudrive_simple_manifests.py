#!/usr/bin/env python3
"""Build reproducible simple-scene manifests from the GPUDrive scene index."""

import argparse
import json
import random
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--eval-seed", type=int, default=10000)
    args = parser.parse_args()

    records = [json.loads(line) for line in args.index.read_text().splitlines() if line]
    train = sorted(
        record["path"] for record in records
        if record["split"] == "training" and record["complexity_tier"] == "simple"
    )
    validation = sorted(
        record["path"] for record in records
        if record["split"] == "validation" and record["complexity_tier"] == "simple"
    )
    rng = random.Random(args.eval_seed)
    eval_scene = rng.choice(validation)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "training_simple.manifest").write_text(
        "\n".join(train) + "\n", encoding="utf-8"
    )
    (args.output_dir / "validation_simple.manifest").write_text(
        "\n".join(validation) + "\n", encoding="utf-8"
    )
    (args.output_dir / "validation_simple_eval1.manifest").write_text(
        eval_scene + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "training_simple": len(train),
        "validation_simple": len(validation),
        "periodic_eval_scene": eval_scene,
        "eval_seed": args.eval_seed,
    }, indent=2))


if __name__ == "__main__":
    main()
