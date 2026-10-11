#!/usr/bin/env python3
"""Index GPUDrive JSON scenes and build deterministic train/eval manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path


VEHICLE_TYPES = {"vehicle", "car"}


def _valid_at(values, index: int) -> bool:
    return index < len(values) and bool(values[index])


def _xy_at(values, index: int):
    if index >= len(values):
        return None
    point = values[index]
    if isinstance(point, dict):
        x, y = point.get("x"), point.get("y")
    elif len(point) >= 2:
        x, y = point[:2]
    else:
        return None
    if not (math.isfinite(x) and math.isfinite(y)):
        return None
    return float(x), float(y)


def _road_fingerprint(roads) -> str:
    """Translation/rotation-invariant approximate fingerprint of road geometry."""
    signatures = []
    for road in roads:
        points = road.get("geometry") or road.get("polyline") or road.get("points") or []
        segment_lengths = []
        for first, second in zip(points, points[1:]):
            try:
                if isinstance(first, dict):
                    dx = float(second["x"]) - float(first["x"])
                    dy = float(second["y"]) - float(first["y"])
                else:
                    dx = float(second[0]) - float(first[0])
                    dy = float(second[1]) - float(first[1])
            except (TypeError, ValueError, IndexError):
                continue
            segment_lengths.append(round(math.hypot(dx, dy), 1))
        signatures.append((str(road.get("type", "")), tuple(sorted(segment_lengths))))
    payload = json.dumps(sorted(signatures), separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def analyze_scene(path: Path, split: str) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        scene = json.load(stream)

    objects = scene.get("objects", [])
    roads = scene.get("roads", [])
    vehicle_count = moving_count = controllable_count = 0
    goal_distances = []
    heading_changes = []
    initial_positions = []
    tracks_to_predict = 0

    for obj in objects:
        if str(obj.get("type", "")).lower() not in VEHICLE_TYPES:
            continue
        vehicle_count += 1
        valid = obj.get("valid", [])
        positions = obj.get("position", [])
        if not _valid_at(valid, 0):
            continue
        start = _xy_at(positions, 0)
        if start is None:
            continue
        initial_positions.append(start)
        valid_positions = [
            xy for idx in range(min(len(valid), len(positions)))
            if _valid_at(valid, idx) and (xy := _xy_at(positions, idx)) is not None
        ]
        goal = obj.get("goalPosition") or obj.get("goal_position")
        if isinstance(goal, dict):
            goal_distance = math.hypot(float(goal["x"]) - start[0], float(goal["y"]) - start[1])
        elif goal and len(goal) >= 2:
            goal_distance = math.hypot(float(goal[0]) - start[0], float(goal[1]) - start[1])
        elif valid_positions:
            goal_distance = math.hypot(
                valid_positions[-1][0] - start[0], valid_positions[-1][1] - start[1]
            )
        else:
            goal_distance = 0.0
        goal_distances.append(goal_distance)
        dynamic = goal_distance >= 0.2
        moving_count += int(dynamic)
        controllable_count += int(dynamic and not bool(obj.get("mark_as_expert", False)))
        tracks_to_predict += int(bool(obj.get("tracks_to_predict", False)))

        headings = obj.get("heading", [])
        valid_headings = [
            float(headings[idx]) for idx in range(min(len(valid), len(headings)))
            if _valid_at(valid, idx) and math.isfinite(float(headings[idx]))
        ]
        if len(valid_headings) > 1:
            unwrapped = [valid_headings[0]]
            for heading in valid_headings[1:]:
                delta = (heading - unwrapped[-1] + math.pi) % (2 * math.pi) - math.pi
                unwrapped.append(unwrapped[-1] + delta)
            heading_changes.append(max(unwrapped) - min(unwrapped))

    close_pairs = 0
    for idx, first in enumerate(initial_positions):
        for second in initial_positions[idx + 1:]:
            close_pairs += int(math.dist(first, second) <= 15.0)

    return {
        "path": str(path.resolve()),
        "split": split,
        "scenario_id": scene.get("scenario_id", path.stem),
        "object_count": len(objects),
        "vehicle_count": vehicle_count,
        "moving_vehicle_count": moving_count,
        "controllable_count": min(controllable_count, 64),
        "road_count": len(roads),
        "max_heading_span_rad": max(heading_changes, default=0.0),
        "mean_goal_distance_m": sum(goal_distances) / max(len(goal_distances), 1),
        "max_goal_distance_m": max(goal_distances, default=0.0),
        "initial_close_pairs_15m": close_pairs,
        "tracks_to_predict": tracks_to_predict,
        "road_fingerprint": _road_fingerprint(roads),
    }


def _rank(values, value):
    if len(values) <= 1:
        return 0.0
    return sum(candidate <= value for candidate in values) / len(values)


def assign_complexity(records):
    dimensions = (
        "controllable_count",
        "road_count",
        "max_heading_span_rad",
        "initial_close_pairs_15m",
    )
    populations = {key: sorted(record[key] for record in records) for key in dimensions}
    for record in records:
        record["complexity_score"] = sum(
            _rank(populations[key], record[key]) for key in dimensions
        ) / len(dimensions)
    ordered = sorted(record["complexity_score"] for record in records)
    low = ordered[len(ordered) // 3]
    high = ordered[(2 * len(ordered)) // 3]
    for record in records:
        score = record["complexity_score"]
        record["complexity_tier"] = "simple" if score <= low else "medium" if score <= high else "complex"


def write_manifest(path: Path, records, rng: random.Random, weighted=False):
    weights = {"simple": 1, "medium": 2, "complex": 4}
    paths = []
    for record in records:
        paths.extend([record["path"]] * (weights[record["complexity_tier"]] if weighted else 1))
    rng.shuffle(paths)
    path.write_text("\n".join(paths) + "\n", encoding="utf-8")
    return len(paths)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    by_split = {}
    all_records = []
    for split in ("training", "validation", "testing"):
        paths = sorted((args.data_root / split).glob("tfrecord*.json"))
        records = [analyze_scene(path, split) for path in paths]
        assign_complexity(records)
        by_split[split] = records
        all_records.extend(records)

    index_path = args.output_dir / "scene_index.jsonl"
    with index_path.open("w", encoding="utf-8") as stream:
        for record in all_records:
            stream.write(json.dumps(record, sort_keys=True) + "\n")

    counts = {}
    for offset, split in enumerate(("training", "validation", "testing")):
        counts[split] = write_manifest(
            args.output_dir / f"{split}.manifest",
            by_split[split],
            random.Random(args.seed + offset),
            weighted=split == "training",
        )

    fingerprints = defaultdict(list)
    for record in all_records:
        fingerprints[record["road_fingerprint"]].append(record)
    cross_split_groups = []
    for fingerprint, group in fingerprints.items():
        if len({record["split"] for record in group}) > 1:
            cross_split_groups.append({
                "road_fingerprint": fingerprint,
                "scenes": [{"split": r["split"], "path": r["path"]} for r in group],
            })

    summary = {
        "seed": args.seed,
        "scene_counts": {split: len(records) for split, records in by_split.items()},
        "manifest_entries": counts,
        "complexity_tiers": {
            split: dict(Counter(record["complexity_tier"] for record in records))
            for split, records in by_split.items()
        },
        "cross_split_road_fingerprint_groups": cross_split_groups,
        "note": "Road fingerprints are approximate geometry signatures, not proof of identical maps.",
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
