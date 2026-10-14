"""Continuously mirror GRAFO evaluation records into one CSV per run.

The exporter reads append-only metrics.jsonl files and never touches a live
trainer or its W&B state. Run with --watch after starting tmux training jobs.
"""

import argparse
import csv
import json
import os
import pathlib
import time


def evaluation_rows(metrics_path):
    records = []
    first_update = {}
    with metrics_path.open(encoding="utf-8") as stream:
        for line in stream:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue  # A trainer may be writing its last line.
            stage = int(record["counters/stage"])
            first_update.setdefault(stage, int(record["counters/update"]))
            if any(key.startswith("eval/") for key in record):
                records.append(record)
    return records, first_update


def export_run(run_dir):
    metrics_path = run_dir / "metrics.jsonl"
    if not metrics_path.exists():
        return 0
    records, first_update = evaluation_rows(metrics_path)
    if not records:
        return 0

    stages = {}
    task_columns = set()
    for record in records:
        stage = int(record["counters/stage"])
        stages.setdefault(stage, set()).update(
            key.split("/")[1] for key in record if key.startswith("eval/"))
    trained_tasks = {}
    previous = set()
    for stage, visible in sorted(stages.items()):
        new = visible - previous
        trained_tasks[stage] = next(iter(new)) if len(new) == 1 else ""
        previous |= visible
        task_columns |= visible

    columns = [
        "run", "stage", "trained_task", "update", "update_in_stage",
        "total_frames", "stage_complete", "AP", "AF", "AFP",
    ]
    for task in sorted(task_columns):
        columns.extend((f"{task}/success_rate",
                        f"{task}/final_success_rate",
                        f"{task}/return_mean"))

    rows = []
    for record in records:
        stage = int(record["counters/stage"])
        update = int(record["counters/update"])
        row = {
            "run": run_dir.name,
            "stage": stage,
            "trained_task": trained_tasks[stage],
            "update": update,
            "update_in_stage": update - first_update[stage] + 1,
            "total_frames": int(record["counters/total_frames"]),
            "stage_complete": int("continual/AP" in record),
            "AP": record.get("continual/AP", ""),
            "AF": record.get("continual/AF", ""),
            "AFP": record.get("continual/AFP", ""),
        }
        for task in task_columns:
            for metric in ("success_rate", "final_success_rate", "return_mean"):
                row[f"{task}/{metric}"] = record.get(
                    f"eval/{task}/{metric}", "")
        rows.append(row)

    output = run_dir / "evaluation.csv"
    temporary = output.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(output)
    return len(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=pathlib.Path,
                        default=pathlib.Path("logs/cgmarl/full_cl"))
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=float, default=30.0)
    args = parser.parse_args()
    while True:
        runs = sorted(path for path in args.root.glob("full_*") if path.is_dir())
        counts = {path.name: export_run(path) for path in runs}
        print(json.dumps(counts, sort_keys=True), flush=True)
        if not args.watch or (runs and all((path / "summary.json").exists()
                                        for path in runs)):
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
