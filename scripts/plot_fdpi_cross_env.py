#!/usr/bin/env python3
"""Aggregate per-run FDPI-style DGPPO diagnostics across environments."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_root", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    root = args.results_root.expanduser().resolve()
    output = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else root
    )
    output.mkdir(parents=True, exist_ok=True)

    records = []
    for summary_path in sorted(root.glob("*/latest_summary.json")):
        env_name = summary_path.parent.name
        sample_path = summary_path.parent / "latest_samples.npz"
        manifest_path = summary_path.parent / "manifest.json"
        if not sample_path.is_file() or not manifest_path.is_file():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        error = np.load(sample_path)["error"].reshape(-1)
        records.append(
            {
                "environment": env_name,
                "family": "LidarEnv" if env_name.startswith("Lidar") else "MPE",
                "run_dir": manifest["run_dir"],
                "error": error,
                "summary": summary,
            }
        )
    if not records:
        raise SystemExit(f"no complete per-environment results found below {root}")

    family_order = {"MPE": 0, "LidarEnv": 1}
    records.sort(key=lambda item: (family_order[item["family"]], item["environment"]))
    colors = {"MPE": "#2878B5", "LidarEnv": "#D95319"}

    figure3, axis3 = plt.subplots(figsize=(10.5, 4.8))
    names = [record["environment"] for record in records]
    ratios = [
        record["summary"]["instantaneous_violation_agent_state_ratio"]
        for record in records
    ]
    axis3.bar(
        names,
        ratios,
        color=[colors[record["family"]] for record in records],
    )
    axis3.set_ylabel("Post-convergence violation-transition ratio")
    axis3.set_title("DGPPO violating samples on 100k policy-collected states")
    axis3.tick_params(axis="x", rotation=30)
    axis3.grid(axis="y", alpha=0.25)
    for index, ratio in enumerate(ratios):
        axis3.text(index, ratio, f"{100 * ratio:.2f}%", ha="center", va="bottom")
    figure3.tight_layout()
    figure3.savefig(output / "figure3_cross_env_violation_ratio.png", dpi=180)
    plt.close(figure3)

    limit = max(
        float(np.quantile(np.abs(record["error"]), 0.995))
        for record in records
    )
    limit = max(limit, 1e-6)
    bins = np.linspace(-limit, limit, 161)
    ncols = 4
    nrows = math.ceil(len(records) / ncols)
    figure4, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.2 * ncols, 3.5 * nrows),
        sharex=True,
        sharey=False,
        squeeze=False,
    )
    for axis, record in zip(axes.flat, records):
        axis.hist(
            record["error"],
            bins=bins,
            density=True,
            histtype="step",
            linewidth=1.5,
            color=colors[record["family"]],
        )
        axis.axvline(0.0, color="black", linestyle="--", linewidth=1.0)
        summary = record["summary"]
        axis.set_title(record["environment"])
        axis.text(
            0.03,
            0.96,
            f"mean={summary['error']['mean']:.3f}\n"
            f"RMSE={summary['error']['rmse']:.3f}\n"
            f"unsafe n={summary['trajectory_unsafe_value_count']}",
            transform=axis.transAxes,
            va="top",
            fontsize=9,
        )
        axis.grid(alpha=0.2)
    for axis in axes.flat[len(records):]:
        axis.set_visible(False)
    figure4.supxlabel("Vh prediction - discounted-max trajectory target")
    figure4.supylabel("Density")
    figure4.suptitle("DGPPO feasibility-function estimation error", y=1.01)
    figure4.tight_layout()
    figure4.savefig(output / "figure4_cross_env_vh_error.png", dpi=180, bbox_inches="tight")
    plt.close(figure4)

    table_rows = []
    for record in records:
        summary = record["summary"]
        unsafe = summary["unsafe_only"]
        table_rows.append(
            {
                "environment": record["environment"],
                "family": record["family"],
                "run_dir": record["run_dir"],
                "sample_states": summary["n_global_states"],
                "violation_agent_state_ratio": summary[
                    "instantaneous_violation_agent_state_ratio"
                ],
                "violation_global_state_ratio": summary[
                    "instantaneous_violation_global_state_ratio"
                ],
                "error_mean": summary["error"]["mean"],
                "error_std": summary["error"]["std"],
                "error_mae": summary["error"]["mae"],
                "error_rmse": summary["error"]["rmse"],
                "unsafe_value_count": summary["trajectory_unsafe_value_count"],
                "unsafe_value_ratio": summary["trajectory_unsafe_value_ratio"],
                "unsafe_error_mean": unsafe["mean_error"],
                "unsafe_error_rmse": unsafe["rmse"],
                "unsafe_underestimate_mean": unsafe["underestimate_mean"],
                "unsafe_false_safe_rate": unsafe["false_safe_rate"],
            }
        )
    with (output / "cross_env_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=table_rows[0].keys())
        writer.writeheader()
        writer.writerows(table_rows)
    with (output / "selected_complete_runs.json").open("w", encoding="utf-8") as file:
        json.dump(table_rows, file, indent=2)
        file.write("\n")


if __name__ == "__main__":
    main()
