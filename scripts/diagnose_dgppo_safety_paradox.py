#!/usr/bin/env python3
"""Diagnose safety-paradox signatures in an instrumented DGPPO JSONL log.

The script does not infer the phenomenon from training loss alone.  It requires
the held-out rollout metrics emitted by ``DGPPO.diagnostic_metrics`` and checks
whether violating training transitions become rarer while empirical future-max
error, underestimation, or false negatives on held-out unsafe examples worsen.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PREFIX = "eval/safety_paradox/"
REQUIRED = (
    "future_unsafe_rate",
    "future_unsafe_count",
    "future_max_mse_unsafe",
    "unsafe_underestimate_mean",
    "dgcbf_false_negative_rate",
)
VIOLATION_RATE = "collection/safety/violation_rate"


def _load(path: Path) -> list[dict[str, float]]:
    records = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from error
            if VIOLATION_RATE in record and all(PREFIX + key in record for key in REQUIRED):
                records.append(record)
    return records


def _mean(records: list[dict], key: str) -> float:
    return float(np.mean([float(record[PREFIX + key]) for record in records]))


def _relative_change(old: float, new: float) -> float:
    return (new - old) / max(abs(old), 1e-12)


def _correlation(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.std(x) == 0.0 or np.std(y) == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def analyze(records: list[dict], window: int, min_dangerous: int) -> dict:
    window = min(window, max(1, len(records) // 3))
    early = records[:window]
    late = records[-window:]
    violation_early = float(np.mean([record[VIOLATION_RATE] for record in early]))
    violation_late = float(np.mean([record[VIOLATION_RATE] for record in late]))
    future_unsafe_early = _mean(early, "future_unsafe_rate")
    future_unsafe_late = _mean(late, "future_unsafe_rate")
    count_late = _mean(late, "future_unsafe_count")

    error_keys = (
        "future_max_mse_unsafe",
        "unsafe_underestimate_mean",
        "dgcbf_false_negative_rate",
    )
    changes = {
        key: {
            "early": _mean(early, key),
            "late": _mean(late, key),
            "relative_change": _relative_change(_mean(early, key), _mean(late, key)),
        }
        for key in error_keys
    }
    violation = np.asarray([record[VIOLATION_RATE] for record in records], dtype=float)
    correlations = {
        key: _correlation(
            violation,
            np.asarray([record[PREFIX + key] for record in records], dtype=float),
        )
        for key in error_keys
    }

    scarcity = violation_late <= 0.5 * violation_early
    degradation = any(value["relative_change"] >= 0.2 for value in changes.values())
    enough_dangerous = count_late >= min_dangerous
    if not enough_dangerous:
        verdict = "inconclusive_too_few_future_unsafe_samples"
    elif scarcity and degradation:
        verdict = "safety_paradox_signature_detected"
    elif scarcity:
        verdict = "unsafe_samples_became_sparse_but_no_error_degradation_detected"
    else:
        verdict = "no_clear_unsafe_sample_scarcity"

    return {
        "verdict": verdict,
        "methodology": (
            "Signature test only: scarcity uses on-policy training violation transitions; "
            "error uses a finite-horizon empirical future-max target on fixed-seed eval "
            "rollouts. This is not a proof of the FDPI CDF error theorem."
        ),
        "n_evaluations": len(records),
        "window": window,
        "training_violation_rate": {"early": violation_early, "late": violation_late},
        "future_unsafe_rate": {
            "early": future_unsafe_early,
            "late": future_unsafe_late,
        },
        "late_future_unsafe_count": count_late,
        "error_changes": changes,
        "training_violation_rate_error_correlations": correlations,
        "criteria": {
            "scarcity": scarcity,
            "degradation": degradation,
            "enough_late_dangerous_samples": enough_dangerous,
        },
    }


def plot(records: list[dict], output: Path) -> None:
    steps = np.asarray([record.get("counters/iter", index) for index, record in enumerate(records)])
    series = (
        (None, "Training violation-transition rate"),
        ("future_max_mse_unsafe", "Future-max MSE on future-unsafe"),
        ("unsafe_underestimate_mean", "Unsafe underestimation"),
        ("dgcbf_false_negative_rate", "DGCBF-threshold false-negative rate"),
    )
    figure, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for axis, (key, label) in zip(axes.flat, series):
        metric_key = VIOLATION_RATE if key is None else PREFIX + key
        values = np.asarray([record[metric_key] for record in records], dtype=float)
        axis.plot(steps, values, linewidth=1.5)
        axis.set_title(label)
        axis.grid(alpha=0.25)
    axes[1, 0].set_xlabel("Training iteration")
    axes[1, 1].set_xlabel("Training iteration")
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path, help="DGPPO training_metrics.jsonl")
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--min-dangerous", type=int, default=20)
    parser.add_argument("--output", type=Path, default=None, help="optional JSON report")
    parser.add_argument("--plot", type=Path, default=None, help="optional PNG plot")
    args = parser.parse_args()
    if args.window <= 0 or args.min_dangerous < 0:
        parser.error("--window must be positive and --min-dangerous non-negative")

    records = _load(args.log)
    if len(records) < 3:
        raise SystemExit(
            "No usable safety-paradox diagnostics found. Retrain DGPPO with the "
            "instrumented code and eval_interval small enough to yield at least 3 evaluations."
        )
    report = analyze(records, args.window, args.min_dangerous)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    if args.plot is not None:
        plot(records, args.plot)


if __name__ == "__main__":
    main()
