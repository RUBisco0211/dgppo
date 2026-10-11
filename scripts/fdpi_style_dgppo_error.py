#!/usr/bin/env python3
"""FDPI-style post-training feasibility-error evaluation for DGPPO.

This script adapts Figures 3 and 4 of the FDPI paper to this on-policy,
multi-agent implementation.  It reports the final-window training violation
ratio and evaluates the distribution of ``V_h prediction - trajectory target``
on independently collected trajectories.  The trajectory target follows the
discounted max Bellman operator used by ``compute_dec_ocp_gae``; it is not the
older, undiscounted future-max proxy.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import re
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import yaml

from dgppo.algo import make_algo
from dgppo.env import make_env


def _cfg(config: Any, name: str, default: Any = None) -> Any:
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


def _load_config(run_dir: Path) -> Any:
    path = run_dir / "config.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"missing config: {path}")
    with path.open("r", encoding="utf-8") as file:
        config = yaml.load(file, Loader=yaml.UnsafeLoader)
    if _cfg(config, "algo") != "dgppo":
        raise ValueError(f"expected a DGPPO run, got algo={_cfg(config, 'algo')!r}")
    return config


def _make_dgppo(config: Any):
    env = make_env(
        env_id=_cfg(config, "env"),
        num_agents=int(_cfg(config, "num_agents")),
        num_obs=int(_cfg(config, "obs")),
        n_rays=int(_cfg(config, "n_rays", 32)),
        full_observation=bool(_cfg(config, "full_observation", False)),
    )
    algo = make_algo(
        algo="dgppo",
        env=env,
        node_dim=env.node_dim,
        edge_dim=env.edge_dim,
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        n_agents=env.num_agents,
        actor_gnn_layers=int(_cfg(config, "actor_gnn_layers", 2)),
        Vl_gnn_layers=int(_cfg(config, "Vl_gnn_layers", 2)),
        Vh_gnn_layers=int(_cfg(config, "Vh_gnn_layers", 1)),
        gamma=float(_cfg(config, "gamma", 0.99)),
        lr_actor=float(_cfg(config, "lr_actor", 3e-4)),
        lr_Vl=float(_cfg(config, "lr_Vl", 1e-3)),
        lr_Vh=float(_cfg(config, "lr_Vh", 1e-3)),
        batch_size=int(_cfg(config, "batch_size", 8192)),
        epoch_ppo=int(_cfg(config, "epoch_ppo", 1)),
        clip_eps=float(_cfg(config, "clip_eps", 0.25)),
        gae_lambda=float(_cfg(config, "gae_lambda", 0.95)),
        coef_ent=float(_cfg(config, "coef_ent", 1e-2)),
        max_grad_norm=float(_cfg(config, "max_grad_norm", 2.0)),
        seed=int(_cfg(config, "seed", 0)),
        use_rnn=bool(_cfg(config, "use_rnn", not _cfg(config, "no_rnn", False))),
        rnn_layers=int(_cfg(config, "rnn_layers", 1)),
        rnn_step=int(_cfg(config, "rnn_step", 16)),
        use_lstm=bool(_cfg(config, "use_lstm", False)),
        alpha=float(_cfg(config, "alpha", 10.0)),
        cbf_eps=float(_cfg(config, "cbf_eps", 1e-2)),
        cbf_weight=float(_cfg(config, "cbf_weight", 1.0)),
        train_steps=int(_cfg(config, "steps", 100_000)),
        cbf_schedule=bool(_cfg(config, "cbf_schedule", True)),
    )
    return env, algo


def _resolve_checkpoint(run_dir: Path, requested: str) -> tuple[str, Path, int]:
    models = run_dir / "models"
    if requested == "latest":
        path = models / "latest"
        markers = sorted(path.glob("latest_iter_*.txt"))
        step = (
            int(markers[-1].stem.removeprefix("latest_iter_"))
            if markers
            else -1
        )
        return "latest", path, step

    direct = models / requested
    diagnostic = models / "diagnostic" / requested
    if direct.is_dir():
        path = direct
    elif diagnostic.is_dir():
        path = diagnostic
    else:
        raise FileNotFoundError(
            f"checkpoint {requested!r} not found below {models}"
        )
    match = re.search(r"(\d+)$", path.name)
    return path.name, path, int(match.group(1)) if match else -1


def _load_params(checkpoint_dir: Path):
    with (checkpoint_dir / "actor.pkl").open("rb") as file:
        actor_params = pickle.load(file)
    with (checkpoint_dir / "Vh.pkl").open("rb") as file:
        vh_params = pickle.load(file)
    return actor_params, vh_params


def _trajectory_target(costs: np.ndarray, gamma: float) -> np.ndarray:
    """Finite-trajectory target for DGPPO's discounted max Bellman operator."""
    target = np.empty_like(costs)
    next_value = np.zeros_like(costs[:, -1])
    for time_index in range(costs.shape[1] - 1, -1, -1):
        current = costs[:, time_index]
        current_max = current.max(axis=-1, keepdims=True)
        discounted = (1.0 - gamma) * current_max + gamma * next_value
        next_value = np.maximum(current, discounted)
        target[:, time_index] = next_value
    return target


def _make_batch_evaluator(env, algo):
    def evaluate_one(actor_params, vh_params, key):
        reset_key, rollout_key = jax.random.split(key)
        graph = env.reset(reset_key)

        def body(carry, action_key):
            current_graph, actor_state = carry
            input_actor_state = actor_state
            action, next_actor_state = algo.act(
                current_graph,
                input_actor_state,
                {"policy": actor_params},
            )
            prediction = algo.get_Vh(
                current_graph,
                input_actor_state,
                {"Vh": vh_params},
            )
            next_graph, _reward, cost, _done, _info = env.step(
                current_graph, action
            )
            return (next_graph, next_actor_state), (prediction, cost)

        action_keys = jax.random.split(rollout_key, env.max_episode_steps)
        _, (predictions, costs) = jax.lax.scan(
            body,
            (graph, algo.init_rnn_state),
            action_keys,
        )
        return predictions, costs

    return jax.jit(jax.vmap(evaluate_one, in_axes=(None, None, 0)))


def _training_violation_ratio(
    metrics_path: Path, checkpoint_step: int, final_fraction: float
) -> tuple[float, int]:
    if not metrics_path.is_file():
        return float("nan"), 0
    records = []
    with metrics_path.open("r", encoding="utf-8") as file:
        for line in file:
            record = json.loads(line)
            if "collection/safety/violation_rate" not in record:
                continue
            step = int(record.get("counters/iter", len(records)))
            if checkpoint_step < 0 or step <= checkpoint_step:
                records.append((step, float(record["collection/safety/violation_rate"])))
    if not records:
        return float("nan"), 0
    final_step = records[-1][0]
    lower = final_step * (1.0 - final_fraction)
    values = [value for step, value in records if step >= lower]
    return float(np.mean(values)), len(values)


def _quantiles(values: np.ndarray) -> dict[str, float]:
    return {
        str(q): float(np.quantile(values, q))
        for q in (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)
    }


def _summarize(
    predictions: np.ndarray,
    targets: np.ndarray,
    costs: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    error = predictions - targets
    unsafe = targets > threshold
    unsafe_count = int(unsafe.sum())
    false_safe = unsafe & (predictions <= threshold)
    if unsafe_count:
        unsafe_error = error[unsafe]
        unsafe_summary = {
            "valid": True,
            "mean_error": float(unsafe_error.mean()),
            "rmse": float(np.sqrt(np.square(unsafe_error).mean())),
            "underestimate_mean": float(
                np.maximum(-unsafe_error, 0.0).mean()
            ),
            "false_safe_rate": float(false_safe.sum() / unsafe_count),
        }
    else:
        unsafe_summary = {
            "valid": False,
            "mean_error": None,
            "rmse": None,
            "underestimate_mean": None,
            "false_safe_rate": None,
        }
    return {
        "n_global_states": int(predictions.shape[0]),
        "n_agent_constraint_values": int(predictions.size),
        "instantaneous_violation_agent_state_ratio": float(
            (costs.max(axis=-1) > 0.0).mean()
        ),
        "instantaneous_violation_global_state_ratio": float(
            (costs.max(axis=(-1, -2)) > 0.0).mean()
        ),
        "trajectory_unsafe_value_ratio": float(unsafe.mean()),
        "trajectory_unsafe_value_count": unsafe_count,
        "error": {
            "definition": "Vh_prediction_minus_discounted_max_trajectory_target",
            "mean": float(error.mean()),
            "std": float(error.std()),
            "mae": float(np.abs(error).mean()),
            "rmse": float(np.sqrt(np.square(error).mean())),
            "quantiles": _quantiles(error.reshape(-1)),
        },
        "unsafe_only": unsafe_summary,
    }


def _plot(results: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    figure, (error_axis, ratio_axis) = plt.subplots(1, 2, figsize=(12, 4.5))
    all_errors = np.concatenate([result["raw_error"] for result in results])
    limit = max(float(np.quantile(np.abs(all_errors), 0.995)), 1e-6)
    bins = np.linspace(-limit, limit, 161)
    for result in results:
        error_axis.hist(
            result["raw_error"],
            bins=bins,
            density=True,
            histtype="step",
            linewidth=1.6,
            label=result["label"],
        )
    error_axis.axvline(0.0, color="black", linewidth=1.0, linestyle="--")
    error_axis.set_xlabel("Vh prediction - trajectory target")
    error_axis.set_ylabel("Density")
    error_axis.set_title("FDPI-style feasibility error distribution")
    error_axis.legend()
    error_axis.grid(alpha=0.2)

    labels = [result["label"] for result in results]
    ratios = [result["training_violation_ratio"] for result in results]
    ratio_axis.bar(labels, ratios)
    ratio_axis.set_ylabel("Violation-transition ratio")
    ratio_axis.set_title("Final 10% of training before checkpoint")
    ratio_axis.tick_params(axis="x", rotation=25)
    ratio_axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_dir / "fdpi_style_comparison.png", dpi=180)
    plt.close(figure)

    figure3, axis3 = plt.subplots(figsize=(6.4, 4.5))
    axis3.bar(labels, ratios)
    axis3.set_ylabel("Violation-transition ratio")
    axis3.set_title("Final 10% training violation-sample ratio")
    axis3.tick_params(axis="x", rotation=25)
    axis3.grid(axis="y", alpha=0.2)
    figure3.tight_layout()
    figure3.savefig(output_dir / "figure3_violation_sample_ratio.png", dpi=180)
    plt.close(figure3)

    figure4, axis4 = plt.subplots(figsize=(7.2, 4.5))
    for result in results:
        axis4.hist(
            result["raw_error"],
            bins=bins,
            density=True,
            histtype="step",
            linewidth=1.6,
            label=result["label"],
        )
    axis4.axvline(0.0, color="black", linewidth=1.0, linestyle="--")
    axis4.set_xlabel("Vh prediction - trajectory target")
    axis4.set_ylabel("Density")
    axis4.set_title("Feasibility-function estimation error")
    axis4.legend()
    axis4.grid(alpha=0.2)
    figure4.tight_layout()
    figure4.savefig(output_dir / "figure4_vh_error_distribution.png", dpi=180)
    plt.close(figure4)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=None,
        help="latest or a diagnostic snapshot name; repeat to compare checkpoints",
    )
    parser.add_argument("--sample-states", type=int, default=100_000)
    parser.add_argument("--batch-rollouts", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--final-training-fraction", type=float, default=0.1)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.sample_states <= 0 or args.batch_rollouts <= 0:
        parser.error("--sample-states and --batch-rollouts must be positive")
    if not 0.0 < args.final_training_fraction <= 1.0:
        parser.error("--final-training-fraction must be in (0, 1]")

    run_dir = args.run_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = _load_config(run_dir)
    env, algo = _make_dgppo(config)
    gamma = float(_cfg(config, "gamma", 0.99))
    threshold = float(_cfg(config, "cbf_eps", 1e-2))
    checkpoints = args.checkpoint or ["latest"]
    n_rollouts = math.ceil(args.sample_states / env.max_episode_steps)
    all_keys = jax.random.split(jax.random.PRNGKey(args.seed), n_rollouts)
    evaluate_batch = _make_batch_evaluator(env, algo)
    results = []

    for requested in checkpoints:
        label, checkpoint_dir, checkpoint_step = _resolve_checkpoint(
            run_dir, requested
        )
        actor_params, vh_params = _load_params(checkpoint_dir)
        prediction_batches = []
        cost_batches = []
        for start in range(0, n_rollouts, args.batch_rollouts):
            keys = all_keys[start:start + args.batch_rollouts]
            predictions, costs = evaluate_batch(actor_params, vh_params, keys)
            prediction_batches.append(np.asarray(predictions))
            cost_batches.append(np.asarray(costs))
        predictions = np.concatenate(prediction_batches, axis=0)
        costs = np.concatenate(cost_batches, axis=0)
        targets = _trajectory_target(costs, gamma)
        predictions = predictions.reshape((-1,) + predictions.shape[2:])[
            :args.sample_states
        ]
        costs = costs.reshape((-1,) + costs.shape[2:])[:args.sample_states]
        targets = targets.reshape((-1,) + targets.shape[2:])[:args.sample_states]
        error = predictions - targets
        training_ratio, training_records = _training_violation_ratio(
            run_dir / "training_metrics.jsonl",
            checkpoint_step,
            args.final_training_fraction,
        )
        summary = _summarize(predictions, targets, costs, threshold)
        summary |= {
            "checkpoint": label,
            "checkpoint_step": checkpoint_step,
            "gamma": gamma,
            "feasibility_threshold": threshold,
            "training_final_window_fraction": args.final_training_fraction,
            "training_final_window_records": training_records,
            "training_violation_transition_ratio": training_ratio,
        }
        np.savez_compressed(
            output_dir / f"{label}_samples.npz",
            prediction=predictions,
            trajectory_target=targets,
            error=error,
            instantaneous_cost=costs,
        )
        with (output_dir / f"{label}_summary.json").open(
            "w", encoding="utf-8"
        ) as file:
            json.dump(summary, file, indent=2, sort_keys=True)
            file.write("\n")
        results.append(
            {
                "label": label,
                "raw_error": error.reshape(-1),
                "training_violation_ratio": training_ratio,
                "summary": summary,
            }
        )
        print(json.dumps(summary, indent=2, sort_keys=True))

    _plot(results, output_dir)
    manifest = {
        "methodology": (
            "FDPI-style adaptation: independently collected deterministic-policy "
            "trajectories, DGPPO discounted-max finite-trajectory targets, and "
            "the final-window on-policy violation ratio. This is not CDF gamma^N."
        ),
        "run_dir": str(run_dir),
        "sample_states_per_checkpoint": args.sample_states,
        "seed": args.seed,
        "checkpoints": [result["summary"] for result in results],
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, sort_keys=True)
        file.write("\n")


if __name__ == "__main__":
    main()
