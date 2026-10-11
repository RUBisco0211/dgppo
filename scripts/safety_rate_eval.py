#!/usr/bin/env python3
"""Evaluate safety and task reach rates over all discovered training seeds."""

import argparse
import contextlib
import io
import os
from pathlib import Path
import sys

import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _resolve_run_dir(path: str) -> Path:
    """Resolve a standard run or an outer two-stage Deep-QP run."""
    run_dir = Path(path).expanduser().resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"run directory does not exist: {run_dir}")

    if run_dir.name == "models":
        run_dir = run_dir.parent
    elif (run_dir / "actor.pkl").is_file() and run_dir.parent.name == "models":
        run_dir = run_dir.parent.parent

    if (run_dir / "config.yaml").is_file() and (run_dir / "models").is_dir():
        return run_dir

    candidates = sorted(
        config_path.parent
        for config_path in (run_dir / "rl").glob("**/config.yaml")
        if (config_path.parent / "models").is_dir()
    )
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(
            f"could not find config.yaml and models/ below run directory: {run_dir}"
        )
    raise ValueError(
        f"multiple RL runs found below {run_dir}; pass one of these directly: "
        + ", ".join(str(path) for path in candidates)
    )


def _load_config(run_dir: Path):
    with (run_dir / "config.yaml").open("r", encoding="utf-8") as file:
        return yaml.load(file, Loader=yaml.UnsafeLoader)


def _validate_runs(run_dirs: list[Path]) -> list:
    configs = [_load_config(run_dir) for run_dir in run_dirs]
    env_ids = [config.env for config in configs]
    unsupported = [env_id for env_id in env_ids if not env_id.startswith(("MPE", "Lidar"))]
    if unsupported:
        raise ValueError(
            "this evaluator supports only MPE and Lidar environments; got: "
            + ", ".join(unsupported)
        )
    signatures = {
        (config.env, config.algo, int(config.num_agents), int(config.obs))
        for config in configs
    }
    if len(signatures) != 1:
        raise ValueError(
            "all discovered runs must use the same environment, algorithm, number "
            "of agents, and number of obstacles"
        )
    return configs


def _has_checkpoint(run_dir: Path, step: str | None) -> bool:
    model_dir = run_dir / "models"
    if step is not None:
        return (model_dir / step / "actor.pkl").is_file()
    if (model_dir / "latest" / "actor.pkl").is_file():
        return True
    return any(
        path.is_dir() and path.name.isdigit() and (path / "actor.pkl").is_file()
        for path in model_dir.iterdir()
    )


def _checkpoint_progress(run_dir: Path, step: str | None) -> int | None:
    model_dir = run_dir / "models"
    if step is None:
        if (model_dir / "latest").is_dir():
            checkpoint_dir = model_dir / "latest"
        else:
            numeric_steps = [
                int(path.name)
                for path in model_dir.iterdir()
                if path.is_dir() and path.name.isdigit()
            ]
            return max(numeric_steps, default=None)
    else:
        checkpoint_dir = model_dir / step

    markers = list(checkpoint_dir.glob("latest_iter_*.txt"))
    if markers:
        return max(int(path.stem.rsplit("_", 1)[-1]) for path in markers)
    if checkpoint_dir.name.isdigit():
        return int(checkpoint_dir.name)
    return None


def _discover_run_dirs(
    env_log_dir: str, algo: str, step: str | None, complete_only: bool
) -> tuple[list[Path], list[Path]]:
    env_dir = Path(env_log_dir).expanduser().resolve()
    if not env_dir.is_dir():
        raise FileNotFoundError(f"environment log directory does not exist: {env_dir}")

    algo_dir = env_dir / algo
    if not algo_dir.is_dir():
        raise FileNotFoundError(f"algorithm log directory does not exist: {algo_dir}")

    seed_dirs = sorted(
        path for path in algo_dir.iterdir()
        if path.is_dir() and path.name.startswith("seed")
    )
    if not seed_dirs:
        raise FileNotFoundError(f"no seed directories found below: {algo_dir}")

    run_dirs = []
    skipped_dirs = []
    for seed_dir in seed_dirs:
        try:
            run_dir = _resolve_run_dir(str(seed_dir))
        except FileNotFoundError:
            skipped_dirs.append(seed_dir)
            continue
        if not _has_checkpoint(run_dir, step):
            skipped_dirs.append(seed_dir)
            continue
        if complete_only:
            config = _load_config(run_dir)
            progress = _checkpoint_progress(run_dir, step)
            if progress is None or progress < int(config.steps):
                skipped_dirs.append(seed_dir)
                continue
        run_dirs.append(run_dir)

    if not run_dirs:
        checkpoint = step if step is not None else "latest or numeric"
        raise FileNotFoundError(
            f"no seed contains a usable {checkpoint} checkpoint below: {algo_dir}"
        )
    return run_dirs, skipped_dirs


def _evaluation_args(args, run_dir: Path) -> argparse.Namespace:
    return argparse.Namespace(
        path=str(run_dir),
        no_video=True,
        epi=args.episodes,
        step=args.step,
        obs=None,
        stochastic=args.stochastic,
        full_observation=args.full_observation,
        debug=False,
        cpu=args.cpu,
        max_step=args.max_step,
        log=False,
        num_agents=None,
        seed=args.eval_seed,
        env=None,
        offset=0,
        dpi=100,
        policy_only=True,
        reach_threshold=args.reach_threshold,
    )


def _display_name(algo: str) -> str:
    return {"dgppo": "DGPPO", "deepqp": "Deep-QP"}.get(algo, algo)


def _plot_summaries(env_name: str, summaries: list[dict], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    markers = ("o", "s", "^", "D", "P", "X")
    fig, ax = plt.subplots(figsize=(3.4, 3.0), dpi=200)
    for index, summary in enumerate(summaries):
        ax.errorbar(
            summary["reach_mean"],
            summary["safety_mean"],
            xerr=summary["reach_std"],
            yerr=summary["safety_std"],
            marker=markers[index % len(markers)],
            markersize=5,
            capsize=3,
            linewidth=1.2,
            label=summary["label"],
        )

    reach_bounds = [
        value
        for summary in summaries
        for value in (
            summary["reach_mean"] - summary["reach_std"],
            summary["reach_mean"] + summary["reach_std"],
        )
    ]
    safety_bounds = [
        value
        for summary in summaries
        for value in (
            summary["safety_mean"] - summary["safety_std"],
            summary["safety_mean"] + summary["safety_std"],
        )
    ]

    def padded_limits(bounds: list[float]) -> tuple[float, float]:
        lower = min(0.0, *bounds)
        upper = max(1.0, *bounds)
        padding = max(0.03, 0.05 * (upper - lower))
        return lower - padding, upper + padding

    ax.set(xlabel="Reach rate", ylabel="Safety rate", title=env_name)
    ax.set_xlim(*padded_limits(reach_bounds))
    ax.set_ylim(*padded_limits(safety_bounds))
    ax.set_xticks(np.linspace(0.0, 1.0, 6))
    ax.set_yticks(np.linspace(0.0, 1.0, 6))
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)
    ax.legend(frameon=False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _evaluate_algorithm(args, algo_name: str) -> dict:
    from test import test as evaluate_run

    run_dirs, skipped_dirs = _discover_run_dirs(
        args.env_log_dir, algo_name, args.step, args.complete_only
    )
    for skipped_dir in skipped_dirs:
        print(f"> Skipping seed without a usable checkpoint: {skipped_dir}")
    configs = _validate_runs(run_dirs)
    all_episode_rates = []
    all_episode_reach_rates = []

    print(
        f"> Environment: {configs[0].env}, algorithm: {configs[0].algo}, "
        f"agents: {configs[0].num_agents}, "
        f"training seeds: {len(run_dirs)}, "
        f"initial conditions per seed: {args.episodes}"
    )
    for index, (run_dir, config) in enumerate(zip(run_dirs, configs)):
        print(f"> Evaluating seed {index + 1}/{len(run_dirs)}: {run_dir}")
        with contextlib.redirect_stdout(io.StringIO()):
            result = evaluate_run(_evaluation_args(args, run_dir))
        episode_rates = np.asarray(result["episode_safe_rates"], dtype=np.float64)
        episode_reach_rates = np.asarray(
            result["episode_reach_rates"], dtype=np.float64
        )
        all_episode_rates.append(episode_rates)
        all_episode_reach_rates.append(episode_reach_rates)
        print(
            f"  training_seed={config.seed}, "
            f"safety_rate={episode_rates.mean() * 100:.3f}% "
            f"± {episode_rates.std() * 100:.3f}%, "
            f"reach_rate={episode_reach_rates.mean() * 100:.3f}% "
            f"± {episode_reach_rates.std() * 100:.3f}%"
        )

    pooled_rates = np.concatenate(all_episode_rates)
    pooled_reach_rates = np.concatenate(all_episode_reach_rates)
    expected_episodes = len(run_dirs) * args.episodes
    if pooled_rates.shape != (expected_episodes,):
        raise RuntimeError(
            f"expected {expected_episodes} episode safety rates, got {pooled_rates.shape}"
        )
    if pooled_reach_rates.shape != (expected_episodes,):
        raise RuntimeError(
            f"expected {expected_episodes} episode reach rates, "
            f"got {pooled_reach_rates.shape}"
        )

    summary = {
        "env": configs[0].env,
        "label": _display_name(algo_name),
        "episodes": expected_episodes,
        "safety_mean": float(pooled_rates.mean()),
        "safety_std": float(pooled_rates.std()),
        "reach_mean": float(pooled_reach_rates.mean()),
        "reach_std": float(pooled_reach_rates.std()),
    }
    print(f"> Aggregate over all evaluation episodes ({summary['label']})")
    print(f"  episodes={expected_episodes}")
    print(f"  safety_rate_mean={summary['safety_mean'] * 100:.3f}%")
    print(f"  safety_rate_std={summary['safety_std'] * 100:.3f}%")
    print(f"  reach_rate_mean={summary['reach_mean'] * 100:.3f}%")
    print(f"  reach_rate_std={summary['reach_std'] * 100:.3f}%")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Discover every seed under an environment/algorithm log directory, "
            "evaluate them on the same initial conditions, and report safety "
            "and task reach rates."
        )
    )
    parser.add_argument(
        "--env-log-dir",
        required=True,
        help="environment log directory, for example logs/MPESpread",
    )
    parser.add_argument(
        "--algo",
        nargs="+",
        required=True,
        help="algorithm subdirectory names, for example dgppo deepqp",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=32,
        help="evaluation initial conditions per trained seed (default: 32)",
    )
    parser.add_argument(
        "--eval-seed",
        type=int,
        default=1234,
        help="seed used to generate the shared evaluation initial conditions",
    )
    parser.add_argument(
        "--step",
        default=None,
        help="checkpoint directory name for every run (default: latest)",
    )
    parser.add_argument("--max-step", type=int, default=None)
    parser.add_argument("--stochastic", action="store_true", default=False)
    parser.add_argument("--full-observation", action="store_true", default=False)
    parser.add_argument("--cpu", action="store_true", default=False)
    parser.add_argument(
        "--complete-only",
        action="store_true",
        default=False,
        help="evaluate only checkpoints whose iteration reached config.steps",
    )
    parser.add_argument(
        "--plot-path",
        default=None,
        help=(
            "comparison PNG path (default: "
            "<env-log-dir>/safety_reach_comparison.png)"
        ),
    )
    parser.add_argument(
        "--reach-threshold",
        type=float,
        default=None,
        help="goal-distance threshold (default: environment dist2goal)",
    )
    args = parser.parse_args()

    if args.episodes <= 0:
        parser.error("--episodes must be positive")
    if args.episodes > 1_000:
        parser.error("--episodes must not exceed 1000")
    if args.reach_threshold is not None and args.reach_threshold <= 0.0:
        parser.error("--reach-threshold must be positive")

    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    if args.cpu:
        os.environ["JAX_PLATFORM_NAME"] = "cpu"
        os.environ["JAX_PLATFORMS"] = "cpu"

    summaries = [_evaluate_algorithm(args, algo) for algo in args.algo]
    env_names = {summary["env"] for summary in summaries}
    if len(env_names) != 1:
        raise ValueError("all algorithm directories must belong to the same environment")

    plot_path = (
        Path(args.plot_path).expanduser().resolve()
        if args.plot_path is not None
        else Path(args.env_log_dir).expanduser().resolve()
        / "safety_reach_comparison.png"
    )
    _plot_summaries(summaries[0]["env"], summaries, plot_path)
    print(f"> Plot saved to: {plot_path}")


if __name__ == "__main__":
    main()
