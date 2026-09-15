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


def _discover_run_dirs(
    env_log_dir: str, algo: str, step: str | None
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
        if _has_checkpoint(run_dir, step):
            run_dirs.append(run_dir)
        else:
            skipped_dirs.append(seed_dir)

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
    )


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
        required=True,
        help="algorithm subdirectory name, for example dgppo or deepqp",
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
    args = parser.parse_args()

    if args.episodes <= 0:
        parser.error("--episodes must be positive")
    if args.episodes > 1_000:
        parser.error("--episodes must not exceed 1000")

    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    if args.cpu:
        os.environ["JAX_PLATFORM_NAME"] = "cpu"
        os.environ["JAX_PLATFORMS"] = "cpu"

    from test import test as evaluate_run

    run_dirs, skipped_dirs = _discover_run_dirs(
        args.env_log_dir, args.algo, args.step
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

    print("> Aggregate over all evaluation episodes")
    print(f"  episodes={expected_episodes}")
    print(f"  safety_rate_mean={pooled_rates.mean() * 100:.3f}%")
    print(f"  safety_rate_std={pooled_rates.std() * 100:.3f}%")
    print(f"  reach_rate_mean={pooled_reach_rates.mean() * 100:.3f}%")
    print(f"  reach_rate_std={pooled_reach_rates.std() * 100:.3f}%")


if __name__ == "__main__":
    main()
