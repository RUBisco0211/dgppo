"""Measure formation precision for a saved single-task GRAFO actor."""

import argparse
import pathlib
import sys
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from train_grafo_continual import make_algorithm, make_environment, make_eval_fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--reach-threshold", type=float, required=True)
    parser.add_argument("--action-encoding", choices=("continuous", "discrete"),
                        default="continuous")
    parser.add_argument("--episodes", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    env = make_environment(args.task, (args.task,), args.action_encoding,
                           args.reach_threshold)
    train_args = SimpleNamespace(
        regularization="none", importance_mode="multi", seed=args.seed,
        batch_size=512, ppo_epochs=4, lr=5e-4, frog=False,
        regularization_coefficient=0.0)
    algo = make_algorithm(env, train_args, 0, [], [])
    algo.load(args.checkpoint, "model")
    keys = jax.random.split(jax.random.PRNGKey(args.seed + 10_000), args.episodes)
    rollout = make_eval_fn(algo, env)(algo.policy_train_state.params, keys)
    distances = jax.vmap(jax.vmap(env._assigned_distances))(rollout.next_graph)
    max_distance = np.asarray(distances.max(axis=-1))
    final = max_distance[:, -1]
    minimum = max_distance.min(axis=1)
    print({
        "episodes": args.episodes,
        "success_rate": float(np.mean(minimum <= args.reach_threshold)),
        "final_team_max_distance_mean": float(final.mean()),
        "final_team_max_distance_median": float(np.median(final)),
        "closest_team_max_distance_mean": float(minimum.mean()),
        "closest_team_max_distance_median": float(np.median(minimum)),
        "closest_team_max_distance_p90": float(np.percentile(minimum, 90)),
        "per_agent_closest_distance_mean": np.asarray(distances.min(axis=1)).mean(axis=0).tolist(),
        "per_agent_final_distance_mean": np.asarray(distances[:, -1]).mean(axis=0).tolist(),
        "fraction_within_0.35": float(np.mean(minimum <= 0.35)),
        "fraction_within_0.15": float(np.mean(minimum <= 0.15)),
        "fraction_within_0.05": float(np.mean(minimum <= 0.05)),
    })


if __name__ == "__main__":
    main()
