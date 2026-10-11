#!/usr/bin/env python3
"""Paired CPU evaluation of handcrafted MPE CBF responsibility allocations.

Use a PD nominal controller or --path RUN_DIR for an existing PPO actor.
LP/QP calls run in a Python loop; only policy/environment calls are JITted.
"""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import pickle
import sys
import time

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import linear_sum_assignment
import yaml

from dgppo.algo.module.cbf_allocation import AllocatedCBFFilter
from dgppo.algo.module.mpe_cbf import MPECBFConfig, MPEPairwiseCBF
from dgppo.env import make_env


def goals_for(env, graph):
    derived = env.render_task_goal_positions(graph)
    return np.asarray(derived if derived is not None else graph.type_states(env.GOAL, env.num_goals)[:, :2])


def goal_distances(env, graph):
    positions = np.asarray(graph.type_states(env.AGENT, env.num_agents))[:, :2]
    goals = goals_for(env, graph)
    if type(env).__name__ == "MPETarget":
        return np.linalg.norm(positions - goals, axis=-1)
    return np.linalg.norm(goals[:, None] - positions[None], axis=-1).min(axis=1)


def load_actor(path, step, config, env):
    """Load only the existing actor, with the same architecture/carry as PPO."""
    from dgppo.algo.module.policy import PPOPolicy

    if config["algo"] not in ("dgppo", "hcbfcrpo", "informarl", "informarl_cbf", "informarl_eval_cbf", "informarl_lagr", "informarl_deep_qp"):
        raise ValueError("Checkpoint must contain a standard DGPPO/InforMARL PPO actor")
    models = path / "models"
    if step is None:
        if (models / "latest" / "actor.pkl").is_file():
            step = "latest"
        else:
            candidates = [int(p.name) for p in models.iterdir()
                          if p.name.isdigit() and (p / "actor.pkl").is_file()]
            if not candidates:
                raise FileNotFoundError(f"No actor checkpoint in {models}")
            step = str(max(candidates))
    checkpoint = models / str(step) / "actor.pkl"
    with checkpoint.open("rb") as file:
        params = pickle.load(file)
    layers = config.get("rnn_layers", 1)
    policy = PPOPolicy(node_dim=env.node_dim, edge_dim=env.edge_dim, n_agents=env.num_agents,
                       action_dim=env.action_dim, use_rnn=config.get("use_rnn", True),
                       rnn_layers=layers, gnn_layers=config.get("actor_gnn_layers", 2),
                       gnn_out_dim=64, use_lstm=config.get("use_lstm", False))
    keys = jax.random.split(jax.random.PRNGKey(config.get("seed", 0)), env.num_agents)
    carry = jax.vmap(policy.initialize_carry)(keys)
    carry = jnp.stack(carry, axis=1) if isinstance(carry, tuple) else carry[:, None, :]
    carry = jnp.repeat(carry[None], layers, axis=0)
    actor = jax.jit(lambda graph, state: policy.get_action(params, graph, state))
    return actor, carry, str(checkpoint.resolve())


def finite_min(values):
    value = min(values, default=np.inf)
    return float(value) if np.isfinite(value) else None


def evaluate_episode(env, graph0, mode, args, config, actor, carry0, step_fn, cost_fn):
    filter_ = None if mode == "none" else MPEPairwiseCBF(
        env, mode, config, slack_penalty=args.slack_penalty, witness_fallback=args.witness_fallback)
    diagnostic = MPEPairwiseCBF(env, config=config)
    initial_rows = diagnostic.rows(graph0)
    if filter_ is not None:
        # Warm the fixed-shape QPs and LP machinery before measuring latency.
        filter_(graph0, np.zeros((env.num_agents, 2)))
    graph, carry = graph0, carry0
    initial_states = np.asarray(graph.type_states(env.AGENT, env.num_agents))
    goals = goals_for(env, graph)
    if type(env).__name__ != "MPETarget":
        _, assignment = linear_sum_assignment(np.linalg.norm(initial_states[:, None, :2] - goals[None], axis=-1))
        goals = goals[assignment]
    unsafe_ever = np.any(np.asarray(cost_fn(graph)) >= 0, axis=-1)
    reached_ever = goal_distances(env, graph) <= args.reach_threshold
    rewards, times, modifications, reserves, margins, violations, all_h = [], [], [], [], [], [], []
    infeasible = solver_failed = fallback = collision_steps = clipped_steps = 0
    all_h.extend(initial_rows.h.tolist())
    for _ in range(args.steps):
        if actor is None:
            state = np.asarray(graph.type_states(env.AGENT, env.num_agents))
            nominal = np.clip((args.kp * (goals - state[:, :2]) - args.kd * state[:, 2:]) / 10, -1, 1)
        else:
            nominal, carry = actor(graph, carry)
            nominal = np.asarray(nominal)  # synchronization outside the filter timer
        start = time.perf_counter()
        if filter_ is None:
            action, info = nominal, None
        else:
            action, info = filter_(graph, nominal)
        times.append(1000 * (time.perf_counter() - start))
        modifications.append(float(np.linalg.norm(action - nominal, axis=-1).mean()))
        if info is not None:
            reserves.append(info["reserve"])
            margins.append(info["allocation_margin"])
            violations.append(info["max_violation"])
            infeasible += int(info["infeasible"])
            solver_failed += int(info["solver_failed"])
            fallback += int(info["fallback"])
        # State clipping can invalidate the continuous dynamics used by CBFs.
        old = np.asarray(graph.type_states(env.AGENT, env.num_agents))
        predicted = old + env.dt * np.concatenate([old[:, 2:], 10 * action], axis=1)
        next_graph, reward, *_ = step_fn(graph, jnp.asarray(action, dtype=jnp.float32))
        new = np.asarray(next_graph.type_states(env.AGENT, env.num_agents))
        clipped_steps += int(not np.allclose(new, predicted, atol=1e-6, rtol=0))
        graph = next_graph
        rewards.append(float(reward))
        unsafe = np.any(np.asarray(cost_fn(graph)) >= 0, axis=-1)
        unsafe_ever |= unsafe
        collision_steps += int(unsafe.any())
        reached_ever |= goal_distances(env, graph) <= args.reach_threshold
        all_h.extend(diagnostic.rows(graph).h.tolist())
    return {
        "mode": mode, "reward": sum(rewards), "safe_agent_rate": float((~unsafe_ever).mean()),
        "collision_run": bool(unsafe_ever.any()), "collision_step_rate": collision_steps / args.steps,
        "reach_ever_rate": float(reached_ever.mean()),
        "reach_final_rate": float((goal_distances(env, graph) <= args.reach_threshold).mean()),
        "final_goal_distance": float(goal_distances(env, graph).mean()),
        "infeasible_step_rate": infeasible / args.steps if filter_ else None,
        "solver_failed_steps": solver_failed, "fallback_steps": fallback,
        "state_clipped_step_rate": clipped_steps / args.steps,
        "mean_action_correction": float(np.mean(modifications)),
        "filter_ms_mean": float(np.mean(times)), "filter_ms_p95": float(np.percentile(times, 95)),
        "min_reserve": finite_min(reserves), "min_allocation_margin": finite_min(margins),
        "max_constraint_violation": max(violations, default=None), "min_h": finite_min(all_h),
        "initial_hocbf_admissible": bool(np.all(initial_rows.h >= 0) and np.all(initial_rows.psi1 >= 0)),
        "initial_constraint_count": len(initial_rows.h),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", choices=["MPETarget", "MPESpread", "MPELine", "MPEFormation", "MPECorridor"])
    parser.add_argument("--path", type=Path, help="Training run containing config.yaml and models/")
    parser.add_argument("--step", help="Actor checkpoint directory, default latest or greatest integer")
    parser.add_argument("--num-agents", type=int)
    parser.add_argument("--obs", type=int)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--modes", nargs="+", choices=["none", *AllocatedCBFFilter.MODES],
                        default=["none", "uniform", "capability", "certificate", "centralized"])
    parser.add_argument("--alpha1", type=float, default=3)
    parser.add_argument("--alpha2", type=float, default=3)
    parser.add_argument("--safety-margin", type=float, default=0)
    parser.add_argument("--sensing-radius", type=float, default=None,
                        help="Experimental distance mask; default uses all geometry")
    parser.add_argument("--slack-penalty", type=float, default=1e6)
    parser.add_argument("--witness-fallback", action="store_true")
    parser.add_argument("--reach-threshold", type=float, default=None)
    parser.add_argument("--kp", type=float, default=3, help="PD acceleration gain")
    parser.add_argument("--kd", type=float, default=3.6, help="PD acceleration damping")
    parser.add_argument("--output", type=Path, default=Path("artifacts/mpe_cbf/evaluation.json"))
    args = parser.parse_args()
    if args.episodes < 1 or args.steps < 1:
        parser.error("episodes and steps must be positive")
    training = {}
    if args.path:
        with (args.path / "config.yaml").open() as file:
            loaded = yaml.load(file, Loader=yaml.UnsafeLoader)
        training = loaded if isinstance(loaded, dict) else vars(loaded)
    env_name = args.env or training.get("env", "MPETarget")
    agents = args.num_agents if args.num_agents is not None else training.get("num_agents", 4)
    obs = args.obs if args.obs is not None else training.get("obs", 3)
    if agents < 1 or obs < 0:
        parser.error("num-agents must be positive and obs nonnegative")
    env = make_env(env_name, agents, num_obs=obs, max_step=args.steps,
                   full_observation=training.get("full_observation", False))
    config = MPECBFConfig(args.alpha1, args.alpha2, args.safety_margin, args.sensing_radius)
    MPEPairwiseCBF(env, config=config)  # reject unsupported dynamics/tasks before rollouts
    args.reach_threshold = env.params["dist2goal"] if args.reach_threshold is None else args.reach_threshold
    if args.reach_threshold <= 0:
        parser.error("reach-threshold must be positive")
    actor, carry, checkpoint = None, None, None
    if args.path:
        actor, carry, checkpoint = load_actor(args.path, args.step, training, env)
    reset_fn, step_fn, cost_fn = jax.jit(env.reset), jax.jit(env.step), jax.jit(env.get_cost)
    episodes = []
    for episode in range(args.episodes):
        seed = args.seed + episode
        graph = reset_fn(jax.random.PRNGKey(seed))
        # Warm environment/policy functions; every mode then uses the same graph.
        warm_graph, *_ = step_fn(graph, jnp.zeros((agents, 2)))
        np.asarray(cost_fn(warm_graph))
        if actor:
            np.asarray(actor(graph, carry)[0])
        for mode in args.modes:
            result = evaluate_episode(env, graph, mode, args, config, actor, carry, step_fn, cost_fn)
            result["seed"] = seed
            episodes.append(result)
            print(f"seed={seed} {mode:11s} safe={result['safe_agent_rate']:.3f} "
                  f"reach={result['reach_ever_rate']:.3f} infeasible={result['infeasible_step_rate']} "
                  f"filter_ms={result['filter_ms_mean']:.2f}", flush=True)
    summaries = {}
    for mode in args.modes:
        records = [record for record in episodes if record["mode"] == mode]
        summary = {}
        for key in records[0]:
            if key in ("mode", "seed"):
                continue
            values = [record[key] for record in records if record[key] is not None]
            summary[key] = float(np.mean(values)) if values else None
        summaries[mode] = summary
    payload = {"env": env_name, "num_agents": agents, "n_obs": obs, "checkpoint": checkpoint,
               "cbf_config": asdict(config), "arguments": {k: str(v) if isinstance(v, Path) else v
                                                            for k, v in vars(args).items()},
               "summary": summaries, "episodes": episodes}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(f"Saved {args.output.resolve()}")


if __name__ == "__main__":
    main()
