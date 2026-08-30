"""Pretrain the distributed Graph-HJ critic independently of task reward."""

import argparse
import os
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import numpy as np
from tqdm import tqdm

from dgppo.algo.module.deep_qp_safety import (
    DeepQPSafetyConfig,
    GraphHJSafetyCritic,
)
from dgppo.env import make_env
from dgppo.env.base import MultiAgentEnv
from dgppo.env.safety_constraint import (
    safety_constraint,
    safety_constraint_metadata,
    safety_node_feature_mask,
)
from dgppo.env.vmas.vmas_navigation import VMASNavigation
from dgppo.trainer.data import SafetyBatch
from dgppo.trainer.safety_buffer import SafetyReplayBuffer
from dgppo.utils.utils import tree_where


def _flatten_env_time(tree):
    return jtu.tree_map(lambda x: x.reshape((-1,) + x.shape[2:]), tree)


def _validate_args(args) -> None:
    positive = {
        "steps": args.steps,
        "n_env": args.n_env,
        "rollout_steps": args.rollout_steps,
        "updates_per_collect": args.updates_per_collect,
        "batch_size": args.batch_size,
        "replay_size": args.replay_size,
        "save_interval": args.save_interval,
        "log_interval": args.log_interval,
    }
    invalid = [name for name, value in positive.items() if value <= 0]
    if invalid:
        raise ValueError("these arguments must be positive: " + ", ".join(invalid))
    if args.warmup < 0:
        raise ValueError("warmup must be non-negative")
    if args.replay_size < max(args.warmup, args.batch_size):
        raise ValueError("replay_size must be at least max(warmup, batch_size)")


def _make_collector(
        env: MultiAgentEnv,
        safety_critic: GraphHJSafetyCritic,
        n_env: int,
        rollout_steps: int,
        agent_margin: float,
        obstacle_margin: float,
        braking_accel: float | None,
):
    def constraint_fn(graph):
        return safety_constraint(
            env,
            graph,
            agent_margin=agent_margin,
            obstacle_margin=obstacle_margin,
            braking_accel=braking_accel,
            maximum_margin=safety_critic.config.constraint_scale,
        )

    def collect_one(key):
        reset_key, mean_key, decay_key, scale_key, key = jr.split(key, 5)
        graph = env.reset(reset_key)
        ou_action = jnp.zeros((env.num_agents, env.action_dim), dtype=graph.nodes.dtype)
        ou_mean = jr.uniform(mean_key, ou_action.shape, minval=-0.5, maxval=0.5)
        ou_decay = jr.uniform(decay_key, (), minval=0.70, maxval=0.98)
        ou_scale = jr.uniform(scale_key, (), minval=0.10, maxval=0.50)

        def body(carry, step_key):
            graph, ou_action = carry
            noise_key, uniform_key, bang_key, mix_key, reset_key = jr.split(step_key, 5)
            ou_action = jnp.clip(
                ou_decay * ou_action
                + (1.0 - ou_decay) * ou_mean
                + ou_scale * jr.normal(noise_key, ou_action.shape),
                -1.0,
                1.0,
            )
            uniform_action = jr.uniform(uniform_key, ou_action.shape, minval=-1.0, maxval=1.0)
            bang_action = jnp.where(
                jr.bernoulli(bang_key, 0.5, ou_action.shape),
                jnp.ones_like(ou_action),
                -jnp.ones_like(ou_action),
            )
            mixture = jr.randint(mix_key, (), minval=0, maxval=3)
            raw_action = jnp.where(
                mixture == 0,
                ou_action,
                jnp.where(mixture == 1, uniform_action, bang_action),
            )

            constraint = constraint_fn(graph)
            next_graph, _, _, _, _ = env.step(graph, raw_action)
            next_constraint = constraint_fn(next_graph)
            done = jnp.any(next_constraint < 0.0)

            reset_graph = env.reset(reset_key)
            carry_graph = tree_where(done, reset_graph, next_graph)
            carry_ou_action = jnp.where(done, jnp.zeros_like(ou_action), ou_action)
            outputs = (graph, raw_action, constraint, next_graph, next_constraint, done)
            return (carry_graph, carry_ou_action), outputs

        keys = jr.split(key, rollout_steps)
        _, outputs = jax.lax.scan(body, (graph, ou_action), keys, length=rollout_steps)
        return outputs

    collect_many = jax.vmap(collect_one)

    @jax.jit
    def collect(keys):
        return collect_many(keys)

    def collect_batch(key):
        keys = jr.split(key, n_env)
        graph, actions, constraints, next_graph, next_constraints, dones = collect(
            keys
        )
        graph = graph._replace(env_states=None)
        next_graph = next_graph._replace(env_states=None)
        return SafetyBatch(
            graph=_flatten_env_time(graph),
            actions=_flatten_env_time(actions),
            constraints=_flatten_env_time(constraints),
            next_graph=_flatten_env_time(next_graph),
            next_constraints=_flatten_env_time(next_constraints),
            dones=_flatten_env_time(dones),
        )

    return collect_batch


def train(args):
    _validate_args(args)
    if args.debug:
        jax.config.update("jax_disable_jit", True)
    env = make_env(
        env_id=args.env,
        num_agents=args.num_agents,
        num_obs=args.obs,
        n_rays=args.n_rays,
        max_step=args.rollout_steps,
        full_observation=args.full_observation,
    )
    if not isinstance(env, VMASNavigation):
        raise ValueError(
            "train_safety_filter.py currently targets VMASNavigation and "
            "VMASNavigationObs scenarios"
        )

    config = DeepQPSafetyConfig(
        gnn_layers=args.gnn_layers,
        gnn_out_dim=args.gnn_out_dim,
        hidden_dim=args.hidden_dim,
        hidden_layers=args.hidden_layers,
        learning_rate=args.lr,
        learning_rate_final=args.lr_final,
        max_grad_norm=args.max_grad_norm,
        target_tau=args.tau,
        dt=env.dt,
        lambda_init=args.lambda_init / env.dt,
        lambda_final=args.lambda_final / env.dt,
        lambda_decay_steps=args.lambda_decay_steps,
        constraint_scale=args.constraint_scale,
    )
    lower, upper = env.action_lim()
    safety_critic = GraphHJSafetyCritic(
        env.action_dim,
        env.num_agents,
        lower,
        upper,
        node_feature_mask=safety_node_feature_mask(env),
        config=config,
    )
    key = jr.PRNGKey(args.seed)
    init_key, reset_key, key = jr.split(key, 3)
    state = safety_critic.initialize(init_key, env.reset(reset_key))
    replay = SafetyReplayBuffer(args.replay_size, seed=args.seed)
    checkpoint_metadata = safety_constraint_metadata(
        env,
        agent_margin=args.agent_margin,
        obstacle_margin=args.obstacle_margin,
        braking_accel=args.braking_accel,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.resume is not None:
        state = safety_critic.load_checkpoint(
            state, args.resume, expected_metadata=checkpoint_metadata
        )
        replay_path = Path(args.resume).parent / "deep_qp_replay.pkl"
        if replay_path.exists():
            replay.load(replay_path, expected_n_agents=env.num_agents)

    collect_batch = _make_collector(
        env, safety_critic, args.n_env, args.rollout_steps,
        args.agent_margin, args.obstacle_margin, args.braking_accel
    )
    initial_step = int(np.asarray(state.online.step))
    progress = tqdm(total=args.steps, initial=initial_step, desc="safety updates")
    last_saved_step = initial_step
    while int(np.asarray(state.online.step)) < args.steps:
        collect_key, key = jr.split(key)
        batch = collect_batch(collect_key)
        replay.append(batch)

        info = None
        if replay.length >= max(args.warmup, args.batch_size):
            remaining = args.steps - int(np.asarray(state.online.step))
            n_updates = min(args.updates_per_collect, remaining)
            for _ in range(n_updates):
                state, info = safety_critic.update(state, replay.sample(args.batch_size))
                if bool(np.asarray(info["safety/has_nan"])):
                    raise FloatingPointError(
                        "non-finite Deep-QP safety update; state was left unchanged"
                    )
            progress.update(n_updates)

        current_step = int(np.asarray(state.online.step))
        if info is not None and current_step % args.log_interval < args.updates_per_collect:
            progress.set_postfix(
                loss=float(np.asarray(info["safety/loss"])),
                value=float(np.asarray(info["safety/value_loss"])),
                deriv=float(np.asarray(info["safety/derivative_loss"])),
                replay=replay.length,
            )
        if current_step > 0 and current_step - last_saved_step >= args.save_interval:
            safety_critic.save_checkpoint(
                state, output_dir / "deep_qp_safety.pkl", metadata=checkpoint_metadata
            )
            replay.save(output_dir / "deep_qp_replay.pkl")
            last_saved_step = current_step

    safety_critic.save_checkpoint(
        state, output_dir / "deep_qp_safety.pkl", metadata=checkpoint_metadata
    )
    replay.save(output_dir / "deep_qp_replay.pkl")
    progress.close()
    print(f"> Saved Graph-HJ critic to {output_dir / 'deep_qp_safety.pkl'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="VMASNavigationObs")
    parser.add_argument("-n", "--num-agents", type=int, default=3)
    parser.add_argument("--obs", type=int, default=3)
    parser.add_argument("--n-rays", type=int, default=32)
    parser.add_argument("--full-observation", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=1_000_000)
    parser.add_argument("--n-env", type=int, default=32)
    parser.add_argument("--rollout-steps", type=int, default=32)
    parser.add_argument("--updates-per-collect", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--replay-size", type=int, default=1_000_000)
    parser.add_argument("--gnn-layers", type=int, default=1)
    parser.add_argument("--gnn-out-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--hidden-layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lr-final", type=float, default=3e-6)
    parser.add_argument("--max-grad-norm", type=float, default=2.0)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--lambda-init", type=float, default=0.1)
    parser.add_argument("--lambda-final", type=float, default=0.0001)
    parser.add_argument("--lambda-decay-steps", type=int, default=1_000_000)
    parser.add_argument("--constraint-scale", type=float, default=0.5)
    parser.add_argument("--agent-margin", type=float, default=0.02)
    parser.add_argument("--obstacle-margin", type=float, default=0.02)
    parser.add_argument("--braking-accel", type=float, default=None)
    parser.add_argument("--output-dir", default="./logs/deep_qp_safety")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--save-interval", type=int, default=10_000)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--debug", action="store_true", default=False)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
