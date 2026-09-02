"""Pretrain the distributed Graph-HJ critic independently of task reward."""

import argparse
import json
import os
import pickle
from time import time
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import numpy as np
import wandb
from tqdm import tqdm

from dgppo.algo.module.deep_qp_safety import (
    DeepQPSafetyConfig,
    GraphHJSafetyCritic,
    environment_cost_metadata,
    graph_hj_node_feature_mask,
    safety_lambda_at,
)
from dgppo.env import make_env
from dgppo.env.base import MultiAgentEnv
from dgppo.env.lidar_env.base import LidarEnv
from dgppo.env.vmas.vmas_navigation import VMASNavigation
from dgppo.trainer.data import SafetyBatch
from dgppo.trainer.safety_buffer import SafetyReplayBuffer
from dgppo.utils.utils import tree_where


def _flatten_env_time(tree):
    return jtu.tree_map(lambda x: x.reshape((-1,) + x.shape[2:]), tree)


def _json_scalars(metrics: dict) -> dict:
    return {
        key: float(np.asarray(value).mean())
        for key, value in metrics.items()
    }


def _training_state_path(output_dir: Path) -> Path:
    return output_dir / "deep_qp_training_state.pkl"


def _save_training_checkpoint(
        safety_critic: GraphHJSafetyCritic,
        state,
        replay: SafetyReplayBuffer,
        key,
        output_dir: Path,
        metadata: dict,
) -> None:
    safety_critic.save_checkpoint(
        state, output_dir / "deep_qp_safety.pkl", metadata=metadata
    )
    replay.save(output_dir / "deep_qp_replay.pkl")
    sidecar_path = _training_state_path(output_dir)
    temporary_path = sidecar_path.with_suffix(".tmp")
    with temporary_path.open("wb") as file:
        pickle.dump({"key": np.asarray(key)}, file)
    os.replace(temporary_path, sidecar_path)


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
        "eval_interval": args.eval_interval,
        "eval_n_env": args.eval_n_env,
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
        n_env: int,
        rollout_steps: int,
):
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

            # The environment owns the collision definition. Its cost channels
            # are unsafe-positive, whereas Graph-HJ consumes one safe-positive
            # scalar per agent.
            constraint = -jnp.max(env.get_cost(graph), axis=-1)
            next_graph, _, _, _, _ = env.step(graph, raw_action)
            next_constraint = -jnp.max(env.get_cost(next_graph), axis=-1)
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
        args.wandb_mode = "disabled"
    env = make_env(
        env_id=args.env,
        num_agents=args.num_agents,
        num_obs=args.obs,
        n_rays=args.n_rays,
        max_step=args.rollout_steps,
        full_observation=args.full_observation,
    )
    if not isinstance(env, (LidarEnv, VMASNavigation)):
        raise ValueError(
            "train_safety_filter.py currently targets LidarEnv and the "
            "VMASNavigation family"
        )
    print("> Graph-HJ safety constraint source: env.get_cost")

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
        node_feature_mask=graph_hj_node_feature_mask(env),
        config=config,
    )
    key = jr.PRNGKey(args.seed)
    init_key, reset_key, key = jr.split(key, 3)
    state = safety_critic.initialize(init_key, env.reset(reset_key))
    replay = SafetyReplayBuffer(args.replay_size, seed=args.seed)
    checkpoint_metadata = environment_cost_metadata(env)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.resume is not None:
        state = safety_critic.load_checkpoint(
            state, args.resume, expected_metadata=checkpoint_metadata
        )
        replay_path = Path(args.resume).parent / "deep_qp_replay.pkl"
        if replay_path.exists():
            replay.load(replay_path, expected_n_agents=env.num_agents)
        training_state_path = _training_state_path(Path(args.resume).parent)
        if training_state_path.exists():
            with training_state_path.open("rb") as file:
                key = jnp.asarray(pickle.load(file)["key"])
        else:
            print(
                "> Resume checkpoint has no collector PRNG sidecar; "
                "collection restarts from --seed."
            )

    collect_batch = _make_collector(
        env, args.n_env, args.rollout_steps
    )
    eval_collect_batch = _make_collector(
        env, args.eval_n_env, args.rollout_steps
    )
    eval_key = jr.fold_in(jr.PRNGKey(args.seed), 0x484A)
    eval_batch = eval_collect_batch(eval_key)

    @jax.jit
    def evaluate(eval_state):
        safety_lambda = safety_lambda_at(
            safety_critic.config, eval_state.online.step
        )
        _, eval_info = safety_critic.loss(
            eval_state.online.params,
            eval_state.target_params,
            eval_batch,
            safety_lambda,
        )
        return eval_info

    initial_step = int(np.asarray(state.online.step))
    progress = tqdm(total=args.steps, initial=initial_step, desc="safety updates")
    last_saved_step = initial_step
    last_logged_step = initial_step
    last_eval_step = initial_step
    run_name = args.wandb_name or (
        f"graph_hj_{args.env}_n{args.num_agents}_o{args.obs}_seed{args.seed}"
    )
    wandb_run = None
    if args.wandb_mode != "disabled":
        if args.wandb_mode == "online":
            wandb.login()
        wandb_run = wandb.init(
            id=args.wandb_run_id,
            resume="allow" if args.wandb_run_id is not None else None,
            name=run_name,
            project=args.wandb_project,
            group=args.env,
            dir=str(output_dir),
            mode=args.wandb_mode,
            config={
                f"deep-qp/{key}": value for key, value in vars(args).items()
            } | {
                f"deep-qp/critic/{key}": value
                for key, value in config.to_dict().items()
            },
        )
        wandb.define_metric("deep-qp/counters/update")
        wandb.define_metric(
            "deep-qp/*", step_metric="deep-qp/counters/update"
        )

    log_path = output_dir / args.log_file
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("a", encoding="utf-8")
    start_time = time()
    try:
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
            should_log = info is not None and (
                current_step == args.steps
                or current_step - last_logged_step >= args.log_interval
            )
            if should_log:
                progress.set_postfix(
                    loss=float(np.asarray(info["safety/loss"])),
                    value=float(np.asarray(info["safety/value_loss"])),
                    deriv=float(np.asarray(info["safety/derivative_loss"])),
                    replay=replay.length,
                )
                log_info = _json_scalars(info) | {
                    "counters/update": current_step,
                    "counters/replay_size": replay.length,
                    "data/constraint_mean": float(np.asarray(batch.constraints).mean()),
                    "data/constraint_min": float(np.asarray(batch.constraints).min()),
                    "data/unsafe_rate": float((np.asarray(batch.constraints) < 0.0).mean()),
                    "time/elapsed_sec": time() - start_time,
                    "performance/updates_per_sec": (
                        (current_step - initial_step) / max(time() - start_time, 1e-8)
                    ),
                }
                if (
                    current_step == args.steps
                    or current_step - last_eval_step >= args.eval_interval
                ):
                    eval_info = evaluate(state)
                    log_info |= {
                        key.replace("safety/", "eval/safety/", 1): value
                        for key, value in _json_scalars(eval_info).items()
                    }
                    log_info |= {
                        "eval/data/constraint_mean": float(
                            np.asarray(eval_batch.constraints).mean()
                        ),
                        "eval/data/constraint_min": float(
                            np.asarray(eval_batch.constraints).min()
                        ),
                        "eval/data/unsafe_rate": float(
                            (np.asarray(eval_batch.constraints) < 0.0).mean()
                        ),
                    }
                    last_eval_step = current_step
                log_info = {
                    f"deep-qp/{key}": value for key, value in log_info.items()
                }
                log_file.write(json.dumps(log_info, sort_keys=True) + "\n")
                log_file.flush()
                if wandb_run is not None:
                    wandb.log(log_info)
                last_logged_step = current_step
            if current_step > 0 and current_step - last_saved_step >= args.save_interval:
                _save_training_checkpoint(
                    safety_critic,
                    state,
                    replay,
                    key,
                    output_dir,
                    checkpoint_metadata,
                )
                tqdm.write(f"> Saved Graph-HJ checkpoint at update {current_step}")
                last_saved_step = current_step

        _save_training_checkpoint(
            safety_critic,
            state,
            replay,
            key,
            output_dir,
            checkpoint_metadata,
        )
    finally:
        log_file.close()
        progress.close()
        if wandb_run is not None:
            wandb.finish()
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
    parser.add_argument("--output-dir", default="./logs/deep_qp_safety")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--save-interval", type=int, default=10_000)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--eval-interval", type=int, default=1_000)
    parser.add_argument("--eval-n-env", type=int, default=8)
    parser.add_argument("--log-file", default="training_metrics.jsonl")
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default="disabled",
    )
    parser.add_argument("--wandb-project", default="dgppo")
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--wandb-run-id", default=None)
    parser.add_argument("--debug", action="store_true", default=False)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
