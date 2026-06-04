"""Train DGPPO on the JAX navigation/navigation_obs environments.

This entry point keeps the CLI and wandb metric names close to the GemsMARL
navigation training scripts while using the original JAX DGPPO implementation.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable

import jax
import jax.random as jr
import jax.tree_util as jtu
import matplotlib
import numpy as np
import wandb
import yaml
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dgppo.algo import make_algo
from dgppo.env import make_env
from dgppo.trainer.data import Rollout
from dgppo.trainer.utils import is_connected, test_rollout
from dgppo.utils.utils import tree_index


SCENARIO_TO_ENV = {
    "navigation": "VMASNavigation",
    "navigation_obs": "VMASNavigationObs",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train original JAX DGPPO on navigation/navigation_obs with GemsMARL-style logging."
    )
    parser.add_argument(
        "--scenario",
        choices=sorted(SCENARIO_TO_ENV),
        default="navigation_obs",
        help="Navigation scenario to train.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda", help="Kept for CLI compatibility; JAX chooses the backend.")
    parser.add_argument("--save-folder", default="outputs")
    parser.add_argument("--render", action="store_true", help="Enable eval video logging. Disabled by default for headless runs.")
    parser.add_argument("--no-render", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument(
        "--wandb-mode",
        choices=["online", "offline", "auto"],
        default="online",
        help="wandb upload mode. Default is online for real-time server sync.",
    )
    parser.add_argument("--project-name", default="benchmarl")
    parser.add_argument("--name", default=None)

    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--n-agents", type=int, default=4)
    parser.add_argument("--n-obstacles", type=int, default=3)
    parser.add_argument("--full-observation", action="store_true", default=False)

    parser.add_argument("--on-policy-n-envs-per-worker", type=int, default=None)
    parser.add_argument("--on-policy-collected-frames-per-batch", type=int, default=6000)
    parser.add_argument("--on-policy-n-minibatch-iters", type=int, default=45)
    parser.add_argument("--on-policy-minibatch-size", type=int, default=400)
    parser.add_argument("--clip-grad-val", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--critic-lr", type=float, default=None)

    parser.add_argument("--evaluation-interval", type=int, default=120_000)
    parser.add_argument("--evaluation-episodes", type=int, default=10)
    parser.add_argument("--checkpoint-interval", type=int, default=0)
    parser.add_argument("--max-n-iters", type=int, default=None)
    parser.add_argument("--max-n-frames", type=int, default=3_000_000)

    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-eps", type=float, default=0.25)
    parser.add_argument("--lagr-init", type=float, default=0.5)
    parser.add_argument("--lr-lagr", type=float, default=1e-7)
    parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument("--cbf-eps", type=float, default=1e-2)
    parser.add_argument("--cbf-weight", type=float, default=1.0)
    parser.add_argument("--no-cbf-schedule", action="store_true", default=False)
    parser.add_argument("--cost-schedule", action="store_true", default=False)
    parser.add_argument("--no-rnn", action="store_true", default=False)
    parser.add_argument("--actor-gnn-layers", type=int, default=2)
    parser.add_argument("--Vl-gnn-layers", type=int, default=2)
    parser.add_argument("--Vh-gnn-layers", type=int, default=1)
    parser.add_argument("--rnn-layers", type=int, default=1)
    parser.add_argument("--use-lstm", action="store_true", default=False)
    parser.add_argument("--coef-ent", type=float, default=1e-2)
    parser.add_argument("--rnn-step", type=int, default=10)
    parser.add_argument("--debug", action="store_true", default=False)
    return parser.parse_args()


def _jsonable(value: Any):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _flatten_scalars(metrics: Dict[str, Any]) -> Dict[str, float]:
    out = {}
    for key, value in metrics.items():
        try:
            arr = np.asarray(value)
            if arr.shape == ():
                out[key] = float(arr)
        except Exception:
            continue
    return out


def _default_max_steps(scenario: str) -> int:
    return 200 if scenario == "navigation" else 100


def _default_n_envs(max_steps: int, frames_per_batch: int) -> int:
    if frames_per_batch % max_steps != 0:
        raise ValueError(
            "--on-policy-collected-frames-per-batch must be divisible by --max-steps "
            f"for original DGPPO full-episode rollouts, got {frames_per_batch} and {max_steps}."
        )
    return frames_per_batch // max_steps


def _resolve_schedule(args: argparse.Namespace) -> Dict[str, int]:
    max_steps = args.max_steps or _default_max_steps(args.scenario)
    if args.on_policy_n_envs_per_worker is None:
        args.on_policy_n_envs_per_worker = _default_n_envs(
            max_steps,
            args.on_policy_collected_frames_per_batch,
        )
    frames_per_iter = args.on_policy_n_envs_per_worker * max_steps
    requested_frames_per_batch = args.on_policy_collected_frames_per_batch
    if requested_frames_per_batch is not None and requested_frames_per_batch != frames_per_iter:
        raise ValueError(
            "Original DGPPO collects a full episode per iteration, so "
            "--on-policy-collected-frames-per-batch must equal "
            "--on-policy-n-envs-per-worker * --max-steps "
            f"({frames_per_iter}), got {requested_frames_per_batch}."
        )

    max_n_frames = args.max_n_frames
    if args.max_n_iters is not None:
        max_n_iters = args.max_n_iters
        max_n_frames = max_n_iters * frames_per_iter
    else:
        max_n_iters = int(np.ceil(max_n_frames / frames_per_iter))

    if args.evaluation_interval % frames_per_iter != 0:
        raise ValueError(
            f"evaluation_interval ({args.evaluation_interval}) must be a multiple "
            f"of frames_per_iter ({frames_per_iter}) for GemsMARL-style frame scheduling."
        )
    eval_interval_iters = max(1, args.evaluation_interval // frames_per_iter)

    if args.checkpoint_interval == 0:
        checkpoint_interval_iters = 0
    else:
        if args.checkpoint_interval % frames_per_iter != 0:
            raise ValueError(
                f"checkpoint_interval ({args.checkpoint_interval}) must be a multiple "
                f"of frames_per_iter ({frames_per_iter})."
            )
        checkpoint_interval_iters = max(1, args.checkpoint_interval // frames_per_iter)

    batch_size = args.on_policy_minibatch_size
    if not (max_steps <= batch_size <= frames_per_iter):
        raise ValueError(
            "--on-policy-minibatch-size must satisfy "
            f"max_steps <= minibatch_size <= frames_per_iter, got {batch_size}, "
            f"valid range [{max_steps}, {frames_per_iter}]."
        )
    if batch_size // max_steps < 1:
        raise ValueError("--on-policy-minibatch-size is too small for DGPPO batching.")
    if args.on_policy_n_envs_per_worker // (batch_size // max_steps) < 1:
        raise ValueError("--on-policy-minibatch-size is too large for the number of envs.")
    if max_steps // args.rnn_step < 1 or max_steps % args.rnn_step != 0:
        raise ValueError("--rnn-step must divide --max-steps for original DGPPO batching.")

    return {
        "max_steps": max_steps,
        "frames_per_iter": frames_per_iter,
        "max_n_iters": max_n_iters,
        "max_n_frames": max_n_frames,
        "eval_interval_iters": eval_interval_iters,
        "checkpoint_interval_iters": checkpoint_interval_iters,
    }


class GemsmarlStyleLogger:
    def __init__(
        self,
        folder: Path,
        run_name: str,
        args: argparse.Namespace,
        schedule: Dict[str, int],
        env_name: str,
        algo_config: Dict[str, Any],
    ):
        self.folder = folder
        self.folder.mkdir(parents=True, exist_ok=True)
        self.metrics_file = (self.folder / "metrics.csv").open("w", newline="")
        self.metrics_writer = csv.writer(self.metrics_file)
        self.metrics_writer.writerow(["step", "key", "value"])
        self.wandb_run = None

        hparams = {
            "algorithm_name": "dgppo",
            "environment_name": "vmas",
            "task_name": args.scenario,
            "model_name": "gnn",
            "seed": args.seed,
            "experiment_config": {
                "max_n_iters": schedule["max_n_iters"],
                "max_n_frames": schedule["max_n_frames"],
                "on_policy_n_envs_per_worker": args.on_policy_n_envs_per_worker,
                "on_policy_collected_frames_per_batch": schedule["frames_per_iter"],
                "on_policy_n_minibatch_iters": args.on_policy_n_minibatch_iters,
                "on_policy_minibatch_size": args.on_policy_minibatch_size,
                "evaluation_interval": args.evaluation_interval,
                "evaluation_episodes": args.evaluation_episodes,
                "checkpoint_interval": args.checkpoint_interval,
                "clip_grad_val": args.clip_grad_val,
                "lr": args.lr,
                "render": args.render and not args.no_render,
                "loggers": [] if args.no_wandb else ["wandb"],
                "wandb_mode": args.wandb_mode,
                "project_name": args.project_name,
            },
            "task_config": {
                "max_steps": schedule["max_steps"],
                "n_agents": args.n_agents,
                "n_obstacles": args.n_obstacles if args.scenario == "navigation_obs" else 0,
            },
            "algorithm_config": algo_config,
            "dgppo_env_name": env_name,
        }
        self.hparams = _jsonable(hparams)
        (self.folder / "hparams.json").write_text(json.dumps(self.hparams, indent=2))
        (self.folder / "config.yaml").write_text(yaml.safe_dump(self.hparams, sort_keys=False))

        if not args.no_wandb:
            if args.wandb_mode == "online":
                os.environ["WANDB_MODE"] = "online"
            elif args.wandb_mode == "offline":
                os.environ["WANDB_MODE"] = "offline"
            elif not is_connected():
                os.environ["WANDB_MODE"] = "offline"
            self.wandb_run = wandb.init(
                project=args.project_name,
                group=args.scenario,
                name=run_name,
                id=run_name,
                dir=str(self.folder),
                config=self.hparams,
            )

    def log(self, metrics: Dict[str, Any], step: int):
        scalars = _flatten_scalars(metrics)
        if self.wandb_run is not None:
            wandb.log(scalars, step=step)
        for key, value in scalars.items():
            self.metrics_writer.writerow([step, key, value])
        self.metrics_file.flush()

    def log_video(self, video_frames: np.ndarray, step: int):
        if self.wandb_run is None or video_frames is None or len(video_frames) <= 1:
            return
        vid = np.transpose(video_frames, (0, 3, 1, 2)).astype(np.uint8)[None]
        wandb.log(
            {"eval/video": wandb.Video(vid, fps=20, format="mp4")},
            step=step,
        )

    def finish(self):
        self.metrics_file.close()
        if self.wandb_run is not None:
            wandb.finish()


def _masked_episode_returns(rewards: np.ndarray, dones: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n_env, horizon = rewards.shape
    returns = np.zeros(n_env, dtype=np.float32)
    lengths = np.zeros(n_env, dtype=np.float32)
    for env_idx in range(n_env):
        done_ids = np.nonzero(dones[env_idx])[0]
        end = int(done_ids[0]) + 1 if len(done_ids) else horizon
        returns[env_idx] = rewards[env_idx, :end].sum()
        lengths[env_idx] = end
    return returns, lengths


def _add_min_mean_max(metrics: Dict[str, float], key: str, values: np.ndarray):
    values = np.asarray(values)
    metrics[f"{key}_min"] = values.min()
    metrics[f"{key}_mean"] = values.mean()
    metrics[f"{key}_max"] = values.max()


def _add_cost_metrics(
    metrics: Dict[str, float],
    costs: np.ndarray,
    prefix: str,
    cost_components: Iterable[str],
):
    positive_costs = np.maximum(costs, 0.0)
    metrics[f"{prefix}/cost/cost_mean"] = positive_costs.mean()
    metrics[f"{prefix}/cost/cost_max"] = positive_costs.max()
    metrics[f"{prefix}/safety/unsafe_rate"] = (positive_costs.max(axis=(-1, -2, -3)) > 0).mean()
    metrics["Safe/unsafe_rate"] = metrics[f"{prefix}/safety/unsafe_rate"]
    metrics["Safe/cost_mean"] = metrics[f"{prefix}/cost/cost_mean"]
    metrics["Safe/cost_max"] = metrics[f"{prefix}/cost/cost_max"]
    for idx, name in enumerate(cost_components):
        key_name = name.replace(" ", "_")
        metrics[f"{prefix}/cost/{key_name}_mean"] = positive_costs[..., idx].mean()


def _collection_metrics(
    rollout: Rollout,
    iteration: int,
    total_frames: int,
    fps: float,
    cost_components: Iterable[str],
) -> Dict[str, float]:
    rewards = np.asarray(rollout.rewards)
    dones = np.asarray(rollout.dones)
    costs = np.asarray(rollout.costs)
    episode_returns, _ = _masked_episode_returns(rewards, dones)

    metrics: Dict[str, float] = {
        "general/iteration": iteration,
        "general/total_frames": total_frames,
        "general/fps": fps,
    }
    _add_min_mean_max(metrics, "collection/reward/reward", rewards)
    _add_min_mean_max(metrics, "collection/agents/reward/episode_reward", episode_returns)
    _add_min_mean_max(metrics, "collection/reward/episode_reward", episode_returns)
    _add_cost_metrics(metrics, costs, "collection", cost_components)
    return metrics


def _training_metrics(update_info: Dict[str, Any]) -> Dict[str, float]:
    metrics = {
        "train/agents/loss_objective": update_info.get("policy/loss", np.nan),
        "train/agents/loss_critic": update_info.get("Vl/loss", np.nan),
        "train/agents/loss_critic_l": update_info.get("Vl/loss", np.nan),
        "train/agents/loss_critic_h": update_info.get("Vh/loss_Vh", update_info.get("Vh/loss", np.nan)),
        "train/agents/entropy": update_info.get("policy/entropy", np.nan),
        "train/agents/clip_fraction": update_info.get("policy/clip_frac", np.nan),
        "train/agents/grad_norm": update_info.get("policy/grad_norm", np.nan),
        "train/agents/actor_grad_norm": update_info.get("policy/grad_norm", np.nan),
        "train/agents/critic_l_grad_norm": update_info.get("Vl/grad_norm", np.nan),
        "train/agents/critic_h_grad_norm": update_info.get(
            "Vh/grad_Vh_norm", update_info.get("Vh/grad_norm", np.nan)
        ),
        "train/agents/safe_ratio": update_info.get("eval/safe_data", np.nan),
    }
    return metrics


def _eval_metrics(
    rollouts: Rollout,
    best_eval_reward: float,
    cost_components: Iterable[str],
) -> tuple[Dict[str, float], float]:
    rewards = np.asarray(rollouts.rewards)
    dones = np.asarray(rollouts.dones)
    costs = np.asarray(rollouts.costs)
    returns, lengths = _masked_episode_returns(rewards, dones)
    metrics: Dict[str, float] = {}
    _add_min_mean_max(metrics, "eval/agents/reward/episode_reward", returns)
    _add_min_mean_max(metrics, "eval/reward/episode_reward", returns)
    metrics["eval/reward/episode_len_mean"] = lengths.mean()
    _add_cost_metrics(metrics, costs, "eval", cost_components)
    best_eval_reward = max(best_eval_reward, float(returns.mean()))
    metrics["eval/reward/best_episode_reward_mean"] = best_eval_reward
    return metrics, best_eval_reward


def _render_eval_frames(env, rollout: Rollout) -> np.ndarray:
    single_rollout = tree_index(rollout, 0)
    env_states = single_rollout.graph.env_states
    costs = np.asarray(single_rollout.costs)
    a_pos = np.asarray(env_states.a_pos)
    goal_pos = np.asarray(env_states.goal_pos)
    o_pos = np.asarray(env_states.o_pos) if hasattr(env_states, "o_pos") else None

    fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=100)
    ax.set_xlim(-1.05 * env.half_width, 1.05 * env.half_width)
    ax.set_ylim(-1.05 * env.half_width, 1.05 * env.half_width)
    ax.set_aspect("equal")
    ax.add_patch(
        plt.Rectangle(
            (-env.half_width, -env.half_width),
            2 * env.half_width,
            2 * env.half_width,
            fc="none",
            ec="0.4",
        )
    )
    obs_patches = []
    if o_pos is not None:
        obs_patches = [
            plt.Circle((0, 0), env.params["obstacle_radius"], color="0.5", alpha=0.65)
            for _ in range(env.params["n_obs"])
        ]
        for patch in obs_patches:
            ax.add_patch(patch)
    goal_patches = [
        plt.Circle((0, 0), env.params["dist2goal"], color=f"C{ii}", alpha=0.25)
        for ii in range(env.num_agents)
    ]
    agent_patches = [
        plt.Circle((0, 0), env.agent_radius, color=f"C{ii}", zorder=5)
        for ii in range(env.num_agents)
    ]
    for patch in goal_patches + agent_patches:
        ax.add_patch(patch)
    text_opts = dict(size=10, color="k", transform=ax.transAxes)
    step_text = ax.text(0.99, 1.01, "kk=0", va="bottom", ha="right", **text_opts)
    cost_text = ax.text(0.99, 1.05, "cost=0", va="bottom", ha="right", **text_opts)

    frames = []
    horizon = max(1, len(a_pos) - 1)
    for kk in range(horizon):
        if o_pos is not None:
            for oo, patch in enumerate(obs_patches):
                patch.set_center(o_pos[kk, oo])
        for ii in range(env.num_agents):
            goal_patches[ii].set_center(goal_pos[kk, ii])
            agent_patches[ii].set_center(a_pos[kk, ii])
        step_text.set_text(f"kk={kk:04}")
        if costs.ndim >= 3:
            cost_text.set_text(
                "cost={}".format(
                    ", ".join([f"{float(v):+.3f}" for v in costs[kk].max(axis=0)])
                )
            )
        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        frames.append(frame)
    plt.close(fig)
    return np.stack(frames, axis=0)


def main():
    args = _parse_args()
    schedule = _resolve_schedule(args)
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    if args.debug:
        os.environ["JAX_DISABLE_JIT"] = "True"
        args.no_wandb = True
    np.random.seed(args.seed)

    env_name = SCENARIO_TO_ENV[args.scenario]
    obs_count = args.n_obstacles if args.scenario == "navigation_obs" else 0
    env = make_env(
        env_id=env_name,
        num_agents=args.n_agents,
        max_step=schedule["max_steps"],
        num_obs=obs_count,
        full_observation=args.full_observation,
    )
    env_test = make_env(
        env_id=env_name,
        num_agents=args.n_agents,
        max_step=schedule["max_steps"],
        num_obs=obs_count,
        full_observation=args.full_observation,
    )

    critic_lr = args.critic_lr if args.critic_lr is not None else args.lr
    algo = make_algo(
        algo="dgppo",
        env=env,
        node_dim=env.node_dim,
        edge_dim=env.edge_dim,
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        n_agents=env.num_agents,
        cost_weight=0.0,
        cbf_weight=args.cbf_weight,
        actor_gnn_layers=args.actor_gnn_layers,
        Vl_gnn_layers=args.Vl_gnn_layers,
        Vh_gnn_layers=args.Vh_gnn_layers,
        rnn_layers=args.rnn_layers,
        lr_actor=args.lr,
        lr_Vl=critic_lr,
        lr_Vh=critic_lr,
        max_grad_norm=args.clip_grad_val,
        alpha=args.alpha,
        cbf_eps=args.cbf_eps,
        seed=args.seed,
        batch_size=args.on_policy_minibatch_size,
        epoch_ppo=args.on_policy_n_minibatch_iters,
        use_rnn=not args.no_rnn,
        use_lstm=args.use_lstm,
        coef_ent=args.coef_ent,
        rnn_step=args.rnn_step,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_eps=args.clip_eps,
        lagr_init=args.lagr_init,
        lr_lagr=args.lr_lagr,
        train_steps=schedule["max_n_iters"],
        cbf_schedule=not args.no_cbf_schedule,
        cost_schedule=args.cost_schedule,
    )

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"{args.name}_" if args.name else ""
    run_name = f"dgppo_{args.scenario}_{suffix}seed{args.seed}_{timestamp}"
    folder = Path(args.save_folder) / args.scenario / "dgppo" / run_name
    logger = GemsmarlStyleLogger(folder, run_name, args, schedule, env_name, algo.config)

    print("[DGPPO navigation] Run configuration")
    print(f"  scenario={args.scenario} env={env_name} seed={args.seed}")
    print(f"  rollout={args.on_policy_n_envs_per_worker} envs x {schedule['max_steps']} steps")
    print(f"  frames_per_iter={schedule['frames_per_iter']} max_n_iters={schedule['max_n_iters']}")
    print(f"  minibatch={args.on_policy_minibatch_size} ppo_epochs={args.on_policy_n_minibatch_iters}")
    print(f"  eval_interval_iters={schedule['eval_interval_iters']} output={folder}")

    init_rnn_state = algo.init_rnn_state

    def test_fn_single(params, key):
        return test_rollout(env_test, lambda graph, rnn_state: algo.act(graph, rnn_state, params), init_rnn_state, key)

    test_fn = jax.jit(lambda params, keys: jax.vmap(lambda key: test_fn_single(params, key))(keys))
    best_eval_reward = -np.inf
    total_frames = 0
    start = time.time()
    train_key = jr.PRNGKey(args.seed)
    eval_key = jr.PRNGKey(args.seed + 10_000)

    try:
        pbar = tqdm(range(schedule["max_n_iters"]), ncols=90)
        for iteration in pbar:
            iter_start = time.time()
            key_x0, train_key = jr.split(train_key)
            train_keys = jr.split(key_x0, args.on_policy_n_envs_per_worker)
            rollouts = algo.collect(algo.params, train_keys)
            total_frames += schedule["frames_per_iter"]
            fps = schedule["frames_per_iter"] / max(time.time() - iter_start, 1e-6)
            collection = _collection_metrics(
                rollouts,
                iteration=iteration,
                total_frames=total_frames,
                fps=fps,
                cost_components=env.cost_components,
            )
            logger.log(collection, step=total_frames)

            update_info = algo.update(rollouts, iteration)
            logger.log(_training_metrics(update_info), step=total_frames)

            if (iteration + 1) % schedule["eval_interval_iters"] == 0:
                eval_key, subkey = jr.split(eval_key)
                eval_keys = jr.split(subkey, args.evaluation_episodes)
                eval_rollouts = test_fn(algo.params, eval_keys)
                metrics, best_eval_reward = _eval_metrics(
                    eval_rollouts, best_eval_reward, env.cost_components
                )
                metrics.update(
                    {
                        "general/iteration": iteration + 1,
                        "general/total_frames": total_frames,
                        "general/fps": total_frames / max(time.time() - start, 1e-6),
                    }
                )
                logger.log(metrics, step=total_frames)
                if args.render and not args.no_render:
                    video_frames = _render_eval_frames(env_test, eval_rollouts)
                    logger.log_video(video_frames, step=total_frames)

            if (
                schedule["checkpoint_interval_iters"]
                and (iteration + 1) % schedule["checkpoint_interval_iters"] == 0
            ):
                algo.save(str(folder / "models"), iteration + 1)

            pbar.set_postfix(
                reward=f"{collection['collection/reward/episode_reward_mean']:.3f}",
                unsafe=f"{collection['Safe/unsafe_rate']:.3f}",
            )
    finally:
        logger.finish()


if __name__ == "__main__":
    main()
