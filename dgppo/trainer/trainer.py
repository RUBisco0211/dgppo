import os
import pathlib
import shutil

import wandb
import numpy as np
import jax
import jax.random as jr
import functools as ft
import jax.numpy as jnp

from time import time
from matplotlib.animation import writers
from tqdm import tqdm

from .data import Rollout
from .utils import test_rollout
from ..env import MultiAgentEnv
from ..algo.base import Algorithm
from ..utils.utils import tree_index


def _scalar(value):
    arr = np.asarray(value)
    if arr.size == 0:
        return np.nan
    return float(arr.mean())


def _add_min_mean_max(metrics: dict, key: str, values):
    values = np.asarray(values)
    metrics[f"{key}_min"] = float(values.min())
    metrics[f"{key}_mean"] = float(values.mean())
    metrics[f"{key}_max"] = float(values.max())


def _episode_returns(rewards):
    rewards = np.asarray(rewards)
    if rewards.ndim < 2:
        return rewards
    returns = rewards.sum(axis=1)
    if returns.ndim > 1 and returns.shape[-1] == 1:
        returns = returns[..., 0]
    return returns


def _cost_metrics(costs, prefix: str, cost_components):
    costs = np.asarray(costs)
    positive_costs = np.maximum(costs, 0.0)
    metrics = {
        f"{prefix}/cost/cost_mean": float(positive_costs.mean()),
        f"{prefix}/cost/cost_max": float(positive_costs.max()),
    }
    if positive_costs.ndim > 1:
        per_episode_cost = positive_costs.max(axis=tuple(range(1, positive_costs.ndim)))
        metrics[f"{prefix}/safety/unsafe_rate"] = float((per_episode_cost > 0).mean())
    for idx, name in enumerate(cost_components):
        if idx < positive_costs.shape[-1]:
            key_name = name.replace(" ", "_")
            metrics[f"{prefix}/cost/{key_name}_mean"] = float(positive_costs[..., idx].mean())
    return metrics


def _rollout_metrics(rollout: Rollout, prefix: str, cost_components):
    rewards = np.asarray(rollout.rewards)
    returns = _episode_returns(rewards)
    metrics = {}
    _add_min_mean_max(metrics, f"{prefix}/reward/reward", rewards)
    _add_min_mean_max(metrics, f"{prefix}/reward/episode_reward", returns)
    _add_min_mean_max(metrics, f"{prefix}/agents/reward/episode_reward", returns)
    metrics |= _cost_metrics(rollout.costs, prefix, cost_components)
    return metrics


def _training_metrics(update_info: dict):
    return {
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


def _wandb_scalars(metrics: dict):
    clean = {}
    for key, value in metrics.items():
        try:
            clean[key] = _scalar(value)
        except (TypeError, ValueError):
            clean[key] = value
    return clean


class Trainer:

    def __init__(
            self,
            env: MultiAgentEnv,
            env_test: MultiAgentEnv,
            algo: Algorithm,
            gamma: float,
            n_env_train: int,
            n_env_test: int,
            log_dir: str,
            seed: int,
            params: dict,
            save_log: bool = True
    ):
        self.env = env
        self.env_test = env_test
        self.algo = algo
        self.gamma = gamma
        self.n_env_train = n_env_train
        self.n_env_test = n_env_test
        self.log_dir = log_dir
        self.seed = seed

        if Trainer._check_params(params):
            self.params = params

        # make dir for the models
        if save_log:
            if not os.path.exists(log_dir):
                os.mkdir(log_dir)
            self.model_dir = os.path.join(log_dir, 'models')
            if not os.path.exists(self.model_dir):
                os.mkdir(self.model_dir)

        wandb.login()
        wandb.init(name=params['run_name'], project='dgppo', group=env.__class__.__name__, dir=self.log_dir)
        wandb.define_metric("counters/total_frames")
        wandb.define_metric("counters/iter", step_metric="counters/total_frames")
        wandb.define_metric("*", step_metric="counters/total_frames")

        self.save_log = save_log

        self.steps = params['training_steps']
        self.eval_interval = params['eval_interval']
        self.eval_epi = params['eval_epi']
        self.save_interval = params['save_interval']
        self.video_interval = params.get('video_interval', self.eval_interval)
        self.log_video = params.get('log_video', True)
        self.video_dpi = params.get('video_dpi', 100)
        self.start_step = params.get('start_step', 0)
        self.best_eval_reward = -np.inf

        self.update_steps = 0
        self.key = jax.random.PRNGKey(seed)

    @staticmethod
    def _check_params(params: dict) -> bool:
        assert 'run_name' in params, 'run_name not found in params'
        assert 'training_steps' in params, 'training_steps not found in params'
        assert 'eval_interval' in params, 'eval_interval not found in params'
        assert params['eval_interval'] > 0, 'eval_interval must be positive'
        assert 'eval_epi' in params, 'eval_epi not found in params'
        assert params['eval_epi'] >= 1, 'eval_epi must be greater than or equal to 1'
        assert 'save_interval' in params, 'save_interval not found in params'
        assert params['save_interval'] > 0, 'save_interval must be positive'
        assert params.get('start_step', 0) >= 0, 'start_step must be non-negative'
        return True

    def _save_latest_checkpoint(self, step: int):
        latest_dir = os.path.join(self.model_dir, 'latest')
        if os.path.exists(latest_dir):
            shutil.rmtree(latest_dir)
        for name in os.listdir(self.model_dir):
            path = os.path.join(self.model_dir, name)
            if path == latest_dir:
                continue
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)
        self.algo.save(self.model_dir, 'latest')
        pathlib.Path(latest_dir, f"latest_iter_{int(step):09d}.txt").touch()

    def _eval_video(self, rollouts: Rollout, step: int):
        if not self.save_log or not self.log_video or step % self.video_interval != 0:
            return {}
        videos_dir = pathlib.Path(self.log_dir) / "videos"
        videos_dir.mkdir(parents=True, exist_ok=True)
        video_format = "mp4" if writers.is_available("ffmpeg") else "gif"
        video_path = videos_dir / f"latest_eval.{video_format}"
        single_rollout = tree_index(rollouts, 0)
        costs = np.asarray(single_rollout.costs)
        Ta_is_unsafe = costs.max(axis=-1) >= 1e-6 if costs.ndim >= 2 else None
        self.env_test.render_video(single_rollout, video_path, Ta_is_unsafe, {}, dpi=self.video_dpi)
        return {"eval/video": wandb.Video(str(video_path), format=video_format)}

    def train(self):
        # record start time
        start_time = time()

        # preprocess the rollout function
        init_rnn_state = self.algo.init_rnn_state

        def test_fn_single(params, key):
            act_fn = ft.partial(self.algo.act, params=params)
            return test_rollout(
                self.env_test,
                act_fn,
                init_rnn_state,
                key
            )

        test_fn = lambda params, keys: jax.vmap(ft.partial(test_fn_single, params))(keys)
        test_fn = jax.jit(test_fn)

        # start training
        test_key = jr.PRNGKey(self.seed)
        assert self.eval_epi <= 1_000, 'eval_epi must be less than or equal to 1_000'
        test_keys = jr.split(test_key, 1_000)[:self.eval_epi]

        remaining_steps = max(0, self.steps - self.start_step)
        pbar = tqdm(total=remaining_steps + 1, initial=0, ncols=80)
        for step in range(self.start_step, self.steps + 1):
            total_frames = step * self.n_env_train * self.env.max_episode_steps
            log_info = {
                "general/iteration": step,
                "counters/iter": step,
                "counters/total_frames": total_frames,
            }
            media_info = {}

            # evaluate the algorithm
            if step % self.eval_interval == 0:
                eval_info = {}
                test_rollouts: Rollout = test_fn(self.algo.params, test_keys)
                total_reward = test_rollouts.rewards.sum(axis=-1)
                reward_min, reward_max = total_reward.min(), total_reward.max()
                reward_mean = np.mean(total_reward)
                reward_final = np.mean(test_rollouts.rewards[:, -1])
                cost = jnp.maximum(test_rollouts.costs, 0.0).max(axis=-1).max(axis=-1).sum(axis=-1).mean()
                unsafe_frac = np.mean(test_rollouts.costs.max(axis=-1).max(axis=-2) >= 1e-6)
                eval_info = eval_info | {
                    "eval/reward": reward_mean,
                    "eval/reward_final": reward_final,
                    "eval/cost": cost,
                    "eval/unsafe_frac": unsafe_frac,
                }
                eval_info |= _rollout_metrics(test_rollouts, "eval", self.env_test.cost_components)
                self.best_eval_reward = max(self.best_eval_reward, eval_info["eval/reward/episode_reward_mean"])
                eval_info["eval/reward/best_episode_reward_mean"] = self.best_eval_reward
                time_since_start = time() - start_time
                eval_verbose = (f'step: {step:3}, time: {time_since_start:5.0f}s, reward: {reward_mean:9.4f}, '
                                f'min/max reward: {reward_min:7.2f}/{reward_max:7.2f}, cost: {cost:8.4f}, '
                                f'unsafe_frac: {unsafe_frac:6.2f}')
                tqdm.write(eval_verbose)
                log_info |= eval_info
                media_info |= self._eval_video(test_rollouts, step)

            # save the model
            if self.save_log and step % self.save_interval == 0:
                self._save_latest_checkpoint(step)

            # collect rollouts
            key_x0, self.key = jax.random.split(self.key)
            key_x0 = jax.random.split(key_x0, self.n_env_train)
            rollouts = self.algo.collect(self.algo.params, key_x0)
            collection_info = _rollout_metrics(rollouts, "collection", self.env.cost_components)
            log_info |= collection_info

            # update the algorithm
            update_info = self.algo.update(rollouts, step)
            log_info |= update_info | _training_metrics(update_info)
            wandb.log(_wandb_scalars(log_info) | media_info, step=self.update_steps)
            self.update_steps += 1

            pbar.update(1)
