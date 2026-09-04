"""Train only a GCBF certificate on a DGPPO LidarEnv.

Unlike ``train.py --algo gcbf+``, this entry point has no actor network.  A
fixed exploratory controller collects transitions and only the parameters of
the graph CBF are optimized.
"""

from __future__ import annotations

import argparse
import datetime as dt
import functools as ft
import json
import math
import os
import pickle
import shutil
from pathlib import Path
from time import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import numpy as np
import optax
import wandb
import yaml
from flax.training.train_state import TrainState
from tqdm import tqdm

from dgppo.algo.gcbf_plus_adapter import LidarGCBFPlusAdapter
from dgppo.algo.module.gcbf_plus import GCBFNetwork
from dgppo.env import make_env
from dgppo.env.lidar_env.base import LidarEnv
from dgppo.trainer.data import GCBFTransitionBatch
from dgppo.trainer.gcbf_buffer import GCBFReplayBuffer
from dgppo.trainer.utils import compute_norm_and_clip, has_any_nan_or_inf
from dgppo.utils.utils import jax_vmap, merge01


class GCBFCertificate:
    """The CBF optimization module; it deliberately owns no control policy."""

    def __init__(
        self,
        env: LidarEnv,
        gnn_layers: int = 1,
        lr_cbf: float = 3e-5,
        alpha: float = 1.0,
        eps: float = 0.02,
        loss_unsafe_coef: float = 1.0,
        loss_safe_coef: float = 1.0,
        loss_h_dot_coef: float = 0.01,
        max_grad_norm: float = 2.0,
        seed: int = 0,
    ):
        self.env = env
        self.alpha = alpha
        self.eps = eps
        self.loss_unsafe_coef = loss_unsafe_coef
        self.loss_safe_coef = loss_safe_coef
        self.loss_h_dot_coef = loss_h_dot_coef
        self.max_grad_norm = max_grad_norm
        self.gnn_layers = gnn_layers
        self.lr_cbf = lr_cbf

        init_key, graph_key = jr.split(jr.PRNGKey(seed))
        self.network = GCBFNetwork(env.num_agents, gnn_layers)
        params = self.network.initialize(init_key, env.reset(graph_key))
        optimizer = optax.apply_if_finite(
            optax.adamw(lr_cbf, weight_decay=1e-3), 1_000_000
        )
        self.state = TrainState.create(
            apply_fn=self.network.get_cbf,
            params=params,
            tx=optimizer,
        )

    @property
    def config(self) -> dict:
        return {
            "gcbf_gnn_layers": self.gnn_layers,
            "gcbf_lr_cbf": self.lr_cbf,
            "gcbf_alpha": self.alpha,
            "gcbf_eps": self.eps,
            "gcbf_loss_unsafe_coef": self.loss_unsafe_coef,
            "gcbf_loss_safe_coef": self.loss_safe_coef,
            "gcbf_loss_h_dot_coef": self.loss_h_dot_coef,
            "max_grad_norm": self.max_grad_norm,
        }

    def get_cbf(self, graph, params=None):
        params = self.state.params if params is None else params
        return self.network.get_cbf(params, graph)

    def _loss(self, params, batch: GCBFTransitionBatch):
        cbf_fn = jax_vmap(lambda graph: self.network.get_cbf(params, graph))
        h = merge01(cbf_fn(batch.graph).squeeze(-1))
        h_next = merge01(cbf_fn(batch.next_graph).squeeze(-1))
        safe = merge01(batch.safe_mask)
        unsafe = merge01(batch.unsafe_mask)

        h_unsafe = jnp.where(unsafe, h, -2.0 * self.eps)
        loss_unsafe = jnp.sum(jax.nn.relu(h_unsafe + self.eps)) / (
            jnp.count_nonzero(unsafe) + 1e-6
        )
        h_safe = jnp.where(safe, h, 2.0 * self.eps)
        loss_safe = jnp.sum(jax.nn.relu(-h_safe + self.eps)) / (
            jnp.count_nonzero(safe) + 1e-6
        )

        h_dot = (h_next - h) / self.env.dt
        loss_h_dot = jnp.mean(
            jax.nn.relu(-h_dot - self.alpha * h + self.eps)
        )
        total = (
            self.loss_unsafe_coef * loss_unsafe
            + self.loss_safe_coef * loss_safe
            + self.loss_h_dot_coef * loss_h_dot
        )
        info = {
            "loss/total": total,
            "loss/unsafe": loss_unsafe,
            "loss/safe": loss_safe,
            "loss/h_dot": loss_h_dot,
            "acc/unsafe": jnp.sum(jnp.where(unsafe, h < 0.0, False))
            / (jnp.count_nonzero(unsafe) + 1e-6),
            "acc/safe": jnp.sum(jnp.where(safe, h > 0.0, False))
            / (jnp.count_nonzero(safe) + 1e-6),
            "acc/h_dot": jnp.mean(h_dot + self.alpha * h > 0.0),
            "data/unsafe_ratio": jnp.mean(unsafe),
        }
        return total, info

    @ft.partial(jax.jit, static_argnums=(0,))
    def evaluate(self, state: TrainState, batch: GCBFTransitionBatch) -> dict:
        return self._loss(state.params, batch)[1]

    @ft.partial(jax.jit, static_argnums=(0,))
    def update(
        self, state: TrainState, batch: GCBFTransitionBatch
    ) -> tuple[TrainState, dict]:
        (_, info), gradient = jax.value_and_grad(self._loss, has_aux=True)(
            state.params, batch
        )
        has_nan = has_any_nan_or_inf(gradient).astype(jnp.float32)
        gradient, grad_norm = compute_norm_and_clip(gradient, self.max_grad_norm)
        state = state.apply_gradients(grads=gradient)
        return state, info | {
            "grad_norm/cbf": grad_norm,
            "grad_has_nan/cbf": has_nan,
        }


@ft.partial(jax.jit, static_argnums=(1,))
def _safe_mask(unsafe_mask: jax.Array, horizon: int) -> jax.Array:
    """Match the original GCBF+ future-horizon safe labeling rule."""
    safe = jnp.ones_like(unsafe_mask, dtype=jnp.bool_)
    for time_idx in range(unsafe_mask.shape[1]):
        start = max(0, time_idx - horizon)
        safe = safe.at[:, start : time_idx + 1].set(
            safe[:, start : time_idx + 1]
            & ~unsafe_mask[:, time_idx : time_idx + 1]
        )
    return safe.at[:, 0].set(True)


def _flatten_env_time(tree):
    return jtu.tree_map(lambda value: value.reshape((-1,) + value.shape[2:]), tree)


def _make_collector(
    env: LidarEnv,
    n_env: int,
    rollout_steps: int,
    horizon: int,
    exploration_probability: float,
    exploration_scale: float,
):
    adapter = LidarGCBFPlusAdapter(env)
    lower, upper = env.action_lim()
    action_scale = (upper - lower) / 2.0

    def collect_one(key):
        reset_key, key = jr.split(key)
        graph = env.reset(reset_key)

        def body(current_graph, step_key):
            noise_key, random_key, choice_key = jr.split(step_key, 3)
            nominal = adapter.nominal_action(current_graph)
            noisy_nominal = nominal + exploration_scale * action_scale * jr.normal(
                noise_key, nominal.shape
            )
            random_action = jr.uniform(
                random_key, nominal.shape, minval=lower, maxval=upper
            )
            explore = jr.bernoulli(choice_key, exploration_probability)
            action = jnp.where(explore, random_action, noisy_nominal)
            action = env.clip_action(action)
            next_graph, _, cost, _, _ = env.step(current_graph, action)
            return next_graph, (current_graph, next_graph, cost)

        step_keys = jr.split(key, rollout_steps)
        _, (graphs, next_graphs, costs) = jax.lax.scan(
            body, graph, step_keys, length=rollout_steps
        )
        return graphs, next_graphs, costs

    collect_many = jax.jit(jax.vmap(collect_one))

    def collect(key) -> GCBFTransitionBatch:
        graphs, next_graphs, costs = collect_many(jr.split(key, n_env))
        unsafe = jnp.max(costs, axis=-1) > 0.0
        safe = _safe_mask(unsafe, horizon)
        return GCBFTransitionBatch(
            graph=_flatten_env_time(graphs),
            next_graph=_flatten_env_time(next_graphs),
            safe_mask=merge01(safe),
            unsafe_mask=merge01(unsafe),
        )

    return collect


def _scalars(values: dict) -> dict[str, float]:
    return {name: float(np.asarray(value).mean()) for name, value in values.items()}


def _checkpoint_dir(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.is_file():
        path = path.parent
    if (path / "cbf.pkl").is_file():
        return path
    candidate = path / "models" / "latest"
    if (candidate / "cbf.pkl").is_file():
        return candidate
    raise FileNotFoundError(f"could not find a GCBF checkpoint under {path}")


def _save_checkpoint(
    run_dir: Path,
    certificate: GCBFCertificate,
    replay: GCBFReplayBuffer,
    key: jax.Array,
    iteration: int,
    metadata: dict,
) -> None:
    models_dir = run_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = models_dir / "latest"
    temporary = models_dir / ".latest.tmp"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir()
    with (temporary / "cbf.pkl").open("wb") as file:
        pickle.dump(certificate.state.params, file)
    with (temporary / "gcbf_training_state.pkl").open("wb") as file:
        pickle.dump(
            {
                "format_version": 1,
                "iteration": iteration,
                "optimizer_step": certificate.state.step,
                "optimizer_state": certificate.state.opt_state,
                "collector_key": np.asarray(key),
                "metadata": metadata,
            },
            file,
        )
    replay.save(temporary / "replay.pkl")
    (temporary / f"latest_iter_{iteration:09d}.txt").touch()
    backup = models_dir / ".latest.old"
    if backup.exists():
        shutil.rmtree(backup)
    if checkpoint.exists():
        os.replace(checkpoint, backup)
    try:
        os.replace(temporary, checkpoint)
    except BaseException:
        if backup.exists() and not checkpoint.exists():
            os.replace(backup, checkpoint)
        raise
    if backup.exists():
        shutil.rmtree(backup)


def _load_checkpoint(
    checkpoint: Path,
    certificate: GCBFCertificate,
    replay: GCBFReplayBuffer,
    expected_metadata: dict,
) -> tuple[jax.Array, int]:
    with (checkpoint / "cbf.pkl").open("rb") as file:
        params = pickle.load(file)
    certificate.state = certificate.state.replace(params=params)
    state_path = checkpoint / "gcbf_training_state.pkl"
    if not state_path.is_file():
        return jr.PRNGKey(expected_metadata["seed"]), 0
    with state_path.open("rb") as file:
        payload = pickle.load(file)
    if payload.get("format_version") != 1:
        raise ValueError("unsupported GCBF-only checkpoint format")
    observed = payload.get("metadata", {})
    mismatches = {
        name: (observed.get(name), value)
        for name, value in expected_metadata.items()
        if name != "seed" and observed.get(name) != value
    }
    if mismatches:
        raise ValueError(f"checkpoint metadata mismatch: {mismatches}")
    certificate.state = certificate.state.replace(
        step=payload["optimizer_step"], opt_state=payload["optimizer_state"]
    )
    replay_path = checkpoint / "replay.pkl"
    if replay_path.is_file():
        replay.load(replay_path)
    return jnp.asarray(payload["collector_key"]), int(payload["iteration"])


def _validate_args(args: argparse.Namespace) -> None:
    positive = {
        "steps": args.steps,
        "num_agents": args.num_agents,
        "rollout_steps": args.rollout_steps,
        "batch_size": args.batch_size,
        "buffer_size": args.buffer_size,
        "inner_epoch": args.inner_epoch,
        "n_env_train": args.n_env_train,
        "n_env_test": args.n_env_test,
        "eval_interval": args.eval_interval,
        "save_interval": args.save_interval,
        "log_interval": args.log_interval,
    }
    invalid = [name for name, value in positive.items() if value <= 0]
    if invalid:
        raise ValueError("these arguments must be positive: " + ", ".join(invalid))
    if not 0.0 <= args.unsafe_fraction <= 1.0:
        raise ValueError("unsafe-fraction must be in [0, 1]")
    if not 0.0 <= args.exploration_probability <= 1.0:
        raise ValueError("exploration-probability must be in [0, 1]")
    if args.exploration_scale < 0.0:
        raise ValueError("exploration-scale must be non-negative")


def train(args: argparse.Namespace) -> Path:
    _validate_args(args)
    if args.debug:
        jax.config.update("jax_disable_jit", True)
        args.wandb_mode = "disabled"
    np.random.seed(args.seed)
    env = make_env(
        args.env,
        args.num_agents,
        num_obs=args.obs,
        n_rays=args.n_rays,
        max_step=args.rollout_steps,
        full_observation=args.full_observation,
    )
    if not isinstance(env, LidarEnv):
        raise ValueError("train_gcbf.py only supports the LidarEnv family")

    certificate = GCBFCertificate(
        env,
        gnn_layers=args.gnn_layers,
        lr_cbf=args.lr_cbf,
        alpha=args.alpha,
        eps=args.eps,
        loss_unsafe_coef=args.loss_unsafe_coef,
        loss_safe_coef=args.loss_safe_coef,
        loss_h_dot_coef=args.loss_h_dot_coef,
        max_grad_norm=args.max_grad_norm,
        seed=args.seed,
    )
    replay_capacity = args.buffer_size * args.rollout_steps
    if replay_capacity < args.batch_size:
        raise ValueError("buffer-size * rollout-steps must be at least batch-size")
    if args.n_env_train * args.rollout_steps < args.batch_size:
        raise ValueError("n-env-train * rollout-steps must be at least batch-size")
    replay = GCBFReplayBuffer(replay_capacity, seed=args.seed + 1)
    metadata = {
        "env": args.env,
        "num_agents": args.num_agents,
        "obs": args.obs,
        "n_rays": args.n_rays,
        "gnn_layers": args.gnn_layers,
        "seed": args.seed,
    }

    timestamp = dt.datetime.now().strftime("%Y%m%d%H%M%S")
    default_run_dir = (
        Path(args.log_dir)
        / args.env
        / "gcbf"
        / f"seed{args.seed}_{timestamp}"
    )
    checkpoint = None
    if args.resume is not None:
        checkpoint = _checkpoint_dir(args.resume)
        default_run_dir = checkpoint.parent.parent
    run_dir = (
        default_run_dir
        if args.output_dir is None
        else args.output_dir.expanduser().resolve()
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    key = jr.PRNGKey(args.seed)
    start_iteration = 0
    if checkpoint is not None:
        key, start_iteration = _load_checkpoint(
            checkpoint, certificate, replay, metadata
        )
        if start_iteration > args.steps:
            raise ValueError(
                f"checkpoint iteration {start_iteration} exceeds --steps {args.steps}"
            )

    argument_config = {
        name: str(value) if isinstance(value, Path) else value
        for name, value in vars(args).items()
    }
    config = argument_config | certificate.config | {
        "algo": "gcbf",
        "training_mode": "cbf_only",
        "gcbf_batch_size": args.batch_size,
        "gcbf_buffer_size": replay_capacity,
        "gcbf_horizon": args.horizon,
        "gcbf_inner_epoch": args.inner_epoch,
        "gcbf_unsafe_fraction": args.unsafe_fraction,
    }
    with (run_dir / "config.yaml").open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, sort_keys=True)

    collect_train = _make_collector(
        env,
        args.n_env_train,
        args.rollout_steps,
        args.horizon,
        args.exploration_probability,
        args.exploration_scale,
    )
    collect_eval = _make_collector(
        env,
        args.n_env_test,
        args.rollout_steps,
        args.horizon,
        0.0,
        0.0,
    )
    samples_per_iteration = args.n_env_train * args.rollout_steps
    minibatches_per_epoch = math.ceil(samples_per_iteration / args.batch_size)

    run_name = args.wandb_name or (
        f"gcbf_{args.env}_n{args.num_agents}_o{args.obs}_seed{args.seed}"
    )
    wandb_run = None
    if args.wandb_mode != "disabled":
        if args.wandb_mode == "online":
            wandb.login()
        wandb_run = wandb.init(
            name=run_name,
            project=args.wandb_project,
            group=args.env,
            dir=str(run_dir),
            mode=args.wandb_mode,
            config=config,
        )
        wandb.define_metric("counters/iteration")
        wandb.define_metric("*", step_metric="counters/iteration")

    metrics_file = (run_dir / "training_metrics.jsonl").open(
        "a", encoding="utf-8"
    )
    progress = tqdm(
        total=args.steps,
        initial=start_iteration,
        desc="GCBF certificate",
    )
    start_time = time()
    last_info = None
    try:
        for iteration in range(start_iteration + 1, args.steps + 1):
            collect_key, key = jr.split(key)
            batch = collect_train(collect_key)
            replay.append(batch)

            for _ in range(args.inner_epoch * minibatches_per_epoch):
                sampled = replay.sample(args.batch_size, args.unsafe_fraction)
                certificate.state, last_info = certificate.update(
                    certificate.state, sampled
                )
                if bool(np.asarray(last_info["grad_has_nan/cbf"])):
                    raise FloatingPointError("non-finite CBF gradient")

            log_info = _scalars(last_info) | {
                "counters/iteration": iteration,
                "counters/optimizer_step": int(np.asarray(certificate.state.step)),
                "counters/current_frames": samples_per_iteration,
                "counters/total_frames": iteration * samples_per_iteration,
                "data/replay_size": replay.length,
                "collection/unsafe_rate": float(
                    np.asarray(batch.unsafe_mask).mean()
                ),
                "time/elapsed_sec": time() - start_time,
            }

            did_evaluate = (
                iteration % args.eval_interval == 0 or iteration == args.steps
            )
            if did_evaluate:
                eval_key = jr.fold_in(jr.PRNGKey(args.seed), iteration)
                eval_batch = collect_eval(eval_key)
                eval_info = _scalars(certificate.evaluate(certificate.state, eval_batch))
                log_info |= {
                    name.replace("loss/", "eval/loss/")
                    .replace("acc/", "eval/acc/")
                    .replace("data/", "eval/data/"): value
                    for name, value in eval_info.items()
                }
                log_info["eval/unsafe_rate"] = float(
                    np.asarray(eval_batch.unsafe_mask).mean()
                )

            if iteration % args.save_interval == 0 or iteration == args.steps:
                _save_checkpoint(
                    run_dir, certificate, replay, key, iteration, metadata
                )

            if (
                iteration % args.log_interval == 0
                or did_evaluate
                or iteration == args.steps
            ):
                metrics_file.write(json.dumps(log_info, sort_keys=True) + "\n")
                metrics_file.flush()
                if wandb_run is not None:
                    wandb.log(log_info, step=iteration)

            progress.update(1)
            progress.set_postfix(
                loss=log_info["loss/total"],
                unsafe=log_info["collection/unsafe_rate"],
            )
    finally:
        progress.close()
        metrics_file.close()
        if wandb_run is not None:
            wandb.finish()
    print(f"> GCBF-only checkpoint: {run_dir / 'models' / 'latest'}")
    return run_dir


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train only a GCBF certificate on a DGPPO LidarEnv"
    )
    parser.add_argument("--env", default="LidarSpread")
    parser.add_argument("-n", "--num-agents", type=int, default=8)
    parser.add_argument("--obs", type=int, default=3)
    parser.add_argument("--n-rays", type=int, default=32)
    parser.add_argument("--full-observation", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=1000)

    # Defaults follow the original gcbfplus training entry point.
    parser.add_argument("--gnn-layers", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--lr-cbf", type=float, default=3e-5)
    parser.add_argument("--eps", type=float, default=0.02)
    parser.add_argument("--loss-unsafe-coef", type=float, default=1.0)
    parser.add_argument("--loss-safe-coef", type=float, default=1.0)
    parser.add_argument("--loss-h-dot-coef", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--buffer-size", type=int, default=512)
    parser.add_argument("--inner-epoch", type=int, default=8)
    parser.add_argument("--n-env-train", type=int, default=16)
    parser.add_argument("--n-env-test", type=int, default=32)
    parser.add_argument("--rollout-steps", type=int, default=128)

    parser.add_argument("--unsafe-fraction", type=float, default=0.5)
    parser.add_argument("--exploration-probability", type=float, default=0.5)
    parser.add_argument("--exploration-scale", type=float, default=0.5)
    parser.add_argument("--eval-interval", type=int, default=1)
    parser.add_argument("--save-interval", type=int, default=10)
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--log-dir", type=Path, default=Path("logs"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default="disabled",
    )
    parser.add_argument("--wandb-project", default="dgppo")
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    train(parse_args())
