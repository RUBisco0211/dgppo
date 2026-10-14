"""Sequential GRAFO training with InforMARL, FROG, and CL regularizers."""

import argparse
import json
import math
import os
import pathlib
import pickle
from time import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import wandb
from tqdm import tqdm

from dgppo.algo.grafo_continual import (
    GrafoContinualInforMARL, output_head, with_output_head,
)
from dgppo.env.mpe.grafo import GRAFO, GRAFO_SEQUENCES, GRAFO_TASKS
from dgppo.trainer.utils import test_rollout
from dgppo.utils.utils import tree_index
from scripts.export_grafo_cl_csv import export_run


def copy_tree(tree):
    # PPO donates train-state arrays to JAX; CL anchors must own their buffers.
    return jtu.tree_map(lambda x: x.copy() if hasattr(x, "copy") else x, tree)


def make_environment(name, tasks=None, action_encoding="continuous",
                     reach_threshold=None):
    if tasks is None or len(tasks) == 1:
        return GRAFO(task=GRAFO_TASKS[name], max_step=50,
                     action_encoding=action_encoding,
                     reach_threshold=reach_threshold)
    return GRAFO(task=GRAFO_TASKS[name], max_step=50,
                 task_index=tasks.index(name), n_tasks=len(tasks),
                 action_encoding=action_encoding,
                 reach_threshold=reach_threshold)


def make_algorithm(env, args, stage, anchors, importances, shuffle_rng=None):
    method = args.regularization if stage else "none"
    if args.importance_mode == "online" and importances:
        anchors = anchors[-1:]
        importances = importances[-1:]
    return GrafoContinualInforMARL(
        env, env.node_dim, env.edge_dim, env.state_dim, env.action_dim,
        env.num_agents, seed=args.seed + stage, batch_size=args.batch_size,
        epoch_ppo=args.ppo_epochs, rnn_step=16, actor_gnn_layers=4,
        Vl_gnn_layers=4, gnn_arch="unimp", hidden_dim=128,
        gnn_key_dim=16, gnn_heads=3, critic_head_layers=1,
        lr_actor=args.lr, lr_Vl=args.lr,
        clip_eps=.2, coef_ent=.001, value_coefficient=.01,
        max_grad_norm=10., use_rnn=True,
        freeze_encoder=args.frog and stage > 0,
        regularization=method,
        regularization_coefficient=args.regularization_coefficient,
        anchors=anchors, importances=importances,
        shuffle_rng=shuffle_rng,
    )


def make_eval_fn(algo, env):
    def rollout_for_actor(actor_params, keys):
        def episode(key):
            return test_rollout(
                env,
                lambda graph, carry: algo.policy.get_action(
                    actor_params, graph, carry),
                algo.init_rnn_state, key)
        return jax.vmap(episode)(keys)

    return jax.jit(rollout_for_actor)


def evaluate(env, eval_fn, actor_params, keys):
    rollouts = eval_fn(actor_params, keys)
    returns = np.asarray(rollouts.rewards.sum(axis=1))
    step_success = np.asarray(jax.vmap(jax.vmap(env.success))(
        rollouts.next_graph))
    return rollouts, {
        "return_mean": float(returns.mean()),
        "return_min": float(returns.min()),
        "return_max": float(returns.max()),
        "success_rate": float(step_success.any(axis=1).mean()),
        "final_success_rate": float(step_success[:, -1].mean()),
    }


def save_checkpoint(path, algo, state):
    path.mkdir(parents=True, exist_ok=True)
    algo.save(str(path), "model")
    temp = path / "runner.pkl.tmp"
    with temp.open("wb") as handle:
        pickle.dump(state, handle)
    temp.replace(path / "runner.pkl")


def load_checkpoint(path):
    with (path / "runner.pkl").open("rb") as handle:
        return pickle.load(handle)


def metrics_for_matrix(matrix):
    k = len(matrix)
    if not k:
        return {}
    result = {"continual/AP": float(np.mean([matrix[i][i] for i in range(k)]))}
    if k > 1:
        result["continual/AFP"] = float(np.mean(matrix[-1][:k]))
        result["continual/AF"] = float(np.mean([
            matrix[-1][j] - matrix[j][j] for j in range(k - 1)]))
    return result


PAPER_REGULARIZATION_COEFFICIENTS = {
    ("l2", "multi", False): 1e3,
    ("l2", "multi", True): 1e2,
    ("ewc", "multi", False): 1e10,
    ("ewc", "multi", True): 1e4,
    ("ewc", "online", False): 1e11,
    ("ewc", "online", True): 1e5,
    ("mas", "multi", False): 1e7,
    ("mas", "multi", True): 1e6,
    ("mas", "online", False): 1e9,
    ("mas", "online", True): 1e4,
}


def resolve_regularization_coefficient(args):
    if args.regularization_coefficient is not None:
        return args.regularization_coefficient
    if args.regularization == "none":
        return 0.0
    mode = "multi" if args.regularization == "l2" else args.importance_mode
    return PAPER_REGULARIZATION_COEFFICIENTS[
        (args.regularization, mode, args.frog)]


def run(args):
    args.regularization_coefficient = resolve_regularization_coefficient(args)
    if args.reach_threshold is None:
        # The cited JaxInforMARL Formation uses overlap of 0.15 and 0.20
        # radii for discrete actions. The authors' GRAFO threshold is unknown.
        args.reach_threshold = .35 if args.action_encoding == "discrete" else .05
    tasks = tuple(args.tasks.split(",")) if args.tasks else GRAFO_SEQUENCES[args.sequence]
    if len(set(tasks)) != len(tasks):
        raise ValueError("task names must be unique within a sequence")
    reproducibility_config = {
        key: getattr(args, key) for key in (
            "frog", "regularization", "regularization_coefficient",
            "importance_mode", "online_decay", "importance_episodes",
            "importance_batch_size", "importance_steps", "frames_per_task", "n_env",
            "batch_size", "ppo_epochs", "lr", "seed")}
    reproducibility_config.update({
        "grafo_env_version": "relative_obs_old_velocity_snap_action_choices_v3",
        "ppo_implementation_version": "feedforward_two_hidden_isolated_shuffle_v5",
        "action_encoding": args.action_encoding,
        "reward": "negative_mean_l2_plus_mean_first_reach_bonus",
        "reach_threshold": args.reach_threshold,
        "reach_bonus": GRAFO.PARAMS["reach_bonus"],
        "task_identity": "agent_node_one_hot_when_multitask",
    })
    for name in tasks:
        if name not in GRAFO_TASKS:
            raise ValueError(f"unknown GRAFO task: {name}")
    if args.frames_per_task < 1 or args.eval_episodes < 1 or args.reach_threshold <= 0:
        raise ValueError("frame count, evaluation count, and reach threshold must be positive")
    if (args.importance_episodes < 1 or args.importance_steps < 1 or
            args.importance_batch_size < 1):
        raise ValueError("importance counts and batch size must be positive")
    if args.regularization_coefficient < 0:
        raise ValueError("regularization coefficient must be nonnegative")
    if args.eval_interval < 1 or args.save_interval < 1:
        raise ValueError("eval/save intervals must be positive")
    if args.n_env < 1 or args.batch_size < 128 or args.n_env * 128 % args.batch_size:
        raise ValueError("batch_size must divide n_env * 128 and be at least 128")
    if args.n_env % (args.batch_size // 128):
        raise ValueError("batch_size must contain a whole number of environments")

    output = pathlib.Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    state = {
        "format_version": 4, "tasks": tasks, "stage": 0, "next_update": 0,
        "frames": 0, "updates": 0, "heads": {}, "anchors": [],
        "importances": [], "matrix": [], "key": np.asarray(jax.random.PRNGKey(args.seed)),
        "last_actor": None, "last_critic": None,
        "minibatch_rng_state": np.random.RandomState(args.seed).get_state(),
        "config": reproducibility_config,
    }
    if args.resume:
        state = load_checkpoint(pathlib.Path(args.resume))
        if state["format_version"] != 4 or tuple(state["tasks"]) != tasks:
            raise ValueError("checkpoint format or task sequence mismatch")
        if state.get("config", reproducibility_config) != reproducibility_config:
            raise ValueError("checkpoint training configuration mismatch")
        if (not 0 <= state["stage"] <= len(tasks) or
                len(state["matrix"]) != state["stage"] or
                any(len(row) != index + 1
                    for index, row in enumerate(state["matrix"]))):
            raise ValueError("checkpoint evaluation matrix does not match completed stages")
    shuffle_rng = np.random.RandomState()
    shuffle_rng.set_state(state["minibatch_rng_state"])
    run = None
    log_file = (output / "metrics.jsonl").open("a", encoding="utf-8")
    updates_per_stage = math.ceil(args.frames_per_task / (args.n_env * 128))
    pbar = tqdm(total=len(tasks) * updates_per_stage, initial=state["updates"],
                unit="update")
    try:
        run = wandb.init(project="cgmarl", name=args.name, mode=args.wandb_mode,
                         dir=str(output), config=reproducibility_config,
                         id=state.get("wandb_run_id"), resume="allow")
        state["wandb_run_id"] = run.id
        for stage in range(state["stage"], len(tasks)):
            name = tasks[stage]
            env = make_environment(name, tasks, args.action_encoding,
                                   args.reach_threshold)
            anchors = state["anchors"]
            importances = state["importances"]
            algo = make_algorithm(env, args, stage, anchors, importances,
                                  shuffle_rng=shuffle_rng)
            if args.resume and stage == state["stage"] and state["next_update"]:
                algo.load(str(pathlib.Path(args.resume)), "model")
            elif stage:
                actor = with_output_head(copy_tree(state["last_actor"]),
                                         output_head(algo.policy_train_state.params))
                algo.policy_train_state = algo.policy_train_state.replace(params=actor)
                algo.Vl_train_state = algo.Vl_train_state.replace(
                    params=copy_tree(state["last_critic"]))

            evaluators = {name: (env, make_eval_fn(algo, env))}
            for previous in tasks[:stage]:
                previous_env = make_environment(previous, tasks,
                                                args.action_encoding,
                                                args.reach_threshold)
                previous_algo = make_algorithm(previous_env, args, stage,
                                               anchors, importances)
                evaluators[previous] = (
                    previous_env, make_eval_fn(previous_algo, previous_env))

            for update in range(state["next_update"] if stage == state["stage"] else 0,
                                updates_per_stage):
                started = time()
                key = jnp.asarray(state["key"])
                collect_key, key = jax.random.split(key)
                state["key"] = np.asarray(key)
                keys = jax.random.split(collect_key, args.n_env)
                rollout = algo.collect(algo.params, keys)
                info = algo.update(rollout, update)
                state["minibatch_rng_state"] = shuffle_rng.get_state()
                state["frames"] += args.n_env * algo.rollout_horizon
                state["updates"] += 1
                state["stage"] = stage
                state["next_update"] = update + 1
                metrics = {
                    "counters/total_frames": state["frames"],
                    "counters/update": state["updates"],
                    "counters/stage": stage + 1,
                    "train/reward_per_step_mean": float(jnp.mean(rollout.rewards)),
                    "time/update_sec": time() - started,
                }
                metrics.update({key: float(np.asarray(value)) for key, value in info.items()
                                if np.asarray(value).ndim == 0})
                media = {}
                evaluate_now = ((update + 1) % args.eval_interval == 0 or
                                update + 1 == updates_per_stage)
                if evaluate_now:
                    state["heads"][name] = copy_tree(
                        output_head(algo.policy_train_state.params))
                    row = []
                    for previous in tasks[:stage + 1]:
                        previous_env, eval_fn = evaluators[previous]
                        actor = with_output_head(algo.policy_train_state.params,
                                                 state["heads"][previous])
                        eval_key = jax.random.fold_in(jax.random.PRNGKey(args.seed + 10_000),
                                                       stage * 1_000_000 + update * 100 +
                                                       tasks.index(previous))
                        eval_keys = jax.random.split(eval_key, args.eval_episodes)
                        eval_rollout, result = evaluate(previous_env, eval_fn,
                                                        actor, eval_keys)
                        metrics.update({f"eval/{previous}/{key}": value
                                        for key, value in result.items()})
                        row.append(result["success_rate"])
                        if previous == name and args.gif:
                            video_path = output / "media" / f"stage{stage+1}_update{update+1}.gif"
                            video_path.parent.mkdir(parents=True, exist_ok=True)
                            try:
                                single_rollout = tree_index(eval_rollout, 0)
                                previous_env.render_video(
                                    single_rollout, video_path,
                                    Ta_is_unsafe=np.zeros(
                                        single_rollout.costs.shape[:2], dtype=bool),
                                    dpi=args.video_dpi)
                                media["eval/video"] = wandb.Video(str(video_path), format="gif")
                            except (OSError, RuntimeError, ValueError, IndexError) as error:
                                tqdm.write(f"GIF unavailable: {error}")
                    if update + 1 == updates_per_stage:
                        state["matrix"].append(row)
                        metrics.update(metrics_for_matrix(state["matrix"]))
                    pbar.set_postfix(task=name, success=f"{row[-1]:.2f}")
                    tqdm.write(f"{name} update {update+1}: success={row[-1]:.3f}")

                log_file.write(json.dumps(metrics, sort_keys=True) + "\n")
                log_file.flush()
                if evaluate_now:
                    try:
                        export_run(output)
                    except OSError as error:
                        tqdm.write(f"CSV export unavailable: {error}")
                wandb.log(metrics | media, step=state["frames"])
                pbar.update(1)
                if ((update + 1) % args.save_interval == 0 or
                        update + 1 == updates_per_stage):
                    save_checkpoint(output / "checkpoints" / "latest", algo, state)

            state["heads"][name] = copy_tree(output_head(algo.policy_train_state.params))
            state["last_actor"] = copy_tree(algo.policy_train_state.params)
            state["last_critic"] = copy_tree(algo.Vl_train_state.params)
            if args.regularization in ("ewc", "mas") and stage < len(tasks) - 1:
                importance_key = jax.random.fold_in(jax.random.PRNGKey(args.seed + 20_000), stage)
                importance_rollout = algo.collect_importance(
                    jax.random.split(importance_key, args.importance_episodes),
                    horizon=args.importance_steps)
                importance = algo.estimate_importance(
                    importance_rollout, method=args.regularization,
                    batch_size=args.importance_batch_size)
                if args.importance_mode == "online" and state["importances"]:
                    importance = jtu.tree_map(
                        lambda old, new: args.online_decay * old + new,
                        state["importances"][-1], importance)
                    state["anchors"] = [copy_tree(algo.policy_train_state.params)]
                    state["importances"] = [importance]
                else:
                    state["anchors"].append(copy_tree(algo.policy_train_state.params))
                    state["importances"].append(importance)
            elif args.regularization == "l2":
                state["anchors"] = [copy_tree(algo.policy_train_state.params)]
                state["importances"] = [None]
            state["stage"] = stage + 1
            state["next_update"] = 0
            save_checkpoint(output / "checkpoints" / "latest", algo, state)
            save_checkpoint(output / "checkpoints" / f"stage_{stage + 1}", algo, state)
        (output / "summary.json").write_text(json.dumps(
            {"tasks": tasks, "matrix": state["matrix"],
             "metrics": metrics_for_matrix(state["matrix"])}, indent=2))
    finally:
        pbar.close()
        log_file.close()
        if run is not None:
            wandb.finish()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", choices=tuple(GRAFO_SEQUENCES), default="TS")
    parser.add_argument("--tasks", help="comma-separated custom task sequence")
    parser.add_argument("--action-encoding", choices=("continuous", "discrete"),
                        default="continuous")
    parser.add_argument("--reach-threshold", type=float,
                        help="goal distance threshold; defaults to 0.05 continuous or 0.35 discrete")
    parser.add_argument("--frog", action="store_true")
    parser.add_argument("--regularization", choices=("none", "l2", "ewc", "mas"),
                        default="none")
    parser.add_argument("--regularization-coefficient", type=float,
                        help="defaults to the paper's coefficient for the selected method")
    parser.add_argument("--importance-mode", choices=("multi", "online"), default="multi")
    parser.add_argument("--online-decay", type=float, default=.9)
    parser.add_argument("--importance-episodes", type=int, default=64)
    parser.add_argument("--importance-steps", type=int, default=50)
    parser.add_argument("--importance-batch-size", type=int, default=16)
    parser.add_argument("--frames-per-task", type=int, default=5_000_000)
    parser.add_argument("--n-env", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--eval-episodes", type=int, default=64)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--save-interval", type=int, default=100)
    parser.add_argument("--gif", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--video-dpi", type=int, default=60)
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"),
                        default="online")
    parser.add_argument("--name")
    parser.add_argument("--output", default="./logs/cgmarl")
    parser.add_argument("--resume", help="checkpoint directory with runner.pkl")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
