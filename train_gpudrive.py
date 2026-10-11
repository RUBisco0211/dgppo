"""Independent DGPPO training entry point for GPUDrive."""

import argparse
import datetime
import json
import os
import string
import time
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

from dgppo.algo.gpudrive_dgppo import GPUDriveDGPPO
from dgppo.algo.gpudrive_informarl import GPUDriveInforMARL
from dgppo.algo.gpudrive_gnn_ippo import GPUDriveGNNIPPO
from dgppo.env.gpudrive import GPUDriveDGPPOAdapter, GPUDriveDGPPOConfig
from dgppo.trainer.gpudrive_runner import GPUDriveRolloutRunner


def make_env(args, seed, worlds):
    import torch
    from gpudrive.env.config import EnvConfig, RenderConfig
    from gpudrive.env.dataset import SceneDataLoader
    from gpudrive.env.env_torch import GPUDriveTorchEnv
    is_gnn_ippo = args.algorithm == "gnn_ippo"
    cfg = EnvConfig(num_worlds=worlds, max_controlled_agents=64,
                    remove_non_vehicles=True, lidar_obs=not is_gnn_ippo,
                    disable_classic_obs=False, collision_behavior="ignore",
                    reward_type="weighted_combination",
                    collision_weight=(-0.5 if is_gnn_ippo else 0.0),
                    off_road_weight=(-0.5 if is_gnn_ippo else 0.0),
                    goal_achieved_weight=1.0,
                    dynamics_model="classic",
                    # Classic dynamics applies tan(steering). Using [-pi, pi]
                    # puts discrete actions on the +/-pi/2 singularities and
                    # turns vehicle states into NaNs. Both discrete and
                    # continuous policies use the same safe physical limit.
                    steer_actions=torch.linspace(
                        -args.max_steer_rad, args.max_steer_rad, 13
                    ))
    loader = SceneDataLoader(root=args.data_dir, batch_size=worlds,
                             dataset_size=args.dataset_size,
                             sample_with_replacement=True, seed=seed,
                             file_prefix=args.scene_file)
    return GPUDriveTorchEnv(cfg, loader, max_cont_agents=64, device=args.device,
                            action_type="continuous", render_config=RenderConfig())


def metrics_from_rollout(rollout):
    alive = np.asarray(rollout.alive_mask)
    rewards = np.asarray(rollout.rewards)
    goal_progress_reward = np.asarray(rollout.goal_progress_reward)
    goal_reward = np.asarray(rollout.goal_reward)
    vehicle_collision_reward = np.asarray(rollout.vehicle_collision_reward)
    offroad_reward = np.asarray(rollout.offroad_reward)
    lane_deviation_reward = np.asarray(rollout.lane_deviation_reward)
    goal_distance = np.asarray(rollout.goal_distance)
    lane_deviation = np.asarray(rollout.lane_deviation)
    costs = np.asarray(rollout.costs)
    worst_cost = costs.max(axis=-1)
    terminated = np.asarray(rollout.terminated)
    goals = np.asarray(rollout.goal_achieved)
    collisions = np.asarray(rollout.native_collision)
    road_collisions = np.asarray(rollout.road_collision)
    cc = np.asarray(rollout.controlled_collision)
    cb = np.asarray(rollout.background_collision)
    clearance = np.asarray(rollout.min_clearance)
    episode_end = np.asarray(rollout.episode_end)
    episode_length = np.asarray(rollout.episode_length)
    actions = np.asarray(rollout.actions)
    denom = max(int(alive.sum()), 1)
    # Episode-level rates must exclude the fixed 64-slot padding.  Averaging
    # over every slot made sparse scenes look artificially unsuccessful/safe.
    valid_agents = alive.any(axis=1)
    valid_agent_denom = max(int(valid_agents.sum()), 1)
    min_clearance = float(np.where(alive, clearance, np.inf).min()) if alive.any() else float("nan")
    metrics = {
        "reward/mean": float(np.where(alive, rewards, 0.0).sum() / denom),
        "reward/chunk_return": float(np.where(alive, rewards, 0.0).sum(axis=(1, 2)).mean()),
        "reward/goal_progress_mean": float(
            np.where(alive, goal_progress_reward, 0.0).sum() / denom),
        "reward/goal_mean": float(np.where(alive, goal_reward, 0.0).sum() / denom),
        "reward/vehicle_collision_mean": float(np.where(alive, vehicle_collision_reward, 0.0).sum() / denom),
        "reward/offroad_mean": float(np.where(alive, offroad_reward, 0.0).sum() / denom),
        "reward/lane_deviation_mean": float(
            np.where(alive, lane_deviation_reward, 0.0).sum() / denom),
        "task/goal_distance_m": float(np.where(alive, goal_distance, 0.0).sum() / denom),
        "task/lane_deviation_m": float(np.where(alive, lane_deviation, 0.0).sum() / denom),
        "diagnostics/padding_state_rate": float(((goal_distance > 1000.0) & alive).sum() / denom),
        "policy/action_abs_mean": float(np.where(alive[..., None], np.abs(actions), 0.0).sum()
                                          / max(int(alive.sum()) * actions.shape[-1], 1)),
        "policy/action_abs_max": float(np.where(alive[..., None], np.abs(actions), 0.0).max()),
        "safety/cost_mean": float(np.where(alive, worst_cost, 0.0).sum() / denom),
        "safety/vehicle_cost_mean": float(np.where(alive, costs[..., 0], 0.0).sum() / denom),
        "safety/road_cost_mean": float(np.where(alive, costs[..., 1], 0.0).sum() / denom),
        "safety/unsafe_rate": float(((worst_cost > 0) & alive).sum() / denom),
        "safety/vehicle_unsafe_rate": float(((costs[..., 0] > 0) & alive).sum() / denom),
        "safety/road_unsafe_rate": float(((costs[..., 1] > 0) & alive).sum() / denom),
        "safety/termination_rate": float(
            (terminated.any(axis=1) & valid_agents).sum() / valid_agent_denom),
        "task/success_rate": float(
            (goals & alive).any(axis=1).sum() / valid_agent_denom),
        "safety/native_collision_rate": float(
            (collisions & alive).any(axis=1).sum() / valid_agent_denom),
        "safety/road_collision_rate": float(
            (road_collisions & alive).any(axis=1).sum() / valid_agent_denom),
        "safety/controlled_controlled_collision_rate": float(
            (cc & alive).any(axis=1).sum() / valid_agent_denom),
        "safety/controlled_background_collision_rate": float(
            (cb & alive).any(axis=1).sum() / valid_agent_denom),
        "safety/min_clearance_m": min_clearance,
        "episode/completed_count": int(episode_end.sum()),
        "episode/mean_alive_agent_steps_per_world": float(alive.sum(axis=(1, 2)).mean()),
    }
    if episode_end.any():
        metrics["episode/mean_length"] = float(episode_length[episode_end].mean())
    return metrics


def evaluate(algo, runner, key, horizon, video_path=None):
    def policy(graph, unused_key):
        action = algo.deterministic_actions(graph)
        return action, jnp.zeros(action.shape[:-1])
    frames = []
    media_failed = [False]
    capture_step = [0]
    controlled_color = "#1565c0"
    background_color = "#4b5563"

    def mark_controlled_vehicles(fig):
        """Make every car visible and emphasize controlled cars and goals."""
        from gpudrive.datatypes.observation import AGENT_SCALE

        ax = fig.axes[0]
        adapter = runner.adapter
        slots = adapter._controlled_slots()[0]
        if adapter._controlled_valid is not None:
            slots = slots[adapter._controlled_valid[0]]
        absolute = adapter.env.sim.absolute_self_observation_tensor().to_torch()[0]
        absolute_np = absolute.detach().cpu().numpy()
        physical = np.isfinite(absolute_np).all(axis=-1) & (absolute_np[:, 10] > 0)
        background = absolute_np[physical]
        state = absolute[slots].detach().cpu().numpy()

        # Screen-space markers remain legible after the full simulator figure
        # is reduced to a 900 px GIF. Native vehicle polygons become nearly
        # invisible at this scale.
        if len(background):
            ax.scatter(
                background[:, 0], background[:, 1], marker="s", s=24,
                facecolors=background_color, edgecolors="white", linewidths=0.45,
                alpha=0.9, zorder=24, label="other vehicle",
            )
        for vehicle in state:
            position, goal = vehicle[:2], vehicle[8:10]
            heading = vehicle[7]
            length, width = vehicle[10:12] * AGENT_SCALE
            direction = np.array([np.cos(heading), np.sin(heading)])
            lateral = np.array([-np.sin(heading), np.cos(heading)])
            corners = np.stack(
                [
                    position + 0.5 * length * direction + 0.5 * width * lateral,
                    position + 0.5 * length * direction - 0.5 * width * lateral,
                    position - 0.5 * length * direction - 0.5 * width * lateral,
                    position - 0.5 * length * direction + 0.5 * width * lateral,
                ]
            )
            outline = np.concatenate([corners, corners[:1]], axis=0)
            ax.fill(
                outline[:, 0], outline[:, 1], facecolor=controlled_color,
                edgecolor="white", linewidth=1.2, alpha=1.0, zorder=31,
            )
            ax.scatter(
                position[0], position[1], marker="o", s=48,
                facecolors=controlled_color, edgecolors="white",
                linewidths=0.8, zorder=32,
            )
            ax.scatter(
                goal[0], goal[1], marker="*", s=115,
                facecolors=controlled_color, edgecolors="white",
                linewidths=0.7, zorder=30,
            )

        # Focus on the agents and controlled goals rather than the full map's
        # empty margins. Include all physical vehicles so traffic context is
        # not silently cropped out.
        focus = background[:, :2] if len(background) else state[:, :2]
        if len(state):
            focus = np.concatenate([focus, state[:, :2], state[:, 8:10]], axis=0)
        focus = focus[np.isfinite(focus).all(axis=-1)]
        if len(focus):
            low, high = focus.min(axis=0), focus.max(axis=0)
            span = np.maximum(high - low, 20.0)
            margin = np.maximum(0.08 * span, 8.0)
            ax.set_xlim(low[0] - margin[0], high[0] + margin[0])
            ax.set_ylim(low[1] - margin[1], high[1] + margin[1])
            ax.set_aspect("equal", adjustable="box")
    def capture():
        capture_step[0] += 1
        if video_path is None or media_failed[0]:
            return
        # Matplotlib rendering dominated short evaluations and previously
        # produced a zero-byte GIF after collecting all 91 full-size frames.
        # Thirty frames are sufficient for inspection and keep eval bounded.
        if capture_step[0] % 3 != 1 and capture_step[0] != horizon:
            return
        try:
            import matplotlib.pyplot as plt
            fig = runner.adapter.env.vis.plot_simulator_state([0])[0]
            mark_controlled_vehicles(fig)
            fig.canvas.draw()
            frame = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
            if max(frame.shape[:2]) > 900:
                from PIL import Image
                image = Image.fromarray(frame)
                image.thumbnail((900, 900))
                frame = np.asarray(image)
            frames.append(frame)
            plt.close(fig)
        except (OSError, RuntimeError, ValueError, ImportError) as error:
            media_failed[0] = True
            tqdm.write(f"Evaluation rendering disabled: {error}")
    rollout, key = runner.collect(policy, key, horizon=horizon, frame_callback=capture)
    if video_path is not None and frames and not media_failed[0]:
        try:
            try:
                import imageio.v2 as imageio
                imageio.mimsave(video_path, frames, fps=10)
            except ImportError:
                from PIL import Image
                images = [Image.fromarray(frame) for frame in frames]
                images[0].save(video_path, save_all=True, append_images=images[1:],
                               duration=100, loop=0)
        except (OSError, RuntimeError, ValueError, ImportError) as error:
            media_failed[0] = True
            tqdm.write(f"Evaluation video encoding disabled: {error}")
    return {f"eval/{k}": v for k, v in metrics_from_rollout(rollout).items()}, key


def main(args):
    for name in ("iterations", "horizon", "eval_horizon", "eval_interval", "save_interval", "video_interval"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if not 0.0 < args.max_steer_rad < 1.4:
        raise ValueError("--max-steer-rad must be in (0, 1.4) to avoid classic-model tan singularities")
    if args.debug:
        args.wandb_mode = "disabled"
    if args.eval_worlds != args.num_worlds:
        raise ValueError(
            "--eval-worlds must equal --num-worlds: GPUDrive/Madrona supports one "
            "CUDA simulator per process, so training, safety, and evaluation share it"
        )
    if args.run_name is None:
        timestamp = datetime.datetime.now().strftime("%m%d%H%M%S")
        random_id = "".join(
            np.random.default_rng().choice(list(string.ascii_uppercase), size=4)
        )
        local_run_name = f"seed{args.seed}_{timestamp}_{random_id}"
        algorithm_label = (
            f"{args.algorithm}_{args.action_mode}"
            if args.algorithm == "gnn_ippo" else args.algorithm
        )
        args.run_name = f"{algorithm_label}_seed{args.seed:03}_{timestamp}_{random_id}"
        run_dir = Path(args.log_dir) / args.algorithm / local_run_name
    else:
        run_dir = Path(args.log_dir) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.json", "w", encoding="utf-8") as config_file:
        json.dump(vars(args), config_file, indent=2, sort_keys=True)
    local_log = open(run_dir / "metrics.jsonl", "a", encoding="utf-8")
    wandb_run = None
    train_env = None
    progress = None
    try:
        if args.wandb_mode != "disabled":
            import wandb
            wandb_run = wandb.init(project=args.wandb_project, name=args.run_name,
                                   group=args.wandb_group, mode=args.wandb_mode,
                                   dir=str(run_dir), config=vars(args))
            wandb_run.define_metric("counters/total_frames")
            wandb_run.define_metric("*", step_metric="counters/total_frames")
        cfg = GPUDriveDGPPOConfig(geometry=args.geometry,
                                  model_profile=args.model_profile,
                                  rollout_horizon=args.horizon,
                                  num_road_segments=(200 if args.algorithm == "gnn_ippo" else 32),
                                  road_feature_dim=(14 if args.algorithm == "gnn_ippo" else 8),
                                  num_controlled=(64 if args.control_all_vehicles else 5),
                                  num_background=(0 if args.control_all_vehicles else 10),
                                  control_all_vehicles=args.control_all_vehicles,
                                  use_native_controlled_mask=args.native_controlled_mask,
                                  keep_all_scene_vehicles=args.keep_all_vehicles,
                                  background_behavior=args.background_behavior,
                                  goal_progress_weight=args.goal_progress_weight)
        train_env = make_env(args, args.seed, args.num_worlds)
        train_adapter = GPUDriveDGPPOAdapter(train_env, cfg)
        train_adapter.prepare_scenes(args.seed)
        scene_vehicle_counts = (
            train_adapter._scene_valid.sum(axis=1).detach().cpu().numpy()
        )
        scene_controlled_counts = (
            train_adapter._controlled_valid.sum(axis=1).detach().cpu().numpy()
        )
        tqdm.write(
            "Prepared physical vehicles per world: "
            f"min={scene_vehicle_counts.min()}, "
            f"mean={scene_vehicle_counts.mean():.1f}, "
            f"max={scene_vehicle_counts.max()}"
        )
        collect_costs = args.algorithm == "dgppo"
        graph_fn = train_adapter.ippo_graph_batch if args.algorithm == "gnn_ippo" else None
        train_runner = GPUDriveRolloutRunner(
            train_adapter, collect_costs=collect_costs, graph_fn=graph_fn)
        actor_batch = (train_adapter.ippo_graph_batch() if args.algorithm == "gnn_ippo"
                       else train_adapter.local_graph_batch())
        actor_graph = jax.tree_util.tree_map(
            lambda value: None if value is None else value[0],
            actor_batch.graph,
            is_leaf=lambda value: value is None,
        )
        critic_graph = jax.tree_util.tree_map(
            lambda value: None if value is None else value[0],
            train_adapter.scene_graph_batch(),
            is_leaf=lambda value: value is None,
        )
        actor_layers, vl_layers, vh_layers = cfg.gnn_layers
        if args.algorithm == "gnn_ippo":
            algo = GPUDriveGNNIPPO(
                actor_graph, action_mode=args.action_mode,
                actor_gnn_layers=actor_layers, critic_gnn_layers=vl_layers,
                seed=args.seed, lr_critic=args.lr_vl, coef_ent=args.coef_ent,
                minibatch_worlds=args.minibatch_worlds,
                epoch_ppo=args.epoch_ppo, n_agents=cfg.num_controlled,
                base_node_dim=18, road_slots=cfg.num_road_segments,
                road_feature_dim=cfg.road_feature_dim)
        elif args.algorithm == "informarl":
            algo = GPUDriveInforMARL(
                actor_graph, critic_graph, actor_gnn_layers=actor_layers,
                Vl_gnn_layers=vl_layers, seed=args.seed, lr_Vl=args.lr_vl,
                coef_ent=args.coef_ent,
                minibatch_worlds=args.minibatch_worlds, epoch_ppo=args.epoch_ppo,
                n_agents=cfg.num_controlled, base_node_dim=18,
                road_slots=cfg.num_road_segments, road_feature_dim=cfg.road_feature_dim,
                behavior_cloning=args.behavior_cloning,
                bc_checkpoint=args.bc_checkpoint, bc_steps=args.bc_steps)
        else:
            algo = GPUDriveDGPPO(actor_graph, critic_graph, actor_gnn_layers=actor_layers,
                                 Vl_gnn_layers=vl_layers, Vh_gnn_layers=vh_layers,
                                 seed=args.seed, dt=args.dt, cbf_alpha=args.cbf_alpha,
                                 minibatch_worlds=args.minibatch_worlds,
                                 epoch_ppo=args.epoch_ppo, n_cost=cfg.n_cost)
        # Madrona aborts when a second CUDA executor is constructed in the same
        # process. Keep nominal/safety/evaluation rollouts logically separate,
        # but run them sequentially from resets of one simulator.
        safety_runner = GPUDriveRolloutRunner(
            train_adapter, collect_costs=collect_costs, graph_fn=graph_fn)
        counters = {"iteration": 0, "total_frames": 0, "updates": 0}
        resume_state = {}
        if args.resume is not None:
            resume_dir, resume_step = os.path.split(os.path.normpath(args.resume))
            resume_state = algo.load(resume_dir, resume_step)
            counters.update({key: resume_state[key] for key in counters if key in resume_state})
            tqdm.write(
                "Resume restores model, optimizer, RNG and counters; GPUDrive simulator "
                "state starts from freshly prepared scenes."
            )

        eval_runner = GPUDriveRolloutRunner(
            train_adapter, collect_costs=collect_costs, graph_fn=graph_fn)
        train_key = jnp.asarray(resume_state.get("train_key", jax.random.PRNGKey(args.seed + 1000)))
        eval_key = jnp.asarray(resume_state.get("eval_key", jax.random.PRNGKey(args.eval_seed + 1000)))
        safety_key = jnp.asarray(resume_state.get("safety_key", jax.random.PRNGKey(args.seed + 6000)))
        def checkpoint_state():
            return counters | {
                "train_key": np.asarray(train_key), "eval_key": np.asarray(eval_key),
                "safety_key": np.asarray(safety_key),
                "environment_state_restored": False,
            }
        progress = tqdm(range(counters["iteration"], args.iterations),
                        initial=counters["iteration"], total=args.iterations,
                        desc=f"GPUDrive {args.algorithm}", unit="iter")
        for iteration in progress:
            wandb_media = None
            started = time.perf_counter()
            # Re-sample which five valid cars are controlled on every training
            # rollout while retaining every physical vehicle in the scene.
            train_adapter.prepare_scenes(args.seed + iteration + 1, fixed_eval=False)
            train_runner.reset_masks()
            def rollout_progress(label):
                return lambda step, total: tqdm.write(
                    f"iteration {iteration + 1}: {label} rollout {step}/{total}"
                )
            rollout, train_key = train_runner.collect(
                algo.sample_actions, train_key, args.horizon,
                progress_callback=rollout_progress("train"))
            def deterministic_policy(graph, unused_key):
                action = algo.deterministic_actions(graph)
                return action, jnp.zeros(action.shape[:-1])
            safety_rollout = None
            if args.algorithm == "dgppo":
                train_adapter.reset_worlds(range(train_adapter.num_worlds))
                safety_runner.reset_masks()
                safety_rollout, safety_key = safety_runner.collect(
                    deterministic_policy, safety_key, args.horizon,
                    progress_callback=rollout_progress("safety"))
            collection_s = time.perf_counter() - started
            update_started = time.perf_counter()
            update_info = algo.update(rollout, safety_rollout)
            update_s = time.perf_counter() - update_started
            counters["iteration"] = iteration + 1
            counters["updates"] += algo.last_update_count
            rollout_count = 2 if args.algorithm == "dgppo" else 1
            if args.algorithm in ("informarl", "gnn_ippo"):
                current_frames = int(np.asarray(rollout.alive_mask).sum())
            else:
                current_frames = rollout_count * args.num_worlds * args.horizon
            counters["total_frames"] += current_frames
            metrics = metrics_from_rollout(rollout)
            metrics.update({k: float(np.asarray(v)) for k, v in update_info.items()})
            metrics.update({"counters/iteration": counters["iteration"],
                            "counters/updates": counters["updates"],
                            "counters/total_frames": counters["total_frames"],
                            "counters/current_frames": current_frames,
                            "timer/collection_s": collection_s, "timer/update_s": update_s,
                            "performance/world_steps_per_s": current_frames / max(collection_s, 1e-9)})
            metrics.update({
                "scene/vehicle_count_min": int(scene_vehicle_counts.min()),
                "scene/vehicle_count_mean": float(scene_vehicle_counts.mean()),
                "scene/vehicle_count_max": int(scene_vehicle_counts.max()),
                "scene/controlled_count_min": int(scene_controlled_counts.min()),
                "scene/controlled_count_mean": float(scene_controlled_counts.mean()),
                "scene/controlled_count_max": int(scene_controlled_counts.max()),
            })
            finite_keys = ["policy/loss", "Vl/loss", "reward/mean", "safety/cost_mean"]
            if args.algorithm == "dgppo":
                finite_keys.extend(["Vh/loss", "safety/cbf_violation"])
            invalid = [key for key in finite_keys if not np.isfinite(metrics[key])]
            if invalid:
                raise FloatingPointError(f"non-finite training metrics: {invalid}")
            if metrics["diagnostics/padding_state_rate"] > 0:
                raise RuntimeError(
                    "an alive controlled agent entered GPUDrive's padding state")
            if counters["iteration"] % args.eval_interval == 0:
                eval_started = time.perf_counter()
                train_adapter.prepare_scenes(args.eval_seed, fixed_eval=True)
                eval_runner.reset_masks()
                video_path = None
                if not args.no_video and counters["iteration"] % args.video_interval == 0:
                    video_path = run_dir / "videos" / f"eval_{counters['iteration']:08d}.gif"
                    video_path.parent.mkdir(parents=True, exist_ok=True)
                eval_metrics, eval_key = evaluate(
                    algo, eval_runner, eval_key, args.eval_horizon, video_path)
                if wandb_run is not None and video_path is not None and video_path.exists():
                    import wandb
                    wandb_media = wandb.Video(str(video_path), format="gif")
                metrics.update(eval_metrics)
                metrics["timer/evaluation_s"] = time.perf_counter() - eval_started
            if counters["iteration"] % args.save_interval == 0:
                save_started = time.perf_counter()
                algo.save(str(run_dir / "checkpoints"), counters["iteration"], checkpoint_state())
                metrics["timer/checkpoint_s"] = time.perf_counter() - save_started
            metrics["timer/iteration_s"] = time.perf_counter() - started
            local_log.write(json.dumps(metrics, sort_keys=True) + "\n"); local_log.flush()
            if wandb_run is not None:
                if wandb_media is not None:
                    wandb_run.log({"eval/video": wandb_media},
                                  step=counters["total_frames"], commit=False)
                wandb_run.log(metrics, step=counters["total_frames"])
            progress.set_postfix(reward=f"{metrics['reward/mean']:.3f}",
                                 unsafe=f"{metrics['safety/unsafe_rate']:.3f}")
            if (args.total_agent_steps is not None
                    and counters["total_frames"] >= args.total_agent_steps):
                tqdm.write(
                    f"Reached target of {args.total_agent_steps:,} valid agent transitions"
                )
                break
        algo.save(str(run_dir / "checkpoints"), counters["iteration"], checkpoint_state())
    finally:
        if progress is not None:
            progress.close()
        local_log.close()
        if train_env is not None:
            train_env.close()
        if wandb_run is not None:
            wandb_run.finish()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", choices=["dgppo", "informarl", "gnn_ippo"], default="dgppo")
    parser.add_argument("--action-mode", choices=["discrete", "continuous"], default="discrete")
    parser.add_argument("--data-dir", default="gpudrive/data/processed/examples")
    parser.add_argument(
        "--scene-file", default="tfrecord-00000-of-01000_402.json",
        help="exact filename/prefix of the single homogeneous source intersection",
    )
    parser.add_argument(
        "--dataset-size", type=int, default=1,
        help="number of source scenes; 1 replicates one scene across all parallel worlds",
    )
    parser.add_argument(
        "--keep-all-vehicles", action="store_true",
        help="retain every physical vehicle while actor graphs keep nearest-neighbor slots",
    )
    parser.add_argument(
        "--control-all-vehicles", action="store_true",
        help="control every native non-trivial vehicle with masked 64-slot graphs",
    )
    parser.add_argument(
        "--native-controlled-mask", action="store_true",
        help="use GPUDrive's native controlled-agent mask without reselection",
    )
    parser.add_argument(
        "--background-behavior", choices=["expert", "static"], default="expert",
        help="expert replays background trajectories; static parks the ten background vehicles",
    )
    parser.add_argument(
        "--goal-progress-weight", type=float, default=1.0,
        help="deprecated compatibility option; official GPUDrive reward ignores it",
    )
    parser.add_argument("--num-worlds", type=int, default=16)
    parser.add_argument("--eval-worlds", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument(
        "--total-agent-steps", type=int,
        help="stop after this many valid agent transitions",
    )
    parser.add_argument("--horizon", type=int, default=91)
    parser.add_argument("--eval-horizon", type=int, default=91)
    parser.add_argument("--model-profile", choices=["lightweight", "dgppo_default"], default="lightweight")
    parser.add_argument("--geometry", choices=["oriented_box", "ellipse"], default="oriented_box")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-seed", type=int, default=10000)
    parser.add_argument("--dt", type=float, default=0.1,
                        help="GPUDrive classic dynamics timestep in seconds")
    parser.add_argument(
        "--max-steer-rad", type=float, default=0.3,
        help="physical classic-model steering limit; avoids tan() singularities",
    )
    parser.add_argument("--cbf-alpha", type=float, default=2.0,
                        help="CBF exponential coefficient")
    parser.add_argument("--minibatch-worlds", type=int, default=4,
                        help="number of worlds per sequential PPO minibatch")
    parser.add_argument("--epoch-ppo", type=int, default=1)
    parser.add_argument("--lr-vl", type=float, default=3e-4)
    parser.add_argument("--coef-ent", type=float, default=1e-4)
    parser.add_argument("--behavior-cloning", action="store_true")
    parser.add_argument("--bc-checkpoint")
    parser.add_argument("--bc-steps", type=int, default=0)
    parser.add_argument("--eval-interval", type=int, default=50)
    parser.add_argument("--save-interval", type=int, default=50)
    parser.add_argument("--video-interval", type=int, default=50)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--log-dir", default="logs/gpudrive")
    parser.add_argument("--run-name")
    parser.add_argument("--wandb-project", default="gpudrive")
    parser.add_argument("--wandb-group")
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="disabled")
    parser.add_argument("--resume")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
