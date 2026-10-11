"""Evaluate a GPUDrive GNN-IPPO checkpoint on one scene without training."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp

from dgppo.algo.gpudrive_gnn_ippo import GPUDriveGNNIPPO
from dgppo.env.gpudrive import GPUDriveDGPPOAdapter, GPUDriveDGPPOConfig
from dgppo.trainer.gpudrive_runner import GPUDriveRolloutRunner
from train_gpudrive import evaluate, make_env


def main(args):
    env_args = SimpleNamespace(
        algorithm="gnn_ippo",
        action_mode=args.action_mode,
        data_dir=args.data_dir,
        dataset_size=1,
        scene_file=args.scene_file,
        device=args.device,
        max_steer_rad=args.max_steer_rad,
    )
    env = None
    try:
        env = make_env(env_args, args.eval_seed, args.worlds)
        cfg = GPUDriveDGPPOConfig(
            model_profile="lightweight",
            rollout_horizon=args.horizon,
            num_road_segments=200,
            road_feature_dim=14,
            num_controlled=64,
            num_background=0,
            control_all_vehicles=True,
            keep_all_scene_vehicles=True,
        )
        adapter = GPUDriveDGPPOAdapter(env, cfg)
        adapter.prepare_scenes(args.eval_seed, fixed_eval=True)
        runner = GPUDriveRolloutRunner(
            adapter, collect_costs=False, graph_fn=adapter.ippo_graph_batch
        )
        actor_graph = jax.tree_util.tree_map(
            lambda value: None if value is None else value[0],
            adapter.ippo_graph_batch().graph,
            is_leaf=lambda value: value is None,
        )
        actor_layers, critic_layers, _ = cfg.gnn_layers
        algo = GPUDriveGNNIPPO(
            actor_graph,
            action_mode=args.action_mode,
            actor_gnn_layers=actor_layers,
            critic_gnn_layers=critic_layers,
            seed=0,
            lr_critic=3e-4,
            coef_ent=1e-2,
            minibatch_worlds=1,
            epoch_ppo=1,
            n_agents=cfg.num_controlled,
            base_node_dim=18,
            road_slots=cfg.num_road_segments,
            road_feature_dim=cfg.road_feature_dim,
        )
        checkpoint = Path(args.checkpoint).resolve()
        algo.load(str(checkpoint.parent.parent), checkpoint.parent.name)

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(args.scene_file).stem
        video_path = output_dir / f"{stem}.gif"
        runner.reset_masks()
        metrics, _ = evaluate(
            algo,
            runner,
            jnp.asarray(jax.random.PRNGKey(args.eval_seed + 1000)),
            args.horizon,
            video_path,
        )
        metrics.update({
            "scene/file": args.scene_file,
            "scene/worlds": args.worlds,
            "scene/physical_vehicle_count_mean": float(
                adapter._scene_valid.sum(axis=1).float().mean().cpu()
            ),
            "scene/controlled_vehicle_count_mean": float(
                adapter._controlled_valid.sum(axis=1).float().mean().cpu()
            ),
            "checkpoint": str(checkpoint),
        })
        (output_dir / f"{stem}.json").write_text(
            json.dumps(metrics, indent=2, sort_keys=True) + "\n"
        )
        print(json.dumps(metrics, sort_keys=True), flush=True)
    finally:
        if env is not None:
            env.close()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--scene-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data-dir", default="gpudrive/data/processed/examples")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--worlds", type=int, default=1)
    parser.add_argument("--horizon", type=int, default=91)
    parser.add_argument("--eval-seed", type=int, default=10000)
    parser.add_argument("--max-steer-rad", type=float, default=0.3)
    parser.add_argument("--action-mode", choices=("discrete", "continuous"), default="discrete")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
