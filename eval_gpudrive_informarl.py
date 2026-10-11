"""Evaluate a GPUDrive InforMARL checkpoint without performing training updates."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp

from dgppo.algo.gpudrive_informarl import GPUDriveInforMARL
from dgppo.env.gpudrive import GPUDriveDGPPOAdapter, GPUDriveDGPPOConfig
from dgppo.trainer.gpudrive_runner import GPUDriveRolloutRunner
from train_gpudrive import evaluate, make_env


def main(args):
    env_args = SimpleNamespace(
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
            rollout_horizon=args.horizon,
            num_controlled=64,
            num_background=0,
            control_all_vehicles=True,
            keep_all_scene_vehicles=True,
        )
        adapter = GPUDriveDGPPOAdapter(env, cfg)
        adapter.prepare_scenes(args.eval_seed, fixed_eval=True)
        runner = GPUDriveRolloutRunner(adapter, collect_costs=False)
        actor_graph = jax.tree_util.tree_map(
            lambda value: None if value is None else value[0],
            adapter.local_graph_batch().graph,
            is_leaf=lambda value: value is None,
        )
        critic_graph = jax.tree_util.tree_map(
            lambda value: None if value is None else value[0],
            adapter.scene_graph_batch(),
            is_leaf=lambda value: value is None,
        )
        actor_layers, vl_layers, _ = cfg.gnn_layers
        algo = GPUDriveInforMARL(
            actor_graph,
            critic_graph,
            actor_gnn_layers=actor_layers,
            Vl_gnn_layers=vl_layers,
            seed=0,
            lr_Vl=3e-4,
            coef_ent=1e-4,
            minibatch_worlds=1,
            epoch_ppo=1,
            n_agents=64,
            base_node_dim=18,
            road_slots=cfg.num_road_segments,
            road_feature_dim=cfg.road_feature_dim,
        )
        checkpoint = Path(args.checkpoint).resolve()
        algo.load(str(checkpoint.parent.parent), checkpoint.parent.name)
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        video_path = output_dir / f"{Path(args.scene_file).stem}.gif"
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
        metrics_path = output_dir / f"{Path(args.scene_file).stem}.json"
        metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
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
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
