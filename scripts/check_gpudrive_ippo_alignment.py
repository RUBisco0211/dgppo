"""Dump native controlled IDs and action/reward settings for comparison."""

import argparse
import dataclasses
import json
from types import SimpleNamespace

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("backend", choices=("gnn", "ippo"))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    env = None
    try:
        if args.backend == "gnn":
            from train_gpudrive import make_env
            from dgppo.env.gpudrive import GPUDriveDGPPOAdapter, GPUDriveDGPPOConfig
            ns = SimpleNamespace(
                algorithm="gnn_ippo", data_dir="gpudrive/data/processed/examples",
                dataset_size=4, scene_file="tfrecord", device="cuda", max_steer_rad=0.3,
            )
            env = make_env(ns, 0, 16)
            cfg = GPUDriveDGPPOConfig(
                num_controlled=64, num_background=0, control_all_vehicles=True,
                use_native_controlled_mask=True, keep_all_scene_vehicles=True,
                num_road_segments=200, road_feature_dim=14, rollout_horizon=91,
            )
            adapter = GPUDriveDGPPOAdapter(env, cfg)
            prepared = adapter.prepare_scenes(0)
            ids = [list(world.controlled_ids) for world in prepared]
        else:
            import yaml
            from box import Box
            from gpudrive.env.config import EnvConfig
            from gpudrive.env.wrappers.sb3_wrapper import SB3MultiAgentEnv
            with open("gpudrive/baselines/ppo/config/ppo_aligned_gnn_comparison.yaml") as f:
                cfg = Box(yaml.safe_load(f))
            cfg.data_dir = "gpudrive/data/processed/examples"
            env_cfg = dataclasses.replace(
                EnvConfig(), num_worlds=16, reward_type=cfg.reward_type,
                episode_len=91, remove_non_vehicles=True, collision_behavior="ignore",
                steer_actions=torch.linspace(-0.3, 0.3, 13),
            )
            env = SB3MultiAgentEnv(env_cfg, cfg, env_cfg.max_num_agents_in_scene, "cuda")
            local = env._env.sim.self_observation_tensor().to_torch().clone()
            ids = [
                [int(v) for v in local[w, env.controlled_agent_mask[w], 7].round().tolist()]
                for w in range(16)
            ]
        payload = {
            "backend": args.backend,
            "controlled_ids": ids,
            "counts": [len(values) for values in ids],
            "steer_actions": [float(v) for v in env._env.config.steer_actions]
                if args.backend == "ippo" else [float(v) for v in env.config.steer_actions],
        }
        with open(args.output, "w") as stream:
            json.dump(payload, stream, indent=2)
        print(json.dumps(payload))
    finally:
        if env is not None:
            env.close() if args.backend == "ippo" else env.close()


if __name__ == "__main__":
    main()
