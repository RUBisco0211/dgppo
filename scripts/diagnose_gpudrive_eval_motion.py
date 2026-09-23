#!/usr/bin/env python3
"""Inspect deterministic controlled-car actions and motion from a checkpoint."""
import argparse
from pathlib import Path
import sys
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import jax
import jax.numpy as jnp
import numpy as np

from dgppo.algo.gpudrive_informarl import GPUDriveInforMARL
from dgppo.env.gpudrive import GPUDriveDGPPOAdapter, GPUDriveDGPPOConfig
from train_gpudrive import make_env


def main(args):
    env_args = SimpleNamespace(
        data_dir=args.data_dir, dataset_size=1, scene_file=args.scene_file,
        max_steer_rad=0.3, device="cuda",
    )
    env = make_env(env_args, 0, 8)
    try:
        cfg = GPUDriveDGPPOConfig(rollout_horizon=91)
        adapter = GPUDriveDGPPOAdapter(env, cfg)
        adapter.prepare_scenes(10000, fixed_eval=True)
        actor_batch = adapter.local_graph_batch().graph
        critic_batch = adapter.scene_graph_batch()
        actor_graph = jax.tree_util.tree_map(
            lambda x: None if x is None else x[0], actor_batch,
            is_leaf=lambda x: x is None)
        critic_graph = jax.tree_util.tree_map(
            lambda x: None if x is None else x[0], critic_batch,
            is_leaf=lambda x: x is None)
        algo = GPUDriveInforMARL(
            actor_graph, critic_graph, n_agents=5, lr_Vl=3e-4,
            coef_ent=1e-4, minibatch_worlds=4, epoch_ppo=1,
            base_node_dim=18, road_slots=32, road_feature_dim=8)
        directory, step = args.checkpoint.rsplit("/", 1)
        algo.load(directory, step)
        slots = adapter._controlled_slots()
        state0 = env.sim.absolute_self_observation_tensor().to_torch()[0, slots[0], :2].cpu().numpy()
        for step_idx in range(args.steps):
            graph = adapter.local_graph_batch().graph
            actions = np.asarray(algo.deterministic_actions(graph))
            local = env.sim.self_observation_tensor().to_torch()[0, slots[0], 0].cpu().numpy()
            if step_idx in (0, 1, args.steps - 1):
                print(f"step={step_idx} action={np.round(actions[0], 4).tolist()} speed={np.round(local, 4).tolist()}")
            adapter.step(jnp.asarray(actions))
        state1 = env.sim.absolute_self_observation_tensor().to_torch()[0, slots[0], :2].cpu().numpy()
        print("displacement_m=", np.round(np.linalg.norm(state1 - state0, axis=-1), 4).tolist())
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--data-dir", default="gpudrive/data/processed/examples")
    parser.add_argument("--scene-file", default="tfrecord-00000-of-01000_402.json")
    main(parser.parse_args())
