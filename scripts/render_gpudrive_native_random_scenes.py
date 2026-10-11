"""Render GPUDrive-native random scene initialization without custom placement."""

import argparse
import dataclasses
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from gpudrive.env.config import EnvConfig, RenderConfig
from gpudrive.env.dataset import SceneDataLoader
from gpudrive.env.env_torch import GPUDriveTorchEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="gpudrive/data/processed/examples")
    parser.add_argument("--num-scenes", type=int, default=4)
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--manifest")
    parser.add_argument("--max-cont-agents", type=int, default=5)
    parser.add_argument("--output-dir", default="artifacts/gpudrive_native_random_scenes")
    args = parser.parse_args()

    config = dataclasses.replace(
        EnvConfig(),
        init_mode="all_non_trivial",
        remove_non_vehicles=True,
        collision_behavior="ignore",
        dynamics_model="classic",
    )
    loader = SceneDataLoader(
        root=args.data_dir,
        batch_size=args.num_scenes,
        dataset_size=args.num_scenes,
        sample_with_replacement=False,
        shuffle=True,
        seed=args.seed,
        manifest_path=args.manifest,
    )
    env = GPUDriveTorchEnv(
        config=config,
        data_loader=loader,
        max_cont_agents=args.max_cont_agents,
        device="cuda",
        render_config=RenderConfig(),
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    try:
        local = env.sim.self_observation_tensor().to_torch()
        response = env.sim.response_type_tensor().to_torch().squeeze(-1)
        physical = (local[..., 7] >= 0) & (local[..., 1] > 0)
        figures = env.vis.plot_simulator_state(list(range(args.num_scenes)))
        for world, figure in enumerate(figures):
            controlled = env.cont_agent_mask[world] & physical[world]
            other = physical[world] & ~controlled
            moving_other = other & (response[world] == 0)
            static_other = other & (response[world] == 2)
            figure.axes[0].set_title(
                f"Native random scene {world} | controlled={int(controlled.sum())}, "
                f"other dynamic={int(moving_other.sum())}, static={int(static_other.sum())}"
            )
            figure.savefig(output / f"scene_{world}.png", dpi=130, bbox_inches="tight")
            plt.close(figure)
            print(
                world, int(controlled.sum()), int(moving_other.sum()),
                int(static_other.sum())
            )
    finally:
        env.close()


if __name__ == "__main__":
    main()
