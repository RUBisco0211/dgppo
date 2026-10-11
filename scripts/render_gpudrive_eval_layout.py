#!/usr/bin/env python3
"""Render the fixed world-zero GPUDrive evaluation layout without training."""

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import torch

from dgppo.env.gpudrive import GPUDriveDGPPOAdapter, GPUDriveDGPPOConfig
from train_gpudrive import make_env


def main(args):
    env = make_env(args, args.seed, args.num_worlds)
    try:
        if args.raw:
            fig = env.vis.plot_simulator_state([0])[0]
            output = Path(args.output).resolve()
            output.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output, dpi=180, bbox_inches="tight")
            plt.close(fig)
            print(f"saved raw layout {output}")
            return
        adapter = GPUDriveDGPPOAdapter(env, GPUDriveDGPPOConfig())
        prepared = adapter.prepare_scenes(args.seed, fixed_eval=True)
        world = 0
        selection = prepared[world]
        fig = env.vis.plot_simulator_state([world])[0]
        ax = fig.axes[0]

        state = env.sim.absolute_self_observation_tensor().to_torch()[world]
        local = env.sim.self_observation_tensor().to_torch()[world]
        ids = local[:, 7].round().to(torch.int64)
        valid = local[:, 1] > 0
        colors = ("#1565c0", "#7b1fa2", "#00897b", "#c62828", "#6d4c41")
        for order, (agent_id, color) in enumerate(zip(selection.controlled_ids, colors), start=1):
            row = torch.where(valid & (ids == agent_id))[0]
            if row.numel() != 1:
                raise RuntimeError(f"cannot locate controlled vehicle {agent_id}")
            values = state[row[0]].detach().cpu().numpy()
            start, goal = values[:2], values[8:10]
            ax.plot([start[0], goal[0]], [start[1], goal[1]], "--", color=color,
                    linewidth=1.8, alpha=0.8, zorder=20)
            ax.scatter(*start, marker="s", s=85, facecolors="none", edgecolors=color,
                       linewidths=2.0, zorder=21)
            ax.scatter(*goal, marker="*", s=170, color=color, zorder=21)
            ax.annotate(f"C{order} start", start, xytext=(5, 5), textcoords="offset points",
                        color=color, fontsize=9, zorder=22)
            ax.annotate(f"C{order} goal", goal, xytext=(5, 5), textcoords="offset points",
                        color=color, fontsize=9, zorder=22)

        for order, agent_id in enumerate(selection.background_ids, start=1):
            row = torch.where(valid & (ids == agent_id))[0]
            if row.numel() != 1:
                raise RuntimeError(f"cannot locate background vehicle {agent_id}")
            position = state[row[0], :2].detach().cpu().numpy()
            ax.scatter(*position, marker="o", s=58, facecolors="#ef6c00",
                       edgecolors="white", linewidths=0.8, zorder=21)
            ax.annotate(f"B{order}", position, xytext=(4, 4), textcoords="offset points",
                        color="#bf360c", fontsize=8, zorder=22)

        center = np.asarray(selection.layout_center)
        ax.scatter(*center, marker="x", s=110, color="#d32f2f", linewidths=2.2,
                   label="controlled layout center", zorder=23)
        ax.set_title("Fixed straight-road evaluation layout: squares=start, stars=goal")
        ax.legend(loc="best")
        output = Path(args.output).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=180, bbox_inches="tight")
        plt.close(fig)
        print(f"saved {output}")
        print(f"controlled IDs: {selection.controlled_ids}")
        print(f"background IDs: {selection.background_ids}")
        print(f"layout center: {selection.layout_center}")
        kept_rows = []
        for agent_id in selection.kept_ids:
            kept_rows.append(torch.where(valid & (ids == agent_id))[0][0])
        kept_position = state[torch.stack(kept_rows), :2]
        print(f"minimum initial center distance: {float(torch.pdist(kept_position).min()):.2f} m")
        unique_controlled = len({item.controlled_ids for item in prepared})
        print(f"unique controlled groups across {len(prepared)} worlds: {unique_controlled}")
        controlled_rows = torch.stack(kept_rows[: adapter.config.num_controlled])
        controlled_state = state[controlled_rows]
        goal_distance = torch.linalg.norm(
            controlled_state[:, 8:10] - controlled_state[:, :2], dim=-1
        )
        print(f"eval goal distances: {[round(float(value), 2) for value in goal_distance]}")
        costs, _ = adapter.continuous_cost_batch()
        eval_costs = np.asarray(costs)[0]
        print(f"eval initial costs [vehicle, road]: {np.round(eval_costs, 3).tolist()}")
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="gpudrive/data/processed/examples")
    parser.add_argument("--scene-file", default="tfrecord-00000-of-01000_402.json")
    parser.add_argument("--dataset-size", type=int, default=1)
    parser.add_argument("--num-worlds", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steer-rad", type=float, default=0.3)
    parser.add_argument("--output", default="artifacts/gpudrive_eval_layout.png")
    parser.add_argument("--raw", action="store_true")
    main(parser.parse_args())
