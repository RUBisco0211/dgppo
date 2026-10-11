"""One-pass CPU driving check of a BC actor before expensive PPO training."""
import argparse
import dataclasses
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "gpudrive"))

import torch
import yaml
from box import Box

from gpudrive.env.config import EnvConfig
from gpudrive.env.wrappers.sb3_wrapper import SB3MultiAgentEnv
from gpudrive.networks.neighbor_attention import NeighborAttentionNet
from gpudrive.networks.perm_eq_late_fusion import LateFusionNet, LateFusionPolicy
from gpudrive.networks.graph_transformer import GraphTransformerNet, GraphTransformerPolicy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--zero-policy", action="store_true")
    parser.add_argument("--manifest")
    parser.add_argument("--output")
    args = parser.parse_args()
    torch.set_num_threads(4)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    algorithm = checkpoint["metadata"]["algorithm"]
    if algorithm == "gnn_ippo":
        config = Box(yaml.safe_load((ROOT / "gpudrive/baselines/ppo/config/ppo_ippo_bc_scenes_10m.yaml").read_text()))
        graph_config = yaml.safe_load(Path(checkpoint["metadata"]["source_config"]).read_text())
        for key in ("gnn_layers", "gnn_heads", "gnn_msg_dim", "gnn_out_dim"):
            config[key] = graph_config[key]
    else:
        config = Box(yaml.safe_load((ROOT / f"gpudrive/baselines/ppo/config/ppo_{algorithm}_ablation.yaml").read_text()))
    config.scene_manifest = args.manifest or config.eval_scene_manifest
    manifest_paths = [line for line in Path(config.scene_manifest).read_text().splitlines() if line.strip()]
    config.num_worlds = len(manifest_paths)
    config.resample_dataset_size = len(manifest_paths)
    config.data_dir = str(Path(manifest_paths[0]).parent)
    config.sample_with_replacement = False
    config.shuffle_dataset = False
    config.eval_no_auto_reset = True
    config.device = "cpu"
    env_config = dataclasses.replace(EnvConfig(), reward_type=config.reward_type,
                                    episode_len=config.episode_len,
                                    remove_non_vehicles=config.remove_non_vehicles,
                                    polyline_reduction_threshold=config.polyline_reduction_threshold,
                                    obs_radius=config.observation_radius,
                                    collision_behavior=config.collision_behavior,
                                    steer_actions=torch.linspace(-config.max_steer_rad,
                                                                 config.max_steer_rad, 13))
    env = SB3MultiAgentEnv(config=env_config, exp_config=config,
                           max_cont_agents=config.max_cont_agents, device="cpu")
    if algorithm == "gnn_ippo":
        policy = GraphTransformerPolicy(env.observation_space, env.action_space,
                                        lambda _: 3e-4, env_config, config,
                                        mlp_class=GraphTransformerNet)
    else:
        cls = LateFusionNet if algorithm == "ippo" else NeighborAttentionNet
        policy = LateFusionPolicy(env.observation_space, env_config, config, env.action_space,
                                  lambda _: 3e-4, mlp_class=cls)
    missing, unexpected = policy.load_state_dict(checkpoint["state_dict"], strict=False)
    assert not unexpected
    assert all(key.startswith(("mlp_extractor.val_", "mlp_extractor.critic.", "value_net."))
               for key in missing), missing
    policy.set_training_mode(False)
    try:
        obs = env.reset()
        mask = env.controlled_agent_mask
        goals = torch.zeros_like(mask)
        collisions = torch.zeros_like(mask)
        offroads = torch.zeros_like(mask)
        action_hist = torch.zeros(env.action_space.n, dtype=torch.long)
        with torch.no_grad():
            for step in range(config.episode_len):
                actions = (torch.full((len(obs),), 45, dtype=torch.long) if args.zero_policy else
                           policy.get_distribution(obs).get_actions(deterministic=True))
                action_hist += torch.bincount(actions, minlength=env.action_space.n)
                obs, _, _, _ = env.step(actions.float())
                info = env._env.sim.info_tensor().to_torch()
                goals |= mask & (info[..., 3] > 0)
                collisions |= mask & (info[..., 1:3].sum(-1) > 0)
                offroads |= mask & (info[..., 0] > 0)
                if (step + 1) % 30 == 0:
                    print(f"eval step {step + 1}/{config.episode_len}", flush=True)
        result = {"algorithm": algorithm, "zero_policy": args.zero_policy,
                          "checkpoint": str(Path(args.checkpoint).resolve()),
                          "manifest": str(Path(config.scene_manifest).resolve()),
                          "worlds": config.num_worlds,
                          "controlled_agents": int(mask.sum()),
                          "goal_rate": float(goals[mask].float().mean()),
                          "collision_rate": float(collisions[mask].float().mean()),
                          "offroad_rate": float(offroads[mask].float().mean()),
                          "action_45_fraction": float(action_hist[45] / action_hist.sum()),
                          "unique_actions": int((action_hist > 0).sum())}
        if args.output:
            Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    main()
