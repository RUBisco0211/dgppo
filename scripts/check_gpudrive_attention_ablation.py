"""CPU structure/masking checks; optionally generate disabled-W&B smoke configs."""
from pathlib import Path
import sys

import torch
import yaml
from box import Box
from gymnasium import spaces
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "gpudrive"))
from gpudrive.env.config import EnvConfig
from gpudrive.networks.perm_eq_late_fusion import LateFusionNet
from gpudrive.networks.neighbor_attention import NeighborAttentionNet
from gpudrive.networks.perm_eq_late_fusion import LateFusionPolicy

torch.set_num_threads(4)

config_path = ROOT / "gpudrive/baselines/ppo/config/ppo_ippo_ablation.yaml"
config = Box(yaml.safe_load(config_path.read_text()))
env = EnvConfig()
space = spaces.Box(-np.inf, np.inf, shape=(1,))
torch.manual_seed(0)
baseline = LateFusionNet(space, env, config).eval()
torch.manual_seed(0)
attention = NeighborAttentionNet(space, env, config).eval()
for key, value in baseline.state_dict().items():
    assert torch.equal(value, attention.state_dict()[key]), key
width = baseline.ego_input_dim + baseline.ro_max * baseline.ro_input_dim + baseline.rg_max * baseline.rg_input_dim
features = torch.randn(3, width)
assert torch.equal(baseline.forward_critic(features), attention.forward_critic(features))
ego, neighbors, roads = attention._unpack_obs(features)
neighbors.zero_()
out = attention.forward_actor(features)
assert torch.isfinite(out).all()
out.square().mean().backward()
assert all(torch.isfinite(p.grad).all() for p in attention.parameters() if p.grad is not None)
neighbors[:, :2, 4:6] = 1
first = attention.forward_actor(features)
permuted = torch.cat((ego, neighbors.flip(1).flatten(1), roads.flatten(1)), 1)
torch.testing.assert_close(first, attention.forward_actor(permuted), atol=1e-5, rtol=1e-5)
policies = []
for cls in (LateFusionNet, NeighborAttentionNet):
    torch.manual_seed(0)
    policies.append(LateFusionPolicy(spaces.Box(-np.inf, np.inf, shape=(width,)), env, config,
                                    spaces.Discrete(39), lambda _: 3e-4, mlp_class=cls))
for key, value in policies[0].state_dict().items():
    assert torch.equal(value, policies[1].state_dict()[key]), key
assert torch.equal(policies[0].predict_values(features), policies[1].predict_values(features))
print("PASS: identical inherited weights/critic, finite empty-neighbor gradients, permutation invariance")
for algorithm in ("ippo", "attention_ippo"):
    path = ROOT / f"gpudrive/baselines/ppo/config/ppo_{algorithm}_ablation.yaml"
    config = yaml.safe_load(path.read_text())
    config.update(num_worlds=32, total_timesteps=500, eval_policy_freq=1, save_policy_freq=1,
                  wandb_mode="disabled", output_root=str(ROOT / "artifacts/gpudrive_attention_smoke"),
                  run_name=algorithm, eval_num_worlds=5)
    path = ROOT / f"artifacts/gpudrive_attention_smoke/{algorithm}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config))
