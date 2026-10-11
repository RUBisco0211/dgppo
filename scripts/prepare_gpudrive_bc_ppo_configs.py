"""Create a matched 10M-step PPO pair initialized from BC actor checkpoints."""
from pathlib import Path
import argparse
import yaml

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--stamp", required=True)
args = parser.parse_args()
for algorithm in ("ippo", "attention_ippo"):
    source = ROOT / f"gpudrive/baselines/ppo/config/ppo_{algorithm}_ablation.yaml"
    config = yaml.safe_load(source.read_text())
    config.update(total_timesteps=10_000_000, eval_policy_freq=100,
                  save_policy_freq=100, group_name="bc_warmstart_seed0",
                  eval_cuda_visible_devices=2,
                  eval_madrona_cache=str(ROOT / ".cache/gpudrive/gpu2_pci_ippo_aligned_16w/madrona_kernel_cache.bin"),
                  output_root=str(ROOT / "logs/gpudrive/bc_warmstart" / algorithm),
                  pretrained_actor_checkpoint=str(ROOT / "logs/gpudrive/bc_pretrain" /
                                                   algorithm / f"bc_{algorithm}_seed000_{args.stamp}" /
                                                   "best_actor.pt"),
                  bc_initial_eval=True,
                  tags=[algorithm, "bc_warmstart", "10m_agent_steps", "5_heldout_eval_scenes"])
    target = ROOT / f"gpudrive/baselines/ppo/config/ppo_{algorithm}_bc_10m.yaml"
    target.write_text(yaml.safe_dump(config, sort_keys=False))
    print(target)
