"""Launch BC actor pretraining or PPO continuation as a matched tmux pair."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("stage", choices=("data", "bc", "ppo"))
parser.add_argument("--stamp")
parser.add_argument("--data-device", choices=("cpu", "cuda"), default="cuda")
parser.add_argument("--algorithm", choices=("ippo", "attention_ippo"))
args = parser.parse_args()
stamp = args.stamp or datetime.now().strftime("%m%d%H%M%S")
root = ROOT / "logs/gpudrive"
dataset = ROOT / "gpudrive/data/derived/bc_seed0_tracking"

if args.stage == "bc":
    for split in ("training", "validation"):
        if not (dataset / split / "audit.json").is_file():
            raise FileNotFoundError(f"Missing completed BC dataset: {split}")
    subprocess.run(["/home/linminlei/.conda/envs/gpudrive/bin/python",
                    str(ROOT / "scripts/check_gpudrive_bc_audit.py")], check=True)
if args.stage == "ppo":
    for algorithm in ("ippo", "attention_ippo"):
        checkpoint = root / "bc_pretrain" / algorithm / f"bc_{algorithm}_seed000_{stamp}" / "best_actor.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
    subprocess.run(["/home/linminlei/.conda/envs/gpudrive/bin/python",
                    str(ROOT / "scripts/prepare_gpudrive_bc_ppo_configs.py"),
                    "--stamp", stamp], check=True)

records = []
names = (("training", "validation") if args.stage == "data" else
         (args.algorithm,) if args.algorithm else ("ippo", "attention_ippo"))
for index, item in enumerate(names):
    gpu = 2 if args.stage != "data" else (1 if index == 0 else 3)
    if args.stage == "data":
        name = f"bc_data_{item}_{stamp}"
        directory = root / "bc_data" / name
        command = ["env", f"GPUDRIVE_BC_DATA_DEVICE={args.data_device}",
                   "bash", str(ROOT / "scripts/run_gpudrive_bc_data.sh"), item]
    elif args.stage == "bc":
        name = f"bc_{item}_seed000_{stamp}"
        directory = root / "bc_pretrain" / item / name
        command = ["bash", str(ROOT / "scripts/run_gpudrive_bc_actor.sh"),
                   item, str(directory), str(dataset)]
    else:
        name = f"{item}_bc_seed000_{stamp}"
        directory = root / "bc_warmstart" / item / name
        command = ["bash", str(ROOT / "scripts/run_gpudrive_bc_ppo.sh"), item,
                   "--run-name", name]
    session = f"drive_bc_{args.stage}_{item}"
    if subprocess.run(["tmux", "has-session", "-t", session], capture_output=True).returncode == 0:
        raise RuntimeError(f"tmux session already exists: {session}")
    directory.mkdir(parents=True, exist_ok=False)
    console = shlex.quote(str(directory / "console.log"))
    shell = f"set -o pipefail; {shlex.join(command)} 2>&1 | tee {console}; status=$?; echo Training_exit_code=$status; exec bash"
    subprocess.run(["tmux", "new-session", "-d", "-s", session, "-c", str(ROOT),
                    "bash", "-lc", shell], check=True)
    records.append({"stage": args.stage, "item": item, "run_name": name,
                    "tmux": session, "gpu": None if args.stage == "data" and args.data_device == "cpu" else gpu,
                    "directory": str(directory),
                    "wandb_url": f"https://wandb.ai/1205456072-southeast-university/gpudrive/runs/{name}"
                    if args.stage != "data" else None})
suffix = f"_{args.algorithm}" if args.algorithm else ""
index_path = root / "bc_warmstart" / f"{args.stage}_{stamp}{suffix}.json"
index_path.parent.mkdir(parents=True, exist_ok=True)
index_path.write_text(json.dumps(records, indent=2))
print(json.dumps(records, indent=2))
