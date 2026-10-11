"""Start the verified pair in persistent tmux sessions and write a run index."""
from datetime import datetime
import json
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[1]
root = ROOT / "logs/gpudrive/neighbor_attention_ablation"
stamp = datetime.now().strftime("%m%d%H%M%S")
specs = [("ippo", "drive_ippo_ablation", 1), ("attention_ippo", "drive_attention_ablation", 3)]
for _, session, _ in specs:
    if subprocess.run(["tmux", "has-session", "-t", session], capture_output=True).returncode == 0:
        raise RuntimeError(f"Session already exists; will not overwrite: {session}")
if (root / "runs.json").exists():
    raise RuntimeError("Run index already exists; inspect before launching another pair")
records = []
for algorithm, session, gpu in specs:
    name = f"{algorithm}_seed000_{stamp}"
    directory = root / algorithm / name
    directory.mkdir(parents=True, exist_ok=False)
    command = shlex.join(["bash", str(ROOT / "scripts/run_gpudrive_attention_ablation.sh"),
                          algorithm, "--run-name", name])
    console = shlex.quote(str(directory / "console.log"))
    # pipefail exposes trainer failures; keep an interactive shell when finished.
    shell = f"set -o pipefail; {command} 2>&1 | tee {console}; status=$?; echo Training_exit_code=$status; exec bash"
    subprocess.run(["tmux", "new-session", "-d", "-s", session, "-c", str(ROOT),
                    "bash", "-lc", shell], check=True)
    records.append({"algorithm": algorithm, "name": name, "tmux": session, "gpu": gpu,
                    "directory": str(directory), "total_agent_steps": 3_000_000,
                    "wandb_url": f"https://wandb.ai/1205456072-southeast-university/gpudrive/runs/{name}"})
(root / "runs.json").write_text(json.dumps(records, indent=2))
(root / "README.md").write_text("# 当前邻车注意力对照\n\n" + "\n".join(
    f"- {r['algorithm']}: `{r['name']}` | tmux `{r['tmux']}` | GPU {r['gpu']} | [W&B]({r['wandb_url']})"
    for r in records) + "\n\n各约300万有效agent-step，32 worlds，10训练/5验证场景。短测在artifacts，不混入本目录。\n")
print(json.dumps(records, indent=2))
