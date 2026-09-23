python \
  scripts/diagnose_dgppo_safety_paradox.py \
  logs/MPELine/dgppo/<run_die>/training_metrics.jsonl \
  --window 5 \
  --min-dangerous 20 \
  --output artifacts/safety_paradox/report.json \
  --plot artifacts/safety_paradox/report.png
