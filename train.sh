#!/usr/bin/env bash

# Fresh training.
python train.py --env LidarSpread --algo dgppo -n 3 --obs 3

# Resume training from the latest checkpoint of a previous run.
python train.py --env LidarSpread --algo dgppo -n 3 --obs 3 \
  --resume-dir "xxxx/models/latest" \
#   --steps 200000

# For small GPUs:
#   --n-env-train 16 --batch-size 2048

# Two-stage Deep-QP training: Graph-HJ pretraining, then constrained InforMARL.
# python train.py --env LidarSpread --algo deepqp -n 3 --obs 3
