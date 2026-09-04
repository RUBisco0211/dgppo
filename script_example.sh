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

# Train only the GCBF certificate (no actor/policy network).
python train_gcbf.py --env LidarSpread -n 3 --obs 3

# GCBF-only GPU-memory presets. The replay buffer is kept in host RAM.
# GTX 1650 Ti / 4 GB (recommended starting point):
# python train_gcbf.py --env LidarSpread -n 3 --obs 3 \
#   --batch-size 16 --n-env-train 2 --n-env-test 2 --rollout-steps 64
# GTX 1650 Ti / 4 GB (try this after the conservative preset is stable):
# python train_gcbf.py --env LidarSpread -n 3 --obs 3 \
#   --batch-size 32 --n-env-train 4 --n-env-test 4 --rollout-steps 64
# 6-8 GB:
# python train_gcbf.py --env LidarSpread -n 3 --obs 3 \
#   --batch-size 64 --n-env-train 8 --n-env-test 8 --rollout-steps 64
# Original GCBF+ reference/default (usually too large for a 4 GB GPU):
# python train_gcbf.py --env LidarSpread -n 3 --obs 3 \
#   --batch-size 256 --n-env-train 16 --n-env-test 32 --rollout-steps 128

# Visualize the trained GCBF contour; replace seed0_xxx with the run name.
python gcbfplus_visualize.py \
  --gcbfplus-dir "logs/LidarSpread/gcbf/seed0_xxx"

# Graph HJ network as GCBF visualization for an ego agent 
python ./deep_qp_visualize.py --policy-dir logs/LidarSpread/dgppo/seed0_831102005_KHPJ --deep-qp-checkpoint logs/LidarSpread/deepqp/seed0_0831113946_FRLF/deep-qp -n 5 --obs 4

python ./dgcbf_visualize.py --dgppo-dir logs/LidarSpread/dgppo/seed0_831102005_KHPJ \
  --cost-channel worst \
  --ego-agents all \
  --output-dir figures/dgbcf-contour
