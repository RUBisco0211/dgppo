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

# GCBF+ training (joint policy and CBF training).
python train.py --env LidarSpread --algo gcbf+ -n 3 --obs 3

# Visualize the trained GCBF+ CBF contour; replace seed0_xxx with the run name.
python gcbfplus_visualize.py \
  --gcbfplus-dir "logs/LidarSpread/gcbf+/seed0_xxx"

# Graph HJ network as GCBF visualization for an ego agent 
python ./deep_qp_visualize.py --policy-dir logs/LidarSpread/dgppo/seed0_831102005_KHPJ --deep-qp-checkpoint logs/LidarSpread/deepqp/seed0_0831113946_FRLF/deep-qp -n 5 --obs 4

python ./dgcbf_visualize.py --dgppo-dir logs/LidarSpread/dgppo/seed0_831102005_KHPJ \
  --cost-channel worst \
  --ego-agents all \
  --output-dir figures/dgbcf-contour
