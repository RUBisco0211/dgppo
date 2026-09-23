env -u LD_LIBRARY_PATH \
CUDA_VISIBLE_DEVICES=0 \
JAX_PLATFORMS=cuda \
python train.py \
  --env MPELine \
  --algo dgppo \
  -n 3 \
  --obs 3 \
  --seed 0 \
  --steps 200000 \
  --n-env-train 128 \
  --batch-size 16384 \
  --n-env-test 32 \
  --eval-interval 1000 \
  --eval-epi 32 \
  --save-interval 1000 \
  --cbf-eps 0.01 \
  --no-video \
  --wandb-mode disabled \
  --name safety_paradox_test
