#!/usr/bin/env bash

# 从头开始训练。
python train.py --env LidarSpread --algo dgppo -n 3 --obs 3

# 从之前训练的最新检查点恢复训练。
python train.py --env LidarSpread --algo dgppo -n 3 --obs 3 \
  --resume-dir "xxxx/models/latest" \
#   --steps 200000

# 小显存 GPU 可使用以下参数：
#   --n-env-train 16 --batch-size 2048

# 两阶段 Deep-QP 训练：先预训练 Graph-HJ，再训练带约束的 InforMARL。
# python train.py --env LidarSpread --algo deepqp -n 3 --obs 3

# MPE 的 DGPPO 训练使用 100,000 次 RL 迭代。
# python train.py --env MPETarget --algo dgppo -n 3 --obs 3 --steps 100000
# python train.py --env MPESpread --algo dgppo -n 3 --obs 3 --steps 100000
# python train.py --env MPEFormation --algo dgppo -n 3 --obs 3 --steps 100000
# python train.py --env MPELine --algo dgppo -n 3 --obs 3 --steps 100000
# python train.py --env MPECorridor --algo dgppo -n 3 --obs 2 --steps 100000
# python train.py --env MPEConnectSpread --algo dgppo -n 3 --obs 1 --steps 100000

# MPE 的 Deep-QP 训练使用 50,000 次 Graph-HJ 预训练更新和 100,000 次 RL 迭代。
# MPEConnectSpread 目前不支持 Deep-QP。
# python train.py --env MPETarget --algo deepqp -n 3 --obs 3 --deep-qp-pretrain-steps 50000 --steps 100000
# python train.py --env MPESpread --algo deepqp -n 3 --obs 3 --deep-qp-pretrain-steps 50000 --steps 100000
# python train.py --env MPEFormation --algo deepqp -n 3 --obs 3 --deep-qp-pretrain-steps 50000 --steps 100000
# python train.py --env MPELine --algo deepqp -n 3 --obs 3 --deep-qp-pretrain-steps 50000 --steps 100000
# python train.py --env MPECorridor --algo deepqp -n 3 --obs 2 --deep-qp-pretrain-steps 50000 --steps 100000

# 评估 MPE/Lidar 的 safety rate 和 reach rate。脚本会自动读取指定环境与算法
# 目录下的全部 seed；每个 seed 默认在相同的 32 个初始条件上执行确定性策略。
# reach rate 使用环境的 dist2goal 阈值，并统计整条轨迹中曾到达目标的任务比例。
# 普通训练目录和 Deep-QP 两阶段训练目录都会自动解析。
# python scripts/safety_rate_eval.py \
#   --env-log-dir "logs/MPESpread" \
#   --algo dgppo
# Lidar 环境的用法相同，例如：
# python scripts/safety_rate_eval.py \
#   --env-log-dir "logs/LidarSpread" \
#   --algo deepqp

# 仅训练 GCBF 证书，不训练 actor/policy 网络。
python train_gcbf.py --env LidarSpread -n 3 --obs 3

# 仅训练 GCBF 时的显存配置预设；经验回放缓冲区保存在主机内存中。
# GTX 1650 Ti / 4 GB（建议从此配置开始）：
# python train_gcbf.py --env LidarSpread -n 3 --obs 3 \
#   --batch-size 16 --n-env-train 2 --n-env-test 2 --rollout-steps 64
# GTX 1650 Ti / 4 GB（保守配置稳定后可尝试此配置）：
# python train_gcbf.py --env LidarSpread -n 3 --obs 3 \
#   --batch-size 32 --n-env-train 4 --n-env-test 4 --rollout-steps 64
# 6–8 GB 显存：
# python train_gcbf.py --env LidarSpread -n 3 --obs 3 \
#   --batch-size 64 --n-env-train 8 --n-env-test 8 --rollout-steps 64
# 原始 GCBF+ 参考/默认配置（通常超出 4 GB GPU 的显存容量）：
# python train_gcbf.py --env LidarSpread -n 3 --obs 3 \
#   --batch-size 256 --n-env-train 16 --n-env-test 32 --rollout-steps 128

# 可视化训练后的 GCBF 等值线；请将 seed0_xxx 替换为实际运行名称。
python gcbfplus_visualize.py \
  --gcbfplus-dir "logs/LidarSpread/gcbf/seed0_xxx"

# 将 Graph-HJ 网络作为 GCBF，为指定自智能体生成可视化。
python ./deep_qp_visualize.py --policy-dir logs/LidarSpread/dgppo/seed0_831102005_KHPJ --deep-qp-checkpoint logs/LidarSpread/deepqp/seed0_0831113946_FRLF/deep-qp -n 5 --obs 4

python ./dgcbf_visualize.py --dgppo-dir logs/LidarSpread/dgppo/seed0_831102005_KHPJ \
  --cost-channel worst \
  --ego-agents all \
  --output-dir figures/dgbcf-contour


python train.py \
  --env LidarSpread \
  -n 3 \
  --algo adversarial_dgppo \
  --obs 3
