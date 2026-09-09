#!/usr/bin/env bash
# 首次 S2 正式训练（1 epoch，10K 子集，从 S1 续跑）
source activate dasr
export CUDA_VISIBLE_DEVICES=5
export HF_HUB_DISABLE_XET=1
cd /home/zzj2002/SpatialSpeechLLM/DirectedASR
python -m dasr.train \
  --stage encoder_lora \
  --config conf/dasr_defaults.yaml \
  --qa-root data/qa --audio-root data \
  --output-dir runs/dasr_stage2 \
  --max-samples 10000 --epochs 1 \
  --resume runs/dasr_stage1/epoch_0_trainable.pt \
  --batch-size 2 --accum 4 --lr 3e-5
echo "S2 done exit=$?"
