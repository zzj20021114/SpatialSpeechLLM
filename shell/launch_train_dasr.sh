#!/usr/bin/env bash
# dasr 一键分阶段训练（自建编码器-解码器）
#   S1 对齐（projector_only）→ S2 联合（encoder_lora，核心）→ S3 继续联合/精调
# 可选 S0：空间编码器 SELD 预训练。
#
# 用法: bash shell/launch_train_dasr.sh [S1|S2|S3|S0] [--data-root DIR] [--gpu ID] [--smoke N]
set -euo pipefail

STAGE="${1:-S2}"
DATA_ROOT="${DATA_ROOT:-.}"
GPU="${GPU:-5}"
SMOKE="${SMOKE:-0}"
export CUDA_VISIBLE_DEVICES="$GPU"
export HF_HUB_DISABLE_XET=1

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
CONFIG="conf/dasr_defaults.yaml"
QA_ROOT="$DATA_ROOT/data/qa"
AUDIO_ROOT="$DATA_ROOT/data"

run() {
  local stage="$1" out="$2"; shift 2
  local extra=()
  if [ "$SMOKE" -gt 0 ]; then extra+=(--max-samples "$SMOKE"); fi
  echo ">>> [$stage] output=$out"
  python -m dasr.train --stage "$stage" --config "$CONFIG" \
    --qa-root "$QA_ROOT" --audio-root "$AUDIO_ROOT" \
    --output-dir "$out" "${extra[@]}" "$@"
}

case "$STAGE" in
  S0)
    python -m dasr.train --stage seld_pretrain --config "$CONFIG" \
      --seld-manifest "$DATA_ROOT/data/seld/train.jsonl" \
      --output-dir "$DATA_ROOT/runs/dasr_stage0"
    ;;
  S1)
    run projector_only "$DATA_ROOT/runs/dasr_stage1" --epochs 1 --batch-size 2 --accum 4 --lr 1e-4
    ;;
  S2)
    run encoder_lora "$DATA_ROOT/runs/dasr_stage2" --epochs 3 --batch-size 2 --accum 4 --lr 3e-5
    ;;
  S3)
    run encoder_lora "$DATA_ROOT/runs/dasr_stage3" --epochs 2 --batch-size 2 --accum 4 --lr 1e-5
    ;;
  *)
    echo "未知 stage: $STAGE（S0|S1|S2|S3）"; exit 1;;
esac

echo "[launch] $STAGE 完成。推理:"
echo "  python -m dasr.generate --run-dir $DATA_ROOT/runs/dasr_stage${STAGE#S} --checkpoint <ckpt> --qa-root $QA_ROOT --audio-root $AUDIO_ROOT --split test"
