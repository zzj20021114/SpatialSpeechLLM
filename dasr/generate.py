"""dasr.generate — 定向 ASR 推理（独立原创实现）。

读取 run-dir 下的 train_args.json（含完整配置），重建模型并加载可训练权重，
对 qa/{split}.jsonl 逐条生成预测，写 predictions.jsonl。

用法：
    python -m dasr.generate \
        --run-dir runs/dasr_stage3 \
        --checkpoint runs/dasr_stage3/epoch_2_trainable.pt \
        --qa-root data/qa --audio-root data --split test
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch

from .data.dataset import QaDataset
from .data.foa import read_foa_wav
from .model import DasrConfig
from .train import build_model_for_stage


def main() -> None:
    parser = argparse.ArgumentParser(description="dasr 推理")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--qa-root", required=True)
    parser.add_argument("--audio-root", default="data")
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-jsonl", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--num-beams", type=int, default=1)
    args = parser.parse_args()

    train_args = json.load(open(os.path.join(args.run_dir, "train_args.json"), encoding="utf-8"))
    cfg = DasrConfig.from_dict(train_args)

    # 用保存的配置重建（含投影器/编码器维度），LoRA 结构需与保存时一致
    stage = train_args.get("train", {}).get("stage", "encoder_lora")
    model = build_model_for_stage(stage, cfg)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state, strict=False)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    quantized = getattr(cfg.decoder, "quantization", "none") != "none" or bool(getattr(cfg.decoder, "device_map", None))
    if not quantized:
        model.to(device)
    else:
        print("[generate] 量化/device_map 加载，跳过 .to()")

    dataset = QaDataset(
        os.path.join(args.qa_root, f"{args.split}.jsonl"),
        audio_roots=[args.audio_root],
        max_samples=args.max_samples,
    )

    out_path = args.output_jsonl or os.path.join(args.run_dir, "bench", args.split, "predictions.jsonl")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as of:
        for i, rec in enumerate(dataset):
            wav = read_foa_wav(rec["audio_path"], cfg.audio_frontend.sample_rate)
            if wav.shape[1] > int(cfg.data.max_audio_seconds * cfg.audio_frontend.sample_rate):
                wav = wav[:, : int(cfg.data.max_audio_seconds * cfg.audio_frontend.sample_rate)]
            audio_t = torch.from_numpy(wav).unsqueeze(0).to(device)
            lengths = torch.tensor([wav.shape[1]], dtype=torch.long, device=device)
            pred = model.generate(
                prompt=rec["prompt"],
                spatial_audio=audio_t,
                spatial_audio_lengths=lengths,
                max_new_tokens=args.max_new_tokens,
                num_beams=args.num_beams,
                do_sample=args.num_beams == 1,
            )
            of.write(json.dumps({
                "pair_id": rec.get("pair_id"),
                "audio_path": rec["audio_path"],
                "prompt": rec.get("prompt"),
                "gold_answer": rec.get("answer"),
                "canonical_answer": rec.get("canonical_answer"),
                "source_refs": rec.get("source_refs"),
                "prediction": pred,
            }, ensure_ascii=False) + "\n")
            if i % 20 == 0:
                print(f"[{time.strftime('%H:%M:%S')}] {i}/{len(dataset)} | pred: {pred[:60]}")

    print(f"完成：{len(dataset)} 条预测 -> {out_path}")


if __name__ == "__main__":
    main()
