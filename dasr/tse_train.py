"""Train the target-conditioned TSE model without loading Qwen.

Stages:
  cue_distill: train enrollment/target speaker representations against an
               optional frozen teacher embedding.
  joint:       train the shared target representation with TSE, speaker,
               VAD, DOA and side objectives.

Example smoke run:
    conda run -n dasr python -m dasr.tse_train \
        --manifest data/tse_public/generated/train/manifest.jsonl \
        --output-dir runs/tse_smoke --stage joint --max-samples 4 \
        --epochs 1 --batch-size 2 --device cuda
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, Iterable

import torch
from torch.utils.data import DataLoader

from .data.tse_dataset import TseCollator, TseDataset
from .model.config import DasrConfig
from .model.joint_model import JointTargetSpeechModel


def _move_batch(batch: Dict, device: torch.device) -> Dict:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _set_trainable(model: JointTargetSpeechModel, stage: str, unfreeze_spatial: bool) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False

    if stage == "cue_distill":
        modules = [
            model.target_encoder.enrollment_encoder,
            model.target_encoder.target_fusion,
            model.tse_heads.speaker,
            model.teacher_projection,
        ]
        for module in modules:
            for parameter in module.parameters():
                parameter.requires_grad = True
    elif stage == "joint":
        for parameter in model.parameters():
            parameter.requires_grad = True
        model.freeze_decoder()
        if not unfreeze_spatial:
            model.enable_spatial_grad(False)
    else:
        raise ValueError(f"unknown TSE stage: {stage}")


def _loss_weights(stage: str, teacher_weight: float) -> Dict[str, float]:
    if stage == "cue_distill":
        return {
            "enrollment_distill": teacher_weight,
            "target_distill": teacher_weight,
        }
    return {
        "si_sdr": 1.0,
        "speaker": 1.0,
        "vad": 0.5,
        "doa": 0.5,
        "side": 0.5,
        "enrollment_distill": teacher_weight,
        "target_distill": teacher_weight,
    }


def _format_losses(losses: Dict[str, torch.Tensor]) -> str:
    return " ".join(
        f"{name}={float(value.detach().cpu()):.4f}"
        for name, value in losses.items()
        if name != "total"
    ) + f" total={float(losses['total'].detach().cpu()):.4f}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练 target-conditioned TSE，不加载 Qwen")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--config", default="conf/dasr_defaults.yaml")
    parser.add_argument("--output-dir", default="runs/tse")
    parser.add_argument("--stage", choices=["cue_distill", "joint"], default="joint")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--teacher-weight", type=float, default=1.0)
    parser.add_argument("--max-audio-seconds", type=float, default=20.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--unfreeze-spatial", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, "train.log")

    def log(message: str) -> None:
        print(message, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(message + "\n")

    cfg = DasrConfig.from_yaml(args.config)
    dataset = TseDataset(
        args.manifest,
        max_samples=args.max_samples,
        verify_audio=True,
    )
    collator = TseCollator(
        sample_rate=cfg.audio_frontend.sample_rate,
        max_audio_seconds=args.max_audio_seconds,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collator,
    )
    model = JointTargetSpeechModel(cfg).to(device)
    _set_trainable(model, args.stage, args.unfreeze_spatial)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("没有可训练参数")
    if args.stage == "cue_distill" and not any(
        "teacher_embedding" in record for record in dataset.records
    ):
        raise ValueError("cue_distill 阶段要求 manifest 包含 teacher_embedding")
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=0.01)
    weights = _loss_weights(args.stage, args.teacher_weight)
    log(json.dumps({
        "stage": args.stage,
        "device": str(device),
        "samples": len(dataset),
        "batches": len(loader),
        "trainable_params": sum(p.numel() for p in trainable),
        "weights": weights,
    }, ensure_ascii=False))

    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        for batch in loader:
            batch = _move_batch(batch, device)
            outputs = model(
                batch["mixture_audio"],
                batch["enrollment_audio"],
                batch["mixture_lengths"],
                batch["enrollment_lengths"],
            )
            losses = model.compute_joint_losses(outputs, batch, weights=weights)
            optimizer.zero_grad(set_to_none=True)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(trainable, 5.0)
            optimizer.step()
            global_step += 1
            if global_step % 10 == 0 or global_step == 1:
                log(f"epoch={epoch} step={global_step} {_format_losses(losses)}")

    checkpoint = os.path.join(args.output_dir, f"{args.stage}_final_trainable.pt")
    torch.save({
        "model": {name: value.detach().cpu() for name, value in model.state_dict().items()
                  if dict(model.named_parameters()).get(name, torch.empty(0)).requires_grad},
        "config": cfg.to_dict(),
        "stage": args.stage,
        "global_step": global_step,
    }, checkpoint)
    log(f"完成：step={global_step} checkpoint={checkpoint} time={time.strftime('%H:%M:%S')}")


if __name__ == "__main__":
    main()
