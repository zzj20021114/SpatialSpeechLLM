"""dasr.train — 训练入口（独立原创实现）。

阶段：
  seld_pretrain  ：空间编码器 + ACCDOA 头，在 FOA SELD 数据上预训练（可选）
  projector_only ：仅训练投影器（编码器/解码器冻结）—— 对齐编码 token 与 LLM 语义
  encoder_lora   ：编码器 + 投影器 + 解码器 LoRA 全开（核心：ASR + 左右定位联合）
  full           ：同 encoder_lora（预留，可扩展全量微调）

用法示例：
  python -m dasr.train --stage projector_only --config conf/dasr_defaults.yaml \
      --qa-root data/qa --audio-root data --output-dir runs/dasr_stage1
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .model import (
    AudioFrontend,
    DasrConfig,
    DasrEncoderDecoder,
    PixelShuffleProjector,
    SeldPretrainModel,
    SpatialEncoder,
)
from .data.dataset import DasrCollator, QaDataset
from .data.seld_dataset import SeldCollator, SeldDataset


def _trainable_state_dict(model: nn.Module):
    """只保存 requires_grad 的参数（LoRA/投影器等），避免写全量权重。"""
    return {k: v.detach().cpu() for k, v in model.named_parameters() if v.requires_grad}


def build_optimizer(model: nn.Module, cfg: DasrConfig) -> torch.optim.Optimizer:
    t = cfg.train
    groups: List[Dict] = []
    proj, enc, lora, other = [], [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("projector"):
            proj.append(p)
        elif name.startswith("frontend") or name.startswith("spatial_encoder"):
            enc.append(p)
        elif "lora" in name.lower():
            lora.append(p)
        else:
            other.append(p)
    if proj:
        groups.append({"params": proj, "lr": t.projector_lr})
    if enc:
        groups.append({"params": enc, "lr": t.spatial_lr})
    if lora:
        groups.append({"params": lora, "lr": t.lora_lr})
    if other:
        groups.append({"params": other, "lr": t.learning_rate})
    return torch.optim.AdamW(groups, weight_decay=t.weight_decay, foreach=False)


def train_step(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    accum: int,
    stage: str,
) -> torch.Tensor:
    if stage == "seld_pretrain":
        loss = model(
            batch["spatial_audio"].to(device),
            batch["spatial_audio_lengths"].to(device),
            batch["activity"].to(device),
            batch["coords"].to(device),
            label_rate=float(batch["label_rate"]),
        )["loss"]
    else:
        out = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            labels=batch["labels"].to(device),
            spatial_audio=batch["spatial_audio"].to(device),
            spatial_audio_lengths=batch["spatial_audio_lengths"].to(device),
        )
        loss = out.loss
    return loss / accum


def build_model_for_stage(stage: str, cfg: DasrConfig) -> nn.Module:
    if stage == "seld_pretrain":
        frontend = AudioFrontend(cfg.audio_frontend)
        encoder = SpatialEncoder(cfg.spatial_encoder)
        return SeldPretrainModel(encoder, cfg.seld_head, frontend=frontend)

    model = DasrEncoderDecoder(cfg)
    if stage in ("encoder_lora", "full"):
        model.apply_lora()
        model.enable_encoder_grad(True)
    else:  # projector_only：仅投影器可训，编码器/解码器全冻结
        model.enable_encoder_grad(False)
        for p in model.decoder.parameters():
            p.requires_grad = False
    return model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="dasr 训练")
    p.add_argument("--stage", choices=["seld_pretrain", "projector_only",
                                       "encoder_lora", "full"], required=True)
    p.add_argument("--config", default="conf/dasr_defaults.yaml")
    p.add_argument("--qa-root", default="data/qa")
    p.add_argument("--audio-root", default="data")
    p.add_argument("--split", default="train")
    p.add_argument("--seld-manifest", default=None, help="S0 阶段的 SELD manifest")
    p.add_argument("--output-dir", default="runs/dasr")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--accum", type=int, default=None)
    p.add_argument("--resume", default=None)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    cfg = DasrConfig.from_yaml(args.config) if os.path.exists(args.config) else DasrConfig()
    t = cfg.train
    if args.epochs:
        t.epochs = args.epochs
    if args.batch_size:
        t.batch_size = args.batch_size
    if args.lr:
        t.learning_rate = args.lr
    if args.accum:
        t.grad_accum_steps = args.accum
    t.output_dir = args.output_dir
    cfg.data.qa_roots = (args.qa_root,)
    cfg.data.audio_root = args.audio_root
    cfg.data.split = args.split

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model_for_stage(args.stage, cfg)

    quantized = getattr(cfg.decoder, "quantization", "none") != "none" or bool(getattr(cfg.decoder, "device_map", None))
    if not quantized:
        model.to(device)
    else:
        print("[train] 量化/device_map 加载，模型已由 accelerate 放置，跳过 .to()")

    if args.stage == "seld_pretrain":
        dataset = SeldDataset(args.seld_manifest, num_classes=cfg.seld_head.num_classes)
        collate_fn = SeldCollator()
    else:
        dataset = QaDataset(
            os.path.join(args.qa_root, f"{args.split}.jsonl"),
            audio_roots=[args.audio_root],
            max_samples=args.max_samples,
            verify_audio=True,
        )
        collate_fn = DasrCollator(
            tokenizer=model.tokenizer,
            speech_token=cfg.decoder.speech_token,
            eos_token=model.tokenizer.eos_token,
            sample_rate=cfg.audio_frontend.sample_rate,
            hop_length=cfg.audio_frontend.hop_length,
            patch_time=cfg.spatial_encoder.patch_time,
            stride_time=cfg.spatial_encoder.stride_time,
            shuffle_factor=model.projector.shuffle_factor,
            max_audio_seconds=cfg.data.max_audio_seconds,
        )

    loader = DataLoader(
        dataset, batch_size=t.batch_size, shuffle=True,
        num_workers=min(4, cfg.data.num_workers), collate_fn=collate_fn,
    )
    optimizer = build_optimizer(model, cfg)
    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location="cpu", weights_only=False), strict=False)

    os.makedirs(t.output_dir, exist_ok=True)
    cfg.save(os.path.join(t.output_dir, "train_args.json"))

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[{args.stage}] trainable={n_trainable / 1e6:.1f}M batches={len(loader)}")

    global_step = 0
    for epoch in range(t.epochs):
        model.train()
        optimizer.zero_grad()
        t0 = time.time()
        for step, batch in enumerate(loader):
            loss = train_step(model, batch, device, t.grad_accum_steps, args.stage)
            loss.backward()
            if (step + 1) % t.grad_accum_steps == 0:
                nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], t.max_grad_norm
                )
                optimizer.step()
                optimizer.zero_grad()
            global_step += 1
            if global_step % t.log_interval == 0:
                print(f"[epoch {epoch} step {step}] loss={loss.item() * t.grad_accum_steps:.4f} "
                      f"lr={optimizer.param_groups[0]['lr']:.2e} time={time.time() - t0:.1f}s")
                t0 = time.time()
            if global_step % t.save_interval == 0:
                ckpt = os.path.join(t.output_dir, f"step_{global_step}_trainable.pt")
                torch.save(_trainable_state_dict(model), ckpt)
        ckpt = os.path.join(t.output_dir, f"epoch_{epoch}_trainable.pt")
        torch.save(_trainable_state_dict(model), ckpt)
        print(f"[epoch {epoch} done] saved {ckpt}")


if __name__ == "__main__":
    main()
