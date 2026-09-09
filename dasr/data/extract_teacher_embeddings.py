"""Extract frozen speaker-teacher embeddings for TSE manifests.

The script creates a new JSONL manifest and never overwrites the input. The
teacher is loaded lazily so data-only tests and manifest inspection do not
require torch or transformers.

Example:
    python -m dasr.data.extract_teacher_embeddings \
        --manifest data/tse/train/manifest.jsonl \
        --output data/tse/train/manifest_teacher.jsonl \
        --model-id microsoft/wavlm-base-plus-sv
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List

import numpy as np

from .generate_tse_mixtures import read_mono


def l2_normalize(embedding: np.ndarray) -> np.ndarray:
    vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm < 1e-8:
        raise ValueError("teacher embedding 范数无效")
    return vector / norm


def load_records(path: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if not records:
        raise ValueError(f"manifest 为空: {path}")
    return records


class HuggingFaceTeacher:
    """Thin wrapper around a frozen Hugging Face speech encoder."""

    def __init__(self, model_id: str, device: str, sample_rate: int) -> None:
        try:
            import torch
            from transformers import AutoFeatureExtractor, AutoModel
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "提取 teacher embedding 需要安装 torch 和 transformers"
            ) from exc
        self.torch = torch
        self.sample_rate = sample_rate
        self.device = torch.device(device)
        self.extractor = AutoFeatureExtractor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(self.device)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        self.model_id = model_id
        self.embedding_dim = int(getattr(self.model.config, "hidden_size", 0))

    @property
    def dimension(self) -> int:
        return self.embedding_dim

    def __call__(self, waveform: np.ndarray) -> np.ndarray:
        inputs = self.extractor(
            waveform,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self.torch.inference_mode():
            hidden = self.model(**inputs).last_hidden_state
        if "attention_mask" in inputs:
            mask = inputs["attention_mask"].to(hidden.dtype).unsqueeze(-1)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        else:
            pooled = hidden.mean(dim=1)
        return l2_normalize(pooled[0].detach().cpu().numpy())


def extract_manifest(
    manifest_path: str,
    output_path: str,
    model_id: str,
    device: str = "cpu",
    sample_rate: int = 16000,
) -> int:
    records = load_records(manifest_path)
    roots = [os.path.dirname(os.path.abspath(manifest_path))]
    teacher = HuggingFaceTeacher(model_id, device, sample_rate)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as out:
        for index, record in enumerate(records, start=1):
            enrollment_path = record["enrollment_audio"]
            if not os.path.isabs(enrollment_path):
                enrollment_path = os.path.join(roots[0], enrollment_path)
            waveform = read_mono(enrollment_path, sample_rate)
            updated = dict(record)
            updated["teacher_embedding"] = teacher(waveform).round(8).tolist()
            updated["teacher_model_id"] = model_id
            updated["teacher_embedding_dim"] = teacher.dimension
            out.write(json.dumps(updated, ensure_ascii=False) + "\n")
            if index % 1000 == 0:
                print(f"  {index}/{len(records)} records")
    return len(records)


def main() -> None:
    parser = argparse.ArgumentParser(description="为 TSE manifest 提取冻结 speaker teacher embedding")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--sample-rate", type=int, default=16000)
    args = parser.parse_args()
    count = extract_manifest(
        args.manifest,
        args.output,
        args.model_id,
        device=args.device,
        sample_rate=args.sample_rate,
    )
    print(f"完成：{count} 条 teacher manifest -> {args.output}")


if __name__ == "__main__":
    main()
