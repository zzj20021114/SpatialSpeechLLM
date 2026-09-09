"""dasr.data.dataset — 定向 ASR QA 数据集与 collator（独立原创实现）。

数据记录 schema（与 build_qa.py 产出一致）：
    {pair_id, split, audio_path(4ch FOA), task_type, task_name,
     prompt, answer, canonical_answer, transcription,
     source_refs: [{azimuth_deg, elevation_deg, distance_m, side}]}

collator 职责：读取 FOA -> 计算编码器 token 占位符数 -> 拼文本
（`<|speech|>`*N + prompt + answer）-> tokenize -> 打包 spatial_audio。
语音特征与方位编码全部在模型内（AudioFrontend + SpatialEncoder）计算。
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from .foa import read_foa_wav


def speech_placeholder_count(
    audio_len: int,
    sample_rate: int,
    hop_length: int,
    patch_time: int,
    stride_time: int,
    shuffle_factor: int,
) -> int:
    """由波形采样数计算 LLM 侧语音占位符数量（与 SpatialEncoder 前向逐 token 对齐）。"""
    T_f = audio_len // hop_length + 1
    n_patch = max(1, (T_f - patch_time) // stride_time + 1)
    return max(1, n_patch // shuffle_factor)


class QaDataset(Dataset):
    def __init__(self, path: str, audio_roots: Optional[List[str]] = None,
                 max_samples: Optional[int] = None, verify_audio: bool = False) -> None:
        self.records: List[Dict[str, Any]] = []
        base_dir = os.path.dirname(os.path.abspath(path))
        roots = [base_dir, os.path.dirname(base_dir)]
        if audio_roots:
            roots = [os.path.abspath(r) for r in audio_roots] + roots

        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if max_samples is not None and i >= max_samples:
                    break
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                ap = rec.get("audio_path")
                if not ap:
                    raise ValueError(f"record {i} 缺 audio_path")
                if not os.path.isabs(ap):
                    for root in roots:
                        cand = os.path.join(root, ap)
                        if os.path.exists(cand):
                            ap = cand
                            break
                    else:
                        ap = os.path.join(base_dir, ap)
                rec["audio_path"] = ap
                if verify_audio:
                    try:
                        import soundfile as sf
                        sf.info(ap)
                    except Exception:
                        print(f"[skip] 无法读取音频: {ap}")
                        continue
                if rec.get("prompt") is None:
                    rec["prompt"] = rec.get("question", "")
                if not rec.get("answer"):
                    raise ValueError(f"record {i} 缺 answer")
                self.records.append(rec)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.records[idx]


class DasrCollator:
    """把记录列表拼成训练 batch。

    文本模板：`<|speech|> * N\n{prompt}\n{answer}{eos}`
    labels：prefix（语音占位符 + prompt）置 -100，仅监督 answer+eos。
    """

    def __init__(
        self,
        tokenizer,
        speech_token: str,
        eos_token: str,
        sample_rate: int = 16000,
        hop_length: int = 160,
        patch_time: int = 16,
        stride_time: int = 8,
        shuffle_factor: int = 1,
        max_audio_seconds: float = 20.0,
    ) -> None:
        self.tokenizer = tokenizer
        self.speech_token = speech_token
        self.eos = eos_token or "<|endoftext|>"
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.patch_time = patch_time
        self.stride_time = stride_time
        self.shuffle_factor = shuffle_factor
        self.max_audio_samples = int(max_audio_seconds * sample_rate)

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        spatial_audio: List[np.ndarray] = []
        spatial_lengths: List[int] = []
        texts: List[str] = []
        answers_sfx: List[str] = []
        meta: List[Dict] = []

        for feat in features:
            wav = read_foa_wav(feat["audio_path"], self.sample_rate)   # [4, T]
            if wav.shape[1] > self.max_audio_samples:
                wav = wav[:, : self.max_audio_samples]
            spatial_audio.append(wav)
            spatial_lengths.append(wav.shape[1])

            n_sp = speech_placeholder_count(
                wav.shape[1], self.sample_rate, self.hop_length,
                self.patch_time, self.stride_time, self.shuffle_factor,
            )
            prompt = str(feat["prompt"]).rstrip()
            answer = str(feat["answer"]).strip()
            texts.append(f"{self.speech_token * n_sp}\n{prompt}\n{answer}{self.eos}")
            answers_sfx.append(answer + self.eos)
            meta.append(feat)

        max_len = max(spatial_lengths)
        sa_t = torch.zeros(len(spatial_audio), 4, max_len, dtype=torch.float32)
        for i, w in enumerate(spatial_audio):
            sa_t[i, :, : w.shape[1]] = torch.from_numpy(w)
        sa_lens = torch.tensor(spatial_lengths, dtype=torch.long)

        tok = self.tokenizer
        prev_side = getattr(tok, "padding_side", "left")
        tok.padding_side = "right"
        try:
            batch = tok(texts, padding=True, return_tensors="pt")
        finally:
            tok.padding_side = prev_side

        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        # labels：prefix 置 -100
        ans_tok = tok(answers_sfx, padding=True, return_tensors="pt", add_special_tokens=False)
        al = ans_tok["attention_mask"].sum(dim=1).long()
        vl = attention_mask.sum(dim=1).long()
        pl = vl - al
        if (pl < 0).any():
            raise ValueError("负的 prefix 长度，检查文本拼接/对齐")
        labels = input_ids.clone()
        labels = labels.masked_fill(attention_mask == 0, -100)
        for i, p in enumerate(pl.tolist()):
            labels[i, :p] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "spatial_audio": sa_t,
            "spatial_audio_lengths": sa_lens,
            "meta": meta,
        }
