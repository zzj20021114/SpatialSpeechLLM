"""dasr.data.seld_dataset — SELD 预训练数据装载（独立原创实现）。

用于 S0 阶段在公开 FOA SELD 数据上预训练空间编码器。

统一采用本项目约定的事件式 manifest（可从 DCASE 等格式转换而来）：
    {"audio_path": ".../foa.wav",
     "events": [{"class_id": 0, "class_name": "speech",
                 "azimuth_deg": 45.0, "elevation_deg": 10.0,
                 "start_s": 0.0, "end_s": 3.2}, ...]}

SeldDataset 把事件式标注转换为帧级 ACCDOA 目标（label_rate Hz，默认 10 Hz）：
    activity  [T_l, C]  0/1
    coords    [T_l, C, 3] 笛卡尔坐标（单位球）
"""
from __future__ import annotations

import json
import math
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .foa import read_foa_wav


def cartesian_from_doa(azimuth_deg: float, elevation_deg: float) -> Tuple[float, float, float]:
    """方位/仰角（度）-> 单位球笛卡尔坐标 (x, y, z)。"""
    az = math.radians(azimuth_deg % 360.0)
    el = math.radians(max(-90.0, min(90.0, elevation_deg)))
    return (
        math.cos(el) * math.cos(az),
        math.cos(el) * math.sin(az),
        math.sin(el),
    )


class SeldDataset(Dataset):
    def __init__(
        self,
        manifest: str,
        sample_rate: int = 16000,
        label_rate: float = 10.0,
        max_audio_seconds: float = 20.0,
        num_classes: Optional[int] = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.label_rate = label_rate
        self.max_audio_samples = int(max_audio_seconds * sample_rate)
        self.num_classes = num_classes
        with open(manifest, encoding="utf-8") as f:
            self.records = [json.loads(line) for line in f if line.strip()]

    def __len__(self) -> int:
        return len(self.records)

    def _frame_labels(self, rec: Dict, n_frames: int) -> Tuple[np.ndarray, np.ndarray]:
        """事件标注 -> (activity[T,C], coords[T,C,3])。"""
        events = rec.get("events", [])
        if events and self.num_classes is None:
            self.num_classes = max(int(e.get("class_id", 0)) for e in events) + 1
        C = self.num_classes or 1
        activity = np.zeros((n_frames, C), dtype=np.float32)
        coords = np.zeros((n_frames, C, 3), dtype=np.float32)
        for e in events:
            cid = int(e.get("class_id", 0))
            x, y, z = cartesian_from_doa(e["azimuth_deg"], e["elevation_deg"])
            lo = max(0, int(math.floor(e.get("start_s", 0.0) * self.label_rate)))
            hi = min(n_frames, int(math.ceil(e.get("end_s", 0.0) * self.label_rate)))
            activity[lo:hi, cid] = 1.0
            coords[lo:hi, cid] = (x, y, z)
        return activity, coords

    def __getitem__(self, idx: int) -> Dict:
        rec = self.records[idx]
        wav = read_foa_wav(rec["audio_path"], self.sample_rate)          # [4, T]
        if wav.shape[1] > self.max_audio_samples:
            wav = wav[:, : self.max_audio_samples]
        n_frames = int(np.ceil(wav.shape[1] / 160))                      # 10ms 帧
        activity, coords = self._frame_labels(rec, n_frames)
        return {
            "spatial_audio": torch.from_numpy(wav),
            "spatial_audio_lengths": torch.tensor(wav.shape[1], dtype=torch.long),
            "activity": torch.from_numpy(activity),
            "coords": torch.from_numpy(coords),
            "label_rate": self.label_rate,
        }


class SeldCollator:
    """把 SeldDataset 样本拼成 batch（FOA 补零 + 标签对齐到最大帧长）。"""

    def __call__(self, samples: List[Dict]) -> Dict[str, torch.Tensor]:
        B = len(samples)
        max_t = max(int(s["spatial_audio"].shape[1]) for s in samples)
        max_frames = max(int(s["activity"].shape[0]) for s in samples)
        sa = torch.zeros(B, 4, max_t, dtype=torch.float32)
        lens = torch.zeros(B, dtype=torch.long)
        act = torch.zeros(B, max_frames, samples[0]["activity"].shape[1], dtype=torch.float32)
        coord = torch.zeros(B, max_frames, samples[0]["coords"].shape[1], 3, dtype=torch.float32)
        for i, s in enumerate(samples):
            sa[i, :, : s["spatial_audio"].shape[1]] = s["spatial_audio"]
            lens[i] = s["spatial_audio_lengths"]
            act[i, : s["activity"].shape[0]] = s["activity"]
            coord[i, : s["coords"].shape[0]] = s["coords"]
        return {
            "spatial_audio": sa,
            "spatial_audio_lengths": lens,
            "activity": act,
            "coords": coord,
            "label_rate": samples[0]["label_rate"],
        }
