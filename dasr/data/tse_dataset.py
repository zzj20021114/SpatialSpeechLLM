"""Dataset and collation utilities for target-speaker extraction.

The dataset consumes manifests produced by ``generate_tse_mixtures.py``.
It deliberately keeps tokenization out of this layer: the batch contains
audio tensors and target metadata, so the same data contract can feed a
standalone extractor or the later ASR/LLM joint model.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .foa import read_foa_wav
from .generate_tse_mixtures import read_mono


AUDIO_FIELDS = (
    "mixture_audio",
    "target_clean_audio",
    "interferer_audio",
    "enrollment_audio",
)
SIDE_TO_ID = {"左": 0, "右": 1}


def _resolve_path(path: str, roots: Sequence[str]) -> str:
    if os.path.isabs(path):
        return path
    for root in roots:
        candidate = os.path.join(root, path)
        if os.path.exists(candidate):
            return os.path.abspath(candidate)
    return os.path.abspath(os.path.join(roots[0], path))


def _source_ref(record: Dict[str, Any], role: str) -> Dict[str, Any]:
    for source in record.get("source_refs", []):
        if source.get("role") == role:
            return source
    raise ValueError(f"record {record.get('pair_id')} 缺少 source_refs[{role!r}]")


class TseDataset:
    """JSONL-backed TSE dataset with path resolution and validation."""

    def __init__(
        self,
        path: str,
        audio_roots: Optional[List[str]] = None,
        max_samples: Optional[int] = None,
        verify_audio: bool = False,
    ) -> None:
        self.path = os.path.abspath(path)
        base_dir = os.path.dirname(self.path)
        roots = [os.path.abspath(root) for root in (audio_roots or [])]
        roots.append(base_dir)
        self.records: List[Dict[str, Any]] = []

        with open(path, encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                if max_samples is not None and len(self.records) >= max_samples:
                    break
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                for field in AUDIO_FIELDS:
                    if not record.get(field):
                        raise ValueError(f"{path}:{line_no} 缺少字段 {field!r}")
                    record[field] = _resolve_path(record[field], roots)
                _source_ref(record, "target")
                _source_ref(record, "interferer")
                if record.get("target_side") not in SIDE_TO_ID:
                    raise ValueError(
                        f"{path}:{line_no} target_side 必须为左或右，"
                        f"得到 {record.get('target_side')!r}"
                    )
                if verify_audio:
                    self._verify_audio(record, line_no)
                self.records.append(record)

    @staticmethod
    def _verify_audio(record: Dict[str, Any], line_no: int) -> None:
        import soundfile as sf

        for field in AUDIO_FIELDS:
            try:
                sf.info(record[field])
            except Exception as exc:
                raise ValueError(
                    f"record {line_no} 的 {field} 无法读取: {record[field]}"
                ) from exc

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.records[index]


def _trim(signal: np.ndarray, max_samples: int) -> np.ndarray:
    return signal[..., :max_samples].astype(np.float32, copy=False)


class TseCollator:
    """Load and pad TSE audio, returning tensors for future joint training."""

    def __init__(
        self,
        sample_rate: int = 16000,
        max_audio_seconds: float = 20.0,
        max_enrollment_seconds: float = 5.0,
        activity_hop_length: int = 160,
    ) -> None:
        self.sample_rate = sample_rate
        self.max_audio_samples = int(max_audio_seconds * sample_rate)
        self.max_enrollment_samples = int(max_enrollment_seconds * sample_rate)
        self.activity_hop_length = activity_hop_length

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError("TseCollator 需要安装 torch") from exc
        if not features:
            raise ValueError("TseCollator 不能处理空 batch")

        mixtures: List[np.ndarray] = []
        targets: List[np.ndarray] = []
        interferers: List[np.ndarray] = []
        enrollments: List[np.ndarray] = []
        target_active_samples: List[int] = []
        interferer_active_samples: List[int] = []

        for record in features:
            mixture = _trim(
                read_foa_wav(record["mixture_audio"], self.sample_rate),
                self.max_audio_samples,
            )
            target = _trim(
                read_mono(record["target_clean_audio"], self.sample_rate),
                self.max_audio_samples,
            )
            interferer = _trim(
                read_mono(record["interferer_audio"], self.sample_rate),
                self.max_audio_samples,
            )
            enrollment = _trim(
                read_mono(record["enrollment_audio"], self.sample_rate),
                self.max_enrollment_samples,
            )
            mixtures.append(mixture)
            targets.append(target)
            interferers.append(interferer)
            enrollments.append(enrollment)
            target_active_samples.append(
                min(int(_source_ref(record, "target").get("active_samples", len(target))),
                    self.max_audio_samples)
            )
            interferer_active_samples.append(
                min(int(_source_ref(record, "interferer").get("active_samples", len(interferer))),
                    self.max_audio_samples)
            )

        max_audio = max(audio.shape[-1] for audio in mixtures)
        max_enrollment = max(audio.shape[-1] for audio in enrollments)
        batch_size = len(features)
        mixture_batch = torch.zeros(batch_size, 4, max_audio, dtype=torch.float32)
        target_batch = torch.zeros(batch_size, max_audio, dtype=torch.float32)
        interferer_batch = torch.zeros(batch_size, max_audio, dtype=torch.float32)
        enrollment_batch = torch.zeros(batch_size, max_enrollment, dtype=torch.float32)
        mixture_lengths = torch.zeros(batch_size, dtype=torch.long)
        enrollment_lengths = torch.zeros(batch_size, dtype=torch.long)

        for index, (mixture, target, interferer, enrollment) in enumerate(
            zip(mixtures, targets, interferers, enrollments)
        ):
            mixture_len = mixture.shape[-1]
            mixture_batch[index, :, :mixture_len] = torch.from_numpy(mixture)
            target_batch[index, : target.shape[-1]] = torch.from_numpy(target)
            interferer_batch[index, : interferer.shape[-1]] = torch.from_numpy(interferer)
            enrollment_batch[index, : enrollment.shape[-1]] = torch.from_numpy(enrollment)
            mixture_lengths[index] = mixture_len
            enrollment_lengths[index] = enrollment.shape[-1]

        num_frames = (max_audio + self.activity_hop_length - 1) // self.activity_hop_length
        target_activity = torch.zeros(batch_size, num_frames, dtype=torch.float32)
        interferer_activity = torch.zeros(batch_size, num_frames, dtype=torch.float32)
        for index in range(batch_size):
            target_frames = min(
                num_frames,
                (target_active_samples[index] + self.activity_hop_length - 1)
                // self.activity_hop_length,
            )
            interferer_frames = min(
                num_frames,
                (interferer_active_samples[index] + self.activity_hop_length - 1)
                // self.activity_hop_length,
            )
            target_activity[index, :target_frames] = 1.0
            interferer_activity[index, :interferer_frames] = 1.0

        return {
            "mixture_audio": mixture_batch,
            "target_audio": target_batch,
            "interferer_audio": interferer_batch,
            "enrollment_audio": enrollment_batch,
            "mixture_lengths": mixture_lengths,
            "enrollment_lengths": enrollment_lengths,
            "target_activity": target_activity,
            "interferer_activity": interferer_activity,
            "target_side": torch.tensor(
                [SIDE_TO_ID[record["target_side"]] for record in features],
                dtype=torch.long,
            ),
            "target_azimuth_deg": torch.tensor(
                [float(record["target_azimuth_deg"]) for record in features],
                dtype=torch.float32,
            ),
            "target_elevation_deg": torch.tensor(
                [float(record["target_elevation_deg"]) for record in features],
                dtype=torch.float32,
            ),
            "target_transcriptions": [
                record.get("target_transcription") or record.get("transcription", "")
                for record in features
            ],
            "target_speaker_ids": [record["target_speaker_id"] for record in features],
            "meta": features,
        }
