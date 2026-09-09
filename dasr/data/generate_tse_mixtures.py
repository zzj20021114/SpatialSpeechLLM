"""Generate ideal two-speaker FOA mixtures for target speaker extraction.

The first TSE data stage is intentionally simple and fully supervised:
two utterances from different speakers overlap from time zero, the target
speaker is spatialized at one DOA, and a different utterance from the same
target speaker is stored as enrollment audio.

Input manifest records must contain ``key``, ``wav`` and ``txt``.  A
``speaker_id`` field is preferred; AISHELL-style keys and LibriSpeech paths
are supported as fallbacks.

Example:
    python -m dasr.data.generate_tse_mixtures \
        --speech-manifest data/speech/smoke_train.jsonl \
        --output-dir data/tse_smoke/train --split train --max-scenes 8
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf

from ..prompts import LEFT, RIGHT
from .foa import encode_point_source


def load_speech_manifest(path: str) -> List[Dict]:
    records: List[Dict] = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            for field in ("key", "wav", "txt"):
                if not rec.get(field):
                    raise ValueError(f"{path}:{line_no} 缺少字段 {field!r}")
            records.append(rec)
    if not records:
        raise ValueError(f"语音 manifest 为空: {path}")
    return records


def infer_speaker_id(record: Dict) -> str:
    """Infer a stable speaker id for common AISHELL/LibriSpeech layouts."""
    for field in ("speaker_id", "speaker", "spk_id"):
        if record.get(field) is not None:
            return str(record[field])

    key = str(record.get("key", ""))
    # AISHELL keys contain e.g. BAC009S0002W0122.
    match = re.search(r"S\d{3,}", key, flags=re.IGNORECASE)
    if match:
        return match.group(0).upper()

    # LibriSpeech paths contain split/speaker/chapter/utterance.flac.
    parts = Path(str(record.get("wav", ""))).parts
    for index, part in enumerate(parts[:-2]):
        if part.startswith(("train-", "dev-", "test-")) and parts[index + 1].isdigit():
            return parts[index + 1]

    raise ValueError(
        f"无法推断说话人 ID: key={key!r} wav={record.get('wav')!r}; "
        "请在 manifest 中增加 speaker_id 字段"
    )


def read_mono(path: str, sample_rate: int) -> np.ndarray:
    data, source_rate = sf.read(path, dtype="float32", always_2d=True)
    mono = data[:, 0].astype(np.float32, copy=False)
    if int(source_rate) == sample_rate:
        return mono
    try:
        from scipy.signal import resample_poly
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError(
            f"音频 {path} 采样率为 {source_rate}，重采样需要 scipy"
        ) from exc
    gcd = math.gcd(int(source_rate), int(sample_rate))
    return resample_poly(
        mono, sample_rate // gcd, int(source_rate) // gcd
    ).astype(np.float32)


def side_from_azimuth(azimuth_deg: float) -> str:
    az = float(azimuth_deg) % 360.0
    if 45.0 <= az <= 135.0:
        return LEFT
    if 225.0 <= az <= 315.0:
        return RIGHT
    raise ValueError(f"方位角不在左右有效区间: {azimuth_deg}")


def _sample_azimuth(rng: random.Random, side_range: Tuple[float, float]) -> Tuple[float, str]:
    side = rng.choice((LEFT, RIGHT))
    if side == LEFT:
        azimuth = rng.uniform(*side_range)
    else:
        azimuth = rng.uniform(360.0 - side_range[1], 360.0 - side_range[0])
    return round(float(azimuth), 1), side


def _rms(signal: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(signal), dtype=np.float64) + 1e-12))


def _match_snr(reference: np.ndarray, interferer: np.ndarray, snr_db: Optional[float]) -> np.ndarray:
    """Scale the interferer to the requested target-to-interferer SNR."""
    if snr_db is None:
        return interferer
    desired_rms = _rms(reference) / (10.0 ** (float(snr_db) / 20.0))
    return interferer * (desired_rms / max(_rms(interferer), 1e-8))


def _pad_to(signal: np.ndarray, length: int) -> np.ndarray:
    if signal.shape[0] >= length:
        return signal[:length].astype(np.float32, copy=False)
    return np.pad(signal, (0, length - signal.shape[0])).astype(np.float32, copy=False)


def make_tse_scene(
    target_record: Dict,
    interferer_record: Dict,
    enrollment_record: Dict,
    rng: random.Random,
    sample_rate: int = 16000,
    max_seconds: float = 20.0,
    elevation_range: Tuple[float, float] = (-30.0, 30.0),
    distance_range: Tuple[float, float] = (1.0, 1.0),
    side_azimuth_range: Tuple[float, float] = (45.0, 135.0),
    snr_db: Optional[float] = None,
) -> Dict:
    """Build one ideal target-speaker scene and its complete metadata."""
    max_samples = int(max_seconds * sample_rate)
    target = read_mono(target_record["wav"], sample_rate)[:max_samples]
    interferer = read_mono(interferer_record["wav"], sample_rate)[:max_samples]
    enrollment = read_mono(enrollment_record["wav"], sample_rate)[:max_samples]
    if not len(target) or not len(interferer) or not len(enrollment):
        raise ValueError("target/interferer/enrollment 不能是空音频")

    target_az, target_side = _sample_azimuth(rng, side_azimuth_range)
    interferer_az, interferer_side = _sample_azimuth(rng, side_azimuth_range)
    target_el = round(rng.uniform(*elevation_range), 1)
    interferer_el = round(rng.uniform(*elevation_range), 1)
    target_distance = round(rng.uniform(*distance_range), 2)
    interferer_distance = round(rng.uniform(*distance_range), 2)

    target_active_samples = len(target)
    interferer_active_samples = len(interferer)
    target_source = target / max(target_distance, 0.1)
    interferer_source = interferer / max(interferer_distance, 0.1)
    interferer_source = _match_snr(target_source, interferer_source, snr_db)
    scene_len = max(len(target_source), len(interferer_source))
    target_source = _pad_to(target_source, scene_len)
    interferer_source = _pad_to(interferer_source, scene_len)

    target_foa = encode_point_source(target_source, target_az, target_el)
    interferer_foa = encode_point_source(interferer_source, interferer_az, interferer_el)
    mixture_foa = target_foa + interferer_foa

    target_id = infer_speaker_id(target_record)
    interferer_id = infer_speaker_id(interferer_record)
    enrollment_id = infer_speaker_id(enrollment_record)
    if target_id != enrollment_id:
        raise ValueError("enrollment 必须来自 target speaker")
    if target_id == interferer_id:
        raise ValueError("target 与 interferer 必须来自不同说话人")

    target_meta = {
        "key": target_record["key"],
        "speaker_id": target_id,
        "transcription": str(target_record["txt"]).strip(),
        "azimuth_deg": target_az,
        "elevation_deg": target_el,
        "distance_m": target_distance,
        "side": target_side,
        "num_samples": int(len(target_source)),
        "active_samples": int(target_active_samples),
    }
    interferer_meta = {
        "key": interferer_record["key"],
        "speaker_id": interferer_id,
        "transcription": str(interferer_record["txt"]).strip(),
        "azimuth_deg": interferer_az,
        "elevation_deg": interferer_el,
        "distance_m": interferer_distance,
        "side": interferer_side,
        "num_samples": int(len(interferer_source)),
        "active_samples": int(interferer_active_samples),
    }
    return {
        "mixture_foa": mixture_foa.astype(np.float32, copy=False),
        "target_foa": target_foa.astype(np.float32, copy=False),
        "interferer_foa": interferer_foa.astype(np.float32, copy=False),
        "target_audio": target_source.astype(np.float32, copy=False),
        "interferer_audio": interferer_source.astype(np.float32, copy=False),
        "enrollment_audio": enrollment.astype(np.float32, copy=False),
        "target": target_meta,
        "interferer": interferer_meta,
        "enrollment": {
            "key": enrollment_record["key"],
            "speaker_id": enrollment_id,
            "num_samples": int(len(enrollment)),
        },
        "snr_db": snr_db,
        "sample_rate": sample_rate,
        "num_samples": int(scene_len),
    }


def _relative_path(path: str, root: str) -> str:
    return os.path.relpath(path, root).replace(os.sep, "/")


def _write_scene(output_dir: str, split: str, index: int, scene: Dict) -> Dict:
    pair_id = f"tse_{split}_{index:07d}"
    roots = {
        "mixture": os.path.join(output_dir, "audio", split),
        "target": os.path.join(output_dir, "target", split),
        "interferer": os.path.join(output_dir, "interferer", split),
        "enrollment": os.path.join(output_dir, "enrollment", split),
    }
    for root in roots.values():
        os.makedirs(root, exist_ok=True)

    paths = {
        "mixture": os.path.join(roots["mixture"], f"{pair_id}.wav"),
        "target": os.path.join(roots["target"], f"{pair_id}.wav"),
        "interferer": os.path.join(roots["interferer"], f"{pair_id}.wav"),
        "enrollment": os.path.join(roots["enrollment"], f"{pair_id}.wav"),
    }
    sf.write(paths["mixture"], scene["mixture_foa"].T, scene["sample_rate"], subtype="FLOAT")
    sf.write(paths["target"], scene["target_audio"], scene["sample_rate"], subtype="FLOAT")
    sf.write(paths["interferer"], scene["interferer_audio"], scene["sample_rate"], subtype="FLOAT")
    sf.write(paths["enrollment"], scene["enrollment_audio"], scene["sample_rate"], subtype="FLOAT")

    target = scene["target"]
    interferer = scene["interferer"]
    source_refs = [
        {"role": "target", **target},
        {"role": "interferer", **interferer},
    ]
    return {
        "pair_id": pair_id,
        "split": split,
        "task_type": "target_speaker_extraction",
        "task_name": "extract_transcribe_and_locate_target",
        "sample_rate": scene["sample_rate"],
        "num_samples": scene["num_samples"],
        "mixture_audio": _relative_path(paths["mixture"], output_dir),
        "target_clean_audio": _relative_path(paths["target"], output_dir),
        "interferer_audio": _relative_path(paths["interferer"], output_dir),
        "enrollment_audio": _relative_path(paths["enrollment"], output_dir),
        "transcription": target["transcription"],
        "target_transcription": target["transcription"],
        "target_speaker_id": target["speaker_id"],
        "target_azimuth_deg": target["azimuth_deg"],
        "target_elevation_deg": target["elevation_deg"],
        "target_side": target["side"],
        "snr_db": scene["snr_db"],
        "enrollment": scene["enrollment"],
        "source_refs": source_refs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="生成理想双说话人 FOA TSE 数据")
    parser.add_argument("--speech-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-scenes", type=int, default=0, help="0 表示使用全部可构造场景")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--max-seconds", type=float, default=20.0)
    parser.add_argument("--elevation-range", nargs=2, type=float, default=(-30.0, 30.0))
    parser.add_argument("--distance-range", nargs=2, type=float, default=(1.0, 1.0))
    parser.add_argument("--side-azimuth-range", nargs=2, type=float, default=(45.0, 135.0))
    parser.add_argument("--snr-db", type=float, default=None, help="目标/干扰 SNR；默认保持原始能量")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    records = load_speech_manifest(args.speech_manifest)
    by_speaker: Dict[str, List[Dict]] = {}
    for record in records:
        speaker_id = infer_speaker_id(record)
        by_speaker.setdefault(speaker_id, []).append(record)
    eligible = [speaker for speaker, items in by_speaker.items() if len(items) >= 2]
    if len(eligible) < 2:
        raise SystemExit("至少需要两个说话人，且目标说话人至少有两条语音")

    rng = random.Random(args.seed)
    max_scenes = args.max_scenes if args.max_scenes > 0 else len(records)
    os.makedirs(args.output_dir, exist_ok=True)
    manifest_path = os.path.join(args.output_dir, "manifest.jsonl")
    written = 0
    with open(manifest_path, "w", encoding="utf-8") as manifest:
        for index in range(max_scenes):
            target_speaker = rng.choice(eligible)
            interferer_speakers = [s for s in eligible if s != target_speaker]
            interferer_speaker = rng.choice(interferer_speakers)
            target_record, enrollment_record = rng.sample(by_speaker[target_speaker], 2)
            interferer_record = rng.choice(by_speaker[interferer_speaker])
            scene = make_tse_scene(
                target_record,
                interferer_record,
                enrollment_record,
                rng,
                sample_rate=args.sample_rate,
                max_seconds=args.max_seconds,
                elevation_range=tuple(args.elevation_range),
                distance_range=tuple(args.distance_range),
                side_azimuth_range=tuple(args.side_azimuth_range),
                snr_db=args.snr_db,
            )
            manifest.write(json.dumps(
                _write_scene(args.output_dir, args.split, index, scene),
                ensure_ascii=False,
            ) + "\n")
            written += 1
            if written % 1000 == 0:
                print(f"  {written}/{max_scenes} scenes")
    print(f"完成：{written} 条 TSE 场景 -> {manifest_path}")


if __name__ == "__main__":
    main()
