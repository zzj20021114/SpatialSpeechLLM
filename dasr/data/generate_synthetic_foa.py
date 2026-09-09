"""dasr.data.generate_synthetic_foa — 合成 FOA 定向语音数据（独立原创实现）。

输入：语音语料 manifest（每行 {key, wav, txt}）。
输出：FOA wav（4ch, 16k, DCASE 序 [W,Y,Z,X]）+ manifest.jsonl，每个片段携带
说话人方位/距离/左右标注。

特性：
  - --workers N：多进程并行（默认 1）。
  - 断点续跑：已存在的 FOA wav 自动跳过，可随时重启。
  - 可复现：每个场景用 (seed, 场景序号) 派生随机种子，与进程数/续跑无关。

用法：
    python -m dasr.data.generate_synthetic_foa \\
        --speech-manifest data/speech/train.jsonl \\
        --output-dir data/synth/train --split train --workers 16
"""
from __future__ import annotations

import argparse
import json
import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf

from ..prompts import LEFT, RIGHT
from .foa import encode_point_source


def load_speech_manifest(path: str) -> List[Dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def read_speech(path: str) -> np.ndarray:
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    mono = data[:, 0] if data.shape[1] > 1 else data[:, 0]
    return mono.astype(np.float32)


def make_scene(
    utterances: List[Dict],
    rng: random.Random,
    np_rng: np.random.Generator,
    sr: int,
    az_range: tuple,
    el_range: tuple,
    dist_range: tuple,
    noise_snr: Optional[float],
    max_seconds: Optional[float],
    leftright: bool = True,
    side_az_range: tuple = (45.0, 135.0),
) -> Dict:
    """从 utterances 构造一个 FOA 场景（单源取首条，多源取全部叠加）。"""
    sources = []
    foas = []
    for utt in utterances:
        el = rng.uniform(*el_range)
        dist = rng.uniform(*dist_range)
        if leftright:
            side = rng.choice([LEFT, RIGHT])
            if side == LEFT:
                az = rng.uniform(*side_az_range)
            else:
                az = rng.uniform(360.0 - side_az_range[1], 360.0 - side_az_range[0])
        else:
            side = None
            az = rng.uniform(*az_range)
        wav = read_speech(utt["wav"])
        ch = encode_point_source(wav, az, el) / max(dist, 0.1)
        foas.append(ch)
        src = {
            "key": utt["key"],
            "txt": utt.get("txt", ""),
            "azimuth_deg": round(float(az), 1),
            "elevation_deg": round(float(el), 1),
            "distance_m": round(float(dist), 2),
        }
        if side is not None:
            src["side"] = side
        sources.append(src)

    max_len = max(ch.shape[1] for ch in foas)
    foa = np.zeros((4, max_len), dtype=np.float32)
    for ch in foas:
        n = ch.shape[1]
        foa[:, :n] += ch[:, :n]

    if noise_snr is not None:
        sig_pow = float(np.mean(foa ** 2))
        noise_pow = sig_pow / (10 ** (noise_snr / 10.0))
        noise = np_rng.normal(0.0, np.sqrt(max(noise_pow, 1e-12)), size=foa.shape)
        foa = foa + noise.astype(np.float32)

    if max_seconds is not None:
        max_len = int(max_seconds * sr)
        foa = foa[:, :max_len]

    peak = float(np.abs(foa).max())
    if peak > 0:
        foa = foa / peak * 0.9

    txt = " ".join(s["txt"] for s in sources)
    return {"foa": foa, "txt": txt, "sources": sources}


# ---------------------------------------------------------------------------
# 多进程 worker（模块级以便 pickle）
# ---------------------------------------------------------------------------
def _worker(task: dict) -> Tuple[int, Dict]:
    """task: {index, group, seed, sr, az_range, el_range, dist_range, noise_snr,
    max_seconds, leftright, side_az_range, split, audio_dir, subtype}"""
    index = task["index"]
    key = f"{task['split']}_{index * task['step']:07d}"
    wav_path = os.path.join(task["audio_dir"], f"{key}.wav")
    if os.path.exists(wav_path):
        return index, {"skipped": True}
    rng = random.Random(task["seed"] * 1_000_003 + index)
    np_rng = np.random.default_rng(task["seed"] + index)
    scene = make_scene(
        task["group"], rng, np_rng, task["sr"],
        task["az_range"], task["el_range"], task["dist_range"],
        task["noise_snr"], task["max_seconds"],
        leftright=task["leftright"], side_az_range=task["side_az_range"],
    )
    sf.write(wav_path, scene["foa"].T, task["sr"], subtype=task["subtype"])
    return index, {
        "key": key,
        "wav": os.path.join("audio", task["split"], f"{key}.wav"),
        "txt": scene["txt"],
        "sources": scene["sources"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="合成 FOA 定向语音数据")
    parser.add_argument("--speech-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--num-sources", type=int, default=1)
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--max-seconds", type=float, default=20.0)
    parser.add_argument("--leftright", action=argparse.BooleanOptionalAction, default=True,
                        help="左右定向模式（默认开）：方位只取左/右半区，记录 side")
    parser.add_argument("--side-azimuth-range", nargs=2, type=float, default=(45.0, 135.0),
                        help="左半区方位范围（右半区自动镜像）")
    parser.add_argument("--azimuth-range", nargs=2, type=float, default=(0.0, 360.0),
                        help="非 leftright 模式时的全局方位范围")
    parser.add_argument("--elevation-range", nargs=2, type=float, default=(-30.0, 30.0))
    parser.add_argument("--distance-range", nargs=2, type=float, default=(1.0, 5.0))
    parser.add_argument("--noise-snr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=1, help="并行进程数（0=自动）")
    parser.add_argument("--max-scenes", type=int, default=0,
                        help="最多生成的场景数（0=全部），便于分批")
    parser.add_argument("--subtype", default="PCM_16",
                        help="FOA wav 采样格式（PCM_16 省空间；FLOAT 精度更高）")
    args = parser.parse_args()

    audio_dir = os.path.join(args.output_dir, "audio", args.split)
    os.makedirs(audio_dir, exist_ok=True)

    records = load_speech_manifest(args.speech_manifest)
    if not records:
        raise SystemExit("语音 manifest 为空")

    step = max(1, args.num_sources)
    total = len(records) // step
    if args.max_scenes > 0:
        total = min(total, args.max_scenes)
    workers = args.workers
    if workers <= 0:
        workers = max(1, os.cpu_count() or 1)

    base_task = {
        "seed": args.seed, "sr": args.sr,
        "az_range": tuple(args.azimuth_range), "el_range": tuple(args.elevation_range),
        "dist_range": tuple(args.distance_range), "noise_snr": args.noise_snr,
        "max_seconds": args.max_seconds, "leftright": args.leftright,
        "side_az_range": tuple(args.side_azimuth_range),
        "split": args.split, "audio_dir": audio_dir, "subtype": args.subtype,
        "step": step,
    }
    tasks = [
        {**base_task, "index": j, "group": records[j * step:(j + 1) * step]}
        for j in range(total)
    ]

    manifest_path = os.path.join(args.output_dir, "manifest.jsonl")
    # 续跑：加载已存在的 manifest（已生成场景的标注复用）
    old_manifest: Dict[str, Dict] = {}
    if os.path.exists(manifest_path):
        with open(manifest_path, encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if ln:
                    r = json.loads(ln)
                    old_manifest[r["key"]] = r

    lines: Dict[int, Dict] = {}
    done = 0
    if workers > 1:
        from multiprocessing import Pool
        with Pool(workers) as pool:
            for j, result in enumerate(pool.imap_unordered(_worker, tasks, chunksize=64)):
                idx, line = result
                lines[idx] = line
                done += 1
                if done % 5000 == 0:
                    print(f"  {done}/{total} 场景")
    else:
        for j, task in enumerate(tasks):
            idx, line = _worker(task)
            lines[idx] = line
            done += 1
            if done % 5000 == 0:
                print(f"  {done}/{total} 场景")

    n_written = 0
    with open(manifest_path, "w", encoding="utf-8") as mf:
        for j in range(total):
            line = lines.get(j)
            if line is not None and line.get("skipped"):
                line = old_manifest.get(line.get("key"))
            if line is None:
                continue
            mf.write(json.dumps(line, ensure_ascii=False) + "\n")
            n_written += 1

    print(f"完成：{n_written}/{total} 个 FOA 场景 -> {manifest_path}")


if __name__ == "__main__":
    main()
