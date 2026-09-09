"""dasr.data.dcase_to_manifest — DCASE SELD 标注 -> 本项目 manifest（独立原创实现）。

把 DCASE Task3（2020–2024）的 `metadata.csv` 帧级标注转换为事件式 manifest，
供 S0 SELD 预训练使用（`dasr/data/seld_dataset.py`）。

DCASE metadata.csv 格式（每行一个 0.1s 帧）：
    audio_filename,event1,start1,end1,event2,start2,end2,...
    事件串形如 `class_azimuth_elevation[_distance]`，如 `speech_45_10`、`alarm_90_0_5`。

用法：
    python -m dasr.data.dcase_to_manifest \\
        --metadata-csv <dcase>/metadata.csv \\
        --audio-root <dcase>/audio \\
        --output seld_manifest.jsonl \\
        --classes classes.txt            # 可选，缺省按出现顺序推断 class_id
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple


def parse_event_str(s: str) -> Optional[Tuple[str, float, float, Optional[float]]]:
    """解析 `class_az_el[_dist]` -> (class_name, az, el, dist)。"""
    s = s.strip()
    if not s or s == "-" or s.lower() == "none":
        return None
    parts = s.split("_")
    if len(parts) == 3:
        cls, az, el = parts
        dist = None
    elif len(parts) == 4:
        cls, az, el, dist = parts
    else:
        raise ValueError(f"无法解析事件串: {s!r}")
    return cls, float(az), float(el), (float(dist) if dist is not None else None)


def _merge_frames(rows: List[Tuple[str, float, float, Optional[float]]],
                  frame_s: float) -> List[Dict]:
    """把相邻同(class,az,el)帧合并为事件区间。"""
    events: List[Dict] = []
    cur: Optional[Dict] = None
    for i, (cls, az, el, dist) in enumerate(rows):
        t_s = round(i * frame_s, 2)
        if cur is not None and cur["class_name"] == cls and abs(cur["azimuth_deg"] - az) < 1e-6 \
                and abs(cur["elevation_deg"] - el) < 1e-6:
            cur["end_s"] = round(t_s + frame_s, 2)
            continue
        if cur is not None:
            events.append(cur)
        cur = {"class_name": cls, "azimuth_deg": az, "elevation_deg": el,
               "start_s": t_s, "end_s": round(t_s + frame_s, 2)}
        if dist is not None:
            cur["distance_m"] = dist
    if cur is not None:
        events.append(cur)
    return events


def convert(
    metadata_csv: str,
    audio_root: str,
    frame_s: float = 0.1,
    classes: Optional[List[str]] = None,
    max_audio_seconds: float = 20.0,
) -> List[Dict]:
    """解析 DCASE metadata.csv，返回 manifest 记录列表。"""
    frames: Dict[str, List[Tuple[str, float, float, Optional[float]]]] = OrderedDict()
    with open(metadata_csv, encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        n_events = (len(header) - 1) // 3
        for row in reader:
            if not row or not row[0]:
                continue
            fname = row[0]
            for i in range(n_events):
                ev = parse_event_str(row[1 + i * 3])
                if ev is None:
                    continue
                frames.setdefault(fname, []).append(ev)

    class_id_map: Dict[str, int] = {}
    if classes:
        class_id_map = {c: i for i, c in enumerate(classes)}
    else:
        seen: List[str] = []
        for evs in frames.values():
            for cls, *_ in evs:
                if cls not in seen:
                    seen.append(cls)
        class_id_map = {c: i for i, c in enumerate(seen)}

    records: List[Dict] = []
    for fname, evs in frames.items():
        events = _merge_frames(evs, frame_s)
        events = [
            {**e, "class_id": class_id_map[e["class_name"]]}
            for e in events if e["end_s"] <= max_audio_seconds
        ]
        records.append({
            "audio_path": os.path.join(audio_root, fname),
            "events": events,
        })
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="DCASE SELD -> manifest")
    parser.add_argument("--metadata-csv", required=True)
    parser.add_argument("--audio-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--classes", default=None, help="类名列表文件（每行一个）")
    parser.add_argument("--frame-s", type=float, default=0.1)
    parser.add_argument("--max-audio-seconds", type=float, default=20.0)
    args = parser.parse_args()

    classes = None
    if args.classes:
        with open(args.classes, encoding="utf-8") as f:
            classes = [ln.strip() for ln in f if ln.strip()]

    records = convert(args.metadata_csv, args.audio_root, args.frame_s,
                      classes, args.max_audio_seconds)
    with open(args.output, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"完成：{len(records)} 段 -> {args.output}")


if __name__ == "__main__":
    main()
