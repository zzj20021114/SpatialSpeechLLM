"""dasr.data.build_qa — 由 FOA manifest 构建定向 ASR QA jsonl（独立原创实现）。

输入 manifest（generate_synthetic_foa 产出）：
    {key, wav, txt, sources: [{azimuth_deg, elevation_deg, distance_m}]}
输出 qa/{split}.jsonl（对齐 dataset.py 的 schema）。

用法：
    python -m dasr.data.build_qa \\
        --manifest data/synth/train/manifest.jsonl \\
        --qa-dir data/qa --split train \\
        --prompt-config conf/directional_asr_prompt.yaml
"""
from __future__ import annotations

import argparse
import json
import os
import random
import uuid

from ..prompts import (
    DIRECTION_TAG,
    LEFT,
    RIGHT,
    TAG_COMBO_TRANSCRIBE_DIRECTION,
    format_answer,
    format_leftright_answer,
    get_prompt,
    load_prompt_templates,
)


def build_answer(txt: str, sources: list, leftright: bool = True) -> str:
    """按声源数量构造答案。

    leftright=True：输出 `<方位> 左/右`（二分类，首期）。
    leftright=False：输出 `方位角=/仰角=`（回归，扩展）。
    """
    if not sources:
        raise ValueError("manifest 记录缺少 sources")
    if leftright:
        sides = [s.get("side") for s in sources]
        if len(sides) == 1:
            return format_leftright_answer(txt, sides[0])
        parts = [txt] + [
            f"第{i}位说话人 {DIRECTION_TAG} {side}"
            for i, side in enumerate(sides, start=1)
        ]
        return " ".join(parts)
    if len(sources) == 1:
        s = sources[0]
        return format_answer(txt, s["azimuth_deg"], s["elevation_deg"])
    parts = [txt]
    for i, s in enumerate(sources, start=1):
        parts.append(
            f"第{i}位说话人 {DIRECTION_TAG} "
            f"方位角={float(s['azimuth_deg']):.1f}, 仰角={float(s['elevation_deg']):.1f}"
        )
    return " ".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="构建定向 ASR QA 数据")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--qa-dir", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--prompt-config", default="conf/directional_asr_prompt.yaml")
    parser.add_argument("--leftright", action=argparse.BooleanOptionalAction, default=True,
                        help="左右定向模式（默认开）：answer 用 <方位> 左/右")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.qa_dir, exist_ok=True)
    templates = load_prompt_templates(args.prompt_config)
    rng = random.Random(args.seed)

    out_path = os.path.join(args.qa_dir, f"{args.split}.jsonl")
    manifest_dir = os.path.dirname(os.path.abspath(args.manifest))
    n = 0
    with open(args.manifest, encoding="utf-8") as mf, open(out_path, "w", encoding="utf-8") as of:
        for line in mf:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            txt = rec.get("txt", "").strip()
            if not txt:
                raise ValueError(f"{rec['key']} 缺转录 txt")
            sources = rec.get("sources", [])
            prompt = get_prompt(templates, TAG_COMBO_TRANSCRIBE_DIRECTION, rng)
            answer = build_answer(txt, sources, leftright=args.leftright)
            audio_path = rec["wav"]
            if not os.path.isabs(audio_path):
                audio_path = os.path.join(manifest_dir, audio_path)
            source_refs = [
                {
                    "class_id": -1,
                    "class_name": "speech",
                    "azimuth_deg": s.get("azimuth_deg"),
                    "elevation_deg": s.get("elevation_deg"),
                    "distance_m": s.get("distance_m"),
                    "side": s.get("side"),
                }
                for s in sources
            ]
            of.write(json.dumps({
                "pair_id": rec.get("key", str(uuid.uuid4())[:8]),
                "split": args.split,
                "audio_path": audio_path,
                "dataset": "synthetic",
                "task_type": "directional_asr",
                "task_name": "transcribe_and_locate",
                "prompt": prompt,
                "question": prompt,
                "answer": answer,
                "canonical_answer": txt,
                "transcription": txt,
                "source_refs": source_refs,
            }, ensure_ascii=False) + "\n")
            n += 1

    print(f"完成：{n} 条定向 ASR QA -> {out_path}")


if __name__ == "__main__":
    main()
