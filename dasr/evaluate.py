"""dasr.evaluate — 定向 ASR 评估（独立原创实现）。

在 predictions.jsonl 上计算：
  - WER / CER            ：转录部分词/字错误率（编辑距离）
  - 方位角 MAE / 仰角 MAE ：角距离（wrap 到 [-180,180]）
  - 方向 bin EM           ：方位角、仰角同时落在阈值内
  - 联合准确率             ：转录 CER<=阈值 且 方向命中

用法：
    python -m dasr.evaluate \
        --predictions-jsonl runs/dasr/bench/test/predictions.jsonl \
        --qa-root data/qa --split test --output-json metrics.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
from typing import Dict, List, Optional, Tuple

from .prompts import parse_answer, parse_leftright


def levenshtein(a: List[str], b: List[str]) -> int:
    """编辑距离（自写实现）。"""
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cur.append(min(
                prev[j] + 1,
                cur[j - 1] + 1,
                prev[j - 1] + (0 if ca == cb else 1),
            ))
        prev = cur
    return prev[-1]


def word_cer(ref: str, hyp: str) -> float:
    """字错误率（中文逐字）。"""
    if not ref:
        return 0.0 if not hyp else 1.0
    dist = levenshtein(list(ref), list(hyp))
    return dist / len(ref)


def word_wer(ref: str, hyp: str) -> float:
    """词错误率（空白分词）。"""
    r = ref.split()
    h = hyp.split()
    if not r:
        return 0.0 if not h else 1.0
    return levenshtein(r, h) / len(r)


def angular_distance(a: float, b: float) -> float:
    """最小角度差（0-180）。"""
    d = abs(float(a) - float(b)) % 360.0
    return min(d, 360.0 - d)


def load_qa_gold(qa_root: str, split: str) -> Dict[str, Dict]:
    gold = {}
    path = os.path.join(qa_root, f"{split}.jsonl")
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            pid = rec.get("pair_id") or rec.get("audio_path")
            src = (rec.get("source_refs") or [{}])[0]
            gold[pid] = {
                "transcription": rec.get("transcription") or rec.get("canonical_answer", ""),
                "azimuth_deg": src.get("azimuth_deg"),
                "elevation_deg": src.get("elevation_deg"),
                "side": src.get("side"),
            }
    return gold


def evaluate(
    predictions: List[Dict],
    gold: Dict[str, Dict],
    az_threshold: float = 15.0,
    el_threshold: float = 15.0,
    cer_threshold: float = 0.2,
) -> Dict[str, float]:
    """评估。首期左右定向：侧别准确率 side_accuracy；转录 CER/WER。

    若 gold 含方位角/仰角（degree 模式），另算方位/仰角 MAE 与 bin EM。
    联合准确率：转录 CER<=阈值 且 方向判定正确（左右正确 或 方位命中）。
    """
    n = len(predictions)
    if n == 0:
        return {}
    az_errs, el_errs = [], []
    cer_list, wer_list = [], []
    az_em = el_em = lr_correct = joint = lr_total = 0
    for pred in predictions:
        pid = pred.get("pair_id") or pred.get("audio_path")
        g = gold.get(pid, {})
        raw = pred.get("prediction", "")

        # 左右模式优先
        parsed_lr = parse_leftright(raw)
        side_hyp = None
        if parsed_lr is not None:
            trans_hyp, side_hyp = parsed_lr
            az_hyp = el_hyp = None
        else:
            parsed = parse_answer(raw)
            if parsed is None:
                continue
            trans_hyp, az_hyp, el_hyp = parsed

        if g.get("transcription"):
            cer_list.append(word_cer(g["transcription"], trans_hyp))
            wer_list.append(word_wer(g["transcription"], trans_hyp))

        # 左右判定
        if g.get("side") is not None and side_hyp is not None:
            lr_total += 1
            if side_hyp == g["side"]:
                lr_correct += 1

        # degree 指标（扩展模式）
        if az_hyp is not None and g.get("azimuth_deg") is not None:
            d = angular_distance(az_hyp, g["azimuth_deg"])
            az_errs.append(d)
            az_em += 1 if d <= az_threshold else 0
        if el_hyp is not None and g.get("elevation_deg") is not None:
            d = angular_distance(el_hyp, g["elevation_deg"])
            el_errs.append(d)
            el_em += 1 if d <= el_threshold else 0

        # 联合
        if not g.get("transcription"):
            continue
        ok_asr = word_cer(g["transcription"], trans_hyp) <= cer_threshold
        if side_hyp is not None and g.get("side") is not None:
            ok_dir = side_hyp == g["side"]
        elif az_hyp is not None and el_hyp is not None \
                and g.get("azimuth_deg") is not None and g.get("elevation_deg") is not None:
            ok_dir = (
                angular_distance(az_hyp, g["azimuth_deg"]) <= az_threshold
                and angular_distance(el_hyp, g["elevation_deg"]) <= el_threshold
            )
        else:
            ok_dir = False
        if ok_asr and ok_dir:
            joint += 1

    return {
        "samples": n,
        "CER": sum(cer_list) / len(cer_list) if cer_list else None,
        "WER": sum(wer_list) / len(wer_list) if wer_list else None,
        "side_accuracy": (lr_correct / lr_total) if lr_total else None,
        "azimuth_MAE_deg": sum(az_errs) / len(az_errs) if az_errs else None,
        "elevation_MAE_deg": sum(el_errs) / len(el_errs) if el_errs else None,
        "azimuth_EM": (az_em / n) if n else None,
        "elevation_EM": (el_em / n) if n else None,
        "joint_accuracy": joint / n if n else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="定向 ASR 评估")
    parser.add_argument("--predictions-jsonl", required=True)
    parser.add_argument("--qa-root", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-json", default="metrics.json")
    parser.add_argument("--az-threshold", type=float, default=15.0)
    parser.add_argument("--el-threshold", type=float, default=15.0)
    parser.add_argument("--cer-threshold", type=float, default=0.2)
    args = parser.parse_args()

    preds = []
    with open(args.predictions_jsonl, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                preds.append(json.loads(line))
    gold = load_qa_gold(args.qa_root, args.split)
    metrics = evaluate(preds, gold, args.az_threshold, args.el_threshold, args.cer_threshold)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
