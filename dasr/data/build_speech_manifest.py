"""dasr.data.build_speech_manifest — 语音语料 -> speech manifest（独立原创实现）。

把任意"单声道 wav + 文本转录"语料转成统一 manifest，供合成 FOA 定向语音使用。

支持两种输入：
  1) 目录扫描（--dir）：遍历子目录，为每个 .wav 找同名 .txt（AISHELL / LibriSpeech 风格）。
  2) kaldi 风格（--wav-scp + --text）：
        wav.scp:  <key> <path>
        text:     <key> <transcription...>

用法：
    python -m dasr.data.build_speech_manifest --dir /data/aishell1 \
        --output data/speech/train.jsonl --lang zh
    python -m dasr.data.build_speech_manifest --wav-scp wav.scp --text text \
        --output data/speech/train.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional


def _scan_dir(root: str, wav_ext: str, txt_ext: str, lang: str,
              concat_chinese: bool = False) -> List[Dict]:
    records: List[Dict] = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if not fn.endswith(wav_ext):
                continue
            stem = fn[: -len(wav_ext)]
            txt_path = os.path.join(dirpath, stem + txt_ext)
            if not os.path.exists(txt_path):
                print(f"[warn] 缺转录: {txt_path}")
                continue
            with open(txt_path, encoding="utf-8") as f:
                txt = f.read().strip()
            if not txt:
                print(f"[warn] 空转录: {txt_path}")
                continue
            if concat_chinese:
                txt = txt.replace(" ", "").replace("\t", "")
            records.append({
                "key": stem,
                "wav": os.path.join(dirpath, fn),
                "txt": txt,
                "lang": lang,
            })
    return records


def _from_kaldi(wav_scp: str, text: str, lang: str, concat_chinese: bool = False) -> List[Dict]:
    wav_map: Dict[str, str] = {}
    with open(wav_scp, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                wav_map[parts[0]] = parts[1]
    records: List[Dict] = []
    with open(text, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) != 2 or parts[0] not in wav_map:
                continue
            txt = parts[1].strip()
            if concat_chinese:
                txt = txt.replace(" ", "").replace("\t", "")
            records.append({
                "key": parts[0],
                "wav": wav_map[parts[0]],
                "txt": txt,
                "lang": lang,
            })
    return records


def _from_global_transcript(root: str, transcript: str, lang: str,
                            concat_chinese: bool = False) -> List[Dict]:
    """AISHELL 风格：`--dir` 下所有 wav + 一个全局转录文件（`key 文本`）。"""
    key_to_wav: Dict[str, str] = {}
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.endswith(".wav"):
                key_to_wav.setdefault(fn[:-4], os.path.join(dirpath, fn))
    records: List[Dict] = []
    no_wav = 0
    with open(transcript, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) != 2:
                continue
            key, txt = parts[0], parts[1].strip()
            if key not in key_to_wav:
                no_wav += 1
                continue
            if concat_chinese:
                txt = txt.replace(" ", "").replace("\t", "")
            if not txt:
                continue
            records.append({"key": key, "wav": key_to_wav[key], "txt": txt, "lang": lang})
    orphan_wavs = len(key_to_wav) - len(records)
    print(f"[pairing] transcript={len(records) + no_wav} wav={len(key_to_wav)} "
          f"matched={len(records)} transcript_no_wav={no_wav} wav_no_transcript={orphan_wavs}")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="构建语音 manifest")
    parser.add_argument("--dir", default=None, help="语料根目录（目录扫描模式）")
    parser.add_argument("--transcript", default=None, help="全局转录文件（key 文本，AISHELL 风格）")
    parser.add_argument("--wav-scp", default=None, help="kaldi 风格 wav.scp")
    parser.add_argument("--text", default=None, help="kaldi 风格 text")
    parser.add_argument("--wav-ext", default=".wav")
    parser.add_argument("--txt-ext", default=".txt")
    parser.add_argument("--lang", default="zh", choices=["zh", "en", "other"])
    parser.add_argument("--concat-chinese", action="store_true", default=True,
                        help="去掉转录中的词间空格（AISHELL 为空格分词，转连续中文字符利于 CER）")
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-duration-s", type=float, default=0.0)
    parser.add_argument("--max-duration-s", type=float, default=20.0)
    args = parser.parse_args()

    if args.dir and args.transcript:
        records = _from_global_transcript(args.dir, args.transcript, args.lang,
                                          concat_chinese=args.concat_chinese)
    elif args.dir:
        records = _scan_dir(args.dir, args.wav_ext, args.txt_ext, args.lang,
                            concat_chinese=args.concat_chinese)
    elif args.wav_scp and args.text:
        records = _from_kaldi(args.wav_scp, args.text, args.lang,
                              concat_chinese=args.concat_chinese)
    else:
        raise SystemExit("必须提供 --dir+--transcript、--dir 或 --wav-scp/--text")

    # 时长过滤（用 wave/soundfile 读取头部）
    if args.min_duration_s > 0 or args.max_duration_s < 1e9:
        import wave
        kept: List[Dict] = []
        for r in records:
            try:
                with wave.open(r["wav"], "rb") as w:
                    dur = w.getnframes() / float(w.getframerate() or 1)
            except Exception:
                dur = -1.0
            if args.min_duration_s <= dur <= args.max_duration_s:
                kept.append(r)
            else:
                print(f"[drop] {r['key']} dur={dur:.2f}s")
        records = kept

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"完成：{len(records)} 条 -> {args.output}")


if __name__ == "__main__":
    main()
