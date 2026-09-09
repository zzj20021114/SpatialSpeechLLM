"""定向 ASR 提示模板与答案编解码（独立原创实现）。

支持两种方位任务模式：
  - leftright（当前首期）：输出 `<方位> 左` / `<方位> 右`，二分类。
  - degree（扩展）：输出 `<方位> 方位角=xx.x, 仰角=xx.x`，回归方位/仰角。
"""
from __future__ import annotations

import random
import re
from typing import Dict, List, Optional, Tuple

import yaml

DIRECTION_TAG = "<方位>"
LEFT = "左"
RIGHT = "右"

# degree 模式
AZIMUTH_PAT = re.compile(r"方位角\s*=\s*(-?\d+(?:\.\d+)?)")
ELEVATION_PAT = re.compile(r"仰角\s*=\s*(-?\d+(?:\.\d+)?)")
# leftright 模式
LR_PAT = re.compile(r"<方位>\s*[：:，,、]?\s*(左|右)")

TAG_COMBO_TRANSCRIBE_DIRECTION = "<TRANSCRIBE> <DIRECTION>"
TAG_COMBO_DIRECTION = "<DIRECTION>"


def load_prompt_templates(path: str) -> Dict[str, List[str]]:
    """加载提示模板 YAML -> {tag_combo: [templates]}。"""
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"prompt config {path} must be a mapping of tag-combo -> list")
    for combo, templates in data.items():
        if not isinstance(templates, list) or not templates:
            raise ValueError(f"tag combo {combo!r} must map to a non-empty list")
    return data


def get_prompt(templates: Dict[str, List[str]], tag_combo: str, rng: Optional[random.Random] = None) -> str:
    """从 tag_combo 的模板列表中随机采样一条提示。"""
    if tag_combo not in templates:
        raise KeyError(f"unknown tag combo {tag_combo!r}; available: {list(templates)}")
    pool = templates[tag_combo]
    if rng is not None:
        return rng.choice(pool)
    return random.choice(pool)


# ---------------------------------------------------------------------------
# leftright 模式（首期）
# ---------------------------------------------------------------------------
def format_leftright_answer(transcription: str, side: str) -> str:
    """``{transcription} <方位> 左`` 或 ``{transcription} <方位> 右``。"""
    if side not in (LEFT, RIGHT):
        raise ValueError(f"side 必须为 '左' 或 '右'，得到 {side!r}")
    return f"{transcription} {DIRECTION_TAG} {side}"


def parse_leftright(text: str) -> Optional[Tuple[str, str]]:
    """从模型输出解析 (transcription, side)。side ∈ {'左','右'}。"""
    m = LR_PAT.search(text)
    if m is None:
        return None
    side = m.group(1)
    tag_idx = text.find(DIRECTION_TAG)
    transcription = text[:tag_idx].strip() if tag_idx >= 0 else ""
    return transcription, side


# ---------------------------------------------------------------------------
# degree 模式（扩展）
# ---------------------------------------------------------------------------
def format_answer(transcription: str, azimuth_deg: float, elevation_deg: float) -> str:
    """``{transcription} <方位> 方位角={azi:.1f}, 仰角={ele:.1f}``。"""
    return (
        f"{transcription} {DIRECTION_TAG} "
        f"方位角={float(azimuth_deg):.1f}, 仰角={float(elevation_deg):.1f}"
    )


def parse_direction(text: str) -> Optional[Tuple[Optional[float], Optional[float]]]:
    """从文本中解析 (azimuth_deg, elevation_deg)；缺失字段为 None。"""
    azi_match = AZIMUTH_PAT.search(text)
    ele_match = ELEVATION_PAT.search(text)
    if azi_match is None and ele_match is None:
        return None
    azi = float(azi_match.group(1)) if azi_match else None
    ele = float(ele_match.group(1)) if ele_match else None
    return azi, ele


def parse_answer(text: str) -> Optional[Tuple[str, Optional[float], Optional[float]]]:
    """解析模型输出 -> (transcription, azimuth_deg, elevation_deg)。"""
    direction = parse_direction(text)
    if direction is None:
        return None
    tag_idx = text.find(DIRECTION_TAG)
    transcription = text[:tag_idx].strip() if tag_idx >= 0 else text.strip()
    return transcription, direction[0], direction[1]
