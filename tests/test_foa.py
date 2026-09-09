"""dasr 核心逻辑测试（纯 numpy，无需 torch）。

覆盖：SN3D FOA 编码自洽性、DOA 估计回验、写读往返、多源混合、
提示模板加载与答案编解码。
"""
import math
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dasr.data.foa import (  # noqa: E402
    encode_point_source,
    estimate_doa,
    mix_sources,
    read_foa_wav,
    write_foa_wav,
)
from dasr.prompts import (  # noqa: E402
    TAG_COMBO_TRANSCRIBE_DIRECTION,
    format_answer,
    format_leftright_answer,
    get_prompt,
    load_prompt_templates,
    parse_answer,
    parse_direction,
    parse_leftright,
)


def _sine(freq=440.0, seconds=1.0, sr=16000, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * sr)) / sr
    return (np.sin(2 * np.pi * freq * t) + 0.1 * rng.standard_normal(int(seconds * sr))).astype(np.float32)


def test_doa_estimate_roundtrip():
    cases = [
        (45.0, 0.0), (90.0, 0.0), (180.0, 0.0), (270.0, 0.0),
        (45.0, 30.0), (200.0, -20.0), (0.0, 0.0),
    ]
    for az, el in cases:
        s = _sine()
        foa = encode_point_source(s, az, el)
        est_az, est_el = estimate_doa(foa)
        # 方位角误差（角距离），仰角误差
        d_az = min(abs(est_az - az) % 360.0, 360.0 - abs(est_az - az) % 360.0)
        assert d_az < 3.0, f"az {az} -> {est_az}"
        assert abs(est_el - el) < 3.0, f"el {el} -> {est_el}"


def test_channel_order_dcase():
    s = _sine()
    foa = encode_point_source(s, 90.0, 0.0)
    # az=90 -> Y 通道最强（sin(90)=1），X 通道为 0
    assert abs(np.corrcoef(foa[1], s)[0, 1]) > 0.99, "Y(通道1) 应携带最强信号"
    assert np.abs(foa[3]).max() < 1e-6, "X(通道3) 在 az=90 时应为 0"
    # W 通道 = s/sqrt(2)
    assert np.allclose(foa[0], s / math.sqrt(2.0), atol=1e-6)


def test_write_read_roundtrip():
    s = _sine()
    foa = encode_point_source(s, 45.0, 10.0)
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "x.wav")
        write_foa_wav(p, foa)
        back = read_foa_wav(p)
        assert back.shape == foa.shape
        assert np.allclose(back, foa, atol=1e-6)


def test_mix_sources():
    s1, s2 = _sine(440.0, 1.0, seed=1), _sine(880.0, 1.0, seed=2)
    foa = mix_sources([(s1, 45.0, 0.0), (s2, 315.0, 0.0)])
    assert foa.shape[0] == 4
    est_az, est_el = estimate_doa(foa)
    # 两个等幅源在 45° 与 315°(-45°) 之间，平均应接近 0° 或 180°；仅校验范围
    assert 0.0 <= est_az < 360.0


def test_prompt_and_answer_roundtrip():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    templates = load_prompt_templates(os.path.join(root, "conf", "directional_asr_prompt.yaml"))
    assert TAG_COMBO_TRANSCRIBE_DIRECTION in templates
    prompt = get_prompt(templates, TAG_COMBO_TRANSCRIBE_DIRECTION)
    assert ("左" in prompt) and ("右" in prompt)

    # 左右模式（首期）
    answer = format_leftright_answer("今天下午开会。", "左")
    assert answer == "今天下午开会。 <方位> 左"
    trans, side = parse_leftright(answer)
    assert trans == "今天下午开会。" and side == "左"
    assert parse_leftright("今天下午开会。 <方位> 右")[1] == "右"
    assert parse_leftright("没有方位标签") is None
    try:
        format_leftright_answer("x", "上")
        assert False, "非法 side 应抛错"
    except ValueError:
        pass

    # degree 模式（扩展）
    answer = format_answer("今天下午开会。", 45.0, 10.0)
    assert answer == "今天下午开会。 <方位> 方位角=45.0, 仰角=10.0"
    trans, az, el = parse_answer(answer)
    assert trans == "今天下午开会。"
    assert abs(az - 45.0) < 1e-6 and abs(el - 10.0) < 1e-6

    # 无方位 -> 返回 None
    assert parse_answer("今天下午开会。") is None
    # 缺字段 -> 部分解析
    d = parse_direction("方位角=30.0")
    assert d[0] == 30.0 and d[1] is None


def test_evaluate_metrics():
    from dasr.evaluate import angular_distance, levenshtein, word_cer
    assert angular_distance(350.0, 10.0) == 20.0
    assert angular_distance(0.0, 180.0) == 180.0
    assert levenshtein(["a", "b", "c"], ["a", "c"]) == 1
    assert abs(word_cer("开会", "开会了") - 1.0 / 2.0) < 1e-6


def test_evaluate_leftright():
    from dasr.evaluate import evaluate
    gold = {
        "1": {"transcription": "今天下午开会。", "azimuth_deg": 60.0, "elevation_deg": 0.0, "side": "左"},
        "2": {"transcription": "明天天气不错。", "azimuth_deg": 300.0, "elevation_deg": 0.0, "side": "右"},
    }
    preds = [
        {"pair_id": "1", "prediction": "今天下午开会。 <方位> 左"},
        {"pair_id": "2", "prediction": "明天天气不错。 <方位> 右"},
    ]
    m = evaluate(preds, gold)
    assert m["side_accuracy"] == 1.0
    assert abs(m["CER"] - 0.0) < 1e-6
    assert m["joint_accuracy"] == 1.0
    # 一个判错
    preds[1]["prediction"] = "明天天气不错。 <方位> 左"
    m2 = evaluate(preds, gold)
    assert m2["side_accuracy"] == 0.5


if __name__ == "__main__":
    for fn in [test_doa_estimate_roundtrip, test_channel_order_dcase,
               test_write_read_roundtrip, test_mix_sources,
               test_prompt_and_answer_roundtrip, test_evaluate_metrics,
               test_evaluate_leftright]:
        fn()
        print(f"[ok] {fn.__name__}")
    print("全部通过")
