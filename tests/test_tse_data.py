"""Tests for the first ideal TSE data contract."""
import os
import sys
import tempfile

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dasr.data.generate_tse_mixtures import (  # noqa: E402
    infer_speaker_id,
    make_tse_scene,
    side_from_azimuth,
)
from dasr.data.build_speech_manifest import _from_wav_scp_global_transcript  # noqa: E402
from dasr.data.tse_dataset import TseDataset  # noqa: E402


def _write(path, signal, sample_rate=16000):
    sf.write(path, signal.astype(np.float32), sample_rate, subtype="FLOAT")


def test_speaker_id_and_side_contract():
    assert infer_speaker_id({"key": "BAC009S0002W0122", "wav": "x.wav"}) == "S0002"
    assert side_from_azimuth(90.0) == "左"
    assert side_from_azimuth(270.0) == "右"


def test_ideal_tse_scene_is_additive_and_target_conditioned():
    sr = 16000
    t = np.arange(sr, dtype=np.float32) / sr
    target = np.sin(2 * np.pi * 440.0 * t).astype(np.float32)
    interferer = np.sin(2 * np.pi * 880.0 * t[: int(sr * 0.75)]).astype(np.float32)
    enrollment = np.sin(2 * np.pi * 330.0 * t[: int(sr * 0.5)]).astype(np.float32)

    with tempfile.TemporaryDirectory() as tmp:
        target_path = os.path.join(tmp, "target.wav")
        interferer_path = os.path.join(tmp, "interferer.wav")
        enrollment_path = os.path.join(tmp, "enrollment.wav")
        _write(target_path, target)
        _write(interferer_path, interferer)
        _write(enrollment_path, enrollment)

        target_rec = {"key": "BAC009S0002W0001", "wav": target_path, "txt": "目标文本"}
        interferer_rec = {"key": "BAC009S0003W0001", "wav": interferer_path, "txt": "干扰文本"}
        enrollment_rec = {"key": "BAC009S0002W0002", "wav": enrollment_path, "txt": "参考文本"}
        scene = make_tse_scene(
            target_rec,
            interferer_rec,
            enrollment_rec,
            rng=__import__("random").Random(7),
            distance_range=(1.0, 1.0),
            elevation_range=(0.0, 0.0),
        )

    assert scene["mixture_foa"].shape == (4, sr)
    assert scene["target_foa"].shape == scene["mixture_foa"].shape
    assert scene["interferer_foa"].shape == scene["mixture_foa"].shape
    assert np.allclose(
        scene["mixture_foa"], scene["target_foa"] + scene["interferer_foa"], atol=1e-6
    )
    assert scene["target"]["speaker_id"] == scene["enrollment"]["speaker_id"]
    assert scene["target"]["speaker_id"] != scene["interferer"]["speaker_id"]
    assert scene["target"]["side"] in ("左", "右")


def test_wav_scp_global_transcript_pairing():
    with tempfile.TemporaryDirectory() as tmp:
        wav_scp = os.path.join(tmp, "train.scp")
        transcript = os.path.join(tmp, "transcript.txt")
        with open(wav_scp, "w", encoding="utf-8") as f:
            f.write("utt_a /old-machine/utt_a.wav\nutt_b /audio/utt_b.wav\n")
        with open(transcript, "w", encoding="utf-8") as f:
            f.write("utt_a 你好 世界\nutt_missing 不应输出\n")

        records = _from_wav_scp_global_transcript(
            wav_scp, transcript, "zh", concat_chinese=True, wav_root="/local/audio"
        )

    assert records == [{
        "key": "utt_a",
        "wav": "/local/audio/utt_a.wav",
        "txt": "你好世界",
        "lang": "zh",
    }]


def test_tse_dataset_resolves_manifest_audio():
    with tempfile.TemporaryDirectory() as tmp:
        mixture = np.zeros((4, 320), dtype=np.float32)
        mono = np.ones(320, dtype=np.float32)
        for name, data in (
            ("mixture.wav", mixture.T),
            ("target.wav", mono),
            ("interferer.wav", mono),
            ("enrollment.wav", mono[:160]),
        ):
            _write(os.path.join(tmp, name), data)
        manifest = os.path.join(tmp, "manifest.jsonl")
        record = {
            "pair_id": "tse_test_0",
            "mixture_audio": "mixture.wav",
            "target_clean_audio": "target.wav",
            "interferer_audio": "interferer.wav",
            "enrollment_audio": "enrollment.wav",
            "target_side": "左",
            "target_azimuth_deg": 90.0,
            "target_elevation_deg": 0.0,
            "target_speaker_id": "spk_a",
            "target_transcription": "测试",
            "source_refs": [
                {"role": "target", "active_samples": 280},
                {"role": "interferer", "active_samples": 320},
            ],
        }
        with open(manifest, "w", encoding="utf-8") as f:
            f.write(__import__("json").dumps(record, ensure_ascii=False) + "\n")

        dataset = TseDataset(manifest, verify_audio=True)
        item = dataset[0]

    assert len(dataset) == 1
    assert item["pair_id"] == "tse_test_0"
    assert os.path.isabs(item["mixture_audio"])
    assert item["target_side"] == "左"
