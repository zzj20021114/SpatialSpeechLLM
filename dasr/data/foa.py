"""dasr.data.foa — SN3D FOA 编解码工具（独立原创实现）。

约定（与空间编码器一致）：
  - 4 通道 DCASE 波形序 [W, Y, Z, X]（内部重排为 [W, X, Y, Z]）。
  - 方位角：0°=正前，逆时针（俯视）增大，范围 [0, 360)。
  - 仰角：[-90, 90]，向上为正。
  - 归一化：SN3D（W 通道 1/sqrt(2)）。
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np

# 存储序 [W, Y, Z, X]（写出文件时使用）
DCASE_CHANNEL_ORDER = ("W", "Y", "Z", "X")


def encode_point_source(
    speech: np.ndarray,
    azimuth_deg: float,
    elevation_deg: float,
) -> np.ndarray:
    """把单声道声源编码为 FOA [4, T]，DCASE 序 [W, Y, Z, X]。

    speech: [T] float32。
    返回 [4, T]：rows = [W, Y, Z, X]。
    """
    s = np.asarray(speech, dtype=np.float32)
    if s.ndim != 1:
        raise ValueError(f"speech 需为一维，得到 {s.ndim}D")
    az = math.radians(azimuth_deg % 360.0)
    el = math.radians(max(-90.0, min(90.0, elevation_deg)))

    w = s / math.sqrt(2.0)
    x = s * math.cos(el) * math.cos(az)
    y = s * math.cos(el) * math.sin(az)
    z = s * math.sin(el)

    # DCASE 序 [W, Y, Z, X]
    return np.stack([w, y, z, x], axis=0).astype(np.float32, copy=False)


def mix_sources(
    sources: list,
    max_seconds: Optional[float] = None,
    sample_rate: int = 16000,
) -> np.ndarray:
    """叠加多个 (speech, azimuth_deg, elevation_deg) 到同一 FOA 场景。

    sources: [(speech_1d, azimuth_deg, elevation_deg), ...]
    返回 [4, T]（按最长源补齐，可选截断到 max_seconds）。
    """
    if not sources:
        raise ValueError("sources 不能为空")
    foas = [
        encode_point_source(s, az, el) for (s, az, el) in sources
    ]
    max_len = max(f.shape[1] for f in foas)
    if max_seconds is not None:
        max_len = min(max_len, int(max_seconds * sample_rate))
    out = np.zeros((4, max_len), dtype=np.float32)
    for f in foas:
        n = min(f.shape[1], max_len)
        out[:, :n] += f[:, :n]
    return out


# ---------------------------------------------------------------------------
# DOA 估计（用于合成数据的自洽性校验 / 简易基线）
# ---------------------------------------------------------------------------
def estimate_doa(foa: np.ndarray, sample_rate: int = 16000) -> Tuple[float, float]:
    """由 FOA 波形估计 (azimuth_deg, elevation_deg)。

    用短时能量加权的强度向量均值：
      IVx = <W*X>、IVy = <W*Y>、IVz = <W*Z>
    （纯消声点源下等价于声源方向；仅用于校验/基线。）
    """
    foa = np.asarray(foa, dtype=np.float32)
    w, y, z, x = foa[0], foa[1], foa[2], foa[3]
    ivx = float(np.mean(w * x))
    ivy = float(np.mean(w * y))
    ivz = float(np.mean(w * z))
    az = math.degrees(math.atan2(ivy, ivx)) % 360.0
    el = math.degrees(math.atan2(ivz, math.hypot(ivx, ivy)))
    return az, el


# ---------------------------------------------------------------------------
# 读写
# ---------------------------------------------------------------------------
def write_foa_wav(path: str, foa: np.ndarray, sample_rate: int = 16000) -> None:
    """写入 [4, T] FOA（DCASE 序）为 4 通道 wav。"""
    import soundfile as sf
    if foa.ndim != 2 or foa.shape[0] != 4:
        raise ValueError(f"需 [4, T]，得到 {foa.shape}")
    sf.write(path, foa.T, sample_rate, subtype="FLOAT")


def read_foa_wav(path: str, sample_rate: int = 16000) -> np.ndarray:
    """读取 4 通道 wav -> [4, T] float32。"""
    import soundfile as sf
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    if sr != sample_rate:
        from scipy.signal import resample_poly
        gcd = math.gcd(int(sr), int(sample_rate))
        up, down = int(sample_rate) // gcd, int(sr) // gcd
        data = resample_poly(data, up, down, axis=0).astype(np.float32)
    if data.shape[1] != 4:
        raise ValueError(f"需 4 通道 FOA，得到 {data.shape[1]} 通道")
    return data.T
