"""dasr.model.audio_frontend — FOA 音频前端（独立原创实现）。

将 4 通道 FOA 波形（DCASE 序 [W, Y, Z, X]）转为 7 通道空间特征图：

    通道 0..3: W / X / Y / Z 的 log-mel（重排为 [W, X, Y, Z] 后计算）
    通道 4..6: 强度向量 IVx / IVy / IVz = Re(conj(W*) · {X, Y, Z})（逐帧，反映到达方向）

实现细节均为公开信号处理公式，自写代码；不复制任何第三方实现。
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn

from .config import AudioFrontendConfig

# 存储序 [W, Y, Z, X] -> 内部规范序 [W, X, Y, Z]
_DCASE_WYZX_TO_WXYZ = (0, 3, 1, 2)


def build_mel_filterbank(
    sample_rate: int,
    n_fft: int,
    num_mel_bins: int,
    fmin: float = 0.0,
    fmax: Optional[float] = None,
) -> torch.Tensor:
    """HTK 风格 mel 滤波器组，返回 [num_mel_bins, n_fft // 2 + 1]。"""
    if fmax is None:
        fmax = float(sample_rate) / 2.0
    num_freqs = n_fft // 2 + 1

    def hz_to_mel(f):
        return 2595.0 * torch.log10(1.0 + f / 700.0)

    def mel_to_hz(m):
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    min_mel = hz_to_mel(torch.tensor(fmin, dtype=torch.float64))
    max_mel = hz_to_mel(torch.tensor(fmax, dtype=torch.float64))
    mel_points = torch.linspace(min_mel, max_mel, num_mel_bins + 2, dtype=torch.float64)
    hz_points = mel_to_hz(mel_points)
    fft_freqs = torch.linspace(0.0, float(sample_rate) / 2.0, num_freqs, dtype=torch.float64)

    banks = torch.zeros((num_mel_bins, num_freqs), dtype=torch.float32)
    for i in range(num_mel_bins):
        f_l, f_c, f_r = hz_points[i], hz_points[i + 1], hz_points[i + 2]
        lo = (fft_freqs - f_l) / (f_c - f_l)
        hi = (f_r - fft_freqs) / (f_r - f_c)
        banks[i] = torch.clamp(torch.minimum(lo, hi), min=0.0)
    return banks


class AudioFrontend(nn.Module):
    """FOA 波形 -> 7 通道空间特征图。

    forward:
        waveform: [B, 4, T] float32（DCASE 序 [W, Y, Z, X]）
        waveform_lengths: [B] 有效采样数（可选）
    返回:
        features: [B, C, T_f, F]
        feature_lengths: [B] 有效帧数（可选输入时提供）
    """

    def __init__(self, cfg: Optional[AudioFrontendConfig] = None) -> None:
        super().__init__()
        self.cfg = cfg or AudioFrontendConfig()
        c = self.cfg

        assert c.feature_channels == 7, "特征图固定为 7 通道（4 log-mel + 3 IV）"
        self.register_buffer(
            "mel_filterbank",
            build_mel_filterbank(c.sample_rate, c.n_fft, c.num_mel_bins, c.fmin, c.fmax),
            persistent=False,
        )
        self.register_buffer("window", torch.hann_window(c.win_length), persistent=False)
        if not c.normalize_logmel:
            raise ValueError("normalize_logmel 当前必须为 True（使用固定 fbank 均值/方差）")

    # ------------------------------------------------------------------
    def _stft(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, C, T] -> complex [B, C, F, T_f]（center=True, pad 对齐 Whisper 前端）。

        torch.stft 仅接受 1D/2D，批量输入先展平为 [B*C, T]。
        """
        c = self.cfg
        B, C, T = x.shape
        flat = x.reshape(B * C, T)
        spec = torch.stft(
            flat,
            n_fft=c.n_fft,
            hop_length=c.hop_length,
            win_length=c.win_length,
            window=self.window.to(x.device),
            center=True,
            return_complex=True,
        )                                       # [B*C, F, T_f]
        return spec.reshape(B, C, *spec.shape[-2:])

    def _log_mel(self, stft: torch.Tensor) -> torch.Tensor:
        """功率谱 -> log-mel，输入/输出 [B, C, F, T_f]。"""
        c = self.cfg
        pow_spec = stft.real ** 2 + stft.imag ** 2           # [B,C,F,T]
        mel = torch.einsum("qf,bcft->bcqt", self.mel_filterbank.to(pow_spec), pow_spec)
        logmel = (mel.clamp_min(1e-10).log() - c.fbank_mean) / (2.0 * c.fbank_std)
        return logmel                                          # [B,C,M,T]

    def _intensity_vectors(self, stft: torch.Tensor) -> torch.Tensor:
        """IVx/y/z = Re(conj(W) · {X, Y, Z})，返回 [B, 3, M, T_f]（已投影到 mel 域）。

        W 为 stft 通道 0（内部序 [W,X,Y,Z]），X/Y/Z 为通道 1/2/3。
        """
        c = self.cfg
        w = stft[:, 0:1]                                       # [B,1,F,T]
        xyz = stft[:, 1:4]                                     # [B,3,F,T]
        iv = (w.conj() * xyz).real / (w.abs().pow(2).sum(1, keepdim=True) + c.iv_eps)
        iv = torch.nan_to_num(iv, nan=0.0, posinf=0.0, neginf=0.0)
        iv = torch.tanh(iv * 0.5)                              # 限制到 (-1, 1)
        # 投影到 mel 域，与 logmel 的频率维对齐
        iv_mel = torch.einsum("qf,bcft->bcqt", self.mel_filterbank.to(iv), iv)
        return iv_mel

    # ------------------------------------------------------------------
    def forward(
        self,
        waveform: torch.Tensor,
        waveform_lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if waveform.ndim != 3 or waveform.shape[1] != 4:
            raise ValueError(f"waveform 需为 [B,4,T]，得到 {tuple(waveform.shape)}")
        waveform = waveform[:, _DCASE_WYZX_TO_WXYZ, :]          # -> [W,X,Y,Z]
        if waveform_lengths is not None:
            max_len = waveform.shape[-1]
            mask = (
                torch.arange(max_len, device=waveform.device).unsqueeze(0)
                < waveform_lengths.unsqueeze(1)
            )
            waveform = waveform * mask.unsqueeze(1)             # 尾部补零防泄漏

        stft = self._stft(waveform)                             # [B,4,F,T_f]
        logmel = self._log_mel(stft)                            # [B,4,M,T_f]
        iv = self._intensity_vectors(stft)                      # [B,3,F,T_f]
        feats = torch.cat([logmel, iv], dim=1)                  # [B,7,M,T_f]

        if waveform_lengths is not None:
            c = self.cfg
            feature_lengths = torch.floor(
                waveform_lengths.float() / c.hop_length
            ).long().clamp(min=1)
            return feats, feature_lengths
        return feats, None
