"""Speaker enrollment encoder for target-conditioned spatial speech."""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import EnrollmentEncoderConfig


class EnrollmentEncoder(nn.Module):
    """Encode a reference waveform into a normalized speaker embedding.

    This is intentionally a compact trainable encoder for the first TSE
    stage. It can later be replaced or initialized from ECAPA/WeSpeaker
    without changing the target-fusion interface.
    """

    def __init__(self, cfg: Optional[EnrollmentEncoderConfig] = None) -> None:
        super().__init__()
        self.cfg = cfg or EnrollmentEncoderConfig()
        c = self.cfg
        self.register_buffer("window", torch.hann_window(c.win_length), persistent=False)
        self.input_norm = nn.LayerNorm(c.n_fft // 2 + 1)
        self.input_proj = nn.Linear(c.n_fft // 2 + 1, c.hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=c.hidden_dim,
            nhead=c.num_heads,
            dim_feedforward=c.hidden_dim * 4,
            dropout=c.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=c.num_layers)
        self.output = nn.Sequential(
            nn.LayerNorm(c.hidden_dim),
            nn.Linear(c.hidden_dim, c.hidden_dim),
            nn.GELU(),
            nn.Linear(c.hidden_dim, c.output_dim),
        )

    def _features(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.ndim == 3:
            if waveform.shape[1] != 1:
                raise ValueError("enrollment waveform 需为 [B,T] 或 [B,1,T]")
            waveform = waveform[:, 0]
        if waveform.ndim != 2:
            raise ValueError(f"enrollment waveform 需为 [B,T]，得到 {tuple(waveform.shape)}")
        spec = torch.stft(
            waveform,
            n_fft=self.cfg.n_fft,
            hop_length=self.cfg.hop_length,
            win_length=self.cfg.win_length,
            window=self.window.to(waveform.device),
            center=True,
            return_complex=True,
        )
        log_mag = spec.abs().clamp_min(1e-5).log().transpose(1, 2)
        return self.input_proj(self.input_norm(log_mag))

    def forward(
        self,
        waveform: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        frames = self._features(waveform)
        padding_mask = None
        if lengths is not None:
            frame_lengths = torch.ceil(lengths.float() / self.cfg.hop_length).long().clamp(
                min=1, max=frames.shape[1]
            )
            padding_mask = torch.arange(frames.shape[1], device=frames.device).unsqueeze(0) >= frame_lengths.unsqueeze(1)
        encoded = self.encoder(frames, src_key_padding_mask=padding_mask)
        if padding_mask is None:
            pooled = encoded.mean(dim=1)
        else:
            valid = (~padding_mask).unsqueeze(-1).to(encoded.dtype)
            pooled = (encoded * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        return F.normalize(self.output(pooled), dim=-1)
