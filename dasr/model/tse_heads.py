"""Multi-task heads for target extraction, activity, DOA and speaker identity."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import TseHeadConfig


def masked_mean(x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
    if lengths is None:
        return x.mean(dim=1)
    mask = (torch.arange(x.shape[1], device=x.device).unsqueeze(0) < lengths.unsqueeze(1)).to(x.dtype)
    return (x * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1.0)


class TseMaskHead(nn.Module):
    """Predict a complex ratio mask at token rate, expandable to STFT frames."""

    def __init__(self, token_dim: int, freq_bins: int) -> None:
        super().__init__()
        self.freq_bins = freq_bins
        self.proj = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, 2 * freq_bins),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.proj(tokens)

    def upsample_to_frames(self, mask_logits: torch.Tensor, num_frames: int) -> torch.Tensor:
        B, T, C = mask_logits.shape
        if C != 2 * self.freq_bins:
            raise ValueError("mask logits 频率维度不匹配")
        return F.interpolate(
            mask_logits.transpose(1, 2), size=num_frames, mode="linear", align_corners=False
        ).transpose(1, 2).reshape(B, num_frames, 2, self.freq_bins).permute(0, 2, 3, 1)


class TseMultiTaskHeads(nn.Module):
    """Shared target tokens -> TSE/VAD/DOA/speaker/side predictions."""

    def __init__(self, cfg: Optional[TseHeadConfig] = None) -> None:
        super().__init__()
        self.cfg = cfg or TseHeadConfig()
        c = self.cfg
        self.mask = TseMaskHead(c.token_dim, c.mask_freq_bins)
        self.vad = nn.Linear(c.token_dim, 1)
        self.doa = nn.Linear(c.token_dim, 3)
        self.speaker = nn.Sequential(
            nn.LayerNorm(c.token_dim),
            nn.Linear(c.token_dim, c.speaker_embed_dim),
        )
        self.side = nn.Sequential(
            nn.LayerNorm(c.token_dim),
            nn.Linear(c.token_dim, c.num_sides),
        )

    def forward(
        self,
        target_tokens: torch.Tensor,
        token_lengths: Optional[torch.Tensor] = None,
    ):
        pooled = masked_mean(target_tokens, token_lengths)
        return {
            "mask_logits": self.mask(target_tokens),
            "vad_logits": self.vad(target_tokens).squeeze(-1),
            "doa_vectors": self.doa(target_tokens),
            "speaker_embedding": F.normalize(self.speaker(pooled), dim=-1),
            "side_logits": self.side(pooled),
        }
