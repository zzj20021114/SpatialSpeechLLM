"""Enrollment-conditioned fusion for selecting a target speaker."""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import TargetFusionConfig


class TargetFusion(nn.Module):
    """Fuse speaker identity into spatial tokens with attention and gating.

    The similarity gate exposes a lightweight target-selection signal while
    cross-attention supplies a learned global target context. Both are kept
    in the token space so TSE and ASR can share the same representation.
    """

    def __init__(self, cfg: Optional[TargetFusionConfig] = None) -> None:
        super().__init__()
        self.cfg = cfg or TargetFusionConfig()
        c = self.cfg
        if c.token_dim % c.num_heads != 0:
            raise ValueError("token_dim 必须能被 num_heads 整除")
        self.token_norm = nn.LayerNorm(c.token_dim)
        self.speaker_proj = nn.Linear(c.speaker_dim, c.token_dim)
        self.cross_attention = nn.MultiheadAttention(
            c.token_dim, c.num_heads, dropout=c.dropout, batch_first=True
        )
        self.film = nn.Linear(c.speaker_dim, 2 * c.token_dim)
        self.fuse = nn.Sequential(
            nn.Linear(2 * c.token_dim + 1, c.token_dim),
            nn.GELU(),
            nn.Dropout(c.dropout),
            nn.Linear(c.token_dim, c.token_dim),
        )
        self.output_norm = nn.LayerNorm(c.token_dim)

    def forward(
        self,
        spatial_tokens: torch.Tensor,
        speaker_embedding: torch.Tensor,
        token_lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if spatial_tokens.ndim != 3 or speaker_embedding.ndim != 2:
            raise ValueError("spatial_tokens 需 [B,T,D]，speaker_embedding 需 [B,E]")
        B, T, D = spatial_tokens.shape
        if speaker_embedding.shape[0] != B or D != self.cfg.token_dim:
            raise ValueError("target fusion 的 batch 或 token 维度不匹配")

        key_padding_mask = None
        if token_lengths is not None:
            key_padding_mask = torch.arange(T, device=spatial_tokens.device).unsqueeze(0) >= token_lengths.unsqueeze(1)

        speaker_token = self.speaker_proj(speaker_embedding).unsqueeze(1)
        context, attention = self.cross_attention(
            speaker_token,
            self.token_norm(spatial_tokens),
            self.token_norm(spatial_tokens),
            key_padding_mask=key_padding_mask,
            need_weights=True,
        )
        context = context.expand(-1, T, -1)

        token_unit = F.normalize(self.token_norm(spatial_tokens), dim=-1)
        speaker_unit = F.normalize(speaker_token.squeeze(1), dim=-1)
        similarity = torch.einsum("btd,bd->bt", token_unit, speaker_unit)
        gate = torch.sigmoid(similarity * self.cfg.similarity_temperature)
        if key_padding_mask is not None:
            gate = gate.masked_fill(key_padding_mask, 0.0)

        scale, bias = self.film(speaker_embedding).chunk(2, dim=-1)
        modulated = spatial_tokens * (1.0 + 0.1 * torch.tanh(scale).unsqueeze(1))
        modulated = modulated + 0.1 * bias.unsqueeze(1)
        fused = self.fuse(torch.cat([modulated, context, gate.unsqueeze(-1)], dim=-1))
        return self.output_norm(spatial_tokens + fused), gate
