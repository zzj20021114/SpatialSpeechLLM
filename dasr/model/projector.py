"""dasr.model.projector — pixel-shuffle MLP 投影器（独立原创实现）。

把空间 token [B, T_s, D_in] 投影到 LLM 隐空间 [B, T_s/k, D_out]：
  1) 沿时间轴每 k 个 token 拼成一个（特征维变为 k*D_in）；
  2) LN -> Linear(k*D_in) -> GELU -> Linear(D_out) -> LN。
token 数降为原来的 1/k，节省 LLM 上下文。
"""
from __future__ import annotations

from typing import Optional

import torch.nn as nn
import torch.nn.functional as F

from .config import ProjectorConfig


class PixelShuffleProjector(nn.Module):
    def __init__(self, cfg: Optional[ProjectorConfig] = None) -> None:
        super().__init__()
        self.cfg = cfg or ProjectorConfig()
        c = self.cfg
        k = int(c.shuffle_factor)
        if k < 1:
            raise ValueError(f"shuffle_factor 必须 >= 1，得到 {k}")
        self.shuffle_factor = k
        self.input_dim = int(c.input_dim)
        self.output_dim = int(c.output_dim)

        self.pre_norm = nn.LayerNorm(c.input_dim * k) if c.pre_norm else nn.Identity()
        self.fc1 = nn.Linear(c.input_dim * k, c.hidden_dim)
        self.act = {"gelu": nn.GELU(), "relu": nn.ReLU()}[c.activation]
        self.fc2 = nn.Linear(c.hidden_dim, c.output_dim)
        self.post_norm = nn.LayerNorm(c.output_dim)

    def forward(self, spatial_tokens):
        """spatial_tokens: [B, T_s, D_in] -> [B, T_s/k, D_out]。"""
        B, T, D = spatial_tokens.shape
        if D != self.input_dim:
            raise ValueError(f"期望输入维度 {self.input_dim}，得到 {D}")
        k = self.shuffle_factor
        if k > 1:
            T_trunc = T - (T % k)
            if T_trunc == 0:
                spatial_tokens = F.pad(spatial_tokens, (0, 0, 0, k - T))
                T_trunc = k
            else:
                spatial_tokens = spatial_tokens[:, :T_trunc]
            spatial_tokens = spatial_tokens.reshape(B, T_trunc // k, D * k)
        x = self.pre_norm(spatial_tokens)
        x = self.act(self.fc1(x))
        return self.post_norm(self.fc2(x))
