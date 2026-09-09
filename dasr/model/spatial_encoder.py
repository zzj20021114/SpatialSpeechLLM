"""dasr.model.spatial_encoder — 自建空间编码器（独立原创实现）。

模块组成：
  SpatialPatchEmbed   ：把 7 通道 FOA 特征图 [B,7,M,T_f] 变成 token 序列 [B,T,D]
  TransformerLayer    ：pre-LN 自注意力 + FFN（支持相对位置偏置）
  SpatialEncoder      ：patch 嵌入 -> Transformer 骨干 -> 时间降采样 -> 空间 token

输出 token 率默认 2.5 Hz（20s 音频 -> 50 token），便于以较低上下文开销注入 LLM。
可选以公开 BEATs 权重热启动（维度需一致）；默认从零初始化。
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import SpatialEncoderConfig


# ---------------------------------------------------------------------------
# patch 嵌入
# ---------------------------------------------------------------------------
class SpatialPatchEmbed(nn.Module):
    """Conv2d 将 [B, C, M, T_f] 映射为 [B, T, D] 的 token 序列。

    freq 核覆盖全部 mel bin（patch_freq == M），时间核/步幅决定原生 token 率。
    """

    def __init__(self, cfg: SpatialEncoderConfig) -> None:
        super().__init__()
        self.cfg = cfg
        # 输入 feats: [B, C, M(频率), T_f(时间)]。Conv2d 的 H=频率、W=时间，
        # 因此 kernel/stride 先频率后时间（patch_freq 覆盖全部 mel bin）。
        self.proj = nn.Conv2d(
            cfg.foa_feature_channels,
            cfg.embed_dim,
            kernel_size=(cfg.patch_freq, cfg.patch_time),
            stride=(cfg.stride_freq, cfg.stride_time),
        )

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        # feats: [B, C, M, T_f] -> tokens: [B, T, D]
        x = self.proj(feats)                    # [B, D, 1, T_p]（频率维折叠为 1）
        return x.squeeze(2).transpose(1, 2)     # [B, T_p, D]


# ---------------------------------------------------------------------------
# 自注意力 / Transformer 层
# ---------------------------------------------------------------------------
class MultiHeadSelfAttention(nn.Module):
    """缩放点积自注意力，支持相对位置偏置（T5/BEATs 风格桶，自写实现）。"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        relative_position_buckets: Optional[int] = None,
        max_distance: int = 1280,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim 必须能被 num_heads 整除"

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.scaling = self.head_dim ** -0.5

        self.use_rel_pos = relative_position_buckets is not None and relative_position_buckets > 0
        if self.use_rel_pos:
            self.relative_position_buckets = int(relative_position_buckets)
            self.max_distance = int(max_distance)
            self.rel_pos_bias = nn.Embedding(self.relative_position_buckets, num_heads)

    # -- 相对位置桶（自写，行为对齐通用文献方案）--------------------------
    def _relative_position_bucket(self, rel_pos: torch.Tensor) -> torch.Tensor:
        # rel_pos: 整数相对距离（[0, 2*max_len) 便于无符号索引）
        max_exact = self.relative_position_buckets // 2
        max_rel = self.max_distance
        out = torch.zeros_like(rel_pos, dtype=torch.long)
        is_small = rel_pos < max_exact
        out[is_small] = rel_pos[is_small]
        large = ~is_small
        bucket = torch.log(rel_pos[large].float() / max_exact) / math.log(
            float(max_rel) / float(max_exact)
        ) * (self.relative_position_buckets - max_exact)
        bucket = bucket.clamp(0, self.relative_position_buckets - max_exact - 1)
        out[large] = bucket.long() + max_exact
        return out

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """x: [B, T, D]；key_padding_mask: [B, T]（True=pad）。返回 [B, T, D]。"""
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)      # [B,H,T,hd]
        q = q * self.scaling
        attn = q @ k.transpose(-1, -2)                       # [B,H,T,T]

        if self.use_rel_pos:
            pos = torch.arange(T, device=x.device)
            rel = (pos.unsqueeze(1) - pos.unsqueeze(0)) + (T - 1)   # [T,T] in [0, 2T-2]
            buckets = self._relative_position_bucket(rel)
            bias = self.rel_pos_bias(buckets).permute(2, 0, 1)      # [H,T,T]
            attn = attn + bias.unsqueeze(0)

        if key_padding_mask is not None:
            attn = attn.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
            )
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, T, D)   # [B,T,D]
        return self.out_proj(out)


class TransformerLayer(nn.Module):
    """pre-LN 层：LN -> attn (+residual) -> LN -> FFN (+residual)。"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.1,
        attn_dropout: float = 0.1,
        activation: str = "gelu",
        layer_norm_first: bool = True,
        relative_position_buckets: Optional[int] = None,
        max_distance: int = 1280,
    ) -> None:
        super().__init__()
        self.layer_norm_first = layer_norm_first
        self.self_attn = MultiHeadSelfAttention(
            embed_dim, num_heads, dropout=attn_dropout,
            relative_position_buckets=relative_position_buckets, max_distance=max_distance,
        )
        self.self_attn_layer_norm = nn.LayerNorm(embed_dim)
        self.ffn_layer_norm = nn.LayerNorm(embed_dim)
        self.fc1 = nn.Linear(embed_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = _ACTIVATIONS[activation]
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.layer_norm_first:
            x = x + self.dropout1(self.self_attn(self.self_attn_layer_norm(x), key_padding_mask))
            x = x + self.dropout2(self.fc2(self.dropout(self.activation(self.fc1(self.ffn_layer_norm(x))))))
        else:
            x = self.self_attn_layer_norm(x + self.dropout1(self.self_attn(x, key_padding_mask)))
            x = self.ffn_layer_norm(x + self.dropout2(self.fc2(self.dropout(self.activation(self.fc1(x))))))
        return x


_ACTIVATIONS = {"gelu": F.gelu, "relu": F.relu, "silu": F.silu}


# ---------------------------------------------------------------------------
# 时间降采样器
# ---------------------------------------------------------------------------
class TemporalResampler(nn.Module):
    """把 [B, T_in, D] 降到目标 token 率。

    mode="conv"：1D 卷积 kernel=stride=factor（可学习，感受野对齐）;
    mode="mean"：均匀分组的均值池化（无参，最稳）。
    """

    def __init__(self, embed_dim: int, factor: int, mode: str = "conv") -> None:
        super().__init__()
        assert factor >= 1 and isinstance(factor, int)
        self.factor = factor
        self.mode = mode
        if mode == "conv":
            self.conv = nn.Conv1d(embed_dim, embed_dim, kernel_size=factor, stride=factor)
        elif mode != "mean":
            raise ValueError(f"unknown resample mode {mode}")

    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None):
        """x: [B,T,D]；lengths: [B] 有效 token 数。返回 (x', lengths')。"""
        B, T, D = x.shape
        if self.factor == 1:
            return x, lengths
        pad = (self.factor - T % self.factor) % self.factor
        if pad:
            x = F.pad(x, (0, 0, 0, pad))
            if lengths is not None:
                lengths = lengths + pad
        if self.mode == "mean":
            x = x.reshape(B, -1, self.factor, D).mean(dim=2)
        else:
            x = self.conv(x.transpose(1, 2)).transpose(1, 2)
        new_lengths = None
        if lengths is not None:
            new_lengths = (lengths // self.factor).clamp(min=1)
        return x, new_lengths


# ---------------------------------------------------------------------------
# 空间编码器
# ---------------------------------------------------------------------------
class SpatialEncoder(nn.Module):
    """FOA 特征图 -> 空间 token 序列（约 target_token_rate Hz）。"""

    def __init__(self, cfg: SpatialEncoderConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.patch_embed = SpatialPatchEmbed(cfg)

        native_rate = cfg.encoder_native_rate
        if native_rate is None:
            # 帧率 100Hz（hop=160@16k）/ stride_time
            native_rate = 100.0 / cfg.stride_time
        self.native_rate = float(native_rate)
        factor = round(self.native_rate / cfg.target_token_rate)
        factor = max(1, factor)
        self.resample_factor = factor
        self.effective_rate = self.native_rate / factor

        self.pos_drop = nn.Dropout(cfg.dropout)
        self.layers = nn.ModuleList(
            TransformerLayer(
                embed_dim=cfg.embed_dim,
                num_heads=cfg.num_heads,
                ffn_dim=cfg.ffn_dim,
                dropout=cfg.dropout,
                attn_dropout=cfg.attn_dropout,
                activation=cfg.activation,
                layer_norm_first=cfg.layer_norm_first,
                relative_position_buckets=cfg.num_position_buckets if cfg.relative_position_embeddings else None,
                max_distance=cfg.max_distance,
            )
            for _ in range(cfg.num_layers)
        )
        self.resampler = TemporalResampler(cfg.embed_dim, factor, mode=cfg.resample_mode)

        if cfg.init_backbone_from:
            self._load_backbone(cfg.init_backbone_from)

        if cfg.freeze_backbone:
            for p in self.parameters():
                p.requires_grad = False

    def _load_backbone(self, path: str) -> None:
        """可选：以公开 BEATs 骨干权重热启动（严格=False，仅匹配维度一致的键）。"""
        try:
            state = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as exc:  # pragma: no cover
            raise OSError(f"无法加载骨干权重 {path}: {exc}") from exc
        trunk = state.get("model", state) if isinstance(state, dict) else state
        missing, unexpected = self.load_state_dict(trunk, strict=False)
        print(f"[SpatialEncoder] init_backbone: missing={len(missing)} unexpected={len(unexpected)}")

    # ------------------------------------------------------------------
    def forward_trunk(
        self,
        feats: torch.Tensor,
        feature_lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """编码器骨干（patch -> Transformer，未降采样）输出 [B, T_p, D]。

        供 SELD 预训练在帧级监督（约 12.5 Hz）上使用。
        """
        x = self.patch_embed(feats)                       # [B, T_p, D]
        B, T, D = x.shape

        mask = None
        if feature_lengths is not None:
            n_patch = torch.ceil(feature_lengths.float() / self.cfg.stride_time).long().clamp(
                min=1, max=T
            )
            mask = torch.arange(T, device=x.device).unsqueeze(0) >= n_patch.unsqueeze(1)
        else:
            n_patch = None

        x = self.pos_drop(x)
        for layer in self.layers:
            x = layer(x, key_padding_mask=mask)
        return x, n_patch

    def forward(
        self,
        feats: torch.Tensor,
        feature_lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """feats: [B, C, M, T_f] -> spatial tokens [B, T_s, D]。

        返回 (tokens, token_lengths)。token_lengths 对应降采样后的有效 token 数。
        """
        x, n_patch = self.forward_trunk(feats, feature_lengths)
        if n_patch is not None:
            tokens, token_lengths = self.resampler(x, n_patch)
            return tokens, token_lengths
        tokens, _ = self.resampler(x, None)
        return tokens, None

    def compute_token_lengths(self, feature_lengths: torch.Tensor) -> torch.Tensor:
        """帧长 -> 降采样后 token 长（供占位符展开对齐）。"""
        n_patch = torch.ceil(
            feature_lengths.float() / self.cfg.stride_time
        ).long().clamp(min=1)
        return (n_patch // self.resample_factor).clamp(min=1)
