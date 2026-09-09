"""dasr.model.seld_heads — SELD 预训练头（独立原创实现，ACCDOA 风格）。

在空间编码器骨干（未降采样，约 12.5 Hz 帧级）上：
  - 对每个 token 帧、每个事件类回归一个 3D 笛卡尔坐标向量 (x, y, z)。
  - 活动 = 向量模长超过阈值；方位角 = atan2(y, x)，仰角 = atan2(z, hypot(x, y))。
损失 = 帧级活动 BCE + 活动帧坐标 MSE。
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import SeldHeadConfig


class AccdoaSeldHead(nn.Module):
    """帧级 ACCDOA 头：token [B, T, D] -> 每类 3D 坐标 [B, T, C, 3]。"""

    def __init__(self, embed_dim: int, cfg: Optional[SeldHeadConfig] = None) -> None:
        super().__init__()
        self.cfg = cfg or SeldHeadConfig()
        self.proj = nn.Linear(embed_dim, self.cfg.num_classes * 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, D] -> [B, T, C, 3]。"""
        B, T, D = x.shape
        return self.proj(x).reshape(B, T, self.cfg.num_classes, 3)

    # ------------------------------------------------------------------
    # 解码：坐标 -> 活动/方位/仰角
    # ------------------------------------------------------------------
    @staticmethod
    def decode(coord: torch.Tensor, threshold: float = 0.5):
        """coord: [B, T, C, 3] -> (activity, azimuth_deg, elevation_deg) 同形状。"""
        magnitude = coord.norm(dim=-1)                       # [B,T,C]
        activity = magnitude >= threshold
        azimuth = torch.atan2(coord[..., 1], coord[..., 0])  # atan2(y, x)
        elevation = torch.atan2(
            coord[..., 2], coord[..., :2].norm(dim=-1).clamp_min(1e-6)
        )
        return activity, azimuth * 180.0 / math.pi, elevation * 180.0 / math.pi


def align_labels_to_token_frames(
    activity: torch.Tensor,
    coords: torch.Tensor,
    label_rate: float,
    token_rate: float,
    label_lengths: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """把 10 Hz 帧级 SELD 标签对齐到 token 帧（token_rate Hz）。

    activity: [B, T_l, C] 0/1；coords: [B, T_l, C, 3]。
    对 token 帧 i 覆盖的标签窗 [i*R, (i+1)*R)：
      活动 = 窗内任一标签活动；坐标 = 窗内活动帧坐标均值（无活动帧为 0）。
    返回 (aligned_activity, aligned_coords, n_tok)。
    """
    R = label_rate / token_rate
    B, T_l, C = activity.shape
    n_tok = int(math.ceil(T_l / R))

    device = activity.device
    idx = (torch.arange(n_tok, device=device, dtype=torch.float32) * R).long().clamp(max=T_l - 1)
    # 每 token 帧的窗口边界 [i*R, (i+1)*R)
    boundaries = torch.linspace(0, n_tok * R, n_tok + 1, device=device)

    out_activity = torch.zeros(B, n_tok, C, device=device, dtype=activity.dtype)
    out_coords = torch.zeros(B, n_tok, C, 3, device=device, dtype=coords.dtype)
    for b in range(B):
        t_valid = T_l
        if label_lengths is not None:
            t_valid = int(label_lengths[b].item())
        for i in range(n_tok):
            lo = int(math.floor(boundaries[i]))
            hi = int(math.ceil(boundaries[i + 1]))
            lo, hi = max(lo, 0), min(hi, t_valid)
            if lo >= hi:
                continue
            win_act = activity[b, lo:hi]                              # [W, C]
            win_coord = coords[b, lo:hi]                              # [W, C, 3]
            act = (win_act.sum(dim=0) > 0).float()                    # [C]
            active_mask = win_act > 0                                 # [W, C]
            # 每个类的活动帧坐标均值（若该类窗内无活动帧则为 0）
            denom = active_mask.sum(dim=0).clamp_min(1)               # [C]
            coord_mean = (win_coord * active_mask.unsqueeze(-1)).sum(dim=0) / denom.unsqueeze(-1)
            coord_mean = coord_mean * act.unsqueeze(-1)
            out_activity[b, i] = act
            out_coords[b, i] = coord_mean
    return out_activity, out_coords, torch.full((B,), n_tok, device=device, dtype=torch.long)


class SeldPretrainModel(nn.Module):
    """空间编码器 + ACCDOA 头 联合，用于 S0 SELD 预训练。"""

    def __init__(
        self,
        spatial_encoder: nn.Module,
        cfg: Optional[SeldHeadConfig] = None,
        frontend: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.spatial_encoder = spatial_encoder
        self.frontend = frontend
        self.head = AccdoaSeldHead(spatial_encoder.cfg.embed_dim, cfg)
        self.cfg = self.head.cfg

    def forward(
        self,
        waveform: torch.Tensor,
        waveform_lengths: torch.Tensor,
        activity: torch.Tensor,
        coords: torch.Tensor,
        label_rate: float = 10.0,
        token_rate: Optional[float] = None,
    ):
        """waveform: [B, 4, T]；activity/coords 为 label_rate Hz 帧级标签。

        若构造时提供了 frontend，则先在内部计算 7 通道特征；否则 waveform 即特征图。
        """
        if self.frontend is not None:
            feats, feature_lengths = self.frontend(waveform, waveform_lengths)
        else:
            feats, feature_lengths = waveform, waveform_lengths
        trunk, n_patch = self.spatial_encoder.forward_trunk(feats, feature_lengths)
        pred = self.head(trunk)                                        # [B, T, C, 3]

        eff_rate = token_rate or self.spatial_encoder.native_rate
        t_act, t_coord, n_tok = align_labels_to_token_frames(
            activity, coords, label_rate, eff_rate
        )

        # 仅监督有效 token 帧
        n_patch_clamped = n_patch.clamp(max=trunk.shape[1])
        valid = (
            torch.arange(trunk.shape[1], device=trunk.device).unsqueeze(0)
            < n_patch_clamped.unsqueeze(1)
        ).unsqueeze(-1)                                                # [B, T, 1]

        pred_act = pred.norm(dim=-1)                                   # [B, T, C]
        act_loss = F.binary_cross_entropy_with_logits(
            pred_act * valid, t_act * valid, reduction="sum"
        ) / valid.sum().clamp_min(1)

        coord_target = t_coord
        active = (t_act > 0).unsqueeze(-1) & valid                    # [B, T, C, 1]
        diff = (pred - coord_target) ** 2
        coord_loss = (diff * active).sum() / active.sum().clamp_min(1)

        loss = (
            self.cfg.activity_weight * act_loss
            + self.cfg.coord_weight * coord_loss * self.cfg.coord_lambda
        )
        return {
            "loss": loss,
            "act_loss": act_loss,
            "coord_loss": coord_loss,
            "pred": pred,
            "activity_target": t_act,
        }
