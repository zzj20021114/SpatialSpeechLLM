"""Differentiable losses for the first target-conditioned TSE stage."""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F


def si_sdr(
    estimate: torch.Tensor,
    target: torch.Tensor,
    lengths: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Return scale-invariant SDR per sample for [B,T] waveforms."""
    if estimate.shape != target.shape or estimate.ndim != 2:
        raise ValueError("estimate 和 target 必须同形状 [B,T]")
    B, T = target.shape
    if lengths is None:
        mask = torch.ones(B, T, device=target.device, dtype=target.dtype)
    else:
        mask = (torch.arange(T, device=target.device).unsqueeze(0) < lengths.unsqueeze(1)).to(target.dtype)
    target = target * mask
    estimate = estimate * mask
    target_energy = (target * target).sum(dim=1, keepdim=True).clamp_min(eps)
    scale = (estimate * target).sum(dim=1, keepdim=True) / target_energy
    projection = scale * target
    residual = estimate - projection
    ratio = projection.pow(2).sum(dim=1) / residual.pow(2).sum(dim=1).clamp_min(eps)
    return 10.0 * torch.log10(ratio.clamp_min(eps))


def si_sdr_loss(estimate: torch.Tensor, target: torch.Tensor, lengths=None) -> torch.Tensor:
    return -si_sdr(estimate, target, lengths).mean()


def speaker_consistency_loss(
    estimated_embedding: torch.Tensor, enrollment_embedding: torch.Tensor
) -> torch.Tensor:
    return (1.0 - F.cosine_similarity(estimated_embedding, enrollment_embedding, dim=-1)).mean()


def vad_loss(logits: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    if mask is not None:
        loss = loss * mask.to(loss.dtype)
        return loss.sum() / mask.sum().clamp_min(1)
    return loss.mean()


def doa_loss(pred_vectors: torch.Tensor, target_vectors: torch.Tensor, active: Optional[torch.Tensor] = None) -> torch.Tensor:
    loss = F.smooth_l1_loss(pred_vectors, target_vectors, reduction="none").mean(dim=-1)
    if active is not None:
        loss = loss * active.to(loss.dtype)
        return loss.sum() / active.sum().clamp_min(1)
    return loss.mean()


def side_loss(logits: torch.Tensor, target_side: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, target_side.long())


def combine_tse_losses(
    losses: Dict[str, torch.Tensor],
    weights: Optional[Dict[str, float]] = None,
) -> torch.Tensor:
    weights = weights or {}
    total = None
    for name, value in losses.items():
        weighted = value * float(weights.get(name, 1.0))
        total = weighted if total is None else total + weighted
    if total is None:
        raise ValueError("losses 不能为空")
    return total
