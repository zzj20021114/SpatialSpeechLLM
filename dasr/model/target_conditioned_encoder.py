"""Compose the spatial encoder, enrollment encoder and target fusion."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class TargetConditionedSpatialEncoder(nn.Module):
    """Produce target-speaker tokens from FOA features and enrollment audio."""

    def __init__(self, spatial_encoder: nn.Module, enrollment_encoder: nn.Module, target_fusion: nn.Module) -> None:
        super().__init__()
        self.spatial_encoder = spatial_encoder
        self.enrollment_encoder = enrollment_encoder
        self.target_fusion = target_fusion

    def forward(
        self,
        spatial_features: torch.Tensor,
        spatial_feature_lengths: Optional[torch.Tensor],
        enrollment_audio: torch.Tensor,
        enrollment_lengths: Optional[torch.Tensor] = None,
    ):
        spatial_tokens, token_lengths = self.spatial_encoder(
            spatial_features, spatial_feature_lengths
        )
        speaker_embedding = self.enrollment_encoder(enrollment_audio, enrollment_lengths)
        target_tokens, target_gate = self.target_fusion(
            spatial_tokens, speaker_embedding, token_lengths
        )
        return {
            "target_tokens": target_tokens,
            "token_lengths": token_lengths,
            "speaker_embedding": speaker_embedding,
            "target_gate": target_gate,
        }
