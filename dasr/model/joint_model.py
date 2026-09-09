"""Joint target-conditioned TSE and ASR representation model.

The model exposes a frozen-Qwen-compatible projected representation, but the
TSE/VAD/DOA/speaker objectives all branch from the same target tokens. It does
not feed a hard separated waveform into ASR, avoiding cascade error propagation.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .audio_frontend import AudioFrontend
from .config import DasrConfig
from .enrollment_encoder import EnrollmentEncoder
from .projector import PixelShuffleProjector
from .spatial_encoder import SpatialEncoder
from .target_conditioned_encoder import TargetConditionedSpatialEncoder
from .target_fusion import TargetFusion
from .tse_heads import TseMultiTaskHeads
from .tse_losses import (
    doa_loss,
    doa_targets_from_angles,
    masked_mean,
    side_loss,
    si_sdr_loss,
    speaker_consistency_loss,
    vad_loss,
    complex_mask_to_waveform,
)


class JointTargetSpeechModel(nn.Module):
    """Shared target representation with parallel TSE and frozen-LLM hooks."""

    def __init__(
        self,
        cfg: DasrConfig,
        decoder: Optional[nn.Module] = None,
        tokenizer: Any = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.frontend = AudioFrontend(cfg.audio_frontend)
        spatial_encoder = SpatialEncoder(cfg.spatial_encoder)
        enrollment_encoder = EnrollmentEncoder(cfg.enrollment_encoder)
        target_fusion = TargetFusion(cfg.target_fusion)
        self.target_encoder = TargetConditionedSpatialEncoder(
            spatial_encoder, enrollment_encoder, target_fusion
        )
        self.tse_heads = TseMultiTaskHeads(cfg.tse_head)
        self.projector = PixelShuffleProjector(cfg.projector)
        self.decoder = decoder
        self.tokenizer = tokenizer

    def encode_target(
        self,
        mixture_audio: torch.Tensor,
        enrollment_audio: torch.Tensor,
        mixture_lengths: Optional[torch.Tensor] = None,
        enrollment_lengths: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        features, feature_lengths = self.frontend(mixture_audio, mixture_lengths)
        encoded = self.target_encoder(
            features,
            feature_lengths,
            enrollment_audio,
            enrollment_lengths,
        )
        predictions = self.tse_heads(
            encoded["target_tokens"], encoded["token_lengths"]
        )
        projected = self.projector(encoded["target_tokens"])
        return {
            **encoded,
            "target_speaker_embedding": predictions["speaker_embedding"],
            "projected_target_tokens": projected,
            "mask_logits": predictions["mask_logits"],
            "vad_logits": predictions["vad_logits"],
            "doa_vectors": predictions["doa_vectors"],
            "side_logits": predictions["side_logits"],
        }

    def forward(
        self,
        mixture_audio: torch.Tensor,
        enrollment_audio: torch.Tensor,
        mixture_lengths: Optional[torch.Tensor] = None,
        enrollment_lengths: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        outputs = self.encode_target(
            mixture_audio,
            enrollment_audio,
            mixture_lengths,
            enrollment_lengths,
        )
        if input_ids is not None:
            if self.decoder is None:
                raise RuntimeError("提供 input_ids 时必须注入 decoder")
            outputs["asr_output"] = self._decode_target_tokens(
                outputs["projected_target_tokens"],
                outputs["token_lengths"],
                input_ids,
                attention_mask,
                labels,
                **kwargs,
            )
        return outputs

    def _decode_target_tokens(
        self,
        projected_tokens: torch.Tensor,
        token_lengths: Optional[torch.Tensor],
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        labels: Optional[torch.Tensor],
        **kwargs: Any,
    ) -> Any:
        decoder_device = next(self.decoder.get_input_embeddings().parameters()).device
        input_ids = input_ids.to(decoder_device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(decoder_device)
        if labels is not None:
            labels = labels.to(decoder_device)
        inputs_embeds = self.decoder.get_input_embeddings()(input_ids)
        speech_token_id = getattr(self.decoder.config, "speech_token_id", None)
        if speech_token_id is None:
            raise RuntimeError("decoder.config.speech_token_id 尚未设置")
        placeholder_mask = input_ids == int(speech_token_id)
        if token_lengths is None:
            token_lengths = torch.full(
                (projected_tokens.shape[0],), projected_tokens.shape[1],
                dtype=torch.long, device=projected_tokens.device,
            )
        flattened = torch.cat([
            projected_tokens[i, : int(token_lengths[i].item())]
            for i in range(projected_tokens.shape[0])
        ], dim=0).to(decoder_device)
        n_placeholders = int(placeholder_mask.sum().item())
        if n_placeholders != flattened.shape[0]:
            raise ValueError(
                f"speech placeholder 数量 {n_placeholders} 与 target token 数量 "
                f"{flattened.shape[0]} 不一致"
            )
        flat_embeds = inputs_embeds.reshape(-1, inputs_embeds.shape[-1])
        flat_mask = placeholder_mask.unsqueeze(-1).expand_as(inputs_embeds).reshape(-1, inputs_embeds.shape[-1])
        flat_embeds = flat_embeds.masked_scatter(
            flat_mask, flattened.to(flat_embeds.dtype).reshape(-1)
        )
        inputs_embeds = flat_embeds.reshape_as(inputs_embeds)
        return self.decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
            **kwargs,
        )

    def compute_joint_losses(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
        weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute auxiliary losses from the shared target representation."""
        weights = weights or {}
        mixture = batch["mixture_audio"]
        estimated = complex_mask_to_waveform(
            outputs["mask_logits"], mixture,
            n_fft=self.cfg.audio_frontend.n_fft,
            hop_length=self.cfg.audio_frontend.hop_length,
            win_length=self.cfg.audio_frontend.win_length,
        )
        losses: Dict[str, torch.Tensor] = {
            "si_sdr": si_sdr_loss(estimated, batch["target_audio"], batch.get("mixture_lengths")),
            "speaker": speaker_consistency_loss(
                outputs["target_speaker_embedding"], outputs["speaker_embedding"]
            ),
            "side": side_loss(outputs["side_logits"], batch["target_side"]),
        }
        target_activity = batch["target_activity"]
        target_activity = F.interpolate(
            target_activity.unsqueeze(1), size=outputs["vad_logits"].shape[1], mode="nearest"
        ).squeeze(1)
        losses["vad"] = vad_loss(outputs["vad_logits"], target_activity)
        target_doa = doa_targets_from_angles(
            batch["target_azimuth_deg"], batch["target_elevation_deg"]
        ).unsqueeze(1).expand_as(outputs["doa_vectors"])
        doa_active = F.interpolate(
            batch["target_activity"].unsqueeze(1), size=outputs["doa_vectors"].shape[1], mode="nearest"
        ).squeeze(1)
        losses["doa"] = doa_loss(outputs["doa_vectors"], target_doa, doa_active > 0)
        if "asr_output" in outputs and getattr(outputs["asr_output"], "loss", None) is not None:
            losses["asr"] = outputs["asr_output"].loss
        losses["total"] = sum(
            value * float(weights.get(name, 1.0))
            for name, value in losses.items()
            if name != "total"
        )
        return losses

    def enable_spatial_grad(self, enabled: bool = True) -> None:
        for parameter in self.target_encoder.spatial_encoder.parameters():
            parameter.requires_grad = enabled

    def freeze_decoder(self) -> None:
        if self.decoder is not None:
            for parameter in self.decoder.parameters():
                parameter.requires_grad = False
