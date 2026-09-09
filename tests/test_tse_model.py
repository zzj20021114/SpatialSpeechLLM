"""Shape and loss tests for the target-conditioned TSE modules."""
import pytest


torch = pytest.importorskip("torch")

from dasr.model.config import (  # noqa: E402
    AudioFrontendConfig,
    DasrConfig,
    EnrollmentEncoderConfig,
    ProjectorConfig,
    SpatialEncoderConfig,
    TargetFusionConfig,
    TseHeadConfig,
)
from dasr.model.enrollment_encoder import EnrollmentEncoder  # noqa: E402
from dasr.model.joint_model import JointTargetSpeechModel  # noqa: E402
from dasr.model.target_fusion import TargetFusion  # noqa: E402
from dasr.model.tse_heads import TseMultiTaskHeads  # noqa: E402
from dasr.model.tse_losses import (  # noqa: E402
    combine_tse_losses,
    complex_mask_to_waveform,
    si_sdr,
    speaker_consistency_loss,
)


def test_target_conditioned_tse_shapes():
    torch.manual_seed(0)
    enrollment = EnrollmentEncoder(EnrollmentEncoderConfig(
        hidden_dim=48, output_dim=16, num_layers=1, num_heads=4
    ))
    fusion = TargetFusion(TargetFusionConfig(
        token_dim=32, speaker_dim=16, num_heads=4
    ))
    heads = TseMultiTaskHeads(TseHeadConfig(
        token_dim=32, mask_freq_bins=9, speaker_embed_dim=16
    ))
    enrollment_embedding = enrollment(torch.randn(2, 1600), torch.tensor([1600, 1200]))
    tokens = torch.randn(2, 10, 32)
    target_tokens, gate = fusion(tokens, enrollment_embedding, torch.tensor([10, 7]))
    outputs = heads(target_tokens, torch.tensor([10, 7]))

    assert enrollment_embedding.shape == (2, 16)
    assert target_tokens.shape == (2, 10, 32)
    assert gate.shape == (2, 10)
    assert outputs["mask_logits"].shape == (2, 10, 18)
    assert outputs["vad_logits"].shape == (2, 10)
    assert outputs["doa_vectors"].shape == (2, 10, 3)
    assert outputs["speaker_embedding"].shape == (2, 16)
    assert outputs["side_logits"].shape == (2, 2)


def test_tse_losses_are_finite():
    estimate = torch.randn(2, 320)
    target = torch.randn(2, 320)
    sdr = si_sdr(estimate, target, torch.tensor([320, 240]))
    speaker = speaker_consistency_loss(torch.randn(2, 16), torch.randn(2, 16))
    total = combine_tse_losses({"sdr": -sdr.mean(), "speaker": speaker})
    assert torch.isfinite(sdr).all()
    assert torch.isfinite(total)


def test_complex_mask_reconstruction_shape():
    torch.manual_seed(1)
    mixture = torch.randn(2, 4, 1600)
    mask_logits = torch.randn(2, 10, 402) * 0.1
    estimated = complex_mask_to_waveform(mask_logits, mixture)
    assert estimated.shape == (2, 1600)
    assert torch.isfinite(estimated).all()


def test_joint_target_speech_model_forward_and_losses():
    torch.manual_seed(2)
    cfg = DasrConfig(
        audio_frontend=AudioFrontendConfig(),
        spatial_encoder=SpatialEncoderConfig(
            embed_dim=32,
            num_layers=1,
            num_heads=4,
            ffn_dim=64,
            freeze_backbone=False,
        ),
        enrollment_encoder=EnrollmentEncoderConfig(
            hidden_dim=48, output_dim=16, num_layers=1, num_heads=4
        ),
        target_fusion=TargetFusionConfig(
            token_dim=32, speaker_dim=16, num_heads=4
        ),
        projector=ProjectorConfig(
            input_dim=32, hidden_dim=32, output_dim=16
        ),
        tse_head=TseHeadConfig(
            token_dim=32, mask_freq_bins=201, speaker_embed_dim=16, teacher_embed_dim=16
        ),
    )
    model = JointTargetSpeechModel(cfg)
    mixture = torch.randn(2, 4, 32000)
    enrollment = torch.randn(2, 16000)
    batch = {
        "mixture_audio": mixture,
        "target_audio": mixture[:, 0],
        "mixture_lengths": torch.tensor([32000, 28000]),
        "target_activity": torch.ones(2, 200),
        "target_side": torch.tensor([0, 1]),
        "target_azimuth_deg": torch.tensor([90.0, 270.0]),
        "target_elevation_deg": torch.zeros(2),
        "teacher_embedding": torch.randn(2, 16),
    }
    outputs = model(mixture, enrollment, batch["mixture_lengths"])
    losses = model.compute_joint_losses(outputs, batch)

    assert outputs["target_tokens"].shape[0] == 2
    assert outputs["projected_target_tokens"].shape[-1] == 16
    assert torch.isfinite(losses["total"])
