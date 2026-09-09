"""dasr.model — 模型模块（自建编码器-解码器，全部独立原创实现）。"""
from .config import (  # noqa: F401
    AudioFrontendConfig,
    DasrConfig,
    DataConfig,
    DecoderConfig,
    EnrollmentEncoderConfig,
    ProjectorConfig,
    SeldHeadConfig,
    SpatialEncoderConfig,
    TargetFusionConfig,
    TrainConfig,
    TseHeadConfig,
)
from .audio_frontend import AudioFrontend  # noqa: F401
from .spatial_encoder import SpatialEncoder  # noqa: F401
from .projector import PixelShuffleProjector  # noqa: F401
from .seld_heads import AccdoaSeldHead, SeldPretrainModel, align_labels_to_token_frames  # noqa: F401
from .encoder_decoder import DasrEncoderDecoder  # noqa: F401
from .enrollment_encoder import EnrollmentEncoder  # noqa: F401
from .target_fusion import TargetFusion  # noqa: F401
from .target_conditioned_encoder import TargetConditionedSpatialEncoder  # noqa: F401
from .tse_heads import TseMaskHead, TseMultiTaskHeads  # noqa: F401
from .joint_model import JointTargetSpeechModel  # noqa: F401


def build_model(cfg: DasrConfig) -> DasrEncoderDecoder:
    """按配置组装自建编码器-解码器模型。"""
    return DasrEncoderDecoder(cfg)
