"""dasr.model.config — 系统配置（独立原创实现）。

架构（自建编码器-解码器）：
  编码器（全部自建、可训练）：
    AudioFrontend  ：FOA -> 7 通道 log-mel + 强度向量特征图
    SpatialEncoder ：特征图 -> 空间/语音 token 序列（12.5 Hz）
    Projector      ：token -> 解码器隐空间（Qwen3-8B hidden=4096）
  解码器：
    Qwen3-8B（文本 LLM，HuggingFace 公开权重）+ LoRA 微调
  输入：FOA 4ch；输出：文本（转录 + <方位> 左/右）。
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional, Union, get_args, get_origin, get_type_hints


def _coerce(field_type, value: Any) -> Any:
    """把 str 数值强转为字段类型（YAML 可能把 1e-4 解析成字符串）。"""
    if not isinstance(value, str):
        return value
    t = field_type
    args = get_args(t)
    if args:
        # Optional[X] / Union：取第一个非 None 成员
        t = next((a for a in args if a is not type(None)), t)
    if t in (float, int):
        try:
            return t(value)
        except (ValueError, TypeError):
            return value
    return value


def _from_dict(cls, data: Optional[Dict[str, Any]]):
    if not data:
        return cls()
    hints = get_type_hints(cls)
    coerced = {k: (_coerce(hints[k], v) if k in hints else v) for k, v in data.items()}
    return cls(**coerced)


@dataclass
class AudioFrontendConfig:
    """FOA 音频前端：逐通道 log-mel + 强度向量（IV）。

    低层 mel 参数与 Whisper 风格前端对齐（16k / 400 / 160 / 128）。
    """
    sample_rate: int = 16000
    n_fft: int = 400
    win_length: int = 400
    hop_length: int = 160
    num_mel_bins: int = 128
    fmin: float = 0.0
    fmax: Optional[float] = None
    normalize_logmel: bool = True
    fbank_mean: float = 15.41663
    fbank_std: float = 6.55582
    iv_eps: float = 1e-10
    feature_channels: int = 7     # 0-3: W/X/Y/Z logmel; 4-6: IVx/y/z


@dataclass
class SpatialEncoderConfig:
    """自建空间/语音编码器：patch 嵌入 -> Transformer 骨干 -> token 序列。

    输出 token 率默认 12.5 Hz（hop160@16k / stride_time 8），足够 ASR 用。
    相对位置偏置自写实现；可选以公开 BEATs 权重热启动（维度需一致）。
    """
    embed_dim: int = 768
    foa_feature_channels: int = 7
    patch_time: int = 16
    patch_freq: int = 128
    stride_time: int = 8
    stride_freq: int = 128
    num_layers: int = 12
    num_heads: int = 12
    ffn_dim: int = 3072
    dropout: float = 0.1
    attn_dropout: float = 0.1
    activation: str = "gelu"
    layer_norm_first: bool = True
    relative_position_embeddings: bool = True
    num_position_buckets: int = 320
    max_distance: int = 1280
    target_token_rate: float = 12.5
    encoder_native_rate: Optional[float] = None
    resample_mode: str = "conv"
    init_backbone_from: Optional[str] = None
    freeze_backbone: bool = True


@dataclass
class EnrollmentEncoderConfig:
    """目标说话人 enrollment 编码器。"""
    n_fft: int = 400
    win_length: int = 400
    hop_length: int = 160
    hidden_dim: int = 384
    output_dim: int = 256
    num_layers: int = 3
    num_heads: int = 6
    dropout: float = 0.1


@dataclass
class TargetFusionConfig:
    """将 enrollment 身份条件注入 FOA 空间 token。"""
    token_dim: int = 768
    speaker_dim: int = 256
    num_heads: int = 12
    dropout: float = 0.1
    similarity_temperature: float = 5.0


@dataclass
class TseHeadConfig:
    """TSE、VAD、DOA 和 speaker consistency 多任务头。"""
    token_dim: int = 768
    mask_freq_bins: int = 201
    speaker_embed_dim: int = 256
    teacher_embed_dim: int = 256
    num_sides: int = 2


@dataclass
class ProjectorConfig:
    """MLP 投影器：编码器 token -> 解码器隐空间（Qwen3-8B hidden=4096）。

    shuffle_factor=1 时退化为纯 MLP（不降采样，保持 12.5 Hz token 率供 ASR）。
    """
    input_dim: int = 768
    shuffle_factor: int = 1
    hidden_dim: int = 1024
    output_dim: int = 4096
    pre_norm: bool = True
    activation: str = "gelu"


@dataclass
class DecoderConfig:
    """Qwen3 文本解码器（HuggingFace 公开权重）+ LoRA。"""
    model_id: str = "Qwen/Qwen3-8B"
    dtype: str = "bfloat16"
    quantization: str = "none"                  # none | 4bit（QLoRA）
    device_map: Optional[str] = None            # "auto" 等
    max_memory: Optional[dict] = None
    speech_token: str = "<|speech|>"            # 自注册的语音 token 占位符
    freeze_llm: bool = True                     # 用 LoRA 微调
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: tuple = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


@dataclass
class SeldHeadConfig:
    """SELD 预训练头（ACCDOA 风格）：帧级活动分类 + 笛卡尔坐标回归。"""
    num_classes: int = 13
    frame_wise: bool = True
    activity_threshold: float = 0.5
    activity_weight: float = 1.0
    coord_weight: float = 1.0
    coord_lambda: float = 1.0


@dataclass
class DataConfig:
    """数据管线配置。"""
    qa_roots: tuple = ("data/qa",)
    audio_root: str = "data"
    split: str = "train"
    sample_rate: int = 16000
    max_audio_seconds: float = 20.0
    prompt_config: str = "conf/directional_asr_prompt.yaml"
    shuffle: bool = True
    num_workers: int = 4


@dataclass
class TrainConfig:
    """训练配置。"""
    stage: str = "encoder_lora"    # seld_pretrain | projector_only | encoder_lora | full
    batch_size: int = 2
    grad_accum_steps: int = 4
    epochs: int = 3
    learning_rate: float = 3e-5
    projector_lr: float = 1e-4
    lora_lr: float = 3e-5
    spatial_lr: float = 1e-5
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    max_grad_norm: float = 5.0
    log_interval: int = 10
    save_interval: int = 1000
    output_dir: str = "runs/dasr"
    resume_from: Optional[str] = None
    seed: int = 42


@dataclass
class DasrConfig:
    """顶层配置。"""
    audio_frontend: AudioFrontendConfig = field(default_factory=AudioFrontendConfig)
    spatial_encoder: SpatialEncoderConfig = field(default_factory=SpatialEncoderConfig)
    enrollment_encoder: EnrollmentEncoderConfig = field(default_factory=EnrollmentEncoderConfig)
    target_fusion: TargetFusionConfig = field(default_factory=TargetFusionConfig)
    tse_head: TseHeadConfig = field(default_factory=TseHeadConfig)
    projector: ProjectorConfig = field(default_factory=ProjectorConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    seld_head: SeldHeadConfig = field(default_factory=SeldHeadConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, path: str) -> None:
        import json
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DasrConfig":
        cfg = cls()
        for section in ("audio_frontend", "spatial_encoder", "enrollment_encoder",
                        "target_fusion", "tse_head", "projector",
                        "decoder", "seld_head", "data", "train"):
            sub = data.get(section)
            if not sub:
                continue
            setattr(cfg, section, _from_dict(getattr(cfg, section).__class__, sub))
        return cfg

    @classmethod
    def from_yaml(cls, path: str) -> "DasrConfig":
        import yaml
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(yaml.safe_load(f) or {})

    def copy(self) -> "DasrConfig":
        return _from_dict(type(self), self.to_dict())
