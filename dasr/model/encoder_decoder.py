"""dasr.model.encoder_decoder — 自建编码器-解码器模型（独立原创实现）。

架构：
  编码器（全部自建、可训练）：
    AudioFrontend   ：FOA 4ch（DCASE 序 [W,Y,Z,X]）-> 7 通道特征图
    SpatialEncoder  ：特征图 -> token 序列（12.5 Hz，承载语音内容 + 左右方位）
    Projector       ：token -> Qwen3-8B 隐空间（4096）
  解码器：
    Qwen3-8B（HuggingFace 公开文本 LLM，LoRA 微调）
  拼接：
    文本中 `<|speech|>` 占位符展开 N 个 -> 把编码器 token 的投影嵌入 masked_scatter
    填入占位符位置 -> 交给 Qwen3 解码器生成文本（转录 + <方位> 左/右）。

个人可训练性：编码器 ~1 亿参数 + Qwen3-8B LoRA（~2 千万）全可训，
单张 24-32GB 卡即可（bf16 或 4-bit）。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from .config import DasrConfig
from .audio_frontend import AudioFrontend
from .spatial_encoder import SpatialEncoder
from .projector import PixelShuffleProjector


class DasrEncoderDecoder(nn.Module):
    def __init__(self, cfg: DasrConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.frontend = AudioFrontend(cfg.audio_frontend)
        self.spatial_encoder = SpatialEncoder(cfg.spatial_encoder)
        self.projector = PixelShuffleProjector(cfg.projector)
        self.decoder, self.tokenizer = self._load_decoder(cfg.decoder)
        self.speech_token_id = self._register_speech_token(cfg.decoder.speech_token)
        self._pin_encoders()

    # ------------------------------------------------------------------
    # 解码器加载 / token 注册
    # ------------------------------------------------------------------
    def _load_decoder(self, dcfg):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(dcfg.model_id, trust_remote_code=True)
        kwargs: Dict[str, Any] = {
            "dtype": {"float16": torch.float16, "bfloat16": torch.bfloat16,
                      "float32": torch.float32}[dcfg.dtype],
            "trust_remote_code": True,
        }
        if dcfg.quantization == "4bit":
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
            )
        if dcfg.device_map:
            kwargs["device_map"] = dcfg.device_map
        if dcfg.max_memory:
            kwargs["max_memory"] = dict(dcfg.max_memory)
        model = AutoModelForCausalLM.from_pretrained(dcfg.model_id, **kwargs)
        return model, tokenizer

    def _register_speech_token(self, token: str) -> int:
        tok = self.tokenizer
        if token not in tok.get_vocab():
            tok.add_special_tokens({"additional_special_tokens": [token]})
        token_id = int(tok.convert_tokens_to_ids(token))
        if len(tok) != self.decoder.get_input_embeddings().num_embeddings:
            self.decoder.resize_token_embeddings(len(tok))
        self.decoder.config.speech_token_id = token_id
        return token_id

    def _pin_encoders(self) -> None:
        try:
            dev = next(self.decoder.get_input_embeddings().parameters()).device
        except StopIteration:
            dev = torch.device("cpu")
        if str(dev) != "meta":
            self.frontend.to(dev)
            self.spatial_encoder.to(dev)
            self.projector.to(dev)

    def _device(self) -> torch.device:
        # 动态取解码器嵌入层当前所在设备（模型可能被 .to(cuda) 或 device_map 移动）
        try:
            return next(self.decoder.get_input_embeddings().parameters()).device
        except StopIteration:
            return torch.device("cpu")

    # ------------------------------------------------------------------
    # LoRA / 冻结
    # ------------------------------------------------------------------
    def apply_lora(self) -> List[str]:
        from peft import LoraConfig, TaskType, get_peft_model
        dcfg = self.cfg.decoder
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=dcfg.lora_r, lora_alpha=dcfg.lora_alpha, lora_dropout=dcfg.lora_dropout,
            target_modules=list(dcfg.lora_target_modules),
        )
        self.decoder = get_peft_model(self.decoder, lora_cfg)
        return [n for n, p in self.decoder.named_parameters() if p.requires_grad]

    def enable_encoder_grad(self, enabled: bool = True) -> None:
        for p in list(self.frontend.parameters()) + list(self.spatial_encoder.parameters()):
            p.requires_grad = enabled

    # ------------------------------------------------------------------
    # 编码器 -> 投影嵌入（打包为 [sum T, D_llm]）
    # ------------------------------------------------------------------
    def _encode(self, spatial_audio, spatial_audio_lengths=None):
        feats, feat_lens = self.frontend(spatial_audio, spatial_audio_lengths)
        tokens, tok_lens = self.spatial_encoder(feats, feat_lens)
        projected = self.projector(tokens)
        if tok_lens is not None:
            chunks = [projected[i, :int(tok_lens[i].item())] for i in range(projected.shape[0])]
            return torch.cat(chunks, dim=0), tok_lens
        return projected.reshape(-1, projected.shape[-1]), None

    @staticmethod
    def _splice(inputs_embeds, new_embeds, mask):
        B, L, D = inputs_embeds.shape
        n_ph = int(mask.sum().item())
        if n_ph == 0:
            return inputs_embeds
        emb = new_embeds
        if emb.shape[0] > n_ph:
            emb = emb[:n_ph]
        elif emb.shape[0] < n_ph:
            emb = torch.cat([emb, emb[-1:].expand(n_ph - emb.shape[0], D)], dim=0)
        flat = inputs_embeds.reshape(-1, D)
        flat_mask = mask.unsqueeze(-1).expand(-1, -1, D).reshape(-1, D)
        flat = flat.masked_scatter(flat_mask, emb.reshape(-1).to(flat.dtype))
        return flat.reshape(B, L, D)

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def forward(
        self,
        input_ids: torch.Tensor,
        spatial_audio: torch.Tensor,
        spatial_audio_lengths: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        **kwargs: Any,
    ) -> Any:
        dev = self._device()
        input_ids = input_ids.to(dev)
        if labels is not None:
            labels = labels.to(dev)
        if attention_mask is not None:
            attention_mask = attention_mask.to(dev)

        projected, _ = self._encode(spatial_audio, spatial_audio_lengths)
        inputs_embeds = self.decoder.get_input_embeddings()(input_ids)
        mask = input_ids == self.speech_token_id
        inputs_embeds = self._splice(inputs_embeds, projected.to(dev), mask)

        return self.decoder(
            inputs_embeds=inputs_embeds,
            labels=labels,
            attention_mask=attention_mask,
            return_dict=return_dict,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # 推理
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        spatial_audio: torch.Tensor,
        spatial_audio_lengths: Optional[torch.Tensor] = None,
        max_new_tokens: int = 96,
        do_sample: bool = False,
        num_beams: int = 1,
        **gen_kwargs: Any,
    ) -> str:
        dev = self._device()
        spatial_audio = spatial_audio.to(dev)
        if spatial_audio_lengths is not None:
            spatial_audio_lengths = spatial_audio_lengths.to(dev)
        feats, feat_lens = self.frontend(spatial_audio, spatial_audio_lengths)
        _, tok_lens = self.spatial_encoder(feats, feat_lens)
        n_sp = int(tok_lens[0].clamp(min=1).item())
        token = self.cfg.decoder.speech_token
        full_text = f"{token * n_sp}\n{prompt}\n"
        enc = self.tokenizer(full_text, return_tensors="pt")
        input_ids = enc["input_ids"].to(dev)

        projected, _ = self._encode(spatial_audio, spatial_audio_lengths)
        inputs_embeds = self.decoder.get_input_embeddings()(input_ids)
        mask = input_ids == self.speech_token_id
        inputs_embeds = self._splice(inputs_embeds, projected.to(dev), mask)
        attn = torch.ones_like(input_ids)

        out = self.decoder.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attn,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            num_beams=num_beams,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            **gen_kwargs,
        )
        return self.tokenizer.decode(out[0], skip_special_tokens=True)
