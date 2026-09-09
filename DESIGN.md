# Spatial-OSUM：定向多任务语音理解大模型 —— 设计方案

> 首期里程碑：**左右定向 ASR（Left/Right Directional ASR）** —— 在 Qwen3 基座上同时完成
> 语音识别（ASR，转录内容）与**声源左右判定**（说话人在左/右）。
> 全方位角/仰角回归作为二期扩展（相关代码已预留 degree 模式）。
>
> **原创性声明**：本项目**参考** OSUM 与 Spatial-Omni 的*研究思想*（ASR+X 联合训练、
> 标签式提示、FOA 空间编码、三阶段训练），但**全部代码为独立原创实现**，不复制/导入
> 上游仓库的代码。仅使用公开的第三方预训练模型权重（如 Qwen3-Omni、BEATs）作为基座，
> 空间编码器完全自建并在公开 SELD 数据上自预训练。

## 1. 背景与目标

现有两个模型各有侧重，二者互补：

| 模型 | 基座 | 核心能力 | 关键机制 |
|---|---|---|---|
| **OSUM** | Whisper 编码器 + Qwen2 文本 LLM | 多任务语音理解（ASR / SRWT / VED / SER / SSR / SGC / SAP / STTC） | **ASR+X 联合训练**：始终与 ASR 一起优化目标任务；任务用标签式提示（`<TRANSCRIBE>` `<EMOTION>` 等）表达；输出 = 转录 + 标签 |
| **Spatial-Omni** | Qwen2.5-Omni / **Qwen3-Omni-MoE** + SO-Encoder | 多声道（FOA）声源定位与空间推理（16 个子任务） | SO-Encoder（BEATs，空间预训练）将 4 通道 FOA 音频编码为 2.5 Hz 空间 token，经 `<|spatial|>` 占位符注入 LLM；三阶段训练 |

**目标**：设计一个 *定向多任务语音理解大模型*，同时具备
① OSUM 的多任务语音理解能力、② Spatial-Omni 的声源方位感知能力，且基座采用 **Qwen3**。

**首期范围（左右定向 ASR）**：单说话人 FOA 语音 → 输出 `转录文本 + 说话人在左/右`。
这是最小可行验证集，跑通后按路线图（§9）扩展到全方位角、完整多任务 + 全部空间子任务。

## 2. 总体架构

```
                    FOA 4 通道音频 (16 kHz, [W,Y,Z,X] DCASE 序)
                                   │
             ┌─────────────────────┴─────────────────────┐
             │ W 通道 (全向) = arr[0]                     │ 完整 4ch FOA
             ▼                                            ▼
   Qwen3-Omni-MoE 单声道音频塔                  SpatialEncoder（本项目自建）
   (公开模型自带, W 通道 ASR)                    (mel+IV 前端 → Transformer 骨干
             │                                     → 时间降采样 → 空间 token)
       <|audio|> 占位符                                  │
             │                                     pixel_shuffle MLP 投影器
             │                                            │
             └────────────────┬───────────────────────────┘
                              ▼
              Qwen3-Omni-MoE LLM（公开模型, LoRA, 本项目自写注入层）
                              ▼
        文本输出：<转录内容> <方位> 方位角=45.0, 仰角=10.0
```

**要点**

1. **单输入、双分支**：一份 FOA 音频同时服务两个分支——W 通道（全向）天然携带完整语音内容，送入 Qwen3-Omni 的单声道音频塔做 ASR/声学理解；完整 4 通道 FOA 送入自建 `SpatialEncoder` 做空间定位。这正符合选定的 *"FOA-primary，W 通道做语音"* 方案。
2. **基座 = 公开 Qwen3-Omni**：从 HuggingFace 加载 `Qwen/Qwen3-Omni-30B-A3B`（公开模型权重），**自写**封装层负责空间 token 注入（`<|spatial|>` 占位符展开 + `masked_scatter`），不复制任何上游封装代码。
3. **空间编码器完全自建**：FOA 前端（逐通道 log-mel + 强度向量 IV）+ 自写 Transformer 骨干 + 时间降采样；在公开 FOA SELD 数据（DCASE 系列）上用 ACCDOA 目标**自预训练**，产出自有权重。
4. **多任务机制来自 OSUM 思想**：标签式提示 + 转录永远在输出中的 *ASR+X* 范式。定向 ASR 中每个样本的 answer 都包含转录，故 ASR 始终与定位联合优化。
5. **训练 = 自写三阶段**：S1 对齐投影器 → S2 LLM LoRA + 投影器（ASR+X 联合）→ S3 解冻空间编码器微调；外加 S0 空间编码器 SELD 预训练。

### 2.1 FOA 输入约定（与 SO-Encoder 训练数据一致）

| 属性 | 约定 |
|---|---|
| 采样率 | 16 kHz |
| 通道数 / 顺序 | 4 通道，**DCASE 波形序 `[W, Y, Z, X]`**（`SOBackbonePreprocessor._reorder_dcase_wyzx_to_wxyz` 内部重排为 `[W,X,Y,Z]`） |
| 归一化 | SN3D（FuMa→SN3D 均可行，需与编码一致） |
| 单源点源编码 | `W=s/√2, X=s·cosφ·cosθ, Y=s·cosφ·sinθ, Z=s·sinφ`（θ 方位角逆时针自正前起，φ 仰角向上为正） |
| 最大时长 | 20 s（对齐 `MAX_AUDIO_SECONDS` / SO-Bench 约定） |

## 3. 任务定义

### 3.1 左右定向 ASR（本期）

- **输入**：FOA 语音（单说话人；扩展：多说话人）
- **输出**：`<转录内容> <方位> 左|右`
- 单源输出示例：
  ```
  今天下午三点在会议室开会。 <方位> 左
  ```
- 方位约定：0°=正前，逆时针增大，90°=左、270°=右；合成时方位只取左/右半区
  （默认左 45°–135°、右 225°–315°，避开正前/正后模糊区）。
- 二期扩展（degree 模式）：输出 `方位角=/仰角=` 回归，脚本已预留。

### 3.2 提示设计（标签式，源自 OSUM）

任务标签组合：`<TRANSCRIBE> <DIRECTION>`。模板见 `conf/directional_asr_prompt.yaml`，
每条指令描述任务并**规定输出格式**（`<转录内容> <方位> 左/右`），保证输出可解析。

与 OSUM 的 `prompt_config.yaml` 结构一致：一个标签组合对应多条模板，训练时随机采样，提升泛化。

## 4. 数据管线

```
语音语料(wav+转录) ─► 1. dasr/data/build_speech_manifest.py ─► speech manifest
  ─► 2. dasr/data/generate_synthetic_foa.py（--leftright）──► FOA wav + manifest.jsonl
                                                            (方位取左右半区, 记录 side,
                                                             SN3D FOA 编码, 可选噪声/多源)
  ─► 3. dasr/data/build_qa.py（--leftright）──────────────► qa/{train,valid,test}.jsonl
```

### 4.1 合成 FOA 数据（首期，可离线）

- 输入语音语料：任意（AISHELL / LibriSpeech / 自有），先经 `build_speech_manifest` 转成
  `{key, wav, txt}` 格式。
- 每个片段（左右模式）：随机决定左/右 → 方位角落在对应半区（默认左 45–135°、右 225–315°，
  避开正前/正后模糊区），仰角如 -30°–30°，按 §2.1 公式编码为 FOA，manifest 记录 `side: 左|右`。
- 增强选项：`--noise-snr`（加性噪声）、`--num-sources>1`（多说话人不同方位叠加）、
  时长截断至 `--max-seconds`。
- 输出：`{key}.wav`（4ch FOA，DCASE 序）+ `manifest.jsonl`（含 `side/azimuth_deg/elevation_deg/distance_m/txt`）。

> 与 SO-Dataset 的构造方式同源（模拟空间化），区别在于声源为**语音**并携带**转录**标注。

### 4.2 QA 数据（对齐 Spatial-Omni schema）

`qa/{split}.jsonl` 每条：

```json
{
  "pair_id": "dasr_00001",
  "split": "train",
  "audio_path": "audio/train/00001.wav",
  "dataset": "synthetic",
  "task_type": "directional_asr",
  "task_name": "transcribe_and_locate",
  "prompt": "请转录音频中的语音内容，并在转录之后给出<方位>标签，格式：<方位> 方位角=xx.x, 仰角=xx.x",
  "question": "同上（prompt 缺失时回退）",
  "answer": "今天下午三点在会议室开会。 <方位> 方位角=45.0, 仰角=10.0",
  "canonical_answer": "今天下午三点在会议室开会。",
  "transcription": "今天下午三点在会议室开会。",
  "source_refs": [{"class_id": -1, "class_name": "speech",
                    "azimuth_deg": 45.0, "elevation_deg": 10.0, "distance_m": 2.0}]
}
```

- `audio_path` 相对 `qa/` 的父目录（与 SO-Dataset 的 `qa/`+`audio/` 兄弟布局一致，
  训练时用 `--audio-root` 解析）。
- 现有训练器 `train_so_qa.py` 的 `SpatialBeatsQACollator` 会自动把
  `prefix = <|audio|><|spatial|>\n{prompt}\n` 拼好，**无需改 collator**。

## 5. 训练策略

融合 OSUM 的 *ASR+X* 与 Spatial-Omni 的三阶段思想，全部由本项目自写训练器实现：

| 阶段 | 模式 | 可训练参数 | 数据 | 说明 |
|---|---|---|---|---|
| S0 空间预训练 | `seld_pretrain` | SpatialEncoder + SELD 头 | DCASE FOA SELD 数据 | ACCDOA 目标，产出自有空间编码器权重 |
| S1 对齐 | `projector_only` | 投影器 | 定向 ASR QA（含转录） | 对齐空间 token 与 LLM 语义；W 通道 ASR 由公开音频塔承担 |
| S2 联合 | `encoder_lora` | LLM LoRA + 投影器 | 定向 ASR QA | **ASR+X 核心阶段**：每个样本均含转录 + 方位，ASR 与定位联合优化 |
| S3 精调 | `spatial_lora` | SpatialEncoder(LoRA) + LoRA + 投影器 | 定向 ASR QA | 解冻空间编码器，进一步压榨方位精度 |

- 基座：公开 `Qwen/Qwen3-Omni-30B-A3B`（HuggingFace 官方 transformers 加载），LoRA 微调注意力投影层。
- 训练细节：bf16 / `attn_impl=sdpa`，LoRA `r=16, alpha=32`（目标 `q/k/v/o`），DeepSpeed ZeRO。
- 数据混合（多任务扩展）：支持多路 QA 根按权重混入（如纯 ASR 单声道数据），对应 OSUM *ASR+X* 数据加权思想。

## 6. 评估

`dasr/evaluate.py` 在 `predictions.jsonl` 上计算（首期以左右为主）：

| 指标 | 说明 | 对标 |
|---|---|---|
| **side_accuracy** | 左右判定正确率（首期核心指标） | 二分类准确率 |
| WER / CER | 转录部分词错误率 / 字错误率（中文用字级编辑距离） | OSUM |
| 联合准确率 | 转录 CER≤阈值 **且** 左右判定正确 | 综合 |
| 方位角 MAE / 仰角 MAE | 二期 degree 模式（gold 含方位时自动计算） | SO-Bench |
| 方向 bin EM | 二期 degree 模式（方位/仰角均在阈值内） | SO-Bench |

生成推理由 `dasr/generate.py` 完成（读 run-dir 配置重建模型 → 逐条生成 → 写 predictions.jsonl），
打分由 `dasr/evaluate.py` 完成（解析 `prediction` 中的转录与左右标签 → 计算上述指标）。

## 7. 代码结构

```
DirectedASR/
├── DESIGN.md                         # 本文档
├── README.md                         # 使用说明
├── conf/
│   ├── directional_asr_prompt.yaml   # 定向 ASR 提示模板（标签式）
│   └── dasr_defaults.yaml            # 模型/数据/训练默认参数
├── dasr/                             # 独立原创包（不导入上游代码）
│   ├── __init__.py
│   ├── prompts.py                    # 模板加载 + 答案格式化/解析
│   ├── model/
│   │   ├── config.py                 # 自建配置 dataclass
│   │   ├── audio_frontend.py         # 自建 FOA→7通道 mel+IV 特征前端
│   │   ├── spatial_encoder.py        # 自建空间编码器（patch+Transformer+降采样）
│   │   ├── projector.py              # 自建 pixel-shuffle MLP 投影器
│   │   ├── seld_heads.py             # 自建 ACCDOA SELD 预训练头
│   │   └── qwen3_omni_wrapper.py     # 自写公开 Qwen3-Omni 封装 + 空间注入
│   ├── data/
│   │   ├── foa.py                    # 自建 SN3D FOA 编码/工具
│   │   ├── dataset.py                # 自建 QA Dataset + Collator
│   │   ├── seld_dataset.py           # 自建 DCASE SELD 数据装载（S0 预训练）
│   │   ├── generate_synthetic_foa.py # 自建合成 FOA 语音脚本
│   │   └── build_qa.py               # 自建 QA jsonl 构建脚本
│   ├── train.py                      # 训练入口（S0–S3）
│   ├── generate.py                   # 推理入口
│   └── evaluate.py                   # WER/CER + 方位 MAE + bin EM
├── shell/
│   └── launch_train_dasr.sh          # 一键三阶段启动脚本
└── tests/
    └── test_foa.py                   # FOA 编码自洽性测试
```

**独立性**：`dasr/` 仅依赖公开库（`torch` / `transformers` / `torchaudio` / `soundfile` /
`numpy` / `scipy`）与公开模型权重；不导入、不复制 OSUM/Spatial-Omni 的任何源码。

## 8. 运行方式（简）

```bash
# 0) 环境
conda create -n dasr python=3.11 && conda activate dasr
pip install torch torchaudio transformers soundfile numpy scipy pyyaml peft accelerate

# 1) 语音语料 -> speech manifest
python -m dasr.data.build_speech_manifest --dir /data/aishell1 \
    --output data/speech/train.jsonl --lang zh

# 2) 合成 FOA 定向语音数据（左右模式，默认开）
python -m dasr.data.generate_synthetic_foa \
    --speech-manifest data/speech/train.jsonl \
    --output-dir data/synth/train --split train   # --leftright 默认开

# 3) 构建 QA（左右答案）
python -m dasr.data.build_qa \
    --manifest data/synth/train/manifest.jsonl \
    --qa-dir data/qa --split train                # --leftright 默认开

# 4) 训练（S1 对齐 → S2 联合 → S3 精调；S0 SELD 预训练可选）
bash shell/launch_train_dasr.sh

# 5) 推理 + 评估
python -m dasr.generate --run-dir runs/dasr_stage3 --checkpoint <ckpt> \
    --qa-root data/qa --audio-root data --split test
python -m dasr.evaluate \
    --predictions-jsonl runs/dasr_stage3/bench/test/predictions.jsonl \
    --qa-root data/qa --split test
```

## 9. 扩展路线图

0. **左右定向（首期，进行中）**：左右二分类 + ASR，端到端 S1→S2→S3。
1. **degree 方位回归（二期）**：恢复全方位角/仰角输出（代码已预留 degree 模式），
   合成数据去掉左右半区限制、评估改回方位 MAE/bin EM。
2. **多说话人定向 ASR**：多源混合数据 + answer 中多段 `<方位>`（按说话人分组）。
3. **完整 OSUM 多任务**：同一 FOA 输入上追加 `<EMOTION>/<STYLE>/<GENDER>/<AGE>/<CAPTION>`
   标签，answer = `转录 + 各标签`，数据按任务加权混合（ASR+X）。
4. **完整空间子任务**：接入公开空间音频评测集（如 SO-Bench 风格）的检测/方位估计/空间推理子任务。
5. **真实场景**：用真实多通道拾音数据替换/补充合成数据；引入 RIR 混响（pyroomacoustics）。
6. **联合解码优化**：对转录与方位分字段的 beam/约束解码。

## 10. 与两个上游模型的关系（思想借鉴，非代码复用）

- **借鉴 OSUM 的思想**：标签式提示体系、ASR+X 训练范式、WER/CER 评估口径。实现为本项目的 `prompts.py` 与 `train.py`（自写）。
- **借鉴 Spatial-Omni 的思想**：FOA（DCASE 序 `[W,Y,Z,X]`）作为空间模态、空间 token 经占位符注入 Omni LLM、分阶段训练。实现为本项目的 `model/` 与 `data/`（自写）。
- **公开依赖**：Qwen3-Omni 模型权重（HuggingFace）、DCASE FOA SELD 公开数据集。空间编码器权重为本项目自预训练产出。
