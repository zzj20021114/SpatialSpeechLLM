# 训练数据需求（定向 ASR · Spatial-OSUM 首期）

> **首期目标：左右定向识别**（LEFT/RIGHT 二分类 + ASR），而非全方位角回归。
> 因此合成数据时方位角只落在左/右两个半区（避开正前/正后模糊区），
> 标注只需 `side: 左|右`；评估指标以侧别准确率 `side_accuracy` 为主。
> （全方位角回归留作二期扩展，相关字段与脚本已预留。）

本模型分四个训练阶段，各阶段数据需求不同：

| 阶段 | 作用 | 需要的数据 |
|---|---|---|
| **S0** `seld_pretrain` | 预训练空间编码器（声源定位能力） | FOA + 声学事件 SELD 标注（类、起止时间、方位/仰角） |
| **S1** `projector_only` | 对齐空间 token 与 LLM 语义 | 定向 ASR QA（FOA 语音 + 转录 + 左右标注） |
| **S2** `encoder_lora` | ASR + 左右定位联合训练（核心） | 定向 ASR QA（必须每样本含转录） |
| **S3** `spatial_lora` | 解冻空间编码器精调 | 定向 ASR QA |

> 左右二分类对空间编码器相对简单：**S0 SELD 预训练可跳过**，直接端到端训练 S1→S2 即可。
> 若追求更强空间鲁棒性，仍可按 §1 补充 SELD 预训练。

**统一的音频底层约定（所有阶段）**

| 属性 | 要求 |
|---|---|
| 采样率 | **16 kHz** |
| FOA 通道 | **4 通道 B-format（SN3D 归一化）**，DCASE 波形序 **`[W, Y, Z, X]`** |
| 方位角 | 0°=正前，逆时针（俯视）增大，`[0, 360)` |
| 仰角 | `[-90, 90]`，向上为正 |
| 距离 | 米，可缺省 |
| 单段时长 | **≤ 20 s**（建议 2–15 s 为主） |
| 事件标注粒度 | 帧级 10 Hz（0.1 s）或事件式（start/end + DOA） |

---

## 1. S0 —— SELD 预训练数据（空间编码器）

**目的**：让空间编码器学会从 FOA 中提取声源到达方向，是空间能力的"底座"。

**格式需求**（本项目 `dasr/data/seld_dataset.py` 的 manifest）：
```json
{
  "audio_path": "audio/train/foa_0001.wav",
  "events": [
    {"class_id": 0, "class_name": "speech",
     "azimuth_deg": 45.0, "elevation_deg": 10.0,
     "start_s": 0.0, "end_s": 3.2},
    {"class_id": 0, "class_name": "speech",
     "azimuth_deg": 90.0, "elevation_deg": 0.0,
     "start_s": 5.0, "end_s": 8.0}
  ]
}
```
> 帧级 DCASE 格式可通过脚本 `dasr/data/dcase_to_manifest.py` 自动转换（见 §5）。

**推荐公开数据集（DCASE 空间声学事件定位与检测任务）**——FOA + 标注完全匹配：

| 数据集 | 版本/年份 | 事件类数 | 说明 | 备注 |
|---|---|---|---|---|
| TAU-NIGENS Spatial Sound Events | 2020 Task 3 | 11 | FOA，模拟场景 | 入门可用 |
| TAU-NIGENS Spatial Sound Events | 2021 Task 3 | 12 | FOA | 标准选择 |
| **TAU Spatial Sound Events** | **2022 Task 3** | 13 | FOA，**真实录音**，带距离 | 强烈推荐（真实感最好） |
| TAU Spatial Sound Events | 2023 Task 3 | 13 | FOA | 推荐 |
| TAU Spatial Sound Events | 2024 Task 3 | 14 | FOA + 距离 | 推荐 |

- 获取：DCASE Challenge 官网 <https://dcase.community/challenge2024/task-sound-event-localization-and-detection>
- 数据量：每版约 1–7 小时标注音频（数千段）。建议 **合并 2–3 个版本（2022+2023+2024）**，约 1–2 万段，足够空间编码器预训练。
- 只要 FOA（`.wav`）+ 标注 `metadata.csv`；数据集内自带的 `foa_dev` 拆分照用即可。

---

## 2. 定向 ASR 训练数据（S1/S2/S3 核心，最重要）

**需求**：FOA 语音片段 + 说话内容转录 + 说话人方位（首期为左右）。

公开数据中**几乎没有**"FOA 语音 + 转录 + DOA"三件套。推荐做法：

### 2.1 推荐方案：合成（用语音语料 + 空间化）

流程（本项目脚本已就绪）：
```
语音语料(wav+txt)  --build_speech_manifest-->  speech manifest
  --generate_synthetic_foa（--leftright）-->  FOA wav + side(左/右) 标注
  --build_qa（--leftright）-->                  qa/{train,valid,test}.jsonl
```

**你需要准备的只是"语音语料"**（单声道 + 转录），要求：

| 要求 | 建议值 |
|---|---|
| 采样率 | 16 kHz（或可由脚本重采样） |
| 声道 | 单声道，单说话人/段 |
| 转录 | 纯文本，与 wav 一一对应（`key` 同名） |
| 单段时长 | 2–15 s（≤20 s） |
| 语言 | 由评测目标决定：中文 / 英文 / 中英混合 |
| 规模（首期） | **≥ 100 小时**（质量优先）；50 小时可冒烟 |
| 规模（完整版） | 500–2000 小时（ASR 与定位才显著受益） |
| 说话人多样性 | 越多越好（100+ 说话人） |

**推荐语音语料（wav + txt 现成）**

| 语料 | 语言 | 规模 | 说明 |
|---|---|---|---|
| **AISHELL-1** | 中文 | ~178 h / 400 人 | 干净朗读，入门首选 |
| AISHELL-2 | 中文 | ~1000 h | 更大 |
| WenetSpeech | 中文 | ~10000 h | 大规模，需筛选干净段 |
| **LibriSpeech** | 英文 | 960 h / 2300+ 人 | 经典，多说话人 |
| Common Voice | 多语言 | 按需 | 需筛选 |
| TED-LIUM | 英文 | ~450 h | 演讲 |

**增强（合成时建议开启，提升真实感）**
- 加性噪声：MUSAN、DNS-Challenge 噪声（SNR 10–20 dB 随机）
- 混响：pyroomacoustics 仿真 RIR（可选，二期再上）

**多说话人场景（扩展，本期可选）**：把多条语音叠加到不同方位，answer 按说话人分组输出。

### 2.2 可选方案：真实采集/真实 RIR
- 自录：单麦克风或麦克风阵列房间录音 + 说话人方位（动捕/坐标标定）+ 人工转写。成本高、规模小，仅作**少量评测**用。
- 真实 RIR + 语音语料卷积（用 pyroomacoustics 或公开 RIR 库生成 FOA），比纯 SN3D 更真实，作为合成增强二期引入。

---

## 3. 单声道 ASR 数据（ASR+X 多任务混合，可选但推荐）

**目的**：定向 ASR 样本中 ASR 始终被训练；额外混入纯单声道 ASR 语料可保持/提升转录能力（对应 OSUM 的 ASR+X 数据加权）。

- 任意标准 ASR 语料（上述 §2.1 语料均可直接复用，作为单声道流）。
- 用法：训练时 `--qa-root` 指向多路 QA，其中一路来自纯 ASR 语料（`build_qa` 时 `task_type=asr_only`）。

---

## 4. 评测数据

- **定向 ASR 测试集**：合成时固定随机种子、预留 `valid/test` 拆分（与训练同分布），数量各 1–5 千段。
- **跨条件测试（建议）**：测试时用与训练不同的方位分布 / SNR / 混响 / 说话人，验证泛化。
- 指标：CER/WER（转录）、方位角/仰角 MAE、方向 bin EM、联合准确率（`dasr/evaluate.py`）。

---

## 5. 数据格式契约（脚本）

| 脚本 | 输入 | 输出 |
|---|---|---|
| `dasr/data/build_speech_manifest.py` | 语音语料目录 或 kaldi 风格 `wav.scp`+`text` | `speech_manifest.jsonl` `{key, wav, txt}` |
| `dasr/data/dcase_to_manifest.py` | DCASE `metadata.csv` + `audio/` | `seld_manifest.jsonl`（S0 用） |
| `dasr/data/generate_synthetic_foa.py` | speech manifest | FOA wav + `manifest.jsonl`（含方位） |
| `dasr/data/build_qa.py` | FOA manifest + 提示模板 | `qa/{split}.jsonl`（S1–S3 用） |

**一期数据获取清单（推荐最小组合）**

1. **S0**：TAU Spatial Sound Events **2022 + 2023**（FOA 各一版，转成 SELD manifest，~1–2 万段）
2. **S1–S3**：**AISHELL-1**（中文 178h）或 **LibriSpeech-100h**（英文），跑合成管线
3. **可选**：MUSAN 噪声（合成增强）
4. **评测**：合成管线的 valid/test 拆分 + 少量人工真实录音（可选）

> 拿到以上任意一部分即可启动对应阶段；S0 与定向 ASR 数据相互独立，可并行准备。
