# 架构总览

voxlink 的设计目标是：把「语音接入语言模型」这件事拆成一串**边界清晰、可单独
替换**的组件，每个组件都有一份不依赖第三方库的参考实现。

## 分层

```
┌─────────────────────────────────────────────────────────────┐
│                      SpeechLMPipeline                        │  端到端编排
├───────────────┬───────────────┬───────────────┬─────────────┤
│   Frontend    │   Quantizer   │ PromptBuilder │  NgramLM    │  核心组件
│  (features)   │   (units)     │   (prompt)    │  (ngram)    │
├───────────────┴───────────────┴───────────────┴─────────────┤
│   dsp / fft / melbank / dct / linalg / mathx / _random      │  数值内核
├─────────────────────────────────────────────────────────────┤
│              types / config / exceptions / io               │  基础设施
└─────────────────────────────────────────────────────────────┘
                          backends（可选 numpy 加速）
```

## 组件职责

| 模块 | 职责 |
|------|------|
| `audio` | WAV 读写、PCM 与浮点互转 |
| `dsp` | 预加重、分帧、加窗、重采样 |
| `fft` | DFT / radix-2 FFT / 功率谱 |
| `melbank` / `dct` | 梅尔滤波器组与 DCT-II |
| `features` | 把波形转成 log-mel / MFCC（`Frontend`） |
| `units` / `kmeans` | k-means 码本，把特征离散化为语音单元 |
| `encoder` | 语音编码器与帧下采样 |
| `adapter` | 线性投影 / 单元嵌入，对齐到语言模型空间 |
| `vocab` | 语音单元 + 文本的联合词表 |
| `prompt` | 把单元与任务指令拼成 token 序列 |
| `llm` / `ngram` | 语言模型协议与 n-gram 参考实现 |
| `decode` | 贪心 / 采样 / beam search |
| `eval` | 编辑距离、WER、CER、语料级聚合 |
| `pipeline` | 训练与推理的端到端编排 |

## 数据表示

为了让纯 Python 内核不依赖 numpy，音频与特征统一用嵌套的 `list[float]` 表示：

- `Waveform`：单声道波形（`samples: list[float]` + `sample_rate`）。
- `FeatureMatrix`：帧 × 维度的特征矩阵，附带 `frame_rate` 用于对齐单元。
- `Utterance`：一条样本，含音频、文本、单元与元信息。

安装了 numpy 时，`backends` 会在热点算子（如 k-means 的成对距离）上切换到
向量化实现，但接口与数值结果保持一致。

## 确定性

所有随机性都通过显式的 `Rng`（xorshift32）注入，不使用全局随机状态，因此同样
的种子在任何平台、任何 Python 版本上都给出同样的结果——这是「离线可复现」的
基础，也让整条链路能在 CI 里断言精确输出。
