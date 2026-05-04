# 使用指南

本指南覆盖 voxlink 的常见用法：特征提取、训练、推理与评估。所有示例都可离线
运行。

## 1. 特征提取

```python
from voxlink import FrontendConfig, Waveform
from voxlink.features import Frontend

frontend = Frontend(FrontendConfig(n_mels=80, n_mfcc=13))
feats = frontend(wave)            # wave: Waveform
print(feats.num_frames, feats.num_dims, feats.frame_rate)
```

可选特征类型：

- `Frontend(..., feature="mfcc")`（默认）：MFCC。
- `Frontend(..., feature="logmel")`：log-mel 频谱。

## 2. 离散语音单元

```python
from voxlink import UnitConfig
from voxlink.units import Quantizer

quant = Quantizer(UnitConfig(n_units=100, seed=0)).fit(all_frames)
units = quant.encode(feats)       # list[int]
quant.save("codebook.json")
```

`UnitConfig.dedup=True` 会折叠相邻的重复单元（游程去重），缩短序列长度。

## 3. 训练一个管线

```python
from voxlink import SpeechLMPipeline, FrontendConfig, UnitConfig, ModelConfig

pipe = SpeechLMPipeline(
    FrontendConfig(n_mfcc=13),
    UnitConfig(n_units=64, seed=0),
    ModelConfig(order=4, smoothing=0.01),
).fit(utterances)                 # utterances: list[Utterance]

pipe.save("model/")
```

## 4. 推理

```python
pipe = SpeechLMPipeline.from_pretrained("model/")

pipe.transcribe(wave)                       # ASR
pipe.answer(wave, "问题文本")                # 语音问答
pipe.follow(wave, "指令文本")                # 指令跟随
pipe.transcribe(wave, beam_size=4)          # 用 beam search
```

## 5. 评估

```python
from voxlink.eval import wer, cer, corpus_metrics

wer("the cat sat", "the cat sit")           # 词错误率
cer("语音识别", "语音时别")                  # 字错误率

m = corpus_metrics(references, hypotheses)  # 语料级聚合
print(m.wer, m.cer, m.num_utts)
```

## 6. 命令行

| 命令 | 说明 |
|------|------|
| `voxlink demo` | 用合成数据端到端演示 |
| `voxlink features audio.wav` | 提取并打印特征信息 |
| `voxlink train --out model/` | 训练码本 + 词表 + 语言模型 |
| `voxlink units audio.wav --model model/` | 打印离散语音单元 |
| `voxlink transcribe audio.wav --model model/` | 转写 |
| `voxlink qa audio.wav "问题" --model model/` | 语音问答 |
| `voxlink eval manifest.jsonl --model model/` | 评估 WER/CER |

manifest 为 JSONL，每行形如：

```json
{"audio": "path/to/a.wav", "text": "one two three"}
```
