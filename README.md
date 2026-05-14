# voxlink

[![CI](https://github.com/livemyu/voxlink/actions/workflows/ci.yml/badge.svg)](https://github.com/livemyu/voxlink/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)
[![Checked with mypy](https://img.shields.io/badge/mypy-strict-blue)](https://mypy-lang.org/)
[![Ruff](https://img.shields.io/badge/lint-ruff-000000)](https://github.com/astral-sh/ruff)

> 把语音编码器 / 离散语音 token 接入大语言模型的离线优先工具包。

`voxlink` 是一个用于 **语音理解、语音问答与语音到文本指令跟随** 的参考实现。
它把整条「语音 → 离散单元 → 语言模型 → 文本」的链路拆成清晰、可替换的组件，
并给出一套 **纯 Python** 的参考内核：无强制运行时依赖，`numpy` / `torch` 后端
可选。所有组件都是确定性的，能够完全离线运行，也因此可以在 CI 中端到端复现。

## 为什么做这个

主流的语音大模型工作大多依赖庞大的预训练权重和 GPU，难以在受限环境里复现或
教学。voxlink 走另一条路：用最小但**真实**的算法（FFT / log-mel / MFCC、k-means
离散化、n-gram 语言模型、beam search、WER/CER）把关键思想串起来，让你能在一台
没有网络、没有 GPU 的机器上，看清「语音是怎么接进语言模型的」。

真实系统可以在保持同样接口的前提下，把任一环节替换成更强的实现（例如把
n-gram 语言模型换成基于 PyTorch 的解码器）。

## 核心链路

```
波形 ──► 特征前端 ──► 离散语音单元 ──► 提示词拼装 ──► 语言模型 ──► 解码 ──► 文本
       (MFCC)      (k-means 码本)   (单元+指令)     (n-gram)    (贪心/beam)
```

## 安装

```bash
pip install voxlink              # 纯 Python 内核
pip install 'voxlink[numpy]'     # 可选 numpy 加速
```

## 快速上手

```python
from voxlink import SpeechLMPipeline, FrontendConfig, UnitConfig, ModelConfig
from voxlink.data import synth_utterance

# 用内置的确定性合成语料训练一个管线
utts = [synth_utterance(s) for s in [["one", "two", "three"], ["yes", "no"]]]
pipe = SpeechLMPipeline(
    FrontendConfig(n_mfcc=13),
    UnitConfig(n_units=24, seed=0),
    ModelConfig(order=4),
).fit(utts)

print(pipe.transcribe(utts[0].audio))   # -> "one two three"
```

命令行同样可用：

```bash
voxlink demo                                   # 端到端演示
voxlink train --out model/                     # 训练并保存
voxlink transcribe audio.wav --model model/    # 转写
voxlink eval manifest.jsonl --model model/     # 评估 WER/CER
```

## 许可证

Apache-2.0，详见 [LICENSE](LICENSE)。
