# 更新日志

本项目遵循 [语义化版本](https://semver.org/lang/zh-CN/)，格式参考
[Keep a Changelog](https://keepachangelog.com/zh-CN/1.0.0/)。

## [0.3.0] - 未发布

### 新增

- 长度适配器 `LengthAdaptor`：拼帧 + 投影，缩短语音序列长度。
- `SpeechLMPipeline.is_trained` 属性与可读的 `repr`。
- 评估结果的一行摘要 `CorpusMetrics.format()`。
- FFT 微基准脚本 `benchmarks/bench_fft.py`。

### 变更

- 采样逻辑统一到 `mathx.sample_categorical`，解码器与语言模型共用。

## [0.2.0] - 未发布

### 新增

- 顶层便捷函数 `voxlink.pipeline(dir)`，一行加载已保存的管线。
- 可选 numpy 加速后端（`voxlink.backends`），k-means 分配步骤自动向量化。
- 语音编码器 `MFCCEncoder` 与模态适配器（线性投影 / 单元嵌入）。

### 修复

- 梅尔滤波器组在小 `n_fft` 下可能出现全零行，现强制 bin 单调递增。
- 合成音频的谐波叠加峰值可能超过 1.0，写 WAV 时会削顶，现统一归一化。

## [0.1.0] - 未发布

### 新增

- 纯 Python 声学特征前端：预加重、分帧、FFT、log-mel、MFCC、CMVN。
- 基于 k-means 的离散语音单元量化器，支持码本保存 / 加载与游程去重。
- 带回退的 n-gram 语言模型（add-k 平滑），实现 `LanguageModel` 协议。
- 解码器：贪心、温度采样、top-k、beam search。
- 端到端管线 `SpeechLMPipeline`：语音转写 / 问答 / 指令跟随。
- 评估：编辑距离、WER、CER 与语料级聚合。
- 命令行 `voxlink`：demo / features / train / units / transcribe / qa / eval。
- 确定性合成语音数据，便于离线演示与测试。

[0.3.0]: https://github.com/livemyu/voxlink/releases/tag/v0.3.0
[0.2.0]: https://github.com/livemyu/voxlink/releases/tag/v0.2.0
[0.1.0]: https://github.com/livemyu/voxlink/releases/tag/v0.1.0
