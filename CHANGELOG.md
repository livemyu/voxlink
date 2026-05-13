# 更新日志

本项目遵循 [语义化版本](https://semver.org/lang/zh-CN/)，格式参考
[Keep a Changelog](https://keepachangelog.com/zh-CN/1.0.0/)。

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

[0.1.0]: https://github.com/livemyu/voxlink/releases/tag/v0.1.0
