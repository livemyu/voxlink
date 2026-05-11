# 示例

这些脚本都用内置的确定性合成语料，可直接离线运行：

```bash
python examples/asr_demo.py            # 端到端语音转写
python examples/qa_demo.py             # 语音问答
python examples/instruction_demo.py    # 语音到文本指令跟随
python examples/evaluate.py            # 在合成测试集上评估 WER/CER
python examples/train_codebook.py      # 单独训练并保存一个离散语音码本
```

每个脚本开头的 docstring 说明了它演示的内容。它们都很短，适合作为阅读 voxlink
接口的入口。

> 提示：这些示例使用玩具级别的合成音频与 n-gram 语言模型，目的在于展示**接口与
> 数据流**，而非真实识别性能。
