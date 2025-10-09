"""voxlink: 离线优先的语音大模型工具包。

把语音编码器 / 离散语音 token 接入语言模型，实现语音理解、语音问答与
语音到文本的指令跟随。核心内核为纯 Python 实现。
"""

__version__ = "0.1.0"

# 计划中的公开子模块（随后逐步实现）：
#   audio      —— WAV 读写与帧切分
#   features   —— FFT / log-mel / MFCC 前端
#   units      —— k-means 离散语音 token
#   encoder    —— 语音编码器
#   adapter    —— 模态投影 / 长度适配
#   llm        —— 语言模型协议与 n-gram 参考实现
#   decode     —— 解码策略
#   eval       —— WER / CER 评估
#   pipeline   —— 端到端推理管线

__all__ = ["__version__"]
