"""端到端管线：把前端、量化器、提示词与语言模型串成可训练、可推理的整体。

训练流程（:meth:`SpeechLMPipeline.fit`）：

1. 用特征前端把每条音频转成特征；
2. 在所有特征帧上训练 k-means 码本，得到离散语音单元；
3. 由「语音单元 + 文本词 + 任务标记 + 特殊符号」构造联合词表；
4. 对每条样本拼出「提示 + 目标文本」的完整 token 序列，喂给 n-gram 语言模型。

推理流程（:meth:`transcribe` / :meth:`answer` / :meth:`follow`）：构造提示后，
用解码器在语言模型上自回归生成答案 token，再还原为文本。

这条链路刻意用轻量的参考组件（纯 Python k-means + n-gram），从而完全离线、
确定性、可在 CI 中端到端跑通；真实系统可在保持同样接口的前提下替换任一环节。
"""

from __future__ import annotations

from .config import FrontendConfig, ModelConfig, UnitConfig
from .decode import beam_search, greedy_decode
from .exceptions import ModelError
from .features import Frontend
from .ngram import NgramLM
from .prompt import ASR, INSTRUCT, QA, PromptBuilder, TaskSpec
from .text import word_tokenize
from .types import Utterance, Waveform
from .units import Quantizer
from .vocab import SpecialTokens, Vocab, build_joint_vocab

__all__ = ["SpeechLMPipeline"]
