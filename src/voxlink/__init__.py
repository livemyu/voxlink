"""voxlink: 离线优先的语音大模型工具包。

把语音编码器 / 离散语音 token 接入语言模型，实现语音理解、语音问答与
语音到文本的指令跟随。核心内核为纯 Python 实现。
"""

from __future__ import annotations

from .config import FrontendConfig, ModelConfig, UnitConfig
from .exceptions import (
    AudioError,
    ConfigError,
    FeatureError,
    ModelError,
    VoxlinkError,
)
from .pipeline import SpeechLMPipeline
from .types import FeatureMatrix, Utterance, Waveform

__version__ = "0.3.0"


def pipeline(directory: str) -> SpeechLMPipeline:
    """便捷入口：从已保存的目录加载一个 :class:`SpeechLMPipeline`。"""
    return SpeechLMPipeline.from_pretrained(directory)


__all__ = [
    "AudioError",
    "ConfigError",
    "FeatureError",
    "FeatureMatrix",
    "FrontendConfig",
    "ModelConfig",
    "ModelError",
    "SpeechLMPipeline",
    "UnitConfig",
    "Utterance",
    "VoxlinkError",
    "Waveform",
    "__version__",
    "pipeline",
]
