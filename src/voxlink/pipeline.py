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


class SpeechLMPipeline:
    """语音接入语言模型的端到端管线。

    Args:
        frontend_config: 特征前端配置。
        unit_config: 离散单元（k-means）配置。
        model_config: 语言模型（n-gram）配置。
    """

    def __init__(
        self,
        frontend_config: FrontendConfig | None = None,
        unit_config: UnitConfig | None = None,
        model_config: ModelConfig | None = None,
    ) -> None:
        self.frontend = Frontend(frontend_config or FrontendConfig())
        self.quantizer = Quantizer(unit_config or UnitConfig())
        self.model_config = model_config or ModelConfig()
        self.vocab: Vocab | None = None
        self.lm: NgramLM | None = None
        self._prompt: PromptBuilder | None = None
        self._tasks: dict[str, TaskSpec] = {"asr": ASR, "qa": QA, "instruct": INSTRUCT}

    # -- 语音侧 --------------------------------------------------------------

    def encode_units(self, wave: Waveform) -> list[int]:
        """把波形编码为离散语音单元序列。"""
        if not self.quantizer.is_fitted:
            raise ModelError("管线尚未训练：码本缺失")
        feats = self.frontend(wave)
        return self.quantizer.encode(feats)
