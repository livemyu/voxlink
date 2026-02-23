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

    # -- 训练 ----------------------------------------------------------------

    def fit(
        self,
        utterances: list[Utterance],
        *,
        task: TaskSpec = ASR,
    ) -> SpeechLMPipeline:
        """在成对的（音频, 文本）数据上训练码本、词表与语言模型。"""
        if not utterances:
            raise ModelError("训练集为空")

        # 1) 训练码本：汇集所有特征帧
        all_frames = []
        feat_cache = []
        for utt in utterances:
            fm = self.frontend(utt.audio)
            feat_cache.append(fm)
            all_frames.extend(fm.frames)
        self.quantizer.fit(all_frames)

        # 2) 构造联合词表：语音单元 + 文本词 + 任务标记
        text_words: list[str] = []
        for utt in utterances:
            text_words.extend(word_tokenize(utt.text))
        task_tokens = [spec.token() for spec in self._tasks.values()]
        self.vocab = build_joint_vocab(
            self.quantizer.n_units, text_words, task_tokens=task_tokens
        )
        self._prompt = PromptBuilder(self.vocab)

        # 3) 拼「提示 + 目标文本」，训练 n-gram 语言模型
        self.lm = NgramLM(len(self.vocab), self.model_config)
        sequences = []
        for utt, fm in zip(utterances, feat_cache, strict=True):
            units = self.quantizer.encode(fm)
            prompt_ids = self._prompt.build(units, task, self._task_text(task, utt))
            answer_ids = self.vocab.encode([*word_tokenize(utt.text), SpecialTokens.EOS])
            sequences.append(prompt_ids + answer_ids)
        self.lm.fit(sequences)
        return self

    @staticmethod
    def _task_text(task: TaskSpec, utt: Utterance) -> str:
        """QA / 指令任务从 meta 里取问题 / 指令文本。"""
        if not task.needs_text:
            return ""
        return utt.meta.get("prompt", "")
