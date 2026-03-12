"""语音问答（Speech QA）示例。

在合成语料上训练管线，并演示给定语音 + 问题文本时的问答接口。参考内核用
n-gram 语言模型，答案来自训练语料的统计规律；重点在于展示「语音单元 + 文本
问题」如何拼进同一个提示喂给语言模型。

运行:
    python examples/qa_demo.py
"""

from __future__ import annotations

from voxlink import FrontendConfig, ModelConfig, SpeechLMPipeline, UnitConfig
from voxlink.data import synth_utterance
from voxlink.prompt import QA
from voxlink.types import Utterance

# 每条样本带一个问题（放在 meta['prompt']）与目标答案（text）。
PAIRS = [
    (["one", "two"], "count", "one two"),
    (["yes"], "reply", "yes"),
    (["no"], "reply", "no"),
    (["up", "down"], "move", "up down"),
]


def main() -> None:
    utts = []
    for words, question, answer in PAIRS:
        u = synth_utterance(words)
        utts.append(Utterance(audio=u.audio, text=answer, meta={"prompt": question}))

    pipe = SpeechLMPipeline(
        FrontendConfig(n_mels=40, n_mfcc=13),
        UnitConfig(n_units=20, seed=0),
        ModelConfig(order=4, smoothing=0.001),
    ).fit(utts, task=QA)

    print("=== 语音问答 ===")
    for (words, question, answer), u in zip(PAIRS, utts, strict=True):
        got = pipe.answer(u.audio, question, max_new_tokens=6)
        print(f"  语音={'/'.join(words):12} 问题={question:8} -> 答案={got!r} (期望 {answer!r})")


if __name__ == "__main__":
    main()
