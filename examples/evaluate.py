"""在合成测试集上评估 WER / CER 的示例。

把合成语料切成训练 / 测试两部分，在训练集上拟合管线，在测试集上报告
字 / 词错误率。展示 voxlink 的评估设施如何串到推理之后。

运行:
    python examples/evaluate.py
"""

from __future__ import annotations

from voxlink import FrontendConfig, ModelConfig, SpeechLMPipeline, UnitConfig
from voxlink.data import synth_utterance
from voxlink.dataset import train_test_split
from voxlink.eval import corpus_metrics

CORPUS = [
    ["one", "two", "three"],
    ["yes", "no"],
    ["up", "down", "up"],
    ["four", "five"],
    ["zero", "one", "two"],
    ["no", "yes", "no"],
    ["three", "four", "five"],
    ["down", "up"],
]


def main() -> None:
    utts = [synth_utterance(s) for s in CORPUS]
    train, test = train_test_split(utts, test_ratio=0.25, seed=1)
    print(f"训练 {len(train)} 条，测试 {len(test)} 条")

    pipe = SpeechLMPipeline(
        FrontendConfig(n_mels=40, n_mfcc=13),
        UnitConfig(n_units=24, seed=0),
        ModelConfig(order=4, smoothing=0.01),
    ).fit(train)

    refs = [u.text for u in test]
    hyps = [pipe.transcribe(u.audio, max_new_tokens=8) for u in test]
    for ref, hyp in zip(refs, hyps, strict=True):
        print(f"  参考: {ref:22} 输出: {hyp}")

    m = corpus_metrics(refs, hyps)
    print(f"\n测试集 WER = {m.wer:.3f}   CER = {m.cer:.3f}")


if __name__ == "__main__":
    main()
