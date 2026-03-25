"""训练并保存一个自定义离散语音码本的示例。

单独展示「离散化」这一环：从若干音频提取 MFCC 特征，训练一个 k-means
码本（即语音单元词表），保存到磁盘并重新加载，验证编码结果一致。

运行:
    python examples/train_codebook.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from voxlink import FrontendConfig, UnitConfig
from voxlink.data import synth_utterance
from voxlink.features import Frontend
from voxlink.units import Quantizer

WORDS = [["one", "two", "three"], ["four", "five"], ["yes", "no"], ["up", "down"]]


def main() -> None:
    frontend = Frontend(FrontendConfig(n_mels=40, n_mfcc=13))
    utts = [synth_utterance(s) for s in WORDS]

    # 汇集所有特征帧用于训练码本
    frames = []
    feats = []
    for u in utts:
        fm = frontend(u.audio)
        feats.append(fm)
        frames.extend(fm.frames)

    quant = Quantizer(UnitConfig(n_units=16, seed=0)).fit(frames)
    print(f"码本大小: {quant.n_units}，特征帧数: {len(frames)}")

    units0 = quant.encode(feats[0])
    print(f"第一条音频的离散单元: {units0}")

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "codebook.json"
        quant.save(str(path))
        reloaded = Quantizer.load(str(path))
        assert reloaded.encode(feats[0]) == units0
        print("保存并重新加载后编码一致 ✓")


if __name__ == "__main__":
    main()
