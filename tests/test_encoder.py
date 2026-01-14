import math

from voxlink.config import FrontendConfig
from voxlink.encoder import (
    MFCCEncoder,
    SpeechEncoder,
    l2_normalize,
    max_pool,
    mean_pool,
    stack_frames,
)
from voxlink.linalg import l2_norm
from voxlink.types import Waveform


def _tone(n: int = 8000, sr: int = 16000) -> Waveform:
    return Waveform([0.3 * math.sin(2 * math.pi * 440 * t / sr) for t in range(n)], sr)


def test_encoder_conforms_to_protocol() -> None:
    enc = MFCCEncoder(FrontendConfig(n_mfcc=13))
    assert isinstance(enc, SpeechEncoder)


def test_encoder_output_dims() -> None:
    enc = MFCCEncoder(FrontendConfig(n_mfcc=13), subsample=2)
    out = enc.encode(_tone())
    assert enc.output_dim == 26
    assert all(len(row) == 26 for row in out.frames)
    assert out.frame_rate == 50.0


def test_stack_frames_halves_length() -> None:
    frames = [[float(i)] for i in range(10)]
    stacked = stack_frames(frames, 2)
    assert len(stacked) == 5
    assert stacked[0] == [0.0, 1.0]


def test_l2_normalize_unit_norm() -> None:
    normed = l2_normalize([[3.0, 4.0]])
    assert abs(l2_norm(normed[0]) - 1.0) < 1e-9


def test_pooling() -> None:
    frames = [[1.0, 2.0], [3.0, 0.0]]
    assert mean_pool(frames) == [2.0, 1.0]
    assert max_pool(frames) == [3.0, 2.0]
