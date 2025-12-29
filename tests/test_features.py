import cmath
import math

from voxlink.config import FrontendConfig
from voxlink.dct import dct2
from voxlink.features import Frontend, log_mel_spectrogram, mfcc
from voxlink.fft import dft, fft, rfft_power
from voxlink.melbank import hz_to_mel, mel_filterbank, mel_to_hz
from voxlink.types import Waveform


def _tone(sr: int, n: int) -> Waveform:
    s = [0.3 * math.sin(2 * math.pi * 440 * t / sr) for t in range(n)]
    return Waveform(s, sr)


def test_fft_matches_dft() -> None:
    signal = [1.0, 2.0, -1.0, 0.5, 0.0, -2.0, 3.0, 1.0]
    fast = fft(signal)
    slow = dft(signal)
    for a, b in zip(fast, slow, strict=True):
        assert cmath.isclose(a, b, abs_tol=1e-9)


def test_mel_scale_roundtrip() -> None:
    for hz in (100.0, 1000.0, 4000.0):
        assert abs(mel_to_hz(hz_to_mel(hz)) - hz) < 1e-6


def test_mel_filterbank_nonnegative_and_shaped() -> None:
    fb = mel_filterbank(n_mels=20, n_fft=512, sample_rate=16000, fmin=20.0, fmax=8000.0)
    assert len(fb) == 20
    assert all(len(row) == 512 // 2 + 1 for row in fb)
    assert all(w >= 0.0 for row in fb for w in row)


def test_mfcc_shape() -> None:
    cfg = FrontendConfig(n_mels=40, n_mfcc=13)
    feats = mfcc(_tone(16000, 8000), cfg)
    assert feats
    assert all(len(f) == 13 for f in feats)


def test_frontend_dims() -> None:
    fe = Frontend(FrontendConfig(n_mels=40, n_mfcc=13))
    fm = fe(_tone(16000, 8000))
    assert fm.num_dims == 13
    assert fm.frame_rate == 100.0


def test_logmel_dims() -> None:
    cfg = FrontendConfig(n_mels=23)
    lm = log_mel_spectrogram(_tone(16000, 8000), cfg)
    assert all(len(row) == 23 for row in lm)


def test_dct_zero_input() -> None:
    assert dct2([0.0, 0.0, 0.0, 0.0], 3) == [0.0, 0.0, 0.0]
