import math

import pytest

from voxlink.audio import float_to_pcm, load_wav, pcm_to_float, save_wav
from voxlink.dsp import resample
from voxlink.exceptions import AudioError
from voxlink.features import power_spectrogram
from voxlink.config import FrontendConfig
from voxlink.types import Waveform


def _tone(sr: int, n: int, freq: float = 440.0) -> list[float]:
    return [0.3 * math.sin(2 * math.pi * freq * t / sr) for t in range(n)]


@pytest.mark.parametrize("bits", [8, 16, 24, 32])
def test_wav_roundtrip_all_bit_depths(tmp_path, bits: int) -> None:
    sr = 16000
    samples = _tone(sr, 2000)
    path = tmp_path / f"t{bits}.wav"
    save_wav(str(path), Waveform(samples, sr), bits=bits)
    loaded = load_wav(str(path))
    assert loaded.sample_rate == sr
    assert loaded.num_samples == len(samples)
    # 8-bit 量化较粗，放宽阈值
    tol = 0.05 if bits == 8 else 1e-2
    assert max(abs(a - b) for a, b in zip(samples, loaded.samples, strict=True)) < tol


def test_load_missing_file_raises(tmp_path) -> None:
    with pytest.raises(AudioError):
        load_wav(str(tmp_path / "nope.wav"))


def test_unsupported_bits_raises() -> None:
    with pytest.raises(AudioError):
        pcm_to_float([1, 2, 3], 12)
    with pytest.raises(AudioError):
        float_to_pcm([0.1], 12)


def test_resample_ratio_preserves_duration() -> None:
    sr = 8000
    samples = _tone(sr, sr)  # 1 秒
    up = resample(samples, sr, 16000)
    # 时长应保持 ~1 秒
    assert abs(len(up) / 16000 - 1.0) < 0.01


def test_stft_frame_dims() -> None:
    cfg = FrontendConfig(n_fft=512)
    spec = power_spectrogram(Waveform(_tone(16000, 8000), 16000), cfg)
    assert all(len(frame) == 512 // 2 + 1 for frame in spec)
    assert all(v >= 0.0 for frame in spec for v in frame)
