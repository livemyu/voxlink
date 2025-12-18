import math

from voxlink.audio import float_to_pcm, load_wav, pcm_to_float, save_wav
from voxlink.types import Waveform


def _tone(freq: float, sr: int, n: int) -> list[float]:
    return [0.5 * math.sin(2 * math.pi * freq * t / sr) for t in range(n)]


def test_wav_roundtrip(tmp_path) -> None:
    sr = 16000
    samples = _tone(440.0, sr, sr // 2)
    path = tmp_path / "tone.wav"
    save_wav(str(path), Waveform(samples, sr))
    loaded = load_wav(str(path))

    assert loaded.sample_rate == sr
    assert loaded.num_samples == len(samples)
    # 16-bit 量化误差应很小
    max_err = max(abs(a - b) for a, b in zip(samples, loaded.samples))
    assert max_err < 1e-3


def test_pcm_conversion_symmetry() -> None:
    ints = [0, 100, -100, 32767, -32767]
    floats = pcm_to_float(ints, 16)
    back = float_to_pcm(floats, 16)
    assert back == ints


def test_float_to_pcm_saturates() -> None:
    assert float_to_pcm([2.0], 16) == [32767]
    assert float_to_pcm([-2.0], 16) == [-32767]
