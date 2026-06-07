import pytest

from voxlink.dsp import (
    apply_window,
    frame_signal,
    hamming_window,
    hann_window,
    resample,
)


def test_frame_signal_empty() -> None:
    assert frame_signal([], 4, 2) == []


def test_frame_signal_invalid_params() -> None:
    with pytest.raises(ValueError):
        frame_signal([1.0, 2.0], 0, 2)


def test_windows_single_sample() -> None:
    assert hann_window(1) == [1.0]
    assert hamming_window(1) == [1.0]


def test_window_invalid_size() -> None:
    with pytest.raises(ValueError):
        hann_window(0)
    with pytest.raises(ValueError):
        hamming_window(0)


def test_apply_window() -> None:
    assert apply_window([1.0, 2.0, 3.0], [1.0, 0.5, 0.0]) == [1.0, 1.0, 0.0]


def test_resample_identity_and_empty() -> None:
    assert resample([1.0, 2.0], 16000, 16000) == [1.0, 2.0]
    assert resample([], 8000, 16000) == []


def test_resample_downsample_shortens() -> None:
    signal = [float(i) for i in range(100)]
    down = resample(signal, 16000, 8000)
    assert len(down) < len(signal)


def test_resample_invalid_rate() -> None:
    with pytest.raises(ValueError):
        resample([1.0], 0, 16000)
