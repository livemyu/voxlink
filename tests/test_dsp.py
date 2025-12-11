from voxlink.dsp import (
    frame_signal,
    hamming_window,
    hann_window,
    preemphasis,
    resample,
)


def test_framing_shapes() -> None:
    signal = [float(i) for i in range(100)]
    frames = frame_signal(signal, frame_length=25, frame_shift=10)
    # 1 + ceil((100-25)/10) = 1 + 8 = 9
    assert len(frames) == 9
    assert all(len(f) == 25 for f in frames)


def test_framing_short_signal_pads() -> None:
    frames = frame_signal([1.0, 2.0, 3.0], frame_length=8, frame_shift=4)
    assert len(frames) == 1
    assert frames[0][:3] == [1.0, 2.0, 3.0]
    assert frames[0][3:] == [0.0] * 5


def test_windows_bounds() -> None:
    for win in (hann_window(16), hamming_window(16)):
        assert len(win) == 16
        assert all(0.0 <= w <= 1.01 for w in win)


def test_preemphasis_first_sample_unchanged() -> None:
    out = preemphasis([1.0, 1.0, 1.0], 0.97)
    assert out[0] == 1.0
    assert abs(out[1] - 0.03) < 1e-9


def test_resample_length() -> None:
    up = resample([0.0, 1.0, 0.0, -1.0], 8000, 16000)
    assert len(up) == 8
