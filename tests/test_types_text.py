import pytest

from voxlink.text import char_tokenize, normalize, word_tokenize
from voxlink.types import FeatureMatrix, Utterance, Waveform


def test_waveform_properties() -> None:
    w = Waveform([0.0] * 8000, 16000)
    assert w.num_samples == 8000
    assert abs(w.duration - 0.5) < 1e-9


def test_waveform_rejects_bad_rate() -> None:
    with pytest.raises(ValueError):
        Waveform([0.0], 0)


def test_feature_matrix_dims() -> None:
    fm = FeatureMatrix([[1.0, 2.0], [3.0, 4.0]], 100.0)
    assert fm.num_frames == 2
    assert fm.num_dims == 2


def test_feature_matrix_empty() -> None:
    fm = FeatureMatrix([], 100.0)
    assert fm.num_frames == 0
    assert fm.num_dims == 0


def test_utterance_defaults() -> None:
    u = Utterance(audio=Waveform([0.0], 16000))
    assert u.text == ""
    assert u.units == []
    assert u.meta == {}


def test_normalize_fullwidth_punct() -> None:
    # NFKC 会把全角标点归一，strip_punct 去除
    assert normalize("Hello，World！") == "hello world"


def test_word_and_char_tokenize() -> None:
    assert word_tokenize("the  cat sat") == ["the", "cat", "sat"]
    assert char_tokenize("语音 识别") == ["语", "音", "识", "别"]


def test_tokenize_empty() -> None:
    assert word_tokenize("") == []
    assert char_tokenize("   ") == []
