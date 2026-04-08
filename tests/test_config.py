import pytest

from voxlink.config import FrontendConfig, ModelConfig, UnitConfig
from voxlink.exceptions import ConfigError


def test_frontend_frame_sizes() -> None:
    cfg = FrontendConfig(sample_rate=16000, frame_length_ms=25.0, frame_shift_ms=10.0)
    assert cfg.frame_length == 400
    assert cfg.frame_shift == 160
    assert cfg.frame_rate == 100.0


def test_frontend_rejects_non_pow2_nfft() -> None:
    with pytest.raises(ConfigError):
        FrontendConfig(n_fft=500)


def test_frontend_rejects_bad_mfcc() -> None:
    with pytest.raises(ConfigError):
        FrontendConfig(n_mels=13, n_mfcc=20)


def test_unit_config_validation() -> None:
    with pytest.raises(ConfigError):
        UnitConfig(n_units=1)


def test_model_config_validation() -> None:
    with pytest.raises(ConfigError):
        ModelConfig(order=0)
    with pytest.raises(ConfigError):
        ModelConfig(smoothing=0.0)


def test_config_to_dict_roundtrip() -> None:
    cfg = FrontendConfig(n_mfcc=20, n_mels=40)
    restored = FrontendConfig(**cfg.to_dict())
    assert restored == cfg
