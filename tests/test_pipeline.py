from voxlink import FrontendConfig, ModelConfig, SpeechLMPipeline, UnitConfig
from voxlink.data import synth_utterance
from voxlink.exceptions import ModelError
from voxlink.prompt import QA
from voxlink.types import Utterance

SENTENCES = [
    ["one", "two", "three"],
    ["yes", "no"],
    ["up", "down", "up"],
    ["four", "five"],
]


def _fit_pipe() -> SpeechLMPipeline:
    utts = [synth_utterance(s) for s in SENTENCES]
    return SpeechLMPipeline(
        FrontendConfig(n_mels=40, n_mfcc=13),
        UnitConfig(n_units=20, seed=0),
        ModelConfig(order=4, smoothing=0.001),
    ).fit(utts)


def test_asr_overfits_training() -> None:
    pipe = _fit_pipe()
    utts = [synth_utterance(s) for s in SENTENCES]
    for u in utts:
        assert pipe.transcribe(u.audio, max_new_tokens=8) == u.text


def test_transcribe_before_fit_raises() -> None:
    pipe = SpeechLMPipeline()
    wave = synth_utterance(["one"]).audio
    try:
        pipe.transcribe(wave)
    except ModelError:
        return
    raise AssertionError("未训练的管线转写应抛出 ModelError")


def test_encode_units_is_deterministic() -> None:
    pipe = _fit_pipe()
    wave = synth_utterance(["one", "two"]).audio
    assert pipe.encode_units(wave) == pipe.encode_units(wave)


def test_qa_pipeline_returns_expected() -> None:
    utts = [
        Utterance(audio=synth_utterance(["yes"]).audio, text="yes", meta={"prompt": "reply"}),
        Utterance(audio=synth_utterance(["no"]).audio, text="no", meta={"prompt": "reply"}),
    ]
    pipe = SpeechLMPipeline(
        FrontendConfig(n_mels=40, n_mfcc=13),
        UnitConfig(n_units=12, seed=0),
        ModelConfig(order=4, smoothing=0.001),
    ).fit(utts, task=QA)
    ans = pipe.answer(utts[0].audio, "reply", max_new_tokens=4)
    assert isinstance(ans, str)


def test_save_load_roundtrip(tmp_path) -> None:
    pipe = _fit_pipe()
    pipe.save(str(tmp_path / "m"))
    restored = SpeechLMPipeline.from_pretrained(str(tmp_path / "m"))
    wave = synth_utterance(["one", "two", "three"]).audio
    assert restored.transcribe(wave, max_new_tokens=8) == pipe.transcribe(wave, max_new_tokens=8)


def test_transcribe_batch() -> None:
    pipe = _fit_pipe()
    waves = [synth_utterance(s).audio for s in SENTENCES]
    outs = pipe.transcribe_batch(waves, max_new_tokens=8)
    assert len(outs) == len(waves)
