from voxlink.audio import save_wav
from voxlink.data import synth_utterance
from voxlink.dataset import load_manifest, train_test_split
from voxlink.io_utils import read_json, write_json, write_jsonl


def test_load_manifest_reads_audio(tmp_path) -> None:
    recs = []
    for i, s in enumerate([["one", "two"], ["yes", "no"]]):
        u = synth_utterance(s)
        p = tmp_path / f"{i}.wav"
        save_wav(str(p), u.audio)
        recs.append({"audio": str(p), "text": u.text, "spk": f"s{i}"})
    manifest = tmp_path / "m.jsonl"
    write_jsonl(str(manifest), recs)

    utts = load_manifest(str(manifest))
    assert len(utts) == 2
    assert utts[0].text == "one two"
    assert utts[0].audio.num_samples > 0
    assert utts[0].meta["spk"] == "s0"


def test_load_manifest_without_audio(tmp_path) -> None:
    manifest = tmp_path / "m.jsonl"
    write_jsonl(str(manifest), [{"text": "hello", "sample_rate": 8000}])
    utts = load_manifest(str(manifest), load_audio=False)
    assert utts[0].audio.num_samples == 0
    assert utts[0].audio.sample_rate == 8000


def test_train_test_split_ratio(tmp_path) -> None:
    utts = [synth_utterance([w]) for w in ["one", "two", "three", "four", "five"]]
    train, test = train_test_split(utts, test_ratio=0.4, seed=0)
    assert len(test) == 2
    assert len(train) == 3


def test_json_roundtrip(tmp_path) -> None:
    path = tmp_path / "x.json"
    obj = {"a": [1, 2, 3], "text": "你好"}
    write_json(str(path), obj)
    assert read_json(str(path)) == obj
