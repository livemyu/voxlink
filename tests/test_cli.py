import voxlink.cli as cli
from voxlink.audio import save_wav
from voxlink.data import synth_utterance
from voxlink.io_utils import write_jsonl


def test_cli_demo(capsys) -> None:
    rc = cli.main(["demo"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "WER=" in out


def test_cli_version(capsys) -> None:
    try:
        cli.main(["--version"])
    except SystemExit as exc:
        assert exc.code == 0
    out = capsys.readouterr().out
    assert "voxlink" in out


def _make_manifest(tmp_path) -> str:
    recs = []
    for i, s in enumerate([["one", "two"], ["yes", "no"], ["up", "down"]]):
        u = synth_utterance(s)
        p = tmp_path / f"{i}.wav"
        save_wav(str(p), u.audio)
        recs.append({"audio": str(p), "text": u.text})
    manifest = tmp_path / "manifest.jsonl"
    write_jsonl(str(manifest), recs)
    return str(manifest)


def test_cli_train_and_transcribe(tmp_path, capsys) -> None:
    manifest = _make_manifest(tmp_path)
    model_dir = tmp_path / "model"
    assert (
        cli.main(
            [
                "train",
                "--manifest",
                manifest,
                "--out",
                str(model_dir),
                "--n-units",
                "12",
                "--order",
                "4",
            ]
        )
        == 0
    )
    capsys.readouterr()
    # 转写第一条音频
    audio = str(tmp_path / "0.wav")
    assert cli.main(["transcribe", audio, "--model", str(model_dir)]) == 0
    out = capsys.readouterr().out.strip()
    assert out  # 有输出


def test_cli_eval(tmp_path, capsys) -> None:
    manifest = _make_manifest(tmp_path)
    model_dir = tmp_path / "model"
    cli.main(["train", "--manifest", manifest, "--out", str(model_dir), "--n-units", "12"])
    capsys.readouterr()
    assert cli.main(["eval", manifest, "--model", str(model_dir)]) == 0
    assert "WER=" in capsys.readouterr().out
