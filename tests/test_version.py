from pathlib import Path

from voxlink import __version__


def test_version_is_string() -> None:
    assert isinstance(__version__, str)
    assert __version__.count(".") == 2


def test_version_matches_pyproject() -> None:
    text = (Path(__file__).resolve().parent.parent / "pyproject.toml").read_text(encoding="utf-8")
    assert f'version = "{__version__}"' in text
