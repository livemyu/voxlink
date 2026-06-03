import pytest

from voxlink.adapter import LengthAdaptor, LinearProjector, UnitEmbedding
from voxlink.exceptions import ModelError


def test_projector_output_dim() -> None:
    proj = LinearProjector(13, 8, seed=0)
    out = proj([[0.1] * 13, [0.2] * 13])
    assert len(out) == 2
    assert all(len(row) == 8 for row in out)


def test_projector_is_deterministic() -> None:
    a = LinearProjector(4, 4, seed=42)
    b = LinearProjector(4, 4, seed=42)
    assert a.weight == b.weight


def test_projector_different_seeds_differ() -> None:
    a = LinearProjector(4, 4, seed=1)
    b = LinearProjector(4, 4, seed=2)
    assert a.weight != b.weight


def test_unit_embedding_lookup() -> None:
    emb = UnitEmbedding(n_units=10, dim=6, seed=0)
    rows = emb([0, 1, 0])
    assert rows[0] == rows[2]  # 同一单元查表结果一致
    assert len(rows[0]) == 6


def test_unit_embedding_out_of_range() -> None:
    emb = UnitEmbedding(n_units=4, dim=3)
    with pytest.raises(ModelError):
        emb.lookup(4)


def test_length_adaptor_shortens_sequence() -> None:
    adaptor = LengthAdaptor(in_dim=4, out_dim=8, stride=2, seed=0)
    frames = [[float(i)] * 4 for i in range(10)]
    out = adaptor(frames)
    assert len(out) == 5
    assert all(len(row) == 8 for row in out)


def test_length_adaptor_rejects_bad_stride() -> None:
    with pytest.raises(ModelError):
        LengthAdaptor(4, 8, stride=0)
