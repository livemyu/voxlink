from voxlink._random import Rng
from voxlink.config import UnitConfig
from voxlink.kmeans import KMeans
from voxlink.types import FeatureMatrix
from voxlink.units import Quantizer, dedup_units
from voxlink.vocab import SpecialTokens, Vocab


def _blobs(seed: int = 0) -> list[list[float]]:
    rng = Rng(seed)
    data: list[list[float]] = []
    for cx, cy in [(0.0, 0.0), (6.0, 6.0), (-6.0, 6.0)]:
        for _ in range(30):
            data.append([cx + rng.gauss(0, 0.25), cy + rng.gauss(0, 0.25)])
    return data


def test_kmeans_separates_clusters() -> None:
    data = _blobs()
    km = KMeans(3, seed=0).fit(data)
    labels = km.predict(data)
    # 每个 30 点的 blob 应落入同一簇
    for start in (0, 30, 60):
        block = labels[start : start + 30]
        assert len(set(block)) == 1


def test_kmeans_inertia_decreases_with_k() -> None:
    data = _blobs()
    small = KMeans(2, seed=0).fit(data).inertia_
    large = KMeans(4, seed=0).fit(data).inertia_
    assert large < small


def test_dedup_units() -> None:
    assert dedup_units([1, 1, 2, 2, 2, 3, 1, 1]) == [1, 2, 3, 1]
    assert dedup_units([]) == []


def test_codebook_roundtrip(tmp_path) -> None:
    data = _blobs()
    q = Quantizer(UnitConfig(n_units=3, dedup=False)).fit(data)
    path = tmp_path / "cb.json"
    q.save(str(path))
    q2 = Quantizer.load(str(path))
    fm = FeatureMatrix(data, 100.0)
    assert q2.encode(fm) == q.encode(fm)


def test_vocab_roundtrip() -> None:
    tokens = [*SpecialTokens.ALL, "u_0", "u_1", "hello"]
    v = Vocab(tokens)
    assert v.decode(v.encode(["hello", "u_1"])) == ["hello", "u_1"]
    # 未登录 token 回退到 <unk>
    assert v.token(v.id("nope")) == SpecialTokens.UNK
