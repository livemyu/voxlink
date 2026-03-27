from voxlink._random import Rng


def test_rng_is_reproducible() -> None:
    a = [Rng(42).random() for _ in range(5)]
    b = [Rng(42).random() for _ in range(5)]
    # 两个同种子生成器的第一个抽样应一致
    assert Rng(42).random() == Rng(42).random()
    # 序列可复现
    ra, rb = Rng(7), Rng(7)
    assert [ra.random() for _ in range(10)] == [rb.random() for _ in range(10)]


def test_rng_range() -> None:
    r = Rng(0)
    for _ in range(1000):
        x = r.random()
        assert 0.0 <= x < 1.0


def test_randint_bounds() -> None:
    r = Rng(3)
    for _ in range(500):
        v = r.randint(2, 5)
        assert 2 <= v < 5


def test_shuffle_is_permutation(rng: Rng) -> None:
    items = list(range(20))
    original = set(items)
    rng.shuffle(items)
    assert set(items) == original
    assert len(items) == 20


def test_different_seeds_diverge() -> None:
    assert Rng(1).random() != Rng(2).random()
