from voxlink._random import Rng
from voxlink.eval import levenshtein


def _random_seq(rng: Rng, n: int, alphabet: int = 4) -> list[str]:
    return [str(rng.randint(0, alphabet)) for _ in range(n)]


def test_levenshtein_is_symmetric() -> None:
    rng = Rng(11)
    for _ in range(50):
        a = _random_seq(rng, rng.randint(0, 8))
        b = _random_seq(rng, rng.randint(0, 8))
        assert levenshtein(a, b) == levenshtein(b, a)


def test_levenshtein_identity() -> None:
    rng = Rng(22)
    for _ in range(20):
        s = _random_seq(rng, rng.randint(1, 10))
        assert levenshtein(s, s) == 0


def test_levenshtein_triangle_inequality() -> None:
    rng = Rng(33)
    for _ in range(50):
        a = _random_seq(rng, rng.randint(0, 6))
        b = _random_seq(rng, rng.randint(0, 6))
        c = _random_seq(rng, rng.randint(0, 6))
        assert levenshtein(a, c) <= levenshtein(a, b) + levenshtein(b, c)


def test_levenshtein_bounded_by_max_length() -> None:
    rng = Rng(44)
    for _ in range(30):
        a = _random_seq(rng, rng.randint(0, 8))
        b = _random_seq(rng, rng.randint(0, 8))
        assert levenshtein(a, b) <= max(len(a), len(b))
