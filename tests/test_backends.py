from voxlink.backends import (
    NumpyBackend,
    PurePythonBackend,
    available_backends,
    get_backend,
)
from voxlink._compat import has_numpy


def test_pure_python_backend_always_available() -> None:
    assert "python" in available_backends()
    b = get_backend(prefer_numpy=False)
    assert isinstance(b, PurePythonBackend)


def test_matmul_reference() -> None:
    b = PurePythonBackend()
    out = b.matmul([[1.0, 2.0], [3.0, 4.0]], [[1.0, 0.0], [0.0, 1.0]])
    assert out == [[1.0, 2.0], [3.0, 4.0]]


def test_pairwise_sq_dist() -> None:
    b = PurePythonBackend()
    d = b.pairwise_sq_dist([[0.0, 0.0]], [[3.0, 4.0]])
    assert abs(d[0][0] - 25.0) < 1e-9


def test_numpy_matches_python_when_available() -> None:
    if not has_numpy():
        return
    pp = PurePythonBackend()
    nb = NumpyBackend()
    pts = [[0.0, 0.0], [1.0, 2.0], [3.0, 1.0]]
    ctr = [[0.0, 0.0], [2.0, 2.0]]
    a = pp.pairwise_sq_dist(pts, ctr)
    b = nb.pairwise_sq_dist(pts, ctr)
    for r1, r2 in zip(a, b, strict=True):
        for x, y in zip(r1, r2, strict=True):
            assert abs(x - y) < 1e-9
