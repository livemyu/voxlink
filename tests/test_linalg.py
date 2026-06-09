import pytest

from voxlink.linalg import (
    add,
    argmax,
    dot,
    l2_norm,
    matmul,
    matvec,
    mean,
    scale,
    sq_euclidean,
    sub,
    transpose,
    zeros,
)


def test_dot_and_norm() -> None:
    assert dot([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]) == 32.0
    assert abs(l2_norm([3.0, 4.0]) - 5.0) < 1e-9


def test_add_sub_scale() -> None:
    assert add([1.0, 2.0], [3.0, 4.0]) == [4.0, 6.0]
    assert sub([5.0, 5.0], [1.0, 2.0]) == [4.0, 3.0]
    assert scale([1.0, 2.0], 3.0) == [3.0, 6.0]


def test_sq_euclidean() -> None:
    assert sq_euclidean([0.0, 0.0], [3.0, 4.0]) == 25.0


def test_mean() -> None:
    assert mean([[1.0, 2.0], [3.0, 4.0]]) == [2.0, 3.0]


def test_mean_empty_raises() -> None:
    with pytest.raises(ValueError):
        mean([])


def test_matmul_and_matvec() -> None:
    a = [[1.0, 2.0], [3.0, 4.0]]
    ident = [[1.0, 0.0], [0.0, 1.0]]
    assert matmul(a, ident) == a
    assert matvec(a, [1.0, 1.0]) == [3.0, 7.0]


def test_transpose_and_zeros() -> None:
    assert transpose([[1.0, 2.0, 3.0]]) == [[1.0], [2.0], [3.0]]
    assert zeros(2, 3) == [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]


def test_dot_length_mismatch() -> None:
    with pytest.raises(ValueError):
        dot([1.0], [1.0, 2.0])


def test_argmax() -> None:
    assert argmax([1.0, 5.0, 3.0]) == 1
    assert argmax([2.0, 2.0]) == 0  # 并列取最小下标
    with pytest.raises(ValueError):
        argmax([])


def test_matmul_empty() -> None:
    assert matmul([], []) == []
