import logging

from voxlink._compat import has_numpy, has_torch, require
from voxlink.logsetup import configure, get_logger


def test_backend_probes_return_bool() -> None:
    assert isinstance(has_numpy(), bool)
    assert isinstance(has_torch(), bool)


def test_require_unknown_backend() -> None:
    try:
        require("nonsense")
    except ValueError:
        return
    raise AssertionError("未知后端应抛 ValueError")


def test_require_torch_when_absent() -> None:
    if has_torch():
        return  # 已安装则跳过
    try:
        require("torch")
    except ImportError as exc:
        assert "torch" in str(exc)
        return
    raise AssertionError("缺少 torch 应抛 ImportError")


def test_get_logger_namespace() -> None:
    log = get_logger("cli")
    assert log.name == "voxlink.cli"
    assert get_logger().name == "voxlink"


def test_configure_is_idempotent() -> None:
    configure(logging.DEBUG)
    root = logging.getLogger("voxlink")
    n = len(root.handlers)
    configure()  # 再次调用不应重复添加 handler
    assert len(root.handlers) == n
