.PHONY: help install lint fmt type test cov check clean

help:
	@echo "voxlink 开发命令："
	@echo "  make install   安装开发依赖"
	@echo "  make lint      运行 ruff 检查"
	@echo "  make fmt       运行 ruff 自动格式化"
	@echo "  make type      运行 mypy 类型检查"
	@echo "  make test      运行测试"
	@echo "  make cov       运行测试并生成覆盖率"
	@echo "  make check     lint + type + test（CI 等价）"

install:
	uv sync --group dev --extra numpy

lint:
	uv run --group dev ruff check .

fmt:
	uv run --group dev ruff format .
	uv run --group dev ruff check --fix .

type:
	uv run --group dev mypy

test:
	uv run --group dev --extra numpy pytest

cov:
	uv run --group dev --extra numpy pytest --cov=voxlink --cov-report=term-missing

check: lint
	uv run --group dev ruff format --check .
	uv run --group dev mypy
	uv run --group dev --extra numpy pytest

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov coverage.xml .coverage dist build
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
