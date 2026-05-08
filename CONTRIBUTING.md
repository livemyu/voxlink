# 贡献指南

欢迎为 voxlink 贡献代码、文档或想法。

## 开发环境

项目使用 [uv](https://github.com/astral-sh/uv) 管理依赖：

```bash
uv sync --group dev --extra numpy
```

常用命令（见 `Makefile`）：

```bash
make lint      # ruff 检查
make fmt       # 自动格式化
make type      # mypy 类型检查
make test      # 运行测试
make cov       # 带覆盖率
make check     # lint + type + test（等价 CI）
```

## 提交前检查

请确保 `make check` 全绿：

- `ruff check` 与 `ruff format --check` 通过；
- `mypy` 无错误（strict 模式）；
- 测试通过，新增行为有对应测试。

## 代码风格

- 纯 Python 内核**不引入**新的第三方运行时依赖；加速实现放在 `backends`，并保持
  与参考实现数值一致。
- 公开函数写类型注解与 docstring；注释只解释「为什么」，不复述「做什么」。
- 随机性通过显式 `Rng` 注入，不使用全局随机状态，保证可复现。

## 提交信息

推荐使用约定式提交（Conventional Commits），例如 `feat(units): ...`、
`fix(features): ...`、`docs: ...`。一个提交只做一件事。

## 报告问题

请用 issue 模板，附上 voxlink 版本、Python 版本、可复现步骤与最小样例。
