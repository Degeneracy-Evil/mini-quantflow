# mini-quantflow

教学型量化研究框架，基于 base-py 工程模板。

## 环境

```bash
uv sync --dev
uv run pre-commit install
```

## 运行

```bash
uv run python experiments/01_synthetic_markets.py
```

## 关键约定

- 最低要求 Python 3.12。
- 使用 uv 管理 Python、虚拟环境、依赖和 `uv.lock`，不要手工维护 requirements 文件。
- 源码使用 `src/quantlab` 布局，并支持 `pip install -e .`。
- Ruff 负责 lint、自动修复和格式化，行宽为 120。
- mypy 使用 strict 模式；新增代码必须有完整、有效的类型标注。
- pytest 是唯一测试框架；测试源码放在 `tests/` 下。
- 测试启用分支覆盖率，最低覆盖率为 90%。
- `uv run python scripts/check.py` 是完整质量检查入口，会自动修复 Ruff 问题并格式化文件，然后运行 mypy 和 pytest。
- pre-commit 处理通用文本问题并自动修复 Ruff 问题；若文件被修改，需要重新暂存并再次提交。
- 行尾统一使用 LF。
- CI 与本地使用同一质量检查入口，并额外运行第一个实验作为冒烟测试。
- 不要求维护开发日志文件。

## 修改后的验证

任何代码或工程配置变更完成后，运行：

```bash
uv run python scripts/check.py
uv run python experiments/01_synthetic_markets.py
```
