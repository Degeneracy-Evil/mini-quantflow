# Mini Quant Research Lab

一个小而完整、源码可读的教学型量化研究框架。它展示历史市场数据如何经过特征、因子或模型预测，转化为组合权重，完成无未来函数的回测并评价策略。

本项目用于学习研究流程，不是实盘交易系统，不包含券商接口、实时行情、订单簿、复杂优化器或黑盒“AI 选股”。任何示例都不构成投资建议。

```text
Market Data
    ↓
Feature / Factor
    ↓
Signal / Prediction
    ↓
Portfolio Weights
    ↓
Backtest + Costs
    ↓
Performance / Factor / Prediction Analysis
```

## 安装

需要 Python 3.12 和 [uv](https://docs.astral.sh/uv/)：

```bash
uv sync --dev
uv run pre-commit install
```

也支持标准 editable install：

```bash
pip install -e .
```

AKShare 和 PyTorch 分别是可选能力：

```bash
pip install -e ".[akshare]"
pip install -e ".[mlp]"
```

核心库、测试和 synthetic experiments 都不需要网络或真实市场数据。开发环境包含 PyTorch，以便运行 MLP 实验。

## 验证

```bash
uv run python scripts/check.py
# 或只运行测试
uv run pytest -q
```

完整检查依次执行 Ruff 自动修复与格式化、mypy strict 和 pytest，并要求分支覆盖率不低于 90%。

## 运行实验

```bash
uv run python experiments/01_synthetic_markets.py
uv run python experiments/02_momentum_strategy.py
uv run python experiments/03_mean_reversion.py
uv run python experiments/04_cross_sectional_factor.py
uv run python experiments/05_factor_analysis.py
uv run python experiments/06_linear_prediction.py
uv run python experiments/07_tree_prediction.py
uv run python experiments/08_mlp_prediction.py
uv run python experiments/09_strategy_comparison.py
```

图片写入被 Git 忽略的 `artifacts/` 目录。

## 目录

```text
src/quantlab/
├── data/           # long-form schema、CSV、synthetic、optional AKShare
├── features/       # 收益率、动量、波动率、均值回归、少量技术特征
├── factors/        # 截面处理、分组收益、IC、因子组合
├── signals/        # 阈值、排名、预测信号
├── portfolio/      # 权重归一化、组合构建、调仓
├── backtest/       # 下一期收益语义、换手、费用和统一结果
├── metrics/        # 收益、风险、因子与预测指标
├── models/         # dataset、线性/逻辑/树/MLP、walk-forward
└── visualization/  # matplotlib 图形
```

建议先阅读 [LEARNING_GUIDE.md](LEARNING_GUIDE.md)，再从 `src/quantlab/data/schema.py` 开始沿数据流阅读。高层开发规格保存在 `docs/MINI_QUANT_RESEARCH_LAB_SPEC.md`。

## 可选 AKShare 示例

```python
from quantlab.data.akshare_source import AKShareDataSource

data = AKShareDataSource("000001", "20240101", "20241231").load()
```

网络错误和缺失依赖会在 adapter 边界给出明确异常；其他模块不依赖 AKShare 的列名或返回格式。

## License

Apache License 2.0。
