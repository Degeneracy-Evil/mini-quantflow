# Mini Quant Research Lab — 开发规格

> 本文档是项目的唯一高层开发规格。请先完整阅读，再开始实现。
>
> 项目目标不是构建生产级交易系统，而是构建一个**小而完整、源码可读、适合系统学习量化研究流程**的教学型量化研究框架。
>
> 代码必须完整实现并可运行，不要留下 `TODO`、占位函数或“留给学习者实现”的部分。

---

## 1. 项目定位

项目名称：

```text
mini-quant-research-lab
```

核心问题：

> 给定历史市场数据，如何从数据中构造特征和因子，形成预测或交易信号，构建投资组合，进行严格回测，并评价策略是否真正有效？

整个项目应该围绕下面这条主线组织：

```text
Market Data
    ↓
Feature / Factor
    ↓
Signal / Prediction
    ↓
Portfolio Construction
    ↓
Backtest
    ↓
Performance Analysis
```

项目应同时覆盖两条量化研究路线：

1. **传统量化 / 因子研究，约占 60%。**
2. **机器学习预测，约占 40%。**

机器学习模型只是研究流程中的一部分，不能让项目退化成普通机器学习分类项目。

---

## 2. 目标读者

目标读者具有较强 Python / 工程背景，但金融和量化基础较弱。

因此代码应该：

- 保持工程结构清晰，但避免过度抽象。
- 重要金融概念在 docstring 或文档中解释“为什么存在”。
- 优先让读者能够顺着调用链阅读源码。
- 不要为了所谓“架构优雅”引入大量基类、工厂、注册器、依赖注入、插件系统。
- 一个核心概念最好能在一个文件或少量相邻文件中读完整。

最终应当让读者通过阅读源码理解：

- 市场数据如何组织。
- 收益率和价格的关系。
- Feature、Factor、Signal、Position、Weight、PnL 的区别。
- 时间序列研究与截面研究的区别。
- 因子研究是如何做的。
- 预测结果如何变成交易仓位。
- 为什么预测准确不等于赚钱。
- 回测如何避免未来函数。
- 交易成本和换手率为什么重要。
- Walk-forward / rolling training 为什么比随机切分更合理。
- Sharpe、最大回撤、IC 等指标分别评价什么。
- 一个完整量化研究流程如何从数据一路走到策略评价。

---

## 3. 明确的非目标

第一版不要实现以下内容：

- 真实券商 API。
- 实盘交易。
- 实时行情。
- 高频交易。
- Level-2 / order book。
- 复杂订单簿撮合。
- 分布式计算。
- GPU 加速。
- Web 前端。
- 数据库服务。
- 微服务。
- Airflow / Kafka / Redis 等基础设施。
- 复杂风险系统。
- 期权定价系统。
- 债券定价系统。
- Black-Scholes。
- CAPM 教学合集。
- 完整 Markowitz 教材。
- 强化学习交易。
- LSTM / Transformer 等复杂时序模型。
- 大规模 AutoML。
- “AI 自动选股”式黑盒功能。

这些内容可能以后扩展，但不要进入第一版。

---

## 4. 技术栈

优先使用：

```text
Python >= 3.11
numpy
pandas
scikit-learn
matplotlib
pytest
```

机器学习部分可以使用：

```text
PyTorch
```

但只用于一个非常小的 MLP 示例。

可选数据源可以支持：

```text
AKShare
```

但 AKShare 必须是 optional dependency。

项目的所有核心测试和核心实验都必须能够在：

- 无网络；
- 无 AKShare；
- 无真实市场数据；

的情况下运行。

不要把 LightGBM、XGBoost、TA-Lib 等重量级或额外编译依赖设为必需依赖。

---

## 5. 代码规模与复杂度要求

目标不是“代码越少越好”，而是让一个有编程经验的人能够完整阅读。

建议范围：

```text
核心 Python 代码：约 3000～6000 行
测试、示例、文档不计入
```

这只是指导，不是硬限制。

应避免：

- 一个文件 1000+ 行。
- 过深继承体系。
- 同一个概念经过五六层 wrapper 才执行。
- 为了统一接口制造大量抽象。
- 依赖隐藏的全局状态。
- 魔法式自动注册。

允许适度重复，只要能显著改善教学可读性。

---

# 6. 总体目录结构

建议使用如下结构，可做小幅调整，但不要改变总体层次：

```text
mini-quant-research-lab/
├── README.md
├── LEARNING_GUIDE.md
├── pyproject.toml
│
├── quantlab/
│   ├── __init__.py
│   │
│   ├── data/
│   │   ├── schema.py
│   │   ├── loader.py
│   │   ├── csv_source.py
│   │   ├── synthetic.py
│   │   └── akshare_source.py
│   │
│   ├── features/
│   │   ├── returns.py
│   │   ├── momentum.py
│   │   ├── volatility.py
│   │   ├── mean_reversion.py
│   │   ├── technical.py
│   │   └── cross_sectional.py
│   │
│   ├── factors/
│   │   ├── base.py
│   │   ├── momentum.py
│   │   ├── volatility.py
│   │   ├── reversal.py
│   │   ├── ranking.py
│   │   └── analysis.py
│   │
│   ├── models/
│   │   ├── dataset.py
│   │   ├── linear.py
│   │   ├── logistic.py
│   │   ├── tree.py
│   │   ├── mlp.py
│   │   └── walk_forward.py
│   │
│   ├── signals/
│   │   ├── threshold.py
│   │   ├── ranking.py
│   │   └── prediction.py
│   │
│   ├── portfolio/
│   │   ├── weights.py
│   │   ├── construction.py
│   │   ├── rebalance.py
│   │   └── state.py
│   │
│   ├── backtest/
│   │   ├── engine.py
│   │   ├── execution.py
│   │   ├── costs.py
│   │   └── result.py
│   │
│   ├── metrics/
│   │   ├── performance.py
│   │   ├── risk.py
│   │   ├── factor.py
│   │   └── prediction.py
│   │
│   └── visualization/
│       ├── equity.py
│       ├── drawdown.py
│       ├── factor.py
│       └── prediction.py
│
├── experiments/
│   ├── 01_synthetic_markets.py
│   ├── 02_momentum_strategy.py
│   ├── 03_mean_reversion.py
│   ├── 04_cross_sectional_factor.py
│   ├── 05_factor_analysis.py
│   ├── 06_linear_prediction.py
│   ├── 07_tree_prediction.py
│   ├── 08_mlp_prediction.py
│   └── 09_strategy_comparison.py
│
├── notebooks/
│   └── optional_exploration.ipynb
│
├── examples/
│   └── sample_csv/
│
├── docs/
│   ├── 01_data_model.md
│   ├── 02_returns_features_factors.md
│   ├── 03_factor_research.md
│   ├── 04_portfolio.md
│   ├── 05_backtesting.md
│   ├── 06_ml_prediction.md
│   ├── 07_walk_forward.md
│   └── 08_common_pitfalls.md
│
└── tests/
```

---

# 7. 数据模型

## 7.1 标准长表格式

核心市场数据统一使用 long-form DataFrame，至少包含：

```text
timestamp
asset
open
high
low
close
volume
```

例如：

```text
timestamp   asset   open   high   low    close  volume
2020-01-01  A       99.0   101    98.5   100    12000
2020-01-01  B       50.2   51.0   49.7   50.5   8000
...
```

要求：

- `timestamp` 明确排序。
- `asset` 是字符串。
- 不允许重复 `(timestamp, asset)`。
- 所有研究函数明确说明输入输出格式。
- 需要时允许转换为宽表，例如 `timestamp × asset` 的 close matrix。

## 7.2 数据源

必须实现：

### SyntheticDataSource

用于生成完全离线的教学市场。

至少支持：

1. Random Walk / Geometric Brownian Motion。
2. Positive Trend Market。
3. Mean-Reverting Process。
4. Regime-Switching Market。
5. Multi-Asset Correlated Market。

多资产模式应支持设置随机种子以完全复现结果。

### CSVDataSource

读取标准 CSV。

需要做基础：

- 日期解析。
- 类型检查。
- 排序。
- 缺失值检查。
- 重复数据检查。

### AKShareDataSource

仅作为 optional adapter。

要求：

- AKShare 不存在时不影响包导入。
- 网络错误给出清晰提示。
- 获取到的数据转换成项目内部统一 schema。
- 不要让其他模块直接依赖 AKShare 返回格式。

---

# 8. Synthetic Market 设计

Synthetic market 是本项目的重要教学组件，而不是临时 mock。

需要能够生成一些“已知结构”的市场，用于检验策略是否能够发现这些结构。

例如：

## 8.1 Trending Market

价格存在可控的正向 drift。

用于观察：

- Momentum 是否可能表现较好。
- Trend Following 是否比纯随机市场更有优势。

## 8.2 Mean-Reverting Market

可以使用简单 Ornstein-Uhlenbeck 风格过程或等价离散模型。

用于观察：

- Mean Reversion 策略。
- 为什么趋势策略可能表现较差。

## 8.3 Random Market

不包含可预测结构。

用于说明：

> 一个策略在随机数据上偶然赚钱，不代表策略存在真正 Alpha。

## 8.4 Regime Switching

例如：

```text
Trend
→ High Volatility
→ Mean Reversion
→ Trend
```

用于研究：

- 策略 regime sensitivity。
- 回测结果的不稳定性。

## 8.5 Multi-Asset Cross-Section

生成 50～200 个资产。

允许人为植入一个弱 factor：

```text
future_return =
    factor_exposure * factor_premium
    + noise
```

用于验证 factor research pipeline。

---

# 9. Feature 层

Feature 是从原始数据计算出的观测量。

至少实现：

## Return

- simple return
- log return
- forward return
- cumulative return

必须特别区分：

```text
past return
future / forward return
```

并通过命名避免误用。

## Momentum

例如：

```text
5-day return
20-day return
60-day return
```

## Volatility

例如：

```text
rolling std
EWMA volatility（可选）
```

## Mean-Reversion / Reversal

例如：

```text
distance from moving average
rolling z-score
short-term reversal
```

## Technical

只实现少量简单指标，例如：

```text
moving average
moving-average ratio
RSI（可选）
```

不要把项目做成技术指标库。

---

# 10. Factor Research

这是传统量化部分的重点。

Factor 与普通 feature 的区别要在文档中明确说明。

核心流程：

```text
Raw Feature
    ↓
Cross-sectional Processing
    ↓
Factor Score
    ↓
Ranking
    ↓
Portfolio / Future Return
    ↓
Factor Evaluation
```

## 10.1 截面排名

实现：

- cross-sectional rank。
- percentile rank。
- z-score。
- winsorization 或简单 clipping。
- standardization。

## 10.2 内置因子

至少实现：

### Momentum Factor

根据过去 N 天收益排序。

### Volatility Factor

根据过去波动率排序。

### Short-Term Reversal Factor

根据短期历史收益构造反转信号。

这些因子主要用于教学，不宣称具有现实盈利能力。

## 10.3 Quantile Analysis

把资产按照 factor score 分成：

```text
Q1
Q2
Q3
Q4
Q5
```

计算每组未来收益。

输出：

- quantile return。
- top-minus-bottom。
- cumulative quantile return。

## 10.4 IC

实现：

### Information Coefficient

factor score 与 forward return 的相关性。

至少支持：

- Pearson IC。
- Rank IC / Spearman correlation。

计算：

- 每期 IC。
- mean IC。
- IC std。
- IC IR。

必须明确处理：

> factor 在 t 时刻计算，只能与 t 之后的 future return 对比。

禁止未来数据泄露。

## 10.5 Long-Short Factor Portfolio

例如：

```text
Long: top 20%
Short: bottom 20%
```

支持：

- equal weight。
- dollar neutral。
- configurable quantile。

输出：

- long return。
- short return。
- long-short return。
- turnover。

---

# 11. Signal 层

Signal 是从 feature / factor / prediction 得到的交易意图。

至少支持：

## Threshold Signal

例如：

```text
zscore < -1 → long
zscore > +1 → short
```

## Ranking Signal

例如：

```text
top 20% → long
bottom 20% → short
```

## Prediction Signal

例如：

```text
predicted_return > threshold
```

或根据预测值截面排序。

Signal 不应直接等同于最终 portfolio weights。

---

# 12. Portfolio Construction

需要明确区分：

```text
Signal
Position
Weight
```

至少实现：

## Equal Weight

对被选资产等权。

## Long-Short Equal Weight

例如：

```text
Long gross exposure  = 1.0
Short gross exposure = 1.0
Net exposure         = 0
```

## Long-Only Top-K

例如只持有 factor ranking 前 K 个资产。

## Weight Normalization

处理：

- long-only。
- long-short。
- gross exposure。
- net exposure。

## Rebalancing

支持简单：

```text
daily
weekly
monthly
```

或者基于明确的 rebalance dates。

不需要复杂 optimizer。

---

# 13. Backtest Engine

这是整个项目最重要的模块之一。

要求实现一个**清晰、逐时间步推进的小型回测引擎**。

推荐流程：

```text
for each timestamp t:

    1. only expose information available at t
    2. compute / read signal available at t
    3. generate target portfolio weights
    4. apply rebalance rule
    5. compute turnover
    6. apply transaction cost
    7. portfolio earns next-period return
    8. update equity
```

实际实现可以做合理调整，但必须确保时间关系非常清楚。

## 13.1 必须防止 Look-Ahead Bias

例如：

如果使用：

```text
close[t]
```

生成 signal，则不能假定在不知道该 close 的情况下提前交易。

必须选择一种清晰的一致语义，例如：

### 方案 A

```text
signal based on close[t]
→ position applies to return[t → t+1]
```

这是推荐方案。

在文档和代码注释中明确说明。

## 13.2 Transaction Cost

实现简单线性费用：

```text
cost = turnover × cost_rate
```

支持配置：

```text
commission_bps
slippage_bps
```

不需要订单簿。

## 13.3 Turnover

清楚实现：

```text
turnover = sum(abs(w_t - w_{t-1}))
```

如果使用不同定义，必须明确记录。

## 13.4 BacktestResult

统一保存：

```text
equity_curve
period_returns
weights
turnover
transaction_costs
benchmark_returns
metrics
```

---

# 14. Performance Metrics

至少实现：

## Return

- total return。
- annualized return / CAGR。

## Risk

- annualized volatility。
- drawdown series。
- maximum drawdown。

## Risk-adjusted

- Sharpe ratio。

允许配置 risk-free rate，但默认可以使用 0。

## Trading

- turnover。
- average turnover。
- total transaction cost。
- hit rate / win rate（只在定义合理时使用）。

## Benchmark

支持策略与 benchmark 对比。

Synthetic experiment 可以使用：

```text
equal-weight universe
```

作为简单 benchmark。

---

# 15. 机器学习预测

机器学习部分目标不是追求 SOTA，而是展示：

```text
feature
→ label
→ time split
→ train
→ predict
→ signal
→ portfolio
→ backtest
```

## 15.1 预测目标

至少覆盖：

### Regression

预测：

```text
next-period return
```

### Classification

预测：

```text
next-period return > 0 ?
```

### Cross-Sectional Ranking

用预测的未来收益对资产排序。

这是重点之一。

---

# 16. Dataset Builder

实现统一的 supervised dataset builder。

输入：

```text
market data
features
prediction horizon
```

输出至少包含：

```text
timestamp
asset
feature_1
feature_2
...
target
```

必须保证：

- 特征仅来自当时及过去数据。
- target 来自未来。
- 缺失值处理显式。
- 不允许随机 shuffle 破坏时间顺序。

---

# 17. 时间序列切分

禁止默认使用：

```python
train_test_split(..., shuffle=True)
```

金融预测必须使用时间顺序。

至少实现：

## Simple Chronological Split

```text
Train
Validation
Test
```

按时间顺序切分。

## Walk-Forward Validation

例如：

```text
Train 2010-2015 → Predict 2016
Train 2011-2016 → Predict 2017
Train 2012-2017 → Predict 2018
...
```

支持：

- expanding window。
- rolling window。

文档必须说明：

> Walk-forward 模拟“在真实历史时间点，当时只能看到过去数据”的训练过程。

---

# 18. 机器学习模型

保持简单。

## Linear Regression

使用 scikit-learn。

用于预测 future return。

## Logistic Regression

用于预测涨跌方向。

## Tree Model

使用：

```text
RandomForestRegressor
```

或 sklearn 自带的简单 Gradient Boosting。

只选一种即可。

## Small MLP

使用 PyTorch，实现一个非常小的：

```text
input
→ hidden
→ activation
→ hidden
→ output
```

模型。

要求：

- 代码可读。
- CPU 可以快速训练。
- 不引入复杂 trainer。
- 自己写一个很短的 training loop。
- 设置随机种子。
- 支持 early stopping 可以有，但不是必须。

不要实现：

- LSTM。
- Transformer。
- Attention。
- GPU 分布式训练。
- 自动超参数搜索。

---

# 19. Prediction Metrics

至少实现：

## Regression

- MSE。
- MAE。
- prediction / target correlation。

## Classification

- accuracy。
- precision / recall 可以选择性加入。

但必须在文档中强调：

> 高预测指标不意味着高交易收益。

## Cross-Sectional

重点支持：

- IC。
- Rank IC。
- top-bottom return。

---

# 20. 必须实现的四条完整实验链

项目不是只有库代码，必须有可以独立运行的 end-to-end experiments。

---

## Experiment A：Momentum on Synthetic Trending Market

流程：

```text
Synthetic Trending Market
→ Momentum Feature
→ Momentum Signal
→ Portfolio
→ Backtest
→ Sharpe / Drawdown
```

同时在 Random Market 上运行对照。

目标：

展示：

> 相同策略在不同 data-generating process 下表现不同。

---

## Experiment B：Cross-Sectional Factor Research

流程：

```text
Synthetic Multi-Asset Market
→ Momentum / Other Factor
→ Cross-sectional Ranking
→ Quantile Analysis
→ IC / Rank IC
→ Long-Short Portfolio
→ Backtest
```

Synthetic 数据中应植入一个弱 factor，使研究流程能够在统计上发现它。

目标：

完整展示：

```text
factor discovery
→ validation
→ portfolio
```

---

## Experiment C：Classical ML Prediction

流程：

```text
Multi-Asset Features
→ Time Ordered Dataset
→ Linear / Tree Model
→ Future Return Prediction
→ Ranking Signal
→ Portfolio
→ Backtest
```

必须输出两类结果：

```text
Prediction Metrics
Trading Metrics
```

并展示两者不完全等价。

---

## Experiment D：MLP + Walk-Forward

流程：

```text
Features
→ Rolling / Expanding Training
→ Small MLP
→ Out-of-Sample Prediction
→ Cross-sectional Ranking
→ Portfolio
→ Backtest
```

目标是理解：

```text
walk-forward training
```

而不是追求深度学习性能。

应与简单 Linear Regression 做对照。

---

# 21. Strategy Comparison

`09_strategy_comparison.py` 至少比较：

```text
Buy & Hold / Equal Weight Benchmark
Momentum
Mean Reversion
Cross-sectional Factor
Linear Regression Prediction
Tree / MLP Prediction
```

在至少两个 synthetic market regime 上比较。

输出：

```text
Total Return
CAGR
Volatility
Sharpe
Max Drawdown
Turnover
Transaction Cost
```

并画 equity curve。

---

# 22. 可视化

使用 matplotlib。

至少提供：

## Equity Curve

```text
strategy vs benchmark
```

## Drawdown

画 drawdown curve。

## Factor Quantile Plot

显示：

```text
Q1 ... Q5
```

未来收益差异。

## IC Time Series

显示 IC 随时间变化。

## Prediction Scatter

例如：

```text
predicted return vs realized return
```

保持图简单，不做 dashboard。

---

# 23. 常见量化陷阱

项目必须专门有：

```text
docs/08_common_pitfalls.md
```

至少解释：

## Look-Ahead Bias

未来信息进入当前决策。

## Data Leakage

训练过程中泄露测试期或未来信息。

## Random Train/Test Split

为什么普通 ML 随机切分通常不适用于时序金融问题。

## Survivorship Bias

即使 synthetic 数据不会发生，也要解释概念。

## Overfitting

为什么尝试大量策略后挑最好的会造成虚假发现。

## Transaction Cost

为什么忽略费用会严重高估策略收益。

## Turnover

为什么频繁换仓会吞噬 Alpha。

## Multiple Testing

为什么试 1000 个因子后找到一个漂亮结果并不神奇。

## Prediction ≠ Profit

为什么：

```text
better MSE / accuracy
```

不必然对应：

```text
better Sharpe
```

---

# 24. 测试

使用 pytest。

所有核心数学和时间语义必须测试。

至少包括：

## Data

- schema validation。
- duplicate timestamp/asset。
- deterministic synthetic generation。

## Returns / Features

- simple return。
- forward return alignment。
- rolling feature alignment。

## Factor

- ranking。
- quantiles。
- IC。
- Rank IC。

## Portfolio

- weight normalization。
- long-short exposure。
- rebalance。

## Backtest

- position shift / next-period return alignment。
- transaction cost。
- turnover。
- equity update。
- benchmark alignment。

## Metrics

- CAGR。
- volatility。
- Sharpe。
- drawdown。
- maximum drawdown。

## ML

- chronological split。
- walk-forward split。
- target alignment。
- no future row accidentally used as feature。

测试优先使用非常小、可以人工验证的数组。

---

# 25. Reproducibility

所有：

- synthetic generator。
- sklearn model。
- PyTorch model。

应尽量支持固定 seed。

Experiment 运行结果应该可重复到合理程度。

---

# 26. README

README 不要写成长篇金融教材。

README 应重点说明：

1. 项目是什么。
2. 项目不是什么。
3. 安装方法。
4. 如何运行测试。
5. 如何运行 experiments。
6. 一张整体 pipeline 图。
7. 主要目录结构。
8. 推荐阅读顺序。
9. optional AKShare 使用方法。
10. 声明项目仅用于教育和研究，不构成投资建议。

---

# 27. LEARNING_GUIDE.md

这是项目非常重要的文件。

它不是 API 文档，而是“如何阅读这个代码库”。

推荐阅读顺序：

```text
Stage 1: Data
Stage 2: Returns & Features
Stage 3: Factor
Stage 4: Signal
Stage 5: Portfolio
Stage 6: Backtest
Stage 7: Factor Analysis
Stage 8: ML Dataset
Stage 9: ML Models
Stage 10: Walk-Forward
Stage 11: End-to-End Experiments
```

每个 Stage 说明：

- 先读哪些文件。
- 这些文件解决什么问题。
- 最关键的 3～5 个概念是什么。
- 读代码时应该重点观察什么。
- 推荐运行哪个 experiment。
- 下一阶段如何依赖当前阶段。

不要要求学习者自己实现代码。

这是一个“阅读源码学习”的项目，不是编程作业。

---

# 28. 代码注释和文档风格

代码注释不要解释 Python 语法。

例如不要写：

```python
# loop over every row
for row in rows:
```

应该解释：

```python
# The signal observed at t is applied to the return from t to t+1.
# This shift prevents the strategy from using future returns.
```

Docstring 优先解释：

- input 的金融语义。
- output 的金融语义。
- 时间对齐关系。
- 关键假设。

重要公式可以写在 docstring 中。

---

# 29. API 风格

优先函数式、显式数据流。

例如：

```python
returns = compute_returns(prices)
factor = momentum_factor(returns, lookback=20)
signal = rank_signal(factor, top_quantile=0.2)
weights = build_long_short_weights(signal)
result = run_backtest(returns, weights, costs)
```

比下面这种更适合本项目：

```python
SystemFactory()
    .register(...)
    .build(...)
    .execute(...)
```

类只在真正有状态或概念上自然时使用，例如：

```text
BacktestResult
SyntheticMarketConfig
WalkForwardConfig
```

不要为了 OOP 而 OOP。

---

# 30. 错误处理

需要对明显错误给出清晰异常，例如：

- 日期未排序。
- 缺少必须列。
- 重复 `(timestamp, asset)`。
- prediction 和 return index 不对齐。
- weight sum / gross exposure 不符合约束。
- forward return horizon 非法。

不要 silent failure。

---

# 31. 示例命令

最终至少可以：

```bash
pip install -e .
pytest -q
```

然后：

```bash
python experiments/01_synthetic_markets.py
python experiments/02_momentum_strategy.py
python experiments/04_cross_sectional_factor.py
python experiments/05_factor_analysis.py
python experiments/06_linear_prediction.py
python experiments/08_mlp_prediction.py
python experiments/09_strategy_comparison.py
```

所有实验默认在 CPU 上合理时间内完成。

---

# 32. 实现顺序

Codex 应按以下顺序开发：

## Phase 1

```text
project skeleton
data schema
synthetic market
CSV loader
returns
basic metrics
tests
```

## Phase 2

```text
features
factor ranking
quantile analysis
IC
factor portfolio
tests
```

## Phase 3

```text
portfolio construction
rebalance
backtest engine
transaction costs
result
tests
```

## Phase 4

```text
linear regression
logistic regression
tree model
dataset builder
chronological split
tests
```

## Phase 5

```text
walk-forward
PyTorch MLP
ML trading pipeline
tests
```

## Phase 6

```text
all experiments
plots
README
docs
LEARNING_GUIDE
```

---

# 33. 验收标准

项目完成必须满足：

## Functional

- 所有模块完整实现。
- 无 `TODO` 核心功能。
- 无占位 `pass`。
- 所有 imports 正常。
- 所有 experiments 可以运行。
- 所有 tests 通过。

## Financial Correctness

特别检查：

- return 时间对齐。
- forward return 时间对齐。
- signal 与持仓错位。
- prediction target 泄露。
- walk-forward 训练窗口。
- transaction cost。
- turnover。
- long-short exposure。

## Engineering

- `pyproject.toml` 正常。
- 包可 editable install。
- pytest 可直接运行。
- 不依赖 notebook 才能工作。
- 不依赖网络才能测试。

## Readability

- 主要模块可以独立阅读。
- 没有不必要的架构层。
- 调用链容易追踪。
- 关键时间关系有明确注释。
- LEARNING_GUIDE 足够让读者从头读完整个项目。

---

# 34. 开发结束前必须执行

Codex 在认为项目完成前，必须亲自执行：

```bash
pip install -e .
pytest -q
```

并逐个运行所有 experiments。

发现：

- exception。
- NaN。
- index alignment error。
- warning。
- 不合理的输出。

都应尽可能修复。

特别注意 pandas 中：

- index alignment。
- `shift`。
- `rolling`。
- `groupby`。
- MultiIndex / long-wide conversion。

这些地方非常容易产生无声的数据错位。

---

# 35. 最终交付说明

开发完成后，请给出一份简短总结，包括：

```text
1. 实现了哪些模块
2. 项目总结构
3. 测试数量与测试结果
4. 已运行哪些 experiments
5. 推荐学习者从哪个文件开始读
6. 当前明确没有实现的内容
```

不要仅仅说“实现完成”。

---

# 36. 最重要的设计原则

如果实现过程中出现取舍，优先级如下：

```text
金融 / 量化语义正确
        >
时间对齐正确、无数据泄露
        >
源码可读
        >
实验链完整
        >
工程优雅
        >
性能
```

这个项目不是生产基础设施。

它真正的价值在于：

> 让一个已经会编程、但刚开始系统学习金融的人，可以通过阅读一个完整的小型量化研究系统，理解真实量化研究到底是如何从数据走到因子、预测、组合、回测和绩效评价的。
