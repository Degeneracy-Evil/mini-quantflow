# 源码学习指南

这不是 API 索引，而是一条从市场数据走到研究结论的阅读路线。每一阶段先读代码，再运行对应实验；无需自行补全任何函数。

## Stage 1：Data

先读 `data/schema.py`，再读 `data/synthetic.py`、`data/csv_source.py`。

- 解决：市场观测如何用统一 long-form 表表达，外部数据怎样在边界处归一化。
- 关键概念：`timestamp`、`asset`、OHLCV、long/wide、可复现随机种子。
- 观察：为什么重复 `(timestamp, asset)`、非正价格和不一致 high/low 会立即报错。
- 运行：`01_synthetic_markets.py`。

下一阶段会把 wide close matrix 变成历史收益和未来标签。

## Stage 2：Returns & Features

读 `features/returns.py`，再读 `momentum.py`、`volatility.py`、`mean_reversion.py`。

- 解决：价格水平如何变成可比较的变化率和滚动观测量。
- 关键概念：simple/log return、past return、forward return、rolling window、缺失期。
- 观察：`forward_return` 的名字和负向 shift 明确表示它只能作为标签或评价数据。
- 运行：`02_momentum_strategy.py`、`03_mean_reversion.py`。

## Stage 3：Factor

读 `factors/ranking.py`，再读三个内置因子。

- 解决：普通历史特征如何在同一时刻的资产之间变成相对分数。
- 关键概念：时间序列、截面、rank、z-score、winsorization。
- 观察：所有截面变换都沿列方向逐行完成，不混合不同日期。
- 运行：`04_cross_sectional_factor.py`。

## Stage 4：Signal

读 `signals/threshold.py`、`signals/ranking.py`、`signals/prediction.py`。

- 解决：连续分数如何表达离散交易意图。
- 关键概念：threshold、top/bottom quantile、long/short/flat、prediction score。
- 观察：signal 只有方向，没有资本约束，因此还不是最终权重。

## Stage 5：Portfolio

读 `portfolio/weights.py`、`construction.py`、`rebalance.py`。

- 解决：选择哪些资产之后，如何分配资本并控制多空敞口。
- 关键概念：position、weight、gross/net exposure、equal weight、rebalance。
- 观察：多头和空头分别归一化；周/月调仓期间持有上一次权重。

## Stage 6：Backtest

依次读 `backtest/execution.py`、`costs.py`、`engine.py`、`result.py`。

- 解决：如何把决策按真实时间顺序映射到下一期盈亏。
- 关键概念：look-ahead、next-period return、turnover、transaction cost、equity。
- 观察：常规收益 `return[t]` 表示 `t-1 → t`，引擎先 `shift(-1)`，使 close `t` 的权重只赚取 `t → t+1`。
- 运行：再次运行 `02_momentum_strategy.py`，对照有无交易成本的指标。

## Stage 7：Factor Analysis

读 `factors/analysis.py` 与 `metrics/factor.py`。

- 解决：因子分数是否与未来收益存在稳定的截面关系。
- 关键概念：quantile return、top-minus-bottom、Pearson IC、Rank IC、IC IR。
- 观察：因子 panel 与 forward return panel 必须完全对齐，否则拒绝静默计算。
- 运行：`04_cross_sectional_factor.py`、`05_factor_analysis.py`。

## Stage 8：ML Dataset

读 `models/dataset.py`。

- 解决：如何把多个 feature panel 变成监督学习表，同时保留时间和资产身份。
- 关键概念：feature、target、horizon、missing value、chronological split。
- 观察：切分边界依据唯一时间戳，而不是随机抽行；同一天的资产不会落入不同集合。

## Stage 9：ML Models

读 `models/linear.py`、`logistic.py`、`tree.py`，最后读 `mlp.py`。

- 解决：几类简单模型如何共享同一研究数据流。
- 关键概念：regression、classification、standardization、nonlinearity、seed。
- 观察：模型只输出预测；预测还需要变成 signal、weights 并经过同一回测。
- 运行：`06_linear_prediction.py`、`07_tree_prediction.py`。

## Stage 10：Walk-Forward

读 `models/walk_forward.py`。

- 解决：在每个历史时点只用此前样本重新训练并产生样本外预测。
- 关键概念：expanding window、rolling window、test window、out-of-sample stitching。
- 观察：每个 split 都满足最大训练日期小于最小测试日期。
- 运行：`08_mlp_prediction.py`。

## Stage 11：End-to-End Experiments

最后顺序阅读 `experiments/01` 到 `09`。

- 观察同一策略为何依赖 data-generating process。
- 对比预测 MSE/相关性与 Sharpe/回撤，它们回答不同问题。
- 检查换手率和费用如何改变高频调仓策略。
- 运行 `09_strategy_comparison.py`，把传统因子和模型策略放到同一张评价表中。

阅读完后，再看 `tests/` 中的小数组案例。它们是时间对齐、组合敞口和指标公式最简洁的可执行说明。
