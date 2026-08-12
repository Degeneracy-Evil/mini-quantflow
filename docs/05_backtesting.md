# 回测与时间语义

项目采用一种一致语义：close `t` 可用于产生当日信号和权重，该权重赚取 `t → t+1` 的收益。输入的常规 `asset_returns[t]` 表示 `t-1 → t`，因此引擎用 `shift(-1)` 把下一行收益对齐到决策日期。最后一行没有可知的下一期收益，会从结果中删除。

每期换手定义为 `sum(abs(w[t] - w[t-1]))`，初始仓位相对全零组合计算。线性费用为：

```text
cost_rate = (commission_bps + slippage_bps) / 10000
cost[t] = turnover[t] * cost_rate
net_return[t] = gross_return[t] - cost[t]
```

BacktestResult 同时保留净收益、权益曲线、实际权重、换手、费用、可选 benchmark 和汇总指标。保留中间序列的目的，是让读者能审计每一步，而不是只相信一个 Sharpe 数字。

该引擎不是撮合器：没有 intraday fill、市场冲击、借券限制或订单簿。其结论只与声明的 close-to-next-close 假设相符。
