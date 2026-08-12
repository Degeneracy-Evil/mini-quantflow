# 数据模型

核心数据采用 long-form DataFrame，每行是一项资产在一个时间戳的 OHLCV 观测：`timestamp, asset, open, high, low, close, volume`。长表适合持久化和数据源适配；计算跨资产收益时使用 `timestamp × asset` 的 wide close matrix。

`validate_market_data` 在数据进入研究流程时统一日期与数值类型、排序，并拒绝缺列、缺失值、重复键、非正价格、不合理 high/low 和负成交量。错误越早暴露，越不容易在收益或回测阶段变成无声的索引错位。

Synthetic market 不是随手 mock。Random、Trend、Mean Reversion、Regime Switching、Correlated 和 Embedded Factor 市场分别提供已知的数据生成机制，让策略在“应该有效”和“不应该稳定有效”的环境中接受对照。

CSV 与 AKShare 都只负责转换边界。研究模块永远不读取外部数据源特有的中文列名。AKShare 是可选依赖；离线测试不导入它。
