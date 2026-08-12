# 组合构建

Signal、Position 和 Weight 不相同：signal 是交易意图，position 是持有方向或数量，weight 是某项资产占组合资本的比例。只有 weight 能直接与资产收益相乘形成组合收益。

Long-only 等权把选中资产的正权重归一化到 gross 1。Long-short 等权分别把正负两侧归一化，例如 long gross 1、short gross 1，因此 net 0、总 gross 2。若一侧没有资产，函数不会凭空创造仓位。

Top-K 根据分数选择固定数量资产；ranking signal 按比例选择，更适合资产数量变化的 universe。

调仓规则决定何时允许 target weight 变成 held weight。非调仓日继续持有上次权重。当前教学引擎不模拟持仓因价格变化产生的盘中漂移，也不使用优化器；这种简化让信号、调仓和换手之间的关系保持透明。
