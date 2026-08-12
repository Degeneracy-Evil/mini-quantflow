# Returns、Feature、Factor

简单收益率为 `r[t] = price[t] / price[t-1] - 1`，对数收益为相邻对数价格之差。累计简单收益必须复利：`prod(1 + r) - 1`，不能直接求和。

`momentum[t]`、滚动波动率和均线距离都是 feature：它们由截至 `t` 可观察的数据构造。`forward_return[t] = price[t+h] / price[t] - 1` 使用未来价格，只能作为监督标签或因子评价目标。

Factor 强调横截面含义。同一时刻的原始 feature 会经过 rank、z-score 或 clipping，使分数表达资产之间的相对预期。一个 20 日动量 feature 说明每项资产自己的历史变化；momentum factor 说明它在当日资产集合中的相对强弱。

务必从金融语义判断某个值是过去观测还是未来结果，而不是仅凭 DataFrame 列名或正负 shift 猜测。
