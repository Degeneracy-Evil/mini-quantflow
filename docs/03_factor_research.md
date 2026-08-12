# 因子研究

标准流程是：原始 feature → 截面处理 → factor score → 分组或组合 → forward return → 评价。

Quantile analysis 把同一日期的资产分成 Q1 到 Q5，然后计算各组未来收益。若因子方向为“越高越好”，理想情况下收益会随 quantile 上升；`Q5 - Q1` 是最直观的单期价差，但仍需检查时间稳定性和换手。

Information Coefficient 是 factor score 与 forward return 的截面相关性。Pearson IC 衡量线性关系；Spearman/Rank IC 先转换为排序，对极端值与非线性单调关系更稳健。IC IR 为平均 IC 除以 IC 时间序列标准差；它描述稳定性，不直接等于组合 Sharpe。

多空因子组合在每期做多顶部资产、做空底部资产。项目默认两侧各 gross 1、net 0。其收益不仅由因子关系决定，也受分组阈值、权重、调仓频率和交易成本影响。
