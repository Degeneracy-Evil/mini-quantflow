# 机器学习预测

监督数据每行由 `timestamp, asset, features..., target` 组成。Feature 必须只依赖当时及此前数据，target 是未来收益或未来收益是否为正。缺少完整历史窗口或未来标签的行会被显式删除。

回归模型预测连续 future return；逻辑回归预测上涨概率；横截面交易则用预测值对当日资产排序。项目只使用线性回归、逻辑回归、一个小型随机森林和两层小 MLP，因为重点是研究链而非模型复杂度。

标准化只能在训练样本上拟合。Linear/Logistic 的 sklearn Pipeline 和 MLP 自己保存的训练均值/标准差都遵守这一点。

MSE、MAE、accuracy 和 prediction correlation 评价预测本身。交易收益还取决于预测排序、阈值、组合敞口、换手、成本与市场状态。因此预测指标更好不保证 Sharpe 更高，所有模型都必须经过与传统因子相同的组合与回测流程。
