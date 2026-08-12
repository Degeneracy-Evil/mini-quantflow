# Walk-Forward

随机 train/test split 会让未来时期的市场状态进入训练集，不符合历史研究时的可用信息集合。Chronological split 按唯一时间戳依次划分 Train、Validation、Test，保证同一天的不同资产不会跨越边界。

Walk-forward 更接近实际过程：先用一段过去窗口训练，只预测随后一段；时间前进后重新拟合。Expanding window 始终保留全部历史，rolling window 只保留最近固定长度。前者样本多但适应状态变化较慢，后者更关注近期但估计方差更大。

实现中的窗口单位是唯一时间戳，而不是 DataFrame 行数。在多资产数据中，这一点避免“200 行”实际只覆盖少数几个交易日。

所有测试窗口都严格位于训练窗口之后，最终预测由多个互不提前观察的样本外片段拼接。Walk-forward 仍不能消除过拟合；模型选择和超参数尝试本身也必须尊重外样本边界。
