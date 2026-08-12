# 常见量化陷阱

## Look-Ahead Bias

在 close `t` 尚未形成时提前按它交易，或让当日权重乘上已经发生的 `t-1 → t` 收益，都会把未来信息放进决策。本项目统一使用 close `t` 决策、赚取 `t → t+1`。

## Data Leakage

用全样本均值标准化、先看测试集再选 feature、把 forward return 混入 feature，都会污染样本外评价。所有预处理参数应只从训练窗口估计。

## Random Train/Test Split

金融数据按时间相关且分布会变化。随机抽行同时破坏因果顺序，并可能把同一日期的资产分到训练和测试两侧。应使用 chronological 或 walk-forward split。

## Survivorship Bias

只使用今天仍存续的资产，会忽略历史上退市、破产或被收购的标的，从而高估过去可实现收益。Synthetic universe 不会自然退市，但真实数据研究必须处理当时可见的 universe。

## Overfitting

策略规则、窗口、模型和超参数越多，越容易把噪声解释成规律。漂亮的回测可能只是研究者反复尝试后的最佳偶然样本。应保留真正未参与选择的外样本，并偏好简单、可解释的假设。

## Transaction Cost 与 Turnover

微弱 alpha 若需要频繁更换整个组合，线性佣金与滑点就足以吞噬收益。始终同时报告 turnover、费用前后收益和对成本假设的敏感性。

## Multiple Testing

同时尝试 1000 个无效因子，也很可能有若干因子得到显著但偶然的结果。经典显著性阈值没有自动补偿试验次数；需要记录全部尝试、做多重检验修正并进行独立复现。

## Prediction 不等于 Profit

更低 MSE 可能只改善了大量接近零、不可交易的预测；更高 accuracy 可能来自正确预测小波动，却错过少数大亏损。组合构建、风险暴露、换手和费用共同决定交易结果，因此预测指标与 Sharpe 必须并列报告。
