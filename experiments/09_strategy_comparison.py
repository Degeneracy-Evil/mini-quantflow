"""Compare benchmark, traditional signals, and ML predictions across regimes."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from common import artifact_path, backtest_scores, feature_panels

from quantlab.backtest import TransactionCostModel, run_backtest
from quantlab.data import SyntheticDataSource, SyntheticMarketConfig, close_matrix
from quantlab.factors.momentum import momentum_factor
from quantlab.features.mean_reversion import rolling_zscore
from quantlab.features.momentum import momentum
from quantlab.models.dataset import build_supervised_dataset, chronological_split, prediction_panel
from quantlab.models.linear import fit_linear_regression, predict_linear
from quantlab.models.tree import fit_random_forest, predict_forest
from quantlab.portfolio.construction import equal_weight
from quantlab.visualization.equity import plot_equity_curves


def evaluate_market(kind: str, seed: int) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    """Evaluate a fixed strategy set on one synthetic market kind."""

    prices = close_matrix(
        SyntheticDataSource(kind, SyntheticMarketConfig(periods=600, assets=30, drift=0.0007, seed=seed)).load()  # type: ignore[arg-type]
    )
    returns = prices.pct_change(fill_method=None)
    benchmark_weights = pd.DataFrame(1.0 / prices.shape[1], index=prices.index, columns=prices.columns)
    benchmark = run_backtest(returns, benchmark_weights, costs=TransactionCostModel())
    momentum_weights = equal_weight((momentum(prices, 20) > 0).astype(float))
    results = {
        "benchmark": benchmark,
        "momentum": run_backtest(returns, momentum_weights, costs=TransactionCostModel(2.0, 3.0)),
        "mean_reversion": backtest_scores(prices, -rolling_zscore(prices, 20)),
        "cross_sectional_factor": backtest_scores(prices, momentum_factor(prices, 20)),
    }
    dataset = build_supervised_dataset(prices, feature_panels(prices))
    split = chronological_split(dataset)
    linear_prediction = predict_linear(fit_linear_regression(split.train), split.test)
    tree_prediction = predict_forest(fit_random_forest(split.train, trees=60), split.test)
    results["linear"] = backtest_scores(prices, prediction_panel(split.test, linear_prediction))
    results["tree"] = backtest_scores(prices, prediction_panel(split.test, tree_prediction))
    rows = []
    for name, result in results.items():
        rows.append({"strategy": name, **result.metrics})
    table = pd.DataFrame(rows).set_index("strategy")
    return table, {name: result.equity_curve for name, result in results.items()}


def main() -> None:
    """Print comparable metrics and save one equity plot per market regime."""

    for kind, seed in (("trend", 41), ("mean_reverting", 43)):
        table, curves = evaluate_market(kind, seed)
        columns = [
            "total_return",
            "cagr",
            "annualized_volatility",
            "sharpe",
            "max_drawdown",
            "average_turnover",
            "total_transaction_cost",
        ]
        print(f"\n{kind}\n{table[columns].round(4).to_string()}")
        plot_equity_curves(curves, title=f"Strategy Comparison: {kind}")
        plt.tight_layout()
        plt.savefig(artifact_path(f"09_strategy_comparison_{kind}.png"))
        plt.close()


if __name__ == "__main__":
    main()
