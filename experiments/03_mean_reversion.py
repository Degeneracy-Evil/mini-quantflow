"""Mean-reversion signal on mean-reverting and trending markets."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from common import artifact_path, backtest_scores, print_metrics

from quantlab.data import SyntheticDataSource, SyntheticMarketConfig, close_matrix
from quantlab.data.synthetic import MarketKind
from quantlab.features.mean_reversion import rolling_zscore
from quantlab.visualization.equity import plot_equity_curves


def main() -> None:
    """Compare an identical contrarian score across two market structures."""

    config = SyntheticMarketConfig(periods=504, assets=30, seed=11)
    curves: dict[str, pd.Series] = {}
    kinds: tuple[MarketKind, ...] = ("mean_reverting", "trend")
    for kind in kinds:
        prices = close_matrix(SyntheticDataSource(kind, config).load())
        result = backtest_scores(prices, -rolling_zscore(prices, 20))
        print_metrics(kind, result)
        curves[kind] = result.equity_curve
    plot_equity_curves(curves, title="Mean Reversion Across Market Structures")
    plt.tight_layout()
    plt.savefig(artifact_path("03_mean_reversion.png"))
    plt.close()


if __name__ == "__main__":
    main()
