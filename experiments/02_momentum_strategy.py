"""Experiment A: momentum on trending and random markets."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from common import artifact_path, print_metrics

from quantlab.backtest import TransactionCostModel, run_backtest
from quantlab.data import SyntheticDataSource, SyntheticMarketConfig, close_matrix
from quantlab.data.synthetic import MarketKind
from quantlab.features.momentum import momentum
from quantlab.portfolio.construction import equal_weight
from quantlab.visualization.equity import plot_equity_curves


def main() -> None:
    """Show that one strategy behaves differently under different processes."""

    config = SyntheticMarketConfig(periods=504, assets=30, drift=0.001, seed=7)
    curves: dict[str, pd.Series] = {}
    kinds: tuple[MarketKind, ...] = ("trend", "random")
    for kind in kinds:
        prices = close_matrix(SyntheticDataSource(kind, config).load())
        # Own past return controls long/flat exposure. A cross-sectional
        # dollar-neutral portfolio would cancel the common drift that this
        # experiment is designed to reveal.
        weights = equal_weight((momentum(prices, 20) > 0).astype(float))
        result = run_backtest(
            prices.pct_change(fill_method=None),
            weights,
            costs=TransactionCostModel(commission_bps=2.0, slippage_bps=3.0),
        )
        print_metrics(kind, result)
        curves[kind] = result.equity_curve
    plot_equity_curves(curves, title="Momentum: Trend vs Random Market")
    plt.tight_layout()
    plt.savefig(artifact_path("02_momentum_strategy.png"))
    plt.close()


if __name__ == "__main__":
    main()
