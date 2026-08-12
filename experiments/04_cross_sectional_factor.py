"""Experiment B: discover and trade an embedded cross-sectional factor."""

from __future__ import annotations

import matplotlib.pyplot as plt
from common import artifact_path, backtest_scores, print_metrics

from quantlab.data import close_matrix
from quantlab.data.synthetic import factor_market
from quantlab.factors.analysis import information_coefficient, quantile_returns, summarize_ic
from quantlab.features.returns import forward_return
from quantlab.visualization.factor import plot_quantile_returns


def main() -> None:
    """Run quantile, IC, and long-short evaluation on a known weak factor."""

    market, factor = factor_market(periods=504, assets=100, factor_premium=0.0007, noise=0.01, seed=19)
    prices = close_matrix(market)
    future = forward_return(prices)
    if not hasattr(future, "columns"):
        raise TypeError("expected return panel")
    returns = quantile_returns(factor, future)
    summary = summarize_ic(information_coefficient(factor, future))
    result = backtest_scores(prices, factor)
    print(f"rank_ic_mean={summary.mean:.4f} rank_ic_ir={summary.information_ratio:.4f}")
    print_metrics("embedded_factor", result)
    plot_quantile_returns(returns)
    plt.tight_layout()
    plt.savefig(artifact_path("04_cross_sectional_factor.png"))
    plt.close()


if __name__ == "__main__":
    main()
