"""Inspect information coefficient through time and factor quantiles."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from common import artifact_path

from quantlab.data import close_matrix
from quantlab.data.synthetic import factor_market
from quantlab.factors.analysis import information_coefficient, quantile_returns, summarize_ic
from quantlab.features.returns import forward_return
from quantlab.visualization.factor import plot_ic, plot_quantile_returns


def main() -> None:
    """Produce the two central factor-research diagnostics."""

    market, factor = factor_market(periods=400, assets=80, seed=23)
    prices = close_matrix(market)
    future = forward_return(prices)
    if not isinstance(future, pd.DataFrame):
        raise TypeError("expected return panel")
    ic = information_coefficient(factor, future)
    summary = summarize_ic(ic)
    print(f"mean_ic={summary.mean:.4f} ic_std={summary.standard_deviation:.4f} ic_ir={summary.information_ratio:.4f}")
    _, axes = plt.subplots(2, 1, figsize=(9, 8))
    plot_ic(ic, ax=axes[0])
    plot_quantile_returns(quantile_returns(factor, future), ax=axes[1])
    plt.tight_layout()
    plt.savefig(artifact_path("05_factor_analysis.png"))
    plt.close()


if __name__ == "__main__":
    main()
