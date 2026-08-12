"""Shared, non-magical helpers for the runnable experiments."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import pandas as pd

from quantlab.backtest import BacktestResult, TransactionCostModel, run_backtest
from quantlab.features.mean_reversion import distance_from_moving_average
from quantlab.features.momentum import momentum
from quantlab.features.returns import simple_return
from quantlab.features.volatility import rolling_volatility
from quantlab.portfolio.construction import long_short_equal_weight
from quantlab.signals.ranking import ranking_signal

matplotlib.use("Agg")


def artifact_path(filename: str) -> Path:
    """Return an ignored output path and create its parent directory."""

    directory = Path("artifacts")
    directory.mkdir(exist_ok=True)
    return directory / filename


def feature_panels(prices: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Build a compact feature set used by multiple prediction experiments."""

    returns = simple_return(prices)
    if not isinstance(returns, pd.DataFrame):
        raise TypeError("expected DataFrame returns")
    return {
        "momentum_5": momentum(prices, 5),
        "momentum_20": momentum(prices, 20),
        "ma_distance_10": distance_from_moving_average(prices, 10),
        "volatility_20": rolling_volatility(returns, 20),
    }


def backtest_scores(prices: pd.DataFrame, scores: pd.DataFrame, quantile: float = 0.2) -> BacktestResult:
    """Rank scores, make a long-short portfolio, and apply modest linear costs."""

    conventional_returns = prices.pct_change(fill_method=None)
    aligned_scores = scores.reindex(index=prices.index, columns=prices.columns)
    signal = ranking_signal(aligned_scores, quantile).fillna(0.0)
    weights = long_short_equal_weight(signal)
    return run_backtest(conventional_returns, weights, costs=TransactionCostModel(2.0, 3.0))


def print_metrics(name: str, result: BacktestResult) -> None:
    """Print the main trading metrics without dumping internal tables."""

    keys = ["total_return", "cagr", "annualized_volatility", "sharpe", "max_drawdown", "average_turnover"]
    values = "  ".join(f"{key}={result.metrics[key]:.4f}" for key in keys)
    print(f"{name}: {values}")
