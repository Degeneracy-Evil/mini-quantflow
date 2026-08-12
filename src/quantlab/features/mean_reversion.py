"""Features useful for mean-reversion and short-term reversal research."""

from __future__ import annotations

import pandas as pd

from quantlab.features.returns import simple_return


def distance_from_moving_average(prices: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Measure relative price distance from its trailing moving average."""

    if window < 2:
        raise ValueError("window must be at least 2")
    average = prices.rolling(window, min_periods=window).mean()
    return prices / average - 1.0


def rolling_zscore(prices: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Standardize current prices using only the trailing window ending at ``t``."""

    if window < 2:
        raise ValueError("window must be at least 2")
    rolling = prices.rolling(window, min_periods=window)
    mean = rolling.mean()
    standard_deviation = rolling.std(ddof=0).replace(0.0, float("nan"))
    return (prices - mean) / standard_deviation


def short_term_reversal(prices: pd.DataFrame, lookback: int = 5) -> pd.DataFrame:
    """Negate trailing return so recent losers receive higher scores."""

    result = simple_return(prices, periods=lookback)
    if not isinstance(result, pd.DataFrame):  # pragma: no cover
        raise TypeError("expected DataFrame result")
    return -result
