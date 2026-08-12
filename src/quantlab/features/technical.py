"""A deliberately small set of readable technical features."""

from __future__ import annotations

import pandas as pd


def moving_average(prices: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Compute a trailing arithmetic moving average."""

    if window < 1:
        raise ValueError("window must be positive")
    return prices.rolling(window, min_periods=window).mean()


def moving_average_ratio(prices: pd.DataFrame, short_window: int = 5, long_window: int = 20) -> pd.DataFrame:
    """Compare short and long trailing averages; values above one imply strength."""

    if short_window < 1 or long_window <= short_window:
        raise ValueError("windows must satisfy 0 < short_window < long_window")
    return moving_average(prices, short_window) / moving_average(prices, long_window)


def rsi(prices: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """Compute Wilder-style relative strength index on a 0-to-100 scale."""

    if window < 2:
        raise ValueError("window must be at least 2")
    changes = prices.diff()
    gains = changes.clip(lower=0.0)
    losses = -changes.clip(upper=0.0)
    average_gain = gains.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    average_loss = losses.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    relative_strength = average_gain / average_loss.replace(0.0, float("nan"))
    result = 100.0 - 100.0 / (1.0 + relative_strength)
    return result.where(average_loss != 0.0, 100.0)
