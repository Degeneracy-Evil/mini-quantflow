"""Return transformations with explicit time alignment."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _validate_prices(prices: pd.Series | pd.DataFrame) -> None:
    if prices.empty:
        raise ValueError("prices must not be empty")
    if (prices <= 0).any().any() if isinstance(prices, pd.DataFrame) else (prices <= 0).any():
        raise ValueError("prices must be strictly positive")


def simple_return(prices: pd.Series | pd.DataFrame, periods: int = 1) -> pd.Series | pd.DataFrame:
    """Return ``price[t] / price[t-periods] - 1``, a past-looking observation."""

    _validate_prices(prices)
    if periods < 1:
        raise ValueError("periods must be positive")
    return prices.pct_change(periods=periods, fill_method=None)


def log_return(prices: pd.Series | pd.DataFrame, periods: int = 1) -> pd.Series | pd.DataFrame:
    """Return the past-looking log price difference over ``periods``."""

    _validate_prices(prices)
    if periods < 1:
        raise ValueError("periods must be positive")
    values = np.log(prices.to_numpy(dtype=float))
    if isinstance(prices, pd.Series):
        return pd.Series(values, index=prices.index, name=prices.name).diff(periods=periods)
    return pd.DataFrame(values, index=prices.index, columns=prices.columns).diff(periods=periods)


def forward_return(prices: pd.Series | pd.DataFrame, horizon: int = 1) -> pd.Series | pd.DataFrame:
    """Return ``price[t+horizon] / price[t] - 1`` aligned at decision time ``t``.

    This is a future label for evaluation or supervised learning. It must never
    be used as a feature available at ``t``.
    """

    _validate_prices(prices)
    if horizon < 1:
        raise ValueError("horizon must be positive")
    return prices.shift(-horizon) / prices - 1.0


def cumulative_return(returns: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Compound simple period returns into wealth growth minus one."""

    return (1.0 + returns.fillna(0.0)).cumprod() - 1.0
