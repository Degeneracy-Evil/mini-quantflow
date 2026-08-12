"""Past-looking volatility features."""

from __future__ import annotations

import pandas as pd


def rolling_volatility(returns: pd.DataFrame, window: int = 20, *, annualization: int | None = None) -> pd.DataFrame:
    """Compute rolling population standard deviation from observed returns."""

    if window < 2:
        raise ValueError("window must be at least 2")
    result = returns.rolling(window=window, min_periods=window).std(ddof=0)
    if annualization is not None:
        if annualization < 1:
            raise ValueError("annualization must be positive")
        result = result * annualization**0.5
    return result


def ewma_volatility(returns: pd.DataFrame, span: int = 20) -> pd.DataFrame:
    """Compute exponentially weighted past volatility."""

    if span < 2:
        raise ValueError("span must be at least 2")
    return returns.ewm(span=span, adjust=False, min_periods=span).std(bias=True)
