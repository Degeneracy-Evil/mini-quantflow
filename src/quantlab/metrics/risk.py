"""Risk statistics derived from realized strategy returns."""

from __future__ import annotations

import numpy as np
import pandas as pd


def annualized_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Scale population period volatility by the square root of time."""

    clean = returns.dropna()
    return float(clean.std(ddof=0) * np.sqrt(periods_per_year)) if not clean.empty else float("nan")


def drawdown_series(returns: pd.Series) -> pd.Series:
    """Compute percentage decline from the running equity peak.

    Initial capital is a peak of 1.0, so a loss in the first observed period is
    immediately a drawdown rather than becoming the series' starting peak.
    """

    equity = (1.0 + returns.fillna(0.0)).cumprod()
    peak = equity.cummax().clip(lower=1.0)
    return equity / peak - 1.0


def maximum_drawdown(returns: pd.Series) -> float:
    """Return the most negative peak-to-trough drawdown."""

    drawdown = drawdown_series(returns)
    return float(drawdown.min()) if not drawdown.empty else float("nan")
