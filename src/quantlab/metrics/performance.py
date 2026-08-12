"""Portfolio return and risk-adjusted performance statistics."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from quantlab.metrics.risk import annualized_volatility, maximum_drawdown


def total_return(returns: pd.Series) -> float:
    """Compound period returns over the complete sample."""

    values = returns.dropna().to_numpy(dtype=float)
    return float(np.prod(1.0 + values) - 1.0)


def annualized_return(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Annualize compounded growth using the number of observed periods."""

    clean = returns.dropna()
    if clean.empty:
        return float("nan")
    growth = float(np.prod(1.0 + clean.to_numpy(dtype=float)))
    if growth <= 0:
        return -1.0
    return float(growth ** (periods_per_year / len(clean)) - 1.0)


def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
    """Annualized Sharpe ratio using a population standard deviation.

    ``risk_free_rate`` is annual and is converted to an equivalent period rate.
    """

    clean = returns.dropna()
    if clean.empty:
        return float("nan")
    period_risk_free = (1.0 + risk_free_rate) ** (1.0 / periods_per_year) - 1.0
    excess = clean - period_risk_free
    volatility = float(excess.std(ddof=0))
    if math.isclose(volatility, 0.0):
        return float("nan")
    return float(np.sqrt(periods_per_year) * excess.mean() / volatility)


def hit_rate(returns: pd.Series) -> float:
    """Fraction of non-missing periods with a strictly positive strategy return."""

    clean = returns.dropna()
    return float((clean > 0).mean()) if not clean.empty else float("nan")


def performance_summary(returns: pd.Series, periods_per_year: int = 252) -> dict[str, float]:
    """Return the core metrics used consistently by experiments and backtests."""

    return {
        "total_return": total_return(returns),
        "cagr": annualized_return(returns, periods_per_year),
        "annualized_volatility": annualized_volatility(returns, periods_per_year),
        "sharpe": sharpe_ratio(returns, periods_per_year=periods_per_year),
        "max_drawdown": maximum_drawdown(returns),
        "hit_rate": hit_rate(returns),
    }
