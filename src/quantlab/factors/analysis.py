"""Factor evaluation against explicitly future returns."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from quantlab.factors.base import require_same_panel
from quantlab.factors.ranking import assign_quantiles

CorrelationMethod = Literal["pearson", "spearman"]


@dataclass(frozen=True)
class ICSummary:
    """Mean, dispersion, and information ratio of an IC time series."""

    mean: float
    standard_deviation: float
    information_ratio: float
    observations: int


@dataclass(frozen=True)
class FactorPortfolioReturns:
    """Returns and turnover of a top-minus-bottom factor portfolio."""

    long_return: pd.Series
    short_return: pd.Series
    long_short_return: pd.Series
    turnover: pd.Series
    weights: pd.DataFrame


def information_coefficient(
    factor_scores: pd.DataFrame,
    future_returns: pd.DataFrame,
    method: CorrelationMethod = "spearman",
    min_assets: int = 3,
) -> pd.Series:
    """Correlate factor at ``t`` with returns strictly after ``t`` for each date."""

    require_same_panel(factor_scores, future_returns)
    if min_assets < 2:
        raise ValueError("min_assets must be at least 2")
    if method not in ("pearson", "spearman"):
        raise ValueError(f"unsupported IC method: {method}")

    values: list[float] = []
    for timestamp in factor_scores.index:
        pair = pd.concat(
            [factor_scores.loc[timestamp].rename("factor"), future_returns.loc[timestamp].rename("future")], axis=1
        ).dropna()
        correlation = pair["factor"].corr(pair["future"], method=method) if len(pair) >= min_assets else np.nan
        values.append(float(correlation))
    return pd.Series(values, index=factor_scores.index, name=f"{method}_ic")


def summarize_ic(ic: pd.Series) -> ICSummary:
    """Summarize IC; IR is mean divided by time-series standard deviation."""

    clean = ic.dropna()
    if clean.empty:
        return ICSummary(np.nan, np.nan, np.nan, 0)
    standard_deviation = float(clean.std(ddof=1)) if len(clean) > 1 else np.nan
    information_ratio = float(clean.mean() / standard_deviation) if standard_deviation > 0 else np.nan
    return ICSummary(float(clean.mean()), standard_deviation, information_ratio, len(clean))


def quantile_returns(factor_scores: pd.DataFrame, future_returns: pd.DataFrame, quantiles: int = 5) -> pd.DataFrame:
    """Average future return of each factor quantile at every timestamp."""

    require_same_panel(factor_scores, future_returns)
    labels = assign_quantiles(factor_scores, quantiles)
    output = pd.DataFrame(index=factor_scores.index, columns=range(1, quantiles + 1), dtype=float)
    for quantile in output.columns:
        output[quantile] = future_returns.where(labels == quantile).mean(axis=1)
    output.columns = [f"Q{number}" for number in output.columns]
    output["top_minus_bottom"] = output[f"Q{quantiles}"] - output["Q1"]
    return output


def cumulative_quantile_returns(returns: pd.DataFrame) -> pd.DataFrame:
    """Compound each quantile's period returns independently."""

    return (1.0 + returns.fillna(0.0)).cumprod() - 1.0


def long_short_factor_portfolio(
    factor_scores: pd.DataFrame, future_returns: pd.DataFrame, quantile: float = 0.2
) -> FactorPortfolioReturns:
    """Build a dollar-neutral, equal-weight top/bottom factor portfolio.

    Weights at ``t`` multiply the supplied future returns aligned at ``t``.
    Long and short books each have gross exposure one when both sides exist.
    """

    require_same_panel(factor_scores, future_returns)
    if not 0.0 < quantile <= 0.5:
        raise ValueError("quantile must be in (0, 0.5]")
    percentile = factor_scores.rank(axis=1, pct=True, method="average")
    long_mask = percentile > 1.0 - quantile
    short_mask = percentile <= quantile
    long_weights = long_mask.div(long_mask.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    short_weights = short_mask.div(short_mask.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    weights = long_weights - short_weights
    long_return = (long_weights * future_returns).sum(axis=1, min_count=1)
    # This is the PnL contribution of the short book, positive when shorted assets fall.
    short_return = (-short_weights * future_returns).sum(axis=1, min_count=1)
    long_short_return = long_return + short_return
    turnover = weights.diff().abs().sum(axis=1).fillna(weights.abs().sum(axis=1))
    return FactorPortfolioReturns(long_return, short_return, long_short_return, turnover, weights)
