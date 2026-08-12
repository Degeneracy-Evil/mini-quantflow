"""Cross-sectional ranking and robust standardization."""

from __future__ import annotations

import numpy as np
import pandas as pd


def cross_sectional_rank(scores: pd.DataFrame, *, percentile: bool = True, ascending: bool = True) -> pd.DataFrame:
    """Rank assets within each row; time never enters the ranking operation."""

    return scores.rank(axis=1, pct=percentile, ascending=ascending, method="average")


def cross_sectional_zscore(scores: pd.DataFrame) -> pd.DataFrame:
    """Center and scale every date's asset scores using population dispersion."""

    means = scores.mean(axis=1)
    standard_deviations = scores.std(axis=1, ddof=0).replace(0.0, np.nan)
    return scores.sub(means, axis=0).div(standard_deviations, axis=0)


def winsorize(scores: pd.DataFrame, lower: float = 0.01, upper: float = 0.99) -> pd.DataFrame:
    """Clip each cross-section to its lower and upper empirical quantiles."""

    if not 0.0 <= lower < upper <= 1.0:
        raise ValueError("quantiles must satisfy 0 <= lower < upper <= 1")
    lower_bounds = scores.quantile(lower, axis=1)
    upper_bounds = scores.quantile(upper, axis=1)
    return scores.clip(lower=lower_bounds, upper=upper_bounds, axis=0)


def assign_quantiles(scores: pd.DataFrame, quantiles: int = 5) -> pd.DataFrame:
    """Assign integer labels 1..N using stable cross-sectional ranks.

    Stable first ranks avoid ``qcut`` failures when many assets share a score.
    Missing scores remain missing.
    """

    if quantiles < 2:
        raise ValueError("quantiles must be at least 2")

    def assign(row: pd.Series) -> pd.Series:
        valid = row.dropna()
        result = pd.Series(np.nan, index=row.index, dtype=float)
        if len(valid) < quantiles:
            return result
        ranks = valid.rank(method="first")
        result.loc[valid.index] = pd.qcut(ranks, quantiles, labels=False).astype(float) + 1.0
        return result

    return scores.apply(assign, axis=1)
