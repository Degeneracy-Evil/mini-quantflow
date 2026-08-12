"""Readable signal-to-weight portfolio constructors."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantlab.portfolio.weights import normalize_long_only, normalize_long_short


def equal_weight(signal: pd.DataFrame) -> pd.DataFrame:
    """Equally weight positive selections in a long-only portfolio."""

    return normalize_long_only((signal > 0).astype(float))


def long_short_equal_weight(signal: pd.DataFrame, gross_each_side: float = 1.0) -> pd.DataFrame:
    """Equally weight both sides, staying in cash when either side is absent."""

    if gross_each_side <= 0:
        raise ValueError("gross_each_side must be positive")
    signed = pd.DataFrame(np.sign(signal.to_numpy()), index=signal.index, columns=signal.columns)
    weights = normalize_long_short(signed, gross_each_side, gross_each_side)
    has_both_sides = signed.gt(0.0).any(axis=1) & signed.lt(0.0).any(axis=1)
    return weights.where(has_both_sides, 0.0, axis=0)


def long_only_top_k(scores: pd.DataFrame, k: int = 10) -> pd.DataFrame:
    """Hold the highest-scoring K available assets at each date with equal weight."""

    if k < 1:
        raise ValueError("k must be positive")
    ranks = scores.rank(axis=1, ascending=False, method="first")
    selected = ((ranks <= k) & scores.notna()).astype(float)
    return normalize_long_only(selected)
