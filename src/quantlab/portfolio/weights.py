"""Explicit portfolio weight normalization."""

from __future__ import annotations

import numpy as np
import pandas as pd


def normalize_long_only(weights: pd.DataFrame, gross_exposure: float = 1.0) -> pd.DataFrame:
    """Clip negative values and scale every invested row to target gross exposure."""

    if gross_exposure <= 0:
        raise ValueError("gross_exposure must be positive")
    positive = weights.clip(lower=0.0)
    totals = positive.sum(axis=1).replace(0.0, np.nan)
    return positive.div(totals, axis=0).fillna(0.0) * gross_exposure


def normalize_long_short(weights: pd.DataFrame, long_gross: float = 1.0, short_gross: float = 1.0) -> pd.DataFrame:
    """Scale positive and negative books independently to requested gross exposure."""

    if long_gross < 0 or short_gross < 0 or long_gross + short_gross == 0:
        raise ValueError("gross exposures must be non-negative and not both zero")
    positive = weights.clip(lower=0.0)
    negative = -weights.clip(upper=0.0)
    long_total = positive.sum(axis=1).replace(0.0, np.nan)
    short_total = negative.sum(axis=1).replace(0.0, np.nan)
    long_book = positive.div(long_total, axis=0).fillna(0.0) * long_gross
    short_book = negative.div(short_total, axis=0).fillna(0.0) * short_gross
    return long_book - short_book


def validate_weights(weights: pd.DataFrame, *, max_gross: float | None = None) -> None:
    """Reject non-finite or over-leveraged weights before backtesting."""

    if not np.isfinite(weights.to_numpy()).all():
        raise ValueError("weights must contain only finite values")
    if max_gross is not None and (weights.abs().sum(axis=1) > max_gross + 1e-10).any():
        raise ValueError("portfolio gross exposure exceeds the configured maximum")
