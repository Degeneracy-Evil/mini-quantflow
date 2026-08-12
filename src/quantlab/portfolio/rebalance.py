"""Rebalance schedules that turn target weights into held weights."""

from __future__ import annotations

from collections.abc import Collection
from typing import Literal

import pandas as pd

RebalanceFrequency = Literal["daily", "weekly", "monthly"]


def rebalance_mask(index: pd.DatetimeIndex, frequency: RebalanceFrequency = "daily") -> pd.Series:
    """Mark the first available observation of each rebalance period."""

    if not index.is_monotonic_increasing or index.has_duplicates:
        raise ValueError("rebalance index must be sorted and unique")
    if frequency == "daily":
        return pd.Series(True, index=index, dtype=bool)
    if frequency == "weekly":
        period = pd.Series(index.to_period("W"), index=index)
        return period.ne(period.shift())
    if frequency == "monthly":
        period = pd.Series(index.to_period("M"), index=index)
        return period.ne(period.shift())
    raise ValueError(f"unsupported rebalance frequency: {frequency}")


def apply_rebalance_schedule(
    target_weights: pd.DataFrame,
    frequency: RebalanceFrequency = "daily",
    dates: Collection[pd.Timestamp] | None = None,
) -> pd.DataFrame:
    """Hold previous weights between rebalance dates; start uninvested if needed."""

    if not isinstance(target_weights.index, pd.DatetimeIndex):
        raise TypeError("weight index must be a DatetimeIndex")
    if dates is None:
        mask = rebalance_mask(target_weights.index, frequency)
    else:
        date_set = set(dates)
        mask = pd.Series(target_weights.index.isin(date_set), index=target_weights.index)
    return target_weights.where(mask, axis=0).ffill().fillna(0.0)
