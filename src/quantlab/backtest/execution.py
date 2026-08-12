"""Alignment checks and simple close-to-next-close execution semantics."""

from __future__ import annotations

import pandas as pd


def validate_backtest_inputs(asset_returns: pd.DataFrame, weights: pd.DataFrame) -> None:
    """Require exact alignment so pandas cannot silently add dates or assets."""

    if not asset_returns.index.equals(weights.index) or not asset_returns.columns.equals(weights.columns):
        raise ValueError("asset returns and weights must have identical index and columns")
    if not asset_returns.index.is_monotonic_increasing or asset_returns.index.has_duplicates:
        raise ValueError("backtest index must be sorted and unique")
    if weights.isna().any().any():
        raise ValueError("weights cannot contain missing values")


def next_period_returns(asset_returns: pd.DataFrame) -> pd.DataFrame:
    """Align conventional return at ``t+1`` to the decision made at close ``t``."""

    return asset_returns.shift(-1)


def portfolio_gross_return(forward_returns: pd.DataFrame, weights: pd.DataFrame) -> pd.Series:
    """Aggregate forward returns and reject missing returns for held assets.

    A completely missing final row is allowed because ``next_period_returns``
    deliberately creates it: there is no return after the final decision date.
    Missing data anywhere else must not silently reduce portfolio exposure.
    """

    missing_held_return = forward_returns.isna() & weights.ne(0.0)
    if not forward_returns.empty and forward_returns.iloc[-1].isna().all():
        missing_held_return.iloc[-1] = False
    if missing_held_return.any().any():
        raise ValueError("forward returns cannot be missing for assets with non-zero weights")

    return (forward_returns * weights).sum(axis=1, min_count=1).rename("gross_return")
