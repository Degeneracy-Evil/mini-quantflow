"""Canonical market-data schema.

Market observations use a sorted long-form table.  One row means one asset at
one timestamp; this makes the time and cross-sectional axes explicit.
"""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

REQUIRED_COLUMNS: tuple[str, ...] = ("timestamp", "asset", "open", "high", "low", "close", "volume")
PRICE_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close")


class MarketDataError(ValueError):
    """Raised when market data violates the canonical schema."""


def validate_market_data(data: pd.DataFrame, *, copy: bool = True) -> pd.DataFrame:
    """Validate and normalize a canonical long-form market table.

    The result is ordered by ``timestamp`` and ``asset``. Duplicate observations
    are rejected because silently choosing one would change calculated returns.
    """

    missing = set(REQUIRED_COLUMNS).difference(data.columns)
    if missing:
        raise MarketDataError(f"missing required columns: {sorted(missing)}")

    result = data.copy() if copy else data
    try:
        result["timestamp"] = pd.to_datetime(result["timestamp"], errors="raise", format="mixed")
    except (TypeError, ValueError) as exc:
        raise MarketDataError("timestamp contains values that cannot be parsed") from exc

    if result["timestamp"].isna().any():
        raise MarketDataError("timestamp contains missing values")
    if result["asset"].isna().any():
        raise MarketDataError("asset contains missing values")
    result["asset"] = result["asset"].astype(str).astype(object)

    numeric_columns: Iterable[str] = (*PRICE_COLUMNS, "volume")
    for column in numeric_columns:
        try:
            result[column] = pd.to_numeric(result[column], errors="raise")
        except (TypeError, ValueError) as exc:
            raise MarketDataError(f"{column} contains non-numeric values") from exc

    if result[list(REQUIRED_COLUMNS)].isna().any().any():
        raise MarketDataError("required market-data columns contain missing values")
    if (result[list(PRICE_COLUMNS)] <= 0).any().any():
        raise MarketDataError("prices must be strictly positive")
    if (result["volume"] < 0).any():
        raise MarketDataError("volume must be non-negative")
    if (result["high"] < result[["open", "close"]].max(axis=1)).any():
        raise MarketDataError("high must be at least open and close")
    if (result["low"] > result[["open", "close"]].min(axis=1)).any():
        raise MarketDataError("low must be at most open and close")
    if result.duplicated(["timestamp", "asset"]).any():
        raise MarketDataError("duplicate (timestamp, asset) observations")

    return result.sort_values(["timestamp", "asset"], kind="stable").reset_index(drop=True)


def close_matrix(data: pd.DataFrame) -> pd.DataFrame:
    """Return a ``timestamp x asset`` close-price matrix from canonical data."""

    validated = validate_market_data(data)
    return validated.pivot(index="timestamp", columns="asset", values="close").sort_index()


def long_from_close(close: pd.DataFrame, *, volume: float = 1_000_000.0) -> pd.DataFrame:
    """Build canonical OHLCV data from closes for synthetic and test markets.

    Open is the previous close (or the initial close), while high and low bound
    open and close. No intraperiod price path is invented.
    """

    if close.empty or close.columns.empty:
        raise MarketDataError("close matrix must contain dates and assets")
    if close.isna().any().any() or (close <= 0).any().any():
        raise MarketDataError("close matrix must contain positive, non-missing prices")

    previous = close.shift(1).fillna(close)
    frame = pd.DataFrame(
        {
            "timestamp": close.index.repeat(len(close.columns)),
            "asset": list(close.columns) * len(close.index),
            "open": previous.to_numpy().ravel(),
            "close": close.to_numpy().ravel(),
        }
    )
    frame["high"] = frame[["open", "close"]].max(axis=1)
    frame["low"] = frame[["open", "close"]].min(axis=1)
    frame["volume"] = volume
    return validate_market_data(frame)
