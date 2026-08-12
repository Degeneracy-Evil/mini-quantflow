"""Tests for schema boundaries and synthetic data sources."""

from __future__ import annotations

import pandas as pd
import pytest

from quantlab.data.csv_source import CSVDataSource
from quantlab.data.loader import load_market_data
from quantlab.data.schema import MarketDataError, close_matrix, long_from_close, validate_market_data
from quantlab.data.synthetic import SyntheticDataSource, SyntheticMarketConfig, factor_market


def test_schema_normalizes_order_and_types(market_data: pd.DataFrame) -> None:
    shuffled = market_data.sample(frac=1.0, random_state=1)
    shuffled["asset"] = shuffled["asset"].astype("string")
    result = validate_market_data(shuffled)
    assert result.equals(result.sort_values(["timestamp", "asset"]).reset_index(drop=True))
    assert result["asset"].dtype == object


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda frame: frame.drop(columns="close"), "missing required"),
        (lambda frame: pd.concat([frame, frame.iloc[[0]]]), "duplicate"),
        (lambda frame: frame.assign(close=-1.0), "strictly positive"),
        (lambda frame: frame.assign(volume=-1.0), "non-negative"),
        (lambda frame: frame.assign(high=0.5), "high must"),
        (lambda frame: frame.assign(low=1_000.0), "low must"),
        (lambda frame: frame.assign(timestamp="not-a-date"), "cannot be parsed"),
        (lambda frame: frame.assign(asset=None), "asset contains"),
        (lambda frame: frame.assign(volume="not-a-number"), "non-numeric"),
    ],
)
def test_schema_rejects_invalid_rows(market_data: pd.DataFrame, mutation: object, message: str) -> None:
    invalid = mutation(market_data.copy())
    with pytest.raises(MarketDataError, match=message):
        validate_market_data(invalid)


def test_close_round_trip(close_prices: pd.DataFrame) -> None:
    assert close_matrix(long_from_close(close_prices)).equals(close_prices.rename_axis(columns="asset"))
    with pytest.raises(MarketDataError, match="must contain dates"):
        long_from_close(pd.DataFrame())
    with pytest.raises(MarketDataError, match="positive"):
        long_from_close(pd.DataFrame({"A": [0.0]}, index=pd.date_range("2024-01-01", periods=1)))


def test_csv_source(tmp_path: pytest.TempPathFactory, market_data: pd.DataFrame) -> None:
    path = tmp_path / "market.csv"
    market_data.to_csv(path, index=False)
    loaded = CSVDataSource(path).load()
    assert loaded.shape == market_data.shape
    assert load_market_data(CSVDataSource(path).load).equals(loaded)
    with pytest.raises(FileNotFoundError):
        CSVDataSource(path.with_name("missing.csv")).load()


@pytest.mark.parametrize("kind", ["random", "trend", "mean_reverting", "regime", "correlated"])
def test_synthetic_markets_are_valid_and_deterministic(kind: str) -> None:
    config = SyntheticMarketConfig(periods=20, assets=3, seed=7)
    first = SyntheticDataSource(kind, config).load()  # type: ignore[arg-type]
    second = SyntheticDataSource(kind, config).load()  # type: ignore[arg-type]
    assert first.equals(second)
    assert first.shape == (60, 7)


def test_synthetic_config_and_kind_validation() -> None:
    with pytest.raises(ValueError, match="periods"):
        SyntheticMarketConfig(periods=1)
    with pytest.raises(ValueError, match="assets"):
        SyntheticMarketConfig(assets=0)
    with pytest.raises(ValueError, match="initial_price"):
        SyntheticMarketConfig(initial_price=0)
    with pytest.raises(ValueError, match="volatility"):
        SyntheticMarketConfig(volatility=-1)
    with pytest.raises(ValueError, match="unsupported"):
        SyntheticDataSource("unknown").load()  # type: ignore[arg-type]


def test_factor_market_embeds_forward_relationship() -> None:
    market, factor = factor_market(periods=80, assets=30, factor_premium=0.01, noise=0.001, seed=3)
    prices = close_matrix(market)
    realized = prices.pct_change(fill_method=None).shift(-1)
    correlations = [factor.iloc[row].corr(realized.iloc[row]) for row in range(len(factor) - 1)]
    assert pd.Series(correlations).mean() > 0.9
    with pytest.raises(ValueError, match="at least two"):
        factor_market(periods=1)
    with pytest.raises(ValueError, match="non-negative"):
        factor_market(noise=-1)
