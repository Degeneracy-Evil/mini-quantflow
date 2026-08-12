"""Tests for return alignment, features, and performance statistics."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from quantlab.features.mean_reversion import distance_from_moving_average, rolling_zscore, short_term_reversal
from quantlab.features.momentum import momentum
from quantlab.features.returns import cumulative_return, forward_return, log_return, simple_return
from quantlab.features.technical import moving_average, moving_average_ratio, rsi
from quantlab.features.volatility import ewma_volatility, rolling_volatility
from quantlab.metrics.performance import annualized_return, hit_rate, performance_summary, sharpe_ratio, total_return
from quantlab.metrics.risk import annualized_volatility, drawdown_series, maximum_drawdown


def test_return_alignment() -> None:
    prices = pd.Series([100.0, 110.0, 121.0], index=pd.date_range("2024-01-01", periods=3))
    assert simple_return(prices).tolist()[1:] == pytest.approx([0.1, 0.1])
    assert log_return(prices).iloc[1] == pytest.approx(math.log(1.1))
    assert forward_return(prices).iloc[:2].tolist() == pytest.approx([0.1, 0.1])
    assert pd.isna(forward_return(prices).iloc[-1])
    assert cumulative_return(pd.Series([0.1, -0.1])).iloc[-1] == pytest.approx(-0.01)


@pytest.mark.parametrize("function", [simple_return, log_return])
def test_return_validation(function: object) -> None:
    with pytest.raises(ValueError, match="empty"):
        function(pd.Series(dtype=float))
    with pytest.raises(ValueError, match="strictly positive"):
        function(pd.Series([0.0, 1.0]))
    with pytest.raises(ValueError, match="positive"):
        function(pd.Series([1.0, 2.0]), periods=0)
    with pytest.raises(ValueError, match="horizon"):
        forward_return(pd.Series([1.0, 2.0]), horizon=0)


def test_rolling_features_use_trailing_values(close_prices: pd.DataFrame) -> None:
    returns = close_prices.pct_change(fill_method=None)
    assert momentum(close_prices, 2).iloc[2, 0] == pytest.approx(0.03)
    assert rolling_volatility(returns, 2).iloc[2, 0] == pytest.approx(np.std(returns["A"].iloc[1:3]))
    assert ewma_volatility(returns, 2).iloc[-1].notna().all()
    assert distance_from_moving_average(close_prices, 2).iloc[1, 0] == pytest.approx(101 / 100.5 - 1)
    assert rolling_zscore(close_prices, 2).iloc[1, 0] == pytest.approx(1.0)
    assert short_term_reversal(close_prices, 1).iloc[1, 0] == pytest.approx(-0.01)
    assert moving_average(close_prices, 2).iloc[1, 0] == pytest.approx(100.5)
    assert moving_average_ratio(close_prices, 2, 3).iloc[2, 0] == pytest.approx(102 / (304 / 3))
    assert rsi(close_prices, 2).dropna().shape[0] > 0


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (lambda prices: momentum(prices, 0), "lookback"),
        (lambda prices: rolling_volatility(prices, 1), "window"),
        (lambda prices: rolling_volatility(prices, 2, annualization=0), "annualization"),
        (lambda prices: ewma_volatility(prices, 1), "span"),
        (lambda prices: distance_from_moving_average(prices, 1), "window"),
        (lambda prices: rolling_zscore(prices, 1), "window"),
        (lambda prices: moving_average(prices, 0), "window"),
        (lambda prices: moving_average_ratio(prices, 3, 2), "windows"),
        (lambda prices: rsi(prices, 1), "window"),
    ],
)
def test_feature_parameter_validation(close_prices: pd.DataFrame, call: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        call(close_prices)


def test_performance_metrics_have_hand_calculable_values() -> None:
    returns = pd.Series([0.1, -0.2, 0.1, 0.0])
    assert total_return(returns) == pytest.approx(1.1 * 0.8 * 1.1 - 1)
    assert annualized_return(returns, 4) == pytest.approx(total_return(returns))
    assert annualized_volatility(returns, 4) == pytest.approx(returns.std(ddof=0) * 2)
    assert hit_rate(returns) == 0.5
    expected_drawdown = pd.Series([0.0, -0.2, -0.12, -0.12])
    assert drawdown_series(returns).tolist() == pytest.approx(expected_drawdown.tolist())
    assert maximum_drawdown(returns) == pytest.approx(-0.2)
    assert performance_summary(returns, 4)["max_drawdown"] == pytest.approx(-0.2)
    assert np.isfinite(sharpe_ratio(returns, periods_per_year=4))


def test_first_period_loss_is_a_drawdown_from_initial_capital() -> None:
    returns = pd.Series([-0.1, 0.05])
    assert drawdown_series(returns).tolist() == pytest.approx([-0.1, -0.055])
    assert maximum_drawdown(returns) == pytest.approx(-0.1)


def test_metric_empty_and_edge_cases() -> None:
    empty = pd.Series(dtype=float)
    assert math.isnan(annualized_return(empty))
    assert math.isnan(annualized_volatility(empty))
    assert math.isnan(sharpe_ratio(empty))
    assert math.isnan(hit_rate(empty))
    assert math.isnan(maximum_drawdown(empty))
    assert math.isnan(sharpe_ratio(pd.Series([0.0, 0.0])))
    assert annualized_return(pd.Series([-1.5]), 1) == -1.0
