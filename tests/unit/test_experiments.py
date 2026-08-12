"""Tests for shared end-to-end experiment semantics."""

from __future__ import annotations

import pandas as pd
import pytest
from common import backtest_scores


def test_score_backtest_excludes_pre_prediction_cash_periods() -> None:
    index = pd.date_range("2024-01-01", periods=6, freq="B")
    prices = pd.DataFrame(
        {
            "A": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0],
            "B": [100.0, 99.0, 98.0, 97.0, 96.0, 95.0],
        },
        index=index,
    )
    scores = pd.DataFrame({"A": [1.0, 1.0, 1.0], "B": [-1.0, -1.0, -1.0]}, index=index[2:5])

    result = backtest_scores(prices, scores, quantile=0.5)

    assert result.period_returns.index.equals(index[2:5])
    assert len(result.period_returns) == len(scores)


def test_score_backtest_requires_an_aligned_observation() -> None:
    index = pd.date_range("2024-01-01", periods=3, freq="B")
    prices = pd.DataFrame({"A": [100.0, 101.0, 102.0]}, index=index)
    scores = pd.DataFrame({"A": [1.0]}, index=pd.DatetimeIndex(["2023-12-01"]))

    with pytest.raises(ValueError, match="no observations"):
        backtest_scores(prices, scores)
