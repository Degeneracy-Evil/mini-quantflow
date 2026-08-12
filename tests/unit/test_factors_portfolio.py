"""Tests for cross-sectional factor and portfolio transformations."""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd
import pytest

from quantlab.factors.analysis import (
    cumulative_quantile_returns,
    information_coefficient,
    long_short_factor_portfolio,
    quantile_returns,
    summarize_ic,
)
from quantlab.factors.momentum import momentum_factor
from quantlab.factors.ranking import assign_quantiles, cross_sectional_rank, cross_sectional_zscore, winsorize
from quantlab.factors.reversal import reversal_factor
from quantlab.factors.volatility import low_volatility_factor
from quantlab.portfolio.construction import equal_weight, long_only_top_k, long_short_equal_weight
from quantlab.portfolio.rebalance import apply_rebalance_schedule, rebalance_mask
from quantlab.portfolio.weights import normalize_long_only, normalize_long_short, validate_weights
from quantlab.signals.prediction import prediction_ranking_signal, prediction_threshold_signal
from quantlab.signals.ranking import ranking_signal
from quantlab.signals.threshold import threshold_signal


def _scores() -> pd.DataFrame:
    return pd.DataFrame(
        [[1.0, 2.0, 3.0, 4.0, 5.0], [5.0, 4.0, 3.0, 2.0, 1.0]],
        index=pd.date_range("2024-01-01", periods=2),
        columns=list("ABCDE"),
    )


def test_cross_sectional_processing() -> None:
    scores = _scores()
    assert cross_sectional_rank(scores).iloc[0].tolist() == pytest.approx([0.2, 0.4, 0.6, 0.8, 1.0])
    assert cross_sectional_zscore(scores).mean(axis=1).tolist() == pytest.approx([0.0, 0.0])
    clipped = winsorize(scores, 0.2, 0.8)
    assert clipped.iloc[0, 0] == pytest.approx(1.8)
    labels = assign_quantiles(scores, 5)
    assert labels.iloc[0].tolist() == [1.0, 2.0, 3.0, 4.0, 5.0]
    too_small = assign_quantiles(scores.iloc[:, :2], 3)
    assert too_small.isna().all().all()


def test_factor_metrics_and_portfolio() -> None:
    scores = _scores()
    future = scores / 100.0
    pearson = information_coefficient(scores, future, "pearson")
    spearman = information_coefficient(scores, future)
    assert pearson.tolist() == pytest.approx([1.0, 1.0])
    assert spearman.tolist() == pytest.approx([1.0, 1.0])
    summary = summarize_ic(pd.Series([0.1, 0.2, 0.3]))
    assert summary.mean == pytest.approx(0.2)
    assert summary.observations == 3
    returns = quantile_returns(scores, future)
    assert returns.loc[returns.index[0], "Q5"] == pytest.approx(0.05)
    assert cast(float, cumulative_quantile_returns(returns).iloc[-1, 0]) > 0
    portfolio = long_short_factor_portfolio(scores, future, 0.2)
    assert portfolio.weights.abs().sum(axis=1).tolist() == pytest.approx([2.0, 2.0])
    assert portfolio.long_short_return.tolist() == pytest.approx([0.04, 0.04])


def test_factor_validation() -> None:
    scores = _scores()
    misaligned = scores.rename(columns={"A": "Z"})
    with pytest.raises(ValueError, match="identical"):
        information_coefficient(scores, misaligned)
    with pytest.raises(ValueError, match="min_assets"):
        information_coefficient(scores, scores, min_assets=1)
    with pytest.raises(ValueError, match="unsupported"):
        information_coefficient(scores, scores, "kendall")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="quantiles"):
        assign_quantiles(scores, 1)
    with pytest.raises(ValueError, match="quantiles"):
        winsorize(scores, 0.8, 0.2)
    with pytest.raises(ValueError, match="quantile"):
        long_short_factor_portfolio(scores, scores, 0.8)
    empty_summary = summarize_ic(pd.Series(dtype=float))
    assert empty_summary.observations == 0
    assert np.isnan(empty_summary.mean)


def test_builtin_factors(close_prices: pd.DataFrame) -> None:
    returns = close_prices.pct_change(fill_method=None)
    assert momentum_factor(close_prices, 2).iloc[-1].notna().all()
    assert reversal_factor(close_prices, 2).iloc[-1].notna().all()
    assert low_volatility_factor(returns, 2).iloc[-1].notna().all()


def test_signals_and_weight_construction() -> None:
    scores = _scores()
    threshold = threshold_signal(scores, long_below=2, short_above=4)
    assert threshold.iloc[0].tolist() == [1.0, 0.0, 0.0, 0.0, -1.0]
    ranks = ranking_signal(scores, 0.2)
    assert ranks.iloc[0].tolist() == [-1.0, 0.0, 0.0, 0.0, 1.0]
    assert prediction_ranking_signal(scores, 0.2).equals(ranks)
    predicted = prediction_threshold_signal(scores - 3.0, 1.0)
    assert predicted.iloc[0].tolist() == [-1.0, 0.0, 0.0, 0.0, 1.0]
    long_short = long_short_equal_weight(ranks)
    assert long_short.sum(axis=1).tolist() == pytest.approx([0.0, 0.0])
    assert long_short.abs().sum(axis=1).tolist() == pytest.approx([2.0, 2.0])
    one_sided = long_short_equal_weight(pd.DataFrame([[1.0, 0.0], [0.0, -1.0]]))
    assert one_sided.eq(0.0).all().all()
    assert equal_weight(ranks).sum(axis=1).tolist() == pytest.approx([1.0, 1.0])
    assert long_only_top_k(scores, 2).sum(axis=1).tolist() == pytest.approx([1.0, 1.0])


def test_signal_and_weight_validation() -> None:
    scores = _scores()
    with pytest.raises(ValueError, match="less"):
        threshold_signal(scores, long_below=1, short_above=1)
    with pytest.raises(ValueError, match="quantile"):
        ranking_signal(scores, 0.8)
    with pytest.raises(ValueError, match="non-negative"):
        prediction_threshold_signal(scores, -1)
    with pytest.raises(ValueError, match="positive"):
        long_only_top_k(scores, 0)
    with pytest.raises(ValueError, match="positive"):
        long_short_equal_weight(scores, 0)
    with pytest.raises(ValueError, match="gross"):
        normalize_long_only(scores, 0)
    with pytest.raises(ValueError, match="gross"):
        normalize_long_short(scores, 0, 0)
    with pytest.raises(ValueError, match="finite"):
        validate_weights(pd.DataFrame([[np.nan]]))
    with pytest.raises(ValueError, match="exceeds"):
        validate_weights(pd.DataFrame([[1.0, 1.0]]), max_gross=1.0)


def test_rebalance_schedule() -> None:
    index = pd.date_range("2024-01-29", periods=8, freq="B")
    weights = pd.DataFrame({"A": range(8)}, index=index, dtype=float)
    daily = apply_rebalance_schedule(weights, "daily")
    weekly = apply_rebalance_schedule(weights, "weekly")
    monthly = apply_rebalance_schedule(weights, "monthly")
    explicit = apply_rebalance_schedule(weights, dates=[index[2]])
    assert daily.equals(weights)
    assert weekly.iloc[1, 0] == weekly.iloc[0, 0]
    assert monthly.loc[pd.Timestamp("2024-02-01"), "A"] == 3.0
    assert explicit.iloc[0, 0] == 0.0 and explicit.iloc[-1, 0] == 2.0
    with pytest.raises(ValueError, match="unsupported"):
        rebalance_mask(index, "yearly")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="sorted"):
        rebalance_mask(index[::-1])
    with pytest.raises(TypeError, match="DatetimeIndex"):
        apply_rebalance_schedule(weights.reset_index(drop=True))
