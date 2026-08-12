"""Tests for time alignment, costs, turnover, and equity updates."""

from __future__ import annotations

import pandas as pd
import pytest

from quantlab.backtest.costs import TransactionCostModel
from quantlab.backtest.engine import run_backtest
from quantlab.backtest.execution import portfolio_gross_return, validate_backtest_inputs


def test_backtest_applies_weights_to_next_return() -> None:
    index = pd.date_range("2024-01-01", periods=4)
    returns = pd.DataFrame({"A": [0.0, 0.1, -0.2, 0.05], "B": [0.0, 0.0, 0.1, -0.1]}, index=index)
    weights = pd.DataFrame({"A": [1.0, 0.0, 1.0, 0.0], "B": [0.0, 1.0, 0.0, 1.0]}, index=index)
    result = run_backtest(returns, weights)
    assert result.period_returns.tolist() == pytest.approx([0.1, 0.1, 0.05])
    assert result.equity_curve.iloc[-1] == pytest.approx(1.1 * 1.1 * 1.05)
    assert result.turnover.tolist() == pytest.approx([1.0, 2.0, 2.0])


def test_transaction_cost_and_benchmark() -> None:
    index = pd.date_range("2024-01-01", periods=3)
    returns = pd.DataFrame({"A": [0.0, 0.1, 0.2], "B": [0.0, 0.0, 0.0]}, index=index)
    weights = pd.DataFrame({"A": [1.0, 0.0, 0.0], "B": [0.0, 1.0, 1.0]}, index=index)
    benchmark = pd.DataFrame(0.5, index=index, columns=returns.columns)
    costs = TransactionCostModel(commission_bps=10, slippage_bps=5)
    result = run_backtest(returns, weights, costs=costs, benchmark_weights=benchmark)
    assert costs.rate == pytest.approx(0.0015)
    assert result.transaction_costs.tolist() == pytest.approx([0.0015, 0.003])
    assert result.period_returns.tolist() == pytest.approx([0.0985, -0.003])
    assert result.benchmark_returns is not None
    assert result.benchmark_returns.tolist() == pytest.approx([0.05, 0.1])
    assert result.summary()["total_transaction_cost"] == pytest.approx(0.0045)


def test_backtest_validation() -> None:
    index = pd.date_range("2024-01-01", periods=2)
    returns = pd.DataFrame({"A": [0.0, 0.1]}, index=index)
    weights = pd.DataFrame({"B": [1.0, 1.0]}, index=index)
    with pytest.raises(ValueError, match="identical"):
        run_backtest(returns, weights)
    with pytest.raises(ValueError, match="missing"):
        validate_backtest_inputs(returns, pd.DataFrame({"A": [float("nan"), 1.0]}, index=index))
    with pytest.raises(ValueError, match="negative"):
        TransactionCostModel(-1)
    with pytest.raises(ValueError, match="negative"):
        TransactionCostModel().calculate(pd.Series([-1.0]))
    with pytest.raises(ValueError, match="exceeds"):
        run_backtest(returns, pd.DataFrame({"A": [2.0, 2.0]}, index=index), max_gross_exposure=1.0)
    with pytest.raises(ValueError, match="sorted"):
        validate_backtest_inputs(returns.iloc[::-1], pd.DataFrame({"A": [1.0, 1.0]}, index=index[::-1]))


def test_portfolio_gross_return_preserves_missing_final_row() -> None:
    index = pd.date_range("2024-01-01", periods=2)
    forward = pd.DataFrame({"A": [0.1, float("nan")]}, index=index)
    weights = pd.DataFrame({"A": [1.0, 1.0]}, index=index)
    result = portfolio_gross_return(forward, weights)
    assert result.iloc[0] == pytest.approx(0.1)
    assert pd.isna(result.iloc[1])
