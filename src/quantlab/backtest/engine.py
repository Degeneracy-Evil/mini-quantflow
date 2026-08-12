"""Backtest target weights with close-to-next-close execution."""

from __future__ import annotations

import pandas as pd

from quantlab.backtest.costs import TransactionCostModel
from quantlab.backtest.execution import next_period_returns, portfolio_gross_return, validate_backtest_inputs
from quantlab.backtest.result import BacktestResult
from quantlab.portfolio.rebalance import RebalanceFrequency, apply_rebalance_schedule
from quantlab.portfolio.weights import validate_weights


def run_backtest(
    asset_returns: pd.DataFrame,
    target_weights: pd.DataFrame,
    *,
    costs: TransactionCostModel | None = None,
    rebalance: RebalanceFrequency = "daily",
    benchmark_weights: pd.DataFrame | None = None,
    periods_per_year: int = 252,
    max_gross_exposure: float | None = None,
) -> BacktestResult:
    """Run a transparent close-to-next-close portfolio backtest.

    ``asset_returns[t]`` is the conventional return from ``t-1`` to ``t``.
    Target weights observed at close ``t`` therefore multiply
    ``asset_returns[t+1]``. This one-row shift is the central no-look-ahead rule.
    The final decision row is omitted because its next-period return is unknown.
    """

    validate_backtest_inputs(asset_returns, target_weights)
    held_weights = apply_rebalance_schedule(target_weights, rebalance)
    validate_weights(held_weights, max_gross=max_gross_exposure)
    turnover = held_weights.diff().abs().sum(axis=1)
    turnover.iloc[0] = held_weights.iloc[0].abs().sum()
    cost_model = costs or TransactionCostModel()
    transaction_costs = cost_model.calculate(turnover)

    forward = next_period_returns(asset_returns)
    gross_return = portfolio_gross_return(forward, held_weights)
    valid_rows = forward.notna().any(axis=1)
    net_return = (gross_return - transaction_costs).loc[valid_rows].rename("strategy_return")

    benchmark_returns: pd.Series | None = None
    if benchmark_weights is not None:
        validate_backtest_inputs(asset_returns, benchmark_weights)
        benchmark_returns = (
            portfolio_gross_return(forward, benchmark_weights).loc[valid_rows].rename("benchmark_return")
        )

    index = net_return.index
    return BacktestResult.from_series(
        net_return,
        held_weights.loc[index],
        turnover.loc[index],
        transaction_costs.loc[index],
        benchmark_returns,
        periods_per_year,
    )
