"""Unified backtest outputs."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from quantlab.metrics.performance import performance_summary


@dataclass(frozen=True)
class BacktestResult:
    """All series needed to audit and evaluate a backtest."""

    equity_curve: pd.Series
    period_returns: pd.Series
    weights: pd.DataFrame
    turnover: pd.Series
    transaction_costs: pd.Series
    benchmark_returns: pd.Series | None = None
    metrics: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_series(
        cls,
        period_returns: pd.Series,
        weights: pd.DataFrame,
        turnover: pd.Series,
        transaction_costs: pd.Series,
        benchmark_returns: pd.Series | None = None,
        periods_per_year: int = 252,
    ) -> BacktestResult:
        """Construct equity and metric outputs from aligned realized series."""

        equity_curve = (1.0 + period_returns).cumprod().rename("equity")
        metrics = performance_summary(period_returns, periods_per_year)
        metrics["average_turnover"] = float(turnover.mean())
        metrics["total_transaction_cost"] = float(transaction_costs.sum())
        return cls(equity_curve, period_returns, weights, turnover, transaction_costs, benchmark_returns, metrics)

    def summary(self) -> pd.Series:
        """Return metrics as a display-friendly named series."""

        return pd.Series(self.metrics, name="strategy")
