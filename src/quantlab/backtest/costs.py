"""Linear transaction-cost assumptions."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class TransactionCostModel:
    """Commission and slippage charged per unit of portfolio turnover."""

    commission_bps: float = 0.0
    slippage_bps: float = 0.0

    def __post_init__(self) -> None:
        if self.commission_bps < 0 or self.slippage_bps < 0:
            raise ValueError("cost rates cannot be negative")

    @property
    def rate(self) -> float:
        """Combined one-way linear cost as a decimal fraction."""

        return (self.commission_bps + self.slippage_bps) / 10_000.0

    def calculate(self, turnover: pd.Series) -> pd.Series:
        """Apply ``cost = turnover * rate`` independently per period."""

        if (turnover < 0).any():
            raise ValueError("turnover cannot be negative")
        return turnover * self.rate
