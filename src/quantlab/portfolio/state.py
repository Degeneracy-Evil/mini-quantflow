"""Small state object used when teaching step-by-step portfolio updates."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class PortfolioState:
    """Portfolio weights and equity immediately after one time step."""

    timestamp: pd.Timestamp
    weights: pd.Series
    equity: float
