"""Cross-sectional momentum factor."""

from __future__ import annotations

import pandas as pd

from quantlab.factors.ranking import cross_sectional_zscore
from quantlab.features.momentum import momentum


def momentum_factor(prices: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """Z-score trailing returns across assets at each timestamp."""

    return cross_sectional_zscore(momentum(prices, lookback))
