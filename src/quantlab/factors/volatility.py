"""Cross-sectional low-volatility factor."""

from __future__ import annotations

import pandas as pd

from quantlab.factors.ranking import cross_sectional_zscore
from quantlab.features.volatility import rolling_volatility


def low_volatility_factor(returns: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Give higher standardized scores to assets with lower trailing volatility."""

    return cross_sectional_zscore(-rolling_volatility(returns, window))
