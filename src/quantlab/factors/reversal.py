"""Cross-sectional short-term reversal factor."""

from __future__ import annotations

import pandas as pd

from quantlab.factors.ranking import cross_sectional_zscore
from quantlab.features.mean_reversion import short_term_reversal


def reversal_factor(prices: pd.DataFrame, lookback: int = 5) -> pd.DataFrame:
    """Standardize negated recent returns across assets."""

    return cross_sectional_zscore(short_term_reversal(prices, lookback))
