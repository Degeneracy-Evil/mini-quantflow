"""Past-return momentum features."""

from __future__ import annotations

import pandas as pd

from quantlab.features.returns import simple_return


def momentum(prices: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """Return each asset's trailing ``lookback``-period simple return."""

    if lookback < 1:
        raise ValueError("lookback must be positive")
    result = simple_return(prices, periods=lookback)
    if not isinstance(result, pd.DataFrame):  # pragma: no cover - input fixes the type
        raise TypeError("expected DataFrame result")
    return result
