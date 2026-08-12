"""Small common interface for market-data sources."""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from quantlab.data.schema import validate_market_data


def load_market_data(loader: Callable[[], pd.DataFrame]) -> pd.DataFrame:
    """Load from any source and enforce schema validation at the shared boundary."""

    return validate_market_data(loader())
