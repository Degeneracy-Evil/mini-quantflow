"""Shared small market fixtures."""

from __future__ import annotations

import pandas as pd
import pytest

from quantlab.data.schema import long_from_close


@pytest.fixture
def close_prices() -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=6, freq="B")
    return pd.DataFrame(
        {
            "A": [100.0, 101.0, 103.0, 102.0, 105.0, 108.0],
            "B": [50.0, 49.0, 48.0, 49.0, 48.0, 47.0],
            "C": [80.0, 82.0, 81.0, 83.0, 84.0, 85.0],
            "D": [40.0, 41.0, 42.0, 43.0, 44.0, 45.0],
            "E": [70.0, 69.0, 68.0, 67.0, 66.0, 65.0],
        },
        index=index,
    )


@pytest.fixture
def market_data(close_prices: pd.DataFrame) -> pd.DataFrame:
    return long_from_close(close_prices)
