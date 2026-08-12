"""CSV adapter for canonical market data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from quantlab.data.schema import validate_market_data


@dataclass(frozen=True)
class CSVDataSource:
    """Read an OHLCV CSV and enforce the internal schema at the boundary."""

    path: str | Path

    def load(self) -> pd.DataFrame:
        """Parse the CSV, including timestamps, and return validated data."""

        path = Path(self.path)
        if not path.is_file():
            raise FileNotFoundError(f"market CSV does not exist: {path}")
        return validate_market_data(pd.read_csv(path))
