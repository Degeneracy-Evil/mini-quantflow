"""Optional AKShare adapter.

AKShare is imported only inside ``load`` so the core package stays usable
offline and without the optional dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from quantlab.data.schema import validate_market_data


@dataclass(frozen=True)
class AKShareDataSource:
    """Download one Chinese A-share symbol and normalize it to our schema."""

    symbol: str
    start_date: str
    end_date: str
    adjust: str = "qfq"

    def load(self) -> pd.DataFrame:
        """Fetch daily history, raising a clear error for dependency/network failures."""

        try:
            import akshare as ak
        except ImportError as exc:
            raise RuntimeError("AKShare is optional; install mini-quantflow[akshare]") from exc

        try:
            raw: Any = ak.stock_zh_a_hist(
                symbol=self.symbol,
                period="daily",
                start_date=self.start_date,
                end_date=self.end_date,
                adjust=self.adjust,
            )
        except Exception as exc:
            raise RuntimeError(f"AKShare request failed for {self.symbol}: {exc}") from exc

        frame = pd.DataFrame(raw).rename(
            columns={
                "日期": "timestamp",
                "开盘": "open",
                "最高": "high",
                "最低": "low",
                "收盘": "close",
                "成交量": "volume",
            }
        )
        frame["asset"] = self.symbol
        return validate_market_data(frame)
