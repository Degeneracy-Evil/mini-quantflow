"""Synthetic markets with known structure for offline research experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from quantlab.data.schema import long_from_close

MarketKind = Literal["random", "trend", "mean_reverting", "regime", "correlated"]


@dataclass(frozen=True)
class SyntheticMarketConfig:
    """Parameters shared by synthetic market generators.

    ``drift`` and ``volatility`` are per-period log-return parameters rather
    than annualized values. A fixed seed reproduces the complete output.
    """

    periods: int = 504
    assets: int = 5
    start: str = "2020-01-01"
    initial_price: float = 100.0
    drift: float = 0.0004
    volatility: float = 0.01
    seed: int = 42

    def __post_init__(self) -> None:
        if self.periods < 2:
            raise ValueError("periods must be at least 2")
        if self.assets < 1:
            raise ValueError("assets must be positive")
        if self.initial_price <= 0:
            raise ValueError("initial_price must be positive")
        if self.volatility < 0:
            raise ValueError("volatility must be non-negative")


class SyntheticDataSource:
    """Generate canonical OHLCV data for a selected synthetic process."""

    def __init__(self, kind: MarketKind = "random", config: SyntheticMarketConfig | None = None) -> None:
        self.kind = kind
        self.config = config or SyntheticMarketConfig()

    def load(self) -> pd.DataFrame:
        """Generate a market; repeated calls are identical for a fixed seed."""

        generators = {
            "random": self._random_or_trend,
            "trend": self._random_or_trend,
            "mean_reverting": self._mean_reverting,
            "regime": self._regime,
            "correlated": self._correlated,
        }
        try:
            close = generators[self.kind]()
        except KeyError as exc:
            raise ValueError(f"unsupported synthetic market kind: {self.kind}") from exc
        return long_from_close(close)

    def _index_and_columns(self) -> tuple[pd.DatetimeIndex, list[str]]:
        config = self.config
        index = pd.date_range(config.start, periods=config.periods, freq="B")
        columns = [f"ASSET_{number:03d}" for number in range(config.assets)]
        return index, columns

    def _prices_from_log_returns(self, log_returns: np.ndarray) -> pd.DataFrame:
        config = self.config
        index, columns = self._index_and_columns()
        log_returns[0, :] = 0.0
        prices = config.initial_price * np.exp(np.cumsum(log_returns, axis=0))
        return pd.DataFrame(prices, index=index, columns=columns)

    def _random_or_trend(self) -> pd.DataFrame:
        config = self.config
        rng = np.random.default_rng(config.seed)
        drift = config.drift if self.kind == "trend" else 0.0
        returns = rng.normal(drift, config.volatility, size=(config.periods, config.assets))
        return self._prices_from_log_returns(returns)

    def _mean_reverting(self) -> pd.DataFrame:
        config = self.config
        rng = np.random.default_rng(config.seed)
        log_price = np.full((config.periods, config.assets), np.log(config.initial_price))
        long_run = np.log(config.initial_price)
        for step in range(1, config.periods):
            shock = rng.normal(0.0, config.volatility, size=config.assets)
            log_price[step] = log_price[step - 1] + 0.12 * (long_run - log_price[step - 1]) + shock
        index, columns = self._index_and_columns()
        return pd.DataFrame(np.exp(log_price), index=index, columns=columns)

    def _regime(self) -> pd.DataFrame:
        config = self.config
        rng = np.random.default_rng(config.seed)
        returns = np.zeros((config.periods, config.assets))
        boundaries = np.linspace(0, config.periods, 5, dtype=int)
        returns[boundaries[0] : boundaries[1]] = rng.normal(
            config.drift, config.volatility, size=(boundaries[1] - boundaries[0], config.assets)
        )
        returns[boundaries[1] : boundaries[2]] = rng.normal(
            0.0, config.volatility * 2.5, size=(boundaries[2] - boundaries[1], config.assets)
        )
        # Negative autocorrelation creates a simple mean-reverting return regime.
        for step in range(max(1, boundaries[2]), boundaries[3]):
            returns[step] = -0.55 * returns[step - 1] + rng.normal(0.0, config.volatility, config.assets)
        returns[boundaries[3] : boundaries[4]] = rng.normal(
            config.drift, config.volatility, size=(boundaries[4] - boundaries[3], config.assets)
        )
        return self._prices_from_log_returns(returns)

    def _correlated(self) -> pd.DataFrame:
        config = self.config
        rng = np.random.default_rng(config.seed)
        common = rng.normal(config.drift, config.volatility, size=(config.periods, 1))
        idiosyncratic = rng.normal(0.0, config.volatility, size=(config.periods, config.assets))
        returns = np.sqrt(0.4) * common + np.sqrt(0.6) * idiosyncratic
        return self._prices_from_log_returns(returns)


def factor_market(
    *,
    periods: int = 504,
    assets: int = 100,
    factor_premium: float = 0.002,
    noise: float = 0.01,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate a cross-sectional market with a weak observable factor.

    At each date, the returned factor score predicts the next period's return.
    The final score has no corresponding future return and is still returned so
    downstream alignment code must handle the unavailable label explicitly.
    """

    if periods < 2 or assets < 2:
        raise ValueError("factor market needs at least two periods and assets")
    if noise < 0:
        raise ValueError("noise must be non-negative")
    rng = np.random.default_rng(seed)
    index = pd.date_range("2020-01-01", periods=periods, freq="B")
    columns = [f"ASSET_{number:03d}" for number in range(assets)]
    raw_score = rng.normal(size=(periods, assets))
    score = (raw_score - raw_score.mean(axis=1, keepdims=True)) / raw_score.std(axis=1, keepdims=True)
    returns = np.zeros((periods, assets))
    returns[1:] = score[:-1] * factor_premium + rng.normal(0.0, noise, size=(periods - 1, assets))
    prices = 100.0 * np.exp(np.cumsum(returns, axis=0))
    close = pd.DataFrame(prices, index=index, columns=columns)
    factor = pd.DataFrame(score, index=index, columns=columns)
    return long_from_close(close), factor
