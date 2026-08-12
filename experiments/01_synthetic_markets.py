"""Inspect known synthetic data-generating processes."""

from __future__ import annotations

from quantlab.data import SyntheticDataSource, SyntheticMarketConfig, close_matrix
from quantlab.data.synthetic import MarketKind


def main() -> None:
    """Generate every offline market and print simple sanity statistics."""

    config = SyntheticMarketConfig(periods=252, assets=5, seed=42)
    kinds: tuple[MarketKind, ...] = ("random", "trend", "mean_reverting", "regime", "correlated")
    for kind in kinds:
        prices = close_matrix(SyntheticDataSource(kind, config).load())
        total = prices.iloc[-1] / prices.iloc[0] - 1.0
        print(f"{kind:15s} mean_total_return={total.mean():.4f} assets={prices.shape[1]} periods={prices.shape[0]}")


if __name__ == "__main__":
    main()
