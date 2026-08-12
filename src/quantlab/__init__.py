"""Mini Quant Research Lab.

The package favors explicit DataFrame transformations so readers can follow a
research idea from market observations to an evaluated portfolio.
"""

from quantlab.data.schema import MarketDataError, validate_market_data

__all__ = ["MarketDataError", "validate_market_data"]
