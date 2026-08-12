"""Drawdown plots."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes

from quantlab.metrics.risk import drawdown_series


def plot_drawdown(returns: pd.Series, *, title: str = "Drawdown", ax: Axes | None = None) -> Axes:
    """Plot peak-to-trough loss from a strategy return series."""

    axis = ax or plt.subplots(figsize=(9, 3))[1]
    drawdown = drawdown_series(returns)
    axis.fill_between(drawdown.index, drawdown, 0.0, alpha=0.35)
    axis.set(title=title, ylabel="Drawdown", xlabel="Date")
    axis.grid(alpha=0.25)
    return axis
