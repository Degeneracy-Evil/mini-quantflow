"""Equity-curve plots."""

from __future__ import annotations

from collections.abc import Mapping

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes


def plot_equity_curves(
    curves: Mapping[str, pd.Series], *, title: str = "Equity Curves", ax: Axes | None = None
) -> Axes:
    """Plot one or more wealth indexes on a shared axis."""

    axis = ax or plt.subplots(figsize=(9, 5))[1]
    for name, curve in curves.items():
        axis.plot(curve.index, curve, label=name)
    axis.set(title=title, ylabel="Equity", xlabel="Date")
    axis.grid(alpha=0.25)
    axis.legend()
    return axis
