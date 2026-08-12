"""Factor quantile and information-coefficient plots."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes


def plot_quantile_returns(returns: pd.DataFrame, *, ax: Axes | None = None) -> Axes:
    """Plot average future return of columns named Q1, Q2, and so on."""

    axis = ax or plt.subplots(figsize=(7, 4))[1]
    quantile_columns = [column for column in returns if str(column).startswith("Q")]
    returns[quantile_columns].mean().plot.bar(ax=axis)
    axis.set(title="Average Future Return by Factor Quantile", ylabel="Return", xlabel="Quantile")
    axis.grid(axis="y", alpha=0.25)
    return axis


def plot_ic(ic: pd.Series, *, ax: Axes | None = None) -> Axes:
    """Plot cross-sectional information coefficient through time."""

    axis = ax or plt.subplots(figsize=(9, 4))[1]
    axis.plot(ic.index, ic, linewidth=1)
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set(title="Information Coefficient", ylabel="IC", xlabel="Date")
    axis.grid(alpha=0.25)
    return axis
