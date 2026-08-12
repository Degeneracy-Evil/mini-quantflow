"""Prediction diagnostics."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes


def plot_prediction_scatter(target: pd.Series, prediction: pd.Series, *, ax: Axes | None = None) -> Axes:
    """Compare predicted and realized future returns on aligned rows."""

    if not target.index.equals(prediction.index):
        raise ValueError("target and prediction indexes must match")
    axis = ax or plt.subplots(figsize=(5, 5))[1]
    axis.scatter(prediction, target, alpha=0.25, s=10)
    axis.set(title="Predicted vs Realized Return", xlabel="Prediction", ylabel="Realized")
    axis.grid(alpha=0.25)
    return axis
