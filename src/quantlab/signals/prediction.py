"""Trading intentions derived from model predictions."""

from __future__ import annotations

import pandas as pd

from quantlab.signals.ranking import ranking_signal


def prediction_threshold_signal(predictions: pd.DataFrame, threshold: float = 0.0) -> pd.DataFrame:
    """Trade only predictions whose absolute magnitude exceeds a threshold."""

    if threshold < 0:
        raise ValueError("threshold must be non-negative")
    signal = pd.DataFrame(0.0, index=predictions.index, columns=predictions.columns)
    signal = signal.mask(predictions > threshold, 1.0)
    signal = signal.mask(predictions < -threshold, -1.0)
    return signal.where(predictions.notna())


def prediction_ranking_signal(predictions: pd.DataFrame, quantile: float = 0.2) -> pd.DataFrame:
    """Select cross-sectional prediction tails."""

    return ranking_signal(predictions, quantile)
