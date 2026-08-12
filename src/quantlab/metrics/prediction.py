"""Prediction metrics kept separate from trading performance."""

from __future__ import annotations

import numpy as np
import pandas as pd


def regression_metrics(target: pd.Series, prediction: pd.Series) -> dict[str, float]:
    """Calculate MSE, MAE, and linear correlation on aligned observations."""

    if not target.index.equals(prediction.index):
        raise ValueError("target and prediction indexes must match")
    pair = pd.concat([target.rename("target"), prediction.rename("prediction")], axis=1).dropna()
    if pair.empty:
        raise ValueError("no valid prediction pairs")
    errors = pair["prediction"] - pair["target"]
    return {
        "mse": float(np.mean(np.square(errors))),
        "mae": float(np.mean(np.abs(errors))),
        "correlation": float(pair["prediction"].corr(pair["target"])),
    }


def classification_metrics(target: pd.Series, prediction: pd.Series) -> dict[str, float]:
    """Calculate accuracy, positive-class precision, and recall."""

    if not target.index.equals(prediction.index):
        raise ValueError("target and prediction indexes must match")
    actual = target.astype(bool)
    predicted = prediction.astype(bool)
    true_positive = int((actual & predicted).sum())
    predicted_positive = int(predicted.sum())
    actual_positive = int(actual.sum())
    return {
        "accuracy": float((actual == predicted).mean()),
        "precision": true_positive / predicted_positive if predicted_positive else 0.0,
        "recall": true_positive / actual_positive if actual_positive else 0.0,
    }
