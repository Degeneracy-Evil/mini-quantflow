"""Logistic next-period direction prediction."""

from __future__ import annotations

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from quantlab.models.dataset import feature_columns


def fit_logistic_regression(dataset: pd.DataFrame, seed: int = 42) -> Pipeline:
    """Fit a standardized binary classifier in chronological row order."""

    columns = feature_columns(dataset)
    if not columns or dataset.empty:
        raise ValueError("training dataset must contain rows and features")
    if dataset["target"].nunique() < 2:
        raise ValueError("classification target must contain both classes")
    model = Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(random_state=seed, max_iter=1_000))])
    model.fit(dataset[columns], dataset["target"].astype(int))
    return model


def predict_probability(model: Pipeline, dataset: pd.DataFrame) -> pd.Series:
    """Return probability of a positive next-period return."""

    values = model.predict_proba(dataset[feature_columns(dataset)])[:, 1]
    return pd.Series(values, index=dataset.index, name="positive_probability", dtype=float)
