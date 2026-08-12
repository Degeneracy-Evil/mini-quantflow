"""Linear future-return prediction."""

from __future__ import annotations

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from quantlab.models.dataset import feature_columns


def fit_linear_regression(dataset: pd.DataFrame) -> Pipeline:
    """Fit standardized ordinary least squares without shuffling rows."""

    columns = feature_columns(dataset)
    if not columns or dataset.empty:
        raise ValueError("training dataset must contain rows and features")
    model = Pipeline([("scale", StandardScaler()), ("model", LinearRegression())])
    model.fit(dataset[columns], dataset["target"])
    return model


def predict_linear(model: Pipeline, dataset: pd.DataFrame) -> pd.Series:
    """Predict future returns and preserve the input row index."""

    values = model.predict(dataset[feature_columns(dataset)])
    return pd.Series(values, index=dataset.index, name="prediction", dtype=float)
