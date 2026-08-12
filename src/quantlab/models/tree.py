"""One intentionally simple nonlinear prediction model."""

from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from quantlab.models.dataset import feature_columns


def fit_random_forest(
    dataset: pd.DataFrame, *, trees: int = 100, max_depth: int = 5, seed: int = 42
) -> RandomForestRegressor:
    """Fit a small deterministic random forest suitable for CPU experiments."""

    if trees < 1 or max_depth < 1:
        raise ValueError("trees and max_depth must be positive")
    columns = feature_columns(dataset)
    if not columns or dataset.empty:
        raise ValueError("training dataset must contain rows and features")
    model = RandomForestRegressor(
        n_estimators=trees,
        max_depth=max_depth,
        min_samples_leaf=5,
        random_state=seed,
        n_jobs=1,
    )
    model.fit(dataset[columns], dataset["target"])
    return model


def predict_forest(model: RandomForestRegressor, dataset: pd.DataFrame) -> pd.Series:
    """Predict future returns while retaining the input row index."""

    return pd.Series(model.predict(dataset[feature_columns(dataset)]), index=dataset.index, name="prediction")
