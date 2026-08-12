"""Rolling and expanding out-of-sample prediction."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Protocol, Self

import numpy as np
import pandas as pd

from quantlab.models.dataset import feature_columns


class Regressor(Protocol):
    """Minimal estimator interface needed by walk-forward training."""

    def fit(self, features: pd.DataFrame, target: pd.Series) -> Self:
        """Fit the estimator and return itself."""

        ...

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Predict one value per feature row."""

        ...


@dataclass(frozen=True)
class WalkForwardConfig:
    """Time-window sizes measured in unique timestamps, not individual rows."""

    minimum_train_periods: int
    test_periods: int = 20
    step_periods: int | None = None
    rolling_train_periods: int | None = None

    def __post_init__(self) -> None:
        if self.minimum_train_periods < 1 or self.test_periods < 1:
            raise ValueError("train and test periods must be positive")
        if self.step_periods is not None and self.step_periods < 1:
            raise ValueError("step_periods must be positive")
        if self.rolling_train_periods is not None and self.rolling_train_periods < self.minimum_train_periods:
            raise ValueError("rolling window cannot be shorter than minimum training history")


def walk_forward_splits(timestamps: pd.Series, config: WalkForwardConfig) -> Iterator[tuple[pd.Index, pd.Index]]:
    """Yield train/test timestamp indexes with every test date strictly after train."""

    unique = pd.Index(pd.to_datetime(timestamps).drop_duplicates().sort_values())
    step = config.step_periods or config.test_periods
    test_start = config.minimum_train_periods
    while test_start < len(unique):
        train_start = 0
        if config.rolling_train_periods is not None:
            train_start = max(0, test_start - config.rolling_train_periods)
        test_end = min(test_start + config.test_periods, len(unique))
        yield unique[train_start:test_start], unique[test_start:test_end]
        test_start += step


def walk_forward_predict(
    dataset: pd.DataFrame,
    estimator_factory: Callable[[], Regressor],
    config: WalkForwardConfig,
) -> pd.Series:
    """Fit only labels known by each test start and return out-of-sample predictions."""

    if dataset.empty:
        raise ValueError("dataset must not be empty")
    columns = feature_columns(dataset)
    predictions = pd.Series(np.nan, index=dataset.index, name="prediction", dtype=float)
    for train_dates, test_dates in walk_forward_splits(dataset["timestamp"], config):
        train_mask = dataset["timestamp"].isin(train_dates)
        if "target_timestamp" in dataset:
            train_mask &= pd.to_datetime(dataset["target_timestamp"]).le(test_dates[0])
        test_mask = dataset["timestamp"].isin(test_dates) & predictions.isna()
        if not test_mask.any():
            continue
        if not train_mask.any():
            raise ValueError("no training labels are observable by the start of the test window")
        model = estimator_factory()
        model.fit(dataset.loc[train_mask, columns], dataset.loc[train_mask, "target"])
        predictions.loc[test_mask] = model.predict(dataset.loc[test_mask, columns])
    return predictions.dropna()
