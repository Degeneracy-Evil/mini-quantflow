"""Build supervised panels without disturbing chronological order."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import pandas as pd

from quantlab.factors.base import require_same_panel
from quantlab.features.returns import forward_return


@dataclass(frozen=True)
class ChronologicalSplit:
    """Time-ordered train, validation, and test row tables."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def _stack_panel(panel: pd.DataFrame) -> pd.Series:
    stacked = panel.stack()
    if isinstance(stacked, pd.DataFrame):  # pragma: no cover - single-level columns return Series
        raise TypeError("feature panels must have one column level")
    return stacked


def build_supervised_dataset(
    prices: pd.DataFrame,
    features: Mapping[str, pd.DataFrame],
    horizon: int = 1,
    *,
    classification: bool = False,
    drop_missing: bool = True,
) -> pd.DataFrame:
    """Combine past-looking feature panels with an explicitly future target.

    Rows stay ordered by timestamp then asset. The label at ``t`` is the return
    from price ``t`` to price ``t+horizon``; callers are responsible for passing
    features that use only information observable through ``t``.
    """

    if horizon < 1:
        raise ValueError("horizon must be positive")
    if not features:
        raise ValueError("at least one feature is required")
    if len(set(features)) != len(features):
        raise ValueError("feature names must be unique")

    columns: dict[str, pd.Series] = {}
    for name, panel in features.items():
        require_same_panel(prices, panel)
        columns[name] = _stack_panel(panel)
    target_panel = forward_return(prices, horizon)
    if not isinstance(target_panel, pd.DataFrame):  # pragma: no cover
        raise TypeError("expected DataFrame target")
    target = _stack_panel(target_panel)
    dataset = pd.DataFrame(columns)
    dataset.index.names = ["timestamp", "asset"]
    dataset["target"] = (target > 0).astype(float).where(target.notna()) if classification else target
    dataset = dataset.reset_index().sort_values(["timestamp", "asset"], kind="stable").reset_index(drop=True)
    feature_names = list(features)
    if drop_missing:
        dataset = dataset.dropna(subset=[*feature_names, "target"]).reset_index(drop=True)
    return dataset


def chronological_split(
    dataset: pd.DataFrame, train_fraction: float = 0.6, validation_fraction: float = 0.2
) -> ChronologicalSplit:
    """Split on unique timestamps so assets from one date never cross boundaries."""

    if "timestamp" not in dataset or "asset" not in dataset:
        raise ValueError("dataset must contain timestamp and asset columns")
    if train_fraction <= 0 or validation_fraction < 0 or train_fraction + validation_fraction >= 1:
        raise ValueError("fractions must leave non-empty train and test intervals")
    timestamps = pd.Index(pd.to_datetime(dataset["timestamp"]).drop_duplicates().sort_values())
    if len(timestamps) < 3:
        raise ValueError("at least three unique timestamps are required")
    train_end = max(1, int(len(timestamps) * train_fraction))
    validation_end = max(train_end, int(len(timestamps) * (train_fraction + validation_fraction)))
    validation_end = min(validation_end, len(timestamps) - 1)
    train_dates = timestamps[:train_end]
    validation_dates = timestamps[train_end:validation_end]
    test_dates = timestamps[validation_end:]
    return ChronologicalSplit(
        dataset[dataset["timestamp"].isin(train_dates)].copy(),
        dataset[dataset["timestamp"].isin(validation_dates)].copy(),
        dataset[dataset["timestamp"].isin(test_dates)].copy(),
    )


def feature_columns(dataset: pd.DataFrame) -> list[str]:
    """Return model input columns from a standard supervised dataset."""

    reserved = {"timestamp", "asset", "target"}
    return [str(column) for column in dataset.columns if column not in reserved]


def prediction_panel(dataset: pd.DataFrame, predictions: pd.Series) -> pd.DataFrame:
    """Pivot row predictions back to ``timestamp x asset`` form for portfolios."""

    if len(dataset) != len(predictions):
        raise ValueError("prediction count must match dataset rows")
    if not dataset.index.equals(predictions.index):
        raise ValueError("prediction index must exactly match dataset row index")
    frame = dataset[["timestamp", "asset"]].copy()
    frame["prediction"] = predictions.to_numpy()
    if frame.duplicated(["timestamp", "asset"]).any():
        raise ValueError("dataset contains duplicate timestamp/asset rows")
    return frame.pivot(index="timestamp", columns="asset", values="prediction").sort_index()
