"""Tests for supervised alignment, prediction, and walk-forward windows."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from quantlab.features.mean_reversion import distance_from_moving_average
from quantlab.features.momentum import momentum
from quantlab.metrics.prediction import classification_metrics, regression_metrics
from quantlab.models.dataset import build_supervised_dataset, chronological_split, prediction_panel
from quantlab.models.linear import fit_linear_regression, predict_linear
from quantlab.models.logistic import fit_logistic_regression, predict_probability
from quantlab.models.mlp import MLPRegressor
from quantlab.models.tree import fit_random_forest, predict_forest
from quantlab.models.walk_forward import WalkForwardConfig, walk_forward_predict, walk_forward_splits


def _dataset(close_prices: pd.DataFrame, *, classification: bool = False) -> pd.DataFrame:
    features = {"momentum_1": momentum(close_prices, 1), "ma_distance": distance_from_moving_average(close_prices, 2)}
    return build_supervised_dataset(close_prices, features, classification=classification)


def test_dataset_target_alignment(close_prices: pd.DataFrame) -> None:
    dataset = _dataset(close_prices)
    row = dataset[(dataset["timestamp"] == close_prices.index[1]) & (dataset["asset"] == "A")].iloc[0]
    assert row["momentum_1"] == pytest.approx(0.01)
    assert row["target"] == pytest.approx(103 / 101 - 1)
    assert dataset.equals(dataset.sort_values(["timestamp", "asset"]).reset_index(drop=True))


def test_dataset_and_split_validation(close_prices: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="at least one"):
        build_supervised_dataset(close_prices, {})
    with pytest.raises(ValueError, match="horizon"):
        build_supervised_dataset(close_prices, {"x": close_prices}, horizon=0)
    with pytest.raises(ValueError, match="identical"):
        build_supervised_dataset(close_prices, {"x": close_prices.rename(columns={"A": "Z"})})
    with pytest.raises(ValueError, match="timestamp"):
        chronological_split(pd.DataFrame({"x": [1]}))
    with pytest.raises(ValueError, match="fractions"):
        chronological_split(_dataset(close_prices), train_fraction=0.9, validation_fraction=0.2)
    with pytest.raises(ValueError, match="three"):
        chronological_split(_dataset(close_prices).iloc[:5])


def test_chronological_split_never_mixes_dates(close_prices: pd.DataFrame) -> None:
    split = chronological_split(_dataset(close_prices), 0.5, 0.25)
    assert split.train["timestamp"].max() < split.validation["timestamp"].min()
    assert split.validation["timestamp"].max() < split.test["timestamp"].min()


def test_classical_models(close_prices: pd.DataFrame) -> None:
    regression = _dataset(close_prices)
    linear = fit_linear_regression(regression)
    linear_prediction = predict_linear(linear, regression)
    forest = fit_random_forest(regression, trees=5, max_depth=2)
    forest_prediction = predict_forest(forest, regression)
    assert len(linear_prediction) == len(regression)
    assert len(forest_prediction) == len(regression)
    classification = _dataset(close_prices, classification=True)
    logistic = fit_logistic_regression(classification)
    probabilities = predict_probability(logistic, classification)
    assert probabilities.between(0, 1).all()


def test_model_validation() -> None:
    empty = pd.DataFrame(columns=["timestamp", "asset", "target"])
    with pytest.raises(ValueError, match="rows and features"):
        fit_linear_regression(empty)
    with pytest.raises(ValueError, match="rows and features"):
        fit_random_forest(empty)
    with pytest.raises(ValueError, match="trees"):
        fit_random_forest(pd.DataFrame(), trees=0)
    one_class = pd.DataFrame({"timestamp": [1], "asset": ["A"], "x": [1.0], "target": [1.0]})
    with pytest.raises(ValueError, match="both classes"):
        fit_logistic_regression(one_class)


def test_walk_forward_splits_and_predictions() -> None:
    timestamps = pd.Series(pd.date_range("2024-01-01", periods=8))
    config = WalkForwardConfig(minimum_train_periods=3, test_periods=2, rolling_train_periods=4)
    splits = list(walk_forward_splits(timestamps, config))
    assert len(splits) == 3
    assert all(train.max() < test.min() for train, test in splits)
    rows = pd.DataFrame(
        {
            "timestamp": timestamps,
            "asset": "A",
            "x": np.arange(8, dtype=float),
            "target": np.arange(8, dtype=float) * 2,
        }
    )
    predictions = walk_forward_predict(rows, LinearRegression, config)
    assert predictions.index.min() == 3
    assert predictions.iloc[-1] == pytest.approx(14.0)


def test_walk_forward_validation() -> None:
    with pytest.raises(ValueError, match="positive"):
        WalkForwardConfig(0)
    with pytest.raises(ValueError, match="step_periods"):
        WalkForwardConfig(1, step_periods=0)
    with pytest.raises(ValueError, match="rolling window"):
        WalkForwardConfig(3, rolling_train_periods=2)
    with pytest.raises(ValueError, match="must not be empty"):
        walk_forward_predict(pd.DataFrame(), LinearRegression, WalkForwardConfig(1))


def test_prediction_panels_and_metrics(close_prices: pd.DataFrame) -> None:
    dataset = _dataset(close_prices)
    prediction = pd.Series(dataset["target"].to_numpy() + 0.01, index=dataset.index)
    panel = prediction_panel(dataset, prediction)
    assert panel.shape[1] == close_prices.shape[1]
    metrics = regression_metrics(dataset["target"], prediction)
    assert metrics["mse"] == pytest.approx(0.0001)
    classes = classification_metrics(pd.Series([1, 1, 0]), pd.Series([1, 0, 0]))
    assert classes == {"accuracy": 2 / 3, "precision": 1.0, "recall": 0.5}
    with pytest.raises(ValueError, match="count"):
        prediction_panel(dataset, prediction.iloc[:-1])
    with pytest.raises(ValueError, match="indexes"):
        regression_metrics(dataset["target"], prediction.reset_index(drop=True).set_axis(range(1, len(prediction) + 1)))


def test_small_mlp_can_fit_and_predict() -> None:
    features = pd.DataFrame({"x": [-1.0, 0.0, 1.0, 2.0], "constant": 1.0})
    target = pd.Series([-2.0, 0.0, 2.0, 4.0])
    model = MLPRegressor(hidden_size=4, epochs=2, seed=1)
    result = model.fit(features, target).predict(features)
    assert result.shape == (4,)
    with pytest.raises(RuntimeError, match="fit"):
        MLPRegressor().predict(features)
    with pytest.raises(ValueError, match="positive"):
        MLPRegressor(hidden_size=0)
