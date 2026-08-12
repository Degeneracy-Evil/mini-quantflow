"""Experiment D: MLP and linear regression under walk-forward training."""

from __future__ import annotations

from common import backtest_scores, feature_panels, print_metrics
from sklearn.linear_model import LinearRegression

from quantlab.data import SyntheticDataSource, SyntheticMarketConfig, close_matrix
from quantlab.metrics.prediction import regression_metrics
from quantlab.models.dataset import build_supervised_dataset, prediction_panel
from quantlab.models.mlp import MLPRegressor
from quantlab.models.walk_forward import WalkForwardConfig, walk_forward_predict


def main() -> None:
    """Compare two model classes using identical expanding out-of-sample windows."""

    prices = close_matrix(SyntheticDataSource("regime", SyntheticMarketConfig(periods=500, assets=30, seed=37)).load())
    dataset = build_supervised_dataset(prices, feature_panels(prices))
    config = WalkForwardConfig(minimum_train_periods=200, test_periods=60)
    factories = {
        "linear": LinearRegression,
        "mlp": lambda: MLPRegressor(hidden_size=12, epochs=20, seed=37),
    }
    for name, factory in factories.items():
        prediction = walk_forward_predict(dataset, factory, config)
        rows = dataset.loc[prediction.index]
        print(name, "prediction:", regression_metrics(rows["target"], prediction))
        print_metrics(name, backtest_scores(prices, prediction_panel(rows, prediction)))


if __name__ == "__main__":
    main()
