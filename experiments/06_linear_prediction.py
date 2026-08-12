"""Experiment C: linear future-return prediction and trading evaluation."""

from __future__ import annotations

import matplotlib.pyplot as plt
from common import artifact_path, backtest_scores, feature_panels, print_metrics

from quantlab.data import SyntheticDataSource, SyntheticMarketConfig, close_matrix
from quantlab.metrics.prediction import regression_metrics
from quantlab.models.dataset import build_supervised_dataset, chronological_split, prediction_panel
from quantlab.models.linear import fit_linear_regression, predict_linear
from quantlab.visualization.prediction import plot_prediction_scatter


def main() -> None:
    """Report prediction statistics alongside economically distinct trading results."""

    prices = close_matrix(SyntheticDataSource("regime", SyntheticMarketConfig(periods=650, assets=40, seed=29)).load())
    dataset = build_supervised_dataset(prices, feature_panels(prices))
    split = chronological_split(dataset)
    model = fit_linear_regression(split.train)
    prediction = predict_linear(model, split.test)
    metrics = regression_metrics(split.test["target"], prediction)
    result = backtest_scores(prices, prediction_panel(split.test, prediction))
    print("prediction:", "  ".join(f"{key}={value:.6f}" for key, value in metrics.items()))
    print_metrics("linear_trading", result)
    plot_prediction_scatter(split.test["target"], prediction)
    plt.tight_layout()
    plt.savefig(artifact_path("06_linear_prediction.png"))
    plt.close()


if __name__ == "__main__":
    main()
