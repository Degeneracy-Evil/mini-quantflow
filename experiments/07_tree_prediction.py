"""Nonlinear tree prediction on the same research pipeline."""

from __future__ import annotations

from common import backtest_scores, feature_panels, print_metrics

from quantlab.data import SyntheticDataSource, SyntheticMarketConfig, close_matrix
from quantlab.metrics.prediction import regression_metrics
from quantlab.models.dataset import build_supervised_dataset, chronological_split, prediction_panel
from quantlab.models.tree import fit_random_forest, predict_forest


def main() -> None:
    """Fit a small forest, then evaluate both statistical and trading outcomes."""

    prices = close_matrix(SyntheticDataSource("regime", SyntheticMarketConfig(periods=650, assets=40, seed=31)).load())
    dataset = build_supervised_dataset(prices, feature_panels(prices))
    split = chronological_split(dataset)
    prediction = predict_forest(fit_random_forest(split.train), split.test)
    print("prediction:", regression_metrics(split.test["target"], prediction))
    print_metrics("tree_trading", backtest_scores(prices, prediction_panel(split.test, prediction)))


if __name__ == "__main__":
    main()
