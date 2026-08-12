"""A tiny PyTorch multilayer perceptron with an intentionally short training loop."""

from __future__ import annotations

from typing import Any, Self

import numpy as np
import pandas as pd


class MLPRegressor:
    """Scikit-like CPU regressor backed by a two-hidden-layer PyTorch network."""

    def __init__(
        self,
        hidden_size: int = 16,
        epochs: int = 40,
        learning_rate: float = 0.01,
        seed: int = 42,
    ) -> None:
        if hidden_size < 1 or epochs < 1 or learning_rate <= 0:
            raise ValueError("hidden_size, epochs, and learning_rate must be positive")
        self.hidden_size = hidden_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.seed = seed
        self.model: Any = None
        self.feature_mean: np.ndarray | None = None
        self.feature_scale: np.ndarray | None = None

    @staticmethod
    def _torch() -> Any:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("PyTorch is optional; install mini-quantflow[mlp]") from exc
        return torch

    def fit(self, features: pd.DataFrame, target: pd.Series) -> Self:
        """Standardize on training data and fit using full-batch Adam on CPU."""

        if features.empty or len(features) != len(target):
            raise ValueError("features and target must have matching non-empty rows")
        torch = self._torch()
        torch.manual_seed(self.seed)
        values = features.to_numpy(dtype=np.float32)
        self.feature_mean = values.mean(axis=0)
        self.feature_scale = values.std(axis=0)
        self.feature_scale[self.feature_scale == 0] = 1.0
        normalized = (values - self.feature_mean) / self.feature_scale
        inputs = torch.tensor(normalized, dtype=torch.float32)
        labels = torch.tensor(target.to_numpy(dtype=np.float32).reshape(-1, 1))
        self.model = torch.nn.Sequential(
            torch.nn.Linear(features.shape[1], self.hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_size, self.hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_size, 1),
        )
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        loss_function = torch.nn.MSELoss()
        self.model.train()
        for _ in range(self.epochs):
            optimizer.zero_grad()
            loss = loss_function(self.model(inputs), labels)
            loss.backward()
            optimizer.step()
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Predict using training-set normalization and no gradient tracking."""

        if self.model is None or self.feature_mean is None or self.feature_scale is None:
            raise RuntimeError("fit must be called before predict")
        torch = self._torch()
        normalized = (features.to_numpy(dtype=np.float32) - self.feature_mean) / self.feature_scale
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(torch.tensor(normalized, dtype=torch.float32)).numpy().ravel()
        return np.asarray(predictions, dtype=float)
