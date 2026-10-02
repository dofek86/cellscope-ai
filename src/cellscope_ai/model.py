"""Small NumPy logistic model used as a dependency-light baseline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


def sigmoid(value: np.ndarray) -> np.ndarray:
    clipped = np.clip(value, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-clipped))


@dataclass
class LogisticBaseline:
    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray
    bias: float

    @classmethod
    def fit(
        cls,
        features: np.ndarray,
        labels: np.ndarray,
        *,
        steps: int = 2500,
        learning_rate: float = 0.05,
        l2: float = 0.01,
    ) -> "LogisticBaseline":
        mean = features.mean(axis=0)
        scale = features.std(axis=0)
        scale[scale < 1e-9] = 1.0
        x = (features - mean) / scale
        y = labels.astype(np.float64)
        weights = np.zeros(x.shape[1], dtype=np.float64)
        bias = 0.0

        count_zero = max(int(np.sum(y == 0)), 1)
        count_one = max(int(np.sum(y == 1)), 1)
        sample_weight = np.where(y == 1, len(y) / (2 * count_one), len(y) / (2 * count_zero))

        for step in range(steps):
            probability = sigmoid(x @ weights + bias)
            error = (probability - y) * sample_weight
            gradient_w = x.T @ error / len(y) + l2 * weights
            gradient_b = float(error.mean())
            rate = learning_rate / np.sqrt(1.0 + step / 250.0)
            weights -= rate * gradient_w
            bias -= rate * gradient_b
        return cls(mean=mean, scale=scale, weights=weights, bias=bias)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        normalized = (features - self.mean) / self.scale
        return sigmoid(normalized @ self.weights + self.bias)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, mean=self.mean, scale=self.scale, weights=self.weights, bias=self.bias)

    @classmethod
    def load(cls, path: Path) -> "LogisticBaseline":
        with np.load(path) as values:
            return cls(
                mean=values["mean"],
                scale=values["scale"],
                weights=values["weights"],
                bias=float(values["bias"]),
            )

    def contributions(self, features: np.ndarray) -> np.ndarray:
        return ((features - self.mean) / self.scale) * self.weights
