"""Abstract base class for prediction tasks."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class BaseTask(ABC):
    """Defines a prediction task: target, feature engineering, and evaluation metrics."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def target_col(self) -> str: ...

    @abstractmethod
    def numeric_feature_cols(self) -> list[str]:
        """Return the names of numeric (vital-like) feature columns.

        Used by the pipeline to restrict simulated missingness to these columns.
        """
        ...

    @abstractmethod
    def build_features(self, df: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, np.ndarray]:
        """Task-specific feature engineering.

        Returns (X, y) where X is a DataFrame and y is a 1-D numpy array.
        NaNs in X are preserved; imputation happens in the pipeline after splitting.
        """
        ...

    @abstractmethod
    def metrics(self) -> dict[str, callable]:
        """Return a dict of metric_name -> fn(y_true, y_pred) -> float."""
        ...

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
        return {name: float(fn(y_true, y_pred)) for name, fn in self.metrics().items()}
