"""Abstract base class for prediction tasks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd


class BaseTask(ABC):
    """Defines a prediction task: target, feature engineering, and evaluation metrics.

    Class attributes (override in subclass):
        task_type  -- "regression" or "classification" (default: "regression")
        n_classes  -- 1 for regression or binary classification (single logit);
                      >1 for multi-class (not yet used)
    """

    task_type: str = "regression"
    n_classes: int = 1

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

    def load_data(self) -> pd.DataFrame:
        """Load the raw DataFrame for this task.

        Default implementation loads the standard MIMIC-IV-ED dataset.
        Override in tasks that need additional joins (e.g. mortality).
        """
        from src.data.preprocessing import load_processed_ed_data
        return load_processed_ed_data()

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

    def plot_predictions(self, y_true: np.ndarray, y_pred: np.ndarray, out_path: Path) -> None:
        """Save a prediction quality plot to out_path.

        Default: regression scatter (actual vs predicted). Override for classification.
        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 6))
        n = min(3000, len(y_true))
        ax.scatter(y_true[:n], y_pred[:n], alpha=0.2, s=5)
        ax.plot([0, 40], [0, 40], "r--", label="Perfect prediction")
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title("Predicted vs Actual")
        ax.legend()
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
