"""Abstract base class for all prediction models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass
class TrainResult:
    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)


class BaseModel(ABC):
    """Unified interface for all downstream models.

    Subclasses declare ``needs_imputed`` to tell the runner which data variant
    to supply: imputed+scaled (True) or raw/unimputed (False).
    """

    needs_imputed: bool = True

    @abstractmethod
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> TrainResult: ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray: ...
