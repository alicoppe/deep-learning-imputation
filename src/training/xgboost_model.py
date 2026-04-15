"""XGBoost model wrapper."""

from __future__ import annotations

import numpy as np
import torch
from xgboost import XGBRegressor

from src.training.base_model import BaseModel, TrainResult


class XGBoostModel(BaseModel):
    needs_imputed = False

    def __init__(
        self,
        n_estimators: int = 2000,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        min_child_weight: int = 1,
        early_stopping_rounds: int = 50,
        n_jobs: int = 4,
        device: str | None = None,
        random_state: int = 42,
        **kwargs,  # absorb unrelated params from shared sweep config
    ):
        xgb_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            min_child_weight=min_child_weight,
            objective="reg:absoluteerror",
            tree_method="hist",
            device=xgb_device,
            early_stopping_rounds=early_stopping_rounds,
            n_jobs=n_jobs,
            random_state=random_state,
            verbosity=0,
        )

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> TrainResult:
        self._model.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=100,
        )
        print()
        evals = self._model.evals_result()
        return TrainResult(
            train_losses=evals["validation_0"]["mae"],
            val_losses=evals["validation_1"]["mae"],
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    @property
    def feature_importances(self) -> np.ndarray:
        return self._model.feature_importances_
