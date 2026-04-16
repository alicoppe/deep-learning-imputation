"""XGBoost model wrapper (regression and binary classification)."""

from __future__ import annotations

import numpy as np
import torch

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
        task_type: str = "regression",
        n_classes: int = 1,
        **kwargs,  # absorb unrelated params from shared sweep config
    ):
        self._xgb_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._kwargs = dict(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            min_child_weight=min_child_weight,
            tree_method="hist",
            device=self._xgb_device,
            early_stopping_rounds=early_stopping_rounds,
            n_jobs=n_jobs,
            random_state=random_state,
            verbosity=0,
        )
        self.task_type = task_type
        self._model = None

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> TrainResult:
        if self.task_type == "classification":
            from xgboost import XGBClassifier
            n_pos = float(y_train.sum())
            n_neg = float(len(y_train) - n_pos)
            spw = n_neg / max(n_pos, 1)
            print(f"  Class balance: {n_pos:.0f} pos / {n_neg:.0f} neg  "
                  f"(scale_pos_weight={spw:.2f})")
            self._model = XGBClassifier(
                **self._kwargs,
                objective="binary:logistic",
                eval_metric="logloss",
                scale_pos_weight=spw,
            )
            loss_key = "logloss"
        else:
            from xgboost import XGBRegressor
            self._model = XGBRegressor(
                **self._kwargs,
                objective="reg:absoluteerror",
            )
            loss_key = "mae"

        self._model.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=100,
        )
        print()
        evals = self._model.evals_result()
        return TrainResult(
            train_losses=evals["validation_0"][loss_key],
            val_losses=evals["validation_1"][loss_key],
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.task_type == "classification":
            return self._model.predict_proba(X)[:, 1]
        return self._model.predict(X)

    @property
    def feature_importances(self) -> np.ndarray:
        return self._model.feature_importances_
