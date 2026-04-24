"""TabPFN v2 wrapper, parallel structure to ContextTabModel.

Uses TabPFN v2 weights explicitly (Hollmann et al. 2025, Nature) rather
than the package default (v2.5). This matches the version cited in the
MECH-PFN paper and uses the Prior Labs Apache-2.0-derived license.

To switch to v2.5, change ``ModelVersion.V2`` -> ``ModelVersion.V2_5`` and
note the license change (v2.5 weights are non-commercial and HF-gated).
"""

from __future__ import annotations

import numpy as np
import torch

from src.training.base_model import BaseModel, TrainResult


class TabPFNModel(BaseModel):
    needs_imputed = True

    def __init__(
        self,
        n_estimators: int = 4,  # Hollmann et al. 2025 Nature default; current pkg default is 8
        device: str | None = None,
        task_type: str = "regression",
        n_classes: int = 1,
        max_context_rows: int = 10000,
        random_state: int = 42,
        **kwargs,  # absorb unrelated sweep-config params
    ):
        self.n_estimators = n_estimators
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.task_type = task_type
        self.n_classes = n_classes
        self.max_context_rows = max_context_rows
        self.random_state = random_state
        self._model = None

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> TrainResult:
        from tabpfn import TabPFNClassifier, TabPFNRegressor
        from tabpfn.constants import ModelVersion

        n_train = len(X_train)
        if n_train > self.max_context_rows:
            rng = np.random.default_rng(self.random_state)
            idx = rng.choice(n_train, size=self.max_context_rows, replace=False)
            X_ctx, y_ctx = X_train[idx], y_train[idx]
            print(f"  Subsampling context: {n_train:,} -> {self.max_context_rows:,}")
        else:
            X_ctx, y_ctx = X_train, y_train

        ModelCls = (
            TabPFNClassifier if self.task_type == "classification" else TabPFNRegressor
        )

        # Pin TabPFN v2 weights explicitly — package default now loads v2.5.
        self._model = ModelCls.create_default_for_version(
            ModelVersion.V2,
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            device=self.device,
        )

        self._model.fit(X_ctx, y_ctx)

        val_pred = self._predict_internal(X_val)
        if self.task_type == "classification":
            from sklearn.metrics import log_loss
            val_loss = float(log_loss(y_val, np.clip(val_pred, 1e-7, 1 - 1e-7)))
            name = "logloss"
        else:
            val_loss = float(np.mean(np.abs(val_pred - y_val)))
            name = "mae"

        print(f"  Context set: n={len(X_ctx):,}  val_{name}={val_loss:.4f}\n")
        return TrainResult(train_losses=[val_loss], val_losses=[val_loss])

    def _predict_internal(self, X: np.ndarray) -> np.ndarray:
        if self.task_type == "classification":
            p = self._model.predict_proba(X)
            return p[:, 1] if p.ndim > 1 else p
        return np.asarray(self._model.predict(X))

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model not fitted. Call train() first.")
        return self._predict_internal(X)
