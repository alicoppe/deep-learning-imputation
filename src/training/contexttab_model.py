"""ConTextTab in-context learner wrapper.

ConTextTab (Spinaci et al. 2025, NeurIPS) is a semantics-aware tabular
in-context learner. Unlike MLP/XGBoost which fit parameters on X_train,
ConTextTab stores (X_train, y_train) as context and runs a forward pass
with the test batch to produce predictions — no gradient step.

The reference implementation was open-sourced by SAP as ``sap-rpt-oss``
(https://github.com/SAP-samples/contexttab); exported classes are
``SAP_RPT_OSS_Classifier`` and ``SAP_RPT_OSS_Regressor``.

For MIMIC-IV-ED (~300k rows) the context set must be subsampled to a
library-friendly size; we use ``max_context_rows`` to control this in
the wrapper (seeded) and also pass it through as the library's own
``max_context_size`` safeguard.
"""

from __future__ import annotations

import numpy as np
import torch

from src.training.base_model import BaseModel, TrainResult


class ContextTabModel(BaseModel):
    # ConTextTab can handle raw NaN natively, but to keep comparisons fair
    # against MLP/XGBoost we consume the same imputed+scaled pipeline output.
    # Flip to False for an ablation that tests ConTextTab's native NaN path.
    needs_imputed = True

    def __init__(
        self,
        bagging: int | str = 8,
        device: str | None = None,
        task_type: str = "regression",
        n_classes: int = 1,
        max_context_rows: int = 8192,
        random_state: int = 42,
        **kwargs,  # absorb unrelated sweep-config params (e.g. MLP hidden_dims)
    ):
        self.bagging = bagging
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
        # Lazy import so the codebase still loads if sap-rpt-oss is absent
        # (e.g. on Intel-Mac where torch 2.7+ wheels aren't available).
        from sap_rpt_oss import SAP_RPT_OSS_Classifier, SAP_RPT_OSS_Regressor

        n_train = len(X_train)
        if n_train > self.max_context_rows:
            rng = np.random.default_rng(self.random_state)
            idx = rng.choice(n_train, size=self.max_context_rows, replace=False)
            X_ctx, y_ctx = X_train[idx], y_train[idx]
            print(f"  Subsampling context: {n_train:,} → {self.max_context_rows:,}")
        else:
            X_ctx, y_ctx = X_train, y_train

        ModelCls = (
            SAP_RPT_OSS_Classifier if self.task_type == "classification" else SAP_RPT_OSS_Regressor
        )
        self._model = ModelCls(
            bagging=self.bagging,
            max_context_size=self.max_context_rows,
        )

        # "Fit" stores context; no gradient step.
        self._model.fit(X_ctx, y_ctx)

        # One val eval for logging consistency with MLP/XGBoost.
        val_pred = self._predict_internal(X_val)
        if self.task_type == "classification":
            from sklearn.metrics import log_loss
            val_loss = float(log_loss(y_val, np.clip(val_pred, 1e-7, 1 - 1e-7)))
            loss_name = "logloss"
        else:
            val_loss = float(np.mean(np.abs(val_pred - y_val)))
            loss_name = "mae"

        print(f"  Context set: n={len(X_ctx):,}  val_{loss_name}={val_loss:.4f}\n")

        # One-point history so the runner's training-curve plotting code
        # doesn't choke on empty lists.
        return TrainResult(train_losses=[val_loss], val_losses=[val_loss])

    def _predict_internal(self, X: np.ndarray) -> np.ndarray:
        if self.task_type == "classification":
            proba = self._model.predict_proba(X)
            return proba[:, 1] if proba.ndim > 1 else proba
        return np.asarray(self._model.predict(X))

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model not fitted. Call train() first.")
        return self._predict_internal(X)
