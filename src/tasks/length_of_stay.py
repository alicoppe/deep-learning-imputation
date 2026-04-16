"""Length-of-stay prediction task (triage → discharge, hours)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from src.data.embeddings import load_chiefcomplaint_embeddings
from src.data.preprocessing import CAT_COLS, NUMERIC_COLS, make_time_features
from src.tasks.base_task import BaseTask

TARGET_COL = "stay_len"
CAT_FEAT_COLS = [c for c in CAT_COLS if c != "disposition"]
NUMERIC_FEAT_COLS = [c for c in NUMERIC_COLS if c != TARGET_COL]


class LOSTask(BaseTask):
    task_type = "regression"
    n_classes = 1

    @property
    def name(self) -> str:
        return "length_of_stay"

    @property
    def target_col(self) -> str:
        return TARGET_COL

    def numeric_feature_cols(self) -> list[str]:
        return NUMERIC_FEAT_COLS

    def build_features(self, df: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, np.ndarray]:
        use_embeddings = config.get("data", {}).get("embeddings", False)

        time_feats = make_time_features(df)
        cat_feats = pd.get_dummies(df[CAT_FEAT_COLS], prefix=CAT_FEAT_COLS, drop_first=False, dtype=float)
        num_feats = df[NUMERIC_FEAT_COLS].copy()
        miss_feats = pd.DataFrame(
            {f"{c}_missing": df[c].isna().astype(float) for c in NUMERIC_FEAT_COLS if df[c].isna().any()},
            index=df.index,
        )
        emb_feats = load_chiefcomplaint_embeddings(df) if use_embeddings else None

        parts = [time_feats, cat_feats, num_feats, miss_feats]
        if emb_feats is not None:
            parts.append(emb_feats)
        X = pd.concat(parts, axis=1)
        y = df[TARGET_COL].values

        print(f"  Feature groups  (total = {X.shape[1]})")
        print(f"    Time features      {time_feats.shape[1]:>4}")
        print(f"    Categorical (OHE)  {cat_feats.shape[1]:>4}")
        print(f"    Triage numerics    {num_feats.shape[1]:>4}")
        print(f"    Missingness flags  {miss_feats.shape[1]:>4}")
        if emb_feats is not None:
            print(f"    CC embeddings      {emb_feats.shape[1]:>4}  (Bio_ClinicalBERT 768-dim + missing flag)")
        print()

        return X, y

    def metrics(self) -> dict[str, callable]:
        return {
            "mae": lambda y_true, y_pred: float(np.mean(np.abs(y_pred - y_true))),
            "rmse": lambda y_true, y_pred: float(np.sqrt(np.mean((y_pred - y_true) ** 2))),
            "r2": lambda y_true, y_pred: float(r2_score(y_true, y_pred)),
        }

    def plot_predictions(self, y_true, y_pred, out_path) -> None:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 6))
        n = min(3000, len(y_true))
        ax.scatter(y_true[:n], y_pred[:n], alpha=0.2, s=5)
        ax.plot([0, 40], [0, 40], "r--", label="Perfect prediction")
        ax.set_xlabel("Actual stay length (h)")
        ax.set_ylabel("Predicted stay length (h)")
        ax.set_title("Predicted vs Actual")
        ax.set_xlim(0, 40)
        ax.set_ylim(0, 40)
        ax.legend()
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()


