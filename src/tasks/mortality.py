"""In-hospital mortality prediction task (binary classification)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    log_loss,
    roc_auc_score,
)

from src.data.embeddings import load_chiefcomplaint_embeddings
from src.data.preprocessing import CAT_COLS, NUMERIC_COLS, load_admissions, load_processed_ed_data, make_time_features
from src.tasks.base_task import BaseTask

TARGET_COL = "hospital_expire_flag"

# Triage vitals — same set as LOS, excluding stay_len (which is a future-leaking LOS target).
NUMERIC_FEAT_COLS = [c for c in NUMERIC_COLS if c != "stay_len"]

# Categorical features excluding disposition — it encodes outcome and would leak for mortality
# (many expired patients have disposition == "EXPIRED" or similar).
CAT_FEAT_COLS = [c for c in CAT_COLS if c != "disposition"]


class InHospitalMortalityTask(BaseTask):
    task_type = "classification"
    n_classes = 1  # binary with single-logit BCE

    @property
    def name(self) -> str:
        return "in_hospital_mortality"

    @property
    def target_col(self) -> str:
        return TARGET_COL

    def numeric_feature_cols(self) -> list[str]:
        return NUMERIC_FEAT_COLS

    def load_data(self) -> pd.DataFrame:
        df = load_processed_ed_data()
        n0 = len(df)
        df = df[df["hadm_id"].notna()].copy()
        print(f"Dropped {n0 - len(df):,} rows: no hadm_id (never admitted)")

        adm = load_admissions()
        df = df.merge(adm, on="hadm_id", how="inner")
        pos = int(df[TARGET_COL].sum())
        print(
            f"Cohort: {len(df):,} ED stays  |  positives: {pos:,} "
            f"({pos / len(df) * 100:.2f}%)"
        )
        return df

    def build_features(self, df: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, np.ndarray]:
        use_embeddings = config.get("data", {}).get("embeddings", False)

        time_feats = make_time_features(df)
        cat_feats = pd.get_dummies(df[CAT_FEAT_COLS], prefix=CAT_FEAT_COLS, drop_first=False, dtype=float)
        num_feats = df[NUMERIC_FEAT_COLS].copy()
        miss_feats = pd.DataFrame(
            {f"{c}_missing": df[c].isna().astype(float) for c in NUMERIC_FEAT_COLS if df[c].isna().any()},
            index=df.index,
        )
        emb_feats = load_chiefcomplaint_embeddings(
            df, cache_name="chiefcomplaint_embeddings_mortality.npy"
        ) if use_embeddings else None

        parts = [time_feats, cat_feats, num_feats, miss_feats]
        if emb_feats is not None:
            parts.append(emb_feats)
        X = pd.concat(parts, axis=1)
        y = df[TARGET_COL].astype(np.float32).values

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
            "auroc":    lambda y, p: float(roc_auc_score(y, p)),
            "auprc":    lambda y, p: float(average_precision_score(y, p)),
            "logloss":  lambda y, p: float(log_loss(y, np.clip(p, 1e-7, 1 - 1e-7))),
            "accuracy": lambda y, p: float(((p >= 0.5) == y).mean()),
            "f1":       lambda y, p: float(f1_score(y, (p >= 0.5).astype(int), zero_division=0)),
        }

    def plot_predictions(self, y_true: np.ndarray, y_pred: np.ndarray, out_path: Path) -> None:
        import matplotlib.pyplot as plt
        from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, roc_curve, precision_recall_curve

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        auroc = roc_auc_score(y_true, y_pred)
        axes[0].plot(fpr, tpr, linewidth=2, label=f"AUROC = {auroc:.3f}")
        axes[0].plot([0, 1], [0, 1], "k--", alpha=0.4)
        axes[0].set_xlabel("False Positive Rate")
        axes[0].set_ylabel("True Positive Rate")
        axes[0].set_title("ROC Curve")
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        # PR curve
        prec, rec, _ = precision_recall_curve(y_true, y_pred)
        auprc = average_precision_score(y_true, y_pred)
        baseline = y_true.mean()
        axes[1].plot(rec, prec, linewidth=2, label=f"AUPRC = {auprc:.3f}")
        axes[1].axhline(baseline, color="k", linestyle="--", alpha=0.4, label=f"Baseline = {baseline:.3f}")
        axes[1].set_xlabel("Recall")
        axes[1].set_ylabel("Precision")
        axes[1].set_title("Precision–Recall Curve")
        axes[1].legend()
        axes[1].grid(alpha=0.3)

        # Confusion matrix at threshold 0.5
        y_bin = (y_pred >= 0.5).astype(int)
        cm = confusion_matrix(y_true.astype(int), y_bin)
        ConfusionMatrixDisplay(cm, display_labels=["Survived", "Died"]).plot(ax=axes[2], colorbar=False)
        axes[2].set_title("Confusion Matrix (threshold = 0.5)")

        fig.suptitle("In-Hospital Mortality — Model Evaluation", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
