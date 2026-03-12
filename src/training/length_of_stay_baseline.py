"""
Baseline models for length-of-stay prediction on MIMIC-IV-ED.

Dataset
-------
MIMIC-IV-ED v2.2  (physionet.org/content/mimic-iv-ed/2.2/)
Target:  stay_len — hours elapsed from triage (intime) to ED discharge (outtime).
Cohort:  ~425 k visits after removing duplicate triage records and non-positive stay lengths.

Feature groups
--------------
Time          Cyclical sin/cos encoding of arrival hour, day-of-week, and month  (6 features)
Categorical   gender, race, arrival_transport — one-hot encoded
              disposition is excluded: it is recorded at discharge, not available at triage.
Numeric       8 triage vitals: temperature, heart rate, respiratory rate, O2 saturation,
              systolic/diastolic BP, pain score, acuity level.
              Out-of-range values are set to NaN by the preprocessing pipeline.
Missingness   Binary indicator for every numeric column that contains any NaN values.
Text (opt)    768-dim Bio_ClinicalBERT sentence embedding of chief complaint (--embeddings).
              Missing complaints are encoded as empty strings; a separate
              chiefcomplaint_missing indicator is appended.

Models
------
Baseline  Mean predictor trained on the training set — trivial lower-bound reference.
MLP       3-hidden-layer network (512 → 256 → 128 → 1), MAE loss, Adam optimiser,
          ReduceLROnPlateau scheduler, early stopping on validation MAE.
XGBoost   Gradient-boosted trees with MAE objective and early stopping on validation MAE.
          Receives raw (unscaled, unimputed) features — XGBoost handles NaN natively.

Split / preprocessing
---------------------
70 / 15 / 15 train / val / test (random_state=42 by default; override with --seed).
Numeric imputation: median, fit on train only; applied before scaling for the MLP.
Scaling:            StandardScaler, fit on train only; MLP input only.

Usage
-----
  python src/training/length_of_stay_baseline.py                               # tabular only
  python src/training/length_of_stay_baseline.py --embeddings                  # + Bio_ClinicalBERT chief complaint
  python src/training/length_of_stay_baseline.py --out results/                # saves to results/tabular/
  python src/training/length_of_stay_baseline.py --embeddings --out results/   # saves to results/embeddings/
  python src/training/length_of_stay_baseline.py --out results/ --run-name exp1
  python src/training/length_of_stay_baseline.py --quick                       # fast smoke-test (10 epochs / 200 rounds)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from xgboost import XGBRegressor

from src.data.preprocessing import (
    CAT_COLS,
    NUMERIC_COLS,
    load_processed_ed_data,
    make_time_features,
)

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
TARGET_COL = "stay_len"
CAT_FEAT_COLS = [c for c in CAT_COLS if c != "disposition"]  # disposition: end-of-visit label
NUMERIC_FEAT_COLS = [c for c in NUMERIC_COLS if c != TARGET_COL]  # exclude target from features


def print_data_summary(df: pd.DataFrame) -> None:
    """Print cohort statistics and per-column missingness rates for triage vitals."""
    tgt = df[TARGET_COL]
    print(f"  Cohort  n={len(df):,}")
    print(f"  Stay length (h)  mean={tgt.mean():.2f}  sd={tgt.std():.2f}  "
          f"median={tgt.median():.2f}  IQR=[{tgt.quantile(0.25):.2f}, {tgt.quantile(0.75):.2f}]  "
          f"range=[{tgt.min():.2f}, {tgt.max():.2f}]")
    print(f"\n  Triage vital missingness:  (each █ ≈ 0.5%)")
    for col in NUMERIC_FEAT_COLS:
        pct = df[col].isna().mean() * 100
        filled = min(20, int(pct * 2))  # 1 char per 0.5%, capped at 20 (= 10%)
        bar = "█" * filled + "░" * (20 - filled)
        print(f"    {col:<20}  {pct:5.1f}%  {bar}")
    print()


def build_features(df: pd.DataFrame, use_embeddings: bool) -> tuple[pd.DataFrame, np.ndarray]:
    """Assemble the feature matrix X and target vector y."""
    time_feats = make_time_features(df)
    cat_feats = pd.get_dummies(df[CAT_FEAT_COLS], prefix=CAT_FEAT_COLS, drop_first=False, dtype=float)
    num_feats = df[NUMERIC_FEAT_COLS].copy()  # NaNs preserved; imputed after split
    miss_feats = pd.DataFrame(
        {f"{c}_missing": df[c].isna().astype(float) for c in NUMERIC_FEAT_COLS if df[c].isna().any()},
        index=df.index,
    )

    emb_feats = _load_embeddings(df) if use_embeddings else None
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


def _load_embeddings(df: pd.DataFrame) -> pd.DataFrame:
    """Load (or compute and cache) Bio_ClinicalBERT sentence embeddings for chief complaint."""
    from sentence_transformers import SentenceTransformer

    cache = REPO_ROOT / "data" / "processed" / "chiefcomplaint_embeddings.npy"
    cache.parent.mkdir(parents=True, exist_ok=True)

    cc_missing = df["chiefcomplaint"].isna()
    print(f"  Chief complaint: {cc_missing.mean()*100:.1f}% missing ({cc_missing.sum():,} rows)")

    if cache.exists():
        print(f"  Loading cached embeddings from {cache.name} ...")
        arr = np.load(cache)
    else:
        print("  Computing Bio_ClinicalBERT embeddings (this may take several minutes) ...")
        arr = SentenceTransformer("emilyalsentzer/Bio_ClinicalBERT").encode(
            df["chiefcomplaint"].fillna("").tolist(),
            batch_size=256,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        np.save(cache, arr)
        print(f"  Saved embeddings → {cache}")

    print(f"  Embedding shape: {arr.shape}\n")
    out = pd.DataFrame(arr, index=df.index, columns=[f"cc_{i}" for i in range(arr.shape[1])])
    out["chiefcomplaint_missing"] = cc_missing.astype(float)
    return out


def prepare_splits(X: pd.DataFrame, y: np.ndarray, seed: int = 42) -> dict:
    """
    Return train/val/test splits as a dict with two variants:
      - Scaled + imputed arrays (X_train, X_val, X_test)  → used by MLP
      - Raw arrays              (X_train_raw, …)          → used by XGBoost (handles NaN natively)

    Imputation:  median, fit on train only.
    Scaling:     StandardScaler, fit on train only.
    Split ratio: 70 / 15 / 15  (test_size=0.15; val is 0.176 of trainval ≈ 15% of total).
    """
    feature_names = list(X.columns)
    numeric_feat_indices = [feature_names.index(c) for c in NUMERIC_FEAT_COLS]

    X_arr = X.values.astype(np.float32)
    y_arr = y.astype(np.float32)

    X_tv, X_test_raw, y_tv, y_test = train_test_split(X_arr, y_arr, test_size=0.15, random_state=seed)
    X_tr_raw, X_val_raw, y_train, y_val = train_test_split(X_tv, y_tv, test_size=0.176, random_state=seed)

    # Impute numeric cols — fit on train only
    imputer = SimpleImputer(strategy="median")
    X_train, X_val, X_test = X_tr_raw.copy(), X_val_raw.copy(), X_test_raw.copy()
    X_train[:, numeric_feat_indices] = imputer.fit_transform(X_tr_raw[:, numeric_feat_indices])
    X_val[:, numeric_feat_indices] = imputer.transform(X_val_raw[:, numeric_feat_indices])
    X_test[:, numeric_feat_indices] = imputer.transform(X_test_raw[:, numeric_feat_indices])

    # Scale — fit on train only
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    print(f"  Split  train={y_train.shape[0]:,}  val={y_val.shape[0]:,}  test={y_test.shape[0]:,}  "
          f"features={X_train.shape[1]}\n")

    return dict(
        X_train=X_train, X_val=X_val, X_test=X_test,
        X_train_raw=X_tr_raw, X_val_raw=X_val_raw, X_test_raw=X_test_raw,
        y_train=y_train, y_val=y_val, y_test=y_test,
        feature_names=feature_names,
    )


class _EDDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


class MLP(nn.Module):
    """Fully-connected network with ReLU activations and dropout."""

    def __init__(self, input_dim: int, hidden_dims: list[int], dropout: float = 0.2):
        super().__init__()
        layers, prev = [], input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_mlp(
    splits: dict,
    device: torch.device,
    hidden_dims: tuple[int, ...] = (512, 256, 128),
    dropout: float = 0.2,
    epochs: int = 60,
    patience: int = 10,
    batch_size: int = 512,
    lr: float = 1e-3,
) -> tuple[MLP, list[float], list[float]]:
    """Train MLP with early stopping; return the best model and loss history."""
    X_train, y_train = splits["X_train"], splits["y_train"]
    X_val, y_val = splits["X_val"], splits["y_val"]

    loader = DataLoader(_EDDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).to(device)

    model = MLP(X_train.shape[1], list(hidden_dims), dropout).to(device)
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=3, factor=0.5, min_lr=1e-5
    )

    best_val, best_state, no_improve = float("inf"), None, 0
    train_losses: list[float] = []
    val_losses: list[float] = []

    for epoch in range(epochs):
        model.train()
        running = 0.0
        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(Xb).squeeze(), yb)
            loss.backward()
            optimizer.step()
            running += loss.item() * len(Xb)

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val_t).squeeze(), y_val_t).item()

        epoch_train = running / len(loader.dataset)
        train_losses.append(epoch_train)
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        improved = val_loss < best_val
        if improved:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        lr_now = optimizer.param_groups[0]["lr"]
        marker = " *" if improved else ""
        print(f"  Epoch {epoch+1:3d}/{epochs}  train={epoch_train:.3f}  val={val_loss:.3f}"
              f"  lr={lr_now:.1e}{marker}")

        if no_improve >= patience:
            print(f"  Early stopping — no val improvement for {patience} consecutive epochs.")
            break

    model.load_state_dict(best_state)
    print(f"  Best val MAE: {best_val:.3f} h\n")
    return model, train_losses, val_losses


def train_xgboost(splits: dict, n_estimators: int = 2000, xgb_device: str = "cuda") -> XGBRegressor:
    """
    Train XGBoost on raw (unimputed, unscaled) features.
    XGBoost handles missing values natively during split-finding.
    """
    model = XGBRegressor(
        n_estimators=n_estimators,
        max_depth=6,
        learning_rate=0.05,
        subsample=1.0,
        colsample_bytree=1.0,
        min_child_weight=1,
        objective="reg:absoluteerror",
        tree_method="hist",
        device=xgb_device,
        early_stopping_rounds=50,
        n_jobs=4,
        random_state=42,
        verbosity=0,
    )
    model.fit(
        splits["X_train_raw"], splits["y_train"],
        eval_set=[(splits["X_val_raw"], splits["y_val"])],
        verbose=100,
    )
    print()
    return model


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Return MAE, RMSE, and R² for a set of predictions."""
    return {
        "mae": float(np.mean(np.abs(y_pred - y_true))),
        "rmse": float(np.sqrt(np.mean((y_pred - y_true) ** 2))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def print_results(rows: list[dict]) -> None:
    """Print a formatted results table with improvement vs. the mean baseline."""
    baseline_mae = next(r["test"]["mae"] for r in rows if r["model"] == "Baseline (mean)")
    header = f"  {'Model':<28}  {'MAE (h)':>8}  {'RMSE (h)':>9}  {'R²':>7}  {'vs baseline':>12}"
    sep = "  " + "─" * (len(header) - 2)
    print(sep)
    print(header)
    print(sep)
    for r in rows:
        t = r["test"]
        delta = (baseline_mae - t["mae"]) / baseline_mae * 100
        print(f"  {r['model']:<28}  {t['mae']:>8.3f}  {t['rmse']:>9.3f}  {t['r2']:>7.3f}  {delta:>+11.1f}%")
    print(sep)
    print()


def save_plots(
    out_dir: Path,
    train_losses: list[float],
    val_losses: list[float],
    y_test: np.ndarray,
    mlp_pred: np.ndarray,
    feat_imp: pd.Series,
    rows: list[dict],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # MLP training curve
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("MAE (hours)")
    plt.title("MLP Training Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "mlp_training_curve.png", dpi=150)
    plt.close()

    # MLP predicted vs actual scatter
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test[:3000], mlp_pred[:3000], alpha=0.2, s=5)
    plt.plot([0, 100], [0, 100], "r--", label="Perfect prediction")
    plt.xlabel("Actual stay length (h)")
    plt.ylabel("Predicted stay length (h)")
    plt.title("MLP — Predicted vs Actual (Test Set)")
    plt.xlim(0, 40)
    plt.ylim(0, 40)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "mlp_predicted_vs_actual.png", dpi=150)
    plt.close()

    # XGBoost feature importance (top 30)
    fig, ax = plt.subplots(figsize=(8, 7))
    feat_imp.head(30).plot(kind="barh", ax=ax)
    ax.invert_yaxis()
    ax.set_xlabel("Importance (gain)")
    ax.set_title("Top 30 XGBoost Feature Importances")
    plt.tight_layout()
    plt.savefig(out_dir / "xgb_feature_importance.png", dpi=150)
    plt.close()

    # Model comparison bar chart
    models = [r["model"] for r in rows]
    maes = [r["test"]["mae"] for r in rows]
    rmses = [r["test"]["rmse"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (ylabel, values) in zip(axes, [("Test MAE (h)", maes), ("Test RMSE (h)", rmses)]):
        bars = ax.bar(models, values, width=0.55)
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, max(values) * 1.25)
        for b, v in zip(bars, values):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.03, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=10)
    fig.suptitle("Model Comparison — Test Set", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_dir / "model_comparison.png", dpi=150)
    plt.close()

    print(f"  Plots saved → {out_dir}/\n")


def main(args: argparse.Namespace) -> None:
    t0 = time.time()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    xgb_device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n{'═' * 60}")
    print(f"  Length-of-Stay Baseline  |  MIMIC-IV-ED")
    print(f"  device={device}  seed={args.seed}  embeddings={args.embeddings}")
    print(f"{'═' * 60}\n")

    # Load data
    print("Loading data ...")
    df = load_processed_ed_data()
    print_data_summary(df)

    # Build features
    print("Building features ...")
    X, y = build_features(df, args.embeddings)
    del df

    # Split & preprocess
    print("Preparing splits ...")
    splits = prepare_splits(X, y, seed=args.seed)
    del X, y

    rows: list[dict] = []

    # Mean baseline
    mean_pred = np.full_like(splits["y_test"], splits["y_train"].mean())
    rows.append({"model": "Baseline (mean)", "test": evaluate(splits["y_test"], mean_pred)})

    # MLP
    mlp_epochs = 10 if args.quick else 60
    print(f"Training MLP  (max {mlp_epochs} epochs, patience=10) ...")
    mlp, train_losses, val_losses = train_mlp(splits, device, epochs=mlp_epochs)

    mlp.eval()
    with torch.no_grad():
        X_test_t = torch.tensor(splits["X_test"], dtype=torch.float32).to(device)
        mlp_pred = mlp(X_test_t).squeeze().cpu().numpy()
    rows.append({"model": "MLP", "test": evaluate(splits["y_test"], mlp_pred)})

    # XGBoost
    xgb_rounds = 200 if args.quick else 2000
    print(f"Training XGBoost  (max {xgb_rounds} rounds, early stopping=50) ...")
    xgb = train_xgboost(splits, n_estimators=xgb_rounds, xgb_device=xgb_device)
    xgb_pred = xgb.predict(splits["X_test_raw"])
    rows.append({"model": "XGBoost", "test": evaluate(splits["y_test"], xgb_pred)})

    # Results table
    print_results(rows)

    # Feature importance — top 20 text table
    feat_imp = pd.Series(
        xgb.feature_importances_, index=splits["feature_names"]
    ).sort_values(ascending=False)
    cc_total = feat_imp[feat_imp.index.str.startswith("cc_")].sum()
    tab_total = feat_imp[~feat_imp.index.str.startswith("cc_")].sum()
    header = "  Top 20 XGBoost features (gain)"
    if args.embeddings:
        header += f"  [CC embeddings total={cc_total:.3f}  tabular total={tab_total:.3f}]"
    print(header)
    print(f"  {'Feature':<35}  {'Importance':>10}")
    print(f"  {'─' * 47}")
    for feat, imp in feat_imp.head(20).items():
        name = feat if len(feat) <= 35 else feat[:32] + "..."
        print(f"  {name:<35}  {imp:>10.4f}")
    print()

    # Save outputs
    if args.out:
        run_name = args.run_name or ("embeddings" if args.embeddings else "tabular")
        out_dir = Path(args.out) / run_name
        save_plots(out_dir, train_losses, val_losses, splits["y_test"], mlp_pred, feat_imp, rows)

        results_json = {
            "seed": args.seed,
            "embeddings": args.embeddings,
            "n_features": splits["X_train"].shape[1],
            "n_train": int(splits["X_train"].shape[0]),
            "n_val": int(splits["X_val"].shape[0]),
            "n_test": int(splits["X_test"].shape[0]),
            "models": rows,
        }
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "results.json").write_text(json.dumps(results_json, indent=2))
        print(f"  Results JSON → {out_dir / 'results.json'}\n")

    elapsed = time.time() - t0
    print(f"  Total runtime: {elapsed / 60:.1f} min")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--embeddings", action="store_true",
        help="Include Bio_ClinicalBERT chief-complaint embeddings",
    )
    ap.add_argument(
        "--out", type=Path, default=None, metavar="DIR",
        help="Base directory for outputs; files go into DIR/tabular or DIR/embeddings (see --run-name)",
    )
    ap.add_argument(
        "--run-name", type=str, default=None, metavar="NAME",
        help="Output subdirectory name (default: 'tabular' or 'embeddings' based on flags)",
    )
    ap.add_argument(
        "--quick", action="store_true",
        help="Smoke-test mode: 10 MLP epochs, 200 XGBoost rounds",
    )
    ap.add_argument(
        "--seed", type=int, default=42, metavar="N",
        help="Random seed for reproducibility",
    )
    main(ap.parse_args())
