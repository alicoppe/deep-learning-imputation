"""
Length-of-stay prediction baseline on MIMIC-IV-ED  (target: hours from triage to discharge).

Models: mean predictor (trivial), MLP, XGBoost.
Features: time (cyclical), categoricals (OHE), triage vitals + missingness indicators,
          optionally Bio_ClinicalBERT embeddings of chief complaint (--embeddings).

Two design decisions worth noting:
  - disposition is excluded — recorded at discharge, not available at triage.
  - XGBoost receives raw (unimputed, unscaled) features; it handles NaN natively.

Usage:
  python src/training/length_of_stay_baseline.py
  python src/training/length_of_stay_baseline.py --embeddings
  python src/training/length_of_stay_baseline.py --run-name exp1
  python src/training/length_of_stay_baseline.py --quick
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
CAT_FEAT_COLS = [c for c in CAT_COLS if c != "disposition"]
NUMERIC_FEAT_COLS = [c for c in NUMERIC_COLS if c != TARGET_COL]


def print_data_summary(df: pd.DataFrame) -> None:
    tgt = df[TARGET_COL]
    print(f"  df length  n={len(df):,}")
    print(f"  Stay length (h)  mean={tgt.mean():.2f}  sd={tgt.std():.2f}  "
          f"median={tgt.median():.2f}  "
          f"range=[{tgt.min():.2f}, {tgt.max():.2f}]")
    print(f"\n  Triage vital missingness:")
    for col in NUMERIC_FEAT_COLS:
        pct = df[col].isna().mean() * 100
        print(f"    {col:<20}  {pct:5.1f}%")
    print()


def build_features(df: pd.DataFrame, use_embeddings: bool) -> tuple[pd.DataFrame, np.ndarray]:
    time_feats = make_time_features(df)
    cat_feats = pd.get_dummies(df[CAT_FEAT_COLS], prefix=CAT_FEAT_COLS, drop_first=False, dtype=float)
    num_feats = df[NUMERIC_FEAT_COLS].copy()  # NaNs preserved; imputed after split
    miss_feats = pd.DataFrame(
        {f"{c}_missing": df[c].isna().astype(float) for c in NUMERIC_FEAT_COLS if df[c].isna().any()},
        index=df.index,
    )
    emb_feats = _load_embeddings(df) if use_embeddings else None

    X = pd.concat([time_feats, cat_feats, num_feats, miss_feats] + ([emb_feats] if emb_feats is not None else []), axis=1)
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
    70 / 15 / 15 train / val / test split.
    Returns two variants of each split:
      - Imputed (median, train only) + scaled (StandardScaler, train only) → X_train / X_val / X_test
      - Raw (unimputed, unscaled)                                          → X_train_raw / X_val_raw / X_test_raw
    """
    feature_names = list(X.columns)
    numeric_feat_indices = [feature_names.index(c) for c in NUMERIC_FEAT_COLS]

    X_arr = X.values.astype(np.float32)
    y_arr = y.astype(np.float32)

    X_tv, X_test_raw, y_tv, y_test = train_test_split(X_arr, y_arr, test_size=0.15, random_state=seed)
    X_tr_raw, X_val_raw, y_train, y_val = train_test_split(X_tv, y_tv, test_size=0.176, random_state=seed)

    imputer = SimpleImputer(strategy="median")
    X_train, X_val, X_test = X_tr_raw.copy(), X_val_raw.copy(), X_test_raw.copy()
    X_train[:, numeric_feat_indices] = imputer.fit_transform(X_tr_raw[:, numeric_feat_indices])
    X_val[:, numeric_feat_indices] = imputer.transform(X_val_raw[:, numeric_feat_indices])
    X_test[:, numeric_feat_indices] = imputer.transform(X_test_raw[:, numeric_feat_indices])

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
    X_train, y_train = splits["X_train"], splits["y_train"]
    X_val, y_val = splits["X_val"], splits["y_val"]

    loader = DataLoader(_EDDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).to(device)

    model = MLP(X_train.shape[1], list(hidden_dims), dropout).to(device)
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5, min_lr=1e-5)

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

        print(f"  Epoch {epoch+1:3d}/{epochs}  train={epoch_train:.3f}  val={val_loss:.3f}"
              f"  lr={optimizer.param_groups[0]['lr']:.1e}" + (" *" if improved else ""))

        if no_improve >= patience:
            print(f"  Early stopping — no val improvement for {patience} consecutive epochs.")
            break

    model.load_state_dict(best_state)
    print(f"  Best val MAE: {best_val:.3f} h\n")
    return model, train_losses, val_losses


def train_xgboost(splits: dict, n_estimators: int = 2000, xgb_device: str = "cuda") -> XGBRegressor:
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
        eval_set=[(splits["X_train_raw"], splits["y_train"]), (splits["X_val_raw"], splits["y_val"])],
        verbose=100,
    )
    print()
    evals = model.evals_result()
    return model, evals["validation_0"]["mae"], evals["validation_1"]["mae"]


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(np.mean(np.abs(y_pred - y_true))),
        "rmse": float(np.sqrt(np.mean((y_pred - y_true) ** 2))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def print_results(rows: list[dict]) -> None:
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
    mlp_train_losses: list[float],
    mlp_val_losses: list[float],
    xgb_train_losses: list[float],
    xgb_val_losses: list[float],
    y_test: np.ndarray,
    mlp_pred: np.ndarray,
    xgb_pred: np.ndarray,
    feat_imp: pd.Series,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(mlp_train_losses, label="Train")
    axes[0].plot(mlp_val_losses, label="Validation")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("MAE (hours)")
    axes[0].set_title("MLP Training Curve"); axes[0].legend()
    axes[1].plot(xgb_train_losses, label="Train")
    axes[1].plot(xgb_val_losses, label="Validation")
    axes[1].set_xlabel("Round"); axes[1].set_ylabel("MAE (hours)")
    axes[1].set_title("XGBoost Training Curve"); axes[1].legend()
    plt.tight_layout()
    plt.savefig(out_dir / "training_curves.png", dpi=150)
    plt.close()

    # Predicted vs actual — MLP and XGBoost side by side
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for ax, pred, title in zip(axes, [mlp_pred, xgb_pred], ["MLP", "XGBoost"]):
        ax.scatter(y_test[:3000], pred[:3000], alpha=0.2, s=5)
        ax.plot([0, 40], [0, 40], "r--", label="Perfect prediction")
        ax.set_xlabel("Actual stay length (h)"); ax.set_ylabel("Predicted stay length (h)")
        ax.set_title(f"{title} — Predicted vs Actual (Test Set)")
        ax.set_xlim(0, 40); ax.set_ylim(0, 40); ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "predicted_vs_actual.png", dpi=150)
    plt.close()

    # XGBoost feature importance
    fig, ax = plt.subplots(figsize=(8, 7))
    feat_imp.head(30).plot(kind="barh", ax=ax)
    ax.invert_yaxis(); ax.set_xlabel("Importance (gain)"); ax.set_title("Top 30 XGBoost Feature Importances")
    plt.tight_layout()
    plt.savefig(out_dir / "xgb_feature_importance.png", dpi=150)
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

    print("Loading data ...")
    df = load_processed_ed_data()
    print_data_summary(df)

    print("Building features ...")
    X, y = build_features(df, args.embeddings)
    del df

    print("Preparing splits ...")
    splits = prepare_splits(X, y, seed=args.seed)
    del X, y

    rows: list[dict] = []

    mean_pred = np.full_like(splits["y_test"], splits["y_train"].mean())
    rows.append({"model": "Baseline (mean)", "test": evaluate(splits["y_test"], mean_pred)})

    mlp_epochs = 10 if args.quick else 60
    print(f"Training MLP  (max {mlp_epochs} epochs, patience=10) ...")
    mlp, train_losses, val_losses = train_mlp(splits, device, epochs=mlp_epochs)
    mlp.eval()
    with torch.no_grad():
        mlp_pred = mlp(torch.tensor(splits["X_test"], dtype=torch.float32).to(device)).squeeze().cpu().numpy()
    rows.append({"model": "MLP", "test": evaluate(splits["y_test"], mlp_pred)})

    xgb_rounds = 200 if args.quick else 2000
    print(f"Training XGBoost  (max {xgb_rounds} rounds, early stopping=50) ...")
    xgb, xgb_train_losses, xgb_val_losses = train_xgboost(splits, n_estimators=xgb_rounds, xgb_device=xgb_device)
    xgb_pred = xgb.predict(splits["X_test_raw"])
    rows.append({"model": "XGBoost", "test": evaluate(splits["y_test"], xgb_pred)})

    print_results(rows)

    feat_imp = pd.Series(xgb.feature_importances_, index=splits["feature_names"]).sort_values(ascending=False)
    cc_total = feat_imp[feat_imp.index.str.startswith("cc_")].sum()
    tab_total = feat_imp[~feat_imp.index.str.startswith("cc_")].sum()
    imp_header = "  Top 20 XGBoost features (gain)"
    if args.embeddings:
        imp_header += f"  [CC={cc_total:.3f}  tabular={tab_total:.3f}]"
    print(imp_header)
    print(f"  {'Feature':<35}  {'Importance':>10}")
    print(f"  {'─' * 47}")
    for feat, imp in feat_imp.head(20).items():
        name = feat if len(feat) <= 35 else feat[:32] + "..."
        print(f"  {name:<35}  {imp:>10.4f}")
    print()

    run_name = args.run_name or ("embeddings" if args.embeddings else "tabular")
    out_dir = Path("results") / run_name
    save_plots(out_dir, train_losses, val_losses, xgb_train_losses, xgb_val_losses, splits["y_test"], mlp_pred, xgb_pred, feat_imp)

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


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings", action="store_true")
    ap.add_argument("--run-name", type=str, default=None, metavar="NAME")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=42, metavar="N")
    main(ap.parse_args())
