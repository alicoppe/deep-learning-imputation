"""Shared data pipeline: load → feature build → split → missingness → impute/scale."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.data.missingness_sim import MissingnessMechanism, simulate_missingness


def run_pipeline(task, config: dict) -> dict:
    """Run the full data pipeline for a given task and config.

    Returns a dict with keys:
      X_train, X_val, X_test          — imputed + scaled (for MLP-like models)
      X_train_raw, X_val_raw, X_test_raw — unimputed, unscaled (for XGBoost-like models)
      y_train, y_val, y_test
      feature_names
    """
    df = task.load_data()
    X, y = task.build_features(df, config)
    del df

    feature_names = list(X.columns)
    X_arr = X.values.astype(np.float32)
    y_arr = y.astype(np.float32)
    del X

    seed = config.get("seed", 42)
    X_tv, X_test_raw, y_tv, y_test = train_test_split(X_arr, y_arr, test_size=0.15, random_state=seed)
    X_tr_raw, X_val_raw, y_train, y_val = train_test_split(X_tv, y_tv, test_size=0.176, random_state=seed)

    miss_cfg = config.get("missingness")
    if miss_cfg and miss_cfg.get("rate", 0) > 0:
        X_tr_raw, X_val_raw, X_test_raw = _inject_missingness(
            X_tr_raw, X_val_raw, X_test_raw, feature_names, task, miss_cfg, seed
        )

    # Imputed + scaled variants (fit only on train)
    numeric_feat_indices = [feature_names.index(c) for c in task.numeric_feature_cols() if c in feature_names]

    imputer = SimpleImputer(strategy="median")
    X_train = X_tr_raw.copy()
    X_val = X_val_raw.copy()
    X_test = X_test_raw.copy()
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


def _inject_missingness(
    X_tr: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    feature_names: list[str],
    task,
    miss_cfg: dict,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply simulated missingness to each split independently."""
    mechanism = MissingnessMechanism(miss_cfg["mechanism"])
    rate = float(miss_cfg["rate"])
    strength = float(miss_cfg.get("strength", 1.0))
    drivers_map = miss_cfg.get("drivers_map")

    # Columns to target — default to task's numeric feature columns
    target_cols = miss_cfg.get("columns") or task.numeric_feature_cols()
    col_indices = [i for i, name in enumerate(feature_names) if name in target_cols]

    if not col_indices:
        return X_tr, X_val, X_test

    print(f"  Injecting {mechanism.value} missingness  rate={rate:.2f}  "
          f"strength={strength:.2f}  cols={len(col_indices)}")

    results = []
    for split_idx, (X_split, split_seed_offset) in enumerate(
        [(X_tr, 0), (X_val, 1), (X_test, 2)]
    ):
        subset_df = pd.DataFrame(
            X_split[:, col_indices],
            columns=[feature_names[i] for i in col_indices],
        )
        mask = simulate_missingness(
            subset_df,
            mechanism=mechanism,
            rate=rate,
            seed=seed + split_seed_offset,
            drivers_map=drivers_map,
            strength=strength,
        )
        X_out = X_split.copy()
        X_out[:, col_indices] = subset_df.where(mask).values
        results.append(X_out)

    print()
    return results[0], results[1], results[2]
