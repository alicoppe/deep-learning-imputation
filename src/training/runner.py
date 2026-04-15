"""Config-driven training orchestrator with sweep support."""

from __future__ import annotations

import copy
import itertools
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.pipeline import run_pipeline
from src.tasks.length_of_stay import LOSTask
from src.training.base_model import TrainResult
from src.training.mlp_model import MLPModel
from src.training.xgboost_model import XGBoostModel

TASK_REGISTRY = {
    "length_of_stay": LOSTask,
}

MODEL_REGISTRY = {
    "mlp": MLPModel,
    "xgboost": XGBoostModel,
}

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())


def run(config: dict) -> dict:
    """Execute a single training run defined by config. Returns metrics dict."""
    t0 = time.time()

    seed = config.get("seed", 42)
    import torch
    torch.manual_seed(seed)

    task_key = config["task"]
    task = TASK_REGISTRY[task_key]()

    model_cfg = config["model"]
    model_key = model_cfg["type"]
    params = dict(model_cfg.get("params", {}))

    # Convert hidden_dims to list (YAML may give it as a list already)
    if "hidden_dims" in params and not isinstance(params["hidden_dims"], list):
        params["hidden_dims"] = list(params["hidden_dims"])

    # Reduce epochs/rounds for --quick mode
    if config.get("quick"):
        if model_key == "mlp":
            params["epochs"] = min(params.get("epochs", 60), 10)
        elif model_key == "xgboost":
            params["n_estimators"] = min(params.get("n_estimators", 2000), 200)

    model = MODEL_REGISTRY[model_key](**params)

    _print_header(config, task, model_key)

    print("Loading data & building features ...")
    splits = run_pipeline(task, config)

    suffix = "" if model.needs_imputed else "_raw"
    X_tr = splits[f"X_train{suffix}"]
    X_val = splits[f"X_val{suffix}"]
    X_te = splits[f"X_test{suffix}"]

    print(f"Training {model_key.upper()} ...")
    result: TrainResult = model.train(X_tr, splits["y_train"], X_val, splits["y_val"])

    y_pred = model.predict(X_te)
    metrics = task.evaluate(splits["y_test"], y_pred)

    elapsed = time.time() - t0
    _print_metrics(model_key, metrics, elapsed)

    out_dir = _resolve_output_dir(config)
    _save_results(out_dir, config, metrics, result, splits, y_pred, model, model_key)

    return metrics


def run_sweep(config: dict) -> None:
    """Expand sweep config into individual runs and execute them all."""
    sweep_params = config["sweep"]
    defaults = config.get("defaults", {})

    # Build list of (key_path, values) pairs
    keys = list(sweep_params.keys())
    value_lists = [sweep_params[k] for k in keys]

    combos = list(itertools.product(*value_lists))
    print(f"Sweep: {len(combos)} runs  ({' × '.join(str(len(v)) for v in value_lists)} combos)\n")

    all_results = []
    for i, combo in enumerate(combos, 1):
        run_cfg = _deep_merge(copy.deepcopy(defaults), {})
        for key_path, value in zip(keys, combo):
            _set_nested(run_cfg, key_path.split("."), value)

        print(f"\n{'═' * 60}")
        print(f"  Run {i}/{len(combos)}  " + "  ".join(f"{k}={v}" for k, v in zip(keys, combo)))
        print(f"{'═' * 60}")

        try:
            metrics = run(run_cfg)
            row = {"run": i, **{k: v for k, v in zip(keys, combo)}, **metrics, "status": "ok"}
        except Exception as e:
            print(f"  ERROR: {e}")
            row = {"run": i, **{k: v for k, v in zip(keys, combo)}, "status": f"error: {e}"}

        all_results.append(row)

    # Write sweep summary
    sweep_dir = REPO_ROOT / "results" / "sweep"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    summary_path = sweep_dir / "summary.csv"
    pd.DataFrame(all_results).to_csv(summary_path, index=False)
    print(f"\nSweep complete. Summary → {summary_path}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _print_header(config: dict, task, model_key: str) -> None:
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    miss_cfg = config.get("missingness", {}) or {}
    miss_str = (
        f"{miss_cfg.get('mechanism', '')} rate={miss_cfg.get('rate', 0)}"
        if miss_cfg.get("rate", 0) > 0 else "none"
    )
    print(f"\n{'═' * 60}")
    print(f"  Task: {task.name}  |  Model: {model_key}  |  device={device}")
    print(f"  seed={config.get('seed', 42)}  embeddings={config.get('data', {}).get('embeddings', False)}")
    print(f"  missingness={miss_str}")
    print(f"{'═' * 60}\n")


def _print_metrics(model_key: str, metrics: dict, elapsed: float) -> None:
    print(f"\n  Results — {model_key.upper()}")
    print(f"  {'─' * 40}")
    for k, v in metrics.items():
        print(f"    {k:<10} {v:.4f}")
    print(f"  Elapsed: {elapsed:.1f}s\n")


def _resolve_output_dir(config: dict) -> Path:
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    task_key = config["task"]
    return REPO_ROOT / "results" / task_key / timestamp


def _save_results(
    out_dir: Path,
    config: dict,
    metrics: dict,
    result: TrainResult,
    splits: dict,
    y_pred: np.ndarray,
    model,
    model_key: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    results_json = {
        "metrics": metrics,
        "n_train": int(splits["X_train"].shape[0]),
        "n_val": int(splits["X_val"].shape[0]),
        "n_test": int(splits["X_test"].shape[0]),
        "n_features": int(splits["X_train"].shape[1]),
    }
    (out_dir / "results.json").write_text(json.dumps(results_json, indent=2))

    import yaml
    (out_dir / "config.yaml").write_text(yaml.dump(config, default_flow_style=False))

    # Training curve
    if result.train_losses:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(result.train_losses, label="Train")
        ax.plot(result.val_losses, label="Validation")
        xlabel = "Epoch" if model_key == "mlp" else "Round"
        ax.set_xlabel(xlabel)
        ax.set_ylabel("MAE (hours)")
        ax.set_title(f"{model_key.upper()} Training Curve")
        ax.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "training_curve.png", dpi=150)
        plt.close()

    # Predicted vs actual
    y_test = splits["y_test"]
    fig, ax = plt.subplots(figsize=(6, 6))
    n = min(3000, len(y_test))
    ax.scatter(y_test[:n], y_pred[:n], alpha=0.2, s=5)
    ax.plot([0, 40], [0, 40], "r--", label="Perfect prediction")
    ax.set_xlabel("Actual stay length (h)")
    ax.set_ylabel("Predicted stay length (h)")
    ax.set_title(f"{model_key.upper()} — Predicted vs Actual")
    ax.set_xlim(0, 40)
    ax.set_ylim(0, 40)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "predicted_vs_actual.png", dpi=150)
    plt.close()

    # XGBoost feature importance
    if model_key == "xgboost" and hasattr(model, "feature_importances"):
        feat_imp = pd.Series(
            model.feature_importances, index=splits["feature_names"]
        ).sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(8, 7))
        feat_imp.head(30).plot(kind="barh", ax=ax)
        ax.invert_yaxis()
        ax.set_xlabel("Importance (gain)")
        ax.set_title("Top 30 XGBoost Feature Importances")
        plt.tight_layout()
        plt.savefig(out_dir / "feature_importance.png", dpi=150)
        plt.close()

    print(f"  Results saved → {out_dir}/\n")


def _deep_merge(base: dict, override: dict) -> dict:
    result = copy.deepcopy(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def _set_nested(d: dict, keys: list[str], value) -> None:
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    d[keys[-1]] = value
