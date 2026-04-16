"""Config-driven training orchestrator with sweep support."""

from __future__ import annotations

import copy
import itertools
import json
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from src.data.pipeline import run_pipeline
from src.tasks.length_of_stay import LOSTask
from src.tasks.mortality import InHospitalMortalityTask
from src.training.base_model import TrainResult
from src.training.mlp_model import MLPModel
from src.training.xgboost_model import XGBoostModel

TASK_REGISTRY = {
    "length_of_stay": LOSTask,
    "in_hospital_mortality": InHospitalMortalityTask,
}

MODEL_REGISTRY = {
    "mlp": MLPModel,
    "xgboost": XGBoostModel,
}

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())


def run(config: dict, out_dir: Path | None = None) -> dict:
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

    # Inject task-type so models can branch on loss / output / predict
    params["task_type"] = task.task_type
    params["n_classes"] = task.n_classes

    # Reduce epochs/rounds for --quick mode
    if config.get("quick"):
        if model_key == "mlp":
            params["epochs"] = min(params.get("epochs", 60), 10)
        elif model_key == "xgboost":
            params["n_estimators"] = min(params.get("n_estimators", 2000), 100)

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

    if out_dir is None:
        task_key = config["task"]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = REPO_ROOT / "results" / task_key / timestamp

    _save_results(out_dir, config, metrics, result, splits, y_pred, model, model_key, task)

    return metrics


def run_sweep(config: dict) -> None:
    """Expand sweep config into individual runs and execute them all."""
    sweep_params = config["sweep"]
    defaults = config.get("defaults", {})

    keys = list(sweep_params.keys())
    value_lists = [sweep_params[k] for k in keys]

    combos = list(itertools.product(*value_lists))
    print(f"Sweep: {len(combos)} runs  ({' × '.join(str(len(v)) for v in value_lists)} combos)\n")

    # Create sweep group folder
    task_key = defaults.get("task", "unknown")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_dir = REPO_ROOT / "results" / task_key / timestamp
    sweep_dir.mkdir(parents=True, exist_ok=True)

    # Save the original sweep config at the group level
    (sweep_dir / "config.yaml").write_text(yaml.dump(config, default_flow_style=False))

    all_results = []
    for i, combo in enumerate(combos, 1):
        run_cfg = _deep_merge(copy.deepcopy(defaults), {})
        for key_path, value in zip(keys, combo):
            _set_nested(run_cfg, key_path.split("."), value)

        # Build a descriptive subfolder name from the swept parameters
        run_label = _make_run_label(keys, combo)
        run_dir = sweep_dir / run_label

        print(f"\n{'═' * 60}")
        print(f"  Run {i}/{len(combos)}  {run_label}")
        print(f"{'═' * 60}")

        try:
            metrics = run(run_cfg, out_dir=run_dir)
            row = {"run": i, "label": run_label,
                   **{k: v for k, v in zip(keys, combo)}, **metrics, "status": "ok"}
        except Exception as e:
            print(f"  ERROR: {e}")
            row = {"run": i, "label": run_label,
                   **{k: v for k, v in zip(keys, combo)}, "status": f"error: {e}"}

        all_results.append(row)

    # Write sweep summary alongside the condition subfolders
    summary_path = sweep_dir / "sweep_summary.csv"
    pd.DataFrame(all_results).to_csv(summary_path, index=False)
    print(f"\nSweep complete. Summary → {summary_path}")

    # Auto-generate plots from the summary
    try:
        from src.analysis.plot_sweep import load, _detect_metric, plot_metric_vs_rate, plot_model_comparison, plot_heatmap, plot_seed_variance
        df_summary = load(summary_path)
        if not df_summary.empty:
            metric, ylabel, higher = _detect_metric(df_summary, None)
            print(f"\nGenerating sweep plots (metric={metric}) ...")
            plot_metric_vs_rate(df_summary, sweep_dir, metric, ylabel)
            plot_model_comparison(df_summary, sweep_dir, metric, ylabel)
            plot_heatmap(df_summary, sweep_dir, metric, ylabel, higher)
            plot_seed_variance(df_summary, sweep_dir, metric, ylabel)
            print(f"Plots → {sweep_dir}/")
    except Exception as e:
        print(f"  Warning: sweep plots failed ({e})")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_run_label(keys: list[str], combo: tuple) -> str:
    """Build a compact subfolder name from swept parameter values.

    Strips common prefixes (e.g. 'missingness.' → '') to keep names short.
    Example: MCAR_mlp_0.25_seed42
    """
    parts = []
    for key, val in zip(keys, combo):
        short_key = key.rsplit(".", 1)[-1]  # missingness.mechanism → mechanism
        # For well-known keys just use the value; for others include the key
        if short_key in ("mechanism", "type"):
            parts.append(str(val))
        elif short_key == "seed":
            parts.append(f"seed{val}")
        elif short_key == "rate":
            parts.append(f"rate{val}")
        else:
            parts.append(f"{short_key}{val}")
    return "_".join(parts)


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


def _save_results(
    out_dir: Path,
    config: dict,
    metrics: dict,
    result: TrainResult,
    splits: dict,
    y_pred: np.ndarray,
    model,
    model_key: str,
    task,
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

    # For direct (non-sweep) runs, save config alongside results
    config_path = out_dir / "config.yaml"
    if not config_path.exists():
        config_path.write_text(yaml.dump(config, default_flow_style=False))

    # Training curve
    if result.train_losses:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(result.train_losses, label="Train")
        ax.plot(result.val_losses, label="Validation")
        xlabel = "Epoch" if model_key == "mlp" else "Round"
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Loss")
        ax.set_title(f"{model_key.upper()} Training Curve")
        ax.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "training_curve.png", dpi=150)
        plt.close()

    # Task-specific prediction plot
    task.plot_predictions(splits["y_test"], y_pred, out_dir / "predictions.png")

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
