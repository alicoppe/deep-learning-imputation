"""
Comprehensive baseline result plots across all 4 sweep runs.

Usage:
  python src/analysis/plot_baselines.py

Expects the 4 most recent sweep results under results/:
  - length_of_stay clean baseline + missingness sweep
  - in_hospital_mortality clean baseline + missingness sweep
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
OUT_DIR = REPO_ROOT / "results" / "figures"

# Style constants
MODEL_COLORS = {"mlp": "#4C72B0", "xgboost": "#DD8452"}
MODEL_LABELS = {"mlp": "MLP", "xgboost": "XGBoost"}
MECHANISM_ORDER = ["MCAR", "MAR", "MNAR"]
MECHANISM_COLORS = {"MCAR": "#4C72B0", "MAR": "#55A868", "MNAR": "#C44E52"}
MECHANISM_MARKERS = {"MCAR": "o", "MAR": "s", "MNAR": "^"}
MECHANISM_LS = {"MCAR": "-", "MAR": "--", "MNAR": ":"}
TASK_LABELS = {"length_of_stay": "Length of Stay", "in_hospital_mortality": "In-Hospital Mortality"}

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
})


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _find_latest_sweeps(task: str, n: int = 2) -> list[Path]:
    """Find the n most recent sweep_summary.csv files for a task."""
    task_dir = REPO_ROOT / "results" / task
    candidates = sorted(task_dir.glob("*/sweep_summary.csv"), key=lambda p: p.stat().st_mtime)
    return candidates[-n:]


def load_clean_baseline(task: str) -> pd.DataFrame:
    """Load the clean (no-missingness) baseline sweep."""
    csvs = _find_latest_sweeps(task, n=2)
    # The baseline sweep has no missingness columns
    for csv in csvs:
        df = pd.read_csv(csv)
        if "missingness.mechanism" not in df.columns:
            df["task"] = task
            return df[df["status"] == "ok"].copy()
    raise FileNotFoundError(f"No clean baseline sweep found for {task}")


def load_missingness_sweep(task: str) -> pd.DataFrame:
    """Load the missingness sweep."""
    csvs = _find_latest_sweeps(task, n=2)
    for csv in reversed(csvs):
        df = pd.read_csv(csv)
        if "missingness.mechanism" in df.columns:
            df["task"] = task
            df["missingness.rate"] = df["missingness.rate"].astype(float)
            df["missingness.mechanism"] = df["missingness.mechanism"].str.upper()
            return df[df["status"] == "ok"].copy()
    raise FileNotFoundError(f"No missingness sweep found for {task}")


def _save(fig: plt.Figure, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


# ---------------------------------------------------------------------------
# Plot 1: Clean baseline comparison (both tasks, both models)
# ---------------------------------------------------------------------------

def plot_clean_baseline_comparison(los_base: pd.DataFrame, mort_base: pd.DataFrame) -> None:
    """Bar chart comparing MLP vs XGBoost on clean data for both tasks."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # LOS: MAE
    ax = axes[0]
    for i, model in enumerate(["mlp", "xgboost"]):
        sub = los_base[los_base["model.type"] == model]
        mean = sub["mae"].mean()
        std = sub["mae"].std()
        bar = ax.bar(i, mean, 0.5, yerr=std, capsize=6, color=MODEL_COLORS[model],
                     edgecolor="black", linewidth=0.5, label=MODEL_LABELS[model])
        ax.text(i, mean + std + 0.01, f"{mean:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["MLP", "XGBoost"])
    ax.set_ylabel("MAE (hours)")
    ax.set_title("Length of Stay", fontsize=13, fontweight="bold")
    ax.set_ylim(bottom=3.3)

    # Mortality: AUROC + AUPRC grouped
    ax = axes[1]
    metrics = ["auroc", "auprc"]
    metric_labels = ["AUROC", "AUPRC"]
    x = np.arange(len(metrics))
    width = 0.3
    for j, model in enumerate(["mlp", "xgboost"]):
        sub = mort_base[mort_base["model.type"] == model]
        means = [sub[m].mean() for m in metrics]
        stds = [sub[m].std() for m in metrics]
        offset = (j - 0.5) * width
        bars = ax.bar(x + offset, means, width, yerr=stds, capsize=5,
                      color=MODEL_COLORS[model], edgecolor="black", linewidth=0.5,
                      label=MODEL_LABELS[model])
        for k, (m, s) in enumerate(zip(means, stds)):
            ax.text(x[k] + offset, m + s + 0.005, f"{m:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylabel("Score")
    ax.set_title("In-Hospital Mortality", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right")
    ax.set_ylim(0, 1.0)

    fig.suptitle("Clean Baseline Performance (no simulated missingness)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save(fig, "clean_baseline_comparison.png")


# ---------------------------------------------------------------------------
# Plot 2: Missingness degradation curves (metric vs rate, per task)
# ---------------------------------------------------------------------------

def plot_degradation_curves(los_miss: pd.DataFrame, mort_miss: pd.DataFrame,
                            los_base: pd.DataFrame, mort_base: pd.DataFrame) -> None:
    """Metric vs missingness rate, one subplot per (task, model), lines per mechanism."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    configs = [
        (axes[0, 0], los_miss, los_base, "mae", "MAE (hours)", "Length of Stay — MLP", "mlp", False),
        (axes[0, 1], los_miss, los_base, "mae", "MAE (hours)", "Length of Stay — XGBoost", "xgboost", False),
        (axes[1, 0], mort_miss, mort_base, "auroc", "AUROC", "Mortality — MLP", "mlp", True),
        (axes[1, 1], mort_miss, mort_base, "auroc", "AUROC", "Mortality — XGBoost", "xgboost", True),
    ]

    for ax, df_miss, df_base, metric, ylabel, title, model, higher_better in configs:
        sub = df_miss[df_miss["model.type"] == model]
        base_val = df_base[df_base["model.type"] == model][metric].mean()

        # Baseline horizontal line
        ax.axhline(base_val, color="black", linestyle="-", linewidth=1.5, alpha=0.5, label="Clean baseline")

        for mech in MECHANISM_ORDER:
            m = sub[sub["missingness.mechanism"] == mech]
            if m.empty:
                continue
            agg = m.groupby("missingness.rate")[metric].agg(["mean", "std"]).reset_index()
            ax.plot(agg["missingness.rate"], agg["mean"],
                    marker=MECHANISM_MARKERS[mech], linestyle=MECHANISM_LS[mech],
                    color=MECHANISM_COLORS[mech], label=mech, linewidth=2, markersize=7)
            ax.fill_between(agg["missingness.rate"],
                            agg["mean"] - agg["std"], agg["mean"] + agg["std"],
                            color=MECHANISM_COLORS[mech], alpha=0.1)

        ax.set_xlabel("Missingness Rate")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        ax.legend(fontsize=9)

    fig.suptitle("Performance Degradation Under Simulated Missingness", fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save(fig, "degradation_curves.png")


# ---------------------------------------------------------------------------
# Plot 3: Relative performance change heatmaps (both tasks side by side)
# ---------------------------------------------------------------------------

def plot_relative_change_heatmaps(los_miss: pd.DataFrame, mort_miss: pd.DataFrame,
                                   los_base: pd.DataFrame, mort_base: pd.DataFrame) -> None:
    """Heatmaps showing % change from clean baseline for each (mechanism, rate, model) combo."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    configs = [
        (axes[0, 0], los_miss, los_base, "mae", "LOS — MLP", "mlp", False),
        (axes[0, 1], los_miss, los_base, "mae", "LOS — XGBoost", "xgboost", False),
        (axes[1, 0], mort_miss, mort_base, "auroc", "Mortality — MLP", "mlp", True),
        (axes[1, 1], mort_miss, mort_base, "auroc", "Mortality — XGBoost", "xgboost", True),
    ]

    for ax, df_miss, df_base, metric, title, model, higher_better in configs:
        base_val = df_base[df_base["model.type"] == model][metric].mean()
        sub = df_miss[df_miss["model.type"] == model]
        rates = sorted(sub["missingness.rate"].unique())
        mechs = [m for m in MECHANISM_ORDER if m in sub["missingness.mechanism"].unique()]

        matrix = np.full((len(mechs), len(rates)), np.nan)
        for r_idx, mech in enumerate(mechs):
            for c_idx, rate in enumerate(rates):
                vals = sub[(sub["missingness.mechanism"] == mech) & (sub["missingness.rate"] == rate)][metric]
                if not vals.empty:
                    pct_change = (vals.mean() - base_val) / abs(base_val) * 100
                    matrix[r_idx, c_idx] = pct_change

        # For MAE (lower is better), positive % = worse; for AUROC, negative % = worse
        # Use diverging colormap centered at 0
        vmax = max(abs(np.nanmin(matrix)), abs(np.nanmax(matrix)))
        cmap = "RdYlGn" if higher_better else "RdYlGn_r"
        im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax)

        ax.set_xticks(range(len(rates)))
        ax.set_xticklabels([f"{r:.0%}" for r in rates])
        ax.set_yticks(range(len(mechs)))
        ax.set_yticklabels(mechs)
        ax.set_xlabel("Missingness Rate")
        ax.set_title(title, fontsize=12, fontweight="bold")

        for r_idx in range(len(mechs)):
            for c_idx in range(len(rates)):
                v = matrix[r_idx, c_idx]
                if not np.isnan(v):
                    sign = "+" if v > 0 else ""
                    ax.text(c_idx, r_idx, f"{sign}{v:.2f}%", ha="center", va="center",
                            fontsize=9, fontweight="bold",
                            color="white" if abs(v) > vmax * 0.6 else "black")

        plt.colorbar(im, ax=ax, label="% change from baseline", shrink=0.8)

    fig.suptitle("Relative Change from Clean Baseline (%)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save(fig, "relative_change_heatmaps.png")


# ---------------------------------------------------------------------------
# Plot 4: MLP vs XGBoost robustness comparison
# ---------------------------------------------------------------------------

def plot_model_robustness(los_miss: pd.DataFrame, mort_miss: pd.DataFrame,
                          los_base: pd.DataFrame, mort_base: pd.DataFrame) -> None:
    """Direct MLP vs XGBoost comparison: metric delta from baseline at each condition."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    task_configs = [
        (axes[0], los_miss, los_base, "mae", "Length of Stay (MAE)", False),
        (axes[1], mort_miss, mort_base, "auroc", "Mortality (AUROC)", True),
    ]

    for ax, df_miss, df_base, metric, title, higher_better in task_configs:
        rates = sorted(df_miss["missingness.rate"].unique())
        mechs = [m for m in MECHANISM_ORDER if m in df_miss["missingness.mechanism"].unique()]

        x_labels = []
        mlp_deltas = []
        xgb_deltas = []
        for mech in mechs:
            for rate in rates:
                x_labels.append(f"{mech}\n{rate:.0%}")
                for model, deltas in [("mlp", mlp_deltas), ("xgboost", xgb_deltas)]:
                    base_val = df_base[df_base["model.type"] == model][metric].mean()
                    sub = df_miss[(df_miss["model.type"] == model) &
                                 (df_miss["missingness.mechanism"] == mech) &
                                 (df_miss["missingness.rate"] == rate)]
                    delta = sub[metric].mean() - base_val
                    # For MAE, positive delta = worse; negate so up = better for both
                    if not higher_better:
                        delta = -delta
                    deltas.append(delta)

        x = np.arange(len(x_labels))
        width = 0.35
        ax.bar(x - width / 2, mlp_deltas, width, color=MODEL_COLORS["mlp"], label="MLP",
               edgecolor="black", linewidth=0.5)
        ax.bar(x + width / 2, xgb_deltas, width, color=MODEL_COLORS["xgboost"], label="XGBoost",
               edgecolor="black", linewidth=0.5)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=8)
        direction = "higher = better" if higher_better else "closer to 0 = better"
        ax.set_ylabel(f"Delta from clean baseline ({direction})")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.legend()

        # Add vertical separators between mechanisms
        for i in range(1, len(mechs)):
            ax.axvline(i * len(rates) - 0.5, color="grey", linewidth=0.5, linestyle="--", alpha=0.5)

    fig.suptitle("Model Robustness to Missingness (delta from clean baseline)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save(fig, "model_robustness.png")


# ---------------------------------------------------------------------------
# Plot 5: Mechanism comparison — % change from baseline (grouped bars)
# ---------------------------------------------------------------------------

def plot_mechanism_comparison(los_miss: pd.DataFrame, mort_miss: pd.DataFrame,
                              los_base: pd.DataFrame, mort_base: pd.DataFrame) -> None:
    """Grouped bar chart of % change from clean baseline, by mechanism and rate."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    task_configs = [
        (axes[0], los_miss, los_base, "mae", "Length of Stay (MAE)", False),
        (axes[1], mort_miss, mort_base, "auroc", "Mortality (AUROC)", True),
    ]

    for ax, df_miss, df_base, metric, title, higher_better in task_configs:
        base_val = df_base[metric].mean()  # averaged across both models
        rates = sorted(df_miss["missingness.rate"].unique())
        mechs = [m for m in MECHANISM_ORDER if m in df_miss["missingness.mechanism"].unique()]

        x = np.arange(len(rates))
        width = 0.25
        offsets = {m: (i - 1) * width for i, m in enumerate(mechs)}

        for mech in mechs:
            pct_changes = []
            for rate in rates:
                sub = df_miss[(df_miss["missingness.mechanism"] == mech) &
                              (df_miss["missingness.rate"] == rate)]
                pct = (sub[metric].mean() - base_val) / abs(base_val) * 100
                pct_changes.append(pct)
            ax.bar(x + offsets[mech], pct_changes, width,
                   color=MECHANISM_COLORS[mech], edgecolor="black", linewidth=0.5,
                   label=mech)
            # Value labels
            for k, v in enumerate(pct_changes):
                sign = "+" if v > 0 else ""
                va = "bottom" if v >= 0 else "top"
                ax.text(x[k] + offsets[mech], v, f"{sign}{v:.1f}%",
                        ha="center", va=va, fontsize=8, fontweight="bold")

        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{r:.0%}" for r in rates])
        ax.set_xlabel("Missingness Rate")
        ax.set_ylabel("% Change from Clean Baseline")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)

    fig.suptitle("Impact of Missingness Mechanism by Rate (% change from baseline)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save(fig, "mechanism_comparison.png")


# ---------------------------------------------------------------------------
# Plot 6: Seed variance (stability analysis)
# ---------------------------------------------------------------------------

def plot_seed_variance(los_base: pd.DataFrame, mort_base: pd.DataFrame,
                       los_miss: pd.DataFrame, mort_miss: pd.DataFrame) -> None:
    """Strip/swarm plot showing individual seed results to visualize variance."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # LOS
    ax = axes[0]
    all_conditions = [("Clean", los_base, None, None)]
    for mech in MECHANISM_ORDER:
        for rate in sorted(los_miss["missingness.rate"].unique()):
            all_conditions.append((f"{mech}\n{rate:.0%}", los_miss, mech, rate))

    x_pos = 0
    tick_positions = []
    tick_labels = []
    for label, df, mech, rate in all_conditions:
        for j, model in enumerate(["mlp", "xgboost"]):
            if mech is None:
                sub = df[df["model.type"] == model]
            else:
                sub = df[(df["model.type"] == model) &
                         (df["missingness.mechanism"] == mech) &
                         (df["missingness.rate"] == rate)]
            jitter = (np.random.default_rng(42).random(len(sub)) - 0.5) * 0.15
            ax.scatter([x_pos + j * 0.4] * len(sub) + jitter, sub["mae"],
                       c=MODEL_COLORS[model], s=30, alpha=0.7, edgecolors="black", linewidths=0.3)
        tick_positions.append(x_pos + 0.2)
        tick_labels.append(label)
        x_pos += 1.2

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=7)
    ax.set_ylabel("MAE (hours)")
    ax.set_title("Length of Stay — Per-Seed Results", fontsize=12, fontweight="bold")
    legend_elements = [Patch(facecolor=MODEL_COLORS["mlp"], label="MLP"),
                       Patch(facecolor=MODEL_COLORS["xgboost"], label="XGBoost")]
    ax.legend(handles=legend_elements, fontsize=9)

    # Mortality
    ax = axes[1]
    all_conditions = [("Clean", mort_base, None, None)]
    for mech in MECHANISM_ORDER:
        for rate in sorted(mort_miss["missingness.rate"].unique()):
            all_conditions.append((f"{mech}\n{rate:.0%}", mort_miss, mech, rate))

    x_pos = 0
    tick_positions = []
    tick_labels = []
    for label, df, mech, rate in all_conditions:
        for j, model in enumerate(["mlp", "xgboost"]):
            if mech is None:
                sub = df[df["model.type"] == model]
            else:
                sub = df[(df["model.type"] == model) &
                         (df["missingness.mechanism"] == mech) &
                         (df["missingness.rate"] == rate)]
            jitter = (np.random.default_rng(42).random(len(sub)) - 0.5) * 0.15
            ax.scatter([x_pos + j * 0.4] * len(sub) + jitter, sub["auroc"],
                       c=MODEL_COLORS[model], s=30, alpha=0.7, edgecolors="black", linewidths=0.3)
        tick_positions.append(x_pos + 0.2)
        tick_labels.append(label)
        x_pos += 1.2

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=7)
    ax.set_ylabel("AUROC")
    ax.set_title("Mortality — Per-Seed Results", fontsize=12, fontweight="bold")
    ax.legend(handles=legend_elements, fontsize=9)

    fig.suptitle("Seed Variance Across All Conditions", fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save(fig, "seed_variance.png")


# ---------------------------------------------------------------------------
# Plot 7: Cross-task summary table as figure
# ---------------------------------------------------------------------------

def plot_summary_table(los_base: pd.DataFrame, mort_base: pd.DataFrame,
                       los_miss: pd.DataFrame, mort_miss: pd.DataFrame) -> None:
    """Comprehensive summary table rendered as a figure."""
    rows = []

    # Clean baselines
    for model in ["mlp", "xgboost"]:
        los_sub = los_base[los_base["model.type"] == model]
        mort_sub = mort_base[mort_base["model.type"] == model]
        rows.append([
            MODEL_LABELS[model], "Clean", "—",
            f"{los_sub['mae'].mean():.3f} ± {los_sub['mae'].std():.3f}",
            f"{mort_sub['auroc'].mean():.3f} ± {mort_sub['auroc'].std():.3f}",
            f"{mort_sub['auprc'].mean():.3f} ± {mort_sub['auprc'].std():.3f}",
        ])

    # Missingness conditions
    for mech in MECHANISM_ORDER:
        for rate in sorted(los_miss["missingness.rate"].unique()):
            for model in ["mlp", "xgboost"]:
                los_sub = los_miss[(los_miss["model.type"] == model) &
                                   (los_miss["missingness.mechanism"] == mech) &
                                   (los_miss["missingness.rate"] == rate)]
                mort_sub = mort_miss[(mort_miss["model.type"] == model) &
                                     (mort_miss["missingness.mechanism"] == mech) &
                                     (mort_miss["missingness.rate"] == rate)]
                rows.append([
                    MODEL_LABELS[model], mech, f"{rate:.0%}",
                    f"{los_sub['mae'].mean():.3f} ± {los_sub['mae'].std():.3f}",
                    f"{mort_sub['auroc'].mean():.3f} ± {mort_sub['auroc'].std():.3f}",
                    f"{mort_sub['auprc'].mean():.3f} ± {mort_sub['auprc'].std():.3f}",
                ])

    col_labels = ["Model", "Mechanism", "Rate", "LOS MAE (h)", "Mort. AUROC", "Mort. AUPRC"]

    fig, ax = plt.subplots(figsize=(14, max(6, 0.35 * len(rows) + 2)))
    ax.axis("off")

    table = ax.table(cellText=rows, colLabels=col_labels, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)

    # Style header
    for j in range(len(col_labels)):
        cell = table[0, j]
        cell.set_facecolor("#2C3E50")
        cell.set_text_props(color="white", fontweight="bold")

    # Alternate row colors and highlight clean baselines
    for i in range(len(rows)):
        for j in range(len(col_labels)):
            cell = table[i + 1, j]
            if i % 2 == 0:
                cell.set_facecolor("#F8F9FA")

    fig.suptitle("Complete Baseline Results Summary", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    _save(fig, "summary_table.png")


# ---------------------------------------------------------------------------
# Plot 8: Mortality multi-metric view
# ---------------------------------------------------------------------------

def plot_mortality_multimetric(mort_miss: pd.DataFrame, mort_base: pd.DataFrame) -> None:
    """Show AUROC, AUPRC, and log loss degradation side by side for mortality."""
    metrics = [("auroc", "AUROC", True), ("auprc", "AUPRC", True), ("logloss", "Log Loss", False)]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, (metric, ylabel, higher_better) in zip(axes, metrics):
        base_mlp = mort_base[mort_base["model.type"] == "mlp"][metric].mean()
        base_xgb = mort_base[mort_base["model.type"] == "xgboost"][metric].mean()

        for model, base_val, ls_offset in [("mlp", base_mlp, -0.005), ("xgboost", base_xgb, 0.005)]:
            ax.axhline(base_val, color=MODEL_COLORS[model], linestyle="--", linewidth=1, alpha=0.5)

            for mech in MECHANISM_ORDER:
                sub = mort_miss[(mort_miss["model.type"] == model) &
                                (mort_miss["missingness.mechanism"] == mech)]
                if sub.empty:
                    continue
                agg = sub.groupby("missingness.rate")[metric].agg(["mean", "std"]).reset_index()
                # Slight x offset so lines don't overlap
                x_offset = ls_offset + (MECHANISM_ORDER.index(mech) - 1) * 0.003
                ax.plot(agg["missingness.rate"] + x_offset, agg["mean"],
                        marker=MECHANISM_MARKERS[mech], linestyle=MECHANISM_LS[mech],
                        color=MODEL_COLORS[model], linewidth=1.5, markersize=5, alpha=0.8)

        ax.set_xlabel("Missingness Rate")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel, fontsize=12, fontweight="bold")
        ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))

    # Shared legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=MODEL_COLORS["mlp"], linewidth=2, label="MLP"),
        Line2D([0], [0], color=MODEL_COLORS["xgboost"], linewidth=2, label="XGBoost"),
        Line2D([0], [0], color="grey", marker="o", linestyle="-", label="MCAR"),
        Line2D([0], [0], color="grey", marker="s", linestyle="--", label="MAR"),
        Line2D([0], [0], color="grey", marker="^", linestyle=":", label="MNAR"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=5, fontsize=10,
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("Mortality — Multi-Metric Degradation Under Missingness", fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save(fig, "mortality_multimetric.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Loading data ...")
    los_base = load_clean_baseline("length_of_stay")
    mort_base = load_clean_baseline("in_hospital_mortality")
    los_miss = load_missingness_sweep("length_of_stay")
    mort_miss = load_missingness_sweep("in_hospital_mortality")

    print(f"  LOS baseline: {len(los_base)} runs")
    print(f"  LOS missingness: {len(los_miss)} runs")
    print(f"  Mortality baseline: {len(mort_base)} runs")
    print(f"  Mortality missingness: {len(mort_miss)} runs")
    print()

    print("Generating plots ...")
    plot_clean_baseline_comparison(los_base, mort_base)
    plot_degradation_curves(los_miss, mort_miss, los_base, mort_base)
    plot_relative_change_heatmaps(los_miss, mort_miss, los_base, mort_base)
    plot_model_robustness(los_miss, mort_miss, los_base, mort_base)
    plot_mechanism_comparison(los_miss, mort_miss, los_base, mort_base)
    plot_seed_variance(los_base, mort_base, los_miss, mort_miss)
    plot_summary_table(los_base, mort_base, los_miss, mort_miss)
    plot_mortality_multimetric(mort_miss, mort_base)

    print(f"\nDone. All plots → {OUT_DIR}/")


if __name__ == "__main__":
    main()
