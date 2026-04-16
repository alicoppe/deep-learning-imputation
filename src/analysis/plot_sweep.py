"""
Plot sweep results from a sweep_summary.csv file.

Usage:
  python src/analysis/plot_sweep.py                          # auto-detect most recent
  python src/analysis/plot_sweep.py --csv results/length_of_stay/20260415_143201/sweep_summary.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())

MECHANISM_ORDER = ["MCAR", "MAR", "MNAR"]
MODEL_COLORS = {"mlp": "#4C72B0", "xgboost": "#DD8452"}
MECHANISM_MARKERS = {"MCAR": "o", "MAR": "s", "MNAR": "^"}


def load(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df[df["status"] == "ok"].copy()
    df["missingness.rate"] = df["missingness.rate"].astype(float)
    df["missingness.mechanism"] = df["missingness.mechanism"].str.upper()
    return df


def _detect_metric(df: pd.DataFrame, override: str | None) -> tuple[str, str, bool]:
    """Return (col, y_label, higher_is_better)."""
    if override:
        col = override
        higher = col in ("auroc", "auprc", "r2", "accuracy", "f1")
        label = col.upper()
        return col, label, higher
    if "mae" in df.columns:
        return "mae", "MAE (hours)", False
    if "auroc" in df.columns:
        return "auroc", "AUROC", True
    raise ValueError(f"Cannot auto-detect metric column. Available: {list(df.columns)}")


def plot_metric_vs_rate(df: pd.DataFrame, out_dir: Path, metric: str, ylabel: str) -> None:
    """Primary metric vs missingness rate, one line per mechanism, faceted by model."""
    models = sorted(df["model.type"].unique())
    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 5), sharey=True)
    if len(models) == 1:
        axes = [axes]

    for ax, model in zip(axes, models):
        sub = df[df["model.type"] == model]
        for mech in MECHANISM_ORDER:
            m = sub[sub["missingness.mechanism"] == mech]
            if m.empty:
                continue
            agg = m.groupby("missingness.rate")[metric].agg(["mean", "std"]).reset_index()
            ax.plot(
                agg["missingness.rate"], agg["mean"],
                marker=MECHANISM_MARKERS[mech], label=mech, linewidth=2, markersize=7,
            )
            ax.fill_between(
                agg["missingness.rate"],
                agg["mean"] - agg["std"],
                agg["mean"] + agg["std"],
                alpha=0.15,
            )
        ax.set_title(model.upper(), fontsize=13, fontweight="bold")
        ax.set_xlabel("Missingness rate")
        ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        ax.legend(title="Mechanism")
        ax.grid(axis="y", alpha=0.4)

    axes[0].set_ylabel(ylabel)
    fig.suptitle(f"{ylabel} vs Missingness Rate", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    _save(fig, out_dir / "metric_vs_rate.png")


def plot_model_comparison(df: pd.DataFrame, out_dir: Path, metric: str, ylabel: str) -> None:
    """Side-by-side primary metric for MLP vs XGBoost at each (mechanism, rate) combo."""
    rates = sorted(df["missingness.rate"].unique())
    mechs = [m for m in MECHANISM_ORDER if m in df["missingness.mechanism"].unique()]

    n_cols = len(rates)
    n_rows = len(mechs)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 3.5 * n_rows), sharey=True, sharex=True)
    axes = np.array(axes).reshape(n_rows, n_cols)

    models = sorted(df["model.type"].unique())
    x = np.arange(len(models))
    width = 0.5

    for r_idx, mech in enumerate(mechs):
        for c_idx, rate in enumerate(rates):
            ax = axes[r_idx, c_idx]
            sub = df[(df["missingness.mechanism"] == mech) & (df["missingness.rate"] == rate)]
            means = [sub[sub["model.type"] == m][metric].mean() for m in models]
            stds = [sub[sub["model.type"] == m][metric].std() for m in models]
            ax.bar(x, means, width, yerr=stds, capsize=4,
                   color=[MODEL_COLORS.get(m, "grey") for m in models])
            ax.set_xticks(x)
            ax.set_xticklabels([m.upper() for m in models])
            ax.set_title(f"{mech}  rate={rate:.0%}", fontsize=9)
            ax.grid(axis="y", alpha=0.4)
            if c_idx == 0:
                ax.set_ylabel(ylabel)

    fig.suptitle(f"Model Comparison by Missingness Condition", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    _save(fig, out_dir / "model_comparison.png")


def plot_heatmap(df: pd.DataFrame, out_dir: Path, metric: str, ylabel: str, higher_is_better: bool) -> None:
    """Heatmap of mean primary metric: rows = mechanism, cols = rate, faceted by model."""
    models = sorted(df["model.type"].unique())
    rates = sorted(df["missingness.rate"].unique())
    mechs = [m for m in MECHANISM_ORDER if m in df["missingness.mechanism"].unique()]

    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 3.5))
    if len(models) == 1:
        axes = [axes]

    all_vals = df.groupby(["model.type", "missingness.mechanism", "missingness.rate"])[metric].mean()
    vmin = all_vals.min()
    vmax = all_vals.max()
    # Good = green. For lower-is-better metrics (MAE, logloss) reverse the colormap.
    cmap = "RdYlGn" if higher_is_better else "RdYlGn_r"

    for ax, model in zip(axes, models):
        matrix = np.full((len(mechs), len(rates)), np.nan)
        for r_idx, mech in enumerate(mechs):
            for c_idx, rate in enumerate(rates):
                try:
                    matrix[r_idx, c_idx] = all_vals[(model, mech, rate)]
                except KeyError:
                    pass
        im = ax.imshow(matrix, aspect="auto", vmin=vmin, vmax=vmax, cmap=cmap)
        ax.set_xticks(range(len(rates)))
        ax.set_xticklabels([f"{r:.0%}" for r in rates])
        ax.set_yticks(range(len(mechs)))
        ax.set_yticklabels(mechs)
        ax.set_xlabel("Missingness rate")
        ax.set_title(model.upper(), fontsize=12, fontweight="bold")
        for r_idx in range(len(mechs)):
            for c_idx in range(len(rates)):
                v = matrix[r_idx, c_idx]
                if not np.isnan(v):
                    ax.text(c_idx, r_idx, f"{v:.3f}", ha="center", va="center", fontsize=9,
                            color="white" if v > (vmin + vmax) / 2 else "black")
        plt.colorbar(im, ax=ax, label=ylabel)

    fig.suptitle(f"Mean {ylabel} Heatmap", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, out_dir / "heatmap.png")



def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=None, help="Path to sweep_summary.csv (auto-detects most recent if omitted)")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--metric", type=str, default=None,
                    help="Metric column to plot (auto-detects mae or auroc if omitted)")
    args = ap.parse_args()

    csv_path = args.csv
    if csv_path is None:
        # Find the most recent sweep_summary.csv under results/
        candidates = sorted(REPO_ROOT.glob("results/**/sweep_summary.csv"), key=lambda p: p.stat().st_mtime)
        if not candidates:
            print("No sweep_summary.csv found under results/. Pass --csv explicitly.")
            sys.exit(1)
        csv_path = candidates[-1]

    out_dir = args.out or csv_path.parent

    print(f"Loading {csv_path} ...")
    df = load(csv_path)
    print(f"  {len(df)} successful runs  ({df['model.type'].nunique()} models, "
          f"{df['missingness.mechanism'].nunique()} mechanisms, "
          f"{df['missingness.rate'].nunique()} rates, "
          f"{df['seed'].nunique()} seeds)\n")

    metric, ylabel, higher = _detect_metric(df, args.metric)
    print(f"  Metric: {metric}  ({ylabel})  higher_is_better={higher}\n")

    print("Generating plots ...")
    plot_metric_vs_rate(df, out_dir, metric, ylabel)
    plot_model_comparison(df, out_dir, metric, ylabel)
    plot_heatmap(df, out_dir, metric, ylabel, higher)
    print(f"\nDone. All plots → {out_dir}/")


if __name__ == "__main__":
    main()
