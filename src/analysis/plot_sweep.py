"""
Plot sweep results from results/sweep/summary.csv.

Usage:
  python src/analysis/plot_sweep.py
  python src/analysis/plot_sweep.py --csv results/sweep/summary.csv --out results/sweep/
"""

from __future__ import annotations

import argparse
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


def plot_mae_vs_rate(df: pd.DataFrame, out_dir: Path) -> None:
    """MAE vs missingness rate, one line per mechanism, faceted by model."""
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
            agg = m.groupby("missingness.rate")["mae"].agg(["mean", "std"]).reset_index()
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

    axes[0].set_ylabel("MAE (hours)")
    fig.suptitle("MAE vs Missingness Rate", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    _save(fig, out_dir / "mae_vs_rate.png")


def plot_model_comparison(df: pd.DataFrame, out_dir: Path) -> None:
    """Side-by-side MAE for MLP vs XGBoost at each (mechanism, rate) combo."""
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
            means = [sub[sub["model.type"] == m]["mae"].mean() for m in models]
            stds = [sub[sub["model.type"] == m]["mae"].std() for m in models]
            bars = ax.bar(x, means, width, yerr=stds, capsize=4,
                          color=[MODEL_COLORS.get(m, "grey") for m in models])
            ax.set_xticks(x)
            ax.set_xticklabels([m.upper() for m in models])
            ax.set_title(f"{mech}  rate={rate:.0%}", fontsize=9)
            ax.grid(axis="y", alpha=0.4)
            if c_idx == 0:
                ax.set_ylabel("MAE (hours)")

    fig.suptitle("Model Comparison by Missingness Condition", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    _save(fig, out_dir / "model_comparison.png")


def plot_heatmap(df: pd.DataFrame, out_dir: Path) -> None:
    """Heatmap of mean MAE: rows = mechanism, cols = rate, faceted by model."""
    models = sorted(df["model.type"].unique())
    rates = sorted(df["missingness.rate"].unique())
    mechs = [m for m in MECHANISM_ORDER if m in df["missingness.mechanism"].unique()]

    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 3.5))
    if len(models) == 1:
        axes = [axes]

    all_vals = df.groupby(["model.type", "missingness.mechanism", "missingness.rate"])["mae"].mean()
    vmin = all_vals.min()
    vmax = all_vals.max()

    for ax, model in zip(axes, models):
        matrix = np.full((len(mechs), len(rates)), np.nan)
        for r_idx, mech in enumerate(mechs):
            for c_idx, rate in enumerate(rates):
                try:
                    matrix[r_idx, c_idx] = all_vals[(model, mech, rate)]
                except KeyError:
                    pass
        im = ax.imshow(matrix, aspect="auto", vmin=vmin, vmax=vmax, cmap="RdYlGn_r")
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
        plt.colorbar(im, ax=ax, label="MAE (hours)")

    fig.suptitle("Mean MAE Heatmap", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, out_dir / "heatmap.png")


def plot_seed_variance(df: pd.DataFrame, out_dir: Path) -> None:
    """Strip plot showing per-seed MAE spread across all conditions."""
    df = df.copy()
    df["condition"] = df["model.type"].str.upper() + " | " + df["missingness.mechanism"] + " " + (df["missingness.rate"] * 100).astype(int).astype(str) + "%"
    conditions = df.groupby("condition")["mae"].mean().sort_values().index.tolist()

    fig, ax = plt.subplots(figsize=(8, max(5, len(conditions) * 0.35)))
    for i, cond in enumerate(conditions):
        vals = df[df["condition"] == cond]["mae"].values
        ax.scatter(vals, [i] * len(vals), s=40, zorder=3,
                   color=MODEL_COLORS.get(cond.split(" | ")[0].lower(), "grey"), alpha=0.8)
        ax.plot([vals.min(), vals.max()], [i, i], color="grey", linewidth=1, zorder=2)
    ax.set_yticks(range(len(conditions)))
    ax.set_yticklabels(conditions, fontsize=8)
    ax.set_xlabel("MAE (hours)")
    ax.set_title("Per-Seed MAE Spread by Condition", fontsize=12, fontweight="bold")
    ax.grid(axis="x", alpha=0.4)
    plt.tight_layout()
    _save(fig, out_dir / "seed_variance.png")


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=REPO_ROOT / "results" / "sweep" / "summary.csv")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    out_dir = args.out or args.csv.parent

    print(f"Loading {args.csv} ...")
    df = load(args.csv)
    print(f"  {len(df)} successful runs  ({df['model.type'].nunique()} models, "
          f"{df['missingness.mechanism'].nunique()} mechanisms, "
          f"{df['missingness.rate'].nunique()} rates, "
          f"{df['seed'].nunique()} seeds)\n")

    print("Generating plots ...")
    plot_mae_vs_rate(df, out_dir)
    plot_model_comparison(df, out_dir)
    plot_heatmap(df, out_dir)
    plot_seed_variance(df, out_dir)
    print(f"\nDone. All plots → {out_dir}/")


if __name__ == "__main__":
    main()
