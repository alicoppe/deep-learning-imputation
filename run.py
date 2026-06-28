"""Entry point for config-driven training runs.

Usage:
  python run.py configs/los_baseline.yaml
  python run.py configs/los_sweep.yaml

Runtime overrides (applied to a single run, or to a sweep's `defaults`):
  python run.py configs/los_mlp.yaml --sample 10000
  python run.py configs/los_mlp.yaml --set data.subset.type=random_sample --set data.subset.n=10000
  python run.py configs/los_mlp.yaml --set seed=123 --set model.params.epochs=10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from src.training.runner import run, run_sweep


def _run_plot_scripts(scripts: list[str]) -> None:
    import subprocess
    for script in scripts:
        print(f"\n{'═' * 60}")
        print(f"  Plotting: {script}")
        print(f"{'═' * 60}\n")
        subprocess.run([sys.executable, script], check=True)


def _set_nested(d: dict, dotted_key: str, value) -> None:
    keys = dotted_key.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def _apply_overrides(config: dict, sets: list[str], sample: int | None) -> None:
    """Mutate config in place. For sweeps, overrides target `defaults`."""
    target = config.setdefault("defaults", {}) if "sweep" in config else config

    for item in sets:
        if "=" not in item:
            raise SystemExit(f"--set expects key=value, got: {item!r}")
        key, raw = item.split("=", 1)
        value = yaml.safe_load(raw)  # parse ints/floats/bools/null/lists like YAML would
        _set_nested(target, key.strip(), value)

    if sample is not None:
        _set_nested(target, "data.subset", {"type": "random_sample", "n": sample})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Config-driven training runs.")
    parser.add_argument("config", type=Path, help="Path to a run / sweep / meta YAML config.")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="Override a config value via dotted key path (repeatable).")
    parser.add_argument("--sample", type=int, default=None, metavar="N",
                        help="Shortcut: randomly subsample the cohort to N stays.")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())

    if "runs" in config:
        if args.set or args.sample is not None:
            print("Note: --set/--sample apply to each sub-config in this meta run.")
        for sub_path in config["runs"]:
            sub_config = yaml.safe_load(Path(sub_path).read_text())
            _apply_overrides(sub_config, args.set, args.sample)
            if "sweep" in sub_config:
                run_sweep(sub_config)
            else:
                run(sub_config)
        if "plots" in config:
            _run_plot_scripts(config["plots"])
    elif "sweep" in config:
        _apply_overrides(config, args.set, args.sample)
        run_sweep(config)
    else:
        _apply_overrides(config, args.set, args.sample)
        run(config)
