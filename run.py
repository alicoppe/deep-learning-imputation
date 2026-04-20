"""Entry point for config-driven training runs.

Usage:
  python run.py configs/los_baseline.yaml
  python run.py configs/los_sweep.yaml
"""

from __future__ import annotations

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


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run.py <config.yaml>")
        sys.exit(1)

    config_path = Path(sys.argv[1])
    config = yaml.safe_load(config_path.read_text())

    if "runs" in config:
        for sub_path in config["runs"]:
            sub_config = yaml.safe_load(Path(sub_path).read_text())
            if "sweep" in sub_config:
                run_sweep(sub_config)
            else:
                run(sub_config)
        if "plots" in config:
            _run_plot_scripts(config["plots"])
    elif "sweep" in config:
        run_sweep(config)
    else:
        run(config)
