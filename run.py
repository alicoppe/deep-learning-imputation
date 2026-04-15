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

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run.py <config.yaml>")
        sys.exit(1)

    config_path = Path(sys.argv[1])
    config = yaml.safe_load(config_path.read_text())

    if "sweep" in config:
        run_sweep(config)
    else:
        run(config)
