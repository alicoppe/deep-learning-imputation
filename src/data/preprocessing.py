import os
from pathlib import Path

import numpy as np
import pandas as pd

NUMERIC_COLS = ["temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp", "pain", "acuity"]
FEASIBLE_RANGES = {
    "temperature": (96, 104), # Fahrenheit
    "heartrate": (20, 300),
    "resprate": (5, 80),
    "o2sat": (70, 100),
    "sbp": (50, 250),
    "dbp": (30, 150),
    "acuity": (1, 5),
}


def _find_repo_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "pyproject.toml").exists():
            return p
    return start


def _resolve_data_root(root: str | Path) -> Path:
    root_path = Path(root)
    if root_path.is_absolute():
        return root_path
    return (_find_repo_root(Path(__file__).resolve()) / root_path).resolve()


ROOT = _resolve_data_root(os.environ["MIMIC_ED_DATA_PATH"])

def load_processed_ed_data(root: str | Path = ROOT) -> pd.DataFrame:
    root_path = _resolve_data_root(root)
    triage = pd.read_csv(root_path / "triage.csv")
    edstays = pd.read_csv(root_path / "edstays.csv")

    triage[NUMERIC_COLS] = triage[NUMERIC_COLS].apply(pd.to_numeric, errors="coerce")

    for col, (low, high) in FEASIBLE_RANGES.items():
        out_of_range = triage[col].notna() & (
            (triage[col] < low) | (triage[col] > high)
        )
        triage.loc[out_of_range, col] = np.nan

    triage_clean = triage.copy()
    triage_dedup = (
        triage_clean.sort_values("stay_id")
        .drop_duplicates(subset="stay_id", keep="first")
        .reset_index(drop=True)
    )

    ed_cols = [
        "stay_id",
        "subject_id",
        "hadm_id",
        "intime",
        "outtime",
        "gender",
        "race",
        "arrival_transport",
        "disposition",
    ]

    ed_triage = edstays[ed_cols].merge(triage_dedup, on="stay_id", how="inner", suffixes=("_stay", ""))
    return ed_triage
