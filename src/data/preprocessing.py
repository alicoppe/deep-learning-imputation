import os
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# internal constants
_TRIAGE_NUMERIC_COLS = ["temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp", "pain", "acuity"]
_FEASIBLE_RANGES = {
    "temperature": (96, 104), # Fahrenheit
    "heartrate": (20, 300),
    "resprate": (5, 80),
    "o2sat": (70, 100),
    "sbp": (50, 250),
    "dbp": (30, 150),
    "acuity": (1, 5),
}

# external constants
ID_COLS      = ["stay_id", "subject_id", "hadm_id"]
NUMERIC_COLS = _TRIAGE_NUMERIC_COLS + ["stay_len"]
CAT_COLS     = ["gender", "race", "arrival_transport", "disposition"]
TEXT_COLS    = ["chiefcomplaint"]
TIME_COLS    = ["intime", "outtime"]


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
    """
    Outputs preprocessed ED + triage data

    Guarantees:
    - Stay length > 0 (outtime > intime)
    """
    root_path = _resolve_data_root(root)
    triage = pd.read_csv(root_path / "triage.csv")
    edstays = pd.read_csv(root_path / "edstays.csv")

    triage[_TRIAGE_NUMERIC_COLS] = triage[_TRIAGE_NUMERIC_COLS].apply(pd.to_numeric, errors="coerce")

    for col, (low, high) in _FEASIBLE_RANGES.items():
        out_of_range = triage[col].notna() & (
            (triage[col] < low) | (triage[col] > high)
        )
        triage.loc[out_of_range, col] = np.nan

    triage_dedup = (
        triage.sort_values("stay_id")
        .drop_duplicates(subset="stay_id", keep="first")
        .reset_index(drop=True)
    )
    print(f"Dropped {len(triage) - len(triage_dedup):,} rows: duplicate triage records for same stay_id")

    ed_cols = ["stay_id", "hadm_id", "intime", "outtime", "gender", "race", "arrival_transport", "disposition"]

    n = len(edstays)
    ed_triage = edstays[ed_cols].merge(triage_dedup, on="stay_id", how="inner")
    print(f"Dropped {n - len(ed_triage):,} rows: no matching triage record (inner join)")

    ed_triage["intime"]  = pd.to_datetime(ed_triage["intime"],  errors="raise")
    ed_triage["outtime"] = pd.to_datetime(ed_triage["outtime"], errors="raise")

    ed_triage["stay_len"] = (
        ed_triage["outtime"] - ed_triage["intime"]
    ).dt.total_seconds() / 3600.0
    n = len(ed_triage)
    ed_triage = ed_triage[ed_triage["stay_len"] > 0].reset_index(drop=True)
    print(f"Dropped {n - len(ed_triage):,} rows: non-positive stay length (outtime <= intime)")

    return ed_triage

def make_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encodes intime as cyclical sin/cos features (hour-of-day, day-of-week, month).
    Returns a new DataFrame with columns: hour_sin, hour_cos, dow_sin, dow_cos, month_sin, month_cos.
    """
    intime = pd.to_datetime(df["intime"])
    hour  = intime.dt.hour + intime.dt.minute / 60.0
    dow   = intime.dt.dayofweek
    month = intime.dt.month
    return pd.DataFrame({
        "hour_sin":  np.sin(2 * np.pi * hour / 24),
        "hour_cos":  np.cos(2 * np.pi * hour / 24),
        "dow_sin":   np.sin(2 * np.pi * dow / 7),
        "dow_cos":   np.cos(2 * np.pi * dow / 7),
        "month_sin": np.sin(2 * np.pi * (month - 1) / 12),
        "month_cos": np.cos(2 * np.pi * (month - 1) / 12),
    }, index=df.index)


if __name__ == "__main__":
    df = load_processed_ed_data()
    print(df.head())
    print(df.columns.tolist())