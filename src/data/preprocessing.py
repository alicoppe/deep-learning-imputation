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


def load_admissions(root: str | Path | None = None) -> pd.DataFrame:
    """Load MIMIC-IV hosp/admissions.csv.gz (just the columns needed for mortality)."""
    root_path = _resolve_data_root(root or os.environ["MIMIC_HOSP_DATA_PATH"])
    cols = ["hadm_id", "hospital_expire_flag"]
    return pd.read_csv(root_path / "admissions.csv.gz", usecols=cols)


def load_processed_ed_data(root: str | Path = ROOT) -> pd.DataFrame:
    """
    Outputs preprocessed ED + triage data

    Guarantees:
    - Stay length > 0 (outtime > intime)
    """
    root_path = _resolve_data_root(root)
    triage  = pd.read_csv(next(p for p in [root_path / "triage.csv.gz",  root_path / "triage.csv"]  if p.exists()))
    edstays = pd.read_csv(next(p for p in [root_path / "edstays.csv.gz", root_path / "edstays.csv"] if p.exists()))

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

    ed_cols = ["stay_id", "subject_id", "hadm_id", "intime", "outtime", "gender", "race", "arrival_transport", "disposition"]

    # subject_id also lives in triage; drop it there so the stay_id merge doesn't collide.
    triage_dedup = triage_dedup.drop(columns=["subject_id"], errors="ignore")

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

def _compute_approx_age(df: pd.DataFrame) -> pd.Series:
    """De-identified age at ED visit: anchor_age + (intime_year - anchor_year), capped at 91.

    Requires ``subject_id`` in df; merges hosp/patients to recover anchor fields.
    """
    pat_root = _resolve_data_root(os.environ["MIMIC_HOSP_DATA_PATH"])
    pat_path = next(p for p in [pat_root / "patients.csv.gz", pat_root / "patients.csv"] if p.exists())
    patients = pd.read_csv(pat_path, usecols=["subject_id", "anchor_age", "anchor_year"])
    merged = df[["subject_id"]].merge(patients, on="subject_id", how="left")
    intime_year = pd.to_datetime(df["intime"]).dt.year.to_numpy()
    age = merged["anchor_age"].to_numpy() + (intime_year - merged["anchor_year"].to_numpy())
    return pd.Series(age, index=df.index)


def _subset_mask(df: pd.DataFrame, f: dict) -> pd.Series:
    """Boolean mask (aligned to df.index) selecting rows matching a single filter spec."""
    ftype = f.get("type")

    if ftype == "acuity":
        values = {float(v) for v in f["values"]}
        return df["acuity"].isin(values)

    if ftype == "age":
        age = _compute_approx_age(df)
        mask = pd.Series(True, index=df.index)
        if f.get("min") is not None:
            mask &= age >= float(f["min"])
        if f.get("max") is not None:
            mask &= age <= float(f["max"])
        return mask

    if ftype == "disposition":
        values = {v.upper() for v in f["values"]}
        return df["disposition"].str.upper().isin(values)

    if ftype == "chief_complaint":
        import re
        pattern = "|".join(re.escape(kw) for kw in f["keywords"])
        return df["chiefcomplaint"].str.contains(pattern, case=False, na=False)

    if ftype == "icd_chapter":
        _ICD_CHAPTER_PREFIXES = {
            "cardiovascular":   "I",
            "respiratory":      "J",
            "gastrointestinal": "K",
            "musculoskeletal":  "M",
            "mental_health":    "F",
            "genitourinary":    "N",
            "injury_trauma":    ("S", "T"),
            "oncology":         "C",
        }
        chapter = f["chapter"]
        prefix = _ICD_CHAPTER_PREFIXES.get(chapter)
        if prefix is None:
            raise ValueError(f"Unknown ICD chapter '{chapter}'. Options: {list(_ICD_CHAPTER_PREFIXES)}")
        diag_path = next(p for p in [ROOT / "diagnosis.csv.gz", ROOT / "diagnosis.csv"] if p.exists())
        diag = pd.read_csv(diag_path, usecols=["stay_id", "seq_num", "icd_code", "icd_version"])
        primary = diag[(diag["seq_num"] == 1) & (diag["icd_version"] == 10)].copy()
        if isinstance(prefix, tuple):
            in_chapter = primary["icd_code"].str.strip().str[0].isin(set(prefix))
        else:
            in_chapter = primary["icd_code"].str.strip().str.startswith(prefix)
        stay_ids = set(primary.loc[in_chapter, "stay_id"])
        return df["stay_id"].isin(stay_ids)

    if ftype == "active_cancer_treatment":
        # Identify patients on active systemic cancer treatment via home med reconciliation
        # (medrecon table, recorded at triage — no outcome leakage).
        #
        # Excluded drug classes that are primarily used for non-cancer indications:
        #   Folic Acid Analogs  → methotrexate for RA / psoriasis
        #   Urea Derivatives    → hydroxyurea for sickle cell disease
        #   Purine Analogs      → azathioprine for IBD / transplant / autoimmune
        #   Dermatological antimetabolites / NSAIDs → topical treatments for actinic keratosis
        #   Mast Cell Stabilizers  → misclassified in drug taxonomy, not chemotherapy
        #   Progestins          → used for non-cancer hormonal indications
        _EXCLUDED_CLASSES = {
            "Antineoplastic - Antimetabolite - Folic Acid Analogs",
            "Antineoplastic - Antimetabolite - Urea Derivatives",
            "Antineoplastic - Antimetabolite - Purine Analogs",
            "Dermatological - Antineoplastic Antimetabolites",
            "Dermatological - Antineoplastic or Premalignant Lesions - NSAID's",
            "Antineoplastic - Mast Cell Stabilizers",
            "Antineoplastic - Progestins",
        }
        medrecon_path = next(
            p for p in [ROOT / "medrecon.csv.gz", ROOT / "medrecon.csv"] if p.exists()
        )
        medrecon = pd.read_csv(medrecon_path, usecols=["stay_id", "etcdescription"])
        desc_lower   = medrecon["etcdescription"].str.lower()
        is_antineopl = desc_lower.str.contains("antineoplast", na=False)
        is_excluded  = desc_lower.isin({c.lower() for c in _EXCLUDED_CLASSES})
        stay_ids = set(medrecon[is_antineopl & ~is_excluded]["stay_id"])
        return df["stay_id"].isin(stay_ids)

    raise ValueError(
        f"Unknown subset filter type '{ftype}'. "
        "Options: acuity, age, disposition, chief_complaint, icd_chapter, active_cancer_treatment"
    )


def apply_subset(df: pd.DataFrame, subset_cfg: dict) -> pd.DataFrame:
    """Filter the ED dataframe to a named clinical subset, then optionally subsample.

    Single filter (a ``type`` at the top level):
      type: acuity            values: [1, 2]
      type: age               min: 75   max: null     # either bound optional
      type: disposition       values: [ADMITTED]
      type: chief_complaint   keywords: [fall, weakness, confusion]
      type: icd_chapter       chapter: cardiovascular
          (cardiovascular | respiratory | gastrointestinal | musculoskeletal
           | mental_health | genitourinary | injury_trauma | oncology)
      type: active_cancer_treatment

    Composite filter (AND-combined) — e.g. Geriatric ESI 1+2:
      name: Geriatric ESI 1+2
      filters:
        - {type: age, min: 75}
        - {type: acuity, values: [1, 2]}

    Random subsample — standalone, or appended to any filter via ``sample_n``:
      type: random_sample   n: 10000   seed: 42
      # or:  filters: [...]   sample_n: 10000   seed: 42
    """
    n_before = len(df)
    subset_type = subset_cfg.get("type")

    if "filters" in subset_cfg:
        filters = subset_cfg["filters"]
    elif subset_type in (None, "random_sample"):
        filters = []
    else:
        filters = [subset_cfg]

    mask = pd.Series(True, index=df.index)
    for f in filters:
        mask &= _subset_mask(df, f)
    df = df[mask].reset_index(drop=True)

    sample_n = subset_cfg.get("n") if subset_type == "random_sample" else subset_cfg.get("sample_n")
    if sample_n is not None and len(df) > sample_n:
        seed = subset_cfg.get("seed", 42)
        df = df.sample(n=int(sample_n), random_state=seed).reset_index(drop=True)

    label = subset_cfg.get("name", subset_type or "composite")
    pct = (len(df) / n_before * 100) if n_before else 0.0
    print(f"  Subset [{label}]: {n_before:,} → {len(df):,} stays ({pct:.1f}%)")
    return df


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