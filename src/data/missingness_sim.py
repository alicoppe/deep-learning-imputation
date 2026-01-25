import pandas as pd
import numpy as np
from enum import Enum

ID_COLS = {"stay_id", "subject_id", "hadm_id", "subject_id_stay"}

class MissingnessMechanism(Enum):
    MCAR = "MCAR"
    MAR = "MAR"
    MNAR = "MNAR"


DriversMap = dict[str, list[str]]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _zscore(s: pd.Series) -> np.ndarray:
    x = s.to_numpy(dtype=float, copy=True)
    mu = np.nanmean(x)
    sd = np.nanstd(x)
    if not np.isfinite(sd) or sd == 0:
        return np.zeros_like(x, dtype=float)
    return (np.nan_to_num(x, nan=mu) - mu) / sd


def _calibrate_intercept(score: np.ndarray, eligible: np.ndarray, target_rate: float) -> float:
    if eligible.sum() == 0:
        return 0.0

    lo, hi = -20.0, 20.0
    for _ in range(50):
        mid = (lo + hi) / 2.0
        p = _sigmoid(mid + score[eligible])
        m = float(p.mean())
        if m < target_rate:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0

def simulate_mar_missingness_mask(
    data: pd.DataFrame,
    rate: float,
    seed: int | None = None,
    drivers_map: DriversMap | None = None,
    strength: float = 1.0,
) -> pd.DataFrame:
    """Simulate MAR missingness and return a boolean mask (False = missing)."""
    if not (0 <= rate <= 1): raise ValueError("rate must be between 0 and 1")

    rng = np.random.default_rng(seed)
    mask = pd.DataFrame(True, index=data.index, columns=data.columns)

    numeric_cols = [
        c for c in data.select_dtypes(include=[np.number]).columns if c not in ID_COLS
    ]
    has_acuity_driver = "acuity" in numeric_cols

    if drivers_map is None:
        assert has_acuity_driver, (
            "MAR requires either drivers_map to be provided or a numeric 'acuity' column."
        )
        targets = [c for c in numeric_cols if c != "acuity"]
        drivers_map = {tgt: ["acuity"] for tgt in targets}
    else:
        targets = numeric_cols

    for tgt in targets:
        assert tgt in drivers_map, (
            f"drivers_map must include an entry for target '{tgt}'."
        )

        eligible = data[tgt].notna().to_numpy()
        if eligible.sum() == 0:
            continue

        drivers = [c for c in drivers_map[tgt] if c in data.columns and c != tgt]
        assert len(drivers) > 0, (
            "Ensure drivers_map lists at least one valid driver column."
        )

        score = np.zeros(len(data), dtype=float)
        for drv in drivers:
            z = _zscore(data[drv])
            w = -1.0 if drv == "acuity" else 1.0
            score += w * z

        score *= float(strength)
        alpha = _calibrate_intercept(score, eligible, rate)
        p = _sigmoid(alpha + score)

        u = rng.random(len(data))
        drop = eligible & (u < p)
        mask.loc[data.index[drop], tgt] = False
    mask &= data.notna()
    return mask

def simulate_mnar_missingness_mask(data: pd.DataFrame, rate: float, seed: int = None, strength: float = 1.0, targets = None) -> pd.DataFrame:
    """Simulate MNAR missingness (self-masking) and return a boolean mask."""
    if not (0 <= rate <= 1):
        raise ValueError("rate must be between 0 and 1")

    rng = np.random.default_rng(seed)
    mask = pd.DataFrame(True, index=data.index, columns=data.columns)

    numeric_cols = [c for c in data.select_dtypes(include=[np.number]).columns if c not in ID_COLS]
    if targets is None:
        targets = numeric_cols

    for tgt in targets:
        if tgt not in data.columns:
            continue

        eligible = data[tgt].notna().to_numpy()
        if eligible.sum() == 0:
            continue

        z = _zscore(data[tgt])
        score = float(strength) * z

        alpha = _calibrate_intercept(score, eligible, rate)
        p = _sigmoid(alpha + score)

        u = rng.random(len(data))
        drop = eligible & (u < p)
        mask.loc[data.index[drop], tgt] = False

    mask &= data.notna()
    return mask


def simulate_missingness(
    data: pd.DataFrame,
    mechanism: MissingnessMechanism,
    rate: float,
    seed: int | None = None,
    *,
    drivers_map: DriversMap | None = None,
    strength: float = 1.0,
) -> pd.DataFrame:
    """Return a boolean mask (False = missing) for the requested mechanism."""
    if not (0 <= rate <= 1): raise ValueError("rate must be between 0 and 1")

    mask = pd.DataFrame(True, index=data.index, columns=data.columns)
    numeric_cols = [c for c in data.select_dtypes(include=[np.number]).columns if c not in ID_COLS]

    match mechanism:
        case MissingnessMechanism.MCAR:
            rng = np.random.default_rng(seed)
            eligible = data.notna().to_numpy()
            u = rng.random(size=eligible.shape)

            submask = np.ones(eligible.shape, dtype=bool)
            submask[eligible] = u[eligible] > rate
            mask.loc[:, numeric_cols] = pd.DataFrame(
                submask, index=data.index, columns=data.columns
            ).loc[:, numeric_cols]
        case MissingnessMechanism.MAR:
            mask = simulate_mar_missingness_mask(
                data, rate, seed, drivers_map=drivers_map, strength=strength
            )
        case MissingnessMechanism.MNAR:
            mask = simulate_mnar_missingness_mask(data, rate, seed, strength=strength)
        case _:
            raise ValueError("Unsupported missingness mechanism")
    mask &= data.notna()
    return mask
