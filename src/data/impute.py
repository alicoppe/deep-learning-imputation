import numpy as np
import pandas as pd

from .missingness_sim import ID_COLS


def _numeric_columns(
    data: pd.DataFrame, columns: list[str] | None, exclude_id: bool
) -> list[str]:
    numeric = [c for c in data.select_dtypes(include=[np.number]).columns]
    if exclude_id:
        numeric = [c for c in numeric if c not in ID_COLS]

    if columns is None:
        return numeric

    missing = [c for c in columns if c not in data.columns]
    if missing:
        raise ValueError(f"Columns not found in data: {missing}")

    non_numeric = [c for c in columns if c not in numeric]
    if non_numeric:
        raise ValueError(f"Columns must be numeric for simple imputation: {non_numeric}")

    return columns


def _impute_with_value(
    data: pd.DataFrame,
    value_fn,
    columns: list[str] | None = None,
    exclude_id: bool = True,
) -> pd.DataFrame:
    targets = _numeric_columns(data, columns, exclude_id)
    out = data.copy()

    for col in targets:
        s = out[col]
        if s.notna().all():
            continue

        value = value_fn(s)

        out[col] = s.fillna(value)

    return out


def mean_impute(
    data: pd.DataFrame, columns: list[str] | None = None, exclude_id: bool = True
) -> pd.DataFrame:
    """Impute missing numeric values with the column mean."""

    def _mean(s: pd.Series):
        value = float(np.nanmean(s.to_numpy(dtype=float)))
        if not np.isfinite(value):
            raise ValueError(f"Cannot compute mean for column '{s.name}' (all values missing).")
        return value

    return _impute_with_value(data, _mean, columns=columns, exclude_id=exclude_id)


def median_impute(
    data: pd.DataFrame, columns: list[str] | None = None, exclude_id: bool = True
) -> pd.DataFrame:
    """Impute missing numeric values with the column median."""

    def _median(s: pd.Series):
        value = float(np.nanmedian(s.to_numpy(dtype=float)))
        if not np.isfinite(value):
            raise ValueError(f"Cannot compute median for column '{s.name}' (all values missing).")
        return value

    return _impute_with_value(data, _median, columns=columns, exclude_id=exclude_id)


def constant_impute(
    data: pd.DataFrame,
    fill_value: float | int,
    columns: list[str] | None = None,
    exclude_id: bool = True,
) -> pd.DataFrame:
    """Impute missing numeric values with a constant."""
    return _impute_with_value(
        data, lambda s: fill_value, columns=columns, exclude_id=exclude_id
    )
