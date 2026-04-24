"""Deep imputation wrappers - GAIN complete, MIWAE/not-MIWAE/CSDI/ReMasker stubs.

Each exposes fit_transform(X_train) / transform(X_test) so it plugs into
pipeline._build_imputer() like sklearn imputers.

third_party/ must contain cloned author repos (see THIRD_PARTY_COMMITS.txt).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
THIRD_PARTY = REPO_ROOT / "third_party"


def build_deep_imputer(method: str, imp_cfg: dict, seed: int):
    kwargs = {k: v for k, v in imp_cfg.items() if k != "method"}
    kwargs.setdefault("seed", seed)
    if method == "gain":     return GAINImputer(**kwargs)
    if method == "miwae":    return MIWAEImputer(**kwargs)
    if method == "notmiwae": return NotMIWAEImputer(**kwargs)
    if method == "csdi":     return CSDIImputer(**kwargs)
    if method == "remasker": return ReMaskerImputer(**kwargs)
    raise ValueError(f"Unknown deep imputer: {method}")


class GAINImputer:
    """Yoon, Jordon, van der Schaar (ICML 2018)."""

    def __init__(self, batch_size=128, hint_rate=0.9, alpha=100.0,
                 iterations=10000, seed=42):
        self.batch_size = batch_size
        self.hint_rate = hint_rate
        self.alpha = alpha
        self.iterations = iterations
        self.seed = seed
        self._X_train = None

    def fit_transform(self, X):
        sys.path.insert(0, str(THIRD_PARTY / "GAIN"))
        from gain import gain
        self._X_train = X.copy()
        params = dict(batch_size=self.batch_size, hint_rate=self.hint_rate,
                      alpha=self.alpha, iterations=self.iterations)
        return gain(X, params)

    def transform(self, X):
        # GAIN's authors' code has no separate transform — concatenate
        # train+test and return the test slice after re-imputation.
        sys.path.insert(0, str(THIRD_PARTY / "GAIN"))
        from gain import gain
        combined = np.vstack([self._X_train, X])
        params = dict(batch_size=self.batch_size, hint_rate=self.hint_rate,
                      alpha=self.alpha, iterations=self.iterations)
        imputed = gain(combined, params)
        return imputed[len(self._X_train):]


class MIWAEImputer:
    """Mattei & Frellsen (ICML 2019). Mika implements post-exams."""
    def __init__(self, **kwargs): self.kwargs = kwargs
    def fit_transform(self, X): raise NotImplementedError("MIWAE wrapper pending")
    def transform(self, X): raise NotImplementedError


class NotMIWAEImputer:
    """Ipsen, Mattei, Frellsen (ICLR 2021)."""
    def __init__(self, **kwargs): self.kwargs = kwargs
    def fit_transform(self, X): raise NotImplementedError("not-MIWAE wrapper pending")
    def transform(self, X): raise NotImplementedError


class CSDIImputer:
    """Tashiro, Song, Song, Ermon (NeurIPS 2021)."""
    def __init__(self, **kwargs): self.kwargs = kwargs
    def fit_transform(self, X): raise NotImplementedError("CSDI wrapper pending")
    def transform(self, X): raise NotImplementedError


class ReMaskerImputer:
    """Du, Melis, Wang (ICLR 2024)."""
    def __init__(self, **kwargs): self.kwargs = kwargs
    def fit_transform(self, X): raise NotImplementedError("ReMasker wrapper pending")
    def transform(self, X): raise NotImplementedError
