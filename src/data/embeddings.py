"""Shared Bio_ClinicalBERT embedding loader with per-task caching."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())


def load_chiefcomplaint_embeddings(df: pd.DataFrame, cache_name: str = "chiefcomplaint_embeddings.npy") -> pd.DataFrame:
    """Encode chief complaint text via Bio_ClinicalBERT, caching the result.

    Args:
        df: DataFrame with a 'chiefcomplaint' column. Row order must be stable.
        cache_name: Filename for the .npy cache under data/processed/.

    Returns:
        DataFrame with columns cc_0 … cc_767 plus chiefcomplaint_missing flag.
    """
    cache = REPO_ROOT / "data" / "processed" / cache_name
    cache.parent.mkdir(parents=True, exist_ok=True)

    cc_missing = df["chiefcomplaint"].isna()
    print(f"  Chief complaint: {cc_missing.mean()*100:.1f}% missing ({cc_missing.sum():,} rows)")

    if cache.exists():
        print(f"  Loading cached embeddings from {cache.name} ...")
        arr = np.load(cache)
    else:
        print("  Computing Bio_ClinicalBERT embeddings (this may take several minutes) ...")
        from sentence_transformers import SentenceTransformer
        arr = SentenceTransformer("emilyalsentzer/Bio_ClinicalBERT").encode(
            df["chiefcomplaint"].fillna("").tolist(),
            batch_size=256,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        np.save(cache, arr)
        print(f"  Saved embeddings → {cache}")

    print(f"  Embedding shape: {arr.shape}\n")
    out = pd.DataFrame(arr, index=df.index, columns=[f"cc_{i}" for i in range(arr.shape[1])])
    out["chiefcomplaint_missing"] = cc_missing.astype(float)
    return out
