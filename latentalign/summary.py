"""Worst-case summaries used in Figs. 4-5 and Supp. Figs. 4-5.

For every size (or size bin) the 100 groups with the highest FMR and,
separately, the 100 groups with the lowest semantic distance are averaged;
the band is mean +/- 1.96 * std / sqrt(n).
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Union

import numpy as np
import pandas as pd

Z95 = 1.96


def _top_k(df: pd.DataFrame, by: Union[str, List[str]], column: str, k: int, largest: bool, prefix: str):
    # equivalent to groupby(by).apply(lambda g: g.nlargest(k, column)) (resp. nsmallest)
    top = df.dropna(subset=[column]).sort_values(column, ascending=not largest, kind="stable")
    top = top.groupby(by, observed=True).head(k)
    s = top.groupby(by, observed=True).agg(
        n=(column, "count"), mean=(column, "mean"), std=(column, "std"), avg_size=("size", "mean")
    ).reset_index()
    half = Z95 * s["std"] / np.sqrt(s["n"])
    s[f"mean_{prefix}"] = s["mean"]
    s[f"mean_{prefix}_lower"] = s["mean"] - half
    s[f"mean_{prefix}_upper"] = s["mean"] + half
    s[f"{prefix}_size"] = s["avg_size"]
    return s.drop(columns=["n", "mean", "std", "avg_size"])


def worst_case_summary(
    df: pd.DataFrame,
    by: Union[str, List[str]] = "size",
    bin_width: Optional[int] = None,
    k: int = 100,
    fmr_col: str = "fmr",
    coherence_col: str = "compactness",
) -> pd.DataFrame:
    """Top-``k`` FMR and top-``k`` coherence per ``by`` (optionally binned sizes).

    * methods with fixed group sizes (LatentAlign, NN, weighted-SVD) are grouped
      by exact size;
    * methods with variable sizes (k-means, spherical k-means, DBSCAN) use bins
      of ``bin_width`` = 10 images.
    """
    df = df.copy()
    if bin_width is not None:
        df["size_bin"] = pd.cut(df["size"], bins=np.arange(0, df["size"].max() + bin_width, bin_width), right=False)
        by = "size_bin"
    out = _top_k(df, by, fmr_col, k, largest=True, prefix="fmr")
    if coherence_col in df:
        out = out.merge(_top_k(df, by, coherence_col, k, largest=False, prefix="compactness"), on=by, how="inner")
    if isinstance(by, str) or bin_width is not None:
        return out
    counts = df.groupby(by).agg(n_groups=("size", "count"), mean_size=("size", "mean")).reset_index()
    return counts.merge(out, on=by, how="inner")
