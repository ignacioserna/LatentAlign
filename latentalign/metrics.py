"""Evaluation metrics: semantic distance, verification scores and error rates.

Conventions (identical to the experiments of the paper)
------------------------------------------------------
* A *pair* is an unordered pair of two different images of a group.
  It is *genuine* if both images share the identity label and *impostor*
  otherwise.
* Scores are cosine similarities of L2-normalised embeddings.
* ``global_threshold``: the threshold at which the impostor pairs of the whole
  dataset reach the target FMR (quantile with ``method="higher"``).
* ``group_fmr``: fraction of the impostor pairs of a group whose score is
  ``>=`` the threshold.
* Semantic distance: mean pairwise L1 distance between the attribute vectors
  of the members of a group. For RFW the attributes are one-hot encoded, so
  the L1 distance is divided by two to obtain a Hamming distance over the ten
  attributes; for CelebA the 40 binary attributes are used directly.

Everything is computed group by group from the embeddings, so the N x N score
matrix (13 GB for RFW in float64) is never materialised.
"""
from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial.distance import pdist

__all__ = [
    "group_pair_scores",
    "global_threshold",
    "group_fmr",
    "mean_impostor_similarity",
    "semantic_distance",
    "eer_and_fnmr",
    "fmr_curve",
    "bootstrap_mean_ci",
    "bootstrap_group_fmr",
    "mean_se_ci",
    "is_biased_group",
    "has_min_identities",
]


# ----------------------------------------------------------------------------
# Pairs within a group
# ----------------------------------------------------------------------------
def group_pair_scores(
    embeddings: np.ndarray, ids: np.ndarray, group: Sequence[int]
) -> Tuple[np.ndarray, np.ndarray]:
    """Impostor and genuine scores of all unordered pairs of a group.

    Pairs of the same image (possible if ``group`` contains repeated indices,
    e.g. in a bootstrap sample) are excluded.
    """
    g = np.asarray(group, dtype=np.int64)
    E = embeddings[g]
    S = E @ E.T
    iu, ju = np.triu_indices(len(g), k=1)
    distinct = g[iu] != g[ju]
    same_id = ids[g][iu] == ids[g][ju]
    scores = S[iu, ju]
    impostor = scores[distinct & ~same_id]
    genuine = scores[distinct & same_id]
    return impostor, genuine


def mean_impostor_similarity(embeddings, ids, group) -> float:
    n, total, _ = _pair_block_stats(embeddings, ids, group)
    return total / n if n else np.nan


def global_threshold(
    embeddings: np.ndarray,
    ids: np.ndarray,
    target_fmr: float = 1e-3,
    block: int = 2048,
    verbose: bool = False,
) -> float:
    """Exact dataset-wide threshold at ``target_fmr``.

    Equivalent to ``np.quantile(all_impostor_scores, 1 - target_fmr,
    method="higher")`` over all impostor pairs ``i < j``, computed in blocks by
    keeping only the ``m`` largest impostor scores.
    """
    ids = np.asarray(ids)
    n = len(embeddings)
    # number of impostor pairs i<j
    _, counts = np.unique(ids, return_counts=True)
    n_pairs = n * (n - 1) // 2
    n_genuine = int(np.sum(counts * (counts - 1) // 2))
    n_imp = n_pairs - n_genuine
    q = 1.0 - target_fmr
    rank = int(np.ceil(q * (n_imp - 1)))  # index in the ascending sorted array
    m = n_imp - rank  # how many of the largest values we need
    top = np.empty(0, dtype=np.float32)
    for start in range(0, n, block):
        stop = min(start + block, n)
        S = embeddings[start:stop] @ embeddings.T
        rows = np.arange(start, stop)[:, None]
        cols = np.arange(n)[None, :]
        mask = (cols > rows) & (ids[start:stop][:, None] != ids[None, :])
        vals = S[mask].astype(np.float32)
        if vals.size > m:
            vals = np.partition(vals, vals.size - m)[-m:]
        top = np.concatenate([top, vals])
        if top.size > m:
            top = np.partition(top, top.size - m)[-m:]
        if verbose and (start // block) % 5 == 0:
            print(f"  global threshold: {stop}/{n} rows")
    return float(np.min(top))


def _pair_block_stats(embeddings, ids, group, threshold=None, block: int = 4096):
    """Streaming statistics over the impostor pairs of a (possibly huge) group:
    number of pairs, sum of scores and number of scores >= ``threshold``."""
    g = np.asarray(group, dtype=np.int64)
    E, I = embeddings[g], ids[g]
    n_pairs = n_hits = 0
    total = 0.0
    for a in range(0, len(g), block):
        for b in range(a, len(g), block):
            S = E[a:a + block] @ E[b:b + block].T
            pa, pb = np.arange(a, min(a + block, len(g)))[:, None], np.arange(b, min(b + block, len(g)))[None, :]
            m = (pb > pa) & (g[pa] != g[pb]) & (I[pa] != I[pb])
            v = S[m]
            n_pairs += v.size
            total += float(v.sum())
            if threshold is not None:
                n_hits += int((v >= threshold).sum())
    return n_pairs, total, n_hits


def group_fmr(embeddings, ids, group, threshold: float) -> float:
    """FMR of a group at a fixed (global) threshold."""
    if len(group) <= 4096:
        imp, _ = group_pair_scores(embeddings, ids, group)
        return float(np.mean(imp >= threshold)) if imp.size else np.nan
    n, _, hits = _pair_block_stats(embeddings, ids, group, threshold)
    return hits / n if n else np.nan


def is_biased_group(embeddings, ids, group, min_ids: int = 20, min_mean_impostor: float = 0.1) -> bool:
    """Filter used for the graph/random-initialised sweeps and spherical k-means:
    at least ``min_ids`` identities and mean impostor similarity above
    ``min_mean_impostor``."""
    if np.unique(ids[np.asarray(group)]).size < min_ids:
        return False
    n, total, _ = _pair_block_stats(embeddings, ids, group)
    return n > 0 and total / n > min_mean_impostor


def has_min_identities(ids, group, min_ids: int = 20) -> bool:
    return np.unique(ids[np.asarray(group)]).size >= min_ids


# ----------------------------------------------------------------------------
# Semantic coherence
# ----------------------------------------------------------------------------
def semantic_distance(X: np.ndarray, group: Sequence[int], scale: float = 1.0, min_size: int = 2) -> float:
    """Mean pairwise L1 distance of the attribute rows of a group, divided by
    ``scale`` (2 for one-hot encodings). Rows that are all-NaN (images without
    annotation) are dropped first.

    For 0/1 attributes the sum of all pairwise L1 distances equals
    ``sum_f c_f (n - c_f)`` (c_f = number of ones of feature f), which gives the
    exact value in O(n F) memory; other inputs fall back to ``pdist``.
    """
    A = X[np.asarray(group, dtype=np.int64)]
    if np.isnan(A).any():
        A = A[~np.isnan(A).all(axis=1)]
    n = len(A)
    if n < min_size:
        return np.nan
    if np.isin(A, (0.0, 1.0)).all():
        c = A.sum(axis=0)
        total = float(np.sum(c * (n - c)))
        return total / (n * (n - 1) / 2) / scale
    return float(np.mean(pdist(A, metric="cityblock")) / scale)


# ----------------------------------------------------------------------------
# Verification error rates within a group
# ----------------------------------------------------------------------------
def eer_and_fnmr(genuine, impostor, fmr_targets=(1e-2, 1e-3)) -> Dict[str, float]:
    """EER and FNMR at target FMRs from the ROC curve of a group (Supp. Tab. 4)."""
    from sklearn.metrics import roc_curve

    y_true = np.concatenate([np.ones_like(genuine), np.zeros_like(impostor)])
    y_score = np.concatenate([genuine, impostor])
    fpr, tpr, thr = roc_curve(y_true, y_score)
    fnr = 1 - tpr
    i = np.nanargmin(np.abs(fpr - fnr))
    out = {"eer": float((fpr[i] + fnr[i]) / 2), "eer_threshold": float(thr[i])}
    for t in fmr_targets:
        j = np.searchsorted(fpr, t)
        out[f"fnmr@{t:g}"] = float(fnr[j]) if j < len(fnr) else np.nan
    return out


def fmr_curve(impostor_scores, n_points: int = 500):
    """FMR as a function of the threshold (Fig. 1c, Supp. Fig. 3)."""
    s = np.asarray(impostor_scores)
    thr = np.linspace(s.min(), s.max(), n_points)
    return thr, np.array([(s >= t).mean() for t in thr])


# ----------------------------------------------------------------------------
# Uncertainty
# ----------------------------------------------------------------------------
def mean_se_ci(values, z: float = 1.96) -> Dict[str, float]:
    """Mean, standard error (population std / sqrt(n)) and z * SE."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    se = np.std(v) / np.sqrt(len(v))
    return {"n": int(len(v)), "mean": float(np.mean(v)), "std": float(np.std(v)), "se": float(se), "ci95": float(z * se)}


def bootstrap_mean_ci(values, n_bootstrap: int = 10000, seed: Optional[int] = None, alpha: float = 0.05):
    """Percentile bootstrap CI of the mean. Returns ``(mean, lower, upper)``."""
    rng = np.random.default_rng(seed)
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    means = rng.choice(v, size=(n_bootstrap, len(v)), replace=True).mean(axis=1)
    return float(np.mean(v)), float(np.percentile(means, 100 * alpha / 2)), float(np.percentile(means, 100 * (1 - alpha / 2)))


def bootstrap_group_fmr(
    embeddings, ids, group, threshold: float, n_boot: int = 1000, alpha: float = 0.05, seed: Optional[int] = None
):
    """Image-level bootstrap of the FMR of a group (Supp. Tab. 5).

    The images of the group are resampled with replacement and the FMR is
    recomputed over the impostor pairs of the resample.
    Returns ``(mean, lower, upper)``.
    """
    rng = np.random.default_rng(seed)
    g = np.asarray(group)
    vals = []
    for _ in range(n_boot):
        s = rng.choice(g, size=len(g), replace=True)
        imp, _ = group_pair_scores(embeddings, ids, s)
        vals.append(np.mean(imp >= threshold))
    return float(np.mean(vals)), float(np.percentile(vals, 100 * alpha / 2)), float(np.percentile(vals, 100 * (1 - alpha / 2)))
