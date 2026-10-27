"""LatentAlign: iterative discovery of subpopulations along latent directions.

This module implements Algorithm 1 of the paper:

    while max(p) >= tau:
        v  <- GetLatentDirection(S)          # inverse-frequency weighted mean (Eq. 1)
        p  <- E . v                           # projection onto the direction
        i  <- argmax_{k not in S} p_k         # most aligned embedding (Eq. 2)
        S  <- S U {i}

Two stopping rules are provided:

* ``grow_threshold``  -- the paper's rule: stop when the best candidate projects
  below ``tau`` (used for Tab. 1, the CelebA visualisations and the
  sensitivity analysis of the supplementary material).
* ``grow_fixed_size`` -- add exactly ``n`` samples (used to obtain groups of a
  controlled size, 50/75/100/125, for the parameter sweeps of Figs. 4 and 5).

All embeddings are assumed to be L2-normalised (rows of ``E``), so the
projection equals cosine similarity.
"""
from __future__ import annotations

from typing import Iterable, List, Sequence

import numpy as np

__all__ = [
    "get_direction",
    "grow_threshold",
    "grow_fixed_size",
    "grow_threshold_batched",
    "grow_many",
]


def get_direction(id_labels: np.ndarray, embeddings: np.ndarray) -> np.ndarray:
    """Inverse-frequency weighted latent direction of a group (Eq. 1).

    Every identity contributes with the same total weight, so that identities
    with many images do not dominate the direction.

    Parameters
    ----------
    id_labels : (n,) array of identity labels of the group members.
    embeddings : (n, d) array of the (L2-normalised) embeddings of the members.
    """
    unique_labels, integer_labels, counts = np.unique(
        id_labels, return_inverse=True, return_counts=True
    )
    weights = 1.0 / counts[integer_labels]
    v = weights.astype(np.float32) @ embeddings
    # The weights sum to the number of unique identities C.
    v /= unique_labels.size
    return v / np.linalg.norm(v)


def _best_candidate(proj: np.ndarray, members: Sequence[int]) -> int:
    """Index of the largest projection among non-members."""
    masked = proj.copy()
    masked[np.asarray(members, dtype=np.int64)] = -np.inf
    return int(np.argmax(masked))


def grow_threshold(
    seed: Iterable[int],
    embeddings: np.ndarray,
    ids: np.ndarray,
    tau: float = 0.3,
    max_iter: int = 1000,
    direction_fn=get_direction,
) -> List[int]:
    """Grow ``seed`` until the most aligned candidate projects below ``tau``.

    ``max_iter`` bounds the number of added samples (the original
    implementation used 1000, i.e. a group can grow by at most 1000 images).
    ``direction_fn(ids_subset, embeddings_subset)`` can be replaced, e.g. by
    the weighted-SVD baseline.
    """
    elements = [int(i) for i in seed]
    for _ in range(max_iter):
        v = direction_fn(ids[elements], embeddings[elements])
        proj = embeddings @ v
        i = _best_candidate(proj, elements)
        if proj[i] < tau:
            return elements
        elements.append(i)
    return elements


def grow_fixed_size(
    seed: Iterable[int],
    embeddings: np.ndarray,
    ids: np.ndarray,
    n: int,
    direction_fn=get_direction,
) -> List[int]:
    """Add exactly ``n`` samples to ``seed`` following the latent direction."""
    elements = [int(i) for i in seed]
    for _ in range(n):
        v = direction_fn(ids[elements], embeddings[elements])
        proj = embeddings @ v
        elements.append(_best_candidate(proj, elements))
    return elements


def grow_many(groups, fn, n_jobs: int = -1, **kwargs):
    """Apply ``fn(group, **kwargs)`` to every group in parallel (joblib/loky)."""
    from joblib import Parallel, delayed

    return Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(fn)(g, **kwargs) for g in groups
    )


def grow_threshold_batched(
    seeds,
    embeddings: np.ndarray,
    ids: np.ndarray,
    tau: float = 0.3,
    max_iter: int = 1000,
    batch: int = 256,
    n_max: int = None,
):
    """Same result as ``[grow_threshold(s, embeddings, ids, tau) for s in seeds]``
    (or ``grow_fixed_size`` with ``n_max`` and ``tau=-inf``), but all groups of a
    batch are grown in lock-step so the projections are one matrix product
    per step. Much faster on large datasets (e.g. CelebA)."""
    seeds = [[int(i) for i in s] for s in seeds]
    out = [None] * len(seeds)
    steps = max_iter if n_max is None else n_max
    for b0 in range(0, len(seeds), batch):
        groups = [list(s) for s in seeds[b0:b0 + batch]]
        active = list(range(len(groups)))
        for _ in range(steps):
            if not active:
                break
            V = np.stack([get_direction(ids[groups[j]], embeddings[groups[j]]) for j in active], axis=1)
            P = embeddings @ V  # (N, n_active)
            for c, j in enumerate(active):
                P[groups[j], c] = -np.inf
            best = np.argmax(P, axis=0)
            still = []
            for c, j in enumerate(active):
                i = int(best[c])
                if P[i, c] < tau:
                    continue
                groups[j].append(i)
                still.append(j)
            active = still
        out[b0:b0 + len(groups)] = groups
    return out
