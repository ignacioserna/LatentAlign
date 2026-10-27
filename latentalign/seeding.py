"""Initial seed groups for LatentAlign (Sec. 3.1, "Initialization").

* ``kmeanspp_seeds``  greedy k-means++ centroids (scikit-learn), each centroid
  becomes a single-image seed. Used for every comparison of Sec. 4.3/4.4 so that
  all methods (except DBSCAN) start from identical seeds.
* ``graph_seeds``     connected components of the cosine-similarity graph
  thresholded at ``threshold`` (plus isolated nodes). Supp. Fig. 4.
* ``random_seeds``    random groups of ``size`` images (sampled with
  replacement, as in the original experiments). Supp. Fig. 5.
"""
from __future__ import annotations

import random
from typing import List, Optional

import numpy as np

__all__ = ["kmeanspp_seeds", "paper_seeds", "graph_seeds", "random_seeds"]


def kmeanspp_seeds(embeddings: np.ndarray, k: int, random_state: int = 42) -> np.ndarray:
    """Indices of ``k`` greedy k-means++ centroids."""
    from sklearn.cluster import kmeans_plusplus

    _, indices = kmeans_plusplus(embeddings, n_clusters=k, random_state=random_state)
    return indices


def graph_seeds(
    embeddings: np.ndarray, threshold: float = 0.5, block: int = 4096
) -> List[List[int]]:
    """Connected components of the graph ``cos(e_i, e_j) > threshold``.

    Images without any neighbour above the threshold are returned as
    singleton groups (after the components, as in the original code).
    The similarity matrix is processed in row blocks to keep memory bounded.
    """
    import networkx as nx

    G = nx.Graph()
    connected = np.zeros(len(embeddings), dtype=bool)
    for start in range(0, len(embeddings), block):
        sims = embeddings[start:start + block] @ embeddings.T
        rows = np.arange(start, min(start + block, len(embeddings)))
        sims[np.arange(len(rows)), rows] = 0.0  # ignore self-similarity
        r, c = np.where(sims > threshold)
        r = r + start
        connected[r] = True
        connected[c] = True
        G.add_edges_from(zip(r.tolist(), c.tolist()))
    clusters = [list(component) for component in nx.connected_components(G)]
    leafs = [[int(i)] for i in np.where(~connected)[0]]
    return clusters + leafs


def random_seeds(
    n_items: int, size: int, n_groups: int = 1000, seed: Optional[int] = None
) -> List[List[int]]:
    """``n_groups`` random groups of ``size`` indices (with replacement)."""
    rng = random.Random(seed)
    indices = list(range(n_items))
    return [rng.choices(indices, k=size) for _ in range(n_groups)]


def paper_seeds(dataset: str, model: str, k: int, embeddings: np.ndarray = None, random_state: int = 42) -> np.ndarray:
    """k-means++ seeds used in the paper.

    ``kmeanspp_seeds`` is not bit-stable across scikit-learn versions (the
    sequence diverges after a few dozen centroids), so the seeds of the RFW
    experiments are shipped in ``data/seeds/rfw_kmeanspp_seeds.json`` and
    returned when available; otherwise they are recomputed.
    """
    import json
    from pathlib import Path

    f = Path(__file__).resolve().parents[1] / "data" / "seeds" / f"{dataset.lower()}_kmeanspp_seeds.json"
    if f.exists():
        table = json.loads(f.read_text())
        if model in table and str(k) in table[model]:
            return np.asarray(table[model][str(k)], dtype=np.int64)
    if embeddings is None:
        raise ValueError(f"No stored seeds for {dataset}/{model}/k={k}; pass the embeddings to compute them")
    return kmeanspp_seeds(embeddings, k, random_state=random_state)
