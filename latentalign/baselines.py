"""Baselines of Sec. 4.3: nearest-neighbour search, k-means, spherical k-means,
DBSCAN and weighted-SVD (same iterative alignment as LatentAlign but with the
first right-singular vector of the inverse-frequency weighted data as direction).
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

__all__ = [
    "nearest_neighbour_groups",
    "kmeans_groups",
    "spherical_kmeans",
    "spherical_kmeans_groups",
    "dbscan_groups",
    "inverse_frequency_weighted_svd",
    "svd_direction",
    "labels_to_groups",
]


def labels_to_groups(labels: np.ndarray, ignore: Sequence[int] = (-1,)) -> List[np.ndarray]:
    """Convert a label vector into a list of index arrays (one per label)."""
    return [np.where(labels == c)[0] for c in np.unique(labels) if c not in ignore]


# ----------------------------------------------------------------------------
# Nearest neighbours
# ----------------------------------------------------------------------------
def nearest_neighbour_groups(
    embeddings: np.ndarray, seeds: Sequence[int], n_neighbors: int, nn_model=None
) -> List[np.ndarray]:
    """The ``n_neighbors`` nearest neighbours (Euclidean, i.e. cosine for
    L2-normalised embeddings) of every seed, the seed itself included."""
    from sklearn.neighbors import NearestNeighbors

    if nn_model is None:
        nn_model = NearestNeighbors(n_neighbors=n_neighbors).fit(embeddings)
    _, idx = nn_model.kneighbors(embeddings[np.asarray(seeds)], n_neighbors=n_neighbors)
    return [row for row in idx]


# ----------------------------------------------------------------------------
# k-means / spherical k-means / DBSCAN
# ----------------------------------------------------------------------------
def kmeans_groups(
    embeddings: np.ndarray, k: int, random_state: int = 42, batch_size: int = 16384
) -> List[np.ndarray]:
    """Mini-batch k-means (k-means++ initialisation) clusters."""
    from sklearn.cluster import MiniBatchKMeans

    km = MiniBatchKMeans(n_clusters=k, random_state=random_state, batch_size=batch_size)
    km.fit(embeddings)
    return labels_to_groups(km.labels_)


def spherical_kmeans(
    X,
    n_clusters: int = 100,
    n_iters: int = 100,
    tol: float = 1e-4,
    device: Optional[str] = None,
    seed: Optional[int] = None,
):
    """Spherical k-means with a k-means++-style initialisation (PyTorch).

    Returns ``(labels, centroids)`` as NumPy arrays. ``seed`` fixes the random
    initialisation (the original experiments did not fix it).
    """
    import torch

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    gen = torch.Generator(device=device)
    if seed is not None:
        gen.manual_seed(seed)
    else:
        gen.seed()
    X = torch.as_tensor(X).to(device)
    X = X / (X.norm(dim=1, keepdim=True) + 1e-8)
    n_samples, n_features = X.shape

    # --- k-means++ initialisation -------------------------------------------
    centroids = torch.empty(n_clusters, n_features, device=device, dtype=X.dtype)
    idx = torch.randint(0, n_samples, (1,), device=device, generator=gen)
    centroids[0] = X[idx]
    closest_dist_sq = torch.clamp(1 - (X @ centroids[0]) ** 2, min=0.0)
    for c in range(1, n_clusters):
        probs = torch.clamp(closest_dist_sq / (closest_dist_sq.sum() + 1e-8), min=0.0)
        if probs.sum() == 0:
            next_idx = torch.randint(0, n_samples, (1,), device=device, generator=gen)
        else:
            next_idx = torch.multinomial(probs, 1, generator=gen)
        centroids[c] = X[next_idx]
        dist_new = torch.clamp(1 - (X @ centroids[c]) ** 2, min=0.0)
        closest_dist_sq = torch.minimum(closest_dist_sq, dist_new)

    # --- Lloyd iterations on the sphere --------------------------------------
    for _ in range(n_iters):
        labels = (X @ centroids.T).argmax(dim=1)
        new_centroids = torch.zeros_like(centroids)
        new_centroids.index_add_(0, labels, X)
        counts = torch.bincount(labels, minlength=n_clusters)
        empty = counts == 0
        new_centroids[empty] = centroids[empty]  # keep old centroid if empty
        new_centroids = new_centroids / (new_centroids.norm(dim=1, keepdim=True) + 1e-8)
        shift = (centroids - new_centroids).norm(dim=1).max()
        centroids = new_centroids
        if shift < tol:
            break
    return labels.cpu().numpy(), centroids.cpu().numpy()


def spherical_kmeans_groups(embeddings: np.ndarray, k: int, n_iters: int = 300, **kw):
    labels, _ = spherical_kmeans(embeddings, n_clusters=k, n_iters=n_iters, **kw)
    return labels_to_groups(labels)


def dbscan_groups(embeddings: np.ndarray, eps: float, min_samples: int) -> List[np.ndarray]:
    """DBSCAN clusters (noise label -1 removed)."""
    from sklearn.cluster import DBSCAN

    labels = DBSCAN(eps=eps, min_samples=min_samples).fit(embeddings).labels_
    return labels_to_groups(labels)


# ----------------------------------------------------------------------------
# weighted-SVD
# ----------------------------------------------------------------------------
def inverse_frequency_weighted_svd(embeddings: np.ndarray, identities: np.ndarray) -> np.ndarray:
    """First right-singular vector of the inverse-frequency weighted, centred data."""
    from scipy.sparse.linalg import svds

    unique_ids, counts = np.unique(identities, return_counts=True)
    weight_map = {uid: 1.0 / c for uid, c in zip(unique_ids, counts)}
    weights = np.array([weight_map[uid] for uid in identities])
    weights = weights * (len(identities) / np.sum(weights))
    weighted_mean = np.average(embeddings, axis=0, weights=weights)
    centred = embeddings - weighted_mean
    weighted = centred * np.sqrt(weights)[:, None]
    _, _, vt = svds(weighted, k=1, return_singular_vectors="vh", tol=1e-6)
    return vt[0]


def svd_direction(id_labels: np.ndarray, embeddings: np.ndarray) -> np.ndarray:
    """Direction function for :func:`latentalign.algorithm.grow_*` (weighted-SVD).

    A single-image group uses the embedding itself as direction.
    """
    if len(embeddings) == 1:
        return embeddings[0]
    return inverse_frequency_weighted_svd(embeddings, id_labels)
