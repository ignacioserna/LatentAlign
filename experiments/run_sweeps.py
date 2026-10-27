"""Parameter sweeps on RFW: every method, every backbone, all parameter values.

Writes one SQLite file per (method, backbone) to ``results/db`` with every
discovered group, its members and its semantic distance. These files are the
input of Figs. 4-5 (``fig4_fig5.py``), the LatentAlign row of Tab. 3
(``table3_latentalign_fmr.py``) and Supp. Figs. 4-5 (``supp_fig4_fig5_sensitivity.py``).

Methods (``--methods``)
-----------------------
latentalign   k-means++ seeds (k in 200..1000), fixed-size growth to 50/75/100/125
weighted_svd  same seeds and sizes, weighted-SVD direction
nn            same seeds, 50/75/100/125 nearest neighbours
kmeans        mini-batch k-means (k in 200..1000), groups passing the bias filter
kmeans_all    mini-batch k-means, all clusters (used for Tab. 1)
skmeans       spherical k-means (k in 200..1000), groups passing the bias filter
dbscan        DBSCAN (eps x min_samples grid), clusters with >= 20 identities
graph_init    LatentAlign with similarity-graph seeds (Supp. Fig. 4)
random_init   LatentAlign with random seeds (Supp. Fig. 5)

"bias filter": >= 20 identities and mean impostor similarity > 0.1.

Example
-------
    python -m experiments.run_sweeps --methods latentalign nn kmeans skmeans dbscan
"""
from __future__ import annotations

import time

import numpy as np

from experiments._common import base_parser, db_path
from latentalign import grow_fixed_size, grow_many, grow_threshold, paper_seeds
from latentalign.baselines import (
    dbscan_groups,
    kmeans_groups,
    nearest_neighbour_groups,
    spherical_kmeans_groups,
    svd_direction,
)
from latentalign.data import load_rfw
from latentalign.metrics import has_min_identities, is_biased_group, semantic_distance
from latentalign.results_db import init_db, insert_groups
from latentalign.seeding import graph_seeds, random_seeds

K_VALUES = [200, 400, 500, 700, 1000]
SIZE_STEPS = [49, 25, 25, 25]            # seed (1) + 49 = 50, then 75, 100, 125
NN_SIZES = [50, 75, 100, 125]
DBSCAN_EPS = [1.07, 1.08, 1.09, 1.1, 1.12, 1.13]
DBSCAN_MIN_SAMPLES = [2, 3, 4, 5]
GRAPH_THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
RANDOM_SEED_SIZES = [2, 3, 4, 5, 6]
TAUS = [0.4, 0.375, 0.35, 0.3]          # applied sequentially (each continues the previous groups)


def compactness(ds, groups):
    return [semantic_distance(ds.attributes, g, scale=ds.attribute_scale) for g in groups]


def _safe_svd_grow(seed, embeddings, ids, n):
    try:
        return grow_fixed_size(seed, embeddings, ids, n, direction_fn=svd_direction)
    except Exception as e:  # ARPACK may fail on degenerate groups
        print(f"  weighted-SVD failed for seed {seed[:3]}...: {e}")
        return None


def sweep_fixed_size(ds, model, method, out_db, n_jobs):
    conn = init_db(out_db, {"k": int}, fresh=True)
    for k in K_VALUES:
        groups = [[int(i)] for i in paper_seeds("rfw", model, k, ds.embeddings)]
        for n in SIZE_STEPS:
            if method == "latentalign":
                groups = grow_many(groups, grow_fixed_size, n_jobs=n_jobs,
                                   embeddings=ds.embeddings, ids=ds.ids, n=n)
            else:
                groups = grow_many(groups, _safe_svd_grow, n_jobs=n_jobs,
                                   embeddings=ds.embeddings, ids=ds.ids, n=n)
                groups = [g for g in groups if g is not None]
            insert_groups(conn, model, {"k": k}, groups, compactness(ds, groups))
            print(f"{method} {model} k={k} size={len(groups[0])}")


def sweep_nn(ds, model, out_db):
    from sklearn.neighbors import NearestNeighbors

    conn = init_db(out_db, {"k": int, "n_neighbors": int}, fresh=True)
    nn_model = NearestNeighbors(n_neighbors=max(NN_SIZES)).fit(ds.embeddings)
    for k in K_VALUES:
        seeds = paper_seeds("rfw", model, k, ds.embeddings)
        for n in NN_SIZES:
            groups = nearest_neighbour_groups(ds.embeddings, seeds, n, nn_model=nn_model)
            insert_groups(conn, model, {"k": k, "n_neighbors": n}, groups, compactness(ds, groups))
            print(f"nn {model} k={k} n={n}")


def sweep_kmeans(ds, model, out_db, spherical=False, filter_biased=True):
    conn = init_db(out_db, {"k": int}, fresh=True)
    for k in K_VALUES:
        if spherical:
            groups = spherical_kmeans_groups(ds.embeddings, k, n_iters=300, seed=42)
        else:
            groups = kmeans_groups(ds.embeddings, k, random_state=42, batch_size=16384)
        if filter_biased:
            groups = [g for g in groups if is_biased_group(ds.embeddings, ds.ids, g)]
        if not groups:
            continue
        insert_groups(conn, model, {"k": k}, groups, compactness(ds, groups))
        print(f"{'skmeans' if spherical else 'kmeans'} {model} k={k}: {len(groups)} groups")


def sweep_dbscan(ds, model, out_db):
    conn = init_db(out_db, {"eps": float, "min_samples": int}, fresh=True)
    for eps in DBSCAN_EPS:
        for ms in DBSCAN_MIN_SAMPLES:
            groups = [g for g in dbscan_groups(ds.embeddings, eps, ms) if has_min_identities(ds.ids, g)]
            insert_groups(conn, model, {"eps": eps, "min_samples": ms}, groups, compactness(ds, groups))
            print(f"dbscan {model} eps={eps} min_samples={ms}: {len(groups)} groups")


def sweep_init(ds, model, out_db, kind, n_jobs, seed=0):
    conn = init_db(out_db, {"tau": float, "threshold": float}, fresh=True)
    values = GRAPH_THRESHOLDS if kind == "graph" else RANDOM_SEED_SIZES
    for v in values:
        groups = graph_seeds(ds.embeddings, v) if kind == "graph" else random_seeds(len(ds), v, 1000, seed=seed + v)
        for tau in TAUS:
            groups = grow_many(groups, grow_threshold, n_jobs=n_jobs, embeddings=ds.embeddings, ids=ds.ids, tau=tau)
            biased = [g for g in groups if is_biased_group(ds.embeddings, ds.ids, g)]
            if not biased:
                continue
            insert_groups(conn, model, {"tau": float(v), "threshold": float(tau)}, biased, compactness(ds, biased))
            print(f"{kind}-init {model} {v} tau={tau}: {len(biased)} groups")


def main():
    p = base_parser(__doc__)
    p.add_argument("--methods", nargs="+",
                   default=["latentalign", "weighted_svd", "nn", "kmeans", "kmeans_all", "skmeans", "dbscan"],
                   choices=["latentalign", "weighted_svd", "nn", "kmeans", "kmeans_all", "skmeans", "dbscan",
                            "graph_init", "random_init"])
    p.add_argument("--db-dir", default=None)
    p.add_argument("--k-values", type=int, nargs="+", default=K_VALUES, help="number of k-means++ seeds / clusters")
    p.add_argument("--dbscan-eps", type=float, nargs="+", default=DBSCAN_EPS)
    args = p.parse_args()
    K_VALUES[:] = args.k_values
    DBSCAN_EPS[:] = args.dbscan_eps

    for model in args.models:
        ds = load_rfw(model, args.data_root, labels="mode")
        for method in args.methods:
            t0 = time.time()
            out = db_path(method, model, args.db_dir)
            if method in ("latentalign", "weighted_svd"):
                sweep_fixed_size(ds, model, method, out, args.n_jobs)
            elif method == "nn":
                sweep_nn(ds, model, out)
            elif method in ("kmeans", "kmeans_all"):
                sweep_kmeans(ds, model, out, filter_biased=(method == "kmeans"))
            elif method == "skmeans":
                sweep_kmeans(ds, model, out, spherical=True)
            elif method == "dbscan":
                sweep_dbscan(ds, model, out)
            elif method == "graph_init":
                sweep_init(ds, model, out, "graph", args.n_jobs)
            elif method == "random_init":
                sweep_init(ds, model, out, "random", args.n_jobs)
            print(f"==> {out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
