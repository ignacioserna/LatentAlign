"""Supp. Tab. 6 -- k-means vs LatentAlign on CelebA at matched group sizes.

k-means: mini-batch k-means with k in {4000, 2000, 1000} (N ~ 50, 100, 200).
LatentAlign: groups initialised with the clusters stored in
``data/celeba_legacy_init/{model}_100_CelebA_clusters.npy`` (the initial
groups of the paper) and grown with a decreasing
sequence of thresholds; the distance of all groups (>2 annotated images) is
reported after each threshold, the rows of the table are the thresholds that
give N ~ 50/100/200 (``TAU_FOR_SIZE``).

    python -m experiments.supp_table6_celeba_kmeans
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from experiments._common import REPO, base_parser
from latentalign import grow_many, grow_threshold
from latentalign.baselines import kmeans_groups, labels_to_groups
from latentalign.data import MODEL_TITLES, load_celeba
from latentalign.metrics import semantic_distance

K_FOR_SIZE = {50: 4000, 100: 2000, 200: 1000}
TAU_FOR_SIZE = {  # thresholds of Supp. Tab. 6, columns in the corrected backbone order (see docs, caveat 8)
    "arcface": {50: 0.42, 100: 0.40, 200: 0.38},
    "cosface": {50: 0.40, 100: 0.39, 200: 0.37},
    "elasticface": {50: 0.42, 100: 0.40, 200: 0.38},
    "partialFC": {50: 0.35, 100: 0.33, 200: 0.31},
}
TAU_SCHEDULE = [0.5, 0.49, 0.48, 0.47, 0.46, 0.45, 0.44, 0.43, 0.42, 0.41, 0.4, 0.39, 0.38, 0.37, 0.36, 0.35,
                0.34, 0.33, 0.32, 0.31, 0.3]


def mean_distance(ds, groups, min_size):
    d = np.array([semantic_distance(ds.attributes, g, 1.0, min_size=min_size) for g in groups])
    sizes = np.array([int((~np.isnan(ds.attributes[np.asarray(g)]).all(axis=1)).sum()) for g in groups])
    ok = ~np.isnan(d)
    return float(d[ok].mean()), float(sizes[ok].mean())


def main():
    p = base_parser(__doc__)
    args = p.parse_args()
    rows = []
    for model in args.models:
        ds = load_celeba(model, args.data_root)
        for n, k in K_FOR_SIZE.items():
            d, s = mean_distance(ds, kmeans_groups(ds.embeddings, k, random_state=42, batch_size=16384), 2)
            rows.append({"method": "k-means", "N": n, "model": model, "param": k, "distance": d, "avg_size": s})
            print(f"k-means {model} k={k}: {d:.2f} (N={s:.1f})")

        labels = np.load(REPO / "data" / "celeba_legacy_init" / f"{model}_100_CelebA_clusters.npy")
        groups = [list(g) for g in labels_to_groups(labels)]
        wanted = {tau: n for n, tau in TAU_FOR_SIZE[model].items()}
        for tau in TAU_SCHEDULE:
            groups = grow_many(groups, grow_threshold, n_jobs=args.n_jobs, embeddings=ds.embeddings, ids=ds.ids, tau=tau)
            d, s = mean_distance(ds, groups, 3)
            print(f"LatentAlign {model} tau={tau}: {d:.2f} (N={s:.1f})")
            if tau in wanted:
                rows.append({"method": "LatentAlign", "N": wanted[tau], "model": model, "param": tau,
                             "distance": d, "avg_size": s})
            if tau <= min(wanted):
                break
    df = pd.DataFrame(rows)
    out = Path(args.out) / "tables"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "supp_table6_celeba_kmeans_vs_latentalign.csv", index=False)
    piv = df.pivot_table(index=["N", "method"], columns="model", values="distance")
    print(piv.rename(columns=MODEL_TITLES).round(2).to_string())


if __name__ == "__main__":
    main()
