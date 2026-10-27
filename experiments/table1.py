"""Table 1 -- average intra-group semantic distance (lower is better).

All methods except DBSCAN start from the same greedy k-means++ seeds
(k = 400 on RFW, k = 2000 on CelebA) and are tuned to an average group size of
about 100 images.

    python -m experiments.table1 --dataset rfw
    python -m experiments.table1 --dataset celeba

Uncertainty
-----------
The script prints, for every cell, the mean, the standard error (SE), the 95%
normal half-width (1.96 SE) and, for k-means / spk-means / DBSCAN on CelebA,
the 95% percentile-bootstrap interval. The paper reports a mix of these (see
``PAPER_ERROR`` below and ``docs/REPRODUCIBILITY.md``); the "paper" column
uses the same convention as the published table so the numbers can be checked
one to one.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments._common import base_parser
from latentalign import grow_many, grow_threshold, grow_threshold_batched, paper_seeds
from latentalign.baselines import (
    dbscan_groups,
    kmeans_groups,
    nearest_neighbour_groups,
    spherical_kmeans_groups,
    svd_direction,
)
from latentalign.data import MODEL_TITLES, load_dataset
from latentalign.metrics import (
    bootstrap_mean_ci,
    has_min_identities,
    is_biased_group,
    mean_se_ci,
    semantic_distance,
)

CONFIG = {
    "rfw": {
        "k_seeds": 400,
        "nn_size": 100,
        "kmeans_k": 400,
        "kmeans_batch_size": 16384,
        # DBSCAN (eps, min_samples) closest to N ~ 100 in the sweep; None = no cluster with >= 20 ids
        "dbscan": {"arcface": (1.09, 4), "cosface": (1.12, 5), "elasticface": (1.1, 5), "partialFC": None},
        "dbscan_size_range": None,
        "latentalign_tau": {"arcface": 0.34, "cosface": 0.34, "elasticface": 0.34, "partialFC": 0.28},  # PartialFC: see docs/REPRODUCIBILITY.md
        "wsvd_tau": {"arcface": 0.28, "cosface": 0.28, "elasticface": 0.28, "partialFC": 0.28},         # see docs/REPRODUCIBILITY.md
        "min_labelled": 2,
    },
    "celeba": {
        "k_seeds": 2000,
        "nn_size": 100,
        "kmeans_k": 2000,
        "kmeans_batch_size": 16384,   # 16384 gives 7.62 on ArcFace (paper: 7.73)
        "dbscan": {"arcface": (1.0, 2), "cosface": (1.1, 2), "elasticface": (1.0, 2), "partialFC": (1.1, 2)},  # corrected backbone order (docs, caveat 8)
        "dbscan_size_range": (55, 10000),
        "latentalign_tau": {"arcface": 0.41, "cosface": 0.41, "elasticface": 0.41, "partialFC": 0.41},  
        "wsvd_tau": {"arcface": 0.31, "cosface": 0.31, "elasticface": 0.31, "partialFC": 0.31},         
        "min_labelled": 2,
    },
}

# Error convention of the published table: "se" = standard error,
# "ci95" = 1.96 * SE, "boot" = half-width of the 95% bootstrap interval.
PAPER_ERROR = {
    "rfw": {"NNS": "ci95", "k-means": "ci95", "spk-means": "ci95", "DBSCAN": "ci95",
            "weighted-SVD": "se", "LatentAlign": "se"},
    "celeba": {"NNS": "se", "k-means": "boot", "spk-means": "boot", "DBSCAN": "boot",
               "weighted-SVD": "se", "LatentAlign": "se"},
}

PAPER_VALUES = {  # for a side-by-side check
    "celeba": {
                "NNS": [8.47, 8.42, 8.43, 8.45], "k-means": [7.73, 7.61, 7.73, 7.66], "spk-means": [8.06, 7.97, 8.07, 8.04],
        "DBSCAN": [6.94, 7.50, 7.17, 7.47], "weighted-SVD": [7.23, 7.17, 7.18, 7.03], "LatentAlign": [6.61, 6.55, 6.57, 6.74],
    },
    "rfw": {
        "NNS": [2.78, 2.58, 2.73, 2.63], "k-means": [3.01, 2.81, 2.96, 3.01], "spk-means": [2.65, 2.38, 2.63, None],
        "DBSCAN": [2.13, 2.29, 2.46, None], "weighted-SVD": [2.07, 2.18, 1.99, 2.16], "LatentAlign": [1.89, 1.63, 1.89, 1.73],
    },
}


def _distances(ds, groups, min_size):
    return np.array([semantic_distance(ds.attributes, g, ds.attribute_scale, min_size=min_size) for g in groups])


def _cell(values, n_boot=10000, seed=0):
    s = mean_se_ci(values)
    m, lo, hi = bootstrap_mean_ci(values, n_bootstrap=n_boot, seed=seed)
    s["boot"] = (hi - lo) / 2
    s["boot_lo"], s["boot_hi"] = lo, hi
    return s


def _safe_svd_grow(seed, embeddings, ids, tau):
    try:
        return grow_threshold(seed, embeddings, ids, tau=tau, direction_fn=svd_direction)
    except Exception as e:
        print(f"  weighted-SVD failed for seed {seed}: {e}")
        return None


def run_model(dataset, model, cfg, data_root, n_jobs, skip=()):
    ds = load_dataset(dataset, model, data_root)
    E, ids = ds.embeddings, ds.ids
    seeds = paper_seeds(dataset, model, cfg["k_seeds"], E)
    rows = {}

    def record(name, groups, min_size=cfg["min_labelled"], keep=None):
        d = _distances(ds, groups, min_size)
        sizes = np.array([len(g) for g in groups])
        if keep is not None:
            d, sizes = d[keep], sizes[keep]
        ok = ~np.isnan(d)
        rows[name] = {**_cell(d[ok]), "avg_size": float(sizes[ok].mean()) if ok.any() else np.nan}
        print(f"  {name:13s} {rows[name]['mean']:.3f}  (n={rows[name]['n']}, size={rows[name]['avg_size']:.1f})")

    if "NNS" not in skip:
        record("NNS", nearest_neighbour_groups(E, seeds, cfg["nn_size"]))
    if "k-means" not in skip:
        record("k-means", kmeans_groups(E, cfg["kmeans_k"], random_state=42, batch_size=cfg["kmeans_batch_size"]))
    if "spk-means" not in skip:
        groups = spherical_kmeans_groups(E, cfg["kmeans_k"], n_iters=300, seed=42)
        if dataset == "rfw":  # on RFW only groups passing the bias filter were kept
            groups = [g for g in groups if is_biased_group(E, ids, g)]
        else:
            groups = [g for g in groups if len(g) <= 10000]
        if groups:
            record("spk-means", groups)
    if "DBSCAN" not in skip and cfg["dbscan"].get(model):
        eps, ms = cfg["dbscan"][model]
        groups = dbscan_groups(E, eps, ms)
        if dataset == "rfw":
            groups = [g for g in groups if has_min_identities(ids, g)]
        else:
            lo, hi = cfg["dbscan_size_range"]
            n_lab = [int((~np.isnan(ds.attributes[g]).all(axis=1)).sum()) for g in groups]
            groups = [g for g, n in zip(groups, n_lab) if lo <= n <= hi]
        if groups:
            record("DBSCAN", groups)
    if "weighted-SVD" not in skip:
        groups = grow_many([[int(i)] for i in seeds], _safe_svd_grow, n_jobs=n_jobs,
                           embeddings=E, ids=ids, tau=cfg["wsvd_tau"][model])
        record("weighted-SVD", [g for g in groups if g is not None], min_size=3 if dataset == "celeba" else 2)
    if "LatentAlign" not in skip:
        groups = grow_threshold_batched([[int(i)] for i in seeds], E, ids, tau=cfg["latentalign_tau"][model])
        record("LatentAlign", groups)
    return rows


def main():
    p = base_parser(__doc__)
    p.add_argument("--dataset", choices=["rfw", "celeba"], default="rfw")
    p.add_argument("--skip", nargs="*", default=[], help="Rows to skip, e.g. DBSCAN spk-means")
    p.add_argument("--wsvd-tau", type=float, default=None, help="override the weighted-SVD tau of every backbone")
    p.add_argument("--latentalign-tau", type=float, default=None, help="override the LatentAlign tau of every backbone")
    args = p.parse_args()
    cfg = dict(CONFIG[args.dataset])
    if args.wsvd_tau is not None:
        cfg["wsvd_tau"] = {m: args.wsvd_tau for m in cfg["wsvd_tau"]}
    if args.latentalign_tau is not None:
        cfg["latentalign_tau"] = {m: args.latentalign_tau for m in cfg["latentalign_tau"]}

    results = {}
    for model in args.models:
        print(f"== {args.dataset} / {model}")
        results[model] = run_model(args.dataset, model, cfg, args.data_root, args.n_jobs, skip=set(args.skip))

    out = Path(args.out) / "tables"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"table1_{args.dataset}.json").write_text(json.dumps(results, indent=2))

    methods = ["NNS", "k-means", "spk-means", "DBSCAN", "weighted-SVD", "LatentAlign"]
    table = []
    for m in methods:
        row = {"Method": m}
        for j, model in enumerate(args.models):
            r = results[model].get(m)
            paper = PAPER_VALUES[args.dataset][m][j] if args.models == list(MODEL_TITLES) else None
            if r is None:
                row[MODEL_TITLES[model]] = "—"
                continue
            err = r[PAPER_ERROR[args.dataset][m]]
            row[MODEL_TITLES[model]] = f"{r['mean']:.2f} ± {err:.2f}" + (f"  [paper {paper}]" if paper else "")
        table.append(row)
    df = pd.DataFrame(table)
    print(df.to_string(index=False))
    df.to_csv(out / f"table1_{args.dataset}.csv", index=False)


if __name__ == "__main__":
    main()
