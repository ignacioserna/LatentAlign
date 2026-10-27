"""Sec. 4.4, "Selection effects and group overlap".

How often do the identities of the most biased LatentAlign groups recur across
groups? 400 k-means++ seeds are grown with ``tau`` and the 100 groups with the
highest mean impostor similarity (>= 20 identities) are analysed.

    python -m experiments.identity_overlap --models elasticface

(The original analysis was run on ElasticFace with tau = 0.34; use
``--rank-by fmr`` to rank by FMR at the global threshold instead.)
"""
from __future__ import annotations

from collections import Counter

import numpy as np

from experiments._common import base_parser
from latentalign import grow_many, grow_threshold, paper_seeds
from latentalign.data import load_rfw
from latentalign.metrics import global_threshold, group_fmr, has_min_identities, mean_impostor_similarity


def main():
    p = base_parser(__doc__)
    p.add_argument("--tau", type=float, default=0.34)
    p.add_argument("--k", type=int, default=400)
    p.add_argument("--top", type=int, default=100)
    p.add_argument("--rank-by", choices=["impostor_similarity", "fmr"], default="impostor_similarity")
    p.set_defaults(models=["elasticface"])
    args = p.parse_args()

    for model in args.models:
        ds = load_rfw(model, args.data_root, labels=None)
        seeds = paper_seeds("rfw", model, args.k, ds.embeddings)
        groups = grow_many([[int(i)] for i in seeds], grow_threshold, n_jobs=args.n_jobs,
                           embeddings=ds.embeddings, ids=ds.ids, tau=args.tau)
        if args.rank_by == "fmr":
            T = global_threshold(ds.embeddings, ds.ids, 1e-3)
            score = [group_fmr(ds.embeddings, ds.ids, g, T) if has_min_identities(ds.ids, g) else 0 for g in groups]
        else:
            score = [mean_impostor_similarity(ds.embeddings, ds.ids, g) if has_min_identities(ds.ids, g) else 0
                     for g in groups]
        top = np.argsort(score)[-args.top:]
        counts = Counter()
        for i in top:
            counts.update(set(ds.ids[groups[i]]))
        dist = Counter(counts.values())
        total = sum(dist.values())
        print(f"== {model}: {total} unique identities in the top-{args.top} groups")
        for n_groups in sorted(dist):
            print(f"  in {n_groups:2d} group(s): {dist[n_groups]:5d} ids ({100 * dist[n_groups] / total:.2f}%)")
        print(f"  in >10 groups: {100 * sum(v for k, v in dist.items() if k > 10) / total:.2f}%")


if __name__ == "__main__":
    main()
