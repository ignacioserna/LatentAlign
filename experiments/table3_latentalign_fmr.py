"""Table 3 (LatentAlign row) and the numbers quoted in Sec. 4.5 / Supp. Tab. 9.

FMR of every 100-image LatentAlign group of the ArcFace sweep (2,800 groups:
k in {200, 400, 500, 700, 1000} k-means++ seeds, grown to exactly 100 images),
evaluated at the dataset-wide threshold where FMR = 1e-3.

    python -m experiments.table3_latentalign_fmr

Expected (ArcFace, threshold 0.2515): range 0.0136 - 0.2396, median 0.1278.
This threshold is computed over all impostor pairs of the 40,602 RFW images;
the Domino rows of Tab. 3 (code in ``domino/``) use the threshold of the
expanded 511,864-pair set (0.3227). Pass ``--threshold 0.3227`` to put both
methods at the same operating point (see docs/REPRODUCIBILITY.md, caveat 4).
"""
from __future__ import annotations

import numpy as np

from experiments._common import base_parser, db_path
from latentalign.data import load_rfw
from latentalign.metrics import global_threshold, group_fmr
from latentalign.results_db import load_groups


def main():
    p = base_parser(__doc__)
    p.add_argument("--db-dir", default=None)
    p.add_argument("--size", type=int, default=100)
    p.add_argument("--threshold", type=float, default=None,
                   help="fixed cosine threshold (default: global FMR = 1e-3 threshold over all RFW impostor pairs)")
    p.set_defaults(models=["arcface"])
    args = p.parse_args()

    for model in args.models:
        ds = load_rfw(model, args.data_root, labels=None)
        T = args.threshold if args.threshold is not None else global_threshold(ds.embeddings, ds.ids, 1e-3)
        df = load_groups(db_path("latentalign", model, args.db_dir), model)
        df["fmr"] = [group_fmr(ds.embeddings, ds.ids, g, T) for g in df["indices"]]
        sub = df[df["size"] == args.size]["fmr"].values
        print(f"{model}: threshold={T:.4f}")
        print(f"  {args.size}-image groups: n={len(sub)}  range {sub.min():.4f} – {sub.max():.4f}  "
              f"median {np.median(sub):.4f}  mean {sub.mean():.4f}")


if __name__ == "__main__":
    main()
