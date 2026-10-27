"""Global FMR = 1e-3 operating points of every backbone.

* all pairs: every impostor pair among the 40,602 RFW images (824,183,509 pairs,
  cross-race pairs included). Used by Figs. 4-5, Tab. 2, Tab. 3 and Supp. Tabs. 5, 9.
* expanded set (``--domino-work``): the impostor pairs of the Domino pipeline
  (12,000 official RFW impostor pairs + 500,000 random within-race pairs, 511,864
  usable), i.e. what ``domino/scripts/03_verification.py --criterion far`` computes.

    python -m experiments.global_thresholds
    python -m experiments.global_thresholds --domino-work domino/work

Writes ``results/tables/global_thresholds.json``; pass one value to
``domino/scripts/03_verification.py --criterion fixed --threshold ...`` or the
file to ``experiments/table2_groups.py --threshold``.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments._common import REPO, base_parser
from latentalign.data import load_rfw, rfw_relative_name
from latentalign.metrics import global_threshold


def expanded_threshold(embeddings, work: Path, fmr=1e-3):
    pairs = pd.read_csv(work / "pairs.csv", usecols=["same", "img_id1", "img_id2"])
    images = pd.read_csv(work / "images.csv")
    order = [l.strip() for l in (REPO / "data" / "rfw_image_order.txt").read_text().split()]
    pos = {rfw_relative_name(o): i for i, o in enumerate(order)}
    mp = np.array([pos.get(rfw_relative_name(p), -1) for p in images["path"]])
    a, b = mp[pairs["img_id1"].to_numpy()], mp[pairs["img_id2"].to_numpy()]
    neg = (pairs["same"].to_numpy() == 0) & (a >= 0) & (b >= 0)
    s = np.einsum("ij,ij->i", embeddings[a[neg]], embeddings[b[neg]])
    return float(np.quantile(s, 1 - fmr, method="higher")), int(neg.sum())


def main():
    p = base_parser(__doc__)
    p.add_argument("--domino-work", default=None, help="Domino work folder (pairs.csv, images.csv)")
    p.add_argument("--fmr", type=float, default=1e-3)
    args = p.parse_args()
    res = {}
    for model in args.models:
        ds = load_rfw(model, args.data_root, labels=None)
        res[model] = {"all_pairs": global_threshold(ds.embeddings, ds.ids, args.fmr)}
        if args.domino_work:
            res[model]["expanded"], n = expanded_threshold(ds.embeddings, Path(args.domino_work), args.fmr)
        print(model, {k: round(v, 4) for k, v in res[model].items()}, flush=True)
    out = Path(args.out) / "tables"
    out.mkdir(parents=True, exist_ok=True)
    (out / "global_thresholds.json").write_text(json.dumps(res, indent=1))
    for kind in ("all_pairs", "expanded"):
        if all(kind in r for r in res.values()):
            (out / f"thresholds_{kind}.json").write_text(json.dumps({m: r[kind] for m, r in res.items()}, indent=1))
    print("Saved", out / "global_thresholds.json")


if __name__ == "__main__":
    main()
