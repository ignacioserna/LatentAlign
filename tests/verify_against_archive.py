"""Check the repository against the stored result databases of the experiments.

1. Recomputes the semantic distance of every stored group with
   ``latentalign.metrics.semantic_distance`` + ``data/annotations`` and compares
   it with the stored ``compactness`` column.
2. Recomputes the RFW Tab. 1 cells that can be obtained from the stored groups.

    python tests/verify_against_archive.py --db-dir /path/to/result_dbs
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from latentalign.data import MODELS, rfw_attribute_matrix  # noqa: E402
from latentalign.metrics import semantic_distance  # noqa: E402
from latentalign.results_db import load_groups  # noqa: E402

TAB1 = {  # cell -> (db, query, published value)
    "NNS": ("RFW_{m}_nn_kmeans.db", "k == 400 and n_neighbors == 100", [2.78, 2.58, 2.73, 2.63]),
    "k-means": ("RFW_{m}_kmeans_all.db", "k == 400", [3.01, 2.81, 2.96, 3.01]),
    "spk-means": ("RFW_{m}_Skmeans.db", "k == 400", [2.65, 2.38, 2.63, None]),
    "DBSCAN": ("RFW_{m}_dbscan.db", {"arcface": "eps == 1.09 and min_samples == 4",
                                     "cosface": "eps == 1.12 and min_samples == 5",
                                     "elasticface": "eps == 1.1 and min_samples == 5", "partialFC": None},
               [2.13, 2.29, 2.46, None]),
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db-dir", required=True)
    p.add_argument("--sample", type=int, default=500)
    args = p.parse_args()
    names = np.array((REPO / "data" / "rfw_image_order.txt").read_text().split())
    X = rfw_attribute_matrix(names, REPO / "data" / "annotations" / "rfw" / "labels_RFW_mode.csv")

    print("1) semantic distance of stored groups")
    for db in sorted(Path(args.db_dir).glob("RFW_*.db")):
        if db.stat().st_size == 0:
            continue
        df = load_groups(db)
        sub = df.sample(min(args.sample, len(df)), random_state=0)
        mine = np.array([semantic_distance(X, g, 2.0) for g in sub["indices"]])
        print(f"  {db.name:35s} max |diff| = {np.nanmax(np.abs(mine - sub['compactness'].values)):.2e}")

    print("2) RFW Tab. 1 from stored groups (mean ± 1.96 SE  [published])")
    for row, (pattern, query, published) in TAB1.items():
        cells = []
        for j, m in enumerate(MODELS):
            q = query[m] if isinstance(query, dict) else query
            path = Path(args.db_dir) / pattern.format(m=m)
            if q is None or not path.exists() or path.stat().st_size == 0:
                cells.append("—")
                continue
            d = load_groups(path, m).query(q)["compactness"].values
            cells.append(f"{d.mean():.2f} ± {1.96 * d.std() / math.sqrt(len(d)):.2f} [{published[j]}]")
        print(f"  {row:10s} " + " | ".join(cells))


if __name__ == "__main__":
    main()
