#!/usr/bin/env python3
"""Tab. 3 / Supp. Tab. 9 numbers of one or more work folders produced by
run_both_thresholds.sh / rescore_threshold.sh (reads <work>/report_n50)."""
import json
import os
import sys

import numpy as np
import pandas as pd

works = sys.argv[1:]
if len(works) >= 2:
    for run in ("run_n10", "run_n50"):
        p = [os.path.join(w, run, "slice_probs.npy") for w in works]
        if all(os.path.exists(x) for x in p):
            same = all(np.array_equal(np.load(p[0], mmap_mode="r"), np.load(x, mmap_mode="r")) for x in p[1:])
            print(f"{run}: identical slices in all folders: {same}")
for w in works:
    r = os.path.join(w, "report_n50")
    thr = json.load(open(os.path.join(w, "metrics.json")))["global_threshold"]
    g = pd.read_csv(os.path.join(r, "slice_groups.csv"))
    d = pd.read_csv(os.path.join(r, "fmr_groups.csv"))
    s = pd.read_csv(os.path.join(r, "fmr_summary.csv")).set_index("method")
    la = d[d.method == "latentalign"]["fmr"]
    t, a = g["topM_fmr"], g["all_fmr"]
    print(f"\n== {w}  (threshold {thr:.4f})")
    print(f"Domino top-100 projection : {t.min():.4f} - {t.max():.4f}  median {t.median():.4f}")
    print(f"Domino as populations     : {a.min():.4f} - {a.max():.4f}  median {a.median():.4f}  "
          f"({(a > 0.001).sum()}/50 above 0.0010)")
    print(f"LatentAlign (n={len(la)})    : {la.min():.4f} - {la.max():.4f}  median {la.median():.4f}")
    print(f"LatentAlign groups above the best Domino slice: {100 * (la > t.max()).mean():.1f}%")
    print(s[["n_candidates", "best_fmr", "mean_fmr_top10", "mean_fmr_top100"]].to_string())
