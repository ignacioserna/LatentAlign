"""Supp. Tab. 2 -- class counts of the merged RFW annotations and VLM agreement.

Agreement of an image = fraction of the VLMs with a valid (non-"unknown")
prediction whose prediction equals the final label; mean and standard
deviation per class.

    python preprocessing/annotation/annotation_stats.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from majority_vote import ANN, ANNOTATORS, ATTRIBUTES, REPO, load_votes


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--labels", default=str(ANN / "labels_RFW.csv"))
    args = p.parse_args()

    final = pd.read_csv(args.labels)
    votes = load_votes(ANN).set_index("File Name").loc[final["File Name"]].reset_index()
    total = len(final)
    rows = []
    for a in ATTRIBUTES:
        lab = final[a].fillna("unknown").astype(str).astype(object)
        V = np.stack([votes[f"A{i}_{a}"].astype(str).values for i in range(1, len(ANNOTATORS) + 1)], axis=1)
        # fraction of the VLMs with a valid (non-unknown) prediction that agree with the final label
        valid = V != "unknown"
        hits = (V == np.asarray(lab.values, dtype=object)[:, None]) & valid
        with np.errstate(invalid="ignore", divide="ignore"):
            agree = hits.sum(axis=1) / valid.sum(axis=1)
        for cls, n in lab.value_counts().items():
            m = lab.values == cls
            rows.append({"Category": a, "Class": cls, "Count": int(n), "%": round(100 * n / total, 1),
                         "Mean": np.nan if cls == "unknown" else round(np.nanmean(agree[m]), 2),
                         "Std": np.nan if cls == "unknown" else round(np.nanstd(agree[m]), 2)})
    df = pd.DataFrame(rows)
    out = REPO / "results" / "tables"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "supp_table2_annotation_stats.csv", index=False)
    print(f"Total samples: {total}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
