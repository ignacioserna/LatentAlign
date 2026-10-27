"""Supp. Tab. 3 and Supp. Fig. 1 -- VLM labels vs. a human-annotated 1K sample.

The 1,000 human annotations (random sample, seed 42, annotated with
``annotate_rfw_sample.ipynb``) are in ``data/annotations/rfw/human_annotations_1k.csv``;
the confusion matrices against the merged VLM labels (rows = human,
columns = VLM) are recomputed from it. ``--from-json`` uses the matrices
stored in ``human_vs_vlm_confusion.json`` instead.

    python preprocessing/annotation/human_validation.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

REPO = Path(__file__).resolve().parents[2]
ANN = REPO / "data" / "annotations" / "rfw"
KEYS = {"Age": "age_df", "Ancestry": "ancestry_df", "Bald": "bald_df", "Bangs": "bangs_df", "Beard": "beard_df",
        "Gender": "gender_df", "Glasses": "glasses_df", "Hair Color": "haircolor_df", "Headwear": "headwear_df",
        "Skin Color": "skintone_df"}


def kappa(M):
    M = np.asarray(M, dtype=float)
    n = M.sum()
    po = np.trace(M) / n
    pe = (M.sum(0) * M.sum(1)).sum() / n ** 2
    return po, (po - pe) / (1 - pe)


def _normalise(v):
    """Same normalisation as the validation used for the paper."""
    if pd.isna(v):
        return np.nan
    v = str(v).strip().lower()
    if v in ["", "na", "n/a", "unknown", "unk", "unspecified", "none"]:
        return np.nan
    return {"m": "male", "man": "male", "f": "female", "woman": "female", "old": "senior", "elderly": "senior",
            "y": "yes", "true": "yes", "n": "no", "false": "no"}.get(v, v)


HUMAN_COLS = {"Gender": "gender", "Age": "age", "Skin Color": "skin_color", "Ancestry": "ancestry",
              "Hair Color": "hair_color", "Bangs": "bangs", "Bald": "bald", "Beard": "beard",
              "Glasses": "glasses", "Headwear": "headwear"}


def from_csv(human_csv, vlm_csv):
    """Confusion matrices human (rows) x VLM (columns); images are matched by file name."""
    h = pd.read_csv(human_csv, dtype=str, keep_default_na=False, na_values=[""])
    v = pd.read_csv(vlm_csv, dtype=str, keep_default_na=False, na_values=[""])
    h["key"] = h["filename"].map(lambda p: Path(str(p)).name)
    v["key"] = v["File Name"].map(lambda p: Path(str(p)).name)
    m = h.merge(v, on="key", how="inner", suffixes=("_human", "_vlm"))
    mats = {}
    for a, hc in HUMAN_COLS.items():
        hv, vv = m[hc].map(_normalise), m[a].map(_normalise)
        ok = hv.notna() & vv.notna()
        mats[a] = pd.crosstab(hv[ok].str.capitalize(), vv[ok].str.capitalize())
        mats[a] = mats[a].reindex(index=sorted(set(mats[a].index) | set(mats[a].columns)),
                                  columns=sorted(set(mats[a].index) | set(mats[a].columns)), fill_value=0)
    return mats


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--human-csv", default=str(ANN / "human_annotations_1k.csv"))
    p.add_argument("--from-json", action="store_true")
    p.add_argument("--vlm-csv", default=str(ANN / "labels_RFW.csv"))
    args = p.parse_args()

    if not args.from_json:
        mats = from_csv(args.human_csv, args.vlm_csv)
    else:
        stored = json.loads((ANN / "human_vs_vlm_confusion.json").read_text())
        mats = {a: pd.DataFrame(stored[k]["matrix"], index=stored[k]["labels_rows"], columns=stored[k]["labels_cols"])
                for a, k in KEYS.items()}

    out_t, out_f = REPO / "results" / "tables", REPO / "results" / "figures"
    out_t.mkdir(parents=True, exist_ok=True)
    out_f.mkdir(parents=True, exist_ok=True)
    rows = []
    for a in sorted(mats):
        po, k = kappa(mats[a].values)
        rows.append({"Attribute": a, "Match Rate": round(po, 4), "Cohen's kappa": round(k, 4)})
        with plt.rc_context(rc={"font.family": "DejaVu Sans Mono"}):
            plt.figure(figsize=(max(4, 0.7 * len(mats[a])), max(3, 0.6 * len(mats[a]))))
            ax = sns.heatmap(mats[a], annot=True, square=True, fmt=".10g", cmap=sns.cubehelix_palette(as_cmap=True), vmin=0)
            ax.set_xlabel("VLM", fontsize=14)
            ax.set_ylabel("Human", fontsize=14)
            ax.xaxis.set_label_position("top")
            ax.xaxis.tick_top()
            ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="left")
            plt.tight_layout()
            plt.savefig(out_f / f"supp_fig1_confusion_{a.replace(' ', '_').lower()}.pdf")
            plt.close()
    df = pd.DataFrame(rows)
    df.to_csv(out_t / "supp_table3_human_validation.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
