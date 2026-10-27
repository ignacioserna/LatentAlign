"""Resolve images in which the VLMs described several faces.

For fewer than 0.1% of the images the VLMs returned one label per detected
face (e.g. ``['male', 'female']``). These cases were inspected manually and
the label of the most central face was kept (Supp. A.1). The manual decisions
are stored in ``data/annotations/rfw/manual_overrides_{strict,mode}.csv`` and
applied here, which reproduces exactly the label files used in the paper.

    python preprocessing/annotation/resolve_multiface.py --rule strict     # -> labels_RFW.csv
    python preprocessing/annotation/resolve_multiface.py --rule plurality  # -> labels_RFW_mode.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
ANN = REPO / "data" / "annotations" / "rfw"
RES = REPO / "results" / "annotations"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rule", choices=["strict", "plurality"], default="strict")
    args = p.parse_args()
    if args.rule == "strict":
        src, ovr, dst = RES / "labels_consensus.csv", ANN / "manual_overrides_strict.csv", RES / "labels_RFW.csv"
        reference = ANN / "labels_RFW.csv"
    else:
        src, ovr, dst = RES / "labels_RFW_mode.csv", ANN / "manual_overrides_mode.csv", RES / "labels_RFW_mode.csv"
        reference = ANN / "labels_RFW_mode.csv"

    df = pd.read_csv(src).set_index("File Name")
    overrides = pd.read_csv(ovr)
    unresolved = 0
    for r in overrides.itertuples(index=False):
        if str(df.at[r[0], r[1]]) != str(r.automatic):
            unresolved += 1
        df.at[r[0], r[1]] = r.manual
    df = df.reset_index()
    df.to_csv(dst, index=False)
    print(f"Saved {dst} ({len(overrides)} manual decisions, {unresolved} did not match the automatic vote)")

    ref = pd.read_csv(reference)
    same = (ref.fillna("").astype(str).values == df.fillna("").astype(str).values).all()
    print("Identical to the file used in the paper:", bool(same))


if __name__ == "__main__":
    main()
