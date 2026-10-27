"""Merge the five VLM annotations of RFW into one label per image and attribute.

Two rules are implemented:

* ``strict`` (Supp. Tabs. 2-3): ``unknown``
  predictions are discarded and the label must receive strictly more than half
  of the remaining votes; otherwise the result is ``unknown``.
  Output: ``labels_consensus.csv``. Images with several faces produce list
  values (e.g. ``['male', 'female']``); they are resolved manually with
  ``resolve_multiface.py`` which writes ``labels_RFW.csv``.
* ``plurality``: most frequent prediction among the five, ``unknown`` counting
  as a regular answer and ties broken alphabetically -- i.e. pandas'
  ``DataFrame.mode(axis=1)[0]`` over the five votes. With the multi-face
  overrides this gives ``labels_RFW_mode.csv``, the label set read by every
  experiment (Tab. 1, Figs. 4-5, Tab. 2); 332 of the 406,070 values are
  ``unknown`` (4,137 with the strict rule).

    python preprocessing/annotation/majority_vote.py --rule strict
    python preprocessing/annotation/majority_vote.py --rule plurality
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from functools import reduce
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
ANN = REPO / "data" / "annotations" / "rfw"
ANNOTATORS = ["internvl", "ovis", "qwen", "sailvl", "ola"]
ATTRIBUTES = ["Gender", "Age", "Skin Color", "Ancestry", "Hair Color", "Bangs", "Bald", "Beard", "Glasses", "Headwear"]


def normalise(value):
    """``"['x']"`` -> ``"x"``; multi-element lists are kept as their string."""
    try:
        v = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value
    if isinstance(v, list) and len(v) == 1:
        return v[0]
    return str(v)


def strict_majority(votes, unknown="unknown"):
    valid = [v for v in votes if v != unknown]
    if not valid:
        return pd.NA
    label, count = Counter(valid).most_common(1)[0]
    if count >= len(valid) // 2 + 1:
        return label
    return unknown


def plurality(votes):
    c = Counter(votes)
    best = max(c.values())
    return sorted(v for v, n in c.items() if n == best)[0]


def load_votes(ann_dir: Path) -> pd.DataFrame:
    dfs = []
    for i, name in enumerate(ANNOTATORS, start=1):
        df = pd.read_csv(ann_dir / f"{name}_labels.csv")
        df[ATTRIBUTES] = df[ATTRIBUTES].astype(str).map(normalise) if hasattr(df, "map") else df[ATTRIBUTES].astype(str).applymap(normalise)
        dfs.append(df.rename(columns={a: f"A{i}_{a}" for a in ATTRIBUTES}))
    return reduce(lambda l, r: pd.merge(l, r, on="File Name"), dfs)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rule", choices=["strict", "plurality"], default="strict")
    p.add_argument("--ann-dir", default=str(ANN), help="Folder with the <vlm>_labels.csv files")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    merged = load_votes(Path(args.ann_dir))
    fn = strict_majority if args.rule == "strict" else plurality
    out = pd.DataFrame({"File Name": merged["File Name"]})
    for a in ATTRIBUTES:
        cols = [f"A{i}_{a}" for i in range(1, len(ANNOTATORS) + 1)]
        out[a] = [fn(list(r)) for r in merged[cols].itertuples(index=False)]
    default = "labels_consensus.csv" if args.rule == "strict" else "labels_RFW_mode.csv"
    path = Path(args.out) if args.out else REPO / "results" / "annotations" / default
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    print("Saved", path)


if __name__ == "__main__":
    main()
