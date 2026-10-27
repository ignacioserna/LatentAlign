#!/usr/bin/env python3
"""Assertions for the synthetic smoke test: Domino must recover the planted
weak-identity slice (African false matches) without access to race labels."""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

WORK = sys.argv[1] if len(sys.argv) > 1 else "smoke_work"
PLANTED_RACE = sys.argv[2] if len(sys.argv) > 2 else "African"

summary = pd.read_csv(Path(WORK) / "report" / "slices_summary.csv")
print(summary.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

top = summary.head(3)
hits = top[
    (top["error_lift"] >= 2.0)
    & (top["top_race"] == PLANTED_RACE)
    & (top["top_race_share"] >= 0.6)
    & (top["false_match_share_of_errors"] >= 0.6)
]

if len(hits) == 0:
    print(f"\nSMOKE TEST FAILED: no slice in the top 3 matches the planted "
          f"failure mode ({PLANTED_RACE} false matches, lift>=2, share>=0.6).")
    sys.exit(1)

r = hits.iloc[0]
print(f"\nSMOKE TEST PASSED: slice {int(r['slice'])} recovered the planted "
      f"slice - {int(r['n_pairs'])} pairs, error rate {r['error_rate']:.2f} "
      f"({r['error_lift']:.1f}x overall), {r['top_race']} share "
      f"{r['top_race_share']:.2f}, FM share "
      f"{r['false_match_share_of_errors']:.2f}.")
