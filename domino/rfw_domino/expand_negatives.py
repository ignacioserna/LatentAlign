"""Stage 01b - expand the impostor set so a global-FMR operating point is measurable.

The RFW protocol ships 3,000 impostor pairs per race, ~12,000 in total. At a
threshold calibrated for a global FMR of 1e-3 that leaves **12 false matches**
in the whole benchmark, which is far too few to estimate a per-slice error rate:
the slice-level confidence intervals span two orders of magnitude and only a
slice built around a single error event can reach nominal significance.

The fix costs nothing but disk. Face and CLIP embeddings are stored per *image*
and pairs merely index into them, so additional impostor pairs need no new
forward passes. This script samples extra within-race, different-identity pairs
from the images already present in `images.csv` and appends them to
`pairs.csv` with `protocol == "expanded"`; the original rows keep
`protocol == "official"` so the LFW-style 10-fold accuracy can still be
computed on the published protocol alone.

    python scripts/expand_negatives.py --work work --n-per-race 125000

125,000 per race gives 500,000 impostor pairs and ~500 expected false matches
at FMR = 1e-3, enough for slice-level rates with usable intervals.

After running this, re-run stages 03 (verification), 05 (slicing) and 06
(report). Stages 02/04 (embeddings) do NOT need to be re-run.
"""

import argparse
import os

import numpy as np
import pandas as pd


def sample_within_race_impostors(images: pd.DataFrame, n_target: int,
                                 rng: np.random.Generator,
                                 exclude: set, max_tries: int = 40):
    """Sample distinct (img_id1, img_id2) pairs of different identities.

    Rejection sampling over the identity-labelled image pool of one race.
    Returns an (n, 2) int array of image ids, img_id1 < img_id2.
    """
    ids = images["image_id"].to_numpy()
    idents = pd.factorize(images["identity"])[0]
    n_img = len(ids)
    if n_img < 2:
        return np.empty((0, 2), dtype=int)

    # upper bound on distinct different-identity pairs available
    _, counts = np.unique(idents, return_counts=True)
    max_pairs = n_img * (n_img - 1) // 2 - int((counts * (counts - 1) // 2).sum())
    if n_target > max_pairs:
        print(f"[expand]   only {max_pairs:,} distinct impostor pairs exist "
              f"for this race; capping at that")
        n_target = max_pairs

    seen = set()
    out = []
    for _ in range(max_tries):
        if len(out) >= n_target:
            break
        need = n_target - len(out)
        a = rng.integers(0, n_img, size=need * 2)
        b = rng.integers(0, n_img, size=need * 2)
        ok = (idents[a] != idents[b])          # different identity
        a, b = a[ok], b[ok]
        lo = np.minimum(ids[a], ids[b])
        hi = np.maximum(ids[a], ids[b])
        for i, j in zip(lo, hi):
            key = (int(i), int(j))
            if key in seen or key in exclude:
                continue
            seen.add(key)
            out.append(key)
            if len(out) >= n_target:
                break
    return np.array(out, dtype=int) if out else np.empty((0, 2), dtype=int)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--work", default="work")
    ap.add_argument("--n-per-race", type=int, default=125000,
                    help="extra impostor pairs to sample per race "
                         "(125000 x 4 races -> ~500 false matches at FMR=1e-3)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None,
                    help="output pairs csv (default: overwrite work/pairs.csv, "
                         "after backing the original up to pairs_official.csv)")
    args = ap.parse_args(argv)

    work_dir = os.path.abspath(args.work)
    pairs_csv = os.path.join(work_dir, "pairs.csv")
    images_csv = os.path.join(work_dir, "images.csv")
    pairs = pd.read_csv(pairs_csv)
    images = pd.read_csv(images_csv)

    if "protocol" not in pairs.columns:
        pairs["protocol"] = "official"
    pairs = pairs[pairs["protocol"] == "official"].copy()

    path_of = dict(zip(images["image_id"], images["path"]))
    ident_of = dict(zip(images["image_id"], images["identity"]))

    existing = set()
    for i, j in zip(pairs["img_id1"], pairs["img_id2"]):
        existing.add((int(min(i, j)), int(max(i, j))))

    rng = np.random.default_rng(args.seed)
    new_rows = []
    next_id = int(pairs["pair_id"].max()) + 1

    for race, img_g in images.groupby("race"):
        print(f"[expand] {race}: {len(img_g):,} images, "
              f"{img_g['identity'].nunique():,} identities")
        sampled = sample_within_race_impostors(
            img_g, args.n_per_race, rng, existing
        )
        print(f"[expand]   sampled {len(sampled):,} new impostor pairs")
        for i, j in sampled:
            new_rows.append({
                "pair_id": next_id, "race": race,
                "fold": int(next_id % 10), "same": 0,
                "name1": ident_of[i], "num1": -1,
                "img_id1": int(i), "path1": path_of[i],
                "name2": ident_of[j], "num2": -1,
                "img_id2": int(j), "path2": path_of[j],
                "protocol": "expanded",
            })
            next_id += 1

    out = pd.concat([pairs, pd.DataFrame(new_rows)], ignore_index=True)
    dest = args.out or pairs_csv
    if dest == pairs_csv:
        backup = os.path.join(work_dir, "pairs_official.csv")
        if not os.path.exists(backup):
            pairs.to_csv(backup, index=False)
            print(f"[expand] backed the official protocol up to {backup}")
    out.to_csv(dest, index=False)

    n_neg = int((out["same"] == 0).sum())
    print(f"[expand] wrote {len(out):,} pairs -> {dest}")
    print(f"[expand]   {int((out['same'] == 1).sum()):,} genuine, "
          f"{n_neg:,} impostor")
    for far in (1e-2, 1e-3, 1e-4):
        print(f"[expand]   expected false matches at FMR={far:.0e}: "
              f"{n_neg * far:,.0f}")
    print("[expand] embeddings are per-image and are unchanged; re-run "
          "stages 03, 05 and 06 only.")


if __name__ == "__main__":
    main()