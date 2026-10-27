"""Stage 06b - project Domino slices onto image sets so they can be compared
to LatentAlign groups on equal terms.

A Domino slice and a LatentAlign group are not the same kind of object:

    LatentAlign group : ~100 IMAGES, exhaustively paired -> ~4,950 impostor
                        pairs. Pair density within the group is 100%.
    Domino slice      : N PAIRS sampled from a pool of ~10k images per race.
                        A 4,575-pair slice may touch several thousand distinct
                        images, so its pair density is on the order of 0.03%.

Matching the two on pair count equalises the variance of the FMR estimate but
not the nature of the claim. "These 100 people are mutually confusable" is an
auditable statement about a population; "these 4,575 pairwise events lie in a
region of CLIP-pair space" is not, and you cannot hand a security auditor a
list of faces from it.

This script makes Domino produce the LatentAlign kind of object. For each
slice it ranks images by aggregated slice membership, keeps the top
--group-size, and then evaluates FMR over ALL within-group impostor pairs at
the global threshold, exactly as LatentAlign is evaluated. The resulting
numbers are directly comparable to Table 2 of the paper.

    python scripts/slice_to_groups.py --work work --group-size 100

Two projections are reported per slice:

    top-M      the M images with the highest aggregated membership; this is
               the fair, favourable-to-Domino comparison, and the one a
               reviewer will ask for.
    all-images every distinct image the slice's pairs touch, exhaustively
               paired; this shows what the slice actually denotes as a
               population, and is usually far weaker.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd


def group_fmr(img_ids, embs, identity, thr):
    """FMR over all within-group impostor pairs, at a fixed global threshold."""
    idx = np.asarray(img_ids)
    e = embs[idx]
    keep = np.isfinite(e).all(axis=1)
    idx, e = idx[keep], e[keep]
    if len(idx) < 2:
        return dict(n_images=len(idx), n_pairs=0, n_fm=0, fmr=float("nan"))
    ident = np.asarray([identity[i] for i in idx])
    sims = e @ e.T
    iu = np.triu_indices(len(idx), k=1)
    s = sims[iu]
    diff = ident[iu[0]] != ident[iu[1]]          # impostor pairs only
    s = s[diff]
    if len(s) == 0:
        return dict(n_images=len(idx), n_pairs=0, n_fm=0, fmr=float("nan"))
    n_fm = int((s >= thr).sum())
    return dict(n_images=len(idx), n_pairs=int(len(s)), n_fm=n_fm,
                fmr=float(n_fm / len(s)))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--work", default="work")
    ap.add_argument("--group-size", type=int, default=100,
                    help="images per projected group (match your LatentAlign "
                         "group size)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    w = os.path.abspath(args.work)
    pairs = pd.read_csv(os.path.join(w, "pairs_scored.csv"))
    images = pd.read_csv(os.path.join(w, "images.csv"))
    assign = pd.read_csv(os.path.join(w, "slices_assign.csv"))
    probs = np.load(os.path.join(w, "slice_probs.npy"))
    embs = np.load(os.path.join(w, "face_embs.npy"))
    with open(os.path.join(w, "metrics.json")) as f:
        thr = json.load(f)["global_threshold"]

    df = pairs.merge(assign[["pair_id"]], on="pair_id")
    assert len(df) == len(probs), (
        f"{len(df)} pairs vs {len(probs)} slice rows - rerun stages 05/06 "
        f"with the same --subset")
    identity = dict(zip(images["image_id"], images["identity"]))
    race = dict(zip(images["image_id"], images["race"]))
    n_img = len(embs)
    i1 = df["img_id1"].to_numpy()
    i2 = df["img_id2"].to_numpy()

    global_fmr = float((pairs.loc[pairs["same"] == 0, "sim"] >= thr).mean())
    print(f"[project] global threshold {thr:.4f}, global FMR {global_fmr:.2e}")
    print(f"[project] projecting {probs.shape[1]} slices onto "
          f"{args.group_size}-image groups\n")

    rows = []
    for k in range(probs.shape[1]):
        wk = probs[:, k]
        # aggregated per-image membership: each pair contributes its slice
        # probability to both of its images
        score = np.zeros(n_img)
        np.add.at(score, i1, wk)
        np.add.at(score, i2, wk)
        touched = np.unique(np.concatenate([i1[wk > 0.5], i2[wk > 0.5]])) \
            if (wk > 0.5).any() else np.unique(np.concatenate([i1, i2]))

        top = np.argsort(-score)[: args.group_size]
        g_top = group_fmr(top, embs, identity, thr)
        g_all = group_fmr(touched, embs, identity, thr)

        rr = pd.Series([race[i] for i in top]).value_counts(normalize=True)
        rows.append({
            "slice": k,
            "top_race": rr.index[0], "top_race_share": round(float(rr.iloc[0]), 3),
            "topM_images": g_top["n_images"], "topM_pairs": g_top["n_pairs"],
            "topM_fm": g_top["n_fm"], "topM_fmr": g_top["fmr"],
            "topM_lift": g_top["fmr"] / global_fmr if global_fmr else np.nan,
            "all_images": g_all["n_images"], "all_pairs": g_all["n_pairs"],
            "all_fmr": g_all["fmr"],
            "all_lift": g_all["fmr"] / global_fmr if global_fmr else np.nan,
        })

    out = pd.DataFrame(rows).sort_values("topM_lift", ascending=False)
    dest = args.out or os.path.join(w, "report", "slice_groups.csv")
    out.to_csv(dest, index=False)
    print(out.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\n[project] wrote {dest}")
    print(f"[project] compare topM_lift against your LatentAlign group lifts; "
          f"both are now FMR over all within-group impostor pairs of an "
          f"{args.group_size}-image set at the same global threshold.")


if __name__ == "__main__":
    main()