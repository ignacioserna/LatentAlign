"""Stage 07 - rank every method's candidate groups by FMR and compare.

Replaces compare_groups.py. Every method is reduced to the same object -- a set of
N images -- and scored the same way: FMR over all within-group impostor pairs at the
single global threshold from `metrics.json`. Run stage 03 with
`--criterion far --far-target 1e-3` and the whole table is at FMR = 1e-3.

Output is exactly the top-K groups by FMR per method, with their FMR and coherence.
No lift, no random-baseline ratios, no demographic breakdown.

Where the groups come from:

  <work>/baselines/*/groups.csv   written by baseline_svm / baseline_b2t /
                                  baseline_spotlight
  domino                          projected from slice_probs.npy the way
                                  slice_to_groups.py does it (each pair's slice
                                  probability is added to both of its images; the
                                  top N images by aggregated membership form the
                                  group)
  --groups NAME=path.csv          any CSV with group_id,image_id[,label]. This is
                                  how LatentAlign joins the table -- repeat the flag
                                  for as many runs as you want.
  random                          N-image groups drawn uniformly, listed as an
                                  ordinary method row so you can see the floor
                                  without any ratio arithmetic. --n-random 0 drops it.

Two columns to read together
----------------------------
`fmr` is measured over all within-group impostor pairs. `fmr_heldout` splits those
pairs in half: candidates are *ranked* on one half and *reported* on the other. The
two agree when a group is genuinely high-FMR and diverge when a method got there by
having many candidates to pick from -- with ~1,700 phrases B2T is choosing its best
group from a much larger pool than a method with 50 candidates, and `fmr` alone
rewards that. `n_candidates` is in the table for the same reason. Use --no-holdout
if you only want the plain number.

    python -m rfw_domino.compare_fmr --work work --group-size 100 \
        --groups latentalign=work/lfa_groups.csv --top 10,100
"""

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

from .image_signal import coherence, load_context, make_scorer


# --------------------------------------------------------------------------
# scoring one group
# --------------------------------------------------------------------------

def group_stats(ids, ctx, seed=0, holdout=True):
    """FMR over all within-group impostor pairs, plus a split-half version."""
    idx = np.unique(np.asarray(ids, dtype=int))
    e = ctx.face_embs[idx]
    keep = np.isfinite(e).all(axis=1)
    idx, e = idx[keep], e[keep]
    out = dict(n_images=len(idx), n_pairs=0, n_fm=0, fmr=np.nan,
               fmr_select=np.nan, fmr_heldout=np.nan,
               coherence=np.nan, coherence_face=np.nan, n_identities=0)
    if len(idx) < 2:
        return out

    ident = np.array([ctx.identity[int(i)] for i in idx])
    _, code = np.unique(ident, return_inverse=True)
    S = e @ e.T
    iu = np.triu_indices(len(idx), k=1)
    s = S[iu]
    impostor = code[iu[0]] != code[iu[1]]
    s = s[impostor]
    out["n_identities"] = int(len(np.unique(ident)))
    out["coherence"] = coherence(ctx.clip_embs[idx])
    out["coherence_face"] = coherence(e)
    if len(s) == 0:
        return out

    hit = s >= ctx.tau
    out["n_pairs"] = int(len(s))
    out["n_fm"] = int(hit.sum())
    out["fmr"] = float(hit.mean())

    if holdout:
        # deterministic per group, so the same image set splits identically
        # whichever method proposed it
        key = int(abs(int(np.sum(idx.astype(np.int64) * (np.arange(len(idx)) + 1)))
                      + seed) % (2 ** 31))
        half = np.random.default_rng(key).random(len(s)) < 0.5
        if half.any() and (~half).any():
            out["fmr_select"] = float(hit[half].mean())
            out["fmr_heldout"] = float(hit[~half].mean())
    return out


# --------------------------------------------------------------------------
# collecting candidate groups
# --------------------------------------------------------------------------

def from_baselines(ctx):
    out = []
    for path in sorted(glob.glob(ctx.work.path("baselines/*/groups.csv"))):
        method = os.path.basename(os.path.dirname(path))
        df = pd.read_csv(path)
        for gid, g in df.groupby("group_id"):
            g = g.sort_values("rank")
            out.append(dict(method=method, group_id=int(gid),
                            native_rank=int(g["native_rank"].iloc[0])
                            if "native_rank" in g else int(gid),
                            label=str(g["label"].iloc[0]),
                            image_ids=g["image_id"].to_numpy()))
        print(f"[cmp] {method}: {df['group_id'].nunique()} candidate groups")
    return out


def from_domino(ctx, group_size):
    w = ctx.work
    if not (os.path.exists(w.slice_probs_npy)
            and os.path.exists(w.slices_assign_csv)):
        return []
    probs = np.load(w.slice_probs_npy)
    assign = pd.read_csv(w.slices_assign_csv)
    df = ctx.pairs.merge(assign[["pair_id"]], on="pair_id")
    if len(df) != len(probs):
        print(f"[cmp] skipping domino: {len(df)} scored pairs match the slice "
              f"assignment but slice_probs has {len(probs)} rows (re-run stage 05 "
              f"with the same --subset/--protocol)")
        return []
    i1, i2 = df["img_id1"].to_numpy(), df["img_id2"].to_numpy()
    n_img = len(ctx.face_embs)
    usable = np.zeros(n_img, dtype=bool)
    usable[ctx.pool] = True

    out = []
    for k in range(probs.shape[1]):
        score = np.zeros(n_img)
        np.add.at(score, i1, probs[:, k])
        np.add.at(score, i2, probs[:, k])
        score[~usable] = -np.inf
        out.append(dict(method="domino", group_id=k, native_rank=k,
                        label=f"slice {k}",
                        image_ids=np.argsort(-score)[:group_size]))
    print(f"[cmp] domino: {len(out)} candidate groups (slices projected to "
          f"{group_size}-image sets)")
    if len(out) < 50:
        print(f"[cmp]   NOTE: only {len(out)} candidates. Other methods here offer "
              f"hundreds, and the top-K-by-FMR comparison favours whoever has more. "
              f"Re-run stage 05 with a larger --n-slices for a matched budget.")
    return out


def from_external(specs, ctx):
    out = []
    for spec in specs or []:
        if "=" not in spec:
            raise SystemExit(f"--groups expects NAME=path.csv, got {spec!r}")
        name, path = spec.split("=", 1)
        df = pd.read_csv(path)
        cols = {c.lower(): c for c in df.columns}
        gcol = next((cols[c] for c in ("group_id", "group", "cluster", "slice")
                     if c in cols), None)
        icol = next((cols[c] for c in ("image_id", "img_id", "image", "id")
                     if c in cols), None)
        if gcol is None or icol is None:
            raise SystemExit(f"{path}: need group_id and image_id columns, got "
                             f"{list(df.columns)}")
        lcol = cols.get("label") or cols.get("name") or cols.get("description")
        for gid, g in df.groupby(gcol):
            out.append(dict(method=name, group_id=int(gid),
                            native_rank=int(gid),
                            label=str(g[lcol].iloc[0]) if lcol else "",
                            image_ids=g[icol].to_numpy().astype(int)))
        print(f"[cmp] {name}: {df[gcol].nunique()} candidate groups from {path}")
    return out


def from_random(ctx, group_size, n, rng):
    out = []
    for k in range(n):
        out.append(dict(method="random", group_id=k, native_rank=k, label="",
                        image_ids=rng.choice(ctx.pool, size=group_size,
                                             replace=False)))
    if n:
        print(f"[cmp] random: {n} candidate groups")
    return out


# --------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", default="work")
    ap.add_argument("--group-size", type=int, default=100)
    ap.add_argument("--top", default="10,100",
                    help="comma-separated K values for the mean-FMR columns")
    ap.add_argument("--groups", action="append", default=None, metavar="NAME=PATH",
                    help="external groups, e.g. latentalign=work/lfa_groups.csv "
                         "(repeatable)")
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--protocol", default=None, choices=["official", "expanded"])
    ap.add_argument("--n-random", type=int, default=200)
    ap.add_argument("--no-holdout", dest="holdout", action="store_false")
    ap.set_defaults(holdout=True)
    ap.add_argument("--no-domino", dest="domino", action="store_false")
    ap.set_defaults(domino=True)
    ap.add_argument("--relabel", action="store_true",
                    help="re-describe every group from the phrase bank, overwriting "
                         "whatever label it came with (useful for LatentAlign and "
                         "domino groups, which arrive unlabelled)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    tops = [int(t) for t in str(args.top).split(",") if t.strip()]
    ctx = load_context(args.work, args.threshold, args.protocol)
    rng = np.random.default_rng(args.seed)

    groups = from_baselines(ctx)
    if args.domino:
        groups += from_domino(ctx, args.group_size)
    groups += from_external(args.groups, ctx)
    groups += from_random(ctx, args.group_size, args.n_random, rng)
    if not groups:
        raise SystemExit("no candidate groups found; run the baselines first")

    scorer = make_scorer(ctx) if args.relabel or any(
        not g["label"] for g in groups) else None

    rows = []
    for g in groups:
        ids = np.asarray(g["image_ids"], dtype=int)[:args.group_size]
        st = group_stats(ids, ctx, args.seed, args.holdout)
        label = g["label"]
        if scorer is not None and (args.relabel or not label):
            _, label = scorer.name(ctx.clip_embs[np.unique(ids)])
        rows.append({"method": g["method"], "group_id": g["group_id"],
                     "native_rank": g["native_rank"], "label": label, **st})

    detail = pd.DataFrame(rows)
    # rank on the selection half when holding out, otherwise on the full FMR
    rank_key = "fmr_select" if (args.holdout
                                and detail["fmr_select"].notna().any()) else "fmr"
    detail["_rank_on"] = detail[rank_key].fillna(detail["fmr"])
    detail = detail.sort_values(["method", "_rank_on"], ascending=[True, False])
    detail["fmr_rank"] = detail.groupby("method").cumcount()
    detail = detail.drop(columns=["_rank_on"])

    summary = []
    for method, g in detail.groupby("method"):
        g = g.sort_values("fmr_rank")
        best = g.iloc[0]
        row = {"method": method, "n_candidates": len(g),
               "best_fmr": float(best["fmr"])}
        for k in tops:
            row[f"mean_fmr_top{k}"] = float(g.head(k)["fmr"].mean())
        if args.holdout:
            for k in tops:
                row[f"mean_fmr_heldout_top{k}"] = float(
                    g.head(k)["fmr_heldout"].mean())
        row[f"mean_coherence_top{tops[0]}"] = float(
            g.head(tops[0])["coherence"].mean())
        row["mean_n_identities_top%d" % tops[0]] = float(
            g.head(tops[0])["n_identities"].mean())
        row["best_label"] = str(best["label"])[:80]
        summary.append(row)
    summary = pd.DataFrame(summary).sort_values(f"mean_fmr_top{tops[0]}",
                                                ascending=False)

    out_dir = args.out or ctx.work.report_dir
    os.makedirs(out_dir, exist_ok=True)
    detail.to_csv(os.path.join(out_dir, "fmr_groups.csv"), index=False)
    summary.to_csv(os.path.join(out_dir, "fmr_summary.csv"), index=False)

    members = []
    for _, r in detail[detail.fmr_rank < max(tops)].iterrows():
        g = next(x for x in groups if x["method"] == r["method"]
                 and x["group_id"] == r["group_id"])
        for img in np.asarray(g["image_ids"], dtype=int)[:args.group_size]:
            members.append({"method": r["method"], "fmr_rank": r["fmr_rank"],
                            "group_id": r["group_id"], "image_id": int(img)})
    pd.DataFrame(members).to_csv(os.path.join(out_dir, "fmr_top_members.csv"),
                                 index=False)

    L = [f"# Top-FMR groups per method\n",
         f"- threshold **{ctx.tau:.4f}** (`{ctx.criterion}`), global FMR "
         f"**{ctx.global_fmr:.3e}** over {ctx.n_impostor:,} impostor pairs",
         f"- group size **{args.group_size}** images; FMR over all within-group "
         f"impostor pairs",
         f"- `coherence` = mean cosine of members to the group centroid in CLIP "
         f"space, in [0,1]\n" if not args.holdout else
         f"- `coherence` = mean cosine of members to the group centroid in CLIP "
         f"space, in [0,1]\n- groups are ranked on one half of their impostor pairs "
         f"and `fmr_heldout` reported on the other; `fmr` uses all pairs\n",
         "## Summary\n", summary.to_markdown(index=False, floatfmt=".4g"), "",
         f"## Top {max(tops)} groups per method\n",
         detail[detail.fmr_rank < max(tops)][
             ["method", "fmr_rank", "fmr", "fmr_heldout", "n_pairs", "n_fm",
              "n_identities", "coherence", "label"]
         ].to_markdown(index=False, floatfmt=".4g"), ""]
    with open(os.path.join(out_dir, "fmr_comparison.md"), "w") as f:
        f.write("\n".join(L))
    with open(os.path.join(out_dir, "fmr_meta.json"), "w") as f:
        json.dump({"tau": ctx.tau, "criterion": ctx.criterion,
                   "global_fmr": ctx.global_fmr, "n_impostor": ctx.n_impostor,
                   "group_size": args.group_size, "tops": tops,
                   "holdout": args.holdout}, f, indent=2)

    print()
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        print(summary.to_string(index=False))
    print(f"\n[cmp] {os.path.join(out_dir, 'fmr_summary.csv')}")
    print(f"[cmp] {os.path.join(out_dir, 'fmr_groups.csv')}  (every candidate)")
    print(f"[cmp] {os.path.join(out_dir, 'fmr_top_members.csv')}  (image ids)")
    print(f"[cmp] {os.path.join(out_dir, 'fmr_comparison.md')}")


if __name__ == "__main__":
    main()
