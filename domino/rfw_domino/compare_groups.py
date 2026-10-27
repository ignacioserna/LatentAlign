"""Stage 07 - one comparison table: LFA vs Domino vs the slice-discovery baselines.

Every method is reduced to the same object -- a set of N images -- and scored the
same way: FMR over **all** within-group impostor pairs at the **single global
threshold** from `metrics.json` (run stage 03 with `--criterion far --far-target
1e-3` and that is FMR = 1e-3). This is exactly the evaluation `slice_to_groups.py`
already applies to Domino, extended to every method, and it is what Table 2 of the
paper reports.

Sources of groups:

  domino          projected from `slice_probs.npy` the way `slice_to_groups.py`
                  does it: aggregate each pair's slice probability onto both of its
                  images, keep the top N by aggregated membership.
  baselines       `work/baselines/<method>/groups.csv`, written by baseline_svm,
                  baseline_b2t and baseline_spotlight, which never leave image space.
  lfa             `--lfa-groups path.csv` with columns (group_id, image_id) and
                  optionally (label). Point this at your own method's output to get
                  it into the same table.

Two random baselines, because the difference between them is the interesting number:

  random          uniform random N-image groups from the same pool. This is the
                  paper's "Random group" column and the denominator for `lift`.
  random_matched  for each discovered group, random groups drawn with the *same race
                  composition*. `lift_matched` is therefore the enrichment a method
                  achieves **beyond** whatever it gained by concentrating on one
                  demographic. A method whose `lift` is large and whose
                  `lift_matched` is ~1 has rediscovered race and nothing else; that
                  distinction is worth a column in the paper and a reviewer will
                  look for it.

    python -m rfw_domino.compare_groups --work work --group-size 100 \
        --lfa-groups work/lfa_groups.csv
"""

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

from .image_signal import load_bank, load_context, name_group
from .report import wilson
from .slice_to_groups import group_fmr


# --------------------------------------------------------------------------
# collecting groups from every source
# --------------------------------------------------------------------------

def domino_groups(ctx, group_size):
    w = ctx.work
    need = [w.slice_probs_npy, w.slices_assign_csv]
    if not all(os.path.exists(p) for p in need):
        print("[cmp] no Domino slices found; skipping (run stage 05)")
        return []
    probs = np.load(w.slice_probs_npy)
    assign = pd.read_csv(w.slices_assign_csv)
    df = ctx.pairs.merge(assign[["pair_id"]], on="pair_id")
    if len(df) != len(probs):
        print(f"[cmp] WARNING: {len(df)} scored pairs match the slice assignment "
              f"but slice_probs has {len(probs)} rows. Re-run stages 05 and 07 "
              f"with the same --subset/--protocol. Skipping Domino.")
        return []
    if (len(df) == len(assign)
            and not (df["pair_id"].to_numpy() == assign["pair_id"].to_numpy()).all()):
        raise RuntimeError(
            "merged pairs are not in the same row order as slices_assign.csv, so "
            "slice_probs would be silently misaligned (same check as report.py).")

    i1 = df["img_id1"].to_numpy()
    i2 = df["img_id2"].to_numpy()
    n_img = len(ctx.face_embs)
    usable = np.zeros(n_img, dtype=bool)
    usable[ctx.pool] = True

    out = []
    for k in range(probs.shape[1]):
        wk = probs[:, k]
        score = np.zeros(n_img)
        np.add.at(score, i1, wk)
        np.add.at(score, i2, wk)
        score[~usable] = -np.inf
        top = np.argsort(-score)[:group_size]
        out.append(dict(method="domino", group_id=k, label=f"slice {k}",
                        image_ids=top, scores=score[top]))
    print(f"[cmp] projected {len(out)} Domino slices onto "
          f"{group_size}-image groups")
    return out


def baseline_groups(ctx):
    out = []
    root = ctx.work.path("baselines")
    for path in sorted(glob.glob(os.path.join(root, "*", "groups.csv"))):
        method = os.path.basename(os.path.dirname(path))
        df = pd.read_csv(path)
        for gid, g in df.groupby("group_id"):
            g = g.sort_values("rank")
            out.append(dict(method=method, group_id=int(gid),
                            label=str(g["label"].iloc[0]),
                            image_ids=g["image_id"].to_numpy(),
                            scores=g["score"].to_numpy()))
        print(f"[cmp] loaded {df['group_id'].nunique()} groups from {method}")
    return out


def lfa_groups(path, method="lfa"):
    if not path:
        return []
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    gcol = next((cols[c] for c in ("group_id", "group", "cluster", "slice")
                 if c in cols), None)
    icol = next((cols[c] for c in ("image_id", "img_id", "image", "id")
                 if c in cols), None)
    if gcol is None or icol is None:
        raise SystemExit(
            f"--lfa-groups needs group_id and image_id columns; got "
            f"{list(df.columns)}")
    lcol = cols.get("label") or cols.get("name") or cols.get("description")
    out = []
    for gid, g in df.groupby(gcol):
        out.append(dict(method=method, group_id=int(gid),
                        label=str(g[lcol].iloc[0]) if lcol else f"group {gid}",
                        image_ids=g[icol].to_numpy().astype(int),
                        scores=np.zeros(len(g))))
    print(f"[cmp] loaded {len(out)} {method} groups from {path}")
    return out


# --------------------------------------------------------------------------
# random baselines
# --------------------------------------------------------------------------

def random_groups_fmr(ctx, size, n_draws, rng, restrict=None):
    """FMR of n_draws random groups; `restrict` is an optional pool of image_ids."""
    pool = ctx.pool if restrict is None else np.asarray(restrict)
    if len(pool) < size + 1:
        return np.array([])
    vals = []
    for _ in range(n_draws):
        pick = rng.choice(pool, size=size, replace=False)
        r = group_fmr(pick, ctx.face_embs, ctx.identity, ctx.tau)
        if r["n_pairs"] > 0 and np.isfinite(r["fmr"]):
            vals.append(r["fmr"])
    return np.asarray(vals)


def pool_by_race(ctx):
    """{race: array of pooled image_ids}, computed once and reused."""
    races = ctx.pool_race()
    return {r: ctx.pool[races == r] for r in np.unique(races)}


def race_matched_fmr(ctx, image_ids, n_draws, rng, by_race):
    """Random groups with the same race composition as `image_ids`."""
    races = pd.Series([ctx.race[int(i)] for i in image_ids])
    counts = races.value_counts().to_dict()
    if any(r not in by_race for r in counts):
        return np.array([])
    vals = []
    for _ in range(n_draws):
        pick = []
        for r, n in counts.items():
            src = by_race[r]
            if len(src) < n:
                return np.array([])
            pick.append(rng.choice(src, size=n, replace=False))
        pick = np.concatenate(pick)
        res = group_fmr(pick, ctx.face_embs, ctx.identity, ctx.tau)
        if res["n_pairs"] > 0 and np.isfinite(res["fmr"]):
            vals.append(res["fmr"])
    return np.asarray(vals)


# --------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", default="work")
    ap.add_argument("--group-size", type=int, default=100,
                    help="images per group. Domino slices and the random baselines "
                         "are projected to this size; groups.csv files are used as "
                         "written (truncated if longer).")
    ap.add_argument("--threshold", type=float, default=None,
                    help="override the global threshold; default is "
                         "metrics.json['global_threshold']. Table 2 of the paper "
                         "uses t = 0.2.")
    ap.add_argument("--protocol", default=None, choices=["official", "expanded"])
    ap.add_argument("--lfa-groups", default=None,
                    help="CSV with group_id,image_id[,label] from your own method")
    ap.add_argument("--lfa-name", default="lfa")
    ap.add_argument("--n-random", type=int, default=200,
                    help="uniform random groups for the 'Random group' baseline")
    ap.add_argument("--n-matched", type=int, default=20,
                    help="race-matched random draws per discovered group "
                         "(0 disables the lift_matched column)")
    ap.add_argument("--top-k", type=int, default=3,
                    help="per-method headline = mean FMR of its top-k groups")
    ap.add_argument("--out", default=None,
                    help="output directory (default <work>/report)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    ctx = load_context(args.work, args.threshold, args.protocol)
    rng = np.random.default_rng(args.seed)
    phrases, text_embs = load_bank(ctx.work)
    ref_proto = ctx.pool_clip().astype(np.float64).mean(axis=0)

    groups = (domino_groups(ctx, args.group_size) + baseline_groups(ctx)
              + lfa_groups(args.lfa_groups, args.lfa_name))
    if not groups:
        raise SystemExit(
            "no groups found. Run stage 05 (Domino), the baselines, and/or pass "
            "--lfa-groups.")

    print(f"\n[cmp] {args.n_random} uniform random groups of {args.group_size} "
          f"images ...")
    rnd = random_groups_fmr(ctx, args.group_size, args.n_random, rng)
    rnd_mean = float(rnd.mean()) if len(rnd) else float("nan")
    print(f"[cmp] random-group FMR: mean {rnd_mean:.3e}, median "
          f"{np.median(rnd):.3e}, p95 {np.percentile(rnd, 95):.3e} "
          f"(global FMR {ctx.global_fmr:.3e})")

    by_race = pool_by_race(ctx)
    rows = []
    for g in groups:
        ids = np.asarray(g["image_ids"], dtype=int)[:args.group_size]
        res = group_fmr(ids, ctx.face_embs, ctx.identity, ctx.tau)
        lo, hi = wilson(res["n_fm"], res["n_pairs"])
        races = pd.Series([ctx.race[int(i)] for i in ids]).value_counts(
            normalize=True)
        matched = (race_matched_fmr(ctx, ids, args.n_matched, rng, by_race)
                   if args.n_matched > 0 else np.array([]))
        m_mean = float(matched.mean()) if len(matched) else float("nan")

        label = g.get("label", "")
        if (not label or label.endswith(":")) and text_embs is not None:
            _, label = name_group(ctx.clip_embs[ids], ref_proto, text_embs, phrases)

        rows.append({
            "method": g["method"], "group_id": g["group_id"], "label": label,
            "n_images": res["n_images"], "n_pairs": res["n_pairs"],
            "n_fm": res["n_fm"], "fmr": res["fmr"],
            "fmr_ci_lo": lo, "fmr_ci_hi": hi,
            "lift_vs_random": res["fmr"] / rnd_mean if rnd_mean else np.nan,
            "lift_vs_global": (res["fmr"] / ctx.global_fmr
                               if ctx.global_fmr else np.nan),
            "random_matched_fmr": m_mean,
            "lift_matched": res["fmr"] / m_mean if m_mean else np.nan,
            "top_race": races.index[0], "top_race_share": float(races.iloc[0]),
            "n_identities": len({ctx.identity[int(i)] for i in ids}),
        })

    detail = pd.DataFrame(rows).sort_values(["method", "fmr"],
                                            ascending=[True, False])

    summary = []
    for method, g in detail.groupby("method"):
        g = g.sort_values("fmr", ascending=False)
        top = g.head(args.top_k)
        best = g.iloc[0]
        summary.append({
            "method": method, "n_groups": len(g),
            "best_fmr": float(best["fmr"]),
            "best_lift": float(best["lift_vs_random"]),
            "best_lift_matched": float(best["lift_matched"]),
            f"mean_fmr_top{args.top_k}": float(top["fmr"].mean()),
            f"mean_lift_top{args.top_k}": float(top["lift_vs_random"].mean()),
            "median_fmr": float(g["fmr"].median()),
            "n_groups_above_random_p95": int(
                (g["fmr"] > (np.percentile(rnd, 95) if len(rnd) else np.inf)).sum()),
            "best_label": str(best["label"])[:90],
            "best_top_race": best["top_race"],
            "best_top_race_share": float(best["top_race_share"]),
        })
    summary = pd.DataFrame(summary).sort_values("best_fmr", ascending=False)

    out_dir = args.out or ctx.work.report_dir
    os.makedirs(out_dir, exist_ok=True)
    detail.to_csv(os.path.join(out_dir, "comparison_groups.csv"), index=False)
    summary.to_csv(os.path.join(out_dir, "comparison_summary.csv"), index=False)

    # ---- markdown
    L = ["# Group-level comparison at a fixed global operating point\n"]
    L.append(f"- threshold **{ctx.tau:.4f}** (criterion `{ctx.criterion}`), "
             f"global FMR **{ctx.global_fmr:.3e}** over "
             f"{ctx.n_impostor:,} impostor pairs ({ctx.n_fm:,} false matches)")
    L.append(f"- group size **{args.group_size}** images; FMR is computed over all "
             f"within-group impostor pairs (~{args.group_size * (args.group_size - 1) // 2:,} "
             f"comparisons per group)")
    L.append(f"- random baseline: {len(rnd)} uniform groups, mean FMR "
             f"**{rnd_mean:.3e}** (median {np.median(rnd):.3e}, "
             f"p95 {np.percentile(rnd, 95):.3e})")
    L.append(f"- `lift` is vs the uniform random group; `lift_matched` is vs random "
             f"groups with the **same race composition**, i.e. the enrichment "
             f"beyond demographic concentration\n")

    L.append("## Summary\n")
    L.append("| method | groups | best FMR | lift | lift (race-matched) | "
             f"mean FMR top-{args.top_k} | > random p95 | best group |")
    L.append("|---|---|---|---|---|---|---|---|")
    for _, r in summary.iterrows():
        L.append(
            f"| `{r['method']}` | {r['n_groups']} | {r['best_fmr']:.3e} "
            f"| {r['best_lift']:.1f}x | {r['best_lift_matched']:.1f}x "
            f"| {r[f'mean_fmr_top{args.top_k}']:.3e} "
            f"| {r['n_groups_above_random_p95']} "
            f"| {r['best_top_race']} {r['best_top_race_share']:.0%} - "
            f"{r['best_label']} |")
    L.append("")

    L.append("## Every group\n")
    L.append("| method | # | FMR | 95% CI | lift | lift matched | pairs | FM | "
             "top race | ids | description |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for _, r in detail.iterrows():
        L.append(
            f"| `{r['method']}` | {r['group_id']} | {r['fmr']:.3e} "
            f"| [{r['fmr_ci_lo']:.1e}, {r['fmr_ci_hi']:.1e}] "
            f"| {r['lift_vs_random']:.1f}x | {r['lift_matched']:.1f}x "
            f"| {r['n_pairs']:,} | {r['n_fm']} "
            f"| {r['top_race']} {r['top_race_share']:.0%} | {r['n_identities']} "
            f"| {str(r['label'])[:80]} |")
    L.append("")

    md = os.path.join(out_dir, "comparison.md")
    with open(md, "w") as f:
        f.write("\n".join(L))
    with open(os.path.join(out_dir, "comparison_meta.json"), "w") as f:
        json.dump({"tau": ctx.tau, "criterion": ctx.criterion,
                   "global_fmr": ctx.global_fmr, "n_impostor": ctx.n_impostor,
                   "group_size": args.group_size,
                   "random_mean_fmr": rnd_mean,
                   "random_p95_fmr": float(np.percentile(rnd, 95)) if len(rnd) else None,
                   "n_random": int(len(rnd))}, f, indent=2)

    print()
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        print(summary.to_string(index=False))
    print(f"\n[cmp] wrote {md}")
    print(f"[cmp] wrote {os.path.join(out_dir, 'comparison_groups.csv')}")


if __name__ == "__main__":
    main()
