"""Method-agnostic evaluation of discovered slices.

Consumes the (slices.csv, members.csv) pair that every baseline in this package
writes, so DOMINO -- or your own method -- can be scored on exactly the same
footing by emitting the same two files.

Metrics
-------
Precision@k, the standard slice-discovery metric (DOMINO, FACTS, ViG-Bias): for
each discovered slice, take its top-k members and compute the fraction that belong
to a ground-truth attribute group; the slice is credited with its best-matching
group.  The headline number is the max over discovered slices, matching how the
published tables are computed; the mean over slices is also reported because the
max rewards a method for one lucky slice out of ten.

Precision@k alone is misleading and this script therefore always reports the
prevalence-corrected version alongside it.  If 47% of your RFW pairs are African,
a slice that is 47% African has discovered nothing, yet it scores Precision@k =
0.47.  ``precision_lift`` (= precision / prevalence) is the number that actually
distinguishes a method from chance, and ``hypergeom_p`` gives the exact
one-sided p-value for the enrichment.

Also reported per slice: error-rate lift (does the slice actually contain errors?)
and, when --restrict-to-errors is set, precision computed within the error subset
only, which is the harsher variant.

Ground-truth attributes
-----------------------
--attributes is a CSV keyed by image_id with one column per binary attribute.
Values > 0 count as positive, so MAAD-Face's {1, -1, 0} encoding works directly and
so does plain {0, 1}.  For RFW the minimum viable ground truth is the four
subset labels; for a richer set, join MAAD-Face annotations or predicted attributes
(and say in the paper which one you used, since predicted attributes cap the
achievable precision).

Pair units are mapped to attribute groups by ``--pair-attr-mode``:
  both     (default) the pair is in group g iff both faces have attribute g.  This
           is the convention in demographic FR evaluation, and it is the only one
           that makes sense for relational slices ("both subjects wear glasses").
  either   the pair is in group g iff at least one face has it.  Use for slices you
           expect to be driven by a single bad probe image.

Usage:
    python -m sdm_baselines.evaluate \
        --runs runs/svm runs/b2t runs/spotlight runs/domino \
        --pairs work/pairs.csv --images work/images.csv \
        --attributes work/attributes.csv --out runs/eval
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import hypergeom

try:
    from .common import load_manifest, load_pairs
except ImportError:
    from common import load_manifest, load_pairs


def load_attributes(path, images):
    att = pd.read_csv(path)
    lower = {c.lower(): c for c in att.columns}
    key = None
    for cand in ("image_id", "id", "idx", "index"):
        if cand in lower:
            key = lower[cand]
            break
    if key is None:
        for cand in ("path", "file", "filename", "image"):
            if cand in lower:
                key = lower[cand]
                break
        if key is None:
            raise SystemExit("attributes CSV needs an image_id or path column")
        lut = {}
        for iid, p in zip(images["image_id"], images["path"]):
            p = str(p).replace("\\", "/")
            lut[p] = iid
            lut[os.path.basename(p)] = iid
            lut["/".join(p.rstrip("/").split("/")[-2:])] = iid
        ids = []
        for v in att[key].astype(str):
            v = v.replace("\\", "/")
            hit = (lut.get(v) or lut.get(os.path.basename(v))
                   or lut.get("/".join(v.rstrip("/").split("/")[-2:])))
            ids.append(-1 if hit is None else int(hit))
        att = att.assign(image_id=ids)
        att = att[att.image_id >= 0]
        key = "image_id"

    cols = [c for c in att.columns if c != key]
    n = int(images["image_id"].max()) + 1
    out = {}
    for c in cols:
        v = pd.to_numeric(att[c], errors="coerce").to_numpy()
        if not np.isfinite(v).any():
            continue
        arr = np.zeros(n, dtype=bool)
        idx = att[key].astype(int).to_numpy()
        keep = (idx >= 0) & (idx < n) & np.isfinite(v)
        arr[idx[keep]] = v[keep] > 0
        if 0 < arr.sum() < n:  # skip constant attributes
            out[c] = arr
    if not out:
        raise SystemExit("no usable binary attribute columns found")
    print(f"[eval] {len(out)} ground-truth attribute groups")
    return out


def pair_membership(attr_vec, pairs, mode="both"):
    a = attr_vec[pairs["image_id_a"].to_numpy()]
    b = attr_vec[pairs["image_id_b"].to_numpy()]
    return (a & b) if mode == "both" else (a | b)


def evaluate_run(run_dir, pairs, images, attrs, args):
    sp = os.path.join(run_dir, "slices.csv")
    mp = os.path.join(run_dir, "members.csv")
    if not (os.path.exists(sp) and os.path.exists(mp)):
        print(f"[eval] skipping {run_dir}: no slices.csv/members.csv")
        return None
    try:
        slices = pd.read_csv(sp)
        members = pd.read_csv(mp)
    except pd.errors.EmptyDataError:
        print(f"[eval] skipping {run_dir}: slices.csv/members.csv are empty")
        return None
    if slices.empty or members.empty:
        print(f"[eval] skipping {run_dir}: no slices to evaluate")
        return None

    # pool definitions per (pair_type, unit_type): what the slice was drawn from
    pool_cache, group_cache = {}, {}

    def pools(pair_type, unit_type):
        key = (pair_type, unit_type)
        if key in pool_cache:
            return pool_cache[key], group_cache[key]
        same = pairs["same"].to_numpy(dtype=bool)
        if pair_type == "genuine":
            pm = same
        elif pair_type == "impostor":
            pm = ~same
        else:
            pm = np.ones(len(pairs), dtype=bool)
        if unit_type == "pair":
            pool = np.flatnonzero(pm)
            groups = {name: pair_membership(v, pairs, args.pair_attr_mode)
                      for name, v in attrs.items()}
            groups = {k: v[pm] for k, v in groups.items()}
            pool_index = {u: i for i, u in enumerate(pool)}
        else:
            sub = pairs.loc[pm]
            pool = np.unique(np.concatenate([sub["image_id_a"].to_numpy(),
                                             sub["image_id_b"].to_numpy()]))
            groups = {name: v[pool] for name, v in attrs.items()}
            pool_index = {u: i for i, u in enumerate(pool)}
        pool_cache[key] = (pool, pool_index)
        group_cache[key] = groups
        return pool_cache[key], groups

    rows = []
    for _, s in slices.iterrows():
        sid = s["slice_id"]
        mem = members[members.slice_id == sid].sort_values("rank_in_slice")
        if len(mem) == 0:
            continue
        k = int(args.k) if args.k else len(mem)
        top = mem.head(k)["unit_id"].to_numpy()
        (pool, pool_index), groups = pools(s.get("pair_type", "all"),
                                           s.get("unit_type", "pair"))
        local = np.array([pool_index[u] for u in top if u in pool_index])
        if len(local) == 0:
            continue

        if args.min_err_lift and pd.notna(s.get("lift")) and \
                float(s.get("lift")) < args.min_err_lift:
            continue

        N = len(pool)
        best = None
        candidates = (groups if not args.gt_groups
                      else {g: groups[g] for g in args.gt_groups if g in groups})
        if args.gt_groups and not candidates:
            raise SystemExit(
                f"--gt-groups {args.gt_groups} matched no attribute columns; "
                f"available: {sorted(groups)}")
        for name, mask in candidates.items():
            K = int(mask.sum())
            if K == 0 or K == N:
                continue
            hit = int(mask[local].sum())
            prec = hit / len(local)
            prev = K / N
            p = float(hypergeom.sf(hit - 1, N, K, len(local)))
            cand = dict(best_group=name, precision=prec, prevalence=prev,
                        precision_lift=prec / prev if prev > 0 else np.nan,
                        hypergeom_p=p, group_size=K, n_eval=len(local))
            if best is None or cand["precision"] > best["precision"]:
                best = cand
        if best is None:
            continue
        rows.append({
            "run": os.path.basename(run_dir.rstrip("/")),
            "method": s.get("method", "?"),
            "slice_id": sid,
            "pair_type": s.get("pair_type", "all"),
            "unit_type": s.get("unit_type", "pair"),
            "rank": s.get("rank", 0),
            "description": s.get("description", ""),
            "slice_err_rate": s.get("err_rate", np.nan),
            "base_err_rate": s.get("base_err_rate", np.nan),
            "err_lift": s.get("lift", np.nan),
            **best,
        })

    if not rows:
        return None
    return pd.DataFrame(rows)


def summarise(df, hit_threshold=0.5):
    out = []
    for (run, method, pt, ut), g in df.groupby(
            ["run", "method", "pair_type", "unit_type"], dropna=False):
        best = g.loc[g["precision"].idxmax()]
        recovered = g.loc[g["precision"] >= hit_threshold, "best_group"].nunique()
        out.append(dict(
            run=run, method=method, pair_type=pt, unit_type=ut,
            n_slices=len(g),
            precision_at_k=float(g["precision"].max()),
            mean_precision_at_k=float(g["precision"].mean()),
            precision_lift=float(best["precision_lift"]),
            best_group=best["best_group"],
            best_description=best["description"],
            best_slice_err_lift=float(best["err_lift"])
            if pd.notna(best["err_lift"]) else np.nan,
            mean_err_lift=float(g["err_lift"].mean(skipna=True)),
            n_groups_recovered=int(recovered),
            n_significant=int((g["hypergeom_p"] < 0.05).sum()),
        ))
    return pd.DataFrame(out).sort_values(
        ["pair_type", "unit_type", "precision_at_k"], ascending=[True, True, False])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True,
                    help="run directories containing slices.csv + members.csv")
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--attributes", required=True,
                    help="CSV: image_id (or path) + one binary column per attribute")
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=None,
                    help="evaluate the top-k members of each slice (default: all)")
    ap.add_argument("--pair-attr-mode", default="both", choices=["both", "either"])
    ap.add_argument("--hit-threshold", type=float, default=0.5)
    ap.add_argument("--gt-groups", nargs="*", default=None,
                    help="restrict ground truth to these attribute columns. Use this "
                         "for the published protocol, where Precision@k is measured "
                         "against a *known* bias-conflicting group. With the default "
                         "(all columns) a method is credited for any coherent group "
                         "it happens to isolate, error-related or not.")
    ap.add_argument("--min-err-lift", type=float, default=0.0,
                    help="ignore slices whose error-rate lift is below this. Set to "
                         "1.0 to stop crediting methods for coherent but harmless "
                         "slices (e.g. a pure 'blurry photo' group with no elevated "
                         "error rate).")
    args = ap.parse_args(argv)

    images = load_manifest(args.images)
    pairs = load_pairs(args.pairs, images)
    attrs = load_attributes(args.attributes, images)

    frames = [f for f in (evaluate_run(r, pairs, images, attrs, args)
                          for r in args.runs) if f is not None]
    if not frames:
        raise SystemExit("nothing to evaluate")
    per_slice = pd.concat(frames, ignore_index=True)
    summary = summarise(per_slice, args.hit_threshold)

    os.makedirs(args.out, exist_ok=True)
    per_slice.to_csv(os.path.join(args.out, "per_slice.csv"), index=False)
    summary.to_csv(os.path.join(args.out, "summary.csv"), index=False)

    with pd.option_context("display.width", 200, "display.max_columns", 40):
        print("\n=== summary ===")
        print(summary.to_string(index=False))
    print(f"\n[out] {os.path.join(args.out, 'per_slice.csv')}")
    print(f"[out] {os.path.join(args.out, 'summary.csv')}")


if __name__ == "__main__":
    main()
