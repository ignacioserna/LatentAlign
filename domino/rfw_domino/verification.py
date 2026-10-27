"""Stage 03 - score verification pairs and derive per-pair model outputs.

For Domino we need, per pair: a binary target (same / different person), the
model's decision at a single deployed operating point, and a calibrated
probability. So we
  1. compute cosine similarity between the two face embeddings of each pair;
  2. pick ONE global decision threshold on the pooled pairs of all four races
     (accuracy-maximizing by default, or equal-error-rate with --criterion eer).
     A single operating point mirrors deployment and is exactly where racial
     threshold mismatch shows up as concentrated errors;
  3. calibrate similarity -> P(same) with Platt scaling (1-d logistic
     regression, plain-numpy IRLS);
  4. also report the standard per-race 10-fold LFW-style accuracies so you can
     sanity-check the face pipeline against published RFW numbers.
"""

import argparse

import numpy as np
import pandas as pd

from .io_utils import Work, save_json


# ---------------------------------------------------------------- thresholds

def best_threshold_accuracy(sims: np.ndarray, y: np.ndarray):
    """Accuracy-maximizing threshold (midpoints between sorted sims)."""
    order = np.argsort(sims)
    s, t = sims[order], y[order]
    n = len(s)
    n_pos = t.sum()
    # threshold below all sims -> predict everything "same"
    # moving the threshold above sim_i flips prediction of i to "different"
    cum_pos = np.cumsum(t)          # positives at or below index i -> misses
    cum_neg = np.cumsum(1 - t)      # negatives at or below index i -> correct rejections
    acc = np.empty(n + 1)
    acc[0] = n_pos / n
    acc[1:] = (n_pos - cum_pos + cum_neg) / n
    k = int(np.argmax(acc))
    if k == 0:
        thr = s[0] - 1e-6
    elif k == n:
        thr = s[-1] + 1e-6
    else:
        thr = (s[k - 1] + s[k]) / 2.0
    return float(thr), float(acc[k])


def eer_threshold(sims: np.ndarray, y: np.ndarray):
    """Threshold where false match rate ~= false non-match rate."""
    thrs = np.unique(sims)
    pos, neg = sims[y == 1], sims[y == 0]
    fnmr = np.searchsorted(np.sort(pos), thrs, side="left") / max(1, len(pos))
    fmr = 1.0 - np.searchsorted(np.sort(neg), thrs, side="right") / max(1, len(neg))
    k = int(np.argmin(np.abs(fnmr - fmr)))
    return float(thrs[k]), float((fnmr[k] + fmr[k]) / 2.0)


def platt_scale(sims: np.ndarray, y: np.ndarray, n_iter: int = 100):
    """Fit P(same|s) = sigmoid(a*s + b) by IRLS; returns (a, b, p)."""
    x = np.stack([sims, np.ones_like(sims)], axis=1).astype(np.float64)
    w = np.zeros(2)
    lam = 1e-6
    for _ in range(n_iter):
        z = x @ w
        p = 1.0 / (1.0 + np.exp(-z))
        g = x.T @ (p - y) + lam * w
        r = np.maximum(p * (1 - p), 1e-9)
        h = (x * r[:, None]).T @ x + lam * np.eye(2)
        step = np.linalg.solve(h, g)
        w -= step
        if np.max(np.abs(step)) < 1e-10:
            break
    p = 1.0 / (1.0 + np.exp(-(x @ w)))
    return float(w[0]), float(w[1]), p.astype(np.float32)


def tenfold_accuracy(df: pd.DataFrame):
    """LFW-protocol accuracy: threshold chosen on 9 folds, tested on the 10th."""
    accs = []
    for fold in sorted(df["fold"].unique()):
        train = df[df["fold"] != fold]
        test = df[df["fold"] == fold]
        thr, _ = best_threshold_accuracy(
            train["sim"].to_numpy(), train["same"].to_numpy()
        )
        pred = (test["sim"].to_numpy() >= thr).astype(int)
        accs.append(float((pred == test["same"].to_numpy()).mean()))
    return float(np.mean(accs)), float(np.std(accs))


# --------------------------------------------------------------------- main

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--work", default="work")
    ap.add_argument("--criterion", choices=["accuracy", "eer", "far", "fixed"],
                    default="accuracy",
                    help="how the single global threshold is chosen; 'far' "
                         "fixes the global false match rate at --far-target "
                         "(security-style operating point, where racial FMR "
                         "disparity is most pronounced)")
    ap.add_argument("--far-target", type=float, default=1e-3,
                    help="target global FMR for --criterion far")
    ap.add_argument("--threshold", type=float, default=None,
                    help="cosine threshold for --criterion fixed, e.g. the global "
                         "FMR = 1e-3 threshold over all RFW impostor pairs computed "
                         "by LatentAlign's experiments/global_thresholds.py")
    args = ap.parse_args(argv)

    work = Work(args.work)
    pairs = work.load_pairs()
    embs = work.load_embs("face")  # already L2-normalized

    e1 = embs[pairs["img_id1"].to_numpy()]
    e2 = embs[pairs["img_id2"].to_numpy()]
    sims = np.sum(e1 * e2, axis=1).astype(np.float32)

    # pairs touching an image without an embedding (e.g. detection failed in a
    # precomputed extraction -> NaN row from stage 02b) are dropped
    valid = np.isfinite(sims)
    if not valid.all():
        print(f"[verify] dropping {int((~valid).sum())}/{len(pairs)} pairs "
              f"with a missing embedding")
        pairs = pairs.loc[valid].reset_index(drop=True)
        sims = sims[valid]

    y = pairs["same"].to_numpy().astype(int)
    pairs["sim"] = sims

    if args.criterion == "accuracy":
        thr, glob_acc = best_threshold_accuracy(sims, y)
    elif args.criterion == "eer":
        thr, eer = eer_threshold(sims, y)
        glob_acc = float(((sims >= thr).astype(int) == y).mean())
    elif args.criterion == "fixed":
        if args.threshold is None:
            raise SystemExit("--criterion fixed needs --threshold")
        thr = float(args.threshold)
        glob_acc = float(((sims >= thr).astype(int) == y).mean())
        neg = sims[y == 0]
        print(f"[verify] fixed threshold {thr:.4f}: FMR on this pair set = "
              f"{float((neg >= thr).mean()):.2e}, {int((neg >= thr).sum())} false matches "
              f"out of {len(neg)} impostor pairs")
    else:  # far: threshold at a fixed global false match rate
        neg = sims[y == 0]
        thr = float(np.quantile(neg, 1.0 - args.far_target, method="higher"))
        glob_acc = float(((sims >= thr).astype(int) == y).mean())
        n_fm = int((neg >= thr).sum())
        print(f"[verify] FMR-targeted threshold: global FMR = "
              f"{float((neg >= thr).mean()):.2e} (target {args.far_target:.0e}), "
              f"{n_fm} false matches out of {len(neg)} impostor pairs")
        if n_fm < 100:
            print(f"[verify] WARNING: only {n_fm} false matches exist at this "
                  f"operating point. Per-slice error rates estimated from "
                  f"{n_fm} events cannot separate slices; ~{100/args.far_target:,.0f} "
                  f"impostor pairs would be needed for 100 events. Either "
                  f"sample more impostor pairs or relax --far-target.")


    a, b, p_same = platt_scale(sims, y.astype(np.float64))
    pred = (sims >= thr).astype(int)
    pairs["p_same"] = p_same
    pairs["pred"] = pred
    pairs["correct"] = (pred == y).astype(int)

    metrics = {
        "criterion": args.criterion,
        "global_threshold": thr,
        "platt_a": a, "platt_b": b,
        "overall": {
            "n": int(len(pairs)),
            "accuracy_at_global_thr": glob_acc,
            "fnmr_at_global_thr": float((pred[y == 1] == 0).mean()),
            "fmr_at_global_thr": float((pred[y == 0] == 1).mean()),
        },
        "per_race": {},
    }
    print(f"[verify] global threshold = {thr:.4f} "
          f"({args.criterion}), overall acc = {glob_acc:.4f}")
    header = (f"{'race':<10} {'n':>6} {'acc@global':>11} {'FNMR':>7} {'FMR':>7} "
              f"{'acc(10-fold)':>13}")
    print(header)
    for race, g in pairs.groupby("race"):
        yr = g["same"].to_numpy()
        pr = g["pred"].to_numpy()
        acc10, std10 = tenfold_accuracy(g)
        m = {
            "n": int(len(g)),
            "accuracy_at_global_thr": float((pr == yr).mean()),
            "fnmr_at_global_thr": float((pr[yr == 1] == 0).mean()),
            "fmr_at_global_thr": float((pr[yr == 0] == 1).mean()),
            "tenfold_accuracy_mean": acc10,
            "tenfold_accuracy_std": std10,
        }
        metrics["per_race"][race] = m
        print(f"{race:<10} {m['n']:>6} {m['accuracy_at_global_thr']:>11.4f} "
              f"{m['fnmr_at_global_thr']:>7.4f} {m['fmr_at_global_thr']:>7.4f} "
              f"{acc10:>9.4f}±{std10:.3f}")

    pairs.to_csv(work.pairs_scored_csv, index=False)
    save_json(work.metrics_json, metrics)
    print(f"[verify] wrote {work.pairs_scored_csv} and {work.metrics_json}")


if __name__ == "__main__":
    main()
