"""Stage 05b - Failure Directions baseline (Jain, Lawrence, Moitra & Madry, ICLR 2023).

What the method does: fit a linear model in CLIP space that predicts the model's
loss from an image's CLIP embedding. The coefficient vector is the "failure
direction"; ranking images by their predicted loss puts the worst-affected ones at
the top, and the top slice is the discovered failure mode.

How it produces many candidate groups here:

  bands       one direction gives one ranking, so the candidates are consecutive
              disjoint blocks of --group-size images down that ranking: images
              1-100, 101-200, and so on. This is what a single failure direction
              actually offers, stated honestly.
  directions  --n-directions > 1 adds diversity by PLS-style deflation. Note that
              deflating only the features (the obvious thing, and what an earlier
              version of this file did) barely changes anything: with 512 CLIP
              dimensions the refit predicts the same loss from the remaining 511 and
              you get near-identical rankings. The **target** has to be deflated too
              -- each new direction is fitted to the residual loss that previous
              directions could not explain. That is the difference between ten
              copies of one group and ten different groups.

Total candidates = n_directions x n_bands.

Two documented deviations from the paper, worth stating accurately:

  * --target margin (default) fits ridge regression on the continuous per-image
    loss instead of LinearSVC on a binary right/wrong label. At FMR = 1e-3 about 1%
    of images carry any false match, so the binary label is nearly constant and a
    balanced SVM fits noise. The coefficient vector plays the same role.
    --target error gives the literal published estimator.

  * Predicted losses are out-of-fold with folds assigned **by identity**. RFW has
    many images per person, so a random split lets the model memorise identities and
    return an identity lookup dressed up as a failure direction. The printed
    `fit` value (Spearman rho out-of-fold, or AUC for --target error) is the check:
    near 0 means CLIP space contains no linear failure direction, which is a result.

    python -m rfw_domino.baseline_svm --work work --group-size 100 \
        --n-directions 5 --n-bands 40
"""

import argparse

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from .image_signal import (add_common_args, coherence, load_context, make_scorer,
                           per_image_signal, write_groups)


def _fit(X, y, binary, seed):
    if binary:
        if len(np.unique(y)) < 2:
            return None
        clf = LinearSVC(C=0.01, class_weight="balanced", max_iter=5000,
                        dual="auto", random_state=seed)
        clf.fit(X, y.astype(int))
        return clf.coef_.ravel().astype(np.float64), clf.decision_function
    clf = RidgeCV(alphas=np.logspace(-2, 4, 13))
    clf.fit(X, y)
    return np.asarray(clf.coef_, dtype=np.float64).ravel(), clf.predict


def _oof(X, y, folds, binary, seed):
    scores = np.full(len(X), np.nan)
    for f in np.unique(folds):
        tr, te = folds != f, folds == f
        if tr.sum() < 50 or te.sum() < 5:
            continue
        if binary and len(np.unique(y[tr])) < 2:
            continue
        sc = StandardScaler().fit(X[tr])
        fit = _fit(sc.transform(X[tr]), y[tr], binary, seed)
        if fit is None:
            continue
        scores[te] = fit[1](sc.transform(X[te]))
    ok = np.isfinite(scores)
    if ok.sum() < 10:
        return scores, np.nan
    if binary:
        q = (roc_auc_score(y[ok], scores[ok])
             if len(np.unique(y[ok])) > 1 else np.nan)
    else:
        q = float(spearmanr(y[ok], scores[ok]).statistic)
    return scores, q


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--target", default="margin", choices=["margin", "error"])
    ap.add_argument("--error-quantile", type=float, default=0.1)
    ap.add_argument("--n-directions", type=int, default=5,
                    help="failure directions, found by PLS deflation of both the "
                         "features and the residual loss")
    ap.add_argument("--n-bands", type=int, default=None,
                    help="disjoint bands per direction (default: "
                         "n_candidates / n_directions)")
    ap.add_argument("--n-folds", type=int, default=4)
    args = ap.parse_args(argv)

    ctx = load_context(args.work, args.threshold, args.protocol)
    sig = per_image_signal(ctx, args.loss, args.knn_k)
    scorer = make_scorer(ctx)
    binary = args.target == "error"
    rng = np.random.default_rng(args.seed)
    n_bands = args.n_bands or max(1, args.n_candidates // max(1, args.n_directions))

    img_ids = sig["image_id"].to_numpy()
    X = ctx.clip_embs[img_ids].astype(np.float64)
    loss = sig["loss"].to_numpy()
    y = (loss.copy() if not binary else
         ((sig["n_fm"].to_numpy() > 0).astype(float) if args.loss == "fm_count"
          else (loss >= np.quantile(loss, 1 - args.error_quantile)).astype(float)))

    ident = sig["identity"].to_numpy()
    uniq = np.unique(ident)
    assign = dict(zip(uniq, rng.integers(0, args.n_folds, size=len(uniq))))
    folds = np.array([assign[i] for i in ident])

    groups, detail = [], []
    Xw, yw = X.copy(), y.copy()
    gid = 0
    for d in range(args.n_directions):
        oof, quality = _oof(Xw, yw, folds, binary, args.seed + d)
        if not np.isfinite(oof).any():
            print("[svm] no usable identity-disjoint fold split; stopping")
            break
        order = np.argsort(-np.where(np.isfinite(oof), oof, -np.inf))
        q = "n/a" if not np.isfinite(quality) else f"{quality:+.3f}"
        print(f"[svm] direction {d}: out-of-fold fit {q}")

        for b in range(n_bands):
            lo, hi = b * args.group_size, (b + 1) * args.group_size
            if hi > len(order):
                break
            top = order[lo:hi]
            ids = img_ids[top]
            name = ""
            if scorer is not None:
                _, name = scorer.name(ctx.clip_embs[ids])
            groups.append(dict(group_id=gid, native_rank=gid,
                               label=f"dir{d}.band{b}: {name}",
                               image_ids=ids, scores=oof[top]))
            detail.append(dict(group_id=gid, direction=d, band=b,
                               oof_fit=quality, target=args.target,
                               mean_loss=float(loss[top].mean()),
                               base_loss=float(loss.mean()),
                               coherence=coherence(ctx.clip_embs[ids]),
                               description=name))
            gid += 1

        if d + 1 < args.n_directions:
            sc = StandardScaler().fit(Xw)
            fit = _fit(sc.transform(Xw), yw, binary, args.seed + d)
            if fit is None:
                break
            w = fit[0] / np.maximum(sc.scale_, 1e-12)
            t = Xw @ w
            tt = float(t @ t)
            if tt < 1e-12:
                break
            Xw = Xw - np.outer(t, t @ Xw) / tt      # deflate the features
            yw = yw - t * float(t @ yw) / tt        # AND the target -- the fix

    if not groups:
        raise SystemExit("[svm] no groups produced")
    print(f"[svm] {args.n_directions} directions x {n_bands} bands = "
          f"{len(groups)} candidate groups")
    write_groups(ctx.work, "failure_directions", groups,
                 params=vars(args) | {"tau": ctx.tau, "global_fmr": ctx.global_fmr},
                 extra_table=pd.DataFrame(detail))


if __name__ == "__main__":
    main()
