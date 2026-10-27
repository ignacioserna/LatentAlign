"""Failure Directions (Jain, Lawrence, Moitra, Madry -- ICLR 2023) for face verification.

Original method: for each class, fit a linear SVM in CLIP latent space separating
correctly- from incorrectly-classified examples.  The SVM's normal vector is the
"failure direction"; its decision value ranks examples by how strongly they belong
to the failure mode, and the top-scoring examples form the discovered slice.

Adaptation to verification, and why each choice was made:

  * "class" becomes pair type.  Verification is binary classification over pairs
    (genuine vs impostor), so we fit one SVM per pair type.  Pooling them would
    average a false-non-match direction with a false-match direction, which point
    in unrelated directions in attribute space.

  * The example becomes a pair.  Represented by a symmetric function of the two
    CLIP embeddings (see common.pair_rep).  The default keeps a CLIP-aligned block
    first, so the learned normal can still be named against a text vocabulary.

  * Binary error -> signed margin.  At operational thresholds errors are far too
    rare for a balanced SVM to be stable.  With --target margin we swap LinearSVC
    for ridge regression on the continuous margin; the coefficient vector plays
    exactly the same role as the SVM normal and the method is otherwise unchanged.
    This is the single most important adaptation: run it both ways and report both.

  * Identity-grouped, out-of-fold scoring.  Faces leak: the same identity appears
    in many pairs, so an in-sample or randomly-split SVM will happily memorise
    identities and report a "failure direction" that is really an identity lookup.
    All reported decision values here are out-of-fold, with folds assigned by
    identity.  The held-out AUC (or Spearman rho for the margin target) is printed
    as the coherence check: near 0.5 (or 0) means there is no linear failure
    direction to find, which is a legitimate finding, not a bug.

  * Multiple slices (extension).  The paper yields one direction per class.  To get
    --n-slices > 1 we deflate: after fitting w_j, project the features onto the
    orthogonal complement of w_j and refit.  Flagged here so you can describe it
    accurately in the paper.

Usage:
    python -m sdm_baselines.failure_directions \
        --pairs work/pairs.csv --images work/images.csv \
        --face-embs work/face_embs.npy --desc-embs work/clip_embs.npy \
        --out runs/svm --target margin --n-slices 3
"""

from __future__ import annotations

import argparse

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

try:
    from .common import (add_common_args, build_pairset, clip_block, describe,
                         dump_config, granularities, l2_normalize, name_vector,
                         pair_types, resolve_text_embeddings, slice_stats,
                         write_slices)
except ImportError:  # allow `python failure_directions.py`
    from common import (add_common_args, build_pairset, clip_block, describe,
                        dump_config, granularities, l2_normalize, name_vector,
                        pair_types, resolve_text_embeddings, slice_stats,
                        write_slices)


def _fit_one(X, y, binary, seed):
    """Fit a single linear model; return (weight_vector, intercept, predict_fn)."""
    if binary:
        if len(np.unique(y)) < 2:
            return None
        clf = LinearSVC(C=0.01, class_weight="balanced", max_iter=5000,
                        dual="auto", random_state=seed)
        clf.fit(X, y.astype(int))
        w = clf.coef_.ravel().astype(np.float64)
        return w, float(clf.intercept_[0]), lambda Z: clf.decision_function(Z)
    clf = RidgeCV(alphas=np.logspace(-2, 4, 13))
    clf.fit(X, y)
    w = np.asarray(clf.coef_, dtype=np.float64).ravel()
    return w, float(clf.intercept_), lambda Z: clf.predict(Z)


def _oof_scores(X, y, groups, binary, seed, n_folds=2):
    """Out-of-fold decision values, with folds assigned by identity group."""
    scores = np.full(len(X), np.nan)
    folds = np.unique(groups[groups >= 0])
    if len(folds) < 2:
        # not enough clean groups: fall back to in-sample, and say so loudly
        scaler = StandardScaler().fit(X)
        fit = _fit_one(scaler.transform(X), y, binary, seed)
        if fit is None:
            return scores, np.nan
        print("[svm] WARNING: could not build identity-disjoint folds; "
              "decision values are IN-SAMPLE and will be optimistic")
        return fit[2](scaler.transform(X)), np.nan

    for f in folds:
        tr = (groups != f) & (groups >= 0)
        te = groups == f
        if tr.sum() < 20 or te.sum() < 5:
            continue
        if binary and len(np.unique(y[tr])) < 2:
            continue
        scaler = StandardScaler().fit(X[tr])
        fit = _fit_one(scaler.transform(X[tr]), y[tr], binary, seed)
        if fit is None:
            continue
        _, _, pred = fit
        scores[te] = pred(scaler.transform(X[te]))

    ok = np.isfinite(scores)
    if ok.sum() < 10:
        coherence = np.nan
    elif binary:
        coherence = (roc_auc_score(y[ok], scores[ok])
                     if len(np.unique(y[ok])) > 1 else np.nan)
    else:
        coherence = float(spearmanr(y[ok], scores[ok]).statistic)
    return scores, coherence


def _deflate(X, w):
    """Remove the component of X along w (used to find additional directions)."""
    wn = w / max(np.linalg.norm(w), 1e-12)
    return X - np.outer(X @ wn, wn)


def run_block(ps, args, pair_type, granularity, terms, text_embs, slices, members,
              slice_counter):
    binary = args.target == "error"

    if granularity == "pair":
        m = ps.mask(pair_type)
        X = ps.rep(args.rep, pair_type).astype(np.float64)
        y = ps.target(args.target, pair_type)
        loss = ps.loss[m]
        is_err = ps.is_error[m]
        unit_ids = np.flatnonzero(m)
        groups = ps.identity_folds(2, args.seed)[m]
        cb = clip_block(args.rep, ps.desc_embs.shape[1])
        rep_for_name = ps.rep("mean", pair_type)
    else:
        view = ps.image_view(pair_type, args.target)
        unit_ids = view["image_id"].to_numpy()
        X = ps.desc_embs[unit_ids].astype(np.float64)
        is_err = (view["n_errors"].to_numpy() > 0)
        loss = view["value"].to_numpy()
        y = is_err.astype(float) if binary else loss
        ident = ps.images.set_index("image_id")["identity"].reindex(unit_ids).to_numpy()
        rng = np.random.default_rng(args.seed)
        uniq = np.unique(ident)
        assign = dict(zip(uniq, rng.integers(0, 2, size=len(uniq))))
        groups = np.array([assign[i] for i in ident])
        cb = slice(0, ps.desc_embs.shape[1])
        rep_for_name = X

    if len(X) < 50:
        print(f"[svm] skipping {pair_type}/{granularity}: only {len(X)} units")
        return
    if binary and len(np.unique(y)) < 2:
        print(f"[svm] skipping {pair_type}/{granularity}: the error label is constant "
              f"at tau ({int(np.sum(y))}/{len(y)} errors). Use --target margin, or a "
              f"threshold that puts this pair type in a non-degenerate regime.")
        return
    if not binary and np.std(y) < 1e-12:
        print(f"[svm] skipping {pair_type}/{granularity}: the margin is constant")
        return

    base_err = float(is_err.mean())
    base_margin = float(loss.mean())
    k = min(args.slice_size, max(10, len(X) // 20))

    Xw = X.copy()
    for j in range(args.n_slices):
        oof, coherence = _oof_scores(Xw, y, groups, binary, args.seed)
        if not np.isfinite(oof).any():
            break
        scaler = StandardScaler().fit(Xw)
        fit = _fit_one(scaler.transform(Xw), y, binary, args.seed)
        if fit is None:
            break
        w_std, _, _ = fit
        # undo standardisation so the direction lives in the original feature space
        w = w_std / np.maximum(scaler.scale_, 1e-12)

        order = np.argsort(-np.where(np.isfinite(oof), oof, -np.inf))
        top = order[:k]
        st = slice_stats(top, is_err, loss, base_err, base_margin)

        # name the direction using only its CLIP-aligned block
        desc_dir, desc_cent = [], []
        if cb is not None and text_embs is not None:
            desc_dir = name_vector(w[cb], text_embs, terms, topk=5, signed=True)
            cent = l2_normalize(rep_for_name[top].mean(axis=0)[None, :])[0]
            desc_cent = name_vector(cent, text_embs, terms, topk=5, signed=False)

        sid = f"svm.{pair_type}.{granularity}.{j}"
        slices.append(dict(
            slice_id=sid, pair_type=pair_type, unit_type=granularity, rank=j,
            k=k, coherence=coherence, target=args.target, rep=args.rep,
            description=describe(desc_cent) if desc_cent else "",
            direction_pos=describe(desc_dir[:5]) if desc_dir else "",
            direction_neg=describe(desc_dir[5:]) if len(desc_dir) > 5 else "",
            **st))
        for r, t in enumerate(top):
            members.append(dict(slice_id=sid, unit_type=granularity,
                                unit_id=int(unit_ids[t]), rank_in_slice=r,
                                weight=float(oof[t])))
        coh = "n/a" if not np.isfinite(coherence) else f"{coherence:.3f}"
        print(f"[svm] {sid}: size={st['size']} err={st['err_rate']:.3%} "
              f"(base {base_err:.3%}, lift {st['lift']:.2f}x) coherence={coh} "
              f"| {slices[-1]['description']}")
        slice_counter[0] += 1

        if j + 1 < args.n_slices:
            Xw = _deflate(Xw, w)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    args = ap.parse_args(argv)

    ps = build_pairset(args)
    try:
        terms, text_embs = resolve_text_embeddings(args, ps.desc_embs.shape[1])
    except SystemExit as e:
        print(f"[text] {e}\n[text] continuing without slice descriptions")
        terms, text_embs = None, None

    slices, members, counter = [], [], [0]
    for pt in pair_types(args):
        for gran in granularities(args):
            run_block(ps, args, pt, gran, terms, text_embs, slices, members, counter)

    dump_config(args.out, args, {"tau": ps.tau, "n_pairs": int(len(ps.pairs))})
    write_slices(args.out, "failure_directions", slices, members)


if __name__ == "__main__":
    main()
