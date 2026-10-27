"""Spotlight (d'Eon, d'Eon, Wright, Leyton-Brown -- FAccT 2022) for face verification.

Original method: find a soft, contiguous region of representation space with high
average loss.  A spotlight is a Gaussian kernel centred at mu with bandwidth sigma,
giving weights

    w_i = exp(-||x_i - mu||^2 / (2 sigma^2)),

and the objective is the weighted mean loss, maximised over mu subject to the
spotlight covering at least ``spotlight_size`` total weight.  Additional spotlights
are found by down-weighting the points the previous ones already claimed.

Why this is a good baseline for a verification paper: Spotlight needs nothing but
an embedding and a per-example loss.  It never touches the model, never retrains,
has no vision-language dependency, and makes no assumption about closed-set labels.
That makes it the cleanest control for "does the vision-language machinery in
DOMINO / B2T actually earn its keep, or would a plain density-weighted loss search
in the same embedding space have found this?"  If your method only beats Spotlight
by a little, that is the number a reviewer will fixate on.

Implementation notes / deviations, worth stating accurately in the paper:

  * The paper enforces the size constraint with a Lagrangian barrier optimised
    jointly with mu.  Here the constraint is solved exactly at every step by
    bisecting on sigma (total weight is monotone in sigma), and only mu is
    optimised by gradient ascent.  Same objective and same constraint, more stable
    optimisation, one fewer hyperparameter to tune.

  * Gradients are analytic:
        dL/dmu = (1/W) * sum_i a_i w_i (loss_i - L) (x_i - mu) / sigma^2
    with W = sum_i a_i w_i and a_i the residual availability after earlier
    spotlights.  No autodiff dependency.

  * The loss is the signed verification margin by default (positive = wrong side of
    the threshold), which suits Spotlight better than any other method here: it was
    designed for a continuous loss and degrades to near-uselessness on a sparse
    binary error indicator.

  * Spotlights are fitted per pair type, so an FMR region is never averaged with an
    FNMR region.

Usage:
    python -m sdm_baselines.spotlight \
        --pairs work/pairs.csv --images work/images.csv \
        --face-embs work/face_embs.npy --desc-embs work/clip_embs.npy \
        --out runs/spotlight --n-slices 5 --spotlight-size 0.02
"""

from __future__ import annotations

import argparse

import numpy as np

try:
    from .common import (add_common_args, build_pairset, describe, dump_config,
                         granularities, l2_normalize, name_vector, pair_types,
                         resolve_text_embeddings, slice_stats, write_slices)
except ImportError:
    from common import (add_common_args, build_pairset, describe, dump_config,
                        granularities, l2_normalize, name_vector, pair_types,
                        resolve_text_embeddings, slice_stats, write_slices)


def _preprocess(X, n_components=None, seed=0):
    """Standardise, then optionally PCA.

    Spotlight's isotropic kernel is only meaningful if the axes are comparable, and
    a 512-d CLIP space is mostly empty, so the paper recommends reducing dimension.
    """
    X = np.asarray(X, dtype=np.float64)
    X = X - X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True)
    X = X / np.maximum(sd, 1e-8)
    if n_components and n_components < X.shape[1]:
        # randomised-free exact PCA via SVD; dims here are small enough
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        comps = Vt[:n_components]
        X = X @ comps.T
        X = X / np.maximum(X.std(axis=0, keepdims=True), 1e-8)
        return X, comps, sd
    return X, None, sd


def _weights(d2, sigma):
    return np.exp(-d2 / (2.0 * sigma ** 2))


def _solve_sigma(d2, avail, target, lo=1e-4, hi=1e4, iters=60):
    """Bisect sigma so that sum(avail * w) == target.  Total weight is monotone."""
    def total(s):
        return float((avail * _weights(d2, s)).sum())

    if total(hi) < target:
        return hi
    if total(lo) > target:
        return lo
    for _ in range(iters):
        mid = np.sqrt(lo * hi)
        if total(mid) < target:
            lo = mid
        else:
            hi = mid
    return np.sqrt(lo * hi)


def _fit_spotlight(X, loss, avail, size, steps=300, lr=0.05, seed=0, n_init=4):
    """Return (weights, mu, sigma, objective) for the best of n_init restarts."""
    rng = np.random.default_rng(seed)
    n = len(X)
    target = max(5.0, size * float(avail.sum()))
    best = None

    # inits: centroid of the worst-loss available points, then random high-loss points
    eff = loss * avail - 1e9 * (avail < 1e-6)
    order = np.argsort(-eff)
    inits = [X[order[:max(5, int(0.01 * n))]].mean(axis=0)]
    for _ in range(max(0, n_init - 1)):
        pick = order[rng.integers(0, max(1, min(len(order), int(0.1 * n) + 1)))]
        inits.append(X[pick] + 0.05 * rng.standard_normal(X.shape[1]))

    for mu0 in inits:
        mu = np.asarray(mu0, dtype=np.float64).copy()
        m = np.zeros_like(mu)
        v = np.zeros_like(mu)
        b1, b2, eps = 0.9, 0.999, 1e-8
        sigma = 1.0
        for t in range(1, steps + 1):
            diff = X - mu
            d2 = np.einsum("ij,ij->i", diff, diff)
            sigma = _solve_sigma(d2, avail, target)
            w = avail * _weights(d2, sigma)
            W = w.sum()
            if W < 1e-9:
                break
            L = float((w * loss).sum() / W)
            grad = ((w * (loss - L))[:, None] * diff).sum(axis=0) / (W * sigma ** 2)
            m = b1 * m + (1 - b1) * grad
            v = b2 * v + (1 - b2) * grad ** 2
            mhat = m / (1 - b1 ** t)
            vhat = v / (1 - b2 ** t)
            mu = mu + lr * mhat / (np.sqrt(vhat) + eps)

        diff = X - mu
        d2 = np.einsum("ij,ij->i", diff, diff)
        sigma = _solve_sigma(d2, avail, target)
        w = avail * _weights(d2, sigma)
        W = w.sum()
        if W < 1e-9:
            continue
        obj = float((w * loss).sum() / W)
        if best is None or obj > best[3]:
            best = (w, mu, sigma, obj)
    return best


def run_block(ps, args, pair_type, granularity, terms, text_embs, slices, members):
    if granularity == "pair":
        m = ps.mask(pair_type)
        unit_ids = np.flatnonzero(m)
        X_raw = ps.rep(args.rep, pair_type)
        clip_rep = ps.rep("mean", pair_type)
        is_err = ps.is_error[m]
        loss_all = ps.loss[m]
    else:
        view = ps.image_view(pair_type, "margin")
        unit_ids = view["image_id"].to_numpy()
        X_raw = ps.desc_embs[unit_ids]
        clip_rep = X_raw
        is_err = (view["n_errors"].to_numpy() > 0)
        loss_all = view["value"].to_numpy()

    if len(unit_ids) < 50:
        print(f"[spot] skipping {pair_type}/{granularity}: only {len(unit_ids)} units")
        return

    loss = loss_all if args.target == "margin" else is_err.astype(np.float64)
    if np.std(loss) < 1e-12:
        print(f"[spot] skipping {pair_type}/{granularity}: the loss is constant at "
              f"tau ({int(is_err.sum())}/{len(is_err)} errors). Use --target margin.")
        return
    X, _, _ = _preprocess(X_raw, args.pca, args.seed)

    base_err = float(is_err.mean())
    base_margin = float(loss_all.mean())
    k = min(args.slice_size, max(10, len(unit_ids) // 20))
    avail = np.ones(len(X))

    for j in range(args.n_slices):
        fit = _fit_spotlight(X, loss, avail, args.spotlight_size,
                             steps=args.steps, lr=args.lr, seed=args.seed + j,
                             n_init=args.n_init)
        if fit is None:
            break
        w, mu, sigma, obj = fit
        top = np.argsort(-w)[:k]
        st = slice_stats(top, is_err, loss_all, base_err, base_margin)

        desc = []
        if text_embs is not None:
            cent = l2_normalize(clip_rep[top].mean(axis=0)[None, :])[0]
            desc = name_vector(cent, text_embs, terms, topk=5, signed=False)

        sid = f"spotlight.{pair_type}.{granularity}.{j}"
        slices.append(dict(
            slice_id=sid, pair_type=pair_type, unit_type=granularity, rank=j, k=k,
            objective=obj, sigma=float(sigma), total_weight=float(w.sum()),
            target=args.target, coherence=np.nan,
            description=describe(desc) if desc else "", **st))
        for rank, t in enumerate(top):
            members.append(dict(slice_id=sid, unit_type=granularity,
                                unit_id=int(unit_ids[t]), rank_in_slice=rank,
                                weight=float(w[t])))
        print(f"[spot] {sid}: size={st['size']} err={st['err_rate']:.3%} "
              f"(base {base_err:.3%}, lift {st['lift']:.2f}x) obj={obj:+.4f} "
              f"| {slices[-1]['description']}")

        # claim these points so the next spotlight has to look elsewhere
        avail = avail * (1.0 - w / max(w.max(), 1e-12))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--spotlight-size", type=float, default=0.02,
                    help="fraction of the (available) units a spotlight must cover")
    ap.add_argument("--pca", type=int, default=64,
                    help="PCA dimension before the isotropic kernel (0 disables)")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--n-init", type=int, default=4)
    args = ap.parse_args(argv)

    ps = build_pairset(args)
    try:
        terms, text_embs = resolve_text_embeddings(args, ps.desc_embs.shape[1])
    except SystemExit as e:
        print(f"[text] {e}\n[text] continuing without slice descriptions")
        terms, text_embs = None, None

    slices, members = [], []
    for pt in pair_types(args):
        for gran in granularities(args):
            run_block(ps, args, pt, gran, terms, text_embs, slices, members)

    dump_config(args.out, args, {"tau": ps.tau})
    write_slices(args.out, "spotlight", slices, members)


if __name__ == "__main__":
    main()
