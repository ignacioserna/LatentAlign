"""Stage 05d - Spotlight baseline (d'Eon, d'Eon, Wright & Leyton-Brown, FAccT 2022).

Plain version of what it does
-----------------------------
Every image is a point in CLIP space. Spotlight shines a soft round torch on that
cloud of points and slides it around until it sits over the region where the average
loss is highest. Brightness falls off with distance from the torch's centre (a
Gaussian), and the torch's width is adjusted at every step so that it always
illuminates a fixed fraction of the images -- that stops it from shrinking onto one
bad image and declaring victory. The candidate group is the --group-size images the
torch lights up most brightly. Then those images are dimmed and the next torch has
to go find somewhere else, which is how you get many different candidate groups.

Printed numbers: `obj` is the brightness-weighted average loss under the torch, in
the same units as the per-image loss printed above it -- higher is better, and it is
what the search maximises. `loss` compares the mean loss of the 100 selected images
against the dataset mean.

Why this baseline earns its place: Spotlight uses no phrases, no captions, no
vision-language model and no labels -- only an embedding and a per-image loss. So it
answers the question a reviewer will ask about any method that does use those
things: would a plain density-weighted loss search in the same space have found the
same subpopulations? Run `--space face` to ask the same question of the recognition
model's own embedding.

Implementation deviations, stated so the paper can describe them accurately:

  * The paper enforces the size constraint with a Lagrangian barrier optimised
    jointly with the centre. Here the constraint is solved exactly at every step by
    bisecting on the torch width (total brightness is monotone in width), and only
    the centre is optimised. Same objective, same constraint, more stable, one fewer
    hyperparameter.
  * Gradients are analytic, so there is no autodiff dependency.

Runtime note: this is the slow baseline -- each candidate is an independent
optimisation over all images. ~100 candidates over 40k images takes a few minutes.
Lower --steps or --n-candidates if that matters.

    python -m rfw_domino.baseline_spotlight --work work --group-size 100 \
        --n-candidates 100
"""

import argparse
import time

import numpy as np
import pandas as pd

from .image_signal import (add_common_args, coherence, load_context, make_scorer,
                           per_image_signal, write_groups)


def _preprocess(X, n_components):
    X = np.asarray(X, dtype=np.float64)
    X = X - X.mean(0, keepdims=True)
    X = X / np.maximum(X.std(0, keepdims=True), 1e-8)
    if n_components and n_components < X.shape[1]:
        _, _, Vt = np.linalg.svd(X, full_matrices=False)
        X = X @ Vt[:n_components].T
        X = X / np.maximum(X.std(0, keepdims=True), 1e-8)
    return X


def _solve_sigma(d2, avail, target, lo=1e-4, hi=1e4, iters=40):
    """Torch width such that the total brightness equals `target`."""
    def total(s):
        return float((avail * np.exp(-d2 / (2.0 * s * s))).sum())
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


def _fit_spotlight(X, loss, avail, size, steps, lr, seed, n_init):
    rng = np.random.default_rng(seed)
    n = len(X)
    target = max(10.0, size * float(avail.sum()))
    best = None

    eff = np.where(avail > 1e-6, loss * avail, -np.inf)
    order = np.argsort(-eff)
    inits = [X[order[:max(10, int(0.01 * n))]].mean(0)]
    for _ in range(max(0, n_init - 1)):
        pick = order[rng.integers(0, max(1, min(len(order), int(0.1 * n) + 1)))]
        inits.append(X[pick] + 0.05 * rng.standard_normal(X.shape[1]))

    for mu0 in inits:
        mu = np.asarray(mu0, dtype=np.float64).copy()
        m = np.zeros_like(mu)
        v = np.zeros_like(mu)
        b1, b2, eps = 0.9, 0.999, 1e-8
        for t in range(1, steps + 1):
            diff = X - mu
            d2 = np.einsum("ij,ij->i", diff, diff)
            sigma = _solve_sigma(d2, avail, target)
            w = avail * np.exp(-d2 / (2.0 * sigma * sigma))
            W = w.sum()
            if W < 1e-9:
                break
            L = float((w * loss).sum() / W)
            g = ((w * (loss - L))[:, None] * diff).sum(0) / (W * sigma * sigma)
            m = b1 * m + (1 - b1) * g
            v = b2 * v + (1 - b2) * g * g
            mu = mu + lr * (m / (1 - b1 ** t)) / (np.sqrt(v / (1 - b2 ** t)) + eps)

        diff = X - mu
        d2 = np.einsum("ij,ij->i", diff, diff)
        sigma = _solve_sigma(d2, avail, target)
        kernel = np.exp(-d2 / (2.0 * sigma * sigma))   # in [0, 1], = 1 at mu
        w = avail * kernel
        if w.sum() < 1e-9:
            continue
        obj = float((w * loss).sum() / w.sum())
        if best is None or obj > best[4]:
            best = (w, kernel, mu, sigma, obj)
    return best


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--space", default="clip", choices=["clip", "face"],
                    help="which embedding the torch searches")
    ap.add_argument("--spotlight-size", type=float, default=0.01,
                    help="fraction of the remaining images each torch must cover")
    ap.add_argument("--pca", type=int, default=64, help="0 disables")
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--n-init", type=int, default=2)
    ap.add_argument("--decay", type=float, default=1.0,
                    help="how hard a found spotlight suppresses its region for the "
                         "next one: avail *= (1 - decay * kernel). 1.0 is the "
                         "paper's rule. Lower values let spotlights overlap more, "
                         "which yields more candidates but more redundant ones.")
    ap.set_defaults(n_candidates=100)   # this one is slow; 100 is the sane default
    args = ap.parse_args(argv)

    ctx = load_context(args.work, args.threshold, args.protocol)
    sig = per_image_signal(ctx, args.loss, args.knn_k)
    scorer = make_scorer(ctx)

    img_ids = sig["image_id"].to_numpy()
    src = ctx.clip_embs if args.space == "clip" else ctx.face_embs
    X = _preprocess(src[img_ids], args.pca)
    loss = sig["loss"].to_numpy()
    if np.std(loss) < 1e-12:
        raise SystemExit("[spot] the loss is constant; nothing to search for")

    avail = np.ones(len(X))
    groups, detail = [], []
    t0 = time.time()
    for gid in range(args.n_candidates):
        fit = _fit_spotlight(X, loss, avail, args.spotlight_size, args.steps,
                             args.lr, args.seed + gid, args.n_init)
        if fit is None:
            print(f"[spot] exhausted after {gid} candidates: every image has been "
                  f"claimed by earlier spotlights, so there is nowhere left to put "
                  f"one. This is a real ceiling of the method on a pool of "
                  f"{len(X):,} images, not an error.")
            print(f"[spot] to raise it: lower --spotlight-size (0.005 or 0.002 "
                  f"gives smaller, finer spotlights and roughly 1.4-1.6x more of "
                  f"them) or lower --decay below 1.0 (more overlap between "
                  f"candidates, so more of them, but more redundant).")
            break
        w, kernel, mu, sigma, obj = fit
        top = np.argsort(-w)[:args.group_size]
        ids = img_ids[top]
        name = ""
        if scorer is not None:
            _, name = scorer.name(ctx.clip_embs[ids])
        groups.append(dict(group_id=gid, native_rank=gid, label=name,
                           image_ids=ids, scores=w[top]))
        detail.append(dict(group_id=gid, objective=obj, sigma=float(sigma),
                           space=args.space, mean_loss=float(loss[top].mean()),
                           base_loss=float(loss.mean()),
                           coherence=coherence(ctx.clip_embs[ids]),
                           description=name))
        if gid < 10 or gid % 25 == 0:
            print(f"[spot] candidate {gid}: obj={obj:+.4f} "
                  f"loss {loss[top].mean():+.4f} vs {loss.mean():+.4f} "
                  f"({time.time() - t0:.0f}s) | {name}")
        # Down-weight what this spotlight claimed, exactly as the paper does: the
        # multiplier is the raw kernel value in [0, 1] (1 at the centre, ~0 far
        # away), NOT the weight renormalised by its own max. w = avail * kernel is
        # already scaled down by earlier rounds, so w.max() is small and w/w.max()
        # is near 1 across a huge region -- dividing by it removed 12-75% of the
        # remaining availability per round instead of the ~1% the size constraint
        # implies, and the search ran out of space after a few dozen spotlights.
        avail = np.maximum(avail * (1.0 - args.decay * kernel), 0.0)

    if not groups:
        raise SystemExit("[spot] no groups produced")
    print(f"[spot] {len(groups)} candidate groups in {time.time() - t0:.0f}s")
    method = "spotlight" if args.space == "clip" else "spotlight_face"
    write_groups(ctx.work, method, groups,
                 params=vars(args) | {"tau": ctx.tau, "global_fmr": ctx.global_fmr},
                 extra_table=pd.DataFrame(detail))


if __name__ == "__main__":
    main()