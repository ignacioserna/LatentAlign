"""Stage 05 - run Domino's error-aware mixture model over verification pairs.

Adaptation of Domino (designed for classification examples) to verification:
  * the *unit* is the pair, not the image;
  * the pair's cross-modal embedding pools the CLIP embeddings of its two
    images (--pair-embed):
        mean     - symmetric, stays close to CLIP's image manifold, so slices
                   remain text-describable (default);
        absdiff  - |e1 - e2|, emphasizes appearance mismatch;
        meandiff - concat(mean, |diff|);
        concat   - concat(e1, e2) (order-sensitive; pairs are unordered, so
                   prefer the symmetric options);
  * targets  = same/different (binary);
  * pred_probs = Platt-calibrated P(same | cosine similarity) at the deployed
    global threshold, as [P(diff), P(same)].

Engine (--impl):
  vendored (default) - faithful numpy port in rfw_domino/domino_vendored.py;
                       no meerkat/torch needed, supports "confusion" init.
  official           - the HazyResearch `domino` package, if installed:
                       pip install "domino[clip,text] @ git+https://github.com/HazyResearch/domino@main"
"""

import argparse

import numpy as np
import pandas as pd

from .io_utils import Work, save_json


def build_pair_embeddings(clip_embs: np.ndarray, pairs: pd.DataFrame,
                          mode: str) -> np.ndarray:
    e1 = clip_embs[pairs["img_id1"].to_numpy()]
    e2 = clip_embs[pairs["img_id2"].to_numpy()]
    if mode == "mean":
        return (e1 + e2) / 2.0
    if mode == "absdiff":
        return np.abs(e1 - e2)
    if mode == "meandiff":
        return np.concatenate([(e1 + e2) / 2.0, np.abs(e1 - e2)], axis=1)
    if mode == "concat":
        return np.concatenate([e1, e2], axis=1)
    raise ValueError(mode)


def make_slicer(impl: str, **kwargs):
    if impl == "official":
        try:
            from domino import DominoSlicer as OfficialSlicer
        except ImportError as e:
            raise SystemExit(
                "--impl official requires the HazyResearch domino package:\n"
                '  pip install "domino[clip,text] @ '
                'git+https://github.com/HazyResearch/domino@main"\n'
                f"(import failed with: {e})"
            )
        return OfficialSlicer(**kwargs)
    from .domino_vendored import DominoSlicer
    return DominoSlicer(**kwargs)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--work", default="work")
    ap.add_argument("--impl", choices=["vendored", "official"], default="vendored")
    ap.add_argument("--pair-embed",
                    choices=["mean", "absdiff", "meandiff", "concat"],
                    default="mean")
    ap.add_argument("--subset", choices=["all", "pos", "neg"], default="all",
                    help="slice only positive (genuine) or only negative "
                         "(impostor) pairs; 'neg' isolates false-match "
                         "behavior, where FMR bias lives")
    ap.add_argument("--n-slices", type=int, default=10)
    ap.add_argument("--n-components", type=int, default=50,
                    help="mixture components (top --n-slices are returned)")
    ap.add_argument("--y-weight", type=float, default=40.0,
                    help="gamma: label log-likelihood weight")
    ap.add_argument("--yhat-weight", type=float, default=40.0,
                    help="gamma-hat: prediction log-likelihood weight")
    ap.add_argument("--pca", type=int, default=128,
                    help="PCA components before the mixture (0 = off)")
    ap.add_argument("--init", choices=["confusion", "kmeans", "random"],
                    default="confusion")
    ap.add_argument("--confusion-noise", type=float, default=1e-3)
    ap.add_argument("--max-iter", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    work = Work(args.work)
    pairs = work.load_pairs(scored=True)
    clip_embs = work.load_embs("clip")

    if args.subset != "all":
        keep = pairs["same"] == (1 if args.subset == "pos" else 0)
        pairs = pairs.loc[keep].reset_index(drop=True)
        n_err = int((pairs["correct"] == 0).sum())
        print(f"[domino] --subset {args.subset}: slicing {len(pairs)} pairs "
              f"({n_err} errors)")
        if n_err == 0:
            raise SystemExit("[domino] no errors in this subset at the current "
                             "threshold - nothing for Domino to find.")
        if n_err < 10 * args.n_slices:
            print(f"[domino] WARNING: {n_err} errors across {args.n_slices} "
                  f"requested slices (fewer than 10 per slice). Per-slice error "
                  f"rates will not be separable from noise.")
        # On a single-class subset the target is constant, so its one-hot is
        # [1,0] for every pair, every component fits p_s = [1,0], and
        # CategoricalVariable.estimate_log_prob returns exactly 0. The y
        # log-likelihood term drops out of the E-step entirely.
        if args.y_weight != 0.0:
            print(f"[domino] NOTE: --subset {args.subset} makes the target "
                  f"constant, so the y log-likelihood term is identically zero "
                  f"and --y-weight {args.y_weight} has no effect. The slice "
                  f"ranking reduces to ordering components by mean P(same), a "
                  f"monotone function of cosine similarity.")


    x = build_pair_embeddings(clip_embs, pairs, args.pair_embed)
    y = pairs["same"].to_numpy().astype(int)
    p_same = pairs["p_same"].to_numpy().astype(np.float64)
    pred_probs = np.stack([1.0 - p_same, p_same], axis=1)

    kwargs = dict(
        n_slices=args.n_slices,
        n_mixture_components=args.n_components,
        y_log_likelihood_weight=args.y_weight,
        y_hat_log_likelihood_weight=args.yhat_weight,
        n_pca_components=(None if args.pca == 0 else args.pca),
        init_params=args.init,
        confusion_noise=args.confusion_noise,
        max_iter=args.max_iter,
        random_state=args.seed,
    )
    print(f"[domino] impl={args.impl} pair_embed={args.pair_embed} "
          f"X={x.shape} {kwargs}")
    try:
        slicer = make_slicer(args.impl, **kwargs)
        slicer.fit(data=None, embeddings=x, targets=y, pred_probs=pred_probs)
    except ValueError as e:
        # current official `main` dropped the "confusion" init - retry kmeans
        if args.impl == "official" and args.init == "confusion":
            print(f"[domino] official init 'confusion' failed ({e}); "
                  f"retrying with kmeans")
            kwargs["init_params"] = "kmeans"
            slicer = make_slicer(args.impl, **kwargs)
            slicer.fit(data=None, embeddings=x, targets=y, pred_probs=pred_probs)
        else:
            raise

    slice_probs = slicer.predict_proba(
        data=None, embeddings=x, targets=y, pred_probs=pred_probs
    )
    np.save(work.slice_probs_npy, slice_probs)

    out = pairs[["pair_id", "race", "same", "sim", "p_same", "pred",
                 "correct"]].copy()
    out["slice_top"] = slice_probs.argmax(axis=1)
    out["slice_top_prob"] = slice_probs.max(axis=1)
    for k in range(slice_probs.shape[1]):
        out[f"slice_{k}_prob"] = slice_probs[:, k]
    out.to_csv(work.slices_assign_csv, index=False)

    info = {
        "impl": args.impl,
        "pair_embed": args.pair_embed,
        "subset": args.subset,
        "params": {k: v for k, v in kwargs.items()},
        "slice_cluster_indices": np.asarray(
            slicer.slice_cluster_indices
        ).tolist(),
        "mixture_weights": np.asarray(slicer.mm.weights_).tolist(),
        "y_probs": np.asarray(slicer.mm.variables[1].probs).tolist(),
        "y_hat_probs": np.asarray(slicer.mm.variables[2].probs).tolist(),
        "em_converged": bool(getattr(slicer.mm, "converged_", True)),
        "em_n_iter": int(getattr(slicer.mm, "n_iter_", -1)),
    }
    save_json(work.slicer_info_json, info)
    print(f"[domino] wrote {slice_probs.shape} slice probs -> "
          f"{work.slice_probs_npy}")
    print(f"[domino] EM converged={info['em_converged']} "
          f"iters={info['em_n_iter']}")


if __name__ == "__main__":
    main()
