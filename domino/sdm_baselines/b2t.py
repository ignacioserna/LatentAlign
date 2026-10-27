"""B2T / Bias-to-Text (Kim, Mo, Kim, Lee, Lee, Shin -- CVPR 2024) for face verification.

Original method: caption the misclassified images, extract candidate keywords from
those captions, then score each keyword by the *CLIP score gap*

    s(t) = mean_{x in Wrong} sim(x, t) - mean_{x in Correct} sim(x, t)

and report the top-scoring keywords as the discovered bias.

This implementation keeps B2T's scoring step verbatim and replaces its keyword
*proposal* step with a fixed face-attribute vocabulary (MAAD-Face's 47 attributes
plus acquisition covariates -- see vocab.py).  Two reasons, both worth stating in
the paper:

  * Captioners collapse on aligned face crops.  BLIP/ClipCap emit near-identical
    strings ("a man with a beard wearing glasses") for most of a face dataset, so
    the proposal step returns a handful of low-variance keywords and B2T's ceiling
    is set by the captioner rather than by its scoring rule.  Fixing the vocabulary
    removes that confound and gives B2T a *strictly easier* problem than it had in
    the paper -- so this is a generous baseline, not a hobbled one.

  * Evaluation alignment.  Scoring the same attribute names you evaluate against
    means Precision@k is not measuring vocabulary mismatch.

If you want the faithful captioner pipeline as well, run alinlab/b2t's caption +
YAKE stage, dump its keywords to a text file and pass it via --terms-file: this
script will then score exactly those keywords.

Verification-specific choices:

  * Wrong/correct groups are formed per pair type, so FMR keywords and FNMR
    keywords never mix.

  * With --target margin the groups are the top and bottom quantiles of the signed
    margin (--margin-quantile) rather than the binary error label.  At strict
    operating points the binary "Wrong" group can be a few dozen pairs, which makes
    a difference-of-means over ~90 terms almost pure noise.

  * Pair-level CLIP scores use the mean of the two image embeddings -- the only
    symmetric pair representation that stays inside CLIP's text-aligned space, so
    sim(pair, t) remains a meaningful quantity.

  * Every keyword gets a Welch t-test and a bootstrap CI.  B2T's raw gap has no
    notion of significance, and on face data a good fraction of the top-ranked
    terms will not survive; reporting that is fairer than reporting a bare top-5.

Usage:
    python -m sdm_baselines.b2t \
        --pairs work/pairs.csv --images work/images.csv \
        --face-embs work/face_embs.npy --desc-embs work/clip_embs.npy \
        --out runs/b2t --target margin
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats

try:
    from .common import (add_common_args, build_pairset, dump_config, granularities,
                         l2_normalize, pair_types, resolve_text_embeddings,
                         slice_stats, write_slices)
except ImportError:
    from common import (add_common_args, build_pairset, dump_config, granularities,
                        l2_normalize, pair_types, resolve_text_embeddings,
                        slice_stats, write_slices)


def _groups(is_err, loss, target, q, rng):
    """Return boolean masks (wrong, correct), or (None, None) if degenerate."""
    if target == "error":
        wrong, correct = is_err.astype(bool), ~is_err.astype(bool)
    else:
        hi = np.quantile(loss, 1.0 - q)
        lo = np.quantile(loss, q)
        wrong, correct = loss >= hi, loss <= lo
    if wrong.sum() < 5 or correct.sum() < 5:
        return None, None
    return wrong, correct


def _bootstrap_gap(sw, sc, n_boot, rng):
    """Percentile bootstrap CI for the difference of means."""
    if n_boot <= 0:
        return np.nan, np.nan
    bw = rng.choice(sw, size=(n_boot, len(sw)), replace=True).mean(axis=1)
    bc = rng.choice(sc, size=(n_boot, len(sc)), replace=True).mean(axis=1)
    d = bw - bc
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def run_block(ps, args, pair_type, granularity, terms, text_embs, slices, members,
              keyword_rows):
    rng = np.random.default_rng(args.seed)

    if granularity == "pair":
        m = ps.mask(pair_type)
        unit_ids = np.flatnonzero(m)
        # mean of the two CLIP embeddings: symmetric AND still text-alignable
        rep = ps.rep("mean", pair_type)
        is_err = ps.is_error[m]
        loss = ps.loss[m]
    else:
        view = ps.image_view(pair_type, args.target)
        unit_ids = view["image_id"].to_numpy()
        rep = ps.desc_embs[unit_ids]
        is_err = (view["n_errors"].to_numpy() > 0)
        loss = view["value"].to_numpy()

    if len(unit_ids) < 50:
        print(f"[b2t] skipping {pair_type}/{granularity}: only {len(unit_ids)} units")
        return

    # (U, T) CLIP scores
    S = l2_normalize(rep) @ text_embs.T
    if args.znorm:
        # comparability across terms: CLIP similarities have very different scales
        # per prompt, and B2T's raw gap silently favours high-variance terms.
        mu, sd = S.mean(axis=0, keepdims=True), S.std(axis=0, keepdims=True)
        S = (S - mu) / np.maximum(sd, 1e-8)

    wrong, correct = _groups(is_err, loss, args.target, args.margin_quantile, rng)
    if wrong is None:
        n_err = int(is_err.sum())
        print(f"[b2t] skipping {pair_type}/{granularity}: cannot form Wrong/Correct "
              f"groups ({n_err}/{len(is_err)} errors at tau). Use --target margin, "
              f"raise --margin-quantile, or loosen --threshold.")
        return
    base_err = float(is_err.mean())
    base_margin = float(loss.mean())
    k = min(args.slice_size, max(10, len(unit_ids) // 20))

    rows = []
    for ti, term in enumerate(terms):
        sw, sc = S[wrong, ti], S[correct, ti]
        gap = float(sw.mean() - sc.mean())
        t_stat, p_val = stats.ttest_ind(sw, sc, equal_var=False)
        lo, hi = _bootstrap_gap(sw, sc, args.n_boot, rng)
        rows.append(dict(term=term, term_index=ti, gap=gap, t=float(t_stat),
                         p=float(p_val), ci_lo=lo, ci_hi=hi,
                         n_wrong=int(wrong.sum()), n_correct=int(correct.sum())))
    kw = pd.DataFrame(rows).sort_values("gap", ascending=False).reset_index(drop=True)

    # Benjamini-Hochberg over the vocabulary
    p = kw["p"].to_numpy()
    order = np.argsort(p)
    ranked = np.empty(len(p), dtype=float)
    ranked[order] = np.minimum.accumulate(
        (p[order] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    kw["q_bh"] = np.clip(ranked, 0, 1)
    kw["pair_type"] = pair_type
    kw["unit_type"] = granularity
    keyword_rows.append(kw)

    for j in range(min(args.n_slices, len(kw))):
        r = kw.iloc[j]
        ti = int(r["term_index"])
        # the keyword induces a slice: the units it describes best
        top = np.argsort(-S[:, ti])[:k]
        st = slice_stats(top, is_err, loss, base_err, base_margin)
        sid = f"b2t.{pair_type}.{granularity}.{j}"
        slices.append(dict(
            slice_id=sid, pair_type=pair_type, unit_type=granularity, rank=j, k=k,
            description=r["term"], gap=float(r["gap"]), t=float(r["t"]),
            p=float(r["p"]), q_bh=float(r["q_bh"]), ci_lo=float(r["ci_lo"]),
            ci_hi=float(r["ci_hi"]), target=args.target, coherence=np.nan, **st))
        for rank, t in enumerate(top):
            members.append(dict(slice_id=sid, unit_type=granularity,
                                unit_id=int(unit_ids[t]), rank_in_slice=rank,
                                weight=float(S[t, ti])))
        sig = "*" if r["q_bh"] < 0.05 else " "
        print(f"[b2t] {sid}: {r['term']!r} gap={r['gap']:+.3f} q={r['q_bh']:.3g}{sig} "
              f"| slice err={st['err_rate']:.3%} (base {base_err:.3%}, "
              f"lift {st['lift']:.2f}x)")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--margin-quantile", type=float, default=0.1,
                    help="with --target margin, the top/bottom quantile that define "
                         "the Wrong / Correct groups")
    ap.add_argument("--n-boot", type=int, default=1000,
                    help="bootstrap resamples for the gap CI (0 disables)")
    ap.add_argument("--no-znorm", dest="znorm", action="store_false",
                    help="use raw CLIP similarities instead of per-term z-scores "
                         "(closer to the paper, but favours high-variance terms)")
    ap.set_defaults(znorm=True)
    args = ap.parse_args(argv)

    ps = build_pairset(args)
    terms, text_embs = resolve_text_embeddings(args, ps.desc_embs.shape[1])
    print(f"[b2t] scoring {len(terms)} vocabulary terms")

    slices, members, keyword_rows = [], [], []
    for pt in pair_types(args):
        for gran in granularities(args):
            run_block(ps, args, pt, gran, terms, text_embs, slices, members,
                      keyword_rows)

    dump_config(args.out, args, {"tau": ps.tau, "n_terms": len(terms)})
    write_slices(args.out, "b2t", slices, members)
    if keyword_rows:
        kw = pd.concat(keyword_rows, ignore_index=True)
        path = os.path.join(args.out, "keywords.csv")
        kw.to_csv(path, index=False)
        print(f"[out] full keyword ranking -> {path}")


if __name__ == "__main__":
    main()
