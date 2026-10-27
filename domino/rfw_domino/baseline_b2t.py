"""Stage 05c - B2T / Bias-to-Text baseline (Kim et al., CVPR 2024).

What the method does: split images into Wrong and Correct by the model's loss, then
score every candidate phrase by the CLIP-score gap between the two groups,

    s(t) = mean_{x in Wrong} sim(x, t) - mean_{x in Correct} sim(x, t).

Phrases with a large positive gap describe what the failures have in common.

Candidate groups: **one per phrase** -- the --group-size images whose CLIP
similarity to that phrase is highest. So the phrase bank *is* the candidate budget.
With stage 04's 53 phrases you get 53 candidate groups; after
`python -m rfw_domino.expand_phrases` you get ~1,700. Run it.

Two adaptations, both worth stating in the paper:

  * B2T proposes its own keywords by captioning the wrong images and running YAKE.
    On aligned face crops a captioner emits near-identical strings across the whole
    dataset, so the proposal step, not the scoring rule, would set the ceiling. The
    phrase bank replaces it. This makes B2T *stronger* than published, which is the
    right way to lose an argument with a reviewer. To run the literal pipeline,
    dump alinlab/b2t's caption+YAKE keywords one per line and pass --phrases-file
    (they must already be in phrases.txt so their embeddings exist).

  * Wrong / Correct come from the per-image verification loss at the global
    threshold, defaulting to the top and bottom --quantile. At FMR = 1e-3 the
    literal "carried a false match" label covers ~1% of images; --use-fm-label
    gives you that version.

Every phrase is reported with a Welch t-test, a bootstrap CI and a BH-corrected
q-value, since over ~1,700 phrases a bare gap ranking has real noise in it.

    python -m rfw_domino.baseline_b2t --work work --group-size 100
"""

import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats

from .image_signal import (add_common_args, coherence, load_bank, load_context,
                           make_scorer, per_image_signal, write_groups)
from .io_utils import l2_normalize


def _bh(p):
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    q = np.empty(len(p))
    q[order] = np.minimum.accumulate(
        (p[order] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    return np.clip(q, 0, 1)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--quantile", type=float, default=0.1)
    ap.add_argument("--use-fm-label", action="store_true",
                    help="literal binary label (image carried >=1 false match); "
                         "faithful but very sparse at FMR=1e-3")
    ap.add_argument("--phrases-file", default=None,
                    help="restrict candidates to these phrases (e.g. alinlab/b2t's "
                         "caption+YAKE output); they must be in phrases.txt already")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--no-znorm", dest="znorm", action="store_false")
    ap.set_defaults(znorm=True)
    args = ap.parse_args(argv)

    ctx = load_context(args.work, args.threshold, args.protocol)
    sig = per_image_signal(ctx, args.loss, args.knn_k)
    phrases, text_embs = load_bank(ctx.work)
    if text_embs is None:
        raise SystemExit("B2T needs the phrase bank: run stage 04, then "
                         "`python -m rfw_domino.expand_phrases --work <work>`.")
    if args.phrases_file:
        want = [ln.strip() for ln in open(args.phrases_file) if ln.strip()]
        index = {p: i for i, p in enumerate(phrases)}
        keep = [index[p] for p in want if p in index]
        if len(want) - len(keep):
            raise SystemExit(f"{len(want) - len(keep)}/{len(want)} phrases from "
                             f"--phrases-file have no cached embedding; append them "
                             f"to phrases.txt and re-run expand_phrases.")
        phrases = [phrases[i] for i in keep]
        text_embs = text_embs[keep]
        print(f"[b2t] restricted to {len(phrases)} phrases from "
              f"{os.path.basename(args.phrases_file)}")
    scorer = make_scorer(ctx)

    rng = np.random.default_rng(args.seed)
    img_ids = sig["image_id"].to_numpy()
    loss = sig["loss"].to_numpy()
    n_fm = sig["n_fm"].to_numpy()

    S = l2_normalize(ctx.clip_embs[img_ids].astype(np.float32)) @ text_embs.T
    S = S.astype(np.float64)
    if args.znorm:
        S = (S - S.mean(0, keepdims=True)) / np.maximum(S.std(0, keepdims=True), 1e-8)

    if args.use_fm_label:
        wrong, correct = n_fm > 0, n_fm == 0
    else:
        hi, lo = np.quantile(loss, 1 - args.quantile), np.quantile(loss, args.quantile)
        wrong, correct = loss >= hi, loss <= lo
    if wrong.sum() < 5 or correct.sum() < 5:
        raise SystemExit(f"Wrong/Correct groups are {int(wrong.sum())}/"
                         f"{int(correct.sum())}; drop --use-fm-label or raise "
                         f"--quantile.")
    print(f"[b2t] {int(wrong.sum()):,} wrong vs {int(correct.sum()):,} correct "
          f"images, scoring {len(phrases)} phrases")

    sw, sc = S[wrong], S[correct]
    gap = sw.mean(0) - sc.mean(0)
    t, p = stats.ttest_ind(sw, sc, equal_var=False, axis=0)
    if args.n_boot > 0:
        bw = np.stack([sw[rng.integers(0, len(sw), len(sw))].mean(0)
                       for _ in range(args.n_boot)])
        bc = np.stack([sc[rng.integers(0, len(sc), len(sc))].mean(0)
                       for _ in range(args.n_boot)])
        d = bw - bc
        ci_lo, ci_hi = np.percentile(d, 2.5, 0), np.percentile(d, 97.5, 0)
    else:
        ci_lo = ci_hi = np.full(len(gap), np.nan)

    kw = pd.DataFrame({"phrase": phrases, "phrase_index": np.arange(len(phrases)),
                       "gap": gap, "t": t, "p": p, "q_bh": _bh(p),
                       "ci_lo": ci_lo, "ci_hi": ci_hi})
    kw = kw.sort_values("gap", ascending=False).reset_index(drop=True)

    groups, detail = [], []
    n_emit = min(args.n_candidates, len(kw))
    for gid in range(n_emit):
        r = kw.iloc[gid]
        ti = int(r["phrase_index"])
        top = np.argsort(-S[:, ti])[:args.group_size]
        ids = img_ids[top]
        groups.append(dict(group_id=gid, native_rank=gid, label=r["phrase"],
                           image_ids=ids, scores=S[top, ti]))
        detail.append(dict(group_id=gid, phrase=r["phrase"], gap=float(r["gap"]),
                           q_bh=float(r["q_bh"]), ci_lo=float(r["ci_lo"]),
                           ci_hi=float(r["ci_hi"]),
                           mean_loss=float(loss[top].mean()),
                           base_loss=float(loss.mean()),
                           coherence=coherence(ctx.clip_embs[ids])))
        if gid < 10:
            star = "*" if r["q_bh"] < 0.05 else " "
            print(f"[b2t] candidate {gid}: gap={r['gap']:+.3f} q={r['q_bh']:.2g}"
                  f"{star} | {r['phrase']}")
    if n_emit > 10:
        print(f"[b2t] ... and {n_emit - 10} more candidates")
    print(f"[b2t] {int((kw['q_bh'] < 0.05).sum())}/{len(kw)} phrase gaps survive BH "
          f"at 0.05")

    write_groups(ctx.work, "b2t", groups,
                 params=vars(args) | {"tau": ctx.tau, "n_phrases": len(phrases),
                                      "global_fmr": ctx.global_fmr},
                 extra_table=pd.DataFrame(detail))


if __name__ == "__main__":
    main()
