"""Shared image-level plumbing for the slice-discovery baselines.

Every baseline produces the same object: a **candidate group** of N images. Each
baseline emits many of them (hundreds), `compare_fmr.py` measures the FMR of every
one at the single global threshold, and the comparison is the top-K groups by FMR.
That is the same shape you can produce from a LatentAlign run by exporting
(group_id, image_id) and letting `compare_fmr.py` score it identically.

Three things live here.

1. **The per-image loss** at the deployed threshold. `metrics.json`'s
   `global_threshold` is read, never recomputed, so stage 03 run with
   `--criterion far --far-target 1e-3` fixes the whole comparison at FMR = 1e-3.
   At that operating point ~1% of images carry any false match, so the default loss
   is `knn_margin`: the mean of each image's top-k same-race, different-identity
   similarities, minus tau. Dense, and it is the quantity that determines whether a
   group will have a high FMR. `pair_margin` and `fm_count` are the protocol-only
   alternatives, one flag away.

2. **Phrase naming that actually discriminates.** `describe.py` scores a phrase by
   `t . (group prototype - dataset prototype)`, which is unnormalised: a handful of
   phrase directions have far more variance than the rest across face crops, so they
   win for *every* group and each slice ends up labelled "a person wearing a sports
   helmet". `PhraseScorer` z-scores each phrase against its own distribution over
   the image pool first, so a phrase can only label a group if that group is unusual
   *for that phrase*. This is the same normalisation B2T applies, which is why B2T's
   labels came out diverse and the others' did not.

3. **Coherence.** `coherence(embs)` is the norm of the mean unit embedding, i.e. the
   mean cosine similarity between members and the group's centroid direction. It
   lands in [0, 1]: 1 is a group of near-identical images, ~0 is an arbitrary
   collection. Computable from image ids alone, so it applies to LatentAlign groups
   unchanged.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from .io_utils import Work, l2_normalize

LOSS_MODES = ("knn_margin", "pair_margin", "fm_count")


# --------------------------------------------------------------------------

class Ctx:
    """Everything a baseline needs, loaded once and validated."""

    def __init__(self, work: Work, pairs, images, face_embs, clip_embs,
                 tau, criterion, pool):
        self.work = work
        self.pairs = pairs
        self.images = images
        self.face_embs = face_embs
        self.clip_embs = clip_embs
        self.tau = float(tau)
        self.criterion = criterion
        self.pool = pool
        self.identity = dict(zip(images["image_id"], images["identity"]))
        self.race = dict(zip(images["image_id"], images["race"])) if \
            "race" in images.columns else {}

        neg = pairs.loc[pairs["same"] == 0, "sim"].to_numpy()
        self.n_impostor = int(len(neg))
        self.global_fmr = float((neg >= self.tau).mean()) if len(neg) else float("nan")
        self.n_fm = int((neg >= self.tau).sum())

    def pool_clip(self):
        return self.clip_embs[self.pool]

    def pool_face(self):
        return self.face_embs[self.pool]

    def pool_race(self):
        if not self.race:
            return np.array(["all"] * len(self.pool))
        return np.array([self.race[int(i)] for i in self.pool])

    def pool_identity(self):
        return np.array([self.identity[int(i)] for i in self.pool])

    def summary(self):
        return (f"[ctx] {len(self.pairs):,} scored pairs "
                f"({self.n_impostor:,} impostor), {len(self.pool):,} usable images | "
                f"tau={self.tau:.4f} ({self.criterion}) | "
                f"global FMR={self.global_fmr:.3e} ({self.n_fm:,} false matches)")


def load_context(work_dir, threshold=None, protocol=None):
    work = Work(work_dir)
    pairs = work.load_pairs(scored=True)
    images = work.load_images()
    face_embs = np.asarray(work.load_embs("face"), dtype=np.float32)
    clip_embs = np.asarray(work.load_embs("clip"), dtype=np.float32)

    if protocol and "protocol" in pairs.columns:
        pairs = pairs[pairs["protocol"] == protocol].reset_index(drop=True)
        print(f"[ctx] restricted to protocol={protocol!r}: {len(pairs):,} pairs")

    criterion = "override"
    if threshold is None:
        with open(work.metrics_json) as f:
            metrics = json.load(f)
        threshold = metrics["global_threshold"]
        criterion = metrics.get("criterion", "?")
        if criterion != "far":
            print(f"[ctx] WARNING: metrics.json was produced with "
                  f"--criterion {criterion}. For an FMR=1e-3 operating point re-run "
                  f"stage 03 with `--criterion far --far-target 1e-3`.")

    n_neg = int((pairs["same"] == 0).sum())
    if n_neg < 100_000:
        print(f"[ctx] WARNING: only {n_neg:,} impostor pairs -> ~{n_neg * 1e-3:.0f} "
              f"false matches at FMR=1e-3. Run "
              f"`python -m rfw_domino.expand_negatives --work <work>`.")

    referenced = np.unique(np.concatenate([pairs["img_id1"].to_numpy(),
                                           pairs["img_id2"].to_numpy()]))
    ok = np.isfinite(face_embs).all(axis=1) & np.isfinite(clip_embs).all(axis=1)
    pool = referenced[ok[referenced]]
    if len(referenced) - len(pool):
        print(f"[ctx] dropped {len(referenced) - len(pool)} images with a missing "
              f"embedding")

    ctx = Ctx(work, pairs, images, face_embs, clip_embs, threshold, criterion, pool)
    print(ctx.summary())
    return ctx


# --------------------------------------------------------------------------
# per-image loss
# --------------------------------------------------------------------------

def _pair_aggregates(ctx):
    neg = ctx.pairs[ctx.pairs["same"] == 0]
    ids = np.concatenate([neg["img_id1"].to_numpy(), neg["img_id2"].to_numpy()])
    sims = np.concatenate([neg["sim"].to_numpy()] * 2)
    n = len(ctx.pool)

    slot = np.full(int(ctx.pool.max()) + 2, -1, dtype=np.int64)
    slot[ctx.pool] = np.arange(n)
    j = np.where(ids <= ctx.pool.max(), slot[np.minimum(ids, len(slot) - 1)], -1)
    keep = j >= 0
    j, sims = j[keep], sims[keep]

    counts, fms, msum = np.zeros(n), np.zeros(n), np.zeros(n)
    np.add.at(counts, j, 1.0)
    np.add.at(msum, j, sims - ctx.tau)
    np.add.at(fms, j, (sims >= ctx.tau).astype(float))
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_margin = np.where(counts > 0, msum / np.maximum(counts, 1), np.nan)
    return counts, fms, mean_margin


def _knn_margin(ctx, k=10, chunk=512, verbose=True):
    face = ctx.pool_face()
    races = ctx.pool_race()
    idents = ctx.pool_identity()
    out = np.full(len(face), np.nan)

    for race in np.unique(races):
        sel = np.flatnonzero(races == race)
        if len(sel) < k + 2:
            continue
        E = face[sel]
        _, code = np.unique(idents[sel], return_inverse=True)
        top = np.empty(len(sel))
        for start in range(0, len(sel), chunk):
            stop = min(start + chunk, len(sel))
            S = E[start:stop] @ E.T
            S[code[start:stop][:, None] == code[None, :]] = -np.inf
            kk = min(k, S.shape[1] - 1)
            top[start:stop] = np.partition(S, -kk, axis=1)[:, -kk:].mean(axis=1)
            if verbose:
                print(f"\r[loss] knn {race}: {stop}/{len(sel)}", end="", flush=True)
        out[sel] = top - ctx.tau
        if verbose:
            print()
    return out


def per_image_signal(ctx, mode="knn_margin", knn_k=10, chunk=512, verbose=True):
    if mode not in LOSS_MODES:
        raise SystemExit(f"unknown --loss {mode!r}; choose from {LOSS_MODES}")
    counts, fms, pair_margin = _pair_aggregates(ctx)
    knn = (_knn_margin(ctx, knn_k, chunk, verbose)
           if mode == "knn_margin" else np.full(len(ctx.pool), np.nan))

    df = pd.DataFrame({
        "image_id": ctx.pool, "race": ctx.pool_race(),
        "identity": ctx.pool_identity(), "n_impostor_pairs": counts,
        "n_fm": fms, "pair_margin": pair_margin, "knn_margin": knn,
    })
    df["loss"] = df[{"fm_count": "n_fm"}.get(mode, mode)].astype(float)

    ok = np.isfinite(df["loss"].to_numpy())
    if ok.sum() < 50:
        raise SystemExit(f"--loss {mode} produced only {int(ok.sum())} finite values")
    if not ok.all():
        df.loc[~ok, "loss"] = float(np.nanmedian(df.loc[ok, "loss"]))
        print(f"[loss] {int((~ok).sum())} images had no {mode}; filled with median")

    if verbose:
        s = df["loss"].to_numpy()
        print(f"[loss] mode={mode} mean={s.mean():+.4f} sd={s.std():.4f} "
              f"p90={np.percentile(s, 90):+.4f} p99={np.percentile(s, 99):+.4f} | "
              f"images with >=1 false match: {int((df['n_fm'] > 0).sum()):,}"
              f"/{len(df):,}")
    return df


# --------------------------------------------------------------------------
# coherence + phrase naming
# --------------------------------------------------------------------------

def coherence(embs):
    """Mean cosine similarity of members to the group's centroid direction, in [0,1].

    Equals ||mean of unit-norm embeddings||. 1.0 = every member points the same way;
    ~0 = an arbitrary collection. Needs only image ids, so LatentAlign groups get the
    same number from the same code.
    """
    e = l2_normalize(np.asarray(embs, dtype=np.float64))
    if len(e) == 0:
        return float("nan")
    return float(np.linalg.norm(e.mean(axis=0)))


class PhraseScorer:
    """Names a group by the phrases it is *unusual* for.

    describe.py's `t . (proto - ref)` is unnormalised, so the few phrase directions
    with the largest spread across face crops win for every group -- which is why
    ten different SVM groups all came back as "a sports helmet / a bald person".
    Here each phrase is first z-scored against its own distribution over the whole
    image pool, so scores are comparable across phrases and a phrase only labels a
    group whose members really are atypical for it.

    The unnormalised MeanDescriber score is kept alongside as `raw`, so a report can
    quote either.
    """

    def __init__(self, phrases, text_embs, pool_clip, chunk=8192, verbose=True):
        self.phrases = phrases
        self.text = np.asarray(text_embs, dtype=np.float32)
        n, T = len(pool_clip), len(self.text)
        s1 = np.zeros(T, dtype=np.float64)
        s2 = np.zeros(T, dtype=np.float64)
        for start in range(0, n, chunk):
            S = (pool_clip[start:start + chunk].astype(np.float32)
                 @ self.text.T).astype(np.float64)
            s1 += S.sum(axis=0)
            s2 += (S * S).sum(axis=0)
        self.mu = s1 / n
        self.sd = np.sqrt(np.maximum(s2 / n - self.mu ** 2, 1e-12))
        self.ref = np.asarray(pool_clip, dtype=np.float64).mean(axis=0)
        if verbose:
            print(f"[name] {T} phrases, z-scored against {n:,} pooled images")
            if T < 200:
                print(f"[name] NOTE: only {T} phrases. Run "
                      f"`python -m rfw_domino.expand_phrases --work <work>` to go to "
                      f"~1,700 -- with a bank this small, unrelated groups get the "
                      f"same label and B2T has only {T} candidate groups to offer.")

    def score(self, member_clip):
        S = np.asarray(member_clip, dtype=np.float32) @ self.text.T
        return (S.mean(axis=0) - self.mu) / self.sd

    def name(self, member_clip, topk=3):
        if len(member_clip) == 0:
            return [], ""
        z = self.score(member_clip)
        idx = np.argsort(-z)[:topk]
        proto = np.asarray(member_clip, dtype=np.float64).mean(axis=0)
        raw = self.text.astype(np.float64) @ (proto - self.ref)
        entries = [{"text": self.phrases[i], "z": float(z[i]), "raw": float(raw[i])}
                   for i in idx]
        return entries, " | ".join(e["text"] for e in entries)


def load_bank(work: Work):
    if not (os.path.exists(work.text_embs_npy) and os.path.exists(work.phrases_txt)):
        print("[name] no text embeddings; groups will be unlabelled. Run stage 04 "
              "then `python -m rfw_domino.expand_phrases`.")
        return None, None
    embs = np.load(work.text_embs_npy)
    with open(work.phrases_txt) as f:
        phrases = [ln.strip() for ln in f if ln.strip()]
    if len(phrases) != len(embs):
        print(f"[name] phrases.txt ({len(phrases)}) and text_embs.npy ({len(embs)}) "
              f"disagree; re-run expand_phrases. Skipping names.")
        return None, None
    return phrases, embs


def make_scorer(ctx):
    phrases, text_embs = load_bank(ctx.work)
    if text_embs is None:
        return None
    return PhraseScorer(phrases, text_embs, ctx.pool_clip())


# --------------------------------------------------------------------------
# candidate-group output (consumed by compare_fmr.py)
# --------------------------------------------------------------------------

GROUP_COLUMNS = ["method", "group_id", "native_rank", "label", "rank", "image_id",
                 "score"]


def write_groups(work: Work, method: str, groups, params=None, extra_table=None):
    """groups: list of {group_id, label, image_ids, scores, native_rank?}."""
    d = work.path(os.path.join("baselines", method))
    os.makedirs(d, exist_ok=True)
    rows = []
    for g in groups:
        for r, (img, sc) in enumerate(zip(g["image_ids"], g["scores"])):
            rows.append({"method": method, "group_id": int(g["group_id"]),
                         "native_rank": int(g.get("native_rank", g["group_id"])),
                         "label": g.get("label", ""), "rank": r,
                         "image_id": int(img), "score": float(sc)})
    df = pd.DataFrame(rows, columns=GROUP_COLUMNS)
    df.to_csv(os.path.join(d, "groups.csv"), index=False)
    if params is not None:
        with open(os.path.join(d, "params.json"), "w") as f:
            json.dump(params, f, indent=2, default=str)
    if extra_table is not None:
        extra_table.to_csv(os.path.join(d, "detail.csv"), index=False)
    print(f"[out] {len(groups)} candidate groups of "
          f"{len(df) // max(1, len(groups))} images -> "
          f"{os.path.join(d, 'groups.csv')}")
    print(f"[out] rank them by FMR with: python -m rfw_domino.compare_fmr "
          f"--work {work.dir}")


def add_common_args(ap):
    ap.add_argument("--work", default="work")
    ap.add_argument("--group-size", type=int, default=100,
                    help="images per candidate group; match your LatentAlign groups")
    ap.add_argument("--n-candidates", "--n-groups", dest="n_candidates",
                    type=int, default=200,
                    help="how many candidate groups to emit. compare_fmr.py scores "
                         "all of them and reports the top-K by FMR, so more "
                         "candidates = a better shot for the method (and the "
                         "candidate count is printed in the table, since it is a "
                         "selection budget).")
    ap.add_argument("--loss", default="knn_margin", choices=list(LOSS_MODES))
    ap.add_argument("--knn-k", type=int, default=10)
    ap.add_argument("--threshold", type=float, default=None,
                    help="override the global threshold (default: "
                         "metrics.json['global_threshold'])")
    ap.add_argument("--protocol", default=None, choices=["official", "expanded"])
    ap.add_argument("--seed", type=int, default=0)
    return ap
