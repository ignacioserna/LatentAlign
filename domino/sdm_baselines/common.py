"""Shared plumbing for running classification-era slice-discovery methods on
face *verification*.

The whole point of this module is to absorb the four structural mismatches
between the SDM literature and verification, so that the individual baselines
(failure_directions.py, b2t.py, spotlight.py) stay recognisably close to their
published form:

  1. Errors live on pairs, not images.  ``PairSet`` carries the pair table and
     exposes both a pair-level view and an image-level view (a pair error
     attributed to both constituent faces), so every method can be run twice and
     you can report the granularity that flatters the baseline.

  2. FNMR and FMR are semantically opposite.  Everything here is keyed by
     ``pair_type`` in {"genuine", "impostor"} and the baselines fit separately per
     type.  Pooling them produces slices that mean nothing.

  3. Errors are threshold-dependent and rare.  ``pick_threshold`` supports EER and
     target-FMR operating points, and every method can consume either the binary
     error label or the *continuous* margin loss.  At operational thresholds the
     binary signal is too sparse to cluster on; the margin is what actually works.

  4. The model's own embedding is identity-discriminative by construction.  All
     representations here are built from a *separate* description space (CLIP by
     default) and the FR embedding is used only to produce scores.  Passing the FR
     embedding as the description space is supported (--rep-space face) as an
     ablation, but expect it to be weak, which is itself a result worth reporting.

Conventions used throughout:

  * ``images``   DataFrame with columns image_id (int, 0..N-1), path, identity
  * ``pairs``    DataFrame with columns image_id_a, image_id_b, same (bool)
  * embeddings   (N, D) float32 arrays indexed by image_id; rows of NaN mean the
                 embedding is missing and every pair touching them is dropped
  * a "unit"     is whatever a slice is a set of: a pair index or an image_id
"""

from __future__ import annotations

import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------

_ALIASES = {
    "image_id_a": ["image_id_a", "img_a", "image_a", "id_a", "a", "idx_a", "left"],
    "image_id_b": ["image_id_b", "img_b", "image_b", "id_b", "b", "idx_b", "right"],
    "same": ["same", "label", "is_genuine", "issame", "genuine", "y", "is_same",
             "match"],
    "path_a": ["path_a", "file_a", "img_path_a"],
    "path_b": ["path_b", "file_b", "img_path_b"],
    "score": ["score", "cos", "cosine", "similarity", "sim"],
    "image_id": ["image_id", "id", "idx", "index"],
    "path": ["path", "file", "filepath", "filename", "image"],
    "identity": ["identity", "id_name", "subject", "person", "class", "label"],
}


def _resolve(df, canonical, required=True):
    lower = {c.lower(): c for c in df.columns}
    for cand in _ALIASES[canonical]:
        if cand in lower:
            return lower[cand]
    if required:
        raise SystemExit(
            f"could not find a '{canonical}' column; looked for "
            f"{_ALIASES[canonical]}, got {list(df.columns)}"
        )
    return None


def load_manifest(path):
    """Load the image manifest.  Adds image_id if absent (row order)."""
    images = pd.read_csv(path)
    c_id = _resolve(images, "image_id", required=False)
    c_path = _resolve(images, "path", required=False)
    c_ident = _resolve(images, "identity", required=False)
    out = pd.DataFrame()
    out["image_id"] = (images[c_id].astype(int) if c_id
                       else np.arange(len(images), dtype=int))
    out["path"] = images[c_path].astype(str) if c_path else ""
    if c_ident:
        out["identity"] = images[c_ident].astype(str)
    else:
        # fall back to the parent directory, which is the identity for RFW/LFW/VGGFace2
        out["identity"] = [
            str(p).replace("\\", "/").rstrip("/").split("/")[-2]
            if "/" in str(p).replace("\\", "/") else str(i)
            for i, p in zip(out["image_id"], out["path"])
        ]
    if out["image_id"].duplicated().any():
        raise SystemExit("manifest has duplicate image_id values")
    return out.sort_values("image_id").reset_index(drop=True)


def load_pairs(path, images=None):
    """Load the pair table.  Accepts image_id columns or path columns."""
    pairs = pd.read_csv(path)
    c_same = _resolve(pairs, "same")
    a = _resolve(pairs, "image_id_a", required=False)
    b = _resolve(pairs, "image_id_b", required=False)
    if a is None or b is None:
        pa = _resolve(pairs, "path_a")
        pb = _resolve(pairs, "path_b")
        if images is None:
            raise SystemExit("pairs given by path but no --images manifest supplied")
        lut = {}
        for iid, p in zip(images["image_id"], images["path"]):
            p = str(p).replace("\\", "/")
            lut[p] = iid
            lut[os.path.basename(p)] = iid
            lut["/".join(p.rstrip("/").split("/")[-2:])] = iid

        def _map(col):
            out = []
            for v in pairs[col].astype(str):
                v = v.replace("\\", "/")
                hit = (lut.get(v) or lut.get(os.path.basename(v))
                       or lut.get("/".join(v.rstrip("/").split("/")[-2:])))
                out.append(-1 if hit is None else int(hit))
            return np.asarray(out, dtype=int)

        ia, ib = _map(pa), _map(pb)
        unresolved = int(((ia < 0) | (ib < 0)).sum())
        if unresolved:
            print(f"[pairs] WARNING: {unresolved} pairs reference images not in the "
                  f"manifest; dropping them")
    else:
        ia = pairs[a].astype(int).to_numpy()
        ib = pairs[b].astype(int).to_numpy()

    same = pairs[c_same]
    if same.dtype == object:
        same = same.astype(str).str.strip().str.lower().isin(
            {"1", "true", "yes", "same", "genuine", "pos", "positive"})
    same = same.astype(bool).to_numpy()

    out = pd.DataFrame({"image_id_a": ia, "image_id_b": ib, "same": same})
    c_score = _resolve(pairs, "score", required=False)
    if c_score:
        out["score"] = pairs[c_score].astype(float).to_numpy()
    # carry through any extra grouping columns (e.g. RFW's subset/race column)
    for extra in ("subset", "group", "race", "fold", "split"):
        if extra in {c.lower() for c in pairs.columns}:
            src = {c.lower(): c for c in pairs.columns}[extra]
            out[extra] = pairs[src].to_numpy()
    out = out[(out.image_id_a >= 0) & (out.image_id_b >= 0)].reset_index(drop=True)
    return out


def load_embs(path, n_images=None, normalize=True, name="embeddings"):
    """Load an (N, D) embedding array indexed by image_id.  NaN rows = missing."""
    arr = np.load(path)
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 2:
        raise SystemExit(f"{name}: expected 2-D array, got {arr.shape}")
    if n_images is not None and len(arr) != n_images:
        raise SystemExit(
            f"{name}: {len(arr)} rows but manifest has {n_images} images; these "
            f"must be aligned by image_id (row i == image_id i)"
        )
    if normalize:
        arr = l2_normalize(arr)
    n_missing = int((~np.isfinite(arr).all(axis=1)).sum())
    if n_missing:
        print(f"[load] {name}: {n_missing} missing rows")
    return arr


def l2_normalize(x, axis=-1, eps=1e-12):
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(n, eps)


# --------------------------------------------------------------------------
# scores, thresholds, error labels
# --------------------------------------------------------------------------

def cosine_scores(face_embs, pairs):
    """Cosine similarity per pair.  NaN where either embedding is missing."""
    a = face_embs[pairs["image_id_a"].to_numpy()]
    b = face_embs[pairs["image_id_b"].to_numpy()]
    return np.einsum("ij,ij->i", a, b).astype(np.float64)


def pick_threshold(scores, same, mode="eer"):
    """Choose an operating threshold.

    mode:
      "eer"          equal error rate
      "fmr:1e-3"     lowest threshold whose FMR is <= the target
      "fnmr:1e-2"    lowest threshold whose FNMR is <= the target
      "value:0.28"   use the number verbatim
    """
    ok = np.isfinite(scores)
    s, y = scores[ok], np.asarray(same, dtype=bool)[ok]
    gen, imp = s[y], s[~y]
    if len(gen) == 0 or len(imp) == 0:
        raise SystemExit("need both genuine and impostor pairs to set a threshold")

    if mode.startswith("value:"):
        return float(mode.split(":", 1)[1])

    grid = np.unique(np.concatenate([gen, imp]))
    # thresholds tried between observed scores
    grid = np.concatenate([[grid[0] - 1e-6], (grid[:-1] + grid[1:]) / 2.0,
                           [grid[-1] + 1e-6]])
    fmr = np.array([(imp >= t).mean() for t in grid])
    fnmr = np.array([(gen < t).mean() for t in grid])

    if mode == "eer":
        return float(grid[int(np.argmin(np.abs(fmr - fnmr)))])
    if mode.startswith("fmr:"):
        target = float(mode.split(":", 1)[1])
        feasible = np.flatnonzero(fmr <= target)
        if len(feasible) == 0:
            warnings.warn(f"FMR never reaches {target}; falling back to EER")
            return pick_threshold(scores, same, "eer")
        return float(grid[feasible[0]])
    if mode.startswith("fnmr:"):
        target = float(mode.split(":", 1)[1])
        feasible = np.flatnonzero(fnmr <= target)
        if len(feasible) == 0:
            warnings.warn(f"FNMR never reaches {target}; falling back to EER")
            return pick_threshold(scores, same, "eer")
        return float(grid[feasible[-1]])
    raise SystemExit(f"unknown threshold mode: {mode}")


def error_signal(scores, same, tau):
    """Return (loss, is_error).

    ``loss`` is a signed margin, positive iff the pair is misclassified, with the
    same units for both error types:

        genuine  pair:  loss = tau - score     (positive => false non-match)
        impostor pair:  loss = score - tau     (positive => false match)

    Use ``loss`` rather than ``is_error`` whenever the operating point is strict:
    at FMR=1e-4 there are too few positives for any of these methods to cluster on,
    but the margin is dense and carries the same ordering information.
    """
    same = np.asarray(same, dtype=bool)
    loss = np.where(same, tau - scores, scores - tau)
    return loss, loss > 0


# --------------------------------------------------------------------------
# pair representations
# --------------------------------------------------------------------------

REPS = ("mean", "absdiff", "prod", "minmax", "mean_absdiff")


def pair_rep(embs, pairs, kind="mean_absdiff"):
    """Build an order-invariant representation of each pair.

    Verification pairs are unordered, so any representation must be symmetric in
    (a, b) or the method will waste capacity learning that (a,b) and (b,a) are the
    same thing.  All options here are symmetric.

    "mean"          (u+v)/2, renormalised.  The only option that stays inside
                    CLIP's text-aligned space, so it is the one to use when the
                    slice has to be *named*.
    "absdiff"       |u-v|.  Captures how the two faces differ, which is what drives
                    false non-matches, but is not text-alignable.
    "prod"          u*v, the elementwise contribution to the cosine score.
    "minmax"        concat(min(u,v), max(u,v)).  Symmetric and lossless.
    "mean_absdiff"  concat(mean, absdiff).  Default: expressive for discovery, and
                    the leading block is still CLIP-aligned so ``clip_block`` can
                    recover a nameable direction.
    """
    ia = pairs["image_id_a"].to_numpy()
    ib = pairs["image_id_b"].to_numpy()
    u, v = embs[ia], embs[ib]
    if kind == "mean":
        return l2_normalize((u + v) / 2.0)
    if kind == "absdiff":
        return np.abs(u - v)
    if kind == "prod":
        return u * v
    if kind == "minmax":
        return np.concatenate([np.minimum(u, v), np.maximum(u, v)], axis=1)
    if kind == "mean_absdiff":
        return np.concatenate([l2_normalize((u + v) / 2.0), np.abs(u - v)], axis=1)
    raise SystemExit(f"unknown pair representation: {kind} (choose from {REPS})")


def clip_block(kind, dim):
    """Slice of a ``pair_rep`` output that still lives in CLIP's joint space.

    Returns a ``slice`` or None.  Used to name a learned direction: only the
    CLIP-aligned block can be compared against text embeddings.
    """
    if kind == "mean":
        return slice(0, dim)
    if kind == "mean_absdiff":
        return slice(0, dim)
    return None


# --------------------------------------------------------------------------
# the pair set: one object every baseline consumes
# --------------------------------------------------------------------------

class PairSet:
    """Pairs + scores + errors + description embeddings, with missing data dropped."""

    def __init__(self, images, pairs, face_embs, desc_embs, threshold_mode="eer"):
        finite_face = np.isfinite(face_embs).all(axis=1)
        finite_desc = np.isfinite(desc_embs).all(axis=1)
        usable = finite_face & finite_desc
        ia = pairs["image_id_a"].to_numpy()
        ib = pairs["image_id_b"].to_numpy()
        keep = usable[ia] & usable[ib]
        n_drop = int((~keep).sum())
        if n_drop:
            print(f"[pairs] dropped {n_drop}/{len(pairs)} pairs with a missing "
                  f"embedding on one side")
        self.pairs = pairs.loc[keep].reset_index(drop=True)
        self.images = images
        self.face_embs = face_embs
        self.desc_embs = desc_embs

        if "score" in self.pairs.columns and self.pairs["score"].notna().all():
            self.scores = self.pairs["score"].to_numpy(dtype=np.float64)
            print("[pairs] using precomputed scores from the pair table")
        else:
            self.scores = cosine_scores(face_embs, self.pairs)

        self.threshold_mode = threshold_mode
        self.tau = pick_threshold(self.scores, self.pairs["same"].to_numpy(),
                                  threshold_mode)
        self.loss, self.is_error = error_signal(
            self.scores, self.pairs["same"].to_numpy(), self.tau)
        self.same = self.pairs["same"].to_numpy(dtype=bool)

        gen_err = float(self.is_error[self.same].mean()) if self.same.any() else 0.0
        imp_err = (float(self.is_error[~self.same].mean())
                   if (~self.same).any() else 0.0)
        print(f"[pairs] {len(self.pairs)} pairs | tau={self.tau:.4f} "
              f"({threshold_mode}) | FNMR={gen_err:.4%} ({int(self.is_error[self.same].sum())} "
              f"errors) | FMR={imp_err:.4%} ({int(self.is_error[~self.same].sum())} errors)")
        if min(int(self.is_error[self.same].sum()),
               int(self.is_error[~self.same].sum())) < 30:
            print("[pairs] NOTE: fewer than 30 errors in one pair type. Binary-error "
                  "discovery will be unstable here; prefer --target margin.")

        # identity of each side, for leakage-free splits
        ident = images.set_index("image_id")["identity"]
        self.ident_a = ident.reindex(self.pairs["image_id_a"]).to_numpy()
        self.ident_b = ident.reindex(self.pairs["image_id_b"]).to_numpy()

    # -- views ------------------------------------------------------------
    def mask(self, pair_type):
        if pair_type == "genuine":
            return self.same
        if pair_type == "impostor":
            return ~self.same
        if pair_type in ("all", None):
            return np.ones(len(self.pairs), dtype=bool)
        raise SystemExit(f"unknown pair_type: {pair_type}")

    def target(self, kind, pair_type="all"):
        """Return the per-pair supervision signal restricted to ``pair_type``."""
        m = self.mask(pair_type)
        if kind == "error":
            return self.is_error[m].astype(np.float64)
        if kind == "margin":
            return self.loss[m]
        raise SystemExit(f"unknown target: {kind} (choose 'error' or 'margin')")

    def rep(self, kind="mean_absdiff", pair_type="all"):
        m = self.mask(pair_type)
        return pair_rep(self.desc_embs, self.pairs.loc[m], kind)

    def image_view(self, pair_type="all", target="margin"):
        """Collapse pair errors onto images.

        Every pair error is attributed to *both* faces, which is what you have to do
        to run these methods in their published (per-image) form.  Returns a
        DataFrame with image_id, n_pairs, n_errors, err_rate, mean_margin.
        """
        m = self.mask(pair_type)
        sub = self.pairs.loc[m]
        val = self.loss[m] if target == "margin" else self.is_error[m].astype(float)
        rows = pd.DataFrame({
            "image_id": np.concatenate([sub["image_id_a"].to_numpy(),
                                        sub["image_id_b"].to_numpy()]),
            "err": np.concatenate([self.is_error[m].astype(float)] * 2),
            "val": np.concatenate([val] * 2),
        })
        agg = rows.groupby("image_id").agg(
            n_pairs=("err", "size"), n_errors=("err", "sum"),
            err_rate=("err", "mean"), value=("val", "mean")).reset_index()
        return agg

    def identity_folds(self, n_folds=2, seed=0):
        """Assign each pair to a fold such that no identity spans two folds.

        Face pairs leak badly if you split at random: the same identity appears in
        many pairs, so a random split lets the model memorise identities rather than
        find a transferable failure direction.  Pairs whose two identities land in
        different folds are marked -1 and excluded from held-out evaluation.
        """
        rng = np.random.default_rng(seed)
        idents = np.unique(np.concatenate([self.ident_a, self.ident_b]))
        assign = {k: int(v) for k, v in
                  zip(idents, rng.integers(0, n_folds, size=len(idents)))}
        fa = np.array([assign[i] for i in self.ident_a])
        fb = np.array([assign[i] for i in self.ident_b])
        folds = np.where(fa == fb, fa, -1)
        return folds


# --------------------------------------------------------------------------
# CLIP text encoding (optional dependency, with an escape hatch)
# --------------------------------------------------------------------------

def encode_text(terms, model_name="ViT-B-32", pretrained="openai", device=None,
                templates=None, batch_size=256):
    """Encode vocabulary terms with prompt ensembling.  Returns (T, D) unit vectors.

    Requires open_clip (preferred) or the original openai clip package.  If you
    would rather not install either, precompute the array yourself and pass
    --text-embs / --terms-file to the baselines instead.
    """
    try:
        from .vocab import TEMPLATES
    except ImportError:
        from vocab import TEMPLATES
    templates = TEMPLATES if templates is None else templates

    try:
        import torch
        import open_clip
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model, _, _ = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=device)
        tokenizer = open_clip.get_tokenizer(model_name)
        encode = model.encode_text
    except ImportError:
        try:
            import torch
            import clip as openai_clip
            device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            model, _ = openai_clip.load(model_name.replace("-", "/"), device=device)
            tokenizer = openai_clip.tokenize
            encode = model.encode_text
        except ImportError:
            raise SystemExit(
                "encode_text needs open_clip (pip install open_clip_torch) or the "
                "openai clip package. Alternatively precompute text embeddings and "
                "pass --text-embs/--terms-file."
            )

    out = []
    with torch.no_grad():
        for t in terms:
            prompts = [tmpl.format(t) for tmpl in templates]
            toks = tokenizer(prompts).to(device)
            feats = encode(toks).float()
            feats = feats / feats.norm(dim=-1, keepdim=True)
            out.append(feats.mean(0).cpu().numpy())
    out = np.stack(out).astype(np.float32)
    return l2_normalize(out)


def resolve_text_embeddings(args, desc_dim):
    """Get (terms, text_embs) from either --text-embs or a live CLIP encode."""
    try:
        from . import vocab as vocab_mod
    except ImportError:
        import vocab as vocab_mod

    if getattr(args, "text_embs", None):
        text_embs = np.asarray(np.load(args.text_embs), dtype=np.float32)
        if getattr(args, "terms_file", None):
            terms = vocab_mod.load_terms(args.terms_file)
        else:
            terms = [f"term_{i}" for i in range(len(text_embs))]
        if len(terms) != len(text_embs):
            raise SystemExit(
                f"--terms-file has {len(terms)} terms but --text-embs has "
                f"{len(text_embs)} rows")
        text_embs = l2_normalize(text_embs)
    else:
        terms = (vocab_mod.load_terms(args.terms_file)
                 if getattr(args, "terms_file", None) else vocab_mod.DEFAULT_VOCAB)
        print(f"[text] encoding {len(terms)} vocabulary terms with "
              f"{args.clip_model}/{args.clip_pretrained} ...")
        text_embs = encode_text(terms, args.clip_model, args.clip_pretrained)

    if text_embs.shape[1] != desc_dim:
        raise SystemExit(
            f"text embeddings are {text_embs.shape[1]}-d but the description "
            f"embeddings are {desc_dim}-d; they must come from the same CLIP model"
        )
    return terms, text_embs


def name_vector(vec, text_embs, terms, topk=5, signed=True):
    """Describe a direction or centroid by its nearest vocabulary terms."""
    v = np.asarray(vec, dtype=np.float32).ravel()
    nrm = np.linalg.norm(v)
    if nrm < 1e-9:
        return []
    sims = text_embs @ (v / nrm)
    order = np.argsort(-sims)
    out = [(terms[i], float(sims[i])) for i in order[:topk]]
    if signed:
        tail = np.argsort(sims)
        out += [(terms[i], float(sims[i])) for i in tail[:topk]]
    return out


def describe(pairs_of_term_score, n=5):
    return ", ".join(f"{t}" for t, _ in pairs_of_term_score[:n])


# --------------------------------------------------------------------------
# standard slice output format (consumed by evaluate.py)
# --------------------------------------------------------------------------

SLICE_COLUMNS = ["slice_id", "pair_type", "unit_type", "rank", "k", "size",
                 "err_rate", "base_err_rate", "lift", "mean_margin", "margin_delta",
                 "coherence", "target", "description"]
MEMBER_COLUMNS = ["slice_id", "unit_type", "unit_id", "rank_in_slice", "weight"]


def write_slices(out_dir, method, slices, members):
    """Write the two-file slice format.

    slices.csv   one row per discovered slice:
                 method, slice_id, pair_type, unit_type, rank, size, k,
                 err_rate, base_err_rate, lift, mean_margin, description, extra
    members.csv  slice_id, unit_type, unit_id, rank_in_slice, weight
    """
    os.makedirs(out_dir, exist_ok=True)
    sdf = pd.DataFrame(slices) if slices else pd.DataFrame(columns=SLICE_COLUMNS)
    mdf = pd.DataFrame(members) if members else pd.DataFrame(columns=MEMBER_COLUMNS)
    if not slices:
        print("[out] WARNING: no slices were produced. Every block was skipped - "
              "scroll up for the reason (usually a degenerate threshold; try "
              "--target margin).")
    sdf.insert(0, "method", method)
    sdf.to_csv(os.path.join(out_dir, "slices.csv"), index=False)
    mdf.to_csv(os.path.join(out_dir, "members.csv"), index=False)
    print(f"[out] {len(sdf)} slices -> {os.path.join(out_dir, 'slices.csv')}")
    print(f"[out] {len(mdf)} memberships -> {os.path.join(out_dir, 'members.csv')}")
    return sdf, mdf


def slice_stats(idx, is_error, loss, base_err, base_margin):
    """Summary statistics for a candidate slice given member indices."""
    idx = np.asarray(idx, dtype=int)
    if len(idx) == 0:
        return dict(size=0, err_rate=np.nan, base_err_rate=base_err, lift=np.nan,
                    mean_margin=np.nan, margin_delta=np.nan)
    er = float(is_error[idx].mean())
    mm = float(loss[idx].mean())
    return dict(
        size=int(len(idx)),
        err_rate=er,
        base_err_rate=float(base_err),
        lift=float(er / base_err) if base_err > 0 else np.nan,
        mean_margin=mm,
        margin_delta=float(mm - base_margin),
    )


# --------------------------------------------------------------------------
# shared CLI
# --------------------------------------------------------------------------

def add_common_args(ap):
    ap.add_argument("--pairs", required=True,
                    help="CSV of verification pairs (image_id_a/image_id_b/same, "
                         "or path_a/path_b/same)")
    ap.add_argument("--images", required=True,
                    help="CSV manifest: image_id, path, identity")
    ap.add_argument("--face-embs", required=True,
                    help="(N, D) .npy of FR embeddings, row i == image_id i")
    ap.add_argument("--desc-embs", required=True,
                    help="(N, C) .npy of description embeddings (CLIP), "
                         "row i == image_id i")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--threshold", default="eer",
                    help="eer | fmr:1e-3 | fnmr:1e-2 | value:0.28")
    ap.add_argument("--pair-type", default="both",
                    choices=["genuine", "impostor", "both"],
                    help="genuine -> FNMR slices, impostor -> FMR slices")
    ap.add_argument("--granularity", default="both",
                    choices=["pair", "image", "both"])
    ap.add_argument("--target", default="margin", choices=["error", "margin"],
                    help="binary error label, or the continuous signed margin "
                         "(default; far more stable at strict thresholds)")
    ap.add_argument("--rep", default="mean_absdiff", choices=list(REPS))
    ap.add_argument("--n-slices", type=int, default=5)
    ap.add_argument("--slice-size", type=int, default=200,
                    help="k: number of units in each reported slice")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--clip-model", default="ViT-B-32")
    ap.add_argument("--clip-pretrained", default="openai")
    ap.add_argument("--text-embs", default=None,
                    help="precomputed (T, C) .npy text embeddings; skips loading CLIP")
    ap.add_argument("--terms-file", default=None,
                    help="newline-delimited vocabulary (defaults to MAAD-Face + "
                         "covariate terms)")
    return ap


def build_pairset(args):
    images = load_manifest(args.images)
    pairs = load_pairs(args.pairs, images)
    face = load_embs(args.face_embs, len(images), normalize=True, name="face embs")
    desc = load_embs(args.desc_embs, len(images), normalize=True, name="desc embs")
    return PairSet(images, pairs, face, desc, args.threshold)


def pair_types(args):
    return ["genuine", "impostor"] if args.pair_type == "both" else [args.pair_type]


def granularities(args):
    return ["pair", "image"] if args.granularity == "both" else [args.granularity]


def dump_config(out_dir, args, extra=None):
    os.makedirs(out_dir, exist_ok=True)
    cfg = {k: v for k, v in vars(args).items()}
    if extra:
        cfg.update(extra)
    with open(os.path.join(out_dir, "config.json"), "w") as fh:
        json.dump(cfg, fh, indent=2, default=str)
