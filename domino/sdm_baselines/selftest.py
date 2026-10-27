"""Planted-slice self-test.

Generates a synthetic dataset shaped like RFW (identities, images, genuine and
impostor pairs) with one *known* failure mode planted in it, then runs all three
baselines end-to-end and checks that each recovers it.

The planted failure: images carrying the latent attribute ``eyeglasses`` are given
a noisier face embedding, so genuine pairs where *both* faces wear glasses score
lower (elevated FNMR) and impostor pairs where both wear glasses score higher
(elevated FMR).  Ground truth is therefore the pair-level group "both faces have
eyeglasses", which is exactly what --pair-attr-mode both evaluates.

This catches the bugs that are otherwise invisible until you have real results:
sign errors in the margin, pair/image index misalignment, identity leakage across
folds, and description vectors that have drifted out of CLIP's text space.

    python -m sdm_baselines.selftest            # run it
    python -m sdm_baselines.selftest --keep /tmp/sdm_demo   # keep the fixtures
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(_HERE))
    __package__ = os.path.basename(_HERE)

from . import b2t, evaluate, failure_directions, spotlight  # noqa: E402
from .common import l2_normalize  # noqa: E402

TERMS = ["eyeglasses", "a hat", "a smile", "a blurry photo", "a beard",
         "a plain background", "brown hair", "a profile view of a face"]
PLANTED = "eyeglasses"


def make_fixtures(out_dir, n_identities=250, per_id=8, desc_dim=48, face_dim=96,
                  n_pairs=3000, attr_rate=0.25, seed=0):
    rng = np.random.default_rng(seed)
    os.makedirs(out_dir, exist_ok=True)

    n_images = n_identities * per_id
    identity = np.repeat(np.arange(n_identities), per_id)
    image_id = np.arange(n_images)
    paths = [f"synth/id{identity[i]:04d}/img{i:06d}.jpg" for i in range(n_images)]

    # latent attributes; the planted one plus decoys
    attrs = {t: rng.random(n_images) < attr_rate for t in TERMS}

    # ---- description space (a stand-in for CLIP) -------------------------
    text_embs = l2_normalize(rng.standard_normal((len(TERMS), desc_dim)))
    desc = rng.standard_normal((n_images, desc_dim)) * 0.55
    for ti, t in enumerate(TERMS):
        desc[attrs[t]] += 1.0 * text_embs[ti]
    desc = l2_normalize(desc).astype(np.float32)

    # ---- face space: identity clusters, degraded by the planted attribute --
    centers = l2_normalize(rng.standard_normal((n_identities, face_dim)))
    noise = np.where(attrs[PLANTED], 0.85, 0.30)[:, None]
    face = centers[identity] + noise * rng.standard_normal((n_images, face_dim))
    face = l2_normalize(face).astype(np.float32)

    # ---- pairs -----------------------------------------------------------
    gen_a, gen_b = [], []
    while len(gen_a) < n_pairs:
        i = rng.integers(0, n_identities)
        x, y = rng.choice(per_id, size=2, replace=False)
        gen_a.append(i * per_id + x)
        gen_b.append(i * per_id + y)
    imp_a, imp_b = [], []
    while len(imp_a) < n_pairs:
        i, j = rng.integers(0, n_identities, size=2)
        if i == j:
            continue
        imp_a.append(i * per_id + rng.integers(0, per_id))
        imp_b.append(j * per_id + rng.integers(0, per_id))

    pairs = pd.DataFrame({
        "image_id_a": np.concatenate([gen_a, imp_a]).astype(int),
        "image_id_b": np.concatenate([gen_b, imp_b]).astype(int),
        "same": np.concatenate([np.ones(n_pairs, bool), np.zeros(n_pairs, bool)]),
    })
    # plant an FMR effect too: impostor pairs where both wear glasses look alike
    both = attrs[PLANTED][pairs.image_id_a.to_numpy()] & \
        attrs[PLANTED][pairs.image_id_b.to_numpy()]
    bump = np.where(both & ~pairs["same"].to_numpy(), 0.35, 0.0)
    face_bump_dir = l2_normalize(rng.standard_normal((1, face_dim)))[0]
    for idx in np.flatnonzero(bump > 0):
        for col in ("image_id_a", "image_id_b"):
            k = int(pairs.iloc[idx][col])
            face[k] = l2_normalize(face[k] + 0.55 * face_bump_dir)

    images = pd.DataFrame({"image_id": image_id, "path": paths,
                           "identity": [f"id{i:04d}" for i in identity]})
    att = pd.DataFrame({"image_id": image_id})
    for t in TERMS:
        att[t.replace(" ", "_")] = attrs[t].astype(int)

    images.to_csv(os.path.join(out_dir, "images.csv"), index=False)
    pairs.to_csv(os.path.join(out_dir, "pairs.csv"), index=False)
    att.to_csv(os.path.join(out_dir, "attributes.csv"), index=False)
    np.save(os.path.join(out_dir, "face_embs.npy"), face)
    np.save(os.path.join(out_dir, "desc_embs.npy"), desc)
    np.save(os.path.join(out_dir, "text_embs.npy"), text_embs.astype(np.float32))
    with open(os.path.join(out_dir, "terms.txt"), "w") as fh:
        fh.write("\n".join(TERMS) + "\n")
    return out_dir


def _base_args(d, out):
    return ["--pairs", os.path.join(d, "pairs.csv"),
            "--images", os.path.join(d, "images.csv"),
            "--face-embs", os.path.join(d, "face_embs.npy"),
            "--desc-embs", os.path.join(d, "desc_embs.npy"),
            "--text-embs", os.path.join(d, "text_embs.npy"),
            "--terms-file", os.path.join(d, "terms.txt"),
            "--out", out]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--keep", default=None, help="directory to keep fixtures in")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    d = args.keep or tempfile.mkdtemp(prefix="sdm_selftest_")
    make_fixtures(d, seed=args.seed)
    print(f"[selftest] fixtures in {d}\n")

    runs = {}
    print("=" * 70, "\nFAILURE DIRECTIONS\n", "=" * 70, sep="")
    runs["svm"] = os.path.join(d, "run_svm")
    failure_directions.main(_base_args(d, runs["svm"]) +
                            ["--target", "margin", "--n-slices", "2",
                             "--slice-size", "150", "--seed", str(args.seed)])

    print("\n", "=" * 70, "\nB2T\n", "=" * 70, sep="")
    runs["b2t"] = os.path.join(d, "run_b2t")
    b2t.main(_base_args(d, runs["b2t"]) +
             ["--target", "margin", "--n-slices", "2", "--slice-size", "150",
              "--n-boot", "200", "--seed", str(args.seed)])

    print("\n", "=" * 70, "\nSPOTLIGHT\n", "=" * 70, sep="")
    runs["spotlight"] = os.path.join(d, "run_spotlight")
    spotlight.main(_base_args(d, runs["spotlight"]) +
                   ["--target", "margin", "--n-slices", "2", "--slice-size", "150",
                    "--pca", "32", "--steps", "150", "--seed", str(args.seed)])

    print("\n", "=" * 70, "\nEVALUATION\n", "=" * 70, sep="")
    eval_out = os.path.join(d, "run_eval")
    evaluate.main(["--runs", *runs.values(),
                   "--pairs", os.path.join(d, "pairs.csv"),
                   "--images", os.path.join(d, "images.csv"),
                   "--attributes", os.path.join(d, "attributes.csv"),
                   "--out", eval_out, "--pair-attr-mode", "both"])

    # ---- assertions ------------------------------------------------------
    summary = pd.read_csv(os.path.join(eval_out, "summary.csv"))
    per_slice = pd.read_csv(os.path.join(eval_out, "per_slice.csv"))
    planted = PLANTED.replace(" ", "_")
    failures = []
    for method in ("failure_directions", "b2t", "spotlight"):
        sub = per_slice[per_slice.method == method]
        if sub.empty:
            failures.append(f"{method}: produced no slices")
            continue
        hit = sub[(sub.best_group == planted) & (sub.precision_lift > 1.2)]
        if hit.empty:
            best = sub.loc[sub.precision_lift.idxmax()]
            failures.append(
                f"{method}: did not recover '{planted}' "
                f"(best was {best.best_group} lift={best.precision_lift:.2f})")
        else:
            b = hit.loc[hit.precision.idxmax()]
            print(f"[ok] {method}: recovered '{planted}' at precision "
                  f"{b.precision:.3f} (prevalence {b.prevalence:.3f}, "
                  f"lift {b.precision_lift:.2f}x, err lift {b.err_lift:.2f}x)")

    print()
    if failures:
        for f in failures:
            print(f"[FAIL] {f}")
        if not args.keep:
            shutil.rmtree(d, ignore_errors=True)
        raise SystemExit(1)
    print("[selftest] all three baselines recovered the planted slice.")
    if not args.keep:
        shutil.rmtree(d, ignore_errors=True)
    else:
        print(f"[selftest] fixtures kept in {d}")


if __name__ == "__main__":
    main()
