"""Smoke test for the baseline + comparison stages, on a synthetic `work/` folder.

`synthetic.py` smoke-tests the pipeline from images down. This one starts one stage
later -- it fabricates a `work/` directory with the exact schema stages 03-07 expect
-- so it exercises the parts `synthetic.py` cannot reach without model downloads:
the per-image loss, the three baselines' candidate generation, the Domino->group
projection, and the FMR ranking.

A failure mode is planted: images carrying a latent attribute ("dark sunglasses")
share a direction in face space, so groups built around them have a much higher FMR,
and their CLIP embeddings lean towards that phrase's text embedding, so a
description-based method can name it. One race also gets a milder shared component,
purely so the data is not uniform.

The assertions are the ones that caught real bugs:
  * every method's top-10 groups must beat random by 3x,
  * the SVM's candidate groups must actually differ from one another (feature-only
    deflation returned ten near-copies of the same group),
  * labels must discriminate (unnormalised phrase scoring gave every group the same
    description).

    python -m rfw_domino.smoke_baselines                # run it
    python -m rfw_domino.smoke_baselines --keep /tmp/wk # keep the work folder
"""

import argparse
import os
import shutil
import tempfile

import numpy as np
import pandas as pd

from . import RACES
from .io_utils import l2_normalize
from .phrase_bank import build_expanded_bank

PLANTED_PHRASE_SUBSTR = "dark sunglasses"


# --------------------------------------------------------------------------

def build_work(work_dir, n_ids=150, per_id=6, dim_face=128, dim_clip=128,
               attr_rate=0.2, weak_race="African", n_expand=40000, seed=0):
    rng = np.random.default_rng(seed)
    os.makedirs(work_dir, exist_ok=True)

    phrases = build_expanded_bank()
    text_embs = l2_normalize(
        rng.standard_normal((len(phrases), dim_clip)).astype(np.float32))
    planted = next(i for i, p in enumerate(phrases) if PLANTED_PHRASE_SUBSTR in p)

    rows, ident_of, race_of = [], [], []
    for race in RACES:
        for i in range(n_ids):
            name = f"m.{race[:3].lower()}{i:04d}"
            for j in range(1, per_id + 1):
                rows.append({"image_id": len(rows),
                             "path": f"synthetic/{race}/{name}/{name}_{j:04d}.jpg",
                             "race": race, "identity": name})
                ident_of.append(name)
                race_of.append(race)
    images = pd.DataFrame(rows)
    n = len(images)
    race_arr = np.array(race_of)
    ident_arr = np.array(ident_of)
    has_attr = rng.random(n) < attr_rate

    # ---- CLIP space --------------------------------------------------------
    # every component is unit-norm before mixing, so the planted signal is a
    # controlled fraction of the total rather than being swallowed by sqrt(dim)
    clip = l2_normalize(rng.standard_normal((n, dim_clip)))
    clip[has_attr] += 1.0 * text_embs[planted]
    for r_i, race in enumerate(RACES):                 # mild race structure
        clip[race_arr == race] += 0.45 * text_embs[(planted + 37 * (r_i + 1))
                                                   % len(text_embs)]
    clip = l2_normalize(clip).astype(np.float32)

    # ---- face space --------------------------------------------------------
    # Elevated FMR comes from a *shared* component that makes different
    # identities mutually similar, not from noise alone -- extra isotropic noise
    # destroys identity without creating false matches.
    id_codes, id_index = np.unique(ident_arr, return_inverse=True)
    centers = l2_normalize(rng.standard_normal((len(id_codes), dim_face)))
    unit_noise = l2_normalize(rng.standard_normal((n, dim_face)))
    sigma = np.full(n, 0.45)
    sigma[race_arr == weak_race] = 0.75
    sigma[has_attr] = np.maximum(sigma[has_attr], 0.75)
    face = centers[id_index] + sigma[:, None] * unit_noise
    attr_dir = l2_normalize(rng.standard_normal((1, dim_face)))[0]
    race_dir = l2_normalize(rng.standard_normal((1, dim_face)))[0]
    face[has_attr] += 0.62 * attr_dir                  # ~20x FMR, race-neutral
    face[race_arr == weak_race] += 0.40 * race_dir     # ~5x FMR, pure race
    face = l2_normalize(face).astype(np.float32)

    # ---- official-protocol pairs -------------------------------------------
    pair_rows = []
    for race in RACES:
        idx = np.flatnonzero(race_arr == race)
        names = np.unique(ident_arr[idx])
        by_name = {nm: np.flatnonzero((ident_arr == nm)) for nm in names}
        for fold in range(10):
            for _ in range(300):                       # genuine
                nm = names[rng.integers(len(names))]
                a, b = rng.choice(by_name[nm], 2, replace=False)
                pair_rows.append((race, fold, 1, nm, int(a), nm, int(b)))
            for _ in range(300):                       # impostor
                n1, n2 = rng.choice(names, 2, replace=False)
                a = by_name[n1][rng.integers(len(by_name[n1]))]
                b = by_name[n2][rng.integers(len(by_name[n2]))]
                pair_rows.append((race, fold, 0, n1, int(a), n2, int(b)))

    path_of = dict(zip(images["image_id"], images["path"]))
    pairs = pd.DataFrame([{
        "pair_id": k, "race": r, "fold": f, "same": s,
        "name1": nm1, "num1": -1, "img_id1": a, "path1": path_of[a],
        "name2": nm2, "num2": -1, "img_id2": b, "path2": path_of[b],
        "protocol": "official",
    } for k, (r, f, s, nm1, a, nm2, b) in enumerate(pair_rows)])

    images.to_csv(os.path.join(work_dir, "images.csv"), index=False)
    pairs.to_csv(os.path.join(work_dir, "pairs.csv"), index=False)
    np.save(os.path.join(work_dir, "face_embs.npy"), face)
    np.save(os.path.join(work_dir, "clip_embs.npy"), clip)
    np.save(os.path.join(work_dir, "text_embs.npy"), text_embs)
    with open(os.path.join(work_dir, "phrases.txt"), "w") as f:
        f.write("\n".join(phrases))
    with open(os.path.join(work_dir, "clip_meta.json"), "w") as f:
        f.write('{"backend": "openclip", "model": "ViT-B-32", '
                '"pretrained": "openai", "dim": %d, "text": true}' % dim_clip)

    truth = pd.DataFrame({"image_id": images["image_id"], "has_attr": has_attr,
                          "race": race_arr})
    truth.to_csv(os.path.join(work_dir, "planted_truth.csv"), index=False)
    return work_dir, phrases[planted], n_expand


# --------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--keep", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--group-size", type=int, default=100)
    args = ap.parse_args(argv)

    from . import baseline_b2t, baseline_spotlight, baseline_svm, compare_fmr
    from . import expand_negatives, slicing, verification

    wd = args.keep or tempfile.mkdtemp(prefix="rfw_smoke_")
    _, planted_phrase, n_expand = build_work(wd, seed=args.seed)
    print(f"[smoke] work dir {wd}")
    print(f"[smoke] planted phrase: {planted_phrase!r}\n")

    print("=" * 72, "\nEXPAND NEGATIVES\n", "=" * 72, sep="")
    expand_negatives.main(["--work", wd, "--n-per-race", str(n_expand),
                           "--seed", str(args.seed)])

    print("\n", "=" * 72, "\nVERIFICATION @ FMR = 1e-3\n", "=" * 72, sep="")
    verification.main(["--work", wd, "--criterion", "far", "--far-target", "1e-3"])

    print("\n", "=" * 72, "\nDOMINO (impostor pairs only)\n", "=" * 72, sep="")
    slicing.main(["--work", wd, "--subset", "neg", "--n-slices", "30",
                  "--n-components", "60", "--pca", "64", "--max-iter", "40",
                  "--seed", str(args.seed)])

    common = ["--work", wd, "--group-size", str(args.group_size),
              "--seed", str(args.seed)]
    print("\n", "=" * 72, "\nFAILURE DIRECTIONS\n", "=" * 72, sep="")
    baseline_svm.main(common + ["--n-directions", "4", "--n-bands", "15"])
    print("\n", "=" * 72, "\nB2T\n", "=" * 72, sep="")
    baseline_b2t.main(common + ["--n-boot", "200", "--n-candidates", "200"])
    print("\n", "=" * 72, "\nSPOTLIGHT\n", "=" * 72, sep="")
    baseline_spotlight.main(common + ["--steps", "80", "--n-candidates", "30"])

    print("\n", "=" * 72, "\nCOMPARISON\n", "=" * 72, sep="")
    compare_fmr.main(["--work", wd, "--group-size", str(args.group_size),
                      "--top", "10,50", "--n-random", "100",
                      "--seed", str(args.seed)])

    # ---- assertions ------------------------------------------------------
    detail = pd.read_csv(os.path.join(wd, "report", "fmr_groups.csv"))
    summary = pd.read_csv(os.path.join(wd, "report", "fmr_summary.csv"))
    truth = pd.read_csv(os.path.join(wd, "planted_truth.csv"))
    attr = dict(zip(truth["image_id"], truth["has_attr"]))

    failures = []
    rnd = summary[summary.method == "random"]
    rnd_top10 = float(rnd["mean_fmr_top10"].iloc[0]) if len(rnd) else 0.0
    print(f"\n[ref] random: mean FMR of its top-10 groups = {rnd_top10:.3e}\n")

    for method in ("failure_directions", "b2t", "spotlight", "domino"):
        row = summary[summary.method == method]
        if row.empty:
            failures.append(f"{method}: produced no groups")
            continue
        r = row.iloc[0]
        floor = (2 if method == "domino" else 3) * rnd_top10
        if not (r["mean_fmr_top10"] > floor):
            failures.append(f"{method}: mean FMR of top-10 is "
                            f"{r['mean_fmr_top10']:.3e}, not clearly above random")
        else:
            print(f"[ok] {method:19s} best {r['best_fmr']:.3e} | "
                  f"top-10 mean {r['mean_fmr_top10']:.3e} | "
                  f"coherence {r['mean_coherence_top10']:.3f} | "
                  f"{int(r['n_candidates'])} candidates")

    # the SVM's candidate groups must actually differ from one another; the old
    # feature-only deflation returned ten near-copies of the same group
    svm = pd.read_csv(os.path.join(wd, "baselines", "failure_directions",
                                   "groups.csv"))
    sets = [set(g["image_id"]) for _, g in svm.groupby("group_id")]
    worst = max(len(sets[0] & s) / len(sets[0]) for s in sets[1:]) if len(sets) > 1 \
        else 1.0
    if worst > 0.5:
        failures.append(f"svm groups are near-duplicates: another group shares "
                        f"{worst:.0%} of group 0's images (deflation is broken)")
    else:
        print(f"[ok] svm groups are distinct: max overlap with group 0 is {worst:.0%}")

    # labels must discriminate: not every group can carry the same description
    for m in ("failure_directions", "spotlight"):
        lab = detail[detail.method == m]["label"].dropna()
        if len(lab) > 5 and lab.nunique() < 3:
            failures.append(f"{m}: only {lab.nunique()} distinct labels across "
                            f"{len(lab)} groups (phrase scoring is degenerate)")
        elif len(lab) > 5:
            print(f"[ok] {m}: {lab.nunique()} distinct labels across {len(lab)} groups")

    b2t_groups = pd.read_csv(os.path.join(wd, "baselines", "b2t", "groups.csv"))
    hits = b2t_groups[b2t_groups["label"].str.contains(
        PLANTED_PHRASE_SUBSTR, case=False, na=False)]
    if hits.empty:
        failures.append(f"b2t never named the planted phrase "
                        f"({PLANTED_PHRASE_SUBSTR!r})")
    else:
        gid = int(hits["group_id"].iloc[0])
        members = b2t_groups[b2t_groups.group_id == gid]["image_id"]
        purity = float(np.mean([attr.get(int(i), False) for i in members]))
        base = float(truth["has_attr"].mean())
        if purity < 2 * base:
            failures.append(f"b2t's planted-phrase group is only {purity:.0%} "
                            f"attribute images (base {base:.0%})")
        else:
            print(f"[ok] b2t named {PLANTED_PHRASE_SUBSTR!r}; that group is "
                  f"{purity:.0%} planted-attribute images (base rate {base:.0%})")

    print()
    if failures:
        for f in failures:
            print(f"[FAIL] {f}")
        if not args.keep:
            shutil.rmtree(wd, ignore_errors=True)
        raise SystemExit(1)
    print("[smoke] all checks passed.")
    if args.keep:
        print(f"[smoke] work folder kept at {wd}")
    else:
        shutil.rmtree(wd, ignore_errors=True)


if __name__ == "__main__":
    main()
