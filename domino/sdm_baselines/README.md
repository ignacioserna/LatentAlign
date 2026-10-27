# Slice-discovery baselines for face verification

Three published slice-discovery methods, adapted to verification, none of which
requires retraining anything:

| script | method | needs |
|---|---|---|
| `failure_directions.py` | Jain, Lawrence, Moitra & Madry, ICLR 2023 | FR embeddings + CLIP embeddings |
| `b2t.py` | Kim et al., CVPR 2024 (Bias-to-Text) | FR embeddings + CLIP image & text embeddings |
| `spotlight.py` | d'Eon et al., FAccT 2022 | FR embeddings + any embedding space |

`evaluate.py` scores all of them — and DOMINO, and your method — with the same
Precision@k protocol. `selftest.py` runs the whole thing on synthetic data with a
planted failure mode and asserts each baseline recovers it.

## Input schema

Four files, all keyed by a contiguous integer `image_id`:

```
images.csv       image_id, path, identity
pairs.csv        image_id_a, image_id_b, same        (or path_a, path_b, same)
face_embs.npy    (N, D) float32   FR embeddings, row i == image_id i
clip_embs.npy    (N, C) float32   CLIP image embeddings, row i == image_id i
```

Rows of `NaN` in either `.npy` mark missing embeddings; pairs touching them are
dropped with a count printed. This is the same contract as your `02b` importer, so
the array it writes to `work.face_embs_npy` drops straight in.

For evaluation you also need ground-truth attributes:

```
attributes.csv   image_id, <attr_1>, <attr_2>, ...
```

Values `> 0` count as positive, so MAAD-Face's `{1, -1, 0}` encoding works as-is.
For RFW the minimum viable ground truth is the four subset labels as four binary
columns.

## Running

```bash
python -m sdm_baselines.selftest        # verify the install; ~30s, no data needed

COMMON="--pairs work/pairs.csv --images work/images.csv \
        --face-embs work/face_embs.npy --desc-embs work/clip_embs.npy"

python -m sdm_baselines.failure_directions $COMMON --out runs/svm       --target margin
python -m sdm_baselines.b2t               $COMMON --out runs/b2t       --target margin
python -m sdm_baselines.spotlight         $COMMON --out runs/spotlight --target margin

python -m sdm_baselines.evaluate \
    --runs runs/svm runs/b2t runs/spotlight runs/domino runs/ours \
    --pairs work/pairs.csv --images work/images.csv \
    --attributes work/attributes.csv --out runs/eval \
    --gt-groups eyeglasses wearing_hat ... --min-err-lift 1.0
```

Every method writes the same two files, so DOMINO and your own method join the
comparison by emitting `slices.csv` + `members.csv` in this format:

```
slices.csv    method, slice_id, pair_type, unit_type, rank, k, size,
              err_rate, base_err_rate, lift, mean_margin, description, ...
members.csv   slice_id, unit_type, unit_id, rank_in_slice, weight
```

`unit_id` is a row index into `pairs.csv` when `unit_type == "pair"`, and an
`image_id` when `unit_type == "image"`.

## The knobs that matter for your results

**`--target margin` vs `--target error`.** This is the single most consequential
flag. At an operational threshold (`--threshold fmr:1e-4`) you may have a few dozen
errors, and every one of these methods was designed for error rates in the 5–30%
band. The binary label makes them all look terrible for a reason that has nothing
to do with their ideas. `--target margin` uses the continuous signed margin
(`tau - score` for genuine, `score - tau` for impostor) and is far more stable.
Run both, report both — "we gave the baselines the margin signal as well" is a
sentence that pre-empts a reviewer.

**`--threshold`.** Slices at `fmr:1e-3` are not slices at `fmr:1e-6`. Report at
least two operating points; the threshold-dependence of the discovered slices is
itself a finding the classification-era literature has no analogue for.

**`--granularity`.** `pair` is the honest framing for verification; `image` runs
the methods in their published per-image form, with each pair error attributed to
both faces. Report image-level too — it is the fairer baseline, and if your method
only wins at pair level, that is the interesting claim, not a weakness.

**`--pair-type`.** FNMR and FMR slices are fitted separately throughout. Never
pool them: a false-non-match direction and a false-match direction point in
unrelated places in attribute space, and pooling produces slices that mean nothing.

**`--rep`.** How two faces become one vector. All options are symmetric in (a, b),
since verification pairs are unordered. `mean_absdiff` (default) is expressive and
keeps a CLIP-aligned leading block so directions can still be named. `mean` alone
is the only fully text-alignable option. Ablating this is worth a table row: it is
a genuine degree of freedom that the original papers never had to face.

**Description space.** `--desc-embs` is CLIP by default. Pointing it at your FR
embeddings instead is a supported ablation and will probably be weak — ArcFace-style
embeddings are trained to suppress exactly the non-identity attributes you are
trying to recover. That negative result is worth a paragraph. The more important
ablation runs the other way: vanilla CLIP is mediocre on fine-grained facial
attributes, so if you can, re-run with FaRL or another face-tuned VLM. Reviewers
will otherwise read your margins as an artefact of a bad embedding space.

## Fidelity notes — state these accurately in the paper

**Failure Directions** is faithful, with two documented changes. Multiple slices
come from deflating the feature space along each found direction and refitting; the
paper produces one direction per class. Decision values are out-of-fold with folds
assigned *by identity*, because random splits leak badly on faces — the same
identity appears in many pairs and an in-sample SVM will report an identity lookup
as a failure direction. The printed `coherence` (held-out AUC, or Spearman rho for
the margin target) is the check: near 0.5 (or 0) means there is no linear failure
direction, which is a result, not a bug.

**B2T** keeps the CLIP-score-gap scoring verbatim and replaces the *keyword
proposal* step with a fixed face-attribute vocabulary (MAAD-Face's 47 attributes
plus acquisition covariates, see `vocab.py`). Captioners emit near-identical strings
across a face dataset, so the proposal step would set B2T's ceiling rather than its
scoring rule. This makes the baseline **stronger**, not weaker — say so. If you
want the faithful pipeline too, run `alinlab/b2t`'s caption + YAKE stage, dump the
keywords to a file, and pass `--terms-file`; this script then scores exactly those.
Added on top of the paper: Welch t-tests, bootstrap CIs, and BH-corrected q-values,
because a bare top-5 keyword list on face data contains a lot of noise.

**Spotlight** keeps the objective and the size constraint, but solves the
constraint exactly by bisecting on the bandwidth at every step instead of the
paper's Lagrangian barrier — same problem, more stable, one fewer hyperparameter.
Gradients are analytic, so there is no autodiff dependency. It is the cleanest
control in the set: no vision-language machinery at all, so it answers "would a
plain density-weighted loss search in the same embedding space have found this?"
If your method only narrowly beats Spotlight, that is the number a reviewer will
fixate on.

## Evaluation caveats

`--gt-groups` restricts ground truth to the attribute columns you nominate, which
is the published protocol (Precision@k against a *known* bias-conflicting group).
Leaving it unset credits a method for any coherent group it isolates, error-related
or not — the self-test demonstrates this, where a method scores 0.98 precision on a
perfectly coherent "blurry photo" slice whose error lift is 1.05.

`precision_at_k` on its own is close to meaningless without prevalence. If 47% of
your RFW pairs are African, a 47%-African slice scores 0.47 and has discovered
nothing. `precision_lift` and the exact hypergeometric p-value are reported
alongside every number; use them.

`--min-err-lift 1.0` drops slices that are coherent but not error-enriched. Worth
reporting with and without: several of these methods find real visual structure
that has nothing to do with failures.

## Dependencies

`numpy`, `pandas`, `scipy`, `scikit-learn`. CLIP text encoding additionally needs
`open_clip_torch` (or the OpenAI `clip` package) — or skip it entirely by
precomputing text embeddings and passing `--text-embs` / `--terms-file`, which is
what the self-test does.
