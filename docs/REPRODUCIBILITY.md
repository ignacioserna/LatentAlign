# Reproducibility guide

For every table and figure of the paper: the script that produces it, the
parameters used, and what has been checked. "Verified" means that the code of
this repository reproduces the published numbers (or the stored outputs of the
original runs) exactly, unless a tolerance is given.

## Operating points

All FMRs of the paper are measured at a single **global threshold where the
whole dataset has FMR = 10⁻³**, computed over all impostor pairs of the 40,602
RFW images (824,183,509 pairs):

| | ArcFace | CosFace | ElasticFace | PartialFC |
|---|---|---|---|---|
| Global threshold (all pairs) | 0.2515 | 0.2393 | 0.2494 | 0.1973 |
| Domino expanded pair set (511,864 within-race impostor pairs) | 0.3227 | 0.3046 | 0.3174 | 0.2598 |

`python -m experiments.global_thresholds [--domino-work domino/work]` computes
both. The second one is only used to fit Domino and to report its slice
statistics (Supp. Tabs. 7-8, Supp. Fig. 13); the group-level comparison with
LatentAlign (Tab. 3, Supp. Tab. 9) uses the global threshold.

## Main paper

| Item | Script | Notes |
|---|---|---|
| Fig. 1(a) | `experiments/table2_groups.py` + images of `data/groups/rfw_groups_ABCD.json` | groups A-D |
| Fig. 1(b), Supp. Fig. 2 | `experiments/table2_groups.py` → `score_distributions_{model}.pdf` | group sizes and pair counts verified (8,211 / 10,001 / 4,464 / 4,831 impostor pairs) |
| Fig. 1(c), Supp. Fig. 3 | `experiments/table2_groups.py` → `fmr_curves_{model}.pdf` | |
| Fig. 2(a) | `traversals/arc2face/traverse_arc2face.py --mode removal` | t = -0.5 |
| Fig. 2(b) | `traversals/arc2face/traverse_arc2face.py --mode addition` | t = 0.45 (0.4-0.5 per image, Supp. B) |
| Fig. 3 | `traversals/attribute_classifier/predict_attributes.py` → `traversals/fig3_aggregate.py` → `traversals/plot_fig3.py` | ResNet-18 attribute classifier trained on the 47 MAAD-Face attribute labels with the protocol and code of Lingenfelter & Hand ([celeba-baselines](https://github.com/blingenf/celeba-baselines)); weights in the v1.0.0 release. The per-image outputs used in the paper are in `data/traversals/classifier_outputs`; aggregating them gives the published values exactly |
| Tab. 1 | `experiments/table1.py --dataset {rfw,celeba}` | see "Tab. 1" below |
| Figs. 4-5 | `experiments/run_sweeps.py`, then `experiments/fig4_fig5.py` | semantic distance of every stored group verified; top-100 groups ranked by semantic distance (Fig. 4) and by FMR (Fig. 5) within each size bin |
| Tab. 2 | `experiments/table2_groups.py` | global threshold; see "Tab. 2" below |
| Sec. 4.4, identity overlap | `experiments/identity_overlap.py` | ElasticFace, τ = 0.34, 400 seeds, top-100 groups by mean impostor similarity: 57% of identities in one group, 0.03% in more than ten |
| Tab. 3 | `experiments/table3_latentalign_fmr.py` (LatentAlign), `domino/scripts/run_both_thresholds.sh` (Domino) | see "Tab. 3 and Supp. Tab. 9" below |

### Tab. 1

Semantic distance is the mean pairwise Hamming distance of the attributes within
a group (RFW: 10 VLM attributes, one-hot, L1/2; CelebA: 40 binary attributes).
All methods except DBSCAN start from the same k-means++ seeds: k = 400 on RFW
(shipped in `data/seeds/rfw_kmeanspp_seeds.json`, because scikit-learn's
k-means++ is not reproducible across versions) and k = 2,000 on CelebA.

| Row | RFW | CelebA |
|---|---|---|
| NNS | 100 nearest neighbours of each seed (verified) | 100 nearest neighbours of each seed |
| k-means | MiniBatchKMeans(k = 400, random_state = 42, batch = 16,384), all clusters (verified) | MiniBatchKMeans(k = 2,000) |
| spk-means | spherical k-means, k = 400, groups with ≥ 20 identities and mean impostor similarity > 0.1 (verified) | k = 2,000, all clusters |
| DBSCAN | (eps, min_samples) = (1.09, 4), (1.12, 5), (1.10, 5); clusters with ≥ 20 identities (verified; ≈ 550-660 images) | eps = 1.0 / 1.1 / 1.0 / 1.1, min_samples = 2; clusters with 55-10,000 annotated images |
| weighted-SVD | threshold growth from the seeds, τ = 0.28 | τ = 0.31 |
| LatentAlign | threshold growth from the seeds, τ = 0.34 (ArcFace, CosFace, ElasticFace), 0.28 (PartialFC) | τ = 0.41 |

With the shipped seeds, `table1.py` gives LatentAlign 1.8905 / 1.6151 / 1.8965 /
1.7526 on RFW (paper: 1.89 / 1.63 / 1.89 / 1.73). The weighted-SVD direction is
the first right singular vector of the centred, identity-weighted group; τ was
chosen to give groups of about 100 images (`--wsvd-tau` and `--latentalign-tau`
override it).

Uncertainties in the table: standard errors, except for k-means, spk-means and
DBSCAN, and NNS on RFW, which report 95% interval half-widths (1.96·SE on RFW,
percentile bootstrap on CelebA). `table1.py` prints all of them.

### Tab. 2

Annotation-defined groups are selected with the attribute filters of
`ANNOTATION_FILTERS` in `table2_groups.py`, on the labels of
`data/annotations/rfw/labels_RFW_mode.csv`. Annotation-defined / LatentAlign FMR:

| Backbone | A | B | C | D |
|---|---|---|---|---|
| ArcFace | 0.003 / 0.018 | 0.019 / 0.077 | 0.062 / 0.090 | 0.052 / 0.153 |
| CosFace | 0.005 / 0.023 | 0.026 / 0.127 | 0.071 / 0.103 | 0.053 / 0.145 |
| ElasticFace | 0.006 / 0.022 | 0.020 / 0.068 | 0.088 / 0.130 | 0.081 / 0.233 |
| PartialFC | 0.006 / 0.017 | 0.026 / 0.112 | 0.067 / 0.086 | 0.056 / 0.149 |

Group R (Supp. Tab. 5) is a random draw of 75 identities with a fixed seed.

### Tab. 3 and Supp. Tab. 9

LatentAlign: the 2,800 groups of exactly 100 images of the ArcFace sweep
(k ∈ {200, 400, 500, 700, 1000} seeds). Domino: the 50 slices of the
`n_slices = 50` run, each projected onto the 100 images with the largest
aggregated slice membership. Domino's slices do not depend on the threshold
(with `--subset neg` it is fitted on CLIP pair embeddings and the
threshold-free Platt-calibrated P(same)), so the same slices are scored at
both operating points. `domino/scripts/run_both_thresholds.sh` reproduces the
slices of the paper exactly (Supp. Tab. 7 row by row) and prints:

| Threshold | LatentAlign (median) | Domino top-100 projection (median) | Domino as populations (median) | LatentAlign groups above the best Domino slice |
|---|---|---|---|---|
| 0.2515, global (paper) | 0.0136-0.2396 (0.1278) | 0.0010-0.0594 (0.0215) | 0.0008-0.0127 (0.0054) | 96.9% |
| 0.3227, expanded set | 0.0010-0.0703 (0.0273) | 0.0000-0.0135 (0.0038) | 0.0000-0.0019 (0.0007) | 93.6% |

`compare_fmr` reports 2,795 LatentAlign groups, because 5 groups lose more than
5% of their images when mapped onto the Domino manifest (median 0.1282 / 0.0274).

## Supplementary material

| Item | Script | Notes |
|---|---|---|
| Supp. Tab. 1 | — | OpenCompass leaderboard values |
| Supp. Tab. 2 | `preprocessing/annotation/annotation_stats.py` | counts and agreement verified from the raw VLM outputs (strict rule) |
| Supp. Tab. 3, Supp. Fig. 1 | `preprocessing/annotation/human_validation.py` | match rates and κ verified from the raw human annotations (strict rule) |
| Supp. Figs. 2-3, Tabs. 4-5 | `experiments/table2_groups.py` | Tab. 4 verified; Tab. 5: image-level bootstrap, 1,000 resamples |
| Supp. Figs. 4-5 | `experiments/run_sweeps.py --methods graph_init random_init`, `experiments/supp_fig4_fig5_sensitivity.py` | top-100 summary verified on ArcFace |
| Supp. Figs. 6-12 | `experiments/supp_figs6_12_celeba_groups.py` | ArcFace, τ = 0.25, seeds in `data/seeds/celeba_seeds.json`; the single-image seeds (Figs. 11-12) reach the 1,000-iteration cap |
| Supp. Tab. 6 | `experiments/supp_table6_celeba_kmeans.py` | initial groups in `data/celeba_legacy_init` |
| Supp. Tabs. 7-9, Supp. Fig. 13 | `domino/` (see `domino/README.md`) | |

## RFW annotations

`preprocessing/annotation/majority_vote.py` implements both merging rules and
reproduces the shipped label files exactly:

* `--rule plurality` (+ `resolve_multiface.py`) → `labels_RFW_mode.csv`, the labels
  used by every experiment: for each attribute, the most frequent of the five
  answers, `unknown` counting as a regular answer, ties broken alphabetically.
* `--rule strict` (+ `resolve_multiface.py`) → `labels_RFW.csv`: `unknown` answers
  discarded and a strict majority of the rest required; used only for the
  statistics of Supp. Tabs. 2-3.

## Notes

1. **Preprocessing.** Faces are fed to the backbones as BGR in [0, 1], without
   mean/std normalisation, as in the original extraction.
2. **ElasticFace checkpoints.** The RFW embeddings use ElasticFace-Cos+ and the
   CelebA embeddings ElasticFace-Arc (`extract_embeddings.py` does the same).
3. **Filters in Figs. 4-5.** k-means and spherical k-means groups are kept only
   if they have ≥ 20 identities and mean impostor similarity > 0.1, DBSCAN
   clusters if they have ≥ 20 identities; LatentAlign, NN and weighted-SVD groups
   are not filtered.
4. **Randomness.** Spherical k-means, group R and the random seeds of Supp.
   Fig. 5 are seeded here; the original runs were not, so these numbers can
   differ slightly from the paper.
5. **Traversal directions.** Each group's direction is recomputed (Eq. 1) from
   its images re-embedded with the decoder's own encoder (antelopev2 ArcFace for
   Arc2Face, the InfiniteYou face pipeline for InfiniteYou).
6. **Domino.** Vendored numpy port (`domino/rfw_domino/domino_vendored.py`);
   mixture components are assigned round-robin to the occupied cells of the
   confusion matrix; the paper's runs use `--y-weight 40 --yhat-weight 5
   --max-iter 500 --seed 42`.

## Runtime (single A100, 512 GB RAM)

Embedding extraction: a few hours. RFW sweeps: minutes to about an hour per
backbone for LatentAlign, NN and weighted-SVD; about an hour for DBSCAN and
spherical k-means. CelebA LatentAlign: hours (`grow_threshold_batched` is much
faster than a sequential loop). Global threshold on RFW: 1-2 minutes per
backbone. Domino (CPU): the 10-slice fit takes minutes, the 50-slice fit several
hours.
