# rfw-domino — slice discovery on Racial Faces in the Wild with Domino

Discover coherent, underperforming slices of a face-verification model on the
[RFW](http://www.whdeng.cn/RFW/testing.html) benchmark using
[Domino](https://github.com/HazyResearch/domino) (Eyuboglu et al., ICLR 2022),
the error-aware mixture model over cross-modal (CLIP) embeddings. The pipeline
evaluates a pretrained ArcFace model (InsightFace `buffalo_l`) on RFW's 24,000
verification pairs, then asks Domino which regions of CLIP space concentrate
the model's errors — without ever showing it the race annotations. The report
then checks, post hoc, how discovered slices align with race and with CLIP
text descriptions (gender, age, accessories, pose, image quality).

## How Domino is adapted to verification

Domino was designed for classification examples; face verification is
pair-based, so:

- the **unit of analysis is the pair**, not the image;
- **embedding**: each pair is embedded in CLIP space by pooling its two image
  embeddings (`--pair-embed`): `mean` (default, symmetric, stays near the
  image manifold so slices remain text-describable), `absdiff`, `meandiff`,
  or `concat`;
- **target** `y`: same-person (1) vs. impostor (0);
- **prediction** `ŷ`: the model's decision at a **single global cosine
  threshold** chosen on the pooled pairs of all four races (accuracy-maximizing
  by default, `--criterion eer` alternative). A single operating point mirrors
  deployment; per-race threshold mismatch then surfaces as concentrated
  errors — exactly what Domino is built to find. `pred_probs` is the
  Platt-calibrated `P(same | similarity)`;
- the standard per-race 10-fold accuracies are also reported so you can
  sanity-check the ArcFace pipeline against published RFW numbers.

## The Domino engine: vendored vs. official

`rfw_domino/domino_vendored.py` is a faithful numpy-only port of
`domino.DominoSlicer` (same generative model, E/M updates, PCA-128
preprocessing, soft-prediction handling, and slice-ranking rule
`argsort(-|max_c(p̂_s - p_s)|)`), used **by default**. Reasons: the official
package has no PyPI release and drags in the unmaintained `meerkat` stack, and
its current `main` branch dropped the paper's "confusion" initialization,
which is implemented here (components assigned round-robin to confusion-matrix
cells, ε ~ U(0, `confusion_noise`]). Differences: only `covariance_type="diag"`
(Domino's default/recommendation) is ported, and PCA/k-means are deterministic
numpy re-implementations, so results are not bitwise-identical to the
sklearn-backed original — the math is the same. Original code is Apache-2.0;
attribution in the file header.

To cross-check with the official implementation:

```bash
pip install "domino[clip,text] @ git+https://github.com/HazyResearch/domino@main"
python scripts/05_run_domino.py --impl official ...
```

## Getting RFW

RFW is license-restricted (non-commercial research; no redistribution).
Request access per the instructions at
<http://www.whdeng.cn/RFW/testing.html> (email from an institutional account).
You need **Test Data_v1** (loosely cropped faces) and ideally
**Test_lmk_v1** (5-point landmarks). Any of the usual layouts works — the
manifest builder searches for `<Race>_pairs.txt` and the per-race image
folders:

```
<rfw_root>/test/data/<Race>/<identity>/<identity>_0001.jpg
<rfw_root>/test/txts/<Race>/<Race>_pairs.txt        (and <Race>_lmk.txt)
```

## Install

Python ≥ 3.9. Recommended: fresh venv.

```bash
pip install -r requirements.txt
```

CPU is fine (ArcFace ONNX + CLIP ViT-B/32 over ~40k images ≈ 1–2 h; GPU:
`pip install onnxruntime-gpu`, use `--ctx-id 0` and `--device cuda`).

## Smoke test (no RFW, no model downloads)

```bash
bash tests/run_smoke.sh
```

Generates a synthetic RFW-shaped dataset with a *planted* failure mode (one
race gets a weak identity signal → concentrated false matches at a global
threshold), runs the full pipeline with pixel-based fake embedders, and
asserts that Domino recovers the planted slice. Should end with
`SMOKE TEST PASSED`.

## Run on real RFW

```bash
W=work   # cached intermediates + report end up here
python scripts/01_build_manifest.py --rfw-root /path/to/RFW --work $W
python scripts/02_embed_faces.py    --work $W --align landmarks --rfw-root /path/to/RFW
python scripts/03_verification.py   --work $W
python scripts/04_embed_clip.py     --work $W
python scripts/05_run_domino.py     --work $W
python scripts/06_report.py         --work $W
```

(or `bash run_all.sh /path/to/RFW work`).

### Using your own precomputed embeddings

If you already extracted face-recognition embeddings (any model), skip stage
02 and import them instead — expected format: an `(N, D)` embedding array
plus a parallel array of image paths (and optionally identities), as produced
by a typical glob + dataloader extraction script:

```bash
python scripts/01_build_manifest.py     --rfw-root $RFW --work work
python scripts/02b_import_embeddings.py --work work \
    --embs $EMB/embd_arcface.npy --names $EMB/x_names.npy --ids $EMB/x_ids.npy
python scripts/03_verification.py       --work work
# then stages 04-06 as usual
```

The importer matches embeddings to the manifest by filename (order doesn't
matter), L2-normalizes raw backbone outputs (`--no-normalize` to skip),
ignores embeddings of images not referenced by any pair, and marks manifest
images without an embedding (e.g. face detection failed during your
extraction) as missing — stage 03 then drops the affected pairs and reports
the count. `--ids` enables a sanity check that your identity labels agree
with the manifest.

Stage 02 alignment modes: `detect`
(RetinaFace detection, default, no landmark files needed), `landmarks` (RFW's
provided landmarks; faster and deterministic), `prealigned` (if you already
have 112×112 crops). Landmark files with unknown coordinate ordering are
auto-disambiguated geometrically.

Key stage-05 knobs (defaults follow the Domino paper/blog demos):
`--n-components 50 --n-slices 10 --y-weight 40 --yhat-weight 40 --pca 128
--init confusion --seed 42`. Larger `γ` weights push slices to be pure in
error type; more components give finer slices. Slice discovery is exploratory:
vary `--seed`, weights, and `--pair-embed` and look for slices that recur.

## Outputs

`<work>/report/report.md` — global and per-race verification metrics, a
ranked slice table (size, error rate and lift, false-match share, race
composition with lift, mean similarity), per-slice CLIP text descriptions,
and a montage PNG of each slice's top pairs (red border = error).
`<work>/slices_assign.csv` has per-pair slice probabilities for downstream
analysis; `<work>/metrics.json` and `<work>/slicer_info.json` record all
parameters (mixture weights, per-component y/ŷ distributions).

## Caveats

- The report's race composition uses RFW's subset labels (which are
  dataset-level annotations, not self-identified ethnicity); the phrase bank
  deliberately excludes race/ethnicity terms since CLIP race attribution is
  unreliable — metadata answers that question exactly.
- Platt calibration is fit on the same pairs it scores (standard for post-hoc
  analysis; split folds if you need honest probabilities).
- Slices are mixture components: expect some incoherent ones, and treat
  descriptions as hypotheses to verify in the montages (the Domino paper
  reports >60% of injected slices are still missed by the best methods).
- RFW's license: research use only, do not redistribute images (including in
  reports you circulate).

## Citations

Domino: Eyuboglu et al., *Domino: Discovering Systematic Errors with
Cross-Modal Embeddings*, ICLR 2022. RFW: Wang et al., *Racial Faces in the
Wild: Reducing Racial Bias by Information Maximization Adaptation Network*,
ICCV 2019. ArcFace: Deng et al., CVPR 2019 (InsightFace implementation).
CLIP: Radford et al., ICML 2021.
