# Comparison with slice discovery: Domino (Sec. 4.5, Supp. G)

`rfw_domino/` adapts [Domino](https://github.com/HazyResearch/domino)
(Eyuboglu et al., ICLR 2022) to pair-based face verification on RFW. Domino is
vendored as a numpy port (`rfw_domino/domino_vendored.py`, Apache-2.0); see
`README_pipeline.md` for the full description of every stage.
`sdm_baselines/` contains three further slice-discovery baselines adapted to
verification (Spotlight, Bias-to-Text, failure directions) that are not
reported in the paper.

## Reproducing Tab. 3, Supp. Tabs. 7-9 and Supp. Fig. 13

Run from this folder. `$D` is the data root (`$LATENTALIGN_DATA`).

```bash
pip install -r requirements.txt
W=work

# 1. manifest of the RFW protocol pairs (needs the RFW txts/ folder)
python scripts/01_build_manifest.py --rfw-root $D/RFW --work $W
# 2. reuse the ArcFace embeddings of the paper (matched by file name)
python scripts/02b_import_embeddings.py --work $W \
    --embs $D/RFW/embeddings/embd_arcface.npy --names $D/RFW/embeddings/x_names.npy --ids $D/RFW/embeddings/x_ids.npy
# 3. expand the impostor set: +125,000 within-race pairs per race (511,864 usable impostor pairs)
python -m rfw_domino.expand_negatives --work $W --n-per-race 125000
# 4. global threshold at FMR = 1e-3 on the expanded set, Platt calibration
python scripts/03_verification.py --work $W --criterion far --far-target 1e-3
# 5. CLIP ViT-B/32 image embeddings + phrase bank
python scripts/04_embed_clip.py --work $W
# 6. Domino on the impostor pairs, mean-pooled CLIP pair embedding
python scripts/05_run_domino.py --work $W --subset neg --pair-embed mean \
    --n-slices 10 --n-components 50 --y-weight 40 --yhat-weight 5 --pca 128 --init confusion --seed 42 --max-iter 500   # Supp. Tab. 7, Fig. 13
python scripts/06_report.py --work $W
python scripts/05_run_domino.py --work $W --subset neg --pair-embed mean \
    --n-slices 50 --n-components 200 --y-weight 40 --yhat-weight 5 --pca 128 --init confusion --seed 42 --max-iter 500   # Supp. Tab. 8, Tab. 3
# 7. project the slices onto 100-image groups (Supp. G.5, Supp. Tab. 9)
python -m rfw_domino.slice_to_groups --work $W --group-size 100
# 8. LatentAlign groups of the sweep (experiments/run_sweeps.py) on the same footing
python -m rfw_domino.lfa_groups --work $W --db ../results/db/RFW_arcface_latentalign.db --model arcface \
    --names $D/RFW/embeddings/x_names.npy --size 100 --bin 1      # the ArcFace DB: the groups must come from the scored backbone
python -m rfw_domino.compare_fmr --work $W --group-size 100 --groups latentalign=$W/lfa_groups.csv --top 10,100
```

`compare_fmr` scores every method with the same protocol (FMR over all
within-group impostor pairs at the single global threshold stored in
`$W/metrics.json`).

## Both operating points in one go

`scripts/run_both_thresholds.sh` runs steps 4-8 twice, in `WORK_expanded`
(global FMR = 1e-3 over the 511,864 pairs, 0.3227 on ArcFace) and `WORK_allpairs`
(`03_verification.py --criterion fixed --threshold 0.2515`, the all-pairs threshold
of Figs. 4-5 and Tab. 2; `python -m experiments.global_thresholds` computes it),
and prints the Tab. 3 / Supp. Tab. 9 numbers of both. The inputs of stages 01-04
are symlinked, not copied. Domino's slices do not depend on the threshold (with
`--subset neg` it is fitted on CLIP pair embeddings and the threshold-free Platt
P(same)); the script checks that both runs give identical slices.

```bash
bash scripts/run_both_thresholds.sh work ../results/db/RFW_arcface_latentalign.db $D/RFW/embeddings/x_names.npy
```

## Notes

* **Operating point.** The global threshold of step 4 is computed on the
  expanded pair set (0.3227 for ArcFace), whereas
  `experiments/table3_latentalign_fmr.py` uses the threshold over *all*
  impostor pairs of the 40,602 RFW images (0.2515). Compare Domino and
  LatentAlign at one threshold: `compare_fmr` (step 8) does so.
* **Component allocation.** `domino_vendored.py` assigns the mixture
  components round-robin to the *occupied* confusion cells.
* **Convergence.** EM stops when the lower bound changes by less than
  `tol = 1e-5` or after `--max-iter` iterations (default 100; the paper used 500).
* **LatentAlign groups.** `lfa_groups.py` must read the DB of the backbone whose
  embeddings are scored (ArcFace); groups from another backbone's DB give
  meaningless FMRs.
* `tests/run_smoke.sh` runs the whole pipeline on a synthetic RFW-shaped
  dataset with a planted failure mode.
