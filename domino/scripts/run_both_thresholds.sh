#!/usr/bin/env bash
# Domino and LatentAlign (Tab. 3, Supp. Tabs. 7-9, Supp. Fig. 13) at the two
# operating points:
#   expanded  - global FMR = 1e-3 over the 511,864 Domino impostor pairs (0.3227 on ArcFace)
#   allpairs  - global FMR = 1e-3 over all 824M impostor pairs of RFW (0.2515 on ArcFace),
#               the threshold of Figs. 4-5 and Tab. 2
#
# Domino's slices do not depend on the threshold (with --subset neg it is fitted on
# the CLIP pair embeddings and the Platt-calibrated P(same), which is threshold-free),
# so it is fitted once and the slices are re-scored at the second threshold; only
# the error counts, the reports and the group FMRs change.
#
# usage (from the domino/ folder):
#   bash scripts/run_both_thresholds.sh WORK LATENTALIGN_DB NAMES [T_ALLPAIRS]
#     WORK            work folder with stages 01-04 done (pairs.csv with the expanded
#                     negatives, images.csv, face_embs.npy, clip_embs.npy, text_embs.npy, ...)
#     LATENTALIGN_DB  SQLite DB of the ArcFace LatentAlign sweep (RFW_arcface_latentalign.db)
#     NAMES           x_names.npy the DB indices refer to
#     T_ALLPAIRS      all-pairs threshold of ArcFace (default 0.2514827251434326;
#                     python -m experiments.global_thresholds --models arcface prints it)
set -euo pipefail
SRC=$(realpath "$1"); DB=$(realpath "$2"); NAMES=$(realpath "$3"); T_ALL=${4:-0.2514827251434326}

INPUTS="images.csv pairs.csv pairs_official.csv face_embs.npy face_meta.csv clip_embs.npy
        clip_meta.json text_embs.npy phrases.txt phrase_bank_meta.json"

# Domino is fitted once, at the expanded operating point; the all-pairs folder
# re-scores the same slices (scripts/rescore_threshold.sh).
W="${SRC}_expanded"
echo "=================== expanded -> $W"
mkdir -p "$W"
for f in $INPUTS; do [ -e "$SRC/$f" ] && ln -sf "$(realpath "$SRC/$f")" "$W/$f"; done

# 1. operating point (also writes pairs_scored.csv with the Platt P(same))
python scripts/03_verification.py --work "$W" --criterion far --far-target 1e-3

# 2. 10 slices (Supp. Tab. 7, Supp. Fig. 13)
python scripts/05_run_domino.py --work "$W" --subset neg --pair-embed mean \
    --n-slices 10 --n-components 50 --y-weight 40 --yhat-weight 5 --pca 128 \
    --init confusion --max-iter 500 --seed 42
python scripts/06_report.py --work "$W"
rm -rf "$W/report_n10" "$W/run_n10"; mv "$W/report" "$W/report_n10"; mkdir -p "$W/run_n10"
cp "$W"/slicer_info.json "$W"/slices_assign.csv "$W"/slice_probs.npy "$W/run_n10/"

# 3. 50 slices (Supp. Tab. 8; Tab. 3 and Supp. Tab. 9 after the projection). Slow:
#    200 components over 511,864 x 128 for up to 500 EM iterations.
python scripts/05_run_domino.py --work "$W" --subset neg --pair-embed mean \
    --n-slices 50 --n-components 200 --y-weight 40 --yhat-weight 5 --pca 128 \
    --init confusion --max-iter 500 --seed 42
python scripts/06_report.py --work "$W"
python -m rfw_domino.slice_to_groups --work "$W" --group-size 100   # -> report/slice_groups.csv

# 4. LatentAlign: the 2,800 100-image ArcFace groups, scored with the same protocol
python -m rfw_domino.lfa_groups --work "$W" --db "$DB" --model arcface --names "$NAMES" \
    --size 100 --bin 1 --out "$W/lfa_groups.csv"
python -m rfw_domino.compare_fmr --work "$W" --group-size 100 \
    --groups latentalign="$W/lfa_groups.csv" --top 10,100
rm -rf "$W/report_n50" "$W/run_n50"; mv "$W/report" "$W/report_n50"; mkdir -p "$W/run_n50"
cp "$W"/slicer_info.json "$W"/slices_assign.csv "$W"/slice_probs.npy "$W/run_n50/"

# 5. the same slices at the all-pairs threshold
bash scripts/rescore_threshold.sh "$W" "${SRC}_allpairs" "$T_ALL" "$DB" "$NAMES"
