#!/usr/bin/env bash
# Re-score an already fitted Domino run at another threshold, without fitting
# Domino again (its slices do not depend on the threshold).
#
#   bash scripts/rescore_threshold.sh FITTED_WORK DEST_WORK THRESHOLD LATENTALIGN_DB NAMES
#
# FITTED_WORK is a folder written by run_both_thresholds.sh (it has run_n10/ and
# run_n50/ with slice_probs.npy, slices_assign.csv, slicer_info.json).
set -euo pipefail
FIT=$(realpath "$1"); W=$(realpath -m "$2"); T=$3; DB=$(realpath "$4"); NAMES=$(realpath "$5")
INPUTS="images.csv pairs.csv pairs_official.csv face_embs.npy face_meta.csv clip_embs.npy
        clip_meta.json text_embs.npy phrases.txt phrase_bank_meta.json"
mkdir -p "$W"
for f in $INPUTS; do [ -e "$FIT/$f" ] && ln -sf "$(realpath "$FIT/$f")" "$W/$f"; done

python scripts/03_verification.py --work "$W" --criterion fixed --threshold "$T"

for RUN in n10 n50; do
  cp "$FIT/run_$RUN/slicer_info.json" "$FIT/run_$RUN/slices_assign.csv" "$FIT/run_$RUN/slice_probs.npy" "$W/"
  rm -rf "$W/report"
  python scripts/06_report.py --work "$W"
  if [ "$RUN" = n50 ]; then
    python -m rfw_domino.slice_to_groups --work "$W" --group-size 100
    python -m rfw_domino.lfa_groups --work "$W" --db "$DB" --model arcface --names "$NAMES" \
        --size 100 --bin 1 --out "$W/lfa_groups.csv"
    python -m rfw_domino.compare_fmr --work "$W" --group-size 100 \
        --groups latentalign="$W/lfa_groups.csv" --top 10,100
  fi
  rm -rf "$W/report_$RUN" "$W/run_$RUN"; mv "$W/report" "$W/report_$RUN"; mkdir -p "$W/run_$RUN"
  mv "$W/slicer_info.json" "$W/slices_assign.csv" "$W/slice_probs.npy" "$W/run_$RUN/"
done
python scripts/summarize_tab3.py "$FIT" "$W"
