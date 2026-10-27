#!/usr/bin/env bash
# Full pipeline on real RFW.  Usage: bash run_all.sh /path/to/RFW [work_dir]
set -euo pipefail
cd "$(dirname "$0")"

RFW="${1:?usage: bash run_all.sh /path/to/RFW [work_dir]}"
WORK="${2:-work}"

python3 scripts/01_build_manifest.py --rfw-root "$RFW" --work "$WORK"
python3 scripts/02_embed_faces.py    --work "$WORK" --align detect
python3 scripts/03_verification.py   --work "$WORK"
python3 scripts/04_embed_clip.py     --work "$WORK"
python3 scripts/05_run_domino.py     --work "$WORK"
python3 scripts/06_report.py         --work "$WORK"

echo "Done - open $WORK/report/report.md"
