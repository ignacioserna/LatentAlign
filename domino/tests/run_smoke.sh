#!/usr/bin/env bash
# End-to-end smoke test on synthetic data with fake (pixel) embedders.
# Needs only: numpy, pandas, Pillow, opencv. No model downloads.
set -euo pipefail
cd "$(dirname "$0")/.."

RFW=smoke_rfw
WORK=smoke_work
rm -rf "$RFW" "$WORK"

python3 scripts/00_make_synthetic_rfw.py --out "$RFW"
python3 scripts/01_build_manifest.py --rfw-root "$RFW" --work "$WORK"
python3 scripts/02_embed_faces.py --work "$WORK" --backend fake
python3 scripts/03_verification.py --work "$WORK"
python3 scripts/04_embed_clip.py --work "$WORK" --backend fake
python3 scripts/05_run_domino.py --work "$WORK" --n-slices 8 --n-components 24
python3 scripts/06_report.py --work "$WORK"
python3 tests/check_smoke.py "$WORK" African
