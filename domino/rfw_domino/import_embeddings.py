"""Stage 02b - import precomputed face-recognition embeddings.

Replaces stage 02 when you already extracted embeddings yourself, e.g. with an
extraction script that saves:

    embd_*.npy   (N, D) float embeddings, in extraction (glob/dataloader) order
    x_names.npy  (N,)   image file paths, same order
    x_ids.npy    (N,)   identity of each image (optional, used as a sanity check)

Handled quirks:
  * ordering: embeddings are matched to the manifest by image filename
    (basename), with an identity-qualified key as fallback if basenames are
    not unique; extra embeddings not referenced by any pair are ignored;
  * images your extraction skipped (e.g. no face detected -> dataset returned
    None) are recorded as missing; stage 03 then drops the affected pairs and
    reports how many;
  * embeddings are L2-normalized on import (raw backbone outputs are fine);
    pass --no-normalize if yours are already unit-length.

Usage:
    python scripts/02b_import_embeddings.py --work work \
        --embs RFW/embd_partialFC.npy --names RFW/x_names.npy --ids RFW/x_ids.npy
"""

import argparse
import os

import numpy as np
import pandas as pd

from .io_utils import Work, l2_normalize


def _basename(p: str) -> str:
    return os.path.basename(str(p).replace("\\", "/"))


def _id_key(p: str) -> str:
    parts = str(p).replace("\\", "/").rstrip("/").split("/")
    return "/".join(parts[-2:])  # identity/filename


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--work", default="work")
    ap.add_argument("--embs", required=True, help="e.g. RFW/embd_partialFC.npy")
    ap.add_argument("--names", required=True, help="e.g. RFW/x_names.npy")
    ap.add_argument("--ids", default=None, help="e.g. RFW/x_ids.npy (optional)")
    ap.add_argument("--no-normalize", action="store_true",
                    help="skip L2 normalization of the imported embeddings")
    ap.add_argument("--min-match", type=float, default=0.9,
                    help="abort if fewer than this fraction of manifest "
                         "images find an embedding")
    args = ap.parse_args(argv)

    work = Work(args.work)
    images = work.load_images()

    embs = np.asarray(np.load(args.embs), dtype=np.float32)
    names = np.load(args.names, allow_pickle=True)
    names = [str(n) for n in np.asarray(names).ravel()]
    if embs.ndim != 2 or len(embs) != len(names):
        raise SystemExit(
            f"shape mismatch: embeddings {embs.shape} vs {len(names)} names"
        )
    print(f"[import] {len(embs)} embeddings of dim {embs.shape[1]} "
          f"from {args.embs}")

    # index the user's embeddings by filename; fall back to identity/filename
    by_base, base_dupes = {}, set()
    by_idkey = {}
    for i, n in enumerate(names):
        b = _basename(n)
        if b in by_base:
            base_dupes.add(b)
        by_base[b] = i
        by_idkey[_id_key(n)] = i
    if base_dupes:
        print(f"[import] note: {len(base_dupes)} duplicate basenames in the "
              f"embedding files - using identity-qualified matching for those")

    n_img = len(images)
    dim = embs.shape[1]
    out = np.full((n_img, dim), np.nan, dtype=np.float32)
    methods = []
    n_missing = 0
    for row in images.itertuples():
        b = _basename(row.path)
        idx = None
        if b in base_dupes or b not in by_base:
            idx = by_idkey.get(_id_key(row.path))
        else:
            idx = by_base[b]
        if idx is None:
            methods.append("missing")
            n_missing += 1
        else:
            out[row.image_id] = embs[idx]
            methods.append("precomputed")

    matched = n_img - n_missing
    print(f"[import] matched {matched}/{n_img} manifest images "
          f"({n_missing} missing - typically images where face detection "
          f"failed during your extraction)")
    if matched / max(1, n_img) < args.min_match:
        raise SystemExit(
            f"only {matched / n_img:.1%} of manifest images matched an "
            f"embedding; check that --names paths correspond to the same "
            f"RFW copy used for the manifest."
        )

    if not args.no_normalize:
        norms = np.linalg.norm(out, axis=1)
        finite = np.isfinite(norms)
        if finite.any():
            print(f"[import] L2-normalizing (raw norms: "
                  f"median {np.median(norms[finite]):.3f}, "
                  f"min {norms[finite].min():.3f}, "
                  f"max {norms[finite].max():.3f})")
        out[finite] = l2_normalize(out[finite])

    if args.ids:
        ids = [str(x) for x in np.asarray(
            np.load(args.ids, allow_pickle=True)).ravel()]
        mismatches = 0
        for row in images.itertuples():
            b = _basename(row.path)
            idx = by_idkey.get(_id_key(row.path)) if (
                b in base_dupes or b not in by_base) else by_base[b]
            if idx is not None and ids[idx] != str(row.identity):
                mismatches += 1
        if mismatches:
            print(f"[import] WARNING: {mismatches} identity mismatches "
                  f"between x_ids and the manifest - verify path parsing.")
        else:
            print("[import] identity sanity check passed (x_ids agree with "
                  "the manifest)")

    np.save(work.face_embs_npy, out)
    pd.DataFrame({"image_id": images["image_id"], "method": methods}).to_csv(
        work.face_meta_csv, index=False
    )
    print(f"[import] wrote {out.shape} -> {work.face_embs_npy}")


if __name__ == "__main__":
    main()
