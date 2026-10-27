"""Stage 02 - face-recognition embeddings for every image in the manifest.

Backends
--------
insightface (default)
    Pretrained ArcFace from the `buffalo_l` model zoo (ResNet50, ONNX).
    Three alignment modes:
      * detect      - RetinaFace/SCRFD detection + 5-point alignment (default;
                      RFW images are only loosely cropped).
      * landmarks   - use RFW's provided `<Race>_lmk.txt` files (Test_lmk_v1)
                      to align without running a detector (faster, deterministic).
      * prealigned  - images are already 112x112 aligned crops; just resize.
fake
    Deterministic pixel-based pseudo-embeddings; used by the synthetic smoke
    test so the pipeline can run without model downloads.
"""

import argparse
import os

import numpy as np
import pandas as pd

from .align import load_lmk_file, norm_crop
from .io_utils import Work, l2_normalize, tqdm


# --------------------------------------------------------------------------

def fake_embed(paths) -> np.ndarray:
    """12x12 grayscale pixels as a 144-d 'identity' embedding (smoke tests)."""
    import cv2

    embs = np.zeros((len(paths), 144), dtype=np.float32)
    for i, p in enumerate(tqdm(paths, desc="fake face embs")):
        img = cv2.imread(p)
        if img is None:
            raise FileNotFoundError(p)
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        g = cv2.resize(g, (12, 12), interpolation=cv2.INTER_AREA)
        embs[i] = g.astype(np.float32).ravel()
    embs -= embs.mean(axis=0, keepdims=True)
    return l2_normalize(embs)


# --------------------------------------------------------------------------

def _load_insightface(model_name: str, ctx_id: int, det_size: int):
    from insightface.app import FaceAnalysis

    app = FaceAnalysis(name=model_name)
    app.prepare(ctx_id=ctx_id, det_size=(det_size, det_size))
    rec = app.models.get("recognition")
    if rec is None:
        raise RuntimeError("No recognition model in the insightface model zoo pack.")
    return app, rec


def _rec_feat(rec, aligned_batch, flip: bool) -> np.ndarray:
    """Run the ArcFace ONNX model on a list of aligned 112x112 BGR crops."""
    import cv2

    feats = rec.get_feat(aligned_batch)
    feats = np.asarray(feats, dtype=np.float32).reshape(len(aligned_batch), -1)
    if flip:
        flipped = [cv2.flip(a, 1) for a in aligned_batch]
        f2 = rec.get_feat(flipped)
        feats = feats + np.asarray(f2, dtype=np.float32).reshape(len(aligned_batch), -1)
    return feats


def insightface_embed(images: pd.DataFrame, model_name: str, ctx_id: int,
                      det_size: int, align_mode: str, rfw_root: str,
                      flip: bool, batch_size: int):
    import cv2

    app, rec = _load_insightface(model_name, ctx_id, det_size)
    n = len(images)
    dim = None
    embs, methods = [None] * n, ["?"] * n

    if align_mode == "landmarks":
        from .manifest import find_txt

        lmk_by_race = {}
        for race in images["race"].unique():
            lmk_by_race[race] = load_lmk_file(find_txt(rfw_root, race, "lmk"))
            print(f"[face] loaded {len(lmk_by_race[race])} landmarks for {race}")

    def read(path):
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(path)
        return img

    if align_mode in ("landmarks", "prealigned"):
        # batchable: alignment does not need the detector
        for start in tqdm(range(0, n, batch_size), desc="face embs (batched)"):
            rows = images.iloc[start:start + batch_size]
            crops, kinds = [], []
            for _, r in rows.iterrows():
                img = read(r["path"])
                if align_mode == "prealigned":
                    crops.append(cv2.resize(img, (112, 112)))
                    kinds.append("prealigned")
                else:
                    lmk = lmk_by_race[r["race"]].get(os.path.basename(r["path"]))
                    if lmk is not None:
                        crops.append(norm_crop(img, lmk))
                        kinds.append("landmarks")
                    else:
                        crops.append(cv2.resize(img, (112, 112)))
                        kinds.append("fallback_resize")
            feats = _rec_feat(rec, crops, flip)
            for j, (f, k) in enumerate(zip(feats, kinds)):
                embs[start + j], methods[start + j] = f, k
            dim = feats.shape[1]
    else:  # detect
        for i in tqdm(range(n), desc="face embs (detect+align)"):
            r = images.iloc[i]
            img = read(r["path"])
            faces = app.get(img)
            if faces:
                face = max(
                    faces,
                    key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
                )
                if flip:
                    crop = norm_crop(img, face.kps)
                    f = _rec_feat(rec, [crop], flip=True)[0]
                else:
                    f = face.embedding.astype(np.float32)
                embs[i], methods[i] = f, "detect"
            else:
                f = _rec_feat(rec, [cv2.resize(img, (112, 112))], flip=flip)[0]
                embs[i], methods[i] = f, "fallback_resize"
            dim = embs[i].shape[0]

    out = np.zeros((n, dim), dtype=np.float32)
    for i, e in enumerate(embs):
        out[i] = e
    n_fb = sum(m == "fallback_resize" for m in methods)
    if n_fb:
        print(f"[face] WARNING: {n_fb}/{n} images fell back to plain resize "
              f"(no detection/landmarks). They are flagged in face_meta.csv.")
    return l2_normalize(out), methods


# --------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--work", default="work")
    ap.add_argument("--backend", choices=["insightface", "fake"],
                    default="insightface")
    ap.add_argument("--model", default="buffalo_l",
                    help="insightface model pack (default buffalo_l = ArcFace R50)")
    ap.add_argument("--align", choices=["detect", "landmarks", "prealigned"],
                    default="detect")
    ap.add_argument("--rfw-root", default=None,
                    help="needed for --align landmarks (to find *_lmk.txt)")
    ap.add_argument("--ctx-id", type=int, default=-1,
                    help="-1 = CPU, >=0 = GPU id")
    ap.add_argument("--det-size", type=int, default=640)
    ap.add_argument("--flip", action="store_true",
                    help="add horizontally-flipped embedding (sum) before norm")
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args(argv)

    work = Work(args.work)
    images = work.load_images()

    if args.backend == "fake":
        embs = fake_embed(images["path"].tolist())
        methods = ["fake"] * len(images)
    else:
        if args.align == "landmarks" and not args.rfw_root:
            ap.error("--align landmarks requires --rfw-root")
        embs, methods = insightface_embed(
            images, args.model, args.ctx_id, args.det_size,
            args.align, args.rfw_root, args.flip, args.batch,
        )

    np.save(work.face_embs_npy, embs)
    pd.DataFrame({"image_id": images["image_id"], "method": methods}).to_csv(
        work.face_meta_csv, index=False
    )
    print(f"[face] wrote {embs.shape} -> {work.face_embs_npy}")


if __name__ == "__main__":
    main()
