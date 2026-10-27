"""Create a small synthetic data root to smoke-test the pipeline (no real data needed).

RFW: the real image list (``data/rfw_image_order.txt``) with random, identity-
clustered embeddings (so the shipped VLM labels and group indices are valid).
CelebA: a few thousand fake images with random attributes and embeddings.

    python tests/make_synthetic_data.py --out /tmp/latentalign_synth --dim 64
    LATENTALIGN_DATA=/tmp/latentalign_synth python -m experiments.table1 --dataset rfw
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
MODELS = ["arcface", "cosface", "elasticface", "partialFC"]


def clustered(ids, dim, rng, spread=0.6):
    uniq, inv = np.unique(ids, return_inverse=True)
    centres = rng.normal(size=(len(uniq), dim))
    attr_dirs = rng.normal(size=(8, dim))
    x = centres[inv] + spread * rng.normal(size=(len(ids), dim))
    x += 0.5 * rng.normal(size=(len(ids), 8)) @ attr_dirs
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--celeba-n", type=int, default=3000)
    args = p.parse_args()
    rng = np.random.default_rng(0)
    out = Path(args.out)

    names = np.array((REPO / "data" / "rfw_image_order.txt").read_text().split())
    ids = np.array([n.split("/")[-2] for n in names])
    e = out / "RFW" / "embeddings"
    e.mkdir(parents=True, exist_ok=True)
    np.save(e / "x_names.npy", np.array(["RFW/" + n for n in names]))
    np.save(e / "x_ids.npy", ids)
    for m in MODELS:
        np.save(e / f"embd_{m}.npy", clustered(ids, args.dim, rng))

    c = out / "CelebA"
    (c / "embeddings").mkdir(parents=True, exist_ok=True)
    cn = np.array([f"img_align_celeba/{i:06d}.jpg" for i in range(1, args.celeba_n + 1)])
    cid = rng.integers(0, args.celeba_n // 20, size=args.celeba_n)
    np.save(c / "embeddings" / "names.npy", cn)
    for m in MODELS:
        np.save(c / "embeddings" / f"embd_{m}_r100.npy", clustered(cid, args.dim, rng))
    pd.DataFrame({"image_id": [n.split("/")[-1] for n in cn], "id": cid}).to_csv(
        c / "identity_CelebA.txt", sep=" ", header=False, index=False)
    attrs = pd.DataFrame(rng.choice([-1, 1], size=(args.celeba_n, 40)), columns=[f"attr_{i}" for i in range(40)])
    attrs.insert(0, "image_id", [n.split("/")[-1] for n in cn])
    attrs.to_csv(c / "label_cleaned_mso.csv", index=False)
    print("Synthetic data written to", out)


if __name__ == "__main__":
    main()
