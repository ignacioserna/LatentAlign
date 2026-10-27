"""Extract the antelopev2 (ArcFace R100, Glint360K) embeddings used by Arc2Face.

These embeddings live in the space Arc2Face is conditioned on, so the latent
directions for the traversals of Sec. 4.2 (Figs. 2-3) are computed with them.

    python preprocessing/embeddings/extract_arc2face_embeddings.py --dataset rfw
    python preprocessing/embeddings/extract_arc2face_embeddings.py --dataset celeba

Requires the ``antelopev2`` InsightFace package in ``./models/antelopev2``
(see the Arc2Face repository). Outputs ``<data-root>/<RFW|CelebA>/embeddings/arc2face/``.
"""
from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


class Faces(torch.utils.data.Dataset):
    def __init__(self, files, root="./"):
        from insightface.app import FaceAnalysis

        self.files = files
        self.app = FaceAnalysis(name="antelopev2", root=root, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        self.app.prepare(ctx_id=0, det_size=(128, 128))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        img = np.array(Image.open(path).convert("RGB"))[:, :, ::-1]
        faces = self.app.get(img)
        if not faces:
            return None
        face = sorted(faces, key=lambda x: (x["bbox"][2] - x["bbox"][0]) * (x["bbox"][3] - x["bbox"][1]))[-1]
        e = torch.tensor(face["embedding"], dtype=torch.float32)
        return e / torch.norm(e), path, os.path.basename(os.path.dirname(path))


def collate(batch):
    batch = [b for b in batch if b is not None]
    return torch.utils.data.dataloader.default_collate(batch)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=["rfw", "celeba"], required=True)
    p.add_argument("--data-root", default=os.environ.get("LATENTALIGN_DATA", "data"))
    p.add_argument("--insightface-root", default="./")
    args = p.parse_args()

    root = Path(args.data_root) / ("RFW" if args.dataset == "rfw" else "CelebA")
    if args.dataset == "rfw":
        # Use the image order of the paper: the group indices shipped in data/groups refer to it.
        order = Path(__file__).resolve().parents[2] / "data" / "rfw_image_order.txt"
        files = [str(root / line) for line in order.read_text().split()] if order.exists() \
            else sorted(glob.glob(str(root / "data" / "*" / "*" / "*")))
    else:
        order = Path(__file__).resolve().parents[2] / "data" / "celeba_image_order.txt"
        files = [str(root / line) for line in order.read_text().split()] if order.exists() \
            else sorted(glob.glob(str(root / "img_align_celeba" / "*")))
    loader = torch.utils.data.DataLoader(Faces(files, args.insightface_root), collate_fn=collate, batch_size=500)
    emb, names, ids = [], [], []
    for e, paths, idents in tqdm(loader):
        emb.append(e.numpy())
        names.extend(paths)
        ids.extend(idents)
    out = root / "embeddings" / "arc2face"
    out.mkdir(parents=True, exist_ok=True)
    prefix = "RFW" if args.dataset == "rfw" else "CelebA"
    np.save(out / f"{prefix}_arcface.npy", np.concatenate(emb))
    np.save(out / f"{prefix}_names.npy", np.array([os.path.relpath(n, root) for n in names]))
    np.save(out / f"{prefix}_ids.npy", np.array(ids))
    print("Saved to", out)


if __name__ == "__main__":
    main()
