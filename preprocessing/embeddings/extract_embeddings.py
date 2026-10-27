"""Extract L2-normalised face embeddings of RFW or CelebA with the four backbones.

Faces are detected with InsightFace (detection module only), aligned to the
standard 112x112 ArcFace template with a similarity transform and fed to an
IResNet-100 backbone (``arcface_torch``).

    python preprocessing/embeddings/extract_embeddings.py --dataset rfw    --models-dir MODELS
    python preprocessing/embeddings/extract_embeddings.py --dataset celeba --models-dir MODELS

Outputs (in ``<data-root>/<RFW|CelebA>/embeddings``):
    RFW:    embd_{model}.npy, x_names.npy, x_ids.npy
    CelebA: embd_{model}_r100.npy, names.npy

Weights: download the IResNet-100 checkpoints from the InsightFace model zoo
(ArcFace MS1MV3, CosFace Glint360K, PartialFC Glint360K) and from the
ElasticFace repository, and place them as in ``WEIGHTS`` below.

Notes
-----
* Images are read with OpenCV (BGR) and converted to [0, 1] tensors without
  mean/std normalisation. This is the preprocessing used to produce the paper
  results; the official arcface_torch inference uses RGB and (x - 0.5) / 0.5.
* Images in which no face is detected are skipped (RFW: 40,602 of 40,607
  images are kept).
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from skimage import transform
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from arcface_torch.backbones import get_model  # noqa: E402

WEIGHTS = {  # relative to --models-dir
    "arcface": "ms1mv3_arcface_r100_fp16/backbone.pth",
    "cosface": "glint360k_cosface_r100_fp16_0.1/backbone.pth",
    "elasticface": "ElasticFace-Cos+/295672backbone.pth",  # RFW; CelebA used ElasticFace-Arc (see ELASTICFACE_CELEBA)
    "partialFC": "partial_fc_glint360k_r100/16_backbone.pth",
}
ELASTICFACE_CELEBA = "ElasticFace-Arc/295672backbone.pth"  # the paper's CelebA ElasticFace embeddings

# ArcFace 5-point template (112x112)
SRC = np.array([[30.2946, 51.6963], [65.5318, 51.5014], [48.0252, 71.7366],
                [33.5493, 92.3655], [62.7299, 92.2041]], dtype=np.float32)
SRC[:, 0] += 8.0


class FaceImages(torch.utils.data.Dataset):
    def __init__(self, files):
        from insightface.app import FaceAnalysis

        self.files = files
        self.app = FaceAnalysis(allowed_modules=["detection"], providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        self.app.prepare(ctx_id=0, det_size=(128, 128))
        self.tform = transform.SimilarityTransform()
        self.to_tensor = T.ToTensor()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        image = cv2.imread(path)
        faces = self.app.get(image)
        if not faces:
            return None
        self.tform.estimate(faces[0]["kps"], SRC)
        aligned = cv2.warpAffine(image, self.tform.params[0:2, :], (112, 112))
        return self.to_tensor(aligned), path, os.path.basename(os.path.dirname(path))


def collate(batch):
    batch = [b for b in batch if b is not None]
    return torch.utils.data.dataloader.default_collate(batch)


def load_backbone(weights, device):
    model = get_model("r100", fp16=False)
    model.load_state_dict(torch.load(weights, map_location=device))
    return model.eval().to(device)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=["rfw", "celeba"], required=True)
    p.add_argument("--data-root", default=os.environ.get("LATENTALIGN_DATA", "data"))
    p.add_argument("--models-dir", default="MODELS")
    p.add_argument("--models", nargs="+", default=list(WEIGHTS))
    p.add_argument("--batch-size", type=int, default=500)
    args = p.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
    out = root / "embeddings"
    out.mkdir(parents=True, exist_ok=True)

    weights = dict(WEIGHTS)
    if args.dataset == "celeba":
        weights["elasticface"] = ELASTICFACE_CELEBA
    models = {m: load_backbone(Path(args.models_dir) / weights[m], device) for m in args.models}
    loader = torch.utils.data.DataLoader(FaceImages(files), collate_fn=collate, batch_size=args.batch_size)
    emb = {m: [] for m in models}
    names, ids = [], []
    for batch, paths, idents in tqdm(loader):
        batch = batch.to(device)
        with torch.no_grad():
            for m, net in models.items():
                emb[m].append(torch.nn.functional.normalize(net(batch)).cpu().numpy())
        names.extend(paths)
        ids.extend(idents)

    rel = [os.path.relpath(n, root) for n in names]   # "data/African/..." or "img_align_celeba/000001.jpg"
    if args.dataset == "rfw":
        np.save(out / "x_names.npy", np.array(rel))
        np.save(out / "x_ids.npy", np.array(ids))
        for m in models:
            np.save(out / f"embd_{m}.npy", np.concatenate(emb[m]))
    else:
        np.save(out / "names.npy", np.array(rel))
        for m in models:
            np.save(out / f"embd_{m}_r100.npy", np.concatenate(emb[m]))
    print("Saved embeddings to", out)


if __name__ == "__main__":
    main()
