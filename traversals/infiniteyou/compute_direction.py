"""Direction of a LatentAlign group in the ArcFace space used by InfiniteYou.

The images of the group are re-embedded with InfiniteYou's face pipeline
(antelopev2 detection + facexlib ArcFace) and the inverse-frequency weighted
direction (Eq. 1) is computed from these embeddings.

    export PYTHONPATH=$PWD/InfiniteYou:$PYTHONPATH
    python traversals/infiniteyou/compute_direction.py --group A --out results/traversals/infiniteyou/group_A.npy

Note: the direction used in the paper was computed from the 94 images of
group A in which this pipeline detected a face.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from latentalign.algorithm import get_direction  # noqa: E402


def main():
    from facexlib.recognition import init_recognition_model
    from insightface.app import FaceAnalysis
    from pipelines.pipeline_infu_flux import extract_arcface_bgr_embedding

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--group", default="A")
    p.add_argument("--data-root", default=os.environ.get("LATENTALIGN_DATA", str(REPO / "data")))
    p.add_argument("--insightface-root", default="models/InfiniteYou/supports/insightface")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    group = json.loads((REPO / "data" / "groups" / "rfw_groups_ABCD.json").read_text())[args.group]
    order = (REPO / "data" / "rfw_image_order.txt").read_text().split()
    app = FaceAnalysis(name="antelopev2", root=args.insightface_root, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(320, 320))
    arcface = init_recognition_model("arcface", device="cuda")

    embs, ids = [], []
    for i in tqdm(group):
        rel = order[i]
        img = cv2.cvtColor(np.array(Image.open(Path(args.data_root) / "RFW" / rel).convert("RGB")), cv2.COLOR_RGB2BGR)
        faces = app.get(img)
        if not faces:
            continue
        face = sorted(faces, key=lambda x: (x["bbox"][2] - x["bbox"][0]) * (x["bbox"][3] - x["bbox"][1]))[-1]
        with torch.no_grad():
            embs.append(extract_arcface_bgr_embedding(img, face["kps"], arcface).cpu().numpy())
        ids.append(rel.split("/")[-2])  # identity folder
    v = get_direction(np.array(ids), np.array(embs))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, v)
    print(f"Direction from {len(embs)}/{len(group)} images saved to {args.out}")


if __name__ == "__main__":
    main()
