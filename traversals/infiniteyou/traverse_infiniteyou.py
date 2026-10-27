"""Latent traversals decoded with InfiniteYou (FLUX.1-dev) -- Fig. 3(b).

InfiniteYou is conditioned on the 512-d ArcFace embedding of an input face
(facexlib ``arcface`` model). We replace that embedding by
``slerp(e, v, t)`` for t in ``--strengths`` and decode it, where ``v`` is the
direction of a LatentAlign group computed in the same embedding space
(``compute_direction.py``).

Setup
-----
    git clone https://github.com/bytedance/InfiniteYou && pip install -r InfiniteYou/requirements.txt
    export PYTHONPATH=$PWD/InfiniteYou:$PYTHONPATH
    huggingface-cli login   # access to black-forest-labs/FLUX.1-dev

Usage
-----
    # direction of group A in InfiniteYou's ArcFace space
    python traversals/infiniteyou/compute_direction.py --group A --out results/traversals/infiniteyou/group_A.npy
    # positive traversal of 1,000 random CelebA faces (Fig. 3b, "addition")
    python traversals/infiniteyou/traverse_infiniteyou.py --direction results/traversals/infiniteyou/group_A.npy \
        --celeba-random 1000 --strengths 0 0.1 0.2 0.3 0.4 0.5
    # negative traversal of the images of group A ("removal")
    python traversals/infiniteyou/traverse_infiniteyou.py --direction results/traversals/infiniteyou/group_A.npy \
        --rfw-group A --strengths 0 -0.1 -0.2 -0.3 -0.4 -0.5

Output images are named ``<image>_<i>.png`` (i = index of the strength) and are
the input of the attribute classifier used for Fig. 3.
This file reuses code from InfiniteYou (Apache-2.0, (c) 2025 Bytedance Ltd.).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from latentalign.traversal import slerp_torch  # noqa: E402

try:
    from pipelines.pipeline_infu_flux import (  # from the InfiniteYou repository
        InfUFluxPipeline,
        extract_arcface_bgr_embedding,
        seed_everything,
    )
except ImportError as e:  # pragma: no cover
    raise SystemExit("Add the InfiniteYou repository to PYTHONPATH (see the docstring).") from e


class InfUFluxTraversal(InfUFluxPipeline):
    @torch.no_grad()
    def id_embedding(self, image: Image.Image):
        img = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
        faces = self._detect_face(img)
        if len(faces) == 0:
            return None
        face = sorted(faces, key=lambda x: (x["bbox"][2] - x["bbox"][0]) * (x["bbox"][3] - x["bbox"][1]))[-1]
        self.arcface_model.to("cuda")
        e = extract_arcface_bgr_embedding(img, face["kps"], self.arcface_model)
        self.arcface_model.cpu()
        torch.cuda.empty_cache()
        return e

    @torch.no_grad()
    def decode(self, emb, prompt="Portrait", width=224, height=224, seed=42, guidance_scale=3.5, num_steps=30):
        id_embed = emb.clone().unsqueeze(0).float().cuda().reshape([1, -1, 512]).to(dtype=torch.bfloat16)
        self.image_proj_model.to("cuda", torch.bfloat16)
        id_embed = self.image_proj_model(id_embed)
        self.image_proj_model.cpu()
        torch.cuda.empty_cache()
        control_image = Image.fromarray(np.zeros([height, width, 3], dtype=np.uint8))
        seed_everything(seed)
        return self.pipe(
            prompt=prompt, controlnet_prompt_embeds=id_embed, control_image=control_image,
            guidance_scale=guidance_scale, num_inference_steps=num_steps, controlnet_guidance_scale=1.0,
            controlnet_conditioning_scale=1.0, control_guidance_start=0.0, control_guidance_end=1.0,
            height=height, width=width,
        ).images[0]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--direction", required=True, help=".npy with the 512-d direction")
    p.add_argument("--strengths", type=float, nargs="+", default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    p.add_argument("--celeba-random", type=int, default=0, help="number of random CelebA images")
    p.add_argument("--rfw-group", default=None, help="traverse the images of this group (A-D)")
    p.add_argument("--images", nargs="*", default=[])
    p.add_argument("--data-root", default=os.environ.get("LATENTALIGN_DATA", str(REPO / "data")))
    p.add_argument("--out", default=str(REPO / "results" / "traversals" / "infiniteyou"))
    p.add_argument("--prompt", default="Portrait")
    p.add_argument("--base-model-path", default="black-forest-labs/FLUX.1-dev")
    p.add_argument("--model-dir", default="ByteDance/InfiniteYou")
    p.add_argument("--model-version", default="sim_stage1")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--size", type=int, default=224)
    args = p.parse_args()

    images = list(args.images)
    root = Path(args.data_root)
    if args.celeba_random:
        all_images = sorted(os.listdir(root / "CelebA" / "img_align_celeba"))
        rng = random.Random(args.seed)
        images += [str(root / "CelebA" / "img_align_celeba" / f) for f in rng.sample(all_images, args.celeba_random)]
    if args.rfw_group:
        group = json.loads((REPO / "data" / "groups" / "rfw_groups_ABCD.json").read_text())[args.rfw_group]
        order = (REPO / "data" / "rfw_image_order.txt").read_text().split()
        images += [str(root / "RFW" / order[i]) for i in group]

    pipe = InfUFluxTraversal(
        base_model_path=args.base_model_path,
        infu_model_path=os.path.join(args.model_dir, "infu_flux_v1.0", args.model_version),
        insightface_root_path=os.path.join(args.model_dir, "supports", "insightface"),
        infu_flux_version="v1.0", model_version=args.model_version,
    )
    v = torch.from_numpy(np.load(args.direction)).to("cuda")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for path in images:
        e = pipe.id_embedding(Image.open(path))
        if e is None:
            print("no face:", path)
            continue
        name = Path(path).stem
        for i, t in enumerate(args.strengths):
            img = pipe.decode(slerp_torch(e, v, t), prompt=args.prompt, width=args.size, height=args.size, seed=args.seed)
            img.save(out / f"{name}_0.{i}.png")


if __name__ == "__main__":
    main()
