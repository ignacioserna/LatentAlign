"""Latent traversals decoded with Arc2Face (Stable Diffusion 1.5) -- Figs. 2 and 3(a).

Arc2Face is conditioned on antelopev2 ArcFace embeddings, so the group
directions are computed from the antelopev2 embeddings of RFW
(``preprocessing/embeddings/extract_arc2face_embeddings.py``).

Setup: clone https://github.com/foivospar/Arc2Face, install its requirements,
download its ``models`` folder and add the repository to ``PYTHONPATH``.

Modes
-----
removal       Fig. 2(a): images of group A-D traversed with t = -0.5
addition      Fig. 2(b): CelebA faces traversed towards groups A-D with t = 0.45
quantitative  Fig. 3(a): 1,000 random CelebA faces (seed 42) with t = 0.1..0.5
              (positive) and the images of group A with t = -0.1..-0.5
              (negative); the outputs are scored with the attribute classifier

    python traversals/arc2face/traverse_arc2face.py --mode removal
    python traversals/arc2face/traverse_arc2face.py --mode addition --groups A B C D
    python traversals/arc2face/traverse_arc2face.py --mode quantitative --groups A
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from latentalign.algorithm import get_direction  # noqa: E402
from latentalign.traversal import slerp  # noqa: E402

# Images shown in Fig. 2(a) (RFW indices) and Fig. 2(b) (CelebA indices)
REMOVAL_FACES = {
    "A": [24294, 21175, 24684, 24297, 29843, 24686, 21530, 21446, 21533, 29495, 21531, 29841, 24295, 21535],
    "B": [36862, 31395, 39952, 39953, 34440, 39956, 40548, 35076, 34299, 36168, 35814, 40547, 36273, 37375],
    "C": [34041, 37763, 33339, 32074, 34040, 31958, 35139, 30747, 39728, 38783, 39502, 30746, 37465, 30727],
    "D": [12456, 11305, 13535, 10962, 15362, 11304, 17706, 14830, 13938, 15627, 19744, 13942, 12014, 17704],
}
ADDITION_FACES = [35087, 34529, 34453, 185854, 165802, 9]


def load_pipeline(models_dir="models"):
    from arc2face import CLIPTextModelWrapper  # from the Arc2Face repository
    from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline, UNet2DConditionModel

    encoder = CLIPTextModelWrapper.from_pretrained(models_dir, subfolder="encoder", torch_dtype=torch.float16)
    unet = UNet2DConditionModel.from_pretrained(models_dir, subfolder="arc2face", torch_dtype=torch.float16)
    pipe = StableDiffusionPipeline.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5", text_encoder=encoder, unet=unet,
        torch_dtype=torch.float16, safety_checker=None)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    return pipe.to("cuda")


def generate(pipe, emb, n_images=4, seed=None):
    from arc2face import project_face_embs

    e = torch.tensor(emb, dtype=torch.float16).unsqueeze(0).to("cuda")
    e = project_face_embs(pipe, e)
    gen = torch.Generator("cuda").manual_seed(seed) if seed is not None else None
    return pipe(prompt_embeds=e, num_inference_steps=25, guidance_scale=3.0, num_images_per_prompt=n_images,
                generator=gen).images


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["removal", "addition", "quantitative"], required=True)
    p.add_argument("--groups", nargs="+", default=["A", "B", "C", "D"])
    p.add_argument("--data-root", default=os.environ.get("LATENTALIGN_DATA", str(REPO / "data")))
    p.add_argument("--models-dir", default="models", help="Arc2Face 'models' folder")
    p.add_argument("--out", default=str(REPO / "results" / "traversals" / "arc2face"))
    p.add_argument("--n-random", type=int, default=1000)
    p.add_argument("--seed", type=int, default=None, help="generator seed (the paper did not fix it)")
    args = p.parse_args()

    emb_dir = Path(args.data_root) / "RFW" / "embeddings" / "arc2face"
    rfw = np.load(emb_dir / "RFW_arcface.npy").squeeze()
    order = (REPO / "data" / "rfw_image_order.txt").read_text().split()
    rfw_ids = np.array([r.split("/")[-2] for r in order])  # identity folder
    celeba = np.load(Path(args.data_root) / "CelebA" / "embeddings" / "arc2face" / "CelebA_arcface.npy").squeeze()
    groups = json.loads((REPO / "data" / "groups" / "rfw_groups_ABCD.json").read_text())
    pipe = load_pipeline(args.models_dir)

    for gname in args.groups:
        g = groups[gname]
        v = get_direction(rfw_ids[g], rfw[g])
        out = Path(args.out) / args.mode / f"group_{gname}"
        out.mkdir(parents=True, exist_ok=True)
        if args.mode == "removal":
            for face in REMOVAL_FACES[gname]:
                for tag, e in [("original", rfw[face]), ("subtraction", slerp(rfw[face], v, -0.5))]:
                    for i, img in enumerate(generate(pipe, e, seed=args.seed)):
                        img.save(out / f"{face}_{tag}_{i}.png")
        elif args.mode == "addition":
            for face in ADDITION_FACES:
                for tag, e in [("original", celeba[face]), ("addition", slerp(celeba[face], v, 0.45))]:
                    for i, img in enumerate(generate(pipe, e, seed=args.seed)):
                        img.save(out / f"{face}_{tag}_{i}.png")
        else:
            rng = random.Random(42)
            for face in rng.sample(range(len(celeba)), args.n_random):
                for s in [0.1, 0.2, 0.3, 0.4, 0.5]:
                    for i, img in enumerate(generate(pipe, slerp(celeba[face], v, s), n_images=2, seed=args.seed)):
                        img.save(out / f"celeba_{face}_addition_{i}_{s}.png")
            for face in g:
                for s in [0.1, 0.2, 0.3, 0.4, 0.5]:
                    for i, img in enumerate(generate(pipe, slerp(rfw[face], v, -s), seed=args.seed)):
                        img.save(out / f"rfw_{face}_subtraction_{i}_{s}.png")
        print("Saved", out)


if __name__ == "__main__":
    main()
