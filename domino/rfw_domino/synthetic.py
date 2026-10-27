"""Stage 00 (optional) - generate a synthetic mini-RFW for smoke-testing.

Creates an RFW-shaped directory tree (test/data/<Race>/<id>/<id>_000k.jpg and
test/txts/<Race>/<Race>_pairs.txt) with procedurally generated 64x64 images:

    image = race base color + identity pattern + per-image noise

A failure mode is *planted*: identities of one race (default African) get a
much weaker identity pattern, so with the pipeline's `fake` pixel embedders
their negative (impostor) pairs look similar and cross a global verification
threshold - i.e. concentrated false matches in one demographic slice. The
smoke test then checks that Domino recovers this slice without being told
about race. A "glasses" attribute adds a nuisance sub-population.

This is purely for plumbing verification - the images are noise fields, not
faces.
"""

import argparse
import os

import numpy as np

from . import RACES

BASE_COLORS = {  # RGB
    "African": (105, 75, 55),
    "Asian": (235, 205, 165),
    "Caucasian": (240, 220, 205),
    "Indian": (185, 135, 95),
}


def make_identity_images(rng, base, amp, n_images, glasses, size=64):
    pattern = rng.normal(0.0, 1.0, (8, 8, 3))
    pattern = np.repeat(np.repeat(pattern, size // 8, 0), size // 8, 1)
    imgs = []
    for _ in range(n_images):
        a = amp * rng.uniform(0.6, 1.4)          # per-image quality jitter
        sigma = rng.uniform(6.0, 18.0)
        img = (
            np.asarray(base, dtype=np.float64)[None, None, :]
            + a * pattern
            + rng.normal(0.0, sigma, (size, size, 3))
            + rng.uniform(-12, 12)               # brightness jitter
        )
        if glasses:
            img[size * 3 // 8: size * 4 // 8, :, :] *= 0.35
        imgs.append(np.clip(img, 0, 255).astype(np.uint8))
    return imgs


def main(argv=None):
    from PIL import Image

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="synthetic_rfw")
    ap.add_argument("--n-ids", type=int, default=50, help="identities per race")
    ap.add_argument("--imgs-per-id", type=int, default=5)
    ap.add_argument("--pairs-per-fold", type=int, default=30,
                    help="positive (=negative) pairs per fold, 10 folds")
    ap.add_argument("--weak-race", default="African",
                    help="race with the planted weak identity signal")
    ap.add_argument("--strong-amp", type=float, default=60.0)
    ap.add_argument("--weak-amp", type=float, default=7.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    root = os.path.abspath(args.out)

    for race in RACES:
        data_dir = os.path.join(root, "test", "data", race)
        txt_dir = os.path.join(root, "test", "txts", race)
        os.makedirs(txt_dir, exist_ok=True)
        amp = args.weak_amp if race == args.weak_race else args.strong_amp

        names = [f"m.{race.lower()}{i:04d}" for i in range(args.n_ids)]
        for name in names:
            id_dir = os.path.join(data_dir, name)
            os.makedirs(id_dir, exist_ok=True)
            glasses = rng.random() < 0.3
            imgs = make_identity_images(
                rng, BASE_COLORS[race], amp, args.imgs_per_id, glasses
            )
            for j, img in enumerate(imgs, start=1):
                Image.fromarray(img).save(
                    os.path.join(id_dir, f"{name}_{j:04d}.jpg"), quality=95
                )

        lines = []
        for _fold in range(10):
            for _ in range(args.pairs_per_fold):  # positive pairs
                name = names[rng.integers(args.n_ids)]
                a, b = rng.choice(
                    np.arange(1, args.imgs_per_id + 1), 2, replace=False
                )
                lines.append(f"{name}\t{a}\t{b}")
            for _ in range(args.pairs_per_fold):  # negative pairs
                i, j = rng.choice(args.n_ids, 2, replace=False)
                a = rng.integers(1, args.imgs_per_id + 1)
                b = rng.integers(1, args.imgs_per_id + 1)
                lines.append(f"{names[i]}\t{a}\t{names[j]}\t{b}")
        with open(os.path.join(txt_dir, f"{race}_pairs.txt"), "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"[synthetic] {race}: {args.n_ids} ids, "
              f"{args.n_ids * args.imgs_per_id} images, {len(lines)} pairs"
              + ("  <- planted weak-identity race" if race == args.weak_race
                 else ""))

    print(f"[synthetic] wrote synthetic RFW to {root}")


if __name__ == "__main__":
    main()
