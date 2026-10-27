"""Supp. Figs. 6-12 -- CelebA groups grown from hand-picked seeds (ArcFace, tau = 0.25).

The single-image seeds of
Supp. Figs. 11-12 do not stop (they reach the 1,000-iteration cap); the figures
show the first images that were added.

Images are shown left-to-right, top-to-bottom in the order in which LatentAlign
added them; the seed images are framed in green. Seeds are listed in
``data/seeds/celeba_seeds.json``.

    python -m experiments.supp_figs6_12_celeba_groups --figure fig6_tennis_cap
    python -m experiments.supp_figs6_12_celeba_groups --all
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import seaborn as sns

from experiments._common import REPO, base_parser
from latentalign import grow_threshold
from latentalign.data import data_root, load_celeba

SEEDS_FILE = REPO / "data" / "seeds" / "celeba_seeds.json"


def grid(paths, n_seed, outfile, n_cols=23, max_rows=20):
    paths = paths[: n_cols * max_rows]
    n_rows = -(-len(paths) // n_cols)
    bg = sns.cubehelix_palette(5, rot=-.5, light=.95)[0]
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.5, n_rows * 2.5), squeeze=False)
    fig.patch.set_facecolor(bg)
    for i, ax in enumerate(axs.flat):
        ax.axis("off")
        if i < len(paths):
            ax.imshow(mpimg.imread(paths[i]))
            if i < n_seed:
                for s in ax.spines.values():
                    s.set_visible(True)
                    s.set_edgecolor("limegreen")
                    s.set_linewidth(6)
                ax.axis("on")
                ax.set_xticks([])
                ax.set_yticks([])
    plt.tight_layout(pad=2.0)
    plt.savefig(outfile, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print("Saved", outfile)


def main():
    p = base_parser(__doc__)
    p.add_argument("--figure", default=None)
    p.add_argument("--all", action="store_true")
    p.add_argument("--model", default="arcface")
    p.add_argument("--tau", type=float, default=0.25)
    args = p.parse_args()
    seeds = json.loads(SEEDS_FILE.read_text())
    todo = [k for k in seeds if not k.startswith("_")] if args.all else [args.figure]
    ds = load_celeba(args.model, args.data_root, attributes=False)
    img_dir = data_root(args.data_root) / "CelebA"
    out = Path(args.out) / "figures"
    out.mkdir(parents=True, exist_ok=True)
    for name in todo:
        seed = seeds[name]["seed"]
        group = grow_threshold(seed, ds.embeddings, ds.ids, tau=args.tau)
        print(f"{name}: {len(group)} images")
        paths = [img_dir / str(n) for n in ds.names[group]]
        grid(paths, len(seed), out / f"supp_{name}.pdf")


if __name__ == "__main__":
    main()
