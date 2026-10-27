"""Fig. 3 -- predicted attribute probability vs. traversal strength (group A).

    python traversals/plot_fig3.py                       # values of the paper
    python traversals/plot_fig3.py --input my_probs.json # same format, your classifier outputs
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

REPO = Path(__file__).resolve().parents[1]
HIGHLIGHT = ("Fully Visible Forehead", "Wearing Hat")


def plot_directions(ax, block, title):
    main_colors, hl_colors = sns.color_palette("Greens", n_colors=4), sns.color_palette("Blues", n_colors=2)
    ls = ["-", "-", ":", ":"]
    m = h = 0
    for attr, values in block["values"].items():
        if attr in HIGHLIGHT:
            kw = dict(color=hl_colors[h % 2], linestyle="--", marker="s", alpha=0.7)
            h += 1
        else:
            kw = dict(color=main_colors[m % 4], linestyle=ls[m % 4], marker="o", alpha=1)
            m += 1
        ax.plot(block["strength"], values, markersize=6, linewidth=2.2, label=attr, **kw)
    ax.set_xlabel("Strength")
    ax.set_title(title, weight="bold")
    ax.set_xticks(block["strength"])
    ax.set_ylim(0, 1)
    ax.grid(True, linestyle="--", alpha=0.6)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", default=str(REPO / "data" / "traversals" / "attribute_probabilities.json"))
    p.add_argument("--out", default=str(REPO / "results" / "figures"))
    args = p.parse_args()
    data = json.loads(Path(args.input).read_text())
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({"axes.titlesize": 16, "axes.labelsize": 14, "legend.fontsize": 12,
                         "xtick.labelsize": 12, "ytick.labelsize": 12})
    Path(args.out).mkdir(parents=True, exist_ok=True)
    for decoder in ("arc2face", "infiniteyou"):
        with plt.rc_context(rc={"font.family": "DejaVu Sans Mono"}):
            fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
            plot_directions(axes[0], data[decoder]["positive"], "Positive direction (attribute addition)")
            plot_directions(axes[1], data[decoder]["negative"], "Negative direction (attribute removal)")
            axes[0].legend(loc="best", ncol=3, fontsize=12, frameon=False)
            axes[0].set_ylabel("Predicted Probability")
            plt.tight_layout()
            f = Path(args.out) / f"fig3_{decoder}.pdf"
            plt.savefig(f)
            plt.close()
            print("Saved", f)


if __name__ == "__main__":
    main()
