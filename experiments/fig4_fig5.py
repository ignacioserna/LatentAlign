"""Figures 4 and 5 -- parameter sweeps on RFW.

Fig. 4: average semantic distance of the 100 most coherent groups per size.
Fig. 5: average FMR (at the threshold where the whole dataset has FMR = 1e-3)
        of the 100 most biased groups per size.

Input: the databases written by ``experiments/run_sweeps.py``
(methods latentalign, nn, kmeans, skmeans, dbscan and optionally weighted_svd).

    python -m experiments.fig4_fig5            # paper version (5 methods)
    python -m experiments.fig4_fig5 --with-svd # also plot weighted-SVD
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from experiments._common import base_parser, db_path
from latentalign.data import load_rfw
from latentalign.metrics import global_threshold, group_fmr
from latentalign.results_db import load_groups
from latentalign.summary import worst_case_summary

# method -> (label, bin width or None for exact sizes)
METHODS = {
    "latentalign": ("LatentAlign", None),
    "nn": ("NN", None),
    "kmeans": ("k-means", 10),
    "skmeans": ("spk-means", 10),
    "dbscan": ("DBSCAN", 10),
    "weighted_svd": ("weighted-SVD", None),
}
COLORS = (sns.color_palette("Blues", 2)[:1] + sns.color_palette("Greens", 2) + sns.color_palette("Reds")[-1:]
          + sns.color_palette("Blues", n_colors=10)[-1:] + sns.color_palette("Oranges", 1))
MARKERS = ["o", "s", "^", "P", "v", "X"]
LINESTYLES = ["-", "--", "-.", ":"]

_E = _IDS = None


def _init(E, ids):
    global _E, _IDS
    _E, _IDS = E, ids


def _fmr(indices, threshold):
    return group_fmr(_E, _IDS, indices, threshold)


def summarise(model, methods, data_root, db_dir, threshold=None):
    ds = load_rfw(model, data_root, labels=None)
    T = threshold if threshold is not None else global_threshold(ds.embeddings, ds.ids, 1e-3)
    print(f"{model}: global threshold @ FMR=1e-3 = {T:.4f}")
    out = {}
    for m in methods:
        try:
            df = load_groups(db_path(m, model, db_dir), model)
        except Exception as e:  # missing database or no group (e.g. DBSCAN on PartialFC)
            print(f"  {m}: skipped ({e})")
            continue
        with ProcessPoolExecutor(initializer=_init, initargs=(ds.embeddings, ds.ids)) as ex:
            df["fmr"] = list(ex.map(partial(_fmr, threshold=T), df["indices"], chunksize=64))
        out[m] = worst_case_summary(df, by="size", bin_width=METHODS[m][1])
        print(f"  {m}: {len(df)} groups")
    return out, T


def plot(summaries, models, methods, kind, outfile, ylim):
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({"axes.titlesize": 16, "axes.labelsize": 14, "legend.fontsize": 12,
                         "xtick.labelsize": 12, "ytick.labelsize": 12})
    col = "fmr" if kind == "fmr" else "compactness"
    with plt.rc_context(rc={"font.family": "DejaVu Sans Mono"}):
        fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 5), sharey=True)
        axes = np.atleast_1d(axes)
        for ax, model in zip(axes, models):
            for i, m in enumerate(methods):
                if m not in summaries[model]:
                    continue
                s = summaries[model][m]
                x, y = s[f"{col}_size"].values, s[f"mean_{col}"].values
                ax.plot(x, y, marker=MARKERS[i], linestyle=LINESTYLES[i % 4], color=COLORS[list(METHODS).index(m)],
                        label=METHODS[m][0])
                ax.fill_between(x, s[f"mean_{col}_lower"].values, s[f"mean_{col}_upper"].values, alpha=0.2,
                                color=ax.lines[-1].get_color())
            ax.set_title(model)
            ax.set_xlabel("Size of cluster")
            ax.set_xlim(47, 127)
            ax.set_ylim(*ylim)
            ax.grid(True)
        if kind == "fmr":
            fig.suptitle(r"FMR (top-100 avg) [$\uparrow$]", fontsize=20, fontweight="bold")
            axes[0].set_ylabel(r"FMR @ Global FMR = $10^{-3}$")
        else:
            fig.suptitle(r"Semantic distance (top-100 avg) [$\downarrow$]", fontsize=20, fontweight="bold")
            axes[0].set_ylabel("Intra-group semantic distance")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=len(methods), frameon=False, bbox_to_anchor=(0.5, 0.88))
        fig.tight_layout()
        fig.subplots_adjust(top=0.75)
        fig.savefig(outfile, format="pdf", bbox_inches="tight")
        plt.close(fig)
    print("Saved", outfile)


def main():
    p = base_parser(__doc__)
    p.add_argument("--db-dir", default=None)
    p.add_argument("--with-svd", action="store_true")
    p.add_argument("--fmr-ylim", type=float, nargs=2, default=(0, 0.6))
    p.add_argument("--coherence-ylim", type=float, nargs=2, default=(0, 4))
    args = p.parse_args()
    methods = ["latentalign", "nn", "kmeans", "skmeans", "dbscan"] + (["weighted_svd"] if args.with_svd else [])

    summaries = {}
    for model in args.models:
        summaries[model], _ = summarise(model, methods, args.data_root, args.db_dir)
    out = Path(args.out) / "figures"
    out.mkdir(parents=True, exist_ok=True)
    for model in args.models:
        for m in summaries[model]:
            summaries[model][m].to_csv(out / f"fig4_fig5_summary_{model}_{m}.csv", index=False)
    plot(summaries, args.models, methods, "coherence", out / "fig4_semantic_distance.pdf", args.coherence_ylim)
    plot(summaries, args.models, methods, "fmr", out / "fig5_fmr.pdf", args.fmr_ylim)


if __name__ == "__main__":
    main()
