"""Supp. Figs. 4-5 -- sensitivity of LatentAlign to the initialisation and to tau.

Supp. Fig. 4: similarity-graph seeds (x axis: graph connectivity threshold).
Supp. Fig. 5: random seeds (x axis: size of the random initial group).
One curve per projection threshold tau. Panels: FMR of the top-100 most biased
groups, semantic distance of the top-100 most coherent groups, average group
size and number of groups (groups with >= 20 identities and mean impostor
similarity > 0.1).

Input: ``run_sweeps.py --methods graph_init random_init``.

    python -m experiments.supp_fig4_fig5_sensitivity --init graph
    python -m experiments.supp_fig4_fig5_sensitivity --init random
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from experiments._common import base_parser, db_path
from experiments.fig4_fig5 import _fmr, _init
from latentalign.data import load_rfw
from latentalign.metrics import global_threshold
from latentalign.results_db import load_groups
from latentalign.summary import worst_case_summary

from concurrent.futures import ProcessPoolExecutor
from functools import partial


def plot(summary, model, outfile, x_label, graph):
    sns.set_theme(style="whitegrid")
    markers, ls = ["o", "s", "^", "P"], ["-", "--", "-.", ":"]
    colors = sns.color_palette("Blues", n_colors=10)[-1:] + sns.color_palette("Blues", 2)[:1] + sns.color_palette("Greens", 2)[::-1]
    with plt.rc_context(rc={"font.family": "DejaVu Sans Mono"}):
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        panels = [
            ("mean_fmr", r"FMR @ Global FMR = $10^{-3}$", r"FMR (top-100 avg) [$\uparrow$]", (0, 0.6), False),
            ("mean_compactness", "Intra-group semantic distance", r"Semantic distance (top-100 avg) [$\downarrow$]", (0, 4), False),
            ("mean_size", "Group size", "Avg group size", (10, 1e5), True),
            ("n_groups", "Number of groups", "Total number of groups", (1, 2e4), True),
        ]
        for ax, (col, ylab, title, ylim, log) in zip(axes, panels):
            for i, thr in enumerate(sorted(summary["threshold"].unique())):
                sub = summary[summary["threshold"] == thr].sort_values("tau")
                if graph:
                    sub = sub[sub.tau > 0.3]
                ax.plot(sub["tau"], sub[col], marker=markers[i % 4], linestyle=ls[i % 4], color=colors[i % 4], label=f"τ = {thr}")
                if f"{col}_lower" in sub:
                    ax.fill_between(sub["tau"], sub[f"{col}_lower"], sub[f"{col}_upper"], alpha=0.2, color=ax.lines[-1].get_color())
            ax.set_xlabel(x_label)
            ax.set_ylabel(ylab)
            ax.set_title(title)
            ax.set_ylim(*ylim)
            if log:
                ax.set_yscale("log")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=len(labels), fontsize=13, frameon=False)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(outfile, dpi=150)
        plt.close()
    print("Saved", outfile)


def main():
    p = base_parser(__doc__)
    p.add_argument("--init", choices=["graph", "random"], default="graph")
    p.add_argument("--db-dir", default=None)
    args = p.parse_args()
    out = Path(args.out) / "figures"
    out.mkdir(parents=True, exist_ok=True)
    for model in args.models:
        ds = load_rfw(model, args.data_root, labels=None)
        T = global_threshold(ds.embeddings, ds.ids, 1e-3)
        df = load_groups(db_path(f"{args.init}_init", model, args.db_dir), model)
        with ProcessPoolExecutor(initializer=_init, initargs=(ds.embeddings, ds.ids)) as ex:
            df["fmr"] = list(ex.map(partial(_fmr, threshold=T), df["indices"], chunksize=64))
        summary = worst_case_summary(df, by=["tau", "threshold"])
        summary.to_csv(out / f"supp_sensitivity_{args.init}_{model}.csv", index=False)
        plot(summary, model, out / f"supp_sensitivity_{args.init}_{model}.png",
             "Graph connectivity threshold" if args.init == "graph" else "Size of initial group", args.init == "graph")


if __name__ == "__main__":
    main()
