"""Groups R and A-D on RFW: Fig. 1(b,c), Tab. 2 and Supp. Tabs. 4-5, Supp. Figs. 2-3.

Groups
------
* A-D: groups discovered by LatentAlign (indices in ``data/groups/rfw_groups_ABCD.json``)
  A: Caucasian blonde females, B: young Indian women,
  C: Indian males with turbans, D: Chinese men with mustaches (B&W photographs).
* R: all images of 75 random identities (``--random-seed``).
* Annot.: the closest groups that can be defined with the VLM annotations
  (``ANNOTATION_FILTERS``), used in Tab. 2.

Operating point
---------------
FMRs are evaluated at the per-backbone threshold where the whole dataset
(all impostor pairs of the 40,602 RFW images) has FMR = 1e-3 (default). ``--threshold``
also accepts a fixed cosine threshold or a JSON file {model: threshold}.

    python -m experiments.table2_groups
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from experiments._common import REPO, base_parser
from latentalign.data import MODEL_TITLES, load_rfw, load_rfw_labels
from latentalign.metrics import (
    bootstrap_group_fmr,
    eer_and_fnmr,
    fmr_curve,
    global_threshold,
    group_pair_scores,
)

GROUPS_FILE = REPO / "data" / "groups" / "rfw_groups_ABCD.json"

ANNOTATION_FILTERS = {
    "A": "Gender == 'female' and Age == 'young' and `Skin Color` == 'light' and Ancestry == 'white' and "
         "`Hair Color` == 'blonde' and Bangs == 'yes' and Bald == 'no' and Beard == 'no' and Headwear == 'no'",
    "B": "Gender == 'female' and Age == 'young' and Ancestry == 'south_asian' and `Hair Color` == 'black' and "
         "Bangs == 'no' and Bald == 'no' and Beard == 'no' and Headwear == 'hijab' and Glasses == 'no'",
    "C": "Gender == 'male' and Headwear == 'turban' and `Skin Color` == 'medium' and Beard == 'full' and "
         "Age != 'young' and Ancestry == 'south_asian'",
    "D": "Gender == 'male' and Age == 'senior' and Beard == 'mustache' and Ancestry == 'asian' and "
         "`Skin Color` == 'light'",
}


def random_group(ids, n_ids=75, seed=0):
    rng = np.random.default_rng(seed)
    chosen = rng.choice(np.unique(ids), n_ids)  # with replacement, as originally
    return np.concatenate([np.where(ids == c)[0] for c in chosen])


def ridge_plot(df, title, outfile, bandwidth=0.5):
    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
    pal = sns.cubehelix_palette(5, rot=-.5, light=.7)
    g = sns.FacetGrid(df, row="Label", hue="Label", aspect=10, height=1.25, palette=pal)
    g.map(sns.kdeplot, "Genuine Score", bw_adjust=bandwidth, clip_on=True, color="lightseagreen", fill=True, alpha=1, linewidth=1.5)
    g.map(sns.kdeplot, "Genuine Score", clip_on=True, color="w", lw=2, bw_adjust=bandwidth)
    g.map(sns.kdeplot, "Impostor Score", bw_adjust=bandwidth, clip_on=True, fill=True, alpha=1, linewidth=1.5)
    g.map(sns.kdeplot, "Impostor Score", clip_on=True, color="w", lw=2, bw_adjust=bandwidth)
    g.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=True)

    def label(x, color, label):
        ax = plt.gca()
        ax.text(0, .2, label, fontweight="bold", color=color, ha="left", va="center", transform=ax.transAxes)

    g.map(label, "Impostor Score")
    g.figure.subplots_adjust(hspace=-.5)
    g.set_titles("")
    g.set(xlim=(-.2, 1), yticks=[], ylabel="")
    g.despine(bottom=True, left=True)
    g.figure.suptitle(title)
    g.savefig(outfile, bbox_inches="tight")
    plt.close(g.figure)


def plot_fmr_curves(curves, title, outfile):
    sns.set_theme(style="dark")
    sns.set_palette(sns.cubehelix_palette(5, rot=-.5, light=.7))
    with plt.rc_context(rc={"font.family": "DejaVu Sans Mono"}):
        plt.figure(figsize=(10, 5))
        for name, (thr, fmr) in curves.items():
            plt.plot(thr, fmr, linewidth=2, label=name)
        plt.yscale("log")
        plt.title(title)
        plt.xlabel("Similarity Threshold")
        plt.ylabel("False Match Rate (FMR)")
        plt.xlim(left=-0.22)
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)
        plt.legend(title="Group", loc="upper right")
        plt.tight_layout()
        plt.savefig(outfile)
        plt.close()


def main():
    p = base_parser(__doc__)
    p.add_argument("--threshold", default="global",
                   help="'global' (FMR = 1e-3 over all impostor pairs, as in the captions), "
                        "a fixed cosine threshold, or a JSON file {model: threshold}")
    p.add_argument("--random-seed", type=int, default=0)
    p.add_argument("--n-boot", type=int, default=1000)
    args = p.parse_args()
    out = Path(args.out)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    (out / "tables").mkdir(parents=True, exist_ok=True)

    found = {k: np.array(v) for k, v in json.loads(GROUPS_FILE.read_text()).items()}
    labels = load_rfw_labels(args.data_root, labels="mode")  # plurality-vote labels, as in the paper
    annot = {k: labels.query(q).index.to_numpy() for k, q in ANNOTATION_FILTERS.items()}

    rows_t2, rows_t4, rows_t5 = [], [], []
    for model in args.models:
        ds = load_rfw(model, args.data_root, labels=None)
        E, ids = ds.embeddings, ds.ids
        if args.threshold == "global":
            T = global_threshold(E, ids, 1e-3)
        elif args.threshold.endswith(".json"):  # {model: threshold}, e.g. from the Domino pair set
            T = float(json.loads(Path(args.threshold).read_text())[model])
        else:
            T = float(args.threshold)
        print(f"{model}: threshold {T:.4f}")
        groups = {"R": random_group(ids, seed=args.random_seed), **found}

        plot_df, curves = [], {}
        for name, g in groups.items():
            imp, gen = group_pair_scores(E, ids, g)
            plot_df += [pd.DataFrame({"Genuine Score": gen, "Label": name}),
                        pd.DataFrame({"Impostor Score": imp, "Label": name})]
            curves[name] = fmr_curve(imp)
            print(f"{model} {name}: imp μ={imp.mean():.3f} σ={imp.std():.2f} gen μ={gen.mean():.3f} "
                  f"σ={gen.std():.2f} #imp={imp.size} #ids={np.unique(ids[g]).size} #img={len(g)}")
            if name != "R":
                rows_t4.append({"model": model, "group": name, **eer_and_fnmr(gen, imp)})
            m, lo, hi = bootstrap_group_fmr(E, ids, g, T, n_boot=args.n_boot, seed=args.random_seed)
            rows_t5.append({"model": model, "group": name, "fmr": float(np.mean(imp >= T)),
                            "boot_mean": m, "ci_low": lo, "ci_high": hi, "pm": (hi - lo) / 2})
        ridge_plot(pd.concat(plot_df, ignore_index=True), MODEL_TITLES[model],
                   out / "figures" / f"score_distributions_{model}.pdf")
        plot_fmr_curves(curves, f"FMR vs Threshold: {MODEL_TITLES[model]}", out / "figures" / f"fmr_curves_{model}.pdf")

        for name in "ABCD":
            imp_a, _ = group_pair_scores(E, ids, annot[name])
            imp_o, _ = group_pair_scores(E, ids, found[name])
            fa, fo = np.mean(imp_a >= T), np.mean(imp_o >= T)
            rows_t2.append({"model": model, "group": name, "annot": fa, "ours": fo, "ratio": fo / fa,
                            "annot_images": len(annot[name]), "ours_images": len(found[name]),
                            "ours_impostor_pairs": imp_o.size})

    t2, t4, t5 = pd.DataFrame(rows_t2), pd.DataFrame(rows_t4), pd.DataFrame(rows_t5)
    t2.to_csv(out / "tables" / "table2_annotations_vs_ours.csv", index=False)
    t4.to_csv(out / "tables" / "supp_table4_eer_fnmr.csv", index=False)
    t5.to_csv(out / "tables" / "supp_table5_fmr_bootstrap.csv", index=False)
    print("\nTable 2\n", t2.to_string(index=False, float_format="%.3f"))
    print("\nSupp. Table 4\n", t4.to_string(index=False, float_format="%.2e"))
    print("\nSupp. Table 5\n", t5.to_string(index=False, float_format="%.4f"))


if __name__ == "__main__":
    main()
