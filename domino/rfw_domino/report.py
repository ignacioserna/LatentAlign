"""Stage 06 - per-slice report: statistics, race composition, pair montages,
and (when CLIP text embeddings are available) natural-language descriptions.

Outputs into <work>/report/:
    report.md            - the full report
    slices_summary.csv   - one row per discovered slice
    slice_<k>.png        - montage of the top pairs of slice k
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

from .io_utils import Work

def wilson(k, n, z=1.96):
    """Wilson score interval. The normal approximation is useless at k < 10."""
    import math
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


# ------------------------------------------------------------------ montage

def pair_tile(path1, path2, correct, caption, thumb=112):
    from PIL import Image, ImageDraw, ImageOps

    border = 4
    cap_h = 18
    color = (60, 179, 113) if correct else (220, 60, 60)

    def load(p):
        img = Image.open(p).convert("RGB")
        return ImageOps.fit(img, (thumb, thumb))

    w = thumb * 2 + border * 3
    h = thumb + border * 2 + cap_h
    tile = Image.new("RGB", (w, h), color)
    tile.paste(load(path1), (border, border))
    tile.paste(load(path2), (thumb + border * 2, border))
    strip = Image.new("RGB", (w - 2 * border, cap_h - 2), (25, 25, 25))
    tile.paste(strip, (border, thumb + border + 1))
    draw = ImageDraw.Draw(tile)
    draw.text((border + 3, thumb + border + 2), caption, fill=(235, 235, 235))
    return tile


def slice_montage(rows, out_path, n_cols=3, n_rows=4, thumb=112):
    from PIL import Image

    tiles = []
    for r in rows[: n_cols * n_rows]:
        cap = (f"sim {r['sim']:+.2f}  "
               f"{'same' if r['same'] else 'diff'}->"
               f"{'same' if r['pred'] else 'diff'}  {r['race'][:3]}")
        tiles.append(pair_tile(r["path1"], r["path2"], bool(r["correct"]), cap,
                               thumb))
    if not tiles:
        return False
    tw, th = tiles[0].size
    pad = 6
    grid = Image.new(
        "RGB",
        (n_cols * tw + (n_cols + 1) * pad,
         ((len(tiles) - 1) // n_cols + 1) * (th + pad) + pad),
        (245, 245, 245),
    )
    for i, t in enumerate(tiles):
        r, c = divmod(i, n_cols)
        grid.paste(t, (pad + c * (tw + pad), pad + r * (th + pad)))
    grid.save(out_path)
    return True


# --------------------------------------------------------------------- main

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--work", default="work")
    ap.add_argument("--top-k", type=int, default=12,
                    help="pairs shown per slice montage")
    ap.add_argument("--no-montages", action="store_true")
    args = ap.parse_args(argv)

    work = Work(args.work)
    pairs = work.load_pairs(scored=True)
    assign = pd.read_csv(work.slices_assign_csv)
    slice_probs = np.load(work.slice_probs_npy)
    with open(work.slicer_info_json) as f:
        info = json.load(f)
    with open(work.metrics_json) as f:
        metrics = json.load(f)

    df = pairs.merge(
        assign[["pair_id", "slice_top", "slice_top_prob"]], on="pair_id"
    )
    
    if (len(df) == len(assign)
            and not (df["pair_id"].to_numpy()
                     == assign["pair_id"].to_numpy()).all()):
        raise RuntimeError(
            "the merged pairs are not in the same row order as the slice "
            "assignment, so slice_probs would be silently misaligned."
        )

    if len(df) != len(slice_probs):
        raise RuntimeError(
            f"{len(df)} pairs match the slice assignment but slice_probs has "
            f"{len(slice_probs)} rows - rerun stages 05/06 with the same "
            f"--subset."
        )
    n_slices = slice_probs.shape[1]
    overall_err = 1.0 - df["correct"].mean()
    race_base = df["race"].value_counts(normalize=True)

    # ---- optional describe step
    descriptions = None
    if os.path.exists(work.text_embs_npy) and os.path.exists(work.phrases_txt):
        from .describe import describe_slices

        clip_embs = work.load_embs("clip")
        text_embs = np.load(work.text_embs_npy)
        with open(work.phrases_txt) as f:
            phrases = [ln.strip() for ln in f if ln.strip()]
        pair_img_ids = df[["img_id1", "img_id2"]].to_numpy()
        descriptions = describe_slices(
            slice_probs, pair_img_ids, clip_embs, text_embs, phrases
        )

    # ---- per-slice stats
    summary_rows = []
    montage_ok = {}
    for k in range(n_slices):
        members = df[df["slice_top"] == k]
        n = len(members)
        err = 1.0 - members["correct"].mean() if n else float("nan")
        errors = members[members["correct"] == 0]
        # NaN, not 0.0: a slice with no errors has no false-match share, and
        # 0.00 is indistinguishable from "all errors are false non-matches"
        fm = ((errors["same"] == 0).mean() if len(errors) else np.nan)
        comp = members["race"].value_counts(normalize=True) if n else race_base
        top_race = comp.idxmax() if n else "-"
        summary_rows.append({
            "slice": k,
            "n_pairs": n,
            "n_errors": int(len(errors)),
            "error_rate": err,
            "error_lift": (err / overall_err) if overall_err > 0 and n else np.nan,
            "false_match_share_of_errors": fm,
            "share_positive_pairs": members["same"].mean() if n else np.nan,
            "top_race": top_race,
            "top_race_share": comp.max() if n else np.nan,
            "top_race_lift": (comp.max() / race_base.get(top_race, np.nan))
            if n else np.nan,
            "mean_sim": members["sim"].mean() if n else np.nan,
        })

        if not args.no_montages:
            top = df.iloc[np.argsort(-slice_probs[:, k])[: args.top_k]]
            out_png = os.path.join(work.report_dir, f"slice_{k}.png")
            montage_ok[k] = slice_montage(
                top.to_dict("records"), out_png
            )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(os.path.join(work.report_dir, "slices_summary.csv"),
                   index=False)

    # ---- markdown report
    lines = []
    lines.append("# Domino slice discovery on RFW - report\n")
    lines.append(f"- pairs: **{len(df)}**, overall error rate at the global "
                 f"threshold: **{overall_err:.3%}** "
                 f"(threshold {metrics['global_threshold']:.4f}, "
                 f"criterion {metrics['criterion']})")
    lines.append(f"- Domino: impl `{info['impl']}`, pair embedding "
                 f"`{info['pair_embed']}`, subset `{info.get('subset', 'all')}` "
                 f"({len(df)} pairs sliced), params `{info['params']}`\n")

    lines.append("## Verification metrics per race\n")
    lines.append("| race | n | acc@global | FNMR | FMR | 10-fold acc |")
    lines.append("|---|---|---|---|---|---|")
    for race, m in metrics["per_race"].items():
        lines.append(
            f"| {race} | {m['n']} | {m['accuracy_at_global_thr']:.4f} "
            f"| {m['fnmr_at_global_thr']:.4f} | {m['fmr_at_global_thr']:.4f} "
            f"| {m['tenfold_accuracy_mean']:.4f} "
            f"± {m['tenfold_accuracy_std']:.3f} |"
        )
    lines.append("")

    lines.append("## Discovered slices\n")
    lines.append("Slices are ordered by Domino's ranking (components whose "
                 "prediction distribution deviates most from their label "
                 "distribution). `error lift` is the slice error rate divided "
                 "by the overall error rate; `FM share` is the fraction of the "
                 "slice's errors that are false matches (an impostor pair "
                 "accepted).\n")
    lines.append("| slice | n | errors | error rate | error lift "
                 "| 95% CI on lift | FM share | pos share "
                 "| top race (share, lift) | mean sim |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in summary_rows:
        lo, hi = wilson(r["n_errors"], r["n_pairs"])
        ci = (f"[{lo/overall_err:.1f}, {hi/overall_err:.1f}]"
              if overall_err > 0 else "-")
        fmv = r["false_match_share_of_errors"]
        fms = "-" if not np.isfinite(fmv) else f"{fmv:.2f}"
        lines.append(
            f"| {r['slice']} | {r['n_pairs']} | {r['n_errors']} "
            f"| {r['error_rate']:.5f} | {r['error_lift']:.1f}x | {ci} | {fms} "
            f"| {r['share_positive_pairs']:.2f} "
            f"| {r['top_race']} ({r['top_race_share']:.2f}, "
            f"{r['top_race_lift']:.1f}x) | {r['mean_sim']:+.3f} |"
        )
    lines.append("")

    for k in range(n_slices):
        r = summary_rows[k]
        lines.append(f"### Slice {k}\n")
        members = df[df["slice_top"] == k]
        lines.append(f"- {r['n_pairs']} pairs, {r['n_errors']} errors, "
                     f"error rate {r['error_rate']:.5f} "
                     f"({r['error_lift']:.1f}x overall)")
        if len(members):
            comp = members["race"].value_counts(normalize=True)
            comp_str = ", ".join(f"{race} {frac:.0%}"
                                 for race, frac in comp.items())
            lines.append(f"- race composition: {comp_str}")
        if descriptions is not None:
            lines.append("- top CLIP descriptions "
                         "(score vs. dataset mean; corr with membership):")
            for d in descriptions[k][:5]:
                lines.append(f"    - `{d['text']}` "
                             f"(score {d['score']:+.4f}, corr {d['corr']:+.3f})")
        if montage_ok.get(k):
            lines.append(f"\n![slice {k}](slice_{k}.png)")
        lines.append("")

    report_path = os.path.join(work.report_dir, "report.md")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"[report] wrote {report_path}")
    print(summary.to_string(index=False,
                            float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
