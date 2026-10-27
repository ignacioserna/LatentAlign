"""Stage 06c - turn a LatentAlign (or k-means / NN / DBSCAN) SQLite DB into a
groups.csv that compare_fmr.py can read.

Your DBs store `indices` as positions into `x_names.npy` / `embd_<model>.npy`.
compare_fmr.py works in the `image_id`s of the Domino manifest. This translates, using the same
basename / identity-qualified matching `import_embeddings.py` uses, and writes

    group_id,image_id,label

which is exactly what `--groups NAME=PATH` wants.

    python -m rfw_domino.lfa_groups --work work \\
        --db ../results/db/RFW_arcface_latentalign.db --model arcface --size 100

    python -m rfw_domino.compare_fmr --work work --group-size 100 \\
        --groups latentalign=work/lfa_groups.csv

Pass several --db to convert your clustering baselines at the same time; the method
name is taken from the filename (RFW_<model>_<method>.db) and each gets its own CSV.

Group sizes are selected by **bin**, the same way compute_summary_binned does it:
`pd.cut(size, bins=arange(0, max+w, w), right=False)`, so `--size 100 --bin 10`
keeps the [100, 110) bin. Each kept group is then subsampled down to exactly --size
images so every group in the CSV has the same number of within-group impostor pairs
and compare_fmr.py's --group-size is a no-op on it. `--bin 1` for exact sizes,
`--size 0` to keep everything, `--no-truncate` to leave native sizes alone.
"""

import argparse
import json
import os
import re
import sqlite3

import numpy as np
import pandas as pd

from .io_utils import Work


def _basename(p):
    return os.path.basename(str(p).replace("\\", "/"))


def _id_key(p):
    return "/".join(str(p).replace("\\", "/").rstrip("/").split("/")[-2:])


def build_reverse_map(images, names_npy):
    """row of x_names.npy -> image_id of the Domino manifest."""
    names = [str(n) for n in np.asarray(np.load(names_npy, allow_pickle=True)).ravel()]
    by_base, dupes, by_idkey = {}, set(), {}
    for i, n in enumerate(names):
        b = _basename(n)
        if b in by_base:
            dupes.add(b)
        by_base[b] = i
        by_idkey[_id_key(n)] = i

    rev = {}
    for row in images.itertuples():
        b = _basename(row.path)
        idx = (by_idkey.get(_id_key(row.path)) if (b in dupes or b not in by_base)
               else by_base[b])
        if idx is not None:
            rev[int(idx)] = int(row.image_id)
    print(f"[lfa] {len(rev):,} of {len(names):,} rows in "
          f"{os.path.basename(names_npy)} map onto the manifest "
          f"({len(images):,} images)")
    if len(rev) < 0.5 * len(images):
        raise SystemExit(
            "fewer than half the manifest images were found in x_names.npy - "
            "check that --names is the file whose row order the DB's indices refer "
            "to (the same one you passed to import_embeddings).")
    return rev


def read_db(db_path, model=None):
    """Load the results table, whatever extra columns this DB variant has."""
    conn = sqlite3.connect(db_path)
    try:
        if model:
            df = pd.read_sql_query("SELECT * FROM results WHERE model = ?", conn,
                                   params=(model,))
            if df.empty:
                have = pd.read_sql_query(
                    "SELECT DISTINCT model FROM results", conn)["model"].tolist()
                raise SystemExit(f"no rows for model {model!r} in {db_path}; "
                                 f"the DB has {have}")
        else:
            df = pd.read_sql_query("SELECT * FROM results", conn)
    finally:
        conn.close()
    for c in ("indices", "group_id"):
        if c not in df.columns:
            raise SystemExit(f"{db_path}: results table has no {c!r} column "
                             f"(got {list(df.columns)})")
    df["indices"] = df["indices"].apply(
        lambda s: json.loads(s) if isinstance(s, str) else s)
    if "size" not in df.columns:
        df["size"] = df["indices"].apply(len)
    return df


def method_name(db_path, model=None):
    stem = os.path.splitext(os.path.basename(db_path))[0]
    m = re.match(r"RFW[_-](.+?)[_-](.+)$", stem)
    if m and (model is None or m.group(1) == model):
        return m.group(2)
    return stem


def _size_hist(sizes, width=10, top=12):
    lo = (np.asarray(sizes) // width) * width
    vc = pd.Series(lo).value_counts().sort_index()
    return ", ".join(f"[{int(b)},{int(b) + width}): {n}" for b, n in
                     vc.head(top).items()) + (" ..." if len(vc) > top else "")


def select_by_size(df, size, bin_width):
    """The bin containing `size`, matching compute_summary_binned's pd.cut."""
    if not size:
        print(f"[lfa]   {len(df)} groups, sizes {df['size'].min()}-"
              f"{df['size'].max()}")
        return df
    w = max(1, int(bin_width))
    lo = (int(size) // w) * w
    hi = lo + w
    sel = df[(df["size"] >= lo) & (df["size"] < hi)]
    if sel.empty:
        raise SystemExit(
            f"no groups in the size bin [{lo},{hi}). This DB's sizes: "
            f"{_size_hist(df['size'].to_numpy(), w)}\n"
            f"Widen with --bin, move --size, or pass --size 0 to keep every size.")
    print(f"[lfa]   {len(sel)}/{len(df)} groups in size bin [{lo},{hi}) "
          f"(sizes {sel['size'].min()}-{sel['size'].max()})")
    return sel


def convert(df, rev, size=None, bin_width=10, min_mapped=0.95, truncate=True,
            seed=0, label=None):
    df = select_by_size(df, size, bin_width)
    rng = np.random.default_rng(seed)

    rows, dropped, short, lost_total = [], 0, 0, 0
    for out_gid, (_, r) in enumerate(df.iterrows()):
        ix = np.asarray(r["indices"], dtype=int)
        keep = [m for m in (rev.get(int(i)) for i in ix) if m is not None]
        if len(keep) < min_mapped * len(ix):
            dropped += 1
            continue
        lost_total += len(ix) - len(keep)

        if truncate and size:
            if len(keep) < size:
                short += 1
                continue
            # uniform subsample: a group is an unordered set, so this is an
            # unbiased subsample of its impostor pairs, not a biased prefix
            keep = list(rng.choice(np.asarray(keep), size=size, replace=False))

        lab = label
        if lab is None:
            bits = [f"{c}={r[c]}" for c in ("k", "n_neighbors", "eps", "min_samples")
                    if c in df.columns and pd.notna(r.get(c))]
            lab = " ".join(bits)
        for img in keep:
            rows.append({"group_id": out_gid, "image_id": int(img), "label": lab})

    if dropped:
        print(f"[lfa]   dropped {dropped} groups that lost more than "
              f"{1 - min_mapped:.0%} of their images in the mapping")
    if short:
        print(f"[lfa]   dropped {short} groups left with fewer than {size} mappable "
              f"images (use --no-truncate to keep them at their native size)")
    if lost_total:
        print(f"[lfa]   {lost_total} individual images were not in the manifest "
              f"(RFW images your pair protocol never references)")
    return pd.DataFrame(rows, columns=["group_id", "image_id", "label"])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", default="work")
    ap.add_argument("--db", nargs="+", required=True,
                    help="one or more SQLite DBs, e.g. "
                         "../results/db/RFW_arcface_latentalign.db")
    ap.add_argument("--model", default=None,
                    help="value of the `model` column to keep (default: all rows)")
    ap.add_argument("--names", default="../data/RFW/embeddings/x_names.npy",
                    help="the x_names.npy the DB's indices refer to")
    ap.add_argument("--size", type=int, default=100,
                    help="target group size; groups are selected by the size BIN "
                         "containing it (see --bin), then subsampled to exactly "
                         "this many images. 0 keeps every size and skips truncation.")
    ap.add_argument("--bin", type=int, default=10,
                    help="size bin width, matching compute_summary_binned: "
                         "--size 100 --bin 10 keeps the [100,110) bin. Use 1 for "
                         "exact sizes.")
    ap.add_argument("--truncate", dest="truncate", action="store_true",
                    help="subsampling each group to --size instead of at its "
                         "native size")
    ap.set_defaults(truncate=False)
    ap.add_argument("--min-mapped", type=float, default=0.95,
                    help="drop a group if fewer than this fraction of its images "
                         "map onto the manifest")
    ap.add_argument("--out", default=None,
                    help="output CSV. Only valid with a single --db; otherwise one "
                         "CSV per DB is written next to --out-dir")
    ap.add_argument("--seed", type=int, default=0,
                    help="seed for the subsample when a group is larger than --size")
    ap.add_argument("--out-dir", default=None,
                    help="directory for the per-DB CSVs (default: <work>)")
    args = ap.parse_args(argv)

    if args.out and len(args.db) > 1:
        raise SystemExit("--out takes a single --db; use --out-dir for several")

    work = Work(args.work)
    images = work.load_images()
    rev = build_reverse_map(images, args.names)
    size = None if args.size in (0, None) else args.size
    out_dir = args.out_dir or work.dir
    os.makedirs(out_dir, exist_ok=True)

    written = []
    for db in args.db:
        name = method_name(db, args.model)
        print(f"[lfa] {os.path.basename(db)} -> method {name!r}")
        df = read_db(db, args.model)
        out = convert(df, rev, size, args.bin, args.min_mapped, args.truncate,
                      args.seed)
        if out.empty:
            print(f"[lfa]   nothing to write for {name}")
            continue
        path = args.out or os.path.join(
            out_dir, "lfa_groups.csv" if name == "latentalign"
            else f"{name}_groups.csv")
        out.to_csv(path, index=False)
        n_g = out["group_id"].nunique()
        gs = out.groupby("group_id").size()
        shape = (f"{gs.iloc[0]}" if gs.nunique() == 1
                 else f"{gs.min()}-{gs.max()}")
        print(f"[out] {n_g} groups x {shape} images -> {path}")
        written.append((name, path))

    if written:
        flags = " ".join(f"--groups {n}={p}" for n, p in written)
        print(f"\n[lfa] now run:\n"
              f"    python -m rfw_domino.compare_fmr --work {work.dir} "
              f"--group-size {size or 100} {flags}")


if __name__ == "__main__":
    main()