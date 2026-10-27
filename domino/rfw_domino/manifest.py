"""Stage 01 - parse the RFW test protocol into a pair manifest.

RFW (http://www.whdeng.cn/RFW/testing.html) ships four verification subsets
(African, Asian, Caucasian, Indian). Each has ~3000 identities and 6000 image
pairs given as 10 disjoint splits of 300 positive + 300 negative pairs in an
LFW-style pairs file:

    <name> <n1> <n2>            # positive pair: <name>_<n1>.jpg vs <name>_<n2>.jpg
    <name1> <n1> <name2> <n2>   # negative pair (same race, different identity)

Typical archive layout (both `test/...` and flat layouts are handled; we also
fall back to a recursive search):

    <rfw_root>/test/data/<Race>/<identity>/<identity>_0001.jpg
    <rfw_root>/test/txts/<Race>/<Race>_pairs.txt
"""

import argparse
import os

import pandas as pd

from . import RACES
from .io_utils import Work


# --------------------------------------------------------------------------
# locating files inside the (slightly variable) RFW archive layout
# --------------------------------------------------------------------------

def find_txt(rfw_root: str, race: str, kind: str) -> str:
    """Find e.g. `African_pairs.txt` under the RFW root."""
    fname = f"{race}_{kind}.txt"
    candidates = [
        os.path.join(rfw_root, "test", "txts", race, fname),
        os.path.join(rfw_root, "txts", race, fname),
        os.path.join(rfw_root, "test", "txts", fname),
        os.path.join(rfw_root, "txts", fname),
        os.path.join(rfw_root, race, fname),
        os.path.join(rfw_root, fname),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    for dirpath, _, filenames in os.walk(rfw_root):
        if fname in filenames:
            return os.path.join(dirpath, fname)
    raise FileNotFoundError(
        f"Could not find {fname} anywhere under {rfw_root}. "
        f"Expected an RFW layout like <root>/test/txts/{race}/{fname}."
    )


def find_data_dir(rfw_root: str, race: str) -> str:
    candidates = [
        os.path.join(rfw_root, "test", "data", race),
        os.path.join(rfw_root, "data", race),
        os.path.join(rfw_root, race),
    ]
    for c in candidates:
        if os.path.isdir(c):
            return c
    for dirpath, dirnames, _ in os.walk(rfw_root):
        if race in dirnames and "txts" not in dirpath:
            return os.path.join(dirpath, race)
    raise FileNotFoundError(
        f"Could not find the image folder for race '{race}' under {rfw_root}."
    )


def index_images(data_dir: str) -> dict:
    """filename -> absolute path, for every image under `data_dir`."""
    index = {}
    exts = (".jpg", ".jpeg", ".png", ".bmp")
    for dirpath, _, filenames in os.walk(data_dir):
        for fn in filenames:
            if fn.lower().endswith(exts):
                index[fn] = os.path.join(dirpath, fn)
    return index


def resolve_image(index: dict, name: str, num: int):
    for cand in (
        f"{name}_{num:04d}.jpg",
        f"{name}_{num:04d}.png",
        f"{name}_{num:03d}.jpg",
        f"{name}_{num}.jpg",
        f"{name}_{num:04d}.jpeg",
    ):
        if cand in index:
            return index[cand]
    return None


# --------------------------------------------------------------------------
# pairs parsing
# --------------------------------------------------------------------------

def parse_pairs_file(path: str):
    """Yield (name1, num1, name2, num2, same) tuples from an LFW-style file."""
    pairs = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            toks = line.strip().split()
            if len(toks) == 3:
                name, n1, n2 = toks
                pairs.append((name, int(n1), name, int(n2), 1))
            elif len(toks) == 4:
                n1_name, n1, n2_name, n2 = toks
                pairs.append((n1_name, int(n1), n2_name, int(n2), 0))
            elif len(toks) in (0, 1, 2):
                # blank line or a fold-count header like LFW's "10 300"
                continue
            else:
                raise ValueError(f"Unparseable pairs line in {path!r}: {line!r}")
    return pairs


def build_manifest(rfw_root: str, work: Work, races=None) -> None:
    races = races or RACES
    pair_rows, image_rows = [], []
    img_id_of = {}  # path -> image_id
    missing = []

    def get_img_id(path: str, race: str, identity: str) -> int:
        if path not in img_id_of:
            img_id_of[path] = len(img_id_of)
            image_rows.append(
                {"image_id": img_id_of[path], "path": path,
                 "race": race, "identity": identity}
            )
        return img_id_of[path]

    for race in races:
        pairs_txt = find_txt(rfw_root, race, "pairs")
        data_dir = find_data_dir(rfw_root, race)
        index = index_images(data_dir)
        pairs = parse_pairs_file(pairs_txt)
        if not pairs:
            raise ValueError(f"No pairs parsed from {pairs_txt}")
        fold_size = max(1, len(pairs) // 10)

        n_before = len(pair_rows)
        for i, (name1, n1, name2, n2, same) in enumerate(pairs):
            p1 = resolve_image(index, name1, n1)
            p2 = resolve_image(index, name2, n2)
            if p1 is None or p2 is None:
                missing.append((race, name1, n1, name2, n2))
                continue
            pair_rows.append({
                "pair_id": len(pair_rows),
                "race": race,
                "fold": min(i // fold_size, 9),
                "same": same,
                "name1": name1, "num1": n1,
                "img_id1": get_img_id(p1, race, name1), "path1": p1,
                "name2": name2, "num2": n2,
                "img_id2": get_img_id(p2, race, name2), "path2": p2,
            })
        print(f"[manifest] {race}: {len(pair_rows) - n_before} pairs "
              f"({len(index)} images indexed) from {pairs_txt}")

    if missing:
        frac = len(missing) / max(1, len(missing) + len(pair_rows))
        print(f"[manifest] WARNING: {len(missing)} pairs skipped "
              f"(image files not found), e.g. {missing[:3]}")
        if frac > 0.05:
            raise RuntimeError(
                f"{frac:.1%} of pairs reference missing images - the RFW root "
                f"layout probably doesn't match; inspect the examples above."
            )

    pairs_df = pd.DataFrame(pair_rows)
    images_df = pd.DataFrame(image_rows).sort_values("image_id")
    pairs_df.to_csv(work.pairs_csv, index=False)
    images_df.to_csv(work.images_csv, index=False)
    print(f"[manifest] wrote {len(pairs_df)} pairs -> {work.pairs_csv}")
    print(f"[manifest] wrote {len(images_df)} unique images -> {work.images_csv}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rfw-root", required=True,
                    help="Root of your local RFW test set")
    ap.add_argument("--work", default="work", help="Working directory")
    ap.add_argument("--races", nargs="+", default=None,
                    help=f"Subset of races (default: {RACES})")
    args = ap.parse_args(argv)
    build_manifest(args.rfw_root, Work(args.work), args.races)


if __name__ == "__main__":
    main()
