"""Shared I/O helpers for the pipeline's cached intermediates."""

import json
import os

import numpy as np
import pandas as pd

try:  # tqdm is optional everywhere in this repo
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(x, **kwargs):
        return x


class Work:
    """Paths of cached intermediates inside the working directory."""

    def __init__(self, work_dir: str):
        self.dir = os.path.abspath(work_dir)
        os.makedirs(self.dir, exist_ok=True)

    def path(self, name: str) -> str:
        return os.path.join(self.dir, name)

    # ---- stage 01
    @property
    def pairs_csv(self):
        return self.path("pairs.csv")

    @property
    def images_csv(self):
        return self.path("images.csv")

    # ---- stage 02
    @property
    def face_embs_npy(self):
        return self.path("face_embs.npy")

    @property
    def face_meta_csv(self):
        return self.path("face_meta.csv")

    # ---- stage 03
    @property
    def pairs_scored_csv(self):
        return self.path("pairs_scored.csv")

    @property
    def metrics_json(self):
        return self.path("metrics.json")

    # ---- stage 04
    @property
    def clip_embs_npy(self):
        return self.path("clip_embs.npy")

    @property
    def clip_meta_json(self):
        return self.path("clip_meta.json")

    @property
    def text_embs_npy(self):
        return self.path("text_embs.npy")

    @property
    def phrases_txt(self):
        return self.path("phrases.txt")

    # ---- stage 05
    @property
    def slice_probs_npy(self):
        return self.path("slice_probs.npy")

    @property
    def slices_assign_csv(self):
        return self.path("slices_assign.csv")

    @property
    def slicer_info_json(self):
        return self.path("slicer_info.json")

    # ---- stage 06
    @property
    def report_dir(self):
        d = self.path("report")
        os.makedirs(d, exist_ok=True)
        return d

    # ---- loaders -------------------------------------------------------
    def load_images(self) -> pd.DataFrame:
        df = pd.read_csv(self.images_csv)
        return df

    def load_pairs(self, scored: bool = False) -> pd.DataFrame:
        path = self.pairs_scored_csv if scored else self.pairs_csv
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} not found - run the earlier pipeline stages first."
            )
        return pd.read_csv(path)

    def load_embs(self, which: str) -> np.ndarray:
        path = {"face": self.face_embs_npy, "clip": self.clip_embs_npy}[which]
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} not found - run the embedding stage first."
            )
        return np.load(path)


def save_json(path: str, obj) -> None:
    def _default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(f"not JSON serializable: {type(o)}")

    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_default)


def l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=axis, keepdims=True), eps)
