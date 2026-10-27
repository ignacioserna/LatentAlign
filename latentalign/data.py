"""Data loading for RFW and CelebA.

Expected layout (see ``data/README.md``)::

    <DATA_ROOT>/
      RFW/
        data/<Race>/<identity>/<image>.jpg          # only needed for extraction / figures
        embeddings/embd_{arcface,cosface,elasticface,partialFC}.npy
        embeddings/x_names.npy                      # image paths, same order as embeddings
        embeddings/x_ids.npy                        # identity labels
        embeddings/arc2face/RFW_arcface.npy        # antelopev2 embeddings (traversals)
      CelebA/
        img_align_celeba/*.jpg
        embeddings/embd_{model}_r100.npy
        embeddings/names.npy
        identity_CelebA.txt
        label_cleaned_mso.csv                       # 40 binary attributes (+ image_id)
        embeddings/arc2face/CelebA_arcface.npy

``DATA_ROOT`` defaults to ``./data`` and can be overridden with the
``LATENTALIGN_DATA`` environment variable or the ``--data-root`` option of every
script. The RFW attribute annotations produced with the VLMs are shipped with
the repository in ``data/annotations/rfw``.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS = ["arcface", "cosface", "elasticface", "partialFC"]
MODEL_TITLES = {"arcface": "ArcFace", "cosface": "CosFace", "elasticface": "ElasticFace", "partialFC": "PartialFC"}
RFW_ATTRIBUTES = [
    "Gender", "Age", "Skin Color", "Ancestry", "Hair Color",
    "Bangs", "Bald", "Beard", "Glasses", "Headwear",
]


def data_root(root: Optional[str] = None) -> Path:
    return Path(root or os.environ.get("LATENTALIGN_DATA", REPO_ROOT / "data"))


def l2_normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


@dataclass
class FaceDataset:
    name: str
    model: str
    embeddings: np.ndarray
    names: np.ndarray
    ids: np.ndarray
    attributes: Optional[np.ndarray] = None   # (N, F) float, NaN rows = no annotation
    attribute_scale: float = 1.0              # 2 for one-hot (RFW), 1 for binary (CelebA)
    extra: dict = field(default_factory=dict)

    def __len__(self):
        return len(self.ids)


# ----------------------------------------------------------------------------
# RFW
# ----------------------------------------------------------------------------
def rfw_relative_name(path: str) -> str:
    """``.../RFW/data/African/m.x/m.x_0001.jpg`` -> ``African/m.x/m.x_0001.jpg``."""
    parts = str(path).replace("\\", "/").split("/")
    return "/".join(parts[-3:])


def rfw_attribute_matrix(names: np.ndarray, labels_csv: Path) -> np.ndarray:
    """One-hot encoded VLM attributes aligned with ``names``.

    The encoder is fitted on all annotated images that are present in
    ``names``; the categories therefore include ``unknown`` (as in the paper).
    """
    from sklearn.preprocessing import OneHotEncoder

    df = pd.read_csv(labels_csv)
    rel = np.array([rfw_relative_name(n) for n in names])
    df = df[df["File Name"].isin(set(rel))].set_index("File Name")
    df = df.loc[[r for r in rel if r in df.index]]
    try:
        enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    except TypeError:  # scikit-learn < 1.2
        enc = OneHotEncoder(sparse=False, handle_unknown="ignore")
    enc_values = enc.fit_transform(df[RFW_ATTRIBUTES].astype(str))
    X = np.full((len(rel), enc_values.shape[1]), np.nan)
    pos = {r: i for i, r in enumerate(rel)}
    X[[pos[r] for r in df.index]] = enc_values
    return X


def load_rfw(
    model: str,
    root: Optional[str] = None,
    labels: str = "mode",
    normalize: bool = True,
) -> FaceDataset:
    """Load RFW embeddings of ``model`` and (optionally) the attribute matrix.

    ``labels``: ``"mode"`` (plurality vote; used by every experiment of the
    paper: Tab. 1, Tab. 2, Figs. 4-5), ``"majority"`` (strict majority vote;
    used for the annotation statistics and the human validation, Supp. Tabs. 2-3)
    or ``None``.
    """
    r = data_root(root) / "RFW" / "embeddings"
    E = np.load(r / f"embd_{model}.npy").squeeze()
    E = l2_normalize(E) if normalize else E
    names = np.load(r / "x_names.npy")
    ids = np.load(r / "x_ids.npy")
    X = None
    if labels:
        csv = REPO_ROOT / "data" / "annotations" / "rfw" / (
            "labels_RFW_mode.csv" if labels == "mode" else "labels_RFW.csv"
        )
        X = rfw_attribute_matrix(names, csv)
    return FaceDataset("RFW", model, E, names, ids, X, attribute_scale=2.0)


def load_rfw_labels(root: Optional[str] = None, labels: str = "majority", names=None) -> pd.DataFrame:
    """VLM labels aligned with the embedding order (one row per image)."""
    csv = REPO_ROOT / "data" / "annotations" / "rfw" / (
        "labels_RFW_mode.csv" if labels == "mode" else "labels_RFW.csv"
    )
    if names is None:
        names = np.load(data_root(root) / "RFW" / "embeddings" / "x_names.npy")
    rel = [rfw_relative_name(n) for n in names]
    df = pd.read_csv(csv).set_index("File Name")
    return df.reindex(rel).reset_index()


# ----------------------------------------------------------------------------
# CelebA
# ----------------------------------------------------------------------------
def celeba_ids(names: np.ndarray, identity_file: Path) -> np.ndarray:
    ids_df = pd.read_csv(identity_file, sep=r"\s+", header=None, names=["image_id", "id"])
    m = dict(zip(ids_df["image_id"], ids_df["id"]))
    return np.array([m.get(os.path.basename(str(n)), -1) for n in names])


def celeba_attribute_matrix(names: np.ndarray, labels_csv: Path) -> np.ndarray:
    """Binary (0/1) CelebA attributes aligned with ``names`` (-1 mapped to 0)."""
    df = pd.read_csv(labels_csv).set_index("image_id").replace(-1, 0)
    base = [os.path.basename(str(n)) for n in names]
    return df.reindex(base).to_numpy(dtype=float)


def load_celeba(model: str, root: Optional[str] = None, attributes: bool = True, normalize: bool = True) -> FaceDataset:
    r = data_root(root) / "CelebA"
    E = np.load(r / "embeddings" / f"embd_{model}_r100.npy").squeeze()
    E = l2_normalize(E) if normalize else E
    names = np.load(r / "embeddings" / "names.npy")
    ids = celeba_ids(names, r / "identity_CelebA.txt")
    X = celeba_attribute_matrix(names, r / "label_cleaned_mso.csv") if attributes else None
    return FaceDataset("CelebA", model, E, names, ids, X, attribute_scale=1.0)


def load_dataset(dataset: str, model: str, root: Optional[str] = None, **kw) -> FaceDataset:
    if dataset.lower() == "rfw":
        return load_rfw(model, root, **kw)
    if dataset.lower() == "celeba":
        return load_celeba(model, root, **kw)
    raise ValueError(dataset)
