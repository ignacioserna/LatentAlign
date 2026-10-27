"""Shared helpers for the experiment scripts."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from latentalign.data import MODELS  # noqa: E402

RESULTS = REPO / "results"

# File names of the result databases (identical to the ones used for the paper).
DB_NAMES = {
    "latentalign": "RFW_{model}_latentalign.db",
    "weighted_svd": "RFW_{model}_weighted-svd.db",
    "nn": "RFW_{model}_nn_kmeans.db",
    "kmeans": "RFW_{model}_kmeans.db",
    "kmeans_all": "RFW_{model}_kmeans_all.db",
    "skmeans": "RFW_{model}_Skmeans.db",
    "dbscan": "RFW_{model}_dbscan.db",
    "graph_init": "RFW_{model}.db",
    "random_init": "RFW_random_{model}.db",
}


def db_path(method: str, model: str, db_dir=None) -> Path:
    return Path(db_dir or RESULTS / "db") / DB_NAMES[method].format(model=model)


def base_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data-root", default=None, help="Root of the datasets (default: $LATENTALIGN_DATA or ./data)")
    p.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    p.add_argument("--out", default=str(RESULTS), help="Output directory")
    p.add_argument("--n-jobs", type=int, default=-1)
    return p
