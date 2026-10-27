"""LatentAlign: discovering intersectional bias via directional alignment in
face recognition embeddings (WACV 2027)."""

from .algorithm import get_direction, grow_fixed_size, grow_many, grow_threshold, grow_threshold_batched
from .seeding import graph_seeds, kmeanspp_seeds, paper_seeds, random_seeds

__version__ = "1.0.0"

__all__ = [
    "get_direction",
    "grow_threshold",
    "grow_fixed_size",
    "grow_many",
    "grow_threshold_batched",
    "kmeanspp_seeds",
    "paper_seeds",
    "graph_seeds",
    "random_seeds",
]
