"""Check that the memory-efficient implementations match the original notebook code.

    python -m pytest tests/test_equivalence.py -q
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from latentalign.algorithm import get_direction, grow_fixed_size, grow_threshold, grow_threshold_batched  # noqa: E402
from latentalign.metrics import _pair_block_stats, global_threshold, group_fmr, group_pair_scores, semantic_distance  # noqa: E402

rng = np.random.default_rng(0)
N, D = 1500, 32
IDS = rng.integers(0, 300, size=N).astype(str)
E = rng.normal(size=(N, D)).astype(np.float32)
E[:, :4] += rng.normal(size=(300, 4))[IDS.astype(int)] * 2
E /= np.linalg.norm(E, axis=1, keepdims=True)


# --- original implementations (copied from the notebooks) --------------------
def get_impostors(id_label):
    comparison_matrix = (id_label[:, None] == id_label[None, :]).astype(int)
    genuines_matrix = comparison_matrix.copy()
    genuines_matrix[np.tril_indices(genuines_matrix.shape[0])] = 0
    impostors_matrix = 1 - comparison_matrix
    impostors_matrix[np.tril_indices(impostors_matrix.shape[0])] = 0
    return impostors_matrix, genuines_matrix


def grow_orig(seed, threshold, embeddings, ids):
    elements = seed
    for _ in range(1000):
        v = get_direction(ids[elements], embeddings[elements])
        proj = embeddings @ v
        for i in np.argsort(proj)[::-1]:
            if i not in elements:
                if proj[i] < threshold:
                    return elements
                elements = elements + [i]
                break
    return elements


def grow_n_orig(seed, n, embeddings, ids):
    elements = seed
    for _ in range(n):
        v = get_direction(ids[elements], embeddings[elements])
        proj = embeddings @ v
        for i in np.argsort(proj)[::-1]:
            if i not in elements:
                elements = elements + [i]
                break
    return elements


SCORES = E @ E.T
IMP, GEN = get_impostors(IDS)


def test_global_threshold():
    all_imp = SCORES[IMP == 1]
    ref = np.quantile(all_imp, 1 - 1e-3, method="higher")
    assert np.isclose(global_threshold(E, IDS, 1e-3, block=128), ref)


def test_group_scores_and_fmr():
    T = 0.3
    for _ in range(20):
        g = list(rng.choice(N, size=80, replace=True))  # with duplicates, as in the bootstrap
        A, B, C = SCORES[g][:, g], IMP[g][:, g], GEN[g][:, g]
        imp, gen = group_pair_scores(E, IDS, g)
        assert np.allclose(np.sort(A[B == 1]), np.sort(imp), atol=1e-6)
        assert np.allclose(np.sort(A[C == 1]), np.sort(gen), atol=1e-6)
        assert np.isclose(np.mean(A[B == 1] >= T), group_fmr(E, IDS, g, T))


def test_streaming_stats_and_semantic_distance():
    from scipy.spatial.distance import pdist

    for _ in range(5):
        g = list(rng.choice(N, size=700, replace=True))
        imp, _ = group_pair_scores(E, IDS, g)
        n, total, hits = _pair_block_stats(E, IDS, g, threshold=0.3, block=97)
        assert n == imp.size and np.isclose(total, imp.sum(), rtol=1e-4) and hits == int((imp >= 0.3).sum())
        X = rng.integers(0, 2, size=(N, 20)).astype(float)
        assert np.isclose(semantic_distance(X, g, 2.0), np.mean(pdist(X[g], "cityblock")) / 2)


def test_grow():
    for s in rng.choice(N, size=10, replace=False):
        assert grow_threshold([int(s)], E, IDS, tau=0.35) == [int(x) for x in grow_orig([int(s)], 0.35, E, IDS)]
        assert grow_fixed_size([int(s)], E, IDS, n=30) == [int(x) for x in grow_n_orig([int(s)], 30, E, IDS)]


def test_batched():
    seeds = [[int(s)] for s in rng.choice(N, size=12, replace=False)]
    assert grow_threshold_batched(seeds, E, IDS, tau=0.35, batch=5) == [grow_threshold(s, E, IDS, tau=0.35) for s in seeds]
    assert grow_threshold_batched(seeds, E, IDS, tau=-np.inf, n_max=20) == [grow_fixed_size(s, E, IDS, n=20) for s in seeds]


if __name__ == "__main__":
    test_global_threshold()
    test_group_scores_and_fmr()
    test_streaming_stats_and_semantic_distance()
    test_grow()
    test_batched()
    print("all equivalence checks passed")
