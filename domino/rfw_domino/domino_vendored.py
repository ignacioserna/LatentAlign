"""Standalone (numpy-only) port of Domino's error-aware mixture model.

This is a faithful port of ``DominoSlicer`` (= ``MixtureSlicer``) from
HazyResearch/domino (Apache-2.0):
    domino/_slice/mixture/__init__.py  and  _generalized.py
which are themselves adapted from scikit-learn's GaussianMixture (BSD-3).
Reference: Eyuboglu et al., "Domino: Discovering Systematic Errors with
Cross-Modal Embeddings", ICLR 2022 (https://arxiv.org/abs/2203.14960).

Changes vs. the original (marked "PORT NOTE" inline):
  * no meerkat / torch / scipy / sklearn dependencies - plain numpy;
  * the paper's "confusion" initialization is implemented here (it is
    documented in the MixtureSlicer docstring and was present in domino
    v0.1.x, but the current `main` branch only ships kmeans/random);
  * PCA and k-means are deterministic numpy re-implementations, so fitted
    parameters are not bitwise-identical to the sklearn-based original -
    the model, E/M updates, hyperparameters, and slice-ranking rule are
    the same.

Model (per example i): slice s ~ Cat(pi);  embeddings z|s ~ N(mu_s, Sigma_s)
(diagonal); label y|s ~ Cat(p_s); prediction yhat|s ~ Cat(phat_s). The
categorical log-likelihood terms are raised to weights gamma / gamma_hat in
the E-step ("soft" predictions are marginalized into the categorical PMF).
After fitting `n_mixture_components` components, the returned slices are the
`n_slices` components whose predicted-class distribution deviates most from
their label distribution:  argsort(-|max_c(phat_s - p_s)|)[:n_slices].
"""

import numpy as np

_EPS = np.finfo(np.float64).eps


# --------------------------------------------------------------------------
# small numpy replacements for scipy/sklearn utilities
# --------------------------------------------------------------------------

def _logsumexp(a: np.ndarray, axis: int = 1) -> np.ndarray:
    m = np.max(a, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    out = np.log(np.sum(np.exp(a - m), axis=axis)) + np.squeeze(m, axis=axis)
    return out


class _PCA:
    """Exact PCA via eigendecomposition of the covariance matrix."""

    def __init__(self, n_components: int):
        self.n_components = n_components

    def fit(self, x: np.ndarray):
        x = np.asarray(x, dtype=np.float64)
        self.mean_ = x.mean(axis=0)
        xc = x - self.mean_
        cov = (xc.T @ xc) / max(1, len(x) - 1)
        evals, evecs = np.linalg.eigh(cov)
        order = np.argsort(evals)[::-1]
        k = min(self.n_components, x.shape[1], len(x))
        self.components_ = evecs[:, order[:k]].T  # (k, d)
        self.explained_variance_ = evals[order[:k]]
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (np.asarray(x, dtype=np.float64) - self.mean_) @ self.components_.T


def _kmeans(x: np.ndarray, k: int, rng: np.random.Generator,
            n_iter: int = 30) -> np.ndarray:
    """k-means++ seeding + Lloyd iterations; returns hard labels."""
    n = len(x)
    centers = np.empty((k, x.shape[1]))
    centers[0] = x[rng.integers(n)]
    d2 = np.sum((x - centers[0]) ** 2, axis=1)
    for j in range(1, k):
        probs = d2 / max(d2.sum(), _EPS)
        centers[j] = x[rng.choice(n, p=probs)]
        d2 = np.minimum(d2, np.sum((x - centers[j]) ** 2, axis=1))
    labels = np.zeros(n, dtype=int)
    for _ in range(n_iter):
        dists = (
            np.sum(x ** 2, axis=1, keepdims=True)
            - 2 * x @ centers.T
            + np.sum(centers ** 2, axis=1)
        )
        new_labels = np.argmin(dists, axis=1)
        if np.array_equal(new_labels, labels) and _ > 0:
            break
        labels = new_labels
        for j in range(k):
            mask = labels == j
            if mask.any():
                centers[j] = x[mask].mean(axis=0)
            else:  # re-seed empty cluster at the farthest point
                centers[j] = x[np.argmax(np.min(dists, axis=1))]
    return labels


# --------------------------------------------------------------------------
# mixture variables (ported from domino _generalized.py)
# --------------------------------------------------------------------------

class CategoricalVariable:
    def __init__(self, name: str, weight: float = 1.0):
        self.name = name
        self.weight = weight
        self.probs = None

    @staticmethod
    def preprocess(value: np.ndarray) -> np.ndarray:
        y = np.asarray(value)
        if np.issubdtype(y.dtype, np.integer):
            n_classes = int(y.max()) + 1
            onehot = np.zeros((len(y), max(n_classes, 2)), dtype=np.float64)
            onehot[np.arange(len(y)), y.astype(int)] = 1.0
            return onehot
        y = y.astype(np.float64)
        if y.ndim == 1:
            y = y[:, None]
        if y.shape[-1] == 1:
            y = np.concatenate([1.0 - y, y], axis=1)
        return y

    def update_parameters(self, value, resp, nk):
        self.probs = (resp.T @ value) / nk[:, None]

    initialize_parameters = update_parameters

    def estimate_log_prob(self, value: np.ndarray) -> np.ndarray:
        return np.log(value @ self.probs.T + _EPS)


class GaussianVariable:
    def __init__(self, name: str, weight: float = 1.0,
                 covariance_type: str = "diag", reg_covar: float = 1e-6):
        if covariance_type != "diag":
            # PORT NOTE: original also supports full/tied/spherical; Domino's
            # default and recommendation is "diag", which is all we port.
            raise ValueError("this port only implements covariance_type='diag'")
        self.name = name
        self.weight = weight
        self.reg_covar = reg_covar

    @staticmethod
    def preprocess(value: np.ndarray) -> np.ndarray:
        return np.asarray(value, dtype=np.float64)

    def update_parameters(self, x, resp, nk):
        means = (resp.T @ x) / nk[:, None]
        avg_x2 = (resp.T @ (x * x)) / nk[:, None]
        avg_x_means = means * (resp.T @ x) / nk[:, None]
        self.means_ = means
        self.covariances_ = avg_x2 - 2 * avg_x_means + means ** 2 + self.reg_covar
        self.precisions_cholesky_ = 1.0 / np.sqrt(self.covariances_)

    initialize_parameters = update_parameters

    def estimate_log_prob(self, x: np.ndarray) -> np.ndarray:
        n_features = x.shape[1]
        prec = self.precisions_cholesky_ ** 2  # (k, d)
        log_det = np.sum(np.log(self.precisions_cholesky_), axis=1)
        log_prob = (
            np.sum(self.means_ ** 2 * prec, axis=1)
            - 2.0 * x @ (self.means_ * prec).T
            + x ** 2 @ prec.T
        )
        return -0.5 * (n_features * np.log(2 * np.pi) + log_prob) + log_det


# --------------------------------------------------------------------------

class DominoMixture:
    """EM for the generalized (error-aware) mixture over a list of variables."""

    def __init__(self, n_components, variables, max_iter=100, tol=1e-5,
                 init_params="confusion", confusion_noise=1e-3,
                 random_state=None, verbose=False):
        self.n_components = n_components
        self.variables = variables
        self.max_iter = max_iter
        self.tol = tol
        self.init_params = init_params
        self.confusion_noise = confusion_noise
        self.rng = np.random.default_rng(random_state)
        self.verbose = verbose

    # ---- initialization ---------------------------------------------------
    def _init_resp(self, values) -> np.ndarray:
        n = len(values[0])
        k = self.n_components
        if self.init_params == "confusion":
            # PORT NOTE: implements the initialization described in the Domino
            # paper / MixtureSlicer docstring: components are assigned
            # round-robin to confusion-matrix cells (y, yhat); an example's
            # initial responsibility is (1 + eps) for components matching its
            # own cell and eps otherwise, eps ~ U(0, confusion_noise].
            y = values[1]
            y_hat = values[2] if len(values) > 2 and values[2] is not None else None
            y_cell = y.argmax(axis=1)
            if y_hat is None:
                cells, n_cells = y_cell, y.shape[1]
            else:
                yh_cell = y_hat.argmax(axis=1)
                n_cells = y.shape[1] * y_hat.shape[1]
                cells = y_cell * y_hat.shape[1] + yh_cell
            # PORT NOTE: when a subset makes y (or y_hat) constant some of
            # the n_cells confusion cells are empty, and every component that
            # round-robins onto an empty cell is seeded from noise alone. Map
            # components only onto the cells that are actually occupied.
            occupied = np.unique(cells)
            if len(occupied) < n_cells:
                comp_cell = occupied[np.arange(k) % len(occupied)]
            else:
                comp_cell = np.arange(k) % n_cells
            resp = self.rng.uniform(0.0, self.confusion_noise, size=(n, k))
            resp += (cells[:, None] == comp_cell[None, :]).astype(np.float64)
            resp /= resp.sum(axis=1, keepdims=True)
        elif self.init_params == "kmeans":
            labels = _kmeans(values[0], k, self.rng)
            resp = np.zeros((n, k))
            resp[np.arange(n), labels] = 1.0
        elif self.init_params == "random":
            resp = self.rng.random((n, k))
            resp /= resp.sum(axis=1, keepdims=True)
        else:
            raise ValueError(f"unknown init_params: {self.init_params!r}")
        return resp

    # ---- E / M steps ------------------------------------------------------
    def _m_step(self, values, resp):
        nk = resp.sum(axis=0) + 10 * _EPS
        for value, variable in zip(values, self.variables):
            variable.update_parameters(value, resp, nk)
        self.weights_ = nk / len(values[0])

    def _weighted_log_prob(self, values) -> np.ndarray:
        log_prob = np.log(self.weights_)[None, :]
        for value, variable in zip(values, self.variables):
            if value is None:
                continue
            log_prob = log_prob + variable.estimate_log_prob(value) * variable.weight
        return log_prob

    def _e_step(self, values):
        wlp = self._weighted_log_prob(values)
        log_norm = _logsumexp(wlp, axis=1)
        log_resp = wlp - log_norm[:, None]
        return float(np.mean(log_norm)), np.exp(log_resp)

    # ---- API --------------------------------------------------------------
    def fit(self, values):
        values = [
            None if v is None else var.preprocess(v)
            for v, var in zip(values, self.variables)
        ]
        resp = self._init_resp(values)
        self._m_step(values, resp)
        lower_bound = -np.inf
        self.converged_ = False
        for it in range(1, self.max_iter + 1):
            prev = lower_bound
            lower_bound, resp = self._e_step(values)
            self._m_step(values, resp)
            if self.verbose and (it % 10 == 0 or it == 1):
                print(f"  [em] iter {it:3d}  lower bound {lower_bound:.6f}")
            if abs(lower_bound - prev) < self.tol:
                self.converged_ = True
                break
        self.n_iter_ = it
        self.lower_bound_ = lower_bound
        return self

    def predict_proba(self, values) -> np.ndarray:
        values = [
            None if v is None else var.preprocess(v)
            for v, var in zip(values, self.variables)
        ]
        _, resp = self._e_step(values)
        return resp


# --------------------------------------------------------------------------

class DominoSlicer:
    """Drop-in equivalent of ``domino.DominoSlicer`` (numpy arrays only).

    Defaults mirror the original class; the Domino blog post's ImageNet demo
    used y/y_hat weights of 40, ~2.4 components per returned slice, PCA 128,
    and confusion initialization.
    """

    def __init__(self, n_slices=5, covariance_type="diag",
                 n_pca_components=128, n_mixture_components=25,
                 y_log_likelihood_weight=1.0, y_hat_log_likelihood_weight=1.0,
                 max_iter=100, init_params="confusion", confusion_noise=1e-3,
                 random_state=None, verbose=False):
        self.config = dict(
            n_slices=n_slices, covariance_type=covariance_type,
            n_pca_components=n_pca_components,
            n_mixture_components=n_mixture_components,
            y_log_likelihood_weight=y_log_likelihood_weight,
            y_hat_log_likelihood_weight=y_hat_log_likelihood_weight,
            max_iter=max_iter, init_params=init_params,
            confusion_noise=confusion_noise, random_state=random_state,
        )
        self.pca = None if n_pca_components is None else _PCA(n_pca_components)
        self.verbose = verbose

    # ------------------------------------------------------------------
    def fit(self, data=None, embeddings=None, targets=None, pred_probs=None):
        embeddings = np.asarray(embeddings, dtype=np.float64)
        c = self.config
        variables = [
            GaussianVariable("embeddings", weight=1.0,
                             covariance_type=c["covariance_type"])
        ]
        values = [embeddings]
        if targets is not None:
            variables.append(
                CategoricalVariable("targets", weight=c["y_log_likelihood_weight"])
            )
            values.append(np.asarray(targets))
        if pred_probs is not None:
            variables.append(
                CategoricalVariable("pred_probs",
                                    weight=c["y_hat_log_likelihood_weight"])
            )
            values.append(np.asarray(pred_probs))

        self.mm = DominoMixture(
            n_components=c["n_mixture_components"], variables=variables,
            max_iter=c["max_iter"], init_params=c["init_params"],
            confusion_noise=c["confusion_noise"],
            random_state=c["random_state"], verbose=self.verbose,
        )

        if self.pca is not None:
            self.pca.fit(embeddings)
            values[0] = self.pca.transform(embeddings)

        self.mm.fit(values)

        # slice ranking - identical to the original implementation
        if len(self.mm.variables) == 2:
            self.slice_cluster_indices = (
                -np.abs(self.mm.variables[1].probs.max(axis=1))
            ).argsort()[: c["n_slices"]]
        else:
            self.slice_cluster_indices = (
                -np.abs(
                    (self.mm.variables[2].probs - self.mm.variables[1].probs
                     ).max(axis=1)
                )
            ).argsort()[: c["n_slices"]]
        return self

    def _values(self, embeddings, targets, pred_probs):
        embeddings = np.asarray(embeddings, dtype=np.float64)
        if self.pca is not None:
            embeddings = self.pca.transform(embeddings)
        values = [embeddings]
        if len(self.mm.variables) > 1:
            values.append(None if targets is None else np.asarray(targets))
        if len(self.mm.variables) > 2:
            values.append(None if pred_probs is None else np.asarray(pred_probs))
        return values

    def predict_proba(self, data=None, embeddings=None, targets=None,
                      pred_probs=None) -> np.ndarray:
        clusters = self.mm.predict_proba(
            self._values(embeddings, targets, pred_probs)
        )
        return clusters[:, self.slice_cluster_indices]

    def predict(self, data=None, embeddings=None, targets=None,
                pred_probs=None) -> np.ndarray:
        probs = self.predict_proba(
            embeddings=embeddings, targets=targets, pred_probs=pred_probs
        )
        preds = np.zeros_like(probs, dtype=np.int32)
        preds[np.arange(len(preds)), probs.argmax(axis=-1)] = 1
        return preds
