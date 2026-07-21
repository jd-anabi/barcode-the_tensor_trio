"""Two-stage clusterer: HDBSCAN for structure + noise handling, then a
Gaussian mixture seeded by the HDBSCAN cluster count for regularized soft assignments,
with a BIC sweep cross-checking that count.
"""

from dataclasses import dataclass, field

import numpy as np
from sklearn.cluster import HDBSCAN
from sklearn.mixture import GaussianMixture
from scipy.optimize import linear_sum_assignment


@dataclass
class ClusterResult:
    labels: np.ndarray            # HDBSCAN hard labels (-1 = noise)
    probabilities: np.ndarray     # HDBSCAN membership strength per point
    outlier_scores: np.ndarray    # 1 - probabilities (proxy; sklearn HDBSCAN exposes no GLOSH)
    n_clusters: int               # HDBSCAN clusters, excluding noise
    noise_fraction: float
    gmm_labels: np.ndarray        # GMM hard assignment (argmax); empty if GMM skipped
    gmm_responsibilities: np.ndarray  # (n, k) soft assignments
    gmm_n_components: int         # seeded from the HDBSCAN count
    gmm_bic_best_k: int           # BIC-optimal count (cross-check; 0 if GMM skipped)
    meta: dict = field(default_factory=dict)


def _hdbscan_cluster_count(labels: np.ndarray) -> int:
    return int(len(set(labels[labels >= 0].tolist())))


def cluster(
    X: np.ndarray,
    min_cluster_size: int = 5,
    min_samples: int = 0,
    run_gmm: bool = True,
    covariance_type: str = "diag",
    gmm_n_init: int = 5,
    bic_max_k: int = 10,
    random_state: int = 0,
) -> ClusterResult:
    X = np.asarray(X, dtype=float)
    n = X.shape[0]

    # --- Stage 1: HDBSCAN (density-based; labels sparse points as noise) ---
    mcs = int(max(2, min_cluster_size))
    ms = None if (min_samples is None or int(min_samples) <= 0) else int(min_samples)
    hdb = HDBSCAN(min_cluster_size=mcs, min_samples=ms)
    labels = hdb.fit_predict(X)
    probs = np.asarray(getattr(hdb, "probabilities_", np.zeros(n)), dtype=float)
    k_hdb = _hdbscan_cluster_count(labels)
    noise_fraction = float(np.mean(labels < 0)) if n else 0.0

    # --- Stage 2: GMM seeded by the HDBSCAN count, + BIC sweep cross-check ---
    gmm_labels = np.array([], dtype=int)
    gmm_resp = np.empty((n, 0), dtype=float)
    k_bic = 0
    if run_gmm and k_hdb >= 1 and n >= 2:
        k_cap = int(min(bic_max_k, n - 1, max(k_hdb + 3, 2)))
        bics = {}
        for k in range(1, k_cap + 1):
            try:
                g = GaussianMixture(n_components=k, covariance_type=covariance_type,
                                    reg_covar=1e-3, n_init=gmm_n_init, random_state=random_state)
                g.fit(X)
                bics[k] = g.bic(X)
            except Exception:
                continue
        if bics:
            k_bic = int(min(bics, key=bics.get))
        k_seed = int(min(max(1, k_hdb), n))
        g = GaussianMixture(n_components=k_seed, covariance_type=covariance_type,
                            reg_covar=1e-3, n_init=gmm_n_init, random_state=random_state)
        g.fit(X)
        gmm_resp = g.predict_proba(X)
        # Align GMM components to HDBSCAN cluster ids so GMM_P{c} genuinely means "probability of belonging to cluster c".
        # Without this the mixture's component order is arbitrary and the soft columns can't be read
        # alongside the hard labels.
        clusters_sorted = sorted(set(labels[labels >= 0].tolist()))
        if len(clusters_sorted) == gmm_resp.shape[1]:
            hard = gmm_resp.argmax(axis=1)
            C = np.zeros((len(clusters_sorted), gmm_resp.shape[1]), dtype=int)
            for r, c in enumerate(clusters_sorted):
                for k in range(gmm_resp.shape[1]):
                    C[r, k] = int(np.sum((labels == c) & (hard == k)))
            rows, cols = linear_sum_assignment(-C)
            perm = np.arange(gmm_resp.shape[1])
            for r, k in zip(rows, cols):
                perm[clusters_sorted[r]] = k
            gmm_resp = gmm_resp[:, perm]
        gmm_labels = gmm_resp.argmax(axis=1)

    return ClusterResult(
        labels=labels,
        probabilities=probs,
        outlier_scores=1.0 - probs,
        n_clusters=k_hdb,
        noise_fraction=noise_fraction,
        gmm_labels=gmm_labels,
        gmm_responsibilities=gmm_resp,
        gmm_n_components=(k_hdb if k_hdb >= 1 else 0),
        gmm_bic_best_k=k_bic,
        meta={"n_samples": n, "min_cluster_size": mcs, "min_samples": ms,
              "covariance_type": covariance_type},
    )