"""PCA diagnostics that keep the reduction honest."""

from typing import Optional

import numpy as np
from sklearn.decomposition import PCA

from analysis.clustering.prepare import standardize
from analysis.clustering.cluster import cluster as _cluster


def regimes_in_retained_pcs(X_raw, max_components: int = 12,
                            cluster_kwargs: Optional[dict] = None, random_state: int = 0):
    """Cluster in an increasing number of PCs.

    Reveals whether structure lives in the leading (high-variance) PCs or only appears
    once a later, low-variance component is included. Variance is not relevance: a
    separating direction can sit below any threshold you pick, so this check is what
    stops a confident-looking 'no structure' conclusion from being an artifact.
    """
    cluster_kwargs = dict(cluster_kwargs or {})
    Z, _ = standardize(np.asarray(X_raw, dtype=float))
    p = PCA(n_components=min(Z.shape), random_state=random_state).fit(Z)
    scores, cum = p.transform(Z), np.cumsum(p.explained_variance_ratio_)

    rows = []
    for k in range(1, min(max_components, scores.shape[1]) + 1):
        res = _cluster(scores[:, :k], run_gmm=False, **cluster_kwargs)
        rows.append({"n_pcs": k, "n_clusters": int(res.n_clusters),
                     "noise_fraction": float(res.noise_fraction),
                     "cumulative_variance": float(cum[k - 1])})
    return rows


def pca_eigenvector_stability(X_raw, n_components: int = 3, n_resamples: int = 100,
                              subsample_fraction: float = 0.8, random_state: int = 0):
    """Do the leading eigenvectors reappear across resamples (up to sign)?

    Mean |cosine similarity| between each full-data PC and its resample counterpart.
    Low values mean the basis itself is unreliable at this n, so interpreting loadings
    is on thin ice. Comparison is positional, so a drop may also mean two PCs swapped
    order, which itself is a sign of instability.
    """
    X = np.asarray(X_raw, dtype=float)
    n = X.shape[0]
    rng = np.random.default_rng(random_state)
    Zf, _ = standardize(X)
    K = int(min(n_components, min(Zf.shape)))
    ref = PCA(n_components=K, random_state=random_state).fit(Zf).components_
    m = max(K + 1, int(round(subsample_fraction * n)))

    sims = []
    for _ in range(n_resamples):
        idx = rng.choice(n, size=min(m, n), replace=False)
        Zs, _ = standardize(X[idx])
        if min(Zs.shape) < K:
            continue
        comp = PCA(n_components=K, random_state=random_state).fit(Zs).components_
        sims.append([abs(float(np.dot(ref[k], comp[k]))) for k in range(K)])

    sims = np.asarray(sims, dtype=float)
    return {"n_components": K, "n_resamples": int(sims.shape[0]),
            "mean_abs_cosine": sims.mean(axis=0).tolist() if sims.size else [],
            "min_abs_cosine": sims.min(axis=0).tolist() if sims.size else []}