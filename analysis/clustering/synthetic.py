"""Planted-cluster synthetic data for validating the clustering engine.

Returns a ChannelTable (so it flows through prepare -> reduce -> cluster -> stability
exactly like real data) plus a ground-truth label array (-1 marks planted outliers).
Use it to check: subset/PCA recover known structure, full-space degrades honestly as
noise dimensions grow, and stability survives for real clusters but washes out for none.
"""

from typing import Tuple

import numpy as np

from analysis.clustering.normalize import ChannelTable


def make_synthetic(
    n_per_cluster: int = 20,
    n_clusters: int = 3,
    n_informative: int = 4,
    n_noise_features: int = 6,
    cluster_sep: float = 6.0,
    n_outliers: int = 0,
    random_state: int = 0,
) -> Tuple[ChannelTable, np.ndarray]:
    rng = np.random.default_rng(random_state)

    # Deterministic, well-separated centers: cluster k sits high on informative axis k
    # (requires n_informative >= n_clusters). Removes seed-dependent center collisions,
    # so every validation run is reproducible.
    centers = np.zeros((n_clusters, n_informative))
    for k in range(n_clusters):
        centers[k, k % n_informative] = cluster_sep
    blocks, labels = [], []
    for k in range(n_clusters):
        blocks.append(rng.normal(centers[k], 1.0, size=(n_per_cluster, n_informative)))
        labels += [k] * n_per_cluster
    X_info = np.vstack(blocks)
    n = X_info.shape[0]

    X_noise = rng.normal(0.0, 1.0, size=(n, n_noise_features))  # pure noise: no cluster signal
    X = np.hstack([X_info, X_noise])
    labels = np.array(labels)

    if n_outliers:
        lo, hi = X.min() - cluster_sep, X.max() + cluster_sep
        X = np.vstack([X, rng.uniform(lo, hi, size=(n_outliers, X.shape[1]))])
        labels = np.concatenate([labels, -np.ones(n_outliers, dtype=int)])

    perm = rng.permutation(X.shape[0])
    X, labels = X[perm], labels[perm]

    names = [f"info_{i}" for i in range(n_informative)] + [f"noise_{i}" for i in range(n_noise_features)]
    m = X.shape[0]
    table = ChannelTable(
        X=X,
        feature_names=names,
        video_ids=[f"synthetic_{i}" for i in range(m)],
        channel_ids=[0] * m,
        flags=["0"] * m,
        datasets=["synthetic"] * m,
    )
    return table, labels