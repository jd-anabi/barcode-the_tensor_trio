"""Bootstrap stability — the actual deliverable.

A single partition at n≈50–200 is a point estimate with large, unquantified sampling
variance. This re-runs the whole standardized -> reduction -> cluster path on many resamples
and reports:
  * co-association matrix M: M[i,j] = fraction of resamples in which BOTH i and j were
    drawn that placed them in the same cluster. Its block structure is the stable grouping.
  * cluster-count distribution across resamples (is k stable, or wandering?)
  * per-point stability (how consistently each channel is co-assigned)
  * noise-fraction distribution

If the block structure washes out, the honest conclusion is that n is too small to support
a partition — a legitimate and important result, not a failure to suppress.

IN-FOLD SCALING: this takes the RAW prepared matrix, never a pre-standardized one. The
scaler is fit inside every resample; fitting globally then resampling leaks
across the fold boundary.
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

import numpy as np

from analysis.clustering.prepare import standardize
from analysis.clustering.reduce import reduce as _reduce
from analysis.clustering.cluster import cluster as _cluster


@dataclass
class StabilityResult:
    co_association: np.ndarray        # (n, n) fraction co-clustered, given both drawn
    co_occurrence: np.ndarray         # (n, n) how many resamples drew both
    per_point_stability: np.ndarray   # (n,) mean co-association with own reference cluster
    cluster_count_distribution: Dict[int, int]
    noise_fractions: np.ndarray       # (B,) noise fraction per resample
    reference_labels: np.ndarray      # full-data partition (for ordering + per-point stability)
    n_resamples: int
    meta: dict = field(default_factory=dict)


def _fit_once(X_sub, reduce_kwargs, cluster_kwargs):
    Z, _ = standardize(X_sub)                       # in-fold: fit on THIS partition only
    Zr, _ = _reduce(Z, **reduce_kwargs)
    return _cluster(Zr, run_gmm=False, **cluster_kwargs)


def cluster_order(labels) -> np.ndarray:
    """Index order grouping points by reference cluster (noise last); for heatmaps."""
    lab = np.asarray(labels)
    order = []
    for c in sorted(set(lab[lab >= 0].tolist())):
        order.extend(np.where(lab == c)[0].tolist())
    order.extend(np.where(lab < 0)[0].tolist())
    return np.array(order, dtype=int)


def stability(
    X_raw: np.ndarray,
    *,
    reduce_kwargs: Optional[dict] = None,
    cluster_kwargs: Optional[dict] = None,
    n_bootstrap: int = 200,
    resample: str = "subsample",        # "subsample" (default) | "bootstrap"
    subsample_fraction: float = 0.8,
    random_state: int = 0,
    progress: Optional[Callable[[int, int], None]] = None,
) -> StabilityResult:
    X_raw = np.asarray(X_raw, dtype=float)
    n = X_raw.shape[0]
    reduce_kwargs = dict(reduce_kwargs or {"mode": "full"})
    cluster_kwargs = dict(cluster_kwargs or {})
    rng = np.random.default_rng(random_state)

    reference_labels = _fit_once(X_raw, reduce_kwargs, cluster_kwargs).labels

    co_cluster = np.zeros((n, n), dtype=float)
    co_occur = np.zeros((n, n), dtype=float)
    k_counts, noise_fracs = [], []
    m = max(4, int(round(subsample_fraction * n)))

    for b in range(n_bootstrap):
        if resample == "bootstrap":
            # dedup: exact duplicate rows create zero-distance pairs that distort
            # HDBSCAN's density estimates, so we keep unique draws.
            idx = np.unique(rng.integers(0, n, size=n))
        else:
            idx = rng.choice(n, size=min(m, n), replace=False)
        if idx.size < 4:
            continue
        try:
            res = _fit_once(X_raw[idx], reduce_kwargs, cluster_kwargs)
        except Exception:
            continue

        lab = res.labels
        valid = lab >= 0
        same = (lab[:, None] == lab[None, :]) & valid[:, None] & valid[None, :]
        grid = np.ix_(idx, idx)
        co_cluster[grid] += same
        co_occur[grid] += 1.0
        k_counts.append(int(res.n_clusters))
        noise_fracs.append(float(res.noise_fraction))
        if progress is not None:
            progress(b + 1, n_bootstrap)

    M = np.where(co_occur > 0, co_cluster / np.maximum(co_occur, 1.0), 0.0)

    per_point = np.full(n, np.nan)
    for i in range(n):
        c = reference_labels[i]
        if c < 0:
            continue                       # noise points have no reference cluster
        peers = np.where(reference_labels == c)[0]
        peers = peers[peers != i]
        if peers.size:
            mask = co_occur[i, peers] > 0
            if mask.any():
                per_point[i] = float(M[i, peers][mask].mean())

    dist: Dict[int, int] = {}
    for k in k_counts:
        dist[k] = dist.get(k, 0) + 1

    return StabilityResult(
        co_association=M,
        co_occurrence=co_occur,
        per_point_stability=per_point,
        cluster_count_distribution=dict(sorted(dist.items())),
        noise_fractions=np.asarray(noise_fracs, dtype=float),
        reference_labels=reference_labels,
        n_resamples=len(k_counts),
        meta={"n_channels": n, "resample": resample,
              "subsample_fraction": subsample_fraction if resample == "subsample" else None,
              "reduce_kwargs": reduce_kwargs, "cluster_kwargs": cluster_kwargs,
              "random_state": random_state},
    )