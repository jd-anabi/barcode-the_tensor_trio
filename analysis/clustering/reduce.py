"""Pluggable reduction stage: standardized matrix in -> reduced matrix + metadata.

All three modes share everything downstream; they differ ONLY here.
"""

from typing import Tuple

from sklearn.decomposition import PCA
import numpy as np


def effective_dimensionality(X_std: np.ndarray) -> float:
    """Participation ratio of the correlation-matrix eigenvalues (variances): (Σλ)² / Σλ².

    Ranges from 1 (all variance in one direction) to d (isotropic / uncorrelated).
    For correlated BARCODE metrics this is typically far below the column count.
    """
    n, d = X_std.shape
    if n < 2 or d == 0:
        return float(d)
    C = np.nan_to_num(np.atleast_2d(np.corrcoef(X_std, rowvar=False)), nan=0.0)
    np.fill_diagonal(C, 1.0)
    eig = np.clip(np.linalg.eigvalsh(C), 0.0, None)
    denom = float(np.sum(eig ** 2))
    return float(eig.sum() ** 2 / denom) if denom > 0 else float(d)


def _reduce_full(X_std: np.ndarray, feature_names) -> Tuple[np.ndarray, dict]:
    meta = {
        "mode": "full",
        "n_components": int(X_std.shape[1]),
        "effective_dimensionality": effective_dimensionality(X_std),
        "feature_names": list(feature_names) if feature_names is not None else None,
    }
    return X_std, meta

def _reduce_pca(X_std, feature_names, variance_threshold=0.9, n_components=0, random_state=0):
    """PCA on the correlation matrix (input is already z-scored, so covariance == correlation).

    Component count comes from cumulative variance by default; Kaiser (eigenvalue > 1) is
    reported as a cross-check only and empirically it under-retains and can drop a
    low-variance separating direction.
    """
    n, d = X_std.shape
    max_k = max(1, min(n, d))
    pca = PCA(n_components=max_k, random_state=random_state)
    scores = pca.fit_transform(X_std)
    evr = pca.explained_variance_ratio_
    cum = np.cumsum(evr)
    eig = pca.explained_variance_          # correlation-matrix eigenvalues

    if n_components and int(n_components) > 0:
        k, rule = int(min(int(n_components), max_k)), "explicit"
    else:
        k = int(max(1, min(int(np.searchsorted(cum, variance_threshold) + 1), max_k)))
        rule = f"cumulative_variance>={variance_threshold}"

    meta = {
        "mode": "pca",
        "n_components": k,
        "selection_rule": rule,
        "cumulative_variance": float(cum[k - 1]),
        "explained_variance_ratio": [float(v) for v in evr[:k]],
        "eigenvalues": [float(v) for v in eig],
        "kaiser_n_components": int(np.sum(eig > 1.0)),   # cross-check only
        "loadings": pca.components_[:k].tolist(),        # (k, d): PC x original feature
        "input_feature_names": list(feature_names) if feature_names is not None else None,
        "effective_dimensionality": effective_dimensionality(X_std),
    }
    return scores[:, :k], meta

def _resolve_subset_columns(feature_names, selected_metrics):
    """Map user-selected metric names to column indices in the prepared matrix.

    Handles the circular-direction encoding: selecting 'Mean Flow Direction' picks up BOTH
    '... (cos)' and '... (sin)' produced by prepare(). Names that no longer exist (e.g. a
    metric dropped by NaN hygiene) come back as `unresolved` rather than silently ignored.
    """
    idx_of = {name: j for j, name in enumerate(feature_names)}
    cols, resolved, unresolved = [], [], []
    for name in selected_metrics:
        hits = []
        if name in idx_of:
            hits.append(idx_of[name])
        for suffix in (" (cos)", " (sin)"):
            if f"{name}{suffix}" in idx_of:
                hits.append(idx_of[f"{name}{suffix}"])
        if hits:
            cols.extend(hits)
            resolved.append(name)
        else:
            unresolved.append(name)
    seen = set()
    return [c for c in cols if not (c in seen or seen.add(c))], resolved, unresolved


def available_subset_metrics(feature_names):
    """Original metric names selectable in subset mode, collapsing the (cos)/(sin)
    encoding back to the single underlying metric. Feed this to the GUI picker so the user
    never sees encoded columns. (The 'one unit system' constraint is automatic: get_data()
    only ever emits the %-FOV set, so the µm² variants can't appear here.)
    """
    out = []
    for name in feature_names:
        base = name
        for suffix in (" (cos)", " (sin)"):
            if name.endswith(suffix):
                base = name[: -len(suffix)]
                break
        if base not in out:
            out.append(base)
    return out


def _reduce_subset(X_std, feature_names, selected_metrics):
    """Cluster in the standardized subspace of the user's chosen metrics.

    No rotation, so interpretability is fully preserved. Nothing forces the chosen axes to
    carry cluster structure, the guardrail is that effective dimensionality AND stability
    get reported, so a poor choice is visibly poor rather than quietly confident.
    """
    if feature_names is None:
        raise ValueError("subset mode requires feature_names")
    selected = list(selected_metrics or [])
    cols, resolved, unresolved = _resolve_subset_columns(feature_names, selected)
    if len(resolved) < 2:
        raise ValueError(
            f"Subset mode needs at least 2 usable metrics; resolved {len(resolved)} "
            f"({resolved}). Unresolved: {unresolved}"
        )
    meta = {
        "mode": "subset",
        "n_components": len(cols),
        "selected_metrics": selected,
        "resolved_metrics": resolved,
        "unresolved_metrics": unresolved,
        "feature_names": [feature_names[j] for j in cols],
        "effective_dimensionality": effective_dimensionality(X_std[:, cols]),
    }
    return X_std[:, cols], meta

def reduce(
    X_std: np.ndarray,
    mode: str = "full",
    *,
    feature_names=None,
    selected_metrics=None,
    pca_variance_threshold: float = 0.9,
    pca_n_components: int = 0,
    random_state: int = 0,
) -> Tuple[np.ndarray, dict]:
    """Reduce a standardized matrix. Returns (X_reduced, reduction_metadata)."""
    if mode == "full":
        return _reduce_full(X_std, feature_names)
    if mode == "pca":
        return _reduce_pca(X_std, feature_names,
                           variance_threshold=pca_variance_threshold,
                           n_components=pca_n_components,
                           random_state=random_state)
    if mode == "subset":
        return _reduce_subset(X_std, feature_names, selected_metrics)
    raise ValueError(f"Unknown reduction mode: {mode!r}")