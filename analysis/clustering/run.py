"""Single window-agnostic entry point.

Both BARCODE windows call exactly this function; only how they obtain the ChannelTable
differs (normalize_generate vs normalize_analyze). No per-mode forks live below here.
"""

import os

from analysis.clustering.prepare import prepare, standardize
from analysis.clustering.reduce import reduce
from analysis.clustering.cluster import cluster
from analysis.clustering.stability import stability
from analysis.clustering.diagnostics import regimes_in_retained_pcs, pca_eigenvector_stability
from analysis.clustering.report import generate_report


def _default_progress(done, total):
    step = max(1, total // 4)
    if done % step == 0 or done == total:
        print(f"  clustering stability: {int(100 * done / total)}%", flush=True)


def run_clustering(table, config, output_dir, results=None, progress=None):
    """Guardrails -> prepare -> standardize -> reduce -> cluster -> stability -> report.

    `config` is a ClusterConfig. `results` is the optional List[ChannelResults] used only
    for the cluster-ordered barcode. Returns {name: path}, or None if a guardrail blocked
    the run (which is a legitimate outcome, not an error).
    """
    clean, prep_report = prepare(table, direction_handling=config.direction_handling)

    if clean.n_channels < config.min_channels:
        print(f"Clustering skipped: only {clean.n_channels} usable channels "
              f"(minimum {config.min_channels}).")
        return None
    if clean.n_features < 2:
        print(f"Clustering skipped: only {clean.n_features} usable metric column(s) "
              f"remain after NaN hygiene.")
        return None
    if config.mode == "subset" and len(config.selected_metrics or []) < 2:
        print("Clustering skipped: subset mode needs at least 2 selected metrics.")
        return None

    reduce_kwargs = {"mode": config.mode,
                     "feature_names": clean.feature_names,
                     "selected_metrics": list(config.selected_metrics or []),
                     "pca_variance_threshold": float(config.pca_variance_threshold),
                     "pca_n_components": int(config.pca_n_components),
                     "random_state": int(config.random_seed)}
    cluster_kwargs = {"min_cluster_size": int(config.min_cluster_size),
                      "min_samples": int(config.min_samples),
                      "random_state": int(config.random_seed)}

    Xs, _ = standardize(clean.X)
    Xr, red_meta = reduce(Xs, **reduce_kwargs)
    res = cluster(Xr, **cluster_kwargs)
    print(f"Clustering: {res.n_clusters} clusters, noise fraction {res.noise_fraction:.2f} "
          f"({clean.n_channels} channels, {red_meta['n_components']} components, mode={config.mode})")

    st = stability(clean.X, reduce_kwargs=reduce_kwargs, cluster_kwargs=cluster_kwargs,
                   n_bootstrap=int(config.n_bootstrap), random_state=int(config.random_seed),
                   progress=progress or _default_progress)
    print(f"Stability over {st.n_resamples} resamples: "
          f"cluster-count distribution {st.cluster_count_distribution}")

    extra = {"cluster_config": config.to_dict()}
    if config.mode == "pca":
        extra["pca_diagnostics"] = {
            "regimes_in_retained_pcs": regimes_in_retained_pcs(
                clean.X, max_components=12, cluster_kwargs=cluster_kwargs),
            "eigenvector_stability": pca_eigenvector_stability(
                clean.X, n_components=min(3, red_meta["n_components"]),
                n_resamples=100, random_state=int(config.random_seed)),
        }

    return generate_report(output_dir, table=clean, prepare_report=prep_report,
                           reduction_meta=red_meta, X_reduced=Xr, cluster_result=res,
                           stability_result=st, results=results, extra=extra)