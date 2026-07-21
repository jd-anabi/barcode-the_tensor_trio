"""Unsupervised clustering of BARCODE metrics (regime discovery).

Pipeline: normalize -> prepare -> reduce -> cluster -> stability -> report.
Only `normalize_generate` / `normalize_analyze` are window-specific; everything
downstream operates on the shared `ChannelTable`.
"""

from analysis.clustering.normalize import (
    ChannelTable,
    to_channel_table,
    combine_tables,
    normalize_generate,
    normalize_analyze,
    load_analyze,
)
from analysis.clustering.prepare import prepare, standardize
from analysis.clustering.synthetic import make_synthetic
from analysis.clustering.reduce import (
    reduce,
    effective_dimensionality,
    available_subset_metrics,
)
from analysis.clustering.cluster import cluster, ClusterResult
from analysis.clustering.stability import stability, StabilityResult, cluster_order
from analysis.clustering.diagnostics import (
    regimes_in_retained_pcs,
    pca_eigenvector_stability,
)
from analysis.clustering.report import (
    video_provenance,
    write_assignments_csv,
    write_pca_loadings_csv,
    write_run_metadata,
    plot_co_association,
    plot_stability_summary,
    plot_cluster_scatter,
    write_cluster_roster_csv,
    plot_coassociation_dendrogram,
    plot_cluster_ordered_barcode,
    generate_report,
)
from analysis.clustering.run import run_clustering

__all__ = [
    # ingestion
    "ChannelTable",
    "to_channel_table",
    "combine_tables",
    "normalize_generate",
    "normalize_analyze",
    "load_analyze",
    # hygiene + scaling
    "prepare",
    "standardize",
    # synthetic validation data
    "make_synthetic",
    # reduction
    "reduce",
    "effective_dimensionality",
    "available_subset_metrics",
    # clustering
    "cluster",
    "ClusterResult",
    # stability (the deliverable)
    "stability",
    "StabilityResult",
    "cluster_order",
    # PCA diagnostics
    "regimes_in_retained_pcs",
    "pca_eigenvector_stability",
    # reporting
    "video_provenance",
    "write_assignments_csv",
    "write_pca_loadings_csv",
    "write_run_metadata",
    "plot_co_association",
    "plot_stability_summary",
    "plot_cluster_scatter",
    "write_cluster_roster_csv",
    "plot_coassociation_dendrogram",
    "plot_cluster_ordered_barcode",
    "generate_report",
    # run
    "run_clustering",
]