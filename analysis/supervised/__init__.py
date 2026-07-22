"""Supervised feature importance over BARCODE metrics.

The labelled counterpart to analysis/clustering: given a BARCODE CSV with a ground-truth
column appended, rank the 25 metrics by how much each one drives that outcome.

Pipeline: load -> prepare target -> split by video -> fit forest -> importances -> report.
Only `load` knows about CSVs; everything downstream operates on the shared SupervisedTable.

Not imported by analysis/__init__.py on purpose — like clustering, this keeps scikit-learn
off the application's startup path.
"""

from analysis.supervised.load import (
    FeatureImportanceError,
    META_COLUMNS,
    SupervisedTable,
    apply_cleaning,
    candidate_target_columns,
    encode_direction,
    fit_cleaning,
    load_supervised,
    read_header,
)
from analysis.supervised.feature_importance import (
    CLASSIFICATION,
    HAVE_SHAP,
    NON_INDEPENDENCE_CAVEAT,
    REGRESSION,
    SORT_METHODS,
    SupervisedResult,
    build_model,
    compute_importances,
    evaluate,
    make_split,
    prepare_target,
    resolve_sort_by,
    shap_importance,
)
from analysis.supervised.report import (
    draw_importances,
    generate_report,
    plot_importances,
    write_importances_csv,
    write_metrics_txt,
    write_predictions_csv,
    write_ranking_agreement_csv,
    write_run_metadata,
    write_tuning_txt,
)
from analysis.supervised.run import run_feature_importance

__all__ = [
    # ingestion
    "FeatureImportanceError",
    "META_COLUMNS",
    "SupervisedTable",
    "candidate_target_columns",
    "read_header",
    "load_supervised",
    "encode_direction",
    # hygiene (fit on training rows only)
    "fit_cleaning",
    "apply_cleaning",
    # modes
    "CLASSIFICATION",
    "REGRESSION",
    "SORT_METHODS",
    "HAVE_SHAP",
    "NON_INDEPENDENCE_CAVEAT",
    # model + importances
    "SupervisedResult",
    "prepare_target",
    "make_split",
    "build_model",
    "evaluate",
    "compute_importances",
    "shap_importance",
    "resolve_sort_by",
    # reporting
    "draw_importances",
    "plot_importances",
    "write_importances_csv",
    "write_ranking_agreement_csv",
    "write_metrics_txt",
    "write_predictions_csv",
    "write_tuning_txt",
    "write_run_metadata",
    "generate_report",
    # run
    "run_feature_importance",
]
