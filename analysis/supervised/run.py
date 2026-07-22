"""Single window-agnostic entry point.

Mirrors analysis/clustering/run.py: guardrails -> load -> prepare -> split -> fit ->
importances -> report. The GUI calls exactly this function and nothing below it.

Guardrails print and return None. "This dataset is too small to answer the question" is a
legitimate outcome, not an error — the same stance the clustering module takes. Genuinely
wrong input (a missing target column, an unreadable CSV) raises FeatureImportanceError,
which the GUI worker already catches and shows.
"""

import os
from typing import List, Optional, Sequence

import numpy as np

from analysis.supervised.load import (
    FeatureImportanceError,
    apply_cleaning,
    fit_cleaning,
    load_supervised,
)
from analysis.supervised.feature_importance import (
    CLASSIFICATION,
    REGRESSION,
    SORT_METHODS,
    SupervisedResult,
    build_model,
    compute_importances,
    evaluate,
    make_split,
    prepare_target,
    resolve_sort_by,
)
from analysis.supervised.report import generate_report


def _default_progress(message):
    print(f"  feature importance: {message}", flush=True)


def run_feature_importance(csv_paths, config, output_dir, progress=None) -> Optional[dict]:
    """Guardrails -> load -> split -> fit -> importances -> report.

    `config` is a FeatureImportanceConfig. Returns {"paths": {name: path},
    "result": SupervisedResult}, or None if a guardrail blocked the run (a legitimate
    outcome, not an error).
    """
    say = progress or _default_progress

    if config.model_type not in (CLASSIFICATION, REGRESSION):
        raise FeatureImportanceError(
            f"Model type must be '{CLASSIFICATION}' or '{REGRESSION}', "
            f"not '{config.model_type}'.")
    if config.sort_by not in SORT_METHODS:
        raise FeatureImportanceError(
            f"Sort method must be one of {', '.join(SORT_METHODS)}, not '{config.sort_by}'.")
    if not 0.0 < float(config.test_size) < 1.0:
        raise FeatureImportanceError(
            f"Test fraction must be between 0 and 1, not {config.test_size}.")

    paths = [p for p in (csv_paths or []) if p]
    say(f"reading {len(paths)} CSV file(s)")
    table = load_supervised(paths, config.target_column)

    # ---- guardrails ----
    if table.n_samples < int(config.min_samples):
        print(f"Feature importance skipped: only {table.n_samples} usable labelled row(s) "
              f"(minimum {config.min_samples}). "
              f"{len(table.report['dropped_flagged'])} row(s) were dropped as flagged and "
              f"{len(table.report['dropped_missing_target'])} for a blank target.")
        return None
    if table.n_features < 2:
        print(f"Feature importance skipped: only {table.n_features} metric column(s) "
              f"available; there is nothing to rank.")
        return None

    y, classes = prepare_target(table.y_raw, config.model_type)

    if config.model_type == CLASSIFICATION:
        if len(classes) < 2:
            print(f"Feature importance skipped: '{table.target_name}' has a single class "
                  f"({classes[0]}). Classification needs at least two.")
            return None
        if len(classes) > table.n_samples // 2:
            print(f"Note: {len(classes)} classes across only {table.n_samples} rows — "
                  f"most classes have too few examples for the score to mean much.")
    else:
        # Regression rows whose target would not parse are dead weight; drop them here so
        # they never reach the model, and keep every parallel array in lockstep.
        keep = np.isfinite(y)
        if not keep.all():
            dropped = int((~keep).sum())
            print(f"Dropping {dropped} row(s) whose '{table.target_name}' value is not numeric.")
            table.X = table.X[keep]
            table.y_raw = [v for v, k in zip(table.y_raw, keep) if k]
            table.groups = [v for v, k in zip(table.groups, keep) if k]
            table.channels = [v for v, k in zip(table.channels, keep) if k]
            y = y[keep]
            table.report["dropped_non_numeric_target"] = dropped
            table.report["output_rows"] = table.n_samples
        if table.n_samples < int(config.min_samples):
            print(f"Feature importance skipped: only {table.n_samples} row(s) with a "
                  f"numeric target (minimum {config.min_samples}).")
            return None

    n_test = int(round(table.n_samples * float(config.test_size)))
    if n_test < 1:
        print(f"Feature importance skipped: a test fraction of {config.test_size} leaves "
              f"no held-out rows out of {table.n_samples}.")
        return None

    print(f"Feature importance: {config.model_type} on '{table.target_name}' — "
          f"{table.n_samples} rows from {len(set(table.groups))} video(s), "
          f"{table.n_features} metrics")
    if classes:
        print(f"  classes: {', '.join(classes)}")
    else:
        print(f"  target range: [{np.min(y):.4g}, {np.max(y):.4g}]")

    # ---- split (by video, so near-duplicate channels do not straddle it) ----
    train_idx, test_idx, split_meta = make_split(
        y, table.groups, config.model_type, float(config.test_size),
        int(config.random_seed), bool(config.group_by_file))
    say(f"split {train_idx.size} train / {test_idx.size} test "
        f"({split_meta['strategy']}-wise)")

    # ---- cleaning, fit on training rows only ----
    keep_cols, medians, feature_names, dropped_cols = fit_cleaning(
        table.X[train_idx], table.feature_names)
    if dropped_cols:
        print(f"  dropping {len(dropped_cols)} sparse metric column(s): "
              f"{', '.join(name for name, _ in dropped_cols)}")
    if len(feature_names) < 2:
        print(f"Feature importance skipped: only {len(feature_names)} metric column(s) "
              f"survive NaN hygiene.")
        return None

    X_train = apply_cleaning(table.X[train_idx], keep_cols, medians)
    X_test = apply_cleaning(table.X[test_idx], keep_cols, medians)
    y_train, y_test = y[train_idx], y[test_idx]

    if config.model_type == CLASSIFICATION and len(set(y_train.tolist())) < 2:
        print("Feature importance skipped: the training split ended up with a single class.")
        return None
    if config.model_type == CLASSIFICATION and len(set(y_test.tolist())) < 2:
        # Accuracy is constant on a one-class test set, so shuffling a column cannot move
        # it — every permutation importance would come back at zero for the wrong reason.
        print("Warning: the held-out videos all carry one class. Test accuracy and "
              "permutation importance are not meaningful; try a different seed or a "
              "larger test fraction.")

    # ---- fit ----
    groups_train = [table.groups[i] for i in train_idx]
    say(f"fitting the forest ({'tuned' if config.tuning else 'default'} settings)")
    model, best_params, best_cv = build_model(
        config.model_type, X_train, y_train, groups_train,
        n_estimators=int(config.n_estimators), tuning=bool(config.tuning),
        seed=int(config.random_seed), progress=say)

    metrics = evaluate(model, config.model_type, X_test, y_test)
    if best_cv is not None:
        metrics["best_cv_score"] = best_cv
    print("  " + " | ".join(f"{k} = {v:.3f}" for k, v in metrics.items()))

    # ---- importances ----
    importances, shap_ok = compute_importances(
        model, config.model_type, X_test, y_test, feature_names,
        n_repeats=int(config.permutation_repeats), seed=int(config.random_seed),
        use_shap=bool(config.use_shap), progress=say)
    values = importances["values"]
    sort_by = resolve_sort_by(config.sort_by, values)

    result = SupervisedResult(
        model_type=config.model_type,
        target_name=table.target_name,
        feature_names=list(feature_names),
        impurity=values["impurity"],
        permutation=values["permutation"],
        permutation_std=importances["permutation_std"],
        shap=values["shap"],
        shap_ok=shap_ok,
        sort_by=sort_by,
        metrics=metrics,
        classes=classes,
        predictions=_predictions(model, config.model_type, table, y, classes,
                                 keep_cols, medians, train_idx, test_idx),
        split_meta=split_meta,
        prepare_report={**table.report, "dropped_sparse_columns": dropped_cols,
                        "features_used": list(feature_names)},
        best_params=best_params,
    )

    say("writing report")
    paths = generate_report(output_dir, result, config)
    for name, path in paths.items():
        print(f"  saved: {path}")
    return {"paths": paths, "result": result}


def _predictions(model, model_type, table, y, classes, keep_cols, medians,
                 train_idx, test_idx) -> List[dict]:
    """Per-row actual vs predicted, with the split label taken from row INDICES.

    The original script labelled rows by matching file names against the test set, which
    marked every channel of a video 'test' as soon as one of its channels was held out.
    """
    X_all = apply_cleaning(table.X, keep_cols, medians)
    predicted = model.predict(X_all)

    split = np.empty(len(y), dtype=object)
    split[train_idx] = "train"
    split[test_idx] = "test"

    rows = []
    for i in range(len(y)):
        row = {"File": os.path.basename(table.groups[i]) or table.groups[i],
               "Channel": table.channels[i],
               "Split": split[i]}
        if model_type == CLASSIFICATION:
            row["Actual"] = classes[int(y[i])]
            row["Predicted"] = classes[int(predicted[i])]
        else:
            row["Actual"] = round(float(y[i]), 6)
            row["Predicted"] = round(float(predicted[i]), 6)
            row["Residual"] = round(float(y[i] - predicted[i]), 6)
        rows.append(row)
    return rows
