"""Reporting: turn a SupervisedResult into artifacts a collaborator can read.

`draw_importances` is deliberately separated from `plot_importances`: the worker thread
saves the PNG through a headless Agg canvas here, while the GUI tab calls the same drawing
function on its own Tk-backed figure from the main thread. Neither thread ever touches a
figure the other owns, and the two renderings cannot drift apart.

Nothing in this module uses pyplot. pyplot keeps a global figure registry that is shared
with the Tk main thread and never garbage-collects on its own; bare Figure objects have
neither problem.
"""

import csv
import json
import os
from typing import Optional

import numpy as np
from scipy.stats import spearmanr

import matplotlib
matplotlib.use("Agg")          # we only ever save PNGs here; matches main.py
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

from analysis.supervised.feature_importance import NON_INDEPENDENCE_CAVEAT, SORT_METHODS

_PURPLE = "#4c2a85"
_METHOD_LABELS = {"permutation": "permutation", "impurity": "impurity", "shap": "SHAP"}


def _jsonable(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def draw_importances(ax, result, max_features: Optional[int] = None) -> None:
    """Draw the grouped horizontal bar chart onto an existing Axes.

    Each method is normalised to its own largest magnitude, because the three are on
    incomparable scales — impurity sums to 1, permutation is a drop in score, SHAP is in
    target units. Normalising by max(|values|) rather than max(values) matters: permutation
    importance is legitimately negative for a feature the model is better off without, and
    dividing by a negative maximum would flip every bar in the chart.
    """
    order = result.order()
    if max_features:
        order = order[:max_features]
    order = order[::-1]  # barh draws bottom-up; most important should end up on top

    methods = [m for m in SORT_METHODS if np.isfinite(result.values[m]).any()]
    names = [result.feature_names[i] for i in order]
    y_pos = np.arange(len(order))
    height = 0.8 / max(len(methods), 1)

    for k, method in enumerate(methods):
        values = np.asarray(result.values[method], dtype=float)[order]
        scale = np.nanmax(np.abs(values)) if np.isfinite(values).any() else 0.0
        normalised = values / scale if scale else np.zeros_like(values)
        ax.barh(y_pos + k * height, np.nan_to_num(normalised), height=height,
                label=_METHOD_LABELS.get(method, method),
                color=None if len(methods) > 1 else _PURPLE)

    ax.set_yticks(y_pos + height * (len(methods) - 1) / 2)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("importance (normalised to each method's largest magnitude)", fontsize=9)
    ax.set_title(f"{result.model_type}: what drives '{result.target_name}'", fontsize=10)
    ax.axvline(0, color="grey", lw=0.6)
    ax.legend(fontsize=8, frameon=False)


def plot_importances(path: str, result) -> str:
    """Save the chart. Height grows with feature count, which is safe at ~25 metrics."""
    n = len(result.feature_names)
    fig = Figure(figsize=(9, max(4.0, 0.4 * n)), dpi=150)
    FigureCanvasAgg(fig)
    draw_importances(fig.add_subplot(111), result)
    fig.tight_layout()
    fig.savefig(path)
    return path


def write_importances_csv(path: str, result) -> str:
    """Per-feature values from all three methods, plus each method's rank."""
    ranks = result.ranks()
    order = result.order()

    def cell(value):
        return "" if not np.isfinite(value) else round(float(value), 6)

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Feature", "Impurity_Importance", "Permutation_Importance",
                    "Permutation_Std", "SHAP_Importance",
                    "Impurity_Rank", "Permutation_Rank", "SHAP_Rank"])
        for i in order:
            w.writerow([result.feature_names[i],
                        cell(result.impurity[i]),
                        cell(result.permutation[i]),
                        cell(result.permutation_std[i]),
                        cell(result.shap[i]),
                        cell(ranks["impurity"][i]),
                        cell(ranks["permutation"][i]),
                        cell(ranks["shap"][i])])
    return path


def write_ranking_agreement_csv(path: str, result) -> Optional[str]:
    """Spearman correlation between the methods' rankings.

    Three methods agreeing is the strongest signal this analysis produces. Two of them
    disagreeing means the top of the chart should not be read as settled.
    """
    ranks = result.ranks()
    usable = [m for m in SORT_METHODS if np.isfinite(ranks[m]).all() and len(ranks[m]) > 1]
    if len(usable) < 2:
        return None

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([""] + [_METHOD_LABELS[m] for m in usable])
        for a in usable:
            row = [_METHOD_LABELS[a]]
            for b in usable:
                rho = 1.0 if a == b else float(spearmanr(ranks[a], ranks[b]).statistic)
                row.append(round(rho, 4))
            w.writerow(row)
    return path


def write_metrics_txt(path: str, result) -> str:
    """Held-out performance, and the caveat that governs how it may be used."""
    labels = {"oob_score": "OOB score", "test_accuracy": "Test accuracy",
              "test_r2": "Test R^2", "mae": "MAE", "rmse": "RMSE"}

    with open(path, "w", encoding="utf-8") as f:
        f.write(f"Mode: {result.model_type}\nTarget: {result.target_name}\n")
        f.write("=" * 60 + "\n")
        for key, value in result.metrics.items():
            f.write(f"{labels.get(key, key)}: {value:.4f}\n")
        if "oob_score" not in result.metrics:
            f.write("OOB score: not available (too few bootstrap samples at this size)\n")

        f.write(f"\nSamples: {result.split_meta.get('n_samples')}"
                f" | Videos: {result.split_meta.get('n_groups')}"
                f" | Features: {len(result.feature_names)}\n")
        f.write(f"Split strategy: {result.split_meta.get('strategy')}\n")
        if result.split_meta.get("fallback_reason"):
            f.write(f"  fell back because: {result.split_meta['fallback_reason']}\n")
        if result.classes:
            f.write(f"Classes: {', '.join(result.classes)}\n")
        if result.best_params:
            f.write(f"\nTuned parameters: {result.best_params}\n")
        f.write(f"\nRanked by: {result.sort_by}\n")
        if not result.shap_ok:
            f.write("SHAP: not computed (package missing or the explainer failed)\n")

        f.write("\n" + "-" * 60 + "\n")
        f.write(NON_INDEPENDENCE_CAVEAT + "\n")
    return path


def write_predictions_csv(path: str, result) -> Optional[str]:
    """Per-row actual vs predicted, labelled by which side of the split it landed on.

    The Split column is derived from row indices, not file names: a video contributes one
    row per channel, so matching on the file name alone would mark every channel of a
    video 'test' as soon as any one of them was held out.
    """
    if not result.predictions:
        return None

    fields = list(result.predictions[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(result.predictions)
    return path


def write_tuning_txt(path: str, result) -> Optional[str]:
    if not result.best_params:
        return None
    with open(path, "w", encoding="utf-8") as f:
        f.write("Best parameters from RandomizedSearchCV\n")
        f.write("=" * 40 + "\n")
        for key, value in result.best_params.items():
            f.write(f"{key}: {value}\n")
        if result.metrics.get("best_cv_score") is not None:
            f.write(f"\nBest CV score: {result.metrics['best_cv_score']:.4f}\n")
    return path


def write_run_metadata(path: str, result, config=None) -> str:
    """Everything needed to reproduce and audit the run."""
    meta = {
        "mode": result.model_type,
        "target": result.target_name,
        "features": list(result.feature_names),
        "split": result.split_meta,
        "metrics": result.metrics,
        "classes": result.classes,
        "sorted_by": result.sort_by,
        "shap_computed": result.shap_ok,
        "prepare": result.prepare_report,
        "caveat": NON_INDEPENDENCE_CAVEAT,
    }
    if result.best_params:
        meta["tuned_parameters"] = result.best_params
    if config is not None:
        meta["feature_importance_config"] = config.to_dict()

    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=_jsonable)
    return path


def generate_report(output_dir: str, result, config=None) -> dict:
    """Write the full artifact set. Returns {name: path}."""
    os.makedirs(output_dir, exist_ok=True)
    paths = {}

    paths["chart"] = plot_importances(
        os.path.join(output_dir, "Feature Importances.png"), result)
    paths["importances"] = write_importances_csv(
        os.path.join(output_dir, "Feature Importances.csv"), result)

    agreement = write_ranking_agreement_csv(
        os.path.join(output_dir, "Ranking Agreement (Spearman).csv"), result)
    if agreement:
        paths["agreement"] = agreement

    paths["metrics"] = write_metrics_txt(
        os.path.join(output_dir, "Model Evaluation Metrics.txt"), result)

    predictions = write_predictions_csv(
        os.path.join(output_dir, "Predictions by File.csv"), result)
    if predictions:
        paths["predictions"] = predictions

    tuning = write_tuning_txt(os.path.join(output_dir, "Tuning Results.txt"), result)
    if tuning:
        paths["tuning"] = tuning

    paths["metadata"] = write_run_metadata(
        os.path.join(output_dir, "Feature Importance Run Metadata.json"), result, config)
    return paths
