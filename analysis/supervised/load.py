"""Ingestion: a BARCODE CSV with a ground-truth column appended -> a learning table.

`read_csv_to_channel_results` cannot be reused here. It asserts the header row equals one
of three known schemas exactly (utils/reader.py), so a CSV carrying an extra target column
is rejected outright. We read the header ourselves and classify every column as metadata,
metric, or candidate target.

Deliberately plain numpy + csv, matching analysis/clustering/normalize.py: pandas is not a
BARCODE dependency and everything needed here is a few lines of numpy.
"""

import csv
import os
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from core import ChannelResults, Metrics

# The three columns BARCODE writes before the metrics. Identifiers, never features.
META_COLUMNS = [Metrics.FILEPATH.value, Metrics.CHANNEL.value, Metrics.FLAGS.value]


class FeatureImportanceError(ValueError):
    """A user-correctable problem with the CSV, the target column, or the settings.

    Distinct from a guardrail: guardrails mean "this dataset cannot answer the question"
    and return None (see run.py). This means "the input is wrong" and is shown to the user.
    """


def _known_metric_names() -> List[str]:
    """Every column BARCODE itself writes as a metric, in both unit conventions.

    A CSV produced with Output Unit Conversion on carries the `* Quantity` area columns
    instead of the %-FOV ones, so both sets count as metrics for classification purposes.
    """
    names = list(ChannelResults.get_headers(just_metrics=True))
    for name in ChannelResults.get_physical_headers(just_metrics=True):
        if name not in names:
            names.append(name)
    return names


def read_header(csv_path: str) -> List[str]:
    """First row of the CSV, stripped. `utf-8-sig` so a BOM does not corrupt column one."""
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.reader(f):
            return [c.strip() for c in row]
    return []


def candidate_target_columns(csv_path: str) -> List[str]:
    """Columns that could be the appended ground truth: anything BARCODE did not write.

    Populates the GUI dropdown, so it reads only the header row. `load_supervised` accepts
    any column name, including a metric — predicting one metric from the others is a valid
    regression — this list is just the sensible default offering.
    """
    header = read_header(csv_path)
    known = set(_known_metric_names()) | set(META_COLUMNS)
    return [c for c in header if c and c not in known]


def _to_float(value: str) -> float:
    """Blank / 'nan' / unparseable -> NaN. Same tolerance as utils/reader.py:80-88."""
    text = (value or "").strip()
    if text == "" or text.lower() == "nan":
        return np.nan
    try:
        return float(text)
    except ValueError:
        return np.nan


@dataclass
class SupervisedTable:
    """One row = one channel (the learning sample). IDs are metadata, NEVER features.

    `X` may contain NaN; imputation is deliberately deferred to after the train/test split
    so the medians can be fit on training rows only.
    """

    X: np.ndarray               # (n_samples, n_features) raw metric values
    feature_names: List[str]    # length n_features
    y_raw: List[str]            # target column exactly as read; encoded later, per mode
    groups: List[str]           # the File column — a video's channels share a group
    channels: List[str]         # length n_samples
    target_name: str
    report: dict = field(default_factory=dict)

    @property
    def n_samples(self) -> int:
        return self.X.shape[0]

    @property
    def n_features(self) -> int:
        return self.X.shape[1]

    def __repr__(self) -> str:
        return (f"SupervisedTable(n_samples={self.n_samples}, n_features={self.n_features}, "
                f"target='{self.target_name}', groups={len(set(self.groups))})")


def load_supervised(csv_paths: Sequence[str], target_column: str,
                    drop_flags: Sequence[int] = (1, 2)) -> SupervisedTable:
    """Read BARCODE CSV(s) into a SupervisedTable, dropping unusable rows.

    Rows go away for two reasons, both recorded in the report:
      1. A disqualifying Flag. Same rule as clustering (analysis/clustering/prepare.py):
         flags 1 (dim) and 2 (saturated) invalidate the channel, while 3 and 4 only cap a
         single metric, so those channels are kept.
      2. A blank target. There is nothing to learn from an unlabelled row.
    """
    if not csv_paths:
        raise FeatureImportanceError("No CSV file selected.")
    if not target_column:
        raise FeatureImportanceError("No target column selected.")

    known = _known_metric_names()
    drop_set = {int(f) for f in drop_flags}
    feature_names: Optional[List[str]] = None
    rows, y_raw, groups, channels = [], [], [], []
    report = {"csv_paths": list(csv_paths), "input_rows": 0,
              "dropped_flagged": [], "dropped_missing_target": []}

    for path in csv_paths:
        header = read_header(path)
        if target_column not in header:
            raise FeatureImportanceError(
                f"'{target_column}' is not a column in {os.path.basename(path)}. "
                f"Columns present: {', '.join(header) if header else '(file is empty)'}")

        metrics_here = [c for c in header if c in known and c != target_column]
        if not metrics_here:
            raise FeatureImportanceError(
                f"{os.path.basename(path)} contains no BARCODE metric columns. "
                "Is this a BARCODE summary CSV?")
        if feature_names is None:
            feature_names = metrics_here
        elif metrics_here != feature_names:
            raise FeatureImportanceError(
                f"{os.path.basename(path)} has a different metric schema than "
                f"{os.path.basename(csv_paths[0])}; they cannot be combined.")

        index = {name: i for i, name in enumerate(header)}

        def cell(raw, name, default=""):
            j = index.get(name)
            return raw[j] if j is not None and j < len(raw) else default

        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.reader(f)
            next(reader, None)  # header
            for raw in reader:
                if not raw or all(c.strip() == "" for c in raw):
                    continue
                report["input_rows"] += 1

                name = cell(raw, Metrics.FILEPATH.value).strip() or f"row_{report['input_rows']}"
                channel = cell(raw, Metrics.CHANNEL.value).strip()

                flags = cell(raw, Metrics.FLAGS.value, "0")
                codes = {int(c) for c in str(flags).split(";")
                         if c.strip().isdigit() and c.strip() != "0"}
                if codes & drop_set:
                    report["dropped_flagged"].append((name, channel, flags))
                    continue

                target = cell(raw, target_column).strip()
                if target == "" or target.lower() == "nan":
                    report["dropped_missing_target"].append((name, channel))
                    continue

                rows.append([_to_float(cell(raw, m)) for m in feature_names])
                y_raw.append(target)
                groups.append(name)
                channels.append(channel)

    feature_names = feature_names or []
    X = (np.array(rows, dtype=float) if rows
         else np.empty((0, len(feature_names)), dtype=float))
    X, feature_names = encode_direction(X, feature_names)

    report["output_rows"] = X.shape[0]
    report["n_groups"] = len(set(groups))
    report["feature_names"] = list(feature_names)

    return SupervisedTable(X, feature_names, y_raw, groups, channels, target_column, report)


def encode_direction(X: np.ndarray, feature_names: List[str]) -> Tuple[np.ndarray, List[str]]:
    """Mean Flow Direction is an angle in radians — 0 and 2*pi are the same state.

    Split it into (cos, sin) so a tree can split on direction without treating the
    wrap-around as a large numeric gap. Same convention and naming as
    analysis/clustering/prepare.py:66-69, so both modules describe features identically.
    """
    theta = Metrics.MEAN_THETA.value
    if theta not in feature_names or X.size == 0:
        return X, list(feature_names)

    j = feature_names.index(theta)
    cos = np.cos(X[:, j]).reshape(-1, 1)
    sin = np.sin(X[:, j]).reshape(-1, 1)
    X = np.hstack([X[:, :j], cos, sin, X[:, j + 1:]])
    names = (list(feature_names[:j]) + [f"{theta} (cos)", f"{theta} (sin)"]
             + list(feature_names[j + 1:]))
    return X, names


def fit_cleaning(X_train: np.ndarray, feature_names: List[str],
                 max_column_nan_fraction: float = 0.20):
    """Decide which columns to drop and what to impute with, from TRAINING ROWS ONLY.

    Fitting these on the full dataset leaks test information into the model. It is the
    same in-fold rule the clustering module enforces for scaling (prepare.standardize),
    applied here to imputation.

    A metric that is NaN for a large share of rows is dropped as a COLUMN rather than
    being allowed to take all those rows down with it — at these sample sizes losing one
    metric is far cheaper than losing half the dataset.

    Returns (keep_mask, medians, kept_names, dropped) where `medians` is aligned to the
    kept columns.
    """
    if X_train.shape[0] == 0:
        raise FeatureImportanceError("No training rows after cleaning.")

    frac = np.isnan(X_train).sum(axis=0) / X_train.shape[0]
    keep = frac <= max_column_nan_fraction
    dropped = [(feature_names[j], round(float(frac[j]), 3))
               for j in range(len(feature_names)) if not keep[j]]
    kept_names = [feature_names[j] for j in range(len(feature_names)) if keep[j]]

    if not keep.any():
        raise FeatureImportanceError(
            "Every metric column is more than "
            f"{int(max_column_nan_fraction * 100)}% empty in the training rows; "
            "there is nothing to learn from.")

    medians = np.nanmedian(X_train[:, keep], axis=0)
    # A kept column is <= max_column_nan_fraction NaN, so its median is finite — but a
    # degenerate tiny training set can still produce NaN. Fall back to 0 rather than
    # propagating NaN into the forest.
    medians = np.where(np.isnan(medians), 0.0, medians)
    return keep, medians, kept_names, dropped


def apply_cleaning(X: np.ndarray, keep: np.ndarray, medians: np.ndarray) -> np.ndarray:
    """Drop the columns `fit_cleaning` rejected and median-impute what remains."""
    cleaned = X[:, keep].astype(float, copy=True)
    missing = np.isnan(cleaned)
    if missing.any():
        cleaned[missing] = np.take(medians, np.where(missing)[1])
    return cleaned
