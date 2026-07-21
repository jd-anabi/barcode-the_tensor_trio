"""Column/row hygiene: raw ChannelTable -> clean feature matrix, plus in-fold scaling.

Order of operations (each recorded in the returned report):
  1. Drop Flags-flagged channels (rows).
  2. Handle the circular Mean Flow Direction metric (encode as cos/sin, or drop).
  3. Drop all-NaN columns (an unused branch, or %-FOV area cols from a physical CSV).
  4. Drop rows that still carry any NaN.

Standardization is deliberately NOT done here, it must be fit inside each resampling
fold, so it lives in `standardize()` and is called by the stability loop.
"""

from typing import List, Sequence, Tuple

import numpy as np
from sklearn.preprocessing import StandardScaler

from core import Metrics
from analysis.clustering.normalize import ChannelTable


def prepare(
    table: ChannelTable,
    direction_handling: str = "encode",   # "encode" (cos,sin) | "drop" | "linear"
    drop_flags: Sequence[int] = (1, 2),   # dim (1) and saturated (2); NOT 3/4 (they cap one metric)
) -> Tuple[ChannelTable, dict]:
    """Return a cleaned ChannelTable (features only in X, IDs alongside) + a report dict."""
    names = list(table.feature_names)
    X = table.X.astype(float, copy=True)
    vids, chans = list(table.video_ids), list(table.channel_ids)
    flags, dsets = list(table.flags), list(table.datasets)
    report = {"input_channels": table.n_channels, "input_features": table.n_features}

    def take_rows(idx: List[int]):
        nonlocal X, vids, chans, flags, dsets
        X = X[idx]
        vids = [vids[i] for i in idx]
        chans = [chans[i] for i in idx]
        flags = [flags[i] for i in idx]
        dsets = [dsets[i] for i in idx]

    # 1. Drop Flags-flagged channels
    drop_set = {int(f) for f in drop_flags}
    keep, dropped_flagged = [], []
    for i, fl in enumerate(flags):
        codes = {int(c) for c in str(fl).split(";") if c not in ("", "0")}
        (dropped_flagged.append((vids[i], chans[i], fl)) if codes & drop_set else keep.append(i))
    take_rows(keep)
    report["dropped_flagged"] = dropped_flagged

    # 2. Circular Mean Flow Direction (θ ∈ [-π, π] radians)
    theta_name = Metrics.MEAN_THETA.value
    if theta_name in names:
        j = names.index(theta_name)
        if direction_handling == "encode":
            cos, sin = np.cos(X[:, j]).reshape(-1, 1), np.sin(X[:, j]).reshape(-1, 1)
            X = np.hstack([X[:, :j], cos, sin, X[:, j + 1:]])
            names = names[:j] + [f"{theta_name} (cos)", f"{theta_name} (sin)"] + names[j + 1:]
        elif direction_handling == "drop":
            X = np.hstack([X[:, :j], X[:, j + 1:]])
            names = names[:j] + names[j + 1:]
        report["direction_handling"] = direction_handling
    else:
        report["direction_handling"] = f"{direction_handling} (no theta column present)"

    # 3. Drop all-NaN columns
    allnan = np.isnan(X).all(axis=0) if X.shape[0] else np.zeros(X.shape[1], bool)
    report["dropped_all_nan_columns"] = [names[j] for j in range(len(names)) if allnan[j]]
    X = X[:, ~allnan]
    names = [names[j] for j in range(len(names)) if not allnan[j]]

    # 4. Drop rows with any residual NaN
    row_nan = np.isnan(X).any(axis=1) if X.shape[0] else np.zeros(0, bool)
    report["dropped_nan_rows"] = [(vids[i], chans[i]) for i in range(len(vids)) if row_nan[i]]
    take_rows([i for i in range(len(vids)) if not row_nan[i]])

    report["output_channels"] = X.shape[0]
    report["output_features"] = X.shape[1]
    report["feature_names"] = list(names)

    return ChannelTable(X, names, vids, chans, flags, dsets), report


def standardize(X_train: np.ndarray, X_apply: np.ndarray = None) -> Tuple[np.ndarray, StandardScaler]:
    """Z-score to mean 0 / var 1 (== PCA on the correlation matrix downstream).

    Fit ONLY on X_train (a bootstrap/CV training partition); apply to X_apply.
    This is the single place that enforces the in-fold rule, never
    fit a scaler on the full dataset and then resample.
    """
    scaler = StandardScaler().fit(X_train)
    return scaler.transform(X_train if X_apply is None else X_apply), scaler