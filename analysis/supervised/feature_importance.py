"""Random-Forest model plus the three importance rankings. No file I/O, no globals.

Classification and regression differ in exactly three places, which is why they share one
module rather than two:
  1. target preparation   (encode class labels  vs  a numeric continuous target)
  2. the estimator        (RandomForestClassifier vs RandomForestRegressor)
  3. the scoring metric   (accuracy              vs  R^2 / MAE / RMSE)
Everything else — cleaning, the split, and all three importance methods — is identical.

Three importances are computed because they disagree in informative ways. Impurity is
free but biased toward high-cardinality features; permutation measures what the fitted
model actually uses on held-out data; SHAP attributes per-sample. Agreement between them
is evidence; disagreement is a warning, which is why the Spearman table is reported too.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import rankdata

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    RandomizedSearchCV,
    train_test_split,
)
from sklearn.preprocessing import LabelEncoder

from analysis.supervised.load import FeatureImportanceError

# SHAP is optional; everything else still runs without it.
try:
    import shap

    HAVE_SHAP = True
except Exception:
    HAVE_SHAP = False

CLASSIFICATION = "classification"
REGRESSION = "regression"

SORT_METHODS = ("permutation", "impurity", "shap")

NON_INDEPENDENCE_CAVEAT = (
    "Channels of one video image the same physical sample and are NOT independent draws. "
    "Train and test are therefore split by video, not by row: had they been split by row, "
    "near-duplicate channels would appear on both sides and the reported score would be "
    "optimistic. Even so, no downstream significance test may treat 2n channels as 2n "
    "independent samples."
)


@dataclass
class SupervisedResult:
    """Everything a report or a GUI needs, with no dependency on where it came from."""

    model_type: str
    target_name: str
    feature_names: List[str]

    impurity: np.ndarray            # (n_features,)
    permutation: np.ndarray         # (n_features,)
    permutation_std: np.ndarray     # (n_features,)
    shap: np.ndarray                # (n_features,) — all-NaN when SHAP did not run
    shap_ok: bool

    sort_by: str                    # the method actually used, after fallback
    metrics: dict                   # OOB / accuracy, or R^2 / MAE / RMSE
    classes: Optional[List[str]]
    predictions: List[dict]         # per row: File, Channel, Split, Actual, Predicted[, Residual]
    split_meta: dict
    prepare_report: dict = field(default_factory=dict)
    best_params: Optional[dict] = None

    @property
    def values(self) -> dict:
        return {"impurity": self.impurity, "permutation": self.permutation, "shap": self.shap}

    def ranks(self) -> dict:
        """1 = most important, per method. NaN stays NaN — an absent method ranks nothing."""
        return {name: _ranks(vals) for name, vals in self.values.items()}

    def order(self) -> np.ndarray:
        """Feature indices sorted by the active method, most important first."""
        active = self.values[self.sort_by]
        return np.argsort(np.where(np.isnan(active), -np.inf, active))[::-1]


def _ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    out = np.full(values.shape, np.nan)
    finite = np.isfinite(values)
    if finite.any():
        out[finite] = rankdata(-values[finite], method="min")
    return out


# ---------------- TARGET PREPARATION (differs by mode) ----------------
def prepare_target(y_raw: Sequence[str], model_type: str) -> Tuple[np.ndarray, Optional[List[str]]]:
    """Return (y, classes). `classes` is None for regression."""
    if model_type == CLASSIFICATION:
        encoder = LabelEncoder()
        y = encoder.fit_transform([str(v) for v in y_raw])
        return y, list(encoder.classes_)

    y = np.array([_to_float_strict(v) for v in y_raw], dtype=float)
    if np.isnan(y).all():
        raise FeatureImportanceError(
            "No row has a numeric value in the target column. Regression needs a "
            "continuous target — did you mean to run the classification branch?")
    return y, None


def _to_float_strict(value) -> float:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return np.nan


# ---------------- SPLIT ----------------
def make_split(y: np.ndarray, groups: Sequence[str], model_type: str,
               test_size: float, seed: int, group_by_file: bool) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Hold out whole videos, not rows.

    A video contributes one row per channel, and those rows are near-duplicates. Splitting
    by row puts a video's channels on both sides, so the model is scored partly on data it
    has effectively already seen. GroupShuffleSplit keeps each video whole.

    Grouping and stratification cannot both be guaranteed, so for classification we retry a
    few group splits looking for one that leaves every class represented in training, and
    only fall back to a stratified row split (loudly) if none does.
    """
    n = len(y)
    is_clf = model_type == CLASSIFICATION
    n_groups = len(set(groups))
    meta = {"n_samples": n, "n_groups": n_groups, "test_size": float(test_size)}

    if group_by_file and n_groups >= 2:
        for attempt in range(10):
            splitter = GroupShuffleSplit(n_splits=1, test_size=test_size,
                                         random_state=seed + attempt)
            train_idx, test_idx = next(splitter.split(np.zeros(n), y, groups=np.asarray(groups)))
            if test_idx.size == 0 or train_idx.size == 0:
                continue
            if is_clf and len(set(y[train_idx].tolist())) < 2:
                continue  # a one-class training set cannot be fit
            meta.update(strategy="group", grouped_by="File", attempts=attempt + 1)
            if is_clf:
                unseen = set(y.tolist()) - set(y[train_idx].tolist())
                if unseen:
                    meta["classes_absent_from_training"] = sorted(unseen)
            return train_idx, test_idx, meta

        print("Note: no split by video left every class in the training set; falling back "
              "to a row-wise stratified split. Scores will be optimistic because channels "
              "of the same video appear on both sides.")
        meta["fallback_reason"] = "group split could not populate the training classes"
    elif group_by_file:
        meta["fallback_reason"] = f"only {n_groups} distinct video(s) — nothing to group by"

    stratify = None
    if is_clf:
        counts = np.bincount(y)
        if counts.size and counts[counts > 0].min() >= 2:
            stratify = y
    train_idx, test_idx = train_test_split(
        np.arange(n), test_size=test_size, random_state=seed, stratify=stratify)
    meta.update(strategy="row", stratified=stratify is not None)
    return train_idx, test_idx, meta


# ---------------- MODEL (differs by mode) ----------------
def build_model(model_type: str, X_train: np.ndarray, y_train: np.ndarray,
                groups_train: Optional[Sequence[str]] = None, *,
                n_estimators: int = 500, tuning: bool = False, seed: int = 42,
                progress=None) -> Tuple[object, Optional[dict], Optional[float]]:
    """Fit the forest. Returns (model, best_params, best_cv_score).

    OOB scoring is requested but not required — sklearn silently declines to set
    `oob_score_` when there are too few bootstrap samples, so every reader of it must
    check first.
    """
    is_clf = model_type == CLASSIFICATION
    Model = RandomForestClassifier if is_clf else RandomForestRegressor
    base = Model(n_estimators=n_estimators, random_state=seed, oob_score=True, n_jobs=-1)

    if not tuning:
        base.fit(X_train, y_train)
        return base, None, None

    param_dist = {
        "n_estimators": [50, 100, 200, 300, 500, 800],
        "max_depth": [None, 5, 10, 20, 30, 50],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", None],
    }
    scoring = "accuracy" if is_clf else "neg_mean_absolute_error"

    # The CV folds inherit the same non-independence problem as the train/test split.
    fit_kwargs = {}
    n_groups = len(set(groups_train)) if groups_train is not None else 0
    if groups_train is not None and n_groups >= 2:
        cv = GroupKFold(n_splits=min(5, n_groups))
        fit_kwargs["groups"] = np.asarray(groups_train)
    else:
        cv = min(5, max(2, len(y_train) // 2))

    if progress:
        progress("tuning hyperparameters (this is the slow step)")
    # verbose=0 on purpose: joblib writes its progress to a stream the GUI log never sees.
    search = RandomizedSearchCV(base, param_dist, n_iter=30, scoring=scoring, cv=cv,
                                random_state=seed, n_jobs=-1, verbose=0)
    search.fit(X_train, y_train, **fit_kwargs)

    model = search.best_estimator_
    if not hasattr(model, "oob_score_"):
        # RandomizedSearchCV refits on the full training set, but a tuned configuration
        # can leave too few bootstrap samples for OOB. Ask again explicitly; if sklearn
        # still declines, evaluate() simply omits the OOB line.
        model.set_params(oob_score=True)
        model.fit(X_train, y_train)
    return model, dict(search.best_params_), float(search.best_score_)


# ---------------- EVALUATION (differs by mode) ----------------
def evaluate(model, model_type: str, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Held-out performance. Nothing below this score is worth interpreting.

    A feature-importance ranking from a model that cannot predict the target is a ranking
    of noise, so these numbers belong next to the chart, not in a separate file only.
    """
    metrics = {}
    if hasattr(model, "oob_score_"):
        metrics["oob_score"] = float(model.oob_score_)

    if model_type == CLASSIFICATION:
        metrics["test_accuracy"] = float(model.score(X_test, y_test))
    else:
        predicted = model.predict(X_test)
        metrics["test_r2"] = float(r2_score(y_test, predicted))
        metrics["mae"] = float(mean_absolute_error(y_test, predicted))
        metrics["rmse"] = float(np.sqrt(mean_squared_error(y_test, predicted)))
    return metrics


def shap_importance(model, X_test: np.ndarray) -> np.ndarray:
    """mean(|SHAP value|) per feature, averaged over ALL classes for a classifier."""
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(X_test)
    if isinstance(values, list):                 # one matrix per class (older API)
        magnitudes = np.mean([np.abs(v) for v in values], axis=0)
    else:
        magnitudes = np.abs(values)
        if magnitudes.ndim == 3:                 # (samples, features, classes)
            magnitudes = magnitudes.mean(axis=2)
    return magnitudes.mean(axis=0)


def resolve_sort_by(requested: str, values: dict) -> str:
    """Fall back when the requested method produced nothing usable.

    The original script defaulted to SHAP, which is all-NaN whenever the optional shap
    package is missing — silently making the sort order, and therefore the chart,
    arbitrary. Fail over to a method that actually ran, and say so.
    """
    order = [requested] + [m for m in SORT_METHODS if m != requested]
    for method in order:
        column = values.get(method)
        if column is not None and np.isfinite(column).any():
            if method != requested:
                print(f"Sorting by {method}: '{requested}' importance is unavailable.")
            return method
    raise FeatureImportanceError("No importance method produced a usable result.")


def compute_importances(model, model_type: str, X_test: np.ndarray, y_test: np.ndarray,
                        feature_names: Sequence[str], *, n_repeats: int = 20,
                        seed: int = 42, use_shap: bool = True,
                        progress=None) -> Tuple[dict, bool]:
    """Impurity, permutation and (optionally) SHAP, all aligned to `feature_names`."""
    n_features = len(feature_names)
    scoring = "accuracy" if model_type == CLASSIFICATION else "r2"

    if progress:
        progress(f"permutation importance ({n_repeats} shuffles per feature)")
    perm = permutation_importance(model, X_test, y_test, scoring=scoring,
                                  n_repeats=n_repeats, random_state=seed, n_jobs=-1)

    values = {
        "impurity": np.asarray(model.feature_importances_, dtype=float),
        "permutation": np.asarray(perm.importances_mean, dtype=float),
        "shap": np.full(n_features, np.nan),
    }
    stds = np.asarray(perm.importances_std, dtype=float)

    shap_ok = False
    if use_shap and HAVE_SHAP:
        if progress:
            progress("SHAP values")
        try:
            values["shap"] = np.asarray(shap_importance(model, X_test), dtype=float)
            shap_ok = True
        except Exception as exc:
            print(f"SHAP failed ({exc}); continuing with the other two methods.")
    elif use_shap and not HAVE_SHAP:
        print("Note: the `shap` package is not installed, so the SHAP column is empty. "
              "Run `pip install shap` to enable it.")

    return {"values": values, "permutation_std": stds}, shap_ok
