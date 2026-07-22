# BARCODE Feature Importance

- [Overview](#overview)
- [Terminology](#terminology)
- [Where to Find It](#where-to-find-it)
- [Input Format](#input-format)
- [Settings](#settings)
  - [Feature Importance Settings](#feature-importance-settings)
  - [Settings Not Shown in the Interface](#settings-not-shown-in-the-interface)
- [How It Works](#how-it-works)
- [Data Handling](#data-handling)
- [Outputs](#outputs)
- [Reading the Output](#reading-the-output)
- [Requirements and Runtime](#requirements-and-runtime)

# Overview

Feature importance is the labelled counterpart of [clustering](CLUSTERING_README.md). Clustering
asks which channels behave alike without being told anything; feature importance is given a
ground truth for every channel and asks **which of the 25 BARCODE metrics actually drive it**.

It takes a normal BARCODE summary CSV with one extra column appended holding the answer for each
row, fits a random forest, and ranks every metric by how much the forest depends on it. Three
independent ranking methods are computed rather than one, because they fail in different ways and
their agreement is the real result.

Two branches share the same machinery:

- **Classification** — the target is a category (a condition, a material name, a treatment).
- **Regression** — the target is a continuous number (a measured modulus, a concentration, a time).

# Terminology

**Target** — the ground-truth column appended to the CSV. One value per row, i.e. per channel.

**Impurity importance** — how much each metric reduced node impurity while the trees were grown.
Free to compute, but biased toward metrics that take many distinct values, and it describes how
the model was *built* rather than what it actually needs.

**Permutation importance** — shuffle one metric's column in the held-out data and measure how far
the score falls. This measures what the fitted model genuinely relies on, on data it has not seen.
It is the default ranking for that reason. It can be **negative**: a metric the model would be
better off without.

**SHAP importance** — the mean absolute SHAP value per metric, which attributes each individual
prediction to each metric and then averages the magnitudes. Optional; see
[Requirements](#requirements-and-runtime).

**Group split** — train and test are divided by video, never by row. See
[Data Handling](#data-handling) for why this is not optional.

# Where to Find It

| Window | Page | Tab | Data comes from |
| - | - | - | - |
| Analyze | Analyze BARCODE Data | Feature Importance | The single CSV selected on that tab |

There is no Process-window equivalent. Unlike clustering, this needs ground-truth labels, which do
not exist while videos are still being processed.

Select the CSV, choose the branch and the target column, then press **Analyze BARCODE Data**. The
chart appears in the tab and the full result is written next to the CSV.

# Input Format

A standard BARCODE summary CSV — `File, Channel, Flags`, then the metric columns — with **one
extra column appended** holding the target:

```
File,Channel,Flags,Connectivity,...,Curl,condition
/data/gel_01.nd2,0,0,12.4,...,0.31,soft
/data/gel_01.nd2,1,0,11.9,...,0.28,soft
/data/gel_02.nd2,0,0,48.2,...,0.77,stiff
```

The column may be named anything. The tab reads the header and offers every column BARCODE did not
write itself as a candidate, so no naming convention is imposed. Rows with a blank target are
skipped rather than failing the run.

Both unit conventions are accepted (%-of-FOV and the `* Quantity` physical-unit columns). Several
CSVs can be combined if they share a metric schema.

> BARCODE's own CSV reader rejects a file with an extra column, because it checks the header
> against an exact list. This module parses the header itself, which is why the appended column
> works here and nowhere else in the program.

# Settings

## Feature Importance Settings

| Setting Name | Description | Limits | Default Value |
| - | - | - | - |
| Select CSV File | The BARCODE CSV carrying the target column. | one `.csv` | none |
| Model Type | Which branch to run. Choosing `regression` for a text column is an error, not a guess. | (classification, regression) | classification |
| Target Column | The ground truth to predict. Populated from the chosen CSV's header. | a column in the file | first candidate |
| Tune hyperparameters | Search for better forest settings by cross-validation before fitting. Roughly 150 extra model fits, and by far the slowest part of a run. | (On, Off) | Off |
| Include SHAP importance | Adds the third ranking. Disabled, and labelled as such, when the `shap` package is not installed. | (On, Off) | On |
| Test Fraction | Share of the data held back to score the model. Held out by video, not by row. | (0.1, 0.5) | 0.25 |
| Random Seed | Fixes the split and the forest so a run reproduces exactly. | (0, 99999) | 42 |

## Settings Not Shown in the Interface

These live on `FeatureImportanceConfig` in `core/config.py` and are reachable in code. They follow
the same precedent as clustering's unexposed parameters.

| Setting Name | Description | Default Value |
| - | - | - |
| `n_estimators` | Trees in the forest when tuning is off. | 500 |
| `permutation_repeats` | Shuffles per metric for permutation importance. More is less noisy and slower. | 20 |
| `group_by_file` | Hold out whole videos rather than individual rows. Turning this off inflates the reported score. | True |
| `min_samples` | Minimum labelled rows required before the analysis will run at all. | 20 |
| `sort_by` | Which ranking orders the chart: `permutation`, `impurity`, or `shap`. Falls back automatically if the chosen method produced nothing. | permutation |

# How It Works

```
one barcode CSV + an appended target column
   -> one row per channel, with File/Channel/Flags held aside as metadata
   -> drop flagged channels and rows with no target, encode circular direction
   -> split into train/test BY VIDEO
   -> fit column drops and median imputation on the TRAINING ROWS ONLY
   -> fit a random forest (optionally hyperparameter-tuned with grouped CV)
   -> score on the held-out videos
   -> impurity + permutation + SHAP importances, and their rank agreement
   -> report
```

Three importance methods are computed because no single one is trustworthy alone. Impurity is
biased by construction, permutation is unbiased but noisy on small held-out sets, and SHAP is
neither but is the most expensive. Where all three put the same metric on top, that metric matters.
Where they disagree, the Spearman table says so explicitly rather than letting one arbitrary
ordering look authoritative.

# Data Handling

**Channels are not independent samples.** A video contributes one row per channel, and those rows
image the same physical sample. Splitting them across train and test puts near-duplicates on both
sides, and the model is then scored partly on data it has effectively already seen. Train/test is
therefore split by video, and the tuning cross-validation folds are grouped the same way. This
makes the reported score lower and honest. The caveat is written into every run's metrics file.

Grouping and class-balancing cannot both be guaranteed. For classification the split is retried a
few times looking for one that leaves every class represented in training; only if none does will
it fall back to a row-wise split, and it says so in the log and the run record when it happens.

**Cleaning is fitted on training rows only.** Which metric columns are too sparse to use, and the
medians used to fill the remaining gaps, are both computed from the training rows and then applied
to the held-out ones. Computing them over the whole dataset first would leak test information into
the model and quietly inflate the score.

**Flagged channels.** Flags 1 (dim) and 2 (saturated) drop the channel, matching clustering. Flags
3 and 4 cap a single metric rather than invalidating the channel, so those rows are kept.

**Sparse metric columns.** A metric that is missing for more than 20% of the training rows is
dropped as a column rather than being allowed to take all those rows down with it. At these sample
sizes losing one metric is far cheaper than losing half the dataset.

**Mean Flow Direction** is an angle in radians, where 0 and 2π are the same state. It is replaced
by `Mean Flow Direction (cos)` and `Mean Flow Direction (sin)`, the same encoding and naming
clustering uses, so a tree never treats the wrap-around as a large numeric gap.

# Outputs

Written to `BARCODE Feature Importance/` beside the input CSV.

| File | Content |
| - | - |
| `Feature Importances.png` | The grouped bar chart, every metric, one bar per method |
| `Feature Importances.csv` | Per-metric values and per-method ranks |
| `Ranking Agreement (Spearman).csv` | Rank correlation between the methods |
| `Model Evaluation Metrics.txt` | Held-out score, split strategy, class list, and the independence caveat |
| `Predictions by File.csv` | Per row: File, Channel, Split, Actual, Predicted (and Residual for regression) |
| `Tuning Results.txt` | Chosen parameters — only when tuning was on |
| `Feature Importance Run Metadata.json` | Full configuration and data-handling record for the run |

The tab shows the strongest 15 metrics so the labels stay readable; the saved PNG always carries
every metric.

# Reading the Output

**Check the score first.** A feature-importance ranking from a model that cannot predict the target
is a ranking of noise. Test accuracy near chance, or a test R² near zero or negative, means the
chart should not be interpreted at all. That number is at the top of
`Model Evaluation Metrics.txt` for exactly this reason.

**Bars are normalised per method,** each to its own largest magnitude, because the three methods are
on incomparable scales. Only the shape within a method is meaningful, never the absolute height
across methods.

**A negative permutation bar** means shuffling that metric *improved* the held-out score — the model
is fitting noise through it. This is information, not an error.

**Read the agreement table.** Two methods correlating at 0.9 means the ordering is stable. At 0.4 it
means the top few are probably real and the rest is arbitrary.

**A metric that is important is not a metric that is causal.** Two correlated metrics split the
credit between them almost arbitrarily, so a low bar does not prove a metric is irrelevant — it may
just be redundant with one that scored higher.

**Change the seed and re-run.** If the top of the chart moves, the dataset is too small to pin the
ordering down, and that is the finding.

# Requirements and Runtime

Feature importance requires **scikit-learn** and **scipy**, both already in `requirements.txt`:

```
pip install -r requirements.txt
```

No new dependency is introduced. The forest, the grouped splitters, permutation importance and the
rank statistics all come from scikit-learn and scipy.

**SHAP is optional.** It is not in `requirements.txt` because the other two rankings are sufficient
and `shap` pulls in a compiled toolchain. Without it the checkbox is disabled and labelled, the SHAP
column is left empty, and the chart falls back to ordering by permutation importance. To enable it:

```
pip install shap
```

A default run on a few hundred rows takes seconds. Turning on **Tune hyperparameters** is roughly
150 extra model fits and dominates the runtime; leave it off for a first look. Everything runs on
BARCODE's existing background thread, so the interface stays responsive, and progress is printed to
the Processing Log.
