# BARCODE Clustering

- [Overview](#overview)
- [Terminology](#terminology)
- [Where to Find It](#where-to-find-it)
- [Settings](#settings)
  - [Clustering Settings](#clustering-settings)
  - [Configuration File Settings](#configuration-file-settings)
  - [Cluster Space Modes](#cluster-space-modes)
- [How It Works](#how-it-works)
- [Data Handling](#data-handling)
- [Outputs](#outputs)
  - [Output Location](#output-location)
  - [Figures](#figures)
  - [Tables](#tables)
  - [Run Record](#run-record)
- [Reading the Output](#reading-the-output)
- [Requirements and Runtime](#requirements-and-runtime)

# Overview

The clustering feature groups video channels that have similar BARCODE metrics, so that
distinct material regimes present in a dataset can be identified without supplying any labels.

It is a toggle inside BARCODE's existing windows rather than a separate program. It uses the
metrics the active window has just computed or loaded, and does not re-read videos or re-run
any of the three analysis branches.

Every run reports the clusters together with a measure of how reliable they are. A dataset of
50-200 channels described by up to 25 metrics is small enough that a single clustering run can
easily reflect which videos happened to be collected rather than any real structure. Each run
therefore re-clusters many resampled subsets of the data and reports whether the same groups
keep appearing. If they do not, the correct conclusion is that the dataset does not support a
grouping, and the output shows that clearly.

# Terminology

**Regime** — a distinct mode of behaviour the material can be in, for example a connected
network with large slow-moving islands versus a fragmented, fast-moving one. Mechanically a
regime is a cluster; the word carries the claim that the group corresponds to a real physical
state rather than an arbitrary statistical grouping.

**Partition** — one specific assignment of every channel to a group. A single run of the
clustering algorithm produces a partition. On its own it says nothing about how much that
assignment can be trusted, which is why the stability measures below are reported with it.

**Noise channel** — a channel that does not fall inside any dense region of the metric space.
HDBSCAN does not force every channel into a cluster, so these are labelled `-1` rather than
being assigned to the nearest group. Some channels are genuinely transitional, and a high
noise fraction is a property of the dataset rather than a failure.

**Co-association** — for a pair of channels, the fraction of resampling rounds that placed them
in the same cluster, counting only rounds in which both channels were drawn.

# Where to Find It

The same controls appear in both windows, under a **Clustering** heading:

| Window | Page | Tab | Metrics come from |
| - | - | - | - |
| Generate | Process Data | Execution Settings | The analysis that just ran, held in memory |
| Analyze | Analyze BARCODE Data | Barcode Generator & CSV Aggregator | The barcode CSV files selected on that tab |

On the Analyze page, clustering runs on whichever CSV files are selected in the CSV chooser. It
is independent of the aggregate CSV and comparison barcode options, so it can be run without
producing either.

# Settings

## Clustering Settings

| Setting Name | Description | Limits | Default Value |
| - | - | - | - |
| Generate clustering information | Master toggle. When off, no clustering is performed and no clustering output is written. | (On, Off) | Off |
| Cluster Space | Selects how the metric space is reduced before clustering; see [Cluster Space Modes](#cluster-space-modes). | (pca, subset, full) | pca |
| PCA Cumulative Variance | Fraction of total variance the retained principal components must explain. Used in `pca` mode only. Low values risk discarding an axis that carries the cluster separation. | (0.5, 0.99) | 0.9 |
| Select Metrics | Metrics to cluster on, used in `subset` mode only. At least 2 must be selected. Selecting Mean Flow Direction applies its circular encoding automatically. | 2+ metrics | none selected |
| Minimum Cluster Size | The smallest number of channels that will be treated as a regime. This is a choice about what counts as a group rather than a tuning parameter. | (2, 100) | 5 |
| Stability Resamples | Number of resampled re-clusterings used to test whether the clusters survive. Higher values give a less noisy estimate but take longer, and this dominates the runtime. | (20, 1000) | 200 |

## Configuration File Settings

These are written to each run's `Settings.yaml` and can be edited there. They are not shown in
the interface.

| Setting Name | Description | Default Value |
| - | - | - |
| `pca_n_components` | Explicit number of principal components to retain. When `0`, the PCA Cumulative Variance threshold is used instead. | 0 |
| `min_samples` | HDBSCAN's `min_samples` parameter. When `0`, the library default (equal to `min_cluster_size`) is used. | 0 |
| `direction_handling` | How the circular Mean Flow Direction metric is treated. `encode` replaces it with `(cos, sin)` columns; `drop` removes it. Other values leave it as a linear coordinate, which is not recommended. | encode |
| `min_channels` | Minimum number of usable channels required before clustering will run. | 10 |
| `random_seed` | Seed for resampling and mixture fitting, so runs can be reproduced. | 0 |

## Cluster Space Modes

The three modes share the same clustering and stability code and differ only in how the
standardized metric matrix is reduced beforehand.

| Mode | What it does | When to use it | What to watch for |
| - | - | - | - |
| pca | Principal component analysis on the correlation matrix, keeping components up to the cumulative variance threshold. | The default. Gives a low-dimensional space suited to this sample size without using any target variable. | PCA is linear, and a high-variance component is not necessarily the one that separates the clusters. The run record reports whether the structure appears in the leading components or only after a later one is included. |
| subset | Clusters using only the metrics you select. No rotation, so the axes stay interpretable. | When you have a specific hypothesis about which metrics distinguish the regimes. | Nothing forces the chosen metrics to carry cluster structure. A poor choice shows up in the noise fraction and stability, not in the effective dimensionality. |
| full | Clusters using every metric that survives the checks below. | Comparison and diagnostics. | With 25 dimensions and this many channels, distances concentrate and density becomes hard to estimate. Treat results as a diagnostic rather than a primary answer. |

# How It Works

```
metrics (in memory, or loaded from barcode CSVs)
   -> one row per channel, with file and channel IDs held aside as metadata
   -> drop flagged channels and unusable metric columns, encode circular direction
   -> standardize (fit within each resampling fold, not once globally)
   -> reduce (pca | subset | full)
   -> cluster (HDBSCAN, then a Gaussian mixture seeded by the HDBSCAN cluster count)
   -> stability (repeated re-clustering of resampled subsets)
   -> report
```

Clustering uses HDBSCAN, which varies the density threshold across the space and extracts
clusters by persistence. This handles clusters of differing density and leaves sparse channels
labelled as noise instead of forcing them into a group. A Gaussian mixture seeded with the
HDBSCAN cluster count then provides soft memberships, and a separate BIC sweep over component
counts is recorded as a cross-check on the number of clusters.

Stability is measured by re-running the whole standardize, reduce and cluster sequence on many
random subsets of the channels: 80% of them per round by default, drawn without replacement.
For each pair of channels the analysis records the fraction of rounds that placed them in the
same cluster, counting only rounds where both were drawn. Groups that reflect real structure
survive regardless of which channels are held out. Groups that depend on a few particular
videos break up when those videos are dropped.

Standardization is fit inside each round rather than once on the full dataset. Fitting it
globally would let the held-out channels influence every round through the shared mean and
variance, which would make the stability estimate optimistic.

Subsampling without replacement is used rather than sampling with replacement, because
duplicate rows produce zero-distance pairs that distort HDBSCAN's density estimates.

# Data Handling

- **Each channel is one sample.** Many useful BARCODE distinctions are between channels of the
  same video, so channels are not merged. Enabling Parse All Channels roughly doubles the
  sample count and is recommended.
- **File and channel identity are metadata, not features.** They are held aside from the metric
  matrix and reattached to the output, so every clustered point can be traced to its source and
  the video-provenance comparison below is possible.
- **Channels from the same video are not independent measurements.** They image the same
  physical sample. Clustering does not assume independence, so the grouping is valid, but a
  later significance test must not treat 2n channels as 2n independent samples.
- **Flagged channels are dropped.** Channels flagged dim (flag 1) or saturated (flag 2) are
  excluded. Flags 3 and 4 indicate a correlation length exceeding the field of view, which
  affects one metric rather than the whole channel, so those channels are kept.
- **Unusable metrics are dropped.** Metric columns that are `NaN` across the whole dataset, such
  as a branch that was not run or a v1-format CSV missing newer metrics, are removed first. Any
  channel still containing a `NaN` is then removed. Everything dropped is listed in the run
  record.
- **One unit system is used.** Clustering always uses the dimensionless (% of FOV) metrics, so
  the two unit versions of the area metrics cannot be mixed. Note that a CSV written with Output
  Unit Conversion enabled stores only the µm² values, so its six area metrics will be dropped as
  unusable.
- **Mean Flow Direction is treated as circular.** An angle of -3.1 and one of +3.1 radians point
  in almost the same direction but sit at opposite ends of a linear axis, so the metric is
  encoded as `(cos, sin)` before any distance is calculated.
- **Minimum sizes are enforced.** Clustering is skipped, with a message in the log, if fewer than
  `min_channels` usable channels remain, if fewer than two metric columns survive, or if `subset`
  mode has fewer than two metrics selected.

# Outputs

## Output Location

| Window | Location |
| - | - |
| Generate | `<dataset name> Clustering/`, beside the Summary CSV |
| Analyze | `BARCODE Clustering/`, beside the first selected CSV file |

## Figures

| File | What it shows |
| - | - |
| Co-association Matrix.png | Every channel pair, with both axes listing the same channels sorted by cluster. Colour is the fraction of resampling rounds that placed the pair in the same cluster. |
| Cluster Dendrogram.png | Tree built from the co-association values, with leaves labelled by file name and channel. Two branches join at a distance of `1 - (fraction of rounds together)`. |
| Stability Summary.png | Three panels: how many clusters each round found, how consistently each channel was assigned, and what fraction of channels were labelled noise in each round. |
| Cluster Ordered Barcode.png | BARCODE's colorized barcode with rows sorted by cluster, so the metric fingerprints of each group can be compared side by side. |
| Cluster Scatter.png | Two-dimensional view of the channels coloured by cluster, noise in grey. Uses PC1 and PC2 in `pca` mode and a two-component projection for display otherwise. |

## Tables

| File | Contents |
| - | - |
| Clustering Assignments.csv | One row per channel: file path, channel, dataset, BARCODE flags, assigned cluster, noise flag, HDBSCAN membership probability, outlier score, per-channel stability, and the mixture model's membership value for each cluster. |
| Cluster Roster.csv | The same assignments grouped by cluster and sorted by stability, so each group's most representative members come first. The quickest way to see which files ended up together. |
| PCA Loadings.csv | `pca` mode only. Each retained component's loading on every original metric, with its explained variance. A component is a combination of metrics rather than a metric, so the loadings are what make the clusters interpretable. |
| Cluster Ordered Barcode Row Order.csv | Maps each row of the ordered barcode image to its cluster, file and channel. |

## Run Record

`Clustering Run Metadata.json` contains what is needed to reproduce and check the run:

- all settings used, including the random seed
- which channels and metric columns were dropped, and why
- the reduction: mode, number of components, cumulative variance, eigenvalues, the Kaiser
  cross-check, and the effective dimensionality of the space
- the clustering summary: number of clusters, noise fraction, and the mixture model's
  BIC-preferred count for comparison with the number HDBSCAN found
- the stability summary: number of rounds, the distribution of cluster counts, and mean
  per-channel stability
- the video-provenance comparison: whether a video's channels ended up in the same cluster or
  in different ones, measured against the cross-video baseline
- in `pca` mode, whether the structure appears in the leading components or only once a
  lower-variance component is included, and how reproducible the leading components are across
  rounds

# Reading the Output

**Co-association matrix.** Both axes list the same channels in the same order. A cell is one
pair, and its colour is how often that pair clustered together. Because the channels are sorted
so that cluster-mates are adjacent, each cluster's members occupy a block of rows and the same
block of columns, which is why solid groups appear as bright squares on the diagonal. Bright
blocks with dark space between them mean the groups are reproducible. A mottled image with no
clear blocks means the dataset does not support a grouping, and the cluster labels should not be
reported as a result.

**Cluster count distribution.** This counts rounds, not channels. Each round produces its own
answer to how many clusters are present, and the chart tallies those answers. A single tall bar
means every round agreed. Bars spread over two or three values mean the number of regimes is
itself uncertain, which a single clustering run would not have revealed.

**Per-channel stability.** How consistently each channel was grouped with the same companions.
Values near 1 are reliable assignments. A long tail towards 0 means some channels move between
groups depending on which subset is drawn.

**Dendrogram.** The leaves are file names, so this is the figure that shows which videos group
together. Two channels joining near distance 0 were clustered together in nearly every round. A
leaf that only joins at distance 1.0 never grouped with anything and is effectively unassigned.

**Cluster ordered barcode.** Read down each metric column and look for colour that changes at
the cluster boundaries: those metrics are the ones separating the groups. A column that looks
the same from top to bottom does not distinguish them.

**Agreement with an external label.** The clustering is given no labels, so if the resulting
groups line up with an experimental condition, that is an observation about the data, not a
confirmation that the method worked.

# Requirements and Runtime

Clustering requires **scikit-learn** 1.5 or newer, which is included in `requirements.txt`:

```
pip install -r requirements.txt
```

HDBSCAN, the Gaussian mixture, PCA and the scaler all come from scikit-learn, so no additional
compiled clustering package is needed.

The stability resampling dominates the runtime: at the default of 200 rounds the analysis is
repeated 200 times. It runs on BARCODE's existing background thread, so the interface stays
responsive, and progress is printed to the Processing Log. Lowering Stability Resamples
shortens the run at the cost of a noisier estimate; values below about 100 are not recommended.
