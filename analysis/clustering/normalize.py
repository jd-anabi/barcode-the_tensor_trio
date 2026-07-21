"""Window-specific ingestion: BARCODE results -> a common channel-level table.

Both BARCODE windows converge on `List[ChannelResults]`:
  - Generate: `all_results` already in memory (core/pipeline.run_analysis).
  - Analyze:  loaded from existing barcode CSV(s) via read_csv_to_channel_results.

`ChannelTable` is the shared schema every later stage consumes. We deliberately
use plain numpy + lists: the sample count is small and the downstream
math (StandardScaler/PCA/HDBSCAN) all speaks ndarray anyway.
"""

import os
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from core import ChannelResults
from utils.reader import read_csv_to_channel_results


@dataclass
class ChannelTable:
    """Channel-level metric table shared by both windows. One row = one channel
    (i.e., the clustering sample). IDs are metadata, NEVER features.

    `X` holds the 25 %-FOV metrics via ChannelResults.get_data(just_metrics=True)
    and MAY contain NaN (unused branches, v1 CSVs, physical-unit CSVs where the
    %-FOV area columns are absent). NaN cleanup happens later in `prepare`, not here.
    """

    X: np.ndarray             # (n_channels, n_features) raw %-FOV metric values
    feature_names: List[str]  # length n_features
    video_ids: List[str]      # length n_channels (the source filepath; groups a video's channels)
    channel_ids: List[int]    # length n_channels
    flags: List[str]          # length n_channels (BARCODE Flags string, e.g. "0", "1;3")
    datasets: List[str]       # length n_channels (external dataset label)

    @property
    def n_channels(self) -> int:
        return self.X.shape[0]

    @property
    def n_features(self) -> int:
        return self.X.shape[1]

    def __repr__(self) -> str:
        return (
            f"ChannelTable(n_channels={self.n_channels}, n_features={self.n_features}, "
            f"videos={len(set(self.video_ids))}, datasets={sorted(set(self.datasets))})"
        )


def to_channel_table(results: List[ChannelResults], dataset: str = "") -> ChannelTable:
    """Build a ChannelTable from a list of ChannelResults (window-agnostic)."""
    feature_names = ChannelResults.get_headers(just_metrics=True)

    rows, video_ids, channel_ids, flags, datasets = [], [], [], [], []
    for r in results:
        # Read the flag string BEFORE get_data(): get_data() recomputes total_flags
        # from the per-branch flags, which are all-zero on CSV-loaded results and
        # would clobber the real flags string that came from the CSV.
        flags.append(r.total_flags)
        rows.append(r.get_data(just_metrics=True))
        video_ids.append(r.filepath)
        channel_ids.append(int(r.channel))
        datasets.append(dataset)

    X = (
        np.array(rows, dtype=float)
        if rows
        else np.empty((0, len(feature_names)), dtype=float)
    )
    return ChannelTable(X, feature_names, video_ids, channel_ids, flags, datasets)


def combine_tables(tables: List[ChannelTable]) -> ChannelTable:
    """Vertically stack ChannelTables that share the same feature schema."""
    names = ChannelResults.get_headers(just_metrics=True)
    tables = [t for t in tables if t.n_channels > 0]
    if not tables:
        return ChannelTable(np.empty((0, len(names)), float), names, [], [], [], [])

    for t in tables:
        assert t.feature_names == names, "Cannot combine tables with different feature schemas"

    X = np.vstack([t.X for t in tables])
    cat = lambda attr: [v for t in tables for v in getattr(t, attr)]
    return ChannelTable(
        X, names, cat("video_ids"), cat("channel_ids"), cat("flags"), cat("datasets")
    )


def normalize_generate(all_results: List[ChannelResults], dataset: str = "") -> ChannelTable:
    """Generate window: metrics already in memory as List[ChannelResults]."""
    return to_channel_table(all_results, dataset=dataset)

def load_analyze(csv_paths: List[str], dataset: Optional[str] = None):
    """Analyze window: return (ChannelTable, List[ChannelResults]) from barcode CSV(s).

    Each CSV becomes its own dataset label (its filename stem) unless `dataset` overrides.
    """
    tables, results = [], []
    for path in csv_paths:
        r = read_csv_to_channel_results(path)
        label = dataset if dataset is not None else os.path.splitext(os.path.basename(path))[0]
        tables.append(to_channel_table(r, dataset=label))
        results.extend(r)
    return combine_tables(tables), results


def normalize_analyze(csv_paths: List[str], dataset: Optional[str] = None) -> ChannelTable:
    """Analyze window: load existing barcode CSV(s) into the common table."""
    return load_analyze(csv_paths, dataset)[0]