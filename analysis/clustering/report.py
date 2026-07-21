"""Reporting: turn clustering + stability into artifacts a collaborator can read.

Everything here re-attaches the IDs that were held out of the feature matrix,
so every clustered channel traces back to its source video and channel.
"""

import csv
import json
import os
from typing import Optional

import numpy as np

import matplotlib
matplotlib.use("Agg")          # we only ever save PNGs; matches main.py
import matplotlib.pyplot as plt

from analysis.clustering.stability import cluster_order

NON_INDEPENDENCE_CAVEAT = (
    "Channels of one video image the same physical sample and are NOT independent draws. "
    "Clustering makes no independence assumption, so this grouping is valid, but no "
    "downstream significance test may treat 2n channels as 2n independent samples. "
    "Note the dual edge: strong same-video co-clustering may partly reflect shared "
    "acquisition conditions, not only shared material dynamics."
)


def _jsonable(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def video_provenance(table, stability_result) -> dict:
    """Do a video's channels co-cluster (co-moving components) or split (distinct dynamics)?

    Compares co-association for same-video channel pairs against cross-video pairs, and
    flags per video whether all its channels landed in one non-noise cluster. Enabled
    purely by the retained video_id. See NON_INDEPENDENCE_CAVEAT.
    """
    M, co = stability_result.co_association, stability_result.co_occurrence
    vids = np.asarray(table.video_ids)
    ref = stability_result.reference_labels
    n = len(vids)

    same, cross = [], []
    for i in range(n):
        for j in range(i + 1, n):
            if co[i, j] <= 0:
                continue
            (same if vids[i] == vids[j] else cross).append(M[i, j])

    per_video = {}
    for v in set(vids.tolist()):
        idx = np.where(vids == v)[0]
        if idx.size < 2:
            continue
        labs = ref[idx]
        per_video[v] = {
            "n_channels": int(idx.size),
            "co_cluster": bool(np.all(labs >= 0) and len(set(labs.tolist())) == 1),
            "labels": labs.tolist(),
        }

    return {
        "mean_same_video_co_association": float(np.mean(same)) if same else float("nan"),
        "mean_cross_video_co_association": float(np.mean(cross)) if cross else float("nan"),
        "n_same_video_pairs": len(same),
        "n_cross_video_pairs": len(cross),
        "videos_with_multiple_channels": len(per_video),
        "videos_whose_channels_co_cluster": sum(1 for d in per_video.values() if d["co_cluster"]),
        "per_video": per_video,
        "caveat": NON_INDEPENDENCE_CAVEAT,
    }


def write_assignments_csv(path, table, cluster_result, stability_result=None) -> str:
    """One row per channel: IDs + hard label + noise flag + soft responsibilities."""
    resp = cluster_result.gmm_responsibilities
    K = resp.shape[1] if resp.size else 0
    header = ["Filepath", "Channel", "Dataset", "Flags", "Cluster", "Is_Noise",
              "HDBSCAN_Probability", "Outlier_Score", "Per_Point_Stability"] + \
             [f"GMM_P{c}" for c in range(K)]

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for i in range(table.n_channels):
            stab = ""
            if stability_result is not None:
                s = stability_result.per_point_stability[i]
                stab = "" if np.isnan(s) else round(float(s), 4)
            w.writerow([table.video_ids[i], table.channel_ids[i], table.datasets[i],
                        table.flags[i], int(cluster_result.labels[i]),
                        int(cluster_result.labels[i] < 0),
                        round(float(cluster_result.probabilities[i]), 4),
                        round(float(cluster_result.outlier_scores[i]), 4), stab]
                       + [round(float(resp[i, c]), 4) for c in range(K)])
    return path


def write_pca_loadings_csv(path, reduction_meta) -> Optional[str]:
    """PCA only: each retained PC's loading on every original metric.

    A component is a linear combination of all metrics, not a metric — writing the
    loadings is what keeps clusters translatable ("PC1 loads high on connectivity and
    island area, negative on void area -> a connectedness axis").
    """
    if reduction_meta.get("mode") != "pca":
        return None
    loadings = np.asarray(reduction_meta["loadings"], dtype=float)
    names = reduction_meta.get("input_feature_names") or \
        [f"feature_{j}" for j in range(loadings.shape[1])]
    evr = reduction_meta.get("explained_variance_ratio", [])

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Component", "Explained_Variance_Ratio"] + list(names))
        for c in range(loadings.shape[0]):
            ratio = round(float(evr[c]), 6) if c < len(evr) else ""
            w.writerow([f"PC{c + 1}", ratio] + [round(float(v), 6) for v in loadings[c]])
    return path


def write_run_metadata(path, *, prepare_report, reduction_meta, cluster_result,
                       stability_result=None, provenance=None, extra=None) -> str:
    """Everything needed to reproduce and audit the run."""
    meta = {
        "prepare": prepare_report,
        "reduction": reduction_meta,
        "clustering": {
            "n_clusters": int(cluster_result.n_clusters),
            "noise_fraction": float(cluster_result.noise_fraction),
            "gmm_n_components": int(cluster_result.gmm_n_components),
            "gmm_bic_best_k": int(cluster_result.gmm_bic_best_k),
            **cluster_result.meta,
        },
    }
    if stability_result is not None:
        pp = stability_result.per_point_stability
        meta["stability"] = {
            "n_resamples": int(stability_result.n_resamples),
            "cluster_count_distribution": stability_result.cluster_count_distribution,
            "mean_per_point_stability": (float(np.nanmean(pp)) if np.isfinite(pp).any() else None),
            "mean_noise_fraction": float(stability_result.noise_fractions.mean())
            if stability_result.noise_fractions.size else None,
            **stability_result.meta,
        }
    if provenance is not None:
        meta["provenance"] = {k: v for k, v in provenance.items() if k != "per_video"}
    if extra:
        meta.update(extra)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=_jsonable)
    return path

_PURPLE = "#4c2a85"


def plot_co_association(path, stability_result, title="Co-association (bootstrap)") -> str:
    """Heatmap of M reordered by reference cluster. Block structure = the stable grouping;
    a washed-out image means n is too small to support a partition (spec §1.4)."""
    M, ref = stability_result.co_association, stability_result.reference_labels
    order = cluster_order(ref)
    Mo = M[np.ix_(order, order)]

    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    im = ax.imshow(Mo, vmin=0, vmax=1, cmap="plasma", interpolation="nearest")
    labs = np.asarray(ref)[order]
    for b in np.where(np.diff(labs) != 0)[0] + 0.5:
        ax.axhline(b, color="white", lw=0.6)
        ax.axvline(b, color="white", lw=0.6)
    ax.set_title(title)
    ax.set_xlabel("channel (ordered by cluster)")
    ax.set_ylabel("channel (ordered by cluster)")
    fig.colorbar(im, ax=ax, label="fraction of resamples co-clustered")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
    return path


def plot_stability_summary(path, stability_result) -> str:
    """Three panels: is k stable, are points consistently assigned, how much is noise?"""
    st = stability_result
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), dpi=300)

    ks = sorted(st.cluster_count_distribution)
    axes[0].bar([str(k) for k in ks], [st.cluster_count_distribution[k] for k in ks], color=_PURPLE)
    axes[0].set_title("Cluster count across resamples")
    axes[0].set_xlabel("n clusters"); axes[0].set_ylabel("resamples")

    pp = st.per_point_stability[~np.isnan(st.per_point_stability)]
    if pp.size:
        axes[1].hist(pp, bins=20, range=(0, 1), color=_PURPLE)
    else:
        axes[1].text(0.5, 0.5, "all points noise", ha="center", va="center", transform=axes[1].transAxes)
    axes[1].set_title("Per-point stability")
    axes[1].set_xlabel("mean co-association with own cluster")

    if st.noise_fractions.size:
        axes[2].hist(st.noise_fractions, bins=20, color=_PURPLE)
    axes[2].set_title("Noise fraction across resamples")
    axes[2].set_xlabel("fraction labelled noise")

    fig.tight_layout(); fig.savefig(path); plt.close(fig)
    return path


def plot_cluster_scatter(path, X_reduced, labels, reduction_meta=None, title=None) -> str:
    """Low-dimensional view for inspection: PC1–PC2 for PCA mode, else a 2-D
    PCA-for-display. Any agreement with an external label is an observation, not
    something the clustering was told."""
    X = np.asarray(X_reduced, dtype=float)
    meta = reduction_meta or {}
    if meta.get("mode") == "pca" and X.shape[1] >= 2:
        XY = X[:, :2]
        evr = meta.get("explained_variance_ratio", [])
        xl = f"PC1 ({evr[0] * 100:.0f}% var)" if len(evr) > 0 else "PC1"
        yl = f"PC2 ({evr[1] * 100:.0f}% var)" if len(evr) > 1 else "PC2"
    else:
        from sklearn.decomposition import PCA
        XY = (PCA(n_components=2, random_state=0).fit_transform(X) if X.shape[1] >= 2
              else np.hstack([X, np.zeros((X.shape[0], 1))]))
        xl, yl = "PCA-for-display 1", "PCA-for-display 2"

    labels = np.asarray(labels)
    noise = labels < 0
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    if noise.any():
        ax.scatter(XY[noise, 0], XY[noise, 1], c="lightgrey", s=22, label="noise", edgecolors="none")
    cl = sorted(set(labels[~noise].tolist()))
    cmap = plt.get_cmap("plasma")
    for i, c in enumerate(cl):
        m = labels == c
        shade = cmap(i / max(1, len(cl) - 1)) if len(cl) > 1 else cmap(0.5)
        ax.scatter(XY[m, 0], XY[m, 1], color=shade, s=26, label=f"cluster {c}", edgecolors="none")
    ax.set_xlabel(xl); ax.set_ylabel(yl)
    ax.set_title(title or "Cluster assignments")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
    return path


def generate_report(output_dir, *, table, prepare_report, reduction_meta, X_reduced,
                    cluster_result, stability_result=None, results=None, extra=None) -> dict:
    """Write the full artifact set. Returns {name: path}."""
    os.makedirs(output_dir, exist_ok=True)
    paths = {}
    prov = video_provenance(table, stability_result) if stability_result is not None else None

    paths["assignments"] = write_assignments_csv(
        os.path.join(output_dir, "Clustering Assignments.csv"), table, cluster_result, stability_result)
    loadings = write_pca_loadings_csv(os.path.join(output_dir, "PCA Loadings.csv"), reduction_meta)
    if loadings:
        paths["loadings"] = loadings
    paths["roster"] = write_cluster_roster_csv(
        os.path.join(output_dir, "Cluster Roster.csv"), table, cluster_result, stability_result)
    if stability_result is not None:
        paths["dendrogram"] = plot_coassociation_dendrogram(
            os.path.join(output_dir, "Cluster Dendrogram.png"), table, stability_result)
    if results:
        bp = plot_cluster_ordered_barcode(
            os.path.join(output_dir, "Cluster Ordered Barcode"), results, table, cluster_result)
        if bp:
            paths["ordered_barcode"] = bp
    paths["scatter"] = plot_cluster_scatter(
        os.path.join(output_dir, "Cluster Scatter.png"), X_reduced, cluster_result.labels, reduction_meta)
    if stability_result is not None:
        paths["co_association"] = plot_co_association(
            os.path.join(output_dir, "Co-association Matrix.png"), stability_result)
        paths["stability_summary"] = plot_stability_summary(
            os.path.join(output_dir, "Stability Summary.png"), stability_result)
    paths["metadata"] = write_run_metadata(
        os.path.join(output_dir, "Clustering Run Metadata.json"),
        prepare_report=prepare_report, reduction_meta=reduction_meta,
        cluster_result=cluster_result, stability_result=stability_result,
        provenance=prov, extra=extra)
    return paths

def write_cluster_roster_csv(path, table, cluster_result, stability_result=None) -> str:
    """Assignments grouped BY CLUSTER, the direct answer to 'which files ended up together'.

    Sorted by cluster (noise last), then per-point stability descending, so each cluster's
    most representative members are at the top.
    """
    rows = []
    for i in range(table.n_channels):
        lab = int(cluster_result.labels[i])
        s = float(stability_result.per_point_stability[i]) if stability_result is not None else float("nan")
        rows.append((lab if lab >= 0 else 10 ** 6, -(0.0 if np.isnan(s) else s),
                     ["noise" if lab < 0 else lab, os.path.basename(table.video_ids[i]),
                      table.channel_ids[i], table.datasets[i],
                      "" if np.isnan(s) else round(s, 4), table.video_ids[i]]))
    rows.sort(key=lambda r: (r[0], r[1]))

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Cluster", "File", "Channel", "Dataset", "Per_Point_Stability", "Filepath"])
        for _, _, r in rows:
            w.writerow(r)
    return path


def plot_coassociation_dendrogram(path, table, stability_result,
                                  title="Which files cluster together") -> str:
    """Hierarchical tree from the co-association matrix, with FILENAMES as leaves.

    Merge height = co-association distance (1 - fraction of resamples co-clustered): two
    channels joining at 0 were always grouped, a join at 1.0 means never. Unlike the
    heatmap, this names the files.
    """
    from scipy.cluster.hierarchy import linkage, dendrogram
    from scipy.spatial.distance import squareform

    D = 1.0 - np.asarray(stability_result.co_association, dtype=float)
    np.fill_diagonal(D, 0.0)
    D = (D + D.T) / 2.0                       # exact symmetry required by squareform
    Z = linkage(squareform(D, checks=False), method="average")
    labels = [f"{os.path.basename(v)} ch{c}" for v, c in zip(table.video_ids, table.channel_ids)]

    fig, ax = plt.subplots(figsize=(9, max(4, 0.16 * len(labels))), dpi=300)
    dendrogram(Z, labels=labels, orientation="right", ax=ax,
               color_threshold=0.5, leaf_font_size=5)
    ax.set_xlabel("co-association distance (1 - fraction co-clustered)")
    ax.set_title(title)
    for x in (0.25, 0.5, 0.75):
        ax.axvline(x, color="grey", ls=":", lw=0.8)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
    return path


def plot_cluster_ordered_barcode(figpath_base, results, table, cluster_result,
                                 physical_units=False, metrics_to_visualize=None):
    """BARCODE's own colorized barcode, with rows ORDERED BY CLUSTER.

    If the clusters are real, the metric fingerprints should form visually distinct
    horizontal bands. Writes '<base>.png' plus a companion CSV mapping row -> file,
    since the barcode renderer does not label rows.
    """
    from visualization.barcode import generate_combined_barcode

    lab = np.asarray(cluster_result.labels)
    key = np.where(lab < 0, (lab.max() + 1) if lab.size else 0, lab)   # noise rows last
    order = np.argsort(key, kind="stable")
    lookup = {(r.filepath, int(r.channel)): r for r in results}

    ordered, rows = [], []
    for i in order:
        k = (table.video_ids[i], int(table.channel_ids[i]))
        if k in lookup:
            ordered.append(lookup[k])
            rows.append([len(ordered), "noise" if lab[i] < 0 else int(lab[i]),
                         os.path.basename(table.video_ids[i]), table.channel_ids[i]])
    if not ordered:
        return None

    # per-row labels only when the figure can carry them; otherwise mark cluster blocks only
    if len(ordered) <= 60:
        row_labels = [f"[{c}] {f} ch{ch}" for _, c, f, ch in rows]
    else:
        seen, row_labels = set(), []
        for _, c, _f, _ch in rows:
            row_labels.append(f"— cluster {c} —" if c not in seen else "")
            seen.add(c)

    generate_combined_barcode(ordered, figpath_base, separate_channels=False,
                              physical_units=physical_units,
                              metrics_to_visualize=metrics_to_visualize,
                              row_labels=row_labels)

    with open(f"{figpath_base} Row Order.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["Row", "Cluster", "File", "Channel"]); w.writerows(rows)
    return f"{figpath_base}.png"