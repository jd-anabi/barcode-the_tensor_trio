"""Shared clustering controls.

Used by BOTH the Execution tab (generate window) and the Barcode Aggregator tab
(analyze window) so the two windows expose an identical UI, mirroring the fact that
they route through one clustering core.
"""

import tkinter as tk
from tkinter import ttk

from core import ChannelResults
from utils.gui import create_option_section, create_popup


def create_clustering_controls(frame, cluster_config, row,
                               header_font=("TkDefaultFont", 15, "bold")):
    """Add the clustering toggle + sub-options. Returns the next free grid row."""
    cl = cluster_config
    metric_names = [m.value for m in ChannelResults.get_metrics(just_metrics=True)]

    tk.Label(frame, text="Clustering", font=header_font).grid(
        row=row, column=0, columnspan=3, sticky="w", padx=(5, 5), pady=(10, 5))
    row += 1

    create_option_section(
        frame, row, cl.enabled,
        "Generate clustering information",
        "Group channels into regimes by their BARCODE metrics, with bootstrap stability. "
        "Adds several minutes, since the stability check re-clusters the data many times. "
        "Most informative with 'Parse All Channels' enabled, which doubles the sample count.")
    row += 2

    mode_label = tk.Label(frame, text="Cluster Space:")
    mode_label.grid(row=row, column=0, sticky="w", padx=5, pady=5)
    mode_menu = ttk.Combobox(frame, textvariable=cl.mode, values=["pca", "subset", "full"],
                             width=8, state="readonly")
    mode_menu.grid(row=row, column=1, sticky="w", padx=5, pady=5)
    create_popup(frame, "pca: unsupervised reduction, the recommended default at this sample size. "
                        "subset: cluster only on metrics you pick. "
                        "full: every metric — a diagnostic view, not a default to trust.",
                 row, mode_label)
    row += 1

    var_label = tk.Label(frame, text="PCA Cumulative Variance:")
    var_label.grid(row=row, column=0, sticky="w", padx=5, pady=5)
    var_spin = ttk.Spinbox(frame, from_=0.5, to=0.99, increment=0.01,
                           textvariable=cl.pca_variance_threshold, width=7)
    var_spin.grid(row=row, column=1, sticky="w", padx=5, pady=5)
    create_popup(frame, "Fraction of variance the retained components must explain. Do not set this "
                        "too low — a cluster-separating direction can sit on a low-variance axis.",
                 row, var_label)
    row += 1

    picker_choice = {}
    picker = tk.Menubutton(frame, text="Select Metrics (subset mode)", relief="raised")
    picker.menu = tk.Menu(picker, tearoff=0)
    picker["menu"] = picker.menu

    def update_selected():
        cl.selected_metrics.clear()
        cl.selected_metrics.extend([m for m, v in picker_choice.items() if bool(v.get())])

    for m in metric_names:
        picker_choice[m] = tk.IntVar(value=0)
        picker.menu.add_checkbutton(label=m, variable=picker_choice[m],
                                    onvalue=1, offvalue=0, command=update_selected)
    picker.grid(row=row, column=0, sticky="w", padx=5, pady=5)
    create_popup(frame, "Choose at least 2 metrics. Only used in subset mode. Selecting "
                        "Mean Flow Direction automatically handles its circular encoding.",
                 row, picker)
    row += 1

    mcs_label = tk.Label(frame, text="Minimum Cluster Size:")
    mcs_label.grid(row=row, column=0, sticky="w", padx=5, pady=5)
    mcs_spin = ttk.Spinbox(frame, from_=2, to=100, increment=1,
                           textvariable=cl.min_cluster_size, width=7)
    mcs_spin.grid(row=row, column=1, sticky="w", padx=5, pady=5)
    create_popup(frame, "Smallest number of channels you are willing to call a regime. "
                        "This is a scientific choice, not a tuning knob.", row, mcs_label)
    row += 1

    boot_label = tk.Label(frame, text="Stability Resamples:")
    boot_label.grid(row=row, column=0, sticky="w", padx=5, pady=5)
    boot_spin = ttk.Spinbox(frame, from_=20, to=1000, increment=10,
                            textvariable=cl.n_bootstrap, width=7)
    boot_spin.grid(row=row, column=1, sticky="w", padx=5, pady=5)
    create_popup(frame, "How many resampled re-clusterings to run when testing whether the "
                        "clusters survive. 100-500 is typical; more is slower but more reliable.",
                 row, boot_label)
    row += 1

    widgets = [mode_menu, var_spin, picker, mcs_spin, boot_spin]

    def refresh(*_):
        on = bool(cl.enabled.get())
        for w in widgets:
            w.config(state="normal" if on else "disabled")
        if on:
            mode_menu.config(state="readonly")
            var_spin.config(state="normal" if cl.mode.get() == "pca" else "disabled")
            picker.config(state="normal" if cl.mode.get() == "subset" else "disabled")

    cl.enabled.trace_add("write", refresh)
    cl.mode.trace_add("write", refresh)
    refresh()

    return row