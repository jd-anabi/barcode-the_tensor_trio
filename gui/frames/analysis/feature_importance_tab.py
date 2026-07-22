"""Supervised feature importance tab (Analyze window).

Takes one BARCODE CSV with a ground-truth column appended and ranks the metrics by how
much each drives that target. Analyze-only on purpose: unlike clustering, this needs
labels, which do not exist while videos are still being processed.

The chart is drawn by `analysis.supervised.report.draw_importances`, the same function
that renders the saved PNG, so the on-screen and on-disk figures cannot drift apart.
"""

import os
import queue
import tkinter as tk
from tkinter import ttk, filedialog

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from gui.config import BarcodeConfigGUI, FeatureImportanceConfigGUI
from utils.gui import create_option_section, create_popup

# 25-odd metric labels are unreadable in a tab-sized figure. The embedded chart shows the
# strongest few; the saved PNG always carries every feature.
MAX_EMBEDDED_FEATURES = 15

# How often the main loop checks for a finished run. Imperceptible, and a no-op tick.
RESULT_POLL_MS = 150


def create_feature_importance_frame(parent, config: BarcodeConfigGUI,
                                    fi_config: FeatureImportanceConfigGUI):
    """Create the supervised feature-importance tab."""
    frame = ttk.Frame(parent)
    fi = fi_config

    # Probed once, here, rather than per run: the checkbox should tell the user up front
    # that SHAP is unavailable instead of silently producing an empty third series.
    try:
        import shap  # noqa: F401
        have_shap = True
    except Exception:
        have_shap = False

    row = 0

    # ---- CSV chooser -------------------------------------------------------------
    tk.Label(frame, text="Select CSV File:").grid(
        row=row, column=0, sticky="w", padx=5, pady=5)
    csv_label = tk.Label(frame, text="No file selected", wraplength=280, justify="left")
    csv_label.grid(row=row, column=1, sticky="w", padx=5, pady=5)

    def browse_csv_file():
        chosen = filedialog.askopenfilename(
            filetypes=[("CSV Files", "*.csv")],
            title="Select a BARCODE CSV with a target column")
        if chosen:
            fi.csv_location.set(chosen)
            csv_label.config(text=os.path.basename(chosen))
        else:
            fi.csv_location.set("")
            csv_label.config(text="No file selected")

    browse_button = tk.Button(frame, text="Select CSV File", command=browse_csv_file)
    browse_button.grid(row=row, column=2, padx=5, pady=5)
    create_popup(frame,
                 "A normal BARCODE summary CSV with one extra column holding the ground "
                 "truth for each row -- a class name for classification, or a number for "
                 "regression. Rows with a blank target are skipped.",
                 row, csv_label)
    row += 1

    # ---- model type --------------------------------------------------------------
    mode_label = tk.Label(frame, text="Model Type:")
    mode_label.grid(row=row, column=0, sticky="w", padx=5, pady=5)
    mode_menu = ttk.Combobox(frame, textvariable=fi.model_type,
                             values=["classification", "regression"],
                             width=14, state="readonly")
    mode_menu.grid(row=row, column=1, sticky="w", padx=5, pady=5)
    create_popup(frame,
                 "classification: the target is a category, such as a condition or "
                 "material name. regression: the target is a continuous number, such as a "
                 "measured modulus or concentration.",
                 row, mode_label)
    row += 1

    # ---- target column -----------------------------------------------------------
    target_label = tk.Label(frame, text="Target Column:")
    target_label.grid(row=row, column=0, sticky="w", padx=5, pady=5)
    target_menu = ttk.Combobox(frame, textvariable=fi.target_column, values=[],
                               width=28, state="disabled")
    target_menu.grid(row=row, column=1, sticky="w", padx=5, pady=5)
    target_hint = tk.Label(frame, text="", wraplength=240, justify="left", fg="grey")
    target_hint.grid(row=row, column=2, sticky="w", padx=5, pady=5)
    create_popup(frame,
                 "The ground-truth column to predict. Populated from the chosen CSV: any "
                 "column BARCODE did not write itself is offered as a candidate.",
                 row, target_label)
    row += 1

    def refresh_targets(*_):
        """Repopulate the dropdown from the chosen CSV. Header row only -- must stay cheap."""
        path = fi.csv_location.get()
        if not path or not os.path.exists(path):
            target_menu.config(values=[], state="disabled")
            fi.target_column.set("")
            target_hint.config(text="")
            return
        try:
            from analysis.supervised import candidate_target_columns
            candidates = candidate_target_columns(path)
        except Exception as exc:
            target_menu.config(values=[], state="disabled")
            fi.target_column.set("")
            target_hint.config(text=f"Could not read the header: {exc}")
            return

        target_menu.config(values=candidates,
                           state="readonly" if candidates else "disabled")
        if not candidates:
            fi.target_column.set("")
            target_hint.config(
                text="No target column found -- this CSV holds only BARCODE's own columns.")
        else:
            # One candidate is the overwhelmingly common case: a target appended on the end.
            if fi.target_column.get() not in candidates:
                fi.target_column.set(candidates[0])
            target_hint.config(text=f"{len(candidates)} candidate column(s)")

    fi.csv_location.trace_add("write", refresh_targets)

    # ---- run settings ------------------------------------------------------------
    create_option_section(
        frame, row, fi.tuning,
        "Tune hyperparameters",
        "Search for better forest settings with cross-validation before fitting. This is "
        "roughly 150 extra model fits and is by far the slowest part of a run. Leave it "
        "off for a first look.")
    row += 2

    shap_text = "Include SHAP importance"
    if not have_shap:
        fi.use_shap.set(False)
        shap_text += "  (shap not installed)"
    create_option_section(
        frame, row, fi.use_shap, shap_text,
        "Adds a third ranking that attributes the prediction of every individual row to "
        "each metric. Where all three rankings agree, the result is solid; where they "
        "disagree, the top of the chart should not be read as settled.")
    if not have_shap:
        # The checkbox is the first widget create_option_section grids on this row.
        for child in frame.grid_slaves(row=row, column=0):
            if isinstance(child, tk.Checkbutton):
                child.config(state="disabled")
    row += 2

    test_label = tk.Label(frame, text="Test Fraction:")
    test_label.grid(row=row, column=0, sticky="w", padx=5, pady=5)
    test_spin = ttk.Spinbox(frame, from_=0.1, to=0.5, increment=0.05,
                            textvariable=fi.test_size, width=7)
    test_spin.grid(row=row, column=1, sticky="w", padx=5, pady=5)
    create_popup(frame,
                 "Share of the data held back to score the model. Videos are held out "
                 "whole -- a video's channels image one physical sample, so splitting them "
                 "across train and test would make the score look better than it is.",
                 row, test_label)
    row += 1

    seed_label = tk.Label(frame, text="Random Seed:")
    seed_label.grid(row=row, column=0, sticky="w", padx=5, pady=5)
    seed_spin = ttk.Spinbox(frame, from_=0, to=99999, increment=1,
                            textvariable=fi.random_seed, width=7)
    seed_spin.grid(row=row, column=1, sticky="w", padx=5, pady=5)
    create_popup(frame,
                 "Fixes the split and the forest so a run is reproducible. Changing it and "
                 "seeing the ranking change is itself informative: it means the dataset is "
                 "too small to pin the order down.",
                 row, seed_label)
    row += 1

    # ---- embedded chart ----------------------------------------------------------
    tk.Label(frame, text="Feature Importance").grid(
        row=row, column=0, sticky="w", padx=5, pady=(10, 0))
    row += 1

    root = parent.winfo_toplevel()
    r, g, b = root.winfo_rgb(root.cget("bg"))
    bg_color = (r / 65535, g / 65535, b / 65535)

    fig = Figure(figsize=(7, 5), facecolor=bg_color)
    ax = fig.add_subplot(111)
    canvas = FigureCanvasTkAgg(fig, master=frame)

    def reset_axes(message):
        ax.clear()
        ax.set_facecolor(bg_color)          # ax.clear() drops the face colour
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.text(0.5, 0.5, message, ha="center", va="center", color="grey",
                fontsize=9, wrap=True, transform=ax.transAxes)
        canvas.draw()

    canvas.get_tk_widget().grid(row=row, column=0, columnspan=3, padx=5, pady=(5, 10))
    reset_axes("Select a CSV, choose a target column,\n"
               "then press Analyze BARCODE Data.")
    row += 1

    def show_result(payload):
        """Draw a finished run. MAIN THREAD ONLY -- see deliver_result."""
        if payload is None:
            reset_axes("The run was skipped.\nSee the Processing Log for the reason.")
            return
        result = payload["result"]
        ax.clear()
        ax.set_facecolor(bg_color)
        from analysis.supervised import draw_importances
        draw_importances(ax, result, max_features=MAX_EMBEDDED_FEATURES)
        if len(result.feature_names) > MAX_EMBEDDED_FEATURES:
            ax.set_title(f"{ax.get_title()}  (top {MAX_EMBEDDED_FEATURES})", fontsize=10)
        fig.tight_layout()
        canvas.draw()

    # Results arrive on a worker thread, and Tk is not thread-safe. Note that `after()` is
    # NOT an escape hatch: it calls Tcl's createcommand to register the callback, so from a
    # worker thread it raises "main thread is not in main loop". The worker therefore only
    # puts a plain Python object on a queue, and a poller owned by the main loop does every
    # widget touch.
    results = queue.Queue()
    polling = {"on": True}

    def poll_results():
        if not polling["on"]:
            return
        while True:
            try:
                payload = results.get_nowait()
            except queue.Empty:
                break
            try:
                show_result(payload)
            except Exception as exc:                 # never let one bad run kill the poller
                print(f"Could not draw the feature-importance chart: {exc}")
        try:
            frame.after(RESULT_POLL_MS, poll_results)
        except tk.TclError:
            polling["on"] = False                    # the page was torn down mid-tick

    def stop_polling(event):
        # <Destroy> bubbles up from children, so only the frame's own teardown counts.
        if event.widget is frame:
            polling["on"] = False

    def deliver_result(payload):
        """Hand a finished run to the GUI. Safe to call from any thread.

        If the user navigated away the poller is already stopped and the payload is simply
        dropped -- the saved PNG is the durable artifact, so that is the right outcome.
        """
        results.put(payload)

    frame.bind("<Destroy>", stop_polling)
    frame.deliver_result = deliver_result
    frame.after(RESULT_POLL_MS, poll_results)

    refresh_targets()
    return frame
