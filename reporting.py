"""
Part E - automated experiment documentation.

Produces, per run:
  outputs/run_<timestamp>/
    weights/   model checkpoints
    figures/   publication-ready PNGs (300 DPI)
    reports/   results_draft.md + results_draft.tex (thesis-draft generator)
  metrics_summary.json

Figure design follows a validated light-mode palette (blue/orange categorical
pair, single-hue blue sequential ramp); per data-viz best practice the
learning curves use two side-by-side panels instead of a dual y-axis.
"""
from __future__ import annotations

import datetime
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import mean_squared_error, r2_score

from config import CLASS_NAMES, IFRI_BASELINES, IFRI_REFERENCE

# --- validated light-mode palette (dataviz reference palette) ---
SERIES_BLUE = "#2a78d6"    # categorical slot 1
SERIES_ORANGE = "#eb6834"  # categorical slot 2
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE_AXIS = "#c3c2b7"
# single-hue sequential ramp (blue), light -> dark
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95"]

plt.rcParams.update({
    "font.family": "sans-serif",
    "axes.edgecolor": BASELINE_AXIS,
    "axes.labelcolor": INK_PRIMARY,
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "axes.titlesize": 12,
    "axes.titlecolor": INK_PRIMARY,
    "axes.labelsize": 11,
    "legend.frameon": False,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.grid": False,
})


# ---------------------------------------------------------------------------
# run directory + JSON summary
# ---------------------------------------------------------------------------

def make_run_dir(output_dir: str = "outputs") -> str:
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_dir, f"run_{stamp}")
    for sub in ("weights", "figures", "reports"):
        os.makedirs(os.path.join(run_dir, sub), exist_ok=True)
    return run_dir


def save_json(obj: dict, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)
    print(f"[report] metrics summary -> {path}")


# ---------------------------------------------------------------------------
# figures (300 DPI PNGs)
# ---------------------------------------------------------------------------

def plot_learning_curves(history: dict, metric_label: str, path: str, dpi: int = 300):
    """Train/val loss (left) and train/val metric (right) as two panels.

    Two side-by-side panels instead of a dual-axis plot: loss and accuracy
    live on unrelated scales, and a shared y-axis would invent a spurious
    relationship between them.
    """
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    fig, (ax_l, ax_m) = plt.subplots(1, 2, figsize=(9.5, 4.2))

    for ax, train_key, val_key, title, ylabel in (
            (ax_l, "train_loss", "val_loss", "Loss", "Loss"),
            (ax_m, "train_metric", "val_metric", metric_label, metric_label)):
        ax.plot(epochs, history[train_key], color=SERIES_BLUE, lw=2, label="Train")
        ax.plot(epochs, history[val_key], color=SERIES_ORANGE, lw=2, label="Validation")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.yaxis.grid(True)          # solid hairlines, horizontal only
        ax.legend(loc="best")
        ax.margins(x=0.02)

    fig.suptitle("Training dynamics", color=INK_PRIMARY, y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[report] figure -> {path}")


def plot_confusion_matrix(cm: np.ndarray, path: str, class_names=None, dpi: int = 300):
    """Annotated heatmap of the test-set confusion matrix.

    Sequential single-hue ramp (magnitude encoding), counts on every cell,
    2px white gaps between cells, and per-cell ink chosen for contrast.
    """
    names = list(class_names) if class_names else list(CLASS_NAMES)
    cmap = LinearSegmentedColormap.from_list("blue_ramp", BLUE_RAMP)
    fig, ax = plt.subplots(figsize=(5.8, 5.0))
    sns.heatmap(cm, annot=True, fmt="d", cmap=cmap, cbar=False, square=True,
                linewidths=2, linecolor="white", ax=ax,
                xticklabels=names, yticklabels=names,
                annot_kws={"size": 11, "fontweight": "bold"})
    cmax = max(cm.max(), 1)
    for text, val in zip(ax.texts, cm.ravel()):
        text.set_color("white" if val > 0.55 * cmax else INK_PRIMARY)
    ax.set_xlabel("Predicted", color=INK_PRIMARY)
    ax.set_ylabel("Actual", color=INK_PRIMARY)
    ax.set_title("Confusion matrix - crop classes (test set)")
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[report] figure -> {path}")


def plot_regression_analysis(y_true: np.ndarray, y_pred: np.ndarray,
                             path: str, dpi: int = 300):
    """Actual vs predicted days-after-planting scatter with the 1:1 reference."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))

    fig, ax = plt.subplots(figsize=(6.2, 6.0))
    ax.scatter(y_true, y_pred, s=16, color=SERIES_BLUE, alpha=0.55,
               edgecolors="white", linewidths=0.5, label="Sample")
    hi = max(y_true.max(), y_pred.max()) * 1.05
    ax.plot([0, hi], [0, hi], color=INK_SECONDARY, lw=1.5, label="1:1 reference")
    ax.text(0.04, 0.96, f"R$^2$ = {r2:.3f}\nRMSE = {rmse:.1f} days",
            transform=ax.transAxes, va="top", ha="left", fontsize=11,
            color=INK_PRIMARY)
    ax.set_xlim(0, hi)
    ax.set_ylim(0, hi)
    ax.set_xlabel("Actual days after planting (HST)")
    ax.set_ylabel("Predicted days after planting (HST)")
    ax.set_title("Crop-phase regression - test set")
    ax.legend(loc="lower right")
    ax.yaxis.grid(True)
    ax.xaxis.grid(True)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[report] figure -> {path}")


def plot_split_map(train_df, val_df, test_df, path, dpi: int = 300):
    """Geographic scatter of the spatially-grouped split, one color per split.

    Makes the spatial holdout visible: validation/test points live in source
    regions the model never trained on. Axes are plain lon/lat (no basemap).
    """
    if not all(c in train_df.columns for c in ("longitude", "latitude")):
        print("[report] warning: skipping split map (no lat/lon columns)")
        return
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    for df, color, label in ((train_df, SERIES_BLUE, "Train"),
                             (val_df, SERIES_ORANGE, "Validation"),
                             (test_df, INK_SECONDARY, "Test")):
        ax.scatter(df["longitude"], df["latitude"], s=5, color=color,
                   alpha=0.5, label=f"{label} ({len(df)})")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Spatially-grouped split - by 1-degree grid cell")
    ax.legend(loc="best", markerscale=3)
    ax.xaxis.grid(True)
    ax.yaxis.grid(True)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[report] figure -> {path}")


# ---------------------------------------------------------------------------
# regional subset distribution map (--region sea)
# ---------------------------------------------------------------------------

SEA_COAST = "#9aa5ad"
SEA_LABEL_INK = "#5f6b76"
SEA_COUNTRY_LABELS = [
    (96.6, 16.0, "MYANMAR"), (99.2, 14.6, "THAILAND"), (102.5, 18.9, "LAOS"),
    (106.9, 15.3, "VIETNAM"), (104.6, 12.1, "CAMBODIA"), (102.7, 3.4, "MALAYSIA"),
    (109.4, -7.4, "INDONESIA\n(West Java)")]


def _coastline_segments():
    """Natural Earth 1:110m coastline segments from assets/; [] when absent.

    Committed alongside the repo (public-domain Natural Earth data), so the
    map renders identically offline instead of fetching at run time.
    """
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "assets", "ne_110m_coastline.geojson")
    try:
        with open(path, encoding="utf-8") as f:
            gj = json.load(f)
    except (OSError, ValueError):
        print("[report] warning: coastline file not found - drawing without basemap")
        return []
    segs = []
    for feat in gj.get("features", []):
        geom = feat.get("geometry") or {}
        t = geom.get("type")
        if t == "LineString":
            segs.append(np.asarray(geom["coordinates"], dtype=float))
        elif t == "MultiLineString":
            segs.extend(np.asarray(p, dtype=float) for p in geom["coordinates"])
    return segs


def plot_sea_distribution(df, path, dpi: int = 300):
    """Spatial sample-distribution map of the SEA subset as small multiples.

    One panel per crop class: a 4-hue categorical palette cannot separate
    Rice/Corn/Sugarcane/Other for color-deficient readers on this surface
    (validated), so the facets carry identity instead - each panel is a
    single series named by its title, and the sparse Corn/Other panels make
    the regional class imbalance visible at a glance.
    """
    if not all(c in df.columns for c in ("longitude", "latitude", "crop_class")):
        print("[report] warning: skipping SEA map (missing lat/lon/class columns)")
        return
    counts = df["crop_class"].value_counts().sort_index().to_dict()
    coasts = _coastline_segments()
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.6))
    for c, ax in enumerate(axes.ravel()):
        ax.set_facecolor("#fbfcfd")
        for xy in coasts:
            ax.plot(xy[:, 0], xy[:, 1], color=SEA_COAST, lw=0.5,
                    solid_capstyle="round", zorder=1)
        sub = df[df["crop_class"] == c]
        ax.scatter(sub["longitude"], sub["latitude"], s=36, color=SERIES_BLUE,
                   edgecolors="white", linewidths=0.6, zorder=4)
        for x, y, txt in SEA_COUNTRY_LABELS:
            ax.text(x, y, txt, color=SEA_LABEL_INK, fontsize=7, style="italic",
                    ha="center", va="center", zorder=3)
        ax.set_xlim(95.5, 110.5)
        ax.set_ylim(-8.2, 19.2)
        ax.set_xticks([98, 104, 110])
        ax.set_yticks([-8, 0, 8, 16])
        ax.set_title(f"{CLASS_NAMES[c]} (n={counts.get(c, 0)})",
                     color=INK_PRIMARY, fontsize=11)
        if ax in axes[:, 0]:           # left column only
            ax.set_ylabel("Latitude", color=INK_MUTED, fontsize=8)
        if ax in axes[-1, :]:          # bottom row only
            ax.set_xlabel("Longitude", color=INK_MUTED, fontsize=8)
    fig.suptitle("CropHarvest Southeast Asia subset - sample distribution by crop class",
                 color=INK_PRIMARY, y=1.0)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[report] figure -> {path}")


# ---------------------------------------------------------------------------
# thesis-draft generator (Markdown + LaTeX)
# ---------------------------------------------------------------------------

def _tex_escape(s) -> str:
    return (str(s).replace("\\", r"\textbackslash{}").replace("&", r"\&")
            .replace("%", r"\%").replace("#", r"\#")
            .replace("_", r"\_").replace("$", r"\$"))


def _is_cropharvest(summary: dict) -> bool:
    return (summary.get("data") or {}).get("source") == "cropharvest"


def _discussion_paragraph(summary: dict) -> str:
    """Template-driven discussion draft assembled from the final metrics."""
    parts = []
    cls = summary.get("classification") or {}
    cnn = cls.get("cnn1d")
    class_names = cls.get("class_names") or list(CLASS_NAMES)
    is_ch = _is_cropharvest(summary)
    split_type = (summary.get("data") or {}).get("split")

    if cnn:
        acc = cnn["test_accuracy_pct"]
        if len(class_names) == 2:
            if is_ch:
                parts.append(
                    f"The binary Rice vs other-crop stage attained {acc:.1f}% test "
                    f"accuracy (macro-F1 {cnn['macro_f1_pct']:.1f}%) on the "
                    f"held-out CropHarvest split.")
            else:
                base = IFRI_BASELINES["binary_accuracy_pct"]
                parts.append(
                    f"The binary Rice/Non-Rice stage attained {acc:.1f}% test accuracy "
                    f"(macro-F1 {cnn['macro_f1_pct']:.1f}%), compared with the IFRI 2024 "
                    f"two-class stage's {base}% (CNN+LSTM on high-resolution data). "
                    f"The padi/non-padi split is the well-conditioned part of the "
                    f"problem - the IFRI report already shows this stage clearing 80% "
                    f"while the 4-class stage stalls near 67% - which is why Stage 2 "
                    f"below is gated on it.")
        else:
            if is_ch:
                base = cls.get("baseline") or {}
                if base:
                    delta = acc - base["test_accuracy_pct"]
                    parts.append(
                        f"The CNN 1D classifier attained a test accuracy of {acc:.1f}% "
                        f"(macro-F1 {cnn['macro_f1_pct']:.1f}%) on the held-out "
                        f"CropHarvest split, {abs(delta):.1f} percentage points "
                        f"{'above' if delta >= 0 else 'below'} the {base['model']} "
                        f"shallow baseline ({base['test_accuracy_pct']}%) trained on "
                        f"the same split.")
                else:
                    parts.append(
                        f"The CNN 1D classifier attained a test accuracy of {acc:.1f}% "
                        f"(macro-F1 {cnn['macro_f1_pct']:.1f}%) on the held-out "
                        f"CropHarvest split.")
            else:
                base = IFRI_BASELINES["classification_accuracy_pct"]
                delta = acc - base
                if delta >= 0:
                    parts.append(
                        f"The CNN 1D classifier attained a test accuracy of {acc:.1f}% "
                        f"(macro-F1 {cnn['macro_f1_pct']:.1f}%), surpassing the IFRI 2024 "
                        f"Gradient Boosting baseline of {base}% by {delta:.1f} percentage points.")
                else:
                    parts.append(
                        f"The CNN 1D classifier attained a test accuracy of {acc:.1f}% "
                        f"(macro-F1 {cnn['macro_f1_pct']:.1f}%), falling {abs(delta):.1f} "
                        f"percentage points short of the IFRI 2024 Gradient Boosting "
                        f"baseline of {base}%.")
            cm = np.array(cnn["confusion_matrix"])
            mixed = int(cm[1, 2] + cm[2, 1])
            corn_tebu_total = int(cm[1].sum() + cm[2].sum())
            if mixed > 0:
                parts.append(
                    f"The model still confused Corn with Sugarcane in {mixed} of "
                    f"{corn_tebu_total} test samples, echoing the IFRI report's "
                    f"conclusion that the non-padi stage (jagung vs tebu) is the "
                    f"principal classification bottleneck.")
            else:
                parts.append(
                    "No Corn-Sugarcane confusion occurred on the test set, although "
                    "this should be treated cautiously given the small held-out sample.")
        if is_ch and len(class_names) != 2:
            if split_type == "spatial-group":
                parts.append(
                    "Because validation and test hold out entire 1-degree grid "
                    "cells, these numbers estimate generalisation to unseen "
                    "geographies rather than within-region interpolation.")
            elif split_type == "stratified-fallback":
                parts.append(
                    "As the spatial-group split could not satisfy the per-class "
                    "coverage requirement, this run fell back to a stratified split "
                    "(see the run log), so the held-out metrics are an upper bound "
                    "that does not test cross-geography generalisation.")
            parts.append(
                "CropHarvest labels come from crowdsourced and institutional "
                "interpretation under heterogeneous protocols, and the coarse "
                "'Other' class merges many crop types, so part of the remaining "
                "error is label noise rather than model error.")
        gap = cnn.get("train_val_gap_pct", 0.0)
        verdict = ("a mild overfitting tendency" if gap > 10
                   else "a well-regularised fit" if gap < 3 else "a balanced fit")
        epochs_run = len((cls.get("history") or {}).get("train_loss", []))
        parts.append(
            f"Training ran for {epochs_run} epochs (early stopping patience "
            f"{summary['config']['early_stop_patience']}); the best checkpoint by "
            f"validation loss was selected at epoch {cnn['best_epoch']}, with "
            f"final train/validation loss "
            f"{cnn['final_train_loss']:.4f}/{cnn['final_val_loss']:.4f} and "
            f"{verdict} indicated by the {gap:.1f} percentage-point "
            f"train-validation {cls.get('metric', 'accuracy')} gap.")
        if summary["config"].get("use_attention"):
            parts.append(
                "Temporal self-attention allowed the model to weight the monthly "
                "sequence dynamically (e.g. emphasising the flooded transplant "
                "window for rice), rather than pooling all months equally.")

    reg = summary.get("regression") or {}
    rcnn = reg.get("cnn1d")
    if rcnn:
        base_rmse = IFRI_BASELINES["regression_rmse_days"]
        parts.append(
            f"For crop phase, the supervised CNN 1D regressor predicted days "
            f"after planting with a test RMSE of {rcnn['test_rmse_days']:.1f} days "
            f"(MAE {rcnn['test_mae_days']:.1f} days, R^2 {rcnn['test_r2']:.2f}), "
            f"a reduction of {base_rmse - rcnn['test_rmse_days']:.1f} days against "
            f"the {base_rmse:.0f}-day RMSE of the IFRI manual flood-threshold "
            f"method - evidence that learned phase estimation can bypass manual "
            f"threshold tuning.")
        chain = reg.get("chain")
        if chain:
            parts.append(
                f"Following the IFRI hierarchical design, Stage 2 was trained "
                f"only on pixels Stage 1 predicted as Rice "
                f"({chain['rice_predicted_rows']} of "
                f"{chain['hst_labelled_rows']} HST-labelled rows).")

    if is_ch:
        parts.append(
            "These results are bounded by the benchmark itself: CropHarvest samples "
            "are point-level, class-imbalanced and geographically uneven, so the "
            "figures should be read as benchmark comparisons rather than estimates "
            "of operational accuracy in East Java.")
    else:
        parts.append(
            "These results are bounded by the training data itself: the IFRI sample "
            "set (~400 field + 400 on-screen points) is small and geographically "
            "concentrated, so further validation against additional field campaigns "
            "is required before drawing general conclusions for East Java.")
    return " ".join(parts)


def _methodology_outline(summary: dict) -> str:
    cfg = summary["config"]
    data = summary["data"]
    pct = lambda key: int(cfg[key] * 100)
    if _is_cropharvest(summary):
        lines = [
            f"* Data: {data['rows']} CropHarvest samples (18 bands x 12 months = "
            f"216 features; Zenodo record {data['zenodo_id']}, {data['license']}). "
            f"Non-crop and unlabelled rows excluded; class counts {data['class_counts']}.",
        ]
        if data.get("split") == "spatial-group":
            lines.append(
                f"* Split: spatially-grouped {pct('train_size')}/{pct('val_size')}/"
                f"{pct('test_size')} by ~1-degree lat/lon grid cells (seed "
                f"{cfg['seed']}); no geographic cell is shared between "
                f"train/validation/test, so held-out metrics reflect "
                f"generalisation to unseen geographies.")
        else:
            lines.append(
                f"* Split: {pct('train_size')}/{pct('val_size')}/{pct('test_size')} "
                f"stratified fallback (seed {cfg['seed']}) - the spatial-group split "
                f"failed its class-coverage check (see the run log).")
        if data.get("region") == "sea":
            bbox = data.get("bbox") or {}
            lines.append(
                f"* Region: filtered to the Southeast Asia bounding box "
                f"(lat {bbox.get('lat_min')}..{bbox.get('lat_max')}, "
                f"lon {bbox.get('lon_min')}..{bbox.get('lon_max')}), leaving "
                f"{data['rows']} samples; single-sample classes are pinned to "
                f"train because they cannot populate all three splits.")
    else:
        lines = [
            f"* Data: {data['rows']} rows from {data['source']} "
            f"(181-column IFRI schema).",
            f"* Split: stratified {pct('train_size')}/{pct('val_size')}/"
            f"{pct('test_size')} train/validation/test (seed {cfg['seed']}).",
        ]
    for key in ("classification", "regression"):
        block = summary.get(key) or {}
        if block.get("cnn1d"):
            t = block["tensor"]
            lines.append(
                f"* {key.capitalize()} input tensor: [B, {t['n_channels']}, {t['seq_len']}] "
                f"temporal ({', '.join(t['channels'])}) + [B, {t['n_static']}] static features.")
            if key == "classification" and block.get("class_names"):
                head_desc = (f"{len(block['class_names'])} classes "
                             f"({', '.join(block['class_names'])})")
            else:
                head_desc = "1 node (HST)"
            lines.append(
                f"* {key.capitalize()} architecture: TemporalCNN1D backbone "
                f"(Conv1d {cfg['conv_channels']}, k={cfg['kernel_size']}, "
                f"BN + ReLU + MaxPool + dropout {cfg['dropout']})"
                + (" -> temporal self-attention" if cfg.get("use_attention") else "")
                + " -> global average pooling"
                + (" -> concat static" if t["n_static"] > 0 else "")
                + f" -> FC {cfg['fc_hidden']} -> {head_desc}.")
            lines.append(
                f"* {key.capitalize()} optimisation: {block['loss']} loss, Adam "
                f"(lr {cfg['lr']}, weight decay {cfg['weight_decay']}), batch "
                f"{cfg['batch_size']}, up to {cfg['epochs']} epochs with early "
                f"stopping (patience {cfg['early_stop_patience']}).")
    if summary.get("classification", {}).get("binary"):
        lines.append("* Stage 1 scheme: binary Rice vs Non-Rice (padi/non-padi split).")
    if cfg.get("use_attention"):
        lines.append(f"* Temporal self-attention: {cfg['num_heads']}-head self-attention "
                     f"over the monthly sequence before global average pooling.")
    if (summary.get("regression") or {}).get("chain"):
        lines.append("* Stage 2 scheme: the HST regressor trains only on pixels "
                     "Stage 1 predicts as Rice (hierarchical chain).")
    return "\n".join(lines)


def _markdown_table(summary: dict) -> str:
    cls = (summary.get("classification") or {}).get("cnn1d")
    base = (summary.get("classification") or {}).get("baseline")
    bases = (summary.get("classification") or {}).get("baselines") or ([base] if base else [])
    if _is_cropharvest(summary):
        rows = ["| Metric | This run |",
                "|--------|----------|"]
        if cls:
            rows.append(
                f"| Classification test accuracy (CNN 1D) | {cls['test_accuracy_pct']}% |")
            rows.append(
                f"| Classification macro-F1 (CNN 1D) | {cls['macro_f1_pct']}% |")
        for b in bases:
            rows.append(
                f"| Classification test accuracy (shallow {b['model']}) | "
                f"{b['test_accuracy_pct']}% |")
            rows.append(
                f"| Classification macro-F1 (shallow {b['model']}) | "
                f"{b['macro_f1_pct']}% |")
        rows.append(f"| Held-out split | {summary['data'].get('split', 'stratified')} |")
        return "\n".join(rows)

    rows = ["| Metric | This run | IFRI 2024 baseline | $\\Delta$ |",
            "|--------|----------|--------------------|-----------|"]
    binary = len((summary.get("classification") or {}).get("class_names", [])) == 2
    base_acc = (IFRI_BASELINES["binary_accuracy_pct"] if binary
                else IFRI_BASELINES["classification_accuracy_pct"])
    base_label = ("2-class stage, CNN+LSTM [1,2]" if binary
                  else "Gradient Boosting [1,2]")
    if cls:
        rows.append(
            f"| Classification test accuracy (CNN 1D) | {cls['test_accuracy_pct']}% | "
            f"{base_acc}% ({base_label}) | {cls['test_accuracy_pct'] - base_acc:+.1f} pp |")
        rows.append(
            f"| Classification macro-F1 (CNN 1D) | {cls['macro_f1_pct']}% | - | - |")
    if base:
        rows.append(
            f"| Classification test accuracy (shallow {base['model']} replication) | "
            f"{base['test_accuracy_pct']}% | {base_acc}% ({base_label}) | "
            f"{base['test_accuracy_pct'] - base_acc:+.1f} pp |")
    reg = (summary.get("regression") or {}).get("cnn1d")
    if reg:
        base_rmse = IFRI_BASELINES["regression_rmse_days"]
        rows.append(
            f"| Regression RMSE (CNN 1D) | {reg['test_rmse_days']} days | "
            f"{base_rmse} days (manual flood threshold [1,2]) | "
            f"{reg['test_rmse_days'] - base_rmse:+.1f} days |")
        rows.append(
            f"| Regression MAE (CNN 1D) | {reg['test_mae_days']} days | - | - |")
    return "\n".join(rows)


def _build_markdown(summary: dict) -> str:
    fig = summary.get("figures", [])
    fig_md = "\n\n".join(
        f"![{os.path.basename(p)}](../figures/{os.path.basename(p)})" for p in fig)
    is_ch = _is_cropharvest(summary)
    results_title = "Results" if is_ch else "Results vs. IFRI 2024 Baseline"
    if is_ch:
        refs = ("[1] G. Tseng, I. Zvonkov, C. L. Nakalembe, H. Kerner, *CropHarvest: A global "
                "dataset for crop-type classification*, NeurIPS Datasets and Benchmarks, 2021. "
                "https://github.com/nasaharvest/cropharvest\n"
                "[2] R. G. Congalton and K. Green, *Assessing the Accuracy of Remotely Sensed Data: "
                "Principles and Practices*, 3rd ed., CRC Press, 2019.")
    else:
        refs = ("[1] Tim IFRI, *Laporan Akhir IFRI 2024* (Metode, Hasil dan Pembahasan, Kesimpulan), 2024.\n"
                "[2] R. G. Congalton and K. Green, *Assessing the Accuracy of Remotely Sensed Data: "
                "Principles and Practices*, 3rd ed., CRC Press, 2019.")
    return f"""# Crop Analysis from Multimodal Remote Sensing - Experiment Results

**Run:** {summary['run_id']}  \
**Date:** {summary['timestamp']}  \
**Seed:** {summary['config']['seed']}

## 1. Methodology Outline

{_methodology_outline(summary)}

## 2. {results_title}

{_markdown_table(summary)}

## 3. Figures

{fig_md}

## 4. Discussion Draft

{_discussion_paragraph(summary)}

## References

{refs}
"""


def _latex_table(summary: dict) -> str:
    cls = (summary.get("classification") or {}).get("cnn1d")
    base = (summary.get("classification") or {}).get("baseline")
    bases = (summary.get("classification") or {}).get("baselines") or ([base] if base else [])
    if _is_cropharvest(summary):
        rows = []
        if cls:
            rows.append(rf"Classification test accuracy (CNN 1D) & {cls['test_accuracy_pct']}\% \\")
            rows.append(rf"Classification macro-F1 (CNN 1D) & {cls['macro_f1_pct']}\% \\")
        for b in bases:
            rows.append(rf"Classification test accuracy (shallow {_tex_escape(b['model'])}) & {b['test_accuracy_pct']}\% \\")
            rows.append(rf"Classification macro-F1 (shallow {_tex_escape(b['model'])}) & {b['macro_f1_pct']}\% \\")
        rows.append(rf"Held-out split & {_tex_escape(summary['data'].get('split', 'stratified'))} \\")
        return "\n".join(rows)

    rows = []
    binary = len((summary.get("classification") or {}).get("class_names", [])) == 2
    base_acc = (IFRI_BASELINES["binary_accuracy_pct"] if binary
                else IFRI_BASELINES["classification_accuracy_pct"])
    base_label = ("2-class stage, CNN+LSTM" if binary else "Gradient Boosting")
    if cls:
        rows.append(rf"Classification test accuracy (CNN 1D) & {cls['test_accuracy_pct']}\% & {base_acc}\% ({base_label}) \cite{{ifri2024,congalton2019}} & {cls['test_accuracy_pct'] - base_acc:+.1f} pp \\")
        rows.append(rf"Classification macro-F1 (CNN 1D) & {cls['macro_f1_pct']}\% & -- & -- \\")
    if base:
        rows.append(rf"Classification test accuracy (shallow {_tex_escape(base['model'])}) & {base['test_accuracy_pct']}\% & {base_acc}\% ({base_label}) \cite{{ifri2024,congalton2019}} & {base['test_accuracy_pct'] - base_acc:+.1f} pp \\")
    reg = (summary.get("regression") or {}).get("cnn1d")
    if reg:
        base_rmse = IFRI_BASELINES["regression_rmse_days"]
        rows.append(rf"Regression RMSE (CNN 1D) & {reg['test_rmse_days']} days & {base_rmse} days \cite{{ifri2024,congalton2019}} & {reg['test_rmse_days'] - base_rmse:+.1f} days \\")
        rows.append(rf"Regression MAE (CNN 1D) & {reg['test_mae_days']} days & -- & -- \\")
    return "\n".join(rows)


def _latex_figures(summary: dict) -> str:
    blocks = []
    for p in summary.get("figures", []):
        base = os.path.basename(p)
        caption = {
            "learning_curves.png": "Training and validation loss / accuracy over epochs.",
            "confusion_matrix.png": "Confusion matrix on the held-out test set.",
            "split_map.png": "Geographic scatter of the spatially-grouped split "
                             "(train / validation / test).",
            "sea_sample_distribution.png": "Spatial distribution of the Southeast Asia "
                                           "subset (one panel per crop class).",
            "learning_curves_regression.png": "Training and validation loss / MAE (days) over epochs.",
            "regression_analysis.png": "Actual vs. predicted days after planting (HST).",
        }.get(base, base.replace("_", " "))
        blocks.append(
            "\\begin{figure}[h]\n\\centering\n"
            f"\\includegraphics[width=0.75\\textwidth]{{{base}}}\n"
            f"\\caption{{{caption}}}\n\\end{{figure}}")
    return "\n\n".join(blocks)


def _build_latex(summary: dict) -> str:
    cfg = summary["config"]
    is_ch = _is_cropharvest(summary)
    outline = (_methodology_outline(summary).replace("* ", "\\item ")
               .replace(" -> ", " $\\to$ "))
    discussion = _discussion_paragraph(summary).replace("R^2", "R$^2$")
    if is_ch:
        results_section = "Results"
        colspec = "lcc"
        header = "Metric & This run \\\\"
        caption = "Final test-set metrics of this run."
        bibliography = (
            "\\bibitem{tseng2021} G. Tseng, I. Zvonkov, C. L. Nakalembe, H. Kerner,\n"
            "\\emph{CropHarvest: A global dataset for crop-type classification},\n"
            "NeurIPS Datasets and Benchmarks, 2021.\n"
            "\\bibitem{congalton2019} R. G. Congalton and K. Green, \\emph{Assessing the Accuracy of\n"
            "Remotely Sensed Data: Principles and Practices}, 3rd ed., CRC Press, 2019.")
    else:
        results_section = "Results vs.\\ IFRI 2024 Baseline"
        colspec = "lccc"
        header = "Metric & This run & IFRI 2024 baseline & $\\Delta$ \\\\"
        caption = ("Final test-set metrics of this run against the benchmarks reported in the\n"
                   "2024 Laporan Akhir IFRI \\cite{ifri2024}.")
        bibliography = (
            "\\bibitem{ifri2024} Tim IFRI, \\emph{Laporan Akhir IFRI 2024} (Metode, Hasil dan\n"
            "Pembahasan, Kesimpulan), 2024.\n"
            "\\bibitem{congalton2019} R. G. Congalton and K. Green, \\emph{Assessing the Accuracy of\n"
            "Remotely Sensed Data: Principles and Practices}, 3rd ed., CRC Press, 2019.")
    return rf"""\documentclass[11pt]{{article}}
\usepackage[utf8]{{inputenc}}
\usepackage[margin=1in]{{geometry}}
\usepackage{{booktabs}}
\usepackage{{graphicx}}
\usepackage[hidelinks]{{hyperref}}
\graphicspath{{{{../figures/}}}}

\title{{Crop Analysis from Multimodal Remote Sensing\\Experiment Results}}
\author{{Ariana Nurlayla Syabandini}}
\date{{{summary['timestamp']} -- run {_tex_escape(summary['run_id'])} (seed {cfg['seed']})}}

\begin{{document}}
\maketitle

\section{{Methodology Outline}}
\begin{{itemize}}
{outline}
\end{{itemize}}

\section{{{results_section}}}
\begin{{table}}[h]
\centering
\begin{{tabular}}{{{colspec}}}
\toprule
{header}
\midrule
{_latex_table(summary)}
\bottomrule
\end{{tabular}}
\caption{{{caption}}}
\label{{tab:results}}
\end{{table}}

\section{{Figures}}
{_latex_figures(summary)}

\section{{Discussion Draft}}
{discussion}

\begin{{thebibliography}}{{9}}
{bibliography}
\end{{thebibliography}}

\end{{document}}
"""


def generate_thesis_report(summary: dict, run_dir: str):
    """Write thesis_results_draft.md and thesis_results_draft.tex from the run summary."""
    reports_dir = os.path.join(run_dir, "reports")
    md_path = os.path.join(reports_dir, "thesis_results_draft.md")
    tex_path = os.path.join(reports_dir, "thesis_results_draft.tex")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(_build_markdown(summary))
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(_build_latex(summary))
    print(f"[report] thesis draft -> {md_path} and {tex_path}")
    return md_path, tex_path
