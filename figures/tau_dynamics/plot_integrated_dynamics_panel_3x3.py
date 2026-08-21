#!/usr/bin/env python
"""Reproduce the integrated 3x3 dynamics figure from submission-ready plot data only."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
DEFAULT_OUTPUT = HERE / "outputs" / "integrated_dynamics_visualization_panel_3x3.png"

STAGE_ORDER = ["stage0", "bin0", "bin1", "bin2", "bin3"]
STAGE_TICK_LABELS = ["s0", "s1-2", "s3-6", "s7-15", "s16-20"]
STAGE_COLORS = ["#455F98", "#4DB3C7", "#00A087", "#EE9577", "#E64B35"]
SUBTYPE_COLORS = {0: "#455F98", 1: "#E64B35"}
METHOD_ORDER = ["persistence", "roi_linear", "latent_linear", "model", "mmfm", "vgfm"]
METHOD_LABELS = ["persist", "ROI", "latent", "OT-CFM", "MMFM", "VGFM"]
METHOD_COLORS = ["#455F98", "#4DB3C7", "#00A087", "#E64B35", "#7E57C2", "#E69F00"]
METRICS = [
    ("energy_distance_roi", "Energy Distance"),
    ("mmd_rbf_roi", "RBF MMD"),
    ("sliced_wasserstein_roi", "Sliced Wasserstein"),
]
TRANSITION_MARKERS = ["o", "s", "^", "D", "P", "X", "v", "<", ">", "h"]
STAGE_LEGEND_LABELS = {"S0", "S1-2", "S3-6", "S7-15", "S16-20", "s0", "s1-2", "s3-6", "s7-15", "s16-20"}


def load_csv(name: str, required: list[str]) -> pd.DataFrame:
    path = DATA / name
    if not path.exists():
        raise FileNotFoundError(f"Missing plotting data: {path}")
    table = pd.read_csv(path)
    missing = [column for column in required if column not in table.columns]
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")
    return table


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 16,
            "font.weight": "bold",
            "axes.labelsize": 16,
            "axes.labelweight": "bold",
            "axes.titlesize": 16,
            "axes.titleweight": "bold",
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 16,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "svg.fonttype": "none",
        }
    )


def subtype_label(subtype: int) -> str:
    return f"Subtype {int(subtype) + 1}"


def subtype_marker(subtype: int) -> str:
    return {0: "o", 1: "h"}.get(int(subtype), "o")


def dedupe_legend(ax, *, loc: str = "best", frameon: bool = False, exclude: set[str] | None = None) -> None:
    handles, labels = ax.get_legend_handles_labels()
    kept_h, kept_l, seen = [], [], set()
    excluded = set() if exclude is None else exclude
    for handle, label in zip(handles, labels):
        if not label or label.startswith("_") or label in seen or label in excluded:
            continue
        seen.add(label)
        kept_h.append(handle)
        kept_l.append(label)
    if kept_h:
        legend = ax.legend(kept_h, kept_l, loc=loc, frameon=frameon)
        for text in legend.get_texts():
            text.set_fontweight("bold")


def draw_identity(ax, x, y) -> None:
    values = np.concatenate([np.asarray(x, float), np.asarray(y, float)])
    values = values[np.isfinite(values)]
    lo, hi = float(values.min()), float(values.max())
    pad = max((hi - lo) * 0.08, 1e-3)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "--", color="0.35", linewidth=1.0)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)


def plot_background(ax, points: pd.DataFrame, *, open_markers: bool, size: float, alpha: float) -> None:
    for stage in sorted(points["dynamics_bin"].dropna().astype(int).unique()):
        group = points.loc[points["dynamics_bin"].astype(int) == stage]
        label = str(group["stage_label"].iloc[0])
        kwargs = {"s": size, "alpha": alpha, "label": label, "zorder": 1}
        if open_markers:
            kwargs.update(facecolors="none", edgecolors=STAGE_COLORS[stage], linewidths=0.75)
        else:
            kwargs.update(color=STAGE_COLORS[stage])
        ax.scatter(group["umap1"], group["umap2"], **kwargs)


def draw_anchors(ax, anchors: pd.DataFrame) -> None:
    root = anchors.loc[anchors["anchor_type"] == "shared_root"]
    root_coord = root[["coord1", "coord2"]].iloc[0].to_numpy(float)
    ax.scatter(*root_coord, color="black", s=78, marker="s", label="Stage0", zorder=9)
    branches = anchors.loc[pd.to_numeric(anchors["subtype"], errors="coerce").notna()].copy()
    for row in branches.itertuples(index=False):
        subtype = int(row.subtype)
        color = SUBTYPE_COLORS[subtype]
        ax.annotate(
            "",
            xy=(row.coord1, row.coord2),
            xytext=(root_coord[0], root_coord[1]),
            arrowprops=dict(
                arrowstyle="-|>", color=color, lw=2.3, mutation_scale=13,
                shrinkA=5, shrinkB=5, alpha=0.9,
            ),
            zorder=7,
        )
        ax.scatter(
            row.coord1, row.coord2, color=color, s=64, marker=subtype_marker(subtype),
            edgecolors="black", linewidths=0.45, zorder=8,
        )


def setup_umap_axis(ax, points: pd.DataFrame) -> None:
    plot_background(ax, points, open_markers=True, size=18, alpha=0.5)
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.grid(alpha=0.22)


def plot_velocity(ax, points: pd.DataFrame, vectors: pd.DataFrame, anchors: pd.DataFrame) -> None:
    setup_umap_axis(ax, points)
    draw_anchors(ax, anchors)
    for subtype, group in vectors.groupby("subtype"):
        subtype = int(subtype)
        point_colors = [STAGE_COLORS[int(value)] for value in group["bin_raw"]]
        marker = str(group["marker_type"].dropna().iloc[0])
        ax.scatter(
            group["coord1"], group["coord2"], s=36, c=point_colors, marker=marker,
            edgecolors="black", linewidths=0.3, alpha=0.86,
            label=f"velocity points subtype {subtype + 1}", zorder=6,
        )
        ax.quiver(
            group["coord1"], group["coord2"], group["arrow_dx_plotted"], group["arrow_dy_plotted"],
            angles="xy", scale_units="xy", scale=1.0, color=SUBTYPE_COLORS[subtype],
            width=0.004, alpha=0.84, label=f"velocity subtype {subtype + 1}", zorder=7,
        )
    dedupe_legend(ax, exclude=STAGE_LEGEND_LABELS)
    ax.set_aspect("auto")


def plot_trajectories(
    ax, points: pd.DataFrame, samples: pd.DataFrame, means: pd.DataFrame, anchors: pd.DataFrame
) -> None:
    setup_umap_axis(ax, points)
    draw_anchors(ax, anchors)
    labeled = set()
    samples = samples.sort_values(["subtype", "sample_id", "time_index"])
    for (subtype, _), group in samples.groupby(["subtype", "sample_id"]):
        subtype = int(subtype)
        ax.plot(
            group["coord1"], group["coord2"], color=SUBTYPE_COLORS[subtype], alpha=0.4,
            linewidth=1.0, label=f"trajectory subtype {subtype + 1}" if subtype not in labeled else None,
            zorder=4,
        )
        labeled.add(subtype)
    means = means.sort_values(["subtype", "time_index"])
    for subtype, group in means.groupby("subtype"):
        subtype = int(subtype)
        ax.plot(
            group["coord1"], group["coord2"], color=SUBTYPE_COLORS[subtype], linewidth=2.8,
            marker=subtype_marker(subtype), markersize=4.5, markeredgecolor="black",
            markeredgewidth=0.35, markevery=10, label=f"mean path subtype {subtype + 1}", zorder=6,
        )
    dedupe_legend(ax, exclude=STAGE_LEGEND_LABELS)
    ax.set_aspect("auto")


def plot_decoded(ax, points: pd.DataFrame, paths: pd.DataFrame) -> None:
    plot_background(ax, points, open_markers=False, size=14, alpha=0.45)
    paths = paths.sort_values(["subtype", "time_index"])
    for subtype, group in paths.groupby("subtype"):
        subtype = int(subtype)
        color = SUBTYPE_COLORS[subtype]
        ax.plot(
            group["umap1_mean"], group["umap2_mean"], color=color, linewidth=2.2,
            marker="o", markersize=2.6, label=f"decoded mean path subtype {subtype + 1}", zorder=5,
        )
        ax.scatter(group["umap1_mean"].iloc[0], group["umap2_mean"].iloc[0], color=color, s=42, marker="s", zorder=6)
        ax.scatter(group["umap1_mean"].iloc[-1], group["umap2_mean"].iloc[-1], color=color, s=55, marker="*", zorder=6)
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.grid(alpha=0.25)
    dedupe_legend(ax, frameon=True)


def plot_mse(ax, mse: pd.DataFrame) -> None:
    values = mse["reconstruction_mse_standardized"].to_numpy(float)
    ax.hist(values, bins=40, color=SUBTYPE_COLORS[0], alpha=0.88, edgecolor="white", linewidth=0.45)
    mean, median = float(np.mean(values)), float(np.median(values))
    ax.axvline(mean, color="#2F2F2F", linewidth=1.4, label=f"Mean: {mean:.3g}")
    ax.axvline(median, color="#E64B35", linewidth=1.2, linestyle="--", label=f"Median: {median:.3g}")
    ax.set_xlabel("Reconstruction MSE")
    ax.set_ylabel("Count")
    ax.grid(axis="y", alpha=0.22)
    dedupe_legend(ax, frameon=True)


def plot_heatmaps(fig, axes, cbar_ax, heatmap: pd.DataFrame) -> None:
    heatmap = heatmap.copy()
    heatmap["roi_index"] = heatmap["roi_index"].astype(int)
    roi_info = heatmap[["roi_index", "roi_short"]].drop_duplicates().sort_values("roi_index")
    roi_indices = roi_info["roi_index"].tolist()
    roi_labels = roi_info["roi_short"].astype(str).tolist()
    vmax = float(np.nanpercentile(heatmap["abs_error"].to_numpy(float), 99.5))
    image = None
    for panel_index, (ax, subtype) in enumerate(zip(axes, sorted(heatmap["subtype"].unique()))):
        group = heatmap.loc[heatmap["subtype"] == subtype]
        pivot = group.pivot(index="observed_group", columns="roi_index", values="abs_error")
        pivot = pivot.reindex(index=STAGE_ORDER, columns=roi_indices)
        image = ax.imshow(pivot.to_numpy(float), aspect="auto", interpolation="nearest", vmin=0, vmax=vmax, cmap="viridis")
        ax.set_yticks(np.arange(len(STAGE_ORDER)))
        ax.set_yticklabels(STAGE_TICK_LABELS)
        ax.set_title(subtype_label(int(subtype)))
        if panel_index == 1:
            ax.set_xticks(np.arange(len(roi_indices)))
            ax.set_xticklabels(roi_labels, rotation=90, fontsize=5, ha="center", va="top")
            ax.tick_params(axis="x", length=1.8, pad=1)
        else:
            ax.set_xticks([])
    colorbar = fig.colorbar(image, cax=cbar_ax)
    colorbar.set_label("Absolute Error", fontsize=16, fontweight="bold")
    colorbar.ax.tick_params(labelsize=12)


def plot_individual(ax, individual: pd.DataFrame) -> None:
    for subtype, group in individual.groupby("subtype"):
        subtype = int(subtype)
        ax.scatter(
            group["delta_global_true"], group["delta_global_pred"], alpha=0.72, s=62,
            color=SUBTYPE_COLORS[subtype], label=subtype_label(subtype), linewidths=0.55,
        )
    draw_identity(ax, individual["delta_global_true"], individual["delta_global_pred"])
    ax.set_xlabel("True global tau change")
    ax.set_ylabel("Predicted global tau change")
    dedupe_legend(ax, loc="lower right", frameon=True)


def plot_population(ax, population: pd.DataFrame) -> None:
    model = population.loc[population["method"] == "model"].copy()
    transitions = sorted({(int(row.source_bin), int(row.target_bin)) for row in model.itertuples()})
    marker_map = {transition: TRANSITION_MARKERS[i] for i, transition in enumerate(transitions)}
    for (subtype, source_bin, target_bin), group in model.groupby(["subtype", "source_bin", "target_bin"]):
        subtype = int(subtype)
        transition = (int(source_bin), int(target_bin))
        ax.scatter(
            group["true_global_mean"], group["pred_global_mean"], s=84, alpha=0.82,
            color=SUBTYPE_COLORS[subtype], marker=marker_map[transition], edgecolors="white", linewidths=0.45,
        )
    draw_identity(ax, model["true_global_mean"], model["pred_global_mean"])
    handles = [Patch(facecolor=SUBTYPE_COLORS[s], edgecolor="white", label=subtype_label(s)) for s in [0, 1]]
    handles += [
        Line2D([0], [0], marker=marker, linestyle="None", markerfacecolor="white", markeredgecolor="0.25",
               color="0.25", markersize=8, label=f"{source}->{target}")
        for (source, target), marker in marker_map.items()
    ]
    legend = ax.legend(handles=handles, loc="lower right", ncol=2, frameon=True, handlelength=1.2,
                       handletextpad=0.35, columnspacing=0.7, labelspacing=0.28, borderpad=0.28)
    for text in legend.get_texts():
        text.set_fontweight("bold")
    ax.set_xlabel("Observed target global mean")
    ax.set_ylabel("Predicted target global mean")


def plot_distribution(ax, population: pd.DataFrame, metric: str, title: str, *, ylabel: bool) -> None:
    positions = np.arange(1, len(METHOD_ORDER) + 1)
    pivot = population.pivot_table(index="task_id", columns="method", values=metric, aggfunc="mean")
    for _, row in pivot.iterrows():
        values = [row.get(method, np.nan) for method in METHOD_ORDER]
        valid = [(x, y) for x, y in zip(positions, values) if np.isfinite(y)]
        if len(valid) >= 2:
            ax.plot([x for x, _ in valid], [y for _, y in valid], color="0.80", linewidth=1.25, alpha=0.58, zorder=0)
    data = [
        pd.to_numeric(population.loc[population["method"] == method, metric], errors="coerce").dropna().to_numpy(float)
        for method in METHOD_ORDER
    ]
    boxes = ax.boxplot(
        data, positions=positions, widths=0.52, patch_artist=True, showfliers=False,
        medianprops={"color": "black", "linewidth": 1.0}, boxprops={"linewidth": 0.7},
        whiskerprops={"linewidth": 0.7}, capprops={"linewidth": 0.7},
    )
    for box, color in zip(boxes["boxes"], METHOD_COLORS):
        box.set_facecolor(color)
        box.set_alpha(0.80)
    rng = np.random.default_rng(17)
    for position, values, color in zip(positions, data, METHOD_COLORS):
        ax.scatter(np.full(values.size, position) + rng.uniform(-0.055, 0.055, values.size), values,
                   s=62, color=color, linewidths=0.55, zorder=3)
    ax.set_xticks(positions)
    ax.set_xticklabels(METHOD_LABELS, rotation=35, ha="right")
    ax.text(0.5, 0.965, title, transform=ax.transAxes, ha="center", va="top", fontsize=16, fontweight="bold")
    if ylabel:
        ax.set_ylabel("Distance")
    ax.grid(axis="y", color="0.88", linewidth=0.65)


def style_axes(axes) -> None:
    for ax in axes:
        for label in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
            label.set_fontweight("bold")
        ax.xaxis.label.set_fontweight("bold")
        ax.yaxis.label.set_fontweight("bold")


def make_figure() -> plt.Figure:
    configure_style()
    mse = load_csv("reconstruction_mse.csv", ["reconstruction_mse_standardized"])
    points = load_csv("umap_background_points.csv", ["umap1", "umap2", "dynamics_bin", "stage_label"])
    vectors = load_csv("velocity_vectors.csv", ["subtype", "bin_raw", "coord1", "coord2", "arrow_dx_plotted", "arrow_dy_plotted", "marker_type"])
    samples = load_csv("sample_trajectories.csv", ["subtype", "sample_id", "time_index", "coord1", "coord2"])
    means = load_csv("mean_trajectories.csv", ["subtype", "time_index", "coord1", "coord2"])
    anchors = load_csv("trajectory_anchors.csv", ["anchor_type", "subtype", "coord1", "coord2"])
    decoded = load_csv("decoded_mean_trajectories.csv", ["subtype", "time_index", "umap1_mean", "umap2_mean"])
    heatmap = load_csv("roi_absolute_error_heatmap.csv", ["subtype", "observed_group", "roi_index", "roi_short", "abs_error"])
    individual = load_csv("individual_global_change_predictions.csv", ["subtype", "delta_global_true", "delta_global_pred"])
    population = load_csv("population_forecast_metrics.csv", ["task_id", "subtype", "source_bin", "target_bin", "method", "true_global_mean", "pred_global_mean", *[metric for metric, _ in METRICS]])

    fig = plt.figure(figsize=(27, 15), constrained_layout=False)
    left, right, bottom, top, vgap = 0.045, 0.985, 0.06, 0.965, 0.09
    cell_h = (top - bottom - 2 * vgap) / 3
    row_y = [top - (i + 1) * cell_h - i * vgap for i in range(3)]

    def row_geometry(n_columns: int, gap: float):
        width = (right - left - (n_columns - 1) * gap) / n_columns
        return [left + i * (width + gap) for i in range(n_columns)], width

    xs3, width3 = row_geometry(3, 0.065)
    xs4, width4 = row_geometry(4, 0.045)
    top_y, middle_y, bottom_y = row_y
    ax_mse = fig.add_axes([xs3[0], top_y, width3, cell_h])
    ax_heatmap_outer = fig.add_axes([xs3[1], top_y, width3, cell_h], frameon=False)
    ax_individual = fig.add_axes([xs3[2], top_y, width3, cell_h])
    ax_velocity = fig.add_axes([xs3[0], middle_y, width3, cell_h])
    ax_trajectories = fig.add_axes([xs3[1], middle_y, width3, cell_h])
    ax_decoded = fig.add_axes([xs3[2], middle_y, width3, cell_h])
    ax_population = fig.add_axes([xs4[0], bottom_y, width4, cell_h])
    distribution_axes = [fig.add_axes([x, bottom_y, width4, cell_h]) for x in xs4[1:]]
    ax_heatmap_outer.set_axis_off()

    stack_h = cell_h * 0.74
    inner_gap = cell_h * 0.085
    heatmap_h = (stack_h - inner_gap) / 2
    stack_y = top_y + (cell_h - stack_h) / 2
    cbar_w, cbar_pad = width3 * 0.035, width3 * 0.035
    heatmap_w = width3 - cbar_w - cbar_pad
    heatmap_axes = [
        fig.add_axes([xs3[1], stack_y + heatmap_h + inner_gap, heatmap_w, heatmap_h]),
        fig.add_axes([xs3[1], stack_y, heatmap_w, heatmap_h]),
    ]
    cbar_ax = fig.add_axes([xs3[1] + heatmap_w + cbar_pad, stack_y, cbar_w, stack_h])

    plot_mse(ax_mse, mse)
    plot_heatmaps(fig, heatmap_axes, cbar_ax, heatmap)
    plot_individual(ax_individual, individual)
    plot_velocity(ax_velocity, points, vectors, anchors)
    plot_trajectories(ax_trajectories, points, samples, means, anchors)
    plot_decoded(ax_decoded, points, decoded)
    plot_population(ax_population, population)
    for index, (axis, (metric, title)) in enumerate(zip(distribution_axes, METRICS)):
        plot_distribution(axis, population, metric, title, ylabel=index == 0)

    style_axes([ax_mse, *heatmap_axes, cbar_ax, ax_individual, ax_velocity, ax_trajectories, ax_decoded, ax_population, *distribution_axes])
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig = make_figure()
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
