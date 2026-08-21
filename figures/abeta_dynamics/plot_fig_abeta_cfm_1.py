#!/usr/bin/env python
"""Reproduce the CFM publication figures from a compact plotting-data package.

Normal use reads only files below ``data/``.  ``--prepare-data`` is provided to
create that package once from the analysis outputs in this repository.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd
from PIL import Image


HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
OUTPUT = HERE / "outputs"
PROJECT = HERE.parents[1]

SCHEMES = list("ABCDEF")
SCHEME_COLORS = ["#3C5488", "#4DBBD5", "#00A087", "#E64B35", "#F39B7F", "#8491B4"]
SUBTYPE_COLORS = {0: "#3C5488", 1: "#E64B35"}
STAGE_COLORS = SCHEME_COLORS
STAGE_GROUPS = ["stage0", "bin0", "bin1", "bin2", "bin3"]
STAGE_LABELS = {"stage0": "s0", "bin0": "s1-3", "bin1": "s4-8", "bin2": "s9-20", "bin3": "s21-23"}


def _read(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing plotting-data file: {path}")
    frame = pd.read_csv(path)
    if columns:
        missing = [name for name in columns if name not in frame.columns]
        if missing:
            raise ValueError(f"{path.name} is missing columns: {missing}")
    return frame


def prepare_data() -> None:
    """Export only values and geometry actually used by the figures."""
    integrated = DATA / "integrated_dynamics"
    brain = DATA / "longitudinal_brain_maps"
    integrated.mkdir(parents=True, exist_ok=True)
    brain.mkdir(parents=True, exist_ok=True)

    individual_rows: list[pd.DataFrame] = []
    distribution_rows: list[pd.DataFrame] = []
    for scheme in SCHEMES:
        validation = PROJECT / "Conditional_OT_CFM" / f"outputs_ae_latent_ot_cfm_bin_scheme{scheme}" / "quantitative_validation"
        ind = pd.read_csv(validation / "individual_transition_metrics.csv")
        ind = ind.loc[(ind["analysis_set"] == "exact_one_bin") & (ind["method"] == "model"), ["rmse_followup"]].copy()
        ind.insert(0, "scheme", scheme)
        individual_rows.append(ind)

        pop = pd.read_csv(validation / "population_bin_forecast_metrics.csv")
        pop = pop.loc[pop["method"] == "model", ["sliced_wasserstein_roi"]].copy()
        pop.insert(0, "scheme", scheme)
        distribution_rows.append(pop)

    pd.concat(individual_rows, ignore_index=True).to_csv(integrated / "individual_followup_rmse.csv", index=False)
    pd.concat(distribution_rows, ignore_index=True).to_csv(integrated / "distribution_sliced_wasserstein.csv", index=False)

    direct = PROJECT / "Conditional_OT_CFM" / "outputs_umap2d_ot_cfm_bin_schemeB"
    decoded = PROJECT / "Conditional_OT_CFM" / "outputs_ae_latent_ot_cfm_bin_schemeB"
    error_dir = PROJECT / "OT_CFM_Visualization" / "both_subtypes_decoded_vs_observed_68roi_analysis_Abeta_bin_schemeB"

    coord = pd.read_csv(direct / "umap2d_coordinates.csv")
    display = coord["used_for_umap2d_display"].astype(str).str.lower().isin(["true", "1", "yes"])
    coord.loc[display, ["umap2d_1", "umap2d_2", "dynamics_bin"]].to_csv(integrated / "direct_umap_background.csv", index=False)
    pd.read_csv(direct / "umap2d_velocity_field_vectors.csv")[["subtype", "bin_raw", "coord1", "coord2", "arrow_dx_plotted", "arrow_dy_plotted", "marker_type"]].to_csv(integrated / "velocity_vectors.csv", index=False)
    pd.read_csv(direct / "sample_trajectories_from_stage1.csv")[["subtype", "sample_id", "time_index", "coord1", "coord2"]].to_csv(integrated / "sample_trajectories.csv", index=False)
    pd.read_csv(direct / "mean_trajectories_shared_root.csv")[["subtype", "time_index", "coord1", "coord2"]].to_csv(integrated / "mean_trajectories.csv", index=False)
    pd.read_csv(direct / "root_and_branch_anchors.csv")[["anchor_type", "subtype", "coord1", "coord2"]].to_csv(integrated / "trajectory_anchors.csv", index=False)

    coord = pd.read_csv(decoded / "umap_real_data_coordinates.csv")
    display = coord["used_for_umap_display"].astype(str).str.lower().isin(["true", "1", "yes"])
    coord.loc[display, ["umap1", "umap2", "dynamics_bin", "dynamics_stage_group_label"]].to_csv(integrated / "decoded_umap_background.csv", index=False)
    pd.read_csv(decoded / "umap_decoded_mean_path_by_subtype.csv")[["branch_subtype", "time_index", "umap1_mean", "umap2_mean"]].to_csv(integrated / "decoded_mean_paths.csv", index=False)

    errors = pd.read_csv(error_dir / "roi_error_long_all_subtypes.csv")
    errors = errors.loc[errors["observed_group"].isin(STAGE_GROUPS), ["subtype", "observed_group", "roi", "abs_error"]]
    errors.to_csv(integrated / "heatmap_absolute_error.csv", index=False)
    pd.read_csv(error_dir / "cortical68_feature_order.csv")[["roi_index", "roi"]].to_csv(integrated / "cortical68_roi_order.csv", index=False)
    pd.DataFrame({"observed_group": STAGE_GROUPS, "display_label": [STAGE_LABELS[x] for x in STAGE_GROUPS]}).to_csv(integrated / "stage_group_labels.csv", index=False)

    source_brain = PROJECT / "OT_CFM_Visualization" / "selected_longitudinal_prediction_lh_lateral_Abeta_bin_schemeB"
    values = pd.read_csv(source_brain / "predicted_vs_observed_long.csv")
    values = values.loc[(values["hemi"] == "LH") & values["PTID"].isin(["021_S_4254", "041_S_0679"]), ["PTID", "model_time", "region", "observed_suvr", "predicted_suvr"]].copy()
    values.insert(1, "research_group", values["PTID"].map({"021_S_4254": "CN", "041_S_0679": "MCI"}))
    values.to_csv(brain / "lh_observed_decoded_values.csv", index=False)
    pd.read_csv(source_brain / "lh_lateral_color_scale.csv").to_csv(brain / "color_scale.csv", index=False)

    surface_source = PROJECT / "SuStaIn_Main" / "Publication_Figure_Source_Data" / "data" / "fig_abeta_sustain_1" / "fsaverage5_surface_data.npz"
    surface = np.load(surface_source)
    np.savez_compressed(
        brain / "fsaverage5_lh_surface.npz",
        coords=surface["lh_coords"], faces=surface["lh_faces"], sulc=surface["lh_sulc"],
        labels=surface["lh_labels"], names=surface["lh_names"],
    )
    print(f"Prepared compact plotting data in: {DATA}")


def configure_style() -> None:
    plt.rcParams.update({
        "font.family": "Times New Roman", "font.weight": "bold", "font.size": 15,
        "axes.titleweight": "bold", "axes.labelweight": "bold", "axes.titlesize": 15,
        "axes.labelsize": 15, "xtick.labelsize": 14, "ytick.labelsize": 14,
        "legend.fontsize": 10, "figure.facecolor": "white", "axes.facecolor": "white",
    })


def _stage_color(value: float) -> str:
    return STAGE_COLORS[int(value) % len(STAGE_COLORS)]


def _umap_background(ax, frame: pd.DataFrame, x: str, y: str, *, filled: bool, legend: bool = False) -> None:
    for stage in sorted(frame["dynamics_bin"].dropna().astype(int).unique()):
        group = frame[frame["dynamics_bin"].astype(int) == stage]
        kwargs = dict(s=15, alpha=0.48, zorder=1, label=f"s{stage}" if legend else "_nolegend_")
        if filled:
            kwargs["color"] = _stage_color(stage)
        else:
            kwargs.update(facecolors="none", edgecolors=_stage_color(stage), linewidths=0.7)
        ax.scatter(group[x], group[y], **kwargs)
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.grid(alpha=0.22)


def _anchors(ax, anchors: pd.DataFrame) -> None:
    root = anchors[anchors["anchor_type"] == "shared_root"]
    if root.empty:
        return
    root_xy = root[["coord1", "coord2"]].iloc[0].to_numpy(float)
    ax.scatter(*root_xy, s=75, marker="s", color="black", label="Stage0", zorder=9)
    for _, row in anchors.dropna(subset=["subtype"]).iterrows():
        subtype = int(row["subtype"])
        target = row[["coord1", "coord2"]].to_numpy(float)
        ax.annotate("", xy=target, xytext=root_xy, arrowprops=dict(arrowstyle="->", color=SUBTYPE_COLORS[subtype], lw=2.2))
        ax.scatter(*target, s=65, marker="o" if subtype == 0 else "h", color=SUBTYPE_COLORS[subtype], edgecolor="black", linewidth=0.4, zorder=8)


def _panel_velocity(ax, data: Path) -> None:
    background = _read(data / "direct_umap_background.csv")
    vectors = _read(data / "velocity_vectors.csv")
    anchors = _read(data / "trajectory_anchors.csv")
    _umap_background(ax, background, "umap2d_1", "umap2d_2", filled=False)
    _anchors(ax, anchors)
    for subtype, group in vectors.groupby("subtype"):
        subtype = int(subtype)
        marker = "o" if subtype == 0 else "h"
        colors = [_stage_color(x) for x in group["bin_raw"].fillna(0)]
        ax.scatter(group["coord1"], group["coord2"], s=34, c=colors, marker=marker, edgecolor="black", linewidth=0.3, alpha=0.86, label=f"velocity points subtype {subtype + 1}", zorder=6)
        ax.quiver(group["coord1"], group["coord2"], group["arrow_dx_plotted"], group["arrow_dy_plotted"], angles="xy", scale_units="xy", scale=1, color=SUBTYPE_COLORS[subtype], width=0.004, alpha=0.84, label=f"velocity subtype {subtype + 1}", zorder=7)
    ax.legend(loc="lower right", frameon=False, fontsize=9)


def _panel_trajectories(ax, data: Path) -> None:
    background = _read(data / "direct_umap_background.csv")
    samples = _read(data / "sample_trajectories.csv")
    means = _read(data / "mean_trajectories.csv")
    anchors = _read(data / "trajectory_anchors.csv")
    _umap_background(ax, background, "umap2d_1", "umap2d_2", filled=False)
    _anchors(ax, anchors)
    labeled: set[int] = set()
    for (subtype, _sample), group in samples.sort_values(["subtype", "sample_id", "time_index"]).groupby(["subtype", "sample_id"]):
        subtype = int(subtype)
        ax.plot(group["coord1"], group["coord2"], color=SUBTYPE_COLORS[subtype], alpha=0.28, lw=0.8, label=f"trajectory subtype {subtype + 1}" if subtype not in labeled else None, zorder=3)
        labeled.add(subtype)
    for subtype, group in means.sort_values(["subtype", "time_index"]).groupby("subtype"):
        subtype = int(subtype)
        ax.plot(group["coord1"], group["coord2"], color=SUBTYPE_COLORS[subtype], lw=2.7, marker="o" if subtype == 0 else "h", markevery=10, ms=4, mec="black", mew=0.3, label=f"mean path subtype {subtype + 1}", zorder=6)
    handles, labels = ax.get_legend_handles_labels()
    keep = [(h, l) for h, l in zip(handles, labels) if l and l != "Stage0"]
    ax.legend([x[0] for x in keep], [x[1] for x in keep], loc="lower right", frameon=False, fontsize=9)


def _panel_decoded(ax, data: Path) -> None:
    background = _read(data / "decoded_umap_background.csv")
    paths = _read(data / "decoded_mean_paths.csv")
    _umap_background(ax, background, "umap1", "umap2", filled=True, legend=True)
    background_handles, background_labels = ax.get_legend_handles_labels()
    for subtype, group in paths.sort_values(["branch_subtype", "time_index"]).groupby("branch_subtype"):
        subtype = int(subtype)
        ax.plot(group["umap1_mean"], group["umap2_mean"], color=SUBTYPE_COLORS[subtype], lw=2.3, marker="o", ms=2.5, label=f"decoded mean path subtype {subtype + 1}", zorder=5)
        ax.scatter(group["umap1_mean"].iloc[0], group["umap2_mean"].iloc[0], color=SUBTYPE_COLORS[subtype], marker="s", s=40, zorder=6)
        ax.scatter(group["umap1_mean"].iloc[-1], group["umap2_mean"].iloc[-1], color=SUBTYPE_COLORS[subtype], marker="*", s=55, zorder=6)
    stage_legend = ax.legend(background_handles, background_labels, loc="upper left", ncol=3, fontsize=8, frameon=True)
    ax.add_artist(stage_legend)
    handles, labels = ax.get_legend_handles_labels()
    paths_only = [(handle, label) for handle, label in zip(handles, labels) if label.startswith("decoded mean path")]
    ax.legend([x[0] for x in paths_only], [x[1] for x in paths_only], loc="lower right", fontsize=8, frameon=True)


def _panel_heatmaps(fig, axes, cax, data: Path) -> None:
    errors = _read(data / "heatmap_absolute_error.csv")
    roi_order = _read(data / "cortical68_roi_order.csv").sort_values("roi_index")["roi"].tolist()
    labels = dict(zip(_read(data / "stage_group_labels.csv")["observed_group"], _read(data / "stage_group_labels.csv")["display_label"]))
    vmax = float(np.nanpercentile(errors["abs_error"], 99.5))
    image = None
    for ax, subtype in zip(axes, [0, 1]):
        pivot = errors[errors["subtype"] == subtype].pivot(index="observed_group", columns="roi", values="abs_error").reindex(index=STAGE_GROUPS, columns=roi_order)
        image = ax.imshow(pivot.to_numpy(), aspect="auto", interpolation="nearest", vmin=0, vmax=vmax, cmap="viridis")
        ax.set_yticks(range(len(STAGE_GROUPS)), [labels[x] for x in STAGE_GROUPS])
        ax.set_xticks([])
        ax.set_title(f"Subtype {subtype + 1}", pad=2)
    colorbar = fig.colorbar(image, cax=cax)
    colorbar.set_label("Absolute Error", fontweight="bold")


def plot_integrated_panel(out_file: Path, dpi: int) -> None:
    data = DATA / "integrated_dynamics"
    configure_style()
    fig = plt.figure(figsize=(18, 9))
    left, right, bottom, top = 0.045, 0.985, 0.075, 0.965
    hgap, vgap = 0.065, 0.095
    width = (right - left - 2 * hgap) / 3
    height = (top - bottom - vgap) / 2
    xs = [left + i * (width + hgap) for i in range(3)]
    ys = [bottom + height + vgap, bottom]

    ax_a = fig.add_axes([xs[0], ys[0], width, height])
    ax_b = fig.add_axes([xs[1], ys[0], width, height])
    ax_d = fig.add_axes([xs[0], ys[1], width, height])
    ax_e = fig.add_axes([xs[1], ys[1], width, height])
    ax_f = fig.add_axes([xs[2], ys[1], width, height])

    individual = _read(data / "individual_followup_rmse.csv")
    medians = [individual.loc[individual["scheme"] == s, "rmse_followup"].median() for s in SCHEMES]
    bars = ax_a.bar(SCHEMES, medians, color=SCHEME_COLORS, edgecolor="black", linewidth=0.6, width=0.68)
    ax_a.bar_label(bars, fmt="%.3f", padding=3, fontsize=11, fontweight="bold")
    ax_a.set(xlabel="Binning scheme", ylabel="Median follow-up RMSE", title="Individual one-step prediction", ylim=(0, 0.15))
    ax_a.grid(axis="y", alpha=0.25)

    distribution = _read(data / "distribution_sliced_wasserstein.csv")
    values = [distribution.loc[distribution["scheme"] == s, "sliced_wasserstein_roi"].dropna().to_numpy() for s in SCHEMES]
    box = ax_b.boxplot(values, labels=SCHEMES, patch_artist=True, widths=0.62, medianprops=dict(color="black", lw=1.5), flierprops=dict(marker="o", markersize=3, markerfacecolor="white", markeredgecolor="#555", alpha=0.7))
    for patch, color in zip(box["boxes"], SCHEME_COLORS):
        patch.set(facecolor=color, alpha=0.82, edgecolor="black")
    for i, vals in enumerate(values, 1):
        q1, q3 = np.nanpercentile(vals, [25, 75])
        upper_limit = q3 + 1.5 * (q3 - q1)
        upper_whisker = np.max(vals[vals <= upper_limit])
        ax_b.text(i, upper_whisker + 0.008, f"{np.nanmedian(vals):.3f}", ha="center", fontsize=10, fontweight="bold")
    ax_b.set(xlabel="Binning scheme", ylabel="Sliced Wasserstein", title="Distribution-level model comparison")
    ax_b.grid(axis="y", alpha=0.25)

    stack_h = height * 0.72
    gap = height * 0.07
    heat_h = (stack_h - gap) / 2
    stack_y = ys[0] + (height - stack_h) / 2
    cbar_w, cbar_pad = width * 0.035, width * 0.035
    heat_w = width - cbar_w - cbar_pad
    heat_axes = [fig.add_axes([xs[2], stack_y + heat_h + gap, heat_w, heat_h]), fig.add_axes([xs[2], stack_y, heat_w, heat_h])]
    cax = fig.add_axes([xs[2] + heat_w + cbar_pad, stack_y, cbar_w, stack_h])
    _panel_heatmaps(fig, heat_axes, cax, data)
    _panel_velocity(ax_d, data)
    _panel_trajectories(ax_e, data)
    _panel_decoded(ax_f, data)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _surface_data() -> dict[str, np.ndarray]:
    path = DATA / "longitudinal_brain_maps" / "fsaverage5_lh_surface.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing plotting-data file: {path}")
    archive = np.load(path)
    return {name: archive[name] for name in archive.files}


def _vertex_map(regions: pd.Series, values: pd.Series, surface: dict[str, np.ndarray]) -> np.ndarray:
    lookup = {str(region).lower(): float(value) for region, value in zip(regions, values)}
    names = [str(x).lower() for x in surface["names"]]
    name_values = np.full(len(names), np.nan)
    for index, name in enumerate(names):
        if name in lookup:
            name_values[index] = lookup[name]
    labels = surface["labels"].astype(int)
    vertex = np.full(labels.shape, np.nan)
    valid = (labels >= 0) & (labels < len(name_values))
    vertex[valid] = name_values[labels[valid]]
    return vertex


def _brain_axis(ax, rows: pd.DataFrame, value_col: str, surface: dict[str, np.ndarray], vmin: float, vmax: float, title: str) -> None:
    from nilearn import plotting

    plotting.plot_surf_stat_map(
        (surface["coords"], surface["faces"]), _vertex_map(rows["region"], rows[value_col], surface),
        bg_map=surface["sulc"], hemi="left", view="lateral", cmap="turbo", colorbar=False,
        threshold=None, bg_on_data=True, alpha=1, vmin=vmin, vmax=vmax, axes=ax, figure=ax.figure,
    )
    ax.set_title(title, fontsize=12, fontweight="bold", pad=0)
    ax.set_axis_off()


def plot_brain_overview(out_file: Path, *, subjects: list[str], dpi: int) -> None:
    data = _read(DATA / "longitudinal_brain_maps" / "lh_observed_decoded_values.csv")
    scale = _read(DATA / "longitudinal_brain_maps" / "color_scale.csv").iloc[0]
    vmin, vmax = float(scale["vmin"]), float(scale["vmax"])
    surface = _surface_data()
    n_subjects, n_times = len(subjects), 4
    fig = plt.figure(figsize=(20.4, 5.3))
    nrows, ncols = n_subjects, 2 * n_times
    for subject_index, subject in enumerate(subjects):
        subject_data = data[data["PTID"] == subject]
        group_label = subject_data["research_group"].iloc[0]
        for time_index, model_time in enumerate(sorted(subject_data["model_time"].unique())):
            rows = subject_data[subject_data["model_time"] == model_time]
            for kind_index, (kind, column) in enumerate([("Obs", "observed_suvr"), ("Decoded", "predicted_suvr")]):
                row = subject_index
                col = time_index + kind_index * n_times
                ax = fig.add_subplot(nrows, ncols, row * ncols + col + 1, projection="3d")
                _brain_axis(ax, rows, column, surface, vmin, vmax, f"{kind} t={int(model_time)}")
                if col == 0:
                    ax.text2D(-0.16, 0.5, group_label, transform=ax.transAxes, rotation=90, va="center", fontsize=12, fontweight="bold")
    sm = ScalarMappable(norm=Normalize(vmin, vmax), cmap="turbo")
    sm.set_array([])
    color_axis = fig.add_axes([0.945, 0.18, 0.012, 0.64])
    colorbar = fig.colorbar(sm, cax=color_axis)
    colorbar.set_label("Tau PET SUVR", fontsize=12, fontweight="bold")
    fig.subplots_adjust(left=0.035, right=0.92, top=0.96, bottom=0.02, wspace=-0.10, hspace=0.02)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=dpi, bbox_inches="tight", pad_inches=0.02, transparent=False, facecolor="white")
    plt.close(fig)


def combine_figures(integrated_file: Path, overview_file: Path, out_file: Path) -> None:
    top = Image.open(integrated_file).convert("RGB")
    bottom = Image.open(overview_file).convert("RGB")
    target_width = max(top.width, bottom.width)
    if top.width != target_width:
        top = top.resize((target_width, round(top.height * target_width / top.width)), Image.Resampling.LANCZOS)
    if bottom.width != target_width:
        bottom = bottom.resize((target_width, round(bottom.height * target_width / bottom.width)), Image.Resampling.LANCZOS)
    gap = round(target_width * 0.012)
    canvas = Image.new("RGB", (target_width, top.height + gap + bottom.height), "white")
    canvas.paste(top, (0, 0))
    canvas.paste(bottom, (0, top.height + gap))
    out_file.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_file, quality=95, subsampling=0, dpi=(300, 300))


def validate_data() -> None:
    integrated = DATA / "integrated_dynamics"
    brain = DATA / "longitudinal_brain_maps"
    required = [
        integrated / "individual_followup_rmse.csv", integrated / "distribution_sliced_wasserstein.csv",
        integrated / "direct_umap_background.csv", integrated / "velocity_vectors.csv",
        integrated / "sample_trajectories.csv", integrated / "mean_trajectories.csv",
        integrated / "trajectory_anchors.csv", integrated / "decoded_umap_background.csv",
        integrated / "decoded_mean_paths.csv", integrated / "heatmap_absolute_error.csv",
        integrated / "cortical68_roi_order.csv", integrated / "stage_group_labels.csv",
        brain / "lh_observed_decoded_values.csv", brain / "color_scale.csv", brain / "fsaverage5_lh_surface.npz",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Run with --prepare-data first. Missing:\n" + "\n".join(missing))
    errors = _read(integrated / "heatmap_absolute_error.csv")
    values = _read(brain / "lh_observed_decoded_values.csv")
    assert len(errors) == 2 * 5 * 68, f"Expected 680 heatmap cells, found {len(errors)}"
    assert len(values) == 2 * 4 * 34, f"Expected 272 LH ROI rows, found {len(values)}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-data", action="store_true", help="Export the compact plotting-data package before plotting.")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    if args.prepare_data:
        prepare_data()
    validate_data()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    integrated = OUTPUT / "integrated_dynamics_visualization_panel_2x3.png"
    overview = OUTPUT / "overview_2ptid_observed_then_decoded_lh_lateral_2x8.png"
    combined = OUTPUT / "fig_abeta_cfm_1.jpg"
    plot_integrated_panel(integrated, args.dpi)
    plot_brain_overview(overview, subjects=["021_S_4254", "041_S_0679"], dpi=args.dpi)
    combine_figures(integrated, overview, combined)
    for path in [integrated, overview, combined]:
        print(path)


if __name__ == "__main__":
    main()
