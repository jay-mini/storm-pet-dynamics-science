#!/usr/bin/env python
"""Recreate the three-panel SuStaIn association figure and Tau-PET stage maps.

The script reads only the compact plotting tables stored beside it. Study-level
source tables and previously rendered panel images are not used.
"""

from __future__ import annotations

import argparse
from io import BytesIO
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd
from PIL import Image, ImageChops
import statsmodels.api as sm


HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data" / "fig_tau_sustain_ap_2"
OUTPUT_DIR = HERE / "outputs"
SURFACE_DATA = DATA_DIR / "fsaverage5_surface_data.npz"
STAGE_GROUPS = ["stage0", "stage1-2", "stage3-6", "stage7-15", "stage16-20"]


def load_surfaces(path: Path) -> dict[str, dict]:
    """Load the bundled fsaverage5 mesh and aparc labels without external files."""
    with np.load(path, allow_pickle=False) as archive:
        return {
            hemi: {
                "mesh": (archive[f"{hemi}_coords"], archive[f"{hemi}_faces"]),
                "sulc": archive[f"{hemi}_sulc"],
                "labels": archive[f"{hemi}_labels"],
                "names": archive[f"{hemi}_names"].astype(str).tolist(),
            }
            for hemi in ("lh", "rh")
        }


def roi_vector_to_vertices(values: pd.Series, surface: dict) -> np.ndarray:
    label_ids = {name.lower(): index for index, name in enumerate(surface["names"])}
    vertices = np.full(surface["labels"].shape, np.nan, dtype=float)
    for roi, value in values.items():
        key = str(roi).lower()
        key = key.removeprefix("ctx_lh_").removeprefix("ctx_rh_").removesuffix("_suvr")
        if key not in label_ids:
            raise ValueError(f"ROI {roi!r} is absent from the fsaverage aparc annotation")
        vertices[surface["labels"] == label_ids[key]] = float(value)
    return vertices


def crop_white(image: Image.Image, padding: int = 3) -> Image.Image:
    image = image.convert("RGB")
    background = Image.new("RGB", image.size, "white")
    box = ImageChops.difference(image, background).getbbox()
    if box is None:
        return image
    left, top, right, bottom = box
    return image.crop(
        (
            max(0, left - padding),
            max(0, top - padding),
            min(image.width, right + padding),
            min(image.height, bottom + padding),
        )
    )


def render_surface_thumbnail(
    surface: dict,
    values: pd.Series,
    *,
    hemi: str,
    view: str,
    cmap: str,
    vmin: float,
    vmax: float,
    dpi: int = 150,
) -> np.ndarray:
    from nilearn import plotting

    vertex_values = roi_vector_to_vertices(values, surface)
    fig = plt.figure(figsize=(2.3, 1.8), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    plotting.plot_surf_stat_map(
        surface["mesh"],
        stat_map=vertex_values,
        bg_map=surface["sulc"],
        hemi="left" if hemi == "lh" else "right",
        view=view,
        cmap=cmap,
        colorbar=False,
        threshold=None,
        bg_on_data=True,
        alpha=1.0,
        vmin=vmin,
        vmax=vmax,
        axes=ax,
        figure=fig,
    )
    ax.set_axis_off()
    buffer = BytesIO()
    fig.savefig(buffer, dpi=dpi, bbox_inches="tight", pad_inches=0.0, facecolor="white")
    plt.close(fig)
    buffer.seek(0)
    return np.asarray(crop_white(Image.open(buffer)))


def load_plot_data() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    brain = pd.read_csv(DATA_DIR / "dk68_residual_brainmap_values.csv")
    corr = pd.read_csv(DATA_DIR / "dk68_cross_stage_correlations.csv")
    mmse = pd.read_csv(DATA_DIR / "mmse_stage_observations.csv")
    apoe = pd.read_csv(DATA_DIR / "apoe4_stage_observations.csv")
    tau = pd.read_csv(DATA_DIR / "tau_lh_stage_mean_values.csv")
    return brain, corr, mmse, apoe, tau


def half_violin_with_points(
    axis: plt.Axes,
    values_by_group: list[np.ndarray],
    labels: list[str],
    colors: list[str],
) -> None:
    rng = np.random.default_rng(42)
    positions = np.arange(1, len(labels) + 1)
    for position, values, color in zip(positions, values_by_group, colors):
        values = np.asarray(values, dtype=float)
        low, high = values.min(), values.max()
        grid = np.linspace(low - 0.5, high + 0.5, 250)
        if len(values) > 1 and np.std(values, ddof=1) > 0:
            bandwidth = max(1.06 * np.std(values, ddof=1) * len(values) ** (-0.2), 1e-6)
            z = (grid[:, None] - values[None, :]) / bandwidth
            density = np.exp(-0.5 * z**2).sum(axis=1)
            density /= density.max()
            axis.fill_betweenx(
                grid,
                position - 0.42 * density,
                position,
                color=color,
                alpha=0.95,
                linewidth=0,
            )
        axis.scatter(
            position + 0.26 + rng.normal(0, 0.055, len(values)),
            values,
            s=16,
            color=color,
            alpha=0.8,
            edgecolors="none",
            zorder=3,
        )
    boxplot = axis.boxplot(
        values_by_group,
        positions=positions + 0.26,
        widths=0.18,
        patch_artist=True,
        showfliers=False,
        manage_ticks=False,
        boxprops={"facecolor": "white", "edgecolor": "black", "linewidth": 1.4},
        whiskerprops={"color": "black", "linewidth": 1.4},
        capprops={"color": "black", "linewidth": 1.4},
        medianprops={"color": "black", "linewidth": 1.8},
    )
    for group in ("boxes", "whiskers", "caps", "medians"):
        for artist in boxplot[group]:
            artist.set_zorder(4)
    axis.set_xticks(positions, labels)
    axis.set_xlim(0.4, len(labels) + 0.6)
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(False)


def make_four_panel_figure(
    brain: pd.DataFrame,
    corr: pd.DataFrame,
    mmse: pd.DataFrame,
    apoe: pd.DataFrame,
    surfaces: dict,
) -> plt.Figure:
    fig = plt.figure(figsize=(30, 6), constrained_layout=False)
    outer = fig.add_gridspec(
        1,
        4,
        width_ratios=[1.35, 1.0, 1.0, 1.0],
        left=0.025,
        right=0.985,
        bottom=0.10,
        top=0.90,
        wspace=0.28,
    )

    residual = brain.pivot(index="roi", columns="hemi", values="mean_residual_z_difference")
    limit = float(np.nanmax(np.abs(residual.to_numpy())))
    brain_grid = outer[0].subgridspec(
        2,
        3,
        width_ratios=[1.0, 1.0, 0.055],
        wspace=0.02,
        hspace=0.02,
    )
    surface_specs = [
        ("lh", "lateral", "L", "LH lateral"),
        ("rh", "lateral", "R", "RH lateral"),
        ("lh", "medial", "L", "LH medial"),
        ("rh", "medial", "R", "RH medial"),
    ]
    for index, (hemi, view, column, title) in enumerate(surface_specs):
        ax = fig.add_subplot(brain_grid[index // 2, index % 2])
        ax.imshow(
            render_surface_thumbnail(
                surfaces[hemi],
                residual[column],
                hemi=hemi,
                view=view,
                cmap="coolwarm",
                vmin=-limit,
                vmax=limit,
                dpi=180,
            )
        )
        ax.set_title(title, fontsize=12, fontweight="bold", pad=1)
        ax.axis("off")
    residual_mappable = ScalarMappable(norm=Normalize(-limit, limit), cmap="coolwarm")
    color_axis = fig.add_subplot(brain_grid[:, 2])
    colorbar = fig.colorbar(residual_mappable, cax=color_axis)
    colorbar.set_label("Average z-score difference", fontsize=11, fontweight="bold")
    brain_title = fig.add_subplot(outer[0], frame_on=False)
    brain_title.set_title("Stage-adjusted spatial residuals", fontsize=16, fontweight="bold", pad=20)
    brain_title.set_axis_off()

    matrix = corr.pivot(
        index="subtype1_stage_group",
        columns="subtype2_stage_group",
        values="pearson_r",
    ).reindex(index=STAGE_GROUPS, columns=STAGE_GROUPS)
    ax_corr = fig.add_subplot(outer[1])
    image = ax_corr.imshow(matrix, cmap="viridis", vmin=float(matrix.min().min()), vmax=1.0)
    ax_corr.set_xticks(range(5), [f"S2 {x}" for x in STAGE_GROUPS], rotation=45, ha="right")
    ax_corr.set_yticks(range(5), [f"S1 {x}" for x in STAGE_GROUPS])
    for row in range(5):
        for column in range(5):
            value = matrix.iloc[row, column]
            ax_corr.text(
                column,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold",
                color="white" if value < 0.65 else "black",
            )
    cbar = fig.colorbar(image, ax=ax_corr, fraction=0.046, pad=0.03)
    cbar.set_label("Pearson r across 68 ROIs", fontsize=11, fontweight="bold")
    ax_corr.set_title("Cross-stage bin correlation", fontsize=16, fontweight="bold", pad=18)

    model_matrix = sm.add_constant(mmse["stage"].astype(float))
    model = sm.OLS(mmse["mmse"].astype(float), model_matrix).fit(
        cov_type="cluster", cov_kwds={"groups": mmse["subject_cluster"]}
    )
    stages = np.arange(21, dtype=float)
    prediction = model.get_prediction(sm.add_constant(stages, has_constant="add")).summary_frame()
    stage_means = mmse.groupby("stage", as_index=False)["mmse"].mean()
    rng = np.random.default_rng(42)
    ax_mmse = fig.add_subplot(outer[2])
    ax_mmse.scatter(
        mmse["stage"].to_numpy() + rng.uniform(-0.18, 0.18, len(mmse)),
        mmse["mmse"],
        s=24,
        alpha=0.15,
        edgecolors="none",
    )
    ax_mmse.plot(stages, prediction["mean"].to_numpy(), linewidth=2)
    ax_mmse.fill_between(
        stages,
        prediction["mean_ci_lower"].to_numpy(),
        prediction["mean_ci_upper"].to_numpy(),
        alpha=0.2,
    )
    ax_mmse.scatter(stage_means["stage"], stage_means["mmse"], s=48, color="#2ca02c", zorder=3)
    ax_mmse.set(xlabel="Stage", ylabel="MMSE", xlim=(-0.6, 20.6))
    ax_mmse.set_title("MMSE linear trend", fontsize=16, fontweight="bold", pad=18)
    intercept, slope = model.params["const"], model.params["stage"]
    ax_mmse.text(
        0.03,
        0.17,
        f"MMSE = {intercept:.2f} - {-slope:.3f} * stage\n"
        f"slope p = {model.pvalues['stage']:.2e}, R^2 = {model.rsquared:.3f}",
        transform=ax_mmse.transAxes,
        fontsize=11,
        fontweight="bold",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
    )
    negative = apoe.loc[apoe["apoe4_group"].eq("APOE4 negative"), "stage"].to_numpy(dtype=float)
    positive = apoe.loc[apoe["apoe4_group"].eq("APOE4 positive"), "stage"].to_numpy(dtype=float)
    ax_apoe = fig.add_subplot(outer[3])
    half_violin_with_points(
        ax_apoe,
        [negative, positive],
        ["APOE4-", "APOE4+"],
        ["#d4743c", "#6f2b6e"],
    )
    ax_apoe.set_ylabel("SuStaIn stage (0-20)")
    ax_apoe.set_title("APOE4 stage distribution", fontsize=16, fontweight="bold", pad=18)
    ax_apoe.text(
        0.03,
        0.97,
        f"APOE4-: mean = {negative.mean():.2f}\nAPOE4+: mean = {positive.mean():.2f}",
        transform=ax_apoe.transAxes,
        va="top",
        fontsize=11,
        fontweight="bold",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
    )
    for axis in (ax_corr, ax_mmse, ax_apoe):
        axis.tick_params(labelsize=11)
    return fig


def tau_rows(tau: pd.DataFrame) -> tuple[list[str], dict[str, pd.Series], float, float]:
    roi_columns = [column for column in tau.columns if column.startswith("CTX_LH_")]
    values = tau.set_index("map_name")[roi_columns]
    all_values = values.to_numpy(dtype=float).ravel()
    vmin, vmax = np.nanpercentile(all_values, [1, 99])
    names = [f"subtype1_stage{stage}" for stage in range(1, 21)]
    second = [f"subtype2_stage{stage}" for stage in range(1, 21)]
    missing = [name for name in names + second if name not in values.index]
    if missing:
        raise ValueError(f"Missing Tau-PET stage maps: {missing}")
    return roi_columns, {name: values.loc[name] for name in set(names + second)}, float(vmin), float(vmax)


def make_combined_figure(
    four_panel_png: Path,
    tau: pd.DataFrame,
    lh_surface: dict,
) -> plt.Figure:
    _, tau_values, vmin, vmax = tau_rows(tau)
    cache = {
        name: render_surface_thumbnail(
            lh_surface,
            values,
            hemi="lh",
            view="lateral",
            cmap="turbo",
            vmin=vmin,
            vmax=vmax,
            dpi=130,
        )
        for name, values in tau_values.items()
    }

    fig = plt.figure(figsize=(18.4, 10.2), constrained_layout=False)
    layout = fig.add_gridspec(
        5,
        2,
        height_ratios=[2.35, 1.0, 1.0, 1.0, 1.0],
        width_ratios=[1.0, 0.025],
        left=0.035,
        right=0.975,
        bottom=0.045,
        top=0.985,
        wspace=0.025,
        hspace=0.025,
    )
    upper = fig.add_subplot(layout[0, :])
    upper.imshow(Image.open(four_panel_png), aspect="auto")
    upper.axis("off")

    row_definitions = [
        ("subtype1", 1),
        ("subtype1", 11),
        ("subtype2", 1),
        ("subtype2", 11),
    ]
    for layout_row, (prefix, first_stage) in enumerate(row_definitions, start=1):
        row = layout[layout_row, 0].subgridspec(1, 10, wspace=0.01)
        for column, stage in enumerate(range(first_stage, first_stage + 10)):
            name = f"{prefix}_stage{stage}"
            axis = fig.add_subplot(row[0, column])
            axis.imshow(cache[name])
            axis.axis("off")

    fig.text(0.020, 0.490, "S1", rotation=90, va="center", fontsize=17, fontweight="bold")
    fig.text(0.020, 0.205, "S2", rotation=90, va="center", fontsize=17, fontweight="bold")
    color_axis = fig.add_subplot(layout[1:, 1])
    colorbar = fig.colorbar(
        ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap="turbo"),
        cax=color_axis,
        orientation="vertical",
    )
    colorbar.set_label("Tau-PET SUVR", fontsize=11, fontweight="bold")
    colorbar.ax.tick_params(labelsize=9)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--surface-data", type=Path, default=SURFACE_DATA)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.weight": "bold",
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "pdf.fonttype": 42,
        }
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    brain, corr, mmse, apoe, tau = load_plot_data()
    surfaces = load_surfaces(args.surface_data)

    upper_path = args.output_dir / "dk68_brain_bin_mmse_apoe4_1x4.png"
    upper = make_four_panel_figure(brain, corr, mmse, apoe, surfaces)
    upper.savefig(upper_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(upper)

    combined = make_combined_figure(upper_path, tau, surfaces["lh"])
    pdf_path = args.output_dir / "fig_tau_sustain_ap_2.pdf"
    combined.savefig(pdf_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(combined)
    print(f"Saved: {upper_path}")
    print(f"Saved: {pdf_path}")


if __name__ == "__main__":
    main()
