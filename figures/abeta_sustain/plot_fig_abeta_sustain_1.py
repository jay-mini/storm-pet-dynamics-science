#!/usr/bin/env python
"""Reproduce fig_abeta_sustain_1 from compact, figure-ready source data."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "sustain_figure_mpl_cache"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd
import statsmodels.api as sm
from nilearn import plotting


HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data" / "fig_abeta_sustain_1"
OUTPUT_DIR = HERE / "outputs"

DESIKAN34 = [
    "BANKSSTS", "CAUDALANTERIORCINGULATE", "CAUDALMIDDLEFRONTAL", "CUNEUS",
    "ENTORHINAL", "FUSIFORM", "INFERIORPARIETAL", "INFERIORTEMPORAL",
    "ISTHMUSCINGULATE", "LATERALOCCIPITAL", "LATERALORBITOFRONTAL", "LINGUAL",
    "MEDIALORBITOFRONTAL", "MIDDLETEMPORAL", "PARAHIPPOCAMPAL", "PARACENTRAL",
    "PARSOPERCULARIS", "PARSORBITALIS", "PARSTRIANGULARIS", "PERICALCARINE",
    "POSTCENTRAL", "POSTERIORCINGULATE", "PRECENTRAL", "PRECUNEUS",
    "ROSTRALANTERIORCINGULATE", "ROSTRALMIDDLEFRONTAL", "SUPERIORFRONTAL",
    "SUPERIORPARIETAL", "SUPERIORTEMPORAL", "SUPRAMARGINAL", "FRONTALPOLE",
    "TEMPORALPOLE", "TRANSVERSETEMPORAL", "INSULA",
]


def configure_style() -> None:
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.weight": "bold",
        "font.size": 10,
        "axes.titleweight": "bold",
        "axes.labelweight": "bold",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def load_surface_data(path: Path) -> dict[str, dict[str, np.ndarray]]:
    raw = np.load(path)
    return {
        hemi: {key: raw[f"{hemi}_{key}"] for key in ("coords", "faces", "sulc", "labels", "names")}
        for hemi in ("lh", "rh")
    }


def roi_values_to_vertices(values: pd.Series, surface: dict[str, np.ndarray]) -> np.ndarray:
    name_to_id = {str(name).lower(): i for i, name in enumerate(surface["names"])}
    vertices = np.full(surface["labels"].shape, np.nan, dtype=float)
    for roi in DESIKAN34:
        vertices[surface["labels"] == name_to_id[roi.lower()]] = float(values.loc[roi])
    return vertices


def draw_surface(ax, surface, values, *, hemi, view, cmap, vmin, vmax, title="") -> None:
    plotting.plot_surf_stat_map(
        (surface["coords"], surface["faces"]),
        stat_map=roi_values_to_vertices(values, surface),
        bg_map=surface["sulc"],
        hemi="left" if hemi == "lh" else "right",
        view=view,
        cmap=cmap,
        colorbar=False,
        threshold=None,
        bg_on_data=True,
        darkness=None,
        alpha=1.0,
        vmin=vmin,
        vmax=vmax,
        axes=ax,
        figure=ax.figure,
    )
    ax.set_title("")
    if title:
        ax.text2D(0.5, 0.94, title, transform=ax.transAxes,
                  ha="center", va="top", fontsize=10, weight="bold")
    ax.set_axis_off()
    for collection in ax.collections:
        collection.set_rasterized(True)


def draw_brain_panel(fig, spec, surfaces, residuals) -> None:
    grid = spec.subgridspec(2, 2, wspace=-0.10, hspace=-0.18)
    vmax = float(np.nanmax(np.abs(residuals["average_z_score_difference"])))
    axes = []
    for index, (hemi, view, title) in enumerate([
        ("lh", "lateral", "LH lateral"), ("rh", "lateral", "RH lateral"),
        ("lh", "medial", "LH medial"), ("rh", "medial", "RH medial"),
    ]):
        ax = fig.add_subplot(grid[index // 2, index % 2], projection="3d")
        values = residuals[residuals["hemisphere"].eq(hemi.upper())].set_index("roi")[
            "average_z_score_difference"
        ].reindex(DESIKAN34)
        draw_surface(ax, surfaces[hemi], values, hemi=hemi, view=view,
                     cmap="coolwarm", vmin=-vmax, vmax=vmax, title=title)
        axes.append(ax)
    x1 = max(ax.get_position().x1 for ax in axes)
    y0 = min(ax.get_position().y0 for ax in axes)
    y1 = max(ax.get_position().y1 for ax in axes)
    cax = fig.add_axes([x1 + 0.004, y0 + 0.25 * (y1 - y0), 0.006, 0.50 * (y1 - y0)])
    cb = fig.colorbar(ScalarMappable(Normalize(-vmax, vmax), cmap="coolwarm"), cax=cax)
    cb.set_label("Average z-score difference", fontsize=9)
    cb.ax.tick_params(labelsize=8)


def draw_mmse_panel(ax, points: pd.DataFrame) -> None:
    mmse = points.dropna(subset=["subject_id", "stage", "MMSE"]).copy()
    model = sm.OLS(mmse["MMSE"].astype(float), sm.add_constant(mmse["stage"].astype(float))).fit(
        cov_type="cluster", cov_kwds={"groups": mmse["subject_id"]}
    )
    stages = np.arange(int(mmse["stage"].max()) + 1)
    pred = model.get_prediction(sm.add_constant(stages, has_constant="add")).summary_frame(alpha=0.05)
    rng = np.random.default_rng(42)
    ax.scatter(mmse["stage"] + rng.uniform(-0.18, 0.18, len(mmse)), mmse["MMSE"],
               s=14, alpha=0.15, edgecolors="none", color="#1f77b4")
    ax.plot(stages, pred["mean"], color="#1f77b4", lw=1.5)
    ax.fill_between(stages, pred["mean_ci_lower"], pred["mean_ci_upper"],
                    color="#f4a261", alpha=0.25)
    means = mmse.groupby("stage", as_index=False)["MMSE"].mean()
    ax.scatter(means["stage"], means["MMSE"], s=28, color="#2ca02c", zorder=3)
    beta0, beta1 = model.params["const"], model.params["stage"]
    ax.text(0.03, 0.17,
            f"MMSE = {beta0:.2f} - {-beta1:.3f} * stage\n"
            f"slope p = {model.pvalues['stage']:.2e}, R^2 = {model.rsquared:.3f}",
            transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.9))
    ax.set(xlabel="Stage", ylabel="MMSE", xlim=(-0.6, stages[-1] + 0.6))


def kde_1d(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    values = np.asarray(values, float)
    if len(values) < 2 or np.std(values, ddof=1) < 1e-12:
        return np.zeros_like(grid)
    bw = max(1.06 * np.std(values, ddof=1) * len(values) ** (-1 / 5), 1e-6)
    z = (grid[:, None] - values[None, :]) / bw
    return np.exp(-0.5 * z ** 2).sum(axis=1) / (len(values) * bw * np.sqrt(2 * np.pi))


def raincloud(ax, groups, labels, colors, *, ylabel) -> None:
    rng = np.random.default_rng(0)
    positions = np.arange(1, len(groups) + 1)
    for pos, values, color in zip(positions, groups, colors):
        values = np.asarray(values, float)
        grid = np.linspace(values.min() - 0.05 * np.ptp(values), values.max() + 0.05 * np.ptp(values), 250)
        density = kde_1d(values, grid)
        if density.max() > 0:
            ax.fill_betweenx(grid, pos - density / density.max() * 0.42, pos,
                             color=color, alpha=0.95, lw=0)
        ax.scatter(pos + 0.26 + rng.normal(0, 0.055, len(values)), values,
                   s=12, color=color, alpha=0.8)
    ax.boxplot(groups, positions=positions + 0.26, widths=0.18, patch_artist=True,
               showfliers=False, manage_ticks=False,
               boxprops=dict(facecolor="white", edgecolor="black", linewidth=1.2),
               whiskerprops=dict(color="black", linewidth=1.2),
               capprops=dict(color="black", linewidth=1.2),
               medianprops=dict(color="black", linewidth=1.4))
    ax.set_xticks(positions, labels)
    ax.set_ylabel(ylabel)
    ax.set_xlim(positions[0] - 0.6, positions[-1] + 0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(False)


def draw_apoe_panel(ax, points: pd.DataFrame) -> None:
    apoe = points.dropna(subset=["stage", "APOE4_carrier"])
    neg = apoe.loc[apoe["APOE4_carrier"].eq(0), "stage"].to_numpy(float)
    pos = apoe.loc[apoe["APOE4_carrier"].eq(1), "stage"].to_numpy(float)
    raincloud(ax, [neg, pos], ["APOE4-", "APOE4+"], ["#d4743c", "#6f2b6e"],
              ylabel=f"Stage (0-{int(apoe['stage'].max())})")
    ax.text(0.03, 0.78, f"APOE4-: mean = {neg.mean():.2f}\nAPOE4+: mean = {pos.mean():.2f}",
            transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white"), zorder=5)


def draw_tau_block(fig, spec, surfaces, tau, display, subtype_label, vmin, vmax) -> None:
    rows = display[display["subtype_label"].eq(subtype_label)].sort_values("display_order")
    grid = spec.subgridspec(2, 6, wspace=-0.22, hspace=-0.20)
    for index, row in enumerate(rows.itertuples(index=False)):
        ax = fig.add_subplot(grid[index // 6, index % 6], projection="3d")
        values = tau[tau["map_name"].eq(row.map_name)].iloc[0, 1:]
        values.index = [str(col).replace("CTX_LH_", "").replace("_SUVR", "") for col in values.index]
        draw_surface(ax, surfaces["lh"], values, hemi="lh", view="lateral",
                     cmap="turbo", vmin=vmin, vmax=vmax)


def build_figure(data_dir: Path, output_pdf: Path, output_png: Path | None) -> None:
    configure_style()
    surfaces = load_surface_data(data_dir / "fsaverage5_surface_data.npz")
    residuals = pd.read_csv(data_dir / "dk68_stage_adjusted_spatial_residuals.csv")
    points = pd.read_csv(data_dir / "dk68_stage_mmse_apoe4_points.csv")
    tau = pd.read_csv(data_dir / "tau_pet_lh_stage_mean_values.csv")
    display = pd.read_csv(data_dir / "tau_pet_display_order.csv")
    scale = pd.read_csv(data_dir / "tau_pet_color_scale.csv").iloc[0]
    vmin, vmax = float(scale["vmin"]), float(scale["vmax"])

    fig = plt.figure(figsize=(17.42, 13.41), constrained_layout=False)
    outer = fig.add_gridspec(3, 1, height_ratios=[1.06, 1.0, 1.0],
                             left=0.025, right=0.965, bottom=0.025, top=0.975,
                             hspace=0.03)
    top = outer[0].subgridspec(
        2, 3, height_ratios=[0.90, 0.10], width_ratios=[1.15, 1.0, 1.0],
        wspace=0.28, hspace=0.0,
    )
    draw_brain_panel(fig, top[0, 0], surfaces, residuals)
    mmse_ax = fig.add_subplot(top[0, 1])
    apoe_ax = fig.add_subplot(top[0, 2])
    draw_mmse_panel(mmse_ax, points)
    draw_apoe_panel(apoe_ax, points)

    for ax, title in ((mmse_ax, "MMSE linear trend"), (apoe_ax, "APOE4 stage distribution")):
        ax.set_title(title, fontsize=12, pad=8)
        ax.tick_params(labelsize=9)
    fig.text(0.16, 0.982, "Stage-adjusted spatial residuals", ha="center", fontsize=12, weight="bold")

    draw_tau_block(fig, outer[1], surfaces, tau, display, "S1", vmin, vmax)
    draw_tau_block(fig, outer[2], surfaces, tau, display, "S2", vmin, vmax)
    fig.text(0.495, 0.606, "S1", ha="center", fontsize=12, weight="bold")
    fig.text(0.495, 0.293, "S2", ha="center", fontsize=12, weight="bold")
    cax = fig.add_axes([0.972, 0.12, 0.012, 0.34])
    cb = fig.colorbar(ScalarMappable(Normalize(vmin, vmax), cmap="turbo"), cax=cax)
    cb.set_label("Tau-PET SUVR", fontsize=10)
    cb.ax.tick_params(labelsize=9)

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf, bbox_inches="tight")
    if output_png is not None:
        fig.savefig(output_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-pdf", type=Path, default=OUTPUT_DIR / "fig_abeta_sustain_1.pdf")
    parser.add_argument("--output-png", type=Path, default=OUTPUT_DIR / "fig_abeta_sustain_1.png")
    args = parser.parse_args()
    build_figure(args.data_dir, args.output_pdf, args.output_png)
    print(f"Saved: {args.output_pdf}")
    print(f"Saved: {args.output_png}")


if __name__ == "__main__":
    main()
