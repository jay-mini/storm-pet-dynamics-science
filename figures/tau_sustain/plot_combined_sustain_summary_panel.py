#!/usr/bin/env python
"""Recreate combined_sustain_summary_panel.jpg from compact plotting data."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data" / "combined_sustain_summary_panel"
DEFAULT_OUTPUT = HERE / "outputs" / "combined_sustain_summary_panel.jpg"
STATE_ORDER = ["S0", "S1", "S2"]


def kde_density(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    if len(values) < 2 or np.std(values, ddof=1) < 1e-12:
        return np.zeros_like(grid)
    bandwidth = max(1.06 * np.std(values, ddof=1) * len(values) ** (-0.2), 1e-6)
    z = (grid[:, None] - values[None, :]) / bandwidth
    return np.exp(-0.5 * z**2).sum(axis=1) / (
        len(values) * bandwidth * np.sqrt(2 * np.pi)
    )


def raincloud(
    axis: plt.Axes,
    data: pd.DataFrame,
    group_column: str,
    value_column: str,
    order: list[str],
    colors: list[str],
    ylabel: str,
) -> None:
    rng = np.random.default_rng(0)
    positions = np.arange(1, len(order) + 1)
    arrays = []
    for position, label, color in zip(positions, order, colors):
        values = pd.to_numeric(
            data.loc[data[group_column].eq(label), value_column], errors="coerce"
        ).dropna().to_numpy(dtype=float)
        arrays.append(values)
        if len(values) == 0:
            continue
        low, high = values.min(), values.max()
        padding = 0.05 * (high - low) if high > low else 0.5
        grid = np.linspace(low - padding, high + padding, 250)
        density = kde_density(values, grid)
        if density.max() > 0:
            width = density / density.max() * 0.42
            axis.fill_betweenx(grid, position - width, position, color=color, alpha=0.95, linewidth=0)
        axis.scatter(
            position + 0.26 + rng.normal(0, 0.055, len(values)),
            values,
            s=22,
            color=color,
            alpha=0.8,
            edgecolors="none",
            zorder=3,
        )

    nonempty = [(position, values) for position, values in zip(positions, arrays) if len(values)]
    if nonempty:
        boxes = axis.boxplot(
            [values for _, values in nonempty],
            positions=[position + 0.26 for position, _ in nonempty],
            widths=0.18,
            patch_artist=True,
            showfliers=False,
            manage_ticks=False,
            boxprops={"facecolor": "white", "edgecolor": "black", "linewidth": 1.8},
            whiskerprops={"color": "black", "linewidth": 1.8},
            capprops={"color": "black", "linewidth": 1.8},
            medianprops={"color": "black", "linewidth": 2.2},
        )
        for group in ("boxes", "whiskers", "caps", "medians"):
            for artist in boxes[group]:
                artist.set_zorder(4)
    axis.set_xticks(positions, order)
    axis.set_ylabel(ylabel)
    axis.set_xlim(positions[0] - 0.6, positions[-1] + 0.6)
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(False)


def make_figure(longitudinal: pd.DataFrame, cross_sectional: pd.DataFrame) -> plt.Figure:
    sns.set_theme(style="ticks")
    fig, axes = plt.subplots(2, 3, figsize=(18.0, 10.2), constrained_layout=True)
    axes[0, 0].axis("off")

    counts = pd.crosstab(
        longitudinal["baseline_state"], longitudinal["followup_state"]
    ).reindex(index=STATE_ORDER, columns=STATE_ORDER, fill_value=0)
    percentages = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0) * 100
    heat = sns.heatmap(
        percentages,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        cbar=True,
        ax=axes[0, 1],
    )
    heat.collections[0].colorbar.set_label("Row percentage (%)")
    axes[0, 1].set_xlabel("First follow-up state")
    axes[0, 1].set_ylabel("Baseline state")

    axes[0, 2].scatter(
        longitudinal["baseline_stage"],
        longitudinal["followup_stage"],
        alpha=0.55,
        s=75,
    )
    maximum = int(
        np.nanmax(
            [longitudinal["baseline_stage"].max(), longitudinal["followup_stage"].max()]
        )
    )
    axes[0, 2].plot([0, maximum], [0, maximum], "--", color="black", linewidth=1)
    axes[0, 2].set_xlabel("Baseline stage")
    axes[0, 2].set_ylabel("First follow-up stage")
    axes[0, 2].grid(False)

    stability = longitudinal[
        longitudinal["baseline_stage"].gt(0) & longitudinal["followup_stage"].gt(0)
    ].copy()
    stability["group"] = np.where(
        stability["baseline_subtype"].eq(stability["followup_subtype"]),
        "Stable",
        "Changed",
    )
    raincloud(
        axes[1, 0],
        stability,
        "group",
        "baseline_subtype_probability",
        ["Stable", "Changed"],
        ["#d4743c", "#6f2b6e"],
        "Baseline subtype probability",
    )
    raincloud(
        axes[1, 1],
        cross_sectional,
        "abeta_group",
        "stage",
        ["A-beta negative", "A-beta positive"],
        ["#1f77b4", "#ff7f0e"],
        "SuStaIn stage",
    )
    axes[1, 1].set_xticklabels(["Aβ-", "Aβ+"])
    raincloud(
        axes[1, 2],
        cross_sectional,
        "research_group",
        "stage",
        ["CN", "MCI", "AD"],
        ["#d4743c", "#6f2b6e", "#2a9d8f"],
        "SuStaIn stage",
    )
    return fig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.weight": "bold",
            "font.size": 16,
            "axes.labelweight": "bold",
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
        }
    )
    longitudinal = pd.read_csv(DATA_DIR / "longitudinal_first_followup.csv")
    cross_sectional = pd.read_csv(DATA_DIR / "cross_sectional_stage_groups.csv")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure = make_figure(longitudinal, cross_sectional)
    figure.savefig(args.output, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
