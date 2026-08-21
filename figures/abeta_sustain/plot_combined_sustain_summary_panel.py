#!/usr/bin/env python
"""Reproduce combined_sustain_summary_panel.jpg from compact source-data CSVs."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "sustain_figure_mpl_cache"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data" / "combined_sustain_summary_panel"
OUTPUT = HERE / "outputs" / "combined_sustain_summary_panel.jpg"


def configure_style() -> None:
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.weight": "bold",
        "font.size": 12,
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
    })


def kde_1d(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if len(values) < 2 or np.std(values, ddof=1) < 1e-12:
        return np.zeros_like(grid)
    bw = max(1.06 * np.std(values, ddof=1) * len(values) ** (-1 / 5), 1e-6)
    z = (grid[:, None] - values[None, :]) / bw
    return np.exp(-0.5 * z ** 2).sum(axis=1) / (len(values) * bw * np.sqrt(2 * np.pi))


def raincloud(ax, dataframe, group_col, value_col, order, ylabel) -> None:
    fixed_colors = {
        "Stable": "#d4743c", "Changed": "#6f2b6e",
        "Aβ-": "#ff7f0e", "Aβ+": "#6f2b6e",
        "CN": "#d4743c", "MCI": "#6f2b6e", "AD": "#2a9d8f",
    }
    groups = [
        dataframe.loc[dataframe[group_col].eq(label), value_col].dropna().to_numpy(float)
        for label in order
    ]
    positions = np.arange(1, len(order) + 1)
    rng = np.random.default_rng(0)
    for pos, label, values in zip(positions, order, groups):
        color = fixed_colors[label]
        span = np.ptp(values)
        grid = np.linspace(values.min() - 0.05 * span, values.max() + 0.05 * span, 250)
        density = kde_1d(values, grid)
        if density.max() > 0:
            ax.fill_betweenx(grid, pos - density / density.max() * 0.42, pos,
                             color=color, alpha=0.95, lw=0, zorder=1)
        ax.scatter(pos + 0.26 + rng.normal(0, 0.055, len(values)), values,
                   s=22, color=color, alpha=0.8, zorder=3)
    ax.boxplot(groups, positions=positions + 0.26, widths=0.18, patch_artist=True,
               showfliers=False, manage_ticks=False,
               boxprops=dict(facecolor="white", edgecolor="black", linewidth=1.8),
               whiskerprops=dict(color="black", linewidth=1.8),
               capprops=dict(color="black", linewidth=1.8),
               medianprops=dict(color="black", linewidth=2.2))
    ax.set_xticks(positions, order)
    ax.set_ylabel(ylabel)
    ax.set_xlim(positions[0] - 0.6, positions[-1] + 0.6)
    values = np.concatenate(groups)
    pad = 0.08 * np.ptp(values) if np.ptp(values) else 0.5
    ax.set_ylim(values.min() - pad, values.max() + pad)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(False)


def plot_probability_heatmap(ax, probability: pd.DataFrame) -> None:
    columns = [col for col in probability if col.startswith("subtype_") and col.endswith("_prob")]
    image = ax.imshow(probability[columns].to_numpy(float).T, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_yticks(np.arange(len(columns)), [f"S{i + 1}" for i in range(len(columns))])
    ax.set_xticks([])
    ax.set_xlabel("stage-positive scans")
    ax.set_ylabel("Subtype")
    colorbar = ax.figure.colorbar(image, ax=ax, fraction=0.035, pad=0.03)
    colorbar.set_label("Posterior probability")


def plot_transition_heatmap(ax, pairs: pd.DataFrame) -> None:
    states = ["S0", "S1", "S2"]
    counts = pd.crosstab(pairs["baseline_state"], pairs["followup_state"]).reindex(
        index=states, columns=states, fill_value=0
    )
    shown = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0) * 100
    image = ax.imshow(shown, cmap="Blues", vmin=0, vmax=np.nanmax(shown.to_numpy()))
    for i in range(3):
        for j in range(3):
            value = shown.iloc[i, j]
            color = "white" if value > 0.55 * np.nanmax(shown.to_numpy()) else "#222222"
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", color=color, weight="bold")
    ax.set_xticks(range(3), states)
    ax.set_yticks(range(3), states)
    ax.set_xlabel("First follow-up state")
    ax.set_ylabel("Baseline state")
    colorbar = ax.figure.colorbar(image, ax=ax, fraction=0.035, pad=0.03)
    colorbar.set_label("Row percentage (%)")


def plot_followup_scatter(ax, pairs: pd.DataFrame) -> None:
    ax.scatter(pairs["baseline_stage"], pairs["followup_stage"], s=45,
               alpha=0.55, color="#4c72b0", edgecolors="none")
    maximum = int(np.nanmax(pairs[["baseline_stage", "followup_stage"]].to_numpy()))
    ax.plot([0, maximum], [0, maximum], "--", color="black", linewidth=1)
    ax.set_xlabel("Baseline stage")
    ax.set_ylabel("First follow-up stage")
    ax.grid(False)


def build_figure(data_dir: Path, output: Path, dpi: int) -> None:
    configure_style()
    probability = pd.read_csv(data_dir / "subtype_probability_heatmap.csv")
    pairs = pd.read_csv(data_dir / "longitudinal_first_followup_pairs.csv")
    scans = pd.read_csv(data_dir / "scan_stage_groups.csv", encoding="utf-8-sig")

    fig, axes = plt.subplots(2, 3, figsize=(18.0, 10.2), constrained_layout=True)
    plot_probability_heatmap(axes[0, 0], probability)
    plot_transition_heatmap(axes[0, 1], pairs)
    plot_followup_scatter(axes[0, 2], pairs)

    positive_pairs = pairs[(pairs["baseline_stage"] > 0) & (pairs["followup_stage"] > 0)].copy()
    positive_pairs["stability"] = np.where(
        positive_pairs["baseline_subtype"].eq(positive_pairs["followup_subtype"]),
        "Stable", "Changed",
    )
    raincloud(axes[1, 0], positive_pairs, "stability", "baseline_subtype_probability",
              ["Stable", "Changed"], "Baseline subtype probability")
    raincloud(axes[1, 1], scans.dropna(subset=["abeta_group"]), "abeta_group", "stage",
              ["Aβ-", "Aβ+"], "Sustain stage")
    raincloud(axes[1, 2], scans.dropna(subset=["research_group"]), "research_group", "stage",
              ["CN", "MCI", "AD"], "Sustain stage")

    for ax in axes.flat:
        ax.tick_params(labelsize=11)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()
    build_figure(args.data_dir, args.output, args.dpi)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
